"""
tamp_manager.py
===============
TAMP (Task and Motion Planning) manager integrating:
  - MediaPipe teleop input (ROS 2)
  - Collision-free approach controller  (integrated from ros2_ws/src/approach_controller.py)
  - Continuously-replanning multi-start IPOPT grasp planner with wrench feasibility ranking
  - Live grasp recommendation browser (overlaid on live teleoperation)
  - Interpolated grasp execution
  - Post-grasp object tracking + wrench monitoring
  - Hand-open detection for object placement

State Machine
-------------
  IDLE          waiting for calibration  (C key)
  APPROACH      collision-free IK tracks hand position; grasp planner reruns every
                  replan_interval_sec seconds in the background; ranked candidates are
                  shown as contact-point markers overlaid on the live robot so the human
                  can browse and accept at any time without interrupting teleoperation
  EXECUTING     interpolate fingers from current config to accepted grasp config
  TRANSPORTING  fingers locked; base tracks hand; monitor contact; detect hand-open
  PLACING       interpolate fingers to open; release object -> back to APPROACH

Key bindings  (MuJoCo viewer must be focused)
---------------------------------------------
  C     calibrate wrist offset           (any state)
  J/K   next / previous candidate        (APPROACH, while candidates exist)
  Y     accept current candidate         (APPROACH)
  N     discard current candidates       (APPROACH)
  O     manually trigger place           (TRANSPORTING)
  R     reset to APPROACH                (any state)
"""

from __future__ import annotations

import csv
import enum
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial import ConvexHull

import casadi as ca
from casadi import Callback as _Callback

# ── ApproachController import ─────────────────────────────────────────────────
# Prefer the version already in teleop_manager_v2 (requires ROS 2 to be
# present on import).  Falls back to a local re-implementation if unavailable.
sys.path.insert(0, str(Path(__file__).parent))

# Shared 
# d-distance callback that operates over nu ACTUATED joints only.
# Accepts nu inputs, expands to full qpos internally, so the optimizer dimension matches q_teleop (nu=8)
# rather than model.nq (=15, which includes the object's free joint).
def _make_sdc_actuated(ca, Callback):
    class _SDCActuated(Callback):
        def __init__(self, name, model, data, gid1, gid2, act_idx, opts={}):
            Callback.__init__(self)
            self.model   = model;    self.data  = data
            self.gid1    = gid1;     self.gid2  = gid2
            self.act_idx = act_idx;  self.fromto = np.zeros(6)
            self.construct(name, opts)
        def get_n_in(self):  return 1
        def get_n_out(self): return 1
        def get_sparsity_in(self, i):  return ca.Sparsity.dense(len(self.act_idx), 1)
        def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
        def has_jacobian(self): return False
        def eval(self, arg):
            q_act = np.array(arg[0]).flatten()
            q_full = self.data.qpos.copy()
            for idx, val in zip(self.act_idx, q_act):
                q_full[idx] = val
            self.data.qpos[:] = q_full
            mujoco.mj_forward(self.model, self.data)
            return [mujoco.mj_geomDistance(self.model, self.data,
                                           self.gid1, self.gid2, 1.0, self.fromto)]
    return _SDCActuated

_SDCActuated = _make_sdc_actuated(ca, _Callback)


class ApproachController:
    """
    Approach controller for the TAMP pipeline.
    - Optimises over nu actuated joints (parent uses nq=15 which includes
        the object free joint, causing a dimension mismatch with q_teleop)
    - Constrains both fingers
    - Adds finger_object_distance() for proximity triggering

    Given a reference q from MediaPipe teleop, returns the nearest q that
    keeps the finger distal geoms outside the object geom by at least
    `clearance` metres.

    Parameters
    ----------
    model     : mujoco.MjModel
    clearance : minimum 
    d distance between finger and object (m)
    max_iter  : IPOPT iteration limit (keep small for real-time use)
    """

    # Geom names that must exist in the XML
    _THUMB_GEOM = "right_thumb_distal_geom"
    _INDEX_GEOM = "right_index_distal_geom"
    _OBJ_GEOM   = "obj1_geom"

    def __init__(self,
                 model:     mujoco.MjModel,
                 clearance: float = 2e-3, 
                 max_iter:  int   = 1000):
        self.model     = model
        self.clearance = clearance   # in meters
        self.max_iter  = max_iter

        # Dedicated data copy — approach controller never shares state
        self._data = mujoco.MjData(model)
        mujoco.mj_forward(model, self._data)

        self._thumb_gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, self._THUMB_GEOM)
        self._index_gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, self._INDEX_GEOM)
        self._obj_gid   = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, self._OBJ_GEOM)

        call_opts = {"enable_fd": True}
        self._dist_index = _SDCActuated(
            "ac_dist_index", model, self._data,
            self._index_gid, self._obj_gid, call_opts)
        self._dist_thumb = _SDCActuated(
            "ac_dist_thumb", model, self._data,
            self._thumb_gid, self._obj_gid, call_opts)
        
        self._act_idx = [
            model.jnt_qposadr[model.actuator_trnid[i, 0]]
            for i in range(model.nu)
        ]

        self._dist_index_act = _SDCActuated(
            "ac_dist_index_act", model, self._data,
            self._index_gid, self._obj_gid, self._act_idx, call_opts)
        self._dist_thumb_act = _SDCActuated(
            "ac_dist_thumb_act", model, self._data,
            self._thumb_gid, self._obj_gid, self._act_idx, call_opts)

        self._solver_opts = {
            "print_time": False,
            "ipopt": {
                "jacobian_approximation": "finite-difference-values",
                "hessian_approximation":  "limited-memory",
                "print_level": 0,
                "sb": "yes",
                "max_iter": self.max_iter,
            },
            "ad_weight_sp": 0,
        }


    def sync_state(self, live_data: mujoco.MjData) -> None:
        """Copy the live simulation qpos/qvel into this controller's data."""
        self._data.qpos[:] = live_data.qpos[:]
        self._data.qvel[:] = live_data.qvel[:]
        mujoco.mj_forward(self.model, self._data)


    def get_collision_free_joint_angles(self, q_ref):
        nu   = len(self._act_idx)
        opti = ca.Opti()
        q    = opti.variable(nu)
        opti.minimize(ca.bilin(ca.diag(np.ones(nu)), q - q_ref, q - q_ref))
        opti.subject_to(self._dist_index_act(q) >= self.clearance)
        opti.subject_to(self._dist_thumb_act(q) >= self.clearance)
        opti.solver("ipopt", self._solver_opts)
        opti.set_initial(q, q_ref)
        try:
            return opti.solve().value(q)
        except Exception:
            try:    return opti.debug.value(q)
            except: return np.asarray(q_ref).copy()


    def finger_object_distance(self, live_data):
        self.sync_state(live_data)
        d_index = mujoco.mj_geomDistance(
            self.model, self._data,
            self._index_gid, self._obj_gid, 1.0, np.zeros(6))
        d_thumb = mujoco.mj_geomDistance(
            self.model, self._data,
            self._thumb_gid, self._obj_gid, 1.0, np.zeros(6))
        return float(min(d_index, d_thumb))


# ── GraspPlanner ──────────────────────────────────────────────────────────────
from grasp_planner import (GraspConfig, GraspPlanner,
                            _FingerPointDistCallback, _SDFCallback,
                            _NonPenetrationCallback)

# ── ROS 2 ────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, Empty
    _HAS_ROS = True
except ImportError:
    _HAS_ROS = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(name)s] %(message)s")
log = logging.getLogger("tamp_manager")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TAMPConfig:
    model_path: str = "../models/planar_two_finger_manipulator.xml"

    # Contact-point enumeration grasp planner
    n_seeds: int = 2   # total seeds distributed across all face pairs (>=6 = 1 per pair)
    ms_max_iter:               int = 500  # IPOPT iteration cap per solve

    # Minimum contact quality gate before EXECUTING -> TRANSPORTING.
    # If either fingertip is farther than this from the object surface the
    # grasp is rejected and the planner is asked to rerun.
    contact_gate_mm: float = 3.0

    # Approach controller
    approach_clearance: float = 0.10   # m
    approach_max_iter:  int   = 1000
    approach_ik_freq:   int   = 5      # run approach IK every N sim steps

    # Continuous background replanning
    replan_interval_sec:  float = 3.0   # seconds between replans while in APPROACH
    proximity_threshold:  float = 0.12  # m -- replanning only starts below this distance

    # Grasp execution interpolation
    grasp_interp_steps: int = 60

    # Hand-open detection (release trigger)
    hand_open_threshold: float = 0.25  # mean |finger angle| rad
    hand_open_window:    int   = 15    # rolling average window

    # Wrench parameters
    mu:       float = 1.0   # friction coefficient (matches XML friction="1 ...")
    obj_mass: float = 0.2   # kg

    # Post-grasp contact monitoring
    wrench_monitor_freq: int = 20  # sim steps between checks

    # Camera -> robot frame scaling
    # MediaPipe: X = right (0-1), Y = down (0-1), Z = depth
    # Robot:     X = right,       Y = up
    scale_x: float = 6.0   # camera X  -> robot X
    scale_y: float = 6.0   # camera Y  -> robot Y  (negated: image-down = robot-down)

    # Teleop smoothing: exponential moving average applied to q_teleop
    # 1.0 = no smoothing (raw), 0.0 = frozen; ~0.2-0.4 works well
    smoothing_alpha: float = 0.3

    # ROM calibration capture window per pose (seconds)
    calib_duration_sec: float = 2.0

    # Calibration mode
    # False (default): two-press full ROM calibration (open hand + pinch)
    # True:            one-press wrist-only calibration; finger angles use a
    #                  fixed mapping: open_offset_deg is the resting curl to
    #                  subtract, human_max_deg is the assumed full ROM span.
    simple_calib:          bool  = False
    calib_open_offset_deg: float = 5.0   # degrees of residual curl at open pose
    calib_human_max_deg:   float = 70.0  # assumed open->pinch span in degrees

    # Debug: log MediaPipe angles -> robot controls every N sim steps (0 = off)
    debug_angles: bool = False

    # CSV log file path ("" to disable).  Written during APPROACH.
    # Columns: time_s, state, 6x human angles (deg), 6x robot angles (deg),
    #          8x ctrl values (rad/m) sent to MuJoCo.
    angle_log_path: str = "tamp_angles.csv"

    # Log one row every this many sim steps (~30 rows/s at 1 kHz sim, matching camera rate)
    log_freq: int = 30


class TAMPState(enum.Enum):
    IDLE         = "IDLE"
    CALIBRATING  = "CALIBRATING"   # ROM calibration: open-hand then fist capture
    APPROACH     = "APPROACH"
    EXECUTING    = "EXECUTING"
    TRANSPORTING = "TRANSPORTING"
    PLACING      = "PLACING"


# =============================================================================
# Wrench utilities
#
# Note: calc_wrench below is adapted from models/friction_wrench_cone.py.
# The original reads obj_pos from module-level MuJoCo data, making it inseparable
# from the simulation setup at the top of that script.  Here it is rewritten as a
# pure function by accepting obj_pos explicitly, so it can be called without
# triggering that script's import-time side effects.
# The body-frame == global-frame assumption (valid while the object is unrotated)
# is preserved as a docstring note rather than a runtime assert.
# =============================================================================

def calc_wrench(sensordata: np.ndarray, mu_t: float, obj_pos: np.ndarray):
    """
    Compute friction-cone wrench edge vectors at one contact in the object
    body frame.

    Adapted from models/friction_wrench_cone.py -- made pure by accepting
    obj_pos as a parameter instead of reading it from module-level simulation
    state.

    Parameters
    ----------
    sensordata : (13,) array
        [0]     found? (0 -> absent, returns 0)
        [1-3]   contact force in contact frame  [Fn, Ft, 0]
        [4-6]   contact position in global frame
        [7-9]   contact normal direction in global frame
        [10-12] contact tangent direction in global frame
    mu_t    : tangential friction coefficient
    obj_pos : (2,) or (3,) object body position in global frame.
              Body frame is assumed equal to global frame (object not rotated).

    Returns
    -------
    (F1_b, F2_b, F0_b) -- three (3,) wrench vectors [Mz, fx, fy] in body frame.
    Returns 0 if contact is absent.
    """
    assert len(sensordata) == 13, "sensordata must be length 13"
    if sensordata[0] == 0:
        return 0

    F_c       = sensordata[1:4]
    Fnormal_c = F_c[0]
    p_gc      = sensordata[4:7]
    normal_g  = sensordata[7:10]
    tangent_g = sensordata[10:13]

    # 2D friction cone edge vectors in contact frame
    f0_c = np.array([[0],         [0]])
    f1_c = np.array([[Fnormal_c], [-mu_t * Fnormal_c]])
    f2_c = np.array([[Fnormal_c], [ mu_t * Fnormal_c]])

    # Contact frame -> global/body frame  (R_gc == R_bc since body == global)
    R_gc = np.array([normal_g[0:2], tangent_g[0:2]])
    f0_b = R_gc @ f0_c
    f1_b = R_gc @ f1_c
    f2_b = R_gc @ f2_c

    # Contact position relative to object centre (2D, body frame)
    p_gb = np.asarray(obj_pos)[:2]
    p_bc = np.array([[p_gc[0] - p_gb[0]], [p_gc[1] - p_gb[1]]])

    # Wrench [Mz, fx, fy] = [r x F, fx, fy]
    F0_b = np.array([p_bc[0]*f0_b[1] - p_bc[1]*f0_b[0], f0_b[0], f0_b[1]]).reshape(3)
    F1_b = np.array([p_bc[0]*f1_b[1] - p_bc[1]*f1_b[0], f1_b[0], f1_b[1]]).reshape(3)
    F2_b = np.array([p_bc[0]*f2_b[1] - p_bc[1]*f2_b[0], f2_b[0], f2_b[1]]).reshape(3)

    return F1_b, F2_b, F0_b


def check_wrench_feasibility(thumb_wrenches, index_wrenches,
                              target_wrench_fxfyMz: np.ndarray):
    """
    Check whether a target wrench lies inside the Minkowski sum of the two
    contacts' friction-cone wrench sets.

    Uses the same pairwise vertex-sum + ConvexHull approach as
    models/friction_wrench_cone.py.

    Parameters
    ----------
    thumb_wrenches / index_wrenches :
        (F1_b, F2_b, F0_b) returned by calc_wrench -- each [Mz, fx, fy].
    target_wrench_fxfyMz : (3,) array  [fx, fy, Mz]
        e.g. [0, obj_mass*g, 0] to test gravity balance.

    Returns
    -------
    feasible : bool
    quality  : float  (positive = inside hull; larger = deeper inside)
    """
    # Convert calc_wrench output [Mz, fx, fy] -> [fx, fy, Mz] to match
    # the axis convention used in friction_wrench_cone.py
    tw = np.roll(np.array(thumb_wrenches), -1, axis=1)   # (3, 3)
    iw = np.roll(np.array(index_wrenches), -1, axis=1)

    # Minkowski sum vertices: all pairwise sums (same as friction_wrench_cone.py)
    vert_sums = np.array([t + i for t in tw for i in iw])  # 9 x 3

    if len(vert_sums) < 4:
        return False, 0.0
    try:
        hull    = ConvexHull(vert_sums)
        margins = hull.equations[:, :3] @ np.asarray(target_wrench_fxfyMz) \
                  + hull.equations[:, 3]
        return bool(np.all(margins <= 1e-8)), float(-np.max(margins))
    except Exception:
        return False, 0.0


# =============================================================================
# Geometry helpers
# =============================================================================

def _box_surface_normal_2d(point, center, hx: float, hy: float) -> np.ndarray:
    """Outward unit normal at a point on/near a 2D axis-aligned box."""
    d  = np.asarray(point, float) - np.asarray(center, float)
    nx = abs(d[0]) / hx
    ny = abs(d[1]) / hy
    if nx > ny:
        return np.array([np.sign(d[0]), 0.0])
    elif ny > nx:
        return np.array([0.0, np.sign(d[1])])
    n = np.array([np.sign(d[0]), np.sign(d[1])], float)
    return n / np.linalg.norm(n)


# =============================================================================
# Pre-grasp wrench check  (estimated forces from planned contact geometry)
# =============================================================================

def _pre_grasp_wrench_check(p1, p2, obj_pos, obj_hx: float, obj_hy: float,
                             obj_mass: float, mu: float):
    """
    Estimate the minimum normal forces at planned contact points p1, p2 that
    balance gravity, then evaluate whether those forces lie within the friction
    cones via calc_wrench + check_wrench_feasibility.

    The minimum-force estimate assumes zero tangential force (conservative):
        [n1 | n2] * [fn1, fn2]^T ~= [0, m*g]   (2x2 least-squares)

    Returns (feasible: bool, quality: float)
    """
    g  = 9.81
    c  = np.asarray(obj_pos, float)
    n1 = _box_surface_normal_2d(p1, c, obj_hx, obj_hy)
    n2 = _box_surface_normal_2d(p2, c, obj_hx, obj_hy)

    A  = np.column_stack([n1, n2])
    b  = np.array([0.0, obj_mass * g])
    try:
        fn, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return False, 0.0
    # For left-right (antipodal) grasps the contact normals are horizontal, so
    # normal forces alone cannot balance vertical gravity — friction must carry the
    # full load.  Use fn_min = m*g/(2*mu) so the friction cone is always large
    # enough to potentially contain the gravity wrench, instead of flooring at 1e-3
    # (which produces a near-zero cone and always fails the check).
    fn_min = (obj_mass * g) / (2.0 * max(mu, 1e-6))
    fn1 = max(fn[0], fn_min)
    fn2 = max(fn[1], fn_min)

    # Tangent directions (90 deg CCW from outward normal)
    t1 = np.array([-n1[1], n1[0]])
    t2 = np.array([-n2[1], n2[0]])

    def _sd(fn_val, p_xy, normal_2d, tangent_2d):
        return np.array([
            1,
            fn_val, 0.0, 0.0,
            p_xy[0], p_xy[1], 0.0,
            normal_2d[0],  normal_2d[1],  0.0,
            tangent_2d[0], tangent_2d[1], 0.0,
        ])

    tw = calc_wrench(_sd(fn1, p1, n1, t1), mu, obj_pos)
    iw = calc_wrench(_sd(fn2, p2, n2, t2), mu, obj_pos)

    if tw == 0 or iw == 0:
        return False, 0.0

    return check_wrench_feasibility(tw, iw, np.array([0.0, obj_mass * g, 0.0]))


# =============================================================================
# Post-grasp contact monitoring  (actual MuJoCo contact forces)
# =============================================================================

def _build_contact_sensordata(model, data,
                               finger_gid: int, obj_gid: int) -> np.ndarray | None:
    """
    Extract the contact between finger_gid and obj_gid from MuJoCo's contact
    list and pack it into the (13,) sensordata format expected by calc_wrench.

    Returns None if no contact is found.
    """
    for i in range(data.ncon):
        c  = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ((g1 == finger_gid and g2 == obj_gid) or
                (g1 == obj_gid    and g2 == finger_gid)):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        # contact.frame rows: [normal, tangent1, tangent2] in global frame
        return np.array([
            1,
            force[0], force[1], force[2],          # [Fn, Ft1, Ft2] in contact frame
            c.pos[0],  c.pos[1],  c.pos[2],        # contact pos global
            c.frame[0], c.frame[1], c.frame[2],    # normal global
            c.frame[3], c.frame[4], c.frame[5],    # tangent global
        ])
    return None


def _post_grasp_wrench_check(model, data,
                              thumb_gid: int, index_gid: int, obj_gid: int,
                              obj_bid: int, obj_mass: float, mu: float):
    """
    Check post-grasp stability using actual MuJoCo contact forces via
    calc_wrench + check_wrench_feasibility.

    Returns (stable: bool, reason: str)
    """
    obj_pos = data.xpos[obj_bid].copy()

    sd_thumb = _build_contact_sensordata(model, data, thumb_gid, obj_gid)
    sd_index = _build_contact_sensordata(model, data, index_gid, obj_gid)

    if sd_thumb is None:
        return False, "thumb contact lost"
    if sd_index is None:
        return False, "index contact lost"

    tw = calc_wrench(sd_thumb, mu, obj_pos)
    iw = calc_wrench(sd_index, mu, obj_pos)

    if tw == 0 or iw == 0:
        return False, "contact force is zero"

    feasible, quality = check_wrench_feasibility(
        tw, iw, np.array([0.0, obj_mass * 9.81, 0.0]))
    reason = "stable" if feasible else \
             f"gravity wrench outside friction cone (margin={quality:.4f})"
    return feasible, reason


# =============================================================================
# Seeded grasp planner  (subclass of GraspPlanner)
# =============================================================================

class _SeededGraspPlanner(GraspPlanner):
    """
    Subclass of GraspPlanner that accepts explicit contact-point warm-starts
    (p1_init, p2_init) in solve().

    Only the two warm-start lines differ from the parent; everything else —
    the optimisation problem, constraints, IPOPT options, and return format —
    is identical.
    """

    def solve(self,
              q_ref:    np.ndarray,
              obj_pos:  np.ndarray,
              p1_init:  np.ndarray | None = None,
              p2_init:  np.ndarray | None = None) -> dict:
        """
        Run the grasp optimisation seeded from (p1_init, p2_init).

        If p1_init / p2_init are None the parent's default warm-start is used
        (left and right face centres), making this a drop-in replacement.
        """
        cfg        = self.cfg
        model      = self.model
        act_idx    = self._act_idx
        obj_center = np.asarray(obj_pos, dtype=float)

        # Isolated MjData — solver never touches the live simulation copy
        data_cb = mujoco.MjData(model)
        data_cb.qpos[:] = self.data.qpos[:]
        data_cb.qvel[:] = self.data.qvel[:]
        mujoco.mj_forward(model, data_cb)

        # Callbacks (identical to GraspPlanner.solve)
        ik_thumb_cb = _FingerPointDistCallback(
            'gp_ik_thumb', model, data_cb,
            self._thumb_gid, self._cp1_gid, self._cp1_mocap, act_idx)
        ik_index_cb = _FingerPointDistCallback(
            'gp_ik_index', model, data_cb,
            self._index_gid, self._cp2_gid, self._cp2_mocap, act_idx)
        sdf_cb = _SDFCallback(
            'gp_sdf', model, data_cb, self._obj_gid, act_idx)
        nonpen_thumb_cb = _NonPenetrationCallback(
            'gp_np_thumb', model, data_cb,
            self._thumb_gid, self._obj_gid, act_idx)
        nonpen_index_cb = _NonPenetrationCallback(
            'gp_np_index', model, data_cb,
            self._index_gid, self._obj_gid, act_idx)

        # Optimisation problem (identical to GraspPlanner.solve)
        opti = ca.Opti()
        q  = opti.variable(model.nu)
        p1 = opti.variable(2)
        p2 = opti.variable(2)

        d_thumb   = ik_thumb_cb(q, p1)
        d_index   = ik_index_cb(q, p2)
        sdf1      = sdf_cb(q, p1)
        sdf2      = sdf_cb(q, p2)
        gap_thumb = nonpen_thumb_cb(q)
        gap_index = nonpen_index_cb(q)

        cost = cfg.w_ik * (d_thumb**2 + d_index**2)
        if not cfg.on_object:
            cost += cfg.w_surface * ((sdf1 / cfg.sdf_scale)**2 +
                                     (sdf2 / cfg.sdf_scale)**2)
        d1 = p1 - obj_center
        d2 = p2 - obj_center
        cost += cfg.w_antipodal * ca.fmax(0, ca.dot(d1, d2))**2
        cost += cfg.w_reg       * ca.sumsqr((q - q_ref) / cfg.q_scale)
        cost += cfg.w_proximity * (ca.sumsqr((p1 - obj_center) / cfg.p_scale) +
                                   ca.sumsqr((p2 - obj_center) / cfg.p_scale))
        opti.minimize(cost)

        if cfg.joint_limits:
            for i in range(model.nu):
                jid = model.actuator_trnid[i, 0]
                if model.jnt_limited[jid]:
                    opti.subject_to(opti.bounded(
                        model.jnt_range[jid, 0], q[i], model.jnt_range[jid, 1]))
        if cfg.on_object:
            # Place the cp1/cp2 dummy-sphere centres 2*r_cp inside the box surface
            # (r_cp == 0.0001 m from XML).  Then d_finger==0 puts the fingertip ON
            # the box surface and gap == -1e-4 (the penetration constraint) is
            # geometrically consistent: all three constraints can be satisfied
            # simultaneously.  With sdf==0 the cp sphere extends 0.1 mm outside
            # the box, making gap==-1e-4 require the finger to be 0.2 mm inside
            # — an irreconcilable 0.2 mm inconsistency that prevents convergence.
            _r_cp = 1e-4   # cp1/cp2 geom radius (matches XML size="0.0001")
            opti.subject_to(sdf1 == -2 * _r_cp)
            opti.subject_to(sdf2 == -2 * _r_cp)
        if cfg.ik_constraint:
            opti.subject_to(d_thumb == 0)
            opti.subject_to(d_index == 0)
        if cfg.penetration_constraint:
            opti.subject_to(gap_thumb == -1e-4)
            opti.subject_to(gap_index == -1e-4)

        # ── warm-start: use provided seeds, fall back to parent defaults ──────
        p1_seed = (np.asarray(p1_init, float) if p1_init is not None
                   else obj_center + np.array([-self._obj_hx, 0.0]))
        p2_seed = (np.asarray(p2_init, float) if p2_init is not None
                   else obj_center + np.array([+self._obj_hx, 0.0]))
        opti.set_initial(q,  q_ref)
        opti.set_initial(p1, p1_seed)
        opti.set_initial(p2, p2_seed)

        ipopt_opts: dict = {
            'jacobian_approximation': 'finite-difference-values',
            'hessian_approximation':  'limited-memory',
            'max_iter':               cfg.max_iter,
            'sb':                     'no',
            'tol':                    1e-6,
            'dual_inf_tol':           1.0,
            'constr_viol_tol':        1e-8,
            'print_level':            0,
        }
        opti.solver('ipopt', {'ipopt': ipopt_opts, 'print_time': False})

        try:
            sol = opti.solve()
            return {
                'success':    True,
                'q':          sol.value(q),
                'p1':         sol.value(p1),
                'p2':         sol.value(p2),
                'cost':       float(sol.value(opti.f)),
                'iterations': sol.stats()['iter_count'],
                'status':     'converged',
            }
        except Exception as e:
            self.log.warning(f"_SeededGraspPlanner.solve: {e}")
            try:
                return {
                    'success':    False,
                    'q':          opti.debug.value(q),
                    'p1':         opti.debug.value(p1),
                    'p2':         opti.debug.value(p2),
                    'cost':       None,
                    'iterations': None,
                    'status':     'best-effort',
                }
            except Exception as e2:
                self.log.error(f"_SeededGraspPlanner.solve debug extraction: {e2}")
                return {
                    'success': False, 'q': None, 'p1': None, 'p2': None,
                    'cost': None, 'iterations': None, 'status': 'failed',
                }


# =============================================================================
# Multi-start grasp planner
# =============================================================================

class MultiStartGraspPlanner:
    """
    Enumerates diverse grasp candidates by seeding contact points from a
    structured grid over face pairs and contact heights, rather than random
    perturbation of joint angles.

    For a box object the set of face pairs is:
        Left-Right, Top-Left, Top-Right, Bottom-Left, Bottom-Right, Top-Bottom

    n_seeds (default 16) seeds are distributed evenly across the 6 face pairs
    (~2-3 per pair).  Within each pair, positions are sampled at the same
    fractional height on both faces simultaneously (paired, not Cartesian
    product), so the total stays exactly n_seeds.  This gives both inter-pair
    diversity (different face combinations) and intra-pair diversity (different
    contact heights within a face pair).

    Results are deduplicated (contact points closer than 5 mm are merged) and
    ranked: convergence -> wrench quality (descending) within feasible grasps
    first, then infeasible grasps by cost.
    """

    # Face definitions: (normal direction, offset sign, free axis)
    # Each entry: (face_label, p_centre_offset, free_axis_index, free_axis_half)
    # p_centre_offset is the vector from object centre to face centre.
    # free_axis_index is the axis along which contact slides (0=x, 1=y).
    _FACES = {
        "left":   ( np.array([-1.0,  0.0]), 0, 1),  # x=-hx, slides in y
        "right":  ( np.array([ 1.0,  0.0]), 0, 1),  # x=+hx, slides in y
        "top":    ( np.array([ 0.0,  1.0]), 1, 0),  # y=+hy, slides in x
        "bottom": ( np.array([ 0.0, -1.0]), 1, 0),  # y=-hy, slides in x
    }

    # Face pairs that produce meaningful antipodal or stable grasps
    _FACE_PAIRS = [
        ("left",   "right"),
        ("top",    "left"),
        ("top",    "right"),
        ("bottom", "left"),
        ("bottom", "right"),
        ("top",    "bottom"),
    ]

    def __init__(self, model, data, cfg: TAMPConfig):
        self.model   = model
        self.data    = data
        self.cfg     = cfg
        self._gcfg   = GraspConfig(max_iter=cfg.ms_max_iter)
        obj_gid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj1_geom")
        self._obj_hx = model.geom_size[obj_gid][0]
        self._obj_hy = model.geom_size[obj_gid][1]

    def plan(self, q_ref: np.ndarray, obj_pos: np.ndarray) -> list[dict]:
        seeds   = self._enumerate_seeds(obj_pos)
        n_total = len(seeds)

        # Snapshot so all solves start from the same simulation state
        data_snap = mujoco.MjData(self.model)
        data_snap.qpos[:] = self.data.qpos[:]
        data_snap.qvel[:] = self.data.qvel[:]
        mujoco.mj_forward(self.model, data_snap)

        planner = _SeededGraspPlanner(self.model, data_snap, cfg=self._gcfg)
        results = []

        for p1_init, p2_init, label in seeds:
            result = planner.solve(q_ref, obj_pos, p1_init=p1_init, p2_init=p2_init)
            if result["q"] is None:
                continue

            feasible, quality = _pre_grasp_wrench_check(
                result["p1"], result["p2"],
                obj_pos, self._obj_hx, self._obj_hy,
                self.cfg.obj_mass, self.cfg.mu,
            )
            result["wrench_feasible"] = feasible
            result["wrench_quality"]  = quality
            result["seed_label"]      = label
            results.append(result)

        results = self._deduplicate(results)
        results.sort(key=self._rank_key)

        n_ok = sum(r["wrench_feasible"] for r in results)
        log.info(f"[GraspPlanner] {len(results)}/{n_total} seeds produced valid results "
                 f"| {n_ok} wrench-feasible")
        return results

    def _enumerate_seeds(self, obj_pos: np.ndarray) -> list[tuple]:
        """
        Distribute n_seeds evenly across face pairs using paired (not Cartesian)
        sampling so the total seed count stays at exactly n_seeds.

        n_per_pair = n_seeds // n_pairs; the first (n_seeds % n_pairs) face pairs
        get one extra seed.  Within each pair, both faces are sampled at the same
        fractional positions along their respective free axes.
        """
        c            = np.asarray(obj_pos, float)
        hx, hy       = self._obj_hx, self._obj_hy
        half_extents = {"left": hy, "right": hy, "top": hx, "bottom": hx}
        n_pairs      = len(self._FACE_PAIRS)
        n_total      = self.cfg.n_seeds
        base, extra  = divmod(n_total, n_pairs)   # seeds per pair

        def _face_points(face_name: str, k: int) -> list[np.ndarray]:
            """k evenly-spaced points along face_name, avoiding corners (80%)."""
            normal, fixed_axis, free_axis = self._FACES[face_name]
            half    = half_extents[face_name]
            offsets = np.linspace(-0.8 * half, 0.8 * half, k)
            pts = []
            for off in offsets:
                p = c.copy()
                p[fixed_axis] += normal[fixed_axis] * (hx if fixed_axis == 0 else hy)
                p[free_axis]  += off
                pts.append(p)
            return pts

        seeds = []
        for i, (f1, f2) in enumerate(self._FACE_PAIRS):
            k = base + (1 if i < extra else 0)   # seeds for this pair
            if k == 0:
                continue
            pts1 = _face_points(f1, k)
            pts2 = _face_points(f2, k)
            for p1, p2 in zip(pts1, pts2):        # paired, not Cartesian
                seeds.append((p1, p2, f"{f1}-{f2}"))
        return seeds

    @staticmethod
    def _deduplicate(results: list[dict], tol: float = 5e-3) -> list[dict]:
        """
        Remove results whose contact points (p1, p2) are within tol metres of
        an already-kept result.  Keeps the first occurrence (highest-ranked
        after sort by wrench_quality so we call this before the final sort).
        """
        kept = []
        for r in results:
            duplicate = False
            for k in kept:
                d1 = np.linalg.norm(r["p1"] - k["p1"])
                d2 = np.linalg.norm(r["p2"] - k["p2"])
                if d1 < tol and d2 < tol:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(r)
        return kept

    @staticmethod
    def _rank_key(r):
        # Primary: converged before best-effort
        # Secondary: wrench-feasible first, ranked by quality (higher = better)
        # Tertiary: optimisation cost (lower = better)
        status_rank  = {"converged": 0, "best-effort": 1, "failed": 2}
        wrench_rank  = 0 if r.get("wrench_feasible") else 1
        quality_rank = -r.get("wrench_quality", 0.0)   # negate: higher quality first
        cost_rank    = r.get("cost") or 1e9
        return (status_rank.get(r["status"], 2), wrench_rank, quality_rank, cost_rank)


# =============================================================================
# Hand gesture detector
# =============================================================================

class HandGestureDetector:
    """
    Detects open-hand (release) gesture from MediaPipe finger joint angles.
    Open = rolling mean of |finger angles| drops below open_threshold (rad).
    Minor curls are smoothed away by the rolling window.
    """

    def __init__(self, open_threshold: float = 0.25, window: int = 15):
        self.open_threshold = open_threshold
        self._buf = deque(maxlen=window)

    def update(self, finger_angles: np.ndarray) -> dict:
        self._buf.append(float(np.mean(np.abs(finger_angles))))
        smoothed = float(np.mean(self._buf))
        return {"openness": smoothed, "is_open": smoothed < self.open_threshold}

    def reset(self):
        self._buf.clear()


# =============================================================================
# TAMP Manager
# =============================================================================

class TAMPManager:
    """
    Orchestrates the full TAMP pipeline.
    Call update() every sim tick; register key_callback with the MuJoCo viewer.
    """

    # Actuator indices (must match XML actuator order)
    _BASE_X  = 0;  _BASE_Y  = 1
    _IDX_MCP = 2;  _IDX_PIP = 3;  _IDX_DIP = 4
    _TH_MCP  = 5;  _TH_PIP  = 6;  _TH_DIP  = 7

    def __init__(self, cfg: TAMPConfig | None = None):
        self.cfg = cfg or TAMPConfig()

        # ── MuJoCo ───────────────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data  = mujoco.MjData(self.model)
        # Start from zero pose (fingers straight). ROM calibration (open + fist
        # poses) maps the operator's full range onto the robot's joint limits, so
        # the starting keyframe does not matter for finger control.
        mujoco.mj_forward(self.model, self.data)

        _gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        _bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        self._thumb_gid = _gid("right_thumb_distal_geom")
        self._index_gid = _gid("right_index_distal_geom")
        self._obj_gid   = _gid("obj1_geom")
        self._obj_bid   = _bid("obj1")
        self._cp1_bid   = _bid("cp1_body")
        self._cp2_bid   = _bid("cp2_body")

        # Use the contact_p1 / contact_p2 SITES (1 cm radius, visible) for
        # visualisation.  The cp1/cp2 geoms (radius=0.0001 m) remain unchanged
        # because they are IK targets inside _FingerPointDistCallback — enlarging
        # them would move the finger 2 cm away from the intended contact point.
        _sid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
        self._cp1_site = _sid("contact_p1")   # red   sphere, thumb contact
        self._cp2_site = _sid("contact_p2")   # green sphere, index contact

        self._act_idx   = [
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ]

        # ── Sub-systems ───────────────────────────────────────────────────────
        self._approach = ApproachController(
            self.model, self.cfg.approach_clearance, self.cfg.approach_max_iter)
        self._ms_plan  = MultiStartGraspPlanner(self.model, self.data, self.cfg)
        self._gesture  = HandGestureDetector(
            self.cfg.hand_open_threshold, self.cfg.hand_open_window)

        # ── State ─────────────────────────────────────────────────────────────
        self._state      = TAMPState.IDLE
        self._state_lock = threading.Lock()

        self._current_wrist: np.ndarray | None = None
        self._wrist_offset:  np.ndarray | None = None
        self._raw_msg:       list        | None = None
        self._last_calib_raw: object            = None   # dedup for calib capture

        # Last MediaPipe → robot angle snapshot (set in _build_q_teleop)
        self._last_human_angles: np.ndarray | None = None
        self._last_robot_angles: np.ndarray | None = None

        # CSV angle log
        self._angle_log_file   = None
        self._angle_log_writer = None
        if self.cfg.angle_log_path:
            self._angle_log_file = open(self.cfg.angle_log_path, "w", newline="")
            self._angle_log_writer = csv.writer(self._angle_log_file)
            self._angle_log_writer.writerow([
                "time_s", "state",
                "h_idx_mcp_deg", "h_idx_pip_deg", "h_idx_dip_deg",
                "h_th_spr_deg",  "h_th_ip_deg",   "h_th_ip2_deg",
                "r_idx_mcp_deg", "r_idx_pip_deg", "r_idx_dip_deg",
                "r_th_mcp_deg",  "r_th_pip_deg",  "r_th_dip_deg",
                "ctrl_base_x",   "ctrl_base_y",
                "ctrl_idx_mcp",  "ctrl_idx_pip",  "ctrl_idx_dip",
                "ctrl_th_mcp",   "ctrl_th_pip",   "ctrl_th_dip",
            ])
            log.info(f"[TAMP] Logging angles to: {self.cfg.angle_log_path}")

        # ROM calibration state: MediaPipe angles captured at open-hand / fist poses
        self._open_angles:       np.ndarray = np.zeros(6)
        self._fist_angles:       np.ndarray = np.ones(6)   # non-zero avoids div/0 before calib
        self._robot_fist_angles: np.ndarray = self._compute_robot_fist_angles()
        # calib_phase: 0=prompt_open, 1=capture_open, 2=prompt_fist, 3=capture_fist
        self._calib_phase: int   = 0
        self._calib_buf:   list  = []
        self._calib_start: float = 0.0

        # EMA state for teleop smoothing
        self._q_teleop_prev: np.ndarray | None = None

        self._candidates:      list[dict]             = []
        self._candidate_idx:   int                    = 0
        self._planning_thread: threading.Thread | None = None
        self._last_replan_time:    float              = -999.0  # wall-clock seconds
        self._last_instruction_t:  float              = -999.0  # throttle for Y-prompt

        self._grasp_q:  np.ndarray | None = None
        self._grasp_p1: np.ndarray | None = None
        self._grasp_p2: np.ndarray | None = None

        self._interp_start: np.ndarray | None = None
        self._interp_step:  int = 0

        self._sim_step = 0
        self._opt_tick = 0
        self._transporting_entry_step: int = 0

        # ── ROS 2 (optional) ─────────────────────────────────────────────────
        self._ros_node = None
        if _HAS_ROS:
            self._init_ros()
        else:
            log.warning("[TAMP] ROS 2 not available -- keyboard-only mode.")

        log.info(
            f"\nTAMP Manager ready  \n"
            f"  Config: n_seeds={self.cfg.n_seeds}  replan_interval={self.cfg.replan_interval_sec}s"
            f"  proximity_threshold={self.cfg.proximity_threshold}m\n"
            f"          mu={self.cfg.mu}  obj_mass={self.cfg.obj_mass}kg"
            f"  approach_clearance={self.cfg.approach_clearance}m\n"
            f"  Grasp planner reruns every {self.cfg.replan_interval_sec}s in background\n"
            f"  C         calibrate  ({'wrist-only (simple)' if self.cfg.simple_calib else 'full ROM: spread -> C, pinch -> C'})\n"
            "  D         toggle debug angle logging (human° -> robot°)\n"
            "  J / K     next / previous candidate        (APPROACH)\n"
            "  Y         accept current candidate         (APPROACH)\n"
            "  N         discard candidates + replan      (APPROACH)\n"
            "  O         manually trigger place           (TRANSPORTING)\n"
            "  R         reset to APPROACH\n"
        )

    # ── Robot joint utilities ────────────────────────────────────────────────

    def _compute_robot_fist_angles(self) -> np.ndarray:
        """Return the joint limit (in flexion direction) for each of the 6 finger actuators.

        Convention: actuators 2-7 are [IDX_MCP, IDX_PIP, IDX_DIP, TH_MCP, TH_PIP, TH_DIP].
        The fist direction is whichever limit has the larger absolute value.

        Falls back to physical defaults when joint limits are not set in the XML
        (jnt_limited == 0 or both limits are 0).  Sign convention follows
        teleop_manager.py: index joints flex negative, thumb joints flex positive.
        """
        # Defaults: index curls negative, thumb curls positive (~90° travel each)
        _DEFAULT = np.array([-1.5, -1.5, -1.5, 1.5, 1.5, 0.75])
        angles = np.zeros(6)
        for local_i, act_i in enumerate(range(2, 8)):
            jid = self.model.actuator_trnid[act_i, 0]
            lo, hi = self.model.jnt_range[jid]
            if not self.model.jnt_limited[jid] or (lo == 0.0 and hi == 0.0):
                angles[local_i] = _DEFAULT[local_i]
            else:
                angles[local_i] = lo if abs(lo) >= abs(hi) else hi
        log.info(f"[TAMP] Robot fist angles: {np.round(np.degrees(angles), 1)} deg "
                 f"(from model limits where set, else defaults)")
        return angles

    # ── State machine ─────────────────────────────────────────────────────────

    @property
    def state(self) -> TAMPState:
        with self._state_lock:
            return self._state

    def _transition(self, new: TAMPState) -> None:
        with self._state_lock:
            old, self._state = self._state, new
        log.info(f"[TAMP] {old.value} -> {new.value}")

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self) -> None:
        """Call once per sim tick from the viewer loop."""
        mujoco.mj_step(self.model, self.data)
        self._sim_step += 1
        s = self.state

        if   s == TAMPState.CALIBRATING:  self._step_calibrating()
        elif s == TAMPState.APPROACH:     self._step_approach()
        elif s == TAMPState.EXECUTING:    self._step_executing()
        elif s == TAMPState.TRANSPORTING: self._step_transporting()
        elif s == TAMPState.PLACING:      self._step_placing()

    # ── APPROACH ──────────────────────────────────────────────────────────────

    def _step_approach(self) -> None:
        if self._wrist_offset is None or self._current_wrist is None:
            return

        # Collision-free teleop
        target_base = self._camera_to_robot(self._current_wrist) + self._wrist_offset
        q_teleop    = self._build_q_teleop(target_base)

        self._opt_tick += 1
        if self._opt_tick % self.cfg.approach_ik_freq == 0:
            self._approach.sync_state(self.data)
            q_safe = self._approach.get_collision_free_joint_angles(q_teleop)
        else:
            q_safe = q_teleop

        for i, idx in enumerate(self._act_idx):
            self.data.qpos[idx] = q_safe[i]
        self.data.ctrl[:] = q_safe
        mujoco.mj_forward(self.model, self.data)

        if self._sim_step % self.cfg.log_freq == 0:
            self._write_angle_log(q_safe)

        # Continuous background replanning — only when close enough to the object
        now  = time.monotonic()
        dist = self._approach.finger_object_distance(self.data)
        if (dist < self.cfg.proximity_threshold
                and now - self._last_replan_time >= self.cfg.replan_interval_sec
                and not (self._planning_thread and self._planning_thread.is_alive())):
            log.info(f"[TAMP] Within proximity ({dist:.3f} m) -- replanning.")
            self._start_replan()

        # Overlay current candidate's contact markers (robot keeps moving freely)
        self._update_candidate_markers()

        # Periodically remind the operator how to proceed
        if self._candidates:
            now_t = time.monotonic()
            if now_t - self._last_instruction_t >= 5.0:
                self._last_instruction_t = now_t
                self._print_accept_instruction()

    def _print_accept_instruction(self) -> None:
        log.info(
            "┌─────────────────────────────────────────────────────┐\n"
            "│  To accept current grasp, press Y                   │\n"
            "│  Keep moving toward object to replan grasp          │\n"
            "│  J / K  browse candidates  |  N  discard & replan   │\n"
            "└─────────────────────────────────────────────────────┘"
        )

    def _write_angle_log(self, q_ctrl: np.ndarray) -> None:
        """Write one row to the angle CSV log (called from _step_approach)."""
        if self._angle_log_writer is None or self._last_human_angles is None:
            return
        hd = np.degrees(self._last_human_angles)
        rd = np.degrees(self._last_robot_angles)
        row = (
            [f"{time.time():.4f}", self.state.value]
            + [f"{v:.2f}" for v in hd]
            + [f"{v:.2f}" for v in rd]
            + [f"{v:.5f}" for v in q_ctrl]
        )
        self._angle_log_writer.writerow(row)
        self._angle_log_file.flush()

    # ── CALIBRATING ───────────────────────────────────────────────────────────

    def _step_calibrating(self) -> None:
        """ROM calibration: accumulates MediaPipe angles during timed capture windows."""
        if self._calib_phase in (0, 2):
            return  # waiting for operator C-press — nothing to do each sim step

        now = time.monotonic()
        elapsed = now - self._calib_start

        # Accumulate new MediaPipe samples (deduplicate by object identity)
        if (self._raw_msg is not None
                and self._raw_msg is not self._last_calib_raw):
            self._calib_buf.append(self._extract_finger_angles(self._raw_msg))
            self._last_calib_raw = self._raw_msg

        # Progress indicator every 5 new samples (~every 0.5s at 30fps camera)
        if len(self._calib_buf) > 0 and len(self._calib_buf) % 5 == 0:
            remaining = max(0.0, self.cfg.calib_duration_sec - elapsed)
            log.info(f"[TAMP/CALIB] Capturing... {remaining:.1f}s remaining "
                     f"({len(self._calib_buf)} samples)")

        if elapsed < self.cfg.calib_duration_sec:
            return  # still capturing

        # Capture complete
        if not self._calib_buf:
            log.warning("[TAMP/CALIB] No MediaPipe samples received — check ROS topic.")
            self._calib_phase = 0 if self._calib_phase == 1 else 2
            return

        angles = np.mean(self._calib_buf, axis=0)
        self._calib_buf.clear()
        self._last_calib_raw = None

        if self._calib_phase == 1:  # open-hand capture just finished
            log.info(f"[TAMP/CALIB] Spread-open captured: "
                     f"{np.round(np.degrees(angles), 1)} deg")
            # Wrist offset is recorded in both modes
            if self._current_wrist is not None:
                robot_wrist = np.array([
                    self.data.qpos[self._act_idx[self._BASE_X]],
                    self.data.qpos[self._act_idx[self._BASE_Y]],
                    0.0,
                ])
                self._wrist_offset = robot_wrist - self._camera_to_robot(self._current_wrist)
                log.info(f"[TAMP/CALIB] Wrist offset: {np.round(self._wrist_offset, 4)}")

            if self.cfg.simple_calib:
                # Fixed ROM mapping: synthesise open/fist angles from config params
                # so _scale_to_robot() works without a second capture press.
                offset = np.radians(self.cfg.calib_open_offset_deg)
                span   = np.radians(self.cfg.calib_human_max_deg)
                self._open_angles = np.full(6, offset)
                self._fist_angles = np.full(6, offset + span)
                self._q_teleop_prev = np.array([self.data.qpos[i] for i in self._act_idx])
                self._calib_phase = 0
                log.info(
                    f"[TAMP/CALIB] Simple calibration complete "
                    f"(open_offset={self.cfg.calib_open_offset_deg}°, "
                    f"human_max={self.cfg.calib_human_max_deg}°) -> APPROACH"
                )
                self._transition(TAMPState.APPROACH)
            else:
                self._open_angles = angles
                self._calib_phase = 2
                log.info("[TAMP/CALIB] *** Now PINCH index + thumb together and press C ***")

        elif self._calib_phase == 3:  # fist capture just finished
            self._fist_angles = angles
            rom = np.degrees(self._fist_angles - self._open_angles)
            robot_fist = np.degrees(self._robot_fist_angles)
            log.info(f"[TAMP/CALIB] Pinch captured:  {np.round(np.degrees(angles), 1)} deg")
            log.info(f"[TAMP/CALIB] Human ROM:      {np.round(rom, 1)} deg")
            log.info(f"[TAMP/CALIB] Robot range:    {np.round(robot_fist, 1)} deg")
            # Seed EMA with current robot state so there's no jump on first teleop frame
            self._q_teleop_prev = np.array([self.data.qpos[i] for i in self._act_idx])
            self._calib_phase = 0
            log.info("[TAMP/CALIB] Calibration complete -> APPROACH")
            self._transition(TAMPState.APPROACH)

    def _start_replan(self) -> None:
        """Launch a background replan from the current robot/object state."""
        self._last_replan_time = time.monotonic()
        q_snap  = np.array([self.data.qpos[i] for i in self._act_idx])
        obj_pos = self.data.xpos[self._obj_bid][:2].copy()

        def _run():
            log.info(f"[GraspPlanner] Replanning ({self.cfg.n_seeds} seeds) ...")
            new_candidates = self._ms_plan.plan(q_snap, obj_pos)
            if new_candidates:
                self._candidates    = new_candidates
                self._candidate_idx = 0
                self._log_candidate(0)
            else:
                log.warning("[GraspPlanner] Replan produced no valid candidates.")

        self._planning_thread = threading.Thread(target=_run, daemon=True)
        self._planning_thread.start()

    def _update_candidate_markers(self) -> None:
        """Move contact-point markers to show the current candidate.

        The visible contact_p1/contact_p2 sites (1 cm radius spheres) are
        repositioned via model.site_pos — these are in the worldbody so their
        model-frame position equals their world position.

        The tiny cp1/cp2 mocap geoms (0.1 mm radius) are also kept in sync
        because _FingerPointDistCallback uses them as IK targets; their size
        must stay at 0.0001 m so the IK places the fingertip accurately.
        """
        if not self._candidates:
            return
        r = self._candidates[self._candidate_idx]
        if r.get("p1") is None or r.get("p2") is None:
            return

        p1 = [r["p1"][0], r["p1"][1], 0.0]
        p2 = [r["p2"][0], r["p2"][1], 0.0]

        # Visible sites — updated in model so mj_forward propagates them
        self.model.site_pos[self._cp1_site] = p1
        self.model.site_pos[self._cp2_site] = p2

        # Tiny mocap geoms — kept in sync for IK callback use
        cp1_mid = self.model.body_mocapid[self._cp1_bid]
        cp2_mid = self.model.body_mocapid[self._cp2_bid]
        self.data.mocap_pos[cp1_mid] = p1
        self.data.mocap_pos[cp2_mid] = p2

    def _next_candidate(self) -> None:
        self._candidate_idx = min(self._candidate_idx + 1, len(self._candidates) - 1)
        self._log_candidate(self._candidate_idx)

    def _prev_candidate(self) -> None:
        self._candidate_idx = max(self._candidate_idx - 1, 0)
        self._log_candidate(self._candidate_idx)

    def _log_candidate(self, idx: int) -> None:
        r = self._candidates[idx]
        log.info(
            f"[TAMP] Candidate {idx+1}/{len(self._candidates)}  "
            f"status={r['status']}  wrench_ok={r['wrench_feasible']}  "
            f"quality={r['wrench_quality']:.3f}  cost={r.get('cost')}"
        )
        self._print_accept_instruction()

    def _accept_candidate(self) -> None:
        r = self._candidates[self._candidate_idx]
        if r.get("q") is None:
            log.warning("[TAMP] Candidate has no valid q.")
            return
        self._grasp_q  = r["q"].copy()
        self._grasp_p1 = r["p1"].copy()
        self._grasp_p2 = r["p2"].copy()
        self._interp_start = np.array([self.data.qpos[i] for i in self._act_idx])
        self._interp_step  = 0
        self._gesture.reset()
        log.info(f"[TAMP] Candidate accepted  "
                 f"p1={np.round(self._grasp_p1,4)}  p2={np.round(self._grasp_p2,4)}")
        self._transition(TAMPState.EXECUTING)

    def _reject_candidates(self) -> None:
        log.info("[TAMP] Candidates discarded; replanner will refresh shortly.")
        self._candidates = []
        self._candidate_idx = 0
        self._last_replan_time = -999.0   # force an immediate replan next tick

    # ── EXECUTING ─────────────────────────────────────────────────────────────

    def _step_executing(self) -> None:
        t = min(self._interp_step / self.cfg.grasp_interp_steps, 1.0)
        q = (1.0 - t) * self._interp_start + t * self._grasp_q
        for i, idx in enumerate(self._act_idx):
            self.data.qpos[idx] = q[i]
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)
        self._interp_step += 1
        if self._interp_step >= self.cfg.grasp_interp_steps:
            obj_pos = self.data.xpos[self._obj_bid][:2].copy()
            _fromto = np.zeros(6)
            d_thumb = mujoco.mj_geomDistance(
                self.model, self.data, self._thumb_gid, self._obj_gid, 1.0, _fromto)
            d_index = mujoco.mj_geomDistance(
                self.model, self.data, self._index_gid, self._obj_gid, 1.0, _fromto)
            log.info(
                f"[TAMP] Grasp config reached"
                f"  obj_pos={np.round(obj_pos, 4)}"
                f"  plan_p1={np.round(self._grasp_p1, 4)}"
                f"  d_thumb={d_thumb*1e3:.2f}mm  d_index={d_index*1e3:.2f}mm"
            )
            # Contact quality gate: reject if either fingertip is too far from
            # the object surface (IK didn't converge to a physical contact).
            gate = self.cfg.contact_gate_mm * 1e-3
            if d_thumb > gate or d_index > gate:
                log.warning(
                    f"[TAMP] Grasp rejected: fingertip(s) not in contact "
                    f"(d_thumb={d_thumb*1e3:.2f}mm, d_index={d_index*1e3:.2f}mm "
                    f"> gate={self.cfg.contact_gate_mm:.1f}mm) — replanning."
                )
                self._grasp_q = self._grasp_p1 = self._grasp_p2 = None
                self._reject_candidates()
                return
            self._transporting_entry_step = self._sim_step
            self._transition(TAMPState.TRANSPORTING)

    # ── TRANSPORTING ─────────────────────────────────────────────────────────

    def _step_transporting(self) -> None:
        # Lock fingers; only base follows teleop
        q = self._grasp_q.copy()
        if self._wrist_offset is not None and self._current_wrist is not None:
            base = self._camera_to_robot(self._current_wrist) + self._wrist_offset
            q[self._BASE_X] = base[0]
            q[self._BASE_Y] = base[1]
        for i, idx in enumerate(self._act_idx):
            self.data.qpos[idx] = q[i]
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

        # Post-grasp wrench monitoring via actual contact forces + calc_wrench.
        # Skip the first 100 steps after entering TRANSPORTING: contact forces need
        # a few physics steps to settle from the kinematic (mj_forward) end-state.
        steps_in_transport = self._sim_step - self._transporting_entry_step
        if (steps_in_transport > 100
                and self._sim_step % self.cfg.wrench_monitor_freq == 0):
            stable, reason = _post_grasp_wrench_check(
                self.model, self.data,
                self._thumb_gid, self._index_gid, self._obj_gid,
                self._obj_bid, self.cfg.obj_mass, self.cfg.mu,
            )
            if not stable:
                obj_pos = self.data.xpos[self._obj_bid][:2].copy()
                _fromto = np.zeros(6)
                d_thumb = mujoco.mj_geomDistance(
                    self.model, self.data, self._thumb_gid, self._obj_gid, 1.0, _fromto)
                d_index = mujoco.mj_geomDistance(
                    self.model, self.data, self._index_gid, self._obj_gid, 1.0, _fromto)
                log.warning(
                    f"[TAMP] Grasp instability: {reason}"
                    f"  obj_pos={np.round(obj_pos, 4)}"
                    f"  d_thumb={d_thumb*1e3:.2f}mm  d_index={d_index*1e3:.2f}mm"
                    f"  ncon={self.data.ncon}"
                )

        # Hand-open detection -> PLACING
        # Use robot-space scaled angles: open hand = 0, so threshold comparison works
        # the same way regardless of the human's anatomical range.
        if self._raw_msg is not None:
            human  = self._extract_finger_angles(self._raw_msg)
            robot  = self._scale_to_robot(human)
            gesture = self._gesture.update(robot)
            if gesture["is_open"]:
                log.info(
                    f"[TAMP] Hand open (openness={gesture['openness']:.3f}) -> PLACING")
                self._begin_placing()

    # ── PLACING ───────────────────────────────────────────────────────────────

    def _begin_placing(self) -> None:
        self._interp_start = np.array([self.data.qpos[i] for i in self._act_idx])
        self._interp_step  = 0
        self._transition(TAMPState.PLACING)

    def _step_placing(self) -> None:
        t = min(self._interp_step / self.cfg.grasp_interp_steps, 1.0)
        q_open = self._interp_start.copy()
        q_open[self._IDX_MCP:] = 0.0       # open all 6 finger joints to zero
        q = (1.0 - t) * self._interp_start + t * q_open
        for i, idx in enumerate(self._act_idx):
            self.data.qpos[idx] = q[i]
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)
        self._interp_step += 1
        if self._interp_step >= self.cfg.grasp_interp_steps:
            log.info("[TAMP] Object released -> APPROACH")
            self._grasp_q = self._grasp_p1 = self._grasp_p2 = None
            self._transition(TAMPState.APPROACH)

    # ── Key callback ──────────────────────────────────────────────────────────

    def key_callback(self, keycode: int) -> None:
        s = self.state
        if   keycode == ord("C"):
            self._do_calibrate()
        elif keycode == ord("J") and s == TAMPState.APPROACH:
            self._next_candidate()
        elif keycode == ord("K") and s == TAMPState.APPROACH:
            self._prev_candidate()
        elif keycode == ord("Y") and s == TAMPState.APPROACH:
            self._accept_candidate()
        elif keycode == ord("N") and s == TAMPState.APPROACH:
            self._reject_candidates()
        elif keycode == ord("O") and s == TAMPState.TRANSPORTING:
            log.info("[TAMP] Manual place trigger.")
            self._begin_placing()
        elif keycode == ord("R"):
            self._reset()
        elif keycode == ord("D"):
            self.cfg.debug_angles = not self.cfg.debug_angles
            log.info(f"[TAMP] Debug angle logging: {'ON' if self.cfg.debug_angles else 'OFF'}")

    # ── Calibration / reset ───────────────────────────────────────────────────

    def _do_calibrate(self) -> None:
        """Multi-step ROM calibration driven by repeated C presses.

        Press sequence:
          C (any state)        -> enter CALIBRATING, prompt for open hand
          C (phase 0 prompt)   -> start 2s open-hand capture
          C (phase 2 prompt)   -> start 2s fist capture; wrist offset recorded on open-hand
        """
        s = self.state
        if s == TAMPState.CALIBRATING:
            if self._calib_phase == 0:   # prompt shown; start open-hand capture
                if self._current_wrist is None:
                    log.warning("[TAMP/CALIB] No wrist detected — point hand at camera.")
                    return
                self._calib_buf.clear()
                self._last_calib_raw = None
                self._calib_start = time.monotonic()
                self._calib_phase = 1
                log.info(f"[TAMP/CALIB] Capturing SPREAD OPEN (index+thumb wide) for "
                         f"{self.cfg.calib_duration_sec:.1f}s — hold still...")
            elif self._calib_phase == 2:  # prompt shown; start fist capture
                self._calib_buf.clear()
                self._last_calib_raw = None
                self._calib_start = time.monotonic()
                self._calib_phase = 3
                log.info(f"[TAMP/CALIB] Capturing PINCH (index+thumb touching) for "
                         f"{self.cfg.calib_duration_sec:.1f}s — hold still...")
        else:
            # First C press: enter calibration state
            self._calib_phase = 0
            self._transition(TAMPState.CALIBRATING)
            if self.cfg.simple_calib:
                log.info(
                    "[TAMP/CALIB] Simple calibration (wrist offset only).\n"
                    "  SPREAD index finger and thumb WIDE APART, press C.\n"
                    f"  Fixed ROM: open_offset={self.cfg.calib_open_offset_deg}°, "
                    f"human_max={self.cfg.calib_human_max_deg}°"
                )
            else:
                log.info(
                    "[TAMP/CALIB] Full ROM calibration.\n"
                    "  Step 1: SPREAD index finger and thumb WIDE APART, press C.\n"
                    "  Step 2: PINCH index finger and thumb TOGETHER, press C.\n"
                    "  Wrist position will be recorded during the spread-open step."
                )

    def _reset(self) -> None:
        log.info("[TAMP] Reset.")
        self._grasp_q = self._grasp_p1 = self._grasp_p2 = None
        self._candidates = []
        self._candidate_idx = 0
        self._last_replan_time = -999.0
        self._gesture.reset()
        self._calib_phase = 0
        self._calib_buf.clear()
        self._q_teleop_prev = None
        self._transition(TAMPState.APPROACH if self._wrist_offset is not None
                         else TAMPState.IDLE)

    # ── ROM scaling ───────────────────────────────────────────────────────────

    def _scale_to_robot(self, human_angles: np.ndarray) -> np.ndarray:
        """Map 6 MediaPipe finger angles to robot joint space using calibrated ROM.

        t = (human - open) / (fist - open), clipped to [0, 1]
        robot_angle = t * robot_fist_limit
        """
        result = np.zeros(6)
        for i in range(6):
            human_rom = self._fist_angles[i] - self._open_angles[i]
            if abs(human_rom) < np.radians(3):   # <3° ROM means bad/uncalibrated joint
                result[i] = 0.0
                continue
            t = float(np.clip(
                (human_angles[i] - self._open_angles[i]) / human_rom, 0.0, 1.0))
            result[i] = t * self._robot_fist_angles[i]   # robot_open = 0
        return result

    # ── MediaPipe helpers ─────────────────────────────────────────────────────

    def on_hand_message(self, data_list: list) -> None:
        """Ingest a MediaPipe /hand/joint_angles message as a plain Python list."""
        self._raw_msg = data_list
        if len(data_list) >= 3:
            self._current_wrist = np.array(data_list[0:3], float)

    def _build_q_teleop(self, target_base: np.ndarray) -> np.ndarray:
        q = np.array([self.data.qpos[i] for i in self._act_idx])
        q[self._BASE_X] = target_base[0]
        q[self._BASE_Y] = target_base[1]

        if self._raw_msg is not None:
            # Scale MediaPipe angles to robot joint space using calibrated ROM.
            # open hand -> robot 0; full fist -> robot joint limit.
            human = self._extract_finger_angles(self._raw_msg)
            robot = self._scale_to_robot(human)
            self._last_human_angles = human.copy()
            self._last_robot_angles = robot.copy()
            q[self._IDX_MCP] = robot[0];  q[self._IDX_PIP] = robot[1]
            q[self._IDX_DIP] = robot[2]
            q[self._TH_MCP]  = robot[3];  q[self._TH_PIP]  = robot[4]
            q[self._TH_DIP]  = robot[5]

            if self.cfg.debug_angles and self._sim_step % 30 == 0:
                hd = np.degrees(human)
                rd = np.degrees(robot)
                log.info(
                    f"[DEBUG] Human(°) idx=[{hd[0]:.1f},{hd[1]:.1f},{hd[2]:.1f}] "
                    f"th=[{hd[3]:.1f},{hd[4]:.1f},{hd[5]:.1f}]  "
                    f"Robot(°) idx=[{rd[0]:.1f},{rd[1]:.1f},{rd[2]:.1f}] "
                    f"th=[{rd[3]:.1f},{rd[4]:.1f},{rd[5]:.1f}]"
                )

        # EMA smoothing — reduces MediaPipe noise on all axes
        if self._q_teleop_prev is None:
            self._q_teleop_prev = q.copy()
        alpha = self.cfg.smoothing_alpha
        q = alpha * q + (1.0 - alpha) * self._q_teleop_prev
        self._q_teleop_prev = q.copy()
        return q

    def _extract_finger_angles(self, raw: list) -> np.ndarray:
        """Return 6 finger joint angles in radians.

        Prefers the inter-segment flexion angles appended at positions 51-56
        (added by mediapipe_joint_angles.py get_flexion_angles).  These are
        always in [0, π/2], monotonically increase with flexion, and have no
        Euler-angle discontinuity issues.

        Falls back to Euler yaw (positions 6+ji*3) if the extended message is
        not yet available (old publisher or pre-calibration).
        """
        if len(raw) >= 57:
            # New format: inter-segment bend angles, degrees -> radians
            return np.radians(np.array(raw[51:57], float))
        # Legacy fallback: Euler ZYX yaw — may have discontinuities
        def _f(ji):
            s = 6 + ji * 3
            return np.radians(float(raw[s])) if s < len(raw) else 0.0
        return np.array([_f(3), _f(4), _f(5), _f(1), _f(2), _f(2) * 0.5])

    def _camera_to_robot(self, cam: np.ndarray) -> np.ndarray:
        # cam[0]: image X (right = +)  -> robot X (negated: mirror image vs. robot frame)
        # cam[1]: image Y (down = +)   -> robot Y (negated: image-down = robot-down)
        return np.array([-cam[0] * self.cfg.scale_x, -cam[1] * self.cfg.scale_y, 0.0])

    # ── ROS 2 ─────────────────────────────────────────────────────────────────

    def _init_ros(self) -> None:
        rclpy.init()
        self._ros_node = Node("tamp_manager")
        self._ros_node.create_subscription(
            Float32MultiArray, "/hand/joint_angles",
            lambda msg: self.on_hand_message(list(msg.data)), 10)
        self._ros_node.create_subscription(
            Empty, "/teleop/trigger_calibration",
            lambda _: self._do_calibrate(), 10)
        log.info("[TAMP] ROS 2 subscriptions active.")

    def spin_ros_once(self) -> None:
        if self._ros_node is not None:
            rclpy.spin_once(self._ros_node, timeout_sec=0.001)

    def shutdown(self) -> None:
        if self._angle_log_file is not None:
            self._angle_log_file.close()
            log.info(f"[TAMP] Angle log saved: {self.cfg.angle_log_path}")
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            rclpy.shutdown()


# =============================================================================
# Entry point
# =============================================================================

def main():
    cfg     = TAMPConfig()
    manager = TAMPManager(cfg)

    with mujoco.viewer.launch_passive(
            manager.model, manager.data,
            key_callback=manager.key_callback) as viewer:

        while viewer.is_running():
            manager.spin_ros_once()
            manager.update()
            viewer.sync()
            time.sleep(manager.model.opt.timestep)

    manager.shutdown()


if __name__ == "__main__":
    main()

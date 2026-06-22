"""
grasp_planner.py
================
Self-contained grasp contact-point solver.

Two public symbols are exported:

    GraspConfig   – Config dataclass
    GraspPlanner  – class with .solve(), .verify(), .show_in_viewer()

Typical usage
-------------
    from grasp_planner import GraspConfig, GraspPlanner

    planner = GraspPlanner(model, data)
    result  = planner.solve(q_ref, obj_pos_2d)   # runs IPOPT, returns dict

    if result["q"] is not None:
        planner.show_in_viewer(model, data, result)
        viewer.sync()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import casadi as ca
from casadi import Callback as _Callback
import mujoco as mj
import numpy as np

# minimum_NCF.py lives one level above simulation/
sys.path.insert(0, str(Path(__file__).parent.parent))
from minimum_NCF import WrenchCheck

# Module-level logger.  If the caller doesn't configure it we emit nothing.
log = logging.getLogger("grasp_planner")
if not log.handlers:
    log.addHandler(logging.NullHandler())


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraspConfig:
    """Grasp optimisation configuration."""

    # Cost weights
    w_ik:      float = 0.0
    w_reg:     float = 0.0
    w_surface: float = 0.0
    w_gamma:   float = 1.0   # penalise squeeze force (minimise internal load)

    q_scale:   float = 1.0
    p_scale:   float = 1.0
    sdf_scale: float = 1.0

    joint_limits:           bool = False
    on_object:              bool = True    # hard-constrain p1,p2 on surface
    ik_constraint:          bool = True    # hard-constrain finger to touch p
    penetration_constraint: bool = True    # gap == -1e-4; prevents finger routing through object
    wrench_hard:            bool = True    # True=hard constraint, False=soft penalty in cost
    w_wrench_pen:           float = 100.0  # penalty weight when wrench_hard=False
    max_iter:               int  = 1000

    # Task wrench bounds (used to compute min squeeze gamma)
    task_accel_x: float = 0.5    # m/s²
    task_accel_y: float = 9.81   # m/s² (gravity dominant)
    task_torque:  float = 0.1    # N·m
    mu:           float = 1.0    # friction coefficient
    obj_mass:     float = 0.2    # kg

    # Geometry / body names – must match your XML
    obj_geom:   str = 'obj1_geom'
    obj_body:   str = 'obj1'
    thumb_geom: str = 'right_thumb_distal_geom'
    index_geom: str = 'right_index_distal_geom'
    cp1_geom:   str = 'cp1'        # dummy mocap sphere – thumb
    cp2_geom:   str = 'cp2'        # dummy mocap sphere – index
    cp1_body:   str = 'cp1_body'
    cp2_body:   str = 'cp2_body'
    site_p1:    str = 'contact_p1' # visualisation site – thumb target
    site_p2:    str = 'contact_p2' # visualisation site – index target


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_actuated_indices(model: mj.MjModel) -> list[int]:
    return [model.jnt_qposadr[model.actuator_trnid[i, 0]]
            for i in range(model.nu)]


def _box_sdf_2d(point, obj_pos, half_x, half_y) -> float:
    d  = np.asarray(point) - np.asarray(obj_pos)
    dx = abs(d[0]) - half_x
    dy = abs(d[1]) - half_y
    return float(np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)
                 + min(max(dx, dy), 0.0))


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


# ─────────────────────────────────────────────────────────────────────────────
# CasADi callbacks
# ─────────────────────────────────────────────────────────────────────────────

class _FingerPointDistCallback(ca.Callback):
    """
    (q, p) → mj_geomDistance(finger_geom, dummy_sphere_at_p)

    Returns surface-to-surface signed distance:
        0  => finger surface is touching p
        >0 => gap
        <0 => penetration
    """
    def __init__(self, name, model, data_cb,
                 finger_gid, cp_gid, cp_mocap_id, act_idx, cutoff=5):
        ca.Callback.__init__(self)
        self.model = model; self.data = data_cb
        self.finger_gid = finger_gid; self.cp_gid = cp_gid
        self.cp_mocap_id = cp_mocap_id; self.act_idx = act_idx
        self.cutoff = cutoff
        self._n_calls = 0; self._t_total = 0.0
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return (ca.Sparsity.dense(len(self.act_idx), 1) if i == 0
                else ca.Sparsity.dense(2, 1))

    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        import time as _time; _t0 = _time.perf_counter()
        q_act = np.array(arg[0]).flatten()
        p     = np.array(arg[1]).flatten()
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val
        self.data.mocap_pos[self.cp_mocap_id] = [p[0], p[1], 0.0]
        mj.mj_forward(self.model, self.data)
        dist = mj.mj_geomDistance(self.model, self.data,
                                   self.finger_gid, self.cp_gid,
                                   self.cutoff, np.zeros(6))
        self._n_calls += 1; self._t_total += _time.perf_counter() - _t0
        return [dist]


class _SDFCallback(ca.Callback):
    """(q, point) → SDF of point relative to the object geom."""

    def __init__(self, name, model, data_cb, geom_id, act_idx):
        ca.Callback.__init__(self)
        self.model = model; self.data = data_cb
        self.geom_id = geom_id; self.act_idx = act_idx
        self._n_calls = 0; self._t_total = 0.0
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return (ca.Sparsity.dense(len(self.act_idx), 1) if i == 0
                else ca.Sparsity.dense(2, 1))

    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        import time as _time; _t0 = _time.perf_counter()
        q_act = np.array(arg[0]).flatten()
        point = np.array(arg[1]).flatten()
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val
        mj.mj_forward(self.model, self.data)
        obj_pos   = self.data.geom_xpos[self.geom_id][:2]
        geom_type = self.model.geom_type[self.geom_id]
        geom_size = self.model.geom_size[self.geom_id]
        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            dist = _box_sdf_2d(point, obj_pos, geom_size[0], geom_size[1])
        else:
            dist = float(np.linalg.norm(point - obj_pos) - geom_size[0])
        self._n_calls += 1; self._t_total += _time.perf_counter() - _t0
        return [dist]


class _NonPenetrationCallback(ca.Callback):
    """
    (q,) → mj_geomDistance(finger_geom, obj_geom)  (no dummy sphere)

        >0 => gap (no penetration)
         0 => touching
        <0 => penetrating
    """
    def __init__(self, name, model, data_cb, finger_gid, obj_gid, act_idx,
                 cutoff=0.5):
        ca.Callback.__init__(self)
        self.model = model; self.data = data_cb
        self.finger_gid = finger_gid; self.obj_gid = obj_gid
        self.act_idx = act_idx; self.cutoff = cutoff
        self._n_calls = 0; self._t_total = 0.0
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i): return ca.Sparsity.dense(len(self.act_idx), 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        import time as _time; _t0 = _time.perf_counter()
        q_act = np.array(arg[0]).flatten()
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val
        mj.mj_forward(self.model, self.data)
        dist = mj.mj_geomDistance(self.model, self.data,
                                   self.finger_gid, self.obj_gid,
                                   self.cutoff, None)
        self._n_calls += 1; self._t_total += _time.perf_counter() - _t0
        return [dist]


class _WrenchFeasCallback(_Callback):
    """
    CasADi external callback: wrench feasibility for all 8 task-wrench corners.

    Input:  (5,1) — [p1_x, p1_y, p2_x, p2_y, gamma]
    Output: scalar — minimum rhs across all hull constraints for all 8 corner
            wrenches.  >= 0 means all corners are wrench-feasible at this gamma.

    Delegates to WrenchCheck from minimum_NCF.py; no wrench logic is
    reimplemented here.
    """

    def __init__(self, name, obj_pos, obj_hx, obj_hy,
                 mu, obj_mass, task_accel_x, task_accel_y, task_torque):
        _Callback.__init__(self)
        self._c  = np.asarray(obj_pos, float)[:2]
        self._hx = obj_hx
        self._hy = obj_hy
        self._mu = mu
        self._fx = obj_mass * task_accel_x   # force bound x
        self._fy = obj_mass * task_accel_y   # force bound y (gravity dominant)
        self._tz = task_torque
        self._n_calls = 0; self._t_total = 0.0
        self._geom_key   = None   # rounded (p1, p2) key for geometry cache
        self._geom_cache = None   # cached (n1,t1,n2,t2,R1,R2,r1,r2,ncf1,tan1,ncf2,tan2)
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(5, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self): return False

    def _antipodal_fallback(self, p1, p2, n1) -> float:
        """Smooth finite signal when ConvexHull is degenerate (typically same-face contacts).
        sep = dot(p2-p1, n1): 0 for same-face, negative for antipodal.
        Returns a value in [-10, 0) that gives IPOPT a gradient pushing contacts apart."""
        sep = float(np.dot(p2 - p1, n1))
        val = -sep / (2.0 * max(self._hx, self._hy)) - 1.0
        return float(np.clip(val, -10.0, 0.0))

    def eval(self, arg):
        import time as _time
        from scipy.spatial import ConvexHull as _ConvexHull
        _t0 = _time.perf_counter()

        x     = np.array(arg[0]).flatten()
        p1    = x[0:2];  p2 = x[2:4];  gamma = float(x[4])
        c     = self._c

        # Cache geometry when p1/p2 haven't changed (e.g. FD perturbing only gamma)
        geom_key = tuple(np.round(x[:4], 6))
        if geom_key != self._geom_key:
            n1 = _box_surface_normal_2d(p1, c, self._hx, self._hy)
            t1 = np.array([-n1[1],  n1[0]])
            n2 = _box_surface_normal_2d(p2, c, self._hx, self._hy)
            t2 = np.array([-n2[1],  n2[0]])
            R1 = np.array([n1, t1]);  R2 = np.array([n2, t2])
            r1 = p1 - c;              r2 = p2 - c

            def _x2d(r, f): return r[0]*f[1] - r[1]*f[0]
            G = np.array([
                [_x2d(r1,n1), _x2d(r1,t1), _x2d(r2,n2), _x2d(r2,t2)],
                [n1[0], t1[0], n2[0], t2[0]],
                [n1[1], t1[1], n2[1], t2[1]],
            ])
            _, _, Vt = np.linalg.svd(G)
            f_int = Vt[-1]
            if f_int[0] < 0:
                f_int = -f_int
            ncf1 = float(f_int[0]); tan1 = float(f_int[1])
            ncf2 = float(f_int[2]); tan2 = float(f_int[3])
            self._geom_key   = geom_key
            self._geom_cache = (n1, t1, n2, t2, R1, R2, r1, r2,
                                 ncf1, tan1, ncf2, tan2)
        else:
            (n1, t1, n2, t2, R1, R2, r1, r2,
             ncf1, tan1, ncf2, tan2) = self._geom_cache

        # Wrench-cone vertices — depend on gamma, computed ONCE per eval
        checker = WrenchCheck(
            r1.reshape(2, 1), r2.reshape(2, 1), R1, R2,
            ncf1, ncf2, tan1, tan2, self._mu, self._mu,
        )
        verts1 = checker.single_wrench_cone([gamma], checker.pos1, R1, ncf1, self._mu)
        verts2 = checker.single_wrench_cone([gamma], checker.pos2, R2, ncf2, self._mu)

        # Minkowski-sum vertices and hull — computed ONCE (not 8×)
        vert_sums = np.array([np.reshape(v1 + v2, (3,))
                               for v1 in verts1 for v2 in verts2])
        if not np.all(np.isfinite(vert_sums)):
            self._n_calls += 1; self._t_total += _time.perf_counter() - _t0
            return [self._antipodal_fallback(p1, p2, n1)]
        try:
            hull = _ConvexHull(vert_sums)
        except Exception:
            self._n_calls += 1; self._t_total += _time.perf_counter() - _t0
            return [self._antipodal_fallback(p1, p2, n1)]

        # Tangential contribution from gamma * f_internal (constant across all 8 wrenches)
        tan1_b = R1 @ np.array([[0.0], [gamma * tan1]])
        tan2_b = R2 @ np.array([[0.0], [gamma * tan2]])
        delta = np.array([0.0,
                          float(tan1_b[0, 0] + tan2_b[0, 0]),
                          float(tan1_b[1, 0] + tan2_b[1, 0])])

        # Check all 8 task-wrench corners against the SAME hull.
        # scipy uses outward normals: eq @ [w, 1] < 0 for INTERIOR (feasible) points.
        # Negate so that min_rhs > 0 when feasible and < 0 when infeasible, matching
        # the constraint wf_cb >= 0.
        min_rhs = np.inf
        for ax in [-self._fx, self._fx]:
            for ay in [-self._fy, self._fy]:
                for tz in [-self._tz, self._tz]:
                    w = np.array([ax, ay, tz]) + delta
                    for eq in hull.equations:
                        val = float(eq[0]*w[0] + eq[1]*w[1] + eq[2]*w[2] + eq[3])
                        min_rhs = min(min_rhs, -val)

        if not np.isfinite(min_rhs):
            self._n_calls += 1; self._t_total += _time.perf_counter() - _t0
            return [self._antipodal_fallback(p1, p2, n1)]
        self._n_calls += 1; self._t_total += _time.perf_counter() - _t0
        return [float(min_rhs)]


# ─────────────────────────────────────────────────────────────────────────────
# GraspPlanner
# ─────────────────────────────────────────────────────────────────────────────

class GraspPlanner:
    """
    Wraps the grasp feasibility solver so it can be imported and called from
    any script without the viewer boilerplate.

    Parameters
    ----------
    model   : mj.MjModel – shared with the rest of the simulation.
    data    : mj.MjData  – shared with the simulation.  The planner reads
              the current qpos/qvel snapshot when solve() is called but uses
              its own internal MjData for all IPOPT evaluations, so the live
              simulation state is never corrupted.
    cfg     : GraspConfig  (optional – defaults used if omitted)
    logger  : logging.Logger (optional)
    log_dir : str – directory for per-solve IPOPT log files (optional)
    """

    def __init__(self,
                 model:   mj.MjModel,
                 data:    mj.MjData,
                 cfg:     GraspConfig | None = None,
                 logger:  logging.Logger | None = None,
                 log_dir: str | None = None):

        self.model   = model
        self.data    = data
        self.cfg     = cfg or GraspConfig()
        self.log     = logger or log
        self.log_dir = log_dir

        # Cache IDs once — raises immediately if XML names are wrong
        c = self.cfg
        self._obj_gid   = self._require_geom(c.obj_geom)
        self._thumb_gid = self._require_geom(c.thumb_geom)
        self._index_gid = self._require_geom(c.index_geom)
        self._cp1_gid   = self._require_geom(c.cp1_geom)
        self._cp2_gid   = self._require_geom(c.cp2_geom)

        cp1_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, c.cp1_body)
        cp2_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, c.cp2_body)
        if cp1_bid == -1 or cp2_bid == -1:
            raise ValueError(
                f"GraspPlanner: cp1_body='{c.cp1_body}' or "
                f"cp2_body='{c.cp2_body}' not found in model.")
        self._cp1_mocap = model.body_mocapid[cp1_bid]
        self._cp2_mocap = model.body_mocapid[cp2_bid]

        self._act_idx = _get_actuated_indices(model)
        self._obj_hx  = model.geom_size[self._obj_gid][0]
        self._obj_hy  = model.geom_size[self._obj_gid][1]

    # ── public API ────────────────────────────────────────────────────────────

    def solve(self,
              q_ref:   np.ndarray,
              obj_pos: np.ndarray,
              p1_init: np.ndarray | None = None,
              p2_init: np.ndarray | None = None) -> dict:
        """
        Run the grasp optimisation (synchronous / blocking).

        Jointly optimises (q, p1, p2, gamma) subject to wrench feasibility.
        gamma is the minimum squeeze (internal force scale) required to
        guarantee no-slip across the full task wrench box.  Wrench feasibility
        is enforced as an IPOPT constraint, so every converged solution is
        wrench-feasible by construction.

        Parameters
        ----------
        q_ref   : (nu,) actuated joint angles used as warm-start and
                  regularisation reference.
        obj_pos : (2,) object XY position in world frame.
        p1_init : (2,) optional warm-start for thumb contact point.
        p2_init : (2,) optional warm-start for index contact point.

        Returns
        -------
        dict
            success    : bool
            q          : (nu,) optimised joint angles  | None on failure
            p1         : (2,)  thumb contact point     | None on failure
            p2         : (2,)  index contact point     | None on failure
            gamma      : float minimum squeeze scale   | None on failure
            cost       : float | None
            iterations : int   | None
            status     : 'converged' | 'best-effort' | 'failed'
        """
        cfg        = self.cfg
        model      = self.model
        act_idx    = self._act_idx
        obj_center = np.asarray(obj_pos, dtype=float)

        # Isolated MjData – solver never touches the live simulation copy
        data_cb = mj.MjData(model)
        data_cb.qpos[:] = self.data.qpos[:]
        data_cb.qvel[:] = self.data.qvel[:]
        mj.mj_forward(model, data_cb)

        # Unique prefix so callback names don't collide across calls
        uid = id(data_cb)

        # ── callbacks ────────────────────────────────────────────────────────
        ik_thumb_cb = _FingerPointDistCallback(
            f'gp_ik_thumb_{uid}', model, data_cb,
            self._thumb_gid, self._cp1_gid, self._cp1_mocap, act_idx)
        ik_index_cb = _FingerPointDistCallback(
            f'gp_ik_index_{uid}', model, data_cb,
            self._index_gid, self._cp2_gid, self._cp2_mocap, act_idx)
        sdf_cb = _SDFCallback(
            f'gp_sdf_{uid}', model, data_cb, self._obj_gid, act_idx)
        nonpen_thumb_cb = _NonPenetrationCallback(
            f'gp_np_thumb_{uid}', model, data_cb,
            self._thumb_gid, self._obj_gid, act_idx)
        nonpen_index_cb = _NonPenetrationCallback(
            f'gp_np_index_{uid}', model, data_cb,
            self._index_gid, self._obj_gid, act_idx)
        wf_cb = _WrenchFeasCallback(
            f'gp_wf_{uid}', obj_center, self._obj_hx, self._obj_hy,
            cfg.mu, cfg.obj_mass, cfg.task_accel_x, cfg.task_accel_y,
            cfg.task_torque)

        # ── optimization problem ─────────────────────────────────────────────
        opti  = ca.Opti()
        q     = opti.variable(model.nu)
        p1    = opti.variable(2)
        p2    = opti.variable(2)
        gamma = opti.variable()   # minimum squeeze — decision variable

        d_thumb   = ik_thumb_cb(q, p1)
        d_index   = ik_index_cb(q, p2)
        sdf1      = sdf_cb(q, p1)
        sdf2      = sdf_cb(q, p2)
        gap_thumb = nonpen_thumb_cb(q)
        gap_index = nonpen_index_cb(q)

        # Cost: IK + surface (if soft) + joint regularisation + squeeze penalty
        cost = cfg.w_ik * (d_thumb**2 + d_index**2)
        if not cfg.on_object:
            cost += cfg.w_surface * ((sdf1 / cfg.sdf_scale)**2 +
                                     (sdf2 / cfg.sdf_scale)**2)
        cost += cfg.w_reg   * ca.sumsqr((q - q_ref) / cfg.q_scale)
        cost += cfg.w_gamma * gamma
        opti.minimize(cost)

        # Geometric constraints
        if cfg.joint_limits:
            for i in range(model.nu):
                jid = model.actuator_trnid[i, 0]
                if model.jnt_limited[jid]:
                    opti.subject_to(opti.bounded(
                        model.jnt_range[jid, 0], q[i], model.jnt_range[jid, 1]))

        if cfg.on_object:
            _r_cp = 1e-4
            opti.subject_to(sdf1 == -2 * _r_cp)
            opti.subject_to(sdf2 == -2 * _r_cp)

        if cfg.ik_constraint:
            opti.subject_to(d_thumb == 0)
            opti.subject_to(d_index == 0)

        if cfg.penetration_constraint:
            opti.subject_to(gap_thumb == -1e-4)
            opti.subject_to(gap_index == -1e-4)

        # Keep contact points within the object bounding box (prevents the
        # optimizer from sending p1/p2 to infinity to trivially satisfy the
        # wrench constraint via large contact-arm leverage).
        margin = max(self._obj_hx, self._obj_hy)
        for p in (p1, p2):
            opti.subject_to(opti.bounded(
                obj_center[0] - self._obj_hx - margin, p[0],
                obj_center[0] + self._obj_hx + margin))
            opti.subject_to(opti.bounded(
                obj_center[1] - self._obj_hy - margin, p[1],
                obj_center[1] + self._obj_hy + margin))

        # Wrench feasibility: all 8 task-wrench corners feasible at gamma
        wf_expr = wf_cb(ca.vertcat(p1, p2, gamma))
        if cfg.wrench_hard:
            opti.subject_to(wf_expr >= 0)
        else:
            cost += cfg.w_wrench_pen * ca.fmax(0, -wf_expr)**2
            opti.minimize(cost)   # re-set cost with penalty included
        opti.subject_to(opti.bounded(1, gamma, 1000))

        # Warm-start
        p1_seed = (np.asarray(p1_init, float) if p1_init is not None
                   else obj_center + np.array([-self._obj_hx, 0.0]))
        p2_seed = (np.asarray(p2_init, float) if p2_init is not None
                   else obj_center + np.array([+self._obj_hx, 0.0]))
        opti.set_initial(q,     q_ref)
        opti.set_initial(p1,    p1_seed)
        opti.set_initial(p2,    p2_seed)
        opti.set_initial(gamma, 100.0)  # large start → wf_cb feasible; IPOPT minimises down

        # IPOPT options
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
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ipopt_opts['output_file']      = os.path.join(
                self.log_dir, f"grasp_ipopt_{ts}.log")
            ipopt_opts['file_print_level'] = 5

        opti.solver('ipopt', {'ipopt': ipopt_opts, 'print_time': False})

        # ── solve ────────────────────────────────────────────────────────────
        def _profile_log():
            cbs = [
                ('ik_thumb', ik_thumb_cb), ('ik_index', ik_index_cb),
                ('sdf',      sdf_cb),      ('np_thumb', nonpen_thumb_cb),
                ('np_index', nonpen_index_cb), ('wf',   wf_cb),
            ]
            parts = [
                f"{name}: {cb._n_calls} calls, {cb._t_total:.1f}s "
                f"({1e3*cb._t_total/max(cb._n_calls,1):.1f}ms/call)"
                for name, cb in cbs
            ]
            self.log.info("[profile] " + " | ".join(parts))

        try:
            sol = opti.solve()
            _profile_log()
            return {
                'success':    True,
                'q':          sol.value(q),
                'p1':         sol.value(p1),
                'p2':         sol.value(p2),
                'gamma':      float(sol.value(gamma)),
                'cost':       float(sol.value(opti.f)),
                'iterations': sol.stats()['iter_count'],
                'status':     'converged',
            }
        except Exception as e:
            self.log.warning(f"GraspPlanner.solve: {e}")
            _profile_log()
            try:
                return {
                    'success':    False,
                    'q':          opti.debug.value(q),
                    'p1':         opti.debug.value(p1),
                    'p2':         opti.debug.value(p2),
                    'gamma':      float(opti.debug.value(gamma)),
                    'cost':       None,
                    'iterations': None,
                    'status':     'best-effort',
                }
            except Exception as e2:
                self.log.error(f"GraspPlanner.solve debug extraction: {e2}")
                return {
                    'success': False, 'q': None, 'p1': None, 'p2': None,
                    'gamma': None, 'cost': None, 'iterations': None,
                    'status': 'failed',
                }

    def verify(self, result: dict) -> dict:
        """
        Post-solve sanity check.  Logs and returns IK errors, non-penetration
        gaps, and SDF values.  Safe to call on any result dict.
        """
        if result.get('q') is None:
            self.log.warning("GraspPlanner.verify: result has no q, skipping.")
            return {}

        model  = self.model
        data_v = mj.MjData(model)
        data_v.qpos[:]                    = self.data.qpos[:]
        data_v.qpos[self._act_idx]        = result['q']
        data_v.mocap_pos[self._cp1_mocap] = [result['p1'][0], result['p1'][1], 0.0]
        data_v.mocap_pos[self._cp2_mocap] = [result['p2'][0], result['p2'][1], 0.0]
        mj.mj_forward(model, data_v)

        ik_t  = mj.mj_geomDistance(model, data_v,
                                    self._thumb_gid, self._cp1_gid, 0.5, None)
        ik_i  = mj.mj_geomDistance(model, data_v,
                                    self._index_gid, self._cp2_gid, 0.5, None)
        gap_t = mj.mj_geomDistance(model, data_v,
                                    self._thumb_gid, self._obj_gid,  0.5, None)
        gap_i = mj.mj_geomDistance(model, data_v,
                                    self._index_gid, self._obj_gid,  0.5, None)
        obj_p = data_v.geom_xpos[self._obj_gid][:2]
        s1    = _box_sdf_2d(result['p1'], obj_p, self._obj_hx, self._obj_hy)
        s2    = _box_sdf_2d(result['p2'], obj_p, self._obj_hx, self._obj_hy)

        info = {
            'ik_thumb_mm':  ik_t  * 1000,
            'ik_index_mm':  ik_i  * 1000,
            'gap_thumb_mm': gap_t * 1000,
            'gap_index_mm': gap_i * 1000,
            'sdf_p1_mm':    s1    * 1000,
            'sdf_p2_mm':    s2    * 1000,
        }
        self.log.info(
            f"[GraspPlanner.verify] "
            f"IK=({ik_t*1e3:.2f},{ik_i*1e3:.2f})mm  "
            f"GAP=({gap_t*1e3:+.2f},{gap_i*1e3:+.2f})mm  "
            f"SDF=({s1*1e3:.2f},{s2*1e3:.2f})mm  "
            f"gamma={result.get('gamma', 'N/A')}"
        )
        return info

    def show_in_viewer(self,
                       model:  mj.MjModel,
                       data:   mj.MjData,
                       result: dict) -> None:
        """
        Write the solver result into *data* so the passive viewer shows the
        proposed grasp.  Call viewer.sync() immediately after this.
        """
        if result.get('q') is None:
            return

        data.qpos[self._act_idx]           = result['q']
        data.mocap_pos[self._cp1_mocap]    = [result['p1'][0], result['p1'][1], 0.0]
        data.mocap_pos[self._cp2_mocap]    = [result['p2'][0], result['p2'][1], 0.0]
        mj.mj_forward(model, data)

        for site_name, pt in [(self.cfg.site_p1, result['p1']),
                               (self.cfg.site_p2, result['p2'])]:
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
            if sid != -1:
                model.site_pos[sid] = [pt[0], pt[1], 0.0]

    # ── private ───────────────────────────────────────────────────────────────

    def _require_geom(self, name: str) -> int:
        gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
        if gid == -1:
            raise ValueError(
                f"GraspPlanner: geom '{name}' not found in model.  "
                f"Check GraspConfig geometry name fields.")
        return gid

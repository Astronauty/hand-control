#!/usr/bin/env python3
# All variable naming notation follows https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html
"""3D analog of internal_force_control.py: RRT + internal-force grasp control for the
Kinova Gen3 + LEAP hand pick-and-place scene (models/scene_pick_place.xml).

Same keyboard-driven REACH/GRASP state machine as internal_force_control.py, generalized
to 3D contacts/wrenches via the spatial classes in grasp_control/. The number of fingers
used in the grasp is configurable via FINGER_SET (see below) — the controller loops over
contacts generically rather than hardcoding 2.
"""
import argparse
import numpy as np
import subprocess
import sys
import time
import threading
import traceback
import queue

# Windows consoles often report a legacy codepage (e.g. cp1252) for stdout even in
# UTF-8-capable terminals (Git Bash/Windows Terminal), which crashes on the arrow
# glyphs (←→↑↓) used in the on-screen control hints below.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import mujoco as mj
import mujoco.viewer  # noqa: F401
from pynput import keyboard as _pynput_kb

from scripts.rrt_planner import RRTPlanner
from grasp_control import SpatialIKSolver, ConstrainedIKSolver, GraspController
from grasp_control.constrained_ik import configure_sqp
from live_dashboard import Dashboard

# NLP grasp recommender (contact_aware_teleop mode). simulation/ hosts the planner;
# _geom_normal_np gives the outward surface normal used to build inward contact frames.
sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/simulation')
from grasp_planner_3d import (GraspConfig3D, MultiStartGraspPlanner3D,  # noqa: E402
                              _geom_normal_np)

# 3D_minimum_NCF.py isn't an importable module name (leading digit), so load it by path.
import importlib.util as _ilu
_ncf_spec = _ilu.spec_from_file_location(
    '_ncf', __file__.rsplit('/', 1)[0] + '/scripts/3D_minimum_NCF.py')
_ncf = _ilu.module_from_spec(_ncf_spec)
_ncf_spec.loader.exec_module(_ncf)


def solve_gamma_live(p_O, R_O_inward, mu, mass, accel_box_xyz, ang_accel_box_xyz,
                     inertia_diag):
    """Minimum internal-force scale gamma that keeps the grasp no-slip for the given
    acceleration/torque disturbance box, from the live grasp geometry. Wraps
    3D_minimum_NCF.min_gamma_for_accel_lp; verified against its native antipodal cases
    (see scratchpad/verify_gamma_solve.py). Handles the two convention mismatches:
      * normal sign: spatial_grasp_map / GraspController give col0 = INWARD normal;
        the NCF cone is built with col0 = OUTWARD (force pushing ON the object) -> flip.
      * unit mass: min_gamma_for_accel_lp assumes m=1, so the "accel box" is really a
        FORCE box (m*a) and the "torque box" a real torque (I*alpha) -> scale here.

    Args:
        p_O:            list of (3,) contact positions in the OBJECT body frame.
        R_O_inward:     list of (3,3) contact->object rotations, col0 = inward normal.
        mu:             list of per-contact friction coefficients.
        mass:           object mass (kg).
        accel_box_xyz:  (ax,ay,az) linear-accel budget incl. gravity, object-body axes.
        ang_accel_box_xyz: (alpha_x,alpha_y,alpha_z) angular-accel budget, PRINCIPAL axes.
        inertia_diag:   (Ix,Iy,Iz) principal moments (model.body_inertia). Multiplies the
                        angular-accel budget into a torque box.

    Returns:
        gamma (float), or None if the grasp geometrically cannot resist the box.
    """
    n = len(p_O)
    R_out = [R.copy() for R in R_O_inward]
    for R in R_out:
        R[:, 0] *= -1.0                                   # inward -> outward normal
    pos = [np.asarray(p, float).reshape(3, 1) for p in p_O]
    fx, fy, fz = (mass * a for a in accel_box_xyz)        # force box = m * a
    tx, ty, tz = (I * al for I, al in zip(inertia_diag, ang_accel_box_xyz))  # torque = I*alpha
    return _ncf.min_gamma_for_accel_lp(
        fx, fy, fz, tx, ty, tz, n, pos, R_out, [1.0] * n,
        [0.0] * n, [0.0] * n, list(mu))


def _reindex_hull(hull):
    """ConvexHull.simplices index the ORIGINAL points; remap to hull.vertices order."""
    old_to_new = {old: new for new, old in enumerate(hull.vertices)}
    return np.array([[old_to_new[i] for i in s] for s in hull.simplices], np.int32)


def _flat_hull(pts3, center):
    """Coplanar point set -> filled 2D polygon as a centroid triangle fan in 3D.
    Used when a wrench subspace collapses to a plane (e.g. the antipodal grasp's
    torque cone lives in Tx=0: a pinch resists no torque about the grasp axis)."""
    from scipy.spatial import ConvexHull
    _, _, vt = np.linalg.svd(pts3 - center)
    p2 = (pts3 - center) @ vt[:2].T
    ring = ConvexHull(p2).vertices
    verts = np.vstack([center, pts3[ring]]).astype(np.float32)
    m = len(ring)
    faces = np.array([[0, 1 + i, 1 + (i + 1) % m] for i in range(m)], np.int32)
    return {'verts': verts, 'faces': faces}


def composite_wrench_cone(gamma, p_O, R_O_inward, mu):
    """Force- and torque-subspace hulls of the composite grasp wrench cone at scale
    gamma: the Minkowski sum of each contact's pyramidal wrench cone — the exact set
    3D_minimum_NCF's LP tests wrench membership in. Returns
        {'force': {'verts','faces'} | None, 'torque': {'verts','faces'} | None}
    for the dashboard's 3D panels (None when a subspace is a point/line). Convention
    matches solve_gamma_live (col0 inward->outward). single_wrench_cone lays out each
    vertex as [Tx,Ty,Tz, Fx,Fy,Fz]."""
    import itertools
    from scipy.spatial import ConvexHull
    n = len(p_O)
    R_out = [R.copy() for R in R_O_inward]
    for R in R_out:
        R[:, 0] *= -1.0
    wc = _ncf.WrenchCheck(n, [np.asarray(p).reshape(3, 1) for p in p_O],
                          R_out, [1.0] * n, [0.0] * n, [0.0] * n, list(mu))
    per_contact = [wc.single_wrench_cone(gamma, np.asarray(p_O[i]).reshape(3, 1),
                                         R_out[i], 1.0, mu[i]) for i in range(n)]
    # Minkowski sum: sum one vertex per contact over all combinations (nverts^n points).
    W = np.array([sum(c) for c in itertools.product(*per_contact)])

    def hull3d(pts3):
        pts3 = np.asarray(pts3, float)
        c = pts3.mean(0)
        s = np.linalg.svd(pts3 - c, compute_uv=False)
        rank = int(np.count_nonzero(s > 1e-9 * (s[0] if s[0] > 0 else 1)))
        if rank < 2:
            return None
        if rank == 2:
            return _flat_hull(pts3, c)
        h = ConvexHull(pts3)
        return {'verts': pts3[h.vertices].astype(np.float32), 'faces': _reindex_hull(h)}

    return {'force': hull3d(W[:, 3:6]), 'torque': hull3d(W[:, 0:3])}


# Fingertip contact sites added in models/build_kinova_leap.py (_add_fingertip_sites),
# named "<finger-body>_tip" with the "leap_" attach() prefix.
FINGER_TIP_SITES = {
    "index":  "leap_if_ds_tip",
    "middle": "leap_mf_ds_tip",
    "ring":   "leap_rf_ds_tip",
    "thumb":  "leap_th_ds_tip",
}
# Collision-geom name prefix for each finger's link chain (bs/px/md/ds + tip), used to
# build the RRT's finger_geom_names list.
FINGER_CODE = {"index": "if", "middle": "mf", "ring": "rf", "thumb": "th"}

# v1: 2-finger pinch grasp — matches the 2 antipodal contact sites currently defined per
# object (obj_xxx_c1/c2 in models/scene_pick_place.xml). Extend by adding more contact
# sites to the object XML and listing more fingers here; the controller below loops over
# however many entries this has.
FINGER_SET = ["index", "thumb"]


# Gen3 arm "home" pose — a natural elbow-bent reach-forward configuration. Read at
# runtime from gen3.xml's "home" keyframe (see GEN3_XML / HOME_ARM in main) rather than
# hardcoded, so it tracks the source model. Used as the IK null-space bias for the 7 arm
# joints: the null-space pull toward this pose produces a forward/lateral approach to
# tabletop objects (confirmed visually) without needing explicit orientation constraints
# on the fingertips — the orientation approach (IKSolver's (local_axis, world_target)
# tuple) was implemented and validated but caused joint-limit clipping instability on
# this 23-DOF redundant chain when combined with position constraints, preventing
# convergence.
GEN3_XML = 'mujoco_menagerie/kinova_gen3/gen3.xml'

# FINGERTIP_POINTING_AXIS is kept for reference; used by _approach_orientation below.
# Not currently used in the main IK loop (position-only + HOME_ARM bias suffices), but
# available if a caller wants to add per-site orientation control in future.
FINGERTIP_POINTING_AXIS = np.array([0.0, -1.0, 0.0])


def _approach_orientation(normal):
    """Return an IKSolver (local_axis, world_target) orientation spec that points a
    LEAP fingertip's tip axis inward along -normal. Not used in the current main()
    control loop (position-only IK + HOME_ARM bias gives sufficient approach geometry),
    but available as a building block for future tighter orientation control."""
    return (FINGERTIP_POINTING_AXIS, -normal)


def _finger_collision_geoms(model, finger):
    """All collision/tip geom names belonging to one LEAP finger's link chain."""
    code = FINGER_CODE[finger]
    prefix = f"leap_{code}_"
    names = []
    for i in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        if name and name.startswith(prefix) and ("collision" in name or name.endswith("_tip")):
            names.append(name)
    return names


def make_key_callback(key_queue):
    """Return a GLFW key callback for the MuJoCo passive viewer.
    Handles only Enter / Q / Esc; target selection is handled via pynput
    (Ctrl+digit) so modifier state is available without fighting GLFW."""
    # Keycodes chosen to NOT collide with the viewer's built-in single-key toggles
    # (mjVISSTRING/mjRNDSTRING shortcuts claim nearly every letter: R=Reflection,
    # K=Skybox, B=Perturb Force, G=Fog, ...). 'N' is the only unbound letter; the
    # viewer only uses digits 0-5 (geom groups), so 6-8 are free. Q still overlaps
    # the Camera-frustum toggle, but we quit on it so the toggle never matters.
    _MAP = {
        257: 'enter',   # GLFW_KEY_ENTER (main keyboard)
        335: 'enter',   # GLFW_KEY_KP_ENTER (numpad)
        81:  'quit',    # Q
        256: 'quit',    # Escape
        78:  'release',  # N — (GRASP) open fingers and return to REACH
        76:  'lock_in',  # L — (contact_aware_teleop) lock in the recommended grasp
                        #     contacts and approach them via RRT
        54:  'ik_vis',  # 6 — cycle IK config visualization
        55:  'bspheres', # 7 — toggle IK collision bounding-sphere overlay
        56:  'teleop_start', # 8 — (dexpilot) start/re-zero tracking at current pose
        57:  'calib_orient', # 9 — (dexpilot) hold hand to match robot wrist, capture
                             #     the constant orientation correction
        # Multi-pose orientation calibration (dexpilot):
        77:  'calib_next',   # M — pose the robot to the next calibration orientation
        67:  'calib_capture', # C — capture (hand matched to current wrist)
        86:  'calib_solve',  # V — solve the full rotation correction from captures
        259: 'reset',   # Backspace — deliberately shadows the viewer's built-in Reset:
                        # the viewer already mj_resetData'd the shared data from its own
                        # thread; this event lets the control loop re-home its state
                        # machine instead of PD-exploding against the qpos0 arm pose.
    }
    def _cb(keycode):
        event = _MAP.get(keycode)
        if event:
            key_queue.put(event)
    return _cb


_RANDOMIZE_OBJ_INFO = [
    # (body_name, geom_name, base_rgb) — pure R/G/B, no colour jitter
    ('obj_red_box',        'obj_red_box_geom',        [1.0, 0.0, 0.0]),
    ('obj_red_sphere',     'obj_red_sphere_geom',     [1.0, 0.0, 0.0]),
    ('obj_blue_cylinder',  'obj_blue_cylinder_geom',  [0.0, 0.0, 1.0]),
    ('obj_blue_capsule',   'obj_blue_capsule_geom',   [0.0, 0.0, 1.0]),
    ('obj_green_box',      'obj_green_box_geom',      [0.0, 1.0, 0.0]),
    ('obj_green_cylinder', 'obj_green_cylinder_geom', [0.0, 1.0, 0.0]),
]


def _randomize_objects(model, data, rng):
    """Randomize positions, colors, and sizes for all pickable objects.

    Must be called after MjData creation and before the first mj_forward.
    Updates both data.qpos and model.qpos0 so mj_resetData preserves
    the randomized object positions throughout the IK precomputation loop.
    """
    PICK_CENTER = np.array([0.5, 0.5])
    PICK_HALF   = np.array([0.27, 0.27])   # centers in [0.23, 0.77]² to keep objects on marker
    MIN_SEP     = 0.10                      # minimum centre-to-centre separation (m)
    MAX_TRIES   = 2000

    n = len(_RANDOMIZE_OBJ_INFO)
    size_scales = rng.uniform(0.88, 1.12, n)  # ±12% uniform size scale per object

    # Rejection-sample 2-D positions with minimum separation
    xy_list: list[np.ndarray] = []
    for _ in range(n):
        for _ in range(MAX_TRIES):
            xy = PICK_CENTER + rng.uniform(-PICK_HALF, PICK_HALF)
            if all(np.linalg.norm(xy - p) >= MIN_SEP for p in xy_list):
                xy_list.append(xy)
                break
        else:
            xy_list.append(PICK_CENTER + rng.uniform(-PICK_HALF, PICK_HALF))

    for i, (bname, gname, base_rgb) in enumerate(_RANDOMIZE_OBJ_INFO):
        s   = float(size_scales[i])
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bname)
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, gname)

        # Size: scale geom_size and bounding sphere uniformly
        model.geom_size[gid]   *= s
        model.geom_rbound[gid] *= s

        # Color: fixed pure R/G/B (no jitter)
        model.geom_rgba[gid, :3] = base_rgb

        # Contact sites: scale the x offset (contact-normal direction) in body frame
        for sit_id in range(model.nsite):
            if model.site_bodyid[sit_id] == bid:
                model.site_pos[sit_id, 0] *= s

        # Position: derive rest height from scaled geom size, then write freejoint qpos
        sz    = model.geom_size[gid]          # already scaled
        gtype = model.geom_type[gid]
        if gtype == mj.mjtGeom.mjGEOM_SPHERE:
            z_rest = float(sz[0])
        elif gtype == mj.mjtGeom.mjGEOM_BOX:
            z_rest = float(sz[2])
        elif gtype == mj.mjtGeom.mjGEOM_CYLINDER:
            z_rest = float(sz[1])             # half-height; rests on flat face
        elif gtype == mj.mjtGeom.mjGEOM_CAPSULE:
            z_rest = float(sz[0] + sz[1])     # radius + half-cylinder-length
        else:
            z_rest = float(sz[0])

        jnt_adr = model.body_jntadr[bid]
        if jnt_adr < 0:
            continue
        qadr = model.jnt_qposadr[jnt_adr]
        pos7 = np.array([xy_list[i][0], xy_list[i][1], z_rest, 1.0, 0.0, 0.0, 0.0])
        data.qpos[qadr:qadr + 7]   = pos7
        model.qpos0[qadr:qadr + 7] = pos7    # preserve across mj_resetData calls


if __name__ == "__main__":
    _arg_parser = argparse.ArgumentParser(
        description="Kinova+LEAP pick-place: RRT + internal-force grasp controller.")
    _arg_parser.add_argument(
        '--viz-only', action='store_true',
        help="Debug mode for IK/RRT: disable arm/hand collision physics and contact-force "
             "viz, and never hand off to mj_step — REACH/GRASP hold their target pose via "
             "kinematic replay (qpos overwrite + mj_forward) forever, so the IK solution and "
             "RRT path can be inspected without any dynamics interference.")
    _arg_parser.add_argument(
        '--dashboard', action='store_true',
        help="Launch a live pyqtgraph metrics dashboard (separate process): planning mode "
             "(Approach/Grasp/Transport), proximity-based active object, scrolling "
             "fingertip→object distances, net hand→object wrench, per-finger contact "
             "normal forces, and a combined RRT+IK planner solution log.")
    _arg_parser.add_argument(
        '--mode',
        choices=['contact_aware_autonomous', 'contact_aware_teleop', 'dexpilot',
                 'rrt'],
        default='contact_aware_autonomous',
        help="contact_aware_autonomous (default): autonomous RRT+IK grasp controller, "
             "plans to predefined per-object contact sites ('rrt' is a deprecated alias). "
             "| contact_aware_teleop: teleop the wrist (DexPilot mapping) with MediaPipe "
             "fingers while an NLP continuously recommends grasp contacts for the nearest "
             "object; press L to lock in and approach via RRT, then GRASP with the NLP's "
             "gamma.  | dexpilot: live MediaPipe kinematic retargeting teleop via ROS 2.")
    _arg_parser.add_argument(
        '--camera', type=int, default=None,
        help="Camera index forwarded to ui/mediapipe_joint_angles.py in dexpilot mode "
             "(default: auto-select — prefers external/USB camera at index ≥1).")
    _arg_parser.add_argument(
        '--position-mode', choices=['relative', 'absolute'], default=None,
        help="dexpilot position mapping. relative: press-8 re-zeroable, robot tracks "
             "abs_scale × (board displacement from press-8). absolute: true absolute, "
             "hand's board position -> fixed robot position (board origin -> base "
             "origin), scaled by abs_scale, no re-zero. Default: from "
             "calibration/teleop_config.json (relative if unset). Other tunables "
             "(abs_scale, world_from_board) live in that config file.")
    _arg_parser.add_argument(
        '--seed', type=int, default=None,
        help="RNG seed for object randomization — the same seed reproduces the same "
             "layout (positions and sizes). Default: fresh entropy every run. "
             "Ignored with --no-randomize.")
    _arg_parser.add_argument(
        '--no-randomize', action='store_true',
        help="Skip object randomization: objects keep the positions, sizes, and colors "
             "authored in models/scene_pick_place.xml (default: randomize positions on "
             "the pick marker, ±12%% sizes, pure R/G/B colors).")
    _arg_parser.add_argument(
        '--hand-self-collision', action='store_true',
        help="Re-enable LEAP hand self-collision (finger↔finger, finger↔palm contact "
             "physics). Disabled by default: hand geoms are moved to contype=2 so "
             "hand↔hand pairs never match, while hand↔object/floor/arm keep colliding.")
    _arg_parser.add_argument(
        '--ik-solver', choices=['sqp', 'ipopt'], default='sqp',
        help="sqp (default): sqpmethod + OSQP + softplus SDF + analytic FK Jacobians — "
             "~3× cheaper per iteration, wins on wall time in most cases  |  "
             "ipopt: IPOPT L-BFGS + finite-difference Jacobians — production baseline.")
    args = _arg_parser.parse_args()
    if args.mode == 'rrt':          # deprecated alias
        args.mode = 'contact_aware_autonomous'

    model = mj.MjModel.from_xml_path('models/scene_pick_place.xml')
    data  = mj.MjData(model)

    N_ROBOT = 23  # joint_1..7 (Gen3 arm) + 16 LEAP finger joints; object joints follow
    # Gen3's odd joints (1,3,5,7 -> indices 0,2,4,6) are continuous revolute with no
    # jnt_range — RRTPlanner samples uniformly from model.jnt_range, so [0,0] would never
    # randomize those joints. Give them a generous sampling bound before the planner reads it.
    for j in (0, 2, 4, 6):
        model.jnt_range[j] = [-np.pi, np.pi]

    # Disable LEAP hand self-collision (finger↔finger, finger↔palm) unless re-enabled
    # via --hand-self-collision. Bitmask trick: hand collision geoms move from the
    # compiled (contype=1, conaffinity=1) to (contype=2, conaffinity=1) — a pair
    # collides iff (contype_A & conaffinity_B) | (contype_B & conaffinity_A), so two
    # hand geoms (2&1 both ways) no longer match, while hand↔object/floor/arm pairs
    # (those stay 1,1) still do via the hand geom's conaffinity. mj_step physics only:
    # IK/RRT clearance checks run mj_geomDistance against objects/floor and never test
    # hand↔hand pairs, so planning is unaffected. Visual geoms (contype 0) are skipped.
    if not args.hand_self_collision:
        _n_selfcol = 0
        for _gi in range(model.ngeom):
            _bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[_gi]) or ''
            if _bname.startswith('leap_') and model.geom_contype[_gi]:
                model.geom_contype[_gi] = 2
                _n_selfcol += 1
        print(f"[collision] LEAP hand self-collision OFF ({_n_selfcol} geoms; "
              f"--hand-self-collision to re-enable)")

    # Object randomization (positions/sizes/colors + matching qpos0 so resets restore
    # the same layout). --no-randomize keeps the XML-authored scene; qpos0 already
    # matches the compiled spawn poses there, so reset recovery works either way.
    if not args.no_randomize:
        _randomize_objects(model, data, np.random.default_rng(args.seed))

    mj.mj_forward(model, data)

    # Snapshot the compiled contype BEFORE --viz-only may zero the robot geoms' contype
    # below: the collision-geom lists (_robot_geom_names etc.) distinguish real collision
    # geoms from visual-only ones by contype==0, and --viz-only zeroes ALL robot contypes —
    # so building those lists off the live contype would (wrongly) drop every hand geom and
    # silently disable IK/RRT collision avoidance in viz-only. Use this snapshot instead.
    _geom_contype0 = model.geom_contype.copy()

    if args.viz_only:
        # IK collision-avoidance (ConstrainedIKSolver) and RRT clearance checks use raw
        # mj_geomDistance, which ignores contype/conaffinity, so this only suppresses
        # mj_step contact forces — those constraints stay active regardless (they read the
        # _geom_contype0 snapshot above, not the zeroed live contype).
        _robot_body_ids_dbg = {model.jnt_bodyid[j] for j in range(model.njnt)
                                if model.jnt_qposadr[j] < N_ROBOT}
        for _gi in range(model.ngeom):
            if model.geom_bodyid[_gi] in _robot_body_ids_dbg:
                model.geom_contype[_gi]     = 0
                model.geom_conaffinity[_gi] = 0
        print("[viz-only] arm/hand collision physics disabled (contype/conaffinity=0)")

    # Fingertip sites (on the active FINGER_SET fingers)
    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f]) for f in FINGER_SET]
    N_FINGERS = len(FINGER_SET)

    STEPS_PER_WP    = 5    # max sim steps before forcing waypoint advance (timeout, 1 step = 1ms)
    WP_REACH_TOL    = 0.02  # joint-space radius to consider a waypoint reached (rad)
    JOG_VEL         = 0.2  # jog speed while arrow key held (m/s)
    # Singularity-robust DLS jog damping (see the GRASP-branch resolved-rate solve):
    # JOG_SING_EPS is the smallest-singular-value threshold below which damping ramps
    # in; JOG_LAM_MAX caps the peak joint-rate gain at ~1/(2*JOG_LAM_MAX). With
    # JOG_VEL=0.10 m/s the worst-case rate is ~JOG_VEL/(2*JOG_LAM_MAX)=1.0 rad/s.
    JOG_SING_EPS    = 0.02  # rad·m onset of damping (raise = damp earlier/more conservative)
    JOG_LAM_MAX     = 0.05  # peak damping (raise = gentler but sloppier near singularities)

    # Object definitions: rigid objects only (obj_soft deferred — vertex-level contact,
    # not a rigid grasp-map problem). Each object maps every FINGER_SET finger to the
    # contact site it's driven to, explicitly by finger name rather than by list
    # position — reassign an entry here to send a finger to a different contact site on
    # the object without touching FINGER_SET or FINGER_TIP_SITES. This is also the
    # natural place to later turn finger->contact-site into an optimized assignment
    # rather than a fixed lookup.
    object_defs = [
        ({'index': 'obj_red_box_c2',        'thumb': 'obj_red_box_c1'},        'obj_red_box'),
        ({'index': 'obj_red_sphere_c2',     'thumb': 'obj_red_sphere_c1'},     'obj_red_sphere'),
        ({'index': 'obj_blue_cylinder_c2',  'thumb': 'obj_blue_cylinder_c1'},  'obj_blue_cylinder'),
        ({'index': 'obj_blue_capsule_c2',   'thumb': 'obj_blue_capsule_c1'},   'obj_blue_capsule'),
        ({'index': 'obj_green_box_c2',      'thumb': 'obj_green_box_c1'},      'obj_green_box'),
        ({'index': 'obj_green_cylinder_c2', 'thumb': 'obj_green_cylinder_c1'}, 'obj_green_cylinder'),
    ]
    objects = []
    for contact_sites, body_name in object_defs:
        missing = [f for f in FINGER_SET if f not in contact_sites]
        assert not missing, (
            f"{body_name} has no contact site mapped for finger(s) {missing}")
        obj = {
            'name':    body_name,
            'id_S':    [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, contact_sites[f])
                        for f in FINGER_SET],
            'id_body': mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name),
            'id_geom': mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, body_name + '_geom'),
        }
        # Initial values only — _run_ik refreshes both from the object's live pose at
        # solve time (the contact sites are children of the object body, so they move
        # with it).
        obj['p_S_W'] = [data.site_xpos[sid].copy() for sid in obj['id_S']]
        # Inward surface normal at each contact site (the site's world-frame local
        # x-axis, per the scene XML convention) — the direction each fingerpad normal is
        # driven to align with in the constrained-IK orientation cost. Read from the same
        # FK state as p_S_W so the two stay consistent.
        obj['inward_S_W'] = [data.site_xmat[sid].reshape(3, 3)[:, 0].copy()
                             for sid in obj['id_S']]
        objects.append(obj)

    dls_ik = SpatialIKSolver(n_robot=N_ROBOT)

    # Collision geoms to keep clear during IK and RRT: EVERY collision geom on all four
    # LEAP fingers + the palm + bracelet_link (wrist), not one representative per body. A
    # single representative is not sufficient — the palm alone has 10 collision geoms and
    # each finger link 3-6, so representative-only checking let the rest of each body clip
    # through objects/floor. The IK affords the full set because its FD cost is now
    # per-arm-geom (one position callback each), not per arm-geom×object pair.
    _active_body_prefixes = tuple(f'leap_{code}_' for code in FINGER_CODE.values())
    _robot_geom_names = []
    for _gi in range(model.ngeom):
        _gname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, _gi)
        if not _gname or _geom_contype0[_gi] == 0:
            continue   # skip unnamed / visual-only geoms (compiled contype, pre-viz-only)
        _bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[_gi]) or ''
        if (any(_bname.startswith(p) for p in _active_body_prefixes)
                or _bname in ('leap_palm', 'bracelet_link')):
            _robot_geom_names.append(_gname)
    # 'floor' is the ground plane — keep every checked hand geom above it (both IK and RRT).
    _OBJ_GEOM_NAMES = [
        'obj_red_box_geom', 'obj_red_sphere_geom',
        'obj_blue_cylinder_geom', 'obj_blue_capsule_geom',
        'obj_green_box_geom', 'obj_green_cylinder_geom',
        'floor',
    ]
    constrained_ik = ConstrainedIKSolver(
        model, N_ROBOT,
        arm_geom_names=_robot_geom_names,
        obj_geom_names=_OBJ_GEOM_NAMES,
        clearance=0.005,
        posture_weight=0.0005,
        pad_axis=(-1.0, 0.0, 0.0),  # LEAP fingerpad normal in the fingertip-site frame
        # Weight the tip-position task above the posture regularizer: at 1.0 (raw m²)
        # a 15mm tip error costs 2e-4 vs a posture term of ~3e-3, so the solver traded
        # ~cm of placement for posture comfort — and the GRASP squeeze then computed
        # its grasp map against contacts that never landed where the plan assumed.
        # orient_weight rides along at tip:pad = 1000:1 (A/B'd across the 6 objects:
        # 10000:1 starved pad alignment to ~58 deg on the capsule, 100:1 let the pad
        # term drag tips into 20mm+ local minima on sphere/capsule; 1000:1 gives
        # ~0.5-2.5mm tips at <5 deg pads on 4/6 objects).
        # NOTE: _SQP_SOLVER_OPTS' tol_du (1e-2) is calibrated to this cost scale.
        tip_weight=20.0,
        orient_weight=1.0,
        max_iter=800,   # DLS warm-start puts us near solution; headroom for the tightened tol
    )
    if args.ik_solver == 'sqp':
        configure_sqp(constrained_ik)
        _ik_mode_str = "sqpmethod/OSQP + softplus SDF + analytic Jacobians"
    else:
        _ik_mode_str = "IPOPT L-BFGS + finite-difference Jacobians"
    print(f"[IK] {len(_robot_geom_names)} robot geoms × {len(_OBJ_GEOM_NAMES)} objects "
          f"= {len(_robot_geom_names) * len(_OBJ_GEOM_NAMES)} collision constraints  "
          f"[solver: {_ik_mode_str}]")

    # Fingertip tip-geom ids for the live dashboard's fingertip→object distance plot
    # (mj_geomDistance from each finger's tip mesh geom to the active object's geom).
    _TIP_GEOM_IDS = {f: mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f'leap_{FINGER_CODE[f]}_tip')
                     for f in FINGER_SET}

    # Pad-surface offset per finger: distance from the fingertip SITE (which sits at the
    # tip mesh's centroid, per build_kinova_leap's FINGER_TIPS) to the actual fingerpad
    # SURFACE along the pad normal (pad_axis = -x of the site frame). The IK targets the
    # site, but physical contact happens at the pad surface — aiming the site directly at
    # a contact site on the object buries the tip mesh ~this deep into the object
    # (measured -8.4mm index / -10.0mm thumb on the red box), which shoves the object
    # around during the GRASP hold and breaks the squeeze's grasp-map geometry.
    # Site and geom are rigid on the same body, so one kinematics pass at any config
    # gives the constant offset: max projection of the tip-mesh vertices onto the pad
    # direction, relative to the site.
    def _pad_surface_offset(f):
        d0 = mj.MjData(model)
        mj.mj_kinematics(model, d0)
        gid = _TIP_GEOM_IDS[f]
        sid = id_C[FINGER_SET.index(f)]
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts_W = (d0.geom_xmat[gid].reshape(3, 3) @ model.mesh_vert[adr:adr + num].T).T \
                  + d0.geom_xpos[gid]
        pad_dir_W = -d0.site_xmat[sid].reshape(3, 3)[:, 0]   # world dir of pad_axis (-x)
        return float(np.max((verts_W - d0.site_xpos[sid]) @ pad_dir_W))

    _PAD_OFFSET = {f: _pad_surface_offset(f) for f in FINGER_SET}
    print(f"[IK] fingerpad surface offsets: "
          + "  ".join(f"{f}={_PAD_OFFSET[f]*1e3:.1f}mm" for f in FINGER_SET))

    # All four fingertip tip geoms — used to pick the proximity-based "active object":
    # the object with the smallest AVERAGE signed tip→object distance. This is what the
    # in-scene hover marker and the dashboard's "active object" label / wrench plots
    # track, independent of which target the user selected.
    _ALL_TIP_GIDS = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f'leap_{c}_tip')
                     for c in FINGER_CODE.values()]

    def _guarded_geom_dist(g1, g2, distmax=2.0):
        """mj_geomDistance guarded with the bounding-sphere lower bound — 3.3.5's GJK
        can return a phantom 0.0 for well-separated convex pairs."""
        lb = (np.linalg.norm(data.geom_xpos[g1] - data.geom_xpos[g2])
              - model.geom_rbound[g1] - model.geom_rbound[g2])
        return max(mj.mj_geomDistance(model, data, g1, g2, distmax, None), lb)

    # Contact bookkeeping for the dashboard's net-wrench and normal-force plots.
    _HAND_GIDS      = {mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g) for g in _robot_geom_names}
    _OBJ_GID_TO_IDX = {o['id_geom']: i for i, o in enumerate(objects)}
    _FINGER_BY_GID  = {mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g): f
                       for f in FINGER_SET for g in _finger_collision_geoms(model, f)}

    def _hand_object_contact_metrics(obj_idx):
        """Scan live contacts once and return
          (f_net, tau_net): net world-frame wrench the HAND applies to objects[obj_idx],
                            torque taken about the object's COM;
          normals: each FINGER_SET finger's summed contact normal force vs ANY object;
          tangentials: same, but the summed tangential (friction) force magnitude —
                       tangential/normal vs mu is the friction-cone utilization, i.e.
                       how close each contact is to slipping.
        Convention (verified empirically, MuJoCo 3.3.5): contact.frame rows are the
        contact-frame axes with row 0 = normal pointing geom1→geom2, and mj_contactForce
        returns the force applied TO geom2 expressed in that frame — so the force on the
        object is +R.T@f when the object is geom2 and -R.T@f when it is geom1."""
        f_net, tau_net = np.zeros(3), np.zeros(3)
        normals     = {f: 0.0 for f in FINGER_SET}
        tangentials = {f: 0.0 for f in FINGER_SET}
        com = data.xipos[objects[obj_idx]['id_body']]
        ft = np.zeros(6)
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1, g2 = con.geom1, con.geom2
            if g1 in _HAND_GIDS and g2 in _OBJ_GID_TO_IDX:
                hand_gid, obj_gid, sgn = g1, g2, 1.0
            elif g2 in _HAND_GIDS and g1 in _OBJ_GID_TO_IDX:
                hand_gid, obj_gid, sgn = g2, g1, -1.0
            else:
                continue
            mj.mj_contactForce(model, data, ci, ft)
            fname = _FINGER_BY_GID.get(hand_gid)
            if fname is not None:
                normals[fname]     += ft[0]   # normal component (>= 0, along frame row 0)
                tangentials[fname] += float(np.hypot(ft[1], ft[2]))
            if _OBJ_GID_TO_IDX[obj_gid] == obj_idx:
                R_con = con.frame.reshape(3, 3)
                f_W = sgn * (R_con.T @ ft[:3])
                f_net   += f_W
                tau_net += np.cross(con.pos - com, f_W) + sgn * (R_con.T @ ft[3:6])
        return f_net, tau_net, normals, tangentials

    # Live metrics dashboard (separate process; opt-in via --dashboard). Started before the
    # IK precompute so the grasp IPOPT solves below are reported too. dash is None
    # when disabled; every push site is guarded on it.
    dash = None
    if args.dashboard:
        dash = Dashboard(FINGER_SET, horizon_s=5.0, dt_hint=3 * model.opt.timestep)
        dash.start()
        print("[dashboard] launched (separate process)")

    def _push_ipopt(obj_name, dls_ms=None, ipopt_ms=None):
        """Forward ConstrainedIKSolver.last_metrics + stage timing to the dashboard's
        combined planner panel. No-op when the dashboard is disabled."""
        if dash is None:
            return
        m = constrained_ik.last_metrics
        dash.push({
            'type':         'ipopt',
            'object':       obj_name,
            'status':       m.get('status', '?'),
            'iters':        m.get('iters', '?'),
            'max_site_mm':  max(m.get('site_err_mm', [0.0])),
            'max_pad_deg':  (max(m['pad_deg']) if m.get('pad_deg') else None),
            'min_slack_mm': m.get('min_slack_mm'),
            'dls_ms':       dls_ms,
            'ipopt_ms':     ipopt_ms,
        })

    def _push_squeeze(on, gamma):
        """Forward the GRASP internal-force state to the dashboard header. gamma is the
        per-object value solved at the REACH->GRASP transition; ~gamma/sqrt(2) N per
        contact for the 2-contact pinch. No-op when the dashboard is disabled."""
        if dash is None:
            return
        dash.push({'type': 'squeeze', 'on': bool(on), 'gamma': gamma,
                   'f_contact': gamma / np.sqrt(2)})

    def _push_wrench_cone(gamma, p_O, R_in, mu):
        """Compute the composite grasp wrench cone at gamma and push its force/torque
        hulls to the dashboard's 3D panels (as vertex/face lists). gamma=None clears
        the meshes (e.g. at reset). No-op when the dashboard is disabled."""
        if dash is None:
            return
        if gamma is None:
            dash.push({'type': 'wrench_cone', 'force': None, 'torque': None})
            return
        try:
            cone = composite_wrench_cone(gamma, p_O, R_in, mu)
        except Exception as _e:
            print(f"\r\n[wrench-cone] skipped ({_e})")
            return
        dash.push({
            'type':   'wrench_cone',
            'force':  None if cone['force'] is None else
                      {'verts': cone['force']['verts'].tolist(),
                       'faces': cone['force']['faces'].tolist()},
            'torque': None if cone['torque'] is None else
                      {'verts': cone['torque']['verts'].tolist(),
                       'faces': cone['torque']['faces'].tolist()},
        })

    def _squeeze_diag(d, period_s=1.0):
        """Once per period while squeezing: commanded vs measured per-contact force,
        friction-cone utilization, and tip↔contact-site slip distance. Separates the
        two failure modes: measured normal ≪ commanded → force-delivery problem (PD
        fight / grasp-map geometry error); utilization near 100% and slip growing →
        tangential shear (raise gamma or stiffen the slip correction)."""
        now = time.time()
        if now - _squeeze_diag.last < period_s:
            return
        _squeeze_diag.last = now
        f_c = grasp_ctrl.last_f_c
        _, _, normals, tangentials = _hand_object_contact_metrics(active_idx)
        mu = float(model.geom_friction[obj_grasp['id_geom'], 0])
        parts = []
        for k, f in enumerate(FINGER_SET):
            cmd = (float(np.linalg.norm(f_c[3 * k:3 * k + 3]))
                   if f_c is not None else float('nan'))
            n_meas, t_meas = normals[f], tangentials[f]
            util = t_meas / (mu * n_meas) if n_meas > 1e-6 else float('inf')
            # Slip vs the pad-offset anchor (where the tip SITE sits when the pad
            # surface is flush), not the raw surface site 10mm ahead of it.
            sid_S = obj_grasp['id_S'][k]
            inward_W = d.site_xmat[sid_S].reshape(3, 3)[:, 0]
            anchor_W = d.site_xpos[sid_S] - _PAD_OFFSET[f] * inward_W
            slip_mm = 1e3 * float(np.linalg.norm(d.site_xpos[id_C[k]] - anchor_W))
            parts.append(f"{f}: cmd={cmd:.1f}N meas={n_meas:.1f}N "
                         f"fric={util:.0%} slip={slip_mm:.1f}mm")
        print("\r\n[Squeeze] " + "  |  ".join(parts))
    _squeeze_diag.last = 0.0

    # Arm home pose from gen3.xml's "home" keyframe (the composite model has its keyframes
    # stripped in build_kinova_leap.py, since a 7-DOF arm keyframe no longer fits the larger
    # composite nq — so read it from the source arm model instead of hardcoding).
    HOME_ARM = mj.MjModel.from_xml_path(GEN3_XML).key('home').qpos[:7].copy()

    # Null-space bias: HOME_ARM pulls the 7 arm joints toward a natural elbow-bent
    # reach-forward pose (gen3.xml "home" keyframe), producing a lateral/forward approach
    # to tabletop objects confirmed via offscreen render. Middle/ring fingers (not in
    # FINGER_SET) are curled to prevent their extended-at-zero default pose from poking
    # the floor when the palm lowers to table height. Joint order per finger: mcp,rot,pip,dip.
    Q_BIAS = np.zeros(N_ROBOT)
    Q_BIAS[:7]    = HOME_ARM          # arm: elbow-bent reach-forward
    Q_BIAS[11:15] = [1.2, 0.0, 0.5, 0.5]  # leap_mf_*: curl out of the way
    Q_BIAS[15:19] = [1.2, 0.0, 0.5, 0.5]  # leap_rf_*: curl out of the way

    # NOTE: do NOT alias model.qpos0 to Q_BIAS. For hinge/slide joints qpos0 is the
    # *kinematic reference* — a body's transform is built from (qpos - qpos0) about the
    # joint axis, and gen3.xml authors the arm so its geometric zero (qpos == qpos0) is
    # the straight-up pose. Setting qpos0 = Q_BIAS would make qpos == Q_BIAS render as
    # that straight-up zero (displacement 0), not the elbow-bent HOME_ARM angles — and it
    # would also point the IK null-space bias (q_bias = Q_BIAS) at the straight-up pose.
    # Q_BIAS is an *absolute* joint target, so qpos0 stays at its compiled default and we
    # set data.qpos = Q_BIAS explicitly wherever we want that starting pose.

    # Two-stage IK: DLS (position-only, fast) then IPOPT (collision-aware refinement).
    # IPOPT warm-started from the DLS result is near the solution and converges in <20
    # iterations. Starting IPOPT cold (from Q_BIAS) requires hundreds of iterations because
    # the FD Jacobian is inaccurate over large displacements and collision constraints bind
    # early, producing constrained local minima far from the position target.
    #
    # Active (FINGER_SET) fingers must reach/wrap the object being grasped. The IK models
    # each geom as a bounding SPHERE (finger links 15-24mm), which is far too coarse to both
    # constrain a finger AND let it approach a 3cm object — but only the DISTAL link and tip
    # actually need to reach the surface. So instead of disabling the whole finger (which
    # let proximal links sit inside the object with unbounded penetration), each active-finger
    # geom gets a clearance tiered by how close its link legitimately comes to the surface:
    #   contact tier  (ds/tip geoms):        disabled — they must touch/wrap the object
    #   adjacent tier (if_md / th_px links): -10mm    — tolerates bounding-sphere slack
    #                                                    (~1-2cm over-approximation) while
    #                                                    still bounding gross penetration
    #   proximal tier (everything else):     +2mm     — should never be near
    # e surface
    # All tiers KEEP full clearance vs the floor plane (the solver never reduces the plane
    # constraint), so no active finger drops underground and grasp goals stay valid.
    # Palm, wrist, and the non-active fingers stay fully constrained against everything.
    # The RRT (exact distances, no bounding sphere) analogously gives active geoms a 0mm
    # clearance vs the TARGET object only — touch allowed, penetration rejected (see _run_rrt).
    ANOBJ_DISABLE = -1.0
    _active_finger_geoms = {g for g in _robot_geom_names
                            if any(g.startswith(f'leap_{FINGER_CODE[f]}_') for f in FINGER_SET)}

    def _active_obj_clearance(g):
        if '_ds_' in g or g.endswith('_tip'):
            return ANOBJ_DISABLE                   # contact tier: must reach the surface
        if g.startswith(('leap_if_md', 'leap_th_px')):
            return -0.010                          # adjacent tier: bounded sphere slack
        return 0.002                               # proximal tier: stay off the surface

    _active_clearance_by_geom = {g: _active_obj_clearance(g) for g in _active_finger_geoms}
    print(f"[IK] active-finger geoms (tiered object clearance, floor kept): "
          f"{len(_active_finger_geoms)}")

    # Dedicated MjData for background IK solves so the IK thread never touches the main
    # simulation data. Object positions are set per solve from the live-scene snapshot
    # taken at selection time (obj_qpos_snap), so IK always targets current object poses.
    _ik_data   = mj.MjData(model)
    _ik_solved = set()   # object indices whose IK has been solved and cached
    # Cached IK is reused only while the scene still matches the snapshot it was solved
    # against (see obj['ik_obj_qpos'] and the selection handler). Component-wise qpos
    # tolerance: ignores settling jitter, catches jogs/pushes/re-placement (~>5mm).
    IK_STALE_TOL = 5e-3

    # Per-joint PD gains for REACH phase: 7 arm joints + 16 LEAP finger joints.
    # Arm gains sized for Gen3's forcerange (±105/±52 Nm); finger gains mirror the small
    # values used for the planar model's tiny finger actuators.
    Kp = np.concatenate([np.full(7, 40.0), np.full(16, 0.8)])
    Kd = np.concatenate([np.full(7, 4.0),  np.full(16, 0.05)])

    # Internal squeeze force scale (GRASP, toggled with Enter): f_c = null(G) @ gamma.
    # gamma is now SOLVED per object at the REACH->GRASP transition (solve_gamma_live)
    # from the live grasp geometry and the acceleration-budget box below, rather than
    # hardcoded. GAMMA_FALLBACK is used only if the LP reports the grasp geometrically
    # cannot resist the box (returns None) — then we squeeze at a fixed value and warn.
    GAMMA_FALLBACK = 250.0

    # Disturbance box for the gamma solve, as ACCELERATION budgets (not forces): the
    # max linear / angular acceleration a jog will ever drive the grasped object
    # through. solve_gamma_live converts these to a force/torque box via the object's
    # own mass and principal inertia (m*a, I*alpha), so gamma auto-rescales per object.
    # Gravity is added to the vertical axis at solve time. The jog is slew-rate limited
    # to NCF_ACCEL_BUDGET_XYZ (see the GRASP branch), so the linear budget is ENFORCED
    # by construction and gamma covers the true worst case at 1.0x margin. The angular
    # budget is a small cushion for parasitic wrist rotation near singularities (the
    # jog commands zero angular velocity).
    NCF_ACCEL_BUDGET_XYZ = (0.5, 0.5, 0.5)   # m/s^2   object-frame linear-accel budget
    NCF_ANG_ACCEL_BUDGET = (1.0, 1.0, 1.0)   # rad/s^2 principal-frame angular-accel budget

    # Conservative multiplier on the solved gamma before it drives the squeeze: the LP
    # gives the MINIMUM no-slip gamma for the box (1.0x margin), which leaves nothing for
    # contact compliance, the finger PD lagging the internal-force torques, the
    # pyramidal-vs-elliptic cone mismatch, and unmodeled dynamics. 2x squeezes twice as
    # hard as the theoretical minimum. Only the value SENT TO THE CONTROLLER is scaled;
    # the wrench-cone viz stays at the raw 1.0x gamma so the drawn cage remains the true
    # feasible boundary the LP computed (the trace then sits well inside it).
    GAMMA_SAFETY_FACTOR = 20.0

    # Softens Kp/Kd on the active (grasping) finger joints while squeezing, via
    # GraspController.effective_gains(). Without this the full-strength joint PD
    # hold fights the internal-force torques: as GAMMA pushes the fingers to press
    # harder, the position spring (anchored at the fixed pre-squeeze q_grasp_hold)
    # pulls back proportionally, so measured contact force saturates well below
    # GAMMA/sqrt(2) instead of scaling with it.
    SQUEEZE_PD_SCALE = 5.0

    # Ramp the squeeze force 0->GAMMA over this many seconds of sim time after each
    # squeeze-on. The internal force pair only cancels once BOTH contacts exist; at
    # toggle time the pads typically hover ~mm off the surface (object settled after
    # REACH), and full force while a finger is still closing that gap arrives as an
    # unbalanced shove that knocks the object across the table (measured: 35N commanded
    # -> box launched 400mm; with the ramp the contacts form at ~N-level forces first).
    SQUEEZE_RAMP_S = 1.0

    # Give the RRT the SAME full hand-geom set the IK constrains (_robot_geom_names) plus
    # the floor, instead of only the fingertips — checking just the tips let the palm /
    # proximal links / wrist sweep straight through objects unnoticed. The historical reason
    # for tips-only (the active fingers legitimately pass near the target at the goal) is now
    # handled by the per-plan target-aware pair-clearance overrides below.
    OBJ_BODIES = [
        'obj_red_box', 'obj_red_sphere',
        'obj_blue_cylinder', 'obj_blue_capsule',
        'obj_green_box', 'obj_green_cylinder',
    ]
    planner = RRTPlanner(model, _robot_geom_names, OBJ_BODIES,
                         extra_obj_geom_names=['floor'], n_robot=N_ROBOT,
                         n_plan=7,            # plan only the 7 arm joints; finger DOF fixed at goal
                         clearance=0.005)     # 5mm clearance, matching the IPOPT solves

    # Geom ids of all ACTIVE-finger geoms — used to build per-plan pair-clearance overrides
    # (0mm vs the TARGET object) so they may approach (touch) the object being grasped
    # without the RRT disqualifying the goal, matching the IK's reduced-clearance treatment.
    # Every other hand geom (palm, proximal links, wrist, non-active fingers) and every
    # geom-vs-floor pair stays checked at the full clearance.
    _ACTIVE_SKIP_GIDS = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g) for g in _active_finger_geoms]

    # Background RRT: result dict shared between thread and main loop
    _plan_result = {}

    # Ghost-path visualization: FK data object + thread-safe marker list.
    # _ghost_markers holds (p_start, p_end, rgba) line-segment tuples connecting each
    # fingertip's position across consecutive sampled waypoints (one polyline per
    # finger), rendered as capsule connectors. _ik_markers_by_obj (below) is a
    # separate, still sphere-based feature for the static grasp IK config.
    _ghost_data         = mj.MjData(model)
    _ghost_markers_lock = threading.Lock()
    _ghost_markers      = []

    _GHOST_SITES    = id_C  # same sites driven by IK, so the ghost path matches the IK target
    _N_GHOST        = 15   # max waypoints to sample for ghost display

    # Static IK config markers: built lazily after each object's IK is solved.
    # gold (1, 0.8, 0) = grasp
    # Indexed by object index; None until the object's IK has been solved.
    def _make_ik_markers(obj, obj_qpos_snap):
        markers = []
        _d = mj.MjData(model)
        _d.qpos[N_ROBOT:] = obj_qpos_snap
        for q_cfg, rgba in [
            (obj['q_target'],   np.array([1.0, 0.8, 0.0, 0.55], dtype=np.float32)),
        ]:
            _d.qpos[:N_ROBOT] = q_cfg
            mj.mj_forward(model, _d)
            positions = [_d.site_xpos[s].copy() for s in _GHOST_SITES]
            markers.append((positions, rgba))
        return markers

    _ik_markers_by_obj = [None] * len(objects)

    def _update_ghost_markers(waypoints):
        n = len(waypoints)
        indices = np.round(np.linspace(0, n - 1, min(_N_GHOST, n))).astype(int)
        obj_qpos = data.qpos[N_ROBOT:].copy()   # snapshot object positions
        segments = []
        prev_positions = None
        for k, idx in enumerate(indices):
            t = k / max(1, len(indices) - 1)
            mj.mj_resetData(model, _ghost_data)
            _ghost_data.qpos[:N_ROBOT] = waypoints[idx]
            _ghost_data.qpos[N_ROBOT:] = obj_qpos
            mj.mj_forward(model, _ghost_data)
            positions = [_ghost_data.site_xpos[s].copy() for s in _GHOST_SITES]
            # Colour: grey-blue at start → orange-green at end, semi-transparent
            rgba = np.array([0.6 - 0.5*t, 0.4 + 0.4*t, 0.8 - 0.5*t, 0.55],
                            dtype=np.float32)
            if prev_positions is not None:
                # One line segment per finger, connecting its position at the
                # previous sampled waypoint to its position at this one.
                segments.extend((p0, p1, rgba)
                                 for p0, p1 in zip(prev_positions, positions))
            prev_positions = positions
        with _ghost_markers_lock:
            _ghost_markers.clear()
            _ghost_markers.extend(segments)

    def _run_rrt(q_start, q_grasp, obj_target):
        # Snapshot current object positions so collision checks reflect the live scene.
        planner._data.qpos[N_ROBOT:] = data.qpos[N_ROBOT:].copy()
        # The goal is the GRASP config itself (fingertips on the contact sites), so the
        # goal node necessarily touches — or, per the IK's disabled contact tier /
        # -10mm adjacent tier, marginally penetrates — the target object. Two-part mask
        # makes it admissible without opening holes elsewhere:
        #   1. Active-finger geoms get 0mm clearance vs the TARGET object only — exact
        #      distance is still checked, so they may touch but never sweep through it.
        #   2. plan()'s endpoint grace then relaxes any pair still violating at the
        #      endpoints (tip/ds at ~0 or slightly inside at the grasp goal, or the start
        #      config hugging an object just released) to its endpoint distance — free to
        #      move away, never deeper.
        # Everything else — palm, proximal links, wrist, non-active fingers, and every
        # geom vs the floor — stays checked at the full clearance.
        pair_clearance = {(g, obj_target['id_geom']): 0.0 for g in _ACTIVE_SKIP_GIDS}
        # Re-branch the goal's continuous (base/wrist) joints onto the turn nearest the
        # current pose, so the arm never unwinds a near-full revolution just because the IK
        # left a joint on a far 2pi branch. Same configuration, planner-friendly numbering.
        q_goal = planner.rebranch(q_start, q_grasp)
        _t0 = time.time()
        path = planner.plan(q_start, q_goal, pair_clearance=pair_clearance)
        plan_time = time.time() - _t0
        fallback = path is None
        if fallback:
            # Linear-interpolation fallback: 100 intermediate configs so the PD
            # controller tracks a smooth sequence rather than jumping to the goal,
            # which would produce explosive torques and numerical instability.
            n_interp = 100
            ts = np.linspace(0, 1, n_interp)
            path = [q_start + t * (q_goal - q_start) for t in ts]
            print(f"\r\n[RRT] Planning failed — linear fallback ({n_interp} steps)")
            for i, o in enumerate(objects):
                print(f"         obj{i+1} pos: {data.xpos[o['id_body']]}")
        else:
            print(f"\r\n[RRT] {len(path)} waypoints")
        if dash is not None:
            dash.push({'type': 'rrt', 'object': obj_target['name'], 'n_wp': len(path),
                       'plan_time': plan_time, 'fallback': fallback})
        _plan_result['waypoints'] = path
        _update_ghost_markers(path)

    def _run_ik(obj_idx, obj, obj_qpos_snap):
        """Solve grasp IK for one object (fingertips directly on the contact sites),
        storing q_target — the single IK solution per object, used both as the RRT goal
        and the GRASP-phase hold pose. Runs inside the background plan_thread — uses
        _ik_data, never touches main data. Each DLS and SQP stage is timed
        individually and pushed to the dashboard."""
        mj.mj_resetData(model, _ik_data)
        _ik_data.qpos[:N_ROBOT] = Q_BIAS
        # Pose every object at the live-scene snapshot (mj_resetData above put them back
        # at model.qpos0, i.e. their spawn poses) so both the fingertip targets and the
        # solver's collision constraints reflect where the objects actually ARE now.
        _ik_data.qpos[N_ROBOT:] = obj_qpos_snap
        mj.mj_forward(model, _ik_data)

        # Refresh the targets from the object's current pose: the contact sites are
        # children of the object body, so this FK carries them with it. Without the
        # refresh, IK for an object that moved (jogged, pushed, released elsewhere)
        # would still aim at wherever it spawned.
        obj['p_S_W']      = [_ik_data.site_xpos[sid].copy() for sid in obj['id_S']]
        obj['inward_S_W'] = [_ik_data.site_xmat[sid].reshape(3, 3)[:, 0].copy()
                             for sid in obj['id_S']]
        # Scene snapshot this solve is valid for — the selection handler re-solves
        # instead of reusing the cache once the scene drifts past IK_STALE_TOL.
        obj['ik_obj_qpos'] = obj_qpos_snap.copy()

        # --- Grasp IK ---
        # Target the tip SITES at the contact sites backed off by the pad-surface
        # offset along each contact's inward normal, so the fingerpad SURFACE (not
        # the tip-mesh centroid the site sits at) lands flush on the object.
        obj['ik_targets'] = [p - _PAD_OFFSET[f] * n
                             for f, p, n in zip(FINGER_SET, obj['p_S_W'],
                                                obj['inward_S_W'])]
        _t0 = time.time()
        q_dls_grasp = dls_ik.solve(model, _ik_data, id_C, obj['ik_targets'],
                                    q_bias=Q_BIAS, null_gain=0.3)
        dls_grasp_ms = (time.time() - _t0) * 1e3
        _t0 = time.time()
        obj['q_target'] = constrained_ik.solve(_ik_data, id_C, obj['ik_targets'],
                                                q_bias=Q_BIAS, q_init=q_dls_grasp,
                                                reduced_clearance_geoms=_active_clearance_by_geom,
                                                inward_dirs=obj['inward_S_W'])
        ipopt_grasp_ms = (time.time() - _t0) * 1e3
        _push_ipopt(obj['name'],
                    dls_ms=dls_grasp_ms, ipopt_ms=ipopt_grasp_ms)

        total_ms = dls_grasp_ms + ipopt_grasp_ms
        print(f"\r\n[IK] obj{obj_idx+1}: grasp DLS {dls_grasp_ms:.0f}ms + SQP {ipopt_grasp_ms:.0f}ms"
              f"  |  total {total_ms:.0f}ms")

        # Tip-error + penetration audit (same checks as the old upfront loop), with the
        # objects posed at the same snapshot the solve used.
        _d_chk = mj.MjData(model)
        _d_chk.qpos[N_ROBOT:] = obj_qpos_snap
        for label, q_sol, tgts in [('grasp', obj['q_target'], obj['ik_targets'])]:
            _d_chk.qpos[:N_ROBOT] = q_sol
            mj.mj_forward(model, _d_chk)
            errs = [f"{np.linalg.norm(_d_chk.site_xpos[s] - t)*1e3:.1f} mm"
                    for s, t in zip(id_C, tgts)]
            print(f"\r\n[IK] obj{obj_idx+1} {label}: tip errors = {errs}")
            _ft  = np.zeros(6)
            _pen = {}
            for g in _active_finger_geoms:
                _gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g)
                _lb  = (np.linalg.norm(_d_chk.geom_xpos[obj['id_geom']] - _d_chk.geom_xpos[_gid])
                        - model.geom_rbound[_gid] - model.geom_rbound[obj['id_geom']])
                _pen[g] = max(mj.mj_geomDistance(model, _d_chk, _gid, obj['id_geom'], 1.0, _ft), _lb)
            _worst_g, _worst_d = min(_pen.items(), key=lambda kv: kv[1])
            _flag = '  ** PENETRATION **' if _worst_d < -0.002 else ''
            print(f"\r\n[IK] obj{obj_idx+1} {label}: min exact dist = "
                  f"{_worst_d*1e3:.1f} mm ({_worst_g}){_flag}")

        _ik_markers_by_obj[obj_idx] = _make_ik_markers(obj, obj_qpos_snap)
        _ik_solved.add(obj_idx)

    def _run_ik_then_rrt(obj_idx, obj, q_start, obj_qpos_snap):
        """Background thread: solve IK for obj (if not cached), then run RRT."""
        _run_ik(obj_idx, obj, obj_qpos_snap)
        _run_rrt(q_start, obj['q_target'], obj)

    def _run_ik_recommended(obj_idx, obj, obj_qpos_snap, rec):
        """contact_aware_teleop lock-in: like _run_ik but the fingertip targets come
        from the NLP-recommended contacts (rec['p1']=thumb, rec['p2']=index in world)
        instead of the authored contact sites. Stores the recommended contacts as
        object-LOCAL frames on obj so the GRASP provider tracks the moving object,
        then reuses the same DLS->SQP grasp IK to get obj['q_target']."""
        mj.mj_resetData(model, _ik_data)
        _ik_data.qpos[:N_ROBOT] = Q_BIAS
        _ik_data.qpos[N_ROBOT:] = obj_qpos_snap
        mj.mj_forward(model, _ik_data)

        # World contacts in FINGER_SET order ([index, thumb]): index<-p2, thumb<-p1.
        _by_finger = {'thumb': rec['p1'], 'index': rec['p2']}
        p_S_W = [np.asarray(_by_finger[f], float).copy() for f in FINGER_SET]
        # Inward surface normals at those contacts (object's live geom pose).
        n1_in, n2_in = _recommended_inward_normals(obj_idx, rec['p1'], rec['p2'])
        _n_by_finger = {'thumb': n1_in, 'index': n2_in}
        inward_S_W = [np.asarray(_n_by_finger[f], float).copy() for f in FINGER_SET]

        obj['p_S_W']       = p_S_W
        obj['inward_S_W']  = inward_S_W
        obj['ik_obj_qpos'] = obj_qpos_snap.copy()

        # Store object-local contact frames for the GRASP-phase provider. col0 of
        # R_O is the inward normal (matching the site convention the provider expects).
        bid   = obj['id_body']
        p_WoO = _ik_data.xpos[bid].copy()
        R_WO  = _ik_data.xmat[bid].reshape(3, 3).copy()

        def _local_frame(p_W, n_in_W):
            # Build a right-handed frame with col0 = inward normal, then express in
            # the object body frame. Tangents are arbitrary (only col0 is used by the
            # grasp map / slip anchor, which take col0 and the position).
            n = n_in_W / (np.linalg.norm(n_in_W) + 1e-12)
            ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t1 = np.cross(n, ref);  t1 /= (np.linalg.norm(t1) + 1e-12)
            t2 = np.cross(n, t1)
            R_W = np.column_stack([n, t1, t2])
            p_O = R_WO.T @ (p_W - p_WoO)
            R_O = R_WO.T @ R_W
            return p_O, R_O

        obj['rec_local'] = [_local_frame(p, n) for p, n in zip(p_S_W, inward_S_W)]

        # Grasp IK: tip SITES aimed at contacts backed off by the pad-surface offset
        # along each inward normal (same as _run_ik).
        obj['ik_targets'] = [p - _PAD_OFFSET[f] * n
                             for f, p, n in zip(FINGER_SET, p_S_W, inward_S_W)]
        _t0 = time.time()
        q_dls_grasp = dls_ik.solve(model, _ik_data, id_C, obj['ik_targets'],
                                    q_bias=Q_BIAS, null_gain=0.3)
        dls_ms = (time.time() - _t0) * 1e3
        _t0 = time.time()
        obj['q_target'] = constrained_ik.solve(_ik_data, id_C, obj['ik_targets'],
                                                q_bias=Q_BIAS, q_init=q_dls_grasp,
                                                reduced_clearance_geoms=_active_clearance_by_geom,
                                                inward_dirs=inward_S_W)
        sqp_ms = (time.time() - _t0) * 1e3
        print(f"\r\n[IK] obj{obj_idx+1} (recommended): grasp DLS {dls_ms:.0f}ms + "
              f"SQP {sqp_ms:.0f}ms")

        _d_chk = mj.MjData(model)
        _d_chk.qpos[N_ROBOT:] = obj_qpos_snap
        _d_chk.qpos[:N_ROBOT] = obj['q_target']
        mj.mj_forward(model, _d_chk)
        errs = [f"{np.linalg.norm(_d_chk.site_xpos[s] - t)*1e3:.1f} mm"
                for s, t in zip(id_C, obj['ik_targets'])]
        print(f"\r\n[IK] obj{obj_idx+1} (recommended): tip errors = {errs}")

        _ik_markers_by_obj[obj_idx] = _make_ik_markers(obj, obj_qpos_snap)
        _ik_solved.add(obj_idx)

    def _run_ik_recommended_then_rrt(obj_idx, obj, q_start, obj_qpos_snap, rec):
        _run_ik_recommended(obj_idx, obj, obj_qpos_snap, rec)
        _run_rrt(q_start, obj['q_target'], obj)

    def _plan_thread_main(fn, *args):
        """Wrapper for every plan_thread target: an uncaught exception in _run_ik /
        _run_rrt must not die silently — before this, a failed solve left _plan_result
        empty and the finished-plan handler crashed the main loop on a missing
        'q_target' (every key, including target selection, then went dead)."""
        try:
            fn(*args)
        except Exception:
            traceback.print_exc()
            _plan_result['error'] = True

    # Build target list: index 0 = home pose, 1..N = one entry per object (label only —
    # q_target is not precomputed; it is solved lazily on first selection).
    targets = [{'label': 'init pose'}]
    for i in range(len(objects)):
        targets.append({'label': f'object {i+1}'})

    keys = queue.Queue()
    print("[Control] Ctrl+0..6: select target  |  ←→: jog x  |  ↑↓: jog z (lift)  |  PgUp/PgDn: jog y (depth)  |  Enter: GRASP / toggle squeeze  |  N: release  |  6: IK vis  |  7: coll spheres  |  Backspace: reset  |  Q/Esc: quit")
    print("[Control] Active target: init pose")

    # Simulation — start at Q_BIAS so PD error at t=0 is zero and qfrc_bias is correct.
    # All-zero initial qpos would produce a huge PD error (arm pointing straight up vs
    # HOME_ARM target) causing explosive qacc on the first step.
    mj.mj_resetData(model, data)
    data.qpos[:N_ROBOT] = Q_BIAS
    mj.mj_forward(model, data)

    # --- DexPilot controller (dexpilot mode + contact_aware_teleop wrist/fingers) ---
    # contact_aware_teleop reuses the SAME DexPilot wrist+finger retargeting stack;
    # it only adds the NLP grasp recommender and the lock-in->RRT->GRASP handoff on
    # top. The difference is hand_tracking: teleop needs real finger curling (the
    # MediaPipe joint angles drive the grasp), while pure dexpilot debugging holds
    # the fingers open.
    _teleop_modes   = ('dexpilot', 'contact_aware_teleop')
    _dexpilot_ctrl  = None
    _mediapipe_proc = None
    if args.mode in _teleop_modes:
        # Launch the MediaPipe publisher as a subprocess so its OpenCV window
        # appears alongside the MuJoCo viewer. The subprocess inherits the
        # current environment (CYCLONEDDS_URI, ROS sourcing, venv Python).
        _mp_cmd = [sys.executable,
                   'ui/mediapipe_joint_angles.py']
        if args.camera is not None:
            _mp_cmd += ['--camera', str(args.camera)]
        _mediapipe_proc = subprocess.Popen(_mp_cmd)
        print(f"[DexPilot] MediaPipe publisher launched (pid {_mediapipe_proc.pid})")

        from teleop.dexpilot_controller import DexPilotController
        from teleop.dexpilot_arm_controller import (load_camera_calibration,
                                                    load_teleop_config)
        # Use the measured ChArUco calibration (camera_extrinsics/intrinsics.json)
        # for the camera->robot rotation and pixel->metre scales. Falls back to a
        # bare identity mapping if the calibration files aren't present.
        try:
            # Z-up world remap: the publisher flips the board frame (camera looks
            # DOWN, board +Z points down) to MuJoCo Z-up via diag([1,-1,-1]).
            # Pass the SAME remap here so R_cam_robot (orientation) lives in the
            # same Z-up frame as the published position — otherwise position and
            # orientation disagree and the wrist rotation maps wrong.
            _world_from_board = np.diag([1.0, -1.0, -1.0])
            _cam_kwargs = load_camera_calibration(world_from_board=_world_from_board)
            # Position tunables from calibration/teleop_config.json (mode,
            # abs_scale, world_from_board). --position-mode CLI overrides the
            # config's mode when given. The publisher sends metric board-frame
            # wrist coords (absolute publish mode), which BOTH position modes
            # consume; only valid when calibration is present.
            _pos_cfg = load_teleop_config()
            if args.position_mode is not None:
                _pos_cfg["position_mode"] = args.position_mode
            _cam_kwargs.update(_pos_cfg)
            # DIRECT hand->wrist orientation: with the arm palm frame now built
            # from the stable IMAGE landmarks, the world->wrist rotation shown in
            # the MediaPipe overlay maps 1:1 to the MuJoCo target — no press-8
            # offset, no stale orientation_correction.json. Set False to re-enable
            # the press-8 auto-calibration.
            _cam_kwargs["identity_orientation"] = True
            print(f"[DexPilot] loaded camera calibration: "
                  f"scale_x={_cam_kwargs['scale_x']:.3f} scale_z={_cam_kwargs['scale_z']:.3f} "
                  f"| position={_cam_kwargs['position_mode']} abs_scale={_cam_kwargs['abs_scale']:.2f} "
                  f"| FULL 3-DOF IDENTITY orientation (direct mapping)")
        except FileNotFoundError:
            _cam_kwargs = {"R_cam_robot": np.eye(3), "position_mode": "legacy",
                           "identity_orientation": True}
            print("[DexPilot] no camera calibration found — using identity "
                  "R_cam_robot, LEGACY delta positioning. Run calibration/charuco_calibration.py.")
        # Palm-DOWN home for teleop: pinch_site palm NORMAL (+X) points down
        # (world -Z) and FINGERS (+Z) point FORWARD (world +X) — palm flat over
        # the table, fingers reaching away. Natural neutral (hold your palm down,
        # fingers forward, press 8). Solved via IK for pinch_site at ~(0.55,0,0.15)
        # — LOWERED from z=0.40 so the sim wrist STARTS near the table: relative
        # positioning then needs only ~0.15 m of descent, so the operator reaches
        # the sim table well within real vertical travel at modest abs_scale (the
        # old 0.40 m start forced high gain / was unreachable). Same orientation
        # (normal=world -Z, fingers=world +X); ori err 0.1deg, manip ~0.07, limited
        # joints 2/4/6 within range (margins >=0.5 rad).
        _HOME_WRIST_DOWN = np.array([-0.217, 1.144, 3.44, -2.011,
                                     -0.087, 1.541, 2.872])

        # Multi-pose orientation calibration: 4 distinct, IK-solved wrist
        # orientations (all reachable, within limits). During calibration (key M)
        # the robot is held at one of these while you match your hand and press C;
        # V solves the full rotation mapping. Fixes wrong RELATIVE rotations that
        # single-point (press-8) alignment can't. Orientations:
        #   1 palm-down fingers-fwd  2 palm-fwd fingers-up
        #   3 palm-left fingers-fwd  4 palm-down fingers-left
        _CALIB_POSES = [
            np.array([-0.271, 0.427, 3.5,  -2.193, 0.066, 1.034, 2.961]),
            np.array([ 0.564, 1.348, 2.089,-1.634, 1.35,  2.09,  0.952]),
            np.array([ 0.001, 0.417, 3.146,-2.183, 0.005, 1.029, 1.566]),
            np.array([ 0.161, 0.742, 3.418,-1.519, 1.667, 1.969, 2.296]),
        ]
        _calib_mode = False
        _calib_idx = -1
        data.qpos[:7] = _HOME_WRIST_DOWN
        data.qvel[:N_ROBOT] = 0.0
        mj.mj_forward(model, data)

        # Bias the arm IK toward the wrist-down home too, so the null-space pull
        # matches the new neutral (Q_BIAS still points at the old forward reach).
        _Q_BIAS_DP = Q_BIAS.copy()
        _Q_BIAS_DP[:7] = _HOME_WRIST_DOWN
        # debug=False silences the per-frame [retarget] print. hand_tracking=False
        # (pure dexpilot) holds the fingers OPEN so the hand orientation is easy to
        # read during orientation calibration; contact_aware_teleop needs real
        # finger curling (the MediaPipe joint angles ARE the grasp), so it tracks.
        _hand_tracking = (args.mode == 'contact_aware_teleop')
        _dexpilot_ctrl = DexPilotController(model, q_bias=_Q_BIAS_DP,
            debug=False, eps=0.005, hand_tracking=_hand_tracking, **_cam_kwargs)
        _dexpilot_ctrl.init_home(data)   # snapshots the wrist-down pose as home
        _dexpilot_ctrl.init_ros()
        print("[DexPilot] ROS subscriber active — waiting for /hand/joint_angles (≥120 floats)")
        print("[DexPilot] Press 8 to start tracking (captures your current wrist "
              "orientation as the robot's home). Q/Esc: quit")

    # --- contact_aware_teleop: NLP grasp recommender machinery ---
    # A per-object MultiStartGraspPlanner3D (built lazily, box-like first) continuously
    # recommends 2-finger contacts for whichever object the fingers are nearest. The
    # best candidate's p1(thumb)/p2(index) are shown live via the rec1/rec2 mocap
    # markers; pressing L locks them in and hands off to the existing IK->RRT->GRASP
    # machinery. The NLP's gamma seeds the squeeze, re-solved on the committed geometry.
    _CAT_MODE       = (args.mode == 'contact_aware_teleop')
    _REC_INTERVAL_S = 2.0     # fixed re-solve cadence (NLP solve ~0.5-2s, runs in a thread)
    _REC_NC         = 3       # planner seeds per solve
    _rec1_mocap = int(model.body_mocapid[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'rec1_body')])
    _rec2_mocap = int(model.body_mocapid[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'rec2_body')])
    _REC_HIDDEN = np.array([10.0, 10.0, 10.0])   # park markers off-scene when idle
    _cat_planners     = {}      # obj_idx -> MultiStartGraspPlanner3D (lazy)
    _cat_planner_lock = threading.Lock()
    _rec_thread       = None    # background recommender thread
    _rec_result       = {}      # {'candidate': {...}, 'obj_idx': int} written by the thread
    _rec_result_lock  = threading.Lock()
    _rec_last_solve   = 0.0     # wall-clock of the last solve start
    _rec_obj_idx      = -1      # object the latest recommendation is for

    # Objects the NLP recommender supports (box-like first, per the plan). The planner
    # is shape-aware but validated on boxes; extend this set as other shapes are proven.
    _CAT_SUPPORTED = {'obj_red_box', 'obj_green_box'}

    # Actuated-joint qpos indices (planner.solve wants q_ref as the nu-length actuated
    # vector, in actuator order) — same as GraspPlanner3D._act_idx.
    _cat_act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]]
                    for i in range(model.nu)]

    def _get_cat_planner(obj_idx):
        """Lazily build (and cache) a MultiStartGraspPlanner3D for one object."""
        with _cat_planner_lock:
            p = _cat_planners.get(obj_idx)
            if p is not None:
                return p
            o = objects[obj_idx]
            cfg = GraspConfig3D(obj_geom=o['name'] + '_geom', obj_body=o['name'],
                                max_iter=120)
            # Own MjData so the background solve never races the viewer's data.
            p = MultiStartGraspPlanner3D(model, mj.MjData(model), cfg)
            _cat_planners[obj_idx] = p
            return p

    def _fire_recommender(obj_idx, q_snap, obj_pos):
        """Start a background NLP solve for obj_idx; store the best candidate."""
        def _run():
            planner = _get_cat_planner(obj_idx)
            # The planner's own data must reflect the live object pose for its
            # collision/surface geometry; sync qpos before solving.
            planner._planner.data.qpos[:] = data.qpos[:]
            mj.mj_forward(model, planner._planner.data)
            _t0 = time.time()
            try:
                res = planner.solve(q_snap, obj_pos, max_seeds=_REC_NC)
            except Exception:
                traceback.print_exc()
                return
            _solve_ms = (time.time() - _t0) * 1e3
            if res.get('p1') is None or res.get('p2') is None:
                return
            with _rec_result_lock:
                _rec_result['candidate'] = {
                    'q':  np.asarray(res['q'], float).copy(),
                    'p1': np.asarray(res['p1'], float).copy(),   # thumb
                    'p2': np.asarray(res['p2'], float).copy(),   # index
                    'status': res.get('status'),
                }
                _rec_result['obj_idx'] = obj_idx

            # --- Push solve stats to the dashboard (verify() gives gamma_min + IK) ---
            if dash is not None:
                _all = res.get('all_results') or [res]
                _nconv = sum(1 for r in _all if r.get('status') == 'converged')
                _vinfo = {}
                try:
                    _vinfo = planner._planner.verify(res) or {}
                except Exception:
                    traceback.print_exc()
                dash.push({
                    'type':            'grasp_rec',
                    'object':          objects[obj_idx]['name'],
                    'status':          res.get('status', '?'),
                    'solve_ms':        _solve_ms,
                    'gamma_min':       _vinfo.get('gamma_min'),
                    'wrench_feasible': bool(_vinfo.get('wrench_feasible', False)),
                    'ik_thumb_mm':     _vinfo.get('ik_thumb_mm'),
                    'ik_index_mm':     _vinfo.get('ik_index_mm'),
                    'n_converged':     _nconv,
                    'n_seeds':         len(_all),
                })
        t = threading.Thread(target=_run, daemon=True, name='cat-recommender')
        t.start()
        return t

    def _recommended_inward_normals(obj_idx, p1, p2):
        """Outward->inward surface normals at p1/p2 for the object's live geom pose,
        using the planner's shape-aware _geom_normal_np. Returns (n1_in, n2_in)."""
        o = objects[obj_idx]
        gid = o['id_geom']
        gtype = int(model.geom_type[gid])
        c   = data.geom_xpos[gid].copy()
        R   = data.geom_xmat[gid].reshape(3, 3).copy()
        size = model.geom_size[gid]
        n1_out = _geom_normal_np(p1, gtype, c, R, size)
        n2_out = _geom_normal_np(p2, gtype, c, R, size)
        return -n1_out, -n2_out

    control_phase  = 'REACH'
    # contact_aware_teleop: True while the operator is teleoping + the NLP recommends
    # contacts (pre-lock-in). Lock-in (L) flips this False and hands off to the shared
    # IK->RRT->GRASP machinery; release (N) flips it back True. Always False otherwise.
    _teleop_active = _CAT_MODE
    active_idx     = 0
    active_tgt     = 0        # index into targets[]
    tau_ctrl       = np.zeros(model.nv)   # full nv for qfrc_applied
    traj_waypoints = [Q_BIAS.copy()]   # seed with home so REACH holds immediately
    traj_wp_idx    = 0
    traj_wp_step   = 0    # counts sim steps since last waypoint advance
    plan_thread    = None
    q_plan_hold    = np.zeros(N_ROBOT)    # robot DOFs only
    _plan_discard  = False         # reset arrived while planning: drop the result
    _last_sim_time = 0.0           # backstop reset detection (UI button / BADQACC)
    _held_arrows   = set()         # arrow keys currently held
    _ctrl_held     = set()         # Ctrl_L / Ctrl_R currently held
    squeeze_on     = False         # GRASP: internal force toggled by Enter
    _squeeze_steps = 0             # sim steps since squeeze-on, drives the force ramp
    grasp_ctrl     = None          # GraspController, built at each REACH→GRASP transition
    gamma_live     = GAMMA_FALLBACK  # per-object squeeze scale solved at that transition
    q_grasp_hold   = None          # GRASP PD target; arm part integrated by arrow-key jog
    _jog_v         = np.zeros(3)   # slew-rate-limited palm velocity command (world x,y,z)
    _PALM_BID      = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'leap_palm')
    _ik_vis_mode   = None          # None | 'grasp': freeze physics to show IK config
    _show_bspheres = False         # 7: overlay the IK's per-geom collision bounding spheres

    # Precomputed (geom_id, bounding-sphere radius) for every hand geom the IK constrains —
    # this is the coarse sphere model the IK actually "sees" (finger links 15-24mm), which
    # is why the fingers can't be both constrained and reach a small object. Toggle with B.
    _BSPHERES = [(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g), float(model.geom_rbound[
                    mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g)]))
                 for g in _robot_geom_names]

    # Dashboard streaming state: wall-clock t0 for the scrolling x-axis, a step counter
    # to throttle streaming pushes, and last-pushed mode/active-object (push on change).
    # _prox_idx is the proximity-based "active object" (min average tip distance) — it
    # drives the in-scene hover marker too, so it is updated even without --dashboard.
    _dash_t0        = time.time()
    _dash_i         = 0
    DASH_PUSH_EVERY = 3            # push streaming metrics every N loop iterations
    _dash_last_mode = None
    _dash_last_prox = None
    _prox_idx       = 0

    def _on_press(key):
        try:
            if key in (_pynput_kb.Key.ctrl_l, _pynput_kb.Key.ctrl_r):
                _ctrl_held.add(key)
            elif _ctrl_held:
                # Ctrl+digit: try char first, fall back to virtual key code.
                char = getattr(key, 'char', None)
                vk   = getattr(key, 'vk',  None)
                digit = None
                if char and char.isdigit():
                    digit = char
                elif vk is not None and 48 <= vk <= 57:
                    digit = str(vk - 48)
                if digit is not None:
                    keys.put(f'sel_{digit}')
            elif key == _pynput_kb.Key.right:     _held_arrows.add('right')
            elif key == _pynput_kb.Key.left:      _held_arrows.add('left')
            elif key == _pynput_kb.Key.up:        _held_arrows.add('up')
            elif key == _pynput_kb.Key.down:      _held_arrows.add('down')
            # PageUp/PageDown jog world +Y/-Y (depth, toward/away from the base) — the
            # out-of-plane axis the arrows don't cover. Handled via pynput like the
            # arrows (not the GLFW callback), so no collision with MuJoCo's viewer binds.
            elif key == _pynput_kb.Key.page_up:   _held_arrows.add('depth_fwd')
            elif key == _pynput_kb.Key.page_down: _held_arrows.add('depth_back')
        except AttributeError:
            pass

    def _on_release(key):
        try:
            if key in (_pynput_kb.Key.ctrl_l, _pynput_kb.Key.ctrl_r):
                _ctrl_held.discard(key)
            elif key == _pynput_kb.Key.right:     _held_arrows.discard('right')
            elif key == _pynput_kb.Key.left:      _held_arrows.discard('left')
            elif key == _pynput_kb.Key.up:        _held_arrows.discard('up')
            elif key == _pynput_kb.Key.down:      _held_arrows.discard('down')
            elif key == _pynput_kb.Key.page_up:   _held_arrows.discard('depth_fwd')
            elif key == _pynput_kb.Key.page_down: _held_arrows.discard('depth_back')
        except AttributeError:
            pass

    _kb_listener = _pynput_kb.Listener(on_press=_on_press, on_release=_on_release)
    _kb_listener.start()

    with mj.viewer.launch_passive(model, data, key_callback=make_key_callback(keys)) as viewer:
        viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
        if args.viz_only:
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            viewer.opt.label = mj.mjtLabel.mjLABEL_NONE
        else:
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            # viewer.opt.label = mj.mjtLabel.mjLABEL_CONTACTFORCE
            viewer.opt.label = mj.mjtLabel.mjLABEL_NONE

        def _draw_bspheres(scn):
            """Append the IK's collision bounding spheres (translucent) to the scene at the
            current pose — one per constrained hand geom, radius = model.geom_rbound. Lets
            you see the coarse sphere model the IK uses. No-op unless the B toggle is on."""
            if not _show_bspheres or scn is None:
                return
            _eye9 = np.eye(3, dtype=np.float64).flatten()
            rgba  = np.array([0.2, 0.6, 1.0, 0.25], dtype=np.float32)  # translucent blue
            for gid, rb in _BSPHERES:
                if scn.ngeom >= scn.maxgeom:
                    break
                mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_SPHERE,
                                np.array([rb, rb, rb]), data.geom_xpos[gid].copy(), _eye9, rgba)
                scn.ngeom += 1

        def _draw_active_marker(scn):
            """Translucent sphere hovering above the proximity-based 'active object'
            (min average fingertip signed distance, _prox_idx) so the viewer always
            shows which object the hand is currently nearest."""
            if scn is None or scn.ngeom >= scn.maxgeom:
                return
            o = objects[_prox_idx]
            pos = data.xpos[o['id_body']].copy()
            pos[2] += model.geom_rbound[o['id_geom']] + 0.06
            mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.022, 0.022, 0.022]), pos,
                            np.eye(3, dtype=np.float64).flatten(),
                            np.array([0.25, 1.0, 0.55, 0.4], dtype=np.float32))
            scn.ngeom += 1

        def _render_kinematic_frame():
            """Push ghost/IK markers to the scene and sync the viewer for one
            non-physics frame (no mj_step) — shared by REACH replay, REACH/GRASP
            final-pose hold in --viz-only mode, and PLAN hold in --viz-only mode."""
            with _ghost_markers_lock:
                markers_snapshot = list(_ghost_markers)
            scn = viewer.user_scn
            if scn is not None:
                scn.ngeom = 0
                _eye9 = np.eye(3, dtype=np.float64).flatten()
                _sz   = np.array([0.018, 0, 0], dtype=np.float64)
                for p0, p1, rgba in markers_snapshot:
                    if scn.ngeom >= scn.maxgeom: break
                    mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_CAPSULE,
                                    np.zeros(3), np.zeros(3), _eye9, rgba)
                    mj.mjv_connector(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_CAPSULE,
                                      0.004, p0, p1)
                    scn.ngeom += 1
                if active_tgt > 0 and _ik_markers_by_obj[active_idx] is not None:
                    for positions, rgba in _ik_markers_by_obj[active_idx]:
                        for pos in positions:
                            if scn.ngeom >= scn.maxgeom: break
                            mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_SPHERE,
                                            _sz, pos, _eye9, rgba)
                            scn.ngeom += 1
                _draw_bspheres(scn)
                _draw_active_marker(scn)
            viewer.sync()
            time.sleep(model.opt.timestep)

        running = True
        while viewer.is_running() and running:
            step_start = time.time()

            # --- contact_aware_teleop: teleop wrist+fingers, NLP recommends contacts.
            # Runs ONLY pre-lock-in; after L it falls through to the shared REACH/GRASP
            # machinery below (like autonomous mode) with recommended contacts. ---
            if _CAT_MODE and _teleop_active:
                _dexpilot_ctrl.spin()

                # Drain keys — quit, teleop start, and lock-in.
                _do_lock_in = False
                while not keys.empty():
                    _k = keys.get_nowait()
                    if _k == 'quit':
                        running = False
                    elif _k == 'teleop_start':
                        _dexpilot_ctrl.start(data)
                        print("[teleop] tracking started — home pose captured.")
                    elif _k == 'lock_in':
                        _do_lock_in = True
                if not running:
                    continue

                # Drive wrist (arm IK) + fingers (MediaPipe) kinematically.
                q_teleop = _dexpilot_ctrl.step(model, data)
                if q_teleop is not None:
                    data.qpos[:N_ROBOT] = q_teleop
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)

                # Proximity object (min average fingertip->object signed distance).
                _avg_d = [np.mean([_guarded_geom_dist(_tg, _o['id_geom'])
                                   for _tg in _ALL_TIP_GIDS]) for _o in objects]
                _prox_idx = int(np.argmin(_avg_d))

                # Fixed-interval background NLP recommend for the nearest SUPPORTED obj.
                _supported = objects[_prox_idx]['name'] in _CAT_SUPPORTED
                _rec_idle  = (_rec_thread is None) or (not _rec_thread.is_alive())
                if (_supported and _rec_idle
                        and (time.time() - _rec_last_solve) >= _REC_INTERVAL_S):
                    _q_snap = np.array([data.qpos[i] for i in _cat_act_idx])
                    _obj_pos = data.xpos[objects[_prox_idx]['id_body']].copy()
                    _rec_thread = _fire_recommender(_prox_idx, _q_snap, _obj_pos)
                    _rec_last_solve = time.time()

                # Show the latest recommendation via the rec1/rec2 markers.
                _cand = None
                with _rec_result_lock:
                    if (_rec_result.get('candidate') is not None
                            and _rec_result.get('obj_idx') == _prox_idx):
                        _cand = _rec_result['candidate']
                        _rec_obj_idx = _prox_idx
                if _cand is not None:
                    data.mocap_pos[_rec1_mocap] = _cand['p1']
                    data.mocap_pos[_rec2_mocap] = _cand['p2']
                else:
                    data.mocap_pos[_rec1_mocap] = _REC_HIDDEN
                    data.mocap_pos[_rec2_mocap] = _REC_HIDDEN

                # Lock-in: snapshot the current recommendation, hand off to IK->RRT.
                if _do_lock_in:
                    if _cand is None:
                        print("[teleop] no recommendation yet for the nearest "
                              "supported object — hold near a box and wait.")
                    elif plan_thread is not None:
                        print("[teleop] still planning — lock-in ignored.")
                    else:
                        active_idx = _prox_idx
                        active_tgt = _prox_idx + 1        # targets[0] is home
                        _teleop_active = False
                        data.mocap_pos[_rec1_mocap] = _REC_HIDDEN
                        data.mocap_pos[_rec2_mocap] = _REC_HIDDEN
                        q_start = data.qpos[:N_ROBOT].copy()
                        q_plan_hold = q_start.copy()
                        _plan_result.clear()
                        obj_qpos_snap = data.qpos[N_ROBOT:].copy()
                        with _ghost_markers_lock:
                            _ghost_markers.clear()
                        control_phase = 'PLAN'
                        plan_thread = threading.Thread(
                            target=_plan_thread_main,
                            args=(_run_ik_recommended_then_rrt, active_idx,
                                  objects[active_idx], q_start, obj_qpos_snap, _cand),
                            daemon=True)
                        plan_thread.start()
                        print(f"[teleop] LOCK-IN {objects[active_idx]['name']} — "
                              f"solving IK + RRT to recommended contacts ...")

                # --- Dashboard streams (same panels as autonomous mode) ---
                if dash is not None:
                    _mode = 'Locking in' if not _teleop_active else 'Teleop'
                    if (_mode, active_tgt) != _dash_last_mode:
                        dash.push({'type': 'mode', 'mode': _mode,
                                   'target': objects[_prox_idx]['name']})
                        _dash_last_mode = (_mode, active_tgt)
                    if _prox_idx != _dash_last_prox:
                        dash.push({'type': 'active_obj',
                                   'name': objects[_prox_idx]['name']})
                        _dash_last_prox = _prox_idx
                    if _dash_i % DASH_PUSH_EVERY == 0:
                        _t = time.time() - _dash_t0
                        _ogid = objects[_prox_idx]['id_geom']
                        _dvals = {f: _guarded_geom_dist(_TIP_GEOM_IDS[f], _ogid)
                                  for f in FINGER_SET}
                        dash.push({'type': 'dist', 't': _t, 'd': _dvals})
                        _f_net, _tau_net, _normals, _ = _hand_object_contact_metrics(_prox_idx)
                        dash.push({'type': 'wrench', 't': _t,
                                   'f': _f_net.tolist(), 'tau': _tau_net.tolist()})
                        dash.push({'type': 'normals', 't': _t, 'n': _normals})
                _dash_i += 1

                # Viz: recommendation markers already set; render one kinematic frame.
                if viewer.user_scn is not None:
                    viewer.user_scn.ngeom = 0
                    _draw_active_marker(viewer.user_scn)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
                continue

            # --- DexPilot teleop mode: bypass the RRT/grasp state machine ---
            if args.mode == 'dexpilot':
                _dexpilot_ctrl.spin()
                # Drain key queue — handle quit and teleop start/re-zero
                while not keys.empty():
                    _k = keys.get_nowait()
                    if _k == 'quit':
                        running = False
                    elif _k == 'teleop_start':
                        # Snapshot current human pose as home and begin tracking.
                        _dexpilot_ctrl.start(data)
                        print("[dexpilot] tracking started — home pose captured "
                              "(hold your hand at the desired neutral orientation).")
                    elif _k == 'calib_orient':
                        # Hold hand to MATCH the robot wrist, then press 9 to
                        # capture the constant orientation correction.
                        _dexpilot_ctrl.calibrate_orientation(data)
                    elif _k == 'calib_next':
                        # Enter/advance multi-pose calibration: pose the robot to
                        # the next fixed orientation and HOLD it (tracking paused)
                        # so you can match your hand to it, then press C.
                        _calib_mode = True
                        _calib_idx = (_calib_idx + 1) % len(_CALIB_POSES)
                        data.qpos[:7] = _CALIB_POSES[_calib_idx]
                        data.qvel[:N_ROBOT] = 0.0
                        mj.mj_forward(model, data)
                        print(f"[dexpilot] calib pose {_calib_idx+1}/{len(_CALIB_POSES)} "
                              f"— MATCH your hand to the wrist, then press C to capture "
                              f"(M=next pose, V=solve).")
                    elif _k == 'calib_capture':
                        _dexpilot_ctrl.capture_calib_pose(data)
                    elif _k == 'calib_solve':
                        _dexpilot_ctrl.solve_calib()
                        _calib_mode = False
                        print("[dexpilot] calibration solved & applied. Resuming tracking.")
                if not running:
                    continue
                # In calibration mode, HOLD the posed orientation (tracking paused)
                # so the robot doesn't chase the hand while you match it.
                if _calib_mode:
                    data.qpos[:7] = _CALIB_POSES[_calib_idx]
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
                    # Still run step() so _last_raw (hand mapping) updates for capture,
                    # but discard its qpos output.
                    _dexpilot_ctrl.step(model, data)
                    data.qpos[:7] = _CALIB_POSES[_calib_idx]
                    mj.mj_forward(model, data)
                else:
                    q_teleop = _dexpilot_ctrl.step(model, data)
                    if q_teleop is not None:
                        data.qpos[:N_ROBOT] = q_teleop
                        data.qvel[:N_ROBOT] = 0.0
                        mj.mj_forward(model, data)
                # Visualise BOTH the IK target frame (thick, the pose the arm IK
                # drives pinch_site toward) AND the robot's CURRENT pinch_site
                # frame (thin). The GAP between them is the live IK error —
                # position gap = translation error, axis mismatch = rotation
                # error. X=red Y=green Z=blue for both.
                scn = viewer.user_scn
                if scn is not None:
                    scn.ngeom = 0

                    def _draw_frame(pos, Rm, alen, arad):
                        cols = [np.array([1., 0, 0, 1.]),
                                np.array([0, 1., 0, 1.]),
                                np.array([0, 0, 1., 1.])]
                        for _i, _rgba in enumerate(cols):
                            if scn.ngeom >= scn.maxgeom:
                                break
                            mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_CAPSULE,
                                            np.zeros(3), np.zeros(3),
                                            np.eye(3).flatten(), _rgba)
                            mj.mjv_connector(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_CAPSULE,
                                             arad, pos, pos + alen * Rm[:, _i])
                            scn.ngeom += 1

                    _psid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'pinch_site')
                    _wrist_pos = data.site_xpos[_psid].copy()
                    _tf = _dexpilot_ctrl.target_frame()
                    if _tf is not None:
                        # Target orientation drawn at the TARGET position (thick).
                        _draw_frame(_tf[0], _tf[1], 0.10, 0.007)
                        # ALSO draw the target ORIENTATION at the actual wrist
                        # position (medium) so you can compare pure orientation
                        # without the position offset making it look 'wrong'.
                        _draw_frame(_wrist_pos, _tf[1], 0.08, 0.005)
                    # current pinch_site frame (thin, short)
                    _draw_frame(_wrist_pos,
                                data.site_xmat[_psid].reshape(3, 3).copy(),
                                0.06, 0.004)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
                continue

            # --- Proximity "active object" (min average tip→object signed distance) +
            # dashboard streams: planning mode, distances, net wrench, normal forces ---
            if _dash_i % DASH_PUSH_EVERY == 0:
                _avg_d = [np.mean([_guarded_geom_dist(_tg, _o['id_geom'])
                                   for _tg in _ALL_TIP_GIDS]) for _o in objects]
                _prox_idx = int(np.argmin(_avg_d))
            if dash is not None:
                # Display terminology: PLAN/REACH → Approach; GRASP → Grasp, or
                # Transport while the object is being jogged (arrow keys held).
                _mode = ('Approach' if control_phase in ('PLAN', 'REACH')
                         else ('Transport' if _held_arrows else 'Grasp'))
                if (_mode, active_tgt) != _dash_last_mode:
                    dash.push({'type': 'mode', 'mode': _mode,
                               'target': targets[active_tgt]['label']})
                    _dash_last_mode = (_mode, active_tgt)
                if _prox_idx != _dash_last_prox:
                    dash.push({'type': 'active_obj', 'name': objects[_prox_idx]['name']})
                    _dash_last_prox = _prox_idx
                if _dash_i % DASH_PUSH_EVERY == 0:
                    _t = time.time() - _dash_t0
                    # distmax=2.0 (scene ~1 m): distances above this clamp to 2.0, so the
                    # cap is set high enough to show most of the reach, not just contact.
                    _ogid = objects[_prox_idx]['id_geom']
                    _dvals = {f: _guarded_geom_dist(_TIP_GEOM_IDS[f], _ogid)
                              for f in FINGER_SET}
                    dash.push({'type': 'dist', 't': _t, 'd': _dvals})
                    _f_net, _tau_net, _normals, _ = _hand_object_contact_metrics(_prox_idx)
                    dash.push({'type': 'wrench', 't': _t,
                               'f': _f_net.tolist(), 'tau': _tau_net.tolist()})
                    dash.push({'type': 'normals', 't': _t, 'n': _normals})
            _dash_i += 1

            # --- Detect out-of-band resets: the viewer UI's Reset button and MuJoCo's
            # BADQACC warning both mj_resetData outside our control (no key event),
            # snapping qpos to qpos0 — the arm's compiled zero is the straight-up pose,
            # NOT Q_BIAS — while the controller keeps pulling toward its pre-reset
            # target. data.time jumping backwards is the fingerprint; route it through
            # the same recovery as an explicit Backspace reset.
            if data.time < _last_sim_time - 1e-12:
                keys.put('reset')
            _last_sim_time = data.time

            # --- Check if background RRT finished ---
            if plan_thread is not None and not plan_thread.is_alive():
                plan_thread = None
                if _plan_discard:
                    # Plan was started before a reset — its start pose no longer
                    # matches the arm, so replaying it would teleport the arm through
                    # the old path. Stay in the post-reset home hold.
                    _plan_discard = False
                    with _ghost_markers_lock:   # thread repopulated ghosts on finish
                        _ghost_markers.clear()
                    print("\r\n[Control] stale plan discarded (reset during planning)")
                else:
                    if 'waypoints' in _plan_result:
                        traj_waypoints = _plan_result['waypoints']
                        print(f"\r\n[Control] REACH  |  path: {len(traj_waypoints)} waypoints")
                    else:
                        # Planning died (see _plan_thread_main traceback) — hold the pose
                        # we were already holding so the sim stays alive; reselect to retry.
                        traj_waypoints = [q_plan_hold.copy()]
                        print("\r\n[Control] REACH  |  planning FAILED — holding pose "
                              "(Ctrl+digit to retry)")
                    traj_wp_idx    = 0
                    traj_wp_step   = 0
                    control_phase  = 'REACH'

            # --- Process key events ---
            _reset_req = False
            while not keys.empty():
                key = keys.get_nowait()

                if key == 'reset':
                    # One physical reset can arrive twice (Backspace key callback AND
                    # the time-jump backstop above) — coalesce, handle after the drain.
                    _reset_req = True

                elif key == 'enter' and control_phase == 'REACH':
                    control_phase = 'GRASP'
                    squeeze_on = False
                    # Object pose frozen at grasp time — reference anchor for the
                    # upcoming w_des (object wrench) tracking; unused by the jog.
                    obj_grasp = objects[active_idx]
                    _bid = obj_grasp['id_body']
                    p_WoO = data.xpos[_bid].copy()
                    R_WO  = data.xmat[_bid].reshape(3, 3).copy()
                    obj_grasp['p_obj0'] = p_WoO
                    obj_grasp['R_obj0'] = R_WO
                    _jog_v[:] = 0.0    # reset the jog velocity ramp for the new grasp

                    # --- Solve the internal-force scale gamma for THIS grasp geometry ---
                    # Contact geometry in the object body frame (same transform the
                    # GraspController's internal_force_torques uses): col0 of R_OSk is the
                    # inward normal per the scene XML convention.
                    # contact_aware_teleop stores object-LOCAL recommended frames on
                    # obj['rec_local']; autonomous mode reads the authored contact sites.
                    # Gamma is re-solved here on the COMMITTED geometry either way.
                    _rec_local = obj_grasp.get('rec_local')
                    _teleop_grasp = _rec_local is not None
                    if _teleop_grasp:
                        _p_O   = [p_O.copy() for (p_O, _R_O) in _rec_local]
                        _R_in  = [R_O.copy() for (_p_O, R_O) in _rec_local]
                    else:
                        _p_O   = [R_WO.T @ (data.site_xpos[s] - p_WoO) for s in obj_grasp['id_S']]
                        _R_in  = [R_WO.T @ data.site_xmat[s].reshape(3, 3) for s in obj_grasp['id_S']]
                    _mu    = [float(model.geom_friction[obj_grasp['id_geom'], 0])] * N_FINGERS
                    _mass  = float(model.body_mass[_bid])
                    # Gravity component per OBJECT-body axis, added to the linear budget.
                    _g_O   = R_WO.T @ model.opt.gravity           # (3,), object frame
                    _accel_box = tuple(NCF_ACCEL_BUDGET_XYZ[i] + abs(_g_O[i]) for i in range(3))
                    _inertia   = model.body_inertia[_bid]         # principal moments (Ix,Iy,Iz)
                    # A 2-contact antipodal pinch geometrically CANNOT resist torque about
                    # the grasp axis (the line through the two contacts) — the friction
                    # cones have no moment arm about it, so any nonzero angular budget on
                    # that axis makes the LP infeasible (matches the Tx=None antipodal case
                    # in 3D_minimum_NCF). Zero the angular budget's grasp-axis component so
                    # the solve stays feasible; resisting that DOF needs a 3rd contact or
                    # soft-finger torsion (deferred).
                    # (Assumes the object's principal-inertia frame ≈ its body frame, so
                    # the body-frame grasp axis aligns with the principal-frame angular
                    # budget — true for the current objects, whose body_iquat is identity.)
                    _grasp_axis = _p_O[0] - _p_O[1]
                    _grasp_axis = _grasp_axis / (np.linalg.norm(_grasp_axis) + 1e-12)
                    _ang_budget = np.array(NCF_ANG_ACCEL_BUDGET, float)
                    _ang_budget -= np.dot(_ang_budget, _grasp_axis) * _grasp_axis
                    _ang_budget = np.abs(_ang_budget)   # per-axis magnitudes for the box
                    _gamma = solve_gamma_live(_p_O, _R_in, _mu, _mass,
                                              _accel_box, tuple(_ang_budget), _inertia)
                    if _gamma is None or not np.isfinite(_gamma) or _gamma <= 0.0:
                        gamma_raw  = GAMMA_FALLBACK      # cone viz uses this too
                        gamma_live = GAMMA_FALLBACK
                        print(f"\r\n[gamma] LP infeasible/degenerate for "
                              f"{obj_grasp['name']} — using fallback {GAMMA_FALLBACK:.0f}")
                    else:
                        # gamma_raw = the LP's minimum no-slip gamma (drives the cone viz,
                        # the true feasible boundary). gamma_live = raw * safety factor is
                        # what actually squeezes.
                        gamma_raw  = float(_gamma)
                        gamma_live = gamma_raw * GAMMA_SAFETY_FACTOR
                        print(f"\r\n[gamma] {obj_grasp['name']}: solved gamma={gamma_raw:.2f} "
                              f"x{GAMMA_SAFETY_FACTOR:.1f} = {gamma_live:.2f} "
                              f"(mass={_mass:.3f}kg mu={_mu[0]:.1f}, "
                              f"~{gamma_live/np.sqrt(2):.2f} N/contact)")

                    # Internal-force machinery for the Enter-toggled squeeze. Only
                    # internal_force_torques() is used — the joint PD hold stays in the
                    # GRASP branch below, on top of the shared bias comp.
                    # Teleop: the recommended contacts have no sites, so pass a live
                    # provider that re-expresses the stored object-LOCAL frames in the
                    # world each step (tracks the moving object). Autonomous: sites.
                    if _teleop_grasp:
                        _rec_local_snap = [(p_O.copy(), R_O.copy())
                                           for (p_O, R_O) in _rec_local]
                        _prov_bid = obj_grasp['id_body']

                        def _make_provider(rec_local, bid):
                            def _provider(d):
                                p_WoO_l = d.xpos[bid]
                                R_WO_l  = d.xmat[bid].reshape(3, 3)
                                return [(p_WoO_l + R_WO_l @ p_O, R_WO_l @ R_O)
                                        for (p_O, R_O) in rec_local]
                            return _provider

                        _grasp_provider = _make_provider(_rec_local_snap, _prov_bid)
                        _grasp_sites    = None
                    else:
                        _grasp_provider = None
                        _grasp_sites    = obj_grasp['id_S']
                    grasp_ctrl = GraspController(
                        model, N_ROBOT,
                        tip_site_ids=id_C,
                        obj_site_ids=_grasp_sites,
                        obj_body_id=obj_grasp['id_body'],
                        kp=Kp, kd=Kd, gamma=gamma_live,
                        squeeze_pd_scale=SQUEEZE_PD_SCALE,
                        support_weight=True,
                        pad_offsets=[_PAD_OFFSET[f] for f in FINGER_SET],
                        obj_contact_provider=_grasp_provider)
                    q_grasp_hold = obj_grasp['q_target'].copy()
                    grasp_ctrl.set_squeeze(False)
                    _push_squeeze(False, gamma_live)
                    # Show the feasible wrench set at the RAW (1.0x) gamma — the true
                    # boundary the LP computed — behind the live trace in the 3D panels.
                    # The controller squeezes at the safety-scaled gamma_live, so the
                    # applied wrench sits comfortably inside this cage.
                    _push_wrench_cone(gamma_raw, _p_O, _R_in, _mu)
                    print(f"\r\n[Control] → GRASP  ({targets[active_tgt]['label']})  "
                          f"|  Enter: toggle squeeze (gamma={gamma_live:.1f})  |  N: release")

                elif key == 'enter' and control_phase == 'GRASP':
                    squeeze_on = not squeeze_on
                    _squeeze_steps = 0   # restart the force ramp on every toggle-on
                    grasp_ctrl.set_squeeze(squeeze_on)
                    _push_squeeze(squeeze_on, gamma_live)
                    print(f"\r\n[Control] squeeze {'ON' if squeeze_on else 'off'}  "
                          f"(gamma={gamma_live:.1f}, ~{gamma_live/np.sqrt(2):.2f} N/contact)")

                elif key == 'release' and control_phase == 'GRASP':
                    # Release: no pregrasp config exists anymore, so open the active
                    # fingers back to their Q_BIAS posture while the arm stays at the
                    # (possibly jogged) grasp arm config q_grasp_hold — snapping back
                    # to the original IK pose would drag the object with it.
                    squeeze_on     = False
                    grasp_ctrl.set_squeeze(False)
                    _push_squeeze(False, gamma_live)
                    _push_wrench_cone(None, None, None, None)   # clear the cone meshes
                    if _CAT_MODE:
                        # Teleop: hand control back to the operator (all 23 DOFs are
                        # teleoped), drop the object, and re-arm the recommender. Clear
                        # the committed contacts so the next lock-in re-solves fresh.
                        objects[active_idx].pop('rec_local', None)
                        _teleop_active = True
                        active_tgt     = 0
                        active_idx     = 0
                        _rec_last_solve = 0.0
                        with _rec_result_lock:
                            _rec_result.clear()
                        control_phase  = 'REACH'
                        print("\r\n[teleop] released — operator back in control, "
                              "recommender re-armed.")
                    else:
                        # Release: no pregrasp config exists anymore, so open the active
                        # fingers back to their Q_BIAS posture while the arm stays at the
                        # (possibly jogged) grasp arm config q_grasp_hold — snapping back
                        # to the original IK pose would drag the object with it.
                        q_release      = q_grasp_hold.copy()
                        q_release[7:]  = Q_BIAS[7:]
                        traj_waypoints = [q_release]
                        traj_wp_idx    = 0
                        traj_wp_step   = 0
                        control_phase  = 'REACH'
                        print(f"\r\n[Control] → REACH  (released — opening fingers)")

                elif key == 'ik_vis' and active_tgt > 0 and active_idx in _ik_solved:
                    _ik_vis_mode = {None: 'grasp', 'grasp': None}[_ik_vis_mode]
                    label = f'showing {_ik_vis_mode} config' if _ik_vis_mode else 'off'
                    print(f"\r\n[IK vis] {label}  (6 to toggle)")

                elif key == 'bspheres':
                    _show_bspheres = not _show_bspheres
                    print(f"\r\n[bspheres] IK collision bounding-sphere overlay "
                          f"{'ON' if _show_bspheres else 'off'}  (7 to toggle)")

                elif key.startswith('sel_') and control_phase == 'GRASP':
                    print("\r\n[Control] target selection blocked during GRASP — "
                          "press N to release first")

                elif key.startswith('sel_') and plan_thread is not None:
                    # One plan at a time: the IK/RRT threads share _ik_data, the
                    # constrained solver, and planner._data — a second concurrent
                    # thread corrupts the running solve.
                    print("\r\n[Control] still planning — selection ignored, retry "
                          "when REACH starts")

                elif key.startswith('sel_'):
                    # Re-selecting the CURRENT target is allowed: it re-plans from the
                    # live scene (IK staleness check below re-solves if objects moved).
                    new_tgt = int(key[4:])
                    if 0 <= new_tgt < len(targets):
                        active_tgt = new_tgt
                        active_idx = max(0, active_tgt - 1)  # map back to objects[]
                        _ik_vis_mode = None   # exit vis mode when switching target
                        q_start = data.qpos[:N_ROBOT].copy()
                        with _ghost_markers_lock:   # clear stale ghosts while planning
                            _ghost_markers.clear()
                        if active_tgt == 0:
                            traj_waypoints = [Q_BIAS.copy()]
                            traj_wp_idx    = 0
                            traj_wp_step   = 0
                            control_phase  = 'REACH'
                        else:
                            obj_i = active_idx
                            _plan_result.clear()
                            q_plan_hold    = q_start.copy()
                            obj_qpos_snap  = data.qpos[N_ROBOT:].copy()
                            # Cached IK is valid only while the scene still matches the
                            # snapshot it was solved against — the targets AND the
                            # collision constraints both depend on object poses, and any
                            # object (not just the target) may have moved into or out of
                            # the way since. mj_step never advances during PLAN/REACH
                            # holds beyond settling jitter, so a component-wise qpos
                            # comparison against IK_STALE_TOL is a stable test.
                            _ik_fresh = (obj_i in _ik_solved
                                         and np.max(np.abs(
                                             obj_qpos_snap
                                             - objects[obj_i]['ik_obj_qpos'])) < IK_STALE_TOL)
                            if _ik_fresh:
                                # IK cached and scene unchanged — go straight to RRT
                                plan_thread = threading.Thread(
                                    target=_plan_thread_main,
                                    args=(_run_rrt, q_start,
                                          objects[obj_i]['q_target'], objects[obj_i]),
                                    daemon=True)
                            else:
                                # IK missing or stale (object moved) — solve, then RRT
                                plan_thread = threading.Thread(
                                    target=_plan_thread_main,
                                    args=(_run_ik_then_rrt, obj_i, objects[obj_i],
                                          q_start, obj_qpos_snap),
                                    daemon=True)
                            plan_thread.start()
                            control_phase = 'PLAN'
                        print(f"\r\n[Control] → {targets[active_tgt]['label']}")

                elif key == 'quit':
                    running = False

            # --- Reset recovery: redo the viewer's reset on OUR terms and re-home the
            # state machine. The viewer's built-in Reset already snapped qpos to qpos0,
            # but the compiled arm zero is the straight-up pose and the controller
            # would keep pulling toward its pre-reset target — exactly the explosion
            # this handler exists to prevent. mj_resetData puts the objects back at
            # their randomized spawn poses (qpos0, maintained by _randomize_objects);
            # cached IK solutions are kept — the selection handler's IK_STALE_TOL check
            # re-solves automatically if the restored scene differs from the snapshot a
            # cache entry was solved against.
            if _reset_req:
                if plan_thread is not None:
                    _plan_discard = True   # in-flight IK/RRT started pre-reset: drop it
                mj.mj_resetData(model, data)
                data.qpos[:N_ROBOT] = Q_BIAS   # arm/hand home (qpos0 zero = straight up)
                mj.mj_forward(model, data)
                control_phase  = 'REACH'
                active_tgt     = 0
                active_idx     = 0
                traj_waypoints = [Q_BIAS.copy()]
                traj_wp_idx    = 0
                traj_wp_step   = 0
                squeeze_on     = False
                grasp_ctrl     = None
                gamma_live     = GAMMA_FALLBACK
                q_grasp_hold   = None
                _jog_v[:]      = 0.0
                _ik_vis_mode   = None
                tau_ctrl       = np.zeros(model.nv)
                _last_sim_time = 0.0
                with _ghost_markers_lock:
                    _ghost_markers.clear()
                _push_squeeze(False, gamma_live)
                _push_wrench_cone(None, None, None, None)   # clear the cone meshes
                if _CAT_MODE:
                    # Return to teleop control and re-arm the recommender.
                    for _o in objects:
                        _o.pop('rec_local', None)
                    _teleop_active  = True
                    _rec_last_solve = 0.0
                    with _rec_result_lock:
                        _rec_result.clear()
                    data.mocap_pos[_rec1_mocap] = _REC_HIDDEN
                    data.mocap_pos[_rec2_mocap] = _REC_HIDDEN
                print("\r\n[Control] RESET — arm home, objects at spawn poses; cached "
                      "IK kept (auto re-solved if stale on next selection)")

            # --- Continuous jog: world-frame palm velocity from currently-held arrow
            # keys, consumed by the GRASP branch's resolved-rate target integration.
            # left/right -> x, up/down -> z (lift), PageUp/PageDown -> y (depth).
            _jv_x = (JOG_VEL if 'right' in _held_arrows else 0) - (JOG_VEL if 'left' in _held_arrows else 0)
            _jv_z = (JOG_VEL if 'up'    in _held_arrows else 0) - (JOG_VEL if 'down' in _held_arrows else 0)
            _jv_y = (JOG_VEL if 'depth_fwd' in _held_arrows else 0) - (JOG_VEL if 'depth_back' in _held_arrows else 0)

            # --- IK visualization: freeze physics, show full arm in stored IK config ---
            if _ik_vis_mode is not None and active_tgt > 0:
                q_vis = objects[active_idx]['q_target']
                data.qpos[:N_ROBOT] = q_vis
                data.qvel[:N_ROBOT] = 0.0
                mj.mj_forward(model, data)
                if viewer.user_scn is not None:
                    viewer.user_scn.ngeom = 0   # suppress ghost markers; arm pose is the vis
                    _draw_bspheres(viewer.user_scn)
                    _draw_active_marker(viewer.user_scn)
                viewer.sync()
                time.sleep(model.opt.timestep)
                continue

            obj = objects[active_idx]

            # --- PLAN: hold position while RRT runs in background ---
            if control_phase == 'PLAN':
                if args.viz_only:
                    data.qpos[:N_ROBOT] = q_plan_hold
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
                    _render_kinematic_frame()
                    continue
                # Zero qvel each step, same as the REACH final-waypoint hold. The Kd term
                # goes through qfrc_applied (explicit), whose stability limit dt < 2*I/Kd
                # sits right at the 1ms timestep for the wrist (and far below it for the
                # fingers) — without this, holding at a pregrasp posture diverges at the
                # wrist DOF within ~30ms (observed x1.77/step growth, no contacts), trips
                # MuJoCo's BADQACC auto-reset to qpos0, and the hold then explodes against
                # the huge post-reset PD error for as long as planning runs.
                data.qvel[:N_ROBOT] = 0.0
                tau_ctrl = np.zeros(model.nv)
                tau_ctrl[:N_ROBOT] = Kp * (q_plan_hold - data.qpos[:N_ROBOT]) + Kd * (0 - data.qvel[:N_ROBOT])

            # --- REACH: kinematic replay while traversing; physics hold at final wp
            # (or, in --viz-only, kinematic hold forever) ---
            elif control_phase == 'REACH':
                if traj_waypoints and traj_wp_idx < len(traj_waypoints) - 1:
                    # Kinematic mode: set qpos directly to sidestep arm→finger inertial
                    # coupling that causes QACC explosions when the arm moves large angles.
                    wp = traj_waypoints[traj_wp_idx]
                    data.qpos[:N_ROBOT] = wp
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
                    traj_wp_step += 1
                    if traj_wp_step >= STEPS_PER_WP:
                        traj_wp_step = 0
                        traj_wp_idx += 1
                    _render_kinematic_frame()
                    continue   # skip physics mj_step at bottom of loop

                elif args.viz_only:
                    # Final waypoint reached: keep holding it kinematically — never
                    # hand off to physics, so the IK solution can be inspected as-is.
                    wp = traj_waypoints[-1] if traj_waypoints else Q_BIAS.copy()
                    data.qpos[:N_ROBOT] = wp
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
                    _render_kinematic_frame()
                    continue

                else:
                    # Last waypoint reached: switch to physics PD hold.
                    wp = traj_waypoints[-1] if traj_waypoints else Q_BIAS.copy()
                    data.qvel[:N_ROBOT] = 0.0   # damp out kinematic-replay residual velocity
                    tau_ctrl = np.zeros(model.nv)
                    tau_ctrl[:N_ROBOT] = Kp * (wp - data.qpos[:N_ROBOT]) + Kd * (0 - data.qvel[:N_ROBOT])

            # --- GRASP: grasp_controller_demo strategy — quasi-static joint-space PD
            # hold of the grasp IK config (gravity comp added at the shared mj_step
            # handoff below), plus a pure internal (pinching) force
            # f_c = null(G) @ GAMMA toggled with Enter, mapped through the fingertip
            # Jacobians. The full-rank joint PD stays on throughout — a Cartesian-only
            # version lived here before but was unstable: the fingertip J^T forces
            # leave the arm's ~17-dim contact null space unstiffened/undamped (see git
            # history). Arrow-key jogging translates the whole grasp by resolved-rate
            # motion of the arm PD target (palm velocity tracked, orientation held).
            # (--viz-only: kinematic hold of the grasp IK pose.) ---
            elif control_phase == 'GRASP':
                if args.viz_only:
                    data.qpos[:N_ROBOT] = obj['q_target']
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
                    _render_kinematic_frame()
                    continue
                # Slew-rate limit the commanded palm velocity toward the raw arrow-key
                # target at NCF_ACCEL_BUDGET_XYZ, per world axis. This caps the commanded
                # palm ACCELERATION at exactly the budget the gamma solve assumed — so
                # pressing/releasing an arrow ramps 0<->JOG_VEL over ~JOG_VEL/budget
                # seconds instead of stepping instantly (an unbounded accel that the old
                # code left the arm inertia + Kp to absorb). The disturbance box is thus
                # enforced by construction, and gamma covers it at 1.0x margin.
                _v_target = np.array([_jv_x, _jv_y, _jv_z])
                _dv_max   = np.array(NCF_ACCEL_BUDGET_XYZ) * model.opt.timestep
                _jog_v   += np.clip(_v_target - _jog_v, -_dv_max, _dv_max)

                # Resolved-rate jog: map the (rate-limited) world-frame palm velocity
                # [vx, vy, vz] (orientation held) to arm joint rates via 6-DOF DLS on
                # the live config, and integrate them into the PD target.
                qdot_jog = np.zeros(7)
                if np.any(_jog_v):
                    Jp = np.zeros((3, model.nv))
                    Jr = np.zeros((3, model.nv))
                    mj.mj_jacBody(model, data, Jp, Jr, _PALM_BID)
                    J6 = np.vstack([Jp[:, :7], Jr[:, :7]])
                    v6 = np.array([_jog_v[0], _jog_v[1], _jog_v[2], 0.0, 0.0, 0.0])
                    # Singularity-robust DLS: the damping lambda^2 grows as the palm
                    # Jacobian's smallest singular value falls below JOG_SING_EPS,
                    # capping the joint-rate gain at ~1/(2*lambda_max) near singular
                    # grasp configs instead of letting it spike to ~1/sigma_min and
                    # lurch the arm off the object (the mechanism that threw the cube:
                    # a runaway qdot_jog written into qvel below drags the fixed-angle
                    # fingers across the object faster than friction can hold it).
                    # Depth (y) jogs approach the arm's reach limits fastest, so this
                    # matters most for the new PageUp/PageDown axis.
                    _sigma_min = np.linalg.svd(J6, compute_uv=False)[-1]
                    _lam2 = (0.0 if _sigma_min >= JOG_SING_EPS
                             else (1.0 - (_sigma_min / JOG_SING_EPS) ** 2) * JOG_LAM_MAX ** 2)
                    qdot_jog = J6.T @ np.linalg.solve(J6 @ J6.T + _lam2 * np.eye(6), v6)
                    q_grasp_hold[:7] += qdot_jog * model.opt.timestep
                # Zero qvel each step like the PLAN / REACH final-waypoint holds — the
                # explicit qfrc_applied damping is marginal at the wrist (dt < 2*I/Kd,
                # see the PLAN comment above). While jogging, inject the commanded
                # joint rates instead of zero: under the quasi-static hold the PD alone
                # would need a huge position error to generate motion (velocity is
                # rebuilt from a single qacc step), so the target would run away from
                # the arm — feedforwarding qdot_jog keeps tracking tight and still
                # pins every DOF's velocity each step.
                data.qvel[:N_ROBOT] = 0.0
                data.qvel[:7] = qdot_jog
                tau_ctrl = np.zeros(model.nv)
                # Softened on the active finger joints while squeezing (see
                # SQUEEZE_PD_SCALE) so the position hold doesn't fight the
                # internal-force torques added below.
                kp_eff, kd_eff = grasp_ctrl.effective_gains()
                tau_ctrl[:N_ROBOT] = (kp_eff * (q_grasp_hold - data.qpos[:N_ROBOT])
                                      + kd_eff * (np.r_[qdot_jog, np.zeros(16)] - data.qvel[:N_ROBOT]))
                if squeeze_on:
                    _squeeze_steps += 1
                    _ramp = min(1.0, _squeeze_steps * model.opt.timestep / SQUEEZE_RAMP_S)
                    tau_ctrl[:N_ROBOT] += grasp_ctrl.internal_force_torques(data, scale=_ramp)
                    # Contact-frame position feedback anchoring each fingertip to its
                    # object contact site — holds the TANGENTIAL friction load that the
                    # softened finger PD can't (the softening that helps normal-force
                    # delivery makes the fingers 4x more compliant in exactly the
                    # direction gravity shears the contact).
                    tau_ctrl[:N_ROBOT] += grasp_ctrl.slip_correction_torques(data)
                    _squeeze_diag(data)

            data.qfrc_applied[:] = tau_ctrl
            data.qfrc_applied[:N_ROBOT] += data.qfrc_bias[:N_ROBOT]
            mj.mj_step(model, data)

            # Render ghost markers: RRT path samples + static IK configs
            if viewer.user_scn is not None:
                with _ghost_markers_lock:
                    markers_snapshot = list(_ghost_markers)
                scn = viewer.user_scn
                scn.ngeom = 0
                _eye9 = np.eye(3, dtype=np.float64).flatten()
                _sz   = np.array([0.018, 0, 0], dtype=np.float64)
                for p0, p1, rgba in markers_snapshot:
                    if scn.ngeom >= scn.maxgeom:
                        break
                    mj.mjv_initGeom(scn.geoms[scn.ngeom],
                                    mj.mjtGeom.mjGEOM_CAPSULE,
                                    np.zeros(3), np.zeros(3), _eye9, rgba)
                    mj.mjv_connector(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_CAPSULE,
                                      0.004, p0, p1)
                    scn.ngeom += 1
                if active_tgt > 0 and _ik_markers_by_obj[active_idx] is not None:
                    for positions, rgba in _ik_markers_by_obj[active_idx]:
                        for pos in positions:
                            if scn.ngeom >= scn.maxgeom:
                                break
                            mj.mjv_initGeom(scn.geoms[scn.ngeom],
                                            mj.mjtGeom.mjGEOM_SPHERE,
                                            _sz, pos, _eye9, rgba)
                            scn.ngeom += 1
                _draw_bspheres(scn)
                _draw_active_marker(scn)

            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    _kb_listener.stop()
    if dash is not None:
        dash.close()
    if _dexpilot_ctrl is not None:
        _dexpilot_ctrl.shutdown()
    if _mediapipe_proc is not None:
        _mediapipe_proc.terminate()
        _mediapipe_proc.wait()

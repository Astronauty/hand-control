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
import json
import numpy as np
import cv2   # camera-feed grid (--camera-views) + post-recalibration GUI reset
import subprocess
import signal
import os
import sys
import time
import threading
import traceback
import queue
from datetime import datetime
from pathlib import Path

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
from trial_logger import (EventLogger, TrialRunner, TrialPhase, TraceBuffer,
                          DexPilotAttemptTrigger, ContactAwareAttemptTrigger,
                          rest_half_height, LIFT_HEIGHT_M, TRIAL_TIMEOUT_S)

# NLP grasp recommender (contact_aware_teleop mode). simulation/ hosts the planner;
# _geom_normal_np gives the outward surface normal used to build inward contact frames.
sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/simulation')
from grasp_planner_3d import (GraspConfig3D, MultiStartGraspPlanner3D,  # noqa: E402
                              _geom_normal_np, _geom_sdf_np)

# 3D_minimum_NCF.py isn't an importable module name (leading digit), so load it by path.
import importlib.util as _ilu
_ncf_spec = _ilu.spec_from_file_location(
    '_ncf', __file__.rsplit('/', 1)[0] + '/scripts/3D_minimum_NCF.py')
_ncf = _ilu.module_from_spec(_ncf_spec)
_ncf_spec.loader.exec_module(_ncf)


def solve_gamma_live(p_O, R_O_inward, mu, mass, accel_box_xyz, ang_accel_box_xyz,
                     inertia_diag, grav_O=None):
    """Minimum internal-force scale gamma that keeps the grasp no-slip for the given
    acceleration/torque disturbance box, from the live grasp geometry. Wraps
    3D_minimum_NCF.min_gamma_for_accel_lp; verified against its native antipodal cases
    (see scratchpad/verify_gamma_solve.py). Handles the two convention mismatches:
      * normal sign: spatial_grasp_map / GraspController give col0 = INWARD normal;
        the NCF cone is built with col0 = OUTWARD (force pushing ON the object) -> flip.
      * unit mass: min_gamma_for_accel_lp assumes m=1, so the "accel box" is really a
        FORCE box (m*a) and the "torque box" a real torque (I*alpha) -> scale here.

    Task definition (grav_O given -> Task B / datum; grav_O None -> Task A / CoM):
      * grav_O is None (legacy CoM mode): the caller must have folded gravity INTO
        accel_box_xyz, and the whole force box is referenced about the CoM. A raised
        or off-center pinch is then correctly reported infeasible for a lateral force.
      * grav_O given (datum mode): accel_box_xyz is the PURE accel budget (no gravity);
        the disturbance is referenced about the GRASP MIDPOINT (moment_ref), so a raised
        symmetric grasp can resist a lateral disturbance at the contact interface; and
        gravity is passed as grav_force with its grasp-axis moment projected out
        (project_grasp_axis_moment) so residual off-CoM drift doesn't spuriously break
        feasibility. This is the hold/transport formulation (see
        RAISED_CONTACT_WRENCH_FINDINGS.md sec 5). Pair it with an angular budget whose
        grasp-axis component has been zeroed (the caller already does that).

    Args:
        p_O:            list of (3,) contact positions in the OBJECT body frame.
        R_O_inward:     list of (3,3) contact->object rotations, col0 = inward normal.
        mu:             list of per-contact friction coefficients.
        mass:           object mass (kg).
        accel_box_xyz:  (ax,ay,az) linear-accel budget, object-body axes. Includes
                        gravity in CoM mode; PURE accel (gravity separate) in datum mode.
        ang_accel_box_xyz: (alpha_x,alpha_y,alpha_z) angular-accel budget, PRINCIPAL axes.
        inertia_diag:   (Ix,Iy,Iz) principal moments (model.body_inertia). Multiplies the
                        angular-accel budget into a torque box.
        grav_O:         (3,) gravitational acceleration in the OBJECT frame (R_WO.T @ g).
                        When given, switches to the datum/Task-B formulation above.

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
    if grav_O is not None and n == 2:
        # Datum / Task-B formulation: reference the disturbance about the grasp midpoint,
        # add gravity as an explicit re-datumed wrench, and project its grasp-axis moment.
        _mref = (0.5 * (pos[0] + pos[1])).reshape(3)
        _grav = mass * np.asarray(grav_O, float)          # gravity force, object frame
        return _ncf.min_gamma_for_accel_lp(
            fx, fy, fz, tx, ty, tz, n, pos, R_out, [1.0] * n,
            [0.0] * n, [0.0] * n, list(mu),
            moment_ref=_mref, grav_force=_grav,
            project_grasp_axis_moment=True,
            project_grasp_axis_torque=True)
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
        80:  'rec_vis',  # P — (contact_aware_teleop) preview the recommender's OWN q
                        #     solution (unconstrained NLP, no collision IK)
        79:  'rec_ik_recq_vis',  # O — preview the collision-aware IK to the recommended
                                #     contacts, warm-started from the recommender's q
        73:  'rec_ik_dls_vis',   # I — same collision-aware IK but warm-started from a
                                #     fresh DLS solve, to A/B the two warm-starts
        82:  'record_sample',   # R — (contact_aware_teleop, --record-samples) append the
                                #     current (pose, object, recommendation) to the tuning
                                #     dataset for offline IK-weight sweeping
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
    ('obj_yellow_box',     'obj_yellow_box_geom',     [0.9, 0.75, 0.1]),
]


def _randomize_objects(model, data, rng):
    """Randomize positions, colors, and sizes for all pickable objects.

    Must be called after MjData creation and before the first mj_forward.
    Updates both data.qpos and model.qpos0 so mj_resetData preserves
    the randomized object positions throughout the IK precomputation loop.
    """
    PICK_CENTER = np.array([0.5, 0.5])
    PICK_HALF   = np.array([0.15, 0.15])   # centers in [0.35, 0.65]² to keep objects on marker,
                                            # and farther from the robot base (0,0,0) to avoid
                                            # near-base IK singularities
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
        if bid < 0 or gid < 0:
            continue   # object commented out of the scene XML

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
        '--collision-view', action='store_true',
        help="Open the viewer showing only collision geoms (group 3) with the LEAP-hand "
             "and arm visual meshes (groups 1 & 2) hidden, so you see the actual boxes the "
             "physics and the recommender's collision model act on. Equivalent to pressing "
             "1 and 2 to hide visuals and 3 to show collision geoms in the viewer.")
    _arg_parser.add_argument(
        '--mode',
        choices=['contact_aware_autonomous', 'contact_aware_teleop', 'dexpilot',
                 'rrt'],
        default='contact_aware_teleop',
        help="contact_aware_teleop (default): teleop the wrist (DexPilot mapping) with "
             "MediaPipe fingers while an NLP continuously recommends grasp contacts for "
             "the nearest object; press L to lock in and approach via RRT, then GRASP "
             "with the NLP's gamma.  | contact_aware_autonomous: autonomous RRT+IK grasp "
             "controller, plans to predefined per-object contact sites ('rrt' is a "
             "deprecated alias).  | dexpilot: live MediaPipe kinematic retargeting "
             "teleop via ROS 2.")
    _arg_parser.add_argument(
        '--recommender-grasp', action='store_true',
        help="contact_aware_autonomous: source grasp contacts from the RECOMMENDER (the same "
             "MultiStartGraspPlanner3D + _commit_recommended_pose path teleop uses) instead of "
             "the authored per-object contact sites. Lets autonomous exercise all the "
             "recommender fixes (finger-link collision, surface-pin, r_tip, orient_weight, "
             "mf/rf ground) and watch the grasp/lift in the viewer with the shared gains.")
    _arg_parser.add_argument(
        '--camera', type=int, default=None,
        help="Force SINGLE-camera teleop on this index, forwarded to "
             "ui/mediapipe_joint_angles.py (the 'Hand Tracking [cam N]' window). "
             "Passing this opts OUT of the default multicam fusion. Omit it to use "
             "the default: auto-discovered multi-camera fusion (see --multicam-auto).")
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
        '--hand-self-collision', action='store_true',
        help="Re-enable LEAP hand self-collision (finger↔finger, finger↔palm contact "
             "physics). Disabled by default: hand geoms are moved to contype=2 so "
             "hand↔hand pairs never match, while hand↔object/floor/arm keep colliding.")
    _arg_parser.add_argument(
        '--ik-solver', choices=['sqp', 'ipopt'], default='sqp',
        help="sqp (default): sqpmethod + OSQP + softplus SDF + analytic FK Jacobians — "
             "~3× cheaper per iteration, wins on wall time in most cases  |  "
             "ipopt: IPOPT L-BFGS + finite-difference Jacobians — production baseline.")
    _arg_parser.add_argument(
        '--record-samples', metavar='PATH', default=None,
        help="contact_aware_teleop: append IK-tuning dataset samples to this JSONL file "
             "when the R key is pressed. Each sample is (q_seed, obj_qpos, object, "
             "recommender candidate {q,p1,p2}) — everything tune_ik_weights.py needs to "
             "re-run the recommender->collision-IK gap offline under swept weights.")
    _arg_parser.add_argument(
        '--multicam', action='append', metavar='NAME:INDEX[:WxH]', default=None,
        help="dexpilot / contact_aware_teleop: auto-launch the multi-camera "
             "hand-tracking pipeline (teleop/run_multicam.py) as a child process "
             "instead of the single-camera publisher, so /hand/joint_angles comes "
             "from the FUSED cameras. Repeat per camera (>=2), e.g. --multicam c0:0 "
             "--multicam c1:2. Implies --no-mediapipe (external publisher). Each "
             "NAME needs camera_intrinsics_<name>.json + camera_extrinsics_<name>.json "
             "in calibration/.")
    _arg_parser.add_argument(
        '--multicam-auto', action='store_true',
        help="Hands-off multi-camera: discover connected cameras and match each to "
             "its calibration by hardware id (no --multicam specs needed). "
             "Equivalent to run_multicam.py --auto. Uses every calibrated camera "
             "currently plugged in (>=2); RealSense auto-flagged. Mutually exclusive "
             "with --multicam. THIS IS THE DEFAULT in dexpilot / contact_aware_teleop "
             "modes — a bare run already auto-discovers and fuses cameras; pass this "
             "explicitly only for clarity. Opt OUT with --no-mediapipe (external "
             "publisher) or --camera N (force the single-cam publisher).")
    _arg_parser.add_argument(
        '--multicam-realsense', action='append', default=[], metavar='NAME',
        help="With --multicam: mark a camera NAME as an Intel RealSense (captures the "
             "COLOR stream via pyrealsense2). Its --multicam :INDEX is ignored (SDK "
             "selects the device); :WxH (or 640x480 default) sets the color size. "
             "The D435I's 1080p color is only 8fps, so default 640x480 @30fps. "
             "Repeatable. Forwards --realsense NAME to run_multicam.py.")
    _arg_parser.add_argument(
        '--recalibrate-extrinsics', action='store_true',
        help="With --multicam: run the INTERACTIVE extrinsics-all walkthrough "
             "(calibration/charuco_calibration.py) BEFORE launching — fix the "
             "ChArUco board at the world origin and press SPACE per camera to "
             "re-solve each camera_extrinsics_<name>.json. Default reuses the saved "
             "extrinsics. Requires per-camera intrinsics to already exist.")
    _arg_parser.add_argument(
        '--square-mm', type=float, default=50.0,
        help="With --recalibrate-extrinsics: MEASURED ChArUco square size in mm, "
             "forwarded to charuco_calibration.py's --square-mm. Must match the "
             "printed board actually on the rig (default 50.0, matching "
             "calibration/charuco_calibration.py's DEFAULT_SQUARE_MM — keep the "
             "two in sync if the board changes).")
    _arg_parser.add_argument(
        '--no-mediapipe', '--external-hand', dest='no_mediapipe',
        action='store_true',
        help="dexpilot / contact_aware_teleop: do NOT spawn the built-in single-camera "
             "publisher (ui/mediapipe_joint_angles.py). Use when an EXTERNAL process "
             "already publishes /hand/joint_angles — e.g. teleop/run_multicam.py — so "
             "the teleop app only subscribes. Avoids two publishers racing on the same "
             "topic (interleaved single-cam + fused poses).")
    _arg_parser.add_argument(
        '--camera-views', action='store_true',
        help="dexpilot / contact_aware_teleop with --multicam: open a window tiling "
             "each camera's live feed + landmark overlay (same as run_multicam.py "
             "--show-fused's camera grid), by subscribing to /hand/cam_<name>/preview. "
             "Camera names come from --multicam.")
    _arg_parser.add_argument(
        '--trial-log', metavar='RUN_DIR', nargs='?', const='', default=None,
        help="Enable automated trial benchmarking (dexpilot / contact_aware_teleop): "
             "logs/<RUN_DIR>/events.jsonl (sparse per-trial-per-phase-per-attempt "
             "event stream) + trial_<id>_<method>_<object>_<outcome>.npz (per-mj_step "
             "fingertip pos/force, object pose/vel). RUN_DIR is optional — bare "
             "--trial-log auto-names it <mode>_<timestamp>. A new trial starts on "
             "each Ctrl+<digit> target selection (dexpilot: pressing 8) and ends on "
             "arrival at the matching place site, a 60s timeout, or an abandoned "
             "mid-trial target switch — see trial_logger.py for the full state "
             "machine. Omit the flag entirely (default) to disable trial logging: "
             "the tool then behaves exactly as without this flag.")
    _arg_parser.add_argument(
        '--grasp-trace', metavar='RUN_DIR', nargs='?', const='', default=None,
        help="Record a per-mj_step GRASP-phase diagnostic trace to "
             "logs/<RUN_DIR>/grasp_trace.npz for offline slip analysis: box pose, per-finger "
             "measured normal/tangential force, commanded contact force |f_c|, tip->contact "
             "slip, jog velocity, squeeze ramp, and palm/box z. Works in ANY mode (unlike "
             "--trial-log). Bare --grasp-trace auto-names the dir.")
    args = _arg_parser.parse_args()
    if args.grasp_trace == '':
        args.grasp_trace = f'{args.mode}_grasp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # In-code defaults for behaviours that used to be always-True CLI flags. These
    # never had a way to turn them OFF from the command line, so they only cluttered
    # --help. They live here as plain constants now; flip one for a run by editing it.
    args.dashboard = True          # live pyqtgraph metrics dashboard (separate proc)
    args.no_randomize = True       # keep scene_pick_place.xml object layout (no randomization)
    args.physics = True            # dexpilot: PD torques + mj_step (physical collisions)
    args.multicam_max_res = True   # open each multicam camera at its highest resolution
    args.skeleton_view = True      # orbitable fused-hand skeleton window (teleop modes)
    if args.mode == 'rrt':          # deprecated alias
        args.mode = 'contact_aware_autonomous'
    if args.multicam and args.multicam_auto:
        _arg_parser.error("--multicam and --multicam-auto are mutually exclusive")
    # Multicam fusion is the DEFAULT hand source in teleop modes: a bare run
    # auto-discovers calibrated cameras and fuses them, so the single-cam publisher
    # (ui/mediapipe_joint_angles.py, the "Hand Tracking [cam N]" window) never opens.
    # Opt out explicitly with --no-mediapipe/--external-hand (external publisher) or
    # --camera N (force the single-cam publisher on that index). Passing --multicam
    # specs or --multicam-auto also takes this path with the given/discovered cameras.
    _wants_single_cam = args.no_mediapipe or args.camera is not None
    if (args.mode in ('dexpilot', 'contact_aware_teleop')
            and not args.multicam and not _wants_single_cam):
        args.multicam_auto = True
    if args.multicam_auto:
        # Discover + match calibrated cameras by hardware id, and synthesise the
        # equivalent --multicam / --multicam-realsense lists so ALL downstream code
        # (name extraction, camera-views subscriptions, the run_multicam command)
        # works unchanged. The child pipeline is launched with the resolved explicit
        # --cam specs (below), and the app needs the resolved NAMES here too.
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'calibration'))
        try:
            from camera_identity import (match_calibrated_cameras,
                                         calibrated_hardware_ids)
        except Exception as _e:
            _arg_parser.error(f"--multicam-auto unavailable (camera discovery): {_e}")
        _calib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'calibration')
        if not calibrated_hardware_ids(_calib_dir):
            _arg_parser.error("--multicam-auto: no calibrated cameras with a stored "
                              "hardware id. Run: python calibration/"
                              "charuco_calibration.py intrinsics-all")
        _specs, _rs = match_calibrated_cameras(_calib_dir)
        if len(_specs) < 2:
            _arg_parser.error(f"--multicam-auto found only {len(_specs)} calibrated "
                              f"camera(s) plugged in; need >= 2.")
        args.multicam = [f"{_nm}:{_idx}" for _nm, _idx in _specs]
        args.multicam_realsense = list(dict.fromkeys(
            list(args.multicam_realsense) + _rs))
        print(f"[multicam-auto] using {len(args.multicam)} cameras: "
              f"{', '.join(args.multicam)}")
    if args.multicam:
        # The fused pipeline is the sole /hand/joint_angles publisher, so never
        # also spawn the single-cam one (two publishers interleave poses).
        args.no_mediapipe = True
        if len(args.multicam) < 2:
            _arg_parser.error("--multicam needs at least 2 cameras for triangulation "
                              "(e.g. --multicam c0:0 --multicam c1:2)")
    elif args.recalibrate_extrinsics:
        _arg_parser.error("--recalibrate-extrinsics requires --multicam")
    if args.camera_views and not args.multicam:
        _arg_parser.error("--camera-views requires --multicam (it needs the camera "
                          "names to subscribe to /hand/cam_<name>/preview)")
    # args.trial_log is None (flag omitted -> trial logging off), '' (bare --trial-log
    # -> auto-name), or an explicit RUN_DIR string. Resolve '' to a timestamped name
    # BEFORE any check below, so every downstream use just sees None or a real name.
    if args.trial_log == '':
        args.trial_log = f'{args.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        print(f"[trial-log] no RUN_DIR given — auto-named logs/{args.trial_log}/")
    if args.trial_log and args.mode not in ('dexpilot', 'contact_aware_teleop'):
        _arg_parser.error("--trial-log only supports --mode dexpilot or "
                          "contact_aware_teleop (the two methods with a defined "
                          "attempt trigger in trial_logger.py).")

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
    # contact_aware_teleop post-grasp wrist tracking: P-gain from wrist pose error to a
    # Cartesian velocity command (1/s). The command is then slew-limited to the NCF accel
    # budget and DLS-mapped to joint rates, so this only sets how briskly the wrist closes
    # a tracking gap; the budget still caps peak acceleration for the no-slip guarantee.
    WRIST_TRACK_GAIN = 3.0

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

    # Tier-1 collision subset for the grasp RECOMMENDER's NLP (contact_aware_teleop).
    # The recommender historically ran with NO arm/object collision (only the middle/ring
    # fingertips, via the legacy path) because feeding it the full 71-geom IK set made the
    # box solve non-converge (returns p1/p2 = None) — see _get_cat_planner. Tier 1 adds
    # back only the palm + wrist geoms: the biggest, bluntest bodies and the most common
    # penetrator (palm driving into the object as the arm reaches in). They sit far from
    # the contacts, so the NLP's proximity pruning (col_prune_margin) drops them until the
    # arm actually gets close — near-zero added cost on most solves. Escalate to more links
    # only if this holds convergence. Empty list restores the old no-arm-collision behavior.
    _REC_ARM_GEOMS = [
        _gname for _gi in range(model.ngeom)
        if (_gname := mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, _gi))
        and _geom_contype0[_gi] != 0
        and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[_gi]) or '')
            in ('leap_palm', 'bracelet_link')
    ]
    print(f"[rec] Tier-1 recommender arm-collision geoms (palm+wrist): "
          f"{len(_REC_ARM_GEOMS)}")
    # Diagonal posture regularization, split arm (first 7 DOF) vs LEAP hand (next 16).
    # The posture term is Σ_i w_i (q_i - q_bias_i)²; a per-block weight lets the arm's
    # redundancy float toward the tip targets (small POSTURE_W_ARM) while the fingers are
    # held closer to their curled/retargeted bias (larger POSTURE_W_HAND), instead of one
    # scalar trading tip accuracy against both. Set both equal to recover the old
    # isotropic behavior. N_ROBOT=23 = 7 arm + 16 LEAP hand joints.
    POSTURE_W_ARM  = 0.1e-4   # arm joints 1..7: loose, so the null space serves the tips
    POSTURE_W_HAND = 0.1e-3   # LEAP 16 finger joints: raise to pin fingers near q_bias
    _posture_w = np.r_[np.full(7, POSTURE_W_ARM), np.full(N_ROBOT - 7, POSTURE_W_HAND)]
    constrained_ik = ConstrainedIKSolver(
        model, N_ROBOT,
        arm_geom_names=_robot_geom_names,
        obj_geom_names=_OBJ_GEOM_NAMES,
        clearance=0.005,
        posture_weight=_posture_w,
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
        tip_weight=100.0,
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

    # --- Trial benchmarking (--trial-log) ------------------------------------------
    # Inert (None) unless --trial-log is passed — every call site below is guarded on
    # _trial_runner is not None, so omitting the flag leaves this file's behavior
    # unchanged. See trial_logger.py for the full state machine this drives.
    _trial_runner  = None
    _trial_state   = None
    _trial_events  = None
    _trial_rest_hh = {}   # obj_idx -> rest half-height (m), for the lift-height check
    _dp_trigger    = None   # DexPilotAttemptTrigger, mode == 'dexpilot'
    _cat_trigger   = None   # ContactAwareAttemptTrigger, mode == 'contact_aware_teleop'
    _trial_dofadr = {}   # obj_idx -> freejoint dof address (qvel[adr:adr+3] = linear v)
    if args.trial_log:
        for _oi, _o in enumerate(objects):
            _gid = _o['id_geom']
            _trial_rest_hh[_oi] = rest_half_height(int(model.geom_type[_gid]),
                                                    model.geom_size[_gid])
            _trial_dofadr[_oi] = int(model.jnt_dofadr[model.body_jntadr[_o['id_body']]])
        _trial_events = EventLogger(Path('logs') / args.trial_log)
        _trial_runner = TrialRunner(_trial_events, Path('logs') / args.trial_log)
        if args.mode == 'dexpilot':
            _dp_trigger = DexPilotAttemptTrigger()
        else:
            _cat_trigger = ContactAwareAttemptTrigger()
        print(f"[trial-log] enabled -> logs/{args.trial_log}/  "
              f"(mode={args.mode})")

    # Always-on pose recorder (independent of trials/phases): one row per loop iteration
    # capturing the full robot+object qpos and wall/sim time, saved to
    # logs/<RUN_DIR>/pose_trace.npz at exit. A PD/gamma tuning harness slices REACH-phase
    # warmstarts out of this by cross-referencing the phase_enter timestamps in
    # events.jsonl. Throttled to ~50 Hz sim-time so the file stays small.
    _pose_trace   = TraceBuffer() if args.trial_log else None
    _pose_last_t  = -1.0
    _POSE_DT      = 0.02   # s sim-time between recorded pose rows (~50 Hz)

    # GRASP-phase diagnostic trace (--grasp-trace): per-mj_step during GRASP, for offline
    # slip analysis. Unthrottled (every step) so the slip-onset transient is captured.
    _grasp_trace  = TraceBuffer() if args.grasp_trace else None

    # obj_<color>_<shape> -> place_<color>_site, by matching the color token common to
    # both names (scene_pick_place.xml's naming convention — see place_red/blue/green/
    # yellow_site). Built once here rather than per-step string parsing in the hot loop.
    _PLACE_COLORS = ('red', 'blue', 'green', 'yellow')
    _trial_place_sid = {}
    if args.trial_log:
        for _oi, _o in enumerate(objects):
            _color = next((c for c in _PLACE_COLORS if c in _o['name']), None)
            _sid = (mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, f'place_{_color}_site')
                    if _color else -1)
            if _sid < 0:
                print(f"[trial-log] WARNING: no place_<color>_site match for "
                      f"'{_o['name']}' — arrival can never be detected for this object.")
            _trial_place_sid[_oi] = _sid

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

    def _actual_contact_geometry(obj_idx):
        """Per FINGER_SET finger: MuJoCo's ACTUAL contact point + INWARD normal on the object,
        force-weighted-averaged over that finger's contacts vs objects[obj_idx]. Returns
        {finger: (pos_W (3,), inward_normal_W (3,)) or None}. Used to compare against the
        RECOMMENDED contact geometry the grasp map / LP certificate assumes — a divergence
        means the commanded 'internal' force pair isn't net-zero in physics and drifts the
        object (the LP-vs-MuJoCo mismatch)."""
        _og = objects[obj_idx]['id_geom']
        acc = {f: [np.zeros(3), np.zeros(3), 0.0] for f in FINGER_SET}  # sum p*fn, sum n*fn, sum fn
        _ft = np.zeros(6)
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1, g2 = con.geom1, con.geom2
            if g1 in _HAND_GIDS and g2 == _og:
                hand_gid, sgn = g1, 1.0     # normal row0 points g1->g2 = hand->obj = inward
            elif g2 in _HAND_GIDS and g1 == _og:
                hand_gid, sgn = g2, -1.0    # row0 points obj->hand = outward -> negate
            else:
                continue
            fname = _FINGER_BY_GID.get(hand_gid)
            if fname is None:
                continue
            mj.mj_contactForce(model, data, ci, _ft)
            fn = float(_ft[0])
            if fn <= 1e-9:
                continue
            n_in_W = sgn * con.frame.reshape(3, 3)[0]   # world inward normal (hand->obj)
            acc[fname][0] += con.pos * fn
            acc[fname][1] += n_in_W * fn
            acc[fname][2] += fn
        out = {}
        for f in FINGER_SET:
            p_sum, n_sum, w = acc[f]
            if w > 1e-9:
                out[f] = (p_sum / w, n_sum / (np.linalg.norm(n_sum) + 1e-12))
            else:
                out[f] = None
        return out

    # Live metrics dashboard (separate process; opt-in via --dashboard). Started before the
    # IK precompute so the grasp IPOPT solves below are reported too. dash is None
    # when disabled; every push site is guarded on it.
    dash = None
    if args.dashboard:
        dash = Dashboard(FINGER_SET, horizon_s=5.0, dt_hint=3 * model.opt.timestep)
        dash.start()
        print("[dashboard] launched (separate process)")
        # Tee trial events to the dashboard's live event log (the logger is created
        # before dash exists, so attach here). No-op when trial logging is disabled.
        if args.trial_log:
            _trial_events._dashboard = dash

    def _push_trial_time(state, t_now):
        """Push the trial countdown to the dashboard each frame. `state` is the current
        TrialState (or None between trials). No-op when the dashboard is disabled."""
        if dash is None:
            return
        if state is None or state.outcome is not None:
            dash.push({'type': 'trial_time', 'remaining': None,
                       'elapsed': None, 'trial_id': None})
            return
        elapsed = t_now - state.t_start
        dash.push({'type': 'trial_time',
                   'remaining': TRIAL_TIMEOUT_S - elapsed,
                   'elapsed': elapsed,
                   'trial_id': state.trial_id})

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
        # Contact reference for the slip anchor. When the grasp came from the RECOMMENDER
        # (rec_local set — teleop lock-in OR --recommender-grasp autonomous), the real
        # contacts are the recommended object-LOCAL frames tracking the object, NOT the
        # authored sites. Measuring slip vs the authored sites in that case reports the fixed
        # authored-vs-recommended OFFSET as phantom slip (e.g. ~35mm at the first frame before
        # any motion). Use rec_local when present; fall back to the authored sites otherwise.
        _rec_local = obj_grasp.get('rec_local')
        parts = []
        for k, f in enumerate(FINGER_SET):
            cmd = (float(np.linalg.norm(f_c[3 * k:3 * k + 3]))
                   if f_c is not None else float('nan'))
            n_meas, t_meas = normals[f], tangentials[f]
            util = t_meas / (mu * n_meas) if n_meas > 1e-6 else float('inf')
            # Slip vs the pad-offset anchor (where the tip SITE sits when the pad surface is
            # flush), not the raw surface point 10mm ahead of it.
            if _rec_local is not None:
                p_O, R_O = _rec_local[k]
                p_WoO_l = d.xpos[obj_grasp['id_body']]
                R_WO_l  = d.xmat[obj_grasp['id_body']].reshape(3, 3)
                contact_W = p_WoO_l + R_WO_l @ p_O
                inward_W  = R_WO_l @ R_O[:, 0]
            else:
                sid_S = obj_grasp['id_S'][k]
                contact_W = d.site_xpos[sid_S]
                inward_W  = d.site_xmat[sid_S].reshape(3, 3)[:, 0]
            anchor_W = contact_W - _PAD_OFFSET[f] * inward_W
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
    # values used for the planar model's tiny finger actuators. Finger gains bumped 1.5x
    # (Kp 0.8->1.2, Kd 0.05->0.075) so the LEAP fingers close harder/faster toward the
    # retargeted/grasp angles. Shared by the dexpilot torque drive and contact_aware
    # REACH/GRASP controller (the dexpilot drive is Kp-only, so Kd[7:] applies to REACH).
    Kp = np.concatenate([np.full(7, 40.0), np.full(16, 1.2)])
    Kd = np.concatenate([np.full(7, 4.0),  np.full(16, 0.075)])

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
    NCF_ACCEL_BUDGET_XYZ = (20.0, 20.0, 20.0)   # m/s^2   object-frame linear-accel budget
    NCF_ANG_ACCEL_BUDGET = (1.0, 1.0, 1.0)   # rad/s^2 principal-frame angular-accel budget

    # Task definition for the gamma LP (see RAISED_CONTACT_WRENCH_FINDINGS.md sec 5):
    #   True  -> DATUM / Task-B: the linear disturbance is referenced at the GRASP
    #            MIDPOINT and gravity is a separate re-datumed wrench with its grasp-axis
    #            moment projected out. This makes a RAISED, reachable antipodal pinch
    #            wrench-feasible for a hold/transport task (the recommender's contacts
    #            lift naturally up the face). Requires the recommended contacts to be a
    #            clean antipodal pair -> pair with w_align on the recommender config.
    #   False -> legacy CoM / Task-A: gravity folded into the accel box, disturbance at
    #            the CoM. A raised/off-center pinch is then reported infeasible (falls
    #            back to GAMMA_FALLBACK), which is correct for a free-body lateral accel.
    NCF_DATUM_MODE = True

    # Conservative multiplier on the solved gamma before it drives the squeeze: the LP
    # gives the MINIMUM no-slip gamma for the box (1.0x margin), which leaves nothing for
    # contact compliance, the finger PD lagging the internal-force torques, the
    # pyramidal-vs-elliptic cone mismatch, and unmodeled dynamics. 2x squeezes twice as
    # hard as the theoretical minimum. Only the value SENT TO THE CONTROLLER is scaled;
    # the wrench-cone viz stays at the raw 1.0x gamma so the drawn cage remains the true
    # feasible boundary the LP computed (the trace then sits well inside it).
    # GAMMA_SAFETY_FACTOR = 50.0
    GAMMA_SAFETY_FACTOR = 5.0



    # Softens Kp/Kd on the active (grasping) finger joints while squeezing, via
    # GraspController.effective_gains(). Without this the full-strength joint PD
    # hold fights the internal-force torques: as GAMMA pushes the fingers to press
    # harder, the position spring (anchored at the fixed pre-squeeze q_grasp_hold)
    # pulls back proportionally, so measured contact force saturates well below
    # GAMMA/sqrt(2) instead of scaling with it.
    # SQUEEZE_PD_SCALE = 50.0
    # 2.0 was too SOFT to hold a sustained lift: the fingers deliver normal force fine
    # (~22N/contact) but are too compliant to resist the TANGENTIAL shear as the object hangs
    # under gravity, so the box slips out over ~2s (a momentary lift, no hold). A grasp-lift
    # integration test (test_grasp_lift.py, live accel=20 budget, 3s hold) found 2.0/3.0
    # slip (sag +7-15mm, box drops) while 5.0 holds (sag ~0, box lifts +73-79mm and stays)
    # robustly across gamma x3-x10 — the fix is finger STIFFNESS, not squeeze force (29N/
    # contact already far exceeds the 0.25kg box's weight). 5.0 is stiff enough to hold the
    # shear while still soft enough to let the internal force deliver the normal squeeze.
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
        # (The magenta "intended contacts" diagnostic markers were removed — they
        # duplicated the rec1/rec2 contact mocaps. The gold ACHIEVED-tip spheres above
        # show where the committed recommender pose lands the fingertips.)
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
        if _trial_events is not None and _trial_state is not None:
            # Wall-clock timestamp (not data.time): this runs on the background
            # plan_thread, which must never touch live `data` concurrently with the
            # main thread's mj_step — sim-time correlation isn't needed for a
            # diagnostic solver-timing entry.
            _trial_events.log_solve(_trial_state.trial_id, time.time(), 'rrt',
                                    plan_time * 1e3, object=obj_target['name'],
                                    n_waypoints=len(path), fallback=fallback)
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
        if _trial_events is not None and _trial_state is not None:
            _now_wc = time.time()
            _trial_events.log_solve(_trial_state.trial_id, _now_wc, 'ik_dls',
                                    dls_grasp_ms, object=obj['name'])
            _trial_events.log_solve(_trial_state.trial_id, _now_wc, 'ik_ipopt',
                                    ipopt_grasp_ms, object=obj['name'])

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

    def _setup_recommended_contact_frames(obj_idx, obj, obj_qpos_snap, rec, q_seed):
        """Shared setup for both the recommender-pose commit (RRT goal) and the
        collision-IK path. Poses _ik_data at q_seed, resolves the recommended contacts
        (rec['p1']=thumb, rec['p2']=index) + inward normals into world + object-LOCAL
        frames, and stores everything the GRASP phase needs on obj:
          p_S_W, inward_S_W, ik_obj_qpos, rec_local, ik_targets.
        Returns (p_S_W, inward_S_W) for callers that continue into an IK solve."""
        mj.mj_resetData(model, _ik_data)
        _ik_data.qpos[:N_ROBOT] = q_seed
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
        return p_S_W, inward_S_W

    def _expand_rec_q(rec, q_seed):
        """Expand the recommender's actuated-joint solution rec['q'] into a full N_ROBOT
        pose over q_seed (the live pose supplies any joints the recommender didn't move).
        Returns None if the candidate carries no q."""
        _rec_q = rec.get('q')
        if _rec_q is None:
            return None
        q_full = np.asarray(q_seed, float).copy()
        for _i, _idx in enumerate(_cat_act_idx):
            q_full[_idx] = _rec_q[_i]
        return q_full

    def _commit_recommended_pose(obj_idx, obj, obj_qpos_snap, rec, q_seed):
        """contact_aware_teleop lock-in (committing path): set the GRASP-phase contact
        frames from the recommended contacts and commit obj['q_target'] to the
        RECOMMENDER's own pose (expanded), WITHOUT running the collision-aware IK.

        This is the pose the RRT then plans toward and the GRASP phase holds. The
        collision-aware IK is no longer on the commit path — it remains available for the
        O/I debug previews (_fire_preview_ik) so the operator can still inspect what the
        constrained solve would produce, but it does not alter the committed grasp."""
        _setup_recommended_contact_frames(obj_idx, obj, obj_qpos_snap, rec, q_seed)
        q_rec = _expand_rec_q(rec, q_seed)
        if q_rec is None:
            # No recommender q (shouldn't happen post-solve) — fall back to the live pose
            # so RRT has a valid, in-place goal rather than crashing.
            print("[teleop] WARNING: recommendation carries no q — committing live pose.")
            q_rec = np.asarray(q_seed, float).copy()
        obj['q_target'] = q_rec

        # Report where the recommender's own pose lands the tip SITES vs its recommended
        # contacts — the committed grasp's tip error (same site->contact metric the O/I
        # previews use, so the two are directly comparable).
        _d_chk = mj.MjData(model)
        _d_chk.qpos[N_ROBOT:] = obj_qpos_snap
        _d_chk.qpos[:N_ROBOT] = q_rec
        mj.mj_forward(model, _d_chk)
        _errs_mm = [float(np.linalg.norm(_d_chk.site_xpos[s] - p) * 1e3)
                    for s, p in zip(id_C, obj['p_S_W'])]
        obj['rec_ik_err_mm']         = _errs_mm
        obj['rec_ik_contact_err_mm'] = _errs_mm
        print(f"\r\n[commit] obj{obj_idx+1} recommender pose committed as RRT goal — "
              f"tip errors (site->contact, mm) {FINGER_SET}: "
              f"{[f'{e:.1f}' for e in _errs_mm]}")
        if dash is not None:
            dash.push({'type': 'tip_err', 'object': obj['name'],
                       'fingers': list(FINGER_SET),
                       'nlp': _errs_mm, 'ik': None, 'rrt': None})

        # ACHIEVED contact points (tip sites pushed to the pad surface) so the markers
        # snap to what the hand actually does — same as the IK path.
        bid   = obj['id_body']
        p_WoO = _d_chk.xpos[bid].copy()
        R_WO  = _d_chk.xmat[bid].reshape(3, 3).copy()
        _achieved, _achieved_O = [], []
        for f, s, n in zip(FINGER_SET, id_C, obj['inward_S_W']):
            p_ach_W = _d_chk.site_xpos[s].copy() + _PAD_OFFSET[f] * n
            _achieved.append(p_ach_W)
            _achieved_O.append(R_WO.T @ (p_ach_W - p_WoO))
        obj['rec_achieved_W'] = _achieved
        obj['rec_achieved_O'] = _achieved_O
        _ik_markers_by_obj[obj_idx] = _make_ik_markers(obj, obj_qpos_snap)
        _ik_solved.add(obj_idx)

    def _audit_finger_penetration(q_full, obj, obj_qpos_snap):
        """Exact per-active-finger-geom signed distance to the target object at pose q_full,
        via mj_geomDistance (the SAME exact box distance ConstrainedIKSolver uses — NOT the
        recommender's bounding-sphere proxy). Returns (worst_geom, worst_dist_m, {geom: dist}).
        Negative = real interpenetration. Used to baseline how deep the recommender's RAW q
        buries a finger link before the collision-aware refinement pulls it out — the ground
        truth for deciding whether the exact-box recommender upgrade is worth its cost."""
        _d = mj.MjData(model)
        _d.qpos[N_ROBOT:] = obj_qpos_snap
        _d.qpos[:N_ROBOT] = q_full
        mj.mj_forward(model, _d)
        _ft = np.zeros(6)
        _og = obj['id_geom']
        _pen = {}
        for g in _active_finger_geoms:
            _gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g)
            # Lower bound guards mj_geomDistance's collision-margin cutoff (returns the
            # margin, not the true deep-penetration distance, past a point) — same pattern
            # as the _run_ik audit at ~L1374.
            _lb = (np.linalg.norm(_d.geom_xpos[_og] - _d.geom_xpos[_gid])
                   - model.geom_rbound[_gid] - model.geom_rbound[_og])
            _pen[g] = max(mj.mj_geomDistance(model, _d, _gid, _og, 1.0, _ft), _lb)
        _wg, _wd = min(_pen.items(), key=lambda kv: kv[1])
        return _wg, _wd, _pen

    def _run_ik_recommended(obj_idx, obj, obj_qpos_snap, rec, q_seed,
                            warmstart='rec_q'):
        """contact_aware_teleop LOCK-IN + O/I DEBUG PREVIEW: like _run_ik but the
        fingertip targets come from the NLP-recommended contacts (rec['p1']=thumb,
        rec['p2']=index in world) instead of the authored contact sites. Runs the
        DLS->SQP collision-aware grasp IK to get obj['q_target'] so the committed grasp
        (and the preview) reflects what the CONSTRAINED solve produces — with finger-link
        penetration bounded — rather than the recommender's raw q (which models fingers as
        points and can bury a link in the object). This backs BOTH the lock-in commit path
        (_run_ik_recommended_then_rrt) and the O/I previews (_fire_preview_ik).

        q_seed is the operator's live teleoped pose — the hand is already positioned at a
        valid, collision-free approach to the object, so we BIAS the IK (posture
        regularization) from it rather than from Q_BIAS.

        warmstart selects the IK's q_init:
          'rec_q' — the recommender's own q (already at the contacts, ~cm residual).
          'dls'   — a fresh DLS solve to the contacts from q_seed.
        Both share the same q_bias=q_seed; only the initial guess differs. Exposed so
        the debug previews can A/B the two warm-starts."""
        p_S_W, inward_S_W = _setup_recommended_contact_frames(
            obj_idx, obj, obj_qpos_snap, rec, q_seed)
        # Warm-start (q_init) selection. 'rec_q': the recommender's own q, expanded to
        # a full-robot pose over q_seed — already at the contacts (~cm residual), a
        # closer start than a fresh DLS. 'dls': a fresh DLS solve to the contacts.
        # Falls back to DLS if 'rec_q' is requested but the candidate carries no q.
        _rec_q = rec.get('q')
        # Recommender's OWN tip error: where the NLP's joint solution lands the tip
        # SITES relative to its OWN recommended contacts p_S_W. The NLP's w_ik cost
        # targets the site directly AT the contact (no pad backoff), so measure against
        # p_S_W, not ik_targets. This is the "low tip error" the user sees reported for
        # the recommender; the collision-IK error below is measured the same way so the
        # two are directly comparable and the discrepancy is attributable.
        obj['rec_nlp_err_mm'] = None
        if _rec_q is not None:
            _q_nlp = np.asarray(q_seed, float).copy()
            for _i, _idx in enumerate(_cat_act_idx):
                _q_nlp[_idx] = _rec_q[_i]
            _d_nlp = mj.MjData(model)
            _d_nlp.qpos[N_ROBOT:] = obj_qpos_snap
            _d_nlp.qpos[:N_ROBOT] = _q_nlp
            mj.mj_forward(model, _d_nlp)
            # Measure against the target the recommender NLP ACTUALLY optimizes:
            # contact + r_tip*outward_normal == p_S_W - r_tip*inward_S_W (grasp_planner_3d.py
            # _run_stage: _tp_tgt = p + r_tip*d_outward). The tip SITE sits at the tip-mesh
            # centroid, so it legitimately stands r_tip (~19mm) proud of the bare contact;
            # measuring site-vs-bare-contact reported that fixed offset as if it were error
            # (the spurious "~24mm plateau"). Use the recommender's OWN r_tip (per finger)
            # from its planner config so the number is true convergence error (sub-mm when
            # the NLP solves well). FINGER_SET order = [index, thumb] -> [r_index, r_thumb].
            _rec_cfg = _get_cat_planner(obj_idx)._planner.cfg
            _r_tip_by_finger = {'index': float(_rec_cfg.r_index),
                                'thumb': float(_rec_cfg.r_thumb)}
            _nlp_tgts = [p - _r_tip_by_finger[f] * n
                         for f, p, n in zip(FINGER_SET, p_S_W, inward_S_W)]
            obj['rec_nlp_err_mm'] = [
                float(np.linalg.norm(_d_nlp.site_xpos[s] - t) * 1e3)
                for s, t in zip(id_C, _nlp_tgts)]
            # BASELINE (Step 2): exact-distance penetration of the recommender's RAW q,
            # before any collision-aware refinement. This is the depth the refinement has
            # to pull out — the number that decides whether the exact-box recommender
            # upgrade is worth it. Stored so the post-refinement audit can print the delta.
            _nlp_wg, _nlp_wd, _ = _audit_finger_penetration(_q_nlp, obj, obj_qpos_snap)
            obj['rec_nlp_pen_mm'] = _nlp_wd * 1e3
            obj['rec_nlp_pen_geom'] = _nlp_wg
        if warmstart == 'rec_q' and _rec_q is not None:
            q_warm = np.asarray(q_seed, float).copy()
            for _i, _idx in enumerate(_cat_act_idx):
                q_warm[_idx] = _rec_q[_i]
            dls_ms = 0.0
            _ws_src = 'recommender-q'
        else:
            _t0 = time.time()
            q_warm = dls_ik.solve(model, _ik_data, id_C, obj['ik_targets'],
                                  q_bias=q_seed, null_gain=0.3)
            dls_ms = (time.time() - _t0) * 1e3
            _ws_src = f'DLS {dls_ms:.0f}ms'
        _t0 = time.time()
        obj['q_target'] = constrained_ik.solve(_ik_data, id_C, obj['ik_targets'],
                                                q_bias=q_seed, q_init=q_warm,
                                                reduced_clearance_geoms=_active_clearance_by_geom,
                                                inward_dirs=inward_S_W)
        sqp_ms = (time.time() - _t0) * 1e3
        print(f"\r\n[IK] obj{obj_idx+1} (recommended): warm-start {_ws_src} + "
              f"SQP {sqp_ms:.0f}ms")

        # BASELINE (Step 2): exact-distance penetration AFTER the collision-aware refinement,
        # and the before->after delta. `raw` is what the recommender's IK-only NLP produces
        # (models fingers as points => can bury a link); `refined` is what constrained_ik
        # pulls it back to (bounded by the tiered clearance). The delta is exactly the
        # collision the second solve exists to fix — if it's small in practice, a lighter
        # recommender collision proxy suffices; if large (~-12mm as previously measured), the
        # exact-box upgrade is justified. All via exact mj_geomDistance (not bounding sphere).
        _ref_wg, _ref_wd, _ = _audit_finger_penetration(obj['q_target'], obj, obj_qpos_snap)
        obj['rec_ik_pen_mm'] = _ref_wd * 1e3
        _raw_pen = obj.get('rec_nlp_pen_mm')
        if _raw_pen is not None:
            _delta = _ref_wd * 1e3 - _raw_pen
            _flag = '  ** RAW q PENETRATES **' if _raw_pen < -2.0 else ''
            print(f"[IK] obj{obj_idx+1} finger-link penetration (exact dist, worst geom):\n"
                  f"       recommender RAW q : {_raw_pen:+6.1f} mm ({obj.get('rec_nlp_pen_geom')}){_flag}\n"
                  f"       collision-aware IK: {_ref_wd*1e3:+6.1f} mm ({_ref_wg})\n"
                  f"       refinement fixed  : {_delta:+6.1f} mm  <- collision the 2nd solve removes")

        _d_chk = mj.MjData(model)
        _d_chk.qpos[N_ROBOT:] = obj_qpos_snap
        _d_chk.qpos[:N_ROBOT] = obj['q_target']
        mj.mj_forward(model, _d_chk)
        _errs_mm = [float(np.linalg.norm(_d_chk.site_xpos[s] - t) * 1e3)
                    for s, t in zip(id_C, obj['ik_targets'])]
        # Attribution metric: each stage's TRUE convergence residual, i.e. how far the tip
        # site lands from the target THAT stage optimizes. The recommender NLP targets
        # p_S_W - r_tip*inward (_nlp_tgts above); the collision-aware IK targets
        # obj['ik_targets'] = p_S_W - _PAD_OFFSET*inward. Both are pad-standoff offsets from
        # the bare contact, just different magnitudes (r_tip vs _PAD_OFFSET). Measuring each
        # site against its OWN target (not the bare p_S_W) removes the fixed standoff that
        # previously showed up as ~19mm phantom "error"; the residuals are now sub-mm when a
        # stage converges, and their DELTA is the real pull-off the clearance constraints add.
        _errs_contact_mm = _errs_mm   # collision-IK residual vs ITS target (ik_targets)
        # Emitted as one multi-line block (each stage on its own line) so the chain reads
        # cleanly in the console instead of running together with prior output.
        _nlp_e = obj.get('rec_nlp_err_mm')
        _nlp_s = (f"{[f'{e:.1f}' for e in _nlp_e]}" if _nlp_e is not None
                  else "n/a (no rec-q)")
        _lines = [
            "",   # leading blank line: separates this block from the SQP-timing line
            f"[IK] obj{obj_idx+1} tip-error attribution (site->stage target, mm), "
            f"fingers {FINGER_SET}:",
            f"       recommender NLP q : {_nlp_s}",
            f"       collision-aware IK: {[f'{e:.1f}' for e in _errs_contact_mm]}",
        ]
        if _nlp_e is not None:
            _delta = [ci - ni for ci, ni in zip(_errs_contact_mm, _nlp_e)]
            _lines.append(
                f"       delta (IK - NLP)  : {[f'{d:+.1f}' for d in _delta]}"
                f"  <- clearance-constraint pull-off")
        print("\n".join(_lines))
        obj['rec_ik_contact_err_mm'] = _errs_contact_mm
        # Push NLP + collision-IK stages to the dashboard's fixed attribution readout.
        # The RRT-end stage is unknown here (planning/replay hasn't run yet) — it's sent
        # as a follow-up 'tip_err' from the Enter->GRASP transition, overwriting this row.
        if dash is not None:
            dash.push({'type': 'tip_err', 'object': obj['name'],
                       'fingers': list(FINGER_SET),
                       'nlp': _nlp_e, 'ik': _errs_contact_mm, 'rrt': None})

        # ACHIEVED contact points: where the fingertip SITES actually land at q_target,
        # pushed to the pad surface along each inward normal. These are the contacts the
        # robot will really make — the markers snap here after lock-in so what you see
        # matches what the hand does, even when the ideal NLP contacts aren't reachable.
        # Stored in FINGER_SET order ([index, thumb]).
        bid   = obj['id_body']
        p_WoO = _d_chk.xpos[bid].copy()
        R_WO  = _d_chk.xmat[bid].reshape(3, 3).copy()
        _achieved = []
        _achieved_O = []
        for f, s, n in zip(FINGER_SET, id_C, inward_S_W):
            p_ach_W = _d_chk.site_xpos[s].copy() + _PAD_OFFSET[f] * n
            _achieved.append(p_ach_W)
            _achieved_O.append(R_WO.T @ (p_ach_W - p_WoO))   # object-local (tracks jog)
        obj['rec_achieved_W'] = _achieved         # world at solve time (object static)
        obj['rec_achieved_O'] = _achieved_O       # object-local; recompute markers live
        obj['rec_ik_err_mm']  = _errs_mm
        _rec_reachable = max(_errs_mm) < REC_REACH_TOL_MM
        if not _rec_reachable:
            print(f"\r\n[teleop] WARNING: recommended grasp only reachable to "
                  f"{max(_errs_mm):.0f} mm (tol {REC_REACH_TOL_MM:.0f} mm) — the "
                  f"markers show the ACHIEVED contacts, not the ideal ones.")

        _ik_markers_by_obj[obj_idx] = _make_ik_markers(obj, obj_qpos_snap)
        _ik_solved.add(obj_idx)

    def _run_ik_recommended_then_rrt(obj_idx, obj, q_start, obj_qpos_snap, rec):
        # Commit path: SINGLE-SOLVE architecture — commit the RECOMMENDER's own q directly
        # as the RRT goal, no second IK. What you see (the recommended markers/pose) is
        # exactly what gets committed.
        #
        # History: the commit path used to re-solve a SEPARATE collision-aware IK
        # (ConstrainedIKSolver) to the recommended contacts, because the old recommender
        # modeled fingers as POINTS and its raw q buried a finger LINK ~12mm inside the
        # object (leap_if_md_collision_5 at -12mm) — committing that verbatim made the RRT
        # goal interpenetrate and the object jumped on the physics handoff. That second
        # solve, however, optimized a DIFFERENT objective (posture bias = live q_seed) with a
        # DIFFERENT tip offset (_PAD_OFFSET 10mm directional vs the recommender's r_tip 19mm
        # radial), so the committed grasp landed at a visibly different pose/contacts than the
        # displayed recommendation even on a perfectly STATIC object — the "grasp changes on
        # lock-in" bug. Now that the recommender's NLP carries the finger-link collision
        # constraints (obj_clearance_by_geom, penetration bounded to the tier, validated
        # 20/20), its raw q is collision-safe to commit, so we drop the second solve entirely.
        # The collision-aware ConstrainedIKSolver remains available for the O/I debug previews
        # (_fire_preview_ik) but is no longer on the commit path.
        _commit_recommended_pose(obj_idx, obj, obj_qpos_snap, rec, q_start)
        _run_rrt(q_start, obj['q_target'], obj)

    def _run_recommender_then_rrt(obj_idx, obj, q_start, obj_qpos_snap):
        """AUTONOMOUS (--recommender-grasp): source the grasp from the RECOMMENDER instead of
        the authored contact sites. Solves the NLP SYNCHRONOUSLY here in the plan thread (no
        background continuous recommender like teleop), commits the recommended pose via the
        same single-solve path teleop uses (_commit_recommended_pose -> recommender q as the
        RRT goal), then plans the RRT. This routes autonomous through every recommender fix
        (finger-link collision, surface-pin frame, r_tip, orient_weight, mf/rf ground) with the
        shared GRASP gains, so the grasp/lift can be watched in the viewer.

        q_start is the arm's current pose; the recommender regularizes toward it (reg_arm_
        toward_current), same as teleop seeds from the operator's live pose."""
        planner = _get_cat_planner(obj_idx)
        planner._planner.data.qpos[:N_ROBOT] = q_start
        planner._planner.data.qpos[N_ROBOT:] = obj_qpos_snap
        mj.mj_forward(model, planner._planner.data)
        _q_snap = np.array([q_start[i] for i in _cat_act_idx])
        _obj_pos = planner._planner.data.xpos[obj['id_body']].copy()
        try:
            res = planner.solve(_q_snap, _obj_pos, max_seeds=_REC_NC)
        except Exception:
            traceback.print_exc()
            res = {}
        if res.get('p1') is None or res.get('p2') is None:
            print(f"\r\n[auto-rec] recommender did not converge for {obj['name']} — "
                  f"falling back to authored-site IK.")
            _run_ik_then_rrt(obj_idx, obj, q_start, obj_qpos_snap)
            return
        # Wrench-feasibility check (same datum LP gate teleop uses to accept a candidate).
        try:
            _wf = bool((planner._planner.verify(res) or {}).get('wrench_feasible', False))
        except Exception:
            _wf = False
        if not _wf:
            print(f"\r\n[auto-rec] WARNING: recommended grasp for {obj['name']} is not "
                  f"wrench-feasible — committing anyway for inspection.")
        print(f"\r\n[auto-rec] {obj['name']}: recommender contacts committed "
              f"(cost={res.get('cost')}, wf={_wf})")
        _commit_recommended_pose(obj_idx, obj, obj_qpos_snap, res, q_start)
        _run_rrt(q_start, obj['q_target'], obj)

    def _fire_preview_ik(obj_idx, obj, q_seed, obj_qpos_snap, rec, warmstart, slot):
        """Background: run the collision-aware lock-in IK (WITHOUT RRT / committing) for
        one warm-start variant so a preview key can hold the robot at the result. Writes
        into _rec_ik_result[slot] = {'q','obj_idx','err_mm'}. 'slot' keys the two
        variants ('rec_q' vs 'dls') so both can be previewed/compared independently."""
        def _run():
            try:
                _run_ik_recommended(obj_idx, obj, obj_qpos_snap, rec, q_seed,
                                    warmstart=warmstart)
            except Exception:
                traceback.print_exc()
                return
            with _rec_ik_lock:
                _rec_ik_result[slot] = {
                    'q':       np.asarray(obj['q_target'], float).copy(),
                    'obj_idx': obj_idx,
                    'err_mm':  list(obj.get('rec_ik_err_mm', [])),
                }
        t = threading.Thread(target=_run, daemon=True,
                             name=f'cat-preview-ik-{slot}')
        t.start()
        return t

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
    _tune_retarget  = False   # dexpilot: hot-reload retarget_config.json edits
    _mediapipe_proc = None
    if args.mode in _teleop_modes:
        # Launch the MediaPipe publisher as a subprocess so its OpenCV window
        # appears alongside the MuJoCo viewer. The subprocess inherits the
        # current environment (CYCLONEDDS_URI, ROS sourcing, venv Python).
        # --no-mediapipe/--external-hand skips this so an external publisher
        # (e.g. teleop/run_multicam.py) can be the SOLE source on
        # /hand/joint_angles — two publishers on one topic interleave poses.
        # --multicam auto-launches that fused pipeline here as a child process.
        if args.multicam:
            _here = __file__.rsplit('/', 1)[0]
            # Optional interactive extrinsics recalibration FIRST (blocking): the
            # board must be fixed at the world origin and SPACE pressed per camera.
            # Reuses charuco_calibration.py extrinsics-all; --cam takes NAME:INDEX
            # (strip any :WxH resolution suffix the pipeline spec may carry).
            if args.recalibrate_extrinsics:
                # Solve extrinsics per-camera (the single 'extrinsics' subcommand),
                # NOT extrinsics-all: each camera must open at the SIZE ITS OWN
                # intrinsics were calibrated at, and cameras here differ (e.g. c0
                # 1920x1080, c1 1280x960, rs 1280x720). extrinsics-all only takes a
                # global --max-res / --width-height, which can't serve mixed sizes
                # (and --max-res over-shoots the RealSense to 1080p). Per-camera we
                # forward each spec's :WxH (or --max-res when a spec has none). The
                # board must stay FIXED across all of these solves for one world frame.
                print("[DexPilot] --recalibrate-extrinsics: solving each camera's "
                      "extrinsic in turn — KEEP THE BOARD FIXED at the world origin "
                      "for ALL cameras (that shared pose is the common frame). SPACE "
                      "to solve each, S to skip, Q to stop. Teleop starts after.")
                _cal = _here + '/calibration/charuco_calibration.py'
                sys.path.insert(0, _here + '/teleop')
                from run_multicam import _intrinsics_res  # noqa: E402
                _rs_names = set(args.multicam_realsense)
                for _spec in args.multicam:
                    _p = _spec.split(':')
                    _nm, _idx = _p[0], _p[1]
                    _is_rs = _nm in _rs_names
                    _ex = [sys.executable, _cal, 'extrinsics',
                           '--camera', _idx, '--name', _nm,
                           '--square-mm', str(args.square_mm)]
                    # A RealSense MUST be captured via pyrealsense2 (same path the
                    # pipeline uses), NOT cv2 — forward --realsense so the solve sees
                    # the identical color stream its intrinsics were read from. cv2
                    # here would open a different stream (or fail on the D435I) and
                    # the pose wouldn't match runtime pixels.
                    if _is_rs:
                        _ex += ['--realsense']
                    # Open at this camera's calibrated size so the solve matches its
                    # intrinsics: explicit :WxH > intrinsics image_size > --max-res.
                    # (--max-res is skipped for a RealSense: its capture size is fixed
                    # at the pyrealsense2 stream, and 1080p color is only 8 fps.)
                    _res = None
                    if len(_p) >= 3 and _p[2]:                 # explicit :WxH
                        _res = tuple(int(v) for v in _p[2].lower().split('x'))
                    else:
                        _res = _intrinsics_res(_nm)
                    if _res is not None:
                        _ex += ['--width', str(_res[0]), '--height', str(_res[1])]
                    elif args.multicam_max_res and not _is_rs:
                        _ex += ['--max-res']
                    print(f"[DexPilot]   extrinsic for '{_nm}' (index {_idx})…")
                    _rc = subprocess.call(_ex)
                    if _rc != 0:
                        print(f"[DexPilot]   '{_nm}' extrinsic exited {_rc}; keeping "
                              f"its existing extrinsic on disk.")
                # The calibration subprocesses ran their own cv2 (Qt5) HighGUI
                # windows on this same $DISPLAY. Reset the parent's cv2 GUI state to
                # a clean slate before we later create the skeleton / camera-feed
                # windows, so a stale Qt window registry left by the child can't make
                # them fail to map / vanish. Pump waitKey so Qt processes the destroy.
                try:
                    cv2.destroyAllWindows()
                    for _ in range(5):
                        cv2.waitKey(1)
                except Exception:
                    pass
            # Launch the fused pipeline as ONE supervised child (run_multicam owns
            # the landmark + fusion subprocesses). We only hold this one handle and
            # terminate it on exit; if it dies, the subscriber simply goes quiet.
            _mc_cmd = [sys.executable, _here + '/teleop/run_multicam.py']
            for _spec in args.multicam:
                _mc_cmd += ['--cam', _spec]
            for _rs_name in args.multicam_realsense:
                _mc_cmd += ['--realsense', _rs_name]
            # Resolution: a per-camera :WxH in the spec wins; else --max-res opens
            # each at its highest size; else run_multicam falls back to 640x480 (which
            # must then match each camera's calibrated intrinsics, or the fusion node
            # rejects the projection as invalid).
            if args.multicam_max_res:
                _mc_cmd += ['--max-res']
            # start_new_session=True puts the multicam supervisor in its OWN
            # session/process group, so a terminal Ctrl-C is delivered ONLY to
            # this (parent) process, NOT directly to the child and its grandchildren
            # (the landmark + fusion nodes). Without this, Ctrl-C hit the whole
            # foreground group at once: the parent could exit while the child was
            # mid-teardown, orphaning the grandchildren (they kept logging + holding
            # /dev/video* after the sim exited). Now teardown is driven solely by our
            # finally block, which signals the child's whole group (see below).
            _mediapipe_proc = subprocess.Popen(_mc_cmd, start_new_session=True)
            print(f"[DexPilot] multicam pipeline launched (pid {_mediapipe_proc.pid}): "
                  f"{len(args.multicam)} cameras -> /hand/joint_angles.")
        elif args.no_mediapipe:
            print("[DexPilot] --no-mediapipe: single-cam publisher NOT started; "
                  "subscribing to an external /hand/joint_angles publisher.")
        else:
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
            # MULTICAM (--no-mediapipe): the fused palm_R is already in the SHARED
            # WORLD frame (triangulated), not a single camera's MediaPipe frame. So
            # the single-cam R_cam_robot (= that one camera's R_world_cam, ~103.7°
            # of rotation) must NOT be applied — it re-rotates an already-world palm
            # and flips the wrist orientation. The orientation chain in step() is
            #   R_des = R_correct @ R_cam_robot @ R_mp_to_cv @ palm_R,
            # with R_correct=I (identity_orientation) and R_mp_to_cv=diag([1,-1,-1]).
            # Choosing R_cam_robot = diag([1,-1,-1]) makes R_cam_robot@R_mp_to_cv = I,
            # so R_des = palm_R maps the fused world palm DIRECTLY to the robot. The
            # single-cam path keeps the measured R_cam_robot (unchanged below).
            if args.no_mediapipe:
                _cam_kwargs["R_cam_robot"] = np.diag([1.0, -1.0, -1.0])
                print("[DexPilot] multicam: R_cam_robot neutralized (fused palm is "
                      "already world-frame) — direct world->robot orientation.")
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
        # --physics: persistent PD target the robot is driven toward when no fresh
        # teleop pose is available (before press-8, or between camera frames), so it
        # holds the wrist-down home instead of drifting under contact. Refreshed to
        # each new q_teleop below. Unused in kinematic mode.
        _dp_target = _Q_BIAS_DP.copy()
        # --physics gains: the flat Kp[:7]=40 (tuned to HOLD a static grasp) is far
        # too soft to TRACK a moving hand — the big base joints (I~1.0) get a ~1 Hz
        # bandwidth and lag badly. Design instead for a uniform ~3 Hz tracking
        # bandwidth per joint: Kp = I * wn^2 (inertia-scaled), with CRITICAL damping
        # Kd = 2*I*wn applied via the model's dof_damping (implicit -> unconditionally
        # stable, no dt limit, so no BADQACC). wn=18 (~3 Hz) keeps peak torque near
        # the Gen3 actuator limits; higher would clip/oscillate. Computed from the
        # inertia at the wrist-down home so it matches the teleop operating region.
        _dp_Kp_arm = np.full(7, 40.0)   # fallback; overwritten just below
        # Cap on physics substeps per loop iteration (real-time catch-up). Bounds
        # the work if a single iteration hitches badly, so it can't spiral; ~5 covers
        # the ~1.7 ms retarget + IK + draw overhead at dt=2 ms with margin.
        # NOTE: _elapsed (below) is measured BEFORE the substep loop, so it only sizes
        # _n_sub to cover spin()+draw+step() cost — anything added AFTER the substep
        # loop each iteration (trial-logging, skeleton/camera draw, viewer.sync()) is
        # structurally invisible to this catch-up mechanism and accrues as pure,
        # uncompensated sim-time lag. DP_PROFILE below measures exactly where that
        # per-iteration wall-time actually goes.
        _DP_MAX_SUBSTEPS = 5
        # Per-iteration wall-time breakdown (opt-in via DP_PROFILE=1), same pattern as
        # GRASP_PROFILE — printed once/sec so a live session can show which section is
        # actually responsible for sim-time falling behind wall-time.
        DP_PROFILE = os.environ.get('DP_PROFILE', '0') == '1'
        _dpp_acc = {'spin_draw': 0.0, 'spin': 0.0, 'skel': 0.0, 'camviews': 0.0,
                    'step': 0.0, 'substeps': 0.0, 'trial_log': 0.0,
                    'viz_sync': 0.0, 'iter': 0.0, 'n': 0}
        _dpp_last = time.time()
        if args.physics:
            _Mfull = np.zeros((model.nv, model.nv))
            mj.mj_fullM(model, _Mfull, data.qM)
            _I_arm = np.clip(np.diag(_Mfull)[:7], 1e-3, None)   # per-arm-joint inertia
            _WN = 100.0                                          # rad/s (~3 Hz) target
            _dp_Kp_arm = _I_arm * _WN**2
            model.dof_damping[:7] = 2.0 * _I_arm * _WN          # critical, implicit
            print(f"[dexpilot] --physics: PD-torque drive + mj_step (collisions ON). "
                  f"arm Kp={np.round(_dp_Kp_arm,0)} (inertia-scaled ~3Hz), "
                  f"dof_damping={np.round(model.dof_damping[:7],1)} (critical, implicit).")
        # debug=False silences the per-frame [retarget] print. Both teleop modes
        # now run live DexPilot finger retargeting: contact_aware_teleop needs the
        # curling (the MediaPipe joint angles ARE the grasp), and pure dexpilot
        # needs it too so the live retarget tuner's sliders have a visible effect
        # (previously dexpilot froze the fingers OPEN for easier orientation
        # reading during multi-pose calibration — that convenience is dropped in
        # favour of seeing the retargeting the sliders tune).
        _hand_tracking = True
        # debug prints the per-frame [retarget] S1 pinch distances (if/mf/rf->th)
        # vs EPS and the palm->tip distances — the numbers you read to set EPS/ETA
        # when tuning. On in dexpilot (the tuning mode); off in contact_aware_teleop
        # so it doesn't spam the grasp path.
        _retarg_debug = (args.mode == 'dexpilot')
        # eps seed in METRES (world-landmark fingertips) — open-hand S1 ~0.07-0.11 m,
        # pinch ~0.01-0.03 m, so ~0.03 sits between the clusters. A saved
        # calibration/retarget_config.json still wins over this seed.
        #
        # Output smoothing (EMA) is a MAJOR lag source. In KINEMATIC mode the EMA is
        # the only smoothing, so a heavy alpha=0.3 (default) is fine. In --physics
        # mode the PD + physics ALREADY smooth, so stacking a heavy EMA on top makes
        # the arm/fingers feel sluggish — use a much CRISPER alpha there (near-raw IK
        # output) and let the physics do the smoothing. hand_alpha likewise.
        _dp_arm_alpha  = 0.9 if args.physics else 0.3
        _dp_hand_alpha = 0.9 if args.physics else 0.3
        _cam_kwargs.setdefault("alpha", _dp_arm_alpha)
        _dexpilot_ctrl = DexPilotController(model, q_bias=_Q_BIAS_DP,
            debug=_retarg_debug, eps=0.03, hand_tracking=_hand_tracking,
            hand_alpha=_dp_hand_alpha, **_cam_kwargs)
        _dexpilot_ctrl.init_home(data)   # snapshots the wrist-down pose as home
        _dexpilot_ctrl.init_ros()

        # Live finger-retargeting tuning by TEXT ENTRY: edit the 7 constants
        # (BETA/GAMMA/EPS/ETA1/ETA2/S1/S2 gains) in calibration/retarget_config.json
        # and save — dexpilot hot-reloads the file onto the live retargeter each
        # frame (poll_retarget_config, mtime-gated). contact_aware_teleop uses the
        # saved constants but isn't the live-tuning surface.
        _tune_retarget = (args.mode == 'dexpilot')
        if _tune_retarget:
            print("[DexPilot] live tuning: edit calibration/retarget_config.json "
                  "and save — changes hot-reload onto the retargeter.")

        print("[DexPilot] ROS subscriber active — waiting for /hand/joint_angles (≥120 floats)")
        print("[DexPilot] Press 8 to start tracking (captures your current wrist "
              "orientation as the robot's home). Q/Esc: quit")

    # --- Optional fused-skeleton viewer (dexpilot / contact_aware_teleop) ---
    # Reuses teleop/skeleton_viewer.SkeletonViewer, fed the 21 WORLD landmarks that
    # already ride in every /hand/joint_angles message (raw[57:120]) — the SAME points
    # the multicam fusion node draws. So this shows the fused skeleton with zero extra
    # ROS wiring; it just visualises what's already reaching the retargeter. Camera
    # positions can't be drawn here (calib lives in the fusion node, not the message),
    # and the camera-preview grid needs the /hand/cam_*/preview topics — use
    # run_multicam.py --show for those. Drawn once per frame from both teleop branches.
    _skel_viewer = None
    if args.skeleton_view and args.mode in _teleop_modes:
        sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/teleop')
        from skeleton_viewer import SkeletonViewer   # noqa: E402
        _skel_viewer = SkeletonViewer(name="teleop hand skeleton")
        print("[teleop] skeleton view on — orbit with mouse, r reset, z flip up.")

    # Wall-clock of the last skeleton draw, for throttling (list = mutable closure cell).
    _skel_last_draw = [0.0]
    SKEL_DRAW_HZ = 30.0   # the fused hand data only updates at ~20 Hz; drawing the
                          # separate GLFW window every ~2ms sim iter (~110 Hz) cost
                          # ~5ms/iter (the bulk of spin_draw) for no new information.

    def _draw_skeleton():
        """Feed the latest ABSOLUTE world-frame hand skeleton to the viewer.

        The /hand/joint_angles world-landmark block (raw[57:120]) is WRIST-RELATIVE
        (world_lm = pts - pts[wrist] in the fusion node), so drawing it alone pins
        the wrist at the origin — the hand articulates but never translates, which
        is NOT what the fusion node's own viewer shows. Re-add the absolute wrist
        position (raw[0:3], the triangulated wrist in the shared world frame) to
        recover the true world pose the multicam viewer draws. Draws empty until a
        hand is tracked; no-op if the viewer isn't enabled.

        Throttled to SKEL_DRAW_HZ: the GLFW render is ~5ms and the source data only
        refreshes at camera rate (~20 Hz), so redrawing every sim iteration just burns
        wall-time (inflating the real-time catch-up deficit) with nothing new to show."""
        if _skel_viewer is None:
            return
        _now = time.time()
        if _now - _skel_last_draw[0] < 1.0 / SKEL_DRAW_HZ:
            return
        _skel_last_draw[0] = _now
        raw = _dexpilot_ctrl.raw_msg
        pts = None
        n_lm = 0
        if raw is not None and len(raw) >= 120:
            wrist_rel = np.asarray(raw[57:120], float).reshape(21, 3)
            wrist_abs = np.asarray(raw[0:3], float)      # absolute wrist (world frame)
            pts = wrist_rel + wrist_abs                  # wrist-relative -> absolute
            n_lm = 21
        _skel_viewer.show(pts, f"world landmarks: {n_lm}/21  "
                               f"wrist=({raw[0]:+.2f},{raw[1]:+.2f},{raw[2]:+.2f})m"
                               if pts is not None else "waiting for hand…")

    # --- Optional per-camera feed grid (dexpilot / contact_aware_teleop --multicam) ---
    # Mirrors run_multicam.py --show-fused's CAMERA GRID window: each camera's live
    # frame + landmark overlay, tiled. The landmark nodes already publish a throttled
    # JPEG preview to /hand/cam_<name>/preview (independent of --show), so we just
    # subscribe here on our OWN small rclpy node (kept separate from the shared
    # DexPilot subscriber) and decode into a CameraGridWindow. Camera names come from
    # the --multicam specs (strip :INDEX[:WxH]). Drawn once per frame like _draw_skeleton.
    _cam_grid       = None
    _cam_grid_node  = None
    _cam_previews   = {}      # name -> latest decoded BGR frame
    if args.camera_views and args.multicam and args.mode in _teleop_modes:
        import rclpy as _rclpy                             # noqa: E402
        from rclpy.node import Node as _RclNode            # noqa: E402
        from sensor_msgs.msg import CompressedImage as _CompressedImage  # noqa: E402
        sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/teleop')
        from skeleton_viewer import CameraGridWindow       # noqa: E402
        from hand_message import sensor_qos                # noqa: E402
        _cam_names = [s.split(':')[0] for s in args.multicam]
        # rclpy is already init()'d by the DexPilot ROSInterface; just add a node.
        _cam_grid_node = _RclNode("teleop_camera_grid")

        def _mk_preview_cb(_nm):
            def _cb(msg):
                buf = np.frombuffer(bytes(msg.data), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is not None:
                    _cam_previews[_nm] = img
            return _cb

        for _nm in _cam_names:
            # BEST_EFFORT sensor QoS: this preview grid must never back-pressure
            # the landmark nodes' preview publisher (the press-8 camera freeze).
            _cam_grid_node.create_subscription(
                _CompressedImage, f"/hand/cam_{_nm}/preview", _mk_preview_cb(_nm),
                sensor_qos())
        _cam_grid = CameraGridWindow(_cam_names, name="teleop camera feeds")
        print(f"[teleop] camera feeds on — tiling {_cam_names} "
              f"from /hand/cam_<name>/preview.")

    def _draw_camera_views():
        """Spin the preview subscriptions and tile the latest per-camera frames.
        No-op unless --camera-views is on. Mirrors the fusion node's camera grid."""
        if _cam_grid is None:
            return
        import rclpy as _rclpy
        _rclpy.spin_once(_cam_grid_node, timeout_sec=0.0)   # drain pending previews
        _cam_grid.show(_cam_previews)

    _pipeline_dead_warned = [False]

    def _check_pipeline_alive():
        """Warn ONCE if the --multicam pipeline child has exited, so a dead pipeline
        (e.g. a camera failed to open) shows a clear message instead of the app just
        freezing with no hand data arriving. No-op unless we launched a pipeline."""
        if _mediapipe_proc is None or not args.multicam:
            return
        if _mediapipe_proc.poll() is not None and not _pipeline_dead_warned[0]:
            _pipeline_dead_warned[0] = True
            print(f"[teleop] WARNING: multicam pipeline (pid {_mediapipe_proc.pid}) "
                  f"exited (code {_mediapipe_proc.returncode}). No /hand/joint_angles "
                  f"will arrive — the robot will hold still. Check the pipeline log "
                  f"above for the failing camera, then relaunch.")

    # --- contact_aware_teleop: NLP grasp recommender machinery ---
    # A per-object MultiStartGraspPlanner3D (built lazily, box-like first) continuously
    # recommends 2-finger contacts for whichever object the fingers are nearest. The
    # best candidate's p1(thumb)/p2(index) are shown live via the rec1/rec2 mocap
    # markers; pressing L locks them in and hands off to the existing IK->RRT->GRASP
    # machinery. The NLP's gamma seeds the squeeze, re-solved on the committed geometry.
    _CAT_MODE       = (args.mode == 'contact_aware_teleop')
    _REC_INTERVAL_S = 2.0     # fixed re-solve cadence (NLP solve ~0.5-2s, runs in a thread)
    _REC_NC         = 3       # planner seeds per solve
    REC_REACH_TOL_MM = 15.0   # lock-in IK residual above which a rec is flagged unreachable
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
    # Anti-jitter for the continuously-firing recommender: warm-start each solve from the
    # last ACCEPTED contacts (so a static object returns to the same basin), and apply
    # display hysteresis (only replace the shown candidate on a moved object, a new object,
    # or a meaningfully-better cost) so the markers stop hopping between local optima.
    _REC_HYST_COST_FRAC = 0.15   # new cost must beat shown by >15% to replace it
    _REC_OBJ_MOVE_M     = 0.010  # object move (m) that forces a fresh recommendation
    _rec_vis          = False   # P: hold the robot at the recommender's OWN q (debug)
    _rec_ik_mode      = None    # None | 'rec_q' (O) | 'dls' (I): which constrained-IK
                                #   preview is held (warm-start variant)
    _rec_ik_thread    = None    # background thread running a preview collision IK
    _rec_ik_result    = {}      # slot ('rec_q'|'dls') -> {'q','obj_idx','err_mm'}
    _rec_ik_lock      = threading.Lock()

    # IK-weight tuning dataset recorder (--record-samples). Each R press appends one JSONL
    # sample capturing everything tune_ik_weights.py needs to re-run the recommender->
    # collision-IK gap offline: the operator's live pose (IK warm-start/bias), the object
    # joint state (object pose), and the recommender candidate (q/p1/p2 — weight-invariant,
    # so it's solved once here, not per tuning iteration).
    _record_path      = args.record_samples
    # One-element list so the nested _record_sample mutates the count in place. The
    # enclosing scope here is the module-level `if __name__` block (not a function), so
    # `nonlocal` is illegal and rebinding a bare int would need `global`; a container
    # mutated in place matches the pattern used elsewhere in this file (_rec_result etc.).
    _n_recorded       = [0]
    if _record_path is not None:
        # Count any pre-existing samples so the on-screen counter is cumulative across runs.
        try:
            with open(_record_path) as _f:
                _n_recorded[0] = sum(1 for _line in _f if _line.strip())
            print(f"[record] appending to {_record_path} "
                  f"({_n_recorded[0]} existing samples)")
        except FileNotFoundError:
            print(f"[record] will create {_record_path} on first R press")

    def _record_sample(q_seed, obj_qpos_snap, obj_idx, cand):
        """Append one tuning sample as a JSON line. Returns the new cumulative count, or
        None if there's no usable recommendation (nothing to correlate against)."""
        if cand is None or cand.get('q') is None:
            return None
        sample = {
            'object':   objects[obj_idx]['name'],
            'q_seed':   np.asarray(q_seed, float).tolist(),        # (N_ROBOT,)
            'obj_qpos': np.asarray(obj_qpos_snap, float).tolist(),  # object joints
            'rec_q':    np.asarray(cand['q'],  float).tolist(),    # actuated-order
            'rec_p1':   np.asarray(cand['p1'], float).tolist(),    # thumb contact (W)
            'rec_p2':   np.asarray(cand['p2'], float).tolist(),    # index contact (W)
            'rec_status': cand.get('status'),
        }
        with open(_record_path, 'a') as _f:
            _f.write(json.dumps(sample) + "\n")
        _n_recorded[0] += 1
        return _n_recorded[0]

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
            # NOTE: passing the FULL arm/hand collision geoms (all 71) into the NLP was
            # tried and made the box solve non-converge from the home approach (the extra
            # ~150 softplus-SDF constraints corrupt L-BFGS curvature; returns p1/p2 = None).
            # We now pass a TIER-1 subset instead — palm + wrist only (_REC_ARM_GEOMS) —
            # so the recommender avoids the most common gross penetration (palm into the
            # object) while staying well inside the constraint budget the solve tolerates.
            # These geoms are proximity-pruned each solve (col_prune_margin), so on most
            # solves they add no active constraints. Finer reachability is still surfaced
            # downstream: the markers snap to the ACHIEVED fingertips (with a residual flag)
            # so an unreachable recommendation is visible rather than hidden.
            # w_align keeps the recommended contacts a CLEAN ANTIPODAL PAIR (grasp axis
            # along the face normal, equal height) — the precondition the datum gamma
            # formulation (NCF_DATUM_MODE) needs to certify the naturally-lifted contacts
            # as wrench-feasible. See RAISED_CONTACT_WRENCH_FINDINGS.md sec 5.
            # edge_margin_m keeps contacts OFF the face rim as a HARD bound (the datum γ
            # formulation lets the IK cost pull contacts toward the edges, where a pinch
            # slips). Hard: a grasp needing a near-edge contact is rejected, not slipped.
            #
            # DECOUPLED architecture: the NLP runs IK-only (wrench_constraint=False) —
            # solving for reachable, antipodal (w_align), off-edge (edge_margin_m) contacts
            # — and γ is certified as a post-solve LP in verify() under DATUM semantics
            # (datum_gamma=True), the SAME formulation the controller's solve_gamma_live
            # uses at grasp time. This aligns the recommender's feasibility definition with
            # the one that actually gates the squeeze, and removes the 630 wrench-cone
            # variables from the NLP. See RAISED_CONTACT_WRENCH_FINDINGS.md sec 5.
            # Certify γ under the SAME disturbance budgets the controller uses at grasp
            # time (solve_gamma_live), so the recommender's feasibility flag agrees with
            # the one that gates the squeeze.
            # FINGER-LINK COLLISION IN THE RECOMMENDER (was: palm+wrist only). The IK-only
            # NLP modeled fingers as points, so its raw q buried a finger link ~13mm inside
            # the object (measured), which the post-solve refinement then had to pull 50mm+
            # off the recommended contacts — abandoning the very grasp the datum-γ was
            # certified for. Adding the active-finger links here lets the NLP place contacts
            # its own q can reach without link penetration, so the γ certificate finally
            # corresponds to a reachable, collision-free pose.
            #
            # Constraint-budget care (see the all-71-geoms non-convergence note above): we
            # add ONLY the active grasping fingers' ADJACENT + PROXIMAL links, and EXCLUDE
            # the contact-tier (ds/tip) geoms — those carry no object constraint anyway
            # (clearance disabled so they can touch), so including them would only add prune
            # cost and floor constraints without protecting anything. Each added geom is
            # proximity-pruned per solve, so on most solves few are active.
            # Clearance TIERS reuse _active_obj_clearance verbatim (adjacent -10mm, proximal
            # +2mm) — the SAME tiering the post-solve ConstrainedIKSolver uses — so the
            # recommender's collision definition matches the refinement's by construction.
            # The negative adjacent tier compensates for the bounding-SPHERE over-approx and
            # goes non-negative once the link model is upgraded to exact boxes.
            _rec_finger_geoms = sorted(
                g for g in _active_finger_geoms
                if not ('_ds_' in g or g.endswith('_tip')))   # drop contact tier
            # NON-ACTIVE fingers (middle/ring) get GROUND-ONLY protection: they don't grasp
            # (so no object constraint — clearance set to the disable sentinel, which skips
            # the object test but KEEPS the floor test in the SDF loop), but their curled
            # links were dipping through the table in the committed recommender q. Proximity
            # pruning (floor-aware) drops the ones already well clear of the floor, so most
            # solves add few active constraints. Contact-tier ds/tip excluded like the active
            # fingers — a non-grasping fingertip has no reason to be near the floor plane the
            # proximal links don't already cover.
            _rec_nonactive_geoms = sorted(
                g for g in _robot_geom_names
                if (g.startswith('leap_mf_') or g.startswith('leap_rf_'))
                and not ('_ds_' in g or g.endswith('_tip')))
            _rec_arm_geoms = list(_REC_ARM_GEOMS) + _rec_finger_geoms + _rec_nonactive_geoms
            _rec_obj_clearance = {g: _active_obj_clearance(g) for g in _rec_finger_geoms}
            _rec_obj_clearance.update({g: ANOBJ_DISABLE for g in _rec_nonactive_geoms})
            print(f"[rec] recommender collision geoms: {len(_REC_ARM_GEOMS)} palm/wrist + "
                  f"{len(_rec_finger_geoms)} active-finger links + "
                  f"{len(_rec_nonactive_geoms)} non-active (ground-only) "
                  f"(tiers: adjacent -10mm, proximal +2mm)")
            cfg = GraspConfig3D(obj_geom=o['name'] + '_geom', obj_body=o['name'],
                                max_iter=120, arm_geom_names=_rec_arm_geoms,
                                obj_clearance_by_geom=_rec_obj_clearance,
                                w_align=10.0, orient_weight=2.0, edge_margin_m=0.015,
                                ground_clearance_m=0.010,   # +5mm over col_clearance for
                                                            # curled middle/ring vs the table
                                wrench_constraint=False, datum_gamma=True,
                                accel_budget_xyz=tuple(NCF_ACCEL_BUDGET_XYZ),
                                ang_accel_budget_xyz=tuple(NCF_ANG_ACCEL_BUDGET))
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
            # Warm-start from the last ACCEPTED contacts for THIS object, so a re-solve on
            # a static object returns to the same basin instead of a fresh random optimum.
            _warm = None
            with _rec_result_lock:
                _prev = _rec_result.get('candidate')
                if (_prev is not None and _rec_result.get('obj_idx') == obj_idx
                        and _prev.get('p1') is not None):
                    _warm = (_prev['p1'].copy(), _prev['p2'].copy())
            _t0 = time.time()
            try:
                res = planner.solve(q_snap, obj_pos, max_seeds=_REC_NC,
                                    warm_contacts=_warm)
            except Exception:
                traceback.print_exc()
                return
            _solve_ms = (time.time() - _t0) * 1e3
            if res.get('p1') is None or res.get('p2') is None:
                return
            # Certify wrench feasibility (datum LP inside verify()) BEFORE deciding whether
            # to show this candidate — the visualization must only ever recommend a
            # wrench-feasible grasp. Also feeds the dashboard below.
            _vinfo = {}
            try:
                _vinfo = planner._planner.verify(res) or {}
            except Exception:
                traceback.print_exc()
            _wf = bool(_vinfo.get('wrench_feasible', False))
            _new_cost = res.get('cost')
            _new_p1 = np.asarray(res['p1'], float).copy()
            _new_p2 = np.asarray(res['p2'], float).copy()
            with _rec_result_lock:
                _prev = _rec_result.get('candidate')
                _same_obj = (_rec_result.get('obj_idx') == obj_idx)
                # WF GATE: never show a grasp that is not wrench-feasible.
                if not _wf:
                    _accept = False
                    # If the object has since MOVED, the previously-shown feasible marker
                    # is stale (wrong position) — drop it rather than display a feasible
                    # grasp at an outdated location. Otherwise keep the last feasible one.
                    if (_same_obj and _prev is not None
                            and _prev.get('obj_pos') is not None
                            and float(np.linalg.norm(obj_pos - _prev['obj_pos']))
                                > _REC_OBJ_MOVE_M):
                        _rec_result.pop('candidate', None)
                else:
                    # Display hysteresis: keep the shown candidate unless (a) it is for
                    # another object, (b) the object moved, or (c) the new solve is
                    # meaningfully better. Prevents frame-to-frame marker hopping between
                    # near-equal local optima. A stale infeasible shown candidate (e.g.
                    # object just moved) is always replaced by a feasible one.
                    _prev_wf = bool(_prev.get('wrench_feasible')) if _prev else False
                    _obj_moved = True
                    if _same_obj and _prev is not None and _prev.get('obj_pos') is not None:
                        _obj_moved = (float(np.linalg.norm(obj_pos - _prev['obj_pos']))
                                      > _REC_OBJ_MOVE_M)
                    _better = True
                    if (_same_obj and not _obj_moved and _prev_wf and _prev is not None
                            and _prev.get('cost') is not None and _new_cost is not None):
                        _better = (_new_cost < _prev['cost'] * (1.0 - _REC_HYST_COST_FRAC))
                    _accept = (not _same_obj) or _obj_moved or (not _prev_wf) or _better
                if _accept:
                    _rec_result['candidate'] = {
                        'q':  np.asarray(res['q'], float).copy(),
                        'p1': _new_p1,   # thumb
                        'p2': _new_p2,   # index
                        'status': res.get('status'),
                        'cost': _new_cost,
                        'obj_pos': np.asarray(obj_pos, float).copy(),
                        'wrench_feasible': _wf,
                    }
                    _rec_result['obj_idx'] = obj_idx

            # --- Push solve stats to the dashboard (verify() gave gamma_min + IK) ---
            if dash is not None:
                _all = res.get('all_results') or [res]
                _nconv = sum(1 for r in _all if r.get('status') == 'converged')
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
            if _trial_events is not None:
                # The recommender fires continuously (proximity-based) BEFORE lock-in
                # starts a trial, so a solve for this object may predate any matching
                # trial — attribute to the current trial only if it's already running
                # for THIS object; otherwise log under trial_id=0 (pre-trial / hover),
                # still useful for the timing table but never counted into a trial's
                # own outcome accounting.
                _rec_trial_id = (_trial_state.trial_id
                                 if (_trial_state is not None
                                     and _trial_state.outcome is None
                                     and _trial_state.object_name == objects[obj_idx]['name'])
                                 else 0)
                _trial_events.log_solve(_rec_trial_id, time.time(), 'grasp_rec',
                                        _solve_ms, object=objects[obj_idx]['name'],
                                        status=res.get('status'),
                                        n_seeds=len(res.get('all_results') or [res]))
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

    if _CAT_MODE:
        print("[teleop] Keys: 8=start tracking | L=lock-in & approach | "
              "P=preview recommender q (unconstrained) | "
              "O=preview constrained IK (warm=recommender-q) | "
              "I=preview constrained IK (warm=DLS) | "
              "N=release | Backspace=reset | Q/Esc=quit")

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
    # contact_aware_teleop: when True, the GRASP phase carries the object by tracking the
    # wrist (DexPilot) instead of the arrow-key jog. Armed at the REACH->GRASP transition.
    _grasp_wrist_track = False
    _ARM_IK_ITER_SAVE  = 500       # arm IK max_iter to restore after wrist-track carry
    _jog_w         = np.zeros(3)   # slew-limited angular velocity cmd (wrist tracking)
    # Wrist-target cache: the DexPilot arm IK solve (step()) is ~ms and the ROS wrist
    # data arrives at camera rate (~30Hz), so re-solving it every 1ms sim step crushes
    # real-time. Refresh the cached target on a wall-clock interval; the P-controller
    # tracks the cached target every step.
    _wrist_tgt     = None          # cached (p_tgt, R_tgt) for pinch_site
    _wrist_tgt_t   = 0.0           # wall-clock of last refresh
    WRIST_TGT_REFRESH_S = 0.033    # ~30 Hz — matches the camera/publisher rate
    # GRASP-branch per-step timing (opt-in via GRASP_PROFILE): accumulates wall-time of
    # the wrist-track compute, torque compute, and mj_step, printing a breakdown once/sec.
    GRASP_PROFILE = os.environ.get('GRASP_PROFILE', '0') == '1'
    # Debug: WRIST_NO_REFRESH=1 freezes the wrist target after the first refresh (never
    # calls step()/spin() again). A/B this vs normal to isolate whether the per-refresh
    # DexPilot step() is the bottleneck: if the sim is smooth with this set but slow
    # without, the cost is in step(); if slow both ways, it's elsewhere (physics/loop).
    WRIST_NO_REFRESH = os.environ.get('WRIST_NO_REFRESH', '0') == '1'
    _gp_acc = {'track': 0.0, 'refresh': 0.0, 'spin': 0.0, 'step_ik': 0.0,
               'torque': 0.0, 'step': 0.0, 'n': 0}
    _gp_last = time.time()
    _PALM_BID      = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'leap_palm')
    _PINCH_SID     = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'pinch_site')
    _ik_vis_mode   = None          # None | 'grasp': freeze physics to show IK config
    _show_bspheres = False         # 7: overlay the IK's per-geom collision bounding spheres

    # Precomputed (geom_id, bounding-sphere radius, tier) for every hand geom the IK
    # constrains — this is the coarse sphere model the IK actually "sees" (finger links
    # 15-24mm), which is why the fingers can't be both constrained and reach a small
    # object. Toggle with 7.
    #
    # `tier` is the recommender-side classification we are about to enforce in the NLP:
    #   'contact'  — active ds/tip geom, clearance disabled (must touch the surface)
    #   'adjacent' — active if_md/th_px link, -10mm bounded-penetration tolerance
    #   'proximal' — active finger link that must stay +2mm off the surface
    #   'passive'  — a non-active-finger geom (palm/wrist/other fingers): full clearance
    # Coloring by tier makes the tolerance profile legible in-scene, and the penetration
    # highlight (see _draw_bspheres) shows when a sphere overlaps the ACTIVE object beyond
    # what its tier permits — exactly the -12mm bury the recommender currently can't see.
    def _bsphere_tier(g):
        if g not in _active_finger_geoms:
            return 'passive'
        clr = _active_obj_clearance(g)
        if clr <= ANOBJ_DISABLE + 1e-9:
            return 'contact'
        if clr < 0.0:
            return 'adjacent'
        return 'proximal'

    _BSPHERE_TIER_RGBA = {
        'contact':  np.array([0.30, 0.85, 0.35, 0.22], np.float32),  # green  — must touch
        'adjacent': np.array([0.95, 0.75, 0.20, 0.22], np.float32),  # amber  — -10mm ok
        'proximal': np.array([0.25, 0.55, 1.00, 0.22], np.float32),  # blue   — stay +2mm off
        'passive':  np.array([0.55, 0.55, 0.55, 0.15], np.float32),  # grey   — full clearance
    }
    _BSPHERE_VIOLATE_RGBA = np.array([1.0, 0.1, 0.1, 0.45], np.float32)  # red — over-tier
    _BSPHERES = [(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g),
                  float(model.geom_rbound[mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g)]),
                  _bsphere_tier(g),
                  _active_obj_clearance(g) if g in _active_finger_geoms else 0.005)
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
        if args.collision_view:
            # Show only the collision geoms (group 3); hide the LEAP-hand (group 1) and arm
            # (group 2) visual meshes. Lets you see the exact boxes the recommender's
            # collision model is (about to be) built from. Toggle back live with 1/2/3.
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[2] = 0
            viewer.opt.geomgroup[3] = 1
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
            you see the coarse sphere model the recommender/IK uses. No-op unless the 7
            toggle is on.

            Each sphere is COLORED BY ITS RECOMMENDER TIER (green=contact/must-touch,
            amber=adjacent/-10mm, blue=proximal/+2mm, grey=passive/full-clearance), so the
            tolerance profile we are about to enforce in the NLP is visible in-scene.

            A sphere turns RED when it penetrates the ACTIVE object's surface deeper than its
            tier permits: the bounding-sphere signed distance to the object is
            `_geom_sdf_np(center) - rb`, and the tier is violated when that drops below the
            tier's clearance. This is exactly the ~-12mm finger-link bury the recommender's
            IK-only NLP currently cannot see — turn it on to watch which link goes red at a
            recommended pose BEFORE we add the constraints, then confirm it stays tier-colored
            AFTER."""
            if not _show_bspheres or scn is None:
                return
            _eye9 = np.eye(3, dtype=np.float64).flatten()
            # Active object's geom for the penetration check (None => no highlight, just tiers).
            _obj = objects[active_idx] if 0 <= active_idx < len(objects) else None
            _ogid = _obj['id_geom'] if _obj is not None else None
            if _ogid is not None:
                _ogtype = int(model.geom_type[_ogid])
                _oc     = data.geom_xpos[_ogid].copy()
                _oR     = data.geom_xmat[_ogid].reshape(3, 3).copy()
                _osize  = model.geom_size[_ogid]
            for gid, rb, tier, clr in _BSPHERES:
                if scn.ngeom >= scn.maxgeom:
                    break
                _center = data.geom_xpos[gid].copy()
                rgba = _BSPHERE_TIER_RGBA[tier]
                if _ogid is not None:
                    # Signed distance of this bounding SPHERE to the active object surface —
                    # the same quantity the recommender's SDF constraint bounds. Below the
                    # tier's clearance => over-tier penetration => red.
                    _sd = _geom_sdf_np(_center, _ogtype, _oc, _oR, _osize) - rb
                    if _sd < clr - 1e-4:
                        rgba = _BSPHERE_VIOLATE_RGBA
                mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_SPHERE,
                                np.array([rb, rb, rb]), _center, _eye9, rgba)
                scn.ngeom += 1

        def _draw_active_marker(scn):
            """No-op: the hover sphere above the nearest object was removed — the
            recommended contact points already show which object is active. Kept as a
            stub so existing call sites need no change."""
            return

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
            _n_sub = 1   # physics catch-up substeps this iteration (dexpilot --physics)

            # Always-on pose recorder: one throttled row per iteration, ALL phases (the
            # trial trace only runs post-lock-in). Phase is reconstructed offline from
            # events.jsonl phase_enter timestamps vs `t` here. Uses last frame's qpos
            # (one-frame lag is immaterial for a warmstart snapshot).
            # (data.time < _pose_last_t marks a reset — sample immediately and re-anchor.)
            if _pose_trace is not None and (data.time < _pose_last_t
                                            or (data.time - _pose_last_t) >= _POSE_DT):
                _pose_last_t = data.time
                _pose_trace.sample(
                    t=float(data.time),          # sim-time; RESETS to 0 on backspace reset
                    t_wall=float(time.time()),   # wall-clock; monotonic across resets
                    q_robot=data.qpos[:N_ROBOT].copy(),
                    obj_qpos=data.qpos[N_ROBOT:].copy(),
                    prox_idx=int(_prox_idx))

            # --- contact_aware_teleop: teleop wrist+fingers, NLP recommends contacts.
            # Runs ONLY pre-lock-in; after L it falls through to the shared REACH/GRASP
            # machinery below (like autonomous mode) with recommended contacts. ---
            if _CAT_MODE and _teleop_active:
                _dexpilot_ctrl.spin()
                _draw_skeleton()   # fused-hand overlay (no-op unless --skeleton-view)
                _draw_camera_views()   # per-camera feed grid (no-op unless --camera-views)
                _check_pipeline_alive()  # warn if the multicam child died
                if _tune_retarget:
                    _dexpilot_ctrl.poll_retarget_config()  # hot-reload retarget_config.json edits

                # Drain keys — quit, teleop start, lock-in, and the 3 debug previews:
                #   P  recommender's raw q (unconstrained)
                #   O  collision-aware IK, warm-started from the recommender's q
                #   I  collision-aware IK, warm-started from a fresh DLS
                _do_lock_in = False
                _do_record  = False        # R: append a tuning sample this frame
                _fire_rec_ik_slot = None   # 'rec_q' | 'dls' when a constrained preview turns on
                _dp_reset_frame = False    # set by a 'reset' key: skip physics this frame
                while not keys.empty():
                    _k = keys.get_nowait()
                    if _k == 'quit':
                        running = False
                    elif _k == 'record_sample':
                        _do_record = True
                    elif _k == 'reset':
                        # Same Backspace->press-8 flow as dexpilot mode: reset robot +
                        # objects to home, FREEZE tracking (hold home, don't chase the
                        # hand), re-anchor home to the reset pose, and clear the live
                        # recommendation so stale markers/candidates don't linger. You
                        # then press 8 to re-capture your current hand pose as the offset.
                        # Backspace is the ONLY abandon path — end any running trial here,
                        # before mj_resetData zeroes data.time (keeps the duration valid).
                        if (_trial_runner is not None and _trial_state is not None
                                and _trial_state.outcome is None):
                            _trial_runner.abandon_trial(_trial_state, data.time)
                        mj.mj_resetData(model, data)
                        data.qpos[:N_ROBOT] = _Q_BIAS_DP
                        # Zero EVERY DOF's velocity/accel/applied-force (not just the
                        # robot's): teleporting the arm home while objects snap to their
                        # spawn poses can leave a penetration that the next mj_step would
                        # resolve into object velocity — the objects then fly off the pick
                        # area. _dp_reset_frame skips physics for this frame so the scene
                        # settles at rest before stepping resumes (matches dexpilot mode;
                        # matters here too now that the free-form drive runs --physics).
                        data.qvel[:]         = 0.0
                        data.qacc[:]         = 0.0
                        data.qfrc_applied[:] = 0.0
                        mj.mj_forward(model, data)
                        _dp_reset_frame = True
                        _dp_target = _Q_BIAS_DP.copy()   # physics hold target back to home
                        _dexpilot_ctrl.stop()            # freeze — no tracking until press-8
                        _dexpilot_ctrl.init_home(data)   # re-anchor home to the reset pose
                        _rec_vis        = False
                        _rec_ik_mode    = None
                        _rec_last_solve = 0.0
                        with _rec_result_lock:
                            _rec_result.clear()
                        with _rec_ik_lock:
                            _rec_ik_result.clear()
                        data.mocap_pos[_rec1_mocap] = _REC_HIDDEN
                        data.mocap_pos[_rec2_mocap] = _REC_HIDDEN
                        _last_sim_time = 0.0
                        print("[teleop] RESET — robot + objects home, tracking FROZEN. "
                              "Press 8 to set the offset (capture your current hand pose).")
                    elif _k == 'teleop_start':
                        _dexpilot_ctrl.start(data)
                        print("[teleop] tracking started — home pose captured.")
                        if _trial_runner is not None:
                            # Trial starts on press-8, same as dexpilot mode. Re-pressing
                            # 8 mid-trial abandons the running one and starts fresh; a
                            # clean end otherwise comes only from timeout or place.
                            if _trial_state is not None and _trial_state.outcome is None:
                                _trial_runner.abandon_trial(_trial_state, data.time)
                            _trial_id = (_trial_state.trial_id + 1
                                        if _trial_state is not None else 1)
                            _trial_state = _trial_runner.start_trial(
                                _trial_id, args.mode, objects[_prox_idx]['name'],
                                data.time)
                            if _cat_trigger is not None:
                                _cat_trigger.reset()
                    elif _k == 'bspheres':
                        _show_bspheres = not _show_bspheres
                        print(f"[teleop] IK collision bounding-sphere overlay "
                              f"{'ON' if _show_bspheres else 'off'}  (7 to toggle)")
                        if _show_bspheres:
                            print("           tiers: green=contact(must touch)  "
                                  "amber=adjacent(-10mm)  blue=proximal(+2mm)  "
                                  "grey=passive(full)  RED=over-tier penetration of active obj")
                    elif _k == 'lock_in':
                        _do_lock_in = True
                    elif _k == 'rec_vis':
                        _rec_vis = not _rec_vis
                        if _rec_vis:
                            _rec_ik_mode = None   # previews are mutually exclusive
                        print(f"[teleop] P: recommender-q preview (unconstrained) "
                              f"{'ON' if _rec_vis else 'off'}")
                    elif _k in ('rec_ik_recq_vis', 'rec_ik_dls_vis'):
                        _slot = 'rec_q' if _k == 'rec_ik_recq_vis' else 'dls'
                        if _rec_ik_mode == _slot:
                            _rec_ik_mode = None            # toggle off
                        else:
                            _rec_ik_mode = _slot
                            _rec_vis = False
                            _fire_rec_ik_slot = _slot      # solve this variant
                        _lbl = {'rec_q': 'O: collision-IK (warm=recommender-q)',
                                'dls':   'I: collision-IK (warm=DLS)'}[_slot]
                        print(f"[teleop] {_lbl} "
                              f"{'ON — solving ...' if _rec_ik_mode == _slot else 'off'}")
                if not running:
                    continue

                # Proximity object (min average fingertip->object signed distance).
                # Computed BEFORE the drive block so the rec-vis preview and the markers
                # both key off the current nearest object.
                _avg_d = [np.mean([_guarded_geom_dist(_tg, _o['id_geom'])
                                   for _tg in _ALL_TIP_GIDS]) for _o in objects]
                _prox_idx = int(np.argmin(_avg_d))

                # Latest recommendation candidate for the nearest object (drives both
                # the markers and the P-preview pose).
                _cand = None
                with _rec_result_lock:
                    if (_rec_result.get('candidate') is not None
                            and _rec_result.get('obj_idx') == _prox_idx):
                        _cand = _rec_result['candidate']
                        _rec_obj_idx = _prox_idx

                # Constrained-IK preview (O/I): on toggle-on, run the collision-aware
                # lock-in IK for the current candidate with the requested warm-start,
                # in the background (non-committing). Guarded so it never races the real
                # plan thread or another preview solve.
                if _fire_rec_ik_slot is not None:
                    if _cand is None:
                        print("[teleop] no recommendation yet — cannot preview collision IK")
                        _rec_ik_mode = None
                    elif (plan_thread is None
                            and (_rec_ik_thread is None or not _rec_ik_thread.is_alive())):
                        with _rec_ik_lock:
                            _rec_ik_result.pop(_fire_rec_ik_slot, None)
                        _rec_ik_thread = _fire_preview_ik(
                            _prox_idx, objects[_prox_idx],
                            data.qpos[:N_ROBOT].copy(), data.qpos[N_ROBOT:].copy(),
                            _cand, warmstart=_fire_rec_ik_slot, slot=_fire_rec_ik_slot)

                # Held constrained-IK result for the active preview mode (if solved for
                # this object).
                _rec_ik_q = None
                if _rec_ik_mode is not None:
                    with _rec_ik_lock:
                        _r = _rec_ik_result.get(_rec_ik_mode)
                        if _r is not None and _r.get('obj_idx') == _prox_idx:
                            _rec_ik_q = _r['q']

                # Drive the robot. Normally: wrist IK + MediaPipe fingers (kinematic).
                # Three mutually-exclusive debug previews hold the robot kinematically:
                #   P  -> the recommender's OWN q (unconstrained, no collision IK)
                #   O  -> collision-aware IK, warm-started from the recommender's q
                #   I  -> collision-aware IK, warm-started from a fresh DLS
                # A/B them to see the collision-vs-reach gap and warm-start effect.
                if _dp_reset_frame:
                    # Reset just teleported the scene; render it at rest without a physics
                    # step (or kinematic drive) so no teleport-penetration impulse spins up
                    # the objects. Tracking is already frozen (init_home in the handler).
                    mj.mj_forward(model, data)
                elif _rec_vis and _cand is not None and _cand.get('q') is not None:
                    data.qpos[:N_ROBOT] = Q_BIAS
                    for _i, _idx in enumerate(_cat_act_idx):
                        data.qpos[_idx] = _cand['q'][_i]
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
                elif _rec_ik_mode is not None and _rec_ik_q is not None:
                    data.qpos[:N_ROBOT] = _rec_ik_q
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
                else:
                    # Drive wrist (arm IK) + fingers (MediaPipe). Same dual-strategy
                    # split as dexpilot mode (see the --physics branch there for the
                    # full rationale): kinematic qpos overwrite by default, or —
                    # with --physics — spring-only PD torque + mj_step catch-up so
                    # the hand/arm actually collide with objects and the environment
                    # instead of teleporting through them.
                    q_teleop = _dexpilot_ctrl.step(model, data)
                    if args.physics:
                        if q_teleop is not None:
                            _dp_target = q_teleop
                        _elapsed = time.time() - step_start
                        _n_sub = int(np.clip(round(_elapsed / model.opt.timestep),
                                             1, _DP_MAX_SUBSTEPS))
                        for _ in range(_n_sub):
                            tau_ctrl[:] = 0.0
                            _err = _dp_target - data.qpos[:N_ROBOT]
                            tau_ctrl[:7]        = _dp_Kp_arm * _err[:7]
                            tau_ctrl[7:N_ROBOT] = Kp[7:] * _err[7:]
                            data.qfrc_applied[:] = tau_ctrl
                            data.qfrc_applied[:N_ROBOT] += data.qfrc_bias[:N_ROBOT]
                            mj.mj_step(model, data)
                    elif q_teleop is not None:
                        data.qpos[:N_ROBOT] = q_teleop
                        data.qvel[:N_ROBOT] = 0.0
                        mj.mj_forward(model, data)

                # Fixed-interval background NLP recommend for the nearest SUPPORTED obj.
                # Skip while previewing so the held pose isn't fed back as a solve seed.
                _supported = objects[_prox_idx]['name'] in _CAT_SUPPORTED
                _rec_idle  = (_rec_thread is None) or (not _rec_thread.is_alive())
                if (_supported and _rec_idle and not _rec_vis
                        and _rec_ik_mode is None
                        and (time.time() - _rec_last_solve) >= _REC_INTERVAL_S):
                    _q_snap = np.array([data.qpos[i] for i in _cat_act_idx])
                    _obj_pos = data.xpos[objects[_prox_idx]['id_body']].copy()
                    _rec_thread = _fire_recommender(_prox_idx, _q_snap, _obj_pos)
                    _rec_last_solve = time.time()

                # Markers: the recommender's ideal contacts p1/p2.
                if _cand is not None:
                    data.mocap_pos[_rec1_mocap] = _cand['p1']
                    data.mocap_pos[_rec2_mocap] = _cand['p2']
                else:
                    data.mocap_pos[_rec1_mocap] = _REC_HIDDEN
                    data.mocap_pos[_rec2_mocap] = _REC_HIDDEN

                # R: record a tuning sample from the LIVE pose + current recommendation.
                # data.qpos already reflects this frame's teleop drive, so it's the exact
                # warm-start/bias the lock-in IK would see. Skipped while a debug preview
                # holds the robot (the held pose isn't the operator's hand).
                if _do_record:
                    if _record_path is None:
                        print("[record] no --record-samples PATH given; R ignored.")
                    elif _rec_vis or _rec_ik_mode is not None:
                        print("[record] a preview is holding the robot — turn it off "
                              "(P/O/I) so the recorded pose is your live hand.")
                    else:
                        _n = _record_sample(data.qpos[:N_ROBOT].copy(),
                                            data.qpos[N_ROBOT:].copy(), _prox_idx, _cand)
                        if _n is None:
                            print("[record] no recommendation for the nearest object yet "
                                  "— hold near a supported box and wait for markers.")
                        else:
                            print(f"[record] sample #{_n} saved "
                                  f"({objects[_prox_idx]['name']}).")

                # Lock-in: snapshot the current recommendation, hand off to IK->RRT.
                if _do_lock_in:
                    if _cand is None:
                        print("[teleop] no recommendation yet for the nearest "
                              "supported object — hold near a box and wait.")
                    elif plan_thread is not None:
                        print("[teleop] still planning — lock-in ignored.")
                    else:
                        # If locking in from a debug preview, the sim is posed at the
                        # held preview (not the operator's hand) — re-drive one teleop
                        # step so the IK seeds from the LIVE pose, not the preview.
                        if _rec_vis or _rec_ik_mode is not None:
                            _rec_vis = False
                            _rec_ik_mode = None
                            _q_live = _dexpilot_ctrl.step(model, data)
                            if _q_live is not None:
                                data.qpos[:N_ROBOT] = _q_live
                                data.qvel[:N_ROBOT] = 0.0
                                mj.mj_forward(model, data)
                        # Lock-in no longer starts or abandons a trial: the trial began
                        # on press-8 and continues through lock-in. It ends only on
                        # timeout / place, or is abandoned by a Backspace reset.
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
                    # Pre-RRT teleop = Approach (standardized). 'Locking in' is the brief
                    # hand-off frame after L before the Pick pipeline takes over.
                    _mode = 'Locking in' if not _teleop_active else 'Approach'
                    if (_mode, active_tgt) != _dash_last_mode:
                        dash.push({'type': 'mode', 'mode': _mode,
                                   'target': objects[_prox_idx]['name']})
                        _dash_last_mode = (_mode, active_tgt)
                    if _prox_idx != _dash_last_prox:
                        dash.push({'type': 'active_obj',
                                   'name': objects[_prox_idx]['name']})
                        _dash_last_prox = _prox_idx
                    if _dash_i % DASH_PUSH_EVERY == 0:
                        _push_trial_time(_trial_state, data.time)
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
                    _draw_bspheres(viewer.user_scn)   # 7-toggle: IK collision spheres
                viewer.sync()
                # Pace to real time. In --physics we advanced _n_sub timesteps this
                # iteration (see the dexpilot --physics branch above for why), so the
                # wall-clock budget scales with that instead of a single timestep.
                _budget = (_n_sub if args.physics else 1) * model.opt.timestep
                time.sleep(max(0, _budget - (time.time() - step_start)))
                continue

            # --- DexPilot teleop mode: bypass the RRT/grasp state machine ---
            if args.mode == 'dexpilot':
                _dpp_iter0 = time.perf_counter() if DP_PROFILE else 0.0
                _dexpilot_ctrl.spin()
                if DP_PROFILE:
                    _t = time.perf_counter(); _dpp_acc['spin'] += _t - _dpp_iter0; _dpp_s = _t
                _draw_skeleton()   # fused-hand overlay (no-op unless --skeleton-view)
                if DP_PROFILE:
                    _t = time.perf_counter(); _dpp_acc['skel'] += _t - _dpp_s; _dpp_s = _t
                _draw_camera_views()   # per-camera feed grid (no-op unless --camera-views)
                if DP_PROFILE:
                    _t = time.perf_counter(); _dpp_acc['camviews'] += _t - _dpp_s
                _check_pipeline_alive()  # warn if the multicam child died
                if _tune_retarget:
                    _dexpilot_ctrl.poll_retarget_config()  # hot-reload retarget_config.json edits
                if DP_PROFILE:
                    _dpp_acc['spin_draw'] += time.perf_counter() - _dpp_iter0
                _dp_reset_frame = False   # set by a 'reset' key: skip physics this frame
                # Drain key queue — handle quit and teleop start/re-zero
                while not keys.empty():
                    _k = keys.get_nowait()
                    if _k == 'quit':
                        running = False
                    elif _k == 'reset':
                        # Reset to the startup state: robot + objects to home, and
                        # FREEZE tracking (stop chasing the hand) so the robot holds
                        # the wrist-down home pose. Just like launch, you then press 8
                        # to re-capture your current hand pose as the offset.
                        if (_trial_runner is not None and _trial_state is not None
                                and _trial_state.outcome is None):
                            _trial_runner.abandon_trial(_trial_state, data.time)
                        mj.mj_resetData(model, data)
                        data.qpos[:N_ROBOT] = _Q_BIAS_DP
                        # Zero EVERY DOF's velocity/accel/applied-force (not just the
                        # robot's): teleporting the arm home while objects snap to their
                        # spawn poses can leave a penetration that the next mj_step would
                        # resolve into object velocity — the objects then fly off the
                        # pick area. _dp_reset_frame skips physics for this frame so the
                        # scene settles at rest before stepping resumes.
                        data.qvel[:]         = 0.0
                        data.qacc[:]         = 0.0
                        data.qfrc_applied[:] = 0.0
                        mj.mj_forward(model, data)
                        _dp_reset_frame = True
                        _dexpilot_ctrl.stop()        # freeze — no tracking until press-8
                        _dexpilot_ctrl.init_home(data)   # re-anchor home to the reset pose
                        _dp_target = _Q_BIAS_DP.copy()   # physics hold target back to home
                        _calib_mode = False
                        print("[dexpilot] RESET — robot + objects home, tracking FROZEN. "
                              "Press 8 to set the offset (capture your current hand pose).")
                    elif _k == 'teleop_start':
                        # Snapshot current human pose as home and begin tracking.
                        _dexpilot_ctrl.start(data)
                        print("[dexpilot] tracking started — home pose captured "
                              "(hold your hand at the desired neutral orientation).")
                        if _trial_runner is not None:
                            # DexPilot has no per-object selection state (see
                            # trial_logger.py wiring notes) — the scene is expected to
                            # contain exactly ONE manipulated object per trial batch, so
                            # objects[0] is unambiguously the target. Re-pressing 8
                            # mid-trial abandons whatever was running, same as the other
                            # modes' mid-trial supersession handling.
                            if _trial_state is not None and _trial_state.outcome is None:
                                _trial_runner.abandon_trial(_trial_state, data.time)
                            _trial_id = (_trial_state.trial_id + 1
                                        if _trial_state is not None else 1)
                            _trial_state = _trial_runner.start_trial(
                                _trial_id, 'dexpilot', objects[0]['name'], data.time)
                            _dp_trigger.reset()
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
                elif _dp_reset_frame:
                    # Reset just teleported the scene; render it at rest without a
                    # physics step so no teleport-penetration impulse spins up the
                    # objects. Tracking is already frozen (init_home above).
                    mj.mj_forward(model, data)
                else:
                    _dpp_t0 = time.perf_counter() if DP_PROFILE else 0.0
                    q_teleop = _dexpilot_ctrl.step(model, data)
                    if DP_PROFILE:
                        _dpp_acc['step'] += time.perf_counter() - _dpp_t0
                    if args.physics:
                        # PHYSICS teleop: drive the robot toward the retarget target
                        # with a SPRING-only torque (Kp) + gravity comp and mj_step,
                        # so the arm/hand physically collide with objects/floor/self.
                        # Damping is NOT applied explicitly here: the earlier
                        # qfrc_applied Kd term violates dt < 2*I/Kd on the low-inertia
                        # finger DOFs (I~2e-4) at dt=2ms and caused BADQACC resets.
                        # Instead we rely on the model's implicit dof_damping (added
                        # for the arm in the --physics setup), which the implicitfast
                        # integrator handles with NO stability limit. Hold the last
                        # target (_dp_target, home until the first pose) when no fresh
                        # teleop pose arrived, so contact can't drift the robot.
                        if q_teleop is not None:
                            _dp_target = q_teleop
                        # REAL-TIME CATCH-UP: the per-iteration work is dominated by
                        # the retarget SLSQP solve (~1.7 ms) + arm IK + draw + sync,
                        # which EXCEEDS one timestep (2 ms). Stepping physics once per
                        # iteration then advances sim slower than wall time -> objects
                        # fall in slow motion. mj_step is cheap (~0.03 ms), so we take
                        # as many steps as fit the elapsed wall time (capped, so a hitch
                        # can't spiral), re-applying the cheap PD torque each substep.
                        # step() (the expensive IK) still runs ONCE per iteration above;
                        # the PD tracks the cached _dp_target across the substeps.
                        _elapsed = time.time() - step_start
                        _n_sub = int(np.clip(round(_elapsed / model.opt.timestep),
                                             1, _DP_MAX_SUBSTEPS))
                        _dpp_t0 = time.perf_counter() if DP_PROFILE else 0.0
                        _dpp_t0 = time.perf_counter() if DP_PROFILE else 0.0
                        for _ in range(_n_sub):
                            tau_ctrl[:] = 0.0
                            # Arm: inertia-scaled Kp (_dp_Kp_arm, ~3 Hz tracking).
                            # Fingers: shared Kp[7:] (light, they're low-inertia).
                            _err = _dp_target - data.qpos[:N_ROBOT]
                            tau_ctrl[:7]        = _dp_Kp_arm * _err[:7]
                            tau_ctrl[7:N_ROBOT] = Kp[7:] * _err[7:]
                            data.qfrc_applied[:] = tau_ctrl
                            data.qfrc_applied[:N_ROBOT] += data.qfrc_bias[:N_ROBOT]
                            mj.mj_step(model, data)
                        if DP_PROFILE:
                            _dpp_acc['substeps'] += time.perf_counter() - _dpp_t0
                    elif q_teleop is not None:
                        # KINEMATIC replay: overwrite qpos (rigid, no collisions).
                        data.qpos[:N_ROBOT] = q_teleop
                        data.qvel[:N_ROBOT] = 0.0
                        mj.mj_forward(model, data)

                _dpp_t0 = time.perf_counter() if DP_PROFILE else 0.0
                # --- Detect out-of-band resets (MuJoCo's BADQACC auto-reset on
                # numerical divergence, or the viewer's own Reset button) — same
                # backstop as the shared autonomous/contact_aware_teleop loop
                # (_last_sim_time, defined above). Without this, an out-of-band reset
                # snaps data.time back near 0 with NO key event to catch it, silently
                # corrupting a running trial's timeout math (t_now - t_start going
                # negative) instead of cleanly abandoning it.
                if data.time < _last_sim_time - 1e-12:
                    if (_trial_runner is not None and _trial_state is not None
                            and _trial_state.outcome is None):
                        _trial_runner.abandon_trial(_trial_state, _last_sim_time)
                _last_sim_time = data.time

                # --- Trial benchmarking (--trial-log, dexpilot): DexPilot has no
                # control_phase / squeeze_on / object-selection state at all (see
                # trial_logger.py wiring notes), so approach-contact-counting and the
                # pick/transport state machine run CONCURRENTLY from trial_start
                # onward, both keyed to objects[0] (the scene's one manipulated
                # object — see the single-object-per-trial design). No-op unless
                # --trial-log was passed (implies --physics, forced above).
                if (_trial_runner is not None and _trial_state is not None
                        and _trial_state.outcome is None):
                    _tnow_dp = data.time
                    _dp_obj  = objects[0]
                    _trial_runner.step_approach(
                        _trial_state, _tnow_dp, data.contact[:data.ncon],
                        _HAND_GIDS, _dp_obj['id_geom'])
                    _hh_dp = _trial_rest_hh[0]
                    _height_above_rest_dp = (float(data.geom_xpos[_dp_obj['id_geom']][2])
                                             - _hh_dp)
                    _d_s1_dp = _dexpilot_ctrl.retargeter.last_d_s1
                    _touching_dp = _dp_obj['id_geom'] in {
                        (c.geom2 if c.geom1 in _HAND_GIDS else c.geom1)
                        for c in data.contact[:data.ncon]
                        if c.geom1 in _HAND_GIDS or c.geom2 in _HAND_GIDS}
                    _fired_dp  = _dp_trigger.update(_d_s1_dp, _touching_dp)
                    _active_dp = (min(_d_s1_dp) < _dp_trigger.eps) and _touching_dp
                    _sid_dp = _trial_place_sid.get(0, -1)
                    _xy_off_dp = _spd_dp = None
                    if _sid_dp >= 0:
                        _xy_off_dp = float(np.linalg.norm(
                            data.geom_xpos[_dp_obj['id_geom']][:2]
                            - data.site_xpos[_sid_dp][:2]))
                        _dofadr_dp = _trial_dofadr[0]
                        _spd_dp = float(np.linalg.norm(data.qvel[_dofadr_dp:_dofadr_dp + 3]))
                    _arrived_dp = _trial_runner.step_pick_or_transport(
                        _trial_state, _tnow_dp, trigger_fired=_fired_dp,
                        trigger_active=_active_dp,
                        height_above_rest=_height_above_rest_dp,
                        place_xy_offset=_xy_off_dp, object_speed=_spd_dp)
                    _dp_thumb_sid = id_C[FINGER_SET.index('thumb')]
                    _dp_index_sid = id_C[FINGER_SET.index('index')]
                    _trial_runner.trace.sample(
                        t=_tnow_dp,
                        p_thumb=data.site_xpos[_dp_thumb_sid].copy(),
                        p_index=data.site_xpos[_dp_index_sid].copy(),
                        obj_pos=data.geom_xpos[_dp_obj['id_geom']].copy(),
                        obj_quat=data.xquat[_dp_obj['id_body']].copy(),
                        obj_linvel=data.qvel[_trial_dofadr[0]:_trial_dofadr[0] + 3].copy(),
                        height_above_rest=_height_above_rest_dp,
                        phase=1 if _trial_state.phase == TrialPhase.TRANSPORT else 0)
                    if (_arrived_dp
                            or _trial_runner.check_timeout(_trial_state, _tnow_dp)):
                        _trial_runner.end_trial(_trial_state, _tnow_dp)
                if DP_PROFILE:
                    _dpp_acc['trial_log'] += time.perf_counter() - _dpp_t0
                    _dpp_t0 = time.perf_counter()

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
                # Trial countdown to the dashboard. The dexpilot branch has its own
                # viewer.sync()+continue and never falls through to the shared dashboard
                # block below, so the timer push lives here (throttled like the shared
                # streams). Event-log rows arrive independently via the EventLogger tee.
                if dash is not None and _dash_i % DASH_PUSH_EVERY == 0:
                    _push_trial_time(_trial_state, data.time)
                _dash_i += 1
                if DP_PROFILE:
                    _dpp_acc['viz_sync'] += time.perf_counter() - _dpp_t0
                    _dpp_acc['iter'] += time.perf_counter() - _dpp_iter0
                    _dpp_acc['n'] += 1
                    if time.time() - _dpp_last >= 1.0:
                        _n = max(_dpp_acc['n'], 1)
                        print(f"\r\n[dexpilot-profile] {_dpp_acc['n']} iters/s | "
                              f"per-iter ms: "
                              f"spin_draw={_dpp_acc['spin_draw']/_n*1e3:.3f} "
                              f"[spin={_dpp_acc['spin']/_n*1e3:.3f} "
                              f"skel={_dpp_acc['skel']/_n*1e3:.3f} "
                              f"camviews={_dpp_acc['camviews']/_n*1e3:.3f}] "
                              f"step={_dpp_acc['step']/_n*1e3:.3f} "
                              f"substeps={_dpp_acc['substeps']/_n*1e3:.3f} "
                              f"trial_log={_dpp_acc['trial_log']/_n*1e3:.3f} "
                              f"viz_sync={_dpp_acc['viz_sync']/_n*1e3:.3f} "
                              f"| TOTAL={_dpp_acc['iter']/_n*1e3:.3f}  "
                              f"(budget/iter={model.opt.timestep*1e3:.2f}ms x up to "
                              f"{_DP_MAX_SUBSTEPS} substeps = "
                              f"{model.opt.timestep*_DP_MAX_SUBSTEPS*1e3:.1f}ms cap)")
                        _dpp_acc = {'spin_draw': 0.0, 'spin': 0.0, 'skel': 0.0,
                                    'camviews': 0.0, 'step': 0.0, 'substeps': 0.0,
                                    'trial_log': 0.0, 'viz_sync': 0.0, 'iter': 0.0,
                                    'n': 0}
                        _dpp_last = time.time()
                # Pace to real time. In --physics we advanced _n_sub timesteps this
                # iteration, so the wall-clock budget is _n_sub*timestep (else one).
                # On a fast machine this sleeps the remainder; when the iteration
                # already overran (the common case — retarget dominates), it's ~0 and
                # the catch-up substeps keep sim time matched to wall time.
                _budget = (_n_sub if args.physics else 1) * model.opt.timestep
                time.sleep(max(0, _budget - (time.time() - step_start)))
                continue

            # --- Proximity "active object" (min average tip→object signed distance) +
            # dashboard streams: planning mode, distances, net wrench, normal forces ---
            if _dash_i % DASH_PUSH_EVERY == 0:
                _avg_d = [np.mean([_guarded_geom_dist(_tg, _o['id_geom'])
                                   for _tg in _ALL_TIP_GIDS]) for _o in objects]
                _prox_idx = int(np.argmin(_avg_d))
            if dash is not None:
                # Standardized display terminology (see TrialPhase): the whole post-lock-in
                # RRT+grasp pipeline (PLAN/REACH/GRASP) is PICK; Transport while the object
                # is being jogged/lifted (arrow keys held). Pre-lock-in teleop shows
                # 'Approach' from the teleop branch's own _mode (see the _teleop_active path).
                _mode = 'Transport' if _held_arrows else 'Pick'
                if (_mode, active_tgt) != _dash_last_mode:
                    dash.push({'type': 'mode', 'mode': _mode,
                               'target': targets[active_tgt]['label']})
                    _dash_last_mode = (_mode, active_tgt)
                if _prox_idx != _dash_last_prox:
                    dash.push({'type': 'active_obj', 'name': objects[_prox_idx]['name']})
                    _dash_last_prox = _prox_idx
                if _dash_i % DASH_PUSH_EVERY == 0:
                    _push_trial_time(_trial_state, data.time)
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
                        # Teleop lock-in: snap the recommendation markers to the ACHIEVED
                        # fingertip contacts (rec_achieved_W in FINGER_SET=[index,thumb]
                        # order → rec1=thumb, rec2=index). These track the object during
                        # transport via the same rec_local frames used by the grasp
                        # provider, so recompute them from the live object pose each frame
                        # below; here just enable them by clearing the hidden flag.
                        _ach = objects[active_idx].get('rec_achieved_W')
                        if _CAT_MODE and _ach is not None:
                            _idx_by_f = {f: i for i, f in enumerate(FINGER_SET)}
                            data.mocap_pos[_rec1_mocap] = _ach[_idx_by_f['thumb']]
                            data.mocap_pos[_rec2_mocap] = _ach[_idx_by_f['index']]
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
                    _jog_w[:] = 0.0
                    _grasp_wrist_track = False   # (re-armed below for teleop)

                    # DIAGNOSTIC: RRT-END tip error. The RRT goal IS obj['q_target'] (now
                    # the recommender's committed pose), so a planned end pose reaches that
                    # pose exactly; any EXTRA error here is the object drifting during replay
                    # (the tips aim at where the object WAS). Measured against the recommended
                    # contacts re-expressed at the object's CURRENT pose (via rec_local, which
                    # tracks the object) so drift shows up. Compare to the committed
                    # recommender-pose tip error printed at lock-in: if this is larger, the
                    # object moved.
                    _rec_local_dbg = objects[active_idx].get('rec_local')
                    if _CAT_MODE and _rec_local_dbg is not None:
                        _if_dbg = {f: i for i, f in enumerate(FINGER_SET)}
                        _end_err = []
                        for f in FINGER_SET:
                            _sid = id_C[_if_dbg[f]]
                            _pO, _ = _rec_local_dbg[_if_dbg[f]]
                            _p_contact_now = p_WoO + R_WO @ _pO   # live object pose
                            _end_err.append(
                                float(np.linalg.norm(data.site_xpos[_sid]
                                                     - _p_contact_now) * 1e3))
                        _ik_ce = objects[active_idx].get('rec_ik_contact_err_mm')
                        _rrt_lines = [
                            "",   # leading blank line separates this from prior output
                            f"[IK] obj{active_idx+1} RRT-END tip errors "
                            f"(site->contact @ live obj, {FINGER_SET}): "
                            f"{[f'{e:.1f}' for e in _end_err]} mm",
                        ]
                        if _ik_ce is not None:
                            _drift = [ee - ce for ee, ce in zip(_end_err, _ik_ce)]
                            _rrt_lines.append(
                                f"       vs lock-in IK "
                                f"{[f'{e:.1f}' for e in _ik_ce]} mm  ->  drift "
                                f"{[f'{d:+.1f}' for d in _drift]} mm "
                                f"(object moved during replay)")
                        print("\n".join(_rrt_lines))
                        # Complete the dashboard's attribution readout with the RRT-end
                        # stage (overwrites the NLP+IK row pushed at lock-in).
                        if dash is not None:
                            dash.push({'type': 'tip_err',
                                       'object': objects[active_idx]['name'],
                                       'fingers': list(FINGER_SET),
                                       'nlp': objects[active_idx].get('rec_nlp_err_mm'),
                                       'ik':  _ik_ce, 'rrt': _end_err})

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
                    # Gravity component per OBJECT-body axis. In CoM mode it is folded
                    # into the linear budget; in datum mode it is passed SEPARATELY to
                    # solve_gamma_live as grav_O (re-datumed, grasp-axis moment projected)
                    # so the budget stays a pure accel box referenced at the grasp datum.
                    _g_O   = R_WO.T @ model.opt.gravity           # (3,), object frame
                    if NCF_DATUM_MODE:
                        _accel_box = tuple(NCF_ACCEL_BUDGET_XYZ)
                    else:
                        _accel_box = tuple(NCF_ACCEL_BUDGET_XYZ[i] + abs(_g_O[i]) for i in range(3))
                    _inertia   = model.body_inertia[_bid]         # principal moments (Ix,Iy,Iz)
                    # A 2-contact antipodal pinch geometrically CANNOT resist torque about
                    # the grasp axis (the line through the two contacts) — the friction
                    # cones have no moment arm about it. In datum mode the LP removes that
                    # component EXACTLY per corner (project_grasp_axis_torque=True inside
                    # solve_gamma_live), so we pass the FULL per-axis angular budget here and
                    # let the LP project — no lossy budget-vector pre-projection (which for a
                    # tilted grasp axis would leave off-axis residuals and mishandle the box
                    # corners). The remaining OFF-axis torque capacity of a raised pinch is
                    # small (see RAISED_CONTACT_WRENCH_FINDINGS.md), so NCF_ANG_ACCEL_BUDGET
                    # is set to a hold-task value.
                    _ang_budget = np.abs(np.array(NCF_ANG_ACCEL_BUDGET, float))
                    _gamma = solve_gamma_live(_p_O, _R_in, _mu, _mass,
                                              _accel_box, tuple(_ang_budget), _inertia,
                                              grav_O=(_g_O if NCF_DATUM_MODE else None))
                    if _gamma is None or not np.isfinite(_gamma) or _gamma <= 0.0:
                        gamma_raw  = GAMMA_FALLBACK
                        gamma_live = GAMMA_FALLBACK
                        print(f"\r\n[gamma] LP infeasible/degenerate for "
                              f"{obj_grasp['name']} — using fallback {GAMMA_FALLBACK:.0f}")
                    else:
                        # gamma_raw = the LP's minimum no-slip gamma (the true feasible
                        # boundary, reported in the log). gamma_live = raw * safety factor
                        # is what actually squeezes AND what the wrench cone is drawn at.
                        gamma_raw  = float(_gamma)
                        gamma_live = gamma_raw * GAMMA_SAFETY_FACTOR
                        print(f"\r\n[gamma] {obj_grasp['name']}: solved gamma={gamma_raw:.2f} "
                              f"x{GAMMA_SAFETY_FACTOR:.1f} = {gamma_live:.2f} "
                              f"(mass={_mass:.3f}kg mu={_mu[0]:.1f}, "
                              f"~{gamma_live/np.sqrt(2):.2f} N/contact)")

                    # --- Grasp-quality diagnostic at the COMMITTED pose ---
                    # The lift test calls a grasp "good" at pad-alignment <~15deg and pad-gap
                    # ~0-8mm. Print the SAME metrics for the live committed grasp so a live
                    # slip can be attributed: an oblique pad (high angle) or a large gap can't
                    # hold the weight even though the contacts "look" antipodal. _p_O/_R_in are
                    # the object-frame contact frames (col0 = inward normal); pad-vs-inward is
                    # the alignment; the tip-geom-to-object exact distance is the pad gap.
                    try:
                        _gq_ft = np.zeros(6)
                        _gq_parts = []
                        for _k, _f in enumerate(FINGER_SET):
                            _nin_W = R_WO @ _R_in[_k][:, 0]        # inward normal, world
                            _pad_W = -data.site_xmat[id_C[_k]].reshape(3, 3)[:, 0]
                            _ang = np.degrees(np.arccos(np.clip(_pad_W @ _nin_W, -1, 1)))
                            _tg = _TIP_GEOM_IDS[_f]
                            _gap = mj.mj_geomDistance(model, data, _tg,
                                                      obj_grasp['id_geom'], 0.1, _gq_ft) * 1e3
                            _gq_parts.append(f"{_f}: pad_align={_ang:.0f}deg gap={_gap:.1f}mm")
                        print("[grasp-quality] " + "  |  ".join(_gq_parts)
                              + "   (good: align<15deg, gap 0-8mm)")
                    except Exception:
                        traceback.print_exc()

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
                    # Teleop: apply the squeeze IMMEDIATELY on the first Enter (no need to
                    # press Enter twice) — the ramp (_squeeze_steps/SQUEEZE_RAMP_S) still
                    # eases the force in so it's not a shove. Autonomous keeps the explicit
                    # toggle so you can inspect the pregrasp before committing force.
                    squeeze_on = bool(_CAT_MODE)
                    _squeeze_steps = 0
                    grasp_ctrl.set_squeeze(squeeze_on)
                    _push_squeeze(squeeze_on, gamma_live)
                    # Draw the wrench cone at the APPLIED squeeze (gamma_live = raw x
                    # GAMMA_SAFETY_FACTOR) so the cage reflects the safety multiplier —
                    # it grows with the factor, making its effect visible as you tune it.
                    # (The raw 1.0x cone is the minimum feasible boundary; the applied
                    # cone is GAMMA_SAFETY_FACTOR times larger and the live trace sits
                    # well inside it.)
                    _push_wrench_cone(gamma_live, _p_O, _R_in, _mu)
                    print(f"\r\n[Control] → GRASP  ({targets[active_tgt]['label']})  "
                          f"|  Enter: toggle squeeze (gamma={gamma_live:.1f})  |  N: release")

                    # contact_aware_teleop: after grasp, the WRIST follows your hand
                    # (position + orientation, like the approach) while the fingers and
                    # squeeze stay frozen. Re-zero DexPilot to the current robot/hand
                    # pose (== press-8 recalibration) so tracking starts jerk-free. The
                    # arm PD target (q_grasp_hold[:7]) then slew-tracks the wrist target
                    # in the GRASP branch below. Autonomous mode keeps the arrow-key jog.
                    if _CAT_MODE:
                        _dexpilot_ctrl.start(data)
                        # Disable finger retargeting (the ~40ms scipy SLSQP solve) for
                        # the carry: the fingers are frozen at the grasp config, so we
                        # only need the wrist pose from step().
                        _dexpilot_ctrl._hand_tracking = False
                        # Cap the arm IK iterations. The wrist target (position + full
                        # orientation) often does NOT converge to the 1e-3 tol from the
                        # grasp arm config, so the DLS burns its full 500-iter cap
                        # (~38ms) EVERY refresh — the real GRASP-phase stall. 20 DLS iters
                        # give a smooth setpoint the slew-limiter tracks anyway (~1.5ms
                        # worst case). Restored on release/reset.
                        _ARM_IK_ITER_SAVE = _dexpilot_ctrl._arm._ik.max_iter
                        _dexpilot_ctrl._arm._ik.max_iter = 20
                        _grasp_wrist_track = True
                        _wrist_tgt   = None    # force a fresh target on the first step
                        _wrist_tgt_t = 0.0
                        print("[teleop] wrist tracking armed — move your hand to carry "
                              "the object (fingers/squeeze held).")

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
                    _grasp_wrist_track = False
                    _wrist_tgt = None
                    _jog_w[:] = 0.0
                    if _CAT_MODE:
                        _dexpilot_ctrl._hand_tracking = True   # restore finger retargeting
                        _dexpilot_ctrl._arm._ik.max_iter = _ARM_IK_ITER_SAVE
                    if _CAT_MODE:
                        # Teleop: hand control back to the operator (all 23 DOFs are
                        # teleoped), drop the object, and re-arm the recommender. Clear
                        # the committed contacts so the next lock-in re-solves fresh.
                        objects[active_idx].pop('rec_local', None)
                        _teleop_active = True
                        _rec_vis       = False
                        _rec_ik_mode   = None
                        active_tgt     = 0
                        active_idx     = 0
                        _rec_last_solve = 0.0
                        with _rec_result_lock:
                            _rec_result.clear()
                        with _rec_ik_lock:
                            _rec_ik_result.clear()
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
                    if _show_bspheres:
                        print("           tiers: green=contact(must touch)  "
                              "amber=adjacent(-10mm)  blue=proximal(+2mm)  "
                              "grey=passive(full)  RED=over-tier penetration of active obj")

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
                        if (_trial_runner is not None and _trial_state is not None
                                and _trial_state.outcome is None):
                            # ANY new selection (home, a different object, or a re-plan
                            # on the SAME object) preempts a still-running trial — force-
                            # end it rather than silently orphaning it below. Re-planning
                            # the same object starts a fresh trial_id, consistent with
                            # the lock-in path's re-lock-in handling.
                            _trial_runner.abandon_trial(_trial_state, data.time)
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
                            if args.recommender_grasp:
                                # Autonomous with RECOMMENDER contacts: always re-solve the NLP
                                # from the current pose (no IK cache reuse — the recommender is
                                # cheap enough and its contacts depend on q_start), commit its
                                # pose, then RRT. Routes through every recommender fix.
                                plan_thread = threading.Thread(
                                    target=_plan_thread_main,
                                    args=(_run_recommender_then_rrt, obj_i, objects[obj_i],
                                          q_start, obj_qpos_snap),
                                    daemon=True)
                            elif _ik_fresh:
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
                            if _trial_runner is not None:
                                _trial_id = (_trial_state.trial_id + 1
                                            if _trial_state is not None else 1)
                                _trial_state = _trial_runner.start_trial(
                                    _trial_id, args.mode, objects[obj_i]['name'],
                                    data.time)
                                if _dp_trigger is not None:
                                    _dp_trigger.reset()
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
                if (_trial_runner is not None and _trial_state is not None
                        and _trial_state.outcome is None):
                    # Backspace re-homes the whole scene mid-trial — force-end it rather
                    # than leaving it orphaned across the reset.
                    _trial_runner.abandon_trial(_trial_state, data.time)
                if plan_thread is not None:
                    _plan_discard = True   # in-flight IK/RRT started pre-reset: drop it
                mj.mj_resetData(model, data)
                data.qpos[:N_ROBOT] = Q_BIAS   # arm/hand home (qpos0 zero = straight up)
                # mj_resetData zeros qvel, but writing the arm to Q_BIAS while the
                # objects snap back to their (possibly gripper-overlapping) spawn poses
                # sets up a penetration that the NEXT mj_step resolves with a large
                # contact impulse — the objects pick up velocity and fly off the pick
                # area. Zero every DOF's velocity and clear any leftover applied force,
                # then skip physics for this frame (render-only) so the reset scene
                # settles at rest before stepping resumes.
                data.qvel[:]         = 0.0
                data.qacc[:]         = 0.0
                data.qfrc_applied[:] = 0.0
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
                _jog_w[:]      = 0.0
                _grasp_wrist_track = False
                _wrist_tgt     = None
                if _CAT_MODE and _dexpilot_ctrl is not None:
                    _dexpilot_ctrl._hand_tracking = True   # restore finger retargeting
                    _dexpilot_ctrl._arm._ik.max_iter = _ARM_IK_ITER_SAVE
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
                    _rec_vis        = False
                    _rec_ik_mode    = None
                    _rec_last_solve = 0.0
                    with _rec_result_lock:
                        _rec_result.clear()
                    with _rec_ik_lock:
                        _rec_ik_result.clear()
                    data.mocap_pos[_rec1_mocap] = _REC_HIDDEN
                    data.mocap_pos[_rec2_mocap] = _REC_HIDDEN
                    # FREEZE tracking so the robot holds home instead of snapping to the
                    # hand — like startup, you press 8 to re-capture the offset.
                    if _dexpilot_ctrl is not None:
                        _dexpilot_ctrl.stop()
                        _dexpilot_ctrl.init_home(data)   # re-anchor home to the reset pose
                print("\r\n[Control] RESET — arm home, objects at spawn poses; cached "
                      "IK kept (auto re-solved if stale on next selection)."
                      + ("  Tracking FROZEN — press 8 to set the offset."
                         if _CAT_MODE and _dexpilot_ctrl is not None else ""))
                # Render this frame WITHOUT mj_step: stepping now would resolve any
                # residual arm/object penetration from the teleport with a contact
                # impulse, re-injecting the very velocity mj_resetData just cleared.
                # Next iteration steps from a settled, zero-velocity rest state.
                _render_kinematic_frame()
                continue

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
                # Build a 6-DOF Cartesian velocity command for the wrist, from one of
                # two sources, then map it to arm joint rates via the SAME singularity-
                # robust DLS + PD-target integration:
                #   autonomous  -> arrow-key palm velocity (position only, orient held)
                #   teleop      -> track the DexPilot wrist target (position+orientation)
                # Both are slew-limited so the commanded palm ACCELERATION stays within
                # NCF_ACCEL_BUDGET_XYZ (linear) — the box the gamma squeeze was solved
                # for — so tracking can lag a fast hand but never exceeds the no-slip
                # guarantee. The site tracked is pinch_site (teleop) / leap_palm (jog).
                _t_track0 = time.perf_counter() if GRASP_PROFILE else 0.0
                _dv_max   = np.array(NCF_ACCEL_BUDGET_XYZ) * model.opt.timestep
                if _grasp_wrist_track:
                    # Refresh the wrist TARGET pose (position + full orientation) at camera
                    # rate only — the arm IK solve inside step() is far too costly to run
                    # every 1ms sim step. Between refreshes, keep tracking the cached
                    # target (it's a setpoint, so holding it a few ms is fine).
                    _now = time.time()
                    if (_now - _wrist_tgt_t >= WRIST_TGT_REFRESH_S
                            and not (WRIST_NO_REFRESH and _wrist_tgt is not None)):
                        _t_ref0 = time.perf_counter() if GRASP_PROFILE else 0.0
                        _dexpilot_ctrl.spin()
                        _t_spin = time.perf_counter() if GRASP_PROFILE else 0.0
                        # step() runs the arm IK for the wrist target. Finger retargeting
                        # (a ~40ms scipy SLSQP solve) is DISABLED here — the fingers are
                        # frozen at the grasp config, so we only need the wrist pose. That
                        # SLSQP was the sole cause of the GRASP-phase stall (~40ms/refresh).
                        _dexpilot_ctrl.step(model, data)   # updates the internal target frame
                        _tf = _dexpilot_ctrl.target_frame()
                        if _tf is not None and _tf[1] is not None:
                            _wrist_tgt = (np.asarray(_tf[0]).copy(),
                                          np.asarray(_tf[1]).copy())
                        _wrist_tgt_t = _now
                        if GRASP_PROFILE:
                            _now2 = time.perf_counter()
                            _gp_acc['refresh'] += _now2 - _t_ref0
                            _gp_acc['spin'] += _t_spin - _t_ref0
                            _gp_acc['step_ik'] += _now2 - _t_spin
                    _p_cur = data.site_xpos[_PINCH_SID]
                    _R_cur = data.site_xmat[_PINCH_SID].reshape(3, 3)
                    if _wrist_tgt is not None:
                        _p_tgt, _R_tgt = _wrist_tgt
                        # P-control toward the target pose: desired Cartesian velocity =
                        # gain * pose error. Slew-limited below so accel stays in budget.
                        _v_lin = WRIST_TRACK_GAIN * (_p_tgt - _p_cur)
                        _R_err = _R_tgt @ _R_cur.T
                        _ang   = np.array([_R_err[2, 1] - _R_err[1, 2],
                                           _R_err[0, 2] - _R_err[2, 0],
                                           _R_err[1, 0] - _R_err[0, 1]]) * 0.5
                        _v_ang = WRIST_TRACK_GAIN * _ang
                    else:
                        _v_lin = np.zeros(3)
                        _v_ang = np.zeros(3)
                    _v_lin = np.clip(_v_lin, -JOG_VEL, JOG_VEL)   # cap peak speed
                    _jog_v += np.clip(_v_lin - _jog_v, -_dv_max, _dv_max)
                    _jog_w += np.clip(_v_ang - _jog_w, -_dv_max, _dv_max)
                    _jac_bid_or_sid = _PINCH_SID
                    _use_site = True
                else:
                    # Arrow-key jog: world-frame palm velocity, orientation held.
                    _v_target = np.array([_jv_x, _jv_y, _jv_z])
                    _jog_v   += np.clip(_v_target - _jog_v, -_dv_max, _dv_max)
                    _jog_w[:] = 0.0
                    _use_site = False

                qdot_jog = np.zeros(7)
                if np.any(_jog_v) or np.any(_jog_w):
                    Jp = np.zeros((3, model.nv))
                    Jr = np.zeros((3, model.nv))
                    if _use_site:
                        mj.mj_jacSite(model, data, Jp, Jr, _PINCH_SID)
                    else:
                        mj.mj_jacBody(model, data, Jp, Jr, _PALM_BID)
                    J6 = np.vstack([Jp[:, :7], Jr[:, :7]])
                    v6 = np.array([_jog_v[0], _jog_v[1], _jog_v[2],
                                   _jog_w[0], _jog_w[1], _jog_w[2]])
                    # Singularity-robust DLS: the damping lambda^2 grows as the palm
                    # Jacobian's smallest singular value falls below JOG_SING_EPS,
                    # capping the joint-rate gain at ~1/(2*lambda_max) near singular
                    # grasp configs instead of letting it spike to ~1/sigma_min and
                    # lurch the arm off the object (the mechanism that threw the cube:
                    # a runaway qdot_jog written into qvel below drags the fixed-angle
                    # fingers across the object faster than friction can hold it).
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
                if GRASP_PROFILE:
                    _gp_acc['track'] += time.perf_counter() - _t_track0
                _t_tq0 = time.perf_counter() if GRASP_PROFILE else 0.0
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

                # GRASP-phase diagnostic trace (every step): everything needed to attribute a
                # slip offline — box pose, per-finger measured normal/tangential force + slip,
                # commanded |f_c|, jog velocity, squeeze ramp, palm/box z.
                if _grasp_trace is not None:
                    _f_c = grasp_ctrl.last_f_c if grasp_ctrl is not None else None
                    _, _, _tr_norm, _tr_tan = _hand_object_contact_metrics(active_idx)
                    _act_geom = _actual_contact_geometry(active_idx)
                    _rec_local_tr = obj_grasp.get('rec_local')
                    _slip = np.zeros(N_FINGERS)
                    _norm_ang = np.full(N_FINGERS, np.nan)   # rec-vs-actual normal angle (deg)
                    _pos_off = np.full(N_FINGERS, np.nan)    # rec-vs-actual contact pos (mm)
                    for _k, _f in enumerate(FINGER_SET):
                        if _rec_local_tr is not None:
                            _pO, _RO = _rec_local_tr[_k]
                            _bid_tr = obj_grasp['id_body']
                            _cW = data.xpos[_bid_tr] + data.xmat[_bid_tr].reshape(3, 3) @ _pO
                            _inW = data.xmat[_bid_tr].reshape(3, 3) @ _RO[:, 0]
                        else:
                            _sid_tr = obj_grasp['id_S'][_k]
                            _cW = data.site_xpos[_sid_tr].copy()
                            _inW = data.site_xmat[_sid_tr].reshape(3, 3)[:, 0]
                        _anchor = _cW - _PAD_OFFSET[_f] * _inW
                        _slip[_k] = float(np.linalg.norm(data.site_xpos[id_C[_k]] - _anchor))
                        # RECOMMENDED (grasp-map) normal/pos vs MuJoCo ACTUAL contact.
                        _ag = _act_geom.get(_f)
                        if _ag is not None:
                            _p_act, _n_act = _ag
                            _norm_ang[_k] = float(np.degrees(np.arccos(
                                np.clip(_inW @ _n_act, -1, 1))))
                            _pos_off[_k] = float(np.linalg.norm(_cW - _p_act) * 1e3)
                    _grasp_trace.sample(
                        t=float(data.time), t_wall=float(time.time()),
                        squeeze_on=int(bool(squeeze_on)),
                        ramp=float(min(1.0, _squeeze_steps * model.opt.timestep / SQUEEZE_RAMP_S)),
                        box_xpos=data.xpos[obj_grasp['id_body']].copy(),
                        box_xquat=data.xquat[obj_grasp['id_body']].copy(),
                        palm_z=float(data.xpos[_PALM_BID][2]),
                        norm_force=np.array([_tr_norm[f] for f in FINGER_SET]),
                        tan_force=np.array([_tr_tan[f] for f in FINGER_SET]),
                        fc_cmd=(np.array([float(np.linalg.norm(_f_c[3*k:3*k+3]))
                                          for k in range(N_FINGERS)])
                                if _f_c is not None else np.full(N_FINGERS, np.nan)),
                        slip=_slip,
                        norm_ang=_norm_ang,   # rec-vs-MuJoCo contact normal angle, deg
                        pos_off=_pos_off,     # rec-vs-MuJoCo contact position offset, mm
                        jog_v=_jog_v.copy(),
                        q_arm=data.qpos[:7].copy(),
                        gamma_live=float(gamma_live))

            if GRASP_PROFILE and control_phase == 'GRASP':
                _gp_acc['torque'] += time.perf_counter() - _t_tq0
            data.qfrc_applied[:] = tau_ctrl
            data.qfrc_applied[:N_ROBOT] += data.qfrc_bias[:N_ROBOT]
            _t_step0 = time.perf_counter() if GRASP_PROFILE else 0.0
            mj.mj_step(model, data)
            if GRASP_PROFILE and control_phase == 'GRASP':
                _gp_acc['step'] += time.perf_counter() - _t_step0
                _gp_acc['n'] += 1
                if time.time() - _gp_last >= 1.0:
                    _n = max(_gp_acc['n'], 1)
                    print(f"\r\n[grasp-profile] {_gp_acc['n']} steps/s | per-step ms: "
                          f"track={_gp_acc['track']/_n*1e3:.3f} "
                          f"(refresh={_gp_acc['refresh']/_n*1e3:.3f}: "
                          f"spin={_gp_acc['spin']/_n*1e3:.3f} "
                          f"step_ik={_gp_acc['step_ik']/_n*1e3:.3f}) "
                          f"torque={_gp_acc['torque']/_n*1e3:.3f} "
                          f"step={_gp_acc['step']/_n*1e3:.3f} "
                          f"hand_track={_dexpilot_ctrl._hand_tracking if _dexpilot_ctrl else '?'}")
                    _gp_acc = {'track': 0.0, 'refresh': 0.0, 'spin': 0.0, 'step_ik': 0.0,
                               'torque': 0.0, 'step': 0.0, 'n': 0}
                    _gp_last = time.time()

            # --- Trial benchmarking (--trial-log): detectors + trace, every real
            # physics step, independent of which control_phase branch ran above. See
            # trial_logger.py for the state machine this drives. No-op when the trial
            # runner wasn't constructed (--trial-log omitted).
            if (_trial_runner is not None and _trial_state is not None
                    and _trial_state.outcome is None):   # skip once ended, until the
                                                           # next trial_start replaces it
                _tid  = _trial_state.trial_id
                _tnow = data.time
                if control_phase in ('PLAN', 'REACH'):
                    _trial_runner.set_phase(_trial_state, _tnow, control_phase)
                    _trial_runner.step_approach(
                        _trial_state, _tnow, data.contact[:data.ncon],
                        _HAND_GIDS, obj['id_geom'])
                    if _trial_runner.check_timeout(_trial_state, _tnow):
                        _trial_runner.end_trial(_trial_state, _tnow)
                elif control_phase == 'GRASP':
                    _hh = _trial_rest_hh[active_idx]
                    _height_above_rest = float(data.geom_xpos[obj['id_geom']][2]) - _hh
                    if _dp_trigger is not None:
                        _d_s1 = (_dexpilot_ctrl.retargeter.last_d_s1
                                 if _dexpilot_ctrl is not None else [float('inf')] * 3)
                        _touching = obj['id_geom'] in {
                            (c.geom2 if c.geom1 in _HAND_GIDS else c.geom1)
                            for c in data.contact[:data.ncon]
                            if c.geom1 in _HAND_GIDS or c.geom2 in _HAND_GIDS}
                        _fired  = _dp_trigger.update(_d_s1, _touching)
                        _active = (min(_d_s1) < _dp_trigger.eps) and _touching
                    else:
                        _fired  = _cat_trigger.update(squeeze_on)
                        _active = squeeze_on
                    _sid = _trial_place_sid.get(active_idx, -1)
                    _xy_off = _spd = None
                    if _sid >= 0:
                        _xy_off = float(np.linalg.norm(
                            data.geom_xpos[obj['id_geom']][:2] - data.site_xpos[_sid][:2]))
                        _dofadr = _trial_dofadr[active_idx]
                        _spd = float(np.linalg.norm(data.qvel[_dofadr:_dofadr + 3]))
                    _arrived = _trial_runner.step_pick_or_transport(
                        _trial_state, _tnow, trigger_fired=_fired,
                        trigger_active=_active, height_above_rest=_height_above_rest,
                        place_xy_offset=_xy_off, object_speed=_spd)
                    _trial_runner.trace.sample(
                        t=_tnow,
                        p_thumb=data.site_xpos[id_C[FINGER_SET.index('thumb')]].copy(),
                        p_index=data.site_xpos[id_C[FINGER_SET.index('index')]].copy(),
                        obj_pos=data.geom_xpos[obj['id_geom']].copy(),
                        obj_quat=data.xquat[obj['id_body']].copy(),
                        obj_linvel=data.qvel[_trial_dofadr[active_idx]:
                                             _trial_dofadr[active_idx] + 3].copy(),
                        height_above_rest=_height_above_rest,
                        phase=1 if _trial_state.phase == TrialPhase.TRANSPORT else 0,
                        # Robot joint config (full arm+hand) — the warmstart q_seed a PD/
                        # gamma tuning harness replays: recommender -> RRT -> grasp -> lift.
                        # Full object qpos (pos+quat) pairs with it as the recommender input.
                        q_robot=data.qpos[:N_ROBOT].copy(),
                        obj_qpos=data.qpos[N_ROBOT:].copy())
                    if _arrived or _trial_runner.check_timeout(_trial_state, _tnow):
                        _trial_runner.end_trial(_trial_state, _tnow)

            # Teleop: keep the achieved-contact markers pinned to the object as it
            # moves (jog during GRASP) by re-expressing the stored object-local
            # achieved contacts in the world. Hidden until a lock-in plan succeeds.
            if _CAT_MODE and not _teleop_active:
                _ach_O = objects[active_idx].get('rec_achieved_O')
                if _ach_O is not None:
                    _bid_m = objects[active_idx]['id_body']
                    _pO_m  = data.xpos[_bid_m]
                    _RO_m  = data.xmat[_bid_m].reshape(3, 3)
                    _if = {f: i for i, f in enumerate(FINGER_SET)}
                    data.mocap_pos[_rec1_mocap] = _pO_m + _RO_m @ _ach_O[_if['thumb']]
                    data.mocap_pos[_rec2_mocap] = _pO_m + _RO_m @ _ach_O[_if['index']]

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
    # Persist the always-on pose recorder (warmstart source for PD/gamma tuning).
    if _pose_trace is not None and len(_pose_trace) > 0:
        _pose_path = Path('logs') / args.trial_log / 'pose_trace.npz'
        _pose_n = len(_pose_trace)
        _pose_trace.save(_pose_path)
        print(f"[pose-trace] saved {_pose_path} ({_pose_n} rows)")
    if _grasp_trace is not None and len(_grasp_trace) > 0:
        _gt_path = Path('logs') / args.grasp_trace / 'grasp_trace.npz'
        _gt_path.parent.mkdir(parents=True, exist_ok=True)
        _gt_n = len(_grasp_trace)
        _grasp_trace.save(_gt_path)
        print(f"[grasp-trace] saved {_gt_path} ({_gt_n} rows)")
    if dash is not None:
        dash.close()
    # (retargeting sliders live in the MediaPipe subprocess window; it cleans up
    # its own cv2 windows on exit — no sim-side teardown needed.)
    if _dexpilot_ctrl is not None:
        _dexpilot_ctrl.shutdown()
    if _skel_viewer is not None:
        _skel_viewer.close()
    if _cam_grid is not None:
        _cam_grid.close()
    if _cam_grid_node is not None:
        try:
            _cam_grid_node.destroy_node()
        except Exception:
            pass
    if _mediapipe_proc is not None:
        # run_multicam.py (the --multicam child) supervises its own landmark +
        # fusion subprocesses and tears them down on SIGINT; SIGTERM would orphan
        # them. The single-cam publisher handles SIGTERM fine. Send SIGINT for the
        # pipeline, then wait (with a grace fallback to terminate).
        #
        # Because the child was launched with start_new_session=True it's the
        # leader of its own process group (pgid == pid), so we signal the WHOLE
        # GROUP via os.killpg — this reaches the grandchildren (landmark + fusion
        # nodes) even if the supervisor itself has already died, which is what
        # left orphans holding /dev/video* before. Fall back to signalling just
        # the child if the group lookup fails (e.g. it already exited).
        def _signal_child_group(sig):
            try:
                os.killpg(os.getpgid(_mediapipe_proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                # Group already gone, or not our leader — signal the child directly.
                try:
                    _mediapipe_proc.send_signal(sig)
                except ProcessLookupError:
                    pass
        if args.multicam:
            _signal_child_group(signal.SIGINT)
            try:
                _mediapipe_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Supervisor didn't tear down in time — SIGKILL the whole group so
                # no landmark/fusion grandchild is left running or holding a camera.
                _signal_child_group(signal.SIGKILL)
                _mediapipe_proc.wait()
        else:
            _mediapipe_proc.terminate()
            _mediapipe_proc.wait()

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

import mujoco as mj
import mujoco.viewer  # noqa: F401
from pynput import keyboard as _pynput_kb

from scripts.rrt_planner import RRTPlanner
from grasp_control import SpatialIKSolver, ConstrainedIKSolver, GraspController
from grasp_control.constrained_ik import configure_sqp
from live_dashboard import Dashboard


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
        54:  'ik_vis',  # 6 — cycle IK config visualization
        55:  'bspheres', # 7 — toggle IK collision bounding-sphere overlay
        56:  'teleop_start', # 8 — (dexpilot) start/re-zero tracking at current pose
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
        '--mode', choices=['rrt', 'dexpilot'], default='rrt',
        help="rrt (default): autonomous RRT+IK grasp controller  |  "
             "dexpilot: live MediaPipe kinematic retargeting teleop via ROS 2.")
    _arg_parser.add_argument(
        '--camera', type=int, default=None,
        help="Camera index forwarded to ui/mediapipe_joint_angles.py in dexpilot mode "
             "(default: auto-select — prefers external/USB camera at index ≥1).")
    _arg_parser.add_argument(
        '--ik-solver', choices=['sqp', 'ipopt'], default='sqp',
        help="sqp (default): sqpmethod + OSQP + softplus SDF + analytic FK Jacobians — "
             "~3× cheaper per iteration, wins on wall time in most cases  |  "
             "ipopt: IPOPT L-BFGS + finite-difference Jacobians — production baseline.")
    args = _arg_parser.parse_args()

    model = mj.MjModel.from_xml_path('models/scene_pick_place.xml')
    data  = mj.MjData(model)

    N_ROBOT = 23  # joint_1..7 (Gen3 arm) + 16 LEAP finger joints; object joints follow
    # Gen3's odd joints (1,3,5,7 -> indices 0,2,4,6) are continuous revolute with no
    # jnt_range — RRTPlanner samples uniformly from model.jnt_range, so [0,0] would never
    # randomize those joints. Give them a generous sampling bound before the planner reads it.
    for j in (0, 2, 4, 6):
        model.jnt_range[j] = [-np.pi, np.pi]

    _randomize_objects(model, data, np.random.default_rng())

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
    JOG_VEL         = 0.05  # jog speed while arrow key held (m/s)

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
        orient_weight=0.01,         # align each fingerpad with the contact inward normal
        max_iter=500,   # DLS warm-start puts us near solution; few iters needed
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
          normals: each FINGER_SET finger's summed contact normal force vs ANY object.
        Convention (verified empirically, MuJoCo 3.3.5): contact.frame rows are the
        contact-frame axes with row 0 = normal pointing geom1→geom2, and mj_contactForce
        returns the force applied TO geom2 expressed in that frame — so the force on the
        object is +R.T@f when the object is geom2 and -R.T@f when it is geom1."""
        f_net, tau_net = np.zeros(3), np.zeros(3)
        normals = {f: 0.0 for f in FINGER_SET}
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
                normals[fname] += ft[0]   # normal component (>= 0, along frame row 0)
            if _OBJ_GID_TO_IDX[obj_gid] == obj_idx:
                R_con = con.frame.reshape(3, 3)
                f_W = sgn * (R_con.T @ ft[:3])
                f_net   += f_W
                tau_net += np.cross(con.pos - com, f_W) + sgn * (R_con.T @ ft[3:6])
        return f_net, tau_net, normals

    # Live metrics dashboard (separate process; opt-in via --dashboard). Started before the
    # IK precompute so the grasp IPOPT solves below are reported too. dash is None
    # when disabled; every push site is guarded on it.
    dash = None
    if args.dashboard:
        dash = Dashboard(FINGER_SET, horizon_s=10.0, dt_hint=3 * model.opt.timestep)
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

    def _push_squeeze(on):
        """Forward the GRASP internal-force state to the dashboard header. The commanded
        value is static for now (gamma set at startup, ~gamma/sqrt(2) N per contact for
        the 2-contact pinch) — becomes live once w_des tracking / LP-gamma lands.
        No-op when the dashboard is disabled."""
        if dash is None:
            return
        dash.push({'type': 'squeeze', 'on': bool(on), 'gamma': GAMMA,
                   'f_contact': GAMMA / np.sqrt(2)})

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

    # Internal squeeze force scale (GRASP, toggled with Enter): f_c = null(G) @ GAMMA,
    # ~GAMMA/sqrt(2) N normal force per contact for the 2-contact antipodal pinch.
    # Sized for lifting: every object is 0.5 kg with mu=1 (scene XML), so supporting
    # mg ~ 4.9 N by friction across 2 contacts needs >= mg/(2*mu) ~ 2.45 N of normal
    # force each. GAMMA=8 -> ~5.7 N per contact, ~2.3x margin for the finger PD
    # fighting the squeeze, contact compliance, and dynamic loads while jogging.
    GAMMA = 50.0

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
        _t0 = time.time()
        q_dls_grasp = dls_ik.solve(model, _ik_data, id_C, obj['p_S_W'],
                                    q_bias=Q_BIAS, null_gain=0.3)
        dls_grasp_ms = (time.time() - _t0) * 1e3
        _t0 = time.time()
        obj['q_target'] = constrained_ik.solve(_ik_data, id_C, obj['p_S_W'],
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
        for label, q_sol, tgts in [('grasp', obj['q_target'], obj['p_S_W'])]:
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
    print("[Control] Ctrl+0..6: select target  |  ←→: jog x  |  ↑↓: jog z  |  Enter: GRASP / toggle squeeze  |  N: release  |  6: IK vis  |  7: coll spheres  |  Q/Esc: quit")
    print("[Control] Active target: init pose")

    # Simulation — start at Q_BIAS so PD error at t=0 is zero and qfrc_bias is correct.
    # All-zero initial qpos would produce a huge PD error (arm pointing straight up vs
    # HOME_ARM target) causing explosive qacc on the first step.
    mj.mj_resetData(model, data)
    data.qpos[:N_ROBOT] = Q_BIAS
    mj.mj_forward(model, data)

    # --- DexPilot controller (--mode dexpilot only) ---
    _dexpilot_ctrl  = None
    _mediapipe_proc = None
    if args.mode == 'dexpilot':
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
        _dexpilot_ctrl = DexPilotController(model, q_bias=Q_BIAS,
            R_cam_robot=np.eye(3), debug=True, eps=0.005)
        _dexpilot_ctrl.init_home(data)
        _dexpilot_ctrl.init_ros()
        print("[DexPilot] ROS subscriber active — waiting for /hand/joint_angles (≥120 floats)")
        print("[DexPilot] Press 8 to start tracking (captures your current wrist "
              "orientation as the robot's home). Q/Esc: quit")
    control_phase  = 'REACH'
    active_idx     = 0
    active_tgt     = 0        # index into targets[]
    tau_ctrl       = np.zeros(model.nv)   # full nv for qfrc_applied
    traj_waypoints = [Q_BIAS.copy()]   # seed with home so REACH holds immediately
    traj_wp_idx    = 0
    traj_wp_step   = 0    # counts sim steps since last waypoint advance
    plan_thread    = None
    q_plan_hold    = np.zeros(N_ROBOT)    # robot DOFs only
    _held_arrows   = set()         # arrow keys currently held
    _ctrl_held     = set()         # Ctrl_L / Ctrl_R currently held
    squeeze_on     = False         # GRASP: internal force toggled by Enter
    grasp_ctrl     = None          # GraspController, built at each REACH→GRASP transition
    q_grasp_hold   = None          # GRASP PD target; arm part integrated by arrow-key jog
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
            elif key == _pynput_kb.Key.right: _held_arrows.add('right')
            elif key == _pynput_kb.Key.left:  _held_arrows.add('left')
            elif key == _pynput_kb.Key.up:    _held_arrows.add('up')
            elif key == _pynput_kb.Key.down:  _held_arrows.add('down')
        except AttributeError:
            pass

    def _on_release(key):
        try:
            if key in (_pynput_kb.Key.ctrl_l, _pynput_kb.Key.ctrl_r):
                _ctrl_held.discard(key)
            elif key == _pynput_kb.Key.right: _held_arrows.discard('right')
            elif key == _pynput_kb.Key.left:  _held_arrows.discard('left')
            elif key == _pynput_kb.Key.up:    _held_arrows.discard('up')
            elif key == _pynput_kb.Key.down:  _held_arrows.discard('down')
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
                if not running:
                    continue
                q_teleop = _dexpilot_ctrl.step(model, data)
                if q_teleop is not None:
                    data.qpos[:N_ROBOT] = q_teleop
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
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
                    _f_net, _tau_net, _normals = _hand_object_contact_metrics(_prox_idx)
                    dash.push({'type': 'wrench', 't': _t,
                               'f': _f_net.tolist(), 'tau': _tau_net.tolist()})
                    dash.push({'type': 'normals', 't': _t, 'n': _normals})
            _dash_i += 1

            # --- Check if background RRT finished ---
            if plan_thread is not None and not plan_thread.is_alive():
                plan_thread = None
                if 'waypoints' in _plan_result:
                    traj_waypoints = _plan_result['waypoints']
                    print(f"\r\n[Control] REACH  |  path: {len(traj_waypoints)} waypoints")
                else:
                    # Planning died (see _plan_thread_main traceback) — hold the pose we
                    # were already holding so the sim stays alive; reselect to retry.
                    traj_waypoints = [q_plan_hold.copy()]
                    print("\r\n[Control] REACH  |  planning FAILED — holding pose "
                          "(Ctrl+digit to retry)")
                traj_wp_idx    = 0
                traj_wp_step   = 0
                control_phase  = 'REACH'

            # --- Process key events ---
            while not keys.empty():
                key = keys.get_nowait()

                if key == 'enter' and control_phase == 'REACH':
                    control_phase = 'GRASP'
                    squeeze_on = False
                    # Object pose frozen at grasp time — reference anchor for the
                    # upcoming w_des (object wrench) tracking; unused by the jog.
                    obj_grasp = objects[active_idx]
                    obj_grasp['p_obj0'] = data.xpos[obj_grasp['id_body']].copy()
                    obj_grasp['R_obj0'] = data.xmat[obj_grasp['id_body']].reshape(3, 3).copy()
                    # Internal-force machinery for the Enter-toggled squeeze. Only
                    # internal_force_torques() is used — the joint PD hold stays in the
                    # GRASP branch below, on top of the shared bias comp.
                    grasp_ctrl = GraspController(
                        model, N_ROBOT,
                        tip_site_ids=id_C,
                        obj_site_ids=obj_grasp['id_S'],
                        obj_body_id=obj_grasp['id_body'],
                        kp=Kp, kd=Kd, gamma=GAMMA)
                    q_grasp_hold = obj_grasp['q_target'].copy()
                    _push_squeeze(False)
                    print(f"\r\n[Control] → GRASP  ({targets[active_tgt]['label']})  "
                          f"|  Enter: toggle squeeze (gamma={GAMMA:.0f})  |  N: release")

                elif key == 'enter' and control_phase == 'GRASP':
                    squeeze_on = not squeeze_on
                    _push_squeeze(squeeze_on)
                    print(f"\r\n[Control] squeeze {'ON' if squeeze_on else 'off'}  "
                          f"(gamma={GAMMA:.0f}, ~{GAMMA/np.sqrt(2):.2f} N/contact)")

                elif key == 'release' and control_phase == 'GRASP':
                    # Release: no pregrasp config exists anymore, so open the active
                    # fingers back to their Q_BIAS posture while the arm stays at the
                    # (possibly jogged) grasp arm config q_grasp_hold — snapping back
                    # to the original IK pose would drag the object with it.
                    squeeze_on     = False
                    _push_squeeze(False)
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

            # --- Continuous jog: world-frame palm velocity from currently-held arrow
            # keys, consumed by the GRASP branch's resolved-rate target integration.
            # left/right -> x, up/down -> z (lift) — y (depth) is never commanded.
            _jv_x = (JOG_VEL if 'right' in _held_arrows else 0) - (JOG_VEL if 'left' in _held_arrows else 0)
            _jv_z = (JOG_VEL if 'up'    in _held_arrows else 0) - (JOG_VEL if 'down' in _held_arrows else 0)

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
                # Resolved-rate jog: map the commanded world-frame palm velocity
                # [vx, 0, vz] (orientation held) to arm joint rates via 6-DOF DLS on
                # the live config, and integrate them into the PD target.
                qdot_jog = np.zeros(7)
                if _jv_x or _jv_z:
                    Jp = np.zeros((3, model.nv))
                    Jr = np.zeros((3, model.nv))
                    mj.mj_jacBody(model, data, Jp, Jr, _PALM_BID)
                    J6 = np.vstack([Jp[:, :7], Jr[:, :7]])
                    v6 = np.array([_jv_x, 0.0, _jv_z, 0.0, 0.0, 0.0])
                    qdot_jog = J6.T @ np.linalg.solve(J6 @ J6.T + 1e-6 * np.eye(6), v6)
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
                tau_ctrl[:N_ROBOT] = (Kp * (q_grasp_hold - data.qpos[:N_ROBOT])
                                      + Kd * (np.r_[qdot_jog, np.zeros(16)] - data.qvel[:N_ROBOT]))
                if squeeze_on:
                    tau_ctrl[:N_ROBOT] += grasp_ctrl.internal_force_torques(data)

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

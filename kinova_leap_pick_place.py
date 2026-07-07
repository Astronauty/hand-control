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
import time
import threading
import queue

import mujoco as mj
import mujoco.viewer  # noqa: F401
from pynput import keyboard as _pynput_kb

from scripts.rrt_planner import RRTPlanner
from grasp_control import SpatialGraspMapComputer, SpatialIKSolver, GraspForceAllocator, ConstrainedIKSolver
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
    _MAP = {
        257: 'enter',   # GLFW_KEY_ENTER (main keyboard)
        335: 'enter',   # GLFW_KEY_KP_ENTER (numpad)
        81:  'quit',    # Q
        256: 'quit',    # Escape
        75:  'ik_vis',  # K — cycle IK config visualization
        66:  'bspheres', # B — toggle IK collision bounding-sphere overlay
    }
    def _cb(keycode):
        event = _MAP.get(keycode)
        if event:
            key_queue.put(event)
    return _cb


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
        help="Launch a live pyqtgraph metrics dashboard (separate process): scrolling "
             "fingertip→object distances, RRT solve metrics, and IPOPT solve metrics.")
    args = _arg_parser.parse_args()

    model = mj.MjModel.from_xml_path('models/scene_pick_place.xml')
    data  = mj.MjData(model)

    N_ROBOT = 23  # joint_1..7 (Gen3 arm) + 16 LEAP finger joints; object joints follow
    # Gen3's odd joints (1,3,5,7 -> indices 0,2,4,6) are continuous revolute with no
    # jnt_range — RRTPlanner samples uniformly from model.jnt_range, so [0,0] would never
    # randomize those joints. Give them a generous sampling bound before the planner reads it.
    for j in (0, 2, 4, 6):
        model.jnt_range[j] = [-np.pi, np.pi]

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

    PREGRASP_OFFSET = 0.02  # metres to offset fingertips along contact-normal for RRT goal
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
        ({'index': 'obj_box_c2', 'thumb': 'obj_box_c1'}, 'obj_box'),
        ({'index': 'obj_cylinder_c2', 'thumb': 'obj_cylinder_c1'}, 'obj_cylinder'),
    ]
    objects = []
    for contact_sites, body_name in object_defs:
        missing = [f for f in FINGER_SET if f not in contact_sites]
        assert not missing, (
            f"{body_name} has no contact site mapped for finger(s) {missing}")
        obj = {
            'id_S':    [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, contact_sites[f])
                        for f in FINGER_SET],
            'id_body': mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name),
            'id_geom': mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, body_name + '_geom'),
        }
        obj['p_S_W'] = [data.site_xpos[sid].copy() for sid in obj['id_S']]
        # Inward surface normal at each contact site (the site's world-frame local
        # x-axis, per the scene XML convention) — the direction each fingerpad normal is
        # driven to align with in the constrained-IK orientation cost. Read from the same
        # FK state as p_S_W so the two stay consistent.
        obj['inward_S_W'] = [data.site_xmat[sid].reshape(3, 3)[:, 0].copy()
                             for sid in obj['id_S']]
        objects.append(obj)

    grasp_map_computer = SpatialGraspMapComputer()
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
    _OBJ_GEOM_NAMES = ['obj_box_geom', 'obj_cylinder_geom', 'floor']
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
    print(f"[IK] {len(_robot_geom_names)} robot geoms × {len(_OBJ_GEOM_NAMES)} objects "
          f"= {len(_robot_geom_names) * len(_OBJ_GEOM_NAMES)} collision constraints"
          f"  ({len(_robot_geom_names) * N_ROBOT} FD evals/iter — one position callback per geom)")

    # Fingertip tip-geom ids for the live dashboard's fingertip→object distance plot
    # (mj_geomDistance from each finger's tip mesh geom to the active object's geom).
    _TIP_GEOM_IDS = {f: mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f'leap_{FINGER_CODE[f]}_tip')
                     for f in FINGER_SET}

    # Live metrics dashboard (separate process; opt-in via --dashboard). Started before the
    # IK precompute so the grasp/pregrasp IPOPT solves below are reported too. dash is None
    # when disabled; every push site is guarded on it.
    dash = None
    if args.dashboard:
        dash = Dashboard(FINGER_SET, horizon_s=10.0, dt_hint=3 * model.opt.timestep)
        dash.start()
        print("[dashboard] launched (separate process)")

    def _push_ipopt(phase):
        """Forward ConstrainedIKSolver.last_metrics from the most recent solve() to the
        dashboard's IPOPT panel. No-op when the dashboard is disabled."""
        if dash is None:
            return
        m = constrained_ik.last_metrics
        dash.push({
            'type':         'ipopt',
            'phase':        phase,
            'status':       m.get('status', '?'),
            'iters':        m.get('iters', '?'),
            'max_site_mm':  max(m.get('site_err_mm', [0.0])),
            'max_pad_deg':  (max(m['pad_deg']) if m.get('pad_deg') else None),
            'min_slack_mm': m.get('min_slack_mm'),
        })

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
    #   proximal tier (everything else):     +2mm     — should never be near the surface
    # All tiers KEEP full clearance vs the floor plane (the solver never reduces the plane
    # constraint), so no active finger drops underground and pregrasp goals stay valid.
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

    for i, obj in enumerate(objects):
        mj.mj_resetData(model, data)
        data.qpos[:N_ROBOT] = Q_BIAS   # DLS warm-start pose (qpos0 is no longer Q_BIAS)
        mj.mj_forward(model, data)
        p_obj   = data.xpos[obj['id_body']].copy()
        normals = [(p - p_obj) / np.linalg.norm(p - p_obj) for p in obj['p_S_W']]

        # Stage 1: DLS IK (position-only, null_gain=0.3 for grasp approach direction)
        print(f"[IK] Object {i+1}: DLS grasp …")
        q_dls_grasp = dls_ik.solve(model, data, id_C, obj['p_S_W'],
                                    q_bias=Q_BIAS, null_gain=0.3)
        # Stage 2: IPOPT collision refinement from DLS warm start
        print(f"[IK] Object {i+1}: IPOPT grasp (collision refinement) …")
        obj['q_target'] = constrained_ik.solve(data, id_C, obj['p_S_W'],
                                                q_bias=Q_BIAS, q_init=q_dls_grasp,
                                                reduced_clearance_geoms=_active_clearance_by_geom,
                                                inward_dirs=obj['inward_S_W'])
        _push_ipopt(f'obj{i+1} grasp')

        mj.mj_resetData(model, data)
        data.qpos[:N_ROBOT] = Q_BIAS   # DLS warm-start pose (qpos0 is no longer Q_BIAS)
        mj.mj_forward(model, data)
        pregrasp_targets = [p + PREGRASP_OFFSET * n for p, n in zip(obj['p_S_W'], normals)]

        # Stage 1: DLS pregrasp (null_gain=0.0 — HOME_ARM bias fights low-z targets)
        print(f"[IK] Object {i+1}: DLS pregrasp …")
        q_dls_pre = dls_ik.solve(model, data, id_C, pregrasp_targets,
                                  q_bias=Q_BIAS, null_gain=0.0)
        q_dls_pre[11:19] = Q_BIAS[11:19]   # warm-start the mf/rf fingers curled (not a final override)
        # Stage 2: IPOPT collision refinement (all geoms constrained). The mf/rf curl is
        # only the warm start above — we let IPOPT keep those joints collision-consistent
        # rather than overwriting them back to Q_BIAS afterward, which previously reinstated
        # a curled-fingertip-into-neighbouring-object penetration that the collision-aware
        # solve had avoided (and made the RRT pregrasp goal infeasible).
        print(f"[IK] Object {i+1}: IPOPT pregrasp (collision refinement) …")
        q_pg = constrained_ik.solve(data, id_C, pregrasp_targets,
                                     q_bias=Q_BIAS, q_init=q_dls_pre,
                                     reduced_clearance_geoms=_active_clearance_by_geom,
                                     inward_dirs=obj['inward_S_W'])
        _push_ipopt(f'obj{i+1} pregrasp')
        obj['q_pregrasp'] = q_pg

        _d_chk = mj.MjData(model)
        for label, q_sol, tgts in [('grasp',    obj['q_target'], obj['p_S_W']),
                                    ('pregrasp', q_pg,            pregrasp_targets)]:
            _d_chk.qpos[:N_ROBOT] = q_sol
            mj.mj_forward(model, _d_chk)
            errs = [f"{np.linalg.norm(_d_chk.site_xpos[s] - t)*1e3:.1f} mm"
                    for s, t in zip(id_C, tgts)]
            print(f"[IK] Object {i+1} {label}: tip errors = {errs}")
            # Exact (mj_geomDistance, no bounding sphere) audit of every active-finger geom
            # vs the target object at the solution — the IK's sphere model can't see true
            # penetration, so this is the guardrail that makes it visible. Contact-tier
            # geoms (ds/tip) legitimately read ~0 at grasp; anything below -2mm is flagged.
            # Each query is clamped from below by the bounding-sphere bound: MuJoCo 3.3.x
            # GJK can spuriously return 0.0 for well-separated box-box pairs, and the true
            # distance can never be under ||c1-c2|| - rb1 - rb2.
            _ft = np.zeros(6)
            _pen = {}
            for g in _active_finger_geoms:
                _gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g)
                _lb  = (np.linalg.norm(_d_chk.geom_xpos[obj['id_geom']] - _d_chk.geom_xpos[_gid])
                        - model.geom_rbound[_gid] - model.geom_rbound[obj['id_geom']])
                _pen[g] = max(mj.mj_geomDistance(model, _d_chk, _gid, obj['id_geom'], 1.0, _ft), _lb)
            _worst_g, _worst_d = min(_pen.items(), key=lambda kv: kv[1])
            _flag = '  ** PENETRATION **' if _worst_d < -0.002 else ''
            print(f"[IK] Object {i+1} {label}: active-finger min exact dist = "
                  f"{_worst_d*1e3:.1f} mm ({_worst_g}){_flag}")

    # Per-joint PD gains for REACH phase: 7 arm joints + 16 LEAP finger joints.
    # Arm gains sized for Gen3's forcerange (±105/±52 Nm); finger gains mirror the small
    # values used for the planar model's tiny finger actuators.
    Kp = np.concatenate([np.full(7, 40.0), np.full(16, 0.8)])
    Kd = np.concatenate([np.full(7, 4.0),  np.full(16, 0.05)])
    Kp_obj     = 50.0  # object position stiffness, N/m (GRASP)
    Kd_obj     = 5.0   # object position damping, N·s/m (GRASP)
    Kp_theta   = 5.0   # object orientation stiffness, N·m/rad (GRASP)
    Kd_theta   = 0.5   # object orientation damping, N·m·s/rad (GRASP)
    Kp_contact = 100.0  # weak per-finger slip-correction stiffness, N/m (GRASP)
    Kd_contact = 10.0   # weak per-finger slip-correction damping, N·s/m (GRASP)
    gamma   = 5.0     # internal squeeze force scale; negate if fingers pull apart
    force_allocator = GraspForceAllocator(gamma)

    # Give the RRT the SAME full hand-geom set the IK constrains (_robot_geom_names) plus
    # the floor, instead of only the fingertips — checking just the tips let the palm /
    # proximal links / wrist sweep straight through objects unnoticed. The historical reason
    # for tips-only (the active fingers legitimately pass near the target at the goal) is now
    # handled by the per-plan target-aware pair-clearance overrides below.
    OBJ_BODIES   = ['obj_box', 'obj_cylinder']
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
    # separate, still sphere-based feature for the static pregrasp/grasp IK configs.
    _ghost_data         = mj.MjData(model)
    _ghost_markers_lock = threading.Lock()
    _ghost_markers      = []

    _GHOST_SITES    = id_C  # same sites driven by IK, so the ghost path matches the IK target
    _N_GHOST        = 15   # max waypoints to sample for ghost display

    # Static IK config markers: computed once at startup for each object.
    # cyan  (0, 0.8, 0.8) = pregrasp   gold (1, 0.8, 0) = grasp
    # Indexed by object index so the render loop shows only the active object's configs.
    def _make_ik_markers(obj):
        markers = []
        _d = mj.MjData(model)
        _d.qpos[N_ROBOT:] = data.qpos[N_ROBOT:].copy()
        for q_cfg, rgba in [
            (obj['q_pregrasp'], np.array([0.0, 0.8, 0.8, 0.55], dtype=np.float32)),
            (obj['q_target'],   np.array([1.0, 0.8, 0.0, 0.55], dtype=np.float32)),
        ]:
            _d.qpos[:N_ROBOT] = q_cfg
            mj.mj_forward(model, _d)
            positions = [_d.site_xpos[s].copy() for s in _GHOST_SITES]
            markers.append((positions, rgba))
        return markers

    _ik_markers_by_obj = [_make_ik_markers(obj) for obj in objects]

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

    def _run_rrt(q_start, q_pregrasp, obj_target):
        # Snapshot current object positions so collision checks reflect the live scene.
        planner._data.qpos[N_ROBOT:] = data.qpos[N_ROBOT:].copy()
        # Active fingers may TOUCH (0mm clearance) the object being grasped — unlike the old
        # boolean ignore_pairs, the exact distance is still checked, so they can approach the
        # target but never sweep through it. Vs the OTHER object they keep the full default
        # clearance; if the start config still hugs the object just released (or the IK left
        # a fingertip marginally inside its allowance at the goal), plan()'s endpoint grace
        # relaxes exactly those pairs to their endpoint distance — free to move away, never
        # deeper. Everything else — palm, proximal links, wrist, non-active fingers, and
        # every geom vs the floor — stays checked at the full clearance.
        pair_clearance = {(g, obj_target['id_geom']): 0.0 for g in _ACTIVE_SKIP_GIDS}
        # Re-branch the goal's continuous (base/wrist) joints onto the turn nearest the
        # current pose, so the arm never unwinds a near-full revolution just because the IK
        # left a joint on a far 2pi branch. Same configuration, planner-friendly numbering.
        q_goal = planner.rebranch(q_start, q_pregrasp)
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
            dash.push({'type': 'rrt', 'n_wp': len(path),
                       'plan_time': plan_time, 'fallback': fallback})
        _plan_result['waypoints'] = path
        _update_ghost_markers(path)

    # Build target list: index 0 = home pose (Q_BIAS), 1..N = object IK solutions
    targets = [{'q_target': Q_BIAS.copy(), 'label': 'init pose'}]
    for i, obj in enumerate(objects):
        targets.append({'q_target': obj['q_target'], 'label': f'object {i+1}'})

    keys = queue.Queue()
    print("[Control] Ctrl+0/1/2: select target  |  ←→: jog x  |  ↑↓: jog z  |  Enter: GRASP  |  K: IK vis  |  B: coll spheres  |  Q/Esc: quit")
    print("[Control] Active target: init pose")

    # Simulation — start at Q_BIAS so PD error at t=0 is zero and qfrc_bias is correct.
    # All-zero initial qpos would produce a huge PD error (arm pointing straight up vs
    # HOME_ARM target) causing explosive qacc on the first step.
    mj.mj_resetData(model, data)
    data.qpos[:N_ROBOT] = Q_BIAS
    mj.mj_forward(model, data)
    control_phase  = 'REACH'
    active_idx     = 0
    active_tgt     = 0        # index into targets[]
    tau_ctrl       = np.zeros(model.nv)   # full nv for qfrc_applied
    traj_waypoints = [Q_BIAS.copy()]   # seed with home so REACH holds immediately
    traj_wp_idx    = 0
    traj_wp_step   = 0    # counts sim steps since last waypoint advance
    plan_thread    = None
    q_plan_hold    = np.zeros(N_ROBOT)    # robot DOFs only
    jog_xz         = np.zeros(2)   # [x, z] manual jog offset, added to the frozen grasp pose
    _held_arrows   = set()         # arrow keys currently held
    _ctrl_held     = set()         # Ctrl_L / Ctrl_R currently held
    _ik_vis_mode   = None          # None | 'pregrasp' | 'grasp': freeze physics to show IK config
    _show_bspheres = False         # B: overlay the IK's per-geom collision bounding spheres

    # Precomputed (geom_id, bounding-sphere radius) for every hand geom the IK constrains —
    # this is the coarse sphere model the IK actually "sees" (finger links 15-24mm), which
    # is why the fingers can't be both constrained and reach a small object. Toggle with B.
    _BSPHERES = [(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g), float(model.geom_rbound[
                    mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g)]))
                 for g in _robot_geom_names]

    # Dashboard streaming state: wall-clock t0 for the scrolling x-axis, a step counter to
    # throttle distance pushes, and the last active-label pushed (so we only push on change).
    _dash_t0        = time.time()
    _dash_i         = 0
    DASH_PUSH_EVERY = 3            # push fingertip distances every N loop iterations
    _dash_last_lbl  = None

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
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.label = mj.mjtLabel.mjLABEL_CONTACTFORCE

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
                if active_tgt > 0:
                    for positions, rgba in _ik_markers_by_obj[active_idx]:
                        for pos in positions:
                            if scn.ngeom >= scn.maxgeom: break
                            mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_SPHERE,
                                            _sz, pos, _eye9, rgba)
                            scn.ngeom += 1
                _draw_bspheres(scn)
            viewer.sync()
            time.sleep(model.opt.timestep)

        running = True
        while viewer.is_running() and running:
            step_start = time.time()

            # --- Dashboard: stream fingertip→active-object distances + active label ---
            if dash is not None:
                _lbl = f"{targets[active_tgt]['label']} — {control_phase}"
                if _lbl != _dash_last_lbl:
                    dash.push({'type': 'active', 'label': _lbl})
                    _dash_last_lbl = _lbl
                if _dash_i % DASH_PUSH_EVERY == 0:
                    _ogid = objects[active_idx]['id_geom']
                    # distmax=2.0 (scene ~1 m): distances above this clamp to 2.0, so the
                    # cap is set high enough to show most of the reach, not just contact.
                    _dvals = {f: mj.mj_geomDistance(model, data, _TIP_GEOM_IDS[f],
                                                    _ogid, 2.0, None)
                              for f in FINGER_SET}
                    dash.push({'type': 'dist', 't': time.time() - _dash_t0, 'd': _dvals})
                _dash_i += 1

            # --- Check if background RRT finished ---
            if plan_thread is not None and not plan_thread.is_alive():
                plan_thread    = None
                traj_waypoints = _plan_result.get('waypoints', [objects[active_idx]['q_target']])
                traj_wp_idx    = 0
                traj_wp_step   = 0
                jog_xz[:]      = 0.0
                control_phase  = 'REACH'
                print(f"\r\n[Control] REACH  |  path: {len(traj_waypoints)} waypoints")

            # --- Process key events ---
            while not keys.empty():
                key = keys.get_nowait()

                if key == 'enter' and control_phase == 'REACH':
                    control_phase = 'GRASP'
                    jog_xz[:] = 0.0
                    # Freeze the object's pose at grasp time — jogging targets an offset
                    # from this frozen pose, not the object's live pose.
                    obj_grasp = objects[active_idx]
                    obj_grasp['p_obj0'] = data.xpos[obj_grasp['id_body']].copy()
                    obj_grasp['R_obj0'] = data.xmat[obj_grasp['id_body']].reshape(3, 3).copy()
                    print(f"\r\n[Control] → GRASP  ({targets[active_tgt]['label']})")

                elif key == 'enter' and control_phase == 'GRASP':
                    q_pre = objects[active_idx]['q_pregrasp'].copy()
                    traj_waypoints = [q_pre]
                    traj_wp_idx    = 0
                    traj_wp_step   = 0
                    control_phase  = 'REACH'
                    print(f"\r\n[Control] → REACH  (released — returning to pregrasp)")

                elif key == 'ik_vis' and active_tgt > 0:
                    _ik_vis_mode = {None: 'pregrasp', 'pregrasp': 'grasp', 'grasp': None}[_ik_vis_mode]
                    label = f'showing {_ik_vis_mode} config' if _ik_vis_mode else 'off'
                    print(f"\r\n[IK vis] {label}  (K to cycle)")

                elif key == 'bspheres':
                    _show_bspheres = not _show_bspheres
                    print(f"\r\n[bspheres] IK collision bounding-sphere overlay "
                          f"{'ON' if _show_bspheres else 'off'}  (B to toggle)")

                elif key.startswith('sel_') and control_phase != 'GRASP':
                    new_tgt = int(key[4:])
                    if 0 <= new_tgt < len(targets) and new_tgt != active_tgt:
                        active_tgt = new_tgt
                        active_idx = max(0, active_tgt - 1)  # map back to objects[]
                        _ik_vis_mode = None   # exit vis mode when switching target
                        q_start = data.qpos[:N_ROBOT].copy()
                        with _ghost_markers_lock:   # clear stale ghosts while planning
                            _ghost_markers.clear()
                        if active_tgt == 0:
                            traj_waypoints = [targets[0]['q_target']]
                            traj_wp_idx    = 0
                            traj_wp_step   = 0
                            control_phase  = 'REACH'
                        else:
                            q_pre = objects[active_idx]['q_pregrasp']
                            _plan_result.clear()
                            q_plan_hold   = q_start.copy()
                            plan_thread   = threading.Thread(
                                target=_run_rrt, args=(q_start, q_pre, objects[active_idx]),
                                daemon=True)
                            plan_thread.start()
                            control_phase = 'PLAN'
                        print(f"\r\n[Control] → {targets[active_tgt]['label']}")

                elif key == 'quit':
                    running = False

            # --- Continuous jog: drive jog_xz from currently-held arrow keys ---
            # left/right -> x, up/down -> z (lift) — y (depth) stays at the frozen value.
            _jv_x = (JOG_VEL if 'right' in _held_arrows else 0) - (JOG_VEL if 'left' in _held_arrows else 0)
            _jv_z = (JOG_VEL if 'up'    in _held_arrows else 0) - (JOG_VEL if 'down' in _held_arrows else 0)
            if _jv_x or _jv_z:
                dt = model.opt.timestep
                jog_xz[0] += _jv_x * dt
                jog_xz[1] += _jv_z * dt

            # --- IK visualization: freeze physics, show full arm in stored IK config ---
            if _ik_vis_mode is not None and active_tgt > 0:
                q_vis = objects[active_idx][
                    'q_pregrasp' if _ik_vis_mode == 'pregrasp' else 'q_target']
                data.qpos[:N_ROBOT] = q_vis
                data.qvel[:N_ROBOT] = 0.0
                mj.mj_forward(model, data)
                if viewer.user_scn is not None:
                    viewer.user_scn.ngeom = 0   # suppress ghost markers; arm pose is the vis
                    _draw_bspheres(viewer.user_scn)
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
                    wp = (traj_waypoints[-1] if traj_waypoints
                          else targets[active_tgt]['q_target'])
                    data.qpos[:N_ROBOT] = wp
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
                    _render_kinematic_frame()
                    continue

                else:
                    # Last waypoint reached: switch to physics PD hold.
                    wp = (traj_waypoints[-1] if traj_waypoints
                          else targets[active_tgt]['q_target'])
                    data.qvel[:N_ROBOT] = 0.0   # damp out kinematic-replay residual velocity
                    tau_ctrl = np.zeros(model.nv)
                    tau_ctrl[:N_ROBOT] = Kp * (wp - data.qpos[:N_ROBOT]) + Kd * (0 - data.qvel[:N_ROBOT])

            # --- GRASP: Cartesian impedance + null-space squeeze, generalized to
            # N_FINGERS contacts (or, in --viz-only, kinematic hold of the grasp IK pose) ---
            elif control_phase == 'GRASP':
                if args.viz_only:
                    data.qpos[:N_ROBOT] = obj['q_target']
                    data.qvel[:N_ROBOT] = 0.0
                    mj.mj_forward(model, data)
                    _render_kinematic_frame()
                    continue
                p_WoO_cur = data.xpos[obj['id_body']]
                R_WO_cur  = data.xmat[obj['id_body']].reshape(3, 3)

                contacts  = []
                inward_dirs = []
                R_WS_cur, p_WoS_cur, J_list = [], [], []
                for k in range(N_FINGERS):
                    R_WSk = data.site_xmat[obj['id_S'][k]].reshape(3, 3)
                    p_WoSk = data.site_xpos[obj['id_S'][k]].copy()
                    R_WS_cur.append(R_WSk)
                    p_WoS_cur.append(p_WoSk)

                    p_OSk_O = R_WO_cur.T @ (p_WoSk - p_WoO_cur)
                    R_OSk   = R_WO_cur.T @ R_WSk
                    contacts.append({'p': p_OSk_O, 'R': R_OSk})

                    if k == 0:
                        inward_dirs.append(R_OSk.T @ (-p_OSk_O / np.linalg.norm(p_OSk_O)))
                    else:
                        inward_dirs.append(None)

                    Jk_full = np.zeros((3, model.nv))
                    mj.mj_jacSite(model, data, Jk_full, None, id_C[k])
                    J_list.append(Jk_full[:3, :N_ROBOT])

                G_cur = grasp_map_computer.compute(contacts)

                # Desired object pose: jog defines an [x, z] offset from the pose frozen
                # at grasp time (y/depth held fixed). Orientation held at its grasp-time value.
                p_obj_des = obj['p_obj0'] + np.array([jog_xz[0], 0.0, jog_xz[1]])
                e_p = p_obj_des - p_WoO_cur
                # SO(3) orientation error: sum of cross products of corresponding columns
                # of current vs. desired rotation matrices (standard Cartesian-impedance
                # error; reduces to the planar scalar e_theta when rotation is about z only).
                R_des = obj['R_obj0']
                e_omega = 0.5 * sum(np.cross(R_WO_cur[:, i], R_des[:, i]) for i in range(3))

                # Object velocity via body Jacobian (no need to track per-object qvel indices)
                Jp_o, Jr_o = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mj.mj_jacBodyCom(model, data, Jp_o, Jr_o, obj['id_body'])
                v_obj     = Jp_o @ data.qvel
                omega_obj = Jr_o @ data.qvel

                # Desired object wrench (world-frame force + moment), rotated into the
                # object frame to match SpatialGraspMapComputer's convention.
                f_des_W = Kp_obj * e_p + Kd_obj * (-v_obj)
                m_des_W = Kp_theta * e_omega + Kd_theta * (-omega_obj)
                w_des_O = np.concatenate([R_WO_cur.T @ f_des_W, R_WO_cur.T @ m_des_W])

                # Allocate desired wrench to contact forces (min-norm) + null-space squeeze
                f_c = force_allocator.allocate(G_cur, w_des_O, contact_dof=3,
                                                inward_dirs=inward_dirs)

                tau_ctrl = np.zeros(model.nv)
                for k in range(N_FINGERS):
                    # Low-gain hybrid contact correction: resists slip without overpowering
                    # the wrench-allocated motion above.
                    dpk = J_list[k] @ data.qvel[:N_ROBOT]
                    f_corr_k = (Kp_contact * (p_WoS_cur[k] - data.site_xpos[id_C[k]])
                                + Kd_contact * (-dpk))
                    f_ck_W = R_WS_cur[k] @ f_c[3*k:3*k+3] + f_corr_k
                    tau_ctrl[:N_ROBOT] += J_list[k].T @ f_ck_W

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
                if active_tgt > 0:
                    for positions, rgba in _ik_markers_by_obj[active_idx]:
                        for pos in positions:
                            if scn.ngeom >= scn.maxgeom:
                                break
                            mj.mjv_initGeom(scn.geoms[scn.ngeom],
                                            mj.mjtGeom.mjGEOM_SPHERE,
                                            _sz, pos, _eye9, rgba)
                            scn.ngeom += 1
                _draw_bspheres(scn)

            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    _kb_listener.stop()
    if dash is not None:
        dash.close()

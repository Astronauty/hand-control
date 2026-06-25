#!/usr/bin/env python3
# All variable naming notation follows https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html
"""3D analog of internal_force_control.py: RRT + internal-force grasp control for the
Kinova Gen3 + LEAP hand pick-and-place scene (models/scene_pick_place.xml).

Same keyboard-driven REACH/GRASP state machine as internal_force_control.py, generalized
to 3D contacts/wrenches via the spatial classes in grasp_control/. The number of fingers
used in the grasp is configurable via FINGER_SET (see below) — the controller loops over
contacts generically rather than hardcoding 2.
"""
import numpy as np
import time
import threading
import queue

import mujoco as mj
import mujoco.viewer  # noqa: F401
from pynput import keyboard as _pynput_kb

from scripts.rrt_planner import RRTPlanner
from grasp_control import SpatialGraspMapComputer, SpatialIKSolver, GraspForceAllocator


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


# Gen3 arm "home" pose — a natural elbow-bent reach-forward configuration taken
# from gen3.xml's keyframe. Used as the IK null-space bias for the 7 arm joints:
# the null-space pull toward this pose produces a forward/lateral approach to tabletop
# objects (confirmed visually) without needing explicit orientation constraints on
# the fingertips — the orientation approach (IKSolver's (local_axis, world_target) tuple)
# was implemented and validated but caused joint-limit clipping instability on this
# 23-DOF redundant chain when combined with position constraints, preventing convergence.
HOME_ARM = np.array([0.0, 0.26179939, 3.14159265, -2.26892803,
                      0.0, 0.95993109, 1.57079633])

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
    Avoids tty raw-mode manipulation so the script works in any terminal."""
    # GLFW key codes
    _MAP = {
        257: 'enter',   # GLFW_KEY_ENTER (main keyboard)
        335: 'enter',   # GLFW_KEY_KP_ENTER (numpad)
        320: 'num0', 321: 'num1', 322: 'num2', 323: 'num3',  # KP_0–KP_3
        81:  'quit',    # Q
        256: 'quit',    # Escape
    }
    def _cb(keycode):
        event = _MAP.get(keycode)
        if event:
            key_queue.put(event)
    return _cb


if __name__ == "__main__":
    model = mj.MjModel.from_xml_path('models/scene_pick_place.xml')
    data  = mj.MjData(model)

    N_ROBOT = 23  # joint_1..7 (Gen3 arm) + 16 LEAP finger joints; object joints follow
    # Gen3's odd joints (1,3,5,7 -> indices 0,2,4,6) are continuous revolute with no
    # jnt_range — RRTPlanner samples uniformly from model.jnt_range, so [0,0] would never
    # randomize those joints. Give them a generous sampling bound before the planner reads it.
    for j in (0, 2, 4, 6):
        model.jnt_range[j] = [-np.pi, np.pi]

    mj.mj_forward(model, data)

    # Fingertip sites (on the active FINGER_SET fingers)
    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f]) for f in FINGER_SET]
    N_FINGERS = len(FINGER_SET)

    PREGRASP_OFFSET = 0.06  # metres to offset fingertips along contact-normal for RRT goal
    STEPS_PER_WP    = 50    # max sim steps before forcing waypoint advance (timeout, 1 step = 1ms)
    WP_REACH_TOL    = 0.02  # joint-space radius to consider a waypoint reached (rad)
    JOG_VEL         = 0.05  # jog speed while arrow key held (m/s)

    # Object definitions: rigid objects only (obj_soft deferred — vertex-level contact,
    # not a rigid grasp-map problem). Each object's contact sites must number N_FINGERS.
    object_defs = [
        (['obj_box_c1', 'obj_box_c2'], 'obj_box'),
        (['obj_cylinder_c1', 'obj_cylinder_c2'], 'obj_cylinder'),
    ]
    objects = []
    for s_names, body_name in object_defs:
        assert len(s_names) == N_FINGERS, (
            f"{body_name} defines {len(s_names)} contact sites but FINGER_SET has "
            f"{N_FINGERS} fingers")
        obj = {
            'id_S':    [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, n) for n in s_names],
            'id_body': mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name),
        }
        obj['p_S_W'] = [data.site_xpos[sid].copy() for sid in obj['id_S']]
        objects.append(obj)

    ik_solver = SpatialIKSolver(n_robot=N_ROBOT)
    grasp_map_computer = SpatialGraspMapComputer()

    # Null-space bias: HOME_ARM pulls the 7 arm joints toward a natural elbow-bent
    # reach-forward pose (gen3.xml "home" keyframe), producing a lateral/forward approach
    # to tabletop objects confirmed via offscreen render. Middle/ring fingers (not in
    # FINGER_SET) are curled to prevent their extended-at-zero default pose from poking
    # the floor when the palm lowers to table height. Joint order per finger: mcp,rot,pip,dip.
    Q_BIAS = np.zeros(N_ROBOT)
    Q_BIAS[:7]    = HOME_ARM          # arm: elbow-bent reach-forward
    Q_BIAS[11:15] = [1.2, 0.0, 1.5, 1.0]  # leap_mf_*: curl out of the way
    Q_BIAS[15:19] = [1.2, 0.0, 1.5, 1.0]  # leap_rf_*: curl out of the way

    # Solve IK for each object — grasp and pregrasp configs
    for i, obj in enumerate(objects):
        # Contact-face normals: vector from object center to each contact site (normalised).
        p_obj = data.xpos[obj['id_body']]
        normals = []
        for p_S_W in obj['p_S_W']:
            n = p_S_W - p_obj
            n /= np.linalg.norm(n)
            normals.append(n)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        obj['q_target'] = ik_solver.solve(model, data, id_C, obj['p_S_W'],
                                           q_bias=Q_BIAS, null_gain=0.3)

        # Pregrasp: fingertips offset PREGRASP_OFFSET metres outward along contact normals.
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        pregrasp_targets = [p + PREGRASP_OFFSET * n for p, n in zip(obj['p_S_W'], normals)]
        obj['q_pregrasp'] = ik_solver.solve(model, data, id_C, pregrasp_targets,
                                             q_bias=Q_BIAS, null_gain=0.3)
        print(f"Object {i+1} q_target:   {obj['q_target']}")
        print(f"Object {i+1} q_pregrasp: {obj['q_pregrasp']}")

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

    # Only check fingertip spheres against objects during RRT path finding.
    # Checking the full finger chain (bs/px/md links) disqualifies the pregrasp goal
    # itself because the index finger's proximal links pass near the object to reach
    # the approach position — an unavoidable consequence of this hand geometry. The
    # distal tip sphere is the only link that must actually clear the object surface.
    FINGER_TIP_GEOMS = [f"leap_{FINGER_CODE[f]}_tip" for f in FINGER_SET]
    OBJ_BODIES   = ['obj_box', 'obj_cylinder']
    planner = RRTPlanner(model, FINGER_TIP_GEOMS, OBJ_BODIES, n_robot=N_ROBOT,
                         n_plan=7,            # plan only the 7 arm joints; finger DOF fixed at goal
                         clearance=0.01)      # 1cm tip clearance; 6cm pregrasp offset keeps goal free

    # Background RRT: result dict shared between thread and main loop
    _plan_result = {}

    def _run_rrt(q_start, q_pregrasp):
        # Snapshot current object positions so collision checks reflect the live scene.
        planner._data.qpos[N_ROBOT:] = data.qpos[N_ROBOT:].copy()
        path = planner.plan(q_start, q_pregrasp)
        if path is None:
            print("\r\n[RRT] Planning failed — falling back to pregrasp direct reach")
            for i, o in enumerate(objects):
                print(f"         obj{i+1} pos: {data.xpos[o['id_body']]}")
            _plan_result['waypoints'] = [q_pregrasp]
        else:
            _plan_result['waypoints'] = path

    # Build target list: index 0 = init pose (all zeros), 1..N = object IK solutions
    targets = [{'q_target': np.zeros(N_ROBOT), 'label': 'init pose'}]
    for i, obj in enumerate(objects):
        targets.append({'q_target': obj['q_target'], 'label': f'object {i+1}'})

    keys = queue.Queue()
    print("[Control] 0/1/2: select target  |  ←→: jog x  |  ↑↓: jog z  |  Enter: GRASP  |  Q/Esc: quit")
    print("[Control] Active target: init pose")

    # Simulation
    mj.mj_resetData(model, data)
    control_phase  = 'REACH'
    active_idx     = 0
    active_tgt     = 0        # index into targets[]
    tau_ctrl       = np.zeros(model.nv)   # full nv for qfrc_applied
    traj_waypoints = []
    traj_wp_idx    = 0
    traj_wp_step   = 0    # counts sim steps since last waypoint advance
    plan_thread    = None
    q_plan_hold    = np.zeros(N_ROBOT)    # robot DOFs only
    jog_xz         = np.zeros(2)   # [x, z] manual jog offset, added to the frozen grasp pose
    _held_arrows   = set()         # arrow keys currently held (populated by pynput)

    def _on_press(key):
        try:
            if   key == _pynput_kb.Key.right: _held_arrows.add('right')
            elif key == _pynput_kb.Key.left:  _held_arrows.add('left')
            elif key == _pynput_kb.Key.up:    _held_arrows.add('up')
            elif key == _pynput_kb.Key.down:  _held_arrows.add('down')
        except AttributeError:
            pass

    def _on_release(key):
        try:
            if   key == _pynput_kb.Key.right: _held_arrows.discard('right')
            elif key == _pynput_kb.Key.left:  _held_arrows.discard('left')
            elif key == _pynput_kb.Key.up:    _held_arrows.discard('up')
            elif key == _pynput_kb.Key.down:  _held_arrows.discard('down')
        except AttributeError:
            pass

    _kb_listener = _pynput_kb.Listener(on_press=_on_press, on_release=_on_release)
    _kb_listener.start()

    with mj.viewer.launch_passive(model, data, key_callback=make_key_callback(keys)) as viewer:
        viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.label = mj.mjtLabel.mjLABEL_CONTACTFORCE
        running = True
        while viewer.is_running() and running:
            step_start = time.time()

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

                elif key in ('num0', 'num1', 'num2') and control_phase != 'GRASP':
                    new_tgt = int(key[-1])
                    if new_tgt < len(targets) and new_tgt != active_tgt:
                        active_tgt = new_tgt
                        active_idx = max(0, active_tgt - 1)  # map back to objects[]
                        q_start = data.qpos[:N_ROBOT].copy()
                        if active_tgt == 0:
                            q_home = targets[0]['q_target']
                            traj_waypoints = [q_home]
                            traj_wp_idx    = 0
                            traj_wp_step   = 0
                            control_phase  = 'REACH'
                        else:
                            q_pre = objects[active_idx]['q_pregrasp']
                            _plan_result.clear()
                            q_plan_hold   = q_start.copy()   # N_ROBOT-sized
                            plan_thread   = threading.Thread(
                                target=_run_rrt, args=(q_start, q_pre), daemon=True)
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

            obj = objects[active_idx]

            # --- PLAN: hold position while RRT runs in background ---
            if control_phase == 'PLAN':
                tau_ctrl = np.zeros(model.nv)
                tau_ctrl[:N_ROBOT] = Kp * (q_plan_hold - data.qpos[:N_ROBOT]) + Kd * (0 - data.qvel[:N_ROBOT])

            # --- REACH: joint-space PD tracking along RRT waypoints ---
            elif control_phase == 'REACH':
                if traj_waypoints:
                    if traj_wp_idx < len(traj_waypoints) - 1:
                        wp = traj_waypoints[traj_wp_idx].copy()
                        # Advance when robot is within tolerance of current waypoint,
                        # or after timeout — ensures the path is actually followed, not shortcut.
                        traj_wp_step += 1
                        at_wp = np.linalg.norm(data.qpos[:N_ROBOT] - wp) < WP_REACH_TOL
                        if at_wp or traj_wp_step >= STEPS_PER_WP:
                            traj_wp_step = 0
                            traj_wp_idx += 1
                    else:
                        # Trajectory complete: hold at the final (pregrasp) waypoint.
                        wp = traj_waypoints[-1].copy()
                else:
                    # No trajectory: hold at the selected target.
                    wp = obj['q_target'].copy()
                tau_ctrl = np.zeros(model.nv)
                tau_ctrl[:N_ROBOT] = Kp * (wp - data.qpos[:N_ROBOT]) + Kd * (0 - data.qvel[:N_ROBOT])

            # --- GRASP: Cartesian impedance + null-space squeeze, generalized to
            # N_FINGERS contacts ---
            elif control_phase == 'GRASP':
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
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    _kb_listener.stop()

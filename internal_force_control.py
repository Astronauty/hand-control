#!/usr/bin/env python3
# All variable naming notation follows https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html
import numpy as np
import time
import threading
import queue

import mujoco as mj
import mujoco.viewer  # noqa: F401
from pynput import keyboard as _pynput_kb

from scripts.rrt_planner import RRTPlanner
from grasp_control import PlanarGraspMapComputer, PlanarIKSolver, GraspForceAllocator


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
    model = mj.MjModel.from_xml_path('models/planar_two_finger_manipulator.xml')
    data  = mj.MjData(model)
    mj.mj_forward(model, data)

    # Fingertip sites (on fingers)
    id_C1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'index_touch')
    id_C2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'thumb_touch')

    N_ROBOT         = 8     # number of robot DOFs (object joints follow in qpos/qvel)
    PREGRASP_OFFSET = 0.05  # metres to offset fingertips along contact-face normal for RRT goal
    RRT_CLEARANCE   = PREGRASP_OFFSET - 0.005  # 0.045 m geom-to-geom gap enforced along RRT path
    SAFE_LIFT       = 0.12  # metres to lift base before handing off to RRT (clears contact zone)
    STEPS_PER_WP    = 50   # max sim steps before forcing waypoint advance (timeout, 1 step = 1ms)
    WP_REACH_TOL    = 0.01  # joint-space radius to consider a waypoint reached (m or rad)
    JOG_VEL         = 1.0  # base jog speed while arrow key held (m/s)

    # Object definitions
    object_defs = [
        ('index_box_touch',   'thumb_box_touch',   'obj1'),
        ('index_box_touch_2', 'thumb_box_touch_2', 'obj2'),
        ('index_cyl_touch',   'thumb_cyl_touch',   'obj3'),
    ]
    objects = []
    for s1_name, s2_name, body_name in object_defs:
        obj = {
            'id_S1':   mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, s1_name),
            'id_S2':   mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, s2_name),
            'id_body': mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name),
        }
        obj['p_S1_W'] = data.site_xpos[obj['id_S1']][:2].copy()
        obj['p_S2_W'] = data.site_xpos[obj['id_S2']][:2].copy()
        objects.append(obj)

    ik_solver = PlanarIKSolver(n_robot=N_ROBOT)
    grasp_map_computer = PlanarGraspMapComputer()

    # Solve IK for each object — grasp and pregrasp configs
    for i, obj in enumerate(objects):
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        obj['q_target'] = ik_solver.solve(model, data, [id_C1, id_C2],
                                           [obj['p_S1_W'], obj['p_S2_W']])

        # Contact-face normals: vector from box center to each contact site (normalised).
        p_box = data.xpos[obj['id_body']][:2]
        n1 = obj['p_S1_W'] - p_box;  n1 /= np.linalg.norm(n1)
        n2 = obj['p_S2_W'] - p_box;  n2 /= np.linalg.norm(n2)

        # Pregrasp: fingertips offset PREGRASP_OFFSET metres outward along face normals.
        # This gives a side approach (along normal) instead of top-down, avoiding corner clips.
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        obj['q_pregrasp'] = ik_solver.solve(model, data, [id_C1, id_C2],
                                            [obj['p_S1_W'] + PREGRASP_OFFSET * n1,
                                             obj['p_S2_W'] + PREGRASP_OFFSET * n2])
        print(f"Object {i+1} q_target:   {obj['q_target']}")
        print(f"Object {i+1} q_pregrasp: {obj['q_pregrasp']}")

    # Per-joint PD gains for REACH phase.
    # Index: [base_x, base_y, idx_MCP, idx_PIP, idx_DIP, th_MCP, th_PIP, th_DIP]
    Kp = np.array([4.0,  4.0,  0.8,  0.8,  0.8,  0.8,  0.8,  0.8])
    Kd = np.array([2.00, 2.00, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    Kp_obj     = 50.0  # object position stiffness, N/m (GRASP)
    Kd_obj     = 5.0   # object position damping, N·s/m (GRASP)
    Kp_theta   = 5.0   # object orientation stiffness, N·m/rad (GRASP)
    Kd_theta   = 0.5   # object orientation damping, N·m·s/rad (GRASP)
    Kp_contact = 100.0   # weak per-finger slip-correction stiffness, N/m (GRASP)
    Kd_contact = 10.0   # weak per-finger slip-correction damping, N·s/m (GRASP)
    gamma   = 50.0    # internal squeeze force scale; negate if fingers pull apart
    force_allocator = GraspForceAllocator(gamma)

    FINGER_GEOMS = ['index_proximal', 'index_medial', 'index_distal',
                    'thumb_proximal', 'thumb_medial', 'thumb_distal']
    OBJ_BODIES   = ['obj1', 'obj2', 'obj3']
    planner = RRTPlanner(model, FINGER_GEOMS, OBJ_BODIES, n_robot=N_ROBOT,
                         clearance=RRT_CLEARANCE)

    # Background RRT: result dict shared between thread and main loop
    _plan_result = {}

    def _run_rrt(q_start, q_pregrasp):
        # Snapshot current object positions so collision checks reflect the live scene.
        planner._data.qpos[N_ROBOT:] = data.qpos[N_ROBOT:].copy()
        q_safe = q_start.copy()
        q_safe[1] = min(q_safe[1] + SAFE_LIFT, model.jnt_range[1, 1])
        path = planner.plan(q_safe, q_pregrasp)
        if path is None:
            print("\r\n[RRT] Planning failed — falling back to pregrasp direct reach")
            for i, o in enumerate(objects):
                print(f"         obj{i+1} pos: {data.xpos[o['id_body']][:2]}")
            _plan_result['waypoints'] = [q_pregrasp]
        else:
            # Prepend densified departure segment (q_start → q_safe).
            # Path ends at q_pregrasp (fingers offset along face normals, in free space).
            # GRASP mode (Enter) closes the fingers inward to contact via Cartesian impedance.
            start_seg = planner._densify([q_start, path[0]])
            _plan_result['waypoints'] = start_seg[:-1] + path

    # Build target list: index 0 = init pose (all zeros), 1..N = object IK solutions
    targets = [{'q_target': np.zeros(N_ROBOT), 'label': 'init pose'}]
    for i, obj in enumerate(objects):
        targets.append({'q_target': obj['q_target'], 'label': f'object {i+1}'})

    keys = queue.Queue()
    print("[Control] 0/1/2/3: select target  |  ←→↑↓: jog base  |  Enter: GRASP  |  Q/Esc: quit")
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
    jog_base       = np.zeros(2)   # [x, y] manual base target
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
        viewer.cam.fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'cam0')
        viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
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
                jog_base[:]    = traj_waypoints[-1][:2]  # seed jog from final waypoint
                control_phase  = 'REACH'
                print(f"\r\n[Control] REACH  |  path: {len(traj_waypoints)} waypoints")

            # --- Process key events ---
            while not keys.empty():
                key = keys.get_nowait()

                if key == 'enter' and control_phase == 'REACH':
                    control_phase = 'GRASP'
                    jog_base[:] = data.qpos[:2].copy()
                    # Freeze the object's pose relative to the base at grasp time —
                    # jogging will target this offset, not the object's live pose.
                    obj_grasp = objects[active_idx]
                    obj_grasp['p_obj_offset'] = data.xpos[obj_grasp['id_body']][:2].copy() - data.qpos[:2].copy()
                    R_WO0 = data.xmat[obj_grasp['id_body']].reshape(3, 3)[:2, :2]
                    obj_grasp['theta_des'] = np.arctan2(R_WO0[1, 0], R_WO0[0, 0])
                    print(f"\r\n[Control] → GRASP  ({targets[active_tgt]['label']})")

                elif key == 'enter' and control_phase == 'GRASP':
                    q_pre = objects[active_idx]['q_pregrasp'].copy()
                    traj_waypoints = [q_pre]
                    traj_wp_idx    = 0
                    traj_wp_step   = 0
                    jog_base[:]    = data.qpos[:2].copy()  # keep base where it is after release
                    control_phase  = 'REACH'
                    print(f"\r\n[Control] → REACH  (released — returning to pregrasp)")

                elif key in ('num0', 'num1', 'num2', 'num3') and control_phase != 'GRASP':
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
                            jog_base[:]    = q_home[:2]
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

            # --- Continuous jog: drive jog_base from currently-held arrow keys ---
            _jv_x = (JOG_VEL if 'right' in _held_arrows else 0) - (JOG_VEL if 'left' in _held_arrows else 0)
            _jv_y = (JOG_VEL if 'up'    in _held_arrows else 0) - (JOG_VEL if 'down' in _held_arrows else 0)
            if _jv_x or _jv_y:
                x_lo, x_hi = model.jnt_range[0]
                y_lo, y_hi = model.jnt_range[1]
                dt = model.opt.timestep
                jog_base[0] = np.clip(jog_base[0] + _jv_x * dt, x_lo, x_hi)
                jog_base[1] = np.clip(jog_base[1] + _jv_y * dt, y_lo, y_hi)

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
                        # Trajectory complete: hold finger joints at pregrasp, base follows jog.
                        wp = traj_waypoints[-1].copy()
                        wp[:2] = jog_base
                else:
                    # No trajectory: base follows jog_base, fingers follow selected target.
                    wp = obj['q_target'].copy()
                    wp[:2] = jog_base
                tau_ctrl = np.zeros(model.nv)
                tau_ctrl[:N_ROBOT] = Kp * (wp - data.qpos[:N_ROBOT]) + Kd * (0 - data.qvel[:N_ROBOT])

            # --- GRASP: Cartesian impedance + null-space squeeze ---
            elif control_phase == 'GRASP':
                p_WoO_cur  = data.xpos[obj['id_body']][:2]
                R_WO_cur   = data.xmat[obj['id_body']].reshape(3, 3)[:2, :2]
                R_WS1_cur  = data.site_xmat[obj['id_S1']].reshape(3, 3)[:2, :2]
                R_WS2_cur  = data.site_xmat[obj['id_S2']].reshape(3, 3)[:2, :2]

                # Object contact site positions (kept live so code generalises to free objects)
                p_WoS1_cur = data.site_xpos[obj['id_S1']][:2]
                p_WoS2_cur = data.site_xpos[obj['id_S2']][:2]

                p_OS1_O = R_WO_cur.T @ (p_WoS1_cur - p_WoO_cur)
                p_OS2_O = R_WO_cur.T @ (p_WoS2_cur - p_WoO_cur)
                R_OS1   = R_WO_cur.T @ R_WS1_cur
                R_OS2   = R_WO_cur.T @ R_WS2_cur

                G_cur = grasp_map_computer.compute([
                    {'p': p_OS1_O, 'R': R_OS1},
                    {'p': p_OS2_O, 'R': R_OS2},
                ])
                # Inward (compressive) squeeze direction, anchored at contact 1: the
                # finger-1 null-space component should point toward the box center.
                inward_c1 = R_OS1.T @ (-p_OS1_O / np.linalg.norm(p_OS1_O))

                # Fingertip Jacobians (world frame, 2D) — restrict to robot DOF columns
                J1_full, J2_full = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mj.mj_jacSite(model, data, J1_full, None, id_C1)
                mj.mj_jacSite(model, data, J2_full, None, id_C2)
                J1 = J1_full[:2, :N_ROBOT]
                J2 = J2_full[:2, :N_ROBOT]

                # Desired object pose: jog defines a target, tracked via the offset
                # frozen at grasp time. Orientation is held at its grasp-time value.
                p_obj_des     = jog_base + obj['p_obj_offset']
                theta_obj_cur = np.arctan2(R_WO_cur[1, 0], R_WO_cur[0, 0])
                e_p     = p_obj_des - p_WoO_cur
                e_theta = np.arctan2(np.sin(obj['theta_des'] - theta_obj_cur),
                                      np.cos(obj['theta_des'] - theta_obj_cur))

                # Object velocity via body Jacobian (no need to track per-object qvel indices)
                Jp_o, Jr_o = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mj.mj_jacBodyCom(model, data, Jp_o, Jr_o, obj['id_body'])
                v_obj     = Jp_o[:2] @ data.qvel
                omega_obj = Jr_o[2]  @ data.qvel

                # Desired object wrench (world-frame force, scalar moment), rotated into
                # the object frame to match PlanarGraspMapComputer's convention.
                f_des_W = Kp_obj * e_p + Kd_obj * (-v_obj)
                m_des   = Kp_theta * e_theta + Kd_theta * (-omega_obj)
                w_des_O = np.concatenate([R_WO_cur.T @ f_des_W, [m_des]])

                # Allocate desired wrench to contact forces (min-norm) + null-space squeeze
                f_c = force_allocator.allocate(G_cur, w_des_O, contact_dof=2,
                                                inward_dirs=[inward_c1, None])

                # Low-gain hybrid contact correction: resists slip without overpowering
                # the wrench-allocated motion above (compare Kp_contact to old Kp_cart=50).
                dp1 = J1 @ data.qvel[:N_ROBOT]
                dp2 = J2 @ data.qvel[:N_ROBOT]
                f_corr_1 = Kp_contact * (p_WoS1_cur - data.site_xpos[id_C1][:2]) + Kd_contact * (-dp1)
                f_corr_2 = Kp_contact * (p_WoS2_cur - data.site_xpos[id_C2][:2]) + Kd_contact * (-dp2)

                f_c1_W = R_WS1_cur @ f_c[:2] + f_corr_1
                f_c2_W = R_WS2_cur @ f_c[2:] + f_corr_2

                tau_ctrl = np.zeros(model.nv)
                tau_ctrl[:N_ROBOT] = J1.T @ f_c1_W + J2.T @ f_c2_W

            data.qfrc_applied[:] = tau_ctrl
            data.qfrc_applied[:N_ROBOT] += data.qfrc_bias[:N_ROBOT]
            mj.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    _kb_listener.stop()

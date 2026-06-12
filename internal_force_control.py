# All variable naming notation follows https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html
import numpy as np
import seaborn as sns
import scipy.linalg
import time
import threading
import queue
import sys
import tty
import termios

import mujoco as mj
import mujoco.viewer  # noqa: F401

from rrt_planner import RRTPlanner


def planar_hat_map(a):
    return np.array([[0, -a], [a, 0]])


def planar_grasp_map_PCWF(p_S1_O, p_S2_O, R_S1, R_S2):
    """Maps stacked contact forces [f1; f2] in respective contact frames to
    a wrench [fx, fy, tau_z] on the object in the object frame.

    Args:
        p_S1_O: Contact site 1 position relative to object center in object frame
        p_S2_O: Contact site 2 position relative to object center in object frame
        R_S1: Rotation of contact frame 1 relative to object frame
        R_S2: Rotation of contact frame 2 relative to object frame
    """
    G_1 = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-p_S1_O[1], p_S1_O[0]],
    ]) @ R_S1

    G_2 = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-p_S2_O[1], p_S2_O[0]],
    ]) @ R_S2

    return np.block([G_1, G_2])


def solve_ik(model, data, id_C1, id_C2, p_S1_target, p_S2_target, n_robot=8):
    """Damped-least-squares IK over the first n_robot joints only.
    Object joints beyond n_robot are left unchanged. Returns n_robot-length q."""
    q = data.qpos[:n_robot].copy()
    for _ in range(500):
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        err = np.concatenate([p_S1_target - data.site_xpos[id_C1][:2],
                               p_S2_target - data.site_xpos[id_C2][:2]])
        if np.linalg.norm(err) < 1e-3:
            break
        J1, J2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, J1, None, id_C1)
        mj.mj_jacSite(model, data, J2, None, id_C2)
        J  = np.vstack([J1[:2, :n_robot], J2[:2, :n_robot]])
        dq = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(4)) @ err
        q += 0.5 * dq
        data.qpos[:n_robot] = q
    return q


def key_listener(key_queue):
    """Read keys from stdin in raw mode, push events to queue."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ('\r', '\n'):
                key_queue.put('enter')
            elif ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if   ch3 == 'A': key_queue.put('up')
                    elif ch3 == 'B': key_queue.put('down')
                    elif ch3 == 'C': key_queue.put('right')
                    elif ch3 == 'D': key_queue.put('left')
            elif ch in ('0', '1', '2', '3'):
                key_queue.put(f'num{ch}')
            elif ch in ('q', '\x03'):   # q or Ctrl+C
                key_queue.put('quit')
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    sns.set_theme(style="ticks")

    model = mj.MjModel.from_xml_path('models/planar_two_finger_manipulator.xml')
    data  = mj.MjData(model)
    mj.mj_forward(model, data)

    # Fingertip sites (on fingers)
    id_C1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'index_touch')
    id_C2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'thumb_touch')

    N_ROBOT         = 8     # number of robot DOFs (object joints follow in qpos/qvel)
    PREGRASP_OFFSET = 0.05  # metres to offset fingertips along contact-face normal for RRT goal
    SAFE_LIFT       = 0.12  # metres to lift base before handing off to RRT (clears contact zone)
    STEPS_PER_WP    = 50    # sim steps between reference waypoint advances (1 step = 1ms)
    JOG_STEP        = 0.04  # metres per arrow-key press

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

    # Solve IK for each object — grasp and pregrasp configs
    for i, obj in enumerate(objects):
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        obj['q_target'] = solve_ik(model, data, id_C1, id_C2, obj['p_S1_W'], obj['p_S2_W'])

        # Contact-face normals: vector from box center to each contact site (normalised).
        p_box = data.xpos[obj['id_body']][:2]
        n1 = obj['p_S1_W'] - p_box;  n1 /= np.linalg.norm(n1)
        n2 = obj['p_S2_W'] - p_box;  n2 /= np.linalg.norm(n2)

        # Pregrasp: fingertips offset PREGRASP_OFFSET metres outward along face normals.
        # This gives a side approach (along normal) instead of top-down, avoiding corner clips.
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        obj['q_pregrasp'] = solve_ik(model, data, id_C1, id_C2,
                                     obj['p_S1_W'] + PREGRASP_OFFSET * n1,
                                     obj['p_S2_W'] + PREGRASP_OFFSET * n2)
        print(f"Object {i+1} q_target:   {obj['q_target']}")
        print(f"Object {i+1} q_pregrasp: {obj['q_pregrasp']}")

    # Per-joint PD gains for REACH phase.
    # Index: [base_x, base_y, idx_MCP, idx_PIP, idx_DIP, th_MCP, th_PIP, th_DIP]
    Kp = np.array([3.0,  3.0,  0.8,  0.8,  0.8,  0.8,  0.8,  0.8])
    Kd = np.array([2.00, 2.00, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    Kp_cart = 50.0   # Cartesian impedance stiffness N/m (GRASP)
    Kd_cart = 5.0    # Cartesian impedance damping N·s/m (GRASP)
    gamma   = 5.0    # internal squeeze force scale; negate if fingers pull apart

    FINGER_GEOMS = ['index_proximal', 'index_medial', 'index_distal',
                    'thumb_proximal', 'thumb_medial', 'thumb_distal']
    OBJ_BODIES   = ['obj1', 'obj2', 'obj3']
    planner = RRTPlanner(model, FINGER_GEOMS, OBJ_BODIES, n_robot=N_ROBOT)

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

    # Start key listener thread
    keys = queue.Queue()
    threading.Thread(target=key_listener, args=(keys,), daemon=True).start()
    print("\r\n[Control] 0/1/2/3: select target  |  ←→↑↓: jog base  |  Enter: GRASP  |  q: quit")
    print(f"[Control] Active target: init pose\r\n")

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
    jog_base       = np.zeros(2)  # [x, y] manual base target

    with mj.viewer.launch_passive(model, data) as viewer:
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
                    print(f"\r\n[Control] → GRASP  ({targets[active_tgt]['label']})")

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

                elif key in ('left', 'right', 'up', 'down') and control_phase != 'GRASP':
                    x_lo, x_hi = model.jnt_range[0]
                    y_lo, y_hi = model.jnt_range[1]
                    if   key == 'right': jog_base[0] = min(jog_base[0] + JOG_STEP, x_hi)
                    elif key == 'left':  jog_base[0] = max(jog_base[0] - JOG_STEP, x_lo)
                    elif key == 'up':    jog_base[1] = min(jog_base[1] + JOG_STEP, y_hi)
                    elif key == 'down':  jog_base[1] = max(jog_base[1] - JOG_STEP, y_lo)

                elif key == 'quit':
                    running = False

            obj = objects[active_idx]

            # --- PLAN: hold position while RRT runs in background ---
            if control_phase == 'PLAN':
                tau_ctrl = np.zeros(model.nv)
                tau_ctrl[:N_ROBOT] = Kp * (q_plan_hold - data.qpos[:N_ROBOT]) + Kd * (0 - data.qvel[:N_ROBOT])

            # --- REACH: joint-space PD tracking along RRT waypoints ---
            elif control_phase == 'REACH':
                if traj_waypoints:
                    if traj_wp_idx < len(traj_waypoints) - 1:
                        # Advance reference at a fixed time rate regardless of robot position.
                        traj_wp_step += 1
                        if traj_wp_step >= STEPS_PER_WP:
                            traj_wp_step = 0
                            traj_wp_idx += 1
                        wp = traj_waypoints[traj_wp_idx].copy()
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

                G_cur      = planar_grasp_map_PCWF(p_OS1_O, p_OS2_O, R_OS1, R_OS2)
                G_null_cur = scipy.linalg.null_space(G_cur).flatten()

                # Fingertip Jacobians (world frame, 2D) — restrict to robot DOF columns
                J1_full, J2_full = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mj.mj_jacSite(model, data, J1_full, None, id_C1)
                mj.mj_jacSite(model, data, J2_full, None, id_C2)
                J1 = J1_full[:2, :N_ROBOT]
                J2 = J2_full[:2, :N_ROBOT]

                # Cartesian impedance forces in world frame
                dp1 = J1 @ data.qvel[:N_ROBOT]
                dp2 = J2 @ data.qvel[:N_ROBOT]
                f_imp_1 = Kp_cart * (p_WoS1_cur - data.site_xpos[id_C1][:2]) + Kd_cart * (-dp1)
                f_imp_2 = Kp_cart * (p_WoS2_cur - data.site_xpos[id_C2][:2]) + Kd_cart * (-dp2)

                # Null-space squeeze: G_null is in contact frames, rotate to world frame
                f_null_1 = R_WS1_cur @ G_null_cur[:2] * gamma
                f_null_2 = R_WS2_cur @ G_null_cur[2:] * gamma

                tau_ctrl = np.zeros(model.nv)
                tau_ctrl[:N_ROBOT] = J1.T @ (f_imp_1 + f_null_1) + J2.T @ (f_imp_2 + f_null_2)

            data.qfrc_applied[:] = tau_ctrl + data.qfrc_bias
            mj.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

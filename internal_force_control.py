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


def solve_ik(model, data, id_C1, id_C2, p_S1_target, p_S2_target):
    """Damped-least-squares IK. Modifies data.qpos in place; returns q_target."""
    q = data.qpos.copy()
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
        J  = np.vstack([J1[:2], J2[:2]])
        dq = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(4)) @ err
        q += 0.5 * dq
        data.qpos[:] = q
    return q


def key_listener(key_queue):
    """Read Enter and arrow keys from stdin in raw mode, push events to queue."""
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
                    if ch3 == 'C':
                        key_queue.put('right')
                    elif ch3 == 'D':
                        key_queue.put('left')
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

    # Object definitions
    object_defs = [
        ('index_box_touch',   'thumb_box_touch',   'obj1'),
        ('index_box_touch_2', 'thumb_box_touch_2', 'obj2'),
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

    # Solve IK for each object
    for i, obj in enumerate(objects):
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        obj['q_target'] = solve_ik(model, data, id_C1, id_C2, obj['p_S1_W'], obj['p_S2_W'])
        print(f"Object {i+1} IK solution: {obj['q_target']}")

    # Control parameters
    Kp      = 0.8    # joint-space PD stiffness (REACH)
    Kd      = 0.05   # joint-space PD damping (REACH)
    Kp_cart = 50.0   # Cartesian impedance stiffness N/m (GRASP)
    Kd_cart = 5.0    # Cartesian impedance damping N·s/m (GRASP)
    gamma   = 5.0    # internal squeeze force scale; negate if fingers pull apart

    # Start key listener thread
    keys = queue.Queue()
    threading.Thread(target=key_listener, args=(keys,), daemon=True).start()
    print("\r\n[Control] REACH  |  Enter: switch to GRASP  |  ←/→: select object  |  q: quit")
    print(f"[Control] Active target: object 1\r\n")

    # Simulation
    mj.mj_resetData(model, data)
    control_phase = 'REACH'
    active_idx    = 0
    tau_ctrl      = np.zeros(model.nv)

    with mj.viewer.launch_passive(model, data) as viewer:
        running = True
        while viewer.is_running() and running:
            step_start = time.time()

            # --- Process key events ---
            while not keys.empty():
                key = keys.get_nowait()
                if key == 'enter' and control_phase == 'REACH':
                    control_phase = 'GRASP'
                    print(f"\r\n[Control] → GRASP  (object {active_idx + 1})")
                elif key == 'right':
                    active_idx = (active_idx + 1) % len(objects)
                    print(f"\r\n[Control] Target: object {active_idx + 1}")
                elif key == 'left':
                    active_idx = (active_idx - 1) % len(objects)
                    print(f"\r\n[Control] Target: object {active_idx + 1}")
                elif key == 'quit':
                    running = False

            obj = objects[active_idx]

            # --- REACH: joint-space PD to IK solution ---
            if control_phase == 'REACH':
                tau_ctrl = Kp * (obj['q_target'] - data.qpos) + Kd * (0 - data.qvel)

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

                # Fingertip Jacobians (world frame, 2D)
                J1_full, J2_full = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mj.mj_jacSite(model, data, J1_full, None, id_C1)
                mj.mj_jacSite(model, data, J2_full, None, id_C2)
                J1 = J1_full[:2]
                J2 = J2_full[:2]

                # Cartesian impedance forces in world frame
                dp1 = J1 @ data.qvel
                dp2 = J2 @ data.qvel
                f_imp_1 = Kp_cart * (p_WoS1_cur - data.site_xpos[id_C1][:2]) + Kd_cart * (-dp1)
                f_imp_2 = Kp_cart * (p_WoS2_cur - data.site_xpos[id_C2][:2]) + Kd_cart * (-dp2)

                # Null-space squeeze: G_null is in contact frames, rotate to world frame
                f_null_1 = R_WS1_cur @ G_null_cur[:2] * gamma
                f_null_2 = R_WS2_cur @ G_null_cur[2:] * gamma

                tau_ctrl = J1.T @ (f_imp_1 + f_null_1) + J2.T @ (f_imp_2 + f_null_2)

            data.qfrc_applied[:] = tau_ctrl + data.qfrc_bias
            mj.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

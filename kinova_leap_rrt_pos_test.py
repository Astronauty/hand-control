#!/usr/bin/env python3
"""RRT path-following test using position-servo actuators.

Loads the position-control scene (kinova_leap_pos.xml), runs IK to compute
pregrasp configs, runs RRT in a background thread, then follows the returned
waypoint path by setting data.ctrl to the current waypoint. No torque/force
control — the built-in PD servos handle trajectory tracking.

Controls:
  1 / 2      select object (numpad)
  Enter      re-trigger RRT for current object
  Q / Esc    quit
"""

import numpy as np
import time
import threading
import queue

import mujoco as mj
import mujoco.viewer

from scripts.rrt_planner import RRTPlanner
from grasp_control import SpatialIKSolver

# ── Constants ──────────────────────────────────────────────────────────────────

FINGER_SET  = ["index", "thumb"]
FINGER_CODE = {"index": "if", "thumb": "th"}
FINGER_TIP_SITES = {
    "index": "leap_if_ds_tip",
    "thumb": "leap_th_ds_tip",
}

HOME_ARM = np.array([0.0, 0.26179939, 3.14159265, -2.26892803,
                     0.0, 0.95993109, 1.57079633])

PREGRASP_OFFSET = 0.06   # metres along contact normal
N_ROBOT         = 23     # 7 arm + 16 LEAP finger joints
FRAMES_PER_WP   = 3      # viewer frames to show each waypoint during kinematic replay


def _make_key_callback(key_queue):
    _MAP = {
        257: 'enter', 335: 'enter',  # Enter / KP_Enter
        49:  'num1',  50:  'num2',   # top-row 1, 2
        81:  'quit',  256: 'quit',   # Q, Esc
    }
    def _cb(keycode):
        ev = _MAP.get(keycode)
        if ev:
            key_queue.put(ev)
    return _cb


if __name__ == "__main__":
    model = mj.MjModel.from_xml_path('models/scene_pick_place_pos.xml')
    data  = mj.MjData(model)

    # Continuous-revolute arm joints have jnt_range=[0,0]; give them a sampling
    # bound for the RRT (doesn't affect control, only planner random sampling).
    for j in (0, 2, 4, 6):
        model.jnt_range[j] = [-np.pi, np.pi]

    mj.mj_forward(model, data)

    # ── IK setup ───────────────────────────────────────────────────────────────
    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
            for f in FINGER_SET]

    Q_BIAS = np.zeros(N_ROBOT)
    Q_BIAS[:7]    = HOME_ARM
    Q_BIAS[11:15] = [1.2, 0.0, 1.5, 1.0]   # mf: curl out of the way
    Q_BIAS[15:19] = [1.2, 0.0, 1.5, 1.0]   # rf: curl out of the way

    ik = SpatialIKSolver(n_robot=N_ROBOT)

    object_defs = [
        (['obj_box_c1',      'obj_box_c2'],      'obj_box'),
        (['obj_cylinder_c1', 'obj_cylinder_c2'], 'obj_cylinder'),
    ]

    objects = []
    for s_names, body_name in object_defs:
        id_S    = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, n) for n in s_names]
        id_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)

        mj.mj_resetData(model, data); mj.mj_forward(model, data)
        p_S_W   = [data.site_xpos[s].copy() for s in id_S]
        p_obj   = data.xpos[id_body].copy()
        normals = [(p - p_obj) / np.linalg.norm(p - p_obj) for p in p_S_W]

        pre_targets = [p + PREGRASP_OFFSET * n for p, n in zip(p_S_W, normals)]
        q_pregrasp  = ik.solve(model, data, id_C, pre_targets, q_bias=Q_BIAS, null_gain=0.3)

        objects.append({
            'label':      body_name,
            'id_body':    id_body,
            'q_pregrasp': q_pregrasp,
        })
        print(f"[IK] {body_name}: pregrasp solved, arm={q_pregrasp[:7]}")

    # ── RRT setup ──────────────────────────────────────────────────────────────
    FINGER_TIP_GEOMS = [f"leap_{FINGER_CODE[f]}_tip" for f in FINGER_SET]
    planner = RRTPlanner(
        model, FINGER_TIP_GEOMS, ['obj_box', 'obj_cylinder'],
        n_robot=N_ROBOT,
        n_plan=7,       # plan only the 7 arm joints
        clearance=0.01, # 1 cm tip-sphere clearance
    )

    _plan_result = {}
    _plan_lock   = threading.Lock()

    def _run_rrt(q_start, q_goal, obj_name):
        with _plan_lock:
            _plan_result.clear()
        planner._data.qpos[N_ROBOT:] = data.qpos[N_ROBOT:].copy()
        path = planner.plan(q_start, q_goal)
        with _plan_lock:
            if path is None:
                print(f"\r\n[RRT] {obj_name}: failed — falling back to direct")
                _plan_result['waypoints'] = [q_goal]
            else:
                print(f"\r\n[RRT] {obj_name}: {len(path)} waypoints")
                _plan_result['waypoints'] = path

    # ── Simulation state ───────────────────────────────────────────────────────
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # Initialise ctrl to current qpos so servos hold the zero pose without a jerk.
    data.ctrl[:N_ROBOT] = data.qpos[:N_ROBOT].copy()

    phase        = 'IDLE'       # IDLE | PLAN | REACH | HOLD
    active_obj   = 0
    traj         = []
    wp_idx       = 0
    wp_step      = 0
    plan_thread  = None
    keys         = queue.Queue()

    print("[Control] 1/2: select object  |  Enter: plan+reach  |  Q/Esc: quit")

    with mj.viewer.launch_passive(model, data,
                                  key_callback=_make_key_callback(keys)) as viewer:
        viewer.opt.frame = mj.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            step_start = time.time()

            # ── Check if RRT thread finished ───────────────────────────────────
            if plan_thread is not None and not plan_thread.is_alive():
                plan_thread = None
                with _plan_lock:
                    traj   = _plan_result.get('waypoints', [])
                wp_idx  = 0
                wp_step = 0
                phase   = 'REACH' if traj else 'IDLE'
                print(f"[Control] REACH: {len(traj)} waypoints")

            # ── Key events ─────────────────────────────────────────────────────
            while not keys.empty():
                key = keys.get_nowait()

                if key in ('num1', 'num2'):
                    active_obj = int(key[-1]) - 1
                    print(f"[Control] selected: {objects[active_obj]['label']}")

                elif key == 'enter':
                    q_start = data.qpos[:N_ROBOT].copy()
                    q_goal  = objects[active_obj]['q_pregrasp']
                    phase   = 'PLAN'
                    plan_thread = threading.Thread(
                        target=_run_rrt,
                        args=(q_start, q_goal, objects[active_obj]['label']),
                        daemon=True)
                    plan_thread.start()
                    print(f"[Control] PLAN: {objects[active_obj]['label']} …")

                elif key == 'quit':
                    viewer.close()
                    break

            # ── Control ────────────────────────────────────────────────────────
            if phase == 'REACH' and traj:
                # Kinematic replay: set qpos directly, no physics stepping so contact
                # forces from proximal links don't fight the path. Advances one waypoint
                # every FRAMES_PER_WP viewer frames to give a watchable playback speed.
                wp = traj[wp_idx]
                data.qpos[:N_ROBOT] = wp
                data.qvel[:N_ROBOT] = 0.0
                mj.mj_forward(model, data)
                wp_step += 1
                if wp_step >= FRAMES_PER_WP:
                    wp_step = 0
                    if wp_idx < len(traj) - 1:
                        wp_idx += 1
                    else:
                        # Path complete — switch to servo hold
                        data.ctrl[:N_ROBOT] = wp
                        data.qvel[:] = 0.0
                        phase = 'HOLD'
                        print("[Control] HOLD: at pregrasp")
                viewer.sync()
            else:
                # Physics stepping for IDLE / PLAN / HOLD
                if phase == 'HOLD' and traj:
                    data.ctrl[:N_ROBOT] = traj[-1]
                mj.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

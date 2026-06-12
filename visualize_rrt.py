"""
Kinematic RRT path visualizer.

Steps the viewer through each planned waypoint by directly setting qpos — no PD
control, no dynamics. Any visible finger-object penetration here is an RRT
clearance/tolerance issue; if the replay is clean but internal_force_control.py
still clips, the cause is PD tracking deviation.

Usage:
    python visualize_rrt.py           # plan obj1 → obj2
    python visualize_rrt.py --reverse # plan obj2 → obj1
    python visualize_rrt.py --dt 0.1  # seconds per waypoint (default 0.05)
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

from rrt_planner import RRTPlanner


# ── Must match internal_force_control.py ─────────────────────────────────────
MODEL_PATH    = 'models/planar_two_finger_manipulator.xml'
FINGER_GEOMS  = ['index_proximal', 'index_medial', 'index_distal',
                 'thumb_proximal', 'thumb_medial', 'thumb_distal']
OBJ_BODIES    = ['obj1', 'obj2']
PREGRASP_OFFSET = 0.05  # must match internal_force_control.py
SAFE_LIFT       = 0.12
# ─────────────────────────────────────────────────────────────────────────────


def solve_ik(model, data, id_C1, id_C2, p1_target, p2_target):
    q = data.qpos.copy()
    for _ in range(500):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err = np.concatenate([p1_target - data.site_xpos[id_C1][:2],
                              p2_target - data.site_xpos[id_C2][:2]])
        if np.linalg.norm(err) < 1e-3:
            break
        J1 = np.zeros((3, model.nv))
        J2 = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, J1, None, id_C1)
        mujoco.mj_jacSite(model, data, J2, None, id_C2)
        J  = np.vstack([J1[:2], J2[:2]])
        dq = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(4)) @ err
        q  += 0.5 * dq
        data.qpos[:] = q
    return q


def min_finger_obj_dist(model, data, finger_geoms, obj_geoms):
    """Return (min_dist, fg_id, og_id) across all finger-object geom pairs."""
    fromto  = np.zeros(6)
    min_d   = np.inf
    min_ids = (None, None)
    for fg in finger_geoms:
        for og in obj_geoms:
            d = mujoco.mj_geomDistance(model, data, fg, og, 10.0, fromto)
            if d < min_d:
                min_d   = d
                min_ids = (fg, og)
    return min_d, min_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverse', action='store_true',
                        help='Plan obj2 → obj1 instead of obj1 → obj2')
    parser.add_argument('--dt', type=float, default=0.05,
                        help='Seconds to display each waypoint (default 0.05)')
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    id_C1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'index_touch')
    id_C2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'thumb_touch')

    object_defs = [
        ('index_box_touch',   'thumb_box_touch',   'obj1'),
        ('index_box_touch_2', 'thumb_box_touch_2', 'obj2'),
    ]
    objs = []
    for s1_name, s2_name, body_name in object_defs:
        id_s1  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s1_name)
        id_s2  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s2_name)
        id_bod = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        p1 = data.site_xpos[id_s1][:2].copy()
        p2 = data.site_xpos[id_s2][:2].copy()

        # Contact-face normals: box center → contact site
        p_box = data.xpos[id_bod][:2].copy()
        n1 = p1 - p_box;  n1 /= np.linalg.norm(n1)
        n2 = p2 - p_box;  n2 /= np.linalg.norm(n2)

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        q_grasp = solve_ik(model, data, id_C1, id_C2, p1, p2)

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        q_pre = solve_ik(model, data, id_C1, id_C2,
                         p1 + PREGRASP_OFFSET * n1,
                         p2 + PREGRASP_OFFSET * n2)
        objs.append({'q_grasp': q_grasp, 'q_pregrasp': q_pre, 'p_S1': p1, 'p_S2': p2})
        print(f"[{body_name}]  grasp IK done  pregrasp IK done")

    src_idx, dst_idx = (1, 0) if args.reverse else (0, 1)
    q_start = objs[src_idx]['q_grasp']
    q_grasp = objs[dst_idx]['q_grasp']
    q_pre   = objs[dst_idx]['q_pregrasp']
    q_safe  = q_start.copy()
    q_safe[1] = min(q_safe[1] + SAFE_LIFT, model.jnt_range[1, 1])

    planner = RRTPlanner(model, FINGER_GEOMS, OBJ_BODIES)
    print(f"\nPlanning obj{src_idx+1} → obj{dst_idx+1}  (clearance={planner.clearance}m) ...")
    path = planner.plan(q_safe, q_pre)

    if path is None:
        print("RRT failed — cannot visualize.")
        return

    # Assemble trajectory matching internal_force_control.py exactly:
    #   start_seg : q_start(grasp src) → q_safe(lifted)  — departure
    #   path      : q_safe → q_pregrasp(offset along normals)  — RRT
    # The final q_grasp approach is shown diagnostically (GRASP mode handles it).
    start_seg  = planner._densify([q_start, path[0]])
    n_depart   = len(start_seg) - 1
    n_rrt      = len(path)
    waypoints  = start_seg[:-1] + path
    # Diagnostic: show the pregrasp→grasp closing motion (not executed in REACH).
    diag_extra = planner._densify([path[-1], q_grasp])[1:]
    n_total    = len(waypoints) + len(diag_extra)
    print(f"Depart: {n_depart} wps | RRT: {n_rrt} wps | "
          f"Diag close (not REACH): {len(diag_extra)} wps | Total: {n_total}")
    print(f"dt={args.dt}s per waypoint\n")

    # Collect geom IDs for distance reporting.
    finger_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                       for n in FINGER_GEOMS]
    obj_geom_ids = []
    for body_name in OBJ_BODIES:
        bid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        start = model.body_geomadr[bid]
        for i in range(model.body_geomnum[bid]):
            obj_geom_ids.append(start + i)

    reach_min = np.inf
    diag_min  = np.inf
    all_wps   = list(waypoints) + list(diag_extra)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i, wp in enumerate(all_wps):
            data.qpos[:] = wp
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            d, _ = min_finger_obj_dist(model, data, finger_geom_ids, obj_geom_ids)
            status = "OK  " if d >= planner.clearance else "CLIP"

            in_reach = i < len(waypoints)
            if in_reach:
                reach_min = min(reach_min, d)
                if i < n_depart:
                    phase = "depart"
                    note  = " (expected — departing grasp)"
                else:
                    phase = "rrt   "
                    note  = ""
            else:
                diag_min = min(diag_min, d)
                phase = "diag  "
                note  = " (not in REACH trajectory)"

            print(f"  wp {i:4d}/{n_total-1}  [{phase}]  min_dist={d:+.4f}m  [{status}]{note}")

            viewer.sync()
            time.sleep(args.dt)

        print(f"\n── REACH trajectory (executed by PD controller) ──")
        print(f"   Min distance: {reach_min:+.4f}m   threshold: {planner.clearance}m")
        depart_min_note = "(CLIPs here are expected — robot departs its previous grasp pose)"
        if reach_min < planner.clearance:
            print(f"   ⚠  CLIP detected — check [depart] vs [rrt] labels above. {depart_min_note}")
        else:
            print(f"   ✓  All REACH waypoints respect clearance.")
            print(f"      Any collision in main sim = PD tracking deviation (increase STEPS_PER_WP).")
        print(f"\n── Final descent (diagnostic only — executed by GRASP mode) ──")
        print(f"   Min distance: {diag_min:+.4f}m")
        if diag_min < 0:
            print(f"   ⚠  Descent clips geometry — correct to use Cartesian impedance for contact.")

        print("\nPress Ctrl+C or close window to exit.")
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == '__main__':
    main()

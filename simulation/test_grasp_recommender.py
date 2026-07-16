#!/usr/bin/env python3
"""
test_grasp_recommender.py
=========================
Isolated test for MultiStartGraspPlanner3D.

Predefined kinematic approach paths (Q_BIAS → DLS pregrasp, linear interp)
replace RRT.  Selecting an object starts the approach AND fires the grasp
planner in parallel.  Contact-point markers appear in the viewer as soon as
the planner finishes.  No force control; purely a recommender/visualiser.

Controls
--------
  Ctrl+1   approach obj_box   → fires grasp planner when wrist within threshold
  Ctrl+2   approach obj_cylinder (no grasp planner — geometry not supported yet)
  Ctrl+0   return to home
  Enter    cycle through planner candidates (best → next)
  Q / Esc  quit
"""
import argparse
import logging
import threading
import queue
import time
from datetime import datetime

import numpy as np
import mujoco as mj
import mujoco.viewer
from pynput import keyboard as _pynput_kb

import os, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))                   # for grasp_control
if str(_REPO_ROOT / 'simulation') not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / 'simulation')) 

from grasp_control import SpatialIKSolver
from grasp_planner_3d import GraspConfig3D, MultiStartGraspPlanner3D

# ── Module-level constants ─────────────────────────────────────────────────────

GEN3_XML = 'mujoco_menagerie/kinova_gen3/gen3.xml'
N_ROBOT  = 23

FINGER_TIP_SITES = {"index": "leap_if_ds_tip", "thumb": "leap_th_ds_tip"}

PREGRASP_OFFSET    = 0.04   # m stand-off from object for DLS pregrasp targets
N_APPROACH_WPS     = 200    # waypoints in Q_BIAS → pregrasp path
STEPS_PER_WP       = 3      # sim steps per waypoint
GRASP_TRIGGER_DIST = 0.30   # m wrist-to-object distance that fires the planner


def make_key_callback(key_queue):
    _MAP = {257: 'enter', 335: 'enter', 81: 'quit', 256: 'quit'}
    def _cb(keycode):
        ev = _MAP.get(keycode)
        if ev:
            key_queue.put(ev)
    return _cb


def main():
    ap = argparse.ArgumentParser(
        description="Grasp recommender test (no RRT, no force ctrl)")
    ap.add_argument('--nc',       type=int, default=3,
                    help="Grasp-planner seeds per solve (default 3)")
    ap.add_argument('--max-iter', type=int, default=120,
                    help="Max solver iterations per seed (default 120)")
    ap.add_argument('--log-prefix', type=str, default='',
                    help="Save logs to logs/<prefix>_YYYYMMDD_HHMMSS/ (omit to skip)")
    args = ap.parse_args()

    log_dir = None
    if args.log_prefix:
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"logs/{args.log_prefix}_{ts}"

    logging.basicConfig(
        level   = logging.INFO,
        format  = '%(asctime)s  %(message)s',
        datefmt = '%H:%M:%S')

    model = mj.MjModel.from_xml_path('models/scene_pick_place.xml')
    data  = mj.MjData(model)

    for j in (0, 2, 4, 6):          # give continuous arm joints a sampling range
        model.jnt_range[j] = [-np.pi, np.pi]
    mj.mj_forward(model, data)

    # Fingertip site IDs (thumb first — order matches id_C in kinova_pick_place)
    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
            for f in ("thumb", "index")]

    # Home pose
    HOME_ARM = mj.MjModel.from_xml_path(GEN3_XML).key('home').qpos[:7].copy()
    Q_BIAS = np.zeros(N_ROBOT)
    Q_BIAS[:7]    = HOME_ARM
    Q_BIAS[11:15] = [1.2, 0.0, 0.5, 0.5]   # mf: curl out of the way
    Q_BIAS[15:19] = [1.2, 0.0, 0.5, 0.5]   # rf: curl out of the way

    wrist_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'bracelet_link')

    # ── IK + grasp planner ─────────────────────────────────────────────────────
    dls_ik = SpatialIKSolver(n_robot=N_ROBOT)

    grasp_cfg = GraspConfig3D(
        obj_geom      = 'obj_red_box_geom',
        obj_body      = 'obj_red_box',
        max_iter      = args.max_iter,
    )
    ms_planner = MultiStartGraspPlanner3D(model, data, grasp_cfg, log_dir=log_dir)
    if log_dir:
        print(f"[log] saving to {log_dir}/")
    print(f"[planner] box half-extents: "
          f"hx={ms_planner._obj_hx:.3f}  "
          f"hy={ms_planner._obj_hy:.3f}  "
          f"hz={ms_planner._obj_hz:.3f}")

    # ── Precompute approach paths ──────────────────────────────────────────────
    OBJECT_DEFS = [
        ('obj_red_box',    'obj_red_box_geom',    True),
        ('obj_red_sphere', 'obj_red_sphere_geom', False),
    ]

    objects = []
    for body_name, geom_name, use_planner in OBJECT_DEFS:
        obj = {
            'name':        body_name,
            'id_body':     mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name),
            'id_geom':     mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name),
            'use_planner': use_planner,
        }

        mj.mj_resetData(model, data)
        data.qpos[:N_ROBOT] = Q_BIAS
        mj.mj_forward(model, data)
        obj_pos = data.xpos[obj['id_body']].copy()

        _d = mj.MjData(model)
        _d.qpos[:N_ROBOT] = Q_BIAS
        mj.mj_forward(model, _d)

        if body_name == 'obj_red_sphere':
            # Task-space straight-line approach: fingertips follow a direct Cartesian
            # path from their home positions to the pregrasp targets.  Joint-space linear
            # interpolation arcs below the target Z; task-space interpolation cannot.
            # We warm-start each DLS solve from the previous waypoint so the arm tracks
            # the line smoothly without diverging.

            tips_home = [_d.site_xpos[sid].copy() for sid in id_C]

            # Pregrasp: flanking the cylinder, slightly in front and above the equator
            pregrasp_pos  = obj_pos + np.array([0.0, -PREGRASP_OFFSET * 2, 0.05])
            pregrasp_tgts = [pregrasp_pos + np.array([-0.05, 0., 0.]),
                             pregrasp_pos + np.array([ 0.05, 0., 0.])]

            print(f"[IK] {body_name}: building task-space approach path "
                  f"({N_APPROACH_WPS} waypoints) ...")
            q_cur = Q_BIAS.copy()
            path  = []
            for i in range(N_APPROACH_WPS):
                alpha  = i / max(N_APPROACH_WPS - 1, 1)
                tgts_i = [tips_home[j] + alpha * (pregrasp_tgts[j] - tips_home[j])
                          for j in range(2)]
                q_cur = dls_ik.solve(model, _d, id_C, tgts_i,
                                     q_bias=q_cur, null_gain=0.1)
                _d.qpos[:N_ROBOT] = q_cur
                mj.mj_forward(model, _d)
                path.append(q_cur.copy())

            q_pre = path[-1]
            obj['approach_path'] = path

            errs = [f"{np.linalg.norm(_d.site_xpos[s] - t)*1e3:.1f} mm"
                    for s, t in zip(id_C, pregrasp_tgts)]
            tips = [f"Z={_d.site_xpos[s][2]*1e3:.1f}mm" for s in id_C]
            print(f"[IK] {body_name} pregrasp: tip errors={errs}  tips={tips}")

        else:
            # Box: single DLS solve + joint-space linear interpolation (works fine for box)
            pregrasp_pos  = obj_pos + np.array([0.0, -PREGRASP_OFFSET * 3, 0.12])
            pregrasp_tgts = [pregrasp_pos + np.array([-0.03, 0., 0.]),
                             pregrasp_pos + np.array([ 0.03, 0., 0.])]

            print(f"[IK] {body_name}: DLS pregrasp ...")
            q_pre = dls_ik.solve(model, data, id_C, pregrasp_tgts,
                                 q_bias=Q_BIAS, null_gain=0.1)

            ts = np.linspace(0., 1., N_APPROACH_WPS)
            obj['approach_path'] = [Q_BIAS + t * (q_pre - Q_BIAS) for t in ts]

            _d.qpos[:N_ROBOT] = q_pre
            mj.mj_forward(model, _d)
            errs = [f"{np.linalg.norm(_d.site_xpos[s] - t)*1e3:.1f} mm"
                    for s, t in zip(id_C, pregrasp_tgts)]
            print(f"[IK] {body_name}: pregrasp tip errors = {errs}")

        obj['q_pregrasp'] = q_pre
        obj['obj_pos']    = obj_pos
        objects.append(obj)

    mj.mj_resetData(model, data)
    data.qpos[:N_ROBOT] = Q_BIAS
    mj.mj_forward(model, data)

    # ── Mocap contact markers (red = thumb, blue = index) ─────────────────────
    cp1_mocap = int(model.body_mocapid[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'cp1_body')])
    cp2_mocap = int(model.body_mocapid[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'cp2_body')])
    _HIDDEN = np.array([10., 10., 10.])
    data.mocap_pos[cp1_mocap] = _HIDDEN
    data.mocap_pos[cp2_mocap] = _HIDDEN

    # ── Mutable loop state ────────────────────────────────────────────────────
    state          = 'HOME'   # HOME | APPROACH | HOLD
    active_idx     = -1
    wp_idx         = 0
    wp_step        = 0
    planner_fired  = False
    planner_thread = None
    candidates     = []
    candidate_idx  = 0

    keys       = queue.Queue()
    _ctrl_held = set()

    # ── Keyboard listener ─────────────────────────────────────────────────────
    def _on_press(key):
        try:
            if key in (_pynput_kb.Key.ctrl_l, _pynput_kb.Key.ctrl_r):
                _ctrl_held.add(key)
            elif _ctrl_held:
                char  = getattr(key, 'char', None)
                vk    = getattr(key, 'vk',  None)
                digit = None
                if char and char.isdigit():
                    digit = char
                elif vk is not None and 48 <= vk <= 57:
                    digit = str(vk - 48)
                if digit is not None:
                    keys.put(f'sel_{digit}')
        except AttributeError:
            pass

    def _on_release(key):
        if key in (_pynput_kb.Key.ctrl_l, _pynput_kb.Key.ctrl_r):
            _ctrl_held.discard(key)

    kb = _pynput_kb.Listener(on_press=_on_press, on_release=_on_release)
    kb.start()

    print("\n[Control]  Ctrl+1: obj_red_box  |  Ctrl+2: obj_red_sphere  |  "
          "Ctrl+0: home  |  Enter: next candidate  |  Q/Esc: quit\n")

    # ── Helper: select object ──────────────────────────────────────────────────
    def _select_object(idx):
        nonlocal active_idx, wp_idx, wp_step, planner_fired, candidates, candidate_idx
        active_idx    = idx
        wp_idx        = 0
        wp_step       = 0
        planner_fired = False
        candidates    = []
        candidate_idx = 0
        data.mocap_pos[cp1_mocap] = _HIDDEN
        data.mocap_pos[cp2_mocap] = _HIDDEN
        print(f"[Control] → approaching {objects[idx]['name']}")
        return 'APPROACH'

    # ── Helper: show one candidate ─────────────────────────────────────────────
    def _show_candidate(idx):
        if not candidates or idx >= len(candidates):
            return
        c  = candidates[idx]
        p1 = c.get('p1')
        p2 = c.get('p2')
        if p1 is not None:
            data.mocap_pos[cp1_mocap] = np.asarray(p1, float)
        if p2 is not None:
            data.mocap_pos[cp2_mocap] = np.asarray(p2, float)
        cost_s = f"{c['cost']:.4f}" if c.get('cost') is not None else "n/a"
        print(f"[candidate {idx+1}/{len(candidates)}]  "
              f"status={c.get('status')}  cost={cost_s}  "
              f"p_thumb={np.round(p1, 3) if p1 is not None else '?'}  "
              f"p_index={np.round(p2, 3) if p2 is not None else '?'}")

    # ── Helper: fire grasp planner in background ───────────────────────────────
    def _fire_planner(obj):
        q_snap  = np.array([data.qpos[i] for i in ms_planner._planner._act_idx])
        obj_pos = data.xpos[obj['id_body']].copy()

        def _run():
            nonlocal candidates, candidate_idx
            print(f"[planner] solving {args.nc} seeds for {obj['name']} ...")
            t0     = time.monotonic()
            result = ms_planner.solve(q_snap, obj_pos, max_seeds=args.nc)
            dt     = time.monotonic() - t0
            all_r  = result.get('all_results') or [result]
            cands  = [r for r in all_r if r.get('q') is not None]
            cands.sort(key=lambda r: (
                {'converged': 0, 'best-effort': 1, 'failed': 2}.get(
                    r.get('status', 'failed'), 2),
                r.get('cost') or 1e9))
            n_ok = sum(1 for r in cands if r.get('status') == 'converged')
            print(f"[planner] done in {dt:.1f}s — {n_ok}/{len(cands)} converged")

            # ── Verify each candidate ──────────────────────────────────────
            planner = ms_planner._planner
            log     = logging.getLogger('verify')
            for i, cand in enumerate(cands):
                info = planner.verify(cand)
                if not info:
                    continue
                _wf = (f"OK γ={info['gamma_min']:.3f}"
                       if info['wrench_feasible'] else 'INFEASIBLE')
                line = (
                    f"[verify {i+1}/{len(cands)}]"
                    f"  status={cand.get('status')}"
                    f"  IK=({info['ik_thumb_mm']:.2f},{info['ik_index_mm']:.2f})mm"
                    f"  gap=({info['gap_thumb_mm']:+.2f},{info['gap_index_mm']:+.2f},"
                    f"{info['gap_middle_mm']:+.2f},{info['gap_ring_mm']:+.2f})mm"
                    f"  sdf_p=({info['sdf_p1_mm']:.2f},{info['sdf_p2_mm']:.2f})mm"
                    f"  WF={_wf}"
                )
                print(line)
                log.info(line)

            candidates    = cands
            candidate_idx = 0
            _show_candidate(0)

        t = threading.Thread(target=_run, daemon=True, name='grasp-planner')
        t.start()
        return t

    # ── Viewer loop ───────────────────────────────────────────────────────────
    with mj.viewer.launch_passive(
            model, data, key_callback=make_key_callback(keys)) as viewer:
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        viewer.opt.label = mj.mjtLabel.mjLABEL_NONE

        running = True
        while viewer.is_running() and running:
            t_start = time.time()

            # Key events
            while not keys.empty():
                key = keys.get_nowait()

                if key == 'quit':
                    running = False

                elif key == 'enter':
                    if candidates:
                        candidate_idx = (candidate_idx + 1) % len(candidates)
                        _show_candidate(candidate_idx)
                    else:
                        print("[Control] no candidates yet — planner still running")

                elif key.startswith('sel_'):
                    digit = int(key[4:])
                    if digit == 0:
                        active_idx = -1
                        data.qpos[:N_ROBOT] = Q_BIAS
                        data.qvel[:N_ROBOT] = 0.
                        data.mocap_pos[cp1_mocap] = _HIDDEN
                        data.mocap_pos[cp2_mocap] = _HIDDEN
                        candidates = []
                        state = 'HOME'
                        print("[Control] → home")
                    elif 1 <= digit <= len(objects):
                        state = _select_object(digit - 1)

            # Sim step
            if state == 'HOME':
                data.qpos[:N_ROBOT] = Q_BIAS
                data.qvel[:N_ROBOT] = 0.

            elif state == 'APPROACH':
                obj  = objects[active_idx]
                path = obj['approach_path']

                if wp_idx < len(path):
                    data.qpos[:N_ROBOT] = path[wp_idx]
                    data.qvel[:N_ROBOT] = 0.
                    wp_step += 1
                    if wp_step >= STEPS_PER_WP:
                        wp_step  = 0
                        wp_idx  += 1

                    # Trigger planner when wrist crosses distance threshold
                    if not planner_fired and obj['use_planner']:
                        wrist_pos = data.xpos[wrist_bid].copy()
                        dist = float(np.linalg.norm(wrist_pos - obj['obj_pos']))
                        if dist < GRASP_TRIGGER_DIST:
                            planner_fired  = True
                            planner_thread = _fire_planner(obj)
                            print(f"[trigger] wrist dist={dist:.3f} m → planner started")
                else:
                    # End of path — hold at pregrasp
                    data.qpos[:N_ROBOT] = obj['q_pregrasp']
                    data.qvel[:N_ROBOT] = 0.
                    # Fallback: fire planner if threshold was never crossed
                    if not planner_fired and obj['use_planner']:
                        planner_fired  = True
                        planner_thread = _fire_planner(obj)
                        print("[trigger] at pregrasp (fallback) → planner started")
                    state = 'HOLD'

            elif state == 'HOLD':
                obj = objects[active_idx]
                data.qpos[:N_ROBOT] = obj['q_pregrasp']
                data.qvel[:N_ROBOT] = 0.
                if planner_thread is not None and not planner_thread.is_alive():
                    planner_thread = None

            mj.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0., model.opt.timestep - (time.time() - t_start)))

    kb.stop()


if __name__ == '__main__':
    main()

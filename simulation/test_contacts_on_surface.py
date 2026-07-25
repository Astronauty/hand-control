"""Test: (1) are the recommender's contacts actually ON the object surface, and (2) IK error
with a REDUCED r_tip offset (5mm instead of the auto-measured ~23.8mm tip radius).

Motivation: contacts sometimes look off-surface. This measures the exact signed distance of
each recommended contact p1/p2 to the object surface (SDF, mm) — 0 == on surface, +out, -in —
independently of the IK. It also re-runs the tip-site IK error against the target the NLP
optimizes (contact + r_tip*outward_normal) but with r_tip forced to 5mm, per request.

Note: r_tip is the physical tip-sphere radius; forcing it to 5mm makes the NLP place the tip
SITE only 5mm off the contact, so the real ~23.8mm tip sphere embeds ~19mm — this is a
DIAGNOSTIC offset, not a shippable value.

Usage:
  python simulation/test_contacts_on_surface.py [--n 20] [--seeds 3] [--rtip 0.005]
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from grasp_planner_3d import MultiStartGraspPlanner3D, _geom_normal_np, _geom_sdf_np  # noqa
import test_recommender_on_dexpilot_poses as base  # noqa: E402

N_ROBOT = base.N_ROBOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--logs', type=int, default=3)
    ap.add_argument('--rng', type=int, default=0)
    ap.add_argument('--rtip', type=float, default=0.005,
                    help='forced tip offset (m) along the outward normal; default 5mm')
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(base.SCENE_XML)
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    obj_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, base.OBJ_GEOM)
    thumb_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_th_ds_tip')
    index_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_if_ds_tip')

    cfg, active_finger_geoms = base.build_live_config(model)
    planner = MultiStartGraspPlanner3D(model, mj.MjData(model), cfg)

    # Force the tip offset AFTER construction (__init__ auto-measures r_thumb/r_index from
    # geometry, ~23.8mm; the request is to test a 5mm offset).
    _r_auto_th, _r_auto_if = planner._planner.cfg.r_thumb, planner._planner.cfg.r_index
    planner._planner.cfg.r_thumb = args.rtip
    planner._planner.cfg.r_index = args.rtip
    print(f"[rtip] auto-measured r_thumb={_r_auto_th*1e3:.1f}mm r_index={_r_auto_if*1e3:.1f}mm "
          f"-> FORCED to {args.rtip*1e3:.1f}mm for this test")

    files = base.latest_logs(None, args.logs)
    frames = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        for i in range(len(d['q_robot'])):
            frames.append((d['q_robot'][i].copy(), d['obj_qpos'][i].copy()))
    obj_qpos_len = len(frames[0][1])
    rng = np.random.default_rng(args.rng)
    idxs = rng.choice(len(frames), size=min(args.n, len(frames)), replace=False)
    print(f"[data] {len(frames)} frames; sampling {len(idxs)}; {args.seeds} seeds/solve\n")

    print(f"{'#':>3} {'solve_ms':>8} {'conv':>5} {'ik_th':>6} {'ik_if':>6} "
          f"{'sdf_p1':>7} {'sdf_p2':>7} {'wf':>3}   surface?")
    print('-' * 74)

    rows = []
    for k, fi in enumerate(idxs):
        q_ref, obj_qpos = frames[fi]
        obj_pos = obj_qpos[:3]
        planner._planner.data.qpos[:] = 0.0
        for i, adr in enumerate(act_idx):
            planner._planner.data.qpos[adr] = q_ref[i]
        planner._planner.data.qpos[N_ROBOT:N_ROBOT + obj_qpos_len] = obj_qpos
        mj.mj_forward(model, planner._planner.data)

        t0 = time.perf_counter()
        try:
            res = planner.solve(q_ref, obj_pos, max_seeds=args.seeds)
        except Exception as e:
            print(f"{k:>3} solve raised: {e}")
            rows.append(dict(conv=False))
            continue
        solve_ms = (time.perf_counter() - t0) * 1e3
        if res.get('p1') is None or res.get('p2') is None:
            print(f"{k:>3} {solve_ms:>8.0f} {'NO':>5}")
            rows.append(dict(conv=False))
            continue

        p1 = np.asarray(res['p1'], float)
        p2 = np.asarray(res['p2'], float)
        q_nlp = np.asarray(res['q'], float)

        # Object geometry at the synced planner pose.
        obj_c = planner._planner.data.geom_xpos[obj_gid].copy()
        obj_R = planner._planner.data.geom_xmat[obj_gid].reshape(3, 3).copy()
        obj_sz = model.geom_size[obj_gid]
        obj_gt = int(model.geom_type[obj_gid])

        # (1) CONTACT-ON-SURFACE test: signed distance of each contact to the object surface.
        sdf_p1 = float(_geom_sdf_np(p1, obj_gt, obj_c, obj_R, obj_sz)) * 1e3
        sdf_p2 = float(_geom_sdf_np(p2, obj_gt, obj_c, obj_R, obj_sz)) * 1e3

        # (2) IK error vs the offset target the NLP optimizes (now with forced r_tip).
        n1 = _geom_normal_np(p1, obj_gt, obj_c, obj_R, obj_sz)
        n2 = _geom_normal_np(p2, obj_gt, obj_c, obj_R, obj_sz)
        dchk = mj.MjData(model)
        for i, adr in enumerate(act_idx):
            dchk.qpos[adr] = q_nlp[i]
        dchk.qpos[N_ROBOT:N_ROBOT + obj_qpos_len] = obj_qpos
        mj.mj_forward(model, dchk)
        ik_th = float(np.linalg.norm(dchk.site_xpos[thumb_sid] - (p1 + args.rtip * n1)) * 1e3)
        ik_if = float(np.linalg.norm(dchk.site_xpos[index_sid] - (p2 + args.rtip * n2)) * 1e3)

        try:
            wf = bool((planner._planner.verify(res) or {}).get('wrench_feasible', False))
        except Exception:
            wf = False

        on_surf = abs(sdf_p1) < 0.5 and abs(sdf_p2) < 0.5   # within 0.5mm of surface
        tag = 'on-surf' if on_surf else '** OFF **'
        print(f"{k:>3} {solve_ms:>8.0f} {'YES':>5} {ik_th:>6.1f} {ik_if:>6.1f} "
              f"{sdf_p1:>7.2f} {sdf_p2:>7.2f} {('Y' if wf else 'n'):>3}   {tag}")
        rows.append(dict(conv=True, solve_ms=solve_ms, ik_th=ik_th, ik_if=ik_if,
                         sdf_p1=sdf_p1, sdf_p2=sdf_p2, wf=wf, on_surf=on_surf))

    conv = [r for r in rows if r.get('conv')]
    print()
    print("=" * 60)
    print(f"SUMMARY over {len(rows)} poses  (r_tip offset = {args.rtip*1e3:.1f}mm)")
    print(f"  converged                   : {len(conv)}/{len(rows)}")
    if conv:
        def st(key, absv=False):
            v = np.array([abs(r[key]) if absv else r[key] for r in conv], float)
            return v.mean(), np.median(v), v.min(), v.max()
        for label, key, absv in [('IK err thumb', 'ik_th', False),
                                 ('IK err index', 'ik_if', False),
                                 ('|SDF p1| (contact)', 'sdf_p1', True),
                                 ('|SDF p2| (contact)', 'sdf_p2', True)]:
            m, md, lo, hi = st(key, absv)
            print(f"  {label:<22}: mean={m:7.2f} median={md:7.2f} "
                  f"min={lo:7.2f} max={hi:7.2f} mm")
        n_surf = sum(1 for r in conv if r['on_surf'])
        print(f"  contacts ON surface (<0.5mm): {n_surf}/{len(conv)}")
        print(f"  wrench-feasible             : {sum(r['wf'] for r in conv)}/{len(conv)}")


if __name__ == '__main__':
    main()

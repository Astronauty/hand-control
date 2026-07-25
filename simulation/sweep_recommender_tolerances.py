"""Sweep recommender solver tolerances + cost weights against a FIXED set of real dexpilot
poses, to see whether the systematic ~24mm IK-error plateau (0/20 reachable in the baseline
benchmark) can be improved.

Reuses the pose sampling / metrics from test_recommender_on_dexpilot_poses.py so every config
is evaluated on the IDENTICAL poses (same rng seed). For each config it reports median IK
error, median solve time, %reachable (<15mm both tips), median link penetration, and %wrench-
feasible.

Two axes:
  A) IPOPT flat-objective acceptance (module global _IPOPT_SOLVER_OPTS): the baseline accepts
     as soon as the objective changes <1% for 4 iters (acceptable_obj_change_tol=1e-2), with
     the tight KKT test (acceptable_tol / dual_inf) effectively disabled. That weak exit was a
     workaround for the wrench-NLP dual degeneracy, which no longer exists in the IK-only live
     recommender. We tighten it so the solver keeps working while IK error is still large.
  B) Cost weights on GraspConfig3D: w_ik (reachability) vs w_align (antipodal). Baseline is
     w_ik=0.70, w_align=10.0 — alignment out-weighs reachability ~14:1, which may be trading
     tip accuracy for alignment.

Usage:
  python simulation/sweep_recommender_tolerances.py [--n 20] [--seeds 3]
"""
import argparse
import os
import sys
import time
import copy

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import grasp_planner_3d as gp3
from grasp_planner_3d import MultiStartGraspPlanner3D  # noqa: E402
import test_recommender_on_dexpilot_poses as base  # noqa: E402

SCENE_XML = base.SCENE_XML
N_ROBOT = base.N_ROBOT


# ── Config axes ────────────────────────────────────────────────────────────────
# Each IPOPT config is a patch dict merged over the baseline _IPOPT_SOLVER_OPTS.
IPOPT_CONFIGS = {
    'baseline':         {},   # as-is: acceptable_obj_change_tol=1e-2, acceptable_iter=4
    'tol_1e-3':         {'acceptable_obj_change_tol': 1e-3},
    'tol_1e-4':         {'acceptable_obj_change_tol': 1e-4},
    'tol_1e-4_it8':     {'acceptable_obj_change_tol': 1e-4, 'acceptable_iter': 8},
    # Re-enable the TIGHT KKT exit (dual-inf) that was disabled for the wrench NLP — the
    # IK-only recommender has no dual degeneracy, so real KKT convergence should be reachable.
    'kkt_on':           {'acceptable_tol': 1e-4, 'acceptable_dual_inf_tol': 1e-2,
                         'acceptable_obj_change_tol': 1e-6, 'acceptable_iter': 8},
}

# Each weight config is (w_ik, w_align). Baseline is (0.70, 10.0).
WEIGHT_CONFIGS = {
    'w_base_ik0.7_al10':  (0.70, 10.0),
    'w_ik2_al10':         (2.00, 10.0),
    'w_ik5_al10':         (5.00, 10.0),
    'w_ik2_al2':          (2.00,  2.0),
    'w_ik5_al2':          (5.00,  2.0),
}


def eval_config(model, planner, frames, idxs, act_idx, obj_gid,
                thumb_sid, index_sid, active_finger_geoms, obj_qpos_len, seeds):
    """Run the recommender over the fixed sampled poses; return aggregate metrics."""
    rows = []
    for fi in idxs:
        q_ref, obj_qpos = frames[fi]
        obj_pos = obj_qpos[:3]
        # Sync planner scene to this frame.
        planner._planner.data.qpos[:] = 0.0
        for i, adr in enumerate(act_idx):
            planner._planner.data.qpos[adr] = q_ref[i]
        planner._planner.data.qpos[N_ROBOT:N_ROBOT + obj_qpos_len] = obj_qpos
        mj.mj_forward(model, planner._planner.data)

        t0 = time.perf_counter()
        try:
            res = planner.solve(q_ref, obj_pos, max_seeds=seeds)
        except Exception:
            rows.append(dict(conv=False))
            continue
        solve_ms = (time.perf_counter() - t0) * 1e3
        if res.get('p1') is None or res.get('p2') is None:
            rows.append(dict(conv=False, solve_ms=solve_ms))
            continue

        p1 = np.asarray(res['p1'], float)
        p2 = np.asarray(res['p2'], float)
        q_nlp = np.asarray(res['q'], float)
        dchk = mj.MjData(model)
        for i, adr in enumerate(act_idx):
            dchk.qpos[adr] = q_nlp[i]
        dchk.qpos[N_ROBOT:N_ROBOT + obj_qpos_len] = obj_qpos
        mj.mj_forward(model, dchk)
        ik_th = float(np.linalg.norm(dchk.site_xpos[thumb_sid] - p1) * 1e3)
        ik_if = float(np.linalg.norm(dchk.site_xpos[index_sid] - p2) * 1e3)
        _, con_d, _, _ = base.audit_penetration(
            model, q_nlp, obj_qpos, obj_gid, active_finger_geoms, act_idx)
        try:
            wf = bool((planner._planner.verify(res) or {}).get('wrench_feasible', False))
        except Exception:
            wf = False
        rows.append(dict(conv=True, solve_ms=solve_ms, ik_th=ik_th, ik_if=ik_if,
                         pen_mm=con_d * 1e3, wf=wf,
                         reach=max(ik_th, ik_if) < 15.0))
    conv = [r for r in rows if r.get('conv')]
    if not conv:
        return dict(n=len(rows), conv=0)
    ik_max = np.array([max(r['ik_th'], r['ik_if']) for r in conv])
    return dict(
        n=len(rows), conv=len(conv),
        ik_med=float(np.median(ik_max)), ik_mean=float(ik_max.mean()),
        ik_min=float(ik_max.min()),
        solve_med=float(np.median([r['solve_ms'] for r in conv])),
        reach=sum(r['reach'] for r in conv),
        pen_med=float(np.median([r['pen_mm'] for r in conv])),
        wf=sum(r['wf'] for r in conv),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--logs', type=int, default=3)
    ap.add_argument('--rng', type=int, default=0)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(SCENE_XML)
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    obj_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, base.OBJ_GEOM)
    thumb_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_th_ds_tip')
    index_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_if_ds_tip')

    cfg0, active_finger_geoms = base.build_live_config(model)

    # Fixed pose sample (identical across all configs).
    files = base.latest_logs(None, args.logs)
    frames = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        for i in range(len(d['q_robot'])):
            frames.append((d['q_robot'][i].copy(), d['obj_qpos'][i].copy()))
    obj_qpos_len = len(frames[0][1])
    rng = np.random.default_rng(args.rng)
    idxs = rng.choice(len(frames), size=min(args.n, len(frames)), replace=False)
    print(f"[sweep] {len(frames)} frames; fixed sample of {len(idxs)} poses; "
          f"{args.seeds} seeds/solve")
    print(f"[sweep] {len(IPOPT_CONFIGS)} IPOPT × {len(WEIGHT_CONFIGS)} weight configs "
          f"= {len(IPOPT_CONFIGS)*len(WEIGHT_CONFIGS)} runs\n")

    baseline_ipopt = copy.deepcopy(gp3._IPOPT_SOLVER_OPTS)

    hdr = (f"{'ipopt_cfg':<14} {'weight_cfg':<20} {'conv':>5} {'ik_med':>7} "
           f"{'ik_min':>7} {'reach':>6} {'pen_med':>8} {'wf':>5} {'solve_med':>10}")
    print(hdr)
    print('-' * len(hdr))

    results = []
    for iname, ipatch in IPOPT_CONFIGS.items():
        # Patch the module-global IPOPT opts (re-read as dict() per solve).
        gp3._IPOPT_SOLVER_OPTS = copy.deepcopy(baseline_ipopt)
        gp3._IPOPT_SOLVER_OPTS.update(ipatch)
        for wname, (w_ik, w_align) in WEIGHT_CONFIGS.items():
            # Fresh planner per config so cached seeds/warm-starts don't leak.
            cfg = copy.copy(cfg0)
            cfg.w_ik = w_ik
            cfg.w_align = w_align
            planner = MultiStartGraspPlanner3D(model, mj.MjData(model), cfg)
            m = eval_config(model, planner, frames, idxs, act_idx, obj_gid,
                            thumb_sid, index_sid, active_finger_geoms, obj_qpos_len,
                            args.seeds)
            if m.get('conv', 0) == 0:
                print(f"{iname:<14} {wname:<20} {'0/'+str(m['n']):>5}  (no convergence)")
                continue
            print(f"{iname:<14} {wname:<20} {m['conv']:>2}/{m['n']:<2} "
                  f"{m['ik_med']:>7.1f} {m['ik_min']:>7.1f} "
                  f"{m['reach']:>2}/{m['conv']:<3} {m['pen_med']:>8.1f} "
                  f"{m['wf']:>2}/{m['conv']:<2} {m['solve_med']:>9.0f}ms")
            results.append((iname, wname, m))

    # Restore global.
    gp3._IPOPT_SOLVER_OPTS = baseline_ipopt

    # Best by median IK error, then by %reachable.
    if results:
        best = min(results, key=lambda r: (r[2]['ik_med'], -r[2]['reach']))
        print()
        print(f"BEST (min median IK err): ipopt={best[0]}  weights={best[1]}  "
              f"ik_med={best[2]['ik_med']:.1f}mm  reach={best[2]['reach']}/{best[2]['conv']}  "
              f"solve_med={best[2]['solve_med']:.0f}ms")


if __name__ == '__main__':
    main()

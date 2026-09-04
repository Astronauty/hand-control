"""Sweep NLP cost weights / tolerances / collision constraints against the scene's DEFINED
KEYFRAMES (pose_00..pose_09 in models/scene_pick_place.xml), measuring BOTH:

  - SEED convergence  : fraction of the per-solve seeds whose NLP reached status 'converged'
                        (from res['all_results']). Low => the seed geometry / face-pin is hard
                        for the NLP from most starts (the multi-start is carrying the solve).
  - NLP  convergence  : fraction of keyframes whose BEST result is 'converged' (vs best-effort
                        / failed / non-converged p1==None). This is what the live recommender
                        actually ships.
  - wrench-feasible   : fraction of converged keyframes whose verify() datum-LP is feasible
                        (the gate the live recommender uses to accept a candidate).
  - IK error / solve  : median max-tip IK residual (mm) and median wall solve time (ms).

The recommender config is built by test_recommender_on_dexpilot_poses.build_live_config — the
documented single source of truth that reproduces the AUTONOMOUS framework's _get_cat_planner
GraspConfig3D exactly — so each swept config differs from the live recommender only in the
knobs under test.

Every config runs on the IDENTICAL 10 keyframes (deterministic: fixed-RNG seeds in
MultiStartGraspPlanner3D), so differences are the config, not sampling.

Usage:
  python simulation/sweep_keyframe_convergence.py [--seeds 5] [--group weights|tol|collision|all]
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import grasp_planner_3d as gp3  # noqa: E402
from grasp_planner_3d import MultiStartGraspPlanner3D  # noqa: E402
import test_recommender_on_dexpilot_poses as base  # noqa: E402

SCENE_XML = base.SCENE_XML
N_ROBOT = base.N_ROBOT
# The app's REAL disturbance budget (kinova_leap_pick_place.py NCF_*). The test module's
# default ang budget is 0.5; the live recommender uses 1.0, so pin it here to stay faithful.
ACCEL_BUDGET = (20.0, 20.0, 20.0)
ANG_BUDGET = (1.0, 1.0, 1.0)


# ── Config axes ────────────────────────────────────────────────────────────────
# Each entry is (name, cfg_patch) where cfg_patch is a dict of GraspConfig3D attrs merged
# over the live baseline. Baseline (name 'baseline') applies no patch. Grouped so a run can
# focus on one lever at a time.
#
# Live baseline (build_live_config): w_ik=0.70, w_reg=0.03, w_align=10.0, orient_weight=2.0,
# edge_margin_m=0.015 (build_live_config default; app uses 0.03), col_clearance_m=0.005,
# ground_clearance_m=0.010, seed_march_jitter_deg=15, n_seeds=5, max_iter=120,
# wrench_constraint=False, datum_gamma=True.

WEIGHT_CONFIGS = [
    ('baseline',              {}),
    # Reachability vs alignment balance. Baseline alignment out-weighs IK ~14:1.
    ('w_ik2',                 {'w_ik': 2.00}),
    ('w_ik5',                 {'w_ik': 5.00}),
    ('w_align5',              {'w_align': 5.0}),
    ('w_align20',             {'w_align': 20.0}),
    ('w_ik2_align5',          {'w_ik': 2.00, 'w_align': 5.0}),
    # Pad-orientation cost — a q-dependent rotational FK term; heavier can hurt convergence.
    ('orient0',               {'orient_weight': 0.0}),
    ('orient4',               {'orient_weight': 4.0}),
    # Posture regulariser.
    ('w_reg0.1',              {'w_reg': 0.10}),
]

TOL_CONFIGS = [
    ('baseline',              {}),
    # Seed march jitter: smaller keeps contact 2 on the OPPOSING face more often (fewer
    # adjacent-face seeds that converge but fail wrench). Larger explores more.
    ('jitter5',               {'seed_march_jitter_deg': 5.0}),
    ('jitter25',              {'seed_march_jitter_deg': 25.0}),
    # Seed budget: more seeds -> higher best-of-N convergence at linear time cost.
    ('seeds8',                {'n_seeds': 8}),
    # NLP per-stage iteration cap.
    ('iter200',               {'max_iter': 200}),
    ('iter60',                {'max_iter': 60}),
    # Picard relinearisations of the (frozen) contact normals.
    ('relin2',                {'n_normal_relinearize': 2}),
    ('relin0',                {'n_normal_relinearize': 0}),
]

COLLISION_CONFIGS = [
    ('baseline',              {}),
    # Object clearance for the constrained finger links (bounding-sphere surface). More
    # negative = looser (fewer active constraints -> easier NLP); less negative = tighter.
    ('col_clear0',            {'col_clearance_m': 0.000}),
    ('col_clear10',           {'col_clearance_m': 0.010}),
    # Edge keep-out band. Baseline app uses 0.03; wider shrinks the usable face (harder),
    # narrower frees the rim (easier but risks near-edge slip).
    ('edge15',                {'edge_margin_m': 0.015}),
    ('edge45',                {'edge_margin_m': 0.045}),
    # Ground clearance for the curled non-active fingers.
    ('ground5',               {'ground_clearance_m': 0.005}),
    ('ground20',              {'ground_clearance_m': 0.020}),
    # Proximity-prune margin: how far a geom must be to be dropped from the SDF loop. Smaller
    # prunes more aggressively (fewer active constraints -> easier), larger keeps more.
    ('prune5cm',              {'col_prune_margin': 0.05}),
]

GROUPS = {
    'weights':   WEIGHT_CONFIGS,
    'tol':       TOL_CONFIGS,
    'collision': COLLISION_CONFIGS,
}


def load_keyframes(model):
    """Return [(name, q_ref(23,), obj_qpos(7,)), ...] for every scene <keyframe>."""
    frames = []
    d = mj.MjData(model)
    for k in range(model.nkey):
        mj.mj_resetDataKeyframe(model, d, k)
        mj.mj_forward(model, d)
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_KEY, k) or f'key_{k}'
        q_ref = d.qpos[:N_ROBOT].copy()
        obj_qpos = d.qpos[N_ROBOT:].copy()
        frames.append((name, q_ref, obj_qpos))
    return frames


def eval_config(model, cfg, frames, act_idx, obj_gid, thumb_sid, index_sid,
                active_finger_geoms, seeds):
    """Run the recommender on every keyframe; return per-keyframe rows + aggregate."""
    planner = MultiStartGraspPlanner3D(model, mj.MjData(model), cfg)
    obj_qpos_len = len(frames[0][2])
    rows = []
    for name, q_ref, obj_qpos in frames:
        # Pose the planner's scene to this keyframe (robot + object).
        planner._planner.data.qpos[:] = 0.0
        for i, adr in enumerate(act_idx):
            planner._planner.data.qpos[adr] = q_ref[i]
        planner._planner.data.qpos[N_ROBOT:N_ROBOT + obj_qpos_len] = obj_qpos
        mj.mj_forward(model, planner._planner.data)
        obj_pos = obj_qpos[:3]

        t0 = time.perf_counter()
        try:
            res = planner.solve(q_ref, obj_pos, max_seeds=seeds)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(name=name, nlp_conv=False, err=str(e)[:60]))
            continue
        solve_ms = (time.perf_counter() - t0) * 1e3

        allr = res.get('all_results') or ([res] if res.get('status') else [])
        n_seed = len(allr)
        n_seed_conv = sum(1 for r in allr if r.get('status') == 'converged')
        best_status = res.get('status', 'none')
        nlp_conv = (best_status == 'converged' and res.get('p1') is not None)

        row = dict(name=name, solve_ms=solve_ms, n_seed=n_seed,
                   n_seed_conv=n_seed_conv, best_status=best_status,
                   nlp_conv=nlp_conv)

        if res.get('p1') is not None and res.get('p2') is not None:
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
            row['ik_max'] = max(ik_th, ik_if)
            try:
                row['wf'] = bool((planner._planner.verify(res) or {}).get(
                    'wrench_feasible', False))
            except Exception:  # noqa: BLE001
                row['wf'] = False
            # Opposition angle between the two final OUTWARD contact normals (n*_final are
            # outward). Opposite faces -> normals point opposite -> ~180deg = perfectly
            # antipodal; adjacent faces -> ~90deg (the rotated-box failure mode).
            try:
                _n1 = res.get('n1_final')
                _n2 = res.get('n2_final')
                if _n1 is not None and _n2 is not None:
                    n1 = np.asarray(_n1, float)
                    n2 = np.asarray(_n2, float)
                    c = float(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
                    row['oppo_deg'] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
            except Exception:  # noqa: BLE001
                pass
        rows.append(row)

    n = len(rows)
    nlp_conv = [r for r in rows if r.get('nlp_conv')]
    # Seed convergence across ALL keyframes (sum converged seeds / sum seeds attempted).
    tot_seed = sum(r.get('n_seed', 0) for r in rows)
    tot_seed_conv = sum(r.get('n_seed_conv', 0) for r in rows)
    agg = dict(
        n=n,
        nlp_conv=len(nlp_conv),
        seed_rate=(tot_seed_conv / tot_seed if tot_seed else 0.0),
        tot_seed=tot_seed, tot_seed_conv=tot_seed_conv,
        wf=sum(1 for r in nlp_conv if r.get('wf')),
        solve_med=float(np.median([r['solve_ms'] for r in rows if 'solve_ms' in r]))
                  if any('solve_ms' in r for r in rows) else float('nan'),
    )
    if nlp_conv:
        ikm = [r['ik_max'] for r in nlp_conv if 'ik_max' in r]
        opp = [r['oppo_deg'] for r in nlp_conv if 'oppo_deg' in r]
        agg['ik_med'] = float(np.median(ikm)) if ikm else float('nan')
        agg['oppo_med'] = float(np.median(opp)) if opp else float('nan')
    else:
        agg['ik_med'] = float('nan')
        agg['oppo_med'] = float('nan')
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5, help='seeds per solve (live default 5)')
    ap.add_argument('--group', choices=['weights', 'tol', 'collision', 'all'],
                    default='all')
    ap.add_argument('--verbose', action='store_true',
                    help='print a per-keyframe row for every config')
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(SCENE_XML)
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    obj_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, base.OBJ_GEOM)
    thumb_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_th_ds_tip')
    index_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_if_ds_tip')

    cfg0, active_finger_geoms = base.build_live_config(
        model, accel_budget=ACCEL_BUDGET, ang_budget=ANG_BUDGET)

    frames = load_keyframes(model)
    print(f"[sweep] {len(frames)} keyframes: {', '.join(f[0] for f in frames)}")
    print(f"[sweep] {args.seeds} seeds/solve | budget accel={ACCEL_BUDGET} "
          f"ang={ANG_BUDGET}\n")

    groups = (GROUPS if args.group == 'all' else {args.group: GROUPS[args.group]})

    hdr = (f"{'group':<10} {'config':<16} {'nlp_conv':>9} {'seed_rate':>10} "
           f"{'wf':>7} {'ik_med':>7} {'oppo':>6} {'solve':>8}")

    all_results = []
    for gname, configs in groups.items():
        print(f"\n===== GROUP: {gname} =====")
        print(hdr)
        print('-' * len(hdr))
        for cname, patch in configs:
            cfg = copy.copy(cfg0)
            for k, v in patch.items():
                setattr(cfg, k, v)
            agg, rows = eval_config(model, cfg, frames, act_idx, obj_gid,
                                    thumb_sid, index_sid, active_finger_geoms,
                                    args.seeds)
            print(f"{gname:<10} {cname:<16} "
                  f"{agg['nlp_conv']:>2}/{agg['n']:<6} "
                  f"{agg['seed_rate']*100:>8.0f}%  "
                  f"{agg['wf']:>2}/{agg['nlp_conv']:<4} "
                  f"{agg['ik_med']:>7.1f} {agg['oppo_med']:>5.0f}° "
                  f"{agg['solve_med']:>6.0f}ms")
            if args.verbose:
                for r in rows:
                    print(f"      {r['name']:<9} conv={r.get('nlp_conv')} "
                          f"seed={r.get('n_seed_conv')}/{r.get('n_seed')} "
                          f"status={r.get('best_status')} "
                          f"ik={r.get('ik_max', float('nan')):.1f} "
                          f"wf={r.get('wf')} oppo={r.get('oppo_deg', float('nan')):.0f}")
            all_results.append((gname, cname, agg))

    # Recommend: rank by (nlp_conv desc, wf desc, ik_med asc) among non-baseline configs,
    # compared to baseline.
    print("\n\n===== RECOMMENDATION =====")
    base_rows = {g: a for g, c, a in all_results if c == 'baseline'}
    for gname in groups:
        gr = [(c, a) for g, c, a in all_results if g == gname]
        b = base_rows.get(gname)
        if b is None:
            continue
        # Score: convergence first, then wrench feasibility, then IK error (lower better).
        def score(a):
            return (a['nlp_conv'], a['wf'], -a['ik_med'] if a['ik_med'] == a['ik_med'] else -1e9)
        best_c, best_a = max(gr, key=lambda ca: score(ca[1]))
        improved = (score(best_a) > score(b)) and best_c != 'baseline'
        tag = 'IMPROVES' if improved else 'no gain over baseline'
        print(f"[{gname:<9}] baseline: nlp={b['nlp_conv']}/{b['n']} "
              f"wf={b['wf']}/{b['nlp_conv']} ik={b['ik_med']:.1f}  |  "
              f"best: '{best_c}' nlp={best_a['nlp_conv']}/{best_a['n']} "
              f"wf={best_a['wf']}/{best_a['nlp_conv']} ik={best_a['ik_med']:.1f}  -> {tag}")


if __name__ == '__main__':
    main()

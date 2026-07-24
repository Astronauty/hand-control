"""LP ground-truth + NLP weight sweep for the fixed-contact wrench problem.

Builds on test_fixed_contacts_wrench.py's finding: with contacts pinned at the known-
good grasp sites (obj_red_box_c1/c2) and a real near-box q_seed, the NLP still fails to
converge when the wrench cone is ON, while it converges with it OFF. Two questions:

  1. Is the pinned grasp actually wrench-FEASIBLE? Run the standalone LP
     (min_gamma_for_accel_lp, the SAME slack-penalty LP the NLP embeds) directly on the
     pinned site geometry. A finite gamma_min = feasible ground truth; the NLP's failure
     is then purely a convergence/formulation problem, not an infeasible task.

  2. Do the NON-gamma objective weights block convergence? The objective is
        w_ik·ik + w_reg·reg + w_gamma·gamma + w_y·y (+ w_slack·slack)
     Scale down w_ik/w_reg/w_y (everything NOT gamma) and see whether the NLP then
     converges toward the LP's gamma. If it does, the non-gamma terms were fighting the
     wrench constraint (a weighting problem); if it still fails at ~zero non-gamma
     weight, the wrench-cone machinery itself is malformed.

Usage:
    python simulation/test_lp_vs_nlp_weights.py --sample 0
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'scripts'))

from grasp_planner_3d import (GraspConfig3D, GraspPlanner3D,  # noqa: E402
                              _geom_normal_np, _build_contact_frame_3d)

_ncf = importlib.import_module('3D_minimum_NCF_slack')
min_gamma_for_accel_lp = _ncf.min_gamma_for_accel_lp

_SCENE = os.path.join(os.path.dirname(_HERE), 'models', 'scene_pick_place.xml')


def _load_sample_scene(model, data, obj_body, sample_idx):
    """Pose the arm at a recorded near-box q_seed and relocate obj_body to its obj_qpos.
    Returns q_ref (actuated). None if sample_idx is None (uses home pose as-is)."""
    if sample_idx is None:
        return None
    import json
    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    s = samples[sample_idx]
    q_ref = np.asarray(s['q_seed'], float)
    oq = np.asarray(s['obj_qpos'], float)
    jadr = model.jnt_qposadr[model.body_jntadr[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body)]]
    data.qpos[jadr:jadr + 7] = oq[:7]
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    for i, idx in enumerate(act_idx):
        data.qpos[idx] = q_ref[i]
    mj.mj_forward(model, data)
    print(f"[sample {sample_idx}] recorded q_seed + obj_qpos "
          f"(orig object {s.get('object')}, relocated {obj_body})")
    return q_ref


def _lp_gamma_at_sites(model, data, cfg, obj_body, p1, p2):
    """Run the standalone slack-penalty LP on the pinned-site geometry — the exact math
    verify() uses, but on the given contacts. Returns (gamma_min, max_slack_norm)."""
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body)
    ogid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, obj_body + '_geom')
    obj_pos = data.geom_xpos[ogid].copy()
    obj_mat = data.geom_xmat[ogid].reshape(3, 3)
    gsize = model.geom_size[ogid]
    gtype = model.geom_type[ogid]

    n1_out = _geom_normal_np(p1, gtype, obj_pos, obj_mat, gsize)
    n2_out = _geom_normal_np(p2, gtype, obj_pos, obj_mat, gsize)
    _, t1a, t2a = _build_contact_frame_3d(-n1_out)
    _, t1b, t2b = _build_contact_frame_3d(-n2_out)
    R1 = np.column_stack([-n1_out, t1a, t2a])
    R2 = np.column_stack([-n2_out, t1b, t2b])

    mass  = float(model.body_mass[bid])
    inert = model.body_inertia[bid]
    aab   = cfg.ang_accel_budget_xyz
    mu    = round(0.8 * float(model.geom_friction[ogid][0]), 3)
    R_WO  = data.xmat[bid].reshape(3, 3)
    g_O   = R_WO.T @ model.opt.gravity
    ab    = cfg.accel_budget_xyz
    accel = tuple(ab[i] + abs(g_O[i]) for i in range(3))
    p1_O  = R_WO.T @ (p1 - obj_pos)
    p2_O  = R_WO.T @ (p2 - obj_pos)
    R1_O  = R_WO.T @ R1
    R2_O  = R_WO.T @ R2
    return min_gamma_for_accel_lp(
        mass * accel[0], mass * accel[1], mass * accel[2],
        float(inert[0]) * aab[0], float(inert[1]) * aab[1], float(inert[2]) * aab[2],
        n=2, pos=[p1_O.reshape(3, 1), p2_O.reshape(3, 1)], R=[R1_O, R2_O],
        ncf=[1.0, 1.0], tan_y=[0.0, 0.0], tan_z=[0.0, 0.0], mu=[mu, mu],
        slack_penalty=cfg.verify_slack_penalty)


def _site_world(model, data, name):
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[sid].copy() if sid >= 0 else None


def _outward_face_dir(model, data, obj_body, p_world):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body)
    R = data.xmat[bid].reshape(3, 3)
    c = data.xpos[bid]
    p_loc = R.T @ (p_world - c)
    ax = int(np.argmax(np.abs(p_loc)))
    d = np.zeros(3); d[ax] = np.sign(p_loc[ax])
    return R @ d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--object', default='obj_red_box')
    ap.add_argument('--sample', type=int, default=0,
                    help="samples.jsonl index for a near-box q_seed (default 0)")
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--max-iter', type=int, default=120)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(_SCENE)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    obj = args.object

    q_ref = _load_sample_scene(model, data, obj, args.sample)
    p1 = _site_world(model, data, obj + '_c1')
    p2 = _site_world(model, data, obj + '_c2')
    d1 = _outward_face_dir(model, data, obj, p1)
    d2 = _outward_face_dir(model, data, obj, p2)
    if q_ref is None:
        act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
        q_ref = np.array([data.qpos[i] for i in act_idx])

    # ── 1. LP ground truth on the pinned sites ─────────────────────────────
    cfg0 = GraspConfig3D(obj_geom=obj + '_geom', obj_body=obj)
    lp_g, lp_slack = _lp_gamma_at_sites(model, data, cfg0, obj, p1, p2)
    print("\n" + "=" * 74)
    print(f"LP ground truth (min_gamma_for_accel_lp, slack-penalty) at pinned sites:")
    if lp_g is None:
        print("  INFEASIBLE — the LP says no gamma resists the task wrench at these")
        print("  contacts. The NLP is then CORRECT to not certify feasibility.")
    else:
        print(f"  FEASIBLE — gamma_min = {lp_g:.4f}   max_slack_norm = "
              f"{lp_slack if lp_slack is None else round(lp_slack, 5)}")
        print("  => the grasp IS wrench-feasible; any NLP non-convergence is formulation.")
    print("=" * 74)

    # ── 2. NLP weight sweep: scale the NON-gamma weights down ───────────────
    base = dict(w_ik=0.80, w_reg=0.06, w_gamma=0.10, w_y=0.04)
    scales = [1.0, 0.1, 0.01, 0.0]   # multiply w_ik/w_reg/w_y by this; w_gamma fixed
    print(f"\nNLP wrench-ON, contacts pinned, non-gamma weights scaled (w_gamma fixed "
          f"at {base['w_gamma']}):")
    print(f"{'scale(ik,reg,y)':<16} {'conv':>7} {'gamma_nlp':>10} {'best_cost':>10} "
          f"{'iters':>7} {'ms':>7}")
    print("-" * 74)
    for sc in scales:
        cfg = GraspConfig3D(
            obj_geom=obj + '_geom', obj_body=obj, max_iter=args.max_iter,
            wrench_constraint=True,
            w_ik=base['w_ik'] * sc, w_reg=base['w_reg'] * sc,
            w_gamma=base['w_gamma'], w_y=base['w_y'] * sc,
        )
        planner = GraspPlanner3D(model, mj.MjData(model), cfg)
        n_conv, gs, cs, its, ms = 0, [], [], [], []
        for _ in range(args.repeats):
            planner.data.qpos[:] = data.qpos[:]
            mj.mj_forward(model, planner.data)
            t0 = time.time()
            res = planner.solve(q_ref, data.xpos[
                mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj)].copy(),
                p1_init=p1, p2_init=p2, d1=d1, d2=d2)
            ms.append((time.time() - t0) * 1e3)
            if res.get('status') == 'converged':
                n_conv += 1
            if res.get('gamma_nlp') is not None:
                gs.append(res['gamma_nlp'])
            if res.get('cost') is not None:
                cs.append(res['cost'])
            if res.get('iterations') is not None:
                its.append(res['iterations'])
        g_s = f"{np.mean(gs):.3f}" if gs else '-'
        c_s = f"{np.mean(cs):.3g}" if cs else '-'
        it_s = f"{np.mean(its):.0f}" if its else '-'
        print(f"{sc:<16} {n_conv}/{args.repeats:>5} {g_s:>10} {c_s:>10} "
              f"{it_s:>7} {np.mean(ms):>7.0f}")
    print("=" * 74)
    if lp_g is not None:
        print(f"Target: gamma_nlp should approach the LP's {lp_g:.4f} as convergence")
        print("improves. If conv climbs as non-gamma weights -> 0, those terms were")
        print("fighting the wrench constraint. If it stays 0/N, the cone is malformed.\n")


if __name__ == '__main__':
    main()

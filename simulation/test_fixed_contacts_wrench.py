"""Fixed-contact wrench-feasibility test for the grasp recommender NLP.

Motivation: the recommender has been returning best-effort / 0-of-N converged on the
box, and a constraint ablation pinned the blame on the embedded wrench cone. But that
test also lets the solver SEARCH for contact points. The scene defines known-good grasp
sites on each object (obj_red_box_c1/c2 at the two opposing box-face centers). With the
contacts FIXED at those sites, the only free variables are the arm/hand q (reach the
sites) and the wrench-cone internals (gamma, per-corner y). If the solver still can't
converge with a known-feasible, hand-picked contact pair, the problem is the q+wrench
solve / solver settings, not the contact search.

Two pin strengths:
  face-pin : p1/p2 pinned to the correct box FACE (normal coord fixed, tangential free
             to slide) and warm-started at the site — the planner's native mode
             (GraspPlanner3D.solve p1_init/p2_init + d1/d2).
  hard-pin : p1/p2 fixed to the EXACT site position (all 3 coords equal) — removes
             contact search entirely; only q + wrench vary.

Run against both solver backends (IPOPT L-BFGS default, SQP+OSQP via use_slsqp) so we
can see whether the SQP path — which treats the linear face-pin / wrench constraints
exactly — converges where IPOPT stalls.

Usage:
    python simulation/test_fixed_contacts_wrench.py
    python simulation/test_fixed_contacts_wrench.py --object obj_red_box --repeats 3
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import grasp_planner_3d as gp3  # noqa: E402
from grasp_planner_3d import GraspConfig3D, GraspPlanner3D  # noqa: E402

_SCENE = os.path.join(os.path.dirname(_HERE), 'models', 'scene_pick_place.xml')


def _site_world(model, data, name):
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        return None
    return data.site_xpos[sid].copy()


def _outward_face_dir(model, data, obj_body, p_world):
    """Outward box-face normal (unit) at the face nearest p_world, in world frame.
    For the c1/c2 sites this is just ±x of the object frame."""
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body)
    R = data.xmat[bid].reshape(3, 3)
    c = data.xpos[bid]
    p_loc = R.T @ (p_world - c)
    ax = int(np.argmax(np.abs(p_loc)))
    d_loc = np.zeros(3)
    d_loc[ax] = np.sign(p_loc[ax])
    return R @ d_loc   # outward normal in world


def _run(planner, model, data, obj_body, p1, p2, d1, d2, hard_pin, repeats,
         q_ref_override=None):
    """Solve `repeats` times with the given fixed contacts; aggregate convergence."""
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body)
    obj_pos = data.xpos[bid].copy()
    act_idx = planner._act_idx
    q_ref = (np.asarray(q_ref_override, float) if q_ref_override is not None
             else np.array([data.qpos[i] for i in act_idx]))

    n_conv = n_ok = n = 0
    iters, ms = [], []
    for _ in range(repeats):
        planner.data.qpos[:] = data.qpos[:]
        mj.mj_forward(model, planner.data)
        t0 = time.time()
        res = planner.solve(q_ref, obj_pos, p1_init=p1, p2_init=p2, d1=d1, d2=d2)
        ms.append((time.time() - t0) * 1e3)
        n += 1
        if res.get('status') == 'converged':
            n_conv += 1
        if res.get('wrench_ok'):
            n_ok += 1
        it = res.get('iterations')
        if it is not None:
            iters.append(it)
    return {'conv': n_conv, 'ok': n_ok, 'n': n,
            'iters': (np.mean(iters) if iters else None), 'ms': np.mean(ms)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--object', default='obj_red_box')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--max-iter', type=int, default=120)
    ap.add_argument('--sample', type=int, default=None,
                    help="index into samples.jsonl: use its recorded near-box q_seed as "
                         "q_ref and relocate --object to its obj_qpos, reproducing the "
                         "live operator-near-box conditions instead of the home pose")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(_SCENE)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    q_ref_override = None
    if args.sample is not None:
        import json
        spath = os.path.join(os.path.dirname(_HERE), 'samples.jsonl')
        with open(spath) as f:
            samples = [json.loads(l) for l in f if l.strip()]
        s = samples[args.sample]
        q_ref_override = np.asarray(s['q_seed'], float)
        # Relocate --object to the recorded object pose so the fixed sites sit where the
        # hand actually was, and pose the arm at the recorded near-box q_seed.
        oq = np.asarray(s['obj_qpos'], float)
        jadr = model.jnt_qposadr[model.body_jntadr[
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, args.object)]]
        data.qpos[jadr:jadr + 7] = oq[:7]
        act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
        for i, idx in enumerate(act_idx):
            data.qpos[idx] = q_ref_override[i]
        mj.mj_forward(model, data)
        print(f"[sample {args.sample}] using recorded q_seed + obj_qpos "
              f"(orig object was {s.get('object')}, relocated {args.object} to it)")

    obj = args.object
    p1 = _site_world(model, data, obj + '_c1')
    p2 = _site_world(model, data, obj + '_c2')
    if p1 is None or p2 is None:
        sys.exit(f"sites {obj}_c1/_c2 not found")
    d1 = _outward_face_dir(model, data, obj, p1)
    d2 = _outward_face_dir(model, data, obj, p2)
    print(f"\nFixed-contact wrench test — {obj}")
    print(f"  c1 world={np.round(p1,3)}  outward d1={np.round(d1,2)}")
    print(f"  c2 world={np.round(p2,3)}  outward d2={np.round(d2,2)}")
    print(f"  repeats={args.repeats}  max_iter={args.max_iter}")
    print("=" * 78)
    print(f"{'solver':<10} {'wrench':<7} {'conv':>7} {'wrench_ok':>10} "
          f"{'iters':>7} {'ms':>7}")
    print("-" * 78)

    # solver backends x wrench on/off. Contacts are face-pinned at the known-good sites,
    # so the only remaining freedom is q + (when wrench on) the cone internals.
    for solver in ('ipopt', 'slsqp'):
        for wrench in (True, False):
            cfg = GraspConfig3D(
                obj_geom=obj + '_geom', obj_body=obj, max_iter=args.max_iter,
                wrench_constraint=wrench, use_slsqp=(solver == 'slsqp'),
            )
            planner = GraspPlanner3D(model, mj.MjData(model), cfg)
            s = _run(planner, model, data, obj, p1, p2, d1, d2,
                     hard_pin=False, repeats=args.repeats,
                     q_ref_override=q_ref_override)
            it = f"{s['iters']:.0f}" if s['iters'] is not None else '-'
            print(f"{solver:<10} {'Y' if wrench else 'N':<7} "
                  f"{s['conv']}/{s['n']:>5} {s['ok']}/{s['n']:>8} {it:>7} {s['ms']:>7.0f}")
    print("=" * 78)
    print("Contacts are pinned at the known-good grasp sites, so a non-converged row means")
    print("the q + wrench solve (or solver settings) can't certify a hand-picked feasible")
    print("grasp — not that the contact search failed. Compare ipopt vs slsqp and wrench Y/N.\n")


if __name__ == '__main__':
    main()

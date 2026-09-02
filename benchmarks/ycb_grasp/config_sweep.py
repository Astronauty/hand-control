"""Solve-time / convergence sweep over random ARM configurations, for one fixed
YCB object placement -- diagnostic only, not a change to the live teleop path.

    python benchmarks/ycb_grasp/config_sweep.py --object 005_tomato_soup_can --n 40

The live grasp recommender always warm-starts q_ref from the operator's actual
current arm pose (one config per call). This sweep instead draws many random
arm configurations -- sampled the SAME way benchmarks/ycb_grasp/workspace.py
samples the reachable-workspace cloud (continuous joints 1/3/5/7 forced to
[-pi, pi], see workspace._arm_bounds) -- and re-solves the grasp NLP from each,
to characterize how solve time and convergence vary with the STARTING arm
configuration rather than with the object.

High-dimensional q_ref (7 arm DOF here) is not itself a useful plot axis, so
each config is reduced to a single scalar: the palm-to-object distance at that
config (FK, no solve needed to compute). Solve time / success are then plotted
against that proximity heuristic, per the plan discussed for this benchmark.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from simulation.grasp_planner_3d import GraspConfig3D, MultiStartGraspPlanner3D  # noqa: E402
from ycb_grasp import scene as S, workspace as W                                # noqa: E402

N_ROBOT = 23


def sample_arm_configs(model, n, seed=0):
    """n random arm-joint vectors, same bounds convention as workspace._arm_bounds
    (continuous Gen3 joints 1/3/5/7 forced to [-pi, pi] instead of the [0,0]
    jnt_range MuJoCo compiles them to)."""
    rng = np.random.default_rng(seed)
    lo, hi = W._arm_bounds(model)
    return rng.uniform(lo, hi, size=(n, W.N_ARM))


def palm_object_distance(model, data, arm_q, obj_pos):
    """FK-only proximity heuristic: palm position at arm_q vs. the object center.
    No solve required -- cheap enough to compute for every sampled config."""
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, W.PALM_BODY)
    d = mj.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qpos[:W.N_ARM] = arm_q
    mj.mj_kinematics(model, d)
    return float(np.linalg.norm(d.xpos[bid] - np.asarray(obj_pos, float)))


def run_sweep(obj_id, n=40, seed=0, obj_pos=(0.55, 0.0, 0.35),
             obj_quat=(1.0, 0.0, 0.0, 0.0), n_seeds=3, max_iter=150):
    model, data, info = S.build([(obj_id, obj_pos, obj_quat)])
    bn = next(iter(info))
    obj_geom0 = S.hull_geoms(model, bn)[0]

    cfg = GraspConfig3D(obj_geom=obj_geom0, obj_body=bn,
                        n_seeds=n_seeds, max_iter=max_iter)
    data.qpos[:N_ROBOT] = 0.0
    mj.mj_forward(model, data)

    planner = MultiStartGraspPlanner3D(model, data, cfg)   # bakes the SDF once

    arm_qs = sample_arm_configs(model, n, seed=seed)
    q_ref = np.zeros(N_ROBOT)   # hand joints fixed at 0 across the sweep

    rows = []
    for i, arm_q in enumerate(arm_qs):
        prox = palm_object_distance(model, data, arm_q, obj_pos)
        q_ref[:W.N_ARM] = arm_q

        t0 = time.perf_counter()
        res = planner.solve(q_ref, np.asarray(obj_pos, float), max_seeds=n_seeds)
        dt_ms = (time.perf_counter() - t0) * 1e3

        rows.append(dict(
            i=i, proximity_m=prox, solve_ms=dt_ms,
            status=res.get('status'), cost=res.get('cost'),
            converged=(res.get('status') == 'converged'),
        ))
        print(f"[{i+1}/{n}] prox={prox:.3f}m  solve={dt_ms:.0f}ms  "
              f"status={res.get('status')}  cost={res.get('cost')}")
    return rows


def plot(rows, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    prox = np.array([r['proximity_m'] for r in rows])
    t_ms = np.array([r['solve_ms'] for r in rows])
    ok   = np.array([r['converged'] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(prox[ok], t_ms[ok], c='tab:green', label='converged', alpha=0.8)
    ax1.scatter(prox[~ok], t_ms[~ok], c='tab:red', label='best-effort/failed', alpha=0.8)
    ax1.set_xlabel('palm-to-object distance at q_ref (m)')
    ax1.set_ylabel('solve time (ms)')
    ax1.set_title('Solve time vs. starting-config proximity')
    ax1.legend()

    # Success rate by proximity bucket (5 bins)
    bins = np.linspace(prox.min(), prox.max() + 1e-9, 6)
    idx = np.digitize(prox, bins) - 1
    rate = [ok[idx == b].mean() if (idx == b).any() else np.nan for b in range(5)]
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax2.bar(centers, rate, width=(bins[1] - bins[0]) * 0.8)
    ax2.set_xlabel('palm-to-object distance (m), binned')
    ax2.set_ylabel('convergence rate')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Convergence rate vs. proximity')

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="005_tomato_soup_can")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="contact seeds per arm config (MultiStart's own budget)")
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "ycb_grasp" / "out"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = run_sweep(args.object, n=args.n, seed=args.seed,
                     n_seeds=args.n_seeds, max_iter=args.max_iter)

    n_ok = sum(r['converged'] for r in rows)
    ts = [r['solve_ms'] for r in rows]
    print(f"\n{n_ok}/{len(rows)} converged  "
          f"solve_ms: mean={np.mean(ts):.0f} p50={np.median(ts):.0f} "
          f"p90={np.percentile(ts, 90):.0f}")

    plot(rows, out / f"config_sweep_{args.object}.png")


if __name__ == "__main__":
    main()

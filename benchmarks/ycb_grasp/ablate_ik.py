"""Ablate IK backend / Jacobian / iterate-tracking across a spread of YCB objects.

    python benchmarks/ycb_grasp/ablate_ik.py                    # default sweep
    python benchmarks/ycb_grasp/ablate_ik.py --seeds 4 --max-iter 200
    python benchmarks/ycb_grasp/ablate_ik.py --arms ipopt sqp

Each arm solves the same pinch targets from the same DLS warm start, so the only
difference is the solver configuration.

Two fairness details that are easy to get wrong:

* `configure_sqp` swaps module-level globals process-wide. Every arm therefore
  calls `configure_ipopt` first, so an IPOPT arm run after an SQP arm measures
  IPOPT rather than leftover SQP internals.
* `_SQP_SOLVER_OPTS` carries its own `max_iter` (800) which would otherwise let
  the SQP arm run 2.7x longer than IPOPT's 300. It is overridden per run.
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

import grasp_control.constrained_ik as cik                      # noqa: E402
from grasp_control import ConstrainedIKSolver, SpatialIKSolver  # noqa: E402
from kinova_leap_pick_place import FINGER_SET, FINGER_TIP_SITES  # noqa: E402
from ycb_grasp import scene as S, workspace as W                # noqa: E402
from ycb_grasp.ik_demo import (clearance_by_geom, home_bias,    # noqa: E402
                               object_hull_verts, pinch_targets_from,
                               place_objects, robot_geom_names)

N_ROBOT = 23

# Spread over hull count (1 -> 55), which is what the per-hull constraint path
# scales with, and over shape family (box / round / elongated / concave).
DEFAULT_OBJECTS = [
    "003_cracker_box", "013_apple", "077_rubiks_cube",           # 1 hull
    "048_hammer", "011_banana", "021_bleach_cleanser",           # 4-5
    "035_power_drill", "006_mustard_bottle",                     # 5-7
    "037_scissors", "025_mug", "005_tomato_soup_can", "024_bowl",  # 36-55
]

# (label, backend, jacobian, track_best, n_starts)
#
# sqp-ms exists because a single SQP solve converges (every run reports
# Solve_Succeeded) but lands under 5 mm only ~58% of the time, against IPOPT's
# 86% -- it finds a genuine local minimum, just not always a good one. At 159 ms
# median a handful of restarts still costs a fraction of IPOPT's 2.9 s, and the
# residual is directly observable, so a bad basin can be detected and retried
# rather than silently returned.
DEFAULT_ARMS = [
    ("ipopt", "ipopt", "auto", True, 1),
    ("sqp", "sqp", "auto", True, 1),
    ("sqp-ms4", "sqp", "auto", True, 4),
]

ACCEPT_MM = 1.0        # residual below which a restart is good enough to stop
RESTART_SIGMA = (0.25, 0.35)   # (arm, hand) joint perturbation for restarts


def make_solver(model, rgeoms, obj_geoms, sdf_bodies, backend, jacobian, max_iter):
    cik.configure_ipopt()          # clear any prior configure_sqp before building
    s = ConstrainedIKSolver(
        model, N_ROBOT, rgeoms, obj_geoms, sdf_bodies=sdf_bodies,
        clearance=0.005, posture_weight=np.r_[np.full(7, 1e-5), np.full(16, 1e-4)],
        tip_weight=100.0, orient_weight=0.0, jacobian=jacobian, max_iter=max_iter,
    )
    if backend == "sqp":
        cik.configure_sqp(s)
        s._solver_opts = dict(cik._SQP_SOLVER_OPTS, max_iter=max_iter)
    return s


def _tip_err_mm(model, data, q, sids, tgt):
    data.qpos[:N_ROBOT] = q
    mj.mj_forward(model, data)
    return max(np.linalg.norm(data.site_xpos[i] - t) for i, t in zip(sids, tgt)) * 1e3


def run_case(model, data, body, tgt, q_bias, sids, arm, max_iter, seed=0):
    label, backend, jacobian, track, n_starts = arm
    rgeoms = robot_geom_names(model)

    dls = SpatialIKSolver(n_robot=N_ROBOT)
    dd = mj.MjData(model)
    dd.qpos[:] = data.qpos
    t0 = time.time()
    q_warm = dls.solve(model, dd, sids, tgt, q_bias=q_bias, null_gain=0.3)
    dls_ms = (time.time() - t0) * 1e3

    rng = np.random.default_rng(1000 + seed)
    best_q, best_e, best_m, total_ms, used = None, np.inf, {}, 0.0, 0
    for k in range(n_starts):
        q0 = q_warm.copy()
        if k:      # restart from a perturbed warm start; k=0 keeps the DLS point
            q0[:7] += rng.normal(0, RESTART_SIGMA[0], 7)
            q0[7:] += rng.normal(0, RESTART_SIGMA[1], N_ROBOT - 7)
            q0[7:] = np.clip(q0[7:], model.jnt_range[7:N_ROBOT, 0],
                             model.jnt_range[7:N_ROBOT, 1])
        s = make_solver(model, rgeoms, ["floor"], (body,), backend, jacobian, max_iter)
        try:
            q = s.solve(data, sids, tgt, q_bias=q_bias, q_init=q0,
                        reduced_clearance_geoms=clearance_by_geom(rgeoms),
                        track_best=track)
        finally:
            cik.configure_ipopt()   # never leave the swap in place for the next arm
        e = _tip_err_mm(model, data, q, sids, tgt)
        total_ms += s.last_metrics.get("t_solve_ms", 0.0)
        used = k + 1
        if e < best_e:
            best_q, best_e, best_m = q, e, dict(s.last_metrics)
        if best_e <= ACCEPT_MM:
            break                    # residual is observable, so stop early

    m = dict(best_m)
    m["err_mm"] = best_e
    m["dls_ms"] = dls_ms
    m["t_solve_ms"] = total_ms      # charge every restart, not just the winner
    m["n_starts_used"] = used
    _tip_err_mm(model, data, best_q, sids, tgt)   # leave data at the winning pose
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of arm labels to run")
    args = ap.parse_args()

    arms = [a for a in DEFAULT_ARMS if args.arms is None or a[0] in args.arms]
    base = mj.MjModel.from_xml_path(str(REPO / "models" / "scene_kinova_leap.xml"))
    ws = W.load_or_build(base, n=200_000, seed=0)
    q_bias = home_bias()

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(100 + seed)
        for oid in args.objects:
            try:
                (_, pos, quat), = place_objects([oid], ws, rng)
            except RuntimeError as e:
                print(f"  skip {oid} seed {seed}: {e}")
                continue
            V = object_hull_verts(oid)
            a, b, _ = pinch_targets_from(V, pos, quat)
            model, data, info = S.build([(oid, pos, quat)])
            body = next(iter(info))
            sids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
                    for f in FINGER_SET]
            tgt = [{"index": a, "thumb": b}[f] for f in FINGER_SET]
            n_hull = len(S.hull_geoms(model, body))

            for arm in arms:
                data.qpos[:N_ROBOT] = q_bias
                mj.mj_forward(model, data)
                m = run_case(model, data, body, tgt, q_bias, sids, arm, args.max_iter, seed)
                rows.append(dict(m, seed=seed, object=oid, hulls=n_hull, arm=arm[0]))
                print(f"  [{seed}] {oid:22s} {arm[0]:6s} "
                      f"err={m['err_mm']:8.2f}mm t={m['t_solve_ms']:7.0f}ms "
                      f"it={m['iters']:4d} starts={m.get('n_starts_used', 1)} "
                      f"{str(m.get('status'))[:26]}", flush=True)

    import csv
    out = REPO / "benchmarks" / "ycb_grasp" / "out" / "ablate_ik.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nper-solve rows -> {out}")

    print("\n" + "=" * 92)
    hdr = (f"{'arm':8s} {'n':>4s} {'err med':>9s} {'err p90':>9s} {'<1mm':>6s} "
           f"{'<5mm':>6s} {'ms med':>8s} {'ms p90':>8s} {'iters med':>10s} {'best_it med':>12s}")
    print(hdr)
    for arm in arms:
        r = [x for x in rows if x["arm"] == arm[0]]
        if not r:
            continue
        e = np.array([x["err_mm"] for x in r])
        t = np.array([x["t_solve_ms"] for x in r])
        it = np.array([x["iters"] for x in r])
        # best_iter is None on a converged solve (nothing beat the final point),
        # which is information rather than a gap -- report it as such.
        bi = [x["best_iter"] for x in r if x.get("best_iter") is not None]
        bi_med = f"{np.median(bi):12.0f}" if bi else f"{'n/a':>12s}"
        print(f"{arm[0]:8s} {len(r):4d} {np.median(e):8.2f}mm {np.percentile(e, 90):8.2f}mm "
              f"{np.mean(e < 1) * 100:5.0f}% {np.mean(e < 5) * 100:5.0f}% "
              f"{np.median(t):7.0f}ms {np.percentile(t, 90):7.0f}ms "
              f"{np.median(it):10.0f} {bi_med}")

    # Where the per-hull path used to blow up, so worth splitting out.
    print(f"\n{'by hull count':<20s}" + "".join(f"{a[0]:>22s}" for a in arms))
    for lo, hi, lbl in [(0, 3, "1-3 hulls"), (4, 20, "4-20"), (21, 999, "21+")]:
        line = f"{lbl:<20s}"
        for arm in arms:
            r = [x for x in rows if x["arm"] == arm[0] and lo <= x["hulls"] <= hi]
            line += (f"{np.median([x['err_mm'] for x in r]):9.2f}mm "
                     f"{np.median([x['t_solve_ms'] for x in r]):7.0f}ms" if r
                     else f"{'-':>22s}")
        print(line)


if __name__ == "__main__":
    main()

"""Ablate the grasp NLP (MultiStartGraspPlanner3D) across YCB objects, using the
SAME randomized object/robot poses AND collision setup as ablate_ik.py's IK
ablation, so the two are actually comparable (see the module docstring history —
earlier versions of this file used GraspPlanner3D's default arm_geom_names=[]
i.e. NO arm-vs-object collision at all, unlike ablate_ik.py's tuned clearance_by_geom
schedule; this version aligns them).

    python benchmarks/ycb_grasp/ablate_grasp.py                  # default sweep, no render
    python benchmarks/ycb_grasp/ablate_grasp.py --render         # + one PNG per solve
    python benchmarks/ycb_grasp/ablate_grasp.py --objects 025_mug 011_banana --seeds 2
    python benchmarks/ycb_grasp/ablate_grasp.py --gws             # turn on w_gws/w_span
    python benchmarks/ycb_grasp/ablate_grasp.py --restarts 4      # q_ref perturbation, like sqp-ms4

Reuses ablate_ik.py's DEFAULT_OBJECTS (spread over hull count 1->55 and shape
family: box/round/elongated/concave) and its place_objects/home_bias pose
sampling, so results are directly comparable to the prior IK-only ablation
(benchmarks/ycb_grasp/IK_SDF_FINDINGS.md Sec 5) — same poses, same collision
tiers, different solver.

Metrics per solve: status (converged/best-effort/failed), IPOPT return_status
(the actual failure-type string, not just the coarse status tag — status is
documented as unreliable on its own, see RECOMMENDER_CONVERGENCE_FINDINGS.md),
TRUE fingertip-to-target residual (err_mm, computed independently of solver
status, matching ablate_ik.py's _tip_err_mm — this is the metric that actually
matters), solve time, iteration count, GWS beta (when --gws), wrench
feasibility. Aggregated by object (and, same as ablate_ik.py, by hull-count
bucket) into a summary table + CSV + plots; optionally one rendered PNG per
solve under out/renders/.
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

from simulation.grasp_planner_3d import (GraspConfig3D, MultiStartGraspPlanner3D,  # noqa: E402
                                         _geom_normal_np)
from ycb_grasp import scene as S, workspace as W                                # noqa: E402
from ycb_grasp.ik_demo import (clearance_by_geom, home_bias, place_objects,     # noqa: E402
                               render, robot_geom_names)
from ycb_grasp.ablate_ik import DEFAULT_OBJECTS                                 # noqa: E402
from kinova_leap_pick_place import FINGER_SET, FINGER_TIP_SITES                 # noqa: E402

N_ROBOT = 23
# Matches clearance_by_geom's own +2mm default tier — GraspConfig3D.col_clearance_m
# previously defaulted to 5mm, a real (if minor) mismatch against ablate_ik.py.
DEFAULT_COL_CLEARANCE_M = 0.002


def _tip_err_mm(model, data, obj_gid, obj_bid, mesh_entry, cfg, q, p1, p2):
    """True fingertip-to-target residual (mm), independent of solver status —
    matches ablate_ik.py's _tip_err_mm. Targets are contact + r_tip*outward_normal,
    the same offset the NLP's own IK cost optimizes for (grasp_planner_3d.py's
    _tp1_tgt/_tp2_tgt), not the bare contact point.

    obj_gid: the object's collision geom id (geom_type/geom_size are geom-indexed).
    obj_bid: the object's body id — pose comes from here for mesh objects, since
        object_sdf's table is baked in BODY frame (see grasp_planner_3d.py's own
        comment at the analogous obj_center_np/obj_R_np resolution).
    """
    data.qpos[:N_ROBOT] = q
    mj.mj_forward(model, data)
    gt = int(model.geom_type[obj_gid])
    gsize = model.geom_size[obj_gid]
    if mesh_entry is not None:
        obj_pos = data.xpos[obj_bid]
        obj_mat = data.xmat[obj_bid].reshape(3, 3)
    else:
        obj_pos = data.geom_xpos[obj_gid]
        obj_mat = data.geom_xmat[obj_gid].reshape(3, 3)
    n1 = _geom_normal_np(p1, gt, obj_pos, obj_mat, gsize, mesh_entry=mesh_entry)
    n2 = _geom_normal_np(p2, gt, obj_pos, obj_mat, gsize, mesh_entry=mesh_entry)
    tgt = {"thumb": p1 + cfg.r_thumb * n1, "index": p2 + cfg.r_index * n2}
    sids = {f: mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f]) for f in FINGER_SET}
    return max(np.linalg.norm(data.site_xpos[sids[f]] - tgt[f]) for f in FINGER_SET) * 1000


def run_case(model, data, body_name, q_ref, obj_pos, cfg, n_seeds):
    obj_geom0 = S.hull_geoms(model, body_name)[0]
    cfg.obj_geom = obj_geom0
    cfg.obj_body = body_name

    planner = MultiStartGraspPlanner3D(model, data, cfg)
    t0 = time.perf_counter()
    res = planner.solve(q_ref, obj_pos, max_seeds=n_seeds)
    dt_ms = (time.perf_counter() - t0) * 1e3

    res["t_solve_ms"] = dt_ms
    res["n_hulls"] = len(S.hull_geoms(model, body_name))

    if res.get("q") is not None and res.get("p1") is not None:
        mesh_entry = planner._planner._mesh_entry
        res["err_mm"] = _tip_err_mm(model, data, planner._planner._obj_gid,
                                    planner._planner._obj_bid, mesh_entry, cfg,
                                    res["q"], np.asarray(res["p1"]), np.asarray(res["p2"]))
    else:
        res["err_mm"] = float("nan")
    return res, planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS)
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of random object-pose trials per object (matches ablate_ik.py)")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="MultiStartGraspPlanner3D's own contact-seed budget per solve "
                         "(contact-pair diversity AND, when --restarts is set, q_ref restarts "
                         "share this same budget — see cfg.qref_restart_sigma_arm/hand)")
    ap.add_argument("--restarts", action="store_true",
                    help="perturb q_ref per seed (matches ablate_ik.py's RESTART_SIGMA=(0.25,0.35) "
                         "rad arm/hand joint noise for the sqp-ms4 backend) instead of leaving "
                         "every seed at the operator's exact q_ref")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--gws", action="store_true",
                    help="GWS-only formulation: wrench_constraint=False (drops the 64-corner "
                         "task-specific wrench-cone LP entirely) + w_gws/w_span on (the beta/"
                         "alpha min-weight LP gates feasibility instead — see "
                         "RECOMMENDER_CONVERGENCE_FINDINGS.md-adjacent session notes: the "
                         "64-corner LP never converged within budget on any object tested, "
                         "GWS-only converges in seconds)")
    ap.add_argument("--w-gws", type=float, default=5.0)
    ap.add_argument("--w-span", type=float, default=1.0)
    ap.add_argument("--soft-finger", action="store_true",
                    help="gws_soft_finger=True: add 2 torsional-spin generators/contact to W "
                         "(mu_t from model geom_friction, MuJoCo-realistic elementwise-max "
                         "combine) so a 2-contact pinch's W isn't structurally rank-deficient "
                         "about the grasp axis. Only meaningful with --gws.")
    ap.add_argument("--uv-atlas", action="store_true",
                    help="use_uv_atlas_contact=True: mesh contacts get a 2-DOF offset in a "
                         "plane fit to the seed's real local mesh neighborhood (UV-atlas chart), "
                         "bounded by that neighborhood's actual boundary, instead of the default "
                         "fixed-size Euclidean tangent-plane box. See GraspConfig3D docstring.")
    ap.add_argument("--uv-rings", type=int, default=2,
                    help="uv_atlas_rings — local-neighborhood size in face-adjacency hops. "
                         "Only meaningful with --uv-atlas.")
    ap.add_argument("--render", action="store_true", help="save one PNG per solve")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "ycb_grasp" / "out"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    render_dir = out / "renders"
    if args.render:
        render_dir.mkdir(parents=True, exist_ok=True)

    base = mj.MjModel.from_xml_path(str(REPO / "models" / "scene_kinova_leap.xml"))
    ws = W.load_or_build(base, n=200_000, seed=0)
    q_bias_full = home_bias()   # (23,) arm(7)+hand(16) — same as ablate_ik.py

    # Collision setup matching ablate_ik.py's ConstrainedIKSolver calls exactly
    # (robot_geom_names + clearance_by_geom, ik_demo.py) — GraspPlanner3D defaults
    # arm_geom_names to [] (NO arm-vs-object collision at all) unless set explicitly.
    _rgeoms = robot_geom_names(base)
    _obj_clr = clearance_by_geom(_rgeoms)

    cfg_kw = dict(n_seeds=args.n_seeds, max_iter=args.max_iter,
                 arm_geom_names=_rgeoms, obj_clearance_by_geom=_obj_clr,
                 col_clearance_m=DEFAULT_COL_CLEARANCE_M)
    if args.gws:
        cfg_kw["wrench_constraint"] = False
        cfg_kw["w_gws"] = args.w_gws
        cfg_kw["w_span"] = args.w_span
        cfg_kw["gws_soft_finger"] = args.soft_finger
    if args.uv_atlas:
        cfg_kw["use_uv_atlas_contact"] = True
        cfg_kw["uv_atlas_rings"] = args.uv_rings
    if args.restarts:
        # Same magnitude as ablate_ik.py's RESTART_SIGMA; stand-in for eventually
        # sampling from a real distribution around the operator's pose (see
        # GraspConfig3D.qref_restart_sigma_arm/hand's docstring).
        cfg_kw["qref_restart_sigma_arm"] = 0.25
        cfg_kw["qref_restart_sigma_hand"] = 0.35

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(100 + seed)
        for oid in args.objects:
            try:
                (_, pos, quat), = place_objects([oid], ws, rng)
            except RuntimeError as e:
                print(f"  skip {oid} seed {seed}: {e}")
                continue

            model, data, info = S.build([(oid, pos, quat)])
            body_name = next(iter(info))
            data.qpos[:N_ROBOT] = q_bias_full
            mj.mj_forward(model, data)

            cfg = GraspConfig3D(**cfg_kw)
            try:
                res, planner = run_case(model, data, body_name, q_bias_full,
                                        np.asarray(pos, float), cfg, args.n_seeds)
            except Exception as e:
                print(f"  [{seed}] {oid:24s} EXCEPTION: {e}")
                rows.append(dict(seed=seed, object=oid, status="exception",
                                 return_status=str(e)[:60], t_solve_ms=float("nan"),
                                 iterations=None, cost=None, gws_beta=None, n_hulls=None,
                                 err_mm=float("nan")))
                continue

            # verify()'s standalone post-solve LP certificate — independent of whatever
            # wrench-related NLP terms/constraints were (or weren't) active this solve,
            # so it's the same ground-truth feasibility check across every --gws
            # combination tested (see the rubiks-cube diagnostic session that motivated
            # this: the embedded 64-corner LP's OWN convergence was not a reliable
            # feasibility signal, verify()'s min_gamma_for_accel_lp was).
            vinfo = {}
            if res.get("q") is not None:
                try:
                    vinfo = planner._planner.verify(res)
                except Exception:
                    vinfo = {}

            row = dict(
                seed=seed, object=oid, hulls=res.get("n_hulls"),
                status=res.get("status"), return_status=res.get("return_status"),
                t_solve_ms=res.get("t_solve_ms"), iterations=res.get("iterations"),
                cost=res.get("cost"), gws_beta=res.get("gws_beta"),
                gamma_nlp=res.get("gamma_nlp"), max_slack_norm=res.get("max_slack_norm"),
                err_mm=res.get("err_mm"),
                verify_wrench_feasible=vinfo.get("wrench_feasible"),
                verify_gamma_min=vinfo.get("gamma_min"),
                verify_max_slack_norm=vinfo.get("max_slack_norm"),
            )
            rows.append(row)
            print(f"  [{seed}] {oid:24s} hulls={row['hulls']:>3}  "
                  f"status={str(row['status']):12s} rs={str(row['return_status'])[:28]:28s} "
                  f"err={row['err_mm']:7.2f}mm "
                  f"t={row['t_solve_ms']:7.0f}ms it={str(row['iterations']):>5s} "
                  f"beta={row['gws_beta']}", flush=True)

            if args.render and res.get("q") is not None:
                data.qpos[:N_ROBOT] = res["q"]
                mj.mj_forward(model, data)
                png = render_dir / f"{oid}_seed{seed}.png"
                try:
                    render(model, data, png, lookat=pos, dist=0.75)
                except Exception as e:
                    print(f"    render failed: {e}")

    import csv
    csv_path = out / "ablate_grasp.csv"
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nper-solve rows -> {csv_path}")

    _summary(rows, args.objects)
    _plot(rows, out / "ablate_grasp_summary.png")


def _summary(rows, objects):
    print("\n" + "=" * 130)
    print(f"{'object':24s} {'n':>3s} {'err med':>9s} {'err p90':>9s} {'<5mm':>6s} "
          f"{'conv%':>6s} {'t med(ms)':>10s} {'it med':>7s} {'beta med':>9s} "
          f"{'wf%':>5s} {'gmin med':>9s}")
    for oid in objects:
        r = [x for x in rows if x["object"] == oid]
        if not r:
            continue
        n = len(r)
        conv = sum(x["status"] == "converged" for x in r) / n * 100
        errs = np.array([x["err_mm"] for x in r
                         if x.get("err_mm") is not None and not np.isnan(x["err_mm"])])
        ts = [x["t_solve_ms"] for x in r if x.get("t_solve_ms") is not None
             and not np.isnan(x["t_solve_ms"])]
        its = [x["iterations"] for x in r if x.get("iterations") is not None]
        betas = [x["gws_beta"] for x in r if x.get("gws_beta") is not None]
        wf_vals = [x["verify_wrench_feasible"] for x in r if x.get("verify_wrench_feasible") is not None]
        gmins = [x["verify_gamma_min"] for x in r if x.get("verify_gamma_min") is not None]
        err_med = np.median(errs) if len(errs) else float("nan")
        err_p90 = np.percentile(errs, 90) if len(errs) else float("nan")
        pct5 = np.mean(errs < 5.0) * 100 if len(errs) else float("nan")
        t_med = np.median(ts) if ts else float("nan")
        it_med = np.median(its) if its else float("nan")
        beta_med = np.median(betas) if betas else float("nan")
        wf_pct = np.mean(wf_vals) * 100 if wf_vals else float("nan")
        gmin_med = np.median(gmins) if gmins else float("nan")
        print(f"{oid:24s} {n:3d} {err_med:8.2f}mm {err_p90:8.2f}mm {pct5:5.0f}% "
              f"{conv:5.0f}% {t_med:9.0f}ms {it_med:6.0f} {beta_med:8.3f} "
              f"{wf_pct:4.0f}% {gmin_med:9.3f}")

    print("\n(wf% = verify()'s STANDALONE post-solve LP feasibility certificate "
          "(min_gamma_for_accel_lp), independent of whichever wrench-related NLP terms "
          "were active this run — the ground-truth check, not the NLP's own status. "
          "gmin med = median required squeeze force gamma at that certificate, N.)")

    print("\n(NOTE: status/conv% is the coarse IPOPT/SQP convergence FLAG, documented as "
          "unreliable on its own — see RECOMMENDER_CONVERGENCE_FINDINGS.md and the "
          "013_apple stage0 case that reported 'converged' at a 26mm residual this "
          "session. err_mm above is the metric that actually matters, matching "
          "ablate_ik.py's _tip_err_mm convention.)")

    print("\nIPOPT return_status breakdown:")
    from collections import Counter
    c = Counter(r.get("return_status") for r in rows)
    for k, v in c.most_common():
        print(f"  {str(k):40s} {v:4d}  ({v/len(rows)*100:.0f}%)")

    print("\nby hull count:")
    for lo, hi, lbl in [(0, 3, "1-3 hulls"), (4, 20, "4-20"), (21, 999, "21+")]:
        r = [x for x in rows if x.get("hulls") is not None and lo <= x["hulls"] <= hi]
        if not r:
            continue
        errs = np.array([x["err_mm"] for x in r
                         if x.get("err_mm") is not None and not np.isnan(x["err_mm"])])
        err_med = np.median(errs) if len(errs) else float("nan")
        ts = [x["t_solve_ms"] for x in r if x.get("t_solve_ms") is not None
             and not np.isnan(x["t_solve_ms"])]
        print(f"  {lbl:<12s} n={len(r):3d}  err_med={err_med:7.2f}mm  "
              f"t_med={np.median(ts) if ts else float('nan'):7.0f}ms")


def _plot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    objects = sorted({r["object"] for r in rows})
    err_med = []
    t_med = []
    for oid in objects:
        r = [x for x in rows if x["object"] == oid]
        errs = np.array([x["err_mm"] for x in r
                         if x.get("err_mm") is not None and not np.isnan(x["err_mm"])])
        err_med.append(np.median(errs) if len(errs) else np.nan)
        ts = [x["t_solve_ms"] for x in r if x.get("t_solve_ms") is not None
             and not np.isnan(x["t_solve_ms"])]
        t_med.append(np.median(ts) if ts else np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(objects) * 0.5), 8))
    ax1.bar(range(len(objects)), err_med, color="tab:orange")
    ax1.set_xticks(range(len(objects)))
    ax1.set_xticklabels(objects, rotation=60, ha="right", fontsize=8)
    ax1.set_ylabel("median tip-to-target error (mm)")
    ax1.axhline(5.0, color="gray", linestyle="--", linewidth=1, label="5mm")
    ax1.legend()
    ax1.set_title("Grasp NLP true residual by object (matches ablate_ik.py's err_mm)")

    ax2.bar(range(len(objects)), t_med, color="tab:blue")
    ax2.set_xticks(range(len(objects)))
    ax2.set_xticklabels(objects, rotation=60, ha="right", fontsize=8)
    ax2.set_ylabel("median solve time (ms)")
    ax2.set_title("Grasp NLP solve time by object")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()

"""FRoGGeR vs this solver: plan-only benchmark over the YCB tabletop scene.

Two arms, differing ONLY in objective structure and normal source:

  ours    -- `for_gws_recommender`: beta as one of ~10 weighted terms (w_gws=5.0
             against w_ik=0.70, which dominates the placement gradient), no
             robustness floor, contact normals from the quadratic patch.
  frogger -- `for_frogger`: beta as the SOLE objective, a hard floor
             l_bar* = n_cols*beta >= k_l (FRoGGeR (7c), k_l=0.3), and contact
             normals from the object SDF gradient (their n = -grad s(p)).

Held in common so the comparison is about the formulation: scene, settle, seeding
pool, collision model, patch POSITION parameterization, and every scored quantity.

Scored per solve:
  l_bar*        n_cols * beta, FRoGGeR's normalized metric, comparable across
                contact counts (raw beta's ceiling is 1/n_cols)
  lp_gap        beta_rep_converged - beta_rep, the embedded LP's own optimality
                gap. A COMMON-MODE limitation of both arms (both use the same
                single-level embedding), reported so a beta difference is not
                mistaken for a formulation difference. See docs/GWS_IMPROVEMENTS.md.
  delta         beta_rep - beta_true, patch vs SDF-gradient normals
  span_margin   geometric closure test on true normals (n=2 only)
  gamma_min     task-specific wrench certificate (verify()); None = infeasible
  t_solve       wall-clock

Usage:
    uv run python -m ycb_grasp.frogger_bench --objects 017_orange --seeds 0,1,2
    uv run python -m ycb_grasp.frogger_bench --arms ours,frogger --fingers thumb,index,middle

IMPORTANT (measured, FROGGER_BENCH sec 8.7): do not run two sweeps concurrently and
do not edit planner source while one is in flight. Under contention, best-effort
cells can land on a different iterate. Treat any overlapped run as void.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco as mj                                          # noqa: E402

from ycb_grasp import out_paths as OP                          # noqa: E402
from ycb_grasp import table_scene as TS                       # noqa: E402
from ycb_grasp.ik_demo import clearance_by_geom, render, robot_geom_names   # noqa: E402
from simulation.beta_audit import audit, audit_embedded_lp     # noqa: E402
from simulation.grasp_config_builder import (                 # noqa: E402
    for_frogger, for_gws_recommender, load_seed_config, parse_fingers)
from simulation.grasp_planner_3d import (                     # noqa: E402
    MultiStartGraspPlanner3D, _contact_friction)

ARMS = ("ours", "frogger")

# The five objects the tabletop solver work is characterized on, plus foam_brick
# (the documented beta-vs-geometry disagreement case).
DEFAULT_OBJECTS = ["036_wood_block", "017_orange", "014_lemon",
                   "056_tennis_ball", "009_gelatin_box", "061_foam_brick"]


def _build_cfg(arm, object_id, body_name, rgeoms, obj_geom0, *,
               n_seeds, max_iter, fingers, k_l, sdf_normals):
    """Config for one arm. Everything not listed is shared by construction --
    both presets route through for_gws_recommender, so seeding, collision,
    trust region and solver backend are identical."""
    from ycb_grasp.pick_and_place import (
        DEFAULT_COL_CLEARANCE_M, NCF_ACCEL_BUDGET_XYZ, NCF_ANG_ACCEL_BUDGET)

    cfg_kw = dict(n_seeds=n_seeds, max_iter=max_iter,
                  obj_geom=obj_geom0,
                  col_clearance_m=DEFAULT_COL_CLEARANCE_M,
                  use_quadratic_contact=True,
                  quadratic_mesh_fit=True,
                  ground_z=TS.TABLE_TOP_Z)
    for k, v in load_seed_config(object_id).items():
        cfg_kw.setdefault(k, v)

    common = dict(accel_budget_xyz=NCF_ACCEL_BUDGET_XYZ,
                  ang_accel_budget_xyz=NCF_ANG_ACCEL_BUDGET,
                  fingers=fingers)
    if arm == "frogger":
        # (7e): the ACTIVE fingers' geoms may interpenetrate the target object
        # slightly. Derived from the same FINGER_CODE prefixes the RRT uses, so the
        # set tracks the pairing rather than being a hardcoded list.
        from kinova_common.constants import FINGER_CODE
        codes = [FINGER_CODE[r] for r in (fingers or []) if r in FINGER_CODE]
        fo = [g for g in rgeoms
              if any(f"_{c}_" in g for c in codes) and ("ds" in g or "tip" in g)]
        return for_frogger(body_name, rgeoms, clearance_by_geom(rgeoms),
                           k_l=k_l, sdf_normals=sdf_normals,
                           finger_obj_geoms=fo, **common, **cfg_kw)
    return for_gws_recommender(body_name, rgeoms, clearance_by_geom(rgeoms),
                               **common, **cfg_kw)


def plan_one(arm, object_id, seed, *, n_seeds=3, max_iter=80, fingers=None,
             k_l=0.3, sdf_normals=True):
    """Plan (not execute) one grasp with one arm, and score it."""
    model, data, info = TS.build([object_id])
    body_name = next(iter(info))
    TS.settle(model, data)
    pos, _ = TS.object_pose(model, data, body_name, info)

    q_home = TS.home_qpos()
    rgeoms = robot_geom_names(model)
    obj_geom0 = TS.hull_geoms(model, body_name)[0]

    cfg = _build_cfg(arm, object_id, body_name, rgeoms, obj_geom0,
                     n_seeds=n_seeds, max_iter=max_iter, fingers=fingers,
                     k_l=k_l, sdf_normals=sdf_normals)

    planner = MultiStartGraspPlanner3D(model, data, cfg, seed=seed)
    t0 = time.time()
    res = planner.solve(np.asarray(q_home, float), np.asarray(pos, float),
                        max_seeds=n_seeds)
    t_solve = time.time() - t0

    row = dict(arm=arm, object=object_id, seed=seed, t_solve_s=t_solve,
               status=res.get("status"), return_status=res.get("return_status"))
    ctx = dict(model=model, data=data, res=res, pos=np.asarray(pos, float))
    if res.get("p1") is None:
        row["plan_failed"] = True
        return row, ctx

    verify_info = planner._planner.verify(res)
    inner = planner._planner

    d_v = mj.MjData(model)
    d_v.qpos[:] = data.qpos[:]
    mj.mj_forward(model, d_v)
    if inner._obj_geom_type == 7:
        obj_center = d_v.xpos[inner._obj_bid].copy()
        obj_R = d_v.xmat[inner._obj_bid].reshape(3, 3).copy()
    else:
        obj_center = d_v.geom_xpos[inner._obj_gid].copy()
        obj_R = d_v.geom_xmat[inner._obj_gid].reshape(3, 3).copy()

    mu, mu_t = _contact_friction(model, inner._obj_gid,
                                 inner._thumb_gid, inner._index_gid)
    a = audit(res, geom_type=inner._obj_geom_type, obj_center=obj_center,
              obj_R=obj_R, obj_size=inner._obj_size,
              mesh_entry=inner._mesh_entry, mu=mu,
              mu_t=mu_t if cfg.gws_soft_finger else 0.0)
    a.update(audit_embedded_lp(res))

    # FRoGGeR's normalized metric, the quantity its k_l floor is stated in.
    b_rep, b_relp = a.get("beta_reported"), a.get("beta_relp")
    m = a.get("n_cols")
    row.update({
        "gamma_min": verify_info.get("gamma_min"),
        "wrench_feasible": verify_info.get("wrench_feasible"),
        "n_contacts_verified": verify_info.get("n_contacts_verified"),
        "l_bar": (b_rep * m) if (b_rep is not None and m) else None,
        "l_bar_converged": (b_relp * m) if (b_relp is not None and m) else None,
        "k_l": k_l if arm == "frogger" else None,
        "mu": float(mu),
        **a,
    })
    # Drop the bulky arrays from the persisted row; they are re-derivable.
    for k in ("gws_W", "gws_alpha"):
        row.pop(k, None)
    return row, ctx


def write_artifacts(model, data, res, row, out_dir, seed, pos):
    """Per-solve artifacts for one arm.

    DELIBERATELY NOT `write_grasp_plots`. That figure reconstructs the local
    paraboloid from the saved `quad_*` frame (kappa0/kappa1/axis0_l/axis1_l/
    grad_norm) and draws the trust region as a bound -- see
    plot_grasp_contacts._patch_points. The frogger arm sets
    use_quadratic_contact=False, so it has no patch, no trust region and no
    per-stage `quad_*` trace; the function returns None for it. Emitting it for
    one arm only would also make the two arms' figure sets non-comparable, which
    is the opposite of what a benchmark artifact is for.

    What IS common to both arms and therefore written here:
      seed<N>_planned.png   the posed scene at the planned grasp, before any
                            squeeze moves the object. Same camera for both arms,
                            so the two are readable side by side.
      seed<N>.json          the scored row, so a figure can be regenerated or
                            audited without re-solving.

    Video is NOT written: this harness is plan-only, so there is no motion to
    record and a clip would be a single held frame. It belongs with the
    execution path (the paper's shaky-pickup protocol), not here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    # Pose the robot at the planned grasp. The solve returns only the ACTUATED
    # joints, so scatter them back by index rather than assuming a contiguous block.
    if res.get("q") is not None:
        try:
            from simulation.grasp_planner_3d import _get_actuated_indices
            d_p = mj.MjData(model)
            d_p.qpos[:] = data.qpos[:]
            for idx, val in zip(_get_actuated_indices(model), res["q"]):
                d_p.qpos[idx] = val
            mj.mj_forward(model, d_p)
            png = out_dir / f"seed{seed}_planned.png"
            # Framing matches pick_and_place's planned-pose render (far-side
            # azimuth, so the arm's own links do not occlude the fingers).
            render(model, d_p, str(png), lookat=pos, dist=0.7, azim=-70, elev=-25)
            written.append(png)
        except Exception as e:
            print(f"[artifacts] planned-pose render failed: {e}")
    try:
        js = out_dir / f"seed{seed}.json"
        js.write_text(json.dumps(row, indent=2, default=str))
        written.append(js)
    except Exception as e:
        print(f"[artifacts] row write failed: {e}")
    return written


def _fmt(v, spec="%+.4f", na="--"):
    return (spec % v) if v is not None and np.isfinite(v) else na


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated subset of %s" % (ARMS,))
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--fingers", default="thumb,index,middle",
                    help="ordered role list in SLOT order. Default is the tripod: a "
                         "2-contact W is rank-5-of-6 and cannot satisfy l_bar*>=0.3 "
                         "in the sense FRoGGeR intends.")
    ap.add_argument("--k-l", type=float, default=0.3,
                    help="FRoGGeR's normalized robustness floor (7c). 0 disables it, "
                         "which isolates the objective change from the constraint.")
    ap.add_argument("--patch-normals", action="store_true",
                    help="run the frogger arm on PATCH normals instead of the SDF "
                         "gradient, isolating objective structure from normal source.")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--no-artifacts", dest="artifacts", action="store_false",
                    help="skip per-solve renders (the scored table is unaffected)")
    OP.add_out_args(ap, OP.TABLETOP)
    args = ap.parse_args()

    objects = [o for o in args.objects.split(",") if o]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            ap.error("unknown arm %r; choose from %s" % (a, ARMS))
    fingers = parse_fingers(args.fingers)

    rows = []
    for obj in objects:
        for sd in seeds:
            for arm in arms:
                print(f"\n===== {arm} | {obj} seed {sd} =====", flush=True)
                try:
                    r, ctx = plan_one(arm, obj, sd, n_seeds=args.n_seeds,
                                      max_iter=args.max_iter, fingers=fingers,
                                      k_l=args.k_l,
                                      sdf_normals=not args.patch_normals)
                    if args.artifacts:
                        # out/tabletop/<tag>/<arm>/<object>/ -- the arm level keeps
                        # the two methods' artifacts from overwriting each other
                        # while staying inside one run's tree.
                        od = OP.resolve_out(args, OP.TABLETOP, arm=arm) / obj
                        for w in write_artifacts(ctx["model"], ctx["data"],
                                                 ctx["res"], r, od, sd, ctx["pos"]):
                            print(f"[artifacts] {w}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    r = dict(arm=arm, object=obj, seed=sd, status=f"ERROR: {e}")
                rows.append(r)

    hdr = ("\n%-9s%-17s%3s %-12s%9s%9s%9s%9s%10s%9s" %
           ("arm", "object", "sd", "status", "l_bar*", "l_bar_c", "lp_gap",
            "delta", "gamma_min", "t(s)"))
    print(hdr)
    print("-" * (len(hdr) + 4))
    for r in rows:
        gm = r.get("gamma_min")
        print("%-9s%-17s%3s %-12s%9s%9s%9s%9s%10s%9s" % (
            r.get("arm", "?"), r["object"], r.get("seed", "?"),
            str(r.get("status"))[:12],
            _fmt(r.get("l_bar")), _fmt(r.get("l_bar_converged")),
            _fmt(r.get("lp_gap")), _fmt(r.get("beta_delta")),
            ("%.3f" % gm) if gm is not None else "INFEAS",
            "%.1f" % r.get("t_solve_s", float("nan"))))

    print("\n--- per-arm summary (solves with a plan) ---")
    for arm in arms:
        g = [r for r in rows if r.get("arm") == arm and r.get("l_bar") is not None]
        if not g:
            print("  %-9s no successful plans" % arm)
            continue
        lb = [r["l_bar"] for r in g]
        wf = sum(1 for r in g if r.get("wrench_feasible"))
        lg = [abs(r["lp_gap"]) for r in g if r.get("lp_gap") is not None]
        ts = [r["t_solve_s"] for r in g]
        print("  %-9s n=%2d  median l_bar*=%+.4f  wrench-feasible %d/%d  "
              "median |lp_gap|=%.5f  median t=%.1fs"
              % (arm, len(g), float(np.median(lb)), wf, len(g),
                 float(np.median(lg)) if lg else float("nan"),
                 float(np.median(ts))))
    n_fail = sum(1 for r in rows if r.get("plan_failed"))
    if n_fail:
        print("  (%d solve(s) returned no plan)" % n_fail)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2, default=str))
        print("wrote %s" % args.json_out)


if __name__ == "__main__":
    main()

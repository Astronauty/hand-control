"""Audit the NLP's reported min-weight beta against the TRUE-normal beta.

Plan-only (no execution): for each object/seed it runs the same planning path
`pick_and_place.run_pick_place` uses -- same scene, same settle, same
`for_gws_recommender` preset -- then recomputes beta from the object's true surface
normals at the solved contact points via `simulation.beta_audit`.

The question this answers is the one `_embed_gws_ca`'s docstring raises and leaves
open: the NLP's beta is the min-weight LP's optimum on a wrench matrix assembled from
the QUADRATIC PATCH's symbolic normals, and on a measured 17% of solves it disagrees
with the geometry. This prints both numbers side by side, per solve, so the
disagreement is a column rather than an anecdote.

    uv run python -m ycb_grasp.beta_sweep --objects 036_wood_block,017_orange --seeds 0,1

Columns:
  beta_rep    what the NLP reported (res['gws_beta'])
  beta_true   the same LP re-solved on true normals at the same points
  delta       beta_rep - beta_true; positive = the NLP was OPTIMISTIC
  span_marg   pure geometric closure test on true normals (n=2); < 0 = closure
              is geometrically impossible at this mu, whatever beta says
  gamma_min   the wrench certificate of record (verify()); None = LP infeasible
  !!          flagged when beta_rep > 0.01 but the geometry says no closure
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco as mj                                          # noqa: E402

from ycb_grasp import table_scene as TS                       # noqa: E402
from ycb_grasp.ik_demo import clearance_by_geom, robot_geom_names   # noqa: E402
from simulation.beta_audit import audit                       # noqa: E402
from simulation.grasp_config_builder import (                 # noqa: E402
    for_gws_recommender, load_seed_config)
from simulation.grasp_planner_3d import (                     # noqa: E402
    MultiStartGraspPlanner3D, _contact_friction)

DEFAULT_OBJECTS = ["036_wood_block", "017_orange", "014_lemon",
                   "056_tennis_ball", "009_gelatin_box"]


def plan_one(object_id, seed, n_seeds=3, max_iter=80, fingers=None):
    """Plan (not execute) one grasp and audit its beta. Mirrors run_pick_place's
    planning half: same TS.build/settle, same preset, same MultiStart seeding."""
    from ycb_grasp.pick_and_place import (
        DEFAULT_COL_CLEARANCE_M, NCF_ACCEL_BUDGET_XYZ, NCF_ANG_ACCEL_BUDGET)

    model, data, info = TS.build([object_id])
    body_name = next(iter(info))
    TS.settle(model, data)
    pos, _ = TS.object_pose(model, data, body_name, info)

    q_home = TS.home_qpos()
    rgeoms = robot_geom_names(model)
    obj_geom0 = TS.hull_geoms(model, body_name)[0]

    # Mirrors run_pick_place's cfg_kw for the terms that shape the SOLVE. The
    # execution-only knobs (contact profile, jog speeds, squeeze gains) are absent
    # because this harness never executes.
    cfg_kw = dict(n_seeds=n_seeds, max_iter=max_iter,
                  obj_geom=obj_geom0,
                  col_clearance_m=DEFAULT_COL_CLEARANCE_M,
                  use_quadratic_contact=True,
                  quadratic_mesh_fit=True,
                  ground_z=TS.TABLE_TOP_Z)
    for k, v in load_seed_config(object_id).items():
        cfg_kw.setdefault(k, v)

    cfg = for_gws_recommender(body_name, rgeoms, clearance_by_geom(rgeoms),
                              accel_budget_xyz=NCF_ACCEL_BUDGET_XYZ,
                              ang_accel_budget_xyz=NCF_ANG_ACCEL_BUDGET,
                              fingers=fingers, **cfg_kw)

    planner = MultiStartGraspPlanner3D(model, data, cfg, seed=seed)
    res = planner.solve(np.asarray(q_home, float), np.asarray(pos, float),
                        max_seeds=n_seeds)
    if res.get("p1") is None:
        return {"object": object_id, "seed": seed, "status": "PLAN_FAILED"}

    verify_info = planner._planner.verify(res)
    inner = planner._planner

    # The object's pose at solve time, in the frame _geom_normal_np expects.
    d_v = mj.MjData(model)
    d_v.qpos[:] = data.qpos[:]
    mj.mj_forward(model, d_v)
    if inner._obj_geom_type == 7:   # mesh
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

    return {
        "object": object_id, "seed": seed,
        "status": res.get("status"),
        "mu": float(mu),
        "gamma_min": verify_info.get("gamma_min"),
        "wrench_feasible": verify_info.get("wrench_feasible"),
        "span_margin_final": res.get("span_margin_final"),
        **a,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--fingers", default=None,
                    help="comma-separated role list, e.g. thumb,index,middle")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    objects = [o for o in args.objects.split(",") if o]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    fingers = args.fingers.split(",") if args.fingers else None

    rows = []
    for obj in objects:
        for sd in seeds:
            print(f"\n===== {obj} seed {sd} =====", flush=True)
            try:
                r = plan_one(obj, sd, n_seeds=args.n_seeds,
                             max_iter=args.max_iter, fingers=fingers)
            except Exception as e:
                import traceback
                traceback.print_exc()
                r = {"object": obj, "seed": sd, "status": f"ERROR: {e}"}
            rows.append(r)

    hdr = (f"\n{'object':<20}{'sd':>3}{'n':>3}{'beta_rep':>11}{'beta_true':>11}"
           f"{'delta':>10}{'span_marg':>11}{'gamma_min':>11}{'wf':>6}  !!")
    print(hdr)
    print("-" * (len(hdr) + 2))
    n_contra = 0
    for r in rows:
        if "beta_true" not in r:
            print(f"{r['object']:<20}{r['seed']:>3}  {r.get('status')}")
            continue
        br, bt = r.get("beta_reported"), r.get("beta_true")
        dl, sm = r.get("beta_delta"), r.get("span_margin")
        gm, wf = r.get("gamma_min"), r.get("wrench_feasible")
        flag = "!!" if r.get("contradiction") else ""
        n_contra += bool(r.get("contradiction"))
        print(f"{r['object']:<20}{r['seed']:>3}{r['n_contacts']:>3}"
              f"{(f'{br:+.5f}' if br is not None else '--'):>11}"
              f"{(f'{bt:+.5f}' if bt is not None else '--'):>11}"
              f"{(f'{dl:+.5f}' if dl is not None else '--'):>10}"
              f"{(f'{sm:+.4f}' if sm is not None else '--'):>11}"
              f"{(f'{gm:.3f}' if gm is not None else 'INFEAS'):>11}"
              f"{str(wf):>6}  {flag}")
    n_ok = sum(1 for r in rows if "beta_true" in r)
    print(f"\n{n_contra}/{n_ok} solves report beta > 0.01 on geometry that cannot close.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

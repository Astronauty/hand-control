"""FRoGGeR vs this solver, EXECUTED: the paper's Table I on our scene.

`frogger_bench.py` is plan-only and scores `l_bar*`. That is a PROXY, and the
paper's own data shows it is a noisy one -- median 0.61 for successes against 0.47
for failures, distributions that overlap heavily. FRoGGeR's headline claim is
**78.8% pick success against a 58.0% baseline**, which no plan-only number can
speak to. This harness closes that gap: it plans with either arm, EXECUTES the
grasp through `pick_and_place`'s measured approach/settle/gap-gate/squeeze path,
and runs FRoGGeR's shaky pickup as the lift.

WHAT THIS ADDS OVER `frogger_bench`:

  % pick success   their Sec. IV criteria -- >30 deg object rotation or >7.5 cm
                   deviation from the pick trajectory fails the run. The column
                   their claim actually lives in.
  epsilon          Ferrari-Canny, the classical quality they SCORE with (as
                   distinct from `l_bar*`, the relaxation they OPTIMIZE). Scored
                   from the same contact geometry the gamma certificate uses.
                   See simulation/epsilon_metric.py.
  lift_ok          this benchmark's own gate (rose >= 80% of commanded AND every
                   fingertip still loaded at the end). Reported ALONGSIDE their
                   criteria, never instead: their test can pass on a grasp that
                   never loaded the fingers, since an object that barely moves
                   neither rotates nor deviates.

HOW THE ARMS SHARE AN EXECUTION PATH. `run_pick_place` previously hardcoded
`for_gws_recommender` and a HOME start, so only `ours` was executable. It now
takes a `plan_override` hook supplying (cfg, q_start); `_arm_override` below
builds the frogger preset through the SAME `_build_cfg` that `frogger_bench`
uses, and seeds it from `_frogger_seed`. Everything after the solve -- gamma,
squeeze ramp, lift, scoring -- is shared by construction, which is what makes the
pick-success difference attributable to the grasp rather than to the executor.

READ BEFORE QUOTING ANY NUMBER FROM THIS (all three are recorded in
docs/FROGGER_STATUS.md and docs/FROGGER_COMPARISON.md, not discovered here):

  * mu. The scene default is 2.0; FRoGGeR simulates at 0.7. `l_bar*` is strongly
    friction-dependent (a 120-degree splay closes at 2.0 and does not at 0.7), so
    results at the default are a BETWEEN-ARM comparison only and are NOT
    comparable to the paper's table. Pass --mu 0.7 for their regime, at the cost
    of comparability with every existing tabletop result.
  * n = 2. The paper uses four Allegro fingers. Our n>=3 path certifies
    wrench-feasible but executes 0/15 with contact 3 landing coincident with
    contact 2, so a three-contact run here would measure that defect rather than
    the formulation. Results are two-contact and must be labelled so.
  * solve time. FROGGER_COMPARISON sec 6 measures our per-solve cost as dominated
    by finite-differenced collision gradients (504,630 evaluations against the
    LP's 589), not by the formulation. `--equal-budget` gives both arms the same
    synthesis budget so the TOTAL-time column is at least self-consistent, but
    neither column should be read against the paper's 0.83 s.

Usage:
    # pilot: 2 objects x 3 seeds, both arms, their execution test
    uv run python -m ycb_grasp.frogger_exec_bench \\
        --objects 017_orange,036_wood_block --seeds 0,1,2

    # the paper's friction regime
    uv run python -m ycb_grasp.frogger_exec_bench --mu 0.7

IMPORTANT (FROGGER_BENCH sec 8.7): do not run two sweeps concurrently and do not
edit planner source while one is in flight. Treat any overlapped run as void.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ycb_grasp import out_paths as OP                               # noqa: E402
from ycb_grasp import table_scene as TS                             # noqa: E402
from ycb_grasp.frogger_bench import (ARMS, DEFAULT_OBJECTS,         # noqa: E402
                                     EXCLUDED_OBJECTS, SYNTH_BUDGET_S,
                                     _build_cfg, _frogger_seed,
                                     _frogger_synthesize)
from simulation.grasp_planner_3d import MultiStartGraspPlanner3D  # noqa: E402
from ycb_grasp.ik_demo import robot_geom_names                      # noqa: E402
from ycb_grasp.pick_and_place import run_pick_place, _fmt_eps       # noqa: E402
from simulation.grasp_config_builder import parse_fingers           # noqa: E402

# Two objects with different failure geometry: a sphere (curvature everywhere, the
# case the patch fit handles cleanly) and a box (flat faces plus creases, where
# edge-seeking and the SDF-Hessian curvature problem both live). A pilot wants the
# two ends of that axis rather than two of the same shape.
PILOT_OBJECTS = ["017_orange", "036_wood_block"]


def _arm_override(arm, *, k_l, sdf_normals, max_attempts, budget_s,
                  sdf_surface=True):
    """Build the `plan_override` hook for one arm, or None for `ours`.

    None is not a stub: `ours` IS `run_pick_place`'s own default path (the
    `for_gws_recommender` preset from HOME), and routing it through an override
    that reconstructed the same preset would risk the two drifting apart. The
    override exists for arms that differ from that default.
    """
    if arm == "ours":
        return None

    def _override(*, body_name, model, data, info, cfg_kw, fingers, seed, pos,
                  q_home):
        # Same cfg construction as the plan-only harness, so an executed frogger
        # grasp and a planned one are the same grasp.
        rgeoms = robot_geom_names(model)
        obj_geom0 = TS.hull_geoms(model, body_name)[0]
        cfg = _build_cfg(arm, _override.object_id, body_name, rgeoms, obj_geom0,
                         n_seeds=cfg_kw.get("n_seeds", 1),
                         max_iter=cfg_kw.get("max_iter", 80),
                         fingers=fingers, k_l=k_l, sdf_normals=sdf_normals,
                         sdf_surface=sdf_surface)
        # FRoGGeR's SYNTHESIS LOOP (Sec. IV / Table I), not a single draw. Their
        # convergence rate is a property of resample-until-feasible: a run converges
        # when it yields a grasp clearing the k_l floor inside 60 s, at a median of
        # 3 solves (IQR 1-6). A single attempt is not their protocol and understates
        # the method by construction.
        #
        # This must happen HERE rather than in run_pick_place, which solves exactly
        # once from whatever start pose it is handed. So the loop runs to completion
        # and returns the ACCEPTED attempt's q0; run_pick_place's own solve then
        # re-solves from that seed, landing on the same grasp the loop accepted.
        #
        # Their sampler is where opposition lives (App. C step 1: palm y-axis to an
        # OBB axis, fingers pre-opened to that edge's width), and the paper reports
        # the method is "highly sensitive to the sampled initial conditions" -- so
        # resampling is load-bearing, not a retry convenience.
        roles = list(fingers or ["thumb", "index"])
        if max_attempts > 1:
            planner = MultiStartGraspPlanner3D(model, data, cfg, seed=seed)
            res, sinfo = _frogger_synthesize(
                planner, model, data, info, body_name,
                np.asarray(q_home, float), roles, seed, pos,
                cfg_kw.get("n_seeds", 1), k_l=k_l,
                max_attempts=max_attempts, budget_s=budget_s)
            _override.seed_info = dict(sinfo)
            # Hand back the ACCEPTED result, not a seed to re-solve from. An
            # earlier version re-drew the seed by attempt index and let
            # run_pick_place solve again; that does not reproduce the accepted
            # grasp (measured contacts 345-484 mm from the hand against an
            # accepted attempt that had them on the object), because the solve is
            # not a pure function of the draw index -- the planner's own restart
            # RNG has advanced. Returning the result makes the executed grasp
            # exactly the one the loop certified.
            return cfg, None, res
        q0, sinfo = _frogger_seed(model, data, info, body_name,
                                  np.asarray(q_home, float), roles, seed)
        _override.seed_info = dict(sinfo)
        return cfg, q0

    _override.seed_info = {}
    _override.object_id = None
    return _override


def run_one(arm, object_id, seed, *, fingers, k_l, sdf_normals, mu,
            max_attempts, budget_s, out_dir, lift_mode, do_transport,
            gap_tol_m=None, sdf_surface=True):
    """Plan + execute one grasp with one arm. Returns the scored row."""
    ov = _arm_override(arm, k_l=k_l, sdf_normals=sdf_normals,
                       max_attempts=max_attempts, budget_s=budget_s,
                       sdf_surface=sdf_surface)
    if ov is not None:
        ov.object_id = object_id

    t0 = time.time()
    res, result = run_pick_place(
        object_id, seed,
        fingers=",".join(fingers) if fingers else None,
        out_dir=out_dir, do_transport=do_transport,
        lift_mode=lift_mode, plan_override=ov,
        # The frogger arm's pad-offset constraint parks the pad further out by
        # construction (see run_pick_place's gate comment), so it gets a tolerance
        # sized for ITS convention. `ours` keeps the 8 mm default every existing
        # tabletop result was measured at.
        gap_tol_m=(gap_tol_m if arm != "ours" else None))
    t_total = time.time() - t0

    row = dict(arm=arm, object=object_id, seed=seed, t_total_s=round(t_total, 2))
    row.update({k: v for k, v in result.items() if k != "phase_log"})
    row["phase_log"] = ",".join(result.get("phase_log", []))
    if ov is not None:
        row.update(ov.seed_info)
    # l_bar*: the plan-only harness's headline, recomputed here so one row carries
    # both the metric and the execution outcome -- the pairing the paper's noisy-
    # predictor claim (0.61 vs 0.47) is about.
    b = res.get("gws_beta")
    W = res.get("gws_W")
    m = (np.asarray(W, float).shape[1] if W is not None else None)
    row["l_bar"] = (float(b) * m) if (b is not None and m) else None
    return row


def _median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def _summarize(rows, arms):
    """Per-arm summary in the shape of the paper's Table I."""
    print("\n--- per-arm summary (the paper's Table I columns) ---")
    print("%-9s %8s %14s %14s %10s %10s" %
          ("arm", "n", "pick success", "lift_ok", "eps(x1e3)", "l_bar*"))
    for arm in arms:
        g = [r for r in rows if r.get("arm") == arm]
        if not g:
            print("  %-9s no runs" % arm)
            continue
        # Pick success is scored over runs that REACHED the lift. A plan failure
        # is a separate outcome (their "converged" column), and folding it into
        # the success rate would conflate two different claims.
        lifted = [r for r in g if "lift_done" in (r.get("phase_log") or "")]
        ps = [r for r in lifted if r.get("pick_success")]
        lo = [r for r in lifted if r.get("lift_ok")]
        eps = _median([r.get("epsilon") for r in g])
        lb = _median([r.get("l_bar") for r in g])
        print("%-9s %8d %14s %14s %10s %10s" % (
            arm, len(g),
            ("%d/%d" % (len(ps), len(lifted))) if lifted else "--",
            ("%d/%d" % (len(lo), len(lifted))) if lifted else "--",
            _fmt_eps(eps).replace("e-3", ""),
            ("%+.4f" % lb) if lb is not None else "--"))
    # Failure-mode breakdown: their criteria name WHICH way a pick failed, which
    # is the diagnostic half of the column.
    reasons = {}
    for r in rows:
        if r.get("pick_fail_reason"):
            reasons[r["pick_fail_reason"]] = reasons.get(r["pick_fail_reason"], 0) + 1
    if reasons:
        print("\n  pick failure modes: " +
              ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    n_deg = sum(1 for r in rows if r.get("epsilon_degenerate"))
    if n_deg:
        print(f"  epsilon degenerate (rank<6, expected at n=2): {n_deg}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", default=",".join(PILOT_OBJECTS),
                    help="comma-separated YCB ids. Default is the 2-object pilot; "
                         "pass '%s' for the plan-only harness's full set."
                         % ",".join(DEFAULT_OBJECTS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--fingers", default="thumb,index",
                    help="ordered role list in SLOT order. Default is the PINCH: "
                         "the n>=3 path executes 0/15 (contact 3 lands coincident "
                         "with contact 2), so a tripod run here would measure that "
                         "defect, not the formulation.")
    ap.add_argument("--k-l", type=float, default=0.3,
                    help="FRoGGeR's normalized robustness floor (7c). 0 disables it.")
    ap.add_argument("--patch-normals", action="store_true",
                    help="run the frogger arm on PATCH normals instead of the SDF "
                         "gradient, isolating objective structure from normal source.")
    ap.add_argument("--mu", type=float, default=None,
                    help="object-geom sliding friction. Default None keeps "
                         "table_scene's 2.0. Pass 0.7 for the paper's regime -- see "
                         "the module docstring on why results are not comparable "
                         "across different values.")
    ap.add_argument("--max-attempts", type=int, default=1,
                    help="FRoGGeR resamples until feasible (median 3 solves, IQR "
                         "1-6). Default 1 is a single solve.")
    ap.add_argument("--lift-mode", choices=["shaky", "standard"], default="shaky",
                    help="shaky (default) is FRoGGeR's execution test and the point "
                         "of this harness; standard runs the 12cm lift instead.")
    ap.add_argument("--transport", dest="do_transport", action="store_true",
                    help="also carry to the bin after the lift. OFF by default: the "
                         "paper's test ends at the hold, and the carry adds a phase "
                         "their criteria say nothing about.")
    ap.add_argument("--patch-contacts", action="store_true",
                    help="run the frogger OBJECTIVE and floor on OUR quadratic-patch "
                         "contact parameterization instead of their (7d) FK contacts. "
                         "This separates their formulation from our implementation "
                         "of it: the fixed body-frame pad point is what leaves the "
                         "pad 8-11 mm clear of the object, so removing it while "
                         "keeping `max l*` and k_l says whether the execution "
                         "failures are theirs or ours. NOT the faithful port.")
    ap.add_argument("--gap-tol-m", type=float, default=0.014,
                    help="pre-squeeze fingertip-gap tolerance for the NON-'ours' "
                         "arms, metres. The 8 mm default is sized for this solver's "
                         "r_tip convention; FRoGGeR's (7d) pins a fixed body-frame "
                         "pad point, which leaves the real pad 8-11 mm clear when "
                         "the contact is off that axis, so the default rejects "
                         "grasps that are otherwise sound. The squeeze and the "
                         "post-lift force test still have to pass. `ours` always "
                         "keeps the 8 mm default.")
    ap.add_argument("--json-out", default=None)
    OP.add_out_args(ap, OP.TABLETOP)
    args = ap.parse_args()

    if args.mu is not None:
        print(f"[note] mu={args.mu} overrides the scene default; results are NOT "
              f"comparable to tabletop numbers measured at 2.0.")
        # table_scene.build's friction is applied per-run inside run_pick_place,
        # which does not currently take a friction argument. Fail loudly rather
        # than silently running at the default and labelling it 0.7.
        raise SystemExit(
            "--mu is not yet plumbed through run_pick_place (it builds the scene "
            "itself). Run the plan-only harness for the paper's friction regime:\n"
            "  uv run python -m ycb_grasp.frogger_bench --mu 0.7")

    objects = [o for o in args.objects.split(",") if o]
    for _o in objects:
        if _o in EXCLUDED_OBJECTS:
            print(f"[note] {_o} is excluded by default ({EXCLUDED_OBJECTS[_o]}); "
                  f"running it because it was named explicitly.")
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
                od = OP.resolve_out(args, OP.TABLETOP, arm=arm) / obj
                od.mkdir(parents=True, exist_ok=True)
                try:
                    r = run_one(arm, obj, sd, fingers=fingers, k_l=args.k_l,
                                sdf_normals=not args.patch_normals, mu=args.mu,
                                max_attempts=args.max_attempts,
                                budget_s=SYNTH_BUDGET_S, out_dir=str(od),
                                lift_mode=args.lift_mode,
                                do_transport=args.do_transport,
                                gap_tol_m=args.gap_tol_m,
                                sdf_surface=not args.patch_contacts)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    r = dict(arm=arm, object=obj, seed=sd, status=f"ERROR: {e}")
                rows.append(r)
                try:
                    (od / f"seed{sd}_exec.json").write_text(
                        json.dumps(r, indent=2, default=str))
                except Exception as e:
                    print(f"[artifacts] row write failed: {e}")

    hdr = ("\n%-9s%-17s%3s %-9s%9s%11s%9s%9s" %
           ("arm", "object", "sd", "pick", "l_bar*", "eps(x1e3)", "lift_ok", "t(s)"))
    print(hdr)
    print("-" * (len(hdr) + 4))
    for r in rows:
        lb = r.get("l_bar")
        print("%-9s%-17s%3s %-9s%9s%11s%9s%9s" % (
            r.get("arm", "?"), r.get("object", "?"), r.get("seed", "?"),
            ("OK" if r.get("pick_success") else
             (r.get("pick_fail_reason") or "--"))[:9],
            ("%+.4f" % lb) if lb is not None else "--",
            _fmt_eps(r.get("epsilon")).replace("e-3", ""),
            str(r.get("lift_ok", "--")),
            "%.0f" % r.get("t_total_s", float("nan"))))

    _summarize(rows, arms)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2, default=str))
        print("wrote %s" % args.json_out)


if __name__ == "__main__":
    main()

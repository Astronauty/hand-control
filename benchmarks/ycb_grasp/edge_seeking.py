"""Edge-seeking: how close do solved contacts sit to a real surface boundary?

FRoGGeR names edge-seeking as the dominant failure mode of BOTH their method and
their baseline, attributes it to the epsilon metric preferring large moment arms,
observes it "yields unstable grasps in practice", and lists combating it as future
work ("In the future, we hope to develop methods to combat edge-seeking behavior").
They give **no metric for it**. This module supplies one.

THE MEASUREMENT. Each contact lives at coordinates `t` on a fitted quadratic patch
whose trust region was sized, per axis, by walking outward until the true SDF departs
from the surrogate by more than `quadratic_sdf_err_tol`. That bound is the patch's
own statement of where the locally-smooth surface STOPS -- i.e. an edge. So

    edge_margin(contact) = min over axes of (t_bound_raw[i] - |t[i]|)

is the distance, in metres along the surface, from the contact to the nearest
measured surface boundary. Small = perched near an edge; large = out on open face.

WHY THE **RAW** BOUND, AND WHY ONLY DIVERGENCE-LIMITED AXES. Two traps, both of
which produce a metric that looks fine and means nothing:

  * `quadratic_bound_inset` shaves a constant (10 mm) off every side AFTER the
    search sizes it. Measuring against the inset bound reports the inset, not the
    surface -- every contact would appear to sit exactly `inset` from its "edge".
    `t_bound_raw_{i}` is the pre-inset value and is what the surface actually said.
  * An axis that ran the entire search without ever exceeding tolerance comes back
    at `quadratic_t_bound_max`, the hard cap. That means "flat as far as we looked",
    NOT "edge here". Counting it would report the middle of a large flat face as
    edge-adjacent -- precisely the case `w_edge_margin` exists to permit. Such axes
    are EXCLUDED, exactly as the solver's own hinge excludes them
    (grasp_planner_3d.py, the `_tb_raw >= _cap` test).

A contact with no divergence-limited axis at all is reported as `None` (on open
face, no measurable edge nearby) and is counted separately rather than folded in as
a large number, which would flatter whichever arm finds more flat faces.

WHAT THIS CAN AND CANNOT SHOW. It is a property of the SOLVED CONTACT against the
patch geometry, computed identically for both arms from data each solve already
records (`quad{1,2,3}_frame`, `t{1,2,3}_sol`). Nothing here is re-derived or
re-fitted, so the two arms cannot differ by how the measurement was taken.

It does NOT by itself show that edge-seeking causes failure. Pair it with execution
outcomes (frogger_exec_bench) to test that, which is the comparison the paper's
"yields unstable grasps in practice" claim actually needs.

Usage:
    uv run python -m ycb_grasp.edge_seeking --objects 017_orange,036_wood_block \\
        --seeds 0,1,2 --arms ours,frogger

    # the w_edge_margin ablation: is the term that exists for this any use?
    uv run python -m ycb_grasp.edge_seeking --arms ours --w-edge-margin 0,2.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ycb_grasp import out_paths as OP                            # noqa: E402
from ycb_grasp import table_scene as TS                          # noqa: E402
from ycb_grasp.frogger_bench import (ARMS, DEFAULT_OBJECTS,      # noqa: E402
                                     _build_cfg, _frogger_seed)
from ycb_grasp.ik_demo import robot_geom_names                   # noqa: E402
from simulation.grasp_config_builder import parse_fingers        # noqa: E402
from simulation.grasp_planner_3d import MultiStartGraspPlanner3D  # noqa: E402

# Matches grasp_planner_3d's own cap test for "this axis never diverged".
_CAP_TOL = 1e-9

# Representation-free edge search (see `geodesic_edge_margin`).
_EDGE_TURN_DEG = 30.0      # normal turn that counts as "an edge is here"
_EDGE_MAX_MM = 40.0        # stop looking this far out; beyond it, open face
_EDGE_STEP_MM = 1.0
_EDGE_N_DIRS = 16


def geodesic_edge_margin(p_world, obj_center, obj_R, mesh_entry,
                         turn_deg=_EDGE_TURN_DEG, max_mm=_EDGE_MAX_MM,
                         step_mm=_EDGE_STEP_MM, n_dirs=_EDGE_N_DIRS):
    """Distance (m) from a contact to the nearest edge, measured on the MESH.

    WHY THIS EXISTS ALONGSIDE `contact_edge_margins`. That one reads the patch's
    own trust-region bounds, which is exact and free -- but it only exists for arms
    that HAVE a patch. The frogger arm sets `frogger_fk_contacts=True`: its contacts
    are forward-kinematics outputs pinned to the surface by (7d), with no local
    parameterization and therefore no bounds. Measured: `quad1_frame` is None on
    15/15 frogger solves. Reporting that as "no edge nearby" would credit their arm
    for a measurement that was never taken, which is the opposite of the truth.

    This metric asks the surface directly and is identical for both arms: from the
    contact, step outward along `n_dirs` tangent directions, reprojecting to the
    surface each step, until the surface NORMAL has turned by more than `turn_deg`
    from its value at the contact. The distance travelled when that first happens,
    minimized over directions, is the margin. A crease turns the normal sharply
    within a step or two; a smooth face turns it slowly or not at all.

    Returns None if no direction reaches `turn_deg` within `max_mm` -- the contact
    is on open face with no edge in range, reported separately rather than as a
    large number.
    """
    fn, gfn = mesh_entry.get("fn"), mesh_entry.get("grad_fn")
    proj = mesh_entry.get("project_fn_short")
    if fn is None or gfn is None or proj is None:
        return None
    # World -> object-local, the frame the mesh functions are defined in.
    p_l = np.asarray(obj_R, float).T @ (np.asarray(p_world, float)
                                        - np.asarray(obj_center, float))

    def _n(p):
        g = np.asarray(gfn(np.asarray(p, float)), float).reshape(3)
        nrm = np.linalg.norm(g)
        return None if nrm < 1e-12 else g / nrm

    n0 = _n(p_l)
    if n0 is None:
        return None
    # Tangent basis at the contact.
    a = np.array([0.0, 0.0, 1.0]) if abs(n0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n0, a)
    t1 /= max(np.linalg.norm(t1), 1e-12)
    t2 = np.cross(n0, t1)

    cos_lim = float(np.cos(np.radians(turn_deg)))
    best = None
    for k in range(int(n_dirs)):
        th = 2.0 * np.pi * k / float(n_dirs)
        d = np.cos(th) * t1 + np.sin(th) * t2
        p = p_l.copy()
        travelled = 0.0
        while travelled < max_mm * 1e-3:
            p = p + d * (step_mm * 1e-3)
            try:
                p = np.asarray(proj(p), float).reshape(3)    # back onto the surface
            except Exception:
                break
            travelled += step_mm * 1e-3
            nk = _n(p)
            if nk is None:
                break
            if float(nk @ n0) < cos_lim:
                best = travelled if best is None else min(best, travelled)
                break
        if best is not None and best <= step_mm * 1e-3:
            break                                            # cannot do better
    return best


def contact_edge_margins(res, cfg):
    """Per-contact distance (m) to the nearest MEASURED surface boundary.

    Returns a list with one entry per contact slot present: a float margin, or
    None when no axis of that contact's patch was divergence-limited (the contact
    is on open face and no edge was found within the search).
    """
    cap = float(getattr(cfg, "quadratic_t_bound_max", 0.0) or 0.0)
    out = []
    for fk, tk in (("quad1_frame", "t1_sol"),
                   ("quad2_frame", "t2_sol"),
                   ("quad3_frame", "t3_sol")):
        frame, t = res.get(fk), res.get(tk)
        if frame is None or t is None:
            continue
        t = np.asarray(t, float).reshape(-1)
        best = None
        for i in range(min(2, t.size)):
            # RAW (pre-inset) bound: what the surface said, not what the inset left.
            raw = frame.get(f"t_bound_raw_{i}")
            if raw is None:
                raw = frame.get(f"t_bound_{i}")
            if raw is None:
                continue
            raw = float(raw)
            if cap > 0.0 and raw >= cap - _CAP_TOL:
                continue                     # capped => flat, not an edge
            m = raw - abs(float(t[i]))
            best = m if best is None else min(best, m)
        out.append(best)
    return out


def _stats(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None
    v = np.asarray(v, float)
    return dict(n=int(v.size), median=float(np.median(v)),
                q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                min=float(v.min()))


def plan_one(arm, object_id, seed, *, fingers, n_seeds, max_iter, k_l,
             sdf_normals, w_edge_margin=None):
    """Plan one grasp and score its contacts' edge margins."""
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
    if w_edge_margin is not None:
        # The ablation knob. Lives on the cost group; set it where the solver reads
        # it rather than on the facade, which would silently not take.
        for holder in (cfg, getattr(cfg, "cost", None)):
            if holder is not None and hasattr(holder, "w_edge_margin"):
                setattr(holder, "w_edge_margin", float(w_edge_margin))

    q0 = np.asarray(q_home, float)
    if arm == "frogger":
        q0, _ = _frogger_seed(model, data, info, body_name, q0,
                              list(fingers or ["thumb", "index"]), seed)

    planner = MultiStartGraspPlanner3D(model, data, cfg, seed=seed)
    t0 = time.time()
    res = planner.solve(q0, np.asarray(pos, float), max_seeds=n_seeds)
    dt = time.time() - t0

    row = dict(arm=arm, object=object_id, seed=seed, t_solve_s=round(dt, 2),
               status=res.get("status"), w_edge_margin=w_edge_margin)
    if res.get("p1") is None:
        row["plan_failed"] = True
        return row

    # REPRESENTATION-FREE margin, computed for BOTH arms from the mesh. This is
    # the comparable number; the patch-bound one below exists only where a patch
    # does (see geodesic_edge_margin's docstring).
    inner = planner._planner
    geo = []
    try:
        import mujoco as _mj
        d_v = _mj.MjData(model)
        d_v.qpos[:] = data.qpos[:]
        _mj.mj_forward(model, d_v)
        if inner._obj_geom_type == 7:
            oc = d_v.xpos[inner._obj_bid].copy()
            oR = d_v.xmat[inner._obj_bid].reshape(3, 3).copy()
        else:
            oc = d_v.geom_xpos[inner._obj_gid].copy()
            oR = d_v.geom_xmat[inner._obj_gid].reshape(3, 3).copy()
        for key in ("p1", "p2", "p3"):
            p = res.get(key)
            if p is None:
                continue
            geo.append(geodesic_edge_margin(np.asarray(p, float), oc, oR,
                                            inner._mesh_entry or {}))
    except Exception as e:
        row["geodesic_error"] = repr(e)[:80]
    row["geo_margins_mm"] = [None if g is None else round(1e3 * g, 2) for g in geo]
    _gf = [g for g in geo if g is not None]
    row["min_geo_margin_mm"] = (round(1e3 * min(_gf), 2) if _gf else None)
    row["n_geo_edge_limited"] = len(_gf)

    margins = contact_edge_margins(res, cfg)
    row["edge_margins_mm"] = [None if m is None else round(1e3 * m, 2)
                              for m in margins]
    finite = [m for m in margins if m is not None]
    row["n_contacts"] = len(margins)
    row["n_edge_limited"] = len(finite)
    # The grasp's own margin is its WORST contact: a tripod with two contacts
    # mid-face and one on a crease is an edge-seeking grasp, and a mean would
    # hide that.
    row["min_edge_margin_mm"] = (round(1e3 * min(finite), 2) if finite else None)
    b, W = res.get("gws_beta"), res.get("gws_W")
    m = (np.asarray(W, float).shape[1] if W is not None else None)
    row["l_bar"] = (float(b) * m) if (b is not None and m) else None
    try:
        vi = planner._planner.verify(res)
        row["wrench_feasible"] = vi.get("wrench_feasible")
        row["gamma_min"] = vi.get("gamma_min")
    except Exception:
        pass
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="ours")
    ap.add_argument("--fingers", default="thumb,index")
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--k-l", type=float, default=0.3)
    ap.add_argument("--patch-normals", action="store_true")
    ap.add_argument("--w-edge-margin", default=None,
                    help="comma-separated w_edge_margin values to sweep, e.g. "
                         "'0,2.0'. Default: leave the preset's own value (0.0, off). "
                         "This is the ablation of the term that exists for exactly "
                         "the failure mode FRoGGeR leaves as future work.")
    ap.add_argument("--json-out", default=None)
    OP.add_out_args(ap, OP.TABLETOP)
    args = ap.parse_args()

    objects = [o for o in args.objects.split(",") if o]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            ap.error("unknown arm %r; choose from %s" % (a, ARMS))
    fingers = parse_fingers(args.fingers)
    wems = ([None] if args.w_edge_margin is None
            else [float(x) for x in args.w_edge_margin.split(",") if x != ""])

    rows = []
    for obj in objects:
        for sd in seeds:
            for arm in arms:
                for wem in wems:
                    tag = arm if wem is None else f"{arm}/w={wem:g}"
                    print(f"\n===== {tag} | {obj} seed {sd} =====", flush=True)
                    try:
                        rows.append(plan_one(arm, obj, sd, fingers=fingers,
                                             n_seeds=args.n_seeds,
                                             max_iter=args.max_iter, k_l=args.k_l,
                                             sdf_normals=not args.patch_normals,
                                             w_edge_margin=wem))
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        rows.append(dict(arm=arm, object=obj, seed=sd,
                                         w_edge_margin=wem, status=f"ERROR: {e}"))

    hdr = ("\n%-9s%-7s%-17s%3s %13s %14s %12s %9s" %
           ("arm", "w_edge", "object", "sd", "geo_edge(mm)", "patch_edge(mm)",
            "per-contact", "l_bar*"))
    print(hdr)
    print("-" * (len(hdr) + 10))
    for r in rows:
        mm = r.get("min_edge_margin_mm")
        lb = r.get("l_bar")
        gm = r.get("min_geo_margin_mm")
        print("%-9s%-7s%-17s%3s %13s %14s %12s %9s" % (
            r.get("arm", "?"),
            "--" if r.get("w_edge_margin") is None else ("%g" % r["w_edge_margin"]),
            r.get("object", "?"), r.get("seed", "?"),
            ("%.2f" % gm) if gm is not None else "no-edge",
            ("%.2f" % mm) if mm is not None else "n/a",
            ",".join("--" if x is None else "%.0f" % x
                     for x in (r.get("edge_margins_mm") or [])) or "--",
            ("%+.3f" % lb) if lb is not None else "--"))

    print("\n--- edge-margin distribution (lower = more edge-seeking) ---")
    keys = sorted({(r.get("arm"), r.get("w_edge_margin")) for r in rows},
                  key=lambda k: (str(k[0]), str(k[1])))
    for arm, wem in keys:
        g = [r for r in rows if r.get("arm") == arm and r.get("w_edge_margin") == wem]
        stg = _stats([r.get("min_geo_margin_mm") for r in g])
        if stg is not None:
            print(f"  {(arm if wem is None else f'{arm} w={wem:g}'):16s} "
                  f"GEODESIC n={stg['n']:2d}  median {stg['median']:6.2f} mm  "
                  f"IQR ({stg['q1']:.2f}, {stg['q3']:.2f})  min {stg['min']:6.2f}")
        st = _stats([r.get("min_edge_margin_mm") for r in g])
        n_noedge = sum(1 for r in g
                       if r.get("min_edge_margin_mm") is None
                       and not r.get("plan_failed"))
        lab = arm if wem is None else f"{arm} w={wem:g}"
        if st is None:
            print(f"  {lab:16s} no edge-limited contacts in {len(g)} solves")
            continue
        print(f"  {lab:16s} n={st['n']:2d}  median {st['median']:6.2f} mm  "
              f"IQR ({st['q1']:.2f}, {st['q3']:.2f})  min {st['min']:5.2f}  "
              f"| {n_noedge} solve(s) with no measured edge nearby")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2, default=str))
        print("wrote %s" % args.json_out)


if __name__ == "__main__":
    main()

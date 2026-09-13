"""One figure showing ALL contact seeds a solve considered — accepted and rejected.

The teleop counterpart of benchmarks/ycb_grasp/plot_seed_quadratic.py, and
deliberately drawn in the SAME grammar (its draw_mesh / draw_seed_rays /
draw_quadratic / quad_frame helpers are imported, not reimplemented), so a teleop
figure and a seed_quadratic figure can be read side by side.

The difference is WHERE the seeds come from. plot_seed_quadratic generates its own
seeds in its own process with its own RNG, so its picture is representative of the
seeding DISTRIBUTION but is not the seed set any particular solve used. This module
draws planner.last_seed_accept_table / last_seed_reject_table -- recorded by
MultiStartGraspPlanner3D.solve() as it gates -- so the figure is PAIRED with the
solve that produced the committed grasp: same seeds, same gates, same RNG draw.

Rejected seeds are drawn too (red, with the gate's own reason). solve() stops at
the first n_seeds ACCEPTED, so the rejects recorded are exactly those the gates
threw away on the way to this grasp -- and on an object where a gate refuses
everything, they are the whole story.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from ycb_grasp import out_paths as OP                       # noqa: E402
from ycb_grasp import plot_seed_quadratic as SQ             # noqa: E402


def _scene_from_planner(planner, model, data):
    """The `sc` dict plot_seed_quadratic's helpers expect, built from a LIVE planner
    instead of from its own build_object(). Mesh pose comes from the BODY (object_sdf
    is body-frame) exactly as verify()/solve() do -- see the frame note in
    grasp_planner_3d.verify()."""
    pl = planner._planner if hasattr(planner, "_planner") else planner
    gid = pl._obj_gid
    gtype = int(model.geom_type[gid])
    me = pl._mesh_entry
    if gtype == SQ._GEOM_TYPE_MESH:
        center = data.xpos[pl._obj_bid].copy()
        R = data.xmat[pl._obj_bid].reshape(3, 3).copy()
    else:
        center = data.geom_xpos[gid].copy()
        R = data.geom_xmat[gid].reshape(3, 3).copy()
    Vvis, Fvis = SQ.oua.body_visual_mesh(model, pl._obj_bid)
    cfg = pl.cfg
    return dict(obj=cfg.obj_body, model=model, data=data, cfg=cfg, planner=pl,
                geom_type=gtype, geom_size=model.geom_size[gid].copy(),
                mesh_entry=me, center=center, R=R, Vvis=Vvis, Fvis=Fvis,
                ground_z=float(cfg.ground_z),
                r_tip=(float(cfg.seed_ground_clearance_m)
                       if cfg.seed_ground_clearance_m is not None
                       else float(getattr(pl, "_r_tip_min", 0.0194))))


def _as_rec(entry, ok):
    """One accept/reject table row -> the `rec` shape draw_seed_rays wants.

    p3s/n3_in ride along when the row has them (a tripod seed that cleared the
    third-contact fan). Absent at n=2, and the caller then draws two contacts --
    never a fabricated third."""
    seed = dict(p1s=np.asarray(entry["p1s"], float),
                p2s=np.asarray(entry["p2s"], float),
                n1_in=np.asarray(entry["n1_in"], float),
                n2_in=np.asarray(entry["n2_in"], float))
    seed["p1"], seed["p2"] = seed["p1s"], seed["p2s"]
    if entry.get("p3s") is not None:
        seed["p3s"] = np.asarray(entry["p3s"], float)
        seed["n3_in"] = np.asarray(entry["n3_in"], float)
        seed["p3"] = seed["p3s"]
    return dict(seed=seed, kind=entry.get("kind", "random"), ok=ok,
                why=entry.get("why", "accepted"),
                ray=dict(origin=None, u=None))


def _contacts_of(rec):
    """[(role_key, point, inward_normal), ...] for whatever contacts this seed has.

    'index' is the SLOT-2 key plot_seed_quadratic colours. The NLP makes the middle
    finger share slot 2's patch (grasp_planner_3d.py _run_stage reconstructs p3 from
    _t2_frame), so no second patch is drawn -- but the SEED does not respect that
    sharing, and the panel says so rather than hiding it. See the in/out-of-bounds
    annotation in _draw_block: measured 47-59mm index-to-middle seed separation
    against a ~19mm patch half-extent on 017_orange, i.e. every third seed starts
    2-3x OUTSIDE the region it is later confined to."""
    s_ = rec["seed"]
    out = [("thumb", s_["p1s"], s_["n1_in"]), ("index", s_["p2s"], s_["n2_in"])]
    if s_.get("p3s") is not None:
        out.append(("middle", s_["p3s"], s_["n3_in"]))
    return out


_MIDDLE_COLOR = "#238b45"   # green -- distinct from thumb orange / index blue


def _draw_block(fig, gs, col0, recs, sc, header, ok):
    """One column block (all accepted, or all rejected), two rows deep."""
    for j, rec in enumerate(recs):
        c = col0 + j
        axr = fig.add_subplot(gs[0, c], projection="3d")
        SQ.draw_mesh(axr, sc)
        try:
            SQ.draw_seed_rays(axr, sc, rec)
        except Exception:
            for key, p, _n in _contacts_of(rec):
                col = (_MIDDLE_COLOR if key == "middle"
                       else SQ.FINGER_COLORS[key]) if ok else "#cb181d"
                axr.scatter(*p, s=34, color=col, depthshade=False, zorder=10)
            axr.plot(*np.array([rec["seed"]["p1s"], rec["seed"]["p2s"]]).T,
                     color="0.35" if ok else "#cb181d", lw=1.1, ls="--", zorder=9)
        if rec["seed"].get("p3s") is not None:
            # Middle-finger seed, drawn on the ray panel too so its position
            # relative to the pinch axis is legible at a glance.
            axr.scatter(*rec["seed"]["p3s"], s=42, marker="^",
                        color=_MIDDLE_COLOR if ok else "#cb181d",
                        edgecolor="k", linewidths=0.4, depthshade=False, zorder=11)
        axr.set_title(f"{rec['kind']} — {'accepted' if ok else rec['why']}",
                      fontsize=8.5, color="0.15" if ok else "#cb181d")
        # WORLD frame. draw_mesh/draw_seed_rays both draw in world, so the limits
        # must be set from the world vertex cloud -- passing the BODY-LOCAL Vvis
        # (its y spans [-0.055,0.019] while the world mesh sits at [0.243,0.317])
        # leaves limits and content disjoint and the panel renders EMPTY.
        SQ._equal_axes(axr, sc["center"] + sc["Vvis"] @ sc["R"].T)
        axr.set_axis_off()

        axp = fig.add_subplot(gs[1, c], projection="3d")
        # MESH UNDER THE PATCH. This call is what the sweep figures were missing:
        # without it the paraboloids float with no surface beneath them. Safe to
        # add because _equal_axes turns autoscale OFF before setting limits, so
        # the full-object shell cannot re-expand a patch-local zoom (the same
        # reason plot_grasp_contacts draws the mesh in every zoomed panel).
        SQ.draw_mesh(axp, sc, alpha=0.16)
        pts = []
        _patch_by_key = {}
        for key, p, nin in _contacts_of(rec):
            if key == "middle":
                continue          # shares slot 2's patch; drawn as a marker below
            try:
                frame = SQ.quad_frame(sc, np.asarray(p, float),
                                      np.asarray(nin, float), sc["cfg"])
            except Exception:
                frame = None
            if frame is None:
                continue
            _patch_by_key[key] = frame
            SQ.draw_quadratic(axp, sc, frame, key)
            pts.append(SQ.patch_points(frame, sc["center"], sc["R"],
                                       (frame["t_lo_0"], frame["t_hi_0"]),
                                       (frame["t_lo_1"], frame["t_hi_1"]),
                                       n=7).reshape(-1, 3))
        # Both slot-2 fingers marked ON the shared patch, which is the thing worth
        # seeing: whether index and middle are far enough apart to contribute
        # independent wrench columns, or collapsed onto each other.
        for key, p, _n in _contacts_of(rec):
            if key == "thumb":
                continue
            col = _MIDDLE_COLOR if key == "middle" else SQ.FINGER_COLORS["index"]
            axp.scatter(*p, s=46, marker="^" if key == "middle" else "o",
                        color=col, edgecolor="k", linewidths=0.4,
                        depthshade=False, zorder=12)
            pts.append(np.asarray(p, float)[None, :])
        _s3 = rec["seed"].get("p3s")
        if _s3 is not None:
            _d23 = float(np.linalg.norm(np.asarray(_s3) - rec["seed"]["p2s"])) * 1e3
            # Is the middle SEED actually inside the slot-2 patch it will be
            # confined to? Compare against that patch's own measured half-extent
            # rather than asserting "shared": when it is outside, the seed and the
            # NLP's parameterization disagree, and the solve can only resolve that
            # by dragging the contact -- which is worth seeing on the figure.
            _fr2 = _patch_by_key.get("index")
            if _fr2 is not None:
                _half = max(abs(_fr2["t_hi_0"]), abs(_fr2["t_lo_0"]),
                            abs(_fr2["t_hi_1"]), abs(_fr2["t_lo_1"])) * 1e3
                _in = _d23 <= _half
                axp.set_title(
                    f"index↔middle seed {_d23:.0f}mm  vs slot-2 patch ±{_half:.0f}mm\n"
                    f"{'INSIDE' if _in else 'OUTSIDE patch'}",
                    fontsize=8, color="0.15" if _in else "#cb181d")
            else:
                axp.set_title(f"index↔middle seed {_d23:.0f}mm", fontsize=8)
        if pts:
            SQ._equal_axes(axp, np.vstack(pts), min_r=0.012)
        axp.set_axis_off()


def write_seed_figure(planner, model, data, out_path, title_extra=""):
    """Draw every seed this solve considered. Returns the written path, or None
    when the planner recorded no seeds (e.g. a solve that never reached seeding)."""
    pl = planner._planner if hasattr(planner, "_planner") else planner
    acc = list(getattr(planner, "last_seed_accept_table", None)
               or getattr(pl, "last_seed_accept_table", None) or [])
    rej = list(getattr(planner, "last_seed_reject_table", None)
               or getattr(pl, "last_seed_reject_table", None) or [])
    if not acc and not rej:
        return None
    sc = _scene_from_planner(planner, model, data)
    recs = [_as_rec(e, True) for e in acc] + [_as_rec(e, False) for e in rej]

    n = len(recs)
    n_acc = len(acc)
    # ACCEPTED LEFT, REJECTED RIGHT, with a narrow spacer column between them, so
    # the two populations read as two groups instead of one undifferentiated run.
    _SPACER = 0.22
    _acc_n = len([r for r in recs if r["ok"]])
    _rej_all = [r for r in recs if not r["ok"]]
    _REJ_CAP = 4
    _rej_n = min(len(_rej_all), _REJ_CAP)
    n_acc, n_rej = _acc_n, _rej_n
    widths = ([1.0] * n_acc) + ([_SPACER] if (n_acc and n_rej) else []) + ([1.0] * n_rej)
    n_cols = len(widths)
    fig = plt.figure(figsize=(max(3.3 * (n_acc + n_rej) + 1.0, 10.0), 7.2))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1.05, 1.0], width_ratios=widths)
    fig.suptitle(
        f"{sc['obj']}  —  contact seeds considered by THIS solve{title_extra}\n"
        f"{n_acc} accepted (left)   |   {len(rej)} rejected (right, showing {n_rej})   "
        f"(kappa gate {sc['cfg'].seed_kappa_max_reject:.0f}, "
        f"DLS pool x{sc['cfg'].seed_dls_rank_pool})\n"
        "top: seed ray → surface   bottom: paraboloid patches "
        "(▲ = middle-finger seed; the NLP confines it to the slot-2 patch)",
        fontsize=10.5)

    acc_recs = [r for r in recs if r["ok"]]
    rej_recs = [r for r in recs if not r["ok"]]
    # CAP the rejected columns. 009_gelatin_box rejects 20 seeds (its faces are
    # mostly within seed_ground_clearance_m of the table), and drawing all of them
    # made the figure 8843px wide with ~2mm panels -- the ACCEPTED seeds, which are
    # the subject, became unreadable. Rejects are a diagnostic tail: a
    # representative handful plus a count carries the same information.
    _REJ_CAP = 4
    _n_rej_total = len(rej_recs)
    if _n_rej_total > _REJ_CAP:
        # Spread the sample across the run rather than taking the first few, so a
        # gate that only fires late is still represented.
        _idx = np.linspace(0, _n_rej_total - 1, _REJ_CAP).astype(int)
        rej_recs = [rej_recs[i] for i in _idx]
    _draw_block(fig, gs, 0, acc_recs, sc, "accepted", True)
    _draw_block(fig, gs, n_acc + (1 if (n_acc and n_rej) else 0),
                rej_recs, sc, "rejected", False)

    # Tight vertical packing. The default 3D-axes bbox leaves most of a panel
    # empty around the rendered sphere, which at two rows read as a large band of
    # whitespace between them; hspace is negative to pull the rows back together.
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.02,
                        wspace=0.02, hspace=-0.22)
    out = OP.savefig(fig, Path(out_path), dpi=115)
    plt.close(fig)
    return out

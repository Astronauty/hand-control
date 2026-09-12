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
    """One accept/reject table row -> the `rec` shape draw_seed_rays wants."""
    seed = dict(p1s=np.asarray(entry["p1s"], float),
                p2s=np.asarray(entry["p2s"], float),
                n1_in=np.asarray(entry["n1_in"], float),
                n2_in=np.asarray(entry["n2_in"], float))
    seed["p1"], seed["p2"] = seed["p1s"], seed["p2s"]
    return dict(seed=seed, kind=entry.get("kind", "random"), ok=ok,
                why=entry.get("why", "accepted"),
                ray=dict(origin=None, u=None))


def _build_seed_figure(planner, model, data, title_extra="", max_tris=3000):
    """Build (but do not save) the seed figure. Returns the matplotlib Figure, or None
    when the planner recorded no seeds. Shared by write_seed_figure (-> file) and
    render_seed_figure_png (-> bytes, for the live dashboard). `max_tris` caps the mesh
    scatter per subplot (draw_mesh already thins triangles) so a live render stays cheap."""
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
    fig = plt.figure(figsize=(max(4.6 * n, 11.0), 8.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1.05, 1.0])
    fig.suptitle(
        f"{sc['obj']}  —  contact seeds considered by THIS solve{title_extra}\n"
        f"{len(acc)} accepted, {len(rej)} rejected "
        f"(kappa gate {sc['cfg'].seed_kappa_max_reject:.0f}, "
        f"DLS pool x{sc['cfg'].seed_dls_rank_pool})\n"
        "top: seed ray -> surface   bottom: that seed's thumb + index paraboloid patches",
        fontsize=10.5)

    for i, rec in enumerate(recs):
        axr = fig.add_subplot(gs[0, i], projection="3d")
        SQ.draw_mesh(axr, sc, max_tris=max_tris)
        try:
            SQ.draw_seed_rays(axr, sc, rec)
        except Exception:
            # ray geometry is optional here -- the recorded tables carry contact
            # points and normals, not the sampling ray that produced them.
            for key, p in (("thumb", rec["seed"]["p1s"]), ("index", rec["seed"]["p2s"])):
                axr.scatter(*p, s=34, color=SQ.FINGER_COLORS[key] if rec["ok"] else "#cb181d",
                            depthshade=False, zorder=10)
            axr.plot(*np.array([rec["seed"]["p1s"], rec["seed"]["p2s"]]).T,
                     color="0.35" if rec["ok"] else "#cb181d", lw=1.1, ls="--", zorder=9)
        axr.set_title(f"{rec['kind']} — {'accepted' if rec['ok'] else rec['why']}",
                      fontsize=9, color="0.15" if rec["ok"] else "#cb181d")
        SQ._equal_axes(axr, sc["Vvis"])
        axr.set_axis_off()

        axp = fig.add_subplot(gs[1, i], projection="3d")
        pts = []
        for key, p, nin in (("thumb", rec["seed"]["p1s"], rec["seed"]["n1_in"]),
                            ("index", rec["seed"]["p2s"], rec["seed"]["n2_in"])):
            try:
                frame = SQ.quad_frame(sc, np.asarray(p, float), np.asarray(nin, float),
                                      sc["cfg"])
            except Exception:
                frame = None
            if frame is None:
                continue
            SQ.draw_quadratic(axp, sc, frame, key)
            pts.append(SQ.patch_points(frame, sc["center"], sc["R"],
                                       (frame["t_lo_0"], frame["t_hi_0"]),
                                       (frame["t_lo_1"], frame["t_hi_1"]),
                                       n=7).reshape(-1, 3))
        if pts:
            SQ._equal_axes(axp, np.vstack(pts), min_r=0.012)
        axp.set_axis_off()

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04,
                        wspace=0.12, hspace=0.16)
    return fig


def write_seed_figure(planner, model, data, out_path, title_extra=""):
    """Draw every seed this solve considered to a PNG file. Returns the written path,
    or None when the planner recorded no seeds. High-fidelity (dpi 115) for offline use."""
    fig = _build_seed_figure(planner, model, data, title_extra=title_extra)
    if fig is None:
        return None
    out = OP.savefig(fig, Path(out_path), dpi=115)
    plt.close(fig)
    return out


def render_seed_figure_png(planner, model, data, title_extra="", dpi=60, max_tris=2000):
    """Same figure rendered to in-memory PNG BYTES (for the live dashboard), or None when
    no seeds were recorded. Lower dpi + a tighter triangle cap keep the live render ~0.4 s
    on the recommender thread; the seed points/patches are unaffected by the mesh thinning."""
    import io
    fig = _build_seed_figure(planner, model, data, title_extra=title_extra, max_tris=max_tris)
    if fig is None:
        return None
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi)
    finally:
        plt.close(fig)
    return buf.getvalue()

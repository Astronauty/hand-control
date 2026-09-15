#!/usr/bin/env python3
"""End-to-end pipeline figure: seeds -> grasp map / wrench space -> gamma scaling.

Three panels, left to right, following one grasp from a ray cast to a certified
squeeze force:

  (a) SEED GENERATION. Rays cast from the mesh's VOLUMETRIC centroid (not the
      geom origin, which on every YCB object sits at the base, outside the
      object). Each ray's surface crossings give a contact pair. Panels are
      marked accepted / rejected / selected, and a linearised friction cone is
      drawn at each contact of the selected pair.

  (b) GRASP MAP -> WRENCH SPACE. Each cone facet maps through the grasp map to
      a wrench column of W (6 x 6n). Drawn in the object frame as the convex
      hull of the force rows. The min-weight metric is the LP
          max beta  s.t.  W alpha = 0, sum(alpha) = 1, alpha >= beta
      so alpha are convex weights whose combination cancels; beta = min(alpha)
      is the certificate, and alpha is shown as a bar per column.

  (c) GAMMA SCALING. Cone vertices scale LINEARLY in the internal force,
      V(gamma) = gamma * V(1), which is what makes the LP in
      min_gamma_for_accel_lp linear. gamma is the smallest scale whose wrench
      set contains every corner of the task disturbance box (m*a and I*alpha
      budgets). Drawn as V(1) inside V(gamma) with the box corners marked.

Usage:
    python figures/make_pipeline_figure.py --object 025_mug --seed 2
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "benchmarks"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mujoco as mj
from kinova_common import seed_figure as SF
from ycb_grasp import plot_seed_quadratic as SQ

COL_IN = 7.16          # IEEE DOUBLE column width (two-column span)
TITLE_PT, LAB_PT, TICK_PT, LEG_PT = 9.0, 8.0, 7.0, 7.5
C_TH, C_IX = "#d94801", "#2171b5"


# Friction coefficient used for the CONE DRAWING only. The scene's measured mu
# is 2.0 (half-angle 63 deg), which renders as a cone wider than it is tall and
# reads as a disc rather than a cone. 0.5 (26.6 deg) is the textbook
# illustration -- see the wrench-cone figure this panel pairs with. The value is
# cosmetic: nothing downstream of this figure uses it.
MU_DRAW = 0.5


def _cone_faces(p, n_in, mu=MU_DRAW, h=0.026, n_side=36, view=None):
    """Friction cone at a contact as a SHADED SURFACE plus its SILHOUETTE.

    Returns (facets, rim, gens) -- apex-rim-rim triangles for a
    Poly3DCollection, the closed rim circle, and the two generator lines that
    form the cone's OUTLINE from the current viewpoint.

    The cone OPENS AWAY from the object: apex at the contact, rim offset along
    the OUTWARD normal, matching the usual wrench-cone convention. n_in is the
    INWARD normal, so it is negated here.

    `gens` are the SILHOUETTE edges, not two arbitrary generators. Picking
    rim[0] and rim[n/2] (the previous attempt) draws whichever pair the
    parameterisation happened to start at, which in general lies INSIDE the
    projected outline and reads as a crease across the face. The silhouette is
    where the surface turns away from the camera: the rim points whose radial
    direction is most perpendicular to the view ray. Falls back to the
    arbitrary pair when no view direction is supplied.
    """
    n = -np.asarray(n_in, float)
    n /= max(np.linalg.norm(n), 1e-9)
    a = np.array([1.0, 0.0, 0.0])
    if abs(a @ n) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    r = h * float(mu)
    th = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
    rim = np.array([p + h * n + r * (np.cos(u) * t1 + np.sin(u) * t2)
                    for u in th])
    apex = np.asarray(p, float)
    facets = [[apex, rim[i], rim[(i + 1) % n_side]] for i in range(n_side)]

    if view is None:
        idx = (0, n_side // 2)
    else:
        v = np.asarray(view, float); v /= max(np.linalg.norm(v), 1e-9)
        # Radial offset of each rim point from the cone axis, projected onto the
        # plane perpendicular to the view: the silhouette maximises |radial . w|
        # where w is perpendicular to both the view and the axis.
        w = np.cross(v, n)
        nw = np.linalg.norm(w)
        if nw < 1e-6:
            idx = (0, n_side // 2)           # cone points at the camera
        else:
            w /= nw
            proj = (rim - (p + h * n)) @ w
            idx = (int(np.argmax(proj)), int(np.argmin(proj)))
    gens = [np.vstack([apex, rim[i]]) for i in idx]
    # CONTACT NORMAL through the cone's axis, run past the rim so it reads as
    # the axis the cone is symmetric about rather than as another generator.
    tip = apex + 1.34 * h * n
    axis = np.vstack([apex, tip])
    # Conical ARROWHEAD at the tip, built the same way as the cone itself so it
    # is a real 3D solid and foreshortens with the view rather than a flat glyph
    # pasted on in screen space.
    # Sized against the cone RIM (h*mu), not the cone height: at h=26mm the
    # rim is 13mm and a head of 0.055*h = 1.4mm disappeared next to it. A
    # quarter of the rim radius reads at column scale.
    hr = 0.25 * h * float(mu)
    hl = 2.0 * hr
    base = tip - hl * n
    head_rim = np.array([base + hr * (np.cos(u) * t1 + np.sin(u) * t2)
                         for u in th])
    head = ([[tip, head_rim[i], head_rim[(i + 1) % n_side]]
             for i in range(n_side)]
            + [[base, head_rim[i], head_rim[(i + 1) % n_side]]
               for i in range(n_side)])
    return facets, np.vstack([rim, rim[:1]]), gens, axis, head


def _view_dir(ax):
    """Unit vector from the scene toward the camera, from the axes' own
    elev/azim, so the silhouette tracks whatever view the panel is set to."""
    e = np.deg2rad(ax.elev); a = np.deg2rad(ax.azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def panel_a(fig, gs, pl, model, data, res, max_tris=2600):
    """Seeds: rays from the volumetric centroid, cones on the selected pair."""
    ax = fig.add_subplot(gs, projection="3d")
    sc = SF._scene_from_planner(pl, model, data)
    acc = list(getattr(pl, "last_seed_accept_table", None)
               or getattr(pl._planner, "last_seed_accept_table", None) or [])
    rej = list(getattr(pl, "last_seed_reject_table", None)
               or getattr(pl._planner, "last_seed_reject_table", None) or [])
    # _scene_from_planner returns the VISUAL mesh as Vvis/Fvis in object-local
    # coords; lift to world with the object's pose.
    Vl, F = np.asarray(sc["Vvis"], float), np.asarray(sc["Fvis"])
    V = Vl @ np.asarray(sc["R"], float).T + np.asarray(sc["center"], float)
    step = max(1, len(F) // max_tris)
    _tris = [V[f] for f in F[::step]]
    if _tris:
        _pc = Poly3DCollection(_tris, linewidths=0.25)
        # Explicit RGBA, not facecolor="none" + alpha: the latter leaves the
        # face colour None, which Poly3DCollection's 3D projection multiplies
        # and raises on.
        _pc.set_facecolor((1, 1, 1, 0.0))
        _pc.set_edgecolor((0.60, 0.60, 0.60, 0.40))
        ax.add_collection3d(_pc)
    # RAY ORIGIN: the mesh's volumetric centroid, which is what _seed_pair casts
    # from. NOT the geom frame origin -- on every YCB object that sits at the
    # object's BASE, a point below it on the table.
    me = sc.get("mesh_entry") or {}
    c_l = me.get("vol_centroid")
    c = (np.asarray(c_l, float) @ np.asarray(sc["R"], float).T
         + np.asarray(sc["center"], float)) if c_l is not None else V.mean(0)
    sel = res.get("seed_index")
    # ONLY THE SELECTED PAIR. Earlier versions also drew the losing candidates
    # (grey dots + pale rays). They were the least informative element in a panel
    # that also carries the mesh, two trust-region patches, two friction cones
    # and the centroid, and at column width they read as noise rather than as
    # alternatives. The seed FIGURE (figures/make_seed_figure_icra.py) is where
    # accepted-vs-rejected belongs; this panel's subject is the one grasp that
    # the rest of the pipeline follows.
    for k, e in enumerate(acc + rej):
        if k != sel:
            continue
        rec = SF._as_rec(e, k < len(acc))
        p1, p2 = rec["seed"].get("p1s"), rec["seed"].get("p2s")
        if p1 is None or p2 is None:
            continue
        for p in (p1, p2):
            # The two rays are one antipodal march through the centroid, so
            # they are drawn as a single solid chord.
            ax.plot(*zip(c, p), color="0.20", lw=1.1, ls="-", zorder=4)
        for p, col in ((p1, C_TH), (p2, C_IX)):
            ax.plot([p[0]], [p[1]], [p[2]], "o", ms=4.8, color=col,
                    mec="k", mew=0.5, zorder=6)

    # cones on the SELECTED pair only
    if sel is not None and sel < len(acc):
        rec = SF._as_rec(acc[sel], True)
        for pk, nk, col, key in (("p1s", "n1_in", C_TH, "thumb"),
                                 ("p2s", "n2_in", C_IX, "index")):
            p, n = rec["seed"].get(pk), rec["seed"].get(nk)
            if p is None or n is None: continue
            # The patch each contact is confined to: the paraboloid fitted at
            # the seed over its MEASURED trust region. Same helper the seed
            # figure uses (plot_seed_quadratic.quad_frame/draw_quadratic), so
            # the two figures cannot disagree about what a patch is.
            try:
                _fr = SQ.quad_frame(sc, np.asarray(p, float),
                                    np.asarray(n, float), sc["cfg"])
                if _fr is not None:
                    SQ.draw_quadratic(ax, sc, _fr, key, color=col,
                                      alpha=0.42, depth_sort=True)
            except Exception:
                pass
            facets, rim, gens, axis, head = _cone_faces(
                np.asarray(p, float), np.asarray(n, float), view=_view_dir(ax))
            _cc = matplotlib.colors.to_rgb(col)
            _cone = Poly3DCollection(facets, linewidths=0.0)
            # Explicit RGBA rather than facecolor + alpha: Poly3DCollection's 3D
            # projection multiplies the face colour and raises when it is None.
            _cone.set_facecolor((*_cc, 0.30))
            _cone.set_edgecolor((*_cc, 0.0))
            ax.add_collection3d(_cone)
            # Rim outline reads the opening angle at column scale, where the
            # shaded surface alone is too pale to.
            ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], "-", color=col,
                    lw=0.7, zorder=7)
            ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], "-", color=col,
                    lw=0.9, zorder=8)
            _hd = Poly3DCollection(head, linewidths=0.0)
            _hd.set_facecolor((*_cc, 1.0))
            _hd.set_edgecolor((*_cc, 0.0))
            ax.add_collection3d(_hd)
    ax.plot([c[0]], [c[1]], [c[2]], "x", ms=4.5, color="k", mew=1.1, zorder=6)
    _iso(ax, V)
    ax.set_title("(a) seed, patch, friction cone", fontsize=TITLE_PT, pad=-2)
    return ax


def panel_b(fig, gs, W, alpha, beta):
    """Wrench space: hull of the FORCE rows of W, with the alpha weights."""
    ax = fig.add_subplot(gs, projection="3d")
    Wf = np.asarray(W, float)[:3, :].T            # force rows, one point per column
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(Wf)
        ax.add_collection3d(Poly3DCollection(
            [Wf[s] for s in hull.simplices], facecolor="#6baed6",
            alpha=0.30, edgecolor="#2171b5", linewidths=0.35))
    except Exception:
        pass
    n = Wf.shape[0] // 2
    for i, p in enumerate(Wf):
        ax.plot([p[0]], [p[1]], [p[2]], "o", ms=2.6,
                color=(C_TH if i < n else C_IX), mec="k", mew=0.25)
    ax.plot([0], [0], [0], "x", ms=4.5, color="k", mew=1.1)
    _iso(ax, Wf)
    ax.set_title(r"(b) wrench space, $\mathbf{W}\alpha=0$",
                 fontsize=TITLE_PT, pad=-2)
    return ax


def panel_c(fig, gs, W, gamma, abox, angbox, mass):
    """gamma scaling: V(1) inside V(gamma), against the task box corners."""
    ax = fig.add_subplot(gs, projection="3d")
    Wf = np.asarray(W, float)[:3, :].T
    try:
        from scipy.spatial import ConvexHull
        for scale, fc, ec, al in ((1.0, "#9ecae1", "#6baed6", 0.22),
                                  (float(gamma), "#fdd0a2", "#d94801", 0.16)):
            P = Wf * scale
            h = ConvexHull(P)
            ax.add_collection3d(Poly3DCollection(
                [P[s] for s in h.simplices], facecolor=fc, alpha=al,
                edgecolor=ec, linewidths=0.35))
    except Exception:
        pass
    # task disturbance box corners: m*a, the FORCE the grasp must resist
    f = float(mass) * np.asarray(abox, float)
    corners = np.array([[sx*f[0], sy*f[1], sz*f[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], "s", ms=2.2,
            color="#238b45", mec="k", mew=0.25, ls="none")
    _iso(ax, np.vstack([Wf*float(gamma), corners]))
    ax.set_title(rf"(c) $\gamma$ = {gamma:.1f} N scaling",
                 fontsize=TITLE_PT, pad=-2)
    return ax


def _iso(ax, P):
    P = np.asarray(P, float)
    c = P.mean(0); r = float(np.abs(P - c).max()) * 1.05 or 1.0
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()
    # elev/azim: swept against two criteria, then checked by rendering.
    #   1. |cos(view, cone axis)| per cone -- near 1 means the cone points at the
    #      camera and projects as an ellipse rather than a triangle;
    #   2. the handle HOLE readable, which needs the view roughly perpendicular
    #      to the handle direction (world [-0.96, 0.17, -0.22]) and from the -x
    #      side, so the opening is not occluded by the cup.
    # (20, -58) failed (1) at 0.59/0.70. (20, -12) was ideal for (1) but looked
    # down the grasp axis and hid the handle entirely. Scores alone favoured
    # azim ~138, which puts the handle BEHIND the cup -- rejected on the render.
    # (30, -60) shows the hole with both cones still reading as cones.
    ax.view_init(elev=30, azim=-60)
    try: ax.set_box_aspect(None, zoom=1.28)
    except TypeError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="025_mug")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import benchmarks.ycb_grasp.pick_and_place as PP
    import simulation.grasp_planner_3d as G
    import kinova_common.wrench as WR

    cap = {}
    _o = G.MultiStartGraspPlanner3D.solve
    def _sp(self, *a, **k):
        r = _o(self, *a, **k); cap["r"], cap["pl"] = r, self; return r
    G.MultiStartGraspPlanner3D.solve = _sp
    _g = WR.solve_gamma_live
    def _gs(*a, **k):
        v = _g(*a, **k); cap["gamma"], cap["ga"] = v, (a, k); raise SystemExit
    WR.solve_gamma_live = _gs; PP.solve_gamma_live = _gs
    try:
        PP.run_pick_place(args.object, args.seed, n_seeds=3,
                          do_transport=False, out_dir=None, view=False)
    except SystemExit:
        pass
    if "r" not in cap or cap.get("gamma") is None:
        print("no feasible gamma for this cell; pick another --seed"); return 1

    r, pl = cap["r"], cap["pl"]
    a, _k = cap["ga"]
    fig = plt.figure(figsize=(COL_IN, COL_IN * 0.30))
    gs = GridSpec(1, 3, figure=fig, wspace=0.02)
    import traceback
    for _nm, _fn in (("a", lambda: panel_a(fig, gs[0, 0], pl, pl._planner.model,
                                          pl._planner.data, r)),
                     ("b", lambda: panel_b(fig, gs[0, 1], r["gws_W"],
                                          np.asarray(r["gws_alpha"]).flatten(),
                                          float(r["gws_beta"]))),
                     ("c", lambda: panel_c(fig, gs[0, 2], r["gws_W"], cap["gamma"],
                                          a[4], a[5], a[3]))):
        try:
            _fn()
        except Exception:
            print(f"panel {_nm} raised:"); traceback.print_exc(limit=3)

    h = [Line2D([], [], ls="", marker="o", ms=3.4, color=C_TH, mec="k", mew=0.3,
                label="thumb"),
         Line2D([], [], ls="", marker="o", ms=3.4, color=C_IX, mec="k", mew=0.3,
                label="index"),
         Line2D([], [], ls="", marker="x", ms=4.0, color="k", mew=1.0,
                label="centroid / origin"),
         Line2D([], [], ls="", marker="s", ms=3.0, color="#238b45", mec="k",
                mew=0.3, label=r"task box $m\mathbf{a}$")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False,
               fontsize=LEG_PT, handletextpad=0.25, columnspacing=1.1,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(rf"{args.object.replace('_',' ')}   "
                 rf"$\beta$ = {float(r['gws_beta']):.3f},   "
                 rf"$\gamma$ = {cap['gamma']:.2f} N",
                 fontsize=TITLE_PT, y=1.06)
    # top < 1 leaves the suptitle its own band; the panel titles sit at
    # pad=-2 inside their axes, so without this the two collide.
    fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.10)
    out = Path(args.out or (REPO / "figures" /
               f"pipeline_{args.object}_s{args.seed}.pdf"))
    fig.savefig(out, format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  beta={float(r['gws_beta']):.5f}  gamma={cap['gamma']:.4f}  "
          f"W={np.asarray(r['gws_W']).shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

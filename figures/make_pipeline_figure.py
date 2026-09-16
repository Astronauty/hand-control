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
from mpl_toolkits.mplot3d import proj3d
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
AXLAB_PT = 9.0         # wrench-axis labels; the panels carry no title now
TITLE_PT, LAB_PT, TICK_PT, LEG_PT = 9.0, 8.0, 7.0, 7.5
C_TH, C_IX = "#d94801", "#2171b5"          # thumb / index: contacts AND patches
# WRENCH HULLS. One colour per panel -- the force and torque plots in a panel
# are two projections of ONE set, so colouring them differently implied they
# were different objects. Both are teal/slate rather than orange or blue, so a
# hull is never read as a contact patch, and the task box stays purple.
C_HULL_B_LO, C_HULL_B_HI = "#9ecae1", "#3182bd"   # (b) V(1)
C_HULL_C_LO, C_HULL_C_HI = "#a1d99b", "#31a354"   # (c) V(gamma)


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

    _cb = []            # contacts painted, for the exploded callout
    # cones on the SELECTED pair only
    if sel is not None and sel < len(acc):
        rec = SF._as_rec(acc[sel], True)
        for pk, nk, col, key in (("p1s", "n1_in", C_TH, "thumb"),
                                 ("p2s", "n2_in", C_IX, "index")):
            p, n = rec["seed"].get(pk), rec["seed"].get(nk)
            if p is None or n is None: continue
            # PATCH at the seed (that is where the paraboloid was fitted), but
            # the CONE at the contact the NLP converged to. The seed is a
            # starting point and the solver moves it -- measured 33.3mm on this
            # grasp -- so a cone drawn at the seed shows a friction cone the
            # grasp never uses. res['p1']/['p2'] are the solved contacts and
            # res['n1_final']/['n2_final'] their OUTWARD normals (negated here,
            # since _cone_faces takes the inward convention).
            _ps, _ns = np.asarray(p, float), np.asarray(n, float)
            _pk_s = {"thumb": "p1", "index": "p2"}[key]
            _nk_s = {"thumb": "n1_final", "index": "n2_final"}[key]
            _psol = res.get(_pk_s)
            _nsol = res.get(_nk_s)
            _pc_ = np.asarray(_psol, float) if _psol is not None else _ps
            if _nsol is not None and np.all(np.isfinite(np.asarray(_nsol, float))):
                _nc_ = -np.asarray(_nsol, float)
            else:
                _nc_ = _ns
            _paint_contact(ax, sc, _ps, _ns, col, key, patch=True,
                           cone=False)
            _paint_contact(ax, sc, _pc_, _nc_, col, key, patch=False,
                           cone=True)
            _cb.append((_pc_, _nc_, col, key))
    ax.plot([c[0]], [c[1]], [c[2]], "x", ms=4.5, color="k", mew=1.1, zorder=6)
    _iso(ax, V)
    ax._callout_contacts = _cb
    ax._callout_sc = sc
    # No axes title: the COLUMN title above the figure names this stage.
    # Both together read as two headings for one panel.
    return ax


def _hull_inradius(P):
    """Distance from the ORIGIN to the nearest facet of conv(P).

    This is the Ferrari-Canny epsilon metric restricted to the subspace P spans:
    the radius of the largest ball centred at the origin that fits inside the
    wrench set. Negative when the origin is outside, i.e. no force closure.
    """
    from scipy.spatial import ConvexHull
    h = ConvexHull(np.asarray(P, float))
    return float((-h.equations[:, -1]
                  / np.linalg.norm(h.equations[:, :3], axis=1)).min())


def _ball(c, r, n=22):
    u = np.linspace(0, np.pi, n); v = np.linspace(0, 2 * np.pi, n)
    return (c[0] + r * np.outer(np.sin(u), np.cos(v)),
            c[1] + r * np.outer(np.sin(u), np.sin(v)),
            c[2] + r * np.outer(np.cos(u), np.ones_like(v)))


def _wrench_hull(ax, P, title, col_lo, col_hi, axlab=None, ball=False,
                 box=None, scale=1.0, box_col="#6a3d9a", alpha_w=None,
                 axis_len=None, lim=None, box_lab=None):
    """Convex hull of wrench columns in one 3-D subspace.

    ball  : draw the largest origin-centred ball inside the hull (Ferrari-Canny).
    box   : (3,) half-extents of the task disturbance box, drawn as a wireframe.
    scale : multiply the hull by this (gamma). V(gamma) = gamma * V(1) exactly,
            which is what makes min_gamma_for_accel_lp linear.
    """
    P = np.asarray(P, float) * float(scale)
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(P)
        _pc = Poly3DCollection([P[s] for s in h.simplices], linewidths=0.30)
        _pc.set_facecolor((*matplotlib.colors.to_rgb(col_lo), 0.22))
        _pc.set_edgecolor((*matplotlib.colors.to_rgb(col_hi), 0.80))
        ax.add_collection3d(_pc)
    except Exception:
        pass

    _pts = [P, np.zeros((1, 3))]
    if box is not None:
        _lim0 = (np.asarray(lim, float) if lim is not None
                 else np.vstack([P, np.zeros(3)]))
        b = np.asarray(box, float)
        sgn = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                        for sz in (-1, 1)], float)
        C = sgn * b
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
                 if np.sum(np.abs(sgn[i] - sgn[j])) == 2]
        for i, j in edges:
            ax.plot(*zip(C[i], C[j]), "-", color=box_col, lw=0.65, zorder=7)
        _pts.append(C)
        # NO IN-PLOT LIMIT LABEL. Three per-axis labels collided into a blob
        # ("2.362.36"); a single label on the longest axis still ran into the
        # wireframe, because that axis points back toward the box in this
        # projection and the box is small relative to the frame -- there is no
        # offset that clears both the marker and the box without leaving the
        # axes. The suptitle already states both limits (ma = 2.36 N/axis,
        # I alpha = 0.15 N m) with their inputs, which is the readable place
        # for a number the reader wants to check rather than locate.

    # SHARED LIMITS between (b) and (c) when given: the two panels only compare
    # if one unit of length means the same thing in both.
    _lim = np.vstack(_pts) if lim is None else np.asarray(lim, float)
    if axlab is not None:
        # PER-AXIS arm length: these hulls are strongly anisotropic, so one
        # global radius buries the arm along a wide direction and leaves the
        # arm along a narrow one floating.
        # EQUAL-LENGTH arms: a triad whose arms differ per axis reads as a
        # scale cue that is not there. One length for all three, sized off the
        # largest extent so none is swallowed.
        _r = 1.34 * float(np.abs(_lim).max())
        _r_ax = np.array([_r, _r, _r]) if axis_len is None else np.asarray(axis_len, float)
        _lim = np.vstack([_lim, np.diag(_r_ax) * 1.24, -np.diag(_r_ax) * 0.12])
    # True isometric: elev atan(1/sqrt(2)), azim 45 -- the three axes subtend
    # equal angles, so no wrench component is visually privileged.
    _iso(ax, _lim, elev=35.264, azim=45)
    if axlab is not None:
        for _k, (_lb, _d) in enumerate(zip(axlab, np.eye(3))):
            _end = _r_ax[_k] * _d
            ax.plot(*zip(np.zeros(3), _end), "-", color="0.35", lw=0.55, zorder=2)
            _tp = 1.30 * _end
            ax.text(_tp[0], _tp[1], _tp[2], _lb, fontsize=AXLAB_PT,
                    color="0.25", ha="center", va="center", zorder=9)
    if title:
        ax._callout_contacts = _cb
    ax.set_title(title, fontsize=TITLE_PT, pad=-4)


def _paint_contact(ax, sc, p, n, col, key, patch=True, cone=True):
    """Draw ONE contact's quadratic patch, friction cone and normal arrow.

    Factored out so the exploded callout re-renders the SAME geometry at a
    different scale rather than a second, hand-matched copy of it.
    """
    if patch:
        try:
            _fr = SQ.quad_frame(sc, p, n, sc["cfg"])
            if _fr is not None:
                SQ.draw_quadratic(ax, sc, _fr, key, color=col, alpha=0.42,
                                  depth_sort=True)
        except Exception:
            pass
    if not cone:
        return
    facets, rim, gens, axis, head = _cone_faces(p, n, view=_view_dir(ax))
    _cc = matplotlib.colors.to_rgb(col)
    _cone = Poly3DCollection(facets, linewidths=0.0)
    _cone.set_facecolor((*_cc, 0.30))
    _cone.set_edgecolor((*_cc, 0.0))
    ax.add_collection3d(_cone)
    ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], "-", color=col, lw=0.7, zorder=7)
    ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], "-", color=col, lw=0.9, zorder=8)
    _hd = Poly3DCollection(head, linewidths=0.0)
    _hd.set_facecolor((*_cc, 1.0))
    _hd.set_edgecolor((*_cc, 0.0))
    ax.add_collection3d(_hd)
    ax.plot([p[0]], [p[1]], [p[2]], "o", ms=4.0, color=col, mec="k", mew=0.5,
            zorder=9)


def _contact_callout(fig, host, draw_fn, center_fig, src_r=0.028,
                     bubble=(0.0, 0.0), r=0.085):
    """Exploded callout: a circled ZOOM of one contact, on a black leader.

    draw_fn(ax) paints the magnified content into a fresh 3D axes -- here the
    friction cone and its quadratic patch, re-drawn at contact scale so the
    cone's opening angle and the patch's extent are both legible. The parent
    panel shows them at object scale, where a 26mm cone on a 90mm mug is small.

    The circle, its source ring and the leaders are drawn in FIGURE coordinates
    so they can sit outside the host axes; an inset inside the host lands on
    its axis labels.
    """
    import matplotlib.patches as mpatches
    cx, cy = bubble
    sx, sy = center_fig
    circ = mpatches.Circle((cx, cy), r, transform=fig.transFigure,
                           facecolor="white", edgecolor="k", lw=0.7, zorder=20)
    fig.patches.append(circ)
    src = mpatches.Circle((sx, sy), src_r, transform=fig.transFigure,
                          facecolor="none", edgecolor="k", lw=0.6, zorder=20)
    fig.patches.append(src)
    d = np.hypot(cx - sx, cy - sy)
    if d > 1e-9:
        ux, uy = (cx - sx) / d, (cy - sy) / d
        nx, ny = -uy, ux
        for s in (+1.0, -1.0):
            fig.add_artist(Line2D([sx + s * nx * src_r, cx + s * nx * r],
                                  [sy + s * ny * src_r, cy + s * ny * r],
                                  transform=fig.transFigure, color="k",
                                  lw=0.6, zorder=19))
    ax = fig.add_axes([cx - r * 0.78, cy - r * 0.78, r * 1.56, r * 1.56],
                      projection="3d", zorder=21)
    ax.patch.set_alpha(0.0)
    draw_fn(ax)
    return ax


def panel_b(fig, gs, W, alpha, beta, box_f=None, box_t=None, lim_f=None,
            lim_t=None):
    """(b) GRASP WRENCH SPACE at unit internal force, V(1).

    W is 6 x 6n and the wrench set lives in R^6, so it cannot be drawn directly.
    Projecting onto (fx,fy,fz) and (tx,ty,tz) is the honest reduction, and the
    two need their own axes: the torque rows run 9-15x the force rows here, so
    a shared scale collapses the force hull to a dot.

    ROW ORDER checked against the code, not assumed: build_W's _col() returns
    ca.vertcat(tau, f_O), so rows 0-2 are TORQUE and rows 3-5 are FORCE.

    The TASK BOX is drawn here too, at the SAME scale and axis limits as panel
    (c). That is the whole b->c story in one comparison: the box does NOT fit
    inside V(1) and DOES fit inside V(gamma). Measured, the force inradius goes
    1.0000 -> 4.3330 (exactly gamma) against a 4.0876 N box corner, so the unit
    grasp cannot resist the disturbance and the scaled one can. An annotation
    marking beta was tried instead and dropped: beta is a property of the
    WEIGHTS and says nothing about this scaling.
    """
    sub = gs.subgridspec(2, 1, hspace=0.0)
    Wa = np.asarray(W, float)
    ax_f = fig.add_subplot(sub[0, 0], projection="3d")
    # NO task box and NO shared limits here. Drawing the box in both panels
    # made the comparison explicit but left V(1) at a quarter of the frame,
    # since the shared limits are set by the LARGER of the two. (c) carries the
    # containment story; (b) just shows the wrench set's shape, framed to
    # itself. The gamma factor is stated in the suptitle either way.
    _wrench_hull(ax_f, Wa[3:, :].T, "", C_HULL_B_LO, C_HULL_B_HI,
                 axlab=(r"$f_x$", r"$f_y$", r"$f_z$"))
    ax_t = fig.add_subplot(sub[1, 0], projection="3d")
    _wrench_hull(ax_t, Wa[:3, :].T, "", C_HULL_B_LO, C_HULL_B_HI,
                 axlab=(r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"))
    return ax_f


def panel_c(fig, gs, W, gamma, box_f=None, box_t=None, lim_f=None,
            lim_t=None):
    """(c) GAMMA SCALING against the task disturbance box.

    Cone vertices scale LINEARLY in the internal force, V(gamma) = gamma * V(1),
    which is exactly what makes min_gamma_for_accel_lp an LP rather than a
    binary search. gamma is the smallest scale whose wrench set contains every
    corner of the task box (m*a in force, I*alpha in torque).

    Drawn per subspace for the same reason as (b), and because the two are NOT
    equally binding: measured on this grasp the gamma-scaled force hull has
    inradius 4.33 N against a 4.09 N box corner (tight), while the torque hull
    has 0.279 N m against 2.25e-4 N m (three orders of slack). FORCE is what
    sets gamma here; a single combined plot would hide that.
    """
    sub = gs.subgridspec(2, 1, hspace=0.0)
    Wa = np.asarray(W, float)
    g = float(gamma)
    ax_f = fig.add_subplot(sub[0, 0], projection="3d")
    _wrench_hull(ax_f, Wa[3:, :].T, "", C_HULL_C_LO, C_HULL_C_HI,
                 axlab=(r"$f_x$", r"$f_y$", r"$f_z$"), scale=g, box=box_f,
                 lim=lim_f, box_col="#6a3d9a", box_lab=r"{v:.2f}")
    ax_t = fig.add_subplot(sub[1, 0], projection="3d")
    _wrench_hull(ax_t, Wa[:3, :].T, "", C_HULL_C_LO, C_HULL_C_HI,
                 axlab=(r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"), scale=g,
                 box=box_t, lim=lim_t, box_col="#6a3d9a", box_lab=r"{v:.2f}")
    return ax_f


def _iso(ax, P, elev=30, azim=-60):
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
    ax.view_init(elev=elev, azim=azim)
    try: ax.set_box_aspect(None, zoom=1.60)
    except TypeError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="025_mug")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--ang-display", type=float, default=None,
                    help="FIGURE ONLY: angular-accel budget (rad/s^2) used to "
                         "draw the torque task box. The true budget is 1.0, "
                         "which renders sub-pixel (the box is 5255x smaller "
                         "than the gamma-scaled torque hull). 1000 puts it at "
                         "~19%% of the hull. Changes nothing the solver did; "
                         "state it in the caption if used.")
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
    _M, _D = pl._planner.model, pl._planner.data
    _inertia = _M.body_inertia[pl._planner._obj_bid].copy()
    # Taller and tighter: (b)/(c) are 2-row subgrids, so the old 0.30 aspect
    # left each hull in a sliver. wspace 0 -- the panels carry their own
    # whitespace from the 3D axes' margins, so any grid spacing is additive.
    # Boxes and axis limits computed ONCE and shared by (b) and (c): the
    # comparison only reads if a unit of length is the same in both panels.
    _ang = ((args.ang_display,) * 3 if args.ang_display else a[5])
    _box_f = float(a[3]) * np.asarray(a[4], float)
    _box_t = np.asarray(_inertia, float) * np.asarray(_ang, float)
    _Wf, _Wt = np.asarray(r["gws_W"], float)[3:, :].T, np.asarray(r["gws_W"], float)[:3, :].T
    _g = float(cap["gamma"])
    _lim_f = np.vstack([_Wf * _g, -_Wf * _g, _box_f, -_box_f])
    _lim_t = np.vstack([_Wt * _g, -_Wt * _g, _box_t, -_box_t])
    fig = plt.figure(figsize=(COL_IN, COL_IN * 0.44))
    gs = GridSpec(1, 3, figure=fig, wspace=0.0)
    import traceback
    for _nm, _fn in (("a", lambda: panel_a(fig, gs[0, 0], pl, pl._planner.model,
                                          pl._planner.data, r)),
                     ("b", lambda: panel_b(fig, gs[0, 1], r["gws_W"],
                                          np.asarray(r["gws_alpha"]).flatten(),
                                          float(r["gws_beta"]),
                                          box_f=_box_f, box_t=_box_t,
                                          lim_f=_lim_f, lim_t=_lim_t)),
                     ("c", lambda: panel_c(fig, gs[0, 2], r["gws_W"],
                                          cap["gamma"],
                                          box_f=_box_f, box_t=_box_t,
                                          lim_f=_lim_f, lim_t=_lim_t))):
        try:
            _fn()
        except Exception:
            print(f"panel {_nm} raised:"); traceback.print_exc(limit=3)

    h = [Line2D([], [], ls="", marker="o", ms=3.4, color=C_TH, mec="k", mew=0.3,
                label="thumb"),
         Line2D([], [], ls="", marker="o", ms=3.4, color=C_IX, mec="k", mew=0.3,
                label="index"),
         Line2D([], [], ls="", marker="x", ms=4.0, color="k", mew=1.0,
                label="ray origin (volumetric centroid)"),
         Line2D([], [], ls="", marker="s", ms=3.0, color="#6a3d9a", mec="k",
                mew=0.3,
                label=r"task box  $m\mathbf{a}$ / $\mathbf{I}\boldsymbol{\alpha}$"),
         ]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False,
               fontsize=LEG_PT, handletextpad=0.25, columnspacing=1.1,
               bbox_to_anchor=(0.5, -0.02))
    # One title per COLUMN, naming the stage, plus the two scalars the figure
    # is about. The per-plot titles were dropped -- the axis labels say which
    # subspace each hull is -- but the columns still need naming.
    for _x, _s in ((0.17, "(a) seeding"),
                   (0.50, r"(b) wrench set $V(1)$"),
                   (0.84, r"(c) $V(\gamma)=\gamma\,V(1)$")):
        fig.text(_x, 0.965, _s, fontsize=TITLE_PT, ha="center", va="bottom")
    # The task box stated as the physical quantities it is, so the reader can
    # check the arithmetic: m*a per axis, its corner, and the gamma that covers
    # it. Without these the green/purple wireframe is an abstract shape.
    _mm, _aa = float(a[3]), float(np.asarray(a[4], float)[0])
    _bf = _mm * _aa
    _angd = float(args.ang_display) if args.ang_display else float(
        np.asarray(a[5], float)[0])
    _bt = np.asarray(_inertia, float) * _angd
    _star = r"$^{*}$" if args.ang_display else ""
    fig.text(0.5, 1.055,
             rf"{args.object.replace('_',' ')}:   $m$ = {_mm:.3f} kg,  "
             rf"$a$ = {_aa:.0f} m/s$^2$ $\Rightarrow$ $ma$ = {_bf:.2f} N/axis"
             rf"      $\alpha${_star} = {_angd:.0f} rad/s$^2$ $\Rightarrow$ "
             rf"$\mathbf{{I}}\alpha$ = {_bt.max():.2f} N$\cdot$m",
             fontsize=TITLE_PT, ha="center", va="bottom")
    fig.text(0.5, 1.005,
             rf"$\beta$ = {float(r['gws_beta']):.3f} "
             rf"($\sum\alpha_i = 1$),    "
             rf"$\gamma$ = {cap['gamma']:.2f} N"
             + (r"        $^{*}$illustrative angular budget; the measured one "
                r"renders sub-pixel" if args.ang_display else ""),
             fontsize=TICK_PT, ha="center", va="bottom", color="0.35")
    # top < 1 leaves the suptitle its own band; the panel titles sit at
    # pad=-2 inside their axes, so without this the two collide.
    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.035)
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

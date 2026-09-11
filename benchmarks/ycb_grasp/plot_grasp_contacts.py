"""Solved-grasp contacts drawn in the seed-quadratic visualizer's style.

plot_quadratic_path.py answers "how did the contact MOVE across Picard stages"
-- a trajectory question, and the right one while tuning the relinearization
loop. This module answers the other one: "what does the final grasp actually
look like on this object" -- one zoomed panel per contact showing its
paraboloid patch against the real surface, exactly the way
plot_seed_quadratic.py shows a seed's patch.

Same visual grammar as plot_seed_quadratic.py, deliberately, so the two read
the same way:
  * transparent visual mesh as a thin-edged shell
  * two colors per finger -- saturated inside the measured trust region,
    pale wireframe beyond it (where the surrogate is extrapolating)
  * per-contact zoom, because a pair's two contacts sit on opposite sides of
    the object and framing both at once makes a ~10mm patch unreadable
  * max |true SDF| over the drawn patch reported per contact, so the picture
    carries its own error bar

Differences from the seed version, all because these are SOLVED contacts:
  * the patch is drawn at the contact the solver RETURNED, not at a seed
  * the solved (t0,t1) is marked inside the trust region, so a contact pinned
    against its own bound (the thing quadratic_pin_frac tests, see
    GraspConfig3D) is visible as a dot on the patch edge
  * an overview panel shows both contacts on the whole object with the grasp
    axis between them

Used by pick_and_place.py in place of the quadratic-path figure. Reads the
same per-stage npz trace and the same attempt-matching logic
(plot_quadratic_path._iter_trace_quadratic_stages), so it selects the same
winning attempt; it just draws the LAST stage rather than all of them.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import out_paths as OP
except ImportError:
    from ycb_grasp import out_paths as OP

# Same finger palette as plot_seed_quadratic.py: (inside trust region, outside).
FINGER_COLORS = {
    1: ("#d94801", "#fdd0a2"),   # thumb  — orange
    2: ("#2171b5", "#c6dbef"),   # index  — blue
}
FINGER_NAMES = {1: "thumb", 2: "index"}


def _patch_points(frame, t0_range, t1_range, n=13):
    """p_local(t0,t1) over a grid -- the same map _mesh_quadratic_contact_ca
    builds symbolically (and plot_quadratic_path._paraboloid_p_local evaluates),
    here with EXPLICIT per-side ranges so the asymmetric trust region
    [t_lo, t_hi] is drawn as it actually is rather than symmetrized."""
    t0 = np.linspace(t0_range[0], t0_range[1], n)
    t1 = np.linspace(t1_range[0], t1_range[1], n)
    T0, T1 = np.meshgrid(t0, t1)
    h = -(float(frame["kappa0"]) * T0**2 + float(frame["kappa1"]) * T1**2) \
        / (2.0 * float(frame["grad_norm"]))
    return (np.asarray(frame["seed_l"], float)[None, None, :]
            + T0[..., None] * np.asarray(frame["axis0_l"], float)[None, None, :]
            + T1[..., None] * np.asarray(frame["axis1_l"], float)[None, None, :]
            + h[..., None] * np.asarray(frame["n_l"], float)[None, None, :])


def _bounds(frame):
    """(lo0,hi0),(lo1,hi1) -- the asymmetric per-side trust region when the
    frame carries it, falling back to the symmetric half-width for traces
    written before t_lo_*/t_hi_* were saved."""
    def _g(k, default):
        v = frame.get(k)
        return float(v) if v is not None else float(default)
    b0, b1 = _g("t_bound_0", 0.01), _g("t_bound_1", 0.01)
    return ((_g("t_lo_0", -b0), _g("t_hi_0", b0)),
            (_g("t_lo_1", -b1), _g("t_hi_1", b1)))


def _draw_mesh(ax, Vw, F, alpha=0.12, max_tris=3000):
    """Thin-edged translucent shell (see plot_seed_quadratic.draw_mesh: filled
    faces at YCB triangle counts stack into an opaque blob)."""
    if len(F) > max_tris:
        F = F[np.linspace(0, len(F) - 1, max_tris).astype(int)]
    ax.add_collection3d(Poly3DCollection(Vw[F], facecolor="0.6", edgecolor="0.45",
                                         linewidths=0.15, alpha=alpha, zsort="min"))


def _equal_axes(ax, pts, pad=0.005, min_r=None):
    pts = np.asarray(pts, float).reshape(-1, 3)
    mid = 0.5 * (pts.max(0) + pts.min(0))
    r = 0.5 * float(np.max(pts.max(0) - pts.min(0))) + pad
    if min_r is not None:
        r = max(r, float(min_r))
    ax.set_autoscale_on(False)
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_box_aspect((1, 1, 1))


def _to_world(P_l, center, R):
    return center + np.asarray(P_l, float) @ np.asarray(R, float).T


def plot_grasp_contacts(V, F, stage, object_id, out_path, sdf_fn=None,
                        verify_info=None, elev=18.0, azim=-60.0, n_relin=None):
    """One figure: overview + one zoomed panel per solved contact.

    V, F     : object visual mesh, BODY frame (object_uv_atlas.body_visual_mesh)
    stage    : ONE per-stage dict from
               plot_quadratic_path._iter_trace_quadratic_stages -- normally the
               LAST (the returned solve); carries obj_center/obj_mat and a
               `contact` map {1: frame, 2: frame} of quad_* params.
    sdf_fn   : optional f(p_local)->signed distance. When given, each panel
               reports max |true SDF| over its drawn patch, the same error bar
               plot_seed_quadratic.py shows.
    """
    center = np.asarray(stage["obj_center"], float)
    R = np.asarray(stage["obj_mat"], float).reshape(3, 3)
    Vw = _to_world(V, center, R)
    contacts = [(ci, stage["contact"][ci]) for ci in (1, 2) if ci in stage["contact"]]
    if not contacts:
        return None

    n_cols = 1 + len(contacts)
    fig = plt.figure(figsize=(max(5.4 * n_cols, 11.0), 6.4))
    sub = f"  ({n_relin} Picard stage{'s' if (n_relin or 0) != 1 else ''})" \
        if n_relin is not None else ""
    head = f"{object_id} — solved grasp contacts{sub}"
    if verify_info:
        _wf = verify_info.get("wrench_feasible")
        _gm = verify_info.get("gamma_min")
        if _wf is not None:
            head += f"   |   wrench_feasible={_wf}"
        if _gm is not None:
            head += f"  gamma_min={float(_gm):.2f}N"
    fig.suptitle(head + "\nsaturated = measured trust region, pale wireframe = "
                 "extrapolation, ✕ = solved contact", fontsize=10.5)

    # ── overview: both contacts on the whole object ────────────────────────
    ax = fig.add_subplot(1, n_cols, 1, projection="3d")
    _draw_mesh(ax, Vw, F, alpha=0.10)
    pts_w = []
    for ci, fr in contacts:
        p_w = _to_world(np.asarray(fr["seed_l"], float)[None, :], center, R)[0]
        pts_w.append(p_w)
        ax.scatter(*p_w, color=FINGER_COLORS[ci][0], s=60, edgecolor="k", lw=0.5,
                   zorder=8)
        ax.text(*p_w, f"  {FINGER_NAMES[ci]}", fontsize=7.5, fontweight="bold",
                color=FINGER_COLORS[ci][0], zorder=9)
    if len(pts_w) == 2:
        # Grasp axis -- the line the two fingers squeeze along.
        ax.plot(*np.array(pts_w).T, color="0.3", ls="-.", lw=1.3, zorder=7)
        _d = np.linalg.norm(pts_w[0] - pts_w[1]) * 1e3
        ax.set_title(f"both contacts  (grasp width {_d:.0f}mm)", fontsize=9)
    _equal_axes(ax, np.vstack([Vw] + [np.asarray(pts_w)]), pad=0.01)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7)
    ax.set_zlabel("z (m)", fontsize=7); ax.tick_params(labelsize=6)

    # ── one zoomed panel per contact ───────────────────────────────────────
    for k, (ci, fr) in enumerate(contacts):
        axq = fig.add_subplot(1, n_cols, 2 + k, projection="3d")
        _draw_mesh(axq, Vw, F, alpha=0.12)
        c_in, c_out = FINGER_COLORS[ci]
        (lo0, hi0), (lo1, hi1) = _bounds(fr)

        # pale extrapolation band, then the saturated valid region
        P_out = _patch_points(fr, (lo0 * 1.8, hi0 * 1.8), (lo1 * 1.8, hi1 * 1.8), n=15)
        Pw = _to_world(P_out.reshape(-1, 3), center, R).reshape(P_out.shape)
        axq.plot_wireframe(Pw[..., 0], Pw[..., 1], Pw[..., 2], color=c_out,
                           lw=0.5, alpha=0.75, rstride=1, cstride=1, zorder=7)
        P_in = _patch_points(fr, (lo0, hi0), (lo1, hi1), n=13)
        Pi = _to_world(P_in.reshape(-1, 3), center, R).reshape(P_in.shape)
        axq.plot_surface(Pi[..., 0], Pi[..., 1], Pi[..., 2], color=c_in, alpha=0.55,
                         linewidth=0, antialiased=True, shade=True, zorder=8)
        for edge in (Pi[0], Pi[-1], Pi[:, 0], Pi[:, -1]):
            axq.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=c_in, lw=1.4, zorder=9)

        # The SOLVED contact. seed_l is this stage's anchor (t=0); the solved
        # offset is t_sol when the trace carries it, so a contact sitting on
        # its own bound -- the pinned case quadratic_pin_frac exists to catch --
        # shows up as a marker against the patch edge rather than at its centre.
        t_sol = fr.get("t_sol")
        if t_sol is not None:
            t_sol = np.asarray(t_sol, float).reshape(-1)
            p_sol_l = _patch_points(fr, (t_sol[0], t_sol[0]), (t_sol[1], t_sol[1]),
                                    n=1)[0, 0]
        else:
            p_sol_l = np.asarray(fr["seed_l"], float)
        p_sol_w = _to_world(p_sol_l[None, :], center, R)[0]
        axq.scatter(*p_sol_w, color="k", marker="X", s=70, zorder=12)

        info = (f"κ=({float(fr['kappa0']):+.1f},{float(fr['kappa1']):+.1f})"
                f"{' [planar]' if abs(float(fr['kappa0'])) < 1e-9 and abs(float(fr['kappa1'])) < 1e-9 else ''}\n"
                f"t0∈[{lo0*1e3:+.0f},{hi0*1e3:+.0f}]mm  "
                f"t1∈[{lo1*1e3:+.0f},{hi1*1e3:+.0f}]mm")
        if sdf_fn is not None:
            try:
                errs = [abs(float(sdf_fn(p))) for p in
                        _patch_points(fr, (lo0, hi0), (lo1, hi1), n=9).reshape(-1, 3)]
                info += f"\nmax|SDF| over patch = {max(errs)*1e3:.2f}mm"
            except Exception:
                pass
        if t_sol is not None:
            # How close the solve sat to its own bound, per axis.
            f0 = abs(t_sol[0]) / max(abs(hi0 if t_sol[0] >= 0 else lo0), 1e-9)
            f1 = abs(t_sol[1]) / max(abs(hi1 if t_sol[1] >= 0 else lo1), 1e-9)
            info += (f"\nsolved t=({t_sol[0]*1e3:+.1f},{t_sol[1]*1e3:+.1f})mm "
                     f"= ({f0*100:.0f}%,{f1*100:.0f}%) of bound")

        _equal_axes(axq, np.vstack([Pw.reshape(-1, 3), p_sol_w[None, :]]), min_r=0.012)
        axq.set_title(f"{FINGER_NAMES[ci]}\n{info}", fontsize=8)
        axq.view_init(elev=elev, azim=azim)
        axq.set_xlabel("x (m)", fontsize=7); axq.set_ylabel("y (m)", fontsize=7)
        axq.set_zlabel("z (m)", fontsize=7); axq.tick_params(labelsize=6)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.05, wspace=0.16)
    out_path = OP.savefig(fig, out_path, dpi=115)
    plt.close(fig)
    return out_path

"""Visualize the SEEDING strategy and its local quadratic surrogates, decoupled
from the NLP / controller.

The seeding pipeline (MultiStartGraspPlanner3D.solve) and the local-quadratic
contact patch (_mesh_quadratic_contact_ca) are normally only observable through
whatever the full solve happens to converge to. This script exercises the SAME
functions in isolation -- no opti, no IK, no collision model -- and draws:

  1. SEED RAYS AND THEIR PROJECTION. Each seed source casts a ray and lands the
     result on the surface with _project_to_surface_np:
     BOTH SOURCES RAY FROM THE SAME ORIGIN: _ray_origin_local, the mesh's
     volumetric centroid, drawn "P" in every ray panel. The geom frame origin
     (the SDF frame centre) is NOT drawn by default -- it is not part of the
     seeding geometry, and on a YCB scan it sits far outside the object where
     it only adds a distracting marker. --ray-origins brings it back for the
     one question it answers: why raying from it was wrong (measured |centroid
     - geom origin|: 036_wood_block 112mm, 065-a_cups 63mm, 017_orange 44mm).
       - minor-axis seed (_fixed_antipodal_seed): the deterministic pair tried
         FIRST by every solve. Rays from the SHARED CENTROID along the object's
         minor principal axis. Deliberately
         NOT the body origin: YCB scans put the origin wherever the capture rig
         had it, commonly the object's base -- 036_wood_block's sits 103mm below
         its centre of mass on a 207mm block, so raying through it exits at the
         bottom RIM and both contacts land on an edge. See that function's
         docstring for the full failure chain. Drawn as a dashed line through
         the object with the two ray endpoints (bbox_r out along +/-axis) marked
         hollow, and an arrow from each endpoint to its surface projection.
       - random seeds (_seed_pair): a random direction from the SHARED CENTROID
         projected to p1s, then a
         z-rotation-jittered march THROUGH the object (_march_sdf_np) to the
         antipodal footprint p2s. Both legs are drawn, so the "project, then
         march" structure is visible rather than inferred.
     Rejected seeds are drawn too (grey/red), since which seeds the curvature
     gate throws away is as informative as which survive.

  2. THE QUADRATIC APPROXIMATION, two colors per finger. For each accepted seed
     contact, _mesh_quadratic_contact_ca is called exactly as _run_stage calls
     it, and its returned `frame` dict is used to reconstruct the paraboloid
       p_l(t1,t2) = seed_l + t1*axis0_l + t2*axis1_l + h(t1,t2)*n_l
       h(t1,t2)   = -(kappa0*t1^2 + kappa1*t2^2) / (2*grad_norm)
     over the ASYMMETRIC per-axis trust region [t_lo_i, t_hi_i]. Each finger gets
     two colors: a saturated surface inside the trust region (where the surrogate
     is measured-good, |SDF| <= sdf_err_tol) and a pale wireframe extension
     beyond it (where the model is extrapolating). That is the whole point of the
     bound, so both sides of it are drawn.

  3. TRANSPARENT OBJECT MESH. The visual mesh (group 2, the same vertices
     _mesh_local_surface_fit_np fits to) as a low-alpha Poly3DCollection, so
     patch-vs-surface agreement is judged against the real geometry.

Usage
-----
    python benchmarks/ycb_grasp/plot_seed_quadratic.py                 # all three objects
    python benchmarks/ycb_grasp/plot_seed_quadratic.py 017_orange
    python benchmarks/ycb_grasp/plot_seed_quadratic.py 036_wood_block --n-random 3
    python benchmarks/ycb_grasp/plot_seed_quadratic.py --no-mesh-fit   # SDF-Hessian curvature

--mesh-fit (default on) matches cfg.quadratic_mesh_fit=True: curvature fitted to
mesh vertices with the plane-vs-quadratic model-selection test, so a flat face
reports kappa=0 and gets a trust region sized by the direct SDF search instead of
a curvature contaminated by a corner 50mm away.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import mujoco as mj                                               # noqa: E402
import casadi as ca                                               # noqa: E402

from ycb_grasp import pick_from_floor as P, scene as S, workspace as W   # noqa: E402
from ycb_grasp import out_paths as OP                                    # noqa: E402
from ycb_grasp.ik_demo import clearance_by_geom, robot_geom_names        # noqa: E402
from grasp_control import object_uv_atlas as oua                         # noqa: E402
from simulation.grasp_config_builder import for_ablation_default         # noqa: E402
from simulation.grasp_config_builder import load_seed_config
from simulation.grasp_planner_3d import (                                # noqa: E402
    _GEOM_TYPE_MESH,
    _fixed_antipodal_seed,
    _geom_normal_np,
    _geom_sdf_np,
    _mesh_quadratic_contact_ca,
    _mesh_surface_kappa_max_np,
    _minor_axis_local,
    _ray_origin_local,
    _project_to_surface_np,
    _reachable_contact,
    _seed_pair,
    MultiStartGraspPlanner3D,
)

DEFAULT_OBJECTS = ["036_wood_block", "017_orange", "065-a_cups"]

# Two colors per finger: (inside trust region, outside / extrapolated).
# One colour per finger, used for EVERY mark that belongs to that contact: its
# ray endpoint and projection arrow in the ray panel, its surface footprint, and
# its patch. There is no second (pale) colour: the patch panel draws the trust
# region only, so there is no extrapolation band for a pale colour to mean.
FINGER_COLORS = {
    "thumb": "#d94801",   # orange
    "index": "#2171b5",   # blue
}


# ─────────────────────────────────────────────────────────────────────────────
# Scene setup — same construction path as plot_mesh_fit.py / pick_from_floor
# ─────────────────────────────────────────────────────────────────────────────
def build_object(obj: str, seed: int = 4):
    """Place `obj` on the floor and settle it, returning everything the seeding
    functions need. Mirrors plot_mesh_fit.py's setup so the geometry seen here is
    the geometry a benchmark run would see."""
    rng = np.random.default_rng(seed)
    base = mj.MjModel.from_xml_path(str(REPO / "models" / "scene_kinova_leap.xml"))
    ws = W.load_or_build(base, n=200_000, seed=0)
    pos, quat = P.place_object_on_floor(obj, ws, rng)
    model, data, info = S.build([(obj, pos, quat)])
    bn = next(iter(info))
    bid = info[bn]["bid"]
    mj.mj_forward(model, data)
    P.settle_object_on_floor(model, data, bid)

    rg = robot_geom_names(model)
    # Same seed/surrogate settings the solver gets (models/grasp_seed_config.json),
    # so this diagnostic cannot drift from the gate it is drawing. plot_object's
    # --sdf-err-tol / --t-bound-max flags are applied AFTER this and still win.
    cfg = for_ablation_default(
        obj_geom=S.hull_geoms(model, bn)[0], obj_body=bn, n_seeds=1,
        arm_geom_names=rg, obj_clearance_by_geom=clearance_by_geom(rg),
        **load_seed_config(obj))
    ms = MultiStartGraspPlanner3D(model, data, cfg)
    pl = ms._planner

    gid = pl._obj_gid
    geom_type = int(model.geom_type[gid])
    geom_size = model.geom_size[gid].copy()
    mesh_entry = pl._mesh_entry

    # MESH pose comes from the BODY (object_sdf's table is body-frame), matching
    # MultiStartGraspPlanner3D.solve's own branch.
    if geom_type == _GEOM_TYPE_MESH:
        center = data.xpos[pl._obj_bid].copy()
        R = data.xmat[pl._obj_bid].reshape(3, 3).copy()
    else:
        center = data.geom_xpos[gid].copy()
        R = data.geom_xmat[gid].reshape(3, 3).copy()

    Vvis, Fvis = oua.body_visual_mesh(model, bid)     # body frame
    return dict(obj=obj, model=model, data=data, body=bn, cfg=cfg, planner=pl,
                geom_type=geom_type, geom_size=geom_size, mesh_entry=mesh_entry,
                center=center, R=R, Vvis=Vvis, Fvis=Fvis,
                # Mirrors solve()'s own _r_tip_min rule (flat
                # seed_ground_clearance_m override, else the bounding-sphere
                # radius) so this plot cannot drift from the real seed gate.
                r_tip=(float(cfg.seed_ground_clearance_m)
                       if cfg.seed_ground_clearance_m is not None
                       else min(cfg.r_thumb, cfg.r_index)),
                ground_z=cfg.ground_z)


# ─────────────────────────────────────────────────────────────────────────────
# Seed generation — replays solve()'s seeding order, recording the RAYS
# ─────────────────────────────────────────────────────────────────────────────
def _kappa_max(sc, p_world):
    """Seed-gate curvature at a world point (solve()'s _seed_kappa_ok inner test)."""
    if sc["geom_type"] != _GEOM_TYPE_MESH or sc["mesh_entry"] is None:
        return 0.0
    p_l = sc["R"].T @ (p_world - sc["center"])
    return float(_mesh_surface_kappa_max_np(sc["mesh_entry"], p_l))


def _gate(sc, seed):
    """Apply solve()'s two seed gates and report WHY a seed was rejected, so the
    plot can show discarded seeds rather than silently dropping them."""
    reach = (_reachable_contact(seed["p1s"], sc["ground_z"], sc["r_tip"]) and
             _reachable_contact(seed["p2s"], sc["ground_z"], sc["r_tip"]))
    k1 = _kappa_max(sc, seed["p1s"])
    k2 = _kappa_max(sc, seed["p2s"])
    k_lim = float(sc["cfg"].seed_kappa_max_reject)
    kappa_ok = (k_lim <= 0) or (max(k1, k2) <= k_lim)
    if not reach:
        return False, "unreachable (too near floor)", (k1, k2)
    if not kappa_ok:
        return False, f"kappa {max(k1, k2):.0f} > {k_lim:.0f}", (k1, k2)
    return True, "accepted", (k1, k2)


def generate_seeds(sc, n_random: int = 3, rng_seed: int = 0):
    """Minor-axis seed then random _seed_pair seeds, each annotated with the RAY
    geometry that produced it (the thing part 1 of this plot is about).

    Deliberately does NOT stop at the first n accepted seeds the way solve() does
    -- rejected seeds are kept and returned so they can be drawn."""
    gt, gs, c, R, me = (sc["geom_type"], sc["geom_size"], sc["center"],
                        sc["R"], sc["mesh_entry"])
    bbox_r = float(np.max(gs)) * 2.5
    out = []

    # ── minor-axis (deterministic, tried FIRST by every solve) ──────────────
    axis_l = _minor_axis_local(gt, gs, mesh_entry=me)
    fs = _fixed_antipodal_seed(gt, gs, c, R, axis_l, mesh_entry=me)
    d_world = R @ axis_l
    d_world /= np.linalg.norm(d_world) + 1e-12
    # THE shared seed ray origin, straight from the planner -- not a local copy
    # of the rule, so this plot cannot drift from what the solver actually does.
    c_ray = c + R @ _ray_origin_local(gt, me)
    ok, why, kk = _gate(sc, fs)
    out.append(dict(seed=fs, kind="minor-axis", ok=ok, why=why, kappa=kk,
                    ray=dict(mode="axis", origin=c_ray, dir=d_world, length=bbox_r,
                             origin_kind="shared centroid", geom_center=c,
                             hull_centroid=c_ray)))

    # ── random _seed_pair seeds ────────────────────────────────────────────
    # solve() resets to a fixed constant every solve (determinism); a plain
    # default_rng here reproduces the same style of stream.
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_random):
        # Re-derive the ray by drawing from a CLONED stream first, so the ray we
        # draw is exactly the one _seed_pair consumed (it draws u, then ang).
        st = rng.bit_generator.state
        probe = np.random.default_rng(0)
        probe.bit_generator.state = st
        u = probe.standard_normal(3)
        u[2] *= 0.5                       # _seed_pair's bias away from top/bottom
        u /= np.linalg.norm(u) + 1e-12

        s = _seed_pair(gt, gs, c, R, bbox_r, rng,
                       delta_max=np.deg2rad(sc["cfg"].seed_march_jitter_deg),
                       mesh_entry=me)
        ok, why, kk = _gate(sc, s)
        out.append(dict(seed=s, kind="random", ok=ok, why=why, kappa=kk,
                        ray=dict(mode="march", origin=c_ray, dir=u, length=bbox_r,
                                 origin_kind="shared centroid", geom_center=c,
                                 hull_centroid=c_ray)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Quadratic patch — calls the REAL _mesh_quadratic_contact_ca, uses its `frame`
# ─────────────────────────────────────────────────────────────────────────────
def quad_frame(sc, p_world, n_in, cfg):
    """Run _mesh_quadratic_contact_ca exactly as _run_stage does and return its
    `frame` dict (the paraboloid's object-local parameters), or None for a
    non-mesh object. A throwaway ca.Opti absorbs the decision variable -- the
    frame itself is pure numpy and is what we draw."""
    if sc["geom_type"] != _GEOM_TYPE_MESH or sc["mesh_entry"] is None:
        return None
    opti = ca.Opti()
    _, _, _, frame = _mesh_quadratic_contact_ca(
        opti, p_world, -np.asarray(n_in, float), sc["center"], sc["R"], sc["mesh_entry"],
        t_bound_max=cfg.quadratic_t_bound_max,
        sdf_err_tol=cfg.quadratic_sdf_err_tol,
        mesh_fit=cfg.quadratic_mesh_fit,
        mesh_fit_radius=cfg.quadratic_mesh_fit_radius,
        mesh_fit_quad_gain_min=cfg.quadratic_mesh_fit_gain_min)
    return frame


def patch_points(frame, center, R, t0_range, t1_range, n=13):
    """Evaluate p_world(t1,t2) on a grid, using the frame's own reconstruction
    formula (documented in _mesh_quadratic_contact_ca's return contract -- valid
    outside the bounds too, which is exactly what the pale extrapolation band
    needs)."""
    t0 = np.linspace(t0_range[0], t0_range[1], n)
    t1 = np.linspace(t1_range[0], t1_range[1], n)
    T0, T1 = np.meshgrid(t0, t1)
    h = -(frame["kappa0"] * T0**2 + frame["kappa1"] * T1**2) / (2.0 * frame["grad_norm"])
    p_l = (frame["seed_l"][None, None, :]
           + T0[..., None] * frame["axis0_l"][None, None, :]
           + T1[..., None] * frame["axis1_l"][None, None, :]
           + h[..., None] * frame["n_l"][None, None, :])
    return center[None, None, :] + p_l @ R.T


def patch_sdf_err(sc, frame, t0_range, t1_range, n=9):
    """Max |true SDF| over the drawn patch -- the quantity the trust region is
    sized against, reported so the picture carries its own error bar."""
    if sc["mesh_entry"] is None:
        return float("nan")
    pts = patch_points(frame, sc["center"], sc["R"], t0_range, t1_range, n=n)
    errs = [abs(_geom_sdf_np(p, sc["geom_type"], sc["center"], sc["R"], sc["geom_size"],
                             mesh_entry=sc["mesh_entry"]))
            for p in pts.reshape(-1, 3)]
    return float(np.max(errs))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def draw_mesh(ax, sc, alpha=0.10, max_tris=3000):
    """Transparent object mesh, world frame.

    Drawn as thin-edged translucent triangles rather than filled faces: a YCB
    scan carries 5-20k triangles, and at that density even alpha=0.05 fills
    stack into an opaque blob that hides the very patches this plot is about.
    Edges at low alpha read as a see-through shell instead. Triangles are
    subsampled (deterministically) above max_tris purely for render time -- the
    silhouette is unaffected at these counts."""
    Vw = sc["center"] + sc["Vvis"] @ sc["R"].T
    F = sc["Fvis"]
    if len(F) > max_tris:
        F = F[np.linspace(0, len(F) - 1, max_tris).astype(int)]
    pc = Poly3DCollection(Vw[F], facecolor="0.6", edgecolor="0.45",
                          linewidths=0.15, alpha=alpha, zsort="min")
    ax.add_collection3d(pc)
    return Vw


def draw_seed_rays(ax, sc, rec, show_geom_origin=False):
    """Part 1: the ray, its endpoint, and the projection onto the surface."""
    s, ray = rec["seed"], rec["ray"]
    ok = rec["ok"]
    col = "0.25" if ok else "#cb181d"
    a = 0.9 if ok else 0.45

    # Per-contact marks (ray endpoint, projection arrow, footprint) take THAT
    # contact's finger colour, so the ray panel and the patch panels below it
    # use one consistent encoding. The ray LINE and the centroid stay neutral:
    # they are shared by both contacts, so colouring them for one finger would
    # be a lie. A rejected seed overrides everything to red.
    def _fcol(key):
        return FINGER_COLORS[key] if ok else "#cb181d"

    if ray["mode"] == "axis":
        # Deterministic minor-axis pair: one line through the object, both
        # endpoints projected inward onto the surface.
        o, d, L = ray["origin"], ray["dir"], ray["length"]
        e1, e2 = o + d * L, o - d * L
        ax.plot(*np.array([e1, e2]).T, color=col, ls="--", lw=1.0, alpha=a, zorder=4)
        for e, p, key in ((e1, s["p1s"], "thumb"), (e2, s["p2s"], "index")):
            fc = _fcol(key)
            ax.scatter(*e, facecolor="none", edgecolor=fc, s=28, lw=1.2, alpha=a, zorder=5)
            _arrow(ax, e, p, fc, a)
    else:
        # Random seed: (a) centre -> ray endpoint -> project to p1s, then
        # (b) march from p1s through the object -> p2s.
        o, d, L = ray["origin"], ray["dir"], ray["length"]
        e1 = o + d * L
        fc = _fcol("thumb")
        ax.plot(*np.array([o, e1]).T, color=col, ls=":", lw=0.9, alpha=a, zorder=4)
        ax.scatter(*e1, facecolor="none", edgecolor=fc, s=28, lw=1.2, alpha=a, zorder=5)
        _arrow(ax, e1, s["p1s"], fc, a)

    # RAY ORIGIN. Both seed sources ray from the same point --
    # _ray_origin_local, the mesh's volumetric centroid. The SDF frame centre
    # (geom origin) is deliberately NOT drawn here: it is not part of the
    # seeding geometry, it is only the frame the SDF/normal functions are
    # defined in, and on a YCB scan it sits far outside the object where it
    # just adds a distracting marker. The --ray-origins flag brings it back
    # for the one question it answers (why the geom origin was the wrong
    # thing to ray from).
    hc = ray.get("hull_centroid")
    if hc is not None:
        ax.scatter(*hc, marker="P", s=95, color=col, edgecolor="k", lw=0.6,
                   alpha=a, zorder=6)
    if show_geom_origin and ray.get("geom_center") is not None:
        gc = ray["geom_center"]
        ax.scatter(*gc, marker="X", s=45, facecolor="none", edgecolor=col,
                   lw=1.1, alpha=a * 0.5, zorder=6)

    # PAIRING: the chord joining this seed's two footprints. Drawn for BOTH
    # seed kinds -- for a random seed it is also the physical march leg
    # (_march_sdf_np's path from p1s through the object to p2s), for the
    # minor-axis seed it coincides with the ray. Its job in either case is to
    # make "these two contacts are one pair" unambiguous within the panel.
    ax.plot(*np.array([s["p1s"], s["p2s"]]).T, color=col, ls="-.", lw=1.2,
            alpha=a * 0.85, zorder=4)

    # Landed surface footprints. Which is thumb and which is index
    # (_assign_seed_by_finger's output) is carried by COLOR alone -- the same
    # color that fills that finger's patch in the rows below -- so the figure
    # needs one shared legend instead of a per-point text tag in every panel.
    for p, key in ((s["p1s"], "thumb"), (s["p2s"], "index")):
        ax.scatter(*p, color=_fcol(key), s=52,
                   edgecolor="k", lw=0.5, alpha=1.0 if ok else 0.55, zorder=6)


def s_pts(rec):
    """This seed's two surface footprints, as a (2,3) array -- the points a ray
    panel must keep in frame regardless of how far out the ray itself starts."""
    return np.array([rec["seed"]["p1s"], rec["seed"]["p2s"]], float)


def _arrow(ax, a, b, color, alpha):
    """Projection arrow: ray endpoint -> its _project_to_surface_np image."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax.plot(*np.array([a, b]).T, color=color, lw=1.4, alpha=alpha, zorder=5)
    ax.scatter(*b, color=color, marker="o", s=10, alpha=alpha, zorder=5)


def draw_quadratic(ax, sc, frame, key):
    """Part 2: the paraboloid over its measured trust region.

    Only the trust region is drawn. An earlier version added a pale wireframe
    at 1.8x the bounds to show where the surrogate is extrapolating, but the
    panel's whole subject IS the trust region -- the bounds are what
    _sdf_axis_bound_np measured -- and a second, larger surface around it reads
    as part of the patch and works against that.
    """
    c_in = FINGER_COLORS[key]
    center, R = sc["center"], sc["R"]
    lo0, hi0 = frame["t_lo_0"], frame["t_hi_0"]
    lo1, hi1 = frame["t_lo_1"], frame["t_hi_1"]

    P_in = patch_points(frame, center, R, (lo0, hi0), (lo1, hi1), n=13)
    ax.plot_surface(P_in[..., 0], P_in[..., 1], P_in[..., 2],
                    color=c_in, alpha=0.55, linewidth=0, antialiased=True,
                    shade=True, zorder=8)
    # Trust-region boundary, drawn solid so the asymmetry (t_lo != t_hi) reads.
    for edge in (P_in[0], P_in[-1], P_in[:, 0], P_in[:, -1]):
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=c_in, lw=1.4, zorder=9)


def _equal_axes(ax, pts, pad=0.005, min_r=None):
    """Equal-aspect cube around `pts`. Aspect must be equal or a paraboloid's
    curvature is a plotting artifact rather than the fit's.

    min_r floors the half-span so a tightly-bounded patch (017_orange's ~6mm
    trust region) still shows some surrounding surface for context instead of
    filling the panel edge to edge."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    mid = 0.5 * (pts.max(0) + pts.min(0))
    r = 0.5 * float(np.max(pts.max(0) - pts.min(0))) + pad
    if min_r is not None:
        r = max(r, float(min_r))
    # autoscale OFF first: an added Poly3DCollection (the object mesh) otherwise
    # re-expands the limits to the whole object, which un-zooms a patch panel
    # back to a few pixels of paraboloid.
    ax.set_autoscale_on(False)
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_box_aspect((1, 1, 1))


def plot_object(obj: str, n_random: int, mesh_fit: bool, rng_seed: int,
                out_dir: Path, elev: float, azim: float,
                show_rejected: bool = False, show_geom_origin: bool = False,
                sdf_err_tol: float | None = None,
                t_bound_max: float | None = None):
    sc = build_object(obj)
    cfg = sc["cfg"]
    cfg.quadratic_mesh_fit = mesh_fit
    # The two knobs that size a patch. sdf_err_tol is the model-validity
    # tolerance _sdf_axis_bound_np binary-searches against (how far the
    # paraboloid may depart from the true SDF before that axis's bound stops);
    # t_bound_max caps the half-width even where the surface stays flat enough
    # to never trip the tolerance. Overriding either here is the whole point of
    # a tuning sweep -- the solver reads the same two cfg fields.
    if sdf_err_tol is not None:
        cfg.quadratic_sdf_err_tol = float(sdf_err_tol)
    if t_bound_max is not None:
        cfg.quadratic_t_bound_max = float(t_bound_max)
    recs = generate_seeds(sc, n_random=n_random, rng_seed=rng_seed)

    accepted = [r for r in recs if r["ok"]]
    # Which seeds get a patch column. Normally the accepted ones -- but on an
    # object where the curvature gate rejects EVERYTHING (065-a_cups: a thin
    # shell whose rim/wall reports kappa 91-323 against a limit of 40), the
    # accepted list is empty and the figure would carry no patches at all. The
    # rejected seeds are then the whole story, so --show-rejected draws their
    # patches too, marked as such: it shows exactly what the gate is refusing.
    panel_recs = accepted if (accepted and not show_rejected) else recs
    # Layout: ONE COLUMN PER SEED, read top to bottom --
    #   row 0: that seed's OWN ray + projection panel, whole-object framing
    #   row 1: its thumb paraboloid   row 2: its index paraboloid
    # Every seed gets its own ray panel rather than sharing one overview,
    # because an overview draws every pair in the same style and gives no way
    # to tell WHICH two footprints belong to the same pair -- the pairing is
    # the thing being inspected. One pair per axes makes it unambiguous, and
    # the two footprints are additionally tied together by the connecting
    # chord and a matching p1s/p2s annotation.
    # Patches get separate axes from the rays (and from each other) because a
    # pair's two contacts sit on opposite sides of the object: framing both in
    # one panel forces an object-scale zoom, which is what makes a 6mm trust
    # region (017_orange) unreadable.
    n_cols = len(panel_recs)
    # Width scales with the column count but has a FLOOR: at one accepted
    # seed (017_orange) a purely proportional width leaves no room for the
    # suptitle or the legend, both of which are figure-wide.
    fig = plt.figure(figsize=(max(4.6 * n_cols, 11.0), 12.4))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[1.05, 1.0, 1.0])
    fit_tag = "mesh-fit curvature" if mesh_fit else "SDF-Hessian curvature"
    n_rej = len(recs) - len(accepted)
    fig.suptitle(
        f"{obj}  —  seeding strategy and local quadratic surrogate  ({fit_tag})\n"
        f"{len(accepted)}/{len(recs)} seeds accepted, {n_rej} rejected "
        f"(kappa gate {cfg.seed_kappa_max_reject:.0f})\n"
        "one column per seed: ray → projection, then its thumb and index paraboloid"
        "  —  patch = the measured trust region (t_lo/t_hi per axis)",
        fontsize=10.5)

    # ── Row 0: each seed's own ray + projection, over the transparent mesh ──
    for i, rec in enumerate(panel_recs):
        axr = fig.add_subplot(gs[0, i], projection="3d")
        Vw = draw_mesh(axr, sc)
        draw_seed_rays(axr, sc, rec, show_geom_origin=show_geom_origin)
        # Frame on the OBJECT (plus a small margin), not on the ray extent.
        # A seed ray starts 2.5*max(geom_size) out, so including its far tail
        # in an equal-aspect cube shrinks the object to a blob in the middle
        # and pushes the footprints on top of each other. The ray tails simply
        # clip at the panel edge, which costs nothing -- the informative part
        # is the last stretch where the ray meets the surface.
        _equal_axes(axr, np.vstack([Vw, s_pts(rec)]), pad=0.03)
        _rej = "" if rec["ok"] else f"\nREJECTED: {rec['why']}"
        _d = np.linalg.norm(np.asarray(rec["ray"]["hull_centroid"], float)
                            - np.asarray(rec["ray"]["geom_center"], float))
        _gap = (f"  (centroid–geom-origin gap {_d*1e3:.0f}mm)"
                if show_geom_origin else "")
        axr.set_title(f"seed {i} ({rec['kind']}) — ray → projection{_gap}"
                      f"{_rej}", fontsize=8.5,
                      color="k" if rec["ok"] else "#cb181d")
        axr.view_init(elev=elev, azim=azim)
        axr.set_xlabel("x (m)", fontsize=7); axr.set_ylabel("y (m)", fontsize=7)
        axr.set_zlabel("z (m)", fontsize=7); axr.tick_params(labelsize=6)

    # ── Rows 1-2: that same seed's two contacts' paraboloids ───────────────
    # Framed on the PATCHES, not the object: a trust region is ~5-30mm on a
    # 60-200mm object, so an object-framed view renders it as a few pixels.
    # The mesh is still drawn (clipped by the zoom) so the patch is read
    # against the real surface it is approximating.
    lines = []
    for i, rec in enumerate(panel_recs):
        s = rec["seed"]
        for row, (p, n_in, key) in enumerate(((s["p1s"], s["n1_in"], "thumb"),
                                              (s["p2s"], s["n2_in"], "index"))):
            axq = fig.add_subplot(gs[1 + row, i], projection="3d")
            draw_mesh(axq, sc, alpha=0.12)
            fr = quad_frame(sc, p, n_in, cfg)
            axq.scatter(*p, color=FINGER_COLORS[key], s=48, edgecolor="k",
                        lw=0.5, zorder=11)
            if fr is None:
                _equal_axes(axq, p.reshape(1, 3), min_r=0.03)
                axq.set_title(f"seed {i} ({rec['kind']}) — {key}\n"
                              "analytic geom (no mesh patch)", fontsize=8)
            else:
                draw_quadratic(axq, sc, fr, key)
                err = patch_sdf_err(sc, fr, (fr["t_lo_0"], fr["t_hi_0"]),
                                    (fr["t_lo_1"], fr["t_hi_1"]))
                planar = (abs(fr["kappa0"]) < 1e-9 and abs(fr["kappa1"]) < 1e-9)
                txt = (f"κ=({fr['kappa0']:+.1f},{fr['kappa1']:+.1f})"
                       f"{' [planar]' if planar else ''}\n"
                       f"t0∈[{fr['t_lo_0']*1e3:+.0f},{fr['t_hi_0']*1e3:+.0f}]mm  "
                       f"t1∈[{fr['t_lo_1']*1e3:+.0f},{fr['t_hi_1']*1e3:+.0f}]mm\n"
                       f"max|SDF| over patch = {err*1e3:.2f}mm "
                       f"(tol {cfg.quadratic_sdf_err_tol*1e3:.1f}mm)")
                # Framed just outside the trust region (not the old 1.8x that
                # matched the removed extrapolation band, which now leaves the
                # patch floating in empty space).
                pts = patch_points(fr, sc["center"], sc["R"],
                                   (fr["t_lo_0"] * 1.25, fr["t_hi_0"] * 1.25),
                                   (fr["t_lo_1"] * 1.25, fr["t_hi_1"] * 1.25),
                                   n=5).reshape(-1, 3)
                _equal_axes(axq, np.vstack([pts, p.reshape(1, 3)]), min_r=0.012)
                _rej = "" if rec["ok"] else f"  [REJECTED: {rec['why']}]"
                axq.set_title(f"seed {i} ({rec['kind']}){_rej} — {key}\n" + txt,
                              fontsize=8,
                              color="k" if rec["ok"] else "#cb181d")
                lines.append(f"  [{obj}] seed{i} {rec['kind']:>10s} {key:>5s}  "
                             + txt.replace("\n", "  "))
            axq.view_init(elev=elev, azim=azim)
            axq.set_xlabel("x (m)", fontsize=7); axq.set_ylabel("y (m)", fontsize=7)
            axq.set_zlabel("z (m)", fontsize=7); axq.tick_params(labelsize=6)

    # Legend describing the two-color-per-finger encoding. Colour is the ONLY
    # thumb/index cue in the panels (the per-point p1s/p2s tags were removed),
    # so each finger's entry names its contact POINT and its patch together.
    handles = [
        plt.Line2D([], [], color=FINGER_COLORS["thumb"], lw=6, marker="o",
                   markeredgecolor="k", markeredgewidth=0.5, ms=9,
                   label="thumb — contact point & patch"),
        plt.Line2D([], [], color=FINGER_COLORS["index"], lw=6, marker="o",
                   markeredgecolor="k", markeredgewidth=0.5, ms=9,
                   label="index — contact point & patch"),
        plt.Line2D([], [], color="0.25", marker="P", ls="none", ms=9,
                   label="volumetric centroid — shared ray origin"),
        plt.Line2D([], [], color="0.25", ls="--", label="minor-axis seed ray"),
        plt.Line2D([], [], color="0.25", ls=":", label="random seed ray"),
    ]
    # --ray-origins draws the geom origin as well; it is off by default, so this
    # entry does not count against the compact five-entry legend.
    if show_geom_origin:
        handles.append(plt.Line2D([], [], markerfacecolor="none",
                                  markeredgecolor="0.25", marker="X", ls="none",
                                  ms=8, label="geom origin (SDF frame centre)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=8, frameon=False, borderaxespad=0.0)

    # One SUBFOLDER per object. A tolerance/cap sweep emits a figure per
    # setting, so a flat directory turns into a pile the moment more than one
    # object is swept. The filename keeps the object token too, so a figure
    # pulled out of its folder is still self-identifying.
    obj_dir = out_dir / obj
    obj_dir.mkdir(parents=True, exist_ok=True)
    # Tag the filename with the tolerance whenever it is overridden, so a sweep
    # writes one figure per setting instead of clobbering a single file.
    _tag = "" if sdf_err_tol is None else f"_tol{cfg.quadratic_sdf_err_tol*1e3:g}mm"
    if t_bound_max is not None:
        _tag += f"_cap{cfg.quadratic_t_bound_max*1e3:g}mm"
    out = obj_dir / f"seedquad_{obj}{'' if mesh_fit else '_sdfhess'}{_tag}.png"
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.07,
                        wspace=0.14, hspace=0.24)
    out = OP.savefig(fig, out, dpi=115)
    plt.close(fig)

    print(f"-> {out}")
    for r in recs:
        if not r["ok"]:
            print(f"  [{obj}] {r['kind']:>10s} REJECTED: {r['why']}")
    for ln in lines:
        print(ln)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Distribution view — many rays at once, all objects in one figure
# ─────────────────────────────────────────────────────────────────────────────
def sample_ray_distribution(sc, n: int, rng_seed: int = 0):
    """`n` random seed directions drawn EXACTLY as _seed_pair draws them, with
    the surface footprint each one lands on.

    Reproduces _seed_pair's own two lines (standard_normal, then u[2] *= 0.5)
    rather than calling it, because we want only the ray and its first
    projection -- not the antipodal march, the gates, or the finger assignment.
    Anything that changes _seed_pair's direction sampling must be mirrored here;
    the shared ray ORIGIN comes from _ray_origin_local so at least that cannot
    drift."""
    gt, gs, c, R, me = (sc["geom_type"], sc["geom_size"], sc["center"],
                        sc["R"], sc["mesh_entry"])
    bbox_r = float(np.max(gs)) * 2.5
    c_ray = c + R @ _ray_origin_local(gt, me)
    rng = np.random.default_rng(rng_seed)
    dirs, hits = [], []
    for _ in range(n):
        u = rng.standard_normal(3)
        u[2] *= 0.5              # _seed_pair's bias away from top/bottom faces
        u /= np.linalg.norm(u) + 1e-12
        dirs.append(u)
        hits.append(_project_to_surface_np(c_ray + u * bbox_r, gt, c, R, gs,
                                           mesh_entry=me))
    return c_ray, np.array(dirs), np.array(hits)


def plot_ray_distribution(objects, n_rays, rng_seed, out_dir, elev, azim):
    """ONE figure, one column per object: where the seed rays actually go.

    Answers a different question from the per-seed panels -- not "what did this
    pair do" but "is the sampling covering the object at all". Two rows:
      top    -- the rays themselves, drawn from the shared centroid outward to
                the surface, over the transparent mesh.
      bottom -- the landed footprints alone, colored by height, plus the
                elevation histogram of the sampled directions. _seed_pair
                applies u[2] *= 0.5 to bias AWAY from the top and bottom faces;
                the histogram is where that bias is actually visible, and the
                footprint cloud is where its effect on coverage shows up.
    """
    n_cols = len(objects)
    fig = plt.figure(figsize=(max(5.2 * n_cols, 11.0), 9.4))
    gs_ = fig.add_gridspec(2, n_cols, height_ratios=[1.35, 1.0])
    fig.suptitle(
        f"seed ray distribution — {n_rays} random directions per object "
        f"(_seed_pair sampling, from the shared volumetric centroid)\n"
        "top: rays to their surface footprint   |   "
        "bottom: where those rays LAND, vs this shape's own surface-area "
        "profile (_seed_pair's u[2] *= 0.5 biases away from top/bottom faces)",
        fontsize=11)

    for i, obj in enumerate(objects):
        sc = build_object(obj)
        c_ray, dirs, hits = sample_ray_distribution(sc, n_rays, rng_seed)

        ax = fig.add_subplot(gs_[0, i], projection="3d")
        Vw = draw_mesh(ax, sc, alpha=0.10)
        # One line per ray, centroid -> footprint. Thin and translucent so the
        # DENSITY reads (a few hundred opaque lines would just be a solid ball).
        segs = np.stack([np.repeat(c_ray[None, :], len(hits), 0), hits], axis=1)
        ax.add_collection3d(Line3DCollection(segs, colors="#2171b5",
                                             linewidths=0.35, alpha=0.28))
        ax.scatter(hits[:, 0], hits[:, 1], hits[:, 2], s=5,
                   c=hits[:, 2], cmap="viridis", depthshade=False, zorder=6)
        ax.scatter(*c_ray, marker="P", s=90, color="#d94801",
                   edgecolor="k", lw=0.6, zorder=8)
        _equal_axes(ax, np.vstack([Vw, hits]), pad=0.01)
        ax.set_title(f"{obj}\n{len(hits)} rays from the shared centroid", fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7)
        ax.set_zlabel("z (m)", fontsize=7); ax.tick_params(labelsize=6)

        # WHERE THE RAYS LAND, as a height histogram over the object's own
        # z-extent. Deliberately NOT a histogram of the sampled directions:
        # _seed_pair's direction sampling is object-INDEPENDENT, so that plot is
        # identical in every column and says nothing per object. Coverage is
        # what differs, and it is what a seeding strategy is judged on -- a face
        # that never gets sampled cannot be grasped no matter how good the
        # downstream solve is.
        #
        # The reference line is the object's own surface-area distribution over
        # height, estimated from the visual mesh's triangle areas: that is what
        # "uniform coverage of THIS shape" would look like. Deviation from it is
        # the sampling bias, separated from the shape's own geometry (a cup is
        # mostly wall, so even perfect sampling is not flat in z).
        axh = fig.add_subplot(gs_[1, i])
        zlo, zhi = float(hits[:, 2].min()), float(hits[:, 2].max())
        _rng_z = max(zhi - zlo, 1e-6)
        axh.hist(hits[:, 2], bins=30, range=(zlo, zhi), color="#2171b5",
                 alpha=0.85, density=True, label="ray footprints")
        # area-weighted reference from the mesh triangles, in WORLD z
        Vw_l = sc["center"] + sc["Vvis"] @ sc["R"].T
        F = np.asarray(sc["Fvis"], int)
        _a, _b, _c3 = Vw_l[F[:, 0]], Vw_l[F[:, 1]], Vw_l[F[:, 2]]
        _ar = 0.5 * np.linalg.norm(np.cross(_b - _a, _c3 - _a), axis=1)
        _zc = (_a[:, 2] + _b[:, 2] + _c3[:, 2]) / 3.0
        _h, _e = np.histogram(_zc, bins=30, range=(zlo, zhi), weights=_ar)
        _w = (_e[1] - _e[0])
        _h = _h / max(_h.sum() * _w, 1e-12)
        axh.plot(0.5 * (_e[:-1] + _e[1:]), _h, color="#cb181d", lw=1.4, ls="--",
                 label="surface area of this shape")
        # Coverage: fraction of height bands that got at least one footprint.
        _cov = float(np.mean(np.histogram(hits[:, 2], bins=20,
                                          range=(zlo, zhi))[0] > 0)) * 100
        axh.set_title(f"footprint height — spans {_rng_z*1e3:.0f}mm, "
                      f"{_cov:.0f}% of bands hit", fontsize=8)
        axh.set_xlabel("footprint z (m)", fontsize=7)
        axh.set_ylabel("density", fontsize=7)
        axh.tick_params(labelsize=6); axh.grid(alpha=0.3)
        axh.legend(fontsize=6.5, frameon=False)

        el = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1, 1)))
        _frac = float(np.mean(np.abs(el) > 60.0)) * 100
        print(f"  [{obj}] {len(hits)} rays  "
              f"footprint z [{zlo:.3f}, {zhi:.3f}]m  "
              f"{_cov:.0f}% of height bands hit  "
              f"({_frac:.0f}% of directions steeper than +/-60deg, isotropic 13%)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "seedray_distribution.png"
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.07,
                        wspace=0.22, hspace=0.28)
    out = OP.savefig(fig, out, dpi=115)
    plt.close(fig)
    print(f"-> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("objects", nargs="*", default=None,
                    help=f"YCB object ids (default: {' '.join(DEFAULT_OBJECTS)})")
    ap.add_argument("--n-random", type=int, default=2,
                    help="random _seed_pair seeds to draw alongside the minor-axis seed")
    ap.add_argument("--no-mesh-fit", action="store_true",
                    help="use the SDF Hessian for curvature instead of the mesh-vertex fit")
    ap.add_argument("--show-rejected", action="store_true",
                    help="also draw patch panels for seeds the gates rejected "
                         "(automatic when NO seed is accepted, e.g. 065-a_cups)")
    ap.add_argument("--ray-origins", action="store_true",
                    help="also mark the geom origin (SDF frame centre) in the ray "
                         "panels; off by default since it is not part of the "
                         "seeding geometry")
    ap.add_argument("--ray-distribution", action="store_true",
                    help="instead of the per-seed panels, draw ONE figure showing "
                         "the distribution of sampled seed rays for every object")
    ap.add_argument("--n-rays", type=int, default=300,
                    help="rays per object for --ray-distribution")
    ap.add_argument("--sdf-err-tol", type=float, default=None,
                    help="metres; max surrogate-vs-true-SDF gap that sizes each "
                         "trust-region axis (default: GraspConfig3D's 5e-4). "
                         "Pass several times via a shell loop to sweep; the "
                         "figure filename carries the value.")
    ap.add_argument("--t-bound-max", type=float, default=None,
                    help="metres; hard cap on each patch half-width even where "
                         "the surface never trips --sdf-err-tol (default 0.05)")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    # out/analysis/seed_quadratic/ -- environment-independent diagnostic, so it
    # belongs under analysis/ rather than beside a floor/tabletop run, same as
    # plot_quadratic_path.py and plot_uv_path.py. See out_paths.py's docstring.
    ap.add_argument("--out-dir", type=Path,
                    default=OP.analysis_dir("seed_quadratic", create=False))
    a = ap.parse_args()

    objs = a.objects if a.objects else DEFAULT_OBJECTS
    if a.ray_distribution:
        plot_ray_distribution(objs, a.n_rays, a.rng_seed, a.out_dir,
                              a.elev, a.azim)
        return
    for obj in objs:
        plot_object(obj, a.n_random, not a.no_mesh_fit, a.rng_seed,
                    a.out_dir, a.elev, a.azim, a.show_rejected, a.ray_origins,
                    a.sdf_err_tol, a.t_bound_max)


if __name__ == "__main__":
    main()

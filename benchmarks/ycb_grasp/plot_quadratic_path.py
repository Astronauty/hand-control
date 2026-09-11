"""Visualize a mesh contact's local-quadratic (paraboloid) surrogate on the mesh.

Runs one GraspPlanner3D solve per (object, seed) with use_quadratic_contact=True
and log_dir set (so grasp_planner_3d.py's _opti_cb/_save_iter_npz record the
per-iteration (t1,t2) trajectory AND the paraboloid frame that trajectory is
defined in -- see _mesh_quadratic_contact_ca's docstring for the frame dict's
fields), then renders the mesh with, for EACH Picard relinearization stage:
  - the paraboloid patch p_local(t1,t2) = seed_l + t1*axis0_l + t2*axis1_l +
    h(t1,t2)*n_l, evaluated over a grid CLIPPED to the per-axis trust region
    (t_bound_0, t_bound_1) -- so the patch's own boundary IS the trust region,
    nothing is drawn past where the optimizer was allowed to go -- and pushed
    slightly off the true surface for visibility (same mplot3d depth-sort
    workaround plot_uv_path.py uses). Colored by FINGER, not stage (thumb=
    blue, index=orange/red, always) so the two contacts are distinguishable
    at a glance; which Picard stage a patch belongs to is instead stamped as
    a numbered circle directly on it, since consecutive re-fits (e.g. after a
    trust-region pin forced relinearization) still need to be told apart
  - the solved (t1,t2) trajectory WITHIN that stage, mapped through the same
    p_local(t1,t2) map (not re-derived by nearest-triangle lookup, unlike the
    UV-atlas visualizer -- the quadratic representation has a closed form)

Unlike plot_uv_path.py (which only loads the LAST stage's npz), this loads
EVERY grasp3d_iter_*.npz written during the solve, in filename (== creation)
order, since each Picard stage gets its own timestamped file and "subsequent
surfaces" is the whole point of this script.

    python benchmarks/ycb_grasp/plot_quadratic_path.py                       # default: 013_apple, seed 100
    python benchmarks/ycb_grasp/plot_quadratic_path.py --object 036_wood_block
    python benchmarks/ycb_grasp/plot_quadratic_path.py --object 025_mug --seed 103
    python benchmarks/ycb_grasp/plot_quadratic_path.py --gws --soft-finger
"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mujoco as mj
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from simulation.grasp_planner_3d import GraspConfig3D, MultiStartGraspPlanner3D  # noqa: E402
from ycb_grasp.plot_uv_path import _view_cam_dir                                # noqa: E402
from ycb_grasp import out_paths as OP                                           # noqa: E402
from ycb_grasp import scene as S, workspace as W                                # noqa: E402
from ycb_grasp.ik_demo import (clearance_by_geom, home_bias, place_objects,     # noqa: E402
                               render, robot_geom_names)
from ycb_grasp.ablate_grasp import _tip_err_mm                                  # noqa: E402
from grasp_control import object_uv_atlas as oua                                # noqa: E402
# Reuse plot_uv_path.py's camera-framing helpers rather than re-derive them --
# see that module for the mplot3d depth-sort / antipodal-pair framing notes
# these encode; nothing about them is UV-atlas-specific.

N_ROBOT = 23
DEFAULT_COL_CLEARANCE_M = 0.002

# One solid color PER FINGER, not per stage -- p1/thumb is always this blue,
# p2/index always this orange/red, regardless of how many Picard stages a
# solve needed. Which stage a given patch belongs to is carried entirely by
# the numbered circle stamped on it (see _draw_paraboloid_patch's stage_num),
# not by shading, so color always means "which finger" at a glance.
_FINGER_COLOR = {1: "#2166AC", 2: "#B2182B"}   # thumb=blue, index=red/orange


def _stage_color(ci: int, si: int) -> str:
    return _FINGER_COLOR[ci]


def _load_stage_npz(path):
    """One grasp3d_iter_*.npz -> the same per-stage dict shape used
    throughout this module (raw arrays + 'contact' sub-dicts keyed 1/2 with
    that contact's quad_* frame params, present only under
    use_quadratic_contact)."""
    trace = np.load(path, allow_pickle=True)
    stage = dict(
        p1=trace["p1"], p2=trace["p2"],
        stage_label=str(trace["stage_label"]), status_tag=str(trace["status_tag"]),
        obj_center=np.asarray(trace["obj_center"], float),
        obj_mat=np.asarray(trace["obj_mat"], float),
        n_iter=len(trace["iter"]),
        contact={},
    )
    for ci in (1, 2):
        prefix = f"quad{ci}_"
        keys = [k for k in trace.files if k.startswith(prefix)]
        if not keys:
            continue
        stage["contact"][ci] = {k[len(prefix):]: trace[k] for k in keys}
    return stage


def _iter_trace_quadratic_stages(log_dir, res: dict | None = None):
    """Load grasp3d_iter_*.npz in log_dir, in filename (== creation) order --
    one per Picard STAGE, unlike plot_uv_path.py's _iter_trace_uv_paths which
    only takes the last file.

    MultiStartGraspPlanner3D tries UP TO n_seeds candidate grasp attempts per
    solve, all logging into the SAME log_dir (GraspPlanner3D.log_dir is set
    once, not per attempt) -- so with n_seeds > 1, this directory can hold
    Picard stages from several UNRELATED attempts back to back, and the
    winning one (picked by cost/feasibility rank, see
    MultiStartGraspPlanner3D.solve's `results.sort(key=_rank); best =
    results[0]` -- NOT necessarily the last one tried) isn't identifiable
    from file order alone. Files are first split into contiguous ATTEMPTS
    (a new attempt starts whenever stage_label resets to 'S1' after a later
    stage); if `res` (the solve()'s returned dict) is given, the attempt
    matched is whichever one's FIRST stage's FIRST recorded point equals
    res['p1_seed']/res['p2_seed'] -- the exact pre-solve seed point
    MultiStartGraspPlanner3D stores on every result (grasp_planner_3d.py's
    `r['p1_seed'] = seed['p1s'].copy()`), fixed and known before any NLP
    iteration runs, so this is an EXACT identity match (sub-micron, not a
    proximity heuristic) rather than comparing post-solve positions, which
    two different attempts' Picard loops can converge close together on
    (measured: matching by final position left an 11.8mm residual on a case
    where two attempts' final contacts genuinely ended up near each other --
    seed-point matching has no such ambiguity, since seeds are drawn from
    different random directions and coincide with each other essentially
    never). Without `res` (or its p1_seed/p2_seed), falls back to the LAST
    attempt with a printed warning, since silently mixing attempts would show
    trust regions/patches from a discarded candidate as if they belonged to
    the returned solve.

    Returns a list of per-stage dicts (empty list if nothing saved)."""
    npz_files = sorted(glob.glob(os.path.join(log_dir, "grasp3d_iter_*.npz")))
    if not npz_files:
        return []
    all_stages = [_load_stage_npz(p) for p in npz_files]

    attempts = []
    for stage in all_stages:
        if stage["stage_label"] == "S1" or not attempts:
            attempts.append([stage])
        else:
            attempts[-1].append(stage)

    if len(attempts) == 1 or res is None or res.get("p1_seed") is None:
        if len(attempts) > 1:
            print(f"  NOTE: {len(attempts)} grasp attempts logged in this solve "
                 "(n_seeds > 1) but no res.p1_seed to match against -- using the "
                 "LAST attempt; it may not be the one actually returned/plotted "
                 "elsewhere in this script's own output.")
        return attempts[-1]

    target_p1_seed = np.asarray(res["p1_seed"], float)
    target_p2_seed = np.asarray(res["p2_seed"], float)
    best_idx, best_err = len(attempts) - 1, float("inf")
    for i, attempt in enumerate(attempts):
        first = attempt[0]
        err = max(np.linalg.norm(first["p1"][0] - target_p1_seed),
                  np.linalg.norm(first["p2"][0] - target_p2_seed))
        if err < best_err:
            best_idx, best_err = i, err
    if best_err > 1e-6:
        print(f"  NOTE: best-matching attempt ({best_idx+1}/{len(attempts)}) is still "
             f"{best_err*1000:.4f}mm from the returned result's p1_seed/p2_seed "
             "(should be ~exact) -- the winning attempt's trace may not have been "
             "logged (e.g. a pre-check LP rejection before any NLP stage ran).")
    return attempts[best_idx]


def _paraboloid_p_local(frame: dict, t0, t1):
    """p_local(t0,t1) = seed_l + t0*axis0_l + t1*axis1_l + h(t0,t1)*n_l --
    the SAME map _mesh_quadratic_contact_ca builds symbolically, evaluated
    here in plain numpy for arbitrary t0/t1 (grid or trajectory), from the
    saved frame dict (see that function's docstring for the derivation).
    t0/t1 may be scalars or same-shape arrays (e.g. a meshgrid)."""
    t0 = np.asarray(t0, float); t1 = np.asarray(t1, float)
    h = -(frame["kappa0"] * t0**2 + frame["kappa1"] * t1**2) / (2.0 * frame["grad_norm"])
    return (frame["seed_l"][None, :] if t0.ndim else frame["seed_l"]) + (
        t0[..., None] * frame["axis0_l"] + t1[..., None] * frame["axis1_l"]
        + h[..., None] * frame["n_l"])


def _paraboloid_patch_mesh(frame: dict, n: int = 14):
    """Grid the paraboloid over EXACTLY [-t_bound_0, t_bound_0] x (axis1
    likewise) -- the trust region itself, so the rendered surface's own
    boundary IS the trust-region edge (no separate wireframe ring needed to
    show where it is, and nothing is drawn past where the optimizer was
    actually allowed to go). Returns (P (n,n,3) object-local grid points,
    T0, T1) the caller can push off-surface and hand to Poly3DCollection."""
    b0 = max(float(frame["t_bound_0"]), 1e-4)
    b1 = max(float(frame["t_bound_1"]), 1e-4)
    t0 = np.linspace(-b0, b0, n)
    t1 = np.linspace(-b1, b1, n)
    T0, T1 = np.meshgrid(t0, t1, indexing="ij")
    P = _paraboloid_p_local(frame, T0, T1)
    return P, T0, T1


def _draw_mesh_3d(ax, V, F, color="#dcdcdc", alpha=None, edge_color="#333333", edge_lw=0.25):
    """Flat-colored mesh render with a thin wireframe over the fill --
    V (V,3)/F (F,3) from object_uv_atlas.body_visual_mesh. The plain
    edgecolors="none" fill (this module's and plot_uv_path.py's original
    convention) makes a mesh's silhouette/geometry hard to read since
    nothing marks triangle or face boundaries; a thin edge line fixes that
    without darkening the fill itself."""
    tv = V[F]
    if alpha is None:
        pc = Poly3DCollection(tv, facecolors=color, edgecolors=edge_color, linewidths=edge_lw)
    else:
        pc = Poly3DCollection(tv, facecolors=color, edgecolors=edge_color, linewidths=edge_lw, alpha=alpha)
    # Rasterize the OBJECT SHELL in vector output: its thin (0.25pt) wireframe
    # is sub-pixel in a raster render but full-weight in pdf/svg, where ~3000
    # strokes bury the contact patch. The patch's own boundary (linewidths=1.4,
    # drawn elsewhere in this module) is a deliberate feature line and stays
    # vector.
    pc.set_rasterized(True)
    # mplot3d sorts each Poly3DCollection as ONE unit by its average depth, not
    # per-triangle -- a many-triangle mesh collection's average can end up
    # "in front of" a small patch collection added later even where the real
    # geometry wouldn't occlude it at all (confirmed: a contact patch with a
    # healthy, non-degenerate trust region rendered fully invisible against
    # this mesh from both opposite camera views, which true per-triangle
    # occlusion couldn't do -- a real surface point front-facing in neither of
    # two antipodal views is impossible). set_sort_zpos(None) tells mplot3d
    # "draw this collection in add-order, don't sort it against others" --
    # already used for the alpha-mesh branch and for the patches/arrows below;
    # applying it here too keeps the (implicitly opaque) default mesh from
    # winning that same coarse sort against collections added after it.
    pc.set_sort_zpos(None)
    ax.add_collection3d(pc)
    return tv


def _draw_paraboloid_patch(ax, frame: dict, offset_mag: float, color: str,
                           label: str | None = None, stage_num: int | None = None,
                           num_offset: float | None = None,
                           cam_dir: np.ndarray | None = None):
    """One stage's paraboloid patch, rendered ONLY over the trust region
    itself (see _paraboloid_patch_mesh) and pushed outward along the patch's
    own normal for the same mplot3d depth-sort reason plot_uv_path.py's
    _draw_path_3d pushes contact paths off-surface. The patch's own outline
    (a thin edge on the boundary quads) IS the trust-region boundary now --
    no separate wireframe ring is drawn, since the surface simply doesn't
    extend past where the optimizer was allowed to go. If stage_num is
    given, stamps that number as a text label directly on the patch (at its
    seed point, offset further along the normal so it sits above the
    surface) -- lets a reader identify which Picard stage a given patch
    belongs to without cross-referencing the legend, which is the whole
    point given consecutive re-fits can visually overlap."""
    P, T0, T1 = _paraboloid_patch_mesh(frame)
    n_l = frame["n_l"]
    P_draw = P + n_l * offset_mag

    # Surface as a quad mesh -> quads for Poly3DCollection (edges shown only
    # on the OUTER boundary quads, giving a crisp trust-region silhouette
    # without a dense interior grid line at every one of the n-1 interior
    # quad seams).
    n0, n1 = P_draw.shape[0] - 1, P_draw.shape[1] - 1
    quads, edge_colors = [], []
    for i in range(n0):
        for j in range(n1):
            quads.append([P_draw[i, j], P_draw[i + 1, j], P_draw[i + 1, j + 1], P_draw[i, j + 1]])
            on_boundary = i in (0, n0 - 1) or j in (0, n1 - 1)
            edge_colors.append("k" if on_boundary else "none")
    pc = Poly3DCollection(quads, facecolors=color, edgecolors=edge_colors, linewidths=1.4, alpha=0.55)
    pc.set_sort_zpos(None)
    ax.add_collection3d(pc)

    # Explicit boundary line loop, ON TOP of the filled quads (high zorder,
    # depthshade off) -- belt-and-suspenders for a patch whose trust region is
    # highly anisotropic (one axis near t_bound_max, the other collapsed to a
    # couple mm by _sdf_axis_bound_np near an edge/corner -- see
    # _mesh_quadratic_contact_ca's own docstring on why that shrinkage is
    # asymmetric). Such a patch is a long, narrow ribbon whose face NORMAL can
    # end up nearly parallel to a camera's view axis by chance (confirmed on
    # 036_wood_block seed 2: a contact right on a corner rib rendered with a
    # completely invisible fill in BOTH opposite iso views -- a real point
    # can't be edge-on to two antipodal cameras at once via self-occlusion, so
    # this was the polygon collapsing to near-zero screen area, not a depth-
    # sort or hidden-surface bug). A closed 3D line loop has no such collapse
    # mode short of the view axis running exactly along the patch's own
    # long edge, which the shared two-iso-view camera pair (derived from the
    # grasp axis, not from any one patch's normal) essentially never hits for
    # both views simultaneously.
    loop = np.array([P_draw[0, 0], P_draw[-1, 0], P_draw[-1, -1], P_draw[0, -1], P_draw[0, 0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], "-", color=color, linewidth=2.2,
           zorder=15, solid_capstyle="round")

    if label is not None:
        seed_draw = frame["seed_l"] + n_l * offset_mag
        ax.scatter(*seed_draw, marker="x", s=80, color=color, linewidths=2.2,
                  zorder=9, depthshade=False, label=label)

    if stage_num is not None:
        # Text label pushed off-surface along the patch normal. The push is
        # sized from the OBJECT (num_offset, a fraction of the object span),
        # not from offset_mag: offset_mag is derived from the patch's own
        # t_bound, which near an edge collapses to a couple of mm, so
        # offset_mag*2.5 left the number sitting essentially ON the surface --
        # where mplot3d's per-collection depth sort buries it behind the
        # object mesh in every view whose camera is on the far side. Sizing
        # the push to the object instead keeps every stage number clear of
        # the mesh regardless of how small its patch got.
        # Fixed outward push (no per-stage radial fan): fanning along the
        # normal moved later stages progressively further from the object,
        # which in projection sent them off the mesh and into the panel title.
        push = float(num_offset) if num_offset is not None else offset_mag * 2.5
        # NO far-side cull: every stage number is drawn in every view. A
        # contact on the far side of a thin object is only centimetres behind
        # the surface, and seeing where it sits is the entire point of the
        # panel -- hiding it left views showing fewer than the full set of
        # stages, which reads as missing data rather than as occlusion.
        # Far-side numbers are drawn at reduced alpha instead, so they are
        # still identifiable as being behind the object.
        facing = (1.0 if cam_dir is None
                  else float(np.dot(n_l, np.asarray(cam_dir, float))))
        num_alpha = 1.0 if facing > 0.0 else 0.45
        # Stages whose seeds nearly coincide (036_wood_block seed 4's stages 3
        # and 4 differ by ~1.4mm) are separated VERTICALLY, not radially: the
        # circle is lifted along the object's own z by a per-stage step, so
        # consecutive stages stack in a readable column beside the contact
        # while staying next to the surface rather than drifting off it.
        z_fan = 0.0 if num_offset is None else float(num_offset) * 1.15 * (stage_num - 1)
        label_pt = frame["seed_l"] + n_l * push + np.array([0.0, 0.0, z_fan])
        # Leader line from the seed out to the number: consecutive Picard
        # stages can land a millimetre apart (036_wood_block seed 4's stages 3
        # and 4 differ by ~1.4mm in z), so their circles overlap almost exactly
        # and read as one marker. The leader ties each circle back to the point
        # it actually belongs to, and the per-stage fan below separates the
        # circles themselves.
        seed_pt = frame["seed_l"] + n_l * offset_mag
        ax.plot([seed_pt[0], label_pt[0]], [seed_pt[1], label_pt[1]],
               [seed_pt[2], label_pt[2]], "-", color=color, linewidth=0.8,
               alpha=num_alpha * 0.8, zorder=19)
        ax.text(*label_pt, str(stage_num), color="white", fontsize=9,
               fontweight="bold", zorder=20, ha="center", va="center", alpha=num_alpha,
               bbox=dict(boxstyle="circle,pad=0.25", facecolor=color, edgecolor="k",
                         linewidth=0.8, alpha=num_alpha))
    return P_draw


def _draw_stage_trajectory(ax, frame: dict, p_world_traj: np.ndarray,
                           obj_center: np.ndarray, obj_mat: np.ndarray,
                           offset_mag: float, color: str, cmap: str):
    """The solved contact's WORLD-frame p1/p2 trajectory for this stage,
    converted to object-local (matching the frame dict's own frame) and
    pushed off-surface for visibility -- drawn as a color-graded path,
    mirroring plot_uv_path.py's _draw_path_3d marker convention (circle=
    start, star=end, color gradient=iteration order)."""
    p_local = np.array([obj_mat.T @ (p - obj_center) for p in p_world_traj])
    draw = p_local + frame["n_l"] * offset_mag
    ax.plot(draw[:, 0], draw[:, 1], draw[:, 2], "-", color=color, linewidth=1.8, zorder=10)
    ax.scatter(draw[:-1, 0], draw[:-1, 1], draw[:-1, 2],
              c=np.arange(len(draw) - 1), cmap=cmap, s=40, zorder=11,
              edgecolors="k", linewidths=0.4, depthshade=False)
    ax.scatter(*draw[0], marker="o", s=140, facecolor="none", edgecolor=color,
              linewidths=2.2, zorder=12, depthshade=False)
    ax.scatter(*draw[-1], marker="*", s=240, color=color, edgecolors="k",
              linewidths=0.8, zorder=12, depthshade=False)
    return draw


def _draw_wrench_arrows(ax, verify_info: dict | None, stages: list[dict],
                        obj_center_world: np.ndarray, obj_mat: np.ndarray,
                        offset_mag: float):
    """Squeeze-force arrows at each contact, along the WRENCH-frame contact
    normal verify() actually certified feasibility against (n1_verify /
    n2_verify -- the frozen/last-linearized normal used to build the wrench
    cone, not necessarily identical to the last Picard stage's quadratic
    frame normal), scaled by gamma_min (the per-contact squeeze force the LP
    found feasible). Silently no-ops if verify() found no feasible gamma
    (nothing meaningful to draw)."""
    if not verify_info or verify_info.get("gamma_min") is None:
        return
    gamma = float(verify_info["gamma_min"])
    last = stages[-1]
    p1_w = np.asarray(last["p1"][-1], float)
    p2_w = np.asarray(last["p2"][-1], float)
    n1_w = np.asarray(verify_info.get("n1_verify"), float) if verify_info.get("n1_verify") else None
    n2_w = np.asarray(verify_info.get("n2_verify"), float) if verify_info.get("n2_verify") else None
    # Arrow length scaled to be visible against the off-surface offset scale
    # already in use for patches/trajectories on this axis, not to gamma's
    # raw N magnitude (a few to tens of N would be imperceptible or
    # page-filling depending on the object). Drawn from OUTSIDE the surface
    # in toward the contact (tail pushed off-surface by the patches'/
    # trajectories' own offset trick, head at the true surface point) --
    # starting the arrow AT the surface and pointing further in (the more
    # obvious "squeeze force" direction) put the entire arrow behind/inside
    # the patch's own Poly3DCollection from every camera angle, which
    # mplot3d's per-collection (not per-pixel) depth sort then reliably hid
    # regardless of zorder (confirmed empirically -- the arrow rendered fine
    # in isolation but vanished once patch collections were also on the
    # axis). Approaching from outside keeps the visible arrowhead in front
    # of the patch instead of underneath it.
    # NOTE: this arrow is drawn along the same contact normal the camera in
    # _normal_view/_iso_views is deliberately aligned WITH -- so in a view
    # that's looking almost straight down that normal, the arrow shaft
    # foreshortens to near-nothing by construction, not as a bug. The
    # diamond marker at the surface point is the part that stays legible
    # regardless of view angle; the shaft only reads clearly in the
    # "opposite side"/oblique views where the normal isn't camera-aligned.
    L = offset_mag * 8.0
    for p_w, n_w, color in ((p1_w, n1_w, "#08306b"), (p2_w, n2_w, "#67000d")):
        if n_w is None:
            continue
        p_l = obj_mat.T @ (p_w - obj_center_world)
        n_l = obj_mat.T @ n_w
        tail_l = p_l + n_l * L
        ax.quiver(tail_l[0], tail_l[1], tail_l[2],
                  p_l[0] - tail_l[0], p_l[1] - tail_l[1], p_l[2] - tail_l[2],
                  color=color, linewidth=3.0, arrow_length_ratio=0.3, zorder=1000)
        ax.scatter(*p_l, marker="D", s=90, color=color, edgecolors="white",
                  linewidths=1.0, zorder=1001, depthshade=False)
    ax.text2D(0.02, 0.02, f"γ_min={gamma:.2f} N", transform=ax.transAxes, fontsize=7,
             color="k", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))


def _render_hand_rgb(model, data, lookat, dist=0.45, w=650, h=520,
                     groups=(0, 1, 2, 5), azim=135, elev=-20):
    """MuJoCo offscreen render straight to an RGB array (not disk) so it can
    be embedded as an imshow panel in this figure -- duplicates
    ik_demo.render's camera/option setup since that helper only writes a
    standalone PNG file."""
    opt = mj.MjvOption()
    mj.mjv_defaultOption(opt)
    for g in range(6):
        opt.geomgroup[g] = 1 if g in groups else 0
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, azim, elev
    r = mj.Renderer(model, h, w)
    try:
        r.update_scene(data, camera=cam, scene_option=opt)
        return r.render()
    finally:
        del r


def plot_quadratic_path(V: np.ndarray, F: np.ndarray, stages: list[dict],
                        object_id: str, out_path: Path,
                        hand_rgb: np.ndarray | None = None,
                        verify_info: dict | None = None):
    """One figure: top row is 4 full-object context views (opposite-side iso
    pair per contact, camera aligned to each contact's own mean surface
    normal across all Picard stages), bottom row is a single wide panel
    showing the solved hand pose (MuJoCo render) with the found wrench
    (squeeze-force arrows + gamma_min/gws_beta/wrench_feasible text). Every
    stage overlaid in the top row, colored by FINGER (thumb=blue family,
    index=orange family) with the stage index stamped on each patch, so
    consecutive re-fits after a trust-region pin are distinguishable from
    each other while still visually grouped by which contact they belong
    to."""
    obj_center_world = stages[-1]["obj_center"]
    obj_mat = stages[-1]["obj_mat"]
    positions = V
    obj_center_local = positions.mean(0)
    obj_span = (positions.max(0) - positions.min(0)).max()
    off_floor = obj_span * 0.004
    off_cap = obj_span * 0.02
    last = stages[-1]

    # PER-CONTACT panels (one per contact), each viewed from that contact's
    # OWN side. Two earlier layouts were tried and both proved unreadable for
    # the question these panels exist to answer -- "where on the object does
    # this contact sit, and how did it move across Picard stages?":
    #
    #   * a grasp-axis-derived iso pair: the angle depended on where the
    #     contacts landed, so no two seeds were comparable;
    #   * four fixed object-frame isometrics: comparable across runs, but BOTH
    #     contacts' stage numbers were drawn in all four panels, so eight
    #     circles competed for the same screen space and thumb/index at the
    #     same stage overlapped each other.
    #
    # One panel per contact removes the overlap at its source: only that
    # contact's stages appear, so N stages means exactly N numbers. The camera
    # faces the contact along its own mean outward normal (averaged over
    # stages, since a contact that migrates keeps roughly the same face), with
    # a small downward tilt so the object still reads as a solid.
    ELEV_DEG = 18.0          # fixed downward tilt for every contact panel
    contact_ids = sorted({ci for st in stages for ci in st["contact"]})

    def _contact_view(ci: int):
        """(elev, azim) looking at contact ci along its own mean outward
        normal, tilted down slightly."""
        ns = [st["contact"][ci]["n_l"] for st in stages if ci in st["contact"]]
        if not ns:
            return (25.0, 45.0)
        n = np.mean(np.asarray(ns, float), axis=0)
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-9 else np.array([1.0, 0.0, 0.0])
        # Azimuth comes from the contact's own normal (so the panel faces the
        # contact); ELEVATION is fixed, not derived. Deriving it from the
        # normal put the camera BELOW the horizon whenever a contact normal
        # tilted downward (036_wood_block seed 4's index contact came out at
        # elev=-14, i.e. looking UP at the block from underneath -- the object
        # read as edge-on and every stage bunched against the top edge). A
        # fixed modest downward tilt keeps the object a readable solid and
        # keeps the bottom face hidden, in every panel and for every object.
        azim = np.degrees(np.arctan2(n[1], n[0]))
        return (ELEV_DEG, float(azim))

    def _draw_all_stages(ax, ci: int, cmap: str, label_stages: bool = True,
                         num_offset: float | None = None,
                         cam_dir: np.ndarray | None = None):
        """Draw every stage's patch/wireframe/trajectory for ONE contact
        index into ax. Returns the drawn (off-surface) PATCH points only
        (not the trajectory) across all stages, concatenated -- used by the
        zoomed panel to size its crop to the mm-scale surfaces, not to
        wherever the trajectory itself wandered (a contact CAN legitimately
        move tens of mm between Picard stages -- confirmed on 036_wood_block
        seed 100's index contact, ~90mm total drift across 3 stages -- and
        sizing the crop to include that would zoom back out past the point
        where the patches themselves are visible, the same problem this
        panel exists to fix, just triggered a different way). The trajectory
        is still DRAWN; it can run through/out of the patch-scale crop."""
        all_patch_draw = []
        for si, stage in enumerate(stages):
            frame = stage["contact"].get(ci)
            if frame is None:
                continue
            scolor = _stage_color(ci, si)
            p_traj = stage[f"p{ci}"]
            # Offset sized off the PATCH's own extent (t_bound), not the
            # trajectory's -- a wandering trajectory would otherwise inflate
            # the off-surface push far past what a mm-scale patch needs.
            patch_extent = max(float(frame["t_bound_0"]), float(frame["t_bound_1"]), 1e-4)
            offset_mag = float(np.clip(patch_extent * 0.3, off_floor, off_cap))
            finger_name = "thumb" if ci == 1 else "index"
            patch_draw = _draw_paraboloid_patch(
                ax, frame, offset_mag, scolor,
                label=(f"{finger_name} stage {si+1} ({stage['stage_label']})" if label_stages else None),
                stage_num=si + 1, num_offset=num_offset, cam_dir=cam_dir)
            _draw_stage_trajectory(ax, frame, p_traj, obj_center_world, obj_mat,
                                  offset_mag, scolor, cmap)
            all_patch_draw.append(patch_draw.reshape(-1, 3))
        return np.concatenate(all_patch_draw, axis=0) if all_patch_draw else np.zeros((0, 3))

    fig = plt.figure(figsize=(14, 12))

    # One panel per CONTACT, each from that contact's own facing view.
    panels = [(ci, _contact_view(ci), ("thumb" if ci == 1 else "index"))
              for ci in contact_ids]
    wrench_offset = float(np.clip(obj_span * 0.006, off_floor, off_cap))
    gs = fig.add_gridspec(2, max(len(panels), 2))

    # ── Row 1: one full-object view per contact ─────────────────────────
    for col, (ci, view, fname) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        _draw_mesh_3d(ax, V, F, alpha=None)
        # ONLY this contact's stages -- that is what keeps the numbering
        # readable (N stages -> exactly N circles in this panel).
        cmap = "viridis" if ci == 1 else "plasma"
        _draw_all_stages(ax, ci, cmap, label_stages=True,
                         num_offset=obj_span * 0.03, cam_dir=_view_cam_dir(view))
        _draw_wrench_arrows(ax, verify_info, stages, obj_center_world, obj_mat, wrench_offset)
        lo, hi = positions.min(0), positions.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(np.maximum(hi - lo, 1e-6))
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_title(f"{fname} contact  (elev {view[0]:.0f}, azim {view[1]:.0f})",
                     fontsize=9, pad=12)
        ax.tick_params(labelsize=5, pad=0)
        # Legend OUTSIDE the axes: the stage numbers drift toward the upper
        # part of the panel (contacts on this object sit high), and an
        # in-axes legend sat on top of them.
        ax.legend(loc="upper left", bbox_to_anchor=(-0.08, 1.02), fontsize=6,
                  framealpha=0.85)

    # ── Row 2: one panel -- solved hand pose + found wrench summary --
    # narrower than the full figure width (a full-width axes made the hand
    # render much larger/more prominent than the mesh panels above it).
    ax_hand = fig.add_subplot(gs[1, :])
    if hand_rgb is not None:
        ax_hand.imshow(hand_rgb)
    else:
        ax_hand.text(0.5, 0.5, "hand render unavailable", ha="center", va="center", fontsize=12)
    ax_hand.set_xticks([]); ax_hand.set_yticks([])
    for spine in ax_hand.spines.values():
        spine.set_visible(False)

    if verify_info:
        gm = verify_info.get("gamma_min")
        wf = verify_info.get("wrench_feasible")
        gws = verify_info.get("gws_beta")
        slack = verify_info.get("max_slack_norm")
        summary = (
            f"wrench_feasible={wf}   gamma_min="
            f"{'n/a' if gm is None else f'{gm:.3f} N'}   "
            f"gws_beta={'n/a' if gws is None else f'{gws:.4f}'}   "
            f"max_slack_norm={'n/a' if slack is None else f'{slack:.4f}'}")
        ax_hand.set_title(summary, fontsize=10, pad=10)
    else:
        ax_hand.set_title("verify() unavailable — no wrench summary", fontsize=10, pad=10)

    fig.suptitle(
        f"{object_id}: local-quadratic contact fit per Picard stage "
        f"({len(stages)} stage{'s' if len(stages) != 1 else ''}, "
        f"final status={last['status_tag']})\n"
        "patch=paraboloid surrogate clipped to its trust region, numbered circle=stage index, "
        "blue=thumb/orange=index, x=seed, circle=path start, star=path end",
        fontsize=9, y=0.985)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, hspace=0.25, wspace=0.05)
    OP.savefig(fig, out_path, dpi=150)
    plt.close(fig)


def _solve_and_plot(model, data, body_name, q_bias_full, pos, cfg, log_dir,
                    object_id, out_path, n_seeds=1, render_path=None):
    """Run one solve with log_dir set and, if any iteration traces were saved,
    render the multi-stage paraboloid figure (now including the solved hand
    pose + verify()'s wrench summary, embedded in the same PNG rather than a
    separate --render output). Mirrors plot_uv_path.py's _solve_and_plot
    (same err_mm handling)."""
    planner = MultiStartGraspPlanner3D(model, data, cfg, log_dir=log_dir)
    res = planner.solve(q_bias_full, np.asarray(pos, float), max_seeds=n_seeds)
    print(f"  status={res.get('status')} rs={res.get('return_status')} "
         f"iterations={res.get('iterations')} quad_pinned={res.get('quad_pinned')}")

    verify_info = None
    hand_rgb = None
    if res.get("q") is not None and res.get("p1") is not None:
        mesh_entry = planner._planner._mesh_entry
        err_mm = _tip_err_mm(model, data, planner._planner._obj_gid,
                             planner._planner._obj_bid, mesh_entry, cfg,
                             res["q"], np.asarray(res["p1"]), np.asarray(res["p2"]))
        print(f"  err_mm={err_mm:.2f}  gws_beta={res.get('gws_beta')}")
        try:
            verify_info = planner._planner.verify(res)
            print(f"  wrench_feasible={verify_info.get('wrench_feasible')} "
                 f"gamma_min={verify_info.get('gamma_min')}")
        except Exception as e:
            print(f"  verify() failed: {e}")

        data.qpos[:N_ROBOT] = res["q"]
        mj.mj_forward(model, data)
        try:
            hand_rgb = _render_hand_rgb(model, data, lookat=pos, dist=0.45)
        except Exception as e:
            print(f"  hand render failed: {e}")
        if render_path is not None:
            try:
                render(model, data, render_path, lookat=pos, dist=0.55)
                print(f"  render -> {render_path}")
            except Exception as e:
                print(f"  render failed: {e}")

    stages = _iter_trace_quadratic_stages(log_dir, res=res)
    if not stages:
        print("  No iteration trace was saved (log_dir wiring or solve failure) — nothing to plot.")
        return res, None
    if not any(stage["contact"] for stage in stages):
        print("  No quad_* frame data in any saved stage — was use_quadratic_contact set, "
             "and is this a mesh object? Nothing to plot.")
        return res, None

    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    V, F = oua.body_visual_mesh(model, bid)
    plot_quadratic_path(V, F, stages, object_id, out_path, hand_rgb=hand_rgb, verify_info=verify_info)
    print(f"  {len(stages)} Picard stage(s) traced  saved -> {out_path}")
    return res, stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", "--objects", dest="objects", nargs="+", default=["013_apple"],
                    help="one or more YCB object ids")
    ap.add_argument("--seed", "--seeds", dest="seeds", type=int, nargs="+", default=[100],
                    help="one or more object-pose RNG seeds (matches ablate_grasp.py's "
                         "100 + seed_index convention when sweeping)")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--gws", action="store_true",
                    help="wrench_constraint=False + w_gws/w_span on — matches "
                         "ablate_grasp.py --gws. Default (off): embedded 64-corner "
                         "wrench-cone LP (wrench_constraint=True).")
    ap.add_argument("--w-gws", type=float, default=5.0)
    ap.add_argument("--w-span", type=float, default=1.0)
    ap.add_argument("--soft-finger", action="store_true")
    ap.add_argument("--n-relin", type=int, default=None,
                    help="n_normal_relinearize override — how many extra Picard "
                         "stages to run (more stages = more consecutive paraboloid "
                         "re-fits to show). Default: GraspConfig3D's own default (1).")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="MultiStartGraspPlanner3D's own contact-seed budget per solve")
    ap.add_argument("--render", action="store_true",
                    help="also save a SEPARATE standalone PNG of the final arm+hand pose "
                         "(the hand pose is already embedded in the main figure by default)")
    ap.add_argument("--out-root", default=str(OP.analysis_dir("quadratic_path")),
                    help="output root; results are grouped per object as "
                         "<out-root>/<object_id>/seed<N>.png")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    log_dir = str(out_root / "_quad_path_tmp_logs")

    base = mj.MjModel.from_xml_path(str(REPO / "models" / "scene_kinova_leap.xml"))
    ws = W.load_or_build(base, n=200_000, seed=0)
    q_bias_full = home_bias()
    rgeoms = robot_geom_names(base)
    obj_clr = clearance_by_geom(rgeoms)

    cfg_kw = dict(n_seeds=args.n_seeds, max_iter=args.max_iter, arm_geom_names=rgeoms,
                 obj_clearance_by_geom=obj_clr, col_clearance_m=DEFAULT_COL_CLEARANCE_M,
                 use_quadratic_contact=True)
    if args.n_relin is not None:
        cfg_kw["n_normal_relinearize"] = args.n_relin
    if args.gws:
        cfg_kw["wrench_constraint"] = False
        cfg_kw["w_gws"] = args.w_gws
        cfg_kw["w_span"] = args.w_span
        cfg_kw["gws_soft_finger"] = args.soft_finger

    for seed_val in args.seeds:
        rng = np.random.default_rng(seed_val)
        for oid in args.objects:
            print(f"=== {oid}  seed={seed_val} ===")
            if os.path.isdir(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)

            try:
                (_, pos, quat), = place_objects([oid], ws, rng)
            except RuntimeError as e:
                print(f"  skip {oid} seed {seed_val}: {e}")
                continue

            model, data, info = S.build([(oid, pos, quat)])
            body_name = next(iter(info))
            data.qpos[:N_ROBOT] = q_bias_full
            mj.mj_forward(model, data)

            obj_geom0 = S.hull_geoms(model, body_name)[0]
            cfg = GraspConfig3D(obj_geom=obj_geom0, obj_body=body_name, **cfg_kw)

            obj_out_dir = out_root / oid
            obj_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = OP.fig_path(obj_out_dir / f"seed{seed_val}.png")
            render_path = (obj_out_dir / f"seed{seed_val}_arm_pose.png") if args.render else None
            try:
                _solve_and_plot(model, data, body_name, q_bias_full, pos, cfg,
                               log_dir, oid, out_path, n_seeds=args.n_seeds,
                               render_path=render_path)
            except Exception as e:
                print(f"  EXCEPTION: {e}")

            shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

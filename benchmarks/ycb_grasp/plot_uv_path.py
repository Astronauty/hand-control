"""Visualize a mesh contact's (u,v) iteration path on the UV atlas.

Runs one GraspPlanner3D solve per (object, seed) with use_uv_atlas_contact=True
and log_dir set (so grasp_planner_3d.py's _opti_cb records the per-iteration
uv1/uv2 trajectory — see GraspConfig3D.use_uv_atlas_contact / object_uv_atlas.py
module docstrings for what these decision variables mean), then maps the
recorded p1/p2 WORLD-frame trajectory back onto the atlas's own (u,v) via
object_uv_atlas.point_3d_to_atlas_uv and plots it two ways side by side: the
raw 3D movement over the object's mesh (object-local frame), and the same
path over the flat UV atlas layout. --object/--seed each accept multiple
values (one PNG per object x seed pair) and --seed uses the RNG value
directly as ablate_grasp.py's own `100 + seed_index` — pass the same
100+seed values to reproduce a specific ablate_grasp.py row's trace.

    python benchmarks/ycb_grasp/plot_uv_path.py                        # default: 013_apple, seed 100
    python benchmarks/ycb_grasp/plot_uv_path.py --object 077_rubiks_cube
    python benchmarks/ycb_grasp/plot_uv_path.py --object 025_mug --seed 103
    python benchmarks/ycb_grasp/plot_uv_path.py --gws --soft-finger     # GWS terms on, embedded wrench-cone LP off
    # reproduce ablate_grasp.py --objects 009_gelatin_box 036_wood_block 017_orange --seeds 3 --uv-atlas --gws:
    python benchmarks/ycb_grasp/plot_uv_path.py \\
        --object 009_gelatin_box 036_wood_block 017_orange --seed 100 101 102 --gws
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
import matplotlib.colors
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mujoco as mj
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from simulation.grasp_planner_3d import GraspConfig3D, MultiStartGraspPlanner3D  # noqa: E402
from ycb_grasp import scene as S, workspace as W                                # noqa: E402
from ycb_grasp.ik_demo import (clearance_by_geom, home_bias, place_objects,     # noqa: E402
                               render, robot_geom_names)
from ycb_grasp.ablate_grasp import _tip_err_mm                                  # noqa: E402
from grasp_control import object_uv_atlas as oua                                # noqa: E402

N_ROBOT = 23
DEFAULT_COL_CLEARANCE_M = 0.002


def _iter_trace_uv_paths(model, body_name, obj_pos, log_dir):
    """After a solve with log_dir set, load the saved grasp3d_iter_*.npz and
    project its p1/p2 world-frame trajectory onto the atlas's (u,v)."""
    npz_files = sorted(glob.glob(os.path.join(log_dir, "grasp3d_iter_*.npz")))
    if not npz_files:
        return None
    trace = np.load(npz_files[-1], allow_pickle=True)

    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    atlas, big, area_frac = oua.load_or_build(model, bid)
    obj_c = np.asarray(trace["obj_center"], float)
    obj_R = np.asarray(trace["obj_mat"], float)

    p1_traj, p2_traj = trace["p1"], trace["p2"]
    p1_l0 = obj_R.T @ (p1_traj[0] - obj_c)
    p2_l0 = obj_R.T @ (p2_traj[0] - obj_c)
    chart1 = oua.nearest_big_chart(atlas, big, p1_l0)
    chart2 = oua.nearest_big_chart(atlas, big, p2_l0)

    p1_local = np.array([obj_R.T @ (p - obj_c) for p in p1_traj])
    p2_local = np.array([obj_R.T @ (p - obj_c) for p in p2_traj])
    uv1_path = np.array([oua.point_3d_to_atlas_uv(atlas, chart1, p) for p in p1_local])
    uv2_path = np.array([oua.point_3d_to_atlas_uv(atlas, chart2, p) for p in p2_local])
    return dict(atlas=atlas, big=big, chart1=chart1, chart2=chart2,
               uv1_path=uv1_path, uv2_path=uv2_path,
               p1_local=p1_local, p2_local=p2_local,
               status=str(trace["status_tag"]), n_iter=len(trace["iter"]))


def _chart_colors(atlas, big):
    rng_c = np.random.default_rng(2)
    palette = rng_c.uniform(0.85, 0.95, size=(atlas["n_charts"], 3))
    is_big = np.zeros(atlas["n_charts"], dtype=bool)
    is_big[big] = True
    return np.where(is_big[:, None], palette, [0.97, 0.97, 0.97])[atlas["chart_id"]]


def _facing_view(center: np.ndarray, target: np.ndarray,
                 elev_offset: float = 0.0, azim_offset: float = 0.0,
                 away_from: np.ndarray | None = None) -> tuple[float, float]:
    """(elev, azim) so the matplotlib 3D camera looks roughly AT `target`
    from outside the object — i.e. the camera sits on the ray from `center`
    through `target`, extended outward, looking back toward the object.
    Fixes the fixed (20,35) default facing away from whichever side a
    contact happens to land on for a given object pose/seed.

    elev_offset/azim_offset : degrees added after the facing computation, in
        WORLD-frame elev/azim — only a good tie-breaker when the object's own
        geometry happens to be roughly axis-aligned with world z/xy (true for
        most of this session's test objects sitting "upright"-ish, but not
        guaranteed). A dead-on (perfectly perpendicular) camera against a FLAT
        surface is a degenerate case for mplot3d's per-collection depth sort
        (no true z-buffer): thin line/scatter artists sitting a few mm off a
        flat face can render UNDER the face's own triangles regardless of
        zorder, because the whole face collection and the path sit at
        near-identical projected depth. A small tilt breaks the tie without
        meaningfully changing what's "facing" the contact (reproduced on the
        rubiks cube's flat faces; curved surfaces like the apple don't hit
        this degenerate case, but the tilt is harmless there).

    away_from : if given (object-local, e.g. the OTHER contact's position for
        a 2-contact grasp), the camera is additionally nudged so `target` and
        `away_from` do NOT project to nearly the same screen point. A camera
        "facing p1" sits on the ray center->p1 by construction — for a good
        ANTIPODAL pair (p2 roughly opposite p1 through center, which is the
        whole point of the chart-pair seeding heuristic), that same camera is
        then ALSO looking almost straight through p1 toward p2 on the far
        side, collapsing both paths onto the same screen location (reproduced
        this session on 009_gelatin_box: p1/p2 8.9cm apart in 3D, camera
        direction anti-aligned with the p1->p2 axis at dot=-0.95, both paths
        rendered as one overlapping blob). A world-frame azim/elev_offset
        tilt does NOT reliably fix this (tried up to 25deg on both axes on a
        similar case — 036_wood_block's blank-panel symptom of the same root
        cause — and it didn't help, since the fix needs to rotate AROUND the
        actual p1-p2 axis, not around world z). This instead computes a
        perpendicular-to-the-grasp-axis direction and blends the camera
        position toward it by AWAY_BLEND, guaranteeing real angular
        separation between target and away_from regardless of the grasp
        axis's orientation in world/object frame."""
    d = np.asarray(target, float) - np.asarray(center, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return 20.0, 35.0
    d = d / n

    if away_from is not None:
        away_d = np.asarray(away_from, float) - np.asarray(center, float)
        away_n = np.linalg.norm(away_d)
        if away_n > 1e-9:
            away_d = away_d / away_n
            # Component of away_d perpendicular to d (Gram-Schmidt) — the
            # direction that most separates target from away_from on screen.
            perp = away_d - np.dot(away_d, d) * d
            perp_n = np.linalg.norm(perp)
            if perp_n > 1e-6:
                perp = perp / perp_n
                AWAY_BLEND = 0.6   # how far to rotate off pure target-facing
                d = d + AWAY_BLEND * perp
                d = d / np.linalg.norm(d)

    elev = np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    azim = np.degrees(np.arctan2(d[1], d[0]))
    # Cap how close to vertical the elevation is allowed to get, not just
    # clamp it into (-90,90). A target near a pole (e.g. a tall box's TOP
    # face centroid, d[2] close to +1) gives elev near +-90deg BEFORE any
    # offset -- a camera looking almost straight down/up renders that face as
    # a near-edge-on sliver, not a recognizable 3D shape, regardless of
    # whether the raw number technically stays inside +-90 (reproduced this
    # session on 036_wood_block's chart 11: even elev clamped to 89deg gave a
    # near-blank panel). ELEV_CAP pulls the camera down to a genuinely oblique
    # angle whenever the natural facing elevation exceeds it, same spirit as
    # elev_offset's flat-face tie-break but a much larger correction since
    # "almost straight down" is a worse degenerate case than "exactly
    # perpendicular". azim is left as computed by arctan2 (or 0 if d is
    # purely vertical, where azim is undefined) -- only elev is capped.
    ELEV_CAP = 55.0
    if abs(elev) > ELEV_CAP:
        elev = np.copysign(ELEV_CAP, elev)
        if abs(d[2]) > 0.999:   # d has ~no xy component -- arctan2(0,0) azim is meaningless
            azim = 0.0
    elev_final = np.clip(float(elev) + elev_offset, -89.0, 89.0)
    return elev_final, float(azim) + azim_offset


def _view_cam_dir(view: tuple[float, float]) -> np.ndarray:
    """Unit vector from object center TOWARD the camera for a (elev, azim)
    view_init pair — inverse of _facing_view's own elev/azim derivation."""
    elev, azim = np.radians(view[0]), np.radians(view[1])
    return np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])


def _uncovered_charts(view1, view2, chart_ids, chart_centroid, obj_center, margin_deg: float = 90.0):
    """Which of chart_ids are NOT within margin_deg of EITHER camera's
    direction — i.e. would render edge-on-or-worse (not just occluded, but
    genuinely outside a reasonable "facing" cone) in BOTH panels. Used to
    check the two-camera layout actually covers every usable chart, not just
    p1's/p2's own chart — see plot_uv_path's view-coverage comment."""
    d1, d2 = _view_cam_dir(view1), _view_cam_dir(view2)
    thresh = np.cos(np.radians(margin_deg))
    missed = []
    for ci in chart_ids:
        cdir = chart_centroid[ci] - obj_center
        n = np.linalg.norm(cdir)
        if n < 1e-9:
            continue
        cdir = cdir / n
        if np.dot(cdir, d1) < thresh and np.dot(cdir, d2) < thresh:
            missed.append(ci)
    return missed


def _mesh_outward_normal(atlas, chart_id, near_point):
    """Face normal of the chart triangle nearest `near_point`, oriented
    outward (away from the object's overall centroid) — used to push a path
    slightly off the surface for rendering (see _draw_path_3d)."""
    tv = atlas["positions"][atlas["indices"]]
    mask = atlas["chart_id"] == chart_id
    chart_tv = tv[mask]
    tri_c = chart_tv.mean(axis=1)
    nearest = int(np.argmin(np.sum((tri_c - near_point) ** 2, axis=1)))
    e1 = chart_tv[nearest, 1] - chart_tv[nearest, 0]
    e2 = chart_tv[nearest, 2] - chart_tv[nearest, 0]
    n = np.cross(e1, e2)
    n = n / (np.linalg.norm(n) + 1e-12)
    if np.dot(n, near_point - atlas["positions"].mean(0)) < 0:
        n = -n
    return n


def _draw_mesh_3d(ax, atlas, colors, alpha=None):
    """Draw the full object mesh once. alpha=None -> opaque (default: each
    panel's camera already faces its own titular contact directly, via
    _facing_view, so translucency isn't needed for visibility — and
    mplot3d's lack of a true z-buffer means a transparent surface makes the
    depth-sort MORE ambiguous, not less). alpha=float -> genuinely
    transparent facets, kept as an option for callers that do need to see
    through the near side of the mesh."""
    tv = atlas["positions"][atlas["indices"]]
    if alpha is None:
        ax.add_collection3d(Poly3DCollection(tv, facecolors=colors, edgecolors="none"))
    else:
        rgba = np.concatenate([colors, np.full((len(colors), 1), alpha)], axis=1)
        pc = Poly3DCollection(tv, facecolors=rgba, edgecolors="none")
        pc.set_sort_zpos(None)  # let mplot3d depth-sort normally; alpha handles occlusion
        ax.add_collection3d(pc)
    return tv


def _draw_path_3d(ax, atlas, chart_id, path_local, color, cmap, label=None,
                  offset_mag=0.005, marker_scale=1.0):
    """Draw one contact's path, pushed slightly OUTWARD along the local mesh
    surface normal — cosmetic only (does not touch path_local). mplot3d has
    no true z-buffer: it depth-sorts whole triangle patches by centroid, not
    per-pixel, so a path sitting within ~1mm of a mesh surface gets buried
    under nearby triangles regardless of zorder (confirmed empirically this
    session) — pushing it a small distance off-surface avoids the ambiguity.
    Direction is the NEAREST TRIANGLE's own face normal (not "away from the
    whole object's centroid", which breaks down at concave regions).

    Returns the drawn (offset) path, e.g. so a caller can center a zoom crop
    on it.
    """
    path_c = path_local.mean(0)
    n_out = _mesh_outward_normal(atlas, chart_id, path_c)
    draw = path_local + n_out * offset_mag

    ax.plot(draw[:, 0], draw[:, 1], draw[:, 2], "-", color=color,
           linewidth=2.2 * marker_scale, zorder=5)
    ax.scatter(draw[:-1, 0], draw[:-1, 1], draw[:-1, 2],
              c=np.arange(len(draw) - 1), cmap=cmap, s=55 * marker_scale, zorder=6,
              edgecolors="k", linewidths=0.5, depthshade=False)
    ax.scatter(*draw[0], marker="o", s=190 * marker_scale, facecolor="none", edgecolor=color,
              linewidths=2.8, zorder=7, depthshade=False)
    ax.scatter(*draw[-1], marker="*", s=300 * marker_scale, color=color, edgecolors="k",
              linewidths=0.9, zorder=7, depthshade=False, label=label)
    return draw


def plot_uv_path(paths: dict, object_id: str, out_path: Path, rank_table: list | None = None):
    atlas, big = paths["atlas"], paths["big"]
    chart1, chart2 = paths["chart1"], paths["chart2"]
    uv1_path, uv2_path = paths["uv1_path"], paths["uv2_path"]
    p1_local, p2_local = paths["p1_local"], paths["p2_local"]

    colors = _chart_colors(atlas, big)
    tris_uv = atlas["uvs"][atlas["indices"]]
    positions = atlas["positions"]
    obj_center = positions.mean(0)
    obj_span = (positions.max(0) - positions.min(0)).max()

    # Path-extent-scaled offset (clamped to a band of obj_span) to push the
    # drawn path clear of the mesh surface — see _draw_path_3d docstring. This
    # only needs to clear mplot3d's per-collection (not per-pixel) depth sort
    # against the path's OWN nearby triangles — a small margin, not something
    # that should read as "floating off the surface".
    #
    # KNOWN UNRESOLVED GAP (not fixed this session, documenting so it isn't
    # re-litigated from scratch): NO fixed offset_mag formula tried was
    # reliable across all objects/chart pairs —
    #   - obj_span*0.035 floor (original): paths visibly floated off the
    #     surface once panels became opaque (too big for apple/cube/orange).
    #   - obj_span*0.004 floor: fixed that, but left BOTH paths fully
    #     invisible on 009_gelatin_box's (196,53) chart-pair render.
    #   - obj_span*0.06-0.12: raised to try to fix the gelatin_box case — did
    #     NOT fix it (confirmed by reproducing the exact failing view/chart
    #     standalone with this larger offset; still fully invisible). The
    #     mesh's own outward normal at that chart faces the camera reasonably
    #     directly (dot=0.72, not edge-on) and the drawn point sits inside
    #     the axis limits, so this isn't simple insufficient-offset or
    #     out-of-frame clipping — it looks like a genuine mplot3d
    #     per-COLLECTION (not per-triangle) depth-sort failure specific to
    #     chart positions near a box EDGE/seam between two faces, distinct
    #     from the flat-face and near-pole cases already handled by
    #     _facing_view's ELEV_CAP/elev_offset. Root cause not found; reverted
    #     to the 0.004/0.02 band since it's still strictly better than
    #     nothing (worked on apple/cube/orange, gelatin_box's edge-chart case
    #     was ALSO broken at 0.004, so this isn't a regression from trying).
    #     Needs real investigation (e.g. per-render visibility verification,
    #     not another guessed formula) before spending more time on it.
    off_floor = obj_span * 0.004
    off_cap   = obj_span * 0.02
    offset1 = np.clip((p1_local.max(0) - p1_local.min(0)).max() * 1.5, off_floor, off_cap)
    offset2 = np.clip((p2_local.max(0) - p2_local.min(0)).max() * 1.5, off_floor, off_cap)

    # A single camera cannot show both contacts face-on when they're close to
    # antipodal (measured dot product of their directions from center as low
    # as -0.996 on a real pinch grasp, i.e. ~180deg apart — genuinely no
    # angle sees both sides at once). So: two full-object panels, each camera
    # facing ONE contact — both paths are drawn in both panels for context,
    # but each panel GUARANTEES its own titular contact is squarely visible.
    # elev/azim tilt breaks mplot3d's dead-on-flat-face depth-sort tie (see
    # _facing_view docstring) — a camera looking straight at a contact can
    # otherwise bury that contact's own path under its near-facing triangles.
    # away_from=the OTHER contact fixes a related, more consequential bug: for
    # a good antipodal pair, "facing p1" is ALSO nearly "looking straight
    # through p1 at p2" (reproduced on 009_gelatin_box: dot=-0.95 between
    # camera dir and the p1->p2 axis), collapsing both paths onto the same
    # screen blob even though they're 8.9cm apart in 3D — see _facing_view's
    # away_from docstring for why a plain world-frame azim/elev_offset tilt
    # (tried up to 25deg on both, on 036_wood_block's blank-panel symptom of
    # this SAME root cause) doesn't reliably fix it, only this axis-relative
    # correction does.
    view1_full = _facing_view(obj_center, p1_local[-1], elev_offset=14.0, azim_offset=9.0,
                              away_from=p2_local[-1])
    view2_full = _facing_view(obj_center, p2_local[-1], elev_offset=14.0, azim_offset=9.0,
                              away_from=p1_local[-1])

    # Coverage check: p1/p2 are usually a good antipodal pair (that's the whole
    # point of the chart-pair seeding heuristic), so facing each of them
    # individually USUALLY also covers every other big chart for free — but
    # that's incidental, not guaranteed (a near-perpendicular or 3+-chart
    # object can leave some big chart edge-on/unseen in BOTH panels). If so,
    # search a small azimuth sweep AROUND view2 (view1 stays anchored on p1 —
    # still guarantees that panel's titular contact is visible) and keep
    # whichever candidate covers the most charts while still keeping p2 in
    # its own reasonably-facing cone (<=70deg) — retargeting to the missed
    # charts' raw mean direction was tried first and made things WORSE when
    # the misses were scattered on opposite sides (mean direction lands
    # somewhere covering neither), so this instead picks among a bounded set
    # of candidates near p2 rather than jumping to an unconstrained target.
    _chart_normal_all, _chart_centroid_all = oua.chart_normals_centroids(atlas)
    _missed = _uncovered_charts(view1_full, view2_full, big, _chart_centroid_all, obj_center)
    if _missed:
        _best_view2, _best_missed = view2_full, _missed
        for _az_off in (-45.0, -30.0, -15.0, 15.0, 30.0, 45.0):
            _cand = _facing_view(obj_center, p2_local[-1],
                                 elev_offset=14.0, azim_offset=9.0 + _az_off)
            _p2_dir = (p2_local[-1] - obj_center)
            _p2_dir = _p2_dir / (np.linalg.norm(_p2_dir) + 1e-12)
            _cand_ang = np.degrees(np.arccos(np.clip(np.dot(_p2_dir, _view_cam_dir(_cand)), -1, 1)))
            if _cand_ang > 70.0:
                continue   # too far off p2 itself — that panel's own titular contact must stay visible
            _cand_missed = _uncovered_charts(view1_full, _cand, big, _chart_centroid_all, obj_center)
            if len(_cand_missed) < len(_best_missed):
                _best_view2, _best_missed = _cand, _cand_missed
        view2_full = _best_view2
        print(f"  [view coverage] charts {_missed} not covered by the p1/p2-facing views"
             + (f" — best azimuth sweep still misses {_best_missed}" if _best_missed
                else " — resolved by azimuth sweep"))

    fig = plt.figure(figsize=(16.5, 5.5))

    for col, (view, label) in enumerate([(view1_full, "thumb (p1)"), (view2_full, "index (p2)")]):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        # Opaque: the facing-camera + offset already guarantee each panel's
        # titular path is visible, so translucency (which mplot3d's lack of
        # a true z-buffer makes ambiguous, not clearer) is no longer needed.
        _draw_mesh_3d(ax, atlas, colors, alpha=None)
        _draw_path_3d(ax, atlas, chart1, p1_local, "#2166AC", "viridis",
                     f"thumb (p1), {len(p1_local)} iters", offset_mag=offset1,
                     marker_scale=1.1)
        _draw_path_3d(ax, atlas, chart2, p2_local, "#B2182B", "plasma",
                     f"index (p2), {len(p2_local)} iters", offset_mag=offset2,
                     marker_scale=1.1)
        lo, hi = positions.min(0), positions.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(np.maximum(hi - lo, 1e-6))
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_title(f"Full object, camera facing {label}", fontsize=10, pad=18)
        ax.tick_params(labelsize=6, pad=0)
        ax.legend(loc="upper left", fontsize=6)

    # ── UV panel (unaffected by 3D camera angle — the flat atlas has no
    # "facing away" problem) ─────────────────────────────────────────────
    axuv = fig.add_subplot(1, 3, 3)
    axuv.add_collection(PolyCollection(tris_uv, facecolors=colors, edgecolors="none"))
    for ci, hl_color in [(chart1, "#4C72B0"), (chart2, "#C44E52")]:
        mask = atlas["chart_id"] == ci
        axuv.add_collection(PolyCollection(tris_uv[mask], facecolors=hl_color, alpha=0.25, edgecolors="none"))
    for uv_path, label, color, cmap in [
        (uv1_path, "thumb (p1)", "#2166AC", "viridis"),
        (uv2_path, "index (p2)", "#B2182B", "plasma"),
    ]:
        axuv.plot(uv_path[:, 0], uv_path[:, 1], "-", color=color, linewidth=1.5, zorder=5,
                 label=f"{label} path ({len(uv_path)} iters)")
        axuv.scatter(uv_path[:-1, 0], uv_path[:-1, 1], c=np.arange(len(uv_path) - 1),
                    cmap=cmap, s=35, zorder=6, edgecolors="k", linewidths=0.4)
        axuv.scatter(*uv_path[0], marker="o", s=140, facecolor="none", edgecolor=color,
                    linewidths=2.5, zorder=7)
        axuv.scatter(*uv_path[-1], marker="*", s=220, color=color, edgecolors="k",
                    linewidths=0.8, zorder=7)
    axuv.set_xlim(0, 1); axuv.set_ylim(0, 1); axuv.set_aspect("equal")
    axuv.set_title("Flat UV atlas", fontsize=10, pad=2)
    axuv.tick_params(labelsize=7)
    axuv.legend(loc="upper right", fontsize=6)

    # ── Chart-pair ranking annotation ────────────────────────────────────
    # rank_table entries: chart_pair, antipodal_score (-||n_i+n_j||, higher/
    # closer-to-0 = better-opposed normals), dls_res_mm (cheap DLS-IK
    # feasibility residual for that pair — see _chart_pair_seeds /
    # MultiStartGraspPlanner3D.solve's chart-aware seeding), accepted (was it
    # actually tried as an NLP seed, i.e. within the n_seeds budget by
    # DLS-residual rank). Printed as plain text since the table is small
    # (usually <=10 rows) and a text block is easier to read exactly than
    # trying to cram it into the plot geometry.
    if rank_table:
        selected = {chart1, chart2}
        lines = ["chart-pair ranking (by DLS residual; * = this figure's pair)"]
        lines.append(f"{'rank':>4s} {'pair':>10s} {'antipodal':>10s} {'dls(mm)':>9s} {'seeded':>7s}")
        for i, row in enumerate(rank_table):
            cp = row.get("chart_pair")
            is_sel = cp is not None and selected == set(cp)
            mark = "*" if is_sel else " "
            lines.append(
                f"{mark}{i+1:>3d} {str(cp):>10s} {row.get('antipodal_score', float('nan')):10.4f} "
                f"{row.get('dls_res_mm', float('nan')):9.1f} {'Y' if row.get('accepted') else 'n':>7s}")
        axuv.text(1.03, 1.0, "\n".join(lines), transform=axuv.transAxes,
                 fontsize=6.5, family="monospace", va="top", ha="left")

    fig.suptitle(f"{object_id}: contact iteration path (status={paths['status']}) — "
                "circle=start, star=end, color gradient=iteration order", fontsize=11, y=0.98)
    # subplots_adjust (not tight_layout) — mplot3d axes don't report accurate
    # bounding boxes to tight_layout, which left a large blank band under
    # the suptitle; explicit margins avoid that.
    fig.subplots_adjust(left=0.02, right=0.80, top=0.92, bottom=0.05, wspace=0.15)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _solve_and_plot(model, data, body_name, q_bias_full, pos, cfg, log_dir,
                    object_id, out_path, n_seeds=1, render_path=None):
    """Run one solve with log_dir set and, if an iteration trace was saved,
    render its UV/3D path figure. Also renders the final arm+hand pose (if
    render_path given) and prints the TRUE fingertip-to-target residual
    (ablate_grasp.py's _tip_err_mm — independent of solver status, the metric
    that actually matters; see that module's docstring). Returns
    (res, paths_or_None)."""
    planner = MultiStartGraspPlanner3D(model, data, cfg, log_dir=log_dir)
    res = planner.solve(q_bias_full, np.asarray(pos, float), max_seeds=n_seeds)
    print(f"  status={res.get('status')} rs={res.get('return_status')} "
         f"iterations={res.get('iterations')}")

    if res.get("q") is not None and res.get("p1") is not None:
        mesh_entry = planner._planner._mesh_entry
        err_mm = _tip_err_mm(model, data, planner._planner._obj_gid,
                             planner._planner._obj_bid, mesh_entry, cfg,
                             res["q"], np.asarray(res["p1"]), np.asarray(res["p2"]))
        print(f"  err_mm={err_mm:.2f}  gws_beta={res.get('gws_beta')}")
        if render_path is not None:
            data.qpos[:N_ROBOT] = res["q"]
            mj.mj_forward(model, data)
            try:
                render(model, data, render_path, lookat=pos, dist=0.55)
                print(f"  render -> {render_path}")
            except Exception as e:
                print(f"  render failed: {e}")

    paths = _iter_trace_uv_paths(model, body_name, pos, log_dir)
    if paths is None:
        print("  No iteration trace was saved (log_dir wiring or solve failure) — nothing to plot.")
        return res, None

    plot_uv_path(paths, object_id, out_path, rank_table=planner.last_chart_rank_table)
    print(f"  thumb chart={paths['chart1']} ({paths['n_iter']} iters)  "
         f"index chart={paths['chart2']}")
    print(f"  saved -> {out_path}")
    return res, paths


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
                         "wrench-cone LP (wrench_constraint=True), matching "
                         "ablate_grasp.py's own default.")
    ap.add_argument("--w-gws", type=float, default=5.0)
    ap.add_argument("--w-span", type=float, default=1.0)
    ap.add_argument("--soft-finger", action="store_true")
    ap.add_argument("--rings", type=int, default=2, help="uv_atlas_rings — local-neighborhood size")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="MultiStartGraspPlanner3D's own contact-seed budget per solve "
                         "(matches ablate_grasp.py's --n-seeds — the fixed minor-axis seed "
                         "plus up to this many chart-pair/random seeds are tried per solve)")
    ap.add_argument("--render", action="store_true",
                    help="also save a PNG of the final arm+hand pose alongside the UV path figure")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "ycb_grasp" / "out" / "uv_atlas_prototype"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = str(out_dir / "_uv_path_tmp_logs")

    base = mj.MjModel.from_xml_path(str(REPO / "models" / "scene_kinova_leap.xml"))
    ws = W.load_or_build(base, n=200_000, seed=0)
    q_bias_full = home_bias()
    rgeoms = robot_geom_names(base)
    obj_clr = clearance_by_geom(rgeoms)

    cfg_kw = dict(n_seeds=args.n_seeds, max_iter=args.max_iter, arm_geom_names=rgeoms,
                 obj_clearance_by_geom=obj_clr, col_clearance_m=DEFAULT_COL_CLEARANCE_M,
                 use_uv_atlas_contact=True, uv_atlas_rings=args.rings)
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

            out_path = out_dir / f"uv_iteration_path_{oid}_seed{seed_val}.png"
            render_path = (out_dir / f"arm_pose_{oid}_seed{seed_val}.png") if args.render else None
            try:
                _solve_and_plot(model, data, body_name, q_bias_full, pos, cfg,
                               log_dir, oid, out_path, n_seeds=args.n_seeds,
                               render_path=render_path)
            except Exception as e:
                print(f"  EXCEPTION: {e}")

            shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

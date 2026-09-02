"""UV-atlas contact parameterization for mesh (YCB) objects.

Why this exists
----------------
grasp_planner_3d.py's _mesh_tangent_contact_ca parameterizes a mesh contact as
a 2-DOF offset in a FIXED Euclidean tangent plane, reprojected onto the SDF
surface each NLP iteration. That plane is built once from a single seed
point's normal with an ARBITRARY Euclidean radius bound (t_bound), unrelated
to the object's real geometry -- it can extend past a true face/chart
boundary in some directions and fall short in others (see session notes:
"seed generation may cause certain locations on the face to become
inaccessible").

This module builds a precomputed UV ATLAS of the object's surface and uses it
to derive a BETTER-FITTED local linearization per NLP solve: instead of an
arbitrary Euclidean box, the tangent-plane-equivalent bound comes from the
actual local neighborhood of triangles around the current contact point (a
fixed-size ring of face-adjacency hops within one CHART -- a large, roughly-
flat, contiguous region the atlas already identified). This local
neighborhood is small and cheap (same CasADi cost as today's tangent-plane
scheme: one best-fit plane, one 2-DOF bounded offset, one reprojection) --
NOT the whole chart's ~1000-5000 triangles, which would need an expensive
piecewise/soft-selected map to stay accurate everywhere and was measured this
session to likely reproduce the earlier 64-corner-wrench-LP iteration-budget
problem.

Reaching the WHOLE chart (not just one local neighborhood) happens across
GraspPlanner3D's existing Picard relinearization loop (see
GraspConfig3D.n_normal_relinearize / the _run_stage loop in solve()): each
stage already re-centers its working contact point and rebuilds the local
frame from there (originally just the frozen surface NORMAL for curved
analytic shapes; extended here to also rebuild the local neighborhood/plane
for mesh objects). A multi-stage solve can walk across an entire large chart
via a sequence of small, cheap local linearizations, the same way the
existing loop already lets a contact's frozen normal "catch up" as the
contact moves on a sphere/cylinder.

Mesh source: the VISUAL mesh (textured.obj / MJCF group=2), NOT the collision
hull (group=3, CoACD-decomposed). Measured this session: the collision hull's
own atlas is badly fragmented (per-triangle normal noise up to 60deg on
sliver triangles from CoACD's triangulation -- see git history / session
notes), giving large charts with 15-40% internal GAPS (regions with no
triangle at all). The visual mesh (denser, more uniform triangulation from
the original 3D scan) unwraps far more cleanly (measured fill ~0.93-0.96 on
5/6 of the apple's large charts, vs ~0.6-0.85 on the collision hull). Since
the collision hull is what physics/the NLP's surface constraint actually use,
every (u,v)-derived point is reprojected onto the collision hull's SDF
surface before use (object_sdf.surface_project) -- this is a small correction
almost everywhere (measured mean 0.57mm, median 0.0mm deviation between the
two meshes on the apple) except at a couple of small, non-convex-ish, high-
curvature dimples (the apple's stem/blossom-end poles measured up to 9mm
deviation) that the chart-area filter below already excludes as too small to
be useful contact regions anyway.

xatlas is the atlas generator (pip package, C++ chart-growing + packing).
Default ChartOptions were used for the visual mesh (session testing found
this mesh unwraps cleanly without needing xatlas's roundness_weight/max_cost
tuning that the collision hull required and still didn't fully fix).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull

try:
    import xatlas
    _XATLAS_AVAILABLE = True
except ImportError:
    xatlas = None
    _XATLAS_AVAILABLE = False

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "assets" / "ycb_uv_atlas"

# A chart must hold at least this fraction of the object's total UV-triangle
# area to be considered a usable contact-seeding target. Everything else
# (fragmented scan-noise slivers, small high-curvature pinch regions -- see
# module docstring) is discarded. 1% chosen from this session's measurements:
# the apple's/cube's large charts all clear this by a wide margin (5-18% each
# on both objects), while the fragment tail sits at a median of ~0.03% -- a
# wide, unambiguous gap between "real region" and "noise", not a fine-tuned
# threshold.
DEFAULT_MIN_CHART_AREA_FRAC = 0.01


def _cache_key(verts: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.blake2b(np.ascontiguousarray(verts, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def body_visual_mesh(model, body_id: int, group: int = 2):
    """Visual-geom triangle mesh for one body, in BODY frame (same geom_pos/
    geom_quat folding convention as object_sdf.body_hull_halfspaces, so a UV
    atlas built from this and the SDF table built from the collision hulls
    agree on what "body frame" means).

    Returns (vertices (V,3), faces (F,3) int) for the FIRST group==group mesh
    geom found on the body. Raises if none exists (every YCB MJCF this repo
    generates has exactly one visual mesh geom per object body).
    """
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id or model.geom_group[g] != group:
            continue
        if model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        vid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[vid], model.mesh_vertnum[vid]
        fa, fn = model.mesh_faceadr[vid], model.mesh_facenum[vid]
        R = np.zeros(9)
        mj.mju_quat2Mat(R, model.geom_quat[g])
        V = model.mesh_vert[va:va + vn] @ R.reshape(3, 3).T + model.geom_pos[g]
        F = model.mesh_face[fa:fa + fn].copy()
        return V, F
    raise ValueError(f"body_visual_mesh: no group={group} mesh geom on body {body_id}")


def build_atlas(vertices: np.ndarray, faces: np.ndarray, chart_kw: dict | None = None):
    """Run xatlas on one mesh. Returns a dict:
        positions   (V',3) float64 -- atlas-reindexed vertex positions (xatlas
                     may duplicate vertices along chart seams)
        indices     (F,3) int32    -- triangle indices into `positions`
        uvs         (V',2) float64 -- per-atlas-vertex UV coordinates
        chart_id    (F,)  int32    -- which chart each triangle belongs to
        n_charts    int
    """
    if not _XATLAS_AVAILABLE:
        raise RuntimeError("build_atlas: xatlas package not installed")
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
    co = xatlas.ChartOptions()
    for k, v in (chart_kw or {}).items():
        setattr(co, k, v)
    atlas.generate(co, xatlas.PackOptions(), verbose=False)
    n_charts = atlas.get_mesh_chart_count(0)
    vmapping, indices, uvs = atlas.get_mesh(0)
    positions = vertices[vmapping].astype(np.float64)
    chart_id = np.full(len(indices), -1, dtype=np.int32)
    for ci in range(n_charts):
        chart = atlas.get_mesh_chart(0, ci)
        chart_id[np.asarray(chart.faces)] = ci
    return dict(positions=positions, indices=indices.astype(np.int64),
               uvs=uvs.astype(np.float64), chart_id=chart_id, n_charts=n_charts)


def _tri_areas_3d(positions, indices):
    tv = positions[indices]
    return 0.5 * np.linalg.norm(np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]), axis=1)


def select_big_charts(atlas: dict, min_area_frac: float = DEFAULT_MIN_CHART_AREA_FRAC):
    """Which chart ids hold at least min_area_frac of the TOTAL mesh surface
    area (3D area, not UV area -- UV area is packing-dependent/arbitrary
    scale, 3D area is the physically meaningful quantity for "how much usable
    contact surface does this chart cover").

    Returns (big_chart_ids sorted by descending area, area_frac array over
    ALL chart ids).
    """
    tri_area = _tri_areas_3d(atlas["positions"], atlas["indices"])
    n_charts = atlas["n_charts"]
    chart_id = atlas["chart_id"]
    areas = np.zeros(n_charts)
    for ci in range(n_charts):
        areas[ci] = tri_area[chart_id == ci].sum()
    total = areas.sum()
    area_frac = areas / total if total > 0 else areas
    big = np.where(area_frac >= min_area_frac)[0]
    big = big[np.argsort(-area_frac[big])]
    return big, area_frac


def chart_normals_centroids(atlas: dict):
    """Per-chart AREA-WEIGHTED mean face normal and 3D centroid, object-local
    frame (same frame as atlas["positions"] -- body_visual_mesh's folded
    geom_pos/geom_quat convention). Indexed over ALL chart ids (0..n_charts-1),
    including small fragment charts -- callers filter with big_chart_ids /
    select_big_charts separately, same pattern as area_frac.

    Used by the chart-pair/triple antipodal seeding heuristic (score charts by
    how their normals oppose each other) and by the bottom-facing filter
    (dot chart_normal against the current placement's up direction) -- see
    GraspPlanner3D seeding notes. NOT unit-length guaranteed if a chart has
    zero triangle area (returns the zero vector; callers should treat that as
    "no usable normal", which only happens for degenerate/empty charts that
    select_big_charts would exclude anyway).

    Returns (chart_normal (n_charts,3), chart_centroid (n_charts,3)).
    """
    positions, indices, chart_id = atlas["positions"], atlas["indices"], atlas["chart_id"]
    tv = positions[indices]                                   # (F,3,3)
    tri_centroid = tv.mean(axis=1)                            # (F,3)
    cross = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    # cross/2 is already the area-weighted face normal (unnormalized) --
    # summing it directly area-weights the per-chart mean without a separate
    # multiply, and its own norm recovers total weight for the average.
    tri_normal_area_wt = 0.5 * cross                          # (F,3)

    n_charts = atlas["n_charts"]
    chart_normal = np.zeros((n_charts, 3))
    chart_centroid = np.zeros((n_charts, 3))
    for ci in range(n_charts):
        mask = chart_id == ci
        w = tri_area[mask]
        wsum = w.sum()
        if wsum < 1e-18:
            continue
        n = tri_normal_area_wt[mask].sum(axis=0)
        nn = np.linalg.norm(n)
        chart_normal[ci] = n / nn if nn > 1e-18 else 0.0
        chart_centroid[ci] = (tri_centroid[mask] * w[:, None]).sum(axis=0) / wsum
    return chart_normal, chart_centroid


def filter_bottom_facing_charts(chart_ids: np.ndarray, chart_normal: np.ndarray,
                                up_local: np.ndarray, max_down_deg: float = 60.0):
    """Drop charts from chart_ids whose area-weighted normal faces too far
    DOWNWARD relative to up_local (object-local "world up" direction at the
    object's CURRENT placement -- gravity-based, not a fixed object-frame
    axis, since the same object can be placed in different orientations; see
    GraspPlanner3D seeding notes).

    A chart resting against/near the table is normally unreachable or an
    unstable contact anyway (matches _reachable_contact's existing spirit for
    raw seed points, just applied at chart-selection time instead of
    per-point) -- excluding it up front keeps the antipodal chart-scoring
    heuristic from ever proposing it.

    max_down_deg : a chart is excluded when the angle between its normal and
        DOWN (-up_local) is <= max_down_deg, i.e. chart_normal . up_local <=
        -cos(max_down_deg). Default 60 deg -- generous (only excludes charts
        genuinely facing mostly downward, not merely tilted), consistent with
        select_big_charts' wide-gap-not-fine-tuned philosophy; tighten if the
        heuristic still proposes awkward near-bottom charts in practice.

    Returns the filtered subset of chart_ids (order preserved), and does NOT
    renormalize/require chart_normal to be unit-length (zero-normal/degenerate
    charts naturally fail the dot-product test and get excluded -- callers
    should already be passing select_big_charts' output here, which excludes
    those on area grounds too).
    """
    up = np.asarray(up_local, float)
    up = up / (np.linalg.norm(up) + 1e-12)
    thresh = -np.cos(np.deg2rad(max_down_deg))
    keep = [ci for ci in chart_ids if np.dot(chart_normal[ci], up) > thresh]
    return np.asarray(keep, dtype=chart_ids.dtype)


def chart_triangles(atlas: dict, chart_id: int):
    """(positions_subset, indices_subset, uvs) for just one chart -- what a
    per-chart CasADi query / seed sampler needs, without touching the rest of
    the atlas."""
    mask = atlas["chart_id"] == chart_id
    fidx = np.where(mask)[0]
    return atlas["positions"], atlas["indices"][fidx], atlas["uvs"]


def load_or_build(model, body_id: int, cache_dir=CACHE_DIR, chart_kw: dict | None = None,
                  min_area_frac: float = DEFAULT_MIN_CHART_AREA_FRAC):
    """Bake once per object *shape* (content-hash-keyed on the visual mesh's
    own vertices/faces -- same convention as object_sdf.load_or_bake, so a
    remesh or asset swap invalidates the cache instead of silently serving a
    stale atlas).

    Returns (atlas, big_chart_ids, area_frac) -- see build_atlas /
    select_big_charts docstrings.
    """
    V, F = body_visual_mesh(model, body_id)
    key = _cache_key(V, F)
    path = Path(cache_dir) / f"{key}.npz"
    if path.exists():
        z = np.load(path)
        atlas = dict(positions=z["positions"], indices=z["indices"],
                    uvs=z["uvs"], chart_id=z["chart_id"], n_charts=int(z["n_charts"]))
    else:
        atlas = build_atlas(V, F, chart_kw=chart_kw)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **atlas)
    big, area_frac = select_big_charts(atlas, min_area_frac=min_area_frac)
    return atlas, big, area_frac


# -----------------------------------------------------------------------------
# Local neighborhood (fixed-size ring) -- the actual NLP-facing parameterization
# -----------------------------------------------------------------------------

def _chart_face_adjacency(atlas: dict, chart_id: int):
    """Face-adjacency list (by shared UV edge) restricted to ONE chart's
    triangles. Returns (local_face_indices (global index into atlas["indices"]),
    adj_list (list of lists, LOCAL indices into local_face_indices)).

    Built from shared UV-vertex pairs (two triangles are adjacent if they
    share an edge in the atlas's own vertex indexing) -- NOT 3D adjacency,
    since xatlas may have duplicated vertices along a chart's OWN internal
    seams (rare, but possible for very elongated charts); UV-vertex sharing is
    the correct adjacency notion for "can I walk from one triangle to this one
    while staying in the SAME uv patch."
    """
    mask = atlas["chart_id"] == chart_id
    local_faces = np.where(mask)[0]
    tris = atlas["indices"][local_faces]  # (n,3) global vertex ids

    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    for li, tri in enumerate(tris):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (a, b) if a < b else (b, a)
            edge_to_tris.setdefault(e, []).append(li)

    adj = [[] for _ in range(len(local_faces))]
    for tri_list in edge_to_tris.values():
        if len(tri_list) == 2:
            i, j = tri_list
            adj[i].append(j)
            adj[j].append(i)
    return local_faces, adj


def nearest_chart_triangle(atlas: dict, chart_id: int, point_3d: np.ndarray):
    """Which of chart_id's triangles is closest to point_3d (by centroid
    distance -- a cheap proxy, adequate for choosing a SEED triangle to build
    a local neighborhood around; not an exact closest-point-on-triangle
    query). Returns a LOCAL index into that chart's own triangle list (0..n-1
    within the chart), matching _chart_face_adjacency's indexing.
    """
    mask = atlas["chart_id"] == chart_id
    local_faces = np.where(mask)[0]
    tris = atlas["positions"][atlas["indices"][local_faces]]  # (n,3,3)
    centroids = tris.mean(axis=1)
    d2 = np.sum((centroids - np.asarray(point_3d, float)) ** 2, axis=1)
    return int(np.argmin(d2))


def local_neighborhood(atlas: dict, chart_id: int, point_3d: np.ndarray, rings: int = 2):
    """Fixed-size local patch around point_3d, within ONE chart: BFS out
    `rings` face-adjacency hops from the nearest triangle. Returns dict:
        tri_positions   (k,3,3) -- 3D vertex positions of the k triangles in
                         the neighborhood (for reference/plotting; the NLP
                         itself only needs the plane fit + bound below)
        plane_centroid  (3,)  -- mean of all neighborhood vertices
        plane_basis     (2,3) -- (t1, t2), the two in-plane SVD directions
        bound_A, bound_b      -- linear inequality A @ [u,v] <= b describing
                         the neighborhood's boundary polygon, PROJECTED into
                         (t1,t2) plane coordinates centered at plane_centroid
                         (half-space form of the 2D convex hull of the
                         neighborhood's vertices in-plane)
        seed_uv         (2,)  -- point_3d's own (u,v) in this local frame
                         (== (0,0) approximately, since point_3d IS the seed
                         the neighborhood was built around -- kept explicit
                         for the caller's opti.set_initial)
    """
    local_faces, adj = _chart_face_adjacency(atlas, chart_id)
    seed_local = nearest_chart_triangle(atlas, chart_id, point_3d)

    visited = {seed_local}
    frontier = [seed_local]
    for _ in range(rings):
        nxt = []
        for f in frontier:
            for nb in adj[f]:
                if nb not in visited:
                    visited.add(nb)
                    nxt.append(nb)
        frontier = nxt
        if not frontier:
            break

    tri_global = local_faces[sorted(visited)]
    tri_positions = atlas["positions"][atlas["indices"][tri_global]]  # (k,3,3)
    verts = tri_positions.reshape(-1, 3)

    centroid = verts.mean(0)
    pts = verts - centroid
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    t1, t2 = Vt[0], Vt[1]

    proj = np.stack([pts @ t1, pts @ t2], axis=1)  # (3k, 2)
    hull = ConvexHull(proj)
    # half-space form: hull.equations rows are [a, b, c] meaning a*x+b*y+c<=0
    bound_A = hull.equations[:, :2]
    bound_b = -hull.equations[:, 2]

    p0 = np.asarray(point_3d, float) - centroid
    seed_uv = np.array([p0 @ t1, p0 @ t2])

    return dict(tri_positions=tri_positions, plane_centroid=centroid,
               plane_basis=np.stack([t1, t2]), bound_A=bound_A, bound_b=bound_b,
               seed_uv=seed_uv)


def nearest_big_chart(atlas: dict, big_chart_ids: np.ndarray, point_3d: np.ndarray):
    """Which of big_chart_ids is closest to point_3d (by nearest-TRIANGLE-
    centroid distance, restricted to just the big/usable charts — the small
    fragment charts are never valid seed targets, see module docstring /
    select_big_charts). Returns the chart id (an element of big_chart_ids).

    Used at SEED time (numpy, before any Opti problem exists) to decide which
    chart a randomly/canonically chosen surface point should be assigned to
    for the whole solve — the chart assignment itself does not change across
    Picard relinearization stages, only the LOCAL neighborhood within it (see
    local_neighborhood).
    """
    p = np.asarray(point_3d, float)
    best_ci, best_d2 = None, np.inf
    for ci in big_chart_ids:
        mask = atlas["chart_id"] == ci
        tri_verts = atlas["positions"][atlas["indices"][mask]]
        centroids = tri_verts.mean(axis=1)
        d2 = np.sum((centroids - p) ** 2, axis=1).min()
        if d2 < best_d2:
            best_d2, best_ci = d2, ci
    return int(best_ci)


def point_3d_to_atlas_uv(atlas: dict, chart_id: int, point_3d: np.ndarray):
    """Project a 3D point (object-local frame, on/near the surface) onto its
    nearest triangle within chart_id and return the atlas's own global (u,v)
    at that point (barycentric interpolation of the triangle's 3 UV corners).

    For VISUALIZATION only (e.g. plotting an optimizer's p1/p2 trajectory on
    top of the flat UV atlas image) — nearest-triangle-by-centroid is a cheap
    proxy, not an exact closest-point-on-mesh query; adequate since the NLP's
    own p1/p2 already sit very close to the true surface (reprojected via
    object_sdf.surface_project every iteration).
    """
    mask = atlas["chart_id"] == chart_id
    fidx = np.where(mask)[0]
    tri3d = atlas["positions"][atlas["indices"][fidx]]     # (n,3,3)
    centroids = tri3d.mean(axis=1)
    p = np.asarray(point_3d, float)
    best = int(np.argmin(np.sum((centroids - p) ** 2, axis=1)))
    tri_idx = fidx[best]

    v0, v1, v2 = atlas["positions"][atlas["indices"][tri_idx]]
    uv0, uv1_, uv2_ = atlas["uvs"][atlas["indices"][tri_idx]]

    # barycentric weights of p w.r.t. the 3D triangle (least-squares onto its
    # own plane, since p need not lie EXACTLY on the triangle)
    e1, e2 = v1 - v0, v2 - v0
    n = np.cross(e1, e2)
    n2 = n @ n
    if n2 < 1e-18:
        return uv0  # degenerate triangle, fall back to a corner
    w = p - v0
    b2 = np.dot(np.cross(e1, w), n) / n2
    b1 = np.dot(np.cross(w, e2), n) / n2
    b0 = 1.0 - b1 - b2
    return b0 * uv0 + b1 * uv1_ + b2 * uv2_

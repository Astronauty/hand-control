"""Surface-distance fit of a convex decomposition, in millimetres.

This is the *diagnostic* companion to the gate in mesh_seal.py: the build decides
hull counts on falsely-solid volume, and records the numbers here alongside so
both readings of "how good is this proxy" stay visible.

Keeping both is worth the trouble because they disagree in a specific,
understandable way. Volume is blind to features that are shallow but broad --
`013_apple`'s stem dimple costs almost no volume, so the volume gate accepts a
single convex hull, while this metric reads 4.75 mm there. Distance is blind to
features that are thin but extensive -- a hull wall 1 mm too thick over a large
cavity barely moves the distance and moves the volume a lot, which is why
`073-a_lego_duplo` measures 1.59 mm and still fails the volume gate.

So passing the volume gate does not bound what this measures: across the built
set, objects at `recovered >= 0.85` run from 0.22 mm to 15.97 mm here. If an
object needs surface fidelity specifically, gate on this too -- the build records
it per object as `overshoot_p99_mm` but does not currently act on it.

**What is measured.** Overshoot: how far the boundary of the hull union bulges
*outside* the scanned surface, against the raw scan rather than the sealed solid,
so no sealing assumption enters. Nothing here needs watertightness, which is why
it was usable before mesh_seal.py existed.

Undershoot (scanned surface left outside every hull) is reported as a guard
rather than a target: hulls are supersets of the geometry they cover, so it is
~0 unless CoACD's preprocessing eroded something.
"""
from dataclasses import dataclass, asdict

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REF_SAMPLES = 300_000      # scan-surface samples backing the nearest-surface lookup
HULL_SAMPLES = 40_000      # samples spread over the hull surfaces, by area
NUDGE = 3e-4               # 0.3 mm, see _union_boundary
SEED = 0                   # sampling is seeded: same inputs -> same numbers


@dataclass
class FitError:
    """Overshoot distances in millimetres, plus the hull count they cost."""
    n_hulls: int
    mean_mm: float
    p95_mm: float
    p99_mm: float
    max_mm: float
    undershoot_p99_mm: float
    undershoot_max_mm: float

    def as_dict(self):
        return asdict(self)


def _halfspaces(vertices, faces):
    """Outward halfspace form (A, b) of a convex part: inside iff A @ x <= b.

    qhull triangulates, so a hull's flat sides arrive as many coplanar triangles
    that all carry the same plane; deduplicating to unique planes typically cuts
    the row count several-fold, and every row is a full pass over the points.
    """
    hull = trimesh.Trimesh(vertices, faces).convex_hull
    A = hull.face_normals
    b = np.einsum("ij,ij->i", A, hull.triangles[:, 0])
    planes = np.column_stack([A, b])
    _, keep = np.unique(planes.round(5), axis=0, return_index=True)
    return hull, A[keep], b[keep]


def _outside(A, b, points, block=4096):
    """Largest halfspace violation: > 0 outside the hull, <= 0 inside.

    Outside a hull this is a lower bound on the true distance (exact when the
    nearest feature is a face, short when it is an edge or vertex), which is all
    the undershoot guard needs.

    Blocked over points because the whole-array form materialises a
    len(points) x len(A) temporary -- hundreds of MB for a detailed hull, which
    turns a cache-resident reduction into a memory-bandwidth one.
    """
    out = np.empty(len(points))
    for start in range(0, len(points), block):
        chunk = points[start:start + block]
        out[start:start + block] = (chunk @ A.T - b).max(axis=1)
    return out


def _union_boundary(hulls, halfspaces, rng):
    """Sample the boundary of the union of the hulls.

    Points are drawn per hull in proportion to area, then each is nudged along
    its face normal and kept only if the nudged copy escapes *every* hull.

    Testing the sample itself instead of the nudged copy is the trap: where two
    hulls abut, the shared face lies on the surface of both and strictly inside
    neither, so those points survive — yet they sit deep inside the object, far
    from any scanned surface. Counting them makes the measured error *grow* with
    hull count, which inverts the whole search.
    """
    areas = np.array([h.area for h in hulls])
    counts = np.maximum((HULL_SAMPLES * areas / areas.sum()).astype(int), 200)

    points, probes = [], []
    for hull, count, seed in zip(hulls, counts, rng.integers(1 << 31, size=len(hulls))):
        pts, face_idx = trimesh.sample.sample_surface(hull, int(count), seed=int(seed))
        points.append(pts)
        probes.append(pts + hull.face_normals[face_idx] * NUDGE)
    points, probes = np.vstack(points), np.vstack(probes)

    inside = np.zeros(len(probes), dtype=bool)
    for A, b in halfspaces:
        inside |= _outside(A, b, probes) < 0.0
    return points[~inside]


class FitMetric:
    """Scores decompositions of one mesh; reusable across a parameter sweep.

    The reference sampling and its KD-tree are the expensive part, so they are
    built once and shared by every call.
    """

    def __init__(self, mesh, ref_samples=REF_SAMPLES, seed=SEED):
        self.reference, _ = trimesh.sample.sample_surface(mesh, ref_samples, seed=seed)
        self._tree = cKDTree(self.reference)
        self._seed = seed

    def __call__(self, parts):
        """Score one decomposition, given CoACD's list of (vertices, faces)."""
        rng = np.random.default_rng(self._seed)
        prepared = [_halfspaces(v, f) for v, f in parts]
        hulls = [h for h, _, _ in prepared]
        halfspaces = [(A, b) for _, A, b in prepared]

        boundary = _union_boundary(hulls, halfspaces, rng)
        # Distance to the nearest scan sample approximates distance to the scan
        # surface; with 300k samples the discretisation floor is well under
        # 0.5 mm on a YCB-sized object, an order below any tolerance worth using.
        overshoot, _ = self._tree.query(boundary)

        # Guard direction: scanned surface that no hull covers. Subsampled — it
        # is a sanity check, not a target, and the full reference set would cost
        # a pass per hull.
        probe = self.reference[:: max(1, len(self.reference) // 50_000)]
        gap = np.full(len(probe), np.inf)
        for A, b in halfspaces:
            gap = np.minimum(gap, np.maximum(_outside(A, b, probe), 0.0))

        return FitError(
            n_hulls=len(parts),
            mean_mm=float(overshoot.mean() * 1000),
            p95_mm=float(np.percentile(overshoot, 95) * 1000),
            # p99, not max, is the gate: the scans carry stray specks and small
            # holes (011_banana at 191 hulls measures p99 1.2 mm against a max of
            # 5.0 mm), and a hull count should not be driven by scan artefacts.
            p99_mm=float(np.percentile(overshoot, 99) * 1000),
            max_mm=float(overshoot.max() * 1000),
            undershoot_p99_mm=float(np.percentile(gap, 99) * 1000),
            undershoot_max_mm=float(gap.max() * 1000),
        )


def convex_hull_baseline(mesh):
    """Score of the single convex hull — the error MuJoCo gives with no decomposition."""
    hull = mesh.convex_hull
    return FitMetric(mesh)([(hull.vertices, hull.faces)])

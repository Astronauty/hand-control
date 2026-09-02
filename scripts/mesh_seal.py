"""Turn an open YCB scan into a watertight solid, and measure decompositions against it.

The scans are not closed surfaces. `025_mug` arrives as 157 disconnected
components with ~4.8k boundary edges, so "inside" is undefined: `trimesh` reports
`is_watertight == False`, `mesh.volume` is meaningless, and a volume-based fit
metric has nothing to compare against. Sealing them is what makes volume a usable
common metric across the set.

**Two routes were measured, on objects with published masses to check the answer.**

*Surface repair* (`pymeshfix`, the MeshFix algorithm) is exact where a scan is one
clean shell, but these are not: it has to be run per connected component, and the
fragments seal badly. The mug's 36 surviving components sealed to 19 cm3, an
implied density of 6226 kg/m3; the bowl came out at 4407. Wrong, and wrong by a
different factor per object, which is precisely what a common metric cannot be.

*Solid voxelisation* -- rasterise the shell, close hairline cracks, flood-fill from
outside, and call everything the flood cannot reach interior. A crack narrower
than the voxel pitch simply closes, so scan damage stops mattering. Implied
densities land where the materials say they should: golf ball 1107 (real ~1130),
sponge 49 (foam ~50), mustard bottle 919 (~1000), cracker box 180 (~200).

Voxelisation is what is used here. Its bias is that a shell rasterises about a
voxel thicker than it is, so thin-walled objects read heavy -- the metal bowl
comes out at 1637 kg/m3 rather than steel's 7800. That bias is bounded by the
pitch, which makes it negligible against the centimetre-scale volumes the fit
metric is weighing, and it cancels out of the inertia tensor because the tensor is
rescaled to the published mass rather than to the voxel volume.
"""
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage

GRID = 192      # voxels along the object's longest extent


def _halfspaces(vertices, faces):
    """Outward halfspace form (A, b) of a convex part: inside iff A @ x <= b."""
    hull = trimesh.Trimesh(vertices, faces).convex_hull
    A = hull.face_normals
    b = np.einsum("ij,ij->i", A, hull.triangles[:, 0])
    planes = np.column_stack([A, b])
    _, keep = np.unique(planes.round(5), axis=0, return_index=True)
    return A[keep], b[keep]


@dataclass
class Solid:
    """A watertight solid as an occupancy grid."""
    occupancy: np.ndarray       # bool, (nx, ny, nz)
    origin: np.ndarray          # world position of voxel (0,0,0)'s corner
    pitch: float                # voxel edge, metres

    @property
    def voxel_volume(self):
        return self.pitch ** 3

    @property
    def volume(self):
        return int(np.count_nonzero(self.occupancy)) * self.voxel_volume

    def centres(self):
        idx = np.argwhere(self.occupancy)
        return self.origin + (idx + 0.5) * self.pitch

    def inertia(self, mass):
        """(centre of mass, 3x3 inertia tensor about it) for a uniform solid of `mass`.

        Uniform density is an assumption -- YCB publishes mass, not tensors -- so
        the grid sets the *shape* of the tensor and the published mass sets its
        scale.
        """
        pts = self.centres()
        if len(pts) == 0:
            raise ValueError("sealed solid is empty; cannot derive an inertia tensor")
        m = mass / len(pts)
        com = pts.mean(axis=0)
        r = pts - com
        # point masses at the voxel centres ...
        tensor = m * ((r ** 2).sum() * np.eye(3) - r.T @ r)
        # ... plus each voxel's own inertia about its centre (cube, side = pitch)
        tensor += np.eye(3) * (len(pts) * m * self.pitch ** 2 / 6.0)
        return com, tensor


def seal(mesh, grid=GRID):
    """Rasterise, close cracks, flood-fill from outside -> a watertight Solid."""
    pitch = float(mesh.extents.max()) / grid
    voxels = mesh.voxelized(pitch=pitch)
    occupancy = np.pad(np.asarray(voxels.matrix, dtype=bool), 2)
    origin = np.asarray(voxels.transform[:3, 3]) - (2 + 0.5) * pitch

    # One dilate/erode pass bridges hairline cracks without thickening the shell.
    # It closes gaps up to ~2 voxels, i.e. ~1% of the longest extent, so a real
    # feature thinner than that would be sealed over as if it were scan damage.
    # The narrowest genuine gaps in the set clear it -- 030_fork's tines are ~4 mm
    # against a 1.1 mm pitch -- but a finer-featured object would need a larger
    # grid rather than this default.
    occupancy = ndimage.binary_closing(occupancy, iterations=1)

    reachable = np.zeros_like(occupancy)
    reachable[0, 0, 0] = True
    reachable = ndimage.binary_propagation(reachable, mask=~occupancy)
    return Solid(occupancy | ~reachable, origin, pitch)


def rasterise(parts, solid):
    """Occupancy of the union of convex parts, on `solid`'s grid.

    Testing every voxel against every plane is O(voxels x planes), and a scanned
    object's convex hull carries ~1000 planes -- for a 192^3 grid that is 7e9
    operations per hull, which made a single object take minutes (and OOM-killed
    the naive whole-array form).

    Convexity gives a much better algorithm. Along any column (x, y) fixed, the
    inside set is a single interval in z, because a convex body meets a line in
    one segment. So each plane is reduced to a bound on z, the bounds are combined
    per column, and the resulting span is filled directly -- O(columns x planes),
    a factor of the z-resolution cheaper, with no per-voxel test at all.
    """
    shape = solid.occupancy.shape
    occupancy = np.zeros(shape, dtype=bool)
    z_index = np.arange(shape[2])

    for vertices, faces in parts:
        A, b = _halfspaces(vertices, faces)
        lo = np.maximum(np.floor((vertices.min(0) - solid.origin) / solid.pitch).astype(int) - 1, 0)
        hi = np.minimum(np.ceil((vertices.max(0) - solid.origin) / solid.pitch).astype(int) + 2, shape)
        if np.any(hi <= lo):
            continue

        xs = solid.origin[0] + (np.arange(lo[0], hi[0]) + 0.5) * solid.pitch
        ys = solid.origin[1] + (np.arange(lo[1], hi[1]) + 0.5) * solid.pitch
        az, upper, lower = A[:, 2], A[:, 2] > 0, A[:, 2] < 0
        flat = ~(upper | lower)                     # planes parallel to z

        for i, x in enumerate(xs):
            # residual per (y, plane): A_z * z <= rhs
            rhs = (b - A[:, 0] * x)[None, :] - np.outer(ys, A[:, 1])

            z_hi = np.full(len(ys), np.inf)
            z_lo = np.full(len(ys), -np.inf)
            if upper.any():
                z_hi = (rhs[:, upper] / az[upper]).min(axis=1)
            if lower.any():
                z_lo = (rhs[:, lower] / az[lower]).max(axis=1)
            feasible = (rhs[:, flat] >= 0).all(axis=1) if flat.any() else True

            k_lo = np.ceil((z_lo - solid.origin[2]) / solid.pitch - 0.5)
            k_hi = np.floor((z_hi - solid.origin[2]) / solid.pitch - 0.5)
            k_lo = np.maximum(k_lo, lo[2])
            k_hi = np.minimum(k_hi, hi[2] - 1)
            span = (k_lo <= k_hi) & feasible
            if not np.any(span):
                continue
            block = ((z_index[None, :] >= k_lo[:, None])
                     & (z_index[None, :] <= k_hi[:, None]) & span[:, None])
            occupancy[lo[0] + i, lo[1]:hi[1], :] |= block
    return occupancy


@dataclass
class VolumeError:
    """Volume agreement between a decomposition and the sealed object."""
    n_hulls: int
    solid_cm3: float            # sealed object
    union_cm3: float            # union of the hulls
    hull_cm3: float             # single convex hull, the no-decomposition baseline
    false_solid_cm3: float      # claimed solid, actually empty
    false_solid_frac: float     # as a fraction of the convex hull -- the gate
    recovered: float            # 1 = perfect, 0 = no better than the convex hull
    missing_cm3: float          # object volume no hull covers

    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def volume_error(solid, parts, hull_occupancy):
    """Score one decomposition. `hull_occupancy` is the convex-hull baseline grid."""
    union = rasterise(parts, solid)
    cm3 = solid.voxel_volume * 1e6
    false_solid = np.count_nonzero(union & ~solid.occupancy)
    hull_excess = np.count_nonzero(hull_occupancy & ~solid.occupancy)
    hull_total = max(np.count_nonzero(hull_occupancy), 1)
    return VolumeError(
        n_hulls=len(parts),
        solid_cm3=np.count_nonzero(solid.occupancy) * cm3,
        union_cm3=np.count_nonzero(union) * cm3,
        hull_cm3=hull_total * cm3,
        false_solid_cm3=false_solid * cm3,
        # Normalised by the convex hull, not by the object: for a thin-walled cup
        # the object's own volume is a sliver of what the proxy can get wrong, so
        # dividing by it gives ratios in the hundreds that compare across nothing.
        # The hull is the worst case any decomposition starts from, so this is
        # bounded in [0, 1] and means the same thing on a mug as on a banana.
        false_solid_frac=false_solid / hull_total,
        recovered=1.0 - false_solid / max(hull_excess, 1),
        missing_cm3=np.count_nonzero(solid.occupancy & ~union) * cm3,
    )

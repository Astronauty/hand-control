"""Reachable workspace of the Kinova+LEAP palm, by forward-kinematic sampling.

Random arm configurations are pushed through FK and the palm position recorded,
then voxelised. Object poses for a trial are drawn from the occupied voxels, so
"floating in the workspace" means something checkable rather than a hand-picked
box that happens to look reasonable.

Raw FK reachability alone is a weak filter -- the palm cloud is close to a
spherical shell around the base and covers most of the table. The filters below
(height band, in front of the base, palm pointing down-ish) are what make the
sampled region somewhere a grasp could actually be attempted.
"""
import hashlib
from pathlib import Path

import mujoco as mj
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "assets" / "kinova_workspace.npz"

N_ARM = 7                  # Gen3 joints; the 16 LEAP joints do not move the palm
PALM_BODY = "leap_palm"

# Gen3 joints 1/3/5/7 (indices 0,2,4,6) are continuous -- jnt_range compiles to
# [0,0], which would collapse the sample to a single configuration.
CONTINUOUS = (0, 2, 4, 6)


def _arm_bounds(model):
    lo, hi = model.jnt_range[:N_ARM, 0].copy(), model.jnt_range[:N_ARM, 1].copy()
    for j in CONTINUOUS:
        lo[j], hi[j] = -np.pi, np.pi
    return lo, hi


def sample_palm_cloud(model, n=200_000, seed=0):
    """Palm world positions over n random arm configurations (fingers at zero)."""
    rng = np.random.default_rng(seed)
    lo, hi = _arm_bounds(model)
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, PALM_BODY)
    if bid < 0:
        raise ValueError(f"no body {PALM_BODY!r} in model")

    data = mj.MjData(model)
    pos = np.empty((n, 3))
    zdir = np.empty(n)
    for i in range(n):
        data.qpos[:N_ARM] = rng.uniform(lo, hi)
        mj.mj_kinematics(model, data)
        pos[i] = data.xpos[bid]
        # palm approach axis in world; used for the "pointing down-ish" filter
        zdir[i] = data.xmat[bid].reshape(3, 3)[2, 2]
    return pos, zdir


def build(model, n=200_000, voxel=0.03, seed=0,
          z_band=(0.05, 0.55), r_min=0.25, r_max=0.85, palm_down=0.0):
    """Voxel occupancy of the *usable* palm workspace.

    z_band / r_min / r_max / palm_down are the filters described in the module
    docstring; palm_down keeps samples whose palm -z has a downward component.
    """
    pos, zdir = sample_palm_cloud(model, n=n, seed=seed)
    r = np.linalg.norm(pos[:, :2], axis=1)
    keep = ((pos[:, 2] > z_band[0]) & (pos[:, 2] < z_band[1])
            & (r > r_min) & (r < r_max)
            & (pos[:, 0] > 0.0)              # in front of the base
            & (-zdir > palm_down))           # palm -z points down-ish
    pts = pos[keep]
    if len(pts) == 0:
        raise RuntimeError("no samples survived the workspace filters")

    origin = np.floor(pts.min(0) / voxel) * voxel
    idx = np.floor((pts - origin) / voxel).astype(int)
    keys = np.unique(idx, axis=0)
    centers = origin + (keys + 0.5) * voxel
    return dict(centers=centers, voxel=voxel, origin=origin,
                n_samples=n, n_kept=int(keep.sum()))


def load_or_build(model, cache=CACHE, **kw):
    """Cached build. The key covers the filter settings, so changing one rebuilds."""
    key = hashlib.blake2b(repr(sorted(kw.items())).encode()).hexdigest()[:12]
    if cache.exists():
        z = np.load(cache)
        if str(z["key"]) == key:
            return dict(centers=z["centers"], voxel=float(z["voxel"]),
                        origin=z["origin"], n_samples=int(z["n_samples"]),
                        n_kept=int(z["n_kept"]))
    ws = build(model, **kw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=key, **ws)
    return ws


def sample_positions(ws, n, rng, margin=0.0):
    """Draw n positions uniformly from the occupied voxels (jittered within each)."""
    c = ws["centers"]
    pick = rng.integers(0, len(c), size=n)
    half = ws["voxel"] / 2 - margin
    return c[pick] + rng.uniform(-half, half, size=(n, 3))


def shell_mesh(ws):
    """Boundary-voxel shell as (verts, faces), for a transparent group-4 geom.

    Only voxels missing at least one 6-neighbour are emitted, so the interior of
    the cloud is not drawn -- the shell is what conveys the reachable envelope.
    """
    v = ws["voxel"]
    keys = np.round((ws["centers"] - ws["origin"]) / v - 0.5).astype(int)
    occupied = {tuple(k) for k in keys}
    offs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    surface = np.array([k for k in keys
                        if any((k[0] + o[0], k[1] + o[1], k[2] + o[2]) not in occupied
                               for o in offs)])
    centers = ws["origin"] + (surface + 0.5) * v

    # one axis-aligned cube per surface voxel
    unit = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * (v / 2)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    verts = (centers[:, None, :] + unit[None]).reshape(-1, 3)
    faces = []
    for i in range(len(centers)):
        b = i * 8
        for a, bb, c2, d in quads:
            faces += [[b + a, b + bb, b + c2], [b + a, b + c2, b + d]]
    return verts, np.array(faces, dtype=np.int32)

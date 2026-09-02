"""Precomputed signed-distance tables for mesh objects, queryable from CasADi.

Why this exists
---------------
`ConstrainedIKSolver` dispatches on geom type and has no mesh branch: a mesh
falls through to `_sphere_sphere_distance` and is modelled by its bounding
sphere (constrained_ik.py:890). Measured on the YCB demo, that costs 6-18x in
fingertip placement error -- the targets are reachable, the sphere proxy is what
blocks them. The grasp recommender is worse off still: `_sym_geom_surface_con`
has no mesh branch at all, so a contact on a mesh gets *no* surface constraint,
and `_geom_normal_np` falls through to the radial direction from the geom
centroid, which is not a surface normal.

What this provides
------------------
One table per object *shape*, holding the signed distance to the union of that
object's collision hulls, sampled on a regular 3-D lattice in the object's BODY
frame. A robot body's distance to the object is then

    d(R_obj^T (x_body - c_obj)) - r_body

so the table is shared by every body -- there is nothing per-pair, and nothing
depends on the object's pose. The robot side is already bounding spheres, which
is what makes one field per object sufficient.

Cost is independent of geometry complexity, which matters because hull count is
now driven by a fidelity target rather than a budget (median 23, max 126 across
the YCB set): evaluating hulls directly scales with that count, a table lookup
does not.

Accuracy notes
--------------
The stored values are `min_j max_i (a_ij . x - b_ij)` over the exact hull
half-spaces: exact inside, and outside exact wherever the nearest feature is a
face, a slight *under*estimate near edges and vertices. Underestimating is the
safe direction for collision avoidance.

The B-spline rounds off creases, which leaves it very slightly optimistic
(reporting more clearance than exists). `bake` measures that overshoot per
object and stores it as `safety_offset`, subtracted on every query, which
restores conservatism.

The field is NOT eikonal: ||grad d|| is ~1 on faces but collapses toward 0 at
concave creases where hulls meet, because the exterior medial axis touches the
surface there. Baking true Euclidean distance does not fix this -- it is a
property of the geometry, not of the representation. Anything that divides by
||grad d||^2 (a Newton step) will blow up there; use `surface_project`, which
steps by the distance along the unit gradient and so cannot overshoot.
"""
import hashlib
from pathlib import Path

import casadi as ca
import mujoco as mj
import numpy as np
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "assets" / "ycb_sdf"

DEFAULT_N = 64
DEFAULT_PAD = 0.02        # metres of grid beyond the hull bounding box
PROBE_N = 4000            # probes used to measure the safety offset
SUBSET_FACES = 24         # faces in the cheap per-hull lower bound (see union_sdf_np)
NORM_EPS = 1e-9           # softening for the far-field norm (see casadi_fn)


# --------------------------------------------------------------------------
# geometry -> half-spaces
# --------------------------------------------------------------------------

def body_hull_halfspaces(model, body_id, group=3):
    """Exact half-space form of every collision hull on a body, in BODY frame.

    Each hull geom carries its own `geom_pos`/`geom_quat`; those are model
    constants, so folding them in here yields one common frame and leaves only
    body->world varying at solve time.

    Returns (hulls, verts) where hulls is [(A, b, centre, radius)] with
    `A x <= b` inside, plus a bounding sphere used only to prune.
    """
    hulls, verts = [], []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id or model.geom_group[g] != group:
            continue
        if model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        vid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[vid], model.mesh_vertnum[vid]
        R = np.zeros(9)
        mj.mju_quat2Mat(R, model.geom_quat[g])
        V = model.mesh_vert[a:a + n] @ R.reshape(3, 3).T + model.geom_pos[g]
        h = ConvexHull(V)
        # Bounding sphere from the vertices. It cannot be read off (A, b): the
        # support value along an axis is an LP, not the b of the face whose
        # normal leans that way -- using the latter under-bounds the hull and
        # silently prunes hulls that were in fact nearest (measured 14.6 mm of
        # error before this was fixed).
        c = (V.max(0) + V.min(0)) / 2
        r = float(np.linalg.norm(V - c, axis=1).max())
        hulls.append((h.equations[:, :3].copy(), -h.equations[:, 3].copy(), c, r))
        verts.append(V)
    if not hulls:
        raise ValueError(f"body {body_id} has no group-{group} mesh geoms")
    return hulls, np.vstack(verts)


def union_sdf_np(P, hulls, chunk=20000):
    """Signed distance to the union of convex hulls. Negative inside.

    Evaluated as min over hulls of the per-hull half-space max. Done naively
    this is (points x faces) per hull -- for a 64^3 grid against a 44-hull mug
    with ~1k faces each that is ~1e10 operations and a multi-GB intermediate.
    Chunking bounds the memory; a two-tier bound cuts the work.

    The bound has to be chosen with care. A bounding sphere is NOT usable here:
    `||P - c|| - r` bounds the true *Euclidean* distance, whereas the value
    being minimised is the half-space max, which is itself an underestimate of
    that distance near edges and vertices. The sphere bound can therefore exceed
    the quantity it is supposed to bound, pruning the hull that actually
    attains the minimum (measured: 17.7 mm of error on the mug).

    What is valid: the max over any *subset* of a hull's faces is a lower bound
    on the max over all of them. So a cheap max over a spread of faces prunes
    soundly, and the full face set is only touched for hulls that survive.
    """
    P = np.atleast_2d(np.asarray(P, float))
    out = np.empty(len(P))
    subs = [np.linspace(0, len(b) - 1, min(SUBSET_FACES, len(b))).astype(int)
            for _, b, _, _ in hulls]

    for s in range(0, len(P), chunk):
        Q = P[s:s + chunk]
        d = np.full(len(Q), np.inf)
        # nearest hulls first, so the running min tightens early and prunes more
        order = np.argsort([np.linalg.norm(Q.mean(0) - c) - r
                            for _, _, c, r in hulls])
        for j in order:
            A, b, _, _ = hulls[j]
            k = subs[j]
            lb = (Q @ A[k].T - b[k]).max(axis=1)       # <= max over all faces
            m = lb < d
            if not m.any():
                continue
            d[m] = np.minimum(d[m], (Q[m] @ A.T - b).max(axis=1))
        out[s:s + chunk] = d
    return out


# --------------------------------------------------------------------------
# bake
# --------------------------------------------------------------------------

def _cache_key(verts, n, pad):
    h = hashlib.blake2b(np.ascontiguousarray(verts, dtype=np.float64).tobytes())
    h.update(f"{n}:{pad}".encode())
    return h.hexdigest()[:16]


def bake(hulls, verts, n=DEFAULT_N, pad=DEFAULT_PAD, seed=0):
    """Sample the union SDF on a lattice spanning the hull bbox + pad.

    Returns a dict: lo, hi, n, values (n,n,n), safety_offset, rbound.
    """
    lo = verts.min(0) - pad
    hi = verts.max(0) + pad
    axes = [np.linspace(lo[k], hi[k], n) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    vals = union_sdf_np(G, hulls).reshape(n, n, n)

    table = dict(lo=lo, hi=hi, n=int(n), values=vals,
                 safety_offset=0.0,
                 rbound=float(np.linalg.norm(verts, axis=1).max()))

    # How optimistic is the interpolant? Probe inside the grid box and take the
    # largest overshoot; subtracting it makes every query conservative.
    rng = np.random.default_rng(seed)
    P = rng.uniform(lo, hi, size=(PROBE_N, 3))
    f = casadi_fn(table)
    est = np.array([float(f(p)) for p in P])
    table["safety_offset"] = float(max((est - union_sdf_np(P, hulls)).max(), 0.0))
    return table


def load_or_bake(model, body_id, n=DEFAULT_N, pad=DEFAULT_PAD,
                 cache_dir=CACHE_DIR, group=3):
    """Bake once per object *shape*; identical shapes share a table.

    Keyed on a hash of the hull vertices rather than a path or body name, so a
    remesh, a scale change or a different `maxhullvert` all invalidate it --
    a name-keyed cache would silently serve a stale field.
    """
    hulls, verts = body_hull_halfspaces(model, body_id, group=group)
    key = _cache_key(verts, n, pad)
    path = Path(cache_dir) / f"{key}.npz"
    if path.exists():
        z = np.load(path)
        return dict(lo=z["lo"], hi=z["hi"], n=int(z["n"]), values=z["values"],
                    safety_offset=float(z["safety_offset"]),
                    rbound=float(z["rbound"])), hulls
    table = bake(hulls, verts, n=n, pad=pad)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **table)
    return table, hulls


# --------------------------------------------------------------------------
# CasADi query
# --------------------------------------------------------------------------

def casadi_fn(table, name="obj_sdf", conservative=True):
    """CasADi Function: object-local point (3,) -> signed distance (1,).

    Outside the grid box the interpolant has no data and decays to 0, which
    would read as *touching* at any distance and make the problem infeasible
    everywhere. The query is therefore clamped into the box and the distance
    travelled added back -- valid, mildly conservative, and differentiable.

    conservative=True subtracts the measured safety offset, which is what a
    collision constraint wants: never report more clearance than exists.

    Use conservative=False for *placing* contacts. Projecting against the
    offset field solves `d - offset = 0`, which lands the contact one offset
    OUTSIDE the true surface -- measured as a systematic 0.64-0.89 mm bias,
    exactly the offset, before this was split out.
    """
    axes = [np.linspace(table["lo"][k], table["hi"][k], table["n"])
            for k in range(3)]
    interp = ca.interpolant(f"{name}_grid", "bspline", axes,
                            np.asarray(table["values"]).ravel(order="F"))
    x = ca.MX.sym("x", 3)
    xc = ca.fmin(ca.fmax(x, ca.DM(table["lo"])), ca.DM(table["hi"]))
    # Softened norm, not ca.norm_2: inside the box x - xc is exactly zero and
    # norm_2 differentiates to 0/0 there, so every gradient in the interior --
    # i.e. everywhere that matters -- comes back NaN. The epsilon adds ~1 nm to
    # the value and makes the derivative vanish smoothly at the origin instead.
    outside = ca.sqrt(ca.sumsqr(x - xc) + NORM_EPS ** 2)
    off = float(table["safety_offset"]) if conservative else 0.0
    return ca.Function(name, [x], [outside + interp(xc) - off])


def casadi_grad_fn(fn, name=None):
    """Gradient of a `casadi_fn`, i.e. the unnormalised outward normal."""
    x = ca.MX.sym("x", 3)
    return ca.Function(name or (fn.name() + "_grad"), [x], [ca.gradient(fn(x), x)])


def casadi_normal_fn(fn, name=None, eps=1e-9):
    """Unit outward surface normal, grad d / ||grad d||.

    Normalising matters here: ||grad d|| is ~1 on faces but sags at creases, so
    the raw gradient is a direction with an unreliable magnitude.
    """
    x = ca.MX.sym("x", 3)
    g = ca.gradient(fn(x), x)
    return ca.Function(name or (fn.name() + "_normal"), [x],
                       [g / (ca.norm_2(g) + eps)])


def surface_project(fn, k=5, eps=1e-9, name=None):
    """CasADi Function mapping a free point onto the object surface.

    Steps by the distance value along the unit gradient, k times, unrolled so
    the whole thing is differentiable. This is the `Pi_surf` parameterisation
    that replaces a hard `SDF(p) = 0` equality.

    The unit-direction step is deliberate. A Newton step divides by
    ||grad d||^2, which at a crease (measured ||grad d|| ~ 0.01) inflates the
    step by four orders of magnitude -- 107 mm to close a 1 mm gap, and up to
    354 mm of excursion across a probe set. Stepping by the distance along the
    unit direction cannot overshoot: the residual is bounded, though it does
    oscillate at creases rather than converging, so more iterations do not
    strictly help (measured max residual: k=3 1.34 mm, k=5 0.84 mm, k=10 0.98 mm).
    """
    # The gradient must be built once against a fresh symbol and then *called*:
    # ca.gradient(fn(p), p) fails from the second unroll step on, because p is
    # an expression by then and Xfunction requires purely symbolic inputs.
    grad = casadi_grad_fn(fn)
    u = ca.MX.sym("u", 3)
    p = u
    for _ in range(k):
        g = grad(p)
        p = p - fn(p) * g / (ca.norm_2(g) + eps)
    return ca.Function(name or (fn.name() + "_project"), [u], [p])

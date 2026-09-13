"""FRoGGeR's heuristic palm sampler (their App. B-C), for the benchmark's frogger arm.

WHAT AN OBB IS. An oriented bounding box is the tightest rectangular box around the
object, aligned to the object's OWN principal axes rather than the world's. An
axis-aligned box around a banana lying diagonally is mostly empty; an OBB rotates to
hug it, so its three edge lengths are meaningful object dimensions (length, width,
thickness) rather than artifacts of how the object happens to be oriented.

WHY THIS EXISTS HERE. FRoGGeR's (7a) is `maximize l*(q)` with no alignment or IK term,
so nothing in their objective prefers OPPOSED contacts over same-side ones. That
preference lives entirely in the SEED: the palm is aligned to an OBB axis and the
fingers are pre-separated by that axis's width, so the hand starts straddling the
object and beta only has to refine what the seed already got structurally right.

Our own arm supplies the same preference through `w_align` and `w_ik`, which the
frogger arm zeroes to match (7a). Measured consequence of zeroing them without this
sampler (017_orange seed 0, tripod): the solve returned all three contacts on one side
with pairwise normal dots +0.914/+0.966/+0.931, i.e. not force closure, and beta
reported -0.0. So this is not a performance tweak -- it is the part of their method
that carries the geometry their objective does not.

STATUS. Steps 1-5 are implemented and the geometry is verified (see the validation
notes on each function). What is NOT yet done is applying step 1's finger SEPARATION
to the hand joints: `preshape_finger_separation` returns the target width, but nothing
converts it into LEAP joint angles. Measured consequence on the tabletop scene, thumb
vs index direction from the object centre: the palm reaches its sampled pose to
sub-millimetre on good draws, but the straddle dot sits at ~+0.3 to +0.6 rather than
negative, i.e. the fingers arrive on the SAME side. Until the preshape lands, this
sampler improves the palm pose without yet supplying the opposed-contact start that
motivated it.

THEIR ALGORITHM (App. B-C), five steps:
  1. Choose an OBB axis to align the palm's finger-SEPARATION direction with,
     with probability proportional to that axis's edge length, and use that edge's
     width to set the initial finger separation.
  2. Of the two remaining axes, choose one for the palm's outward normal (the
     APPROACH direction).
  3. Perturb the palm frame with von Mises-Fisher noise on the 2-sphere.
  4. Place the palm ~4 cm from the object's surface, approximated by the box.
  5. Solve IK for q0, WITHOUT collision constraints and WITHOUT forcing fingertips
     onto the surface -- deliberately coarse, since their point is robustness to a
     poor initial guess.

PALM AXIS CONVENTION. Theirs is x = outward palm normal, z = finger direction, y =
right-handed. LEAP's palm frame is NOT that, so writing their convention literally
would aim the palm sideways. Measured at the home pose (thumb/index tip positions in
the palm frame):

    thumb-index separation = [-0.197, 0.114, -0.001]  -> dominated by palm X
    mean finger direction  = [ 0.277, -0.354, -0.893] -> along palm -Z

giving the mapping

    their y (separation) -> our +x
    their z (fingers)    -> our -z
    their x (approach)   -> our +y

which is what `PALM_AXES` below records. Re-measure it if the hand model changes.
"""
from __future__ import annotations

import numpy as np

# LEAP palm frame, measured (see module docstring). Indices into the palm rotation's
# columns, with a sign, for each of FRoGGeR's three semantic directions.
PALM_AXES = {
    "separation": (0, +1.0),   # their y: thumb->index closing direction
    "approach":   (1, +1.0),   # their x: outward palm normal
    "fingers":    (2, -1.0),   # their z: where the fingers point
}

PALM_STANDOFF_M = 0.04         # their "roughly 4cm from the surface"


def object_obb(vertices, R_WO=None, p_WO=None):
    """Oriented bounding box of a point set, by PCA on the vertices.

    FRAME. `vertices` may be given in the object's BODY frame (which is what
    `table_scene.hull_vertices` returns) together with that body's world pose, in
    which case the box is transformed into WORLD coordinates before being returned.
    Pass R_WO/p_WO whenever the caller will use the result against world-frame
    quantities -- a palm pose, the table height, the arm's reach. Omitting them on
    body-frame vertices silently produces a box centred near the origin: measured on
    the tabletop scene, an object whose true centre is z = 0.625 comes back at
    z = 0.035, and every sampled palm pose then aims at the floor under the table.

    Returns (center, axes, half_extents) with `axes` a 3x3 whose COLUMNS are the
    box's unit axes, sorted by DESCENDING edge length so axis 0 is the longest.

    open3d's `get_oriented_bounding_box` is what the paper used; PCA on the hull
    vertices is the same construction for our purposes and avoids the dependency.
    It is approximate for shapes whose covariance does not align with the minimal
    box (an L-shape, say), which is also true of open3d's fast path and is why the
    paper calls the sampler coarse.
    """
    V = np.asarray(vertices, float).reshape(-1, 3)
    c = V.mean(0)
    # Columns of Vt.T are the principal directions.
    _, _, Vt = np.linalg.svd(V - c, full_matrices=True)
    A = Vt.T
    proj = (V - c) @ A
    lo, hi = proj.min(0), proj.max(0)
    half = 0.5 * (hi - lo)
    # Re-center: PCA centers on the vertex mean, which is not the box center unless
    # the point distribution is symmetric along every axis.
    c = c + A @ (0.5 * (hi + lo))
    order = np.argsort(-half)                      # longest edge first
    A = A[:, order]
    half = half[order]
    if R_WO is not None:
        R_WO = np.asarray(R_WO, float).reshape(3, 3)
        A = R_WO @ A
        c = R_WO @ c + (np.zeros(3) if p_WO is None else np.asarray(p_WO, float))
    elif p_WO is not None:
        c = c + np.asarray(p_WO, float)
    # SVD returns an orthonormal triple that may be LEFT-handed, and reordering the
    # columns flips handedness again on odd permutations. The sampler crosses two of
    # these axes to build the third palm direction, so a left-handed box silently
    # mirrors the resulting palm frame. Fix the sign here, at the source.
    if np.linalg.det(A) < 0:
        A[:, 2] *= -1.0
    return c, A, half


def _vmf_perturb(v, kappa, rng):
    """Sample a unit vector from the von Mises-Fisher distribution on S^2 about `v`.

    kappa is the concentration: large = tightly clustered about v, 0 = uniform.
    Uses Wood's method, which for S^2 reduces to a closed form for the polar angle.
    """
    v = np.asarray(v, float)
    v = v / (np.linalg.norm(v) + 1e-12)
    if kappa <= 0:
        w = rng.normal(size=3)
        return w / np.linalg.norm(w)
    # w = cos(theta), density proportional to exp(kappa * w) on [-1, 1].
    u = rng.random()
    w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    # Uniform azimuth in the plane perpendicular to v.
    t = rng.normal(size=3)
    t -= t.dot(v) * v
    t /= (np.linalg.norm(t) + 1e-12)
    return w * v + np.sqrt(max(0.0, 1.0 - w * w)) * t


def sample_palm_pose(vertices, rng, kappa=30.0, standoff=PALM_STANDOFF_M,
                     min_height_axis=None, obb=None):
    """One palm pose from the heuristic sampler. Steps 1-4 of App. B-C.

    Returns a dict with
        R_WP      (3,3) desired palm rotation, columns = palm x/y/z in world
        p_WP      (3,)  desired palm origin in world
        width     float the chosen OBB edge's full width -- the initial finger
                        separation their step 1 fixes
        sep_axis  (3,)  the separation direction actually used (post-noise)
        approach  (3,)  the approach direction actually used (post-noise)

    min_height_axis : world direction that counts as "up". When given, an approach
        whose outward normal does not point at least somewhat downward onto the
        object is re-drawn -- their "for very short objects, we only accepted a palm
        frame whose x-axis approached the object from above to avoid heavy
        collisions with the tabletop".
    """
    c, A, half = obb if obb is not None else object_obb(vertices)
    edges = 2.0 * half

    # (1) separation axis, P(axis) proportional to its edge length.
    p = edges / edges.sum()
    i_sep = int(rng.choice(3, p=p))
    width = float(edges[i_sep])
    sep = A[:, i_sep] * (1.0 if rng.random() < 0.5 else -1.0)

    # (2) approach axis: one of the two remaining, sign chosen below.
    rest = [k for k in range(3) if k != i_sep]
    i_app = int(rng.choice(rest))
    app = A[:, i_app] * (1.0 if rng.random() < 0.5 else -1.0)

    # (3) von Mises-Fisher noise on both directions, then re-orthogonalize.
    sep = _vmf_perturb(sep, kappa, rng)
    app = _vmf_perturb(app, kappa, rng)
    app = app - app.dot(sep) * sep
    n_app = np.linalg.norm(app)
    if n_app < 1e-6:                      # degenerate draw; fall back to clean axes
        app = A[:, i_app].copy()
        app = app - app.dot(sep) * sep
        n_app = np.linalg.norm(app)
    app /= n_app

    if min_height_axis is not None:
        up = np.asarray(min_height_axis, float)
        up /= (np.linalg.norm(up) + 1e-12)
        # The palm should look DOWN at the object, i.e. its outward normal points
        # up-ish so that -x (the approach travel) goes down onto it.
        if app.dot(up) < 0.0:
            app = -app

    fing = np.cross(sep, app)             # right-handed third direction
    fing /= (np.linalg.norm(fing) + 1e-12)

    # (4) palm origin: standoff from the box face along the approach direction.
    reach = float(np.abs(np.asarray(half) @ np.abs(A.T @ app)))
    p_WP = c + app * (reach + standoff)

    # Assemble the palm rotation in OUR convention (see PALM_AXES).
    cols = [None, None, None]
    for name, vec in (("separation", sep), ("approach", app), ("fingers", fing)):
        k, s = PALM_AXES[name]
        cols[k] = s * vec
    R = np.column_stack(cols)
    # Re-orthonormalize: the three assembled columns are orthogonal by construction
    # but the signed remap can flip handedness, and a left-handed "rotation" makes
    # every downstream frame silently mirrored.
    U, _, Vt2 = np.linalg.svd(R)
    R = U @ Vt2
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return dict(R_WP=R, p_WP=p_WP, width=width, sep_axis=sep, approach=app,
                obb_center=c, obb_axes=A, obb_half=half)


def solve_palm_ik(model, data, palm_bid, R_WP, p_WP, q_init, n_robot,
                  iters=200, step=0.5, damping=1e-3, tol_pos=1e-3, tol_rot=1e-2):
    """Step 5: IK for q0 from a desired PALM pose, by damped least squares on the
    BODY Jacobian.

    Their step 5 solves for the palm frame's pose and explicitly does NOT enforce
    collision constraints or put the fingertips on the surface -- "we only consider
    infeasible candidate grasps", the point being that refinement is robust to a
    poor initial guess. So this is deliberately a bare 6-DOF pose solve rather than
    our collision-aware `constrained_ik`.

    Written against `mj_jacBody` rather than reusing grasp_control.ik: that solver
    is built on `mj_jacSite`, and the LEAP palm body carries NO site in this model
    (verified: sites exist only on the four fingertips, the hand mount, and two
    scene markers), so there is nothing for it to target.

    Only the 7 ARM joints are driven. The hand joints do not move the palm at all --
    they are distal to it -- so including them adds null-space columns that the
    damped solve would happily use to no effect, and would corrupt the finger
    preshape the caller sets separately.

    Returns the n_robot-length q, or None if the solve diverges.
    """
    import mujoco as mj
    d = mj.MjData(model)
    d.qpos[:] = data.qpos[:]
    q = np.asarray(q_init, float)[:n_robot].copy()
    R_des = np.asarray(R_WP, float)
    p_des = np.asarray(p_WP, float)
    n_arm = 7
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(int(iters)):
        d.qpos[:n_robot] = q
        mj.mj_kinematics(model, d)
        mj.mj_comPos(model, d)
        p_cur = d.xpos[palm_bid]
        R_cur = d.xmat[palm_bid].reshape(3, 3)
        e_p = p_des - p_cur
        # Orientation error as the axis-angle of R_des @ R_cur^T, via the same
        # cross-product form grasp_control.ik uses for its SO(3) targets.
        e_r = 0.5 * sum(np.cross(R_cur[:, i], R_des[:, i]) for i in range(3))
        if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
            break
        mj.mj_jacBody(model, d, jacp, jacr, palm_bid)
        J = np.vstack([jacp[:, :n_arm], jacr[:, :n_arm]])
        e = np.concatenate([e_p, e_r])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
        q[:n_arm] = q[:n_arm] + step * dq
    if not np.all(np.isfinite(q)):
        return None
    # Respect joint limits: an out-of-range seed is not reachable by anything
    # downstream that samples within range (the RRT, notably).
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            a = model.jnt_qposadr[j]
            if a < n_robot:
                q[a] = np.clip(q[a], model.jnt_range[j, 0], model.jnt_range[j, 1])
    return q


def preshape_finger_separation(width_m):
    """Their step 1 also fixes an initial finger SEPARATION from the chosen OBB edge.

    Returned as the target thumb-to-opposing-fingertip distance so a caller can
    pre-shape the hand; the LEAP hand's finger joints are not exposed as a single
    'span' DOF, so translating this into joint angles is the caller's job (the
    planner's own seeding does it by placing contacts, which is equivalent for our
    purposes and is why this is advisory rather than applied here).

    Clamped to what the hand can actually span, since an OBB edge can exceed it --
    a 210 mm wood block against a hand whose thumb-index reach is ~0.23 m at full
    extension. An unreachably wide preshape is worse than a merely wide one: it
    saturates every finger joint at its limit and the IK then has no gradient left.
    """
    return float(np.clip(float(width_m), 0.02, 0.18))

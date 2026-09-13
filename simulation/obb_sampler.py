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

STATUS: implemented, IMPROVING BUT NOT YET SUFFICIENT. Steps 1-5 are all present,
including the preshape (`solve_preshape`) and the measured hand frame
(`palm_frame_for_preshape`). Quality is measured as the STRADDLE DOT -- the cosine
between the thumb and index directions from the object centre, where negative means
the two fingers are on opposite sides, which is what a force-closure seed needs.

Measured over 40 draws per object, counting only draws the arm can actually reach:

    object            reachable   opposed (dot<0)   best dot
    017_orange          14/40           0            +0.67
    036_wood_block      13/40           0            +0.00
    011_banana           4/40           2            -0.18

Each fix moved this in the right direction (the block went from ~+0.45 to ~+0.05,
and the banana now yields genuinely opposed seeds), but most draws still put both
fingers on the same side, so the sampler does NOT yet reliably supply the opposed
start that motivated it. Two candidates remain, in order: the LEAP thumb may simply
not oppose the index across most of its span without also rotating the abduction
DOFs this solver leaves fixed; and the 4 cm standoff may be too large for a hand
whose fingers curl, so the tips pass the object rather than closing on it.

Do not treat this as a finished replication of their sampler.

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

which held at the HOME pose only -- see the note above `PALM_STANDOFF_M`. The
closing direction is preshape-dependent, so it is measured per draw by
`palm_frame_for_preshape` rather than fixed.
"""
from __future__ import annotations

import numpy as np

# LEAP palm frame, measured (see module docstring). Indices into the palm rotation's
# columns, with a sign, for each of FRoGGeR's three semantic directions.
# DO NOT reintroduce a fixed "separation axis" constant here.
#
# An earlier version assumed the thumb-to-index closing direction was a fixed palm
# axis (measured +x at the home pose). It is not: the direction SWINGS with the
# preshape, because the thumb's four joints move it across the palm as the span
# changes. Measured in the palm frame, thumb-index direction by target span:
#
#     30 mm -> [ 0.947,  0.191,  0.260]      100 mm -> [-0.644, 0.645, 0.411]
#     50 mm -> [ 0.947,  0.191,  0.260]      150 mm -> [-0.784, 0.592, 0.187]
#     72 mm -> [-0.389,  0.645,  0.658]      200 mm -> [-0.839, 0.541, 0.058]
#
# i.e. it rotates by more than 90 degrees across the useful range. Aligning the
# palm using the home-pose value put the achieved closing direction 67 degrees off
# the sampled OBB axis (|dot| = 0.389 against a target of 1.0), which is why the
# fingers never straddled however the standoff was corrected.
#
# `palm_frame_for_preshape` below MEASURES the frame from the preshape instead.

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
                     min_height_axis=None, obb=None, hand_frame=None):
    """One palm pose from the heuristic sampler. Steps 1-4 of App. B-C.

    Returns a dict with
        R_WP      (3,3) desired palm rotation, columns = palm x/y/z in world
        p_WP      (3,)  desired palm origin in world
        width     float the chosen OBB edge's full width -- the initial finger
                        separation their step 1 fixes
        sep_axis  (3,)  the separation direction actually used (post-noise)
        approach  (3,)  the approach direction actually used (post-noise)

    hand_frame : (sep_hat, mid, out_hat) measured from the preshape the caller will
        apply -- see `palm_frame_for_preshape`. REQUIRED: the hand's closing
        direction is preshape-dependent on this hand, so it cannot be assumed.

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

    # (4) palm origin, and the palm rotation.
    #
    # Both come from the PRESHAPE's measured geometry, not from a fixed axis
    # convention: `hand_frame` supplies (sep_hat, mid) in the palm frame for the
    # posture the caller is about to use. We then solve for the palm rotation R
    # that maps the hand's own closing direction onto the sampled OBB axis:
    #
    #     R @ sep_hat  == sep        (close along the chosen box edge)
    #     R @ out_hat  == app        (approach along the chosen face normal)
    #
    # which is a standard frame-to-frame alignment, built by putting both triads in
    # matrix form and composing. Without it the achieved closing direction sits up
    # to 67 degrees off the request (measured |dot| = 0.389), and no standoff
    # correction can make the fingers straddle.
    if hand_frame is None:
        raise ValueError("sample_palm_pose needs hand_frame=(sep_hat, mid, out_hat) "
                         "measured from the preshape; see palm_frame_for_preshape")
    sep_hat, mid_P, out_hat = hand_frame
    sep_hat = np.asarray(sep_hat, float); sep_hat /= np.linalg.norm(sep_hat) + 1e-12
    out_hat = np.asarray(out_hat, float)
    out_hat = out_hat - out_hat.dot(sep_hat) * sep_hat
    out_hat /= np.linalg.norm(out_hat) + 1e-12
    H = np.column_stack([sep_hat, out_hat, np.cross(sep_hat, out_hat)])
    Wf = np.column_stack([sep, app, np.cross(sep, app)])
    R = Wf @ H.T
    U, _, Vt2 = np.linalg.svd(R)
    R = U @ Vt2
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0

    # Place the PINCH -- not the palm origin -- at the standoff distance. FRoGGeR
    # measures 4 cm to the palm, which suits the Allegro; LEAP's thumb-index
    # midpoint sits 53-112 mm out from the palm frame (varying with span), so a
    # palm-referenced standoff put the fingertips 155 mm past the object.
    reach = float(np.abs(np.asarray(half) @ np.abs(A.T @ app)))
    p_pinch = c + app * (reach + standoff)
    p_WP = p_pinch - R @ np.asarray(mid_P, float)

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


# LEAP hand joint layout (qpos indices within the 23-DOF robot block), measured
# from the model rather than assumed. Each finger is 4-DOF; there is no single
# "span" joint, which is why the preshape below is solved numerically.
FINGER_QPOS = {
    "index":  (7, 8, 9, 10),      # mcp, rot, pip, dip
    "middle": (11, 12, 13, 14),
    "ring":   (15, 16, 17, 18),
    "thumb":  (19, 20, 21, 22),   # cmc, axl, mcp, ipl
}

# Measured reachable thumb-index fingertip separation on this hand, sampling the
# ten flexion DOFs over their full joint ranges (4000 draws): 8.9 mm to 283.8 mm.
# Every OBB edge in the YCB tabletop set falls inside that, so the clamp below only
# guards genuinely degenerate widths.
SEPARATION_RANGE_M = (0.012, 0.275)


def preshape_finger_separation(width_m):
    """Their step 1 target: the initial thumb-to-opposing-fingertip separation, set
    from the chosen OBB edge's width.

    Clamped to what this hand can actually span (SEPARATION_RANGE_M). An
    unreachably wide preshape is worse than a merely wide one: it saturates every
    finger joint at its limit, leaving the solve no gradient.
    """
    lo, hi = SEPARATION_RANGE_M
    return float(np.clip(float(width_m), lo, hi))


def solve_preshape(model, q, target_sep_m, roles=("thumb", "index"),
                   iters=60, tol=1e-3, rng=None):
    """Set the hand joints so the two named fingertips sit `target_sep_m` apart.

    This is the half of their step 1 that the palm pose does not carry: "use the
    width of this box edge to fix an initial guess for the separation of the hand's
    fingers". Without it the palm arrives correctly aligned but the fingers stay in
    whatever posture q came with, so they land on the SAME side of the object --
    measured straddle dot ~+0.3 to +0.6 rather than negative, which is exactly the
    same-side start that leaves beta with no gradient toward opposition.

    Solved numerically because the LEAP hand exposes no span DOF: each finger is
    4-DOF (mcp/rot/pip/dip, or cmc/axl/mcp/ipl for the thumb) and separation is a
    nonlinear function of all eight. A damped 1-D secant on a single scalar
    "closure" parameter is enough -- the map from closure to separation is monotone
    over the useful range -- and is far cheaper than a full IK.

    The closure parameter drives the primary FLEXION joints of both fingers from
    their extended limit toward their flexed limit; the abduction/rotation DOFs are
    left at q's values, so a caller that has set them deliberately keeps them.

    Returns a new q (copy). Never raises: on failure the input q is returned
    unchanged, since a poor preshape is a worse seed, not a broken one.
    """
    import mujoco as mj
    from kinova_common.constants import FINGER_TIP_SITES

    q = np.asarray(q, float).copy()
    try:
        a, b = roles[0], roles[1]
        ja = FINGER_QPOS[a]
        jb = FINGER_QPOS[b]
        sa = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[a])
        sb = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[b])
        if sa < 0 or sb < 0:
            return q
    except Exception:
        return q

    # Flexion joints only: index 0 (mcp/cmc) and 2 (pip/mcp) of each finger's tuple.
    flex = [ja[0], ja[2], jb[0], jb[2]]
    lim = {}
    for adr in flex:
        j = next((jj for jj in range(model.njnt)
                  if model.jnt_qposadr[jj] == adr), None)
        if j is None:
            return q
        lim[adr] = (float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1]))

    d = mj.MjData(model)

    def sep_at(u):
        """Separation with every flexion joint at fraction u of its range."""
        qq = q.copy()
        for adr in flex:
            lo, hi = lim[adr]
            qq[adr] = lo + float(np.clip(u, 0.0, 1.0)) * (hi - lo)
        d.qpos[:] = 0.0
        d.qpos[:len(qq)] = qq
        mj.mj_kinematics(model, d)
        return float(np.linalg.norm(d.site_xpos[sa] - d.site_xpos[sb])), qq

    target = preshape_finger_separation(target_sep_m)
    # Bracket: u = 0 is fully extended (widest), u = 1 fully flexed (narrowest).
    u_lo, u_hi = 0.0, 1.0
    s_lo, _ = sep_at(u_lo)
    s_hi, _ = sep_at(u_hi)
    if not (min(s_lo, s_hi) - 1e-6 <= target <= max(s_lo, s_hi) + 1e-6):
        # Target outside what this posture can reach; take the closer endpoint
        # rather than returning an unrelated q.
        _, qq = sep_at(0.0 if abs(s_lo - target) < abs(s_hi - target) else 1.0)
        return qq
    # Bisection: monotone in u over this range, and robust where secant is not.
    best = None
    for _ in range(int(iters)):
        u = 0.5 * (u_lo + u_hi)
        s, qq = sep_at(u)
        best = qq
        if abs(s - target) < tol:
            break
        # s decreases as u increases (more flexion -> tips closer).
        if s > target:
            u_lo = u
        else:
            u_hi = u
    return best if best is not None else q


def palm_frame_for_preshape(model, q, roles=("thumb", "index"),
                            palm_body="leap_palm"):
    """The hand's own grasp frame at posture q, expressed in the PALM frame.

    Returns (sep_hat, mid, out_hat):
      sep_hat  unit thumb -> opposing-finger direction. THIS IS NOT A FIXED PALM
               AXIS. Measured across spans it rotates by more than 90 degrees
               (30 mm: [0.947, 0.191, 0.260]; 200 mm: [-0.839, 0.541, 0.058]),
               because the thumb's joints carry it across the palm as the span
               opens. Assuming the home-pose value left the achieved closing
               direction 67 degrees off the sampled OBB axis.
      mid      thumb-opposing-finger midpoint, i.e. where the pinch actually is.
               53-112 mm from the palm origin on this hand, which is why the
               standoff is referenced to this point rather than to the palm.
      out_hat  the approach direction: from the palm origin toward the pinch,
               orthogonalized against sep_hat. This is the direction that should
               face the object, and it is derived rather than assumed for the same
               reason sep_hat is.

    `sample_palm_pose` consumes this as `hand_frame`, solving for the palm rotation
    that carries (sep_hat, out_hat) onto the sampled (OBB axis, face normal).
    """
    import mujoco as mj
    from kinova_common.constants import FINGER_TIP_SITES
    pb = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, palm_body)
    if pb < 0:
        return None
    d = mj.MjData(model)
    d.qpos[:] = 0.0
    q = np.asarray(q, float)
    d.qpos[:len(q)] = q
    mj.mj_kinematics(model, d)
    P = d.xpos[pb].copy()
    R = d.xmat[pb].reshape(3, 3).copy()
    tips = []
    for r in roles[:2]:
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[r])
        if sid < 0:
            return None
        tips.append(d.site_xpos[sid].copy())
    sep = R.T @ (tips[0] - tips[1])
    n = np.linalg.norm(sep)
    if n < 1e-9:
        return None
    sep /= n
    mid = R.T @ (0.5 * (tips[0] + tips[1]) - P)
    out = mid - mid.dot(sep) * sep
    n_out = np.linalg.norm(out)
    if n_out < 1e-9:
        # Pinch sits on the separation axis through the palm origin; fall back to
        # any perpendicular so the frame stays well defined.
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(tmp.dot(sep)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        out = tmp - tmp.dot(sep) * sep
        n_out = np.linalg.norm(out)
    return sep, mid, out / n_out

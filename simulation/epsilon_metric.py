"""Ferrari-Canny epsilon: the largest wrench ball the grasp can resist.

FRoGGeR's Table I reports `epsilon (x1e3)` alongside the normalized min-weight
metric `l_bar*`, and the two are NOT the same quantity. `l_bar*` is a RELAXATION of
epsilon -- it is cheap, differentiable, and the thing their NLP maximizes; epsilon is
the classical Ferrari-Canny quality they SCORE with. Reporting only `l_bar*` compares
each arm against the objective one of them was built to maximize, which is the
comparison FRoGGeR's own table avoids by reporting both.

DEFINITION. Let the grasp wrench set be

    W = ConvexHull( Minkowski-sum over contacts of each pyramidal wrench cone )

in 6D `[Tx,Ty,Tz, Fx,Fy,Fz]`. Then

    epsilon = min over faces of W of the face's signed distance to the ORIGIN

i.e. the radius of the largest origin-centred ball contained in W. `epsilon > 0` iff
the grasp is force closure. This is the L2 (Euclidean) ball, which is the standard
Ferrari-Canny reading and what FRoGGeR reports.

WHY IT REUSES `composite_wrench_cone`'s MACHINERY RATHER THAN ITS OUTPUT. That
function returns the FORCE and TORQUE hulls SEPARATELY -- two 3D projections of the
6D set. epsilon is not recoverable from projections: a set can project to a wide
force hull and a wide torque hull while being thin in a mixed force-torque direction,
and epsilon is exactly the radius in the worst such direction. So this module builds
the same per-contact cones through the same `_ncf.WrenchCheck.single_wrench_cone`
call and hulls the 6D sum directly. The cone geometry, the inward->outward normal
flip and the `gamma` scaling therefore cannot drift from the certificate's.

UNITS. Force in N and torque in N*m are summed in one Euclidean norm, so epsilon
carries the usual Ferrari-Canny units ambiguity: rescaling the object changes the
torque rows' magnitude relative to the force rows. FRoGGeR inherits this too (their
Table I quotes bare `epsilon x 1e3`), so the number is comparable BETWEEN ARMS on the
same object -- which is the comparison being run -- and should not be read as an
absolute grasp quality across objects of different size. `torque_scale` divides the
torque rows by a characteristic length when a dimensionless value is wanted; it
defaults to 1.0, the paper's convention.

GAMMA. epsilon scales LINEARLY in the internal-force scale `gamma` (every cone vertex
does), so a fair between-arm comparison must hold it fixed. `gamma=1.0` is the
default and the right choice for scoring: it reports the grasp's GEOMETRY, not the
squeeze force the executor happened to solve for. Passing each arm's own solved gamma
would confound grasp quality with how hard that arm decided to squeeze.
"""
from __future__ import annotations

import numpy as np

from kinova_common.wrench import _ncf


def _hull_offsets(pts, qhull_opts="Qx"):
    """Signed distances from the origin to every facet of conv(pts), in 6D.

    scipy's ConvexHull gives facets as `equations` = [normal | offset] with
    `normal . x + offset <= 0` inside, and `normal` unit. The origin's signed
    distance to a facet is therefore `-offset`, positive when the origin is on the
    interior side. Returns None when the point set is degenerate (rank < 6), which
    is the honest answer: a grasp whose wrench set does not span all six dimensions
    contains NO origin-centred ball, so epsilon is not merely small but undefined
    by this construction. Callers report that as non-closure.
    """
    from scipy.spatial import ConvexHull

    pts = np.asarray(pts, float)
    if pts.shape[0] < pts.shape[1] + 1:
        return None
    # Rank test BEFORE calling qhull: a rank-deficient set makes qhull raise, and
    # distinguishing "degenerate" from "qhull failed" matters for the caller.
    s = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
    if int(np.count_nonzero(s > 1e-9 * max(s[0], 1e-30))) < pts.shape[1]:
        return None
    try:
        h = ConvexHull(pts, qhull_options=qhull_opts)
    except Exception:
        return None
    return -h.equations[:, -1]


def grasp_wrench_set(p_O, R_O_inward, mu, gamma=1.0, torque_scale=1.0):
    """The 6D grasp wrench set's vertices, `[Tx,Ty,Tz, Fx,Fy,Fz]` per row.

    Conventions follow `solve_gamma_live` / `composite_wrench_cone` exactly:
    `R_O_inward` has col0 = INWARD normal and is flipped to the outward normal the
    NCF cone is built with. Each contact contributes 5 vertices (the origin plus 4
    pyramid edges -- the origin is legitimate HERE, where "this contact carries no
    force" is a real mode of the wrench set, unlike in the min-weight LP where it
    admits a degenerate optimum; see FROGGER_COMPARISON sec 7.1).

    The Minkowski sum is 5^n points: 25 at n=2, 125 at n=3. Small enough to hull
    directly.
    """
    n = len(p_O)
    R_out = [np.asarray(R, float).copy() for R in R_O_inward]
    for R in R_out:
        R[:, 0] *= -1.0                                  # inward -> outward
    pos = [np.asarray(p, float).reshape(3, 1) for p in p_O]
    wc = _ncf.WrenchCheck(n, pos, R_out, [1.0] * n, [0.0] * n, [0.0] * n, list(mu))
    per_contact = [np.asarray(wc.single_wrench_cone(gamma, pos[i], R_out[i],
                                                    1.0, mu[i]), float)
                   for i in range(n)]
    # Minkowski sum over one vertex per contact.
    import itertools
    W = np.array([np.sum(c, axis=0) for c in itertools.product(*per_contact)], float)
    if torque_scale != 1.0:
        W = W.copy()
        W[:, 0:3] /= float(torque_scale)
    return W


def epsilon_quality(p_O, R_O_inward, mu, gamma=1.0, torque_scale=1.0):
    """Ferrari-Canny epsilon for one grasp. Returns a dict, never raises.

    Args mirror `solve_gamma_live`: contact positions and inward-normal frames in
    the OBJECT body frame, per-contact friction. See the module docstring on
    `gamma` and `torque_scale`.

    Returns:
        epsilon        float; the largest origin-centred ball radius inside the
                       grasp wrench set. <= 0 means the origin is on or outside a
                       facet, i.e. NOT force closure. None when the set is
                       rank-deficient (also not closure -- see `degenerate`).
        force_closure  bool; epsilon is not None and > 0.
        degenerate     bool; the wrench set does not span 6D. At n=2 this is the
                       EXPECTED outcome, not a fault: a two-contact pinch has a
                       rank-5-of-6 wrench matrix (it resists no torque about the
                       line through its contacts), so its wrench set is flat in
                       that direction and contains no 6-ball. This is the same
                       structural fact that makes `project_grasp_axis_torque`
                       necessary in the gamma certificate, and it is why the
                       paper's epsilon column is a FOUR-finger number.
        n_contacts     int, for labelling.
    """
    out = dict(epsilon=None, force_closure=False, degenerate=False,
               n_contacts=len(p_O))
    try:
        W = grasp_wrench_set(p_O, R_O_inward, mu, gamma=gamma,
                             torque_scale=torque_scale)
    except Exception as e:
        out["error"] = repr(e)
        return out
    d = _hull_offsets(W)
    if d is None:
        out["degenerate"] = True
        return out
    eps = float(np.min(d))
    out["epsilon"] = eps
    out["force_closure"] = bool(eps > 0.0)
    return out


def epsilon_subspace(p_O, R_O_inward, mu, gamma=1.0, which="force"):
    """epsilon restricted to the FORCE or TORQUE subspace.

    A fallback for the n=2 case, where the full 6D epsilon is structurally
    degenerate (see `epsilon_quality`). The force-subspace radius is still a
    meaningful, comparable quality number -- it asks how large a pure-force
    disturbance the pinch resists in the worst direction -- and it is defined
    wherever the 3D projection is full rank.

    Reported SEPARATELY from `epsilon` and never as a substitute: a 3D subspace
    radius is not the Ferrari-Canny quantity and is not comparable to the paper's
    column.

    SATURATION -- measured, and the reason this number is nearly useless at this
    scene's friction. On an antipodal pinch the NORMAL direction is bounded by the
    normal force itself (two unit contacts at gamma=1 reach +-1 and no further),
    while the tangential extent grows as mu. Once mu >= 1 the normal axis becomes
    the binding face and the radius CLAMPS:

        mu    force-subspace epsilon
        0.5   0.667
        1.0   1.000
        2.0   1.000
        4.0   1.000

    So at `table_scene`'s mu = 2.0 this returns exactly 1.0 for every pinch,
    carrying no information about grasp geometry. It only discriminates below
    mu = 1, e.g. in FRoGGeR's own mu = 0.7 regime. Do not rank grasps by it at the
    default friction.
    """
    cols = slice(3, 6) if which == "force" else slice(0, 3)
    try:
        W = grasp_wrench_set(p_O, R_O_inward, mu, gamma=gamma)[:, cols]
    except Exception:
        return None
    d = _hull_offsets(W)
    return None if d is None else float(np.min(d))

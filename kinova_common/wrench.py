"""Grasp wrench-cone geometry helpers shared by the pick-and-place control loop and
diagnostic tooling: live gamma (internal-force margin) solves and the composite
wrench-cone hull used for dashboard visualization. Extracted verbatim from
kinova_leap_pick_place.py.
"""
import numpy as np

# 3D_minimum_NCF.py isn't an importable module name (leading digit), so load it by path.
import importlib.util as _ilu
from pathlib import Path as _Path

_ncf_spec = _ilu.spec_from_file_location(
    '_ncf', str(_Path(__file__).resolve().parents[1] / 'scripts' / '3D_minimum_NCF.py'))
_ncf = _ilu.module_from_spec(_ncf_spec)
_ncf_spec.loader.exec_module(_ncf)


def solve_gamma_live(p_O, R_O_inward, mu, mass, accel_box_xyz, ang_accel_box_xyz,
                     inertia_diag, grav_O=None):
    """Minimum internal-force scale gamma that keeps the grasp no-slip for the given
    acceleration/torque disturbance box, from the live grasp geometry. Wraps
    3D_minimum_NCF.min_gamma_for_accel_lp; verified against its native antipodal cases
    (see scratchpad/verify_gamma_solve.py). Handles the two convention mismatches:
      * normal sign: spatial_grasp_map / GraspController give col0 = INWARD normal;
        the NCF cone is built with col0 = OUTWARD (force pushing ON the object) -> flip.
      * unit mass: min_gamma_for_accel_lp assumes m=1, so the "accel box" is really a
        FORCE box (m*a) and the "torque box" a real torque (I*alpha) -> scale here.

    Task definition (grav_O given -> Task B / datum; grav_O None -> Task A / CoM):
      * grav_O is None (legacy CoM mode): the caller must have folded gravity INTO
        accel_box_xyz, and the whole force box is referenced about the CoM. A raised
        or off-center pinch is then correctly reported infeasible for a lateral force.
      * grav_O given (datum mode): accel_box_xyz is the PURE accel budget (no gravity);
        the disturbance is referenced about the GRASP MIDPOINT (moment_ref), so a raised
        symmetric grasp can resist a lateral disturbance at the contact interface; and
        gravity is passed as grav_force with its grasp-axis moment projected out
        (project_grasp_axis_moment) so residual off-CoM drift doesn't spuriously break
        feasibility. This is the hold/transport formulation (see
        RAISED_CONTACT_WRENCH_FINDINGS.md sec 5). Pair it with an angular budget whose
        grasp-axis component has been zeroed (the caller already does that).

    Args:
        p_O:            list of (3,) contact positions in the OBJECT body frame.
        R_O_inward:     list of (3,3) contact->object rotations, col0 = inward normal.
        mu:             list of per-contact friction coefficients.
        mass:           object mass (kg).
        accel_box_xyz:  (ax,ay,az) linear-accel budget, object-body axes. Includes
                        gravity in CoM mode; PURE accel (gravity separate) in datum mode.
        ang_accel_box_xyz: (alpha_x,alpha_y,alpha_z) angular-accel budget, PRINCIPAL axes.
        inertia_diag:   (Ix,Iy,Iz) principal moments (model.body_inertia). Multiplies the
                        angular-accel budget into a torque box.
        grav_O:         (3,) gravitational acceleration in the OBJECT frame (R_WO.T @ g).
                        When given, switches to the datum/Task-B formulation above.

    Returns:
        gamma (float), or None if the grasp geometrically cannot resist the box.
    """
    n = len(p_O)
    R_out = [R.copy() for R in R_O_inward]
    for R in R_out:
        R[:, 0] *= -1.0                                   # inward -> outward normal
    pos = [np.asarray(p, float).reshape(3, 1) for p in p_O]
    fx, fy, fz = (mass * a for a in accel_box_xyz)        # force box = m * a
    tx, ty, tz = (I * al for I, al in zip(inertia_diag, ang_accel_box_xyz))  # torque = I*alpha
    if grav_O is not None and n == 2:
        # Datum / Task-B formulation: reference the disturbance about the grasp midpoint,
        # add gravity as an explicit re-datumed wrench, and project its grasp-axis moment.
        _mref = (0.5 * (pos[0] + pos[1])).reshape(3)
        _grav = mass * np.asarray(grav_O, float)          # gravity force, object frame
        return _ncf.min_gamma_for_accel_lp(
            fx, fy, fz, tx, ty, tz, n, pos, R_out, [1.0] * n,
            [0.0] * n, [0.0] * n, list(mu),
            moment_ref=_mref, grav_force=_grav,
            project_grasp_axis_moment=True,
            project_grasp_axis_torque=True)
    return _ncf.min_gamma_for_accel_lp(
        fx, fy, fz, tx, ty, tz, n, pos, R_out, [1.0] * n,
        [0.0] * n, [0.0] * n, list(mu))


def _reindex_hull(hull):
    """ConvexHull.simplices index the ORIGINAL points; remap to hull.vertices order."""
    old_to_new = {old: new for new, old in enumerate(hull.vertices)}
    return np.array([[old_to_new[i] for i in s] for s in hull.simplices], np.int32)


def _flat_hull(pts3, center):
    """Coplanar point set -> filled 2D polygon as a centroid triangle fan in 3D.
    Used when a wrench subspace collapses to a plane (e.g. the antipodal grasp's
    torque cone lives in Tx=0: a pinch resists no torque about the grasp axis)."""
    from scipy.spatial import ConvexHull
    _, _, vt = np.linalg.svd(pts3 - center)
    p2 = (pts3 - center) @ vt[:2].T
    ring = ConvexHull(p2).vertices
    verts = np.vstack([center, pts3[ring]]).astype(np.float32)
    m = len(ring)
    faces = np.array([[0, 1 + i, 1 + (i + 1) % m] for i in range(m)], np.int32)
    return {'verts': verts, 'faces': faces}


def hull3d(pts3):
    """Rank-aware convex hull: point (rank<2) -> None, coplanar (rank==2) -> _flat_hull,
    else a real 3D scipy ConvexHull re-expressed as {'verts','faces'} with faces
    reindexed to hull.vertices order. Shared by composite_wrench_cone and any other
    caller needing a degenerate-safe 3D hull for a wrench-cone-shaped point cloud."""
    from scipy.spatial import ConvexHull
    pts3 = np.asarray(pts3, float)
    c = pts3.mean(0)
    s = np.linalg.svd(pts3 - c, compute_uv=False)
    rank = int(np.count_nonzero(s > 1e-9 * (s[0] if s[0] > 0 else 1)))
    if rank < 2:
        return None
    if rank == 2:
        return _flat_hull(pts3, c)
    h = ConvexHull(pts3)
    return {'verts': pts3[h.vertices].astype(np.float32), 'faces': _reindex_hull(h)}


def composite_wrench_cone(gamma, p_O, R_O_inward, mu):
    """Force- and torque-subspace hulls of the composite grasp wrench cone at scale
    gamma: the Minkowski sum of each contact's pyramidal wrench cone — the exact set
    3D_minimum_NCF's LP tests wrench membership in. Returns
        {'force': {'verts','faces'} | None, 'torque': {'verts','faces'} | None}
    for the dashboard's 3D panels (None when a subspace is a point/line). Convention
    matches solve_gamma_live (col0 inward->outward). single_wrench_cone lays out each
    vertex as [Tx,Ty,Tz, Fx,Fy,Fz]."""
    import itertools
    n = len(p_O)
    R_out = [R.copy() for R in R_O_inward]
    for R in R_out:
        R[:, 0] *= -1.0
    wc = _ncf.WrenchCheck(n, [np.asarray(p).reshape(3, 1) for p in p_O],
                          R_out, [1.0] * n, [0.0] * n, [0.0] * n, list(mu))
    per_contact = [wc.single_wrench_cone(gamma, np.asarray(p_O[i]).reshape(3, 1),
                                         R_out[i], 1.0, mu[i]) for i in range(n)]
    # Minkowski sum: sum one vertex per contact over all combinations (nverts^n points).
    W = np.array([sum(c) for c in itertools.product(*per_contact)])

    return {'force': hull3d(W[:, 3:6]), 'torque': hull3d(W[:, 0:3])}

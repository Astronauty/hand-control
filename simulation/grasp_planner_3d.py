"""
grasp_planner_3d.py
===================
3D grasp contact-point solver — IPOPT formulation (Kinova Gen3 + LEAP hand).

Seeding (MultiStartGraspPlanner3D)
-----------------------------------
    _seed_pair generates one candidate per call:
      1. Sphere-trace a random direction to surface point p1s.
      2. Perturb the inward normal by up to delta_max and sphere-march
         through the object to find the antipodal footprint p2s.
      3. Use p1s/p2s directly as the NLP warm-start (p1, p2) — no
         off-surface jitter; both land exactly on the surface.
    Contact frames are frozen from the surface normals at p1s/p2s.
    ~20 seeds per call; no filtering or sorting.

    Pre-check LP: min_gamma_for_accel_lp on (p1s, p2s) before building the NLP,
    slack-relaxed (cfg.precheck_slack_penalty) so one geometrically-unreachable
    corner doesn't report the whole seed infeasible. Seeds with γ > 50 are skipped.

    Post-solve: wrench_ok = (status == 'converged') AND (max_slack_norm <
    cfg.slack_tol_abs) — the embedded wrench LP (see below) is slack-relaxed, so
    solver convergence alone no longer certifies exact wrench feasibility; the
    slack magnitude must also be negligible. Results are ranked with wrench_ok
    candidates first.

Decision variables
------------------
    q        (nu,)          actuated joint angles (7 Kinova + 16 LEAP)
    p1       (3,)           thumb contact point, world frame
    p2       (3,)           index-finger contact point, world frame
    γ        scalar         wrench quality margin (minimized; smaller = better geometry)
    y1_k,y2_k (5,) each    friction-cone mixing weights per contact per load corner k
    s_k      (6,) each    wrench-balance slack per load corner k (penalized, not minimized to 0)

    Wrench feasibility is enforced as a slack-relaxed equality per load corner k
    (jointly sign-expanded over torque AND force, in the object body frame),
    matching min_gamma_for_accel_lp's slack_penalty mode
    (scripts/3D_minimum_NCF.py) exactly — see _embed_wrench_cone_ca. A single
    unreachable corner (e.g. near-zero moment arm about one torque axis) no
    longer makes the whole NLP infeasible; it shows up as nonzero slack on that
    corner instead, penalized in the cost and checked post-solve via wrench_ok.

Cost (all terms normalized — each ≈ 1 at its reference level, weights are pure priorities)
------------------------------------------------------------------------------------------
    w_ik    * 0.5*(‖Δp1‖²+‖Δp2‖²) / d_ref²          d_ref = 5mm
    + w_reg   * ‖(q−q_neutral)/q_scale‖² / n_dof
    + w_gamma * γ / g_ref                              g_ref = ‖task_force‖ N
    + w_y     * Σ_k (‖y1_k‖²+‖y2_k‖²) / (n_c·10·g_ref²)
    + w_slack * Σ_k ‖s_k / ref6‖² / n_c                 ref6 = per-row [t_ref×3, g_ref×3]

    IK target includes fingertip radius offset so the tip sphere surface
    touches the contact point (r_thumb/r_index measured from model geom size).

Constraints
-----------
    1. Joint limits    — opti.bounded(lo, q, hi)
    2. Surface contact — linear face-pin (BOX); analytic equality (sphere/cylinder)
    3. Wrench LP       — embedded: slack-relaxed equality per corner over the
                         full [Tx,Ty,Tz,Fx,Fy,Fz] wrench (w + s_k == w_k);
                         slack penalized in the cost, not hard-bounded to 0
    4. Arm collision   — proximity-pruned softplus SDFs (constrained_ik)

Contact frame convention
------------------------
    R = opti.parameter(3,3): [n_in | t1 | t2]
    R[:,0] = inward normal  (compressive, into object)
    Frozen from surface footprint normals for the single NLP solve


Wrench convention (consistent with 3D_minimum_NCF.py)
------------------------------------------------------
    [Tx, Ty, Tz, Fx, Fy, Fz]  (torque first)
"""

from __future__ import annotations

import sys
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# 3D wrench check lives in scripts/ (module name starts with digit — use importlib)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    import importlib as _importlib
    _ncf_mod = _importlib.import_module('3D_minimum_NCF_slack')
    min_gamma_for_accel_lp = _ncf_mod.min_gamma_for_accel_lp
    _NCF_AVAILABLE = True

except Exception as e:
    min_gamma_for_accel_lp = None
    _NCF_AVAILABLE = False

# The DATUM/Task-B certificate uses the HARD LP from 3D_minimum_NCF (the SAME function
# kinova_leap_pick_place.solve_gamma_live calls at grasp time), so the recommender's
# feasibility flag is byte-for-byte the grasp-time definition — no slack masking. That
# module carries the moment_ref / grav_force / project_grasp_axis_moment datum params;
# the slack module above does not. Kept as a separate handle so the legacy CoM path is
# unchanged.
try:
    _ncf_hard_mod = _importlib.import_module('3D_minimum_NCF')
    min_gamma_for_accel_lp_hard = _ncf_hard_mod.min_gamma_for_accel_lp
except Exception:
    min_gamma_for_accel_lp_hard = None

try:
    import casadi as ca
    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False

try:
    import mujoco as mj
    _MJ_AVAILABLE = True
except ImportError:
    _MJ_AVAILABLE = False

try:
    from grasp_control import SpatialIKSolver
    from grasp_control.constrained_ik import (
        _SitePositionCallbackAnalytic,
        _SiteAxisCallbackAnalytic,
        _BatchedGeomPositionCallbackAnalytic,
        _softplus_sphere_box_distance,
        _softplus_sphere_cylinder_distance,
        _sphere_plane_distance,
        _sphere_sphere_distance,
    )
    _CIK_AVAILABLE = True
except ImportError:
    _CIK_AVAILABLE = False

log = logging.getLogger("grasp_planner_3d")
if not log.handlers:
    log.addHandler(logging.NullHandler())

if _NCF_AVAILABLE and _CASADI_AVAILABLE and _MJ_AVAILABLE and _CIK_AVAILABLE:
    log.info("all dependencies available")
else:
    log.warning("some dependencies are missing; grasp planning will fail if invoked")
# ─────────────────────────────────────────────────────────────────────────────
# SQP solver options — copied verbatim from constrained_ik._SQP_SOLVER_OPTS
# (private name; do not import — copy the dict to avoid coupling)
# ─────────────────────────────────────────────────────────────────────────────

_SQP_SOLVER_OPTS = {
    'verbose':               False,
    'print_time':            True,
    'qpsol':                 'osqp',
    'qpsol_options':         {'error_on_fail': True,
                              'verbose': False,
                              'print_problem': False,
                              'osqp': {'verbose': False, 'polish': False}},
    'max_iter':              500,
    'hessian_approximation': 'limited-memory',
    'lbfgs_memory':          20,
    'convexify_strategy':    'regularize',
    'print_iteration':       False,
    'print_header':          False,
    'print_status':          False,
}

_IPOPT_SOLVER_OPTS = {
    'hessian_approximation':      'limited-memory',
    'limited_memory_max_history': 20,       # more curvature pairs for near-singular reduced space
    'max_iter':                   500,
    'sb':                         'no',
    'print_level':                0,
    'mu_strategy':                'adaptive',
    # Accept flat-objective convergence — the dual residual is non-convergent when the
    # active set is degenerate (minimax γ with antipodal symmetry).
    'acceptable_tol':             1e4,     # effectively disabled — dominated by dual inf
    'acceptable_constr_viol_tol': 1e-6,    # the real feasibility test
    'acceptable_compl_inf_tol':   1e2,
    'acceptable_dual_inf_tol':    1e3,
    'acceptable_obj_change_tol':  1e-2,    # ← the criterion that matters
    'acceptable_iter':            4,
    'nlp_scaling_method':         'gradient-based',
}   


# ─────────────────────────────────────────────────────────────────────────────
# Geometry primitives
# ─────────────────────────────────────────────────────────────────────────────

def _box_sdf_3d(point, center, hx: float, hy: float, hz: float) -> float:
    """Signed distance from point to 3D axis-aligned box. Negative = inside."""
    d = np.asarray(point, float) - np.asarray(center, float)
    q = np.array([abs(d[0]) - hx, abs(d[1]) - hy, abs(d[2]) - hz])
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))


def _box_surface_normal_3d(point, center, hx: float, hy: float, hz: float) -> np.ndarray:
    """Outward unit normal at nearest face of a 3D axis-aligned box."""
    d = np.asarray(point, float) - np.asarray(center, float)
    nx = abs(d[0]) / hx
    ny = abs(d[1]) / hy
    nz = abs(d[2]) / hz
    if nx >= ny and nx >= nz:
        return np.array([np.sign(d[0]), 0.0, 0.0])
    elif ny >= nz:
        return np.array([0.0, np.sign(d[1]), 0.0])
    else:
        return np.array([0.0, 0.0, np.sign(d[2])])


def _build_contact_frame_3d(inward_normal: np.ndarray):
    """
    Build right-handed orthonormal frame (n, t1, t2) from an inward contact normal.

    Returns
    -------
    n, t1, t2 : np.ndarray (3,) each
    """
    n = np.asarray(inward_normal, float)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("inward_normal is degenerate (near-zero norm)")
    n = n / norm
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref);  t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1);   t2 /= np.linalg.norm(t2)
    return n, t1, t2


def _symbolic_contact_frame_ca(n_in_sym):
    """
    Build right-handed (n_in, t1, t2) contact frame as a CasADi MX expression.

    Mirrors _build_contact_frame_3d but operates on symbolic MX vectors so the
    frame is part of the NLP's symbolic graph.  CasADi can autodiff through it.

    Parameters
    ----------
    n_in_sym : (3,) CasADi MX — inward contact normal (need not be unit-length;
                normalisation is applied internally).

    Returns
    -------
    R : (3,3) CasADi MX  — columns are [n_in, t1, t2]
    """
    eps = 1e-12
    n = n_in_sym / (ca.norm_2(n_in_sym) + eps)
    # Choose reference that is not parallel to n
    e0 = ca.DM([1.0, 0.0, 0.0])
    e1 = ca.DM([0.0, 1.0, 0.0])
    # Blend: use e1 when |n[0]| >= 0.9, else e0
    w = ca.fabs(n[0]) - 0.9
    alpha = 0.5 * (1.0 + ca.tanh(w / 0.01))   # smooth 0→1 as |n[0]| crosses 0.9
    ref = (1.0 - alpha) * e0 + alpha * e1
    t1 = ca.cross(n, ref);  t1 = t1 / (ca.norm_2(t1) + eps)
    t2 = ca.cross(n, t1);   t2 = t2 / (ca.norm_2(t2) + eps)
    return ca.horzcat(n, t1, t2)                # 3×3 MX


def _sym_inward_normal_ca(p, geom_type, center_dm, Rt_dm, size):
    """
    Symbolic inward normal at world-frame point p (CasADi MX).

    Supported geom types:
      2 (SPHERE)   — radial direction
      5 (CYLINDER) — lateral direction (assumes contact on curved surface)
    Box normals are piecewise-constant and not differentiable symbolically;
    use frozen parameters (the default) for geom_type==6.
    """
    eps = 1e-12
    p_loc = Rt_dm @ (p - center_dm)        # object-local frame
    if geom_type == 2:                      # SPHERE
        return -p_loc / (ca.norm_2(p_loc) + eps)
    elif geom_type == 5:                    # CYLINDER (lateral surface)
        r_xy = ca.sqrt(p_loc[0]**2 + p_loc[1]**2 + eps**2)
        n_lat_loc = ca.vertcat(-p_loc[0] / r_xy,
                               -p_loc[1] / r_xy,
                               ca.DM(0.0))
        return n_lat_loc
    raise ValueError(f"symbolic normals not implemented for geom_type={geom_type}")


def _geom_sdf_np(point, geom_type: int, center, mat, size) -> float:
    """Shape-agnostic signed distance (numpy). Positive = outside."""
    p = np.asarray(point, float)
    c = np.asarray(center, float)
    R = np.asarray(mat).reshape(3, 3)
    p_l = R.T @ (p - c)
    if geom_type == 6:   # BOX
        return _box_sdf_3d(p_l, np.zeros(3), size[0], size[1], size[2])
    elif geom_type == 2:  # SPHERE
        return float(np.linalg.norm(p_l) - size[0])
    elif geom_type == 5:  # CYLINDER  radius=size[0], half-height=size[1]
        r_xy = float(np.linalg.norm(p_l[:2])) - size[0]
        r_z  = abs(float(p_l[2])) - size[1]
        return float(np.sqrt(max(r_xy, 0)**2 + max(r_z, 0)**2) + min(max(r_xy, r_z), 0.0))
    return float(np.linalg.norm(p - c))


def _geom_normal_np(point, geom_type: int, center, mat, size) -> np.ndarray:
    """Outward unit surface normal at nearest surface point (numpy)."""
    p = np.asarray(point, float)
    c = np.asarray(center, float)
    R = np.asarray(mat).reshape(3, 3)
    p_l = R.T @ (p - c)
    if geom_type == 6:   # BOX
        # Select the nearest face and its normal in the OBJECT-LOCAL frame (p_l, centered
        # at the origin), then rotate the local normal back to world (R @ n_l) — same as
        # the sphere/cylinder branches. Passing the WORLD point/center picks the face in
        # world axes and returns a world-axis normal, which is only correct for an
        # axis-aligned box (R=I); for a rotated box it is wrong by the box's rotation
        # (a 45deg-yawed box gave a full 45deg normal error), breaking the grasp-axis
        # alignment term on non-world-aligned objects.
        n_l = _box_surface_normal_3d(p_l, np.zeros(3), size[0], size[1], size[2])
        return R @ n_l
    elif geom_type == 2:  # SPHERE
        n_l = p_l / (np.linalg.norm(p_l) + 1e-12)
        return R @ n_l
    elif geom_type == 5:  # CYLINDER
        rxy = np.linalg.norm(p_l[:2])
        if rxy - size[0] >= abs(p_l[2]) - size[1]:
            n_l = np.array([p_l[0] / (rxy + 1e-12), p_l[1] / (rxy + 1e-12), 0.0])
        else:
            n_l = np.array([0.0, 0.0, np.sign(p_l[2])])
        return R @ n_l
    n = p - c
    return n / (np.linalg.norm(n) + 1e-12)


def _angle_deg_between(a, b) -> float:
    """Angle in degrees between two 3-vectors, or nan if either is None."""
    if a is None or b is None:
        return float('nan')
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _span_margin(n1_out: np.ndarray, n2_out: np.ndarray, mu: float) -> float:
    """Positive = force closure geometrically achievable with friction coefficient mu."""
    c = float(np.dot(n1_out / (np.linalg.norm(n1_out) + 1e-12),
                     n2_out / (np.linalg.norm(n2_out) + 1e-12)))
    return float(np.arccos(np.clip(c, -1.0, 1.0)) - (np.pi - 2.0 * np.arctan(mu)))


def _project_to_surface_np(point, geom_type, center, mat, size, iters=12):
    """Move a world-frame point to the nearest surface point.

    BOX: exact closed form — clamp to box in local frame, snap dominant axis to
    its face.  One step, always exact, works from corners and edges where the
    SDF gradient is diagonal and a single Newton step falls short.

    Other geometries: iterated Newton steps (12 iterations converges from
    anywhere within a few bbox radii at negligible cost).
    """
    p = np.asarray(point, float)
    c = np.asarray(center, float)
    R = np.asarray(mat).reshape(3, 3)

    if geom_type == 6:  # BOX — exact
        q = R.T @ (p - c)
        q = np.clip(q, -np.asarray(size, float), np.asarray(size, float))
        k = int(np.argmax(np.abs(q) / (np.asarray(size, float) + 1e-12)))
        q[k] = np.sign(q[k]) * float(size[k])
        return c + R @ q

    for _ in range(iters):
        s = _geom_sdf_np(p, geom_type, c, mat, size)
        if abs(s) < 1e-9:
            break
        p = p - s * _geom_normal_np(p, geom_type, c, mat, size)
    return p


def _march_sdf_np(p_start, direction, geom_type, center, mat, size,
                  max_steps=80):
    """Sphere-march from p_start along direction until exiting the object, then project to surface."""
    d_unit = np.asarray(direction, float)
    d_unit = d_unit / (np.linalg.norm(d_unit) + 1e-12)
    p = np.asarray(p_start, float)
    for _ in range(max_steps):
        sdf = _geom_sdf_np(p, geom_type, center, mat, size)
        if sdf > 1e-5:
            return _project_to_surface_np(p, geom_type, center, mat, size)
        step = max(abs(sdf) * 0.5, 1e-3)
        p = p + step * d_unit
    return _project_to_surface_np(p, geom_type, center, mat, size)


def _seed_pair(geom_type, size, center, obj_mat, bbox_r, rng,
               delta_max=np.deg2rad(45)):
    """
    One antipodal seed pair, sampled exactly on the object surface.

    Sphere-traces to a random surface point p1s, then marches through the
    object along a perturbed antipodal direction to find p2s. Both p1s/p2s
    are exact surface points (no off-surface jitter) — used directly as the
    NLP warm-start p1/p2 as well as the pre-check LP / frame-freezing
    footprints.

    Returns dict with keys:
        p1, p2       — NLP warm-start positions (== p1s/p2s, on surface)
        p1s, p2s     — surface footprints (for pre-check LP and frame freezing)
        n1_in, n2_in — inward normals at surface footprints
        delta_deg    — jitter angle applied to march direction
    """
    c = np.asarray(center, float)
    u = rng.standard_normal(3)
    u[2] *= 0.5                              # bias toward side faces, away from top/bottom
    u /= np.linalg.norm(u) + 1e-12
    p1s = _project_to_surface_np(c + u * bbox_r, geom_type, c, obj_mat, size)
    n1_in = -_geom_normal_np(p1s, geom_type, c, obj_mat, size)

    # Rotate march direction about world z only — prevents downward tilt that
    # exits through the bottom face and hits _reachable_contact rejection.
    ang  = rng.uniform(-delta_max, delta_max)
    ca_, sa_ = np.cos(ang), np.sin(ang)
    Rz = np.array([[ca_, -sa_, 0.0], [sa_, ca_, 0.0], [0.0, 0.0, 1.0]])
    d  = Rz @ n1_in
    delta = abs(ang)

    p2s = _march_sdf_np(p1s + 1e-3 * d, d, geom_type, c, obj_mat, size)
    n2_in = -_geom_normal_np(p2s, geom_type, c, obj_mat, size)

    return {
        'p1':       p1s.copy(),
        'p2':       p2s.copy(),
        'p1s':      p1s,
        'p2s':      p2s,
        'n1_in':    n1_in,
        'n2_in':    n2_in,
        'offsets':  (0.0, 0.0),
        'delta_deg': float(np.rad2deg(delta)),
    }


def _assign_seed_by_finger(seed, live_thumb, live_index):
    """Orient a seed's contact labels to the operator's actual hand: p1/p1s/n1_in is the
    THUMB seed, p2/p2s/n2_in the INDEX seed. _seed_pair labels the two contacts by a random
    march direction, so the thumb/index assignment is a coin flip — the cause of run-to-run
    finger flips and of awkward assignments the pinned-face NLP can't undo. Swap in place iff
    doing so reduces the total (thumb->contact) + (index->contact) distance, i.e. put the
    thumb on whichever contact it actually reaches. Mutates `seed`."""
    p1, p2 = seed['p1s'], seed['p2s']
    lt, li = np.asarray(live_thumb, float), np.asarray(live_index, float)
    d_keep = np.linalg.norm(p1 - lt) + np.linalg.norm(p2 - li)   # p1=thumb, p2=index
    d_swap = np.linalg.norm(p2 - lt) + np.linalg.norm(p1 - li)   # swapped
    if d_swap < d_keep:
        seed['p1'],    seed['p2']    = seed['p2'],    seed['p1']
        seed['p1s'],   seed['p2s']   = seed['p2s'],   seed['p1s']
        seed['n1_in'], seed['n2_in'] = seed['n2_in'], seed['n1_in']


def _fixed_antipodal_seed(geom_type, size, center, obj_mat, local_axis):
    """
    One deterministic, perfectly antipodal seed pair along a fixed axis
    (given in the OBJECT's local frame) through the object center — zero
    angular jitter, unlike _seed_pair's randomized march direction.

    Two of these (local x and local y) are tried first by
    MultiStartGraspPlanner3D.solve(), ahead of the randomized seeds, as a
    canonical "grab from the side" starting point for every object.

    local_axis : (3,) unit vector in the object's local frame, e.g. [1,0,0].

    Returns the same dict schema as _seed_pair (offsets=(0,0), delta_deg=0
    since both contacts land exactly on the surface with no jitter).
    """
    c = np.asarray(center, float)
    d_world = obj_mat @ np.asarray(local_axis, float)
    d_world /= np.linalg.norm(d_world) + 1e-12
    bbox_r = float(np.max(size)) * 2.5

    p1s = _project_to_surface_np(c + d_world * bbox_r, geom_type, c, obj_mat, size)
    n1_in = -_geom_normal_np(p1s, geom_type, c, obj_mat, size)
    p2s = _project_to_surface_np(c - d_world * bbox_r, geom_type, c, obj_mat, size)
    n2_in = -_geom_normal_np(p2s, geom_type, c, obj_mat, size)

    return {
        'p1':        p1s.copy(),
        'p2':        p2s.copy(),
        'p1s':       p1s,
        'p2s':       p2s,
        'n1_in':     n1_in,
        'n2_in':     n2_in,
        'offsets':   (0.0, 0.0),
        'delta_deg': 0.0,
    }


def _reachable_contact(p, ground_z, r_tip, z_margin=0.002):
    """Returns False if the contact is too close to the support surface for the fingertip sphere."""
    if p[2] < ground_z + r_tip + z_margin:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CasADi symbolic SDF helpers
# ─────────────────────────────────────────────────────────────────────────────

def _symbolic_box_sdf(p_sym: ca.MX,
                      center: np.ndarray, R: np.ndarray,
                      hx: float, hy: float, hz: float) -> ca.MX:
    """Standard CasADi box SDF (not C∞ — has kinks at face/edge transitions)."""
    p_local = ca.DM(R.T) @ (p_sym - ca.DM(center))
    q_vec   = ca.fabs(p_local) - ca.DM([hx, hy, hz])
    outside = ca.sqrt(ca.sumsqr(ca.fmax(q_vec, 0)) + 1e-12)
    inside  = ca.fmin(ca.fmax(ca.fmax(q_vec[0], q_vec[1]), q_vec[2]), 0)
    return outside + inside


def _symbolic_box_sdf_smooth(p_sym: ca.MX,
                              center: np.ndarray, R: np.ndarray,
                              hx: float, hy: float, hz: float,
                              alpha: float = 40.0) -> ca.MX:
    """C∞ smooth CasADi box SDF — required for SQP convergence."""
    eps = (1.0 / alpha) ** 2
    p_local = ca.DM(R.T) @ (p_sym - ca.DM(center))
    abs_local = ca.vertcat(
        ca.sqrt(p_local[0] ** 2 + eps),
        ca.sqrt(p_local[1] ** 2 + eps),
        ca.sqrt(p_local[2] ** 2 + eps),
    )
    q = abs_local - ca.DM([hx, hy, hz])

    def _sm0(x):
        return (x + ca.sqrt(x ** 2 + eps)) * 0.5

    sp0, sp1, sp2 = _sm0(q[0]), _sm0(q[1]), _sm0(q[2])
    outside = ca.sqrt(sp0 ** 2 + sp1 ** 2 + sp2 ** 2 + 1e-12)

    def _sm2(a, b):
        return (a + b + ca.sqrt((a - b) ** 2 + eps)) * 0.5

    q_max  = _sm2(_sm2(q[0], q[1]), q[2])
    inside = -_sm0(-q_max)
    return outside + inside


def _sym_geom_surface_con(opti, p_sym, d, geom_type: int,
                           center_np, mat_np, size, edge_margin: float = 0.0):
    """Apply shape-appropriate surface constraint to p_sym in opti.

    edge_margin (BOX only): keep the contact at least this far (m) from every face
    EDGE by shrinking the tangential bounds. HARD — a contact cannot be placed in the
    rim band, so a grasp that needs a near-edge contact becomes INFEASIBLE (the solve
    returns no contacts) rather than slipping. Per-axis auto-clamped so the band can
    never exceed the half-extent (would empty the feasible set); a warning-free clamp
    leaves a thin sliver at the face center for very small faces.
    """
    if d is None:
        return
    p_loc = ca.DM(mat_np.T) @ (p_sym - ca.DM(center_np))
    if geom_type == 6:   # BOX
        # Pin in the OBJECT LOCAL frame (p_loc), NOT world (p_sym). For a ROTATED box the
        # face is a tilted plane; constraining the world coordinate p_sym[ax]==const pins the
        # contact to a world-axis plane that is not the face, so the contact floats off the
        # surface by several mm (measured: up to +8mm on rotated boxes). p_loc is already
        # object-centred, so the face is simply p_loc[ax] == ±size[ax] and the tangential
        # bounds are on p_loc[ta] (matching the cylinder/sphere branches below). The
        # face-normal axis is chosen from the normal expressed in the LOCAL frame.
        d_loc = mat_np.T @ np.asarray(d, float)
        ax    = int(np.argmax(np.abs(d_loc)))
        coord = float(np.sign(d_loc[ax]) * float(size[ax]))
        opti.subject_to(p_loc[ax] == coord)
        for ta in [i for i in range(3) if i != ax]:
            # Half-extent minus the keep-out margin, clamped to a small positive sliver
            # so the box never collapses to an empty interval on tiny faces.
            _half = max(float(size[ta]) - float(edge_margin),
                        0.05 * float(size[ta]))
            opti.subject_to(opti.bounded(-_half, p_loc[ta], _half))
    elif geom_type == 5:  # CYLINDER
        R_c, H = float(size[0]), float(size[1])
        d_loc  = mat_np.T @ np.asarray(d, float)
        if abs(d_loc[2]) > 0.7:   # cap
            cap_z = H * float(np.sign(d_loc[2]))
            opti.subject_to(p_loc[2] == cap_z)
            opti.subject_to(ca.sumsqr(p_loc[0:2]) <= R_c ** 2)
        else:                      # curved surface
            opti.subject_to(ca.sumsqr(p_loc[0:2]) == R_c ** 2)
            opti.subject_to(opti.bounded(-H, p_loc[2], H))
    elif geom_type == 2:  # SPHERE
        r = float(size[0])
        opti.subject_to(ca.sumsqr(p_loc) == r ** 2)


def _embed_wrench_cone_ca(opti, p1, p2,
                           R1_param, R2_param,
                           obj_center_np, obj_R_np,
                           mu, task_f_np, task_t_np,
                           gamma_max,
                           use_slack: bool = False):
    """
    Add embedded LP wrench constraints to a CasADi Opti problem.

    Mirrors min_gamma_for_accel_lp (scripts/3D_minimum_NCF.py): a per-corner
    equality over the full [Tx,Ty,Tz,Fx,Fy,Fz] wrench, jointly sign-expanded
    over torque and force, evaluated in the object body frame (torque bounds
    derive from body-frame principal inertia, so the wrench balance must be
    computed there too).

    use_slack : bool — when False (default), the equality is HARD (w == w_k):
        a single geometrically-unreachable corner (e.g. exact zero moment arm
        about one torque axis) makes the whole NLP infeasible, but a converged
        solve is an exact certificate of wrench feasibility (matches
        min_gamma_for_accel_lp's default slack_penalty=None behavior).
        When True, a free per-corner slack variable s_k absorbs any residual
        (w + s_k == w_k instead of w == w_k), so one unreachable corner can't
        poison the whole solve — the caller must penalize ‖s_k‖ in the cost
        and gate wrench_ok on its post-solve magnitude instead of relying on
        solver convergence alone (matches min_gamma_for_accel_lp's
        slack_penalty=<float> mode).

    R1_param, R2_param : opti.parameter(3,3) — contact frame [n_in | t1 | t2],
        in WORLD frame. Computed numerically by caller via
        _build_contact_frame_3d(-d_lp). Held constant per NLP solve;
        updated between Picard iterations.
        Eliminates Frisvad singularity at n=[0,0,-1] (top-face inward normal).

    task_f_np : (3,) force magnitudes [fx, fy, fz], object body frame.
    task_t_np : (3,) torque magnitudes [τx, τy, τz], object body frame.
        Both sign-expanded jointly into up to 2^6 corners.

    Returns: gamma, y1_list, y2_list, corners, s_list
        corners : list of (6,) [τx,τy,τz,fx,fy,fz] wrenches, one per entry of
            y1_list/y2_list/s_list, in matching order — lets a caller warm-start
            each corner's y from an external per-corner LP solve.
        s_list  : list of opti.variable(6), per-corner wrench slack, when
            use_slack=True — inspect post-solve (opti.value / opti.debug.value)
            to judge how much of the solve relied on slack rather than exact
            wrench resistance. Empty list when use_slack=False.
    """
    import itertools

    nverts  = 5
    _gamma  = opti.variable()
    opti.subject_to(opti.bounded(0, _gamma, gamma_max))

    verts_c = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, -mu],
                         [1.0, -mu, 0.0],
                         [1.0,  mu, 0.0],
                         [1.0, 0.0,  mu]])
    forces1 = [R1_param @ ca.DM(v) for v in verts_c]
    forces2 = [R2_param @ ca.DM(v) for v in verts_c]
    obj_c   = ca.DM(obj_center_np)
    R_ow    = ca.DM(obj_R_np.T)   # world → object body-frame rotation

    def _wrench_sum(p, forces, y):
        w   = ca.MX.zeros(6)
        p_O = R_ow @ (p - obj_c)
        for j, f_j in enumerate(forces):
            f_O = R_ow @ f_j
            w += y[j] * ca.vertcat(ca.cross(p_O, f_O), f_O)
        return w

    # Joint corner expansion over torque AND force together — matches
    # min_gamma_for_accel_lp's `corners` set (3D_minimum_NCF.py:320-330).
    task6  = np.concatenate([task_t_np, task_f_np])
    nz_idx = np.where(np.abs(task6) > 1e-10)[0]
    seen, corners = set(), []
    for signs in itertools.product([-1, 1], repeat=len(nz_idx)):
        w_k = task6.copy()
        for i, idx in enumerate(nz_idx):
            w_k[idx] *= signs[i]
        key = tuple(np.round(w_k, 12))
        if key not in seen:
            seen.add(key); corners.append(w_k)
    if not corners:
        corners = [task6.copy()]

    # Per-row reference scale for the wrench-balance equality itself (not
    # just the cost, which already normalizes its own slack term the same
    # way — see w_slack cost at the call site). Torque rows (N*m) and force
    # rows (N) can differ by 4-5 orders of magnitude depending on
    # ang_accel_budget_xyz/object inertia; dividing both sides of the
    # equality by a nonzero constant leaves the solution set unchanged, it
    # only rescales the constraint Jacobian row IPOPT actually sees.
    _f_ref = max(float(np.linalg.norm(task_f_np)), 1e-6)
    _t_ref = max(float(np.linalg.norm(task_t_np)), 1e-6)
    ref6   = ca.DM([_t_ref] * 3 + [_f_ref] * 3)

    y1_list, y2_list, s_list = [], [], []
    for w_k in corners:
        _y1_k = opti.variable(nverts)
        _y2_k = opti.variable(nverts)
        opti.subject_to(_y1_k >= 0)
        opti.subject_to(_y2_k >= 0)
        opti.subject_to(ca.sum1(_y1_k) <= _gamma)
        opti.subject_to(ca.sum1(_y2_k) <= _gamma)
        w = _wrench_sum(p1, forces1, _y1_k) + _wrench_sum(p2, forces2, _y2_k)
        if use_slack:
            # Slack-relaxed equality across all 6 rows — mirrors
            # min_gamma_for_accel_lp's slack_penalty mode. Always feasible;
            # caller must penalize/gate on ‖s_k‖. _s_k itself stays in RAW
            # N/N*m units (not divided by ref6) so slack_tol_abs/
            # max_slack_norm/the existing cost-side ref6 normalization all
            # keep working unmodified — only the constraint equation is rescaled.
            _s_k = opti.variable(6)   # free per-corner wrench slack (penalized in caller's cost)
            opti.subject_to((w + _s_k) / ref6 == ca.DM(w_k) / ref6)
            s_list.append(_s_k)
        else:
            # Hard equality across all 6 rows — matches min_gamma_for_accel_lp's
            # default (slack_penalty=None) exactly.
            opti.subject_to(w / ref6 == ca.DM(w_k) / ref6)
        y1_list.append(_y1_k)
        y2_list.append(_y2_k)

    return _gamma, y1_list, y2_list, corners, s_list


# ─────────────────────────────────────────────────────────────────────────────
# GraspConfig3D
# ─────────────────────────────────────────────────────────────────────────────

# obj_clearance_by_geom values at or below this turn the arm-collision OBJECT constraint
# OFF for that geom (contact-tier fingertips/distal links that must touch the object). The
# geom is still kept for the FLOOR constraint. Chosen well below any physically-meaningful
# negative clearance (bounding-sphere phantom penetration tops out ~-0.02 m).
_COL_DISABLE_SENTINEL = -0.5   # m

# Fixed RNG seed reset at the start of every MultiStartGraspPlanner3D.solve() so the random
# seed stream is identical across solves — see the re-seed note in solve(). A constant, NOT a
# per-pose hash (which would be discontinuous under sub-mm teleop jitter).
_SEED_RNG_CONST = 42


@dataclass
class GraspConfig3D:
    """Configuration for the 3D grasp planner (Kinova Gen3 + LEAP hand)."""

    # Cost weights — dimensionless priorities (each term normalized to ≈1 at its reference)
    # d_ref=5mm, g_ref=task_load_N, y_ref=g_ref
    w_ik:     float = 0.70   # reachability — dominant until IK < 5mm
    w_reg:    float = 0.03   # posture tie-breaker
    w_gamma:  float = 0.15   # grasp quality (was 0.1, now normalized by task load)
    w_y:      float = 0.6   # min-norm force distribution tie-breaker
    # penalty on embedded wrench-cone slack (per-corner ‖s_k‖²). None (default)
    # => hard-equality NLP, no slack variables at all (pre-slack behavior).
    # Set to a float (e.g. 1e4) to opt into the slack-relaxed formulation.
    # w_slack:  float | None = None
    w_slack:  float = 1

    q_scale:  float = 1.0

    # Wrench-cone slack tolerances (see _embed_wrench_cone_ca / min_gamma_for_accel_lp
    # slack_penalty mode). A converged NLP no longer certifies exact wrench
    # resistance by itself — wrench_ok additionally requires max_slack_norm below
    # these thresholds. Absolute units: N for force rows, N*m for torque rows.
    slack_tol_abs:          float = 1e1   # gates NLP result's wrench_ok
    verify_slack_penalty:   float = 1e3    # slack_penalty passed to verify()'s diagnostic LP
    verify_slack_tol:       float = 1e1   # gates verify()'s wf_tag OK vs SLACK
    precheck_slack_penalty: float = 1e3    # slack_penalty passed to the seed-loop pre-check LP

    # Floor on the torque reference scale used to normalize the wrench-cone
    # slack COST (‖s_k / ref6‖²) — separate from the 1e-6 floor above, which
    # only exists to avoid a literal divide-by-zero. Torque task budgets can
    # legitimately be tiny (e.g. ang_accel_budget_xyz small relative to
    # object inertia), and d(cost_slack)/d(s_k) scales as 1/t_ref**2 — with
    # a several-1e-4 N*m t_ref this amplifies torque-row slack gradients by
    # millions, letting the slack term hijack the search direction away from
    # IK entirely (confirmed via the gradN_* per-term gradient diagnostic:
    # gradN_slack reached 10-30x gradN_ik within ~20 iterations on a
    # cylinder case with t_ref≈4.9e-4 N*m). This floor decouples "how much
    # torque resistance the task requires" from "how sensitive the penalty
    # is to violating it slightly" — small task budgets no longer create an
    # outsized gradient purely from the normalization, independent of
    # slack_tol_abs (which still gates feasibility on the true, unfloored
    # slack magnitude).
    slack_cost_t_ref_floor: float = 1e-1   # N*m



    # Fingertip mesh effective radius (site centroid to contact surface distance).
    # Always measured from model geometry in GraspPlanner3D.__init__ — never a
    # hardcoded guess; None here only until that measurement runs.
    r_thumb:  float | None = None   # m
    r_index:  float | None = None   # m
    r_middle: float | None = None   # m  — used for middle/ring ground clearance
    r_ring:   float | None = None   # m

    # Constraint flags
    joint_limits:      bool  = True   # active joint limit constraints in NLP
    wrench_constraint: bool  = True   # embedded LP wrench constraint in NLP
    max_iter:          int   = 50    # per-stage max iterations (Picard loop uses n_normal_relinearize+1 stages)

    # Grasp-axis alignment cost: penalize the grasp axis (p2-p1, normalized) deviating
    # from the contact inward normal, so the two contacts stay ANTIPODALLY OPPOSED (the
    # squeeze force routes straight between them). Shape-agnostic force-closure geometry:
    # box -> opposing faces, sphere/cyl -> diameter. Keeps IK-only solves from lifting the
    # contacts into a wrench-infeasible offset. 0.0 = off. Uses the frozen normal (box) /
    # symbolic normal (curved), same as the wrench frame.
    w_align:           float = 0.0

    # Fingertip PAD alignment cost: penalize each tip's pad axis (R_tip(q) @ pad_axis, the
    # fingerpad normal in world) deviating from the contact INWARD surface normal, so the pad
    # meets the surface flush rather than at an oblique edge-contact. This is the recommender
    # analog of ConstrainedIKSolver.orient_weight (same _SiteAxisCallbackAnalytic FK and same
    # pad_axis convention), added so the SINGLE committed solve (the recommender) can control
    # pad orientation directly instead of relying on the removed post-solve collision IK.
    # Distinct from w_align: w_align aligns the GRASP AXIS (p2-p1, contact geometry); this
    # aligns each FINGERPAD (tip orientation, a function of q). 0.0 = off. Adds one
    # _SiteAxisCallbackAnalytic per contact finger to the NLP (a q-dependent rotational FK —
    # keep the weight modest and watch convergence, per the symbolic-normals lesson).
    orient_weight:     float = 0.0
    # Fingerpad normal direction in the tip SITE frame (unit). Matches
    # ConstrainedIKSolver.pad_axis: -x of the LEAP fingertip site frame.
    pad_axis:          tuple = (-1.0, 0.0, 0.0)

    # Edge margin (HARD): keep contacts at least this far from every FACE EDGE, where a
    # pinch slips (short moment arm, friction cone falls off the face). Implemented by
    # shrinking the tangential face BOUNDS in _sym_geom_surface_con — a contact cannot be
    # placed in the rim band, so a grasp that needs a near-edge contact becomes INFEASIBLE
    # (solve returns no contacts) rather than slipping. Per-axis auto-clamped so the band
    # never empties a small face. BOX only. 0.0 = off (full face usable).
    edge_margin_m:     float = 0.015   # hard keep-out band width from each face edge (m)

    col_clearance_m:  float = 0.005

    # FLOOR clearance (m) for the ground constraint, separate from the object clearance above.
    # None -> use col_clearance_m (back-compat). Set slightly larger than col_clearance_m to
    # give the curled non-active fingers (middle/ring) extra margin against the table: their
    # links are modeled as coarse bounding SPHERES, so a sphere center clearing col_clearance_m
    # can still have the real thin link dip below the floor. A few mm of extra floor margin
    # absorbs that bounding-sphere slack without affecting the object grasp geometry.
    ground_clearance_m: float | None = None

    # Full-arm collision (proximity-pruned softplus SDFs).
    arm_geom_names:   list  = field(default_factory=list)
    col_prune_margin: float = 0.10
    col_use_ground:   bool  = True

    # Per-geom OBJECT clearance override for the arm-collision SDF loop (section 5a).
    # Maps an arm_geom_name -> the minimum sphere-vs-object signed distance required for
    # THAT geom (metres). Geoms absent from the dict use the scalar col_clearance_m. A
    # value <= _COL_DISABLE_SENTINEL (see below) turns the object constraint OFF for that
    # geom entirely — used for the CONTACT-tier fingertip/distal geoms, which must be free
    # to touch/wrap the object. The FLOOR constraint is never affected by this dict; every
    # geom keeps full col_clearance_m vs the ground plane so no link can drop underground.
    # This mirrors the ConstrainedIKSolver's reduced_clearance_geoms tiering so the
    # recommender's collision definition can be made to match the post-solve refinement's.
    # Negative values are legitimate here for the SAME reason they are in
    # _active_clearance_by_geom: each arm geom is modeled as its coarse bounding SPHERE
    # (geom_rbound), which over-approximates a thin link, so a link legitimately at the
    # surface reads several mm of phantom penetration. Once the link model is upgraded to
    # exact boxes these can go non-negative.
    obj_clearance_by_geom: dict = field(default_factory=dict)

    # Support-plane z coordinate (world frame) used to filter unreachable seeds.
    # Contacts whose outward normal points more than 60° below horizontal, or whose
    # z-position leaves no room for the fingertip above the table, are rejected.
    # Matches the hardcoded _ground_p=[0,0,0] in the NLP collision constraint.
    ground_z: float = 0.0

    # Regularisation target.  None → use q_dls (DLS warm-start) per seed.
    # Set to a fixed palm-down neutral configuration so regularisation is
    # consistent across seeds and drives the arm toward a natural pose.
    # GraspPlanner3D.__init__ builds a default from the model 'home' keyframe
    # + palm-down arm override; override here for custom neutral poses.
    q_neutral: np.ndarray | None = None

    # False (default) — arm joints regularize toward the fixed palm-down
    # pose baked into q_neutral/_q_neutral_default, same as the hand joints.
    # True — arm joints regularize toward q_ref (the caller's actual current
    # pose passed into solve()) instead, so the tie-breaker minimizes arm
    # movement from wherever the robot currently is rather than pulling it
    # toward one fixed reference regardless of the object's location. Hand
    # joints are unaffected either way (always the fixed neutral). Ignored
    # if q_neutral is set explicitly (an explicit override always wins).
    reg_arm_toward_current: bool = True

    # Profiling / Picard loop
    # extra Picard iterations (n total solves); 
    # n_normal_relinearize can be zero for box, search stays within the seed faces
    # n_normal_relinearize = 2 for curved surfaces, since normals genuinely rotate there
    n_seeds:             int  = 5    # seed pairs per multi-start (each from _seed_pair)
    n_normal_relinearize: int  = 1
    verbose_profile:      bool = False

    # Antipodal-march jitter (deg): _seed_pair marches from contact 1 along its inward
    # normal rotated by up to +/- this angle to find contact 2. Large values (the old 45)
    # let the march exit through an ADJACENT box face ~half the time; since each contact
    # is then HARD-PINNED to its seed face, the solve returns a non-opposite grasp that
    # w_align cannot rescue. A smaller angle keeps contact 2 on the opposing face while
    # still allowing some obliqueness.
    seed_march_jitter_deg: float = 15.0

    # Solver — use_slsqp=True  → SQP+OSQP (default; linear face-pin + wrench constraints exact)
    #           use_slsqp=False → IPOPT (interior-point; use for non-box geometries)
    use_slsqp:  bool  = False
    # symbolic_normals: build the wrench contact frame as a CasADi MX expression
    # of p1/p2 rather than a frozen parameter.  The frame is then re-evaluated at
    # every NLP function call (eval_f / eval_grad_f), making the contact frame
    # genuinely dynamic within the solve.  Supported for sphere and cylinder only
    # (box face normals are piecewise-constant and require frozen parameters).
    # This causes L-BFGS to accumulate curvature pairs (s_k, y_k) that are
    # inconsistent across iterations because ∇f changes not just due to x movement
    # but also due to the frame rotating with x — demonstrably harder to solve.
    
    symbolic_normals: bool = True
    smooth_sdf: bool  = True
    slsqp_alpha: float = 400.0  # smooth SDF alpha (collision avoidance SDF only)


    # Acceleration budgets for mass-scaled task wrench computation.
    # Used in solve() to replace fixed task_fx/fy/fz with mass*(accel+gravity),
    # and in verify() for the post-solve gamma check.
    # Defaults match kinova_leap_pick_place.py NCF_ACCEL_BUDGET_XYZ / NCF_ANG_ACCEL_BUDGET.
    accel_budget_xyz:     tuple = (0.25, 0.25, 0.25)   # m/s² linear, per world axis
    ang_accel_budget_xyz: tuple = (0.5, 0.5, 0.5)   # rad/s² angular, principal axes
    # accel_budget_xyz:     tuple = (0.5, 0.5, 0.5)   # m/s² linear, per world axis
    # ang_accel_budget_xyz: tuple = (0.05, 0.05, 0.05)   # rad/s² angular, principal axes

    # Datum / Task-B semantics for verify()'s post-solve gamma LP (aligns it with
    # kinova_leap_pick_place.solve_gamma_live). When True: reference the linear disturbance
    # at the GRASP MIDPOINT (moment_ref), pass gravity as a separate re-datumed wrench with
    # its grasp-axis moment projected out, and project the grasp-axis component of the
    # angular budget. This certifies a RAISED, reachable antipodal pinch as wrench-feasible
    # for a hold/transport task (see RAISED_CONTACT_WRENCH_FINDINGS.md sec 5). When False:
    # legacy CoM formulation (gravity folded into the accel box). Intended to pair with
    # wrench_constraint=False so the NLP is IK-only and verify() is the datum certificate.
    datum_gamma:          bool = False

    gamma_max: float = 25   # N — hard upper bound on the wrench-cone squeeze force gamma


    # Geometry names (must match scene XML)
    obj_geom:    str = 'obj_red_box_geom'
    obj_body:    str = 'obj_red_box'
    thumb_site:  str = 'leap_th_ds_tip'
    index_site:  str = 'leap_if_ds_tip'
    thumb_geom:  str = 'leap_th_tip'
    index_geom:  str = 'leap_if_tip'
    middle_geom: str = 'leap_mf_tip'
    ring_geom:   str = 'leap_rf_tip'
    cp1_geom:    str = 'cp1'
    cp2_geom:    str = 'cp2'
    cp3_geom:    str = 'cp3'
    cp4_geom:    str = 'cp4'
    cp1_body:    str = 'cp1_body'
    cp2_body:    str = 'cp2_body'
    cp3_body:    str = 'cp3_body'
    cp4_body:    str = 'cp4_body'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_actuated_indices(model) -> list[int]:
    """qpos address for each actuated joint, in actuator order."""
    return [model.jnt_qposadr[model.actuator_trnid[i, 0]]
            for i in range(model.nu)]


# ─────────────────────────────────────────────────────────────────────────────
# GraspPlanner3D
# ─────────────────────────────────────────────────────────────────────────────

class GraspPlanner3D:
    """
    3D grasp contact-point solver — SQP formulation (mirrors ConstrainedIKSolver).

    Decision variables: q[nu], p1[3] (thumb contact), p2[3] (index contact).

    Parameters
    ----------
    model   : mj.MjModel
    data    : mj.MjData
    cfg     : GraspConfig3D (optional)
    logger  : logging.Logger (optional)
    log_dir : str (optional — enables per-solve log files)
    """

    def __init__(self, model, data,
                 cfg: GraspConfig3D | None = None,
                 logger=None,
                 log_dir: str | None = None,
                 dashboard=None):
        self.model   = model
        self.data    = data
        self.cfg     = cfg or GraspConfig3D()
        self.log     = logger or log
        self.log_dir = log_dir
        self.dash    = dashboard

        if log_dir and logger is None:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use a unique per-instance logger rather than the shared
            # module-level 'grasp_planner_3d' singleton. Constructing several
            # GraspPlanner3D instances in one process (e.g. one per object,
            # as test_grasp_recommender.py does at startup) would otherwise
            # keep calling self.log.addHandler(fh) on the SAME shared logger
            # with no de-dup guard — every instance's FileHandler stays
            # attached for the rest of the process, so every subsequent log
            # call gets written once per accumulated handler (observed as
            # literal duplicate/triplicate log lines, and ~3x the per-
            # iteration file-write overhead). Propagation to the root logger
            # stays on (default), so console output via logging.basicConfig
            # is unaffected.
            self.log = logging.getLogger(f"grasp_planner_3d.{id(self)}")
            fh = logging.FileHandler(
                os.path.join(log_dir, f"grasp3d_{ts}.log"), encoding='utf-8')
            fh.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
            fh.setLevel(logging.DEBUG)
            self.log.setLevel(logging.DEBUG)
            self.log.addHandler(fh)

        c = self.cfg
        self._obj_gid    = self._require_geom(c.obj_geom)
        self._thumb_sid  = self._require_site(c.thumb_site)
        self._index_sid  = self._require_site(c.index_site)
        self._thumb_gid  = self._require_geom(c.thumb_geom)
        self._index_gid  = self._require_geom(c.index_geom)
        self._middle_gid = self._require_geom(c.middle_geom)
        self._ring_gid   = self._require_geom(c.ring_geom)
        self._cp1_gid    = self._optional_geom(c.cp1_geom)
        self._cp2_gid    = self._optional_geom(c.cp2_geom)
        self._cp3_gid    = self._optional_geom(c.cp3_geom)
        self._cp4_gid    = self._optional_geom(c.cp4_geom)

        # Fingertip effective radius (site centroid to contact-surface distance) is always
        # measured from the model geometry — never a hardcoded guess. Used as the IK offset:
        # the tip SITE is targeted at contact + r_tip*outward_normal so the pad surface (not
        # the site centroid) lands on the object.
        #
        # For a MESH tip (the LEAP fingertips), geom_rbound is the bounding sphere about the
        # mesh FRAME ORIGIN, which over-estimates the pad extent from the SITE (measured:
        # rbound 23.8mm; farthest mesh vertex from the site 19.5mm). We use the MAX vertex
        # distance from the site when a site id is given — the honest "max pad offset".
        #
        # Choice of MAX (not mean/min): the offset sets the IK target contact + r_tip*normal.
        # The cost is ASYMMETRIC — too small drives the pad INTO the object (penetration the
        # squeeze only worsens as it presses deeper), while too large leaves a GAP the squeeze
        # closes. A headless physics test (test_squeeze_closes_gap.py) showed the squeeze
        # closes a gap up to ~4mm object shift GENTLY. Measured pad-to-box gaps at recommended
        # q: MEAN offset (16.8mm) centres the gap on ZERO -> ~half the contacts PENETRATE (to
        # -6.1mm) — the exact failure to avoid; MAX offset (19.5mm) shifts the whole
        # distribution ~+2.7mm so contacts hover on the SAFE side with worst-case gap inside
        # the squeeze's gentle-closing window. So MAX is the correctly-calibrated choice: it
        # keeps the worst contact non-penetrating without exceeding the shove threshold.
        # (Directional sampling along the actual contact normal would tighten the gap toward 0
        # but needs the tip orientation as a symbolic function of q inside the NLP — a
        # non-smooth support function that degrades L-BFGS convergence — not worth it.)
        def _tip_radius(gid, sid=None):
            gt = int(model.geom_type[gid])
            gs = model.geom_size[gid]
            if gt == 2:   # mjGEOM_SPHERE: size[0] = radius
                return float(gs[0])
            if gt == 3:   # mjGEOM_CAPSULE: size[0] = radius
                return float(gs[0])
            if gt == 6:   # mjGEOM_BOX: use min half-extent as a conservative radius
                return float(np.min(gs[:3]))
            if gt == 7 and sid is not None:   # mjGEOM_MESH with a known site
                did  = int(model.geom_dataid[gid])
                if did >= 0:
                    vadr = int(model.mesh_vertadr[did])
                    vnum = int(model.mesh_vertnum[did])
                    V = model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3)   # mesh local
                    # Site in the mesh(geom) local frame; the site pose relative to the geom is
                    # model-static, so read it at qpos0.
                    _d0 = mj.MjData(model)
                    mj.mj_forward(model, _d0)
                    gpos = _d0.geom_xpos[gid]
                    gmat = _d0.geom_xmat[gid].reshape(3, 3)
                    site_local = gmat.T @ (_d0.site_xpos[sid] - gpos)
                    return float(np.max(np.linalg.norm(V - site_local, axis=1)))
            # MESH without a site (middle/ring) or other: MuJoCo's bounding radius.
            return float(model.geom_rbound[gid])

        c.r_thumb  = _tip_radius(self._thumb_gid, self._thumb_sid)
        c.r_index  = _tip_radius(self._index_gid, self._index_sid)
        c.r_middle = _tip_radius(self._middle_gid)
        c.r_ring   = _tip_radius(self._ring_gid)
        self.log.info(
            f"[tip_radius] r_thumb={c.r_thumb*1e3:.1f}mm  r_index={c.r_index*1e3:.1f}mm  "
            f"r_middle={c.r_middle*1e3:.1f}mm  r_ring={c.r_ring*1e3:.1f}mm  "
            f"(geom types: thumb={int(model.geom_type[self._thumb_gid])}  "
            f"index={int(model.geom_type[self._index_gid])})")

        def _maybe_mocap(bname):
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bname)
            if bid == -1:
                return None
            mid = model.body_mocapid[bid]
            return int(mid) if mid >= 0 else None

        self._cp1_mocap   = _maybe_mocap(c.cp1_body)
        self._cp2_mocap   = _maybe_mocap(c.cp2_body)
        self._cp3_mocap   = _maybe_mocap(c.cp3_body)
        self._cp4_mocap   = _maybe_mocap(c.cp4_body)
        self._has_markers = all(m is not None
                                for m in (self._cp1_mocap, self._cp2_mocap))

        self._act_idx = _get_actuated_indices(model)
        n_act = len(self._act_idx)

        # Palm-down neutral q for regularisation.
        # Start from the model 'home' keyframe (which has good hand joint values),
        # then override the first 7 arm joints with the palm-down wrist pose from
        # kinova_leap_pick_place.py (_HOME_WRIST_DOWN).
        _PALM_DOWN_ARM = np.array([-0.217, 1.144, 3.44, -2.011, -0.087, 1.541, 2.872])
        _key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'home')
        if _key_id != -1:
            _home_full = model.key_qpos[_key_id].copy()
        else:
            _home_full = model.qpos0.copy()
        _q_neutral_full = _home_full.copy()
        _q_neutral_full[:min(7, model.nq)] = _PALM_DOWN_ARM[:min(7, model.nq)]
        self._q_neutral_default = np.array([_q_neutral_full[i] for i in self._act_idx])
        # Arm-joint count within the actuator-ordered q vector (arm first,
        # hand after — matches _act_idx / q_ref convention throughout this
        # file). Used by cfg.reg_arm_toward_current to split the
        # regularization target between arm and hand joints.
        self._n_arm_joints = min(len(_PALM_DOWN_ARM), n_act)

        gs = model.geom_size[self._obj_gid]
        self._obj_hx        = float(gs[0])
        self._obj_hy        = float(gs[1])
        self._obj_hz        = float(gs[2])
        self._obj_size      = model.geom_size[self._obj_gid].copy()
        self._obj_geom_type = int(model.geom_type[self._obj_gid])

        # Joint limit vectors (vectorized constraint — mirrors constrained_ik)
        lo_list, hi_list = [], []
        for i in range(n_act):
            jid = model.actuator_trnid[i, 0]
            if model.jnt_limited[jid]:
                lo_list.append(float(model.jnt_range[jid, 0]))
                hi_list.append(float(model.jnt_range[jid, 1]))
            else:
                lo_list.append(-np.pi)
                hi_list.append( np.pi)
        self._lo_vec = np.array(lo_list)
        self._hi_vec = np.array(hi_list)

        # Full-arm collision geoms. _arm_obj_clearance is the per-geom OBJECT clearance
        # (section 5a), resolved once here from cfg.obj_clearance_by_geom with the scalar
        # col_clearance_m as the default. A geom whose resolved clearance is the disable
        # sentinel gets NO object constraint (contact-tier fingertips/distal links that must
        # touch); it is still kept for the FLOOR constraint. Parallel-indexed with
        # _arm_gids/_arm_radii so the solve loop can look up by the same _ai.
        self._arm_gids  = []
        self._arm_radii = []
        self._arm_obj_clearance = []
        _obj_clr_map = dict(getattr(c, 'obj_clearance_by_geom', None) or {})
        for gname in (c.arm_geom_names or []):
            gid = self._optional_geom(gname)
            if gid is not None:
                self._arm_gids.append(gid)
                self._arm_radii.append(float(model.geom_rbound[gid]))
                self._arm_obj_clearance.append(
                    float(_obj_clr_map.get(gname, c.col_clearance_m)))
            else:
                self.log.warning(f"GraspPlanner3D: arm_geom '{gname}' not found — skipped")

        self._dls_ik   = SpatialIKSolver(n_robot=n_act)
        self._dls_data = mj.MjData(model)

    # ── public API ─────────────────────────────────────────────────────────────

    def solve(self,
              q_ref:   np.ndarray,
              obj_pos: np.ndarray,
              p1_init: np.ndarray | None = None,
              p2_init: np.ndarray | None = None,
              d1:      np.ndarray | None = None,
              d2:      np.ndarray | None = None,
              iter_callback=None,
              update_normals_in_callback: bool = False,
              gamma_init: float | None = None,
              y_by_corner_init: dict | None = None) -> dict:
        """
        Run 3D grasp optimisation (synchronous / blocking).

        Parameters
        ----------
        q_ref   : (nu,) warm-start / regularisation target joint angles.
        obj_pos : (3,)  object center in world frame.
        p1_init : (3,)  optional thumb contact seed.
        p2_init : (3,)  optional index contact seed.
        d1, d2  : face-direction unit vectors (from MultiStart seeds).
                  When provided, pins p1/p2 to the corresponding geom face.
        gamma_init       : optional γ from the caller's pre-solver LP check
                            (min_gamma_for_accel_lp), used to warm-start the
                            embedded wrench-cone LP's γ variable instead of 1.0.
        y_by_corner_init : optional {corner_tuple: y*} dict from the same LP
                            check (return_y=True), used to warm-start the
                            per-corner cone coefficients y1/y2 instead of a
                            uniform 1/5 split. Keyed by rounded
                            (Tx,Ty,Tz,Fx,Fy,Fz) corner wrench, matching the
                            corners generated in _embed_wrench_cone_ca.

        Returns
        -------
        dict — success, q, p1, p2, cost, iterations, status
        """
        cfg     = self.cfg
        model   = self.model
        act_idx = self._act_idx
        n_act   = len(act_idx)
        hx, hy, hz   = self._obj_hx, self._obj_hy, self._obj_hz
        geom_type    = self._obj_geom_type
        geom_size    = self._obj_size

        if self.dash is not None:
            self.dash.push({'type': 'active', 'label': 'grasp3d'})

        # ── Freeze object pose ──────────────────────────────────────────────
        data_cb = mj.MjData(model)
        data_cb.qpos[:] = self.data.qpos[:]
        data_cb.qvel[:] = self.data.qvel[:]
        mj.mj_forward(model, data_cb)

        obj_center_np = data_cb.geom_xpos[self._obj_gid].copy()
        obj_R_np      = data_cb.geom_xmat[self._obj_gid].reshape(3, 3).copy()
        n_obj_dof     = model.nq - n_act
        obj_qpos_snap = data_cb.qpos[n_act:].copy() if n_obj_dof > 0 else None

        # Task wrench from object mass/inertia and configured acceleration budgets.
        _bid    = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.obj_body)
        _mass   = float(model.body_mass[_bid])
        _inert  = model.body_inertia[_bid]
        _g_O    = obj_R_np.T @ model.opt.gravity
        _ab     = cfg.accel_budget_xyz
        _aab    = cfg.ang_accel_budget_xyz
        _mu     = round(1 * float(model.geom_friction[self._obj_gid][0]), 3)
        _nlp_fx = _mass * (_ab[0] + abs(_g_O[0]))
        _nlp_fy = _mass * (_ab[1] + abs(_g_O[1]))
        _nlp_fz = _mass * (_ab[2] + abs(_g_O[2]))
        _nlp_tx = float(_inert[0]) * _aab[0]
        _nlp_ty = float(_inert[1]) * _aab[1]
        _nlp_tz = float(_inert[2]) * _aab[2]
        _f_ref_dbg = float(np.linalg.norm([_nlp_fx, _nlp_fy, _nlp_fz]))
        _t_ref_dbg = float(np.linalg.norm([_nlp_tx, _nlp_ty, _nlp_tz]))
        self.log.info(
            f"[wrench_budget] force=[{_nlp_fx:.4f},{_nlp_fy:.4f},{_nlp_fz:.4f}]N "
            f"(|F|={_f_ref_dbg:.4f})  "
            f"torque=[{_nlp_tx:.6f},{_nlp_ty:.6f},{_nlp_tz:.6f}]N*m "
            f"(|T|={_t_ref_dbg:.6f})  "
            f"ratio |F|/|T|={_f_ref_dbg / max(_t_ref_dbg, 1e-12):.1f}  "
            f"accel_budget={_ab}  ang_accel_budget={_aab}  inertia={list(np.round(_inert, 6))}")

        obj_center = np.asarray(obj_pos, dtype=float)
        margin     = max(hx, hy, hz)

        # ── Seed contact points ─────────────────────────────────────────────
        p1_seed = (np.asarray(p1_init, float) if p1_init is not None
                   else obj_center + np.array([0.0, -hy, 0.0]))
        p2_seed = (np.asarray(p2_init, float) if p2_init is not None
                   else obj_center + np.array([0.0,  hy, 0.0]))

        # ── DLS warm-start IK ───────────────────────────────────────────────
        _t_ws = time.perf_counter()
        self._dls_data.qpos[:n_act] = q_ref
        if obj_qpos_snap is not None:
            self._dls_data.qpos[n_act:] = obj_qpos_snap
        q_dls = self._dls_ik.solve(
            self.model, self._dls_data,
            [self._thumb_sid, self._index_sid],
            [p1_seed, p2_seed],
            q_bias=q_ref, null_gain=0.3)
        mj.mj_kinematics(self.model, self._dls_data)
        
        # DLS target is the contact point; IK error is ‖site − (p + r·n_out)‖.
        _d1_ws = (np.asarray(d1, float) if d1 is not None
                  else _geom_normal_np(p1_seed, geom_type, obj_center_np, obj_R_np, geom_size))
        _d2_ws = (np.asarray(d2, float) if d2 is not None
                  else _geom_normal_np(p2_seed, geom_type, obj_center_np, obj_R_np, geom_size))
        _err_th = float(np.linalg.norm(
            self._dls_data.site_xpos[self._thumb_sid] - (p1_seed + cfg.r_thumb * _d1_ws)))
        _err_if = float(np.linalg.norm(
            self._dls_data.site_xpos[self._index_sid] - (p2_seed + cfg.r_index * _d2_ws)))
        self.log.info(
            f"[dls_ws] th={_err_th*1e3:.1f}mm  idx={_err_if*1e3:.1f}mm  "
            f"dt={1e3*(time.perf_counter()-_t_ws):.0f}ms")

        # ── Proximity pruning: arm geoms vs object for q_dls arm config ──────────────────────────
        _active_arm = []
        if self._arm_gids:
            _data_prune = mj.MjData(model)
            _data_prune.qpos[:n_act] = q_dls
            if obj_qpos_snap is not None:
                _data_prune.qpos[n_act:] = obj_qpos_snap
            mj.mj_kinematics(model, _data_prune)
            for _ai, (_agid, _ar) in enumerate(zip(self._arm_gids, self._arm_radii)):
                _gp = _data_prune.geom_xpos[_agid]
                _arm_dist = (_geom_sdf_np(_gp, geom_type, obj_center_np,
                                          obj_R_np, geom_size) - _ar)
                # Prune against THIS geom's own object clearance, not the scalar default —
                # a proximal finger link with a tighter required clearance must be kept even
                # when it sits slightly farther than col_clearance_m would demand. Contact-
                # tier geoms (object constraint disabled) use the sentinel as their prune
                # threshold, so the object test never keeps them for the object's sake.
                _clr_ai   = float(self._arm_obj_clearance[_ai])
                _near_obj = (_clr_ai > _COL_DISABLE_SENTINEL
                             and _arm_dist < _clr_ai + cfg.col_prune_margin)
                # Floor proximity: every geom needs the ground constraint (section 5a), so a
                # geom close to the floor is kept even when far from the object. z of the
                # sphere surface vs ground_z.
                _clr_gr = float(cfg.ground_clearance_m
                                if cfg.ground_clearance_m is not None else cfg.col_clearance_m)
                _near_flr = (cfg.col_use_ground
                             and (float(_gp[2]) - _ar - float(cfg.ground_z))
                                  < _clr_gr + cfg.col_prune_margin)
                if _near_obj or _near_flr:
                    _active_arm.append(_ai)

        # ── inner: build + run one Opti problem ────────────────────────────
        def _run_stage(q_ws: np.ndarray,
                       p1_ws: np.ndarray,
                       p2_ws: np.ndarray,
                       include_surface: bool,
                       d1_lp: np.ndarray | None = None,
                       d2_lp: np.ndarray | None = None,
                       max_iter_override: int | None = None,
                       stage_label: str = '',
                       iter_callback=None,
                       update_normals_in_callback: bool = False) -> dict:

            _t_stage_start = time.perf_counter()
            _uid = id(q_ws)

            # ── FK callbacks (analytic Jacobians via mj_jacSite) ───────────
            thumb_cb = _SitePositionCallbackAnalytic(
                f'gp3_th_{_uid}', model, self._thumb_sid, n_act, obj_qpos_snap)
            index_cb = _SitePositionCallbackAnalytic(
                f'gp3_if_{_uid}', model, self._index_sid, n_act, obj_qpos_snap)

            # Fingerpad-axis FK (R_tip(q) @ pad_axis in world) for the orient_weight cost —
            # only built when the term is active, since each adds a q-dependent rotational
            # callback to the NLP.
            thumb_axis_cb = index_axis_cb = None
            if cfg.orient_weight > 0.0:
                _pad_ax = np.asarray(cfg.pad_axis, float)
                thumb_axis_cb = _SiteAxisCallbackAnalytic(
                    f'gp3_th_ax_{_uid}', model, self._thumb_sid, _pad_ax, n_act, obj_qpos_snap)
                index_axis_cb = _SiteAxisCallbackAnalytic(
                    f'gp3_if_ax_{_uid}', model, self._index_sid, _pad_ax, n_act, obj_qpos_snap)

            # ── Arm collision callback (active pairs only) ─────────────────
            arm_col_cb = None
            if _active_arm:
                arm_col_cb = _BatchedGeomPositionCallbackAnalytic(
                    f'gp3_arm_{_uid}', model,
                    [self._arm_gids[_ai] for _ai in _active_arm],
                    n_act, obj_qpos_snap)

            # embedded mode only — no callback wrench

            # ── Build Opti ────────────────────────────────────────────────
            _opti = ca.Opti()
            _q  = _opti.variable(n_act)
            _p1 = _opti.variable(3)
            _p2 = _opti.variable(3)

            _tp1   = thumb_cb(_q)
            _tp2   = index_cb(_q)
            # IK cost: fingertip center should be at contact point + r_tip * outward_normal.
            # Without the offset the tip sphere embeds r_tip mm into the object surface.
            _tp1_tgt = _p1 + ca.DM(float(cfg.r_thumb) * d1_lp)
            _tp2_tgt = _p2 + ca.DM(float(cfg.r_index) * d2_lp)
            _d1_sq = ca.sumsqr(_tp1 - _tp1_tgt)   # m²
            _d2_sq = ca.sumsqr(_tp2 - _tp2_tgt)   # m²

            # ── SDF for surface constraints ─────────────────────────────────
            if cfg.smooth_sdf:
                def _sdf(p):
                    return _symbolic_box_sdf_smooth(
                        p, obj_center_np, obj_R_np, hx, hy, hz,
                        alpha=cfg.slsqp_alpha)
            else:
                def _sdf(p):
                    return _symbolic_box_sdf(
                        p, obj_center_np, obj_R_np, hx, hy, hz)

            # ── Regularisation target ─────────────────────────────────────
            # Use a fixed palm-down neutral rather than the per-seed DLS
            # solution so regularisation is consistent across all seeds.
            _q_reg = (cfg.q_neutral if cfg.q_neutral is not None
                      else self._q_neutral_default)
            if cfg.q_neutral is None and cfg.reg_arm_toward_current:
                # Arm joints regularize toward the caller's actual current
                # pose (q_ref, closed over from solve()) instead of the
                # fixed palm-down reference — minimizes arm movement rather
                # than pulling toward one fixed posture regardless of where
                # the object is. Hand joints keep the fixed neutral.
                _q_reg = _q_reg.copy()
                _n_arm = self._n_arm_joints
                _q_reg[:_n_arm] = np.asarray(q_ref, float)[:_n_arm]

            # ── Cost (normalized — each term ≈ 1 at its reference level) ──
            # Reference scales encode what "good enough" means for each term.
            _d_ref  = 0.005                       # m   — acceptable IK residual
            _n_dof  = int(ca.MX(_q).numel())
            _cost_ik  = 0.5 * (_d1_sq + _d2_sq) / _d_ref**2
            _cost_reg = ca.sumsqr((_q - ca.DM(_q_reg)) / cfg.q_scale) / _n_dof
            _cost = cfg.w_ik * _cost_ik + cfg.w_reg * _cost_reg

            # ── Grasp-axis alignment (shape-agnostic force-closure geometry) ──
            # Penalize the grasp axis g_hat = (p2 - p1)/||p2 - p1|| deviating from the
            # contact-1 inward normal n1_in. When aligned, the two contacts are directly
            # opposed and the squeeze force routes straight between them (resistable);
            # when offset, squeezing makes an unresistable couple (the IK-only failure).
            # Uses the frozen inward normal from the seed face direction d1_lp (box) — a
            # constant per face, so this is a smooth quadratic in p1/p2 only.
            if cfg.w_align > 0.0 and d1_lp is not None:
                _n1_in_al = ca.DM(-np.asarray(d1_lp, float)
                                  / (np.linalg.norm(d1_lp) + 1e-12))
                _dp = _p2 - _p1
                _g_hat = _dp / (ca.norm_2(_dp) + 1e-9)
                _cost_align = ca.sumsqr(_g_hat - _n1_in_al)
                _cost = _cost + cfg.w_align * _cost_align

            # ── Fingerpad-normal alignment (orient_weight) ────────────────────
            # Penalize each tip's pad axis (R_tip(q) @ pad_axis, world) deviating from that
            # contact's INWARD surface normal, so the pad meets the face flush. Same term as
            # ConstrainedIKSolver's orient_weight (‖R_tip@pad_axis − n_in‖² per contact). The
            # inward normals are the frozen seed directions (-d1_lp thumb, -d2_lp index),
            # constant per stage like the wrench frame, so the only q-dependence is the tip
            # rotation (via the axis callbacks). Sentinel-zero when off so the log helper can
            # always evaluate it.
            _cost_orient = ca.DM(0.0)
            if (cfg.orient_weight > 0.0 and thumb_axis_cb is not None
                    and d1_lp is not None and d2_lp is not None):
                _n1_in_or = ca.DM(-np.asarray(d1_lp, float)
                                  / (np.linalg.norm(d1_lp) + 1e-12))
                _n2_in_or = ca.DM(-np.asarray(d2_lp, float)
                                  / (np.linalg.norm(d2_lp) + 1e-12))
                _e_th = thumb_axis_cb(_q) - _n1_in_or
                _e_if = index_axis_cb(_q) - _n2_in_or
                _cost_orient = ca.dot(_e_th, _e_th) + ca.dot(_e_if, _e_if)
                _cost = _cost + cfg.orient_weight * _cost_orient

            # Edge margin is now a HARD constraint (tightened tangential face bounds in
            # _sym_geom_surface_con via cfg.edge_margin_m) — no cost term needed.

            # ── 1. Joint limits (vectorized) ──────────────────────────────
            if cfg.joint_limits:
                _opti.subject_to(_opti.bounded(
                    ca.DM(self._lo_vec), _q, ca.DM(self._hi_vec)))

            # ── 2. Surface constraints ────────────────────────────────────────
            if include_surface:
                _Rt_dm = ca.DM(obj_R_np.T)
                _c_dm  = ca.DM(obj_center_np)
                for _p, _d_lp in ((_p1, d1_lp), (_p2, d2_lp)):
                    _pl = _Rt_dm @ (_p - _c_dm)   # local frame
                    _sym_geom_surface_con(_opti, _p, _d_lp,
                                            geom_type, obj_center_np, obj_R_np, geom_size,
                                            edge_margin=cfg.edge_margin_m)


            # Bounding box: prevents p1/p2 from flying to infinity
            for _p in (_p1, _p2):
                for _i, _h in enumerate([hx, hy, hz]):
                    _opti.subject_to(_opti.bounded(
                        obj_center[_i] - _h - margin,
                        _p[_i],
                        obj_center[_i] + _h + margin))

            # ── 3. Wrench feasibility (embedded LP) ───────────────────────
            _gamma_lp    = None
            _y1_list     = []
            _y2_list     = []
            _s_list      = []
            _R1_param    = None   # hoisted so _opti_cb closure can see them
            _R2_param    = None
            _n1_in       = None   # frozen contact normals actually used this stage
            _n2_in       = None
            if cfg.wrench_constraint:
                use_sym_normals = (cfg.symbolic_normals and
                                   geom_type in (2, 5))  # sphere / cylinder only
                if use_sym_normals:
                    # Contact frame built as a CasADi MX expression of _p1/_p2.
                    # CasADi re-evaluates this at every eval_f / eval_grad_f call,
                    # so the frame tracks the current contact position throughout
                    # the solve.  L-BFGS sees inconsistent curvature pairs because
                    # ∇f changes both from x movement AND frame rotation.
                    _c_dm  = ca.DM(obj_center_np)
                    _Rt_dm = ca.DM(obj_R_np.T)
                    _n1_in_sym = _sym_inward_normal_ca(_p1, geom_type, _c_dm, _Rt_dm, geom_size)
                    _n2_in_sym = _sym_inward_normal_ca(_p2, geom_type, _c_dm, _Rt_dm, geom_size)
                    _R1_expr = _symbolic_contact_frame_ca(_n1_in_sym)
                    _R2_expr = _symbolic_contact_frame_ca(_n2_in_sym)
                else:
                    # Default: contact frames as 3×3 parameter — frozen per NLP solve,
                    # updated between Picard iterations by the outer loop.
                    _n1_in = -(d1_lp if d1_lp is not None else
                                _geom_normal_np(p1_ws, geom_type, obj_center_np, obj_R_np, geom_size))
                    _n2_in = -(d2_lp if d2_lp is not None else
                                _geom_normal_np(p2_ws, geom_type, obj_center_np, obj_R_np, geom_size))
                    _R1_param = _opti.parameter(3, 3)
                    _R2_param = _opti.parameter(3, 3)
                    _opti.set_value(_R1_param, np.column_stack(_build_contact_frame_3d(_n1_in)))
                    _opti.set_value(_R2_param, np.column_stack(_build_contact_frame_3d(_n2_in)))
                    _R1_expr = _R1_param
                    _R2_expr = _R2_param

                # Grasp-axis torque projection. A 2-contact pinch geometrically CANNOT
                # resist torque about the grasp axis (the line through the two contacts):
                # the friction-cone forces have no moment arm about it, so any nonzero
                # torque budget on that axis makes every wrench corner infeasible (the
                # Tx=None antipodal case in 3D_minimum_NCF). Zero the torque budget's
                # grasp-axis component so the hard-equality cone stays feasible — the same
                # projection solve_gamma_live already applies in the live GRASP path.
                # Frozen (not symbolic): the face-pin keeps each box contact on its seed
                # face, so p1_ws/p2_ws (hence the grasp axis) are static across the solve;
                # updated between Picard iterations along with the normals. Computed in the
                # OBJECT body frame, matching the torque budget's frame.
                _tb = np.array([_nlp_tx, _nlp_ty, _nlp_tz], float)
                _ga_w = np.asarray(p2_ws, float) - np.asarray(p1_ws, float)
                _ga_n = float(np.linalg.norm(_ga_w))
                if _ga_n > 1e-9:
                    _ga_O = (obj_R_np.T @ _ga_w) / _ga_n          # grasp axis, object frame
                    _tb = _tb - np.dot(_tb, _ga_O) * _ga_O        # kill grasp-axis torque
                    _tb = np.abs(_tb)                             # per-axis magnitudes
                _gamma_lp, _y1_list, _y2_list, _corners, _s_list = _embed_wrench_cone_ca(
                    _opti, _p1, _p2,
                    _R1_expr, _R2_expr,
                    obj_center_np, obj_R_np, _mu,
                    np.array([_nlp_fx, _nlp_fy, _nlp_fz]),
                    _tb,
                    cfg.gamma_max,
                    use_slack=(cfg.w_slack is not None))
                
                
                # Note: Do not warm-start with LP wrench solutions. It makes it much worse. 
                # 
                # # Warm-start γ and the per-corner cone y's from the caller's
                # # pre-solver LP check (min_gamma_for_accel_lp) when available,
                # # falling back to a naive uniform init otherwise.
                # _opti.set_initial(_gamma_lp,
                #                    float(gamma_init) if gamma_init is not None else 1.0)
                # _y_init = np.ones(5) / 5.0
                # for _w_k, _y1_k, _y2_k in zip(_corners, _y1_list, _y2_list):
                #     _y_star = (y_by_corner_init.get(tuple(np.round(_w_k, 9)))
                #                if y_by_corner_init is not None else None)
                #     if _y_star is not None:
                #         _opti.set_initial(_y1_k, _y_star[:5])
                #         _opti.set_initial(_y2_k, _y_star[5:])
                #     else:
                #         _opti.set_initial(_y1_k, _y_init)
                #         _opti.set_initial(_y2_k, _y_init)

                # Cold-start γ and y's to uniform values instead of LP solutions. 
                _opti.set_initial(_gamma_lp, 1.0)
                for _w_k, _y1_k, _y2_k in zip(_corners, _y1_list, _y2_list):
                    _opti.set_initial(_y1_k, np.ones(5) / 5.0)
                    _opti.set_initial(_y2_k, np.ones(5) / 5.0)





            # Finalize cost (gamma + y regularizer + wrench-cone slack — all normalized)
            # Sentinel zero expressions let the log helper always evaluate all terms.
            _cost_gamma = ca.DM(0.0)
            _cost_y     = ca.DM(0.0)
            _cost_slack = ca.DM(0.0)
            if _gamma_lp is not None:
                _g_ref  = float(np.linalg.norm([_nlp_fx, _nlp_fy, _nlp_fz]))  # N task load
                _t_ref  = float(np.linalg.norm([_nlp_tx, _nlp_ty, _nlp_tz]))  # N*m task load
                _y_ref  = _g_ref                                                 # N force scale
                _n_c    = max(len(_y1_list), 1)
                _n_y    = _n_c * 10                                              # 5 verts × 2 contacts per corner
                _cost_gamma = _gamma_lp / max(_g_ref, 1e-6)
                _cost_y     = sum(ca.sumsqr(y1k) + ca.sumsqr(y2k)
                                  for y1k, y2k in zip(_y1_list, _y2_list)) / (_n_y * _y_ref**2)
                _opti_cost = (_cost
                              + cfg.w_gamma * _cost_gamma
                              + cfg.w_y     * _cost_y)
                if cfg.w_slack is not None and _s_list:
                    # Per-row normalization: rows 0-2 are torque (N*m), rows 3-5
                    # are force (N) — mixed units, so each needs its own
                    # reference scale before summing squares (same pattern as
                    # _g_ref elsewhere). The torque reference is floored at
                    # cfg.slack_cost_t_ref_floor (not just 1e-6) — unlike the
                    # constraint-equation row-normalization in
                    # _embed_wrench_cone_ca (where a tiny t_ref correctly maps
                    # a tiny torque requirement to an O(1) target), here a tiny
                    # t_ref amplifies d(cost_slack)/d(s_k) by 1/t_ref**2 and
                    # lets the slack term hijack the gradient away from IK —
                    # see cfg.slack_cost_t_ref_floor's docstring.
                    _t_ref_slack = max(_t_ref, cfg.slack_cost_t_ref_floor)
                    _ref6 = ca.DM([_t_ref_slack] * 3 + [max(_g_ref, 1e-6)] * 3)
                    _cost_slack = (sum(ca.sumsqr(s_k / _ref6) for s_k in _s_list)
                                   / max(len(_s_list), 1))
                    _opti_cost = _opti_cost + cfg.w_slack * _cost_slack
                _opti.minimize(_opti_cost)
            else:
                _opti.minimize(_cost)

            # ── Per-term gradient norms — "what's driving the search" ──────
            # ‖∇(w_i * J_i)‖ w.r.t. the full stacked variable vector opti.x,
            # evaluated at each iterate via opti.debug.value alongside the
            # existing weighted cost VALUES above. A value can be small while
            # its gradient still dominates the step (or vice versa near a
            # stationary point of that term alone) — this is what actually
            # explains which term is pushing the solver at a given iteration,
            # not just which term's current value is largest.
            _grad_w_slack = ((cfg.w_slack if cfg.w_slack is not None else 0.0)
                              * _cost_slack)
            _grad_terms = {
                'ik':    cfg.w_ik    * _cost_ik,
                'reg':   cfg.w_reg   * _cost_reg,
                'gamma': cfg.w_gamma * _cost_gamma,
                'y':     cfg.w_y     * _cost_y,
                'slack': _grad_w_slack,
            }
            _grad_norm_exprs = {
                name: ca.norm_2(ca.gradient(term, _opti.x))
                for name, term in _grad_terms.items()
            }
            _grad_norm_total = ca.norm_2(ca.gradient(_opti.f, _opti.x))

            # ── 5a. Full-arm collision (geometry-appropriate softplus SDF) ─
            _ground_n = ca.DM([0.0, 0.0, 1.0])
            _ground_p = ca.DM([0.0, 0.0, float(cfg.ground_z)])
            # Floor clearance: ground_clearance_m if set, else col_clearance_m (back-compat).
            _clr_ground = float(cfg.ground_clearance_m
                                if cfg.ground_clearance_m is not None else cfg.col_clearance_m)
            if arm_col_cb is not None:
                _arm_pos = arm_col_cb(_q)   # (3*n_active,) CasADi vector
                _obj_R_dm = ca.DM(obj_R_np)
                _obj_c_dm = ca.DM(obj_center_np)
                for _j, _ai in enumerate(_active_arm):
                    _gp = _arm_pos[3*_j : 3*_j+3]
                    _r  = float(self._arm_radii[_ai])
                    # Per-geom OBJECT clearance (defaults to col_clearance_m; see
                    # GraspConfig3D.obj_clearance_by_geom). At/below the disable sentinel the
                    # object constraint is skipped entirely for this geom (contact-tier
                    # fingertips/distal links that must touch), but the FLOOR constraint below
                    # is still applied so it can never drop underground.
                    _clr_obj = float(self._arm_obj_clearance[_ai])
                    if _clr_obj > _COL_DISABLE_SENTINEL:
                        if geom_type == 6:   # BOX
                            _d_obj = _softplus_sphere_box_distance(
                                _gp, _r, _obj_c_dm, _obj_R_dm, ca.DM([hx, hy, hz]))
                        elif geom_type == 5:  # CYLINDER
                            _d_obj = _softplus_sphere_cylinder_distance(
                                _gp, _r, _obj_c_dm, _obj_R_dm,
                                float(geom_size[0]), float(geom_size[1]))
                        else:                 # SPHERE or fallback
                            _obj_r = float(geom_size[0])
                            _d_obj = _sphere_sphere_distance(_gp, _r, _obj_c_dm, _obj_r)
                        _opti.subject_to(_d_obj >= _clr_obj)
                    if cfg.col_use_ground:
                        _opti.subject_to(
                            _sphere_plane_distance(_gp, _r, _ground_p, _ground_n)
                            >= _clr_ground)

            # ── 5b. Thumb + index vs ground (unconditional) ────────────────
            # These two tips are absent from the arm_col loop.
            # They're the contact fingers — the most likely to sink into the table.
            if cfg.col_use_ground:
                for _tp, _r in ((_tp1, float(cfg.r_thumb)),
                                (_tp2, float(cfg.r_index))):
                    _opti.subject_to(
                        _sphere_plane_distance(_tp, _r, _ground_p, _ground_n)
                        >= _clr_ground)

            # ── Initial guess ─────────────────────────────────────────────
            _opti.set_initial(_q,  q_ws)
            _opti.set_initial(_p1, p1_ws)
            _opti.set_initial(_p2, p2_ws)

            # ── Solver ────────────────────────────────────────────────────
            _n_iter = max_iter_override or cfg.max_iter
            if cfg.use_slsqp:
                _sqp_opts = dict(_SQP_SOLVER_OPTS)
                _sqp_opts['max_iter'] = _n_iter
                self.log.info(f"[{stage_label}|solver_opts] sqpmethod  {_sqp_opts}")
                _opti.solver('sqpmethod', _sqp_opts)
            else:
                _ipopt_opts = dict(_IPOPT_SOLVER_OPTS)
                _ipopt_opts['max_iter'] = _n_iter
                if self.log_dir:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    _ipopt_opts['output_file']      = os.path.join(
                        self.log_dir, f"grasp3d_ipopt_{ts}.log")
                    _ipopt_opts['file_print_level'] = 5
                self.log.info(f"[{stage_label}|solver_opts] ipopt  {_ipopt_opts}")
                _opti.solver('ipopt',
                             {'print_time': False},
                             _ipopt_opts)

            # ── Per-iteration callback: normal update + optional logger ───
            # Build log expression list regardless (empty when log_dir is None).
            _log_tag   = f'[{stage_label}|iter] ' if stage_label else '[iter] '
            # Cost components always logged (DEBUG); geometry extras only when log_dir set.
            _log_exprs: list[tuple[str, object]] = [
                ('f',          _opti.f),
                ('ik_th_mm',   ca.sqrt(_d1_sq) * 1e3),
                ('ik_if_mm',   ca.sqrt(_d2_sq) * 1e3),
                ('Jik',        _cost_ik),
                ('Jreg',       _cost_reg),
                ('Jgam',       _cost_gamma),
                ('Jy',         _cost_y),
                ('Jslack',     _cost_slack),
            ]
            if self.log_dir:
                if include_surface:
                    if geom_type == 6:
                        _log_exprs.append(('sdf_p1_mm', _sdf(_p1) * 1e3))
                        _log_exprs.append(('sdf_p2_mm', _sdf(_p2) * 1e3))
                # Per-term gradient norms are full-vector reverse-mode AD
                # evaluations (~110+ DOF) — meaningfully more expensive per
                # iteration than the plain scalar terms above, so unlike
                # those they're only computed when actually being logged
                # (diagnostic/visualization runs), not on every production
                # solve (e.g. test_grasp_recommender.py's live loop).
                _log_exprs.append(('gradN_ik',    _grad_norm_exprs['ik']))
                _log_exprs.append(('gradN_reg',   _grad_norm_exprs['reg']))
                _log_exprs.append(('gradN_gamma', _grad_norm_exprs['gamma']))
                _log_exprs.append(('gradN_y',     _grad_norm_exprs['y']))
                _log_exprs.append(('gradN_slack', _grad_norm_exprs['slack']))
                _log_exprs.append(('gradN_total', _grad_norm_total))

            # Per-iteration logger.  Parameter updates are intentionally absent:
            # calling opti.set_value() here corrupts the L-BFGS curvature pairs
            # and causes numerical divergence.  The iter_callback receives a
            # read-only snapshot of current contact positions and their actual
            # surface normals for diagnostic / visualisation purposes only.
            _d1_frozen = d1_lp if d1_lp is not None else np.zeros(3)
            _d2_frozen = d2_lp if d2_lp is not None else np.zeros(3)

            # Structured per-iteration trace (q/p1/p2/gamma/slack + the cost
            # breakdown above) — saved to grasp3d_iter_<ts>.npz alongside the
            # IPOPT log for the visualizer. None when log_dir is unset.
            _iter_rec: list[dict] | None = [] if self.log_dir else None

            def _opti_cb(i):
                if _log_exprs:
                    parts = [f"i={i:3d}"]
                    _rec_vals = {}
                    for _lbl, _expr in _log_exprs:
                        try:
                            v = float(_opti.debug.value(_expr))
                            parts.append(f"{_lbl}={v:+.3e}")
                            _rec_vals[_lbl] = v
                        except Exception:
                            parts.append(f"{_lbl}=?")
                    self.log.debug(_log_tag + "  ".join(parts))
                    if _iter_rec is not None:
                        try:
                            _rec = dict(_rec_vals)
                            _rec['iter'] = i
                            _rec['q']  = np.asarray(_opti.debug.value(_q),  float).flatten()
                            _rec['p1'] = np.asarray(_opti.debug.value(_p1), float).flatten()
                            _rec['p2'] = np.asarray(_opti.debug.value(_p2), float).flatten()
                            if _gamma_lp is not None:
                                _rec['gamma'] = float(_opti.debug.value(_gamma_lp))
                                if _s_list:
                                    _rec['slack_norm'] = np.array(
                                        [float(np.linalg.norm(_opti.debug.value(sk)))
                                         for sk in _s_list])
                            _iter_rec.append(_rec)
                        except Exception:
                            pass
                if iter_callback is not None or update_normals_in_callback:
                    try:
                        _p1v = np.asarray(_opti.debug.value(_p1), float).flatten()
                        _p2v = np.asarray(_opti.debug.value(_p2), float).flatten()
                        _n1a = _geom_normal_np(_p1v, geom_type,
                                               obj_center_np, obj_R_np, geom_size)
                        _n2a = _geom_normal_np(_p2v, geom_type,
                                               obj_center_np, obj_R_np, geom_size)
                        _cos1 = float(np.clip(np.dot(_d1_frozen, _n1a), -1.0, 1.0))
                        _cos2 = float(np.clip(np.dot(_d2_frozen, _n2a), -1.0, 1.0))
                        _mismatch = max(np.degrees(np.arccos(_cos1)),
                                        np.degrees(np.arccos(_cos2)))
                        if update_normals_in_callback and _R1_param is not None:
                            # Deliberately update the wrench-frame parameters mid-solve.
                            # This re-linearizes the contact frame every SQP/IPOPT
                            # iteration, which corrupts the L-BFGS curvature pairs
                            # (y = ∇f_new − ∇f_old is inconsistent when f changed
                            # between iterates due to the parameter shift).
                            _new_R1 = np.column_stack(
                                _build_contact_frame_3d(-_n1a))
                            _new_R2 = np.column_stack(
                                _build_contact_frame_3d(-_n2a))
                            _opti.set_value(_R1_param, _new_R1)
                            _opti.set_value(_R2_param, _new_R2)
                            _d1_frozen[:] = _n1a
                            _d2_frozen[:] = _n2a
                        if iter_callback is not None:
                            iter_callback({
                                'iter':         i,
                                'stage':        stage_label,
                                'p1':           _p1v,
                                'p2':           _p2v,
                                'n1_actual':    _n1a,
                                'n2_actual':    _n2a,
                                'n1_frozen':    _d1_frozen.copy(),
                                'n2_frozen':    _d2_frozen.copy(),
                                'mismatch_deg': _mismatch,
                                'cost':         float(_opti.debug.value(_opti.f)),
                            })
                    except Exception:
                        pass

            _opti.callback(_opti_cb)

            # ── Profile helper ────────────────────────────────────────────
            def _plog(stats=None):
                t_now    = time.perf_counter()
                dt_graph = _t_solve_start - _t_stage_start
                dt_solve = t_now - _t_solve_start
                dt_total = t_now - _t_stage_start
                tag      = f'[{stage_label}] ' if stage_label else ''

                th_evals  = thumb_cb.eval_count
                if_evals  = index_cb.eval_count
                arm_evals = arm_col_cb.eval_count if arm_col_cb is not None else 0
                _wrench_str = 'embedded' if cfg.wrench_constraint else 'N'

                con_str = (
                    f"surface={'Y' if include_surface else 'N'}  "
                    f"wrench={_wrench_str}  "
                    f"arm_col={'Y(' + str(len(_active_arm)) + 'geoms)' if _active_arm else 'N'}")
                lines = [
                    f"{tag}--- solve profile ------------------------------------------",
                    f"{tag}  constraints : {con_str}",
                    f"{tag}  DOF={n_act}  cbs:"
                    f"  thumb={th_evals}  index={if_evals}"
                    f"  arm={arm_evals}",
                    f"{tag}  graph_build : {dt_graph*1e3:6.0f} ms",
                    f"{tag}  {'SQP' if cfg.use_slsqp else 'IPOPT'}_solve : {dt_solve*1e3:6.0f} ms",
                    f"{tag}  total_wall  : {dt_total*1e3:6.0f} ms",
                ]
                if stats:
                    n_iters   = max(stats.get('iter_count', 0), 1)
                    tw_solver = stats.get('t_wall_solver', 0.0)
                    lines += [
                        f"{tag}  iters={n_iters}  ms/iter={dt_solve*1e3/n_iters:.0f}",
                        f"{tag}  t_wall_solver={tw_solver*1e3:.0f}ms",
                    ]
                lines.append(f"{tag}------------------------------------------------------------")
                for ln in lines:
                    self.log.info(ln)
                if cfg.verbose_profile:
                    for ln in lines:
                        print(ln)

            # ── Solve ─────────────────────────────────────────────────────
            _t_solve_start = time.perf_counter()

            def _cost_breakdown(val_fn) -> str:
                """One-line weighted cost summary for INFO logging."""
                def _v(expr):
                    try:    return float(val_fn(expr))
                    except: return float('nan')
                ik   = _v(_cost_ik);   reg  = _v(_cost_reg)
                gam  = _v(_cost_gamma); y = _v(_cost_y); slack = _v(_cost_slack)
                f    = _v(_opti.f)
                _w_slack_disp = cfg.w_slack if cfg.w_slack is not None else 0.0
                return (
                    f"f={f:.4f}  "
                    f"ik={cfg.w_ik*ik:.4f}({ik:.3f})  "
                    f"reg={cfg.w_reg*reg:.4f}({reg:.3f})  "
                    f"γ={cfg.w_gamma*gam:.4f}({gam:.3f})  "
                    f"y={cfg.w_y*y:.4f}({y:.3f})  "
                    f"slack={_w_slack_disp*slack:.4f}({slack:.3f})"
                )

            def _torque_diagnostic(val_fn) -> str:
                """
                Geometric torque from unit normal forces at the solution contacts.
                τ_geom = (p1-c)×n1_out + (p2-c)×n2_out  (assuming N=1 at each tip).
                Compare against [_nlp_tx, _nlp_ty, _nlp_tz] to check whether the
                height difference between contacts makes the torque bound binding.
                """
                try:
                    _p1v = np.asarray(val_fn(_p1), float)
                    _p2v = np.asarray(val_fn(_p2), float)
                    _n1o = _geom_normal_np(_p1v, geom_type, obj_center_np, obj_R_np, geom_size)
                    _n2o = _geom_normal_np(_p2v, geom_type, obj_center_np, obj_R_np, geom_size)
                    _r1  = _p1v - obj_center_np
                    _r2  = _p2v - obj_center_np
                    _tau = np.cross(_r1, _n1o) + np.cross(_r2, _n2o)
                    _bnd = np.array([_nlp_tx, _nlp_ty, _nlp_tz])
                    _frac = np.abs(_tau) / np.maximum(_bnd, 1e-9)
                    return (
                        f"τ_geom=[{_tau[0]:+.4f},{_tau[1]:+.4f},{_tau[2]:+.4f}]N·m  "
                        f"budget=[{_bnd[0]:.4f},{_bnd[1]:.4f},{_bnd[2]:.4f}]N·m  "
                        f"fill%=[{_frac[0]*100:.0f},{_frac[1]*100:.0f},{_frac[2]*100:.0f}]%  "
                        f"dz={(_p2v-_p1v)[2]*1e3:+.1f}mm"
                    )
                except Exception as _te:
                    return f"torque_diag_err={_te}"

            def _save_iter_npz(status_tag: str):
                """Dump the per-iteration trace collected by _opti_cb to
                grasp3d_iter_<ts>.npz, paired with the IPOPT log of the same
                timestamp (ts is set above when self.log_dir is truthy)."""
                if not (self.log_dir and _iter_rec):
                    return
                try:
                    npz_path = os.path.join(self.log_dir, f"grasp3d_iter_{ts}.npz")
                    _out = {
                        'iter':        np.array([r['iter'] for r in _iter_rec]),
                        'q':           np.stack([r['q']  for r in _iter_rec]),
                        'p1':          np.stack([r['p1'] for r in _iter_rec]),
                        'p2':          np.stack([r['p2'] for r in _iter_rec]),
                        'stage_label': stage_label,
                        'status_tag':  status_tag,
                        'geom_type':   geom_type,
                        'geom_size':   np.asarray(geom_size, float),
                        'obj_center':  np.asarray(obj_center_np, float),
                        'obj_mat':     np.asarray(obj_R_np, float),
                    }
                    for _key in ('f', 'ik_th_mm', 'ik_if_mm', 'Jik', 'Jreg',
                                 'Jgam', 'Jy', 'Jslack', 'sdf_p1_mm', 'sdf_p2_mm',
                                 'gamma', 'gradN_ik', 'gradN_reg', 'gradN_gamma',
                                 'gradN_y', 'gradN_slack', 'gradN_total'):
                        if _key in _iter_rec[0]:
                            _out[_key] = np.array(
                                [r.get(_key, np.nan) for r in _iter_rec])
                    if 'slack_norm' in _iter_rec[0]:
                        _out['slack_norm'] = np.stack(
                            [r['slack_norm'] for r in _iter_rec])
                    np.savez(npz_path, **_out)
                    self.log.info(f"[{stage_label}] iter trace saved -> {npz_path}")
                except Exception as _e_npz:
                    self.log.warning(
                        f"GraspPlanner3D._run_stage: failed to save iter npz: {_e_npz}")

            def _stability_diag(window: int = 20) -> dict | None:
                """Min/max/std of ik_th_mm/ik_if_mm over the last `window`
                recorded iterations. A non-converged solve's reported result
                is just opti.debug.value at whatever iteration the budget ran
                out on — this lets a genuinely settled result be told apart
                from one that landed on a lucky snapshot mid-oscillation."""
                if not _iter_rec:
                    return None
                _last = _iter_rec[-window:]
                _th = np.array([r.get('ik_th_mm', np.nan) for r in _last], float)
                _if = np.array([r.get('ik_if_mm', np.nan) for r in _last], float)
                return {
                    'n':            len(_last),
                    'ik_th_mm_min': float(np.nanmin(_th)),
                    'ik_th_mm_max': float(np.nanmax(_th)),
                    'ik_th_mm_std': float(np.nanstd(_th)),
                    'ik_if_mm_min': float(np.nanmin(_if)),
                    'ik_if_mm_max': float(np.nanmax(_if)),
                    'ik_if_mm_std': float(np.nanstd(_if)),
                }

            def _stability_str(_st: dict | None) -> str:
                if _st is None:
                    return "no iter_rec"
                return (
                    f"n={_st['n']}  "
                    f"ik_th_mm[min={_st['ik_th_mm_min']:.2f} max={_st['ik_th_mm_max']:.2f} "
                    f"std={_st['ik_th_mm_std']:.2f}]  "
                    f"ik_if_mm[min={_st['ik_if_mm_min']:.2f} max={_st['ik_if_mm_max']:.2f} "
                    f"std={_st['ik_if_mm_std']:.2f}]"
                )

            try:
                _sol = _opti.solve()
                _plog(_sol.stats())
                _tag = f'[{stage_label}|cost] ' if stage_label else '[cost] '
                self.log.info(_tag + _cost_breakdown(_sol.value))
                self.log.info(f'[{stage_label}|torque] ' + _torque_diagnostic(_sol.value))
                _stab = _stability_diag()
                self.log.info(f'[{stage_label}|stability] ' + _stability_str(_stab))
                _save_iter_npz('converged')
                return {
                    'success':    True,
                    'q':          _sol.value(_q),
                    'p1':         _sol.value(_p1),
                    'p2':         _sol.value(_p2),
                    'cost':       float(_sol.value(_opti.f)),
                    'iterations': _sol.stats()['iter_count'],
                    'status':     'converged',
                    'gamma_nlp':     float(_sol.value(_gamma_lp)) if _gamma_lp is not None else None,
                    'slack_norms':   ([float(np.linalg.norm(_sol.value(sk))) for sk in _s_list]
                                       if _s_list else None),
                    'max_slack_norm': (float(max(np.linalg.norm(_sol.value(sk)) for sk in _s_list))
                                        if _s_list else None),
                    'n1_frozen':     _n1_in.tolist() if _n1_in is not None else None,
                    'n2_frozen':     _n2_in.tolist() if _n2_in is not None else None,
                    'stability_last20': _stab,
                }
            except Exception as _e:
                self.log.warning(f"GraspPlanner3D._run_stage({stage_label}): {_e}")
                try:    _st = _opti.stats()
                except: _st = None
                _plog(_st)
                try:
                    _tag = f'[{stage_label}|cost] ' if stage_label else '[cost] '
                    self.log.info(_tag + _cost_breakdown(_opti.debug.value))
                    self.log.info(f'[{stage_label}|torque] ' + _torque_diagnostic(_opti.debug.value))
                    _stab = _stability_diag()
                    self.log.info(f'[{stage_label}|stability] ' + _stability_str(_stab))
                    _save_iter_npz('best-effort')
                    return {
                        'success':    False,
                        'q':          _opti.debug.value(_q),
                        'p1':         _opti.debug.value(_p1),
                        'p2':         _opti.debug.value(_p2),
                        'cost':       _opti.debug.value(_opti.f) if _st else None,
                        'iterations': (_st or {}).get('iter_count'),
                        'status':     'best-effort',
                        'gamma_nlp':     float(_opti.debug.value(_gamma_lp)) if _gamma_lp is not None else None,
                        'slack_norms':   ([float(np.linalg.norm(_opti.debug.value(sk))) for sk in _s_list]
                                           if _s_list else None),
                        'max_slack_norm': (float(max(np.linalg.norm(_opti.debug.value(sk)) for sk in _s_list))
                                            if _s_list else None),
                        'n1_frozen':     _n1_in.tolist() if _n1_in is not None else None,
                        'n2_frozen':     _n2_in.tolist() if _n2_in is not None else None,
                        'stability_last20': _stab,
                    }
                except Exception as _e2:
                    self.log.error(f"GraspPlanner3D debug extraction: {_e2}")
                    _save_iter_npz('failed')
                    return {'success': False, 'q': None, 'p1': None, 'p2': None,
                            'cost': None, 'iterations': None, 'status': 'failed',
                            'gamma_nlp': None, 'slack_norms': None, 'max_slack_norm': None}

        # ── Run optimisation ─────────────────────────────────────────────────
        # Outer re-linearisation loop.  Contact normals are frozen per NLP solve
        # (opti.parameter — safe for L-BFGS), then updated between solves.
        # Early-exit when the contact positions have converged (position shift
        # < tol_p) or the normal mismatch is negligible (< tol_deg).
        _d1_lp = (np.asarray(d1, float) if d1 is not None
                  else _geom_normal_np(p1_seed, geom_type, obj_center_np, obj_R_np, geom_size))
        _d2_lp = (np.asarray(d2, float) if d2 is not None
                  else _geom_normal_np(p2_seed, geom_type, obj_center_np, obj_R_np, geom_size))
        _p1_ws, _p2_ws, _q_ws = p1_seed, p2_seed, q_dls
        _tol_p_m   = 5e-4    # 0.5 mm position shift → converged
        _tol_deg   = 2.0     # 2° normal mismatch → normals are accurate enough
        res = {}
        _best_res  = {}      # best stage result by cost (Picard has no descent guarantee)
        # Box (geom_type 6) needs NO Picard relinearization: the face-pin surface
        # constraint keeps each contact on its seed face, where the inward normal is
        # CONSTANT (a flat face doesn't rotate as the contact slides), so the frozen
        # normal set at seed time is already exact for the whole solve. Curved surfaces
        # (sphere/cylinder) still relinearize since their normals genuinely turn with p.
        _n_relin = 0 if geom_type == 6 else cfg.n_normal_relinearize
        for _ri in range(_n_relin + 1):
            res = _run_stage(_q_ws, _p1_ws, _p2_ws,
                             include_surface=True,
                             d1_lp=_d1_lp,
                             d2_lp=_d2_lp,
                             max_iter_override=cfg.max_iter,
                             stage_label=f'S{_ri+1}',
                             iter_callback=iter_callback,
                             update_normals_in_callback=update_normals_in_callback)
            # Keep the cheapest stage result — relinearization has no descent guarantee.
            if (res.get('cost') is not None and
                    (not _best_res or res['cost'] < _best_res.get('cost', float('inf')))):
                _best_res = res
            if _ri >= _n_relin or res.get('p1') is None:
                break

            _p1r = np.asarray(res['p1'])
            _p2r = np.asarray(res['p2'])

            # ── Convergence checks ────────────────────────────────────────
            # 1. Position shift (trust-region proxy): how far did p move?
            _dp = max(np.linalg.norm(_p1r - _p1_ws),
                      np.linalg.norm(_p2r - _p2_ws))

            # 2. Normal mismatch: angle between frozen normal (used this solve)
            #    and actual surface normal at the new contact position.
            _n1_actual = _geom_normal_np(
                _p1r, geom_type, obj_center_np, obj_R_np, geom_size)
            _n2_actual = _geom_normal_np(
                _p2r, geom_type, obj_center_np, obj_R_np, geom_size)
            _cos1 = float(np.clip(np.dot(_d1_lp, _n1_actual), -1.0, 1.0))
            _cos2 = float(np.clip(np.dot(_d2_lp, _n2_actual), -1.0, 1.0))
            _mismatch_deg = max(np.degrees(np.arccos(_cos1)),
                                np.degrees(np.arccos(_cos2)))

            self.log.info(
                f"[relinearize S{_ri+1}→S{_ri+2}] "
                f"dp={_dp*1e3:.2f}mm  mismatch={_mismatch_deg:.1f}°")

            if _dp < _tol_p_m and _mismatch_deg < _tol_deg:
                self.log.info(
                    f"[relinearize] converged after S{_ri+1} "
                    f"(dp={_dp*1e3:.2f}mm, mismatch={_mismatch_deg:.1f}°)")
                break

            # Update normals and warm-start for next solve
            _d1_lp = _n1_actual
            _d2_lp = _n2_actual
            _p1_ws = _p1r
            _p2_ws = _p2r
            if res.get('q') is not None:
                _q_ws = np.asarray(res['q'])

        res = _best_res if _best_res else res   # use cheapest stage, not necessarily last
        if self.dash is not None:
            self.dash.push({
                'type':   'ipopt',
                'phase':  'grasp3d',
                'status': res.get('status', '?'),
                'iters':  res.get('iterations', '?'),
            })
        if res.get('p1') is not None:
            _p1f  = np.asarray(res['p1'])
            _p2f  = np.asarray(res['p2'])
            _n1f  = _geom_normal_np(_p1f, geom_type, obj_center_np, obj_R_np, geom_size)
            _n2f  = _geom_normal_np(_p2f, geom_type, obj_center_np, obj_R_np, geom_size)
            _dot12 = float(np.dot(_n1f, _n2f))
            _sm    = _span_margin(_n1f, _n2f, _mu)
            self.log.info(
                f"[solve|final] n1={np.round(_n1f, 3).tolist()}  n2={np.round(_n2f, 3).tolist()}  "
                f"dot={_dot12:+.3f}  span_margin={_sm:+.4f}rad")
            res['n1_final'] = _n1f.tolist()
            res['n2_final'] = _n2f.tolist()
            res['span_margin_final'] = _sm
        return res

    def verify(self, result: dict) -> dict:
        """Post-solve sanity check: IK residuals, geom gaps, LP wrench feasibility."""
        if result.get('q') is None:
            return {}
        model  = self.model
        data_v = mj.MjData(model)
        data_v.qpos[:] = self.data.qpos[:]
        for idx, val in zip(self._act_idx, result['q']):
            data_v.qpos[idx] = val
        if self._has_markers:
            data_v.mocap_pos[self._cp1_mocap] = result['p1']
            data_v.mocap_pos[self._cp2_mocap] = result['p2']
        mj.mj_forward(model, data_v)

        obj_pos = data_v.geom_xpos[self._obj_gid].copy()
        obj_mat = data_v.geom_xmat[self._obj_gid].reshape(3, 3)

        # IK residual uses the same offset target as the NLP: p + r·n_out
        # Measuring ‖site − p‖ would always return ~r_tip regardless of convergence.
        _p1_np  = np.asarray(result['p1'], float)
        _p2_np  = np.asarray(result['p2'], float)
        _n1_out = _geom_normal_np(_p1_np, self._obj_geom_type, obj_pos, obj_mat, self._obj_size)
        _n2_out = _geom_normal_np(_p2_np, self._obj_geom_type, obj_pos, obj_mat, self._obj_size)
        _tgt1   = _p1_np + self.cfg.r_thumb * _n1_out
        _tgt2   = _p2_np + self.cfg.r_index * _n2_out
        ik_t = float(np.linalg.norm(data_v.site_xpos[self._thumb_sid] - _tgt1))
        ik_i = float(np.linalg.norm(data_v.site_xpos[self._index_sid] - _tgt2))

        gap_t = mj.mj_geomDistance(model, data_v, self._thumb_gid,  self._obj_gid, 0.5, None)
        gap_i = mj.mj_geomDistance(model, data_v, self._index_gid,  self._obj_gid, 0.5, None)
        gap_m = mj.mj_geomDistance(model, data_v, self._middle_gid, self._obj_gid, 0.5, None)
        gap_r = mj.mj_geomDistance(model, data_v, self._ring_gid,   self._obj_gid, 0.5, None)

        # Floor gaps: geom_xpos[gid][2] is geom-center z; subtract radius and ground_z.
        # Negative = penetration.
        _gz = float(self.cfg.ground_z)
        gap_floor_t = float(data_v.geom_xpos[self._thumb_gid][2])  - float(self.cfg.r_thumb)  - _gz
        gap_floor_i = float(data_v.geom_xpos[self._index_gid][2])  - float(self.cfg.r_index)  - _gz
        gap_floor_m = float(data_v.geom_xpos[self._middle_gid][2]) - float(self.cfg.r_middle) - _gz
        gap_floor_r = float(data_v.geom_xpos[self._ring_gid][2])   - float(self.cfg.r_ring)   - _gz

        def _sdf3(p):
            return _geom_sdf_np(p, self._obj_geom_type, obj_pos, obj_mat, self._obj_size)

        s1 = _sdf3(result['p1'])
        s2 = _sdf3(result['p2'])
        s3 = _sdf3(data_v.geom_xpos[self._middle_gid])
        s4 = _sdf3(data_v.geom_xpos[self._ring_gid])

        # Wrench feasibility (post-solve LP for debugging/analysis)
        cfg            = self.cfg
        gamma_min      = None
        max_slack_norm = None
        wf_feasible    = False
        wf_tag         = 'SKIP'
        n1_out = n2_out = None
        try:
            if (_NCF_AVAILABLE
                    and result.get('p1') is not None
                    and result.get('p2') is not None):
                p1_np = np.asarray(result['p1'], float)
                p2_np = np.asarray(result['p2'], float)
                n1_out = _geom_normal_np(p1_np, self._obj_geom_type,
                                          obj_pos, obj_mat, self._obj_size)
                n2_out = _geom_normal_np(p2_np, self._obj_geom_type,
                                          obj_pos, obj_mat, self._obj_size)
                _, t1_1, t2_1 = _build_contact_frame_3d(-n1_out)
                _, t1_2, t2_2 = _build_contact_frame_3d(-n2_out)
                R1 = np.column_stack([-n1_out, t1_1, t2_1])
                R2 = np.column_stack([-n2_out, t1_2, t2_2])
                # Mass-scaled gamma — same approach as solve_gamma_live in
                # kinova_leap_pick_place.py. Contacts expressed in object body frame.
                _bid_v   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.obj_body)
                _mass_v  = float(model.body_mass[_bid_v])
                _inert_v = model.body_inertia[_bid_v]
                _aab_v   = cfg.ang_accel_budget_xyz
                _mu_v    = round(0.8 * float(model.geom_friction[self._obj_gid][0]), 3)
                R_WO_v   = data_v.xmat[_bid_v].reshape(3, 3)
                _g_O_v   = R_WO_v.T @ model.opt.gravity
                _ab_v    = cfg.accel_budget_xyz
                _p1_O_v  = R_WO_v.T @ (p1_np - obj_pos)
                _p2_O_v  = R_WO_v.T @ (p2_np - obj_pos)
                R1_O     = R_WO_v.T @ R1
                R2_O     = R_WO_v.T @ R2
                _pos_v   = [_p1_O_v.reshape(3, 1), _p2_O_v.reshape(3, 1)]
                # Torque box = I * angular budget (principal axes).
                _T_v = np.array([float(_inert_v[i]) * _aab_v[i] for i in range(3)])
                if cfg.datum_gamma:
                    # Datum / Task-B: match solve_gamma_live exactly. Reference the
                    # disturbance at the grasp midpoint, add gravity as a separate
                    # re-datumed / grasp-axis-projected wrench, and let the LP project the
                    # grasp-axis torque out of EACH corner (project_grasp_axis_torque) — the
                    # exact per-corner removal, so the FULL per-axis budget _T_v is passed
                    # (no lossy budget-vector pre-projection). Accel box is a PURE force box.
                    _mref_v = (0.5 * (_p1_O_v + _p2_O_v)).reshape(3)
                    _grav_v = _mass_v * _g_O_v
                    # HARD LP (no slack): returns a single γ or None — a true feasibility
                    # gate identical to solve_gamma_live. max_slack_norm is N/A here.
                    gamma_min = min_gamma_for_accel_lp_hard(
                        _mass_v * _ab_v[0], _mass_v * _ab_v[1], _mass_v * _ab_v[2],
                        _T_v[0], _T_v[1], _T_v[2],
                        n=2, pos=_pos_v, R=[R1_O, R2_O],
                        ncf=[1.0, 1.0], tan_y=[0.0, 0.0], tan_z=[0.0, 0.0],
                        mu=[_mu_v, _mu_v],
                        moment_ref=_mref_v, grav_force=_grav_v,
                        project_grasp_axis_moment=True,
                        project_grasp_axis_torque=True,
                    )
                    max_slack_norm = None
                else:
                    # Legacy CoM / Task-A: gravity folded into the accel budget.
                    _accel_v = tuple(_ab_v[i] + abs(_g_O_v[i]) for i in range(3))
                    gamma_min, max_slack_norm = min_gamma_for_accel_lp(
                        _mass_v * _accel_v[0], _mass_v * _accel_v[1], _mass_v * _accel_v[2],
                        _T_v[0], _T_v[1], _T_v[2],
                        n=2, pos=_pos_v, R=[R1_O, R2_O],
                        ncf=[1.0, 1.0], tan_y=[0.0, 0.0], tan_z=[0.0, 0.0],
                        mu=[_mu_v, _mu_v],
                        slack_penalty=cfg.verify_slack_penalty,
                    )
                wf_feasible = (gamma_min is not None)
                _slack_bad = (max_slack_norm is not None
                              and max_slack_norm > cfg.verify_slack_tol)
                if _slack_bad:
                    wf_tag = f'SLACK(γ_min={gamma_min:.3f}, slack={max_slack_norm:.4f})'
                elif wf_feasible:
                    wf_tag = f'OK(γ_min={gamma_min:.3f})'
                else:
                    wf_tag = 'INFEASIBLE'
        except Exception as _e:
            wf_tag = f'ERROR: {_e}'

        info = {
            'ik_thumb_mm':          ik_t * 1000,
            'ik_index_mm':          ik_i * 1000,
            'gap_thumb_mm':         gap_t * 1000,
            'gap_index_mm':         gap_i * 1000,
            'gap_middle_mm':        gap_m * 1000,
            'gap_ring_mm':          gap_r * 1000,
            'gap_floor_thumb_mm':   gap_floor_t * 1000,
            'gap_floor_index_mm':   gap_floor_i * 1000,
            'gap_floor_middle_mm':  gap_floor_m * 1000,
            'gap_floor_ring_mm':    gap_floor_r * 1000,
            'sdf_p1_mm':            s1   * 1000,
            'sdf_p2_mm':            s2   * 1000,
            'sdf_middle_tip_mm':    s3   * 1000,
            'sdf_ring_tip_mm':      s4   * 1000,
            'wrench_feasible':      wf_feasible,
            'gamma_min':            gamma_min,
            'max_slack_norm':       max_slack_norm,
            'gamma_nlp':            result.get('gamma_nlp'),
            'n1_verify':            n1_out.tolist() if n1_out is not None else None,
            'n2_verify':            n2_out.tolist() if n2_out is not None else None,
        }
        self.log.info(
            f"[verify3d] IK=({ik_t*1e3:.2f},{ik_i*1e3:.2f})mm "
            f"GAP_obj=({gap_t*1e3:+.2f},{gap_i*1e3:+.2f},"
            f"{gap_m*1e3:+.2f},{gap_r*1e3:+.2f})mm "
            f"GAP_floor=({gap_floor_t*1e3:+.2f},{gap_floor_i*1e3:+.2f},"
            f"{gap_floor_m*1e3:+.2f},{gap_floor_r*1e3:+.2f})mm "
            f"WF={wf_tag}")
        return info

    # ── private ────────────────────────────────────────────────────────────────

    def _require_geom(self, name: str) -> int:
        gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
        if gid == -1:
            raise ValueError(
                f"GraspPlanner3D: geom '{name}' not found. "
                f"Check GraspConfig3D geometry name fields.")
        return gid

    def _optional_geom(self, name: str):
        gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
        return int(gid) if gid != -1 else None

    def _require_site(self, name: str) -> int:
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, name)
        if sid == -1:
            raise ValueError(
                f"GraspPlanner3D: site '{name}' not found. "
                f"Check GraspConfig3D thumb_site / index_site fields.")
        return sid


# ─────────────────────────────────────────────────────────────────────────────
# Multi-start wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MultiStartGraspPlanner3D:
    """
    Runs GraspPlanner3D from each of the shape-appropriate canonical seeds
    and returns the best result ranked by: converged > best-effort, then cost.

    Parameters
    ----------
    model, data, cfg, logger, log_dir : forwarded to GraspPlanner3D
    """

    def __init__(self, model, data,
                 cfg: GraspConfig3D | None = None,
                 logger=None, log_dir: str | None = None,
                 dashboard=None):
        self._planner       = GraspPlanner3D(model, data, cfg, logger, log_dir,
                                             dashboard=dashboard)
        self._obj_hx        = self._planner._obj_hx
        self._obj_hy        = self._planner._obj_hy
        self._obj_hz        = self._planner._obj_hz
        self._obj_geom_type = self._planner._obj_geom_type
        self._obj_size      = self._planner._obj_size
        self._fk_data       = mj.MjData(model)           # FK queries for seed generation
        self._rng           = np.random.default_rng(42)  # reproducible random seeds
        # Fingertip effective radii (r_thumb/r_index/r_middle/r_ring) are
        # measured from model geometry inside GraspPlanner3D.__init__ above —
        # nothing left to do here.
        _pl = self._planner

        # Log friction and torque bounds that will be computed inline at solve time.
        _obj_gid = _pl._obj_gid
        _mu_obj  = float(model.geom_friction[_obj_gid][0])
        _mu_init = round(0.8 * _mu_obj, 3)
        _pl.log.info(
            f"[friction] obj_mu={_mu_obj:.3f}  effective_mu={_mu_init:.3f} (0.8x safety margin)")

        _bid_init   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _pl.cfg.obj_body)
        _inert_init = model.body_inertia[_bid_init]
        _aab_init   = _pl.cfg.ang_accel_budget_xyz
        _tx = float(_inert_init[0]) * _aab_init[0]
        _ty = float(_inert_init[1]) * _aab_init[1]
        _tz = float(_inert_init[2]) * _aab_init[2]
        _pl.log.info(
            f"[task_torque] tx={_tx:.4f}  ty={_ty:.4f}  tz={_tz:.4f} N·m  "
            f"(I={np.round(_inert_init,6).tolist()}  α={list(_aab_init)})")

    def solve(self, q_ref: np.ndarray, obj_pos: np.ndarray,
              max_seeds: int | None = None,
              warm_contacts: tuple | None = None) -> dict:
        """Try _seed_pair seeds (up to max_seeds), return best result ranked by cost.

        warm_contacts : optional (p1_world, p2_world) from a PRIOR accepted solve. When
            given, a deterministic seed reconstructed from those contacts (normals recom-
            puted from the current object geometry) is tried FIRST, ahead of the random
            seeds. This anchors a repeated solve on a static object back to the previous
            basin, so the recommended contacts stop jumping between local optima frame to
            frame. It counts against the seed budget; the random seeds still run for
            exploration, and _rank keeps whichever is genuinely best.
        """
        c       = np.asarray(obj_pos, float)
        cfg     = self._planner.cfg
        dash    = self._planner.dash
        model   = self._planner.model
        log     = self._planner.log
        act_idx = self._planner._act_idx

        obj_gid      = self._planner._obj_gid
        geom_type    = self._obj_geom_type
        geom_size    = self._obj_size
        obj_center_np = self._planner.data.geom_xpos[obj_gid].copy()
        obj_R_np      = self._planner.data.geom_xmat[obj_gid].reshape(3, 3).copy()

        # DETERMINISTIC seeding: reset self._rng to a FIXED seed at the START of every solve.
        # Without this the RNG advances across solves, so the same (or a marginally-moved)
        # starting pose draws a DIFFERENT random stream on a re-solve -> different best-of-N
        # winner -> the run-to-run finger/contact flip. Resetting to a constant makes the
        # STREAM of random draws identical every solve, so _seed_pair (which is continuous in
        # the object pose) maps a nearby pose to nearby contacts instead of hopping to an
        # unrelated basin. NOTE: deliberately NOT a hash of the pose — a hash is discontinuous,
        # so sub-mm teleop jitter straddling a quantization boundary would flip the seed and
        # reintroduce the instability; a fixed constant + the pose-continuity of _seed_pair is
        # what actually gives stability under marginal pose changes. Frame-to-frame stickiness
        # on a static object is further handled by the warm-start seed + display hysteresis.
        self._rng = np.random.default_rng(_SEED_RNG_CONST)

        # Operator's LIVE thumb/index tip positions at q_ref, for KINEMATIC finger
        # assignment of each seed pair (below). Each _seed_pair labels its two contacts
        # p1s/p2s by a RANDOM march direction, so which physical contact becomes the THUMB
        # seed (p1, hard-pinned to its face for the solve) vs the INDEX seed (p2) is
        # arbitrary — the source of the run-to-run finger-assignment flip AND of awkward
        # assignments the pinned-face NLP cannot escape. We reassign so p1 goes to whichever
        # contact is nearer the operator's actual thumb tip (and p2 to the index), matching
        # the hand's real geometry instead of a coin flip. FK on _fk_data (the seed-gen
        # buffer): object qpos is already synced into self._planner.data by the caller, so
        # carry it over and overwrite only the actuated robot joints with q_ref.
        _fkd = self._fk_data
        _fkd.qpos[:] = self._planner.data.qpos[:]
        _fkd.qpos[act_idx] = np.asarray(q_ref, float)[:len(act_idx)]
        mj.mj_kinematics(model, _fkd)
        _live_th = _fkd.site_xpos[self._planner._thumb_sid].copy()
        _live_if = _fkd.site_xpos[self._planner._index_sid].copy()

        # Task wrench bounds (mirrors GraspPlanner3D.solve logic)
        _bid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.obj_body)
        _mass  = float(model.body_mass[_bid])
        _inert = model.body_inertia[_bid]
        _g_O   = obj_R_np.T @ model.opt.gravity
        _ab    = cfg.accel_budget_xyz
        _aab   = cfg.ang_accel_budget_xyz
        _mu    = round(0.8 * float(model.geom_friction[obj_gid][0]), 3)
        _nlp_fx = _mass * (_ab[0] + abs(_g_O[0]))
        _nlp_fy = _mass * (_ab[1] + abs(_g_O[1]))
        _nlp_fz = _mass * (_ab[2] + abs(_g_O[2]))
        _nlp_tx = float(_inert[0]) * _aab[0]
        _nlp_ty = float(_inert[1]) * _aab[1]
        _nlp_tz = float(_inert[2]) * _aab[2]

        bbox_r    = float(np.max(geom_size)) * 2.5
        n_seeds   = max_seeds if max_seeds is not None else cfg.n_seeds
        max_attempts = 40 * n_seeds

        def _sdf(p):
            return _geom_sdf_np(p, geom_type, obj_center_np, obj_R_np, geom_size)

        _ground_z  = cfg.ground_z
        _r_tip_min = min(cfg.r_thumb, cfg.r_index)  # conservative: unknown which tip goes where

        _t0 = time.perf_counter()

        # ── Fixed canonical seeds: perfectly antipodal, center-aligned pairs
        # along the object's local x and y axes, tried before the randomized
        # seeds below (counts against the n_seeds budget).
        seeds, attempts, rejected = [], 0, 0
        # Fixed canonical seeds FIRST: perfectly antipodal pairs through the object center
        # along the box's OWN local x and y axes (d_world = obj_mat @ local_axis), so they are
        # opposite-face for ANY object orientation — unlike _seed_pair, whose march is rotated
        # about WORLD z and can exit an adjacent face on a substantially +z-rotated box
        # (contacts ~90deg apart -> NLP converges but verify()'s wrench LP is infeasible). These
        # guarantee >=1 clean antipodal candidate for the ranker to pick on rotated boxes.
        # _assign_seed_by_finger orients thumb/index to the live hand, same as the random seeds.
        for _axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
            if len(seeds) >= n_seeds:
                break
            _fs = _fixed_antipodal_seed(geom_type, geom_size, c, obj_R_np, _axis)
            if (not _reachable_contact(_fs['p1s'], _ground_z, _r_tip_min) or
                    not _reachable_contact(_fs['p2s'], _ground_z, _r_tip_min)):
                log.debug(f"[seed_gen] fixed seed along axis {_axis.tolist()} "
                          f"unreachable — skipped")
                continue
            _assign_seed_by_finger(_fs, _live_th, _live_if)
            seeds.append(_fs)

        # sample seeds for solver
        while len(seeds) < n_seeds and attempts < max_attempts:
            attempts += 1
            s = _seed_pair(geom_type, geom_size, c, obj_R_np, bbox_r, self._rng,
                           delta_max=np.deg2rad(cfg.seed_march_jitter_deg))
            # # Hemisphere check — n1 and n2 must point into opposing hemispheres
            # if float(np.dot(s['n1_in'], s['n2_in'])) >= 0:
            #     rejected += 1
            #     continue
            # Reachability check — reject contacts below the table or facing downward
            n1_out = -s['n1_in'];  n2_out = -s['n2_in']
            if (not _reachable_contact(s['p1s'], _ground_z, _r_tip_min) or
                    not _reachable_contact(s['p2s'], _ground_z, _r_tip_min)):
                rejected += 1
                continue
            _assign_seed_by_finger(s, _live_th, _live_if)
            seeds.append(s)

        if len(seeds) < n_seeds:
            log.warning(
                f"[seed_gen] only {len(seeds)}/{n_seeds} valid seeds after "
                f"{attempts} attempts ({rejected} rejected)")
        log.info(
            f"[seed_gen] {len(seeds)} seeds in {(time.perf_counter()-_t0)*1e3:.1f}ms "
            f"({attempts} attempts, {rejected} rejected)")

        # DEPRECATED — the live caller no longer passes warm_contacts (kept only for API
        # compatibility). Warm-starting from a prior CONVERGED grasp is counterproductive: the
        # solution sits ON the constraint boundary (surface + edge-margin + wrench-cone), so
        # seeding the interior-point NLP jammed against those constraints bounces it into a
        # WORSE basin (measured: a cost-0.14 solution warm-started to cost 3.48), and it
        # displaced a good fresh seed, collapsing convergence (1/3 -> 0/3) on re-solve. Fresh
        # fixed-RNG seeds are deterministic and already return to the same basin on a static
        # object. Do NOT re-enable without re-checking that regression.
        if warm_contacts is not None:
            try:
                _wp1 = _project_to_surface_np(np.asarray(warm_contacts[0], float),
                                              geom_type, c, obj_R_np, geom_size)
                _wp2 = _project_to_surface_np(np.asarray(warm_contacts[1], float),
                                              geom_type, c, obj_R_np, geom_size)
                _wn1 = -_geom_normal_np(_wp1, geom_type, c, obj_R_np, geom_size)
                _wn2 = -_geom_normal_np(_wp2, geom_type, c, obj_R_np, geom_size)
                if (_reachable_contact(_wp1, _ground_z, _r_tip_min) and
                        _reachable_contact(_wp2, _ground_z, _r_tip_min)):
                    _warm_seed = {'p1': _wp1.copy(), 'p2': _wp2.copy(),
                                  'p1s': _wp1, 'p2s': _wp2,
                                  'n1_in': _wn1, 'n2_in': _wn2,
                                  'offsets': (0.0, 0.0), 'delta_deg': 0.0}
                    # Same kinematic finger assignment as the random seeds (the prior
                    # contacts were already assigned, so this normally keeps them).
                    _assign_seed_by_finger(_warm_seed, _live_th, _live_if)
                    if len(seeds) >= n_seeds and seeds:
                        seeds[-1] = _warm_seed        # replace a random seed
                    else:
                        seeds.append(_warm_seed)
                    seeds.insert(0, seeds.pop())      # try the warm seed FIRST
                    log.info("[seed_gen] warm-start seed from prior contacts prepended")
            except Exception as _e:
                log.debug(f"[seed_gen] warm-start seed skipped: {_e}")

        results = []
        for i, seed in enumerate(seeds):
            if dash is not None:
                dash.push({'type': 'active', 'label': f'grasp3d seed {i+1}/{n_seeds}'})

            # ── Pre-check LP on surface footprints (slack-relaxed — see
            #    cfg.precheck_slack_penalty) ───────────────────────────────────
            g_pre = None
            y_by_corner_pre = None
            max_slack_pre = None
            if _NCF_AVAILABLE and min_gamma_for_accel_lp is not None:
                try:
                    _, t1_1, t2_1 = _build_contact_frame_3d(seed['n1_in'])
                    _, t1_2, t2_2 = _build_contact_frame_3d(seed['n2_in'])
                    R1 = np.column_stack([seed['n1_in'], t1_1, t2_1])
                    R2 = np.column_stack([seed['n2_in'], t1_2, t2_2])
                    p1s_O = obj_R_np.T @ (seed['p1s'] - obj_center_np)
                    p2s_O = obj_R_np.T @ (seed['p2s'] - obj_center_np)
                    R1_O  = obj_R_np.T @ R1
                    R2_O  = obj_R_np.T @ R2
                    g_pre, y_by_corner_pre, max_slack_pre, _ = min_gamma_for_accel_lp(
                        _nlp_fx, _nlp_fy, _nlp_fz,
                        _nlp_tx, _nlp_ty, _nlp_tz,
                        n=2,
                        pos=[p1s_O.reshape(3, 1), p2s_O.reshape(3, 1)],
                        R=[R1_O, R2_O],
                        ncf=[1.0, 1.0],
                        tan_y=[0.0, 0.0],
                        tan_z=[0.0, 0.0],
                        mu=[_mu, _mu],
                        return_y=True,
                        slack_penalty=cfg.precheck_slack_penalty,
                    )
                    log.debug(f"[seed {i+1}] pre-check LP γ={g_pre:.3f} "
                              f"slack={max_slack_pre:.4f}")
                except Exception as _lp_e:
                    log.debug(f"[seed {i+1}] pre-check LP error: {_lp_e}")

            # Remove seeds that are obviously infeasible (γ > 500, or the pre-check
            # relies heavily on slack to reach a finite γ) to avoid wasting time on
            # NLP solves.
            g_pre_str = f'{g_pre:.2f}' if g_pre is not None else 'N/A'
            if g_pre is not None and g_pre > 500.0:
                log.debug(
                    f"[seed {i+1}/{n_seeds}] pre-check γ={g_pre:.1f} > 500 — skip  "
                    f"p1s={np.round(seed['p1s'], 4).tolist()} "
                    f"p2s={np.round(seed['p2s'], 4).tolist()}")
                continue

            # ── Run NLP (warm-started from the pre-check LP's γ and cone y's) ──
            r = self._planner.solve(q_ref, obj_pos,
                                    p1_init=seed['p1'],
                                    p2_init=seed['p2'],
                                    d1=-seed['n1_in'],   # outward normal
                                    d2=-seed['n2_in'],
                                    gamma_init=g_pre,
                                    y_by_corner_init=y_by_corner_pre)

            # ── Post-solve diagnostics ────────────────────────────────────────
            sdf_p1 = sdf_p2 = float('nan')
            ik_th  = ik_if  = float('nan')
            if r.get('p1') is not None:
                _p1f = np.asarray(r['p1'])
                _p2f = np.asarray(r['p2'])
                sdf_p1 = _geom_sdf_np(_p1f, geom_type, obj_center_np, obj_R_np, geom_size)
                sdf_p2 = _geom_sdf_np(_p2f, geom_type, obj_center_np, obj_R_np, geom_size)
                if r.get('q') is not None:
                    self._fk_data.qpos[act_idx] = np.asarray(r['q'], float)[:len(act_idx)]
                    mj.mj_kinematics(model, self._fk_data)
                    # Use offset target p + r·n_out to match the NLP objective.
                    _n1_out = _geom_normal_np(_p1f, geom_type, obj_center_np, obj_R_np, geom_size)
                    _n2_out = _geom_normal_np(_p2f, geom_type, obj_center_np, obj_R_np, geom_size)
                    _tgt1   = _p1f + cfg.r_thumb * _n1_out
                    _tgt2   = _p2f + cfg.r_index * _n2_out
                    ik_th = float(np.linalg.norm(
                        self._fk_data.site_xpos[self._planner._thumb_sid] - _tgt1)) * 1e3
                    ik_if = float(np.linalg.norm(
                        self._fk_data.site_xpos[self._planner._index_sid] - _tgt2)) * 1e3

                # The embedded LP is a slack-relaxed, corner-coupled equality
                # (matches verify()'s min_gamma_for_accel_lp slack_penalty mode) —
                # solver convergence alone no longer certifies exact wrench
                # feasibility, since a corner's slack can absorb an unresolved
                # residual. wrench_ok additionally requires that residual be
                # negligible (cfg.slack_tol_abs).
                _max_slack = r.get('max_slack_norm')
                _slack_ok  = (_max_slack is None) or (_max_slack < cfg.slack_tol_abs)
                r['wrench_ok'] = (r.get('status') == 'converged') and _slack_ok

            o1, o2 = seed['offsets']
            _n1_ang_sf = _angle_deg_between(seed.get('n1_in'), r.get('n1_frozen'))
            _n1_ang_ff = _angle_deg_between(r.get('n1_frozen'), r.get('n1_final'))
            _n2_ang_sf = _angle_deg_between(seed.get('n2_in'), r.get('n2_frozen'))
            _n2_ang_ff = _angle_deg_between(r.get('n2_frozen'), r.get('n2_final'))
            log.info(
                f"[seed {i+1}/{n_seeds}] "
                f"p1s={np.round(seed['p1s'], 4).tolist()} "
                f"p2s={np.round(seed['p2s'], 4).tolist()} "
                f"o=({o1*1e3:+.1f},{o2*1e3:+.1f})mm δ={seed['delta_deg']:.0f}° "
                f"γ_pre={g_pre_str} γ_nlp={r.get('gamma_nlp') or float('nan'):.3f} "
                f"wrench_ok={r.get('wrench_ok')} "
                f"sdf=({sdf_p1*1e3:.2f},{sdf_p2*1e3:.2f})mm "
                f"IK=({ik_th:.1f},{ik_if:.1f})mm "
                f"n1(seed→frozen→final)={_n1_ang_sf:.1f}°/{_n1_ang_ff:.1f}° "
                f"n2(seed→frozen→final)={_n2_ang_sf:.1f}°/{_n2_ang_ff:.1f}° "
                f"iters={r.get('iterations', '?')} → {r.get('status', '?')}")

            r['p1_seed']  = seed['p1s'].copy()
            r['p2_seed']  = seed['p2s'].copy()
            r['seed_meta'] = seed
            results.append(r)

        if not results:
            return {'success': False, 'q': None, 'p1': None, 'p2': None,
                    'cost': None, 'status': 'failed', 'all_results': []}

        def _rank(r):
            ok = (r.get('p1') is not None
                  and r.get('status') != 'failed'
                  and r.get('wrench_ok', True))
            return (0 if ok else 1, r.get('cost') or 1e9)

        results.sort(key=_rank)
        best = results[0]
        best['all_results'] = results

        # ── Summary: seed vs. final contact-normal divergence for the best result ──
        _best_seed = best.get('seed_meta') or {}
        _n1_seed, _n1_final = _best_seed.get('n1_in'), best.get('n1_final')
        _n2_seed, _n2_final = _best_seed.get('n2_in'), best.get('n2_final')
        _n1_div = _angle_deg_between(_n1_seed, _n1_final)
        _n2_div = _angle_deg_between(_n2_seed, _n2_final)
        best['n1_seed']      = _n1_seed
        best['n2_seed']      = _n2_seed
        best['n1_divergence_deg'] = _n1_div
        best['n2_divergence_deg'] = _n2_div
        log.info(
            f"[summary] best seed idx={results.index(best)+1}/{len(results)} "
            f"n1(seed→final)={np.round(_n1_seed, 3).tolist() if _n1_seed is not None else None}"
            f"→{np.round(_n1_final, 3).tolist() if _n1_final is not None else None} "
            f"Δ={_n1_div:.1f}° "
            f"n2(seed→final)={np.round(_n2_seed, 3).tolist() if _n2_seed is not None else None}"
            f"→{np.round(_n2_final, 3).tolist() if _n2_final is not None else None} "
            f"Δ={_n2_div:.1f}° "
            f"γ_nlp={best.get('gamma_nlp') or float('nan'):.3f} "
            f"status={best.get('status', '?')}")

        return best

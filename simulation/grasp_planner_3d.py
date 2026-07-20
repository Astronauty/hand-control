"""
grasp_planner_3d.py
===================
3D grasp contact-point solver — IPOPT formulation (Kinova Gen3 + LEAP hand).

Seeding (MultiStartGraspPlanner3D)
-----------------------------------
    _seed_pair generates one candidate per call:
      1. Sphere-trace a random direction to surface point p1s.
      2. Perturb the inward normal by up to 25° and sphere-march through the
         object to find the antipodal footprint p2s.
      3. Apply uniform offsets o1, o2 ∈ [−4, +10] mm along the inward normals
         to create an off-surface NLP warm-start (p1, p2).
    Contact frames are frozen from the surface normals at p1s/p2s, not from
    the jittered warm-start.  ~20 seeds per call; no filtering or sorting.

    Pre-check LP: min_gamma_for_accel_lp on (p1s, p2s) before building the NLP.
    Seeds with γ > 50 are skipped.

    Post-solve filter: results with max_slack > 0.05 N are marked
    'wrench-infeasible' and ranked below feasible results.

Decision variables
------------------
    q        (nu,)          actuated joint angles (7 Kinova + 16 LEAP)
    p1       (3,)           thumb contact point, world frame
    p2       (3,)           index-finger contact point, world frame
    γ        scalar         wrench quality margin (minimized; smaller = better geometry)
    y1_k,y2_k (5,) each    friction-cone mixing weights per contact per load corner k
    s_k      (3,)  each     force-row slack per load corner k

Cost (all terms normalized — each ≈ 1 at its reference level, weights are pure priorities)
------------------------------------------------------------------------------------------
    w_ik    * 0.5*(‖Δp1‖²+‖Δp2‖²) / d_ref²          d_ref = 5mm
    + w_reg   * ‖(q−q_neutral)/q_scale‖² / n_dof
    + w_gamma * γ / g_ref                              g_ref = ‖task_force‖ N
    + w_slack * Σ_k ‖s_k‖² / (n_c · s_ref²)          s_ref = 5mN
    + w_y     * Σ_k (‖y1_k‖²+‖y2_k‖²) / (n_c·10·g_ref²)

    IK target includes fingertip radius offset so the tip sphere surface
    touches the contact point (r_thumb/r_index measured from model geom size).

Constraints
-----------
    1. Joint limits    — opti.bounded(lo, q, hi)
    2. Surface contact — linear face-pin (BOX); analytic equality (sphere/cylinder)
    3. Wrench LP       — embedded: torque rows as box inequality ±[tx,ty,tz];
                         force rows balanced with per-corner slack s_k
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
    # _ncf_mod = _importlib.import_module('3D_minimum_NCF_soft')
    _ncf_mod = _importlib.import_module('3D_minimum_NCF')
    min_gamma_for_accel_lp = _ncf_mod.min_gamma_for_accel_lp
    _NCF_AVAILABLE = True

except Exception as e:
    min_gamma_for_accel_lp = None
    _NCF_AVAILABLE = False

try:
    import casadi as ca
    from casadi import Callback as _CasadiCallback
    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False
    _CasadiCallback = object

try:
    import mujoco as mj
    _MJ_AVAILABLE = True
except ImportError:
    _MJ_AVAILABLE = False

try:
    from grasp_control import SpatialIKSolver
    from grasp_control.constrained_ik import (
        _SitePositionCallbackAnalytic,
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
    print("[grasp_planner_3d] all dependencies available")
else:
    print("[grasp_planner_3d] WARNING: some dependencies are missing; "
          "grasp planning will fail if invoked")
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
    'max_iter':                   120,
    'sb':                         'no',
    'print_level':                0,
    'mu_strategy':                'adaptive',
    # Accept flat-objective convergence — the dual residual is non-convergent when the
    # active set is degenerate (minimax γ with antipodal symmetry).
    'acceptable_tol':             1e4,     # effectively disabled — dominated by dual inf
    'acceptable_constr_viol_tol': 1e-6,    # the real feasibility test
    'acceptable_compl_inf_tol':   1e4,
    'acceptable_dual_inf_tol':    1e6,
    'acceptable_obj_change_tol':  1e-5,    # ← the criterion that matters
    'acceptable_iter':            5,
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
        return _box_surface_normal_3d(p, c, size[0], size[1], size[2])
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


def _rotate_random_vec(v, angle_rad, rng):
    """Rotate v by angle_rad about a random axis perpendicular to v (Rodrigues)."""
    ax = rng.standard_normal(3)
    ax -= np.dot(ax, v) * v
    ax_norm = np.linalg.norm(ax)
    if ax_norm < 1e-12:
        ax = np.array([1.0, 0.0, 0.0])
        ax -= np.dot(ax, v) * v
        ax_norm = np.linalg.norm(ax)
    ax /= ax_norm + 1e-12
    return v * np.cos(angle_rad) + np.cross(ax, v) * np.sin(angle_rad)


def _seed_pair(geom_type, size, center, obj_mat, bbox_r, rng,
               delta_max=np.deg2rad(25), off_lo=-0.004, off_hi=0.010):
    """
    One antipodal seed pair with off-surface jitter.

    Sphere-traces to a random surface point p1s, marches through the object
    along a perturbed antipodal direction to find p2s, then applies uniform
    offsets along the inward normals to create an off-surface NLP warm-start.

    Contact frames are frozen from p1s/p2s (surface normals), not from the
    jittered warm-start positions.

    Returns dict with keys:
        p1, p2       — NLP warm-start positions (may be off surface)
        p1s, p2s     — surface footprints (for pre-check LP and frame freezing)
        n1_in, n2_in — inward normals at surface footprints
        offsets      — (o1, o2) in metres, positive = outward
        delta_deg    — jitter angle applied to march direction
    """
    c = np.asarray(center, float)
    u = rng.standard_normal(3)
    u[2] *= 0.3                              # bias toward side faces, away from top/bottom
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

    o1, o2 = rng.uniform(off_lo, off_hi, size=2)
    return {
        'p1':       p1s - o1 * n1_in,
        'p2':       p2s - o2 * n2_in,
        'p1s':      p1s,
        'p2s':      p2s,
        'n1_in':    n1_in,
        'n2_in':    n2_in,
        'offsets':  (float(o1), float(o2)),
        'delta_deg': float(np.rad2deg(delta)),
    }


def _reachable_contact(p, n_out, ground_z, r_tip,
                       n_down_max=-0.5, z_margin=0.002):
    """
    Returns False if this contact can't be approached from above the support plane.

    Rejects two cases:
      - Normal points more than 60° below horizontal (n_out[2] < -0.5): the finger
        would have to come from underneath the table.
      - Contact too close to the support surface: the fingertip sphere (radius r_tip)
        would collide with the table before reaching the contact point.
    """
    # if n_out[2] < n_down_max:
    #     return False
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
                           center_np, mat_np, size):
    """Apply shape-appropriate surface constraint to p_sym in opti."""
    if d is None:
        return
    p_loc = ca.DM(mat_np.T) @ (p_sym - ca.DM(center_np))
    if geom_type == 6:   # BOX
        ax    = int(np.argmax(np.abs(d)))
        coord = float(center_np[ax] + np.sign(d[ax]) * float(size[ax]))
        opti.subject_to(p_sym[ax] == coord)
        for ta in [i for i in range(3) if i != ax]:
            opti.subject_to(opti.bounded(
                float(center_np[ta] - size[ta]),
                p_sym[ta],
                float(center_np[ta] + size[ta])))
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
                           obj_center_np,
                           mu, task_f_np, task_t_bound_np):
    """
    Add embedded LP wrench constraints to a CasADi Opti problem.

    R1_param, R2_param : opti.parameter(3,3) — contact frame [n_in | t1 | t2].
        Computed numerically by caller via _build_contact_frame_3d(-d_lp).
        Held constant per NLP solve; updated between Picard iterations.
        Eliminates Frisvad singularity at n=[0,0,-1] (top-face inward normal).

    task_f_np       : (3,) force bounds [fx, fy, fz] — sign-expanded to corners.
    task_t_bound_np : (3,) torque bounds [τx, τy, τz] — shared box inequality.

    Returns: gamma, slack_list, y1_list, y2_list
        slack_list : list of (3,) MX slack variables (one per corner, on force rows)
    """
    import itertools

    nverts  = 5
    _gamma  = opti.variable()
    opti.subject_to(_gamma >= 0)

    verts_c = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, -mu],
                         [1.0, -mu, 0.0],
                         [1.0,  mu, 0.0],
                         [1.0, 0.0,  mu]])
    forces1 = [R1_param @ ca.DM(v) for v in verts_c]
    forces2 = [R2_param @ ca.DM(v) for v in verts_c]
    obj_c   = ca.DM(obj_center_np)

    def _wrench_sum(p, forces, y):
        w = ca.MX.zeros(6)
        for j, f_j in enumerate(forces):
            w += y[j] * ca.vertcat(ca.cross(p - obj_c, f_j), f_j)
        return w

    # Corner expansion on force components only
    nz_idx = np.where(np.abs(task_f_np) > 1e-10)[0]
    seen, corners = set(), []
    for signs in itertools.product([-1, 1], repeat=len(nz_idx)):
        t_k = task_f_np.copy()
        for i, idx in enumerate(nz_idx):
            t_k[idx] *= signs[i]
        key = tuple(np.round(t_k, 12))
        if key not in seen:
            seen.add(key); corners.append(t_k)
    if not corners:
        corners = [task_f_np.copy()]

    _t_bnd = ca.DM(task_t_bound_np)
    slack_list, y1_list, y2_list = [], [], []
    for t_f_k in corners:
        _y1_k = opti.variable(nverts)
        _y2_k = opti.variable(nverts)
        _s_k  = opti.variable(3)           # slack on force-row balance
        opti.subject_to(_y1_k >= 0)
        opti.subject_to(_y2_k >= 0)
        opti.subject_to(ca.sum1(_y1_k) <= _gamma)
        opti.subject_to(ca.sum1(_y2_k) <= _gamma)
        w = _wrench_sum(p1, forces1, _y1_k) + _wrench_sum(p2, forces2, _y2_k)
        # Torque rows: box inequality (same bound for all corners)
        opti.subject_to(opti.bounded(-_t_bnd, w[0:3], _t_bnd))
        # Force rows: equality with slack (absorbs infeasibility on bad seeds)
        
        # with slack
        opti.subject_to(w[3:6] == ca.DM(t_f_k) + _s_k)
        slack_list.append(_s_k)
        
        # # without slack
        # opti.subject_to(w[3:6] == ca.DM(t_f_k))
        
        y1_list.append(_y1_k)
        y2_list.append(_y2_k)

    return _gamma, slack_list, y1_list, y2_list


# ─────────────────────────────────────────────────────────────────────────────
# GraspConfig3D
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraspConfig3D:
    """Configuration for the 3D grasp planner (Kinova Gen3 + LEAP hand)."""

    # Cost weights — dimensionless priorities (each term normalized to ≈1 at its reference)
    # d_ref=5mm, g_ref=task_load_N, s_ref=5mN, y_ref=g_ref
    w_ik:     float = 0.80   # reachability — dominant until IK < 5mm
    w_reg:    float = 0.06   # posture tie-breaker
    w_gamma:  float = 0.10   # grasp quality (was 0.1, now normalized by task load)
    w_y:      float = 0.04   # min-norm force distribution tie-breaker
    
    w_slack:  float = 10.00   # wrench feasibility — never lose to a preference term

    
    q_scale:  float = 1.0

    # Fingertip mesh effective radius (site centroid to contact surface distance).
    # Measured from model geom at init (MultiStartGraspPlanner3D.__init__).
    r_thumb:  float = 0.005   # m
    r_index:  float = 0.005   # m
    r_middle: float = 0.005   # m  — used for middle/ring ground clearance
    r_ring:   float = 0.005   # m

    # Constraint flags
    joint_limits:      bool  = True   # active joint limit constraints in NLP
    on_object:         bool  = True   # hard surface constraint when d1/d2 not provided
    wrench_constraint: bool  = True   # embedded LP wrench constraint in NLP
    max_iter:          int   = 120    # per-stage max iterations (Picard loop uses n_normal_relinearize+1 stages)

    # Middle/ring collision avoidance (legacy — used when arm_geom_names is empty)
    col_constraint:   bool  = True
    col_clearance_m:  float = 0.005
    finger_radius_m:  float = 0.005

    # Full-arm collision (proximity-pruned softplus SDFs).
    # When arm_geom_names is non-empty, these replace the legacy col_constraint.
    arm_geom_names:   list  = field(default_factory=list)
    col_prune_margin: float = 0.10
    col_use_ground:   bool  = True

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

    # Profiling / Picard loop
    # extra Picard iterations (n total solves); 
    # n_normal_relinearize can be zero for box, search stays within the seed faces
    # n_normal_relinearize = 2 for curved surfaces, since normals genuinely rotate there
    n_seeds:             int  = 10    # seed pairs per multi-start (each from _seed_pair)
    n_normal_relinearize: int  = 2
    verbose_profile:      bool = False

    # Solver — use_slsqp=True  → SQP+OSQP (default; linear face-pin + wrench constraints exact)
    #           use_slsqp=False → IPOPT (interior-point; use for non-box geometries)
    use_slsqp:  bool  = False
    smooth_sdf: bool  = True
    slsqp_alpha: float = 400.0  # smooth SDF alpha (collision avoidance SDF only)

    # Wrench LP task bounds.
    # task_tx/ty/tz >= 0.5*hx (half face width) keeps torque inequality feasible
    # for contacts displaced from face centre (0.03 Nm = full-face coverage for 30mm box).
    mu:       float = 1.0
    task_fx:  float = 0.5
    task_fy:  float = 0.5
    task_fz:  float = 2.0
    task_tx:  float = 0.03   # torque bound (box inequality), Nm
    task_ty:  float = 0.03
    task_tz:  float = 0.03
    torque_budget_scale: float = 1.0   # multiply _nlp_tx/ty/tz; set >1 to test torque-tightness

    # Acceleration budgets for mass-scaled task wrench computation.
    # Used in solve() to replace fixed task_fx/fy/fz with mass*(accel+gravity),
    # and in verify() for the post-solve gamma check.
    # Defaults match kinova_leap_pick_place.py NCF_ACCEL_BUDGET_XYZ / NCF_ANG_ACCEL_BUDGET.
    accel_budget_xyz:     tuple = (0.5, 0.5, 0.5)   # m/s² linear, per world axis
    ang_accel_budget_xyz: tuple = (1.0, 1.0, 1.0)   # rad/s² angular, principal axes

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
                lo_list.append(-np.pi * 10)
                hi_list.append( np.pi * 10)
        self._lo_vec = np.array(lo_list)
        self._hi_vec = np.array(hi_list)

        # Full-arm collision geoms
        self._arm_gids  = []
        self._arm_radii = []
        for gname in (c.arm_geom_names or []):
            gid = self._optional_geom(gname)
            if gid is not None:
                self._arm_gids.append(gid)
                self._arm_radii.append(float(model.geom_rbound[gid]))
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
              d2:      np.ndarray | None = None) -> dict:
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

        # Mass-scaled task wrench for the NLP wrench constraint.
        # Replaces fixed task_fx/fy/fz so the NLP targets the correct squeeze
        # capability for the actual object mass: f = mass*(accel+gravity).
        _bid    = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.obj_body)
        _mass   = float(model.body_mass[_bid])
        _inert  = model.body_inertia[_bid]                 # (Ix, Iy, Iz) principal
        _g_O    = obj_R_np.T @ model.opt.gravity            # gravity in object geom frame
        _ab     = cfg.accel_budget_xyz
        _aab    = cfg.ang_accel_budget_xyz
        _nlp_fx = _mass * (_ab[0] + abs(_g_O[0]))
        _nlp_fy = _mass * (_ab[1] + abs(_g_O[1]))
        _nlp_fz = _mass * (_ab[2] + abs(_g_O[2]))
        # Torques: inertia-scaled, clamped to cfg.task_tx/ty/tz floor so the embedded
        # LP stays feasible for off-center contacts (needs tx/ty >= 0.5*hx).
        _nlp_tx = max(float(_inert[0]) * _aab[0], cfg.task_tx) * cfg.torque_budget_scale
        _nlp_ty = max(float(_inert[1]) * _aab[1], cfg.task_ty) * cfg.torque_budget_scale
        _nlp_tz = max(float(_inert[2]) * _aab[2], cfg.task_tz) * cfg.torque_budget_scale

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

        # ── Proximity pruning: arm geoms vs object ──────────────────────────
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
                if _arm_dist < cfg.col_clearance_m + cfg.col_prune_margin:
                    _active_arm.append(_ai)

        # ── inner: build + run one Opti problem ────────────────────────────
        def _run_stage(q_ws: np.ndarray,
                       p1_ws: np.ndarray,
                       p2_ws: np.ndarray,
                       include_surface: bool,
                       d1_lp: np.ndarray | None = None,
                       d2_lp: np.ndarray | None = None,
                       max_iter_override: int | None = None,
                       stage_label: str = '') -> dict:

            _t_stage_start = time.perf_counter()
            _uid = id(q_ws)

            # ── FK callbacks (analytic Jacobians via mj_jacSite) ───────────
            thumb_cb = _SitePositionCallbackAnalytic(
                f'gp3_th_{_uid}', model, self._thumb_sid, n_act, obj_qpos_snap)
            index_cb = _SitePositionCallbackAnalytic(
                f'gp3_if_{_uid}', model, self._index_sid, n_act, obj_qpos_snap)

            # ── middle+ring collision (only when arm_geom_names empty) ──
            use_legacy_col = cfg.col_constraint and not self._arm_gids
            col_cb = None
            if use_legacy_col:
                col_cb = _BatchedGeomPositionCallbackAnalytic(
                    f'gp3_col_{_uid}', model,
                    [self._middle_gid, self._ring_gid],
                    n_act, obj_qpos_snap)

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

            # ── SDF for surface constraints / legacy col ───────────────────
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

            # ── Cost (normalized — each term ≈ 1 at its reference level) ──
            # Reference scales encode what "good enough" means for each term.
            _d_ref  = 0.005                       # m   — acceptable IK residual
            _n_dof  = int(ca.MX(_q).numel())
            _cost_ik  = 0.5 * (_d1_sq + _d2_sq) / _d_ref**2
            _cost_reg = ca.sumsqr((_q - ca.DM(_q_reg)) / cfg.q_scale) / _n_dof
            _cost = cfg.w_ik * _cost_ik + cfg.w_reg * _cost_reg
            
            # Embedded mode: gamma variable added to cost below after _embed_wrench_cone_ca
            _gamma_lp = None

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
                    if geom_type == 6:   # BOX: linear face-pin (exact constraint, works for IPOPT and SQP)
                        _sym_geom_surface_con(_opti, _p, _d_lp,
                                              geom_type, obj_center_np, obj_R_np, geom_size)
                    elif geom_type == 2:  # SPHERE: |p_loc|² = r²
                        _opti.subject_to(
                            ca.sumsqr(_pl) == float(geom_size[0])**2)
                    elif geom_type == 5:  # CYLINDER: curved surface
                        _R_c = float(geom_size[0])
                        _H   = float(geom_size[1])
                        _opti.subject_to(ca.sumsqr(_pl[0:2]) == _R_c**2)
                        _opti.subject_to(_opti.bounded(-_H, _pl[2], _H))

            # Bounding box: prevents p1/p2 from flying to infinity
            for _p in (_p1, _p2):
                for _i, _h in enumerate([hx, hy, hz]):
                    _opti.subject_to(_opti.bounded(
                        obj_center[_i] - _h - margin,
                        _p[_i],
                        obj_center[_i] + _h + margin))

            # ── 3. Wrench feasibility (embedded LP) ───────────────────────
            _gamma_lp    = None
            _slack_list  = []
            if cfg.wrench_constraint:
                # Contact frames as 3×3 parameter — computed numerically, no Frisvad singularity.
                # Held constant per solve; updated between Picard iterations by outer loop.
                _n1_in = -(d1_lp if d1_lp is not None else
                            _geom_normal_np(p1_ws, geom_type, obj_center_np, obj_R_np, geom_size))
                _n2_in = -(d2_lp if d2_lp is not None else
                            _geom_normal_np(p2_ws, geom_type, obj_center_np, obj_R_np, geom_size))
                _R1_param = _opti.parameter(3, 3)
                _R2_param = _opti.parameter(3, 3)
                _opti.set_value(_R1_param, np.column_stack(_build_contact_frame_3d(_n1_in)))
                _opti.set_value(_R2_param, np.column_stack(_build_contact_frame_3d(_n2_in)))

                _gamma_lp, _slack_list, _y1_list, _y2_list = _embed_wrench_cone_ca(
                    _opti, _p1, _p2,
                    _R1_param, _R2_param,
                    obj_center_np, cfg.mu,
                    np.array([_nlp_fx, _nlp_fy, _nlp_fz]),
                    np.array([_nlp_tx, _nlp_ty, _nlp_tz]))
                _opti.set_initial(_gamma_lp, 1.0)
                _y_init = np.ones(5) / 5.0
                for _y1_k, _y2_k, _s_k in zip(_y1_list, _y2_list, _slack_list):
                    _opti.set_initial(_y1_k, _y_init)
                    _opti.set_initial(_y2_k, _y_init)
                    _opti.set_initial(_s_k, np.zeros(3))

            # Finalize cost (gamma + slack penalty + y regularizer — all normalized)
            # Sentinel zero expressions let the log helper always evaluate all terms.
            _cost_gamma = ca.DM(0.0)
            _cost_slack = ca.DM(0.0)
            _cost_y     = ca.DM(0.0)
            if _gamma_lp is not None:
                _g_ref  = float(np.linalg.norm([_nlp_fx, _nlp_fy, _nlp_fz]))  # N task load
                _s_ref  = 0.005                                                  # N acceptable slack
                _y_ref  = _g_ref                                                 # N force scale
                _n_c    = max(len(_slack_list), 1)
                _n_y    = _n_c * 10                                              # 5 verts × 2 contacts per corner
                _cost_gamma = _gamma_lp / max(_g_ref, 1e-6)
                _cost_slack = sum(ca.sumsqr(sk) for sk in _slack_list) / (_n_c * _s_ref**2)
                _cost_y     = sum(ca.sumsqr(y1k) + ca.sumsqr(y2k)
                                  for y1k, y2k in zip(_y1_list, _y2_list)) / (_n_y * _y_ref**2)
                _opti.minimize(_cost
                             + cfg.w_gamma * _cost_gamma
                             + cfg.w_slack * _cost_slack
                             + cfg.w_y     * _cost_y)
            else:
                _opti.minimize(_cost)

            # ── 5a. Full-arm collision (geometry-appropriate softplus SDF) ─
            _ground_n = ca.DM([0.0, 0.0, 1.0])
            _ground_p = ca.DM([0.0, 0.0, float(cfg.ground_z)])
            if arm_col_cb is not None:
                _arm_pos = arm_col_cb(_q)   # (3*n_active,) CasADi vector
                _obj_R_dm = ca.DM(obj_R_np)
                _obj_c_dm = ca.DM(obj_center_np)
                for _j, _ai in enumerate(_active_arm):
                    _gp = _arm_pos[3*_j : 3*_j+3]
                    _r  = float(self._arm_radii[_ai])
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
                    _opti.subject_to(_d_obj >= cfg.col_clearance_m)
                    if cfg.col_use_ground:
                        _opti.subject_to(
                            _sphere_plane_distance(_gp, _r, _ground_p, _ground_n)
                            >= cfg.col_clearance_m)

            # ── 5b. Legacy middle+ring collision (object + ground) ────────
            if use_legacy_col:
                _col_out  = col_cb(_q)
                _tm, _tr  = _col_out[0:3], _col_out[3:6]
                _d_min    = float(cfg.finger_radius_m + cfg.col_clearance_m)
                _opti.subject_to(_sdf(_tm) >= _d_min)
                _opti.subject_to(_sdf(_tr) >= _d_min)
                if cfg.col_use_ground:
                    for _gp, _r in ((_tm, float(cfg.r_middle)),
                                    (_tr, float(cfg.r_ring))):
                        _opti.subject_to(
                            _sphere_plane_distance(_gp, _r, _ground_p, _ground_n)
                            >= cfg.col_clearance_m)

            # ── 5c. Thumb + index vs ground (unconditional) ────────────────
            # These two tips are absent from both arm_col and legacy_col loops.
            # They're the contact fingers — the most likely to sink into the table.
            if cfg.col_use_ground:
                for _tp, _r in ((_tp1, float(cfg.r_thumb)),
                                (_tp2, float(cfg.r_index))):
                    _opti.subject_to(
                        _sphere_plane_distance(_tp, _r, _ground_p, _ground_n)
                        >= cfg.col_clearance_m)

            # ── Initial guess ─────────────────────────────────────────────
            _opti.set_initial(_q,  q_ws)
            _opti.set_initial(_p1, p1_ws)
            _opti.set_initial(_p2, p2_ws)

            # ── Solver ────────────────────────────────────────────────────
            _n_iter = max_iter_override or cfg.max_iter
            if cfg.use_slsqp:
                _sqp_opts = dict(_SQP_SOLVER_OPTS)
                _sqp_opts['max_iter'] = _n_iter
                _opti.solver('sqpmethod', _sqp_opts)
            else:
                _ipopt_opts = dict(_IPOPT_SOLVER_OPTS)
                _ipopt_opts['max_iter'] = _n_iter
                if self.log_dir:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    _ipopt_opts['output_file']      = os.path.join(
                        self.log_dir, f"grasp3d_ipopt_{ts}.log")
                    _ipopt_opts['file_print_level'] = 5
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
                ('Jslk',       _cost_slack),
                ('Jy',         _cost_y),
            ]
            if self.log_dir:
                if include_surface:
                    if geom_type == 6:
                        _log_exprs.append(('sdf_p1_mm', _sdf(_p1) * 1e3))
                        _log_exprs.append(('sdf_p2_mm', _sdf(_p2) * 1e3))
                if use_legacy_col:
                    _d_min = float(cfg.finger_radius_m + cfg.col_clearance_m)
                    _log_exprs += [
                        ('col_mf', _sdf(_tm) - _d_min),
                        ('col_rf', _sdf(_tr) - _d_min),
                    ]

            # Per-iteration logger (logging only — no parameter updates here;
            # updating opti parameters inside the callback corrupts the L-BFGS
            # curvature pairs and causes numerical divergence).
            def _opti_cb(i):
                if _log_exprs:
                    parts = [f"i={i:3d}"]
                    for _lbl, _expr in _log_exprs:
                        try:
                            v = float(_opti.debug.value(_expr))
                            parts.append(f"{_lbl}={v:+.3e}")
                        except Exception:
                            parts.append(f"{_lbl}=?")
                    self.log.debug(_log_tag + "  ".join(parts))

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
                col_evals = col_cb.eval_count if col_cb is not None else 0
                arm_evals = arm_col_cb.eval_count if arm_col_cb is not None else 0
                wf_calls  = 0  # embedded mode: no callback
                _wrench_str = 'embedded' if cfg.wrench_constraint else 'N'

                con_str = (
                    f"surface={'Y' if include_surface else 'N'}  "
                    f"wrench={_wrench_str}  "
                    f"col_legacy={'Y' if use_legacy_col else 'N'}  "
                    f"arm_col={'Y(' + str(len(_active_arm)) + 'geoms)' if _active_arm else 'N'}")
                lines = [
                    f"{tag}--- solve profile ------------------------------------------",
                    f"{tag}  constraints : {con_str}",
                    f"{tag}  DOF={n_act}  cbs:"
                    f"  thumb={th_evals}  index={if_evals}"
                    f"  col={col_evals}  arm={arm_evals}  wf_lp={wf_calls}",
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
                gam  = _v(_cost_gamma); slk = _v(_cost_slack); y = _v(_cost_y)
                f    = _v(_opti.f)
                return (
                    f"f={f:.4f}  "
                    f"ik={cfg.w_ik*ik:.4f}({ik:.3f})  "
                    f"reg={cfg.w_reg*reg:.4f}({reg:.3f})  "
                    f"γ={cfg.w_gamma*gam:.4f}({gam:.3f})  "
                    f"slk={cfg.w_slack*slk:.4f}({slk:.3f})  "
                    f"y={cfg.w_y*y:.4f}({y:.3f})"
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

            def _max_slack(val_fn):
                if not _slack_list:
                    return 0.0
                try:
                    return float(max(np.max(np.abs(val_fn(sk))) for sk in _slack_list))
                except Exception:
                    return float('nan')

            def _slack_vectors(val_fn):
                result = []
                for sk in _slack_list:
                    try:
                        result.append(np.asarray(val_fn(sk)).flatten().tolist())
                    except Exception:
                        result.append([float('nan'), float('nan'), float('nan')])
                return result

            try:
                _sol = _opti.solve()
                _plog(_sol.stats())
                _tag = f'[{stage_label}|cost] ' if stage_label else '[cost] '
                self.log.info(_tag + _cost_breakdown(_sol.value))
                self.log.info(f'[{stage_label}|torque] ' + _torque_diagnostic(_sol.value))
                return {
                    'success':    True,
                    'q':          _sol.value(_q),
                    'p1':         _sol.value(_p1),
                    'p2':         _sol.value(_p2),
                    'cost':       float(_sol.value(_opti.f)),
                    'iterations': _sol.stats()['iter_count'],
                    'status':     'converged',
                    'gamma_nlp':     float(_sol.value(_gamma_lp)) if _gamma_lp is not None else None,
                    'max_slack':     _max_slack(_sol.value),
                    'slack_vectors': _slack_vectors(_sol.value),
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
                    return {
                        'success':    False,
                        'q':          _opti.debug.value(_q),
                        'p1':         _opti.debug.value(_p1),
                        'p2':         _opti.debug.value(_p2),
                        'cost':       _opti.debug.value(_opti.f) if _st else None,
                        'iterations': (_st or {}).get('iter_count'),
                        'status':     'best-effort',
                        'gamma_nlp':     float(_opti.debug.value(_gamma_lp)) if _gamma_lp is not None else None,
                        'max_slack':     _max_slack(_opti.debug.value),
                        'slack_vectors': _slack_vectors(_opti.debug.value),
                    }
                except Exception as _e2:
                    self.log.error(f"GraspPlanner3D debug extraction: {_e2}")
                    return {'success': False, 'q': None, 'p1': None, 'p2': None,
                            'cost': None, 'iterations': None, 'status': 'failed',
                            'gamma_nlp': None, 'max_slack': float('nan'),
                            'slack_vectors': []}

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
        for _ri in range(cfg.n_normal_relinearize + 1):
            res = _run_stage(_q_ws, _p1_ws, _p2_ws,
                             include_surface=True,
                             d1_lp=_d1_lp,
                             d2_lp=_d2_lp,
                             max_iter_override=cfg.max_iter,
                             stage_label=f'S{_ri+1}')
            # Keep the cheapest stage result — relinearization has no descent guarantee.
            if (res.get('cost') is not None and
                    (not _best_res or res['cost'] < _best_res.get('cost', float('inf')))):
                _best_res = res
            if _ri >= cfg.n_normal_relinearize or res.get('p1') is None:
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
                f"dp={_dp*1e3:.2f}mm  mismatch={_mismatch_deg:.1f}°  "
                f"max_slack={res.get('max_slack', float('nan')):.3e}")

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
            _sm    = _span_margin(_n1f, _n2f, cfg.mu)
            self.log.info(
                f"[solve|final] n1={np.round(_n1f, 3).tolist()}  n2={np.round(_n2f, 3).tolist()}  "
                f"dot={_dot12:+.3f}  span_margin={_sm:+.4f}rad")
            _svecs = res.get('slack_vectors', [])
            if _svecs:
                self.log.info(
                    f"[solve|slack] per_corner={[[round(x, 4) for x in sv] for sv in _svecs]}")
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
        cfg         = self.cfg
        gamma_min   = None
        wf_feasible = False
        wf_tag      = 'SKIP'
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
                R_WO_v   = data_v.xmat[_bid_v].reshape(3, 3)
                _g_O_v   = R_WO_v.T @ model.opt.gravity
                _ab_v    = cfg.accel_budget_xyz
                _accel_v = tuple(_ab_v[i] + abs(_g_O_v[i]) for i in range(3))
                # Torque bounds: same formula as NLP (_nlp_tx/ty/tz), converted to
                # angular acceleration for min_gamma_for_accel_lp.
                _nlp_tx_v = max(float(_inert_v[0]) * cfg.ang_accel_budget_xyz[0], cfg.task_tx)
                _nlp_ty_v = max(float(_inert_v[1]) * cfg.ang_accel_budget_xyz[1], cfg.task_ty)
                _nlp_tz_v = max(float(_inert_v[2]) * cfg.ang_accel_budget_xyz[2], cfg.task_tz)
                _aab_v = np.array([
                    _nlp_tx_v / max(float(_inert_v[0]), 1e-12),
                    _nlp_ty_v / max(float(_inert_v[1]), 1e-12),
                    _nlp_tz_v / max(float(_inert_v[2]), 1e-12),
                ])
                _p1_O_v  = R_WO_v.T @ (p1_np - obj_pos)
                _p2_O_v  = R_WO_v.T @ (p2_np - obj_pos)
                R1_O     = R_WO_v.T @ R1
                R2_O     = R_WO_v.T @ R2
                gamma_min = min_gamma_for_accel_lp(
                    _mass_v * _accel_v[0], _mass_v * _accel_v[1], _mass_v * _accel_v[2],
                    _inert_v[0] * _aab_v[0], _inert_v[1] * _aab_v[1], _inert_v[2] * _aab_v[2],
                    n=2,
                    pos=[_p1_O_v.reshape(3, 1), _p2_O_v.reshape(3, 1)],
                    R=[R1_O, R2_O],
                    ncf=[1.0, 1.0],
                    tan_y=[0.0, 0.0],
                    tan_z=[0.0, 0.0],
                    mu=[cfg.mu, cfg.mu],
                )
                wf_feasible = (gamma_min is not None)
                wf_tag = f'OK(γ_min={gamma_min:.3f})' if wf_feasible else 'INFEASIBLE'
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
            'gamma_nlp':            result.get('gamma_nlp'),
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

        # Measure actual fingertip effective radius (site-to-contact-surface distance)
        # from the model geom, accounting for all geom types.
        _pl = self._planner

        def _tip_radius(gid):
            gt = int(model.geom_type[gid])
            gs = model.geom_size[gid]
            if gt == 2:   # mjGEOM_SPHERE: size[0] = radius
                return float(gs[0])
            if gt == 3:   # mjGEOM_CAPSULE: size[0] = radius
                return float(gs[0])
            if gt == 6:   # mjGEOM_BOX: use min half-extent as a conservative radius
                return float(np.min(gs[:3]))
            # mjGEOM_MESH (7) and others: geom_rbound is MuJoCo's bounding radius
            return float(model.geom_rbound[gid])

        _pl.cfg.r_thumb  = _tip_radius(_pl._thumb_gid)
        _pl.cfg.r_index  = _tip_radius(_pl._index_gid)
        _pl.cfg.r_middle = _tip_radius(_pl._middle_gid)
        _pl.cfg.r_ring   = _tip_radius(_pl._ring_gid)
        _pl.log.info(
            f"[tip_radius] "
            f"r_thumb={_pl.cfg.r_thumb*1e3:.1f}mm  "
            f"r_index={_pl.cfg.r_index*1e3:.1f}mm  "
            f"r_middle={_pl.cfg.r_middle*1e3:.1f}mm  "
            f"r_ring={_pl.cfg.r_ring*1e3:.1f}mm  "
            f"(geom types: thumb={int(model.geom_type[_pl._thumb_gid])}  "
            f"index={int(model.geom_type[_pl._index_gid])})")

        # Measure friction from the object geom and set cfg.mu with a 0.8 safety margin.
        # MuJoCo friction[0] is sliding; effective pair value = sqrt(mu1*mu2) but we
        # can only read one side here, so apply margin conservatively.
        _obj_gid = _pl._obj_gid
        _mu_obj  = float(model.geom_friction[_obj_gid][0])
        _pl.cfg.mu = round(0.8 * _mu_obj, 3)
        _pl.log.info(
            f"[friction] obj_mu={_mu_obj:.3f}  cfg.mu={_pl.cfg.mu:.3f} (0.8x safety margin)")

    def solve(self, q_ref: np.ndarray, obj_pos: np.ndarray,
              max_seeds: int | None = None) -> dict:
        """Try _seed_pair seeds (up to max_seeds), return best result ranked by cost."""
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

        # Task wrench bounds (mirrors GraspPlanner3D.solve logic)
        _bid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.obj_body)
        _mass  = float(model.body_mass[_bid])
        _inert = model.body_inertia[_bid]
        _g_O   = obj_R_np.T @ model.opt.gravity
        _ab    = cfg.accel_budget_xyz
        _aab   = cfg.ang_accel_budget_xyz
        _nlp_fx = _mass * (_ab[0] + abs(_g_O[0]))
        _nlp_fy = _mass * (_ab[1] + abs(_g_O[1]))
        _nlp_fz = _mass * (_ab[2] + abs(_g_O[2]))
        _nlp_tx = max(float(_inert[0]) * _aab[0], cfg.task_tx) * cfg.torque_budget_scale
        _nlp_ty = max(float(_inert[1]) * _aab[1], cfg.task_ty) * cfg.torque_budget_scale
        _nlp_tz = max(float(_inert[2]) * _aab[2], cfg.task_tz) * cfg.torque_budget_scale

        bbox_r    = float(np.max(geom_size)) * 2.5
        n_seeds   = max_seeds if max_seeds is not None else cfg.n_seeds
        max_attempts = 40 * n_seeds

        def _sdf(p):
            return _geom_sdf_np(p, geom_type, obj_center_np, obj_R_np, geom_size)

        _ground_z  = cfg.ground_z
        _r_tip_min = min(cfg.r_thumb, cfg.r_index)  # conservative: unknown which tip goes where

        _t0 = time.perf_counter()
        seeds, attempts, rejected = [], 0, 0
        while len(seeds) < n_seeds and attempts < max_attempts:
            attempts += 1
            s = _seed_pair(geom_type, geom_size, c, obj_R_np, bbox_r, self._rng)
            # Hemisphere check — n1 and n2 must point into opposing hemispheres
            if float(np.dot(s['n1_in'], s['n2_in'])) >= 0:
                rejected += 1
                continue
            # Reachability check — reject contacts below the table or facing downward
            n1_out = -s['n1_in'];  n2_out = -s['n2_in']
            if (not _reachable_contact(s['p1s'], n1_out, _ground_z, _r_tip_min) or
                    not _reachable_contact(s['p2s'], n2_out, _ground_z, _r_tip_min)):
                rejected += 1
                continue
            seeds.append(s)

        if len(seeds) < n_seeds:
            log.warning(
                f"[seed_gen] only {len(seeds)}/{n_seeds} valid seeds after "
                f"{attempts} attempts ({rejected} rejected)")
        log.info(
            f"[seed_gen] {len(seeds)} seeds in {(time.perf_counter()-_t0)*1e3:.1f}ms "
            f"({attempts} attempts, {rejected} rejected)")

        results = []
        for i, seed in enumerate(seeds):
            if dash is not None:
                dash.push({'type': 'active', 'label': f'grasp3d seed {i+1}/{n_seeds}'})

            # ── Pre-check LP on surface footprints ────────────────────────────
            g_pre = None
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
                    g_pre = min_gamma_for_accel_lp(
                        _nlp_fx, _nlp_fy, _nlp_fz,
                        _nlp_tx, _nlp_ty, _nlp_tz,
                        n=2,
                        pos=[p1s_O.reshape(3, 1), p2s_O.reshape(3, 1)],
                        R=[R1_O, R2_O],
                        ncf=[1.0, 1.0],
                        tan_y=[0.0, 0.0],
                        tan_z=[0.0, 0.0],
                        mu=[cfg.mu, cfg.mu],
                    )
                except Exception as _lp_e:
                    log.debug(f"[seed {i+1}] pre-check LP error: {_lp_e}")

            g_pre_str = f'{g_pre:.2f}' if g_pre is not None else 'N/A'
            if g_pre is not None and g_pre > 50.0:
                log.debug(
                    f"[seed {i+1}/{n_seeds}] pre-check γ={g_pre:.1f} > 50 — skip")
                continue

            # ── Run NLP ───────────────────────────────────────────────────────
            r = self._planner.solve(q_ref, obj_pos,
                                    p1_init=seed['p1'],
                                    p2_init=seed['p2'],
                                    d1=-seed['n1_in'],   # outward normal
                                    d2=-seed['n2_in'])

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

                # Post-solve filter: large slack means force rows unresolvable
                if r.get('max_slack', float('inf')) > 0.05:
                    r['status'] = 'wrench-infeasible'

            o1, o2 = seed['offsets']
            log.info(
                f"[seed {i+1}/{n_seeds}] "
                f"o=({o1*1e3:+.1f},{o2*1e3:+.1f})mm δ={seed['delta_deg']:.0f}° "
                f"γ_pre={g_pre_str} γ_nlp={r.get('gamma_nlp') or float('nan'):.3f} "
                f"slack={r.get('max_slack', float('nan')):.3f}N "
                f"sdf=({sdf_p1*1e3:.2f},{sdf_p2*1e3:.2f})mm "
                f"IK=({ik_th:.1f},{ik_if:.1f})mm "
                f"iters={r.get('iterations', '?')} → {r.get('status', '?')}")

            r['p1_seed']  = seed['p1s'].copy()
            r['p2_seed']  = seed['p2s'].copy()
            r['seed_meta'] = seed
            results.append(r)

        if not results:
            return {'success': False, 'q': None, 'p1': None, 'p2': None,
                    'cost': None, 'status': 'failed', 'all_results': []}

        def _rank(r):
            ok  = r.get('p1') is not None and r.get('status') != 'wrench-infeasible'
            return (0 if ok else 1, r.get('cost') or 1e9)

        results.sort(key=_rank)
        best = results[0]
        best['all_results'] = results
        return best

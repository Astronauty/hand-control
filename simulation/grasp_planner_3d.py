"""
grasp_planner_3d.py
===================
3D grasp contact-point solver — SQP formulation (Kinova Gen3 + LEAP hand).

Decision variables
------------------
    q  (nu,)   actuated joint angles
    p1 (3,)    thumb contact point in world frame
    p2 (3,)    index-finger contact point in world frame

Cost
----
    w_ik  * (‖FK_thumb(q)−p1‖² + ‖FK_index(q)−p2‖²)
    + w_reg * ‖(q−q_ref)/q_scale‖²

Constraints
-----------
    1. Joint limits — vectorized opti.bounded(lo, q, hi)
    2. Surface contact — _sym_geom_surface_con(p1) + _sym_geom_surface_con(p2) + bbox guard
    3. Hard IK (optional) — ‖FK(q)−p_i‖² ≤ 1e-8
    4. Wrench feasibility (LP existence) — _WrenchFeasCallback3D(cat(p1,p2)) >= 0
    5. Arm collision (proximity-pruned, softplus SDFs from constrained_ik)

Contact frame convention
------------------------
    R[:,0] = inward normal  (compressive, into object)
    R[:,1] = first tangent
    R[:,2] = second tangent

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
    'print_time':            True,
    'qpsol':                 'osqp',
    'qpsol_options':         {'error_on_fail': True,
                              'osqp': {'verbose': True, 'polish': True}},
    'max_iter':              500,
    'hessian_approximation': 'limited-memory',
    'lbfgs_memory':          20,
    'convexify_strategy':    'regularize',
    'print_iteration':       True,
    'print_header':          True,
    'print_status':          True,
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


def _geom_seeds(geom_type: int, size, n_radial: int = 4):
    """
    Shape-appropriate 2-finger contact seeds.

    Returns list of (d_thumb, d_index, p1_offset, p2_offset).
    All in the object's local frame; MultiStart rotates by the actual mat.
    """
    if geom_type == 6:   # BOX — 6 face pairs
        seeds = []
        for d_th, d_if in _FACE_SEEDS_2F_UNIT:
            p1_off = d_th * np.array([size[0], size[1], size[2]])
            p2_off = d_if * np.array([size[0], size[1], size[2]])
            seeds.append((d_th.copy(), d_if.copy(), p1_off, p2_off))
        return seeds
    elif geom_type == 5:  # CYLINDER — n_radial radial + 1 axial
        R, H = float(size[0]), float(size[1])
        seeds = []
        for k in range(n_radial):
            theta = k * np.pi / n_radial
            d = np.array([np.cos(theta), np.sin(theta), 0.0])
            seeds.append((d.copy(), -d.copy(), d * R, -d * R))
        seeds.append((np.array([0., 0., -1.]), np.array([0., 0., 1.]),
                      np.array([0., 0., -H]),  np.array([0., 0.,  H])))
        return seeds
    else:   # SPHERE or fallback
        r = float(size[0])
        seeds = []
        for k in range(n_radial):
            theta = k * np.pi / n_radial
            d = np.array([np.cos(theta), np.sin(theta), 0.0])
            seeds.append((d.copy(), -d.copy(), d * r, -d * r))
        return seeds


def _hat(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix: hat(v) @ u == cross(v, u)."""
    v = np.asarray(v, float).flatten()
    return np.array([[ 0.0,  -v[2],  v[1]],
                     [ v[2],  0.0,  -v[0]],
                     [-v[1],  v[0],  0.0 ]])


def _ca_cross(a, b):
    """Cross product for 3×1 CasADi MX/DM vectors."""
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# 2-finger contact seeds: (d_thumb, d_index) face direction pairs.
_FACE_SEEDS_2F_UNIT = [
    (np.array([ 0.0, -1.0,  0.0]), np.array([ 0.0,  1.0,  0.0])),
    (np.array([ 0.0,  1.0,  0.0]), np.array([ 0.0, -1.0,  0.0])),
    (np.array([-1.0,  0.0,  0.0]), np.array([ 1.0,  0.0,  0.0])),
    (np.array([ 1.0,  0.0,  0.0]), np.array([-1.0,  0.0,  0.0])),
    (np.array([ 0.0,  0.0, -1.0]), np.array([ 0.0,  0.0,  1.0])),
    (np.array([ 0.0,  0.0,  1.0]), np.array([ 0.0,  0.0, -1.0])),
]


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


# ─────────────────────────────────────────────────────────────────────────────
# Wrench feasibility CasADi callbacks (LP existence constraint inside NLP)
# ─────────────────────────────────────────────────────────────────────────────

class _WrenchFeasJacCallback3D(ca.Callback):
    """
    Companion Jacobian for _WrenchFeasCallback3D.

    Input 0: x (6×1) — [p1, p2]
    Input 1: y (1×1) — parent output (not used)
    Output:  J (1×6) — d(gamma_min)/d[p1, p2]

    Returns cached LP dual-variable gradient when the input matches the
    parent's last forward evaluation; zeros otherwise (fallback).
    """

    def __init__(self, name: str, parent: '_WrenchFeasCallback3D'):
        ca.Callback.__init__(self)
        self._parent = parent
        self.construct(name, {})

    def get_n_in(self):  return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return (ca.Sparsity.dense(6, 1) if i == 0
                else ca.Sparsity.dense(1, 1))

    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 6)
    def has_jacobian(self):        return False

    def eval(self, arg):
        p = self._parent
        p12 = np.array(arg[0]).flatten()[:6]
        if (p._jac_p12 is not None
                and p._jac_cached is not None
                and np.max(np.abs(p12 - p._jac_p12)) <= 1e-8):
            return [p._jac_cached.reshape(1, 6)]
        return [np.zeros((1, 6))]


class _WrenchFeasCallback3D(ca.Callback):
    """
    [p1(3), p2(3)] → gamma_min (LP feasible) or elastic infeasibility measure (LP infeasible).

    Input:  x = vertcat(p1, p2)  shape (6, 1)
    Output: gamma_min >= 0 if wrench is resistible;
            -elastic_M * max_violation < 0 otherwise  shape (1, 1)

    At each NLP evaluation the callback runs min_gamma_for_accel_lp with
    return_sensitivity=True, caches the LP dual-variable gradient, and
    exposes it through _WrenchFeasJacCallback3D.  CasADi then uses this
    analytic Jacobian instead of FD-perturbing the callback, saving 6 LP
    solves per NLP gradient evaluation.

    When infeasible, the Phase-I LP returns a negative value with a nonzero
    gradient (from Phase-I LP duals), giving the NLP solver a recovery direction
    instead of a zero-row QP subproblem.

    The NLP constraint is: _WrenchFeasCallback3D(vertcat(p1, p2)) >= 0
    SQP drives p1/p2 toward a configuration with gamma_min >= 0.
    """

    def __init__(self, name: str,
                 geom_type: int,
                 obj_center: np.ndarray,
                 obj_R:      np.ndarray,
                 obj_size:   np.ndarray,
                 task_fx: float, task_fy: float, task_fz: float,
                 task_tx: float, task_ty: float, task_tz: float,
                 mu: float,
                 elastic_M: float = 1000.0):
        ca.Callback.__init__(self)
        self._geom_type  = geom_type
        self._obj_center = np.asarray(obj_center, float)
        self._obj_R      = np.asarray(obj_R, float).reshape(3, 3)
        self._obj_size   = np.asarray(obj_size, float)
        self._task_fx    = task_fx
        self._task_fy    = task_fy
        self._task_fz    = task_fz
        self._task_tx    = task_tx
        self._task_ty    = task_ty
        self._task_tz    = task_tz
        self._mu         = mu
        self._elastic_M  = elastic_M   # 0.0 → hard mode (returns None on infeasible)
        self._jac_p12    = None
        self._jac_cached = None
        self._n_calls    = 0
        # Jacobian callback must be alive for as long as this callback lives
        self._jac_cb     = _WrenchFeasJacCallback3D(name + '_J', self)
        self.construct(name, {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(6, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return True
    def get_jacobian(self, *_):    return self._jac_cb

    def eval(self, arg):
        p12 = np.array(arg[0]).flatten()
        p1, p2 = p12[0:3], p12[3:6]

        # Outward normals → inward contact normals → contact rotation matrices
        n1_out = _geom_normal_np(p1, self._geom_type,
                                  self._obj_center, self._obj_R, self._obj_size)
        _, t1_1, t2_1 = _build_contact_frame_3d(-n1_out)
        R1 = np.column_stack([-n1_out, t1_1, t2_1])   # R[:,0]=inward normal

        n2_out = _geom_normal_np(p2, self._geom_type,
                                  self._obj_center, self._obj_R, self._obj_size)
        _, t1_2, t2_2 = _build_contact_frame_3d(-n2_out)
        R2 = np.column_stack([-n2_out, t1_2, t2_2])

        # pos = offset from object centre (frame-agnostic torque calculation)
        pos1 = (p1 - self._obj_center).reshape(3, 1)
        pos2 = (p2 - self._obj_center).reshape(3, 1)

        result = min_gamma_for_accel_lp(
            self._task_fx, self._task_fy, self._task_fz,
            self._task_tx, self._task_ty, self._task_tz,
            n=2,
            pos=[pos1, pos2],
            R=[R1, R2],
            ncf=[1.0, 1.0],
            tan_y=[0.0, 0.0],
            tan_z=[0.0, 0.0],
            mu=[self._mu, self._mu],
            return_sensitivity=True,
            elastic_M=self._elastic_M,
        )

        if result[0] is None:
            # Phase-I LP failed unexpectedly — should not happen with elastic formulation
            self._jac_cached = np.zeros(6)
            self._jac_p12    = p12.copy()
            return [-1.0]

        # result[0]: gamma_min >= 0 (feasible) or -elastic_M * violation < 0 (infeasible)
        # result[1], result[2]: d(value)/dp1, d(value)/dp2 — nonzero in both cases
        g_val, dg_dp1, dg_dp2 = result
        self._jac_cached = np.concatenate([dg_dp1, dg_dp2])   # (6,) — no sign flip
        self._jac_p12    = p12.copy()
        self._n_calls   += 1
        return [float(g_val)]


# ─────────────────────────────────────────────────────────────────────────────
# Embedded wrench LP helper
# ─────────────────────────────────────────────────────────────────────────────

def _embed_wrench_cone_ca(opti, p1, p2, obj_center_np, d1, d2, mu, task_wrench_np):
    """
    Add worst-case LP wrench feasibility constraints directly to a CasADi Opti problem.

    One copy of the LP (y1_k, y2_k variables + wrench balance) is added per unique
    sign-combination of the task wrench components, matching what min_gamma_for_accel_lp
    checks.  All copies share a single gamma, so gamma is the worst-case NCF scale
    across all acceleration directions.

    Variables added to opti:
        gamma        (scalar)     — shared worst-case NCF scale
        y1_k, y2_k  (5,) each    — cone coefficients for the k-th corner, k=0..n_corners-1

    Constraints per corner k:
        y1_k >= 0,  y2_k >= 0
        sum(y1_k) <= gamma,   sum(y2_k) <= gamma
        wrench_1(p1, y1_k) + wrench_2(p2, y2_k) == signs_k * task_wrench

    Parameters
    ----------
    d1, d2         : (3,) outward face normal at each contact (inward = -d).
    task_wrench_np : (6,) [Tx, Ty, Tz, Fx, Fy, Fz] — only nonzero entries generate
                    distinct sign-combinations; zero entries are held fixed.

    Returns
    -------
    gamma    : CasADi MX scalar
    y1_list  : list of n_corners CasADi MX (5,) variables
    y2_list  : list of n_corners CasADi MX (5,) variables
    """
    import itertools

    nverts = 5  # hard point-contact pyramid, 5 vertices

    _gamma = opti.variable()
    opti.subject_to(_gamma >= 0)

    # Contact frames from seed directions (fixed throughout the NLP)
    n1 = np.asarray(-d1, float)
    _, t1_1, t2_1 = _build_contact_frame_3d(n1)
    R1 = np.column_stack([n1, t1_1, t2_1])

    n2 = np.asarray(-d2, float)
    _, t1_2, t2_2 = _build_contact_frame_3d(n2)
    R2 = np.column_stack([n2, t1_2, t2_2])

    # Unit cone vertex forces in world frame (fixed)
    verts_c = np.array([
        [0.0, 0.0,  0.0],   # zero force
        [1.0, 0.0, -mu ],
        [1.0, -mu,  0.0],
        [1.0,  mu,  0.0],
        [1.0, 0.0,  mu ],
    ])
    forces1 = [ca.DM(R1 @ v) for v in verts_c]
    forces2 = [ca.DM(R2 @ v) for v in verts_c]
    obj_c   = ca.DM(obj_center_np)

    def _wrench_sum(p, forces, y):
        w = ca.MX.zeros(6)
        for j, f_j in enumerate(forces):
            w += y[j] * ca.vertcat(ca.cross(p - obj_c, f_j), f_j)
        return w

    # Build unique sign-combination corners (deduplicate zero components)
    nz_idx = np.where(np.abs(task_wrench_np) > 1e-10)[0]
    seen   = set()
    corners = []
    for signs in itertools.product([-1, 1], repeat=len(nz_idx)):
        t_k = task_wrench_np.copy()
        for i, idx in enumerate(nz_idx):
            t_k[idx] *= signs[i]
        key = tuple(np.round(t_k, 12))
        if key not in seen:
            seen.add(key)
            corners.append(t_k)
    if not corners:        # all-zero task wrench — trivially feasible
        corners = [task_wrench_np]

    y1_list, y2_list = [], []
    for t_k in corners:
        _y1_k = opti.variable(nverts)
        _y2_k = opti.variable(nverts)
        opti.subject_to(_y1_k >= 0)
        opti.subject_to(_y2_k >= 0)
        opti.subject_to(ca.sum1(_y1_k) <= _gamma)
        opti.subject_to(ca.sum1(_y2_k) <= _gamma)
        opti.subject_to(_wrench_sum(p1, forces1, _y1_k)
                        + _wrench_sum(p2, forces2, _y2_k) == ca.DM(t_k))
        y1_list.append(_y1_k)
        y2_list.append(_y2_k)

    return _gamma, y1_list, y2_list


# ─────────────────────────────────────────────────────────────────────────────
# GraspConfig3D
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraspConfig3D:
    """Configuration for the 3D grasp planner (Kinova Gen3 + LEAP hand)."""

    # Cost weights
    w_ik:     float = 0.8
    w_reg:    float = 0.005
    q_scale:  float = 1.0

    # Fingertip mesh effective radius (site centroid to contact surface distance)
    # leap_th_ds_tip and leap_if_ds_tip both have site size=0.005 m
    r_thumb:  float = 0.005   # m
    r_index:  float = 0.005   # m

    # Constraint flags
    joint_limits:     bool = True
    on_object:        bool = True    # hard surface constraint when d1/d2 not provided
    wrench_constraint: bool = True   # LP existence constraint inside the NLP
    max_iter:         int  = 120

    # Wrench feasibility mode — three formulations available for comparison:
    #   'hard'     — external callback; returns -1.0 with ZERO gradient when infeasible.
    #                Original behaviour; NLP stalls when LP is infeasible.
    #   'elastic'  — external callback with Phase-I fallback; returns -elastic_M*violation
    #                with a NONZERO gradient from Phase-I LP duals when infeasible.
    #                Gives NLP solvers a recovery direction.
    #   'embedded' — LP constraints (wrench balance + cone membership) added directly to
    #                NLP as bilinear CasADi expressions.  gamma added to cost with w_gamma.
    #                No external callback.  Fully smooth everywhere — bilinear in (p, y).
    wrench_mode: str   = 'embedded'   # 'hard' | 'elastic' | 'embedded'
    elastic_M:   float = 1000.0      # Phase-I penalty (elastic mode only)
    w_gamma:     float = 0.01      # cost weight for gamma (embedded mode only)

    # Middle/ring collision avoidance (legacy — used when arm_geom_names is empty)
    col_constraint:   bool  = True
    col_clearance_m:  float = 0.005
    finger_radius_m:  float = 0.005

    # Full-arm collision (proximity-pruned softplus SDFs).
    # When arm_geom_names is non-empty, these replace the legacy col_constraint.
    arm_geom_names:   list  = field(default_factory=list)
    col_prune_margin: float = 0.10
    col_use_ground:   bool  = True

    # Profiling
    n_radial_seeds:   int  = 4
    verbose_profile:  bool = True

    # Solver — use_slsqp=True → SQP+OSQP (faster, analytic Jacobians)
    #           use_slsqp=False → IPOPT (interior-point, more robust to infeasibility)
    use_slsqp:        bool  = False
    smooth_sdf:       bool  = True
    slsqp_alpha:      float = 40.0

    # Wrench LP task bounds.
    # task_tx/ty MUST be >= 0.5 * hx (half the face width) to keep the LP feasible
    # for contacts displaced up to hx from face centre.  For a 30 mm half-box:
    #   min task_tx/ty = 0.5 * 0.03 = 0.015 Nm.  0.03 Nm gives full-face coverage.
    # Setting them to 0.0 forces zero net torque, which is infeasible for any
    # contact pair that is not perfectly antisymmetric on their faces.
    mu:       float = 1.0
    task_fx:  float = 0.5
    task_fy:  float = 0.5
    task_fz:  float = 2.0
    task_tx:  float = 0.0
    task_ty:  float = 0.0
    task_tz:  float = 0.0

    # Geometry names (must match scene XML)
    obj_geom:    str = 'obj_box_geom'
    obj_body:    str = 'obj_box'
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
        _err_th = float(np.linalg.norm(
            self._dls_data.site_xpos[self._thumb_sid] - p1_seed))
        _err_if = float(np.linalg.norm(
            self._dls_data.site_xpos[self._index_sid] - p2_seed))
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

            # ── Wrench feasibility callback ────────────────────────────────
            wrench_cb = None
            if cfg.wrench_constraint and _NCF_AVAILABLE and cfg.wrench_mode != 'embedded':
                _em = 0.0 if cfg.wrench_mode == 'hard' else cfg.elastic_M
                wrench_cb = _WrenchFeasCallback3D(
                    f'gp3_wf_{_uid}',
                    geom_type, obj_center_np, obj_R_np, geom_size,
                    cfg.task_fx, cfg.task_fy, cfg.task_fz,
                    cfg.task_tx, cfg.task_ty, cfg.task_tz,
                    cfg.mu,
                    elastic_M=_em)

            # ── Build Opti ────────────────────────────────────────────────
            _opti = ca.Opti()
            _q  = _opti.variable(n_act)
            _p1 = _opti.variable(3)
            _p2 = _opti.variable(3)

            _tp1   = thumb_cb(_q)
            _tp2   = index_cb(_q)
            _d1_sq = ca.sumsqr(_tp1 - (_p1 + cfg.r_thumb * ca.DM(d1)))   # m²
            _d2_sq = ca.sumsqr(_tp2 - (_p2 + cfg.r_index * ca.DM(d2)))   # m²

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

            # ── Cost ──────────────────────────────────────────────────────
            _cost = (cfg.w_ik  * (_d1_sq + _d2_sq) +
                     cfg.w_reg * ca.sumsqr((_q - ca.DM(q_ws)) / cfg.q_scale))
            # Embedded mode: gamma variable added to cost below after _embed_wrench_cone_ca
            _gamma_lp = None

            # ── 1. Joint limits (vectorized) ──────────────────────────────
            if cfg.joint_limits:
                _opti.subject_to(_opti.bounded(
                    ca.DM(self._lo_vec), _q, ca.DM(self._hi_vec)))

            # ── 2. Surface constraints ────────────────────────────────────
            if include_surface:
                _sym_geom_surface_con(_opti, _p1, d1, geom_type,
                                      obj_center_np, obj_R_np, geom_size)
                _sym_geom_surface_con(_opti, _p2, d2, geom_type,
                                      obj_center_np, obj_R_np, geom_size)
                if cfg.on_object and d1 is None and d2 is None:
                    _opti.subject_to(_sdf(_p1) == 0)
                    _opti.subject_to(_sdf(_p2) == 0)

            # Bounding box: prevents p1/p2 from flying to infinity
            for _p in (_p1, _p2):
                for _i, _h in enumerate([hx, hy, hz]):
                    _opti.subject_to(_opti.bounded(
                        obj_center[_i] - _h - margin,
                        _p[_i],
                        obj_center[_i] + _h + margin))

            # ── 3. Wrench feasibility ─────────────────────────────────────
            if wrench_cb is not None:
                # hard / elastic: external LP callback
                _opti.subject_to(wrench_cb(ca.vertcat(_p1, _p2)) >= 0)
            elif (cfg.wrench_constraint and _NCF_AVAILABLE
                  and cfg.wrench_mode == 'embedded'):
                # embedded: LP constraints added directly to NLP as bilinear exprs
                _task_w_np = np.array([cfg.task_tx, cfg.task_ty, cfg.task_tz,
                                       cfg.task_fx, cfg.task_fy, cfg.task_fz])
                _gamma_lp, _y1_list, _y2_list = _embed_wrench_cone_ca(
                    _opti, _p1, _p2, obj_center_np, d1, d2, cfg.mu, _task_w_np)
                _opti.set_initial(_gamma_lp, 1.0)
                _y_init = np.ones(5) / 5.0
                for _y1_k, _y2_k in zip(_y1_list, _y2_list):
                    _opti.set_initial(_y1_k, _y_init)
                    _opti.set_initial(_y2_k, _y_init)

            # Finalize cost (add gamma term after embedded variables exist)
            if _gamma_lp is not None:
                _opti.minimize(_cost + cfg.w_gamma * _gamma_lp)
            else:
                _opti.minimize(_cost)

            # ── 5a. Full-arm collision (geometry-appropriate softplus SDF) ─
            if arm_col_cb is not None:
                _arm_pos = arm_col_cb(_q)   # (3*n_active,) CasADi vector
                _obj_R_dm = ca.DM(obj_R_np)
                _obj_c_dm = ca.DM(obj_center_np)
                _ground_n = ca.DM([0.0, 0.0, 1.0])
                _ground_p = ca.DM([0.0, 0.0, 0.0])
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

            # ── 5b. Legacy middle+ring collision ──────────────────────────
            elif use_legacy_col:
                _col_out  = col_cb(_q)
                _tm, _tr  = _col_out[0:3], _col_out[3:6]
                _d_min    = float(cfg.finger_radius_m + cfg.col_clearance_m)
                _opti.subject_to(_sdf(_tm) >= _d_min)
                _opti.subject_to(_sdf(_tr) >= _d_min)

            # ── Initial guess ─────────────────────────────────────────────
            _opti.set_initial(_q,  q_ws)
            _opti.set_initial(_p1, p1_ws)
            _opti.set_initial(_p2, p2_ws)

            # ── Solver ────────────────────────────────────────────────────
            if cfg.use_slsqp:
                _sqp_opts = dict(_SQP_SOLVER_OPTS)
                _sqp_opts['max_iter'] = max_iter_override or cfg.max_iter
                _opti.solver('sqpmethod', _sqp_opts)
            else:
                # IPOPT path — uses analytic Jacobians from all callbacks
                # (FK via mj_jacSite, wrench via LP duals, collision via mj_jac).
                # Do NOT set jacobian_approximation='finite-difference-values': that
                # would override the wrench callback's LP-dual Jacobian with FD of
                # a piecewise-constant function, producing garbage search directions.
                _ipopt_opts: dict = {
                    'hessian_approximation':  'limited-memory',
                    'max_iter':               max_iter_override or cfg.max_iter,
                    'sb':                     'no',
                    'tol':                    1e-4,
                    'dual_inf_tol':           1.0,
                    'constr_viol_tol':        1e-6,
                    'print_level':            0,
                    'mu_strategy':            'adaptive',
                    'acceptable_tol':         1e-3,
                    'acceptable_iter':        20,
                    'acceptable_constr_viol_tol': 1e-3,
                }
                if self.log_dir:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    _ipopt_opts['output_file']      = os.path.join(
                        self.log_dir, f"grasp3d_ipopt_{ts}.log")
                    _ipopt_opts['file_print_level'] = 5
                _opti.solver('ipopt', {'ipopt': _ipopt_opts, 'print_time': True})

            # ── Per-iteration logger (DEBUG → file only) ──────────────────
            if self.log_dir:
                _tag = f'[{stage_label}|iter] ' if stage_label else '[iter] '
                _iter_exprs: list[tuple[str, object]] = [
                    ('f',        _opti.f),
                    ('ik_th_mm', ca.sqrt(_d1_sq) * 1e3),
                    ('ik_if_mm', ca.sqrt(_d2_sq) * 1e3),
                ]
                if include_surface and d1 is not None:
                    _ax1 = int(np.argmax(np.abs(d1)))
                    _fc1 = float(obj_center_np[_ax1] +
                                 np.sign(d1[_ax1]) * float(geom_size[_ax1]))
                    _iter_exprs.append(('face_p1_mm',
                                        ca.fabs(_p1[_ax1] - _fc1) * 1e3))
                if include_surface and d2 is not None:
                    _ax2 = int(np.argmax(np.abs(d2)))
                    _fc2 = float(obj_center_np[_ax2] +
                                 np.sign(d2[_ax2]) * float(geom_size[_ax2]))
                    _iter_exprs.append(('face_p2_mm',
                                        ca.fabs(_p2[_ax2] - _fc2) * 1e3))
                if use_legacy_col:
                    _d_min = float(cfg.finger_radius_m + cfg.col_clearance_m)
                    _iter_exprs += [
                        ('col_mf', _sdf(_tm) - _d_min),
                        ('col_rf', _sdf(_tr) - _d_min),
                    ]

                def _iter_cb(i):
                    parts = [f"i={i:3d}"]
                    for _lbl, _expr in _iter_exprs:
                        try:
                            v = float(_opti.debug.value(_expr))
                            parts.append(f"{_lbl}={v:+.3e}")
                        except Exception:
                            parts.append(f"{_lbl}=?")
                    self.log.debug(_tag + "  ".join(parts))

                _opti.callback(_iter_cb)

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
                wf_calls  = wrench_cb._n_calls if wrench_cb is not None else 0

                if wrench_cb is not None:
                    _wrench_str = cfg.wrench_mode   # 'hard' or 'elastic'
                elif cfg.wrench_constraint and _NCF_AVAILABLE and cfg.wrench_mode == 'embedded':
                    _wrench_str = 'embedded'
                else:
                    _wrench_str = 'N'

                con_str = (
                    f"surface={'Y' if include_surface else 'N'}  "
                    f"wrench={_wrench_str}  "
                    f"col_legacy={'Y' if use_legacy_col else 'N'}  "
                    f"arm_col={'Y(' + str(len(_active_arm)) + 'geoms)' if _active_arm else 'N'}")
                lines = [
                    f"{tag}─── solve profile ──────────────────────────────────────",
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
                lines.append(f"{tag}────────────────────────────────────────────────────────")
                for ln in lines:
                    self.log.info(ln)
                if cfg.verbose_profile:
                    for ln in lines:
                        print(ln)

            # ── Solve ─────────────────────────────────────────────────────
            _t_solve_start = time.perf_counter()
            try:
                _sol = _opti.solve()
                _plog(_sol.stats())
                return {
                    'success':    True,
                    'q':          _sol.value(_q),
                    'p1':         _sol.value(_p1),
                    'p2':         _sol.value(_p2),
                    'cost':       float(_sol.value(_opti.f)),
                    'iterations': _sol.stats()['iter_count'],
                    'status':     'converged',
                }
            except Exception as _e:
                self.log.warning(f"GraspPlanner3D._run_stage({stage_label}): {_e}")
                try:    _st = _opti.stats()
                except: _st = None
                _plog(_st)
                try:
                    return {
                        'success':    False,
                        'q':          _opti.debug.value(_q),
                        'p1':         _opti.debug.value(_p1),
                        'p2':         _opti.debug.value(_p2),
                        'cost':       _opti.debug.value(_opti.f) if _st else None,
                        'iterations': (_st or {}).get('iter_count'),
                        'status':     'best-effort',
                    }
                except Exception as _e2:
                    self.log.error(f"GraspPlanner3D debug extraction: {_e2}")
                    return {'success': False, 'q': None, 'p1': None, 'p2': None,
                            'cost': None, 'iterations': None, 'status': 'failed'}

        # ── Run optimisation ─────────────────────────────────────────────────
        res = _run_stage(q_dls, p1_seed, p2_seed,
                         include_surface=True,
                         stage_label='S1')

        if self.dash is not None:
            self.dash.push({
                'type':   'ipopt',
                'phase':  'grasp3d',
                'status': res.get('status', '?'),
                'iters':  res.get('iterations', '?'),
            })
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

        ik_t = float(np.linalg.norm(data_v.site_xpos[self._thumb_sid] - result['p1']))
        ik_i = float(np.linalg.norm(data_v.site_xpos[self._index_sid] - result['p2']))

        gap_t = mj.mj_geomDistance(model, data_v, self._thumb_gid,  self._obj_gid, 0.5, None)
        gap_i = mj.mj_geomDistance(model, data_v, self._index_gid,  self._obj_gid, 0.5, None)
        gap_m = mj.mj_geomDistance(model, data_v, self._middle_gid, self._obj_gid, 0.5, None)
        gap_r = mj.mj_geomDistance(model, data_v, self._ring_gid,   self._obj_gid, 0.5, None)

        obj_pos = data_v.geom_xpos[self._obj_gid].copy()
        obj_mat = data_v.geom_xmat[self._obj_gid].reshape(3, 3)

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
                gamma_min = min_gamma_for_accel_lp(
                    cfg.task_fx, cfg.task_fy, cfg.task_fz,
                    cfg.task_tx, cfg.task_ty, cfg.task_tz,
                    n=2,
                    pos=[(p1_np - obj_pos).reshape(3, 1),
                         (p2_np - obj_pos).reshape(3, 1)],
                    R=[R1, R2],
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
            'ik_thumb_mm':       ik_t * 1000,
            'ik_index_mm':       ik_i * 1000,
            'gap_thumb_mm':      gap_t * 1000,
            'gap_index_mm':      gap_i * 1000,
            'gap_middle_mm':     gap_m * 1000,
            'gap_ring_mm':       gap_r * 1000,
            'sdf_p1_mm':         s1   * 1000,
            'sdf_p2_mm':         s2   * 1000,
            'sdf_middle_tip_mm': s3   * 1000,
            'sdf_ring_tip_mm':   s4   * 1000,
            'wrench_feasible':   wf_feasible,
            'gamma_min':         gamma_min,
        }
        self.log.info(
            f"[verify3d] IK=({ik_t*1e3:.2f},{ik_i*1e3:.2f})mm "
            f"GAP=({gap_t*1e3:+.2f},{gap_i*1e3:+.2f},"
            f"{gap_m*1e3:+.2f},{gap_r*1e3:+.2f})mm "
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

    def solve(self, q_ref: np.ndarray, obj_pos: np.ndarray,
              max_seeds: int | None = None) -> dict:
        """Try shape-appropriate seeds (up to max_seeds), return best result."""
        c    = np.asarray(obj_pos, float)
        cfg  = self._planner.cfg
        dash = self._planner.dash

        obj_gid = self._planner._obj_gid
        obj_mat = self._planner.data.geom_xmat[obj_gid].reshape(3, 3).copy()

        all_seeds = _geom_seeds(self._obj_geom_type, self._obj_size,
                                n_radial=cfg.n_radial_seeds)
        seeds   = all_seeds[:max_seeds] if max_seeds else all_seeds
        n_seeds = len(seeds)
        results = []

        for i, (d_th, d_if, p1_off, p2_off) in enumerate(seeds):
            d_th_w  = obj_mat @ d_th
            d_if_w  = obj_mat @ d_if
            p1_init = c + obj_mat @ p1_off
            p2_init = c + obj_mat @ p2_off

            if dash is not None:
                dash.push({'type': 'active',
                           'label': f'grasp3d seed {i+1}/{n_seeds}'})

            r = self._planner.solve(q_ref, obj_pos,
                                    p1_init=p1_init, p2_init=p2_init,
                                    d1=d_th_w, d2=d_if_w)
            results.append(r)
            if r['status'] == 'converged':
                self._planner.log.info(
                    f"[multistart] seed {i+1}/{n_seeds} converged — stopping early")
                break

        def _rank(r):
            s = {'converged': 0, 'best-effort': 1, 'failed': 2}.get(
                r.get('status', 'failed'), 2)
            return (s, r.get('cost') or 1e9)

        results.sort(key=_rank)
        best = results[0]
        best['all_results'] = results
        return best

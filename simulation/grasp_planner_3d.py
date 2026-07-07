"""
grasp_planner_3d.py
===================
3D grasp contact-point solver components.

This module contains geometry utilities and CasADi callbacks for a 3D
two-finger grasp optimiser targeting the Kinova Gen3 + LEAP hand system
with a 6-DOF free object.

Public symbols
--------------
    _box_sdf_3d               – signed distance from point to 3D box
    _box_surface_normal_3d    – outward normal on nearest box face
    _build_contact_frame_3d   – orthonormal (n, t1, t2) from a normal
    _WrenchFeasCallback3D     – CasADi callback: 7D input, scalar output

Wrench convention (throughout this file)
-----------------------------------------
    [Tx, Ty, Tz, Fx, Fy, Fz]  (torque first, consistent with 3D_minimum_NCF.py)

Contact frame convention
------------------------
    R[:,0] = inward normal (compressive direction, into object)
    R[:,1] = first tangent
    R[:,2] = second tangent

    Force in contact frame: f_c = [f_n, f_t1, f_t2]
    Force in world frame:   f_w = R @ f_c

Friction cone (Coulomb, square pyramid approximation)
------------------------------------------------------
    f_n ≥ 0,   sqrt(f_t1² + f_t2²) ≤ mu * f_n
    Pyramid vertices: apex (0,0,0) + 4 edges
        (f_n, ±mu*f_n, 0) and (f_n, 0, ±mu*f_n)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 3D WrenchCheck lives in scripts/ (one level above simulation/)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    import casadi as ca
    from casadi import Callback as _CasadiCallback
    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False
    _CasadiCallback = object   # dummy so class body can parse


# ─────────────────────────────────────────────────────────────────────────────
# Geometry primitives
# ─────────────────────────────────────────────────────────────────────────────

def _box_sdf_3d(point, center, hx: float, hy: float, hz: float) -> float:
    """
    Signed distance from *point* to the surface of an axis-aligned box.

    Returns
    -------
    float
        < 0  inside the box
          0  on the surface
        > 0  outside the box
    """
    d = np.asarray(point, float) - np.asarray(center, float)
    q = np.array([abs(d[0]) - hx, abs(d[1]) - hy, abs(d[2]) - hz])
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))


def _box_surface_normal_3d(point, center, hx: float, hy: float, hz: float) -> np.ndarray:
    """
    Outward unit normal at the nearest face of a 3D axis-aligned box.

    The face is determined by which normalised axis the point is furthest
    along relative to the box half-extents.  Corner/edge ties are broken
    by axis order (x > y > z).
    """
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
    Build a right-handed orthonormal frame from an inward contact normal.

    Parameters
    ----------
    inward_normal : array (3,)
        Unit vector pointing INTO the object (compressive direction).

    Returns
    -------
    n, t1, t2 : np.ndarray (3,) each
        n  = normalised inward_normal
        t1 = first tangent (perp to n)
        t2 = second tangent (perp to n and t1, right-handed)
    """
    n = np.asarray(inward_normal, float)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("inward_normal is degenerate (near-zero norm)")
    n = n / norm
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    return n, t1, t2


def _hat(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix such that hat(v) @ u == cross(v, u)."""
    v = np.asarray(v, float).flatten()
    return np.array([[ 0.0,  -v[2],  v[1]],
                      [ v[2],  0.0,  -v[0]],
                      [-v[1],  v[0],  0.0 ]])


# ─────────────────────────────────────────────────────────────────────────────
# Wrench cone utilities (stand-alone, no CasADi)
# ─────────────────────────────────────────────────────────────────────────────

def _friction_cone_verts_3d(ncf_scale: float, gamma: float, mu: float,
                             R: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Compute wrench cone vertices for one contact in 3D.

    Uses a square pyramid to approximate the circular Coulomb cone:
        apex:   f_c = [0, 0, 0]
        edge 1: f_c = [γ*ncf, 0,        -mu*γ*ncf]
        edge 2: f_c = [γ*ncf, -mu*γ*ncf, 0       ]
        edge 3: f_c = [γ*ncf, +mu*γ*ncf, 0       ]
        edge 4: f_c = [γ*ncf, 0,        +mu*γ*ncf]

    Parameters
    ----------
    ncf_scale : float
        Normal contact force component of the null-space internal force (unit).
    gamma : float
        Internal force scale (the optimisation variable).
    mu : float
        Coulomb friction coefficient.
    R : np.ndarray (3,3)
        Contact frame: R[:,0]=inward_normal, R[:,1]=t1, R[:,2]=t2.
    r : np.ndarray (3,)
        Contact arm vector: p_contact - object_center.

    Returns
    -------
    verts : np.ndarray (5, 6)
        Five 6D wrench vertices [Tx, Ty, Tz, Fx, Fy, Fz] in body/world frame.
    """
    g_ncf = gamma * ncf_scale
    fc_verts = np.array([
        [0.0,    0.0,        0.0      ],   # apex (zero force)
        [g_ncf,  0.0,       -mu*g_ncf],   # edge -t2
        [g_ncf, -mu*g_ncf,  0.0      ],   # edge -t1
        [g_ncf, +mu*g_ncf,  0.0      ],   # edge +t1
        [g_ncf,  0.0,       +mu*g_ncf],   # edge +t2
    ])  # (5, 3) in contact frame

    verts = []
    for f_c in fc_verts:
        f_w = R @ f_c               # (3,) force in world frame
        t_w = np.cross(r, f_w)     # (3,) torque about object center
        verts.append(np.concatenate([t_w, f_w]))  # [Tx,Ty,Tz,Fx,Fy,Fz]
    return np.array(verts)         # (5, 6)


def compute_internal_force_3d(p1: np.ndarray, p2: np.ndarray, c: np.ndarray,
                               hx: float, hy: float, hz: float):
    """
    Compute the contact geometry and internal-force direction for two contacts
    on a 3D axis-aligned box.

    The internal force is the null-space direction of the 3D grasp matrix G.
    G maps stacked contact forces [f1_c; f2_c] in their contact frames to the
    6D object wrench [Tx, Ty, Tz, Fx, Fy, Fz].

    Parameters
    ----------
    p1, p2 : np.ndarray (3,)
        Contact point positions in world frame.
    c : np.ndarray (3,)
        Object center in world frame.
    hx, hy, hz : float
        Object half-extents.

    Returns
    -------
    dict with keys:
        n1_out, n2_out  : (3,) outward normals
        R1, R2          : (3,3) contact frames (inward normal convention)
        r1, r2          : (3,) contact arm vectors
        ncf1, tan_y1, tan_z1 : internal force components at contact 1 (unit)
        ncf2, tan_y2, tan_z2 : internal force components at contact 2 (unit)
    """
    n1_out = _box_surface_normal_3d(p1, c, hx, hy, hz)
    n2_out = _box_surface_normal_3d(p2, c, hx, hy, hz)

    _, t1_1, t2_1 = _build_contact_frame_3d(-n1_out)  # inward normal
    _, t1_2, t2_2 = _build_contact_frame_3d(-n2_out)

    R1 = np.column_stack([-n1_out, t1_1, t2_1])  # (3,3): columns = [n_in, t1, t2]
    R2 = np.column_stack([-n2_out, t1_2, t2_2])

    r1 = p1 - c
    r2 = p2 - c

    # 3D grasp matrix G (6×6): maps [f1_c; f2_c] → [Tx,Ty,Tz,Fx,Fy,Fz]
    G = np.block([
        [_hat(r1) @ R1,  _hat(r2) @ R2],   # torque rows  (3×6)
        [R1,              R2            ],   # force rows   (3×6)
    ])

    _, _, Vt = np.linalg.svd(G)
    f_int = Vt[-1]  # (6,) in [f1n, f1t1, f1t2, f2n, f2t1, f2t2] contact frame

    # Ensure contact 1 pushes INTO the object (positive normal)
    if f_int[0] < 0:
        f_int = -f_int

    return {
        'n1_out': n1_out, 'n2_out': n2_out,
        'R1': R1, 'R2': R2,
        'r1': r1, 'r2': r2,
        'ncf1':   float(f_int[0]), 'tan_y1': float(f_int[1]), 'tan_z1': float(f_int[2]),
        'ncf2':   float(f_int[3]), 'tan_y2': float(f_int[4]), 'tan_z2': float(f_int[5]),
    }


def wrench_feas_margin_3d(p1: np.ndarray, p2: np.ndarray, gamma: float,
                           obj_center: np.ndarray,
                           hx: float, hy: float, hz: float,
                           mu: float,
                           task_bounds: dict,
                           geom: dict | None = None) -> float:
    """
    Compute the wrench feasibility margin for a 3D two-finger grasp.

    Checks whether all 2^6 = 64 corners of the task wrench box can be
    balanced by contact forces inside the friction cones at squeeze scale γ.

    Parameters
    ----------
    p1, p2 : np.ndarray (3,)
        Contact positions in world frame.
    gamma : float
        Internal force scale.
    obj_center : np.ndarray (3,)
        Object center in world frame.
    hx, hy, hz : float
        Object half-extents.
    mu : float
        Coulomb friction coefficient.
    task_bounds : dict
        Keys: fx, fy, fz, tx, ty, tz  (half-widths of the task wrench box).
        Wrench convention: [Tx, Ty, Tz, Fx, Fy, Fz].
    geom : dict or None
        Pre-computed geometry from compute_internal_force_3d() (avoids
        recomputation when only gamma changes).

    Returns
    -------
    float
        > 0  : margin (all task wrench corners are feasible)
          0  : on the boundary
        < 0  : infeasible by this amount (most-violated corner)
        -inf : completely degenerate (fallback)
    """
    from scipy.spatial import ConvexHull

    c = np.asarray(obj_center, float)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)

    if geom is None:
        geom = compute_internal_force_3d(p1, p2, c, hx, hy, hz)

    R1, R2 = geom['R1'], geom['R2']
    r1, r2 = geom['r1'], geom['r2']
    ncf1, tan_y1, tan_z1 = geom['ncf1'], geom['tan_y1'], geom['tan_z1']
    ncf2, tan_y2, tan_z2 = geom['ncf2'], geom['tan_y2'], geom['tan_z2']
    n1_out = geom['n1_out']

    # Wrench cone vertices per contact (5 vertices each)
    verts1 = _friction_cone_verts_3d(ncf1, gamma, mu, R1, r1)  # (5,6)
    verts2 = _friction_cone_verts_3d(ncf2, gamma, mu, R2, r2)  # (5,6)

    # Minkowski sum: 5 × 5 = 25 vertices in 6D wrench space
    vert_sums = np.array([v1 + v2 for v1 in verts1 for v2 in verts2])  # (25,6)

    if not np.all(np.isfinite(vert_sums)):
        return _antipodal_fallback_3d(p1, p2, n1_out, hx, hy, hz)

    # Reduce to non-degenerate dimensions (std > tol)
    active = np.where(vert_sums.std(axis=0) > 1e-10)[0]
    if len(active) < 2:
        return _antipodal_fallback_3d(p1, p2, n1_out, hx, hy, hz)

    try:
        hull = ConvexHull(vert_sums[:, active])
    except Exception:
        return _antipodal_fallback_3d(p1, p2, n1_out, hx, hy, hz)

    # Tangential internal force contribution to wrench (delta)
    tan1_c = np.array([0.0, gamma * tan_y1, gamma * tan_z1])
    tan2_c = np.array([0.0, gamma * tan_y2, gamma * tan_z2])
    tan1_w = R1 @ tan1_c           # (3,) force in world frame
    tan2_w = R2 @ tan2_c
    delta = np.concatenate([
        np.cross(r1, tan1_w) + np.cross(r2, tan2_w),   # [dTx, dTy, dTz]
        tan1_w + tan2_w,                                # [dFx, dFy, dFz]
    ])  # (6,)

    # Task wrench bounds → 2^6 = 64 corners
    tx = task_bounds.get('tx', 0.0)
    ty = task_bounds.get('ty', 0.0)
    tz = task_bounds.get('tz', 0.0)
    fx = task_bounds.get('fx', 0.0)
    fy = task_bounds.get('fy', 0.0)
    fz = task_bounds.get('fz', 0.0)

    min_rhs = np.inf
    for sign_tx in (-tx, tx):
        for sign_ty in (-ty, ty):
            for sign_tz in (-tz, tz):
                for sign_fx in (-fx, fx):
                    for sign_fy in (-fy, fy):
                        for sign_fz in (-fz, fz):
                            w = np.array([sign_tx, sign_ty, sign_tz,
                                          sign_fx, sign_fy, sign_fz]) + delta
                            w_a = w[active]
                            for eq in hull.equations:
                                # scipy hull: eq[:-1] @ x + eq[-1] <= 0 (interior)
                                # Negate: positive = interior (feasible)
                                val = float(-(np.dot(eq[:-1], w_a) + eq[-1]))
                                if val < min_rhs:
                                    min_rhs = val

    if not np.isfinite(min_rhs):
        return _antipodal_fallback_3d(p1, p2, n1_out, hx, hy, hz)
    return float(min_rhs)


def _antipodal_fallback_3d(p1: np.ndarray, p2: np.ndarray,
                            n1_out: np.ndarray,
                            hx: float, hy: float, hz: float) -> float:
    """Smooth, finite signal for degenerate contact configurations.

    Returns a value in [-10, 0) that gives IPOPT a gradient pushing
    contacts toward antipodal configuration (opposite faces).

    sep > 0 → contacts on opposite sides (antipodal) → neg return ≈ -1
    sep < 0 → contacts on same side (bad)            → neg return → -10
    """
    sep = float(np.dot(p2 - p1, n1_out))
    ref = 2.0 * max(hx, hy, hz)
    val = -sep / ref - 1.0
    return float(np.clip(val, -10.0, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# CasADi callback
# ─────────────────────────────────────────────────────────────────────────────

class _WrenchFeasCallback3D(_CasadiCallback):
    """
    CasADi external callback: 3D wrench feasibility for two contacts on a box.

    Input:  (7,1) — [p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, gamma]
    Output: scalar — min margin across all 64 task-wrench corners.
            >= 0  →  all corners wrench-feasible at this gamma
             < 0  →  violated (infeasible)

    Geometry (contact normals, frames, internal force direction) is cached
    whenever p1/p2 are unchanged so only the wrench cone vertices (which
    depend on gamma) are recomputed each IPOPT iteration.

    Task wrench box: [−tx, tx] × [−ty, ty] × [−tz, tz] × [−fx, fx] × [−fy, fy] × [−fz, fz]
    Convention: [Tx, Ty, Tz, Fx, Fy, Fz] (torque first).
    """

    def __init__(self, name: str,
                 obj_pos,
                 obj_hx: float, obj_hy: float, obj_hz: float,
                 mu: float, obj_mass: float,
                 task_accel_x: float, task_accel_y: float, task_accel_z: float,
                 task_torque_x: float, task_torque_y: float, task_torque_z: float):
        if not _CASADI_AVAILABLE:
            raise ImportError("casadi is required for _WrenchFeasCallback3D")
        _CasadiCallback.__init__(self)

        self._c   = np.asarray(obj_pos, float)[:3]
        self._hx  = obj_hx
        self._hy  = obj_hy
        self._hz  = obj_hz
        self._mu  = mu

        # Task wrench half-widths
        self._tx = task_torque_x
        self._ty = task_torque_y
        self._tz = task_torque_z
        self._fx = obj_mass * task_accel_x
        self._fy = obj_mass * task_accel_y
        self._fz = obj_mass * task_accel_z

        self._n_calls  = 0
        self._t_total  = 0.0
        self._geom_key   = None
        self._geom_cache = None   # dict from compute_internal_force_3d()

        self.construct(name, {"enable_fd": True})

    # ── CasADi interface ──────────────────────────────────────────────────────

    def get_n_in(self):   return 1
    def get_n_out(self):  return 1

    def get_sparsity_in(self, i):  return ca.Sparsity.dense(7, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    # ── eval ─────────────────────────────────────────────────────────────────

    def eval(self, arg):
        import time as _time
        _t0 = _time.perf_counter()

        x     = np.array(arg[0]).flatten()
        p1    = x[0:3]
        p2    = x[3:6]
        gamma = float(x[6])

        # Cache geometry (expensive SVD) when p1/p2 unchanged
        geom_key = tuple(np.round(x[:6], 6))
        if geom_key != self._geom_key:
            try:
                self._geom_cache = compute_internal_force_3d(
                    p1, p2, self._c, self._hx, self._hy, self._hz)
            except Exception:
                self._geom_cache = None
            self._geom_key = geom_key

        if self._geom_cache is None:
            self._n_calls += 1
            self._t_total += _time.perf_counter() - _t0
            n1_out = _box_surface_normal_3d(p1, self._c, self._hx, self._hy, self._hz)
            return [_antipodal_fallback_3d(p1, p2, n1_out, self._hx, self._hy, self._hz)]

        margin = wrench_feas_margin_3d(
            p1, p2, gamma,
            self._c,
            self._hx, self._hy, self._hz,
            self._mu,
            task_bounds=dict(
                tx=self._tx, ty=self._ty, tz=self._tz,
                fx=self._fx, fy=self._fy, fz=self._fz,
            ),
            geom=self._geom_cache,
        )

        self._n_calls += 1
        self._t_total += _time.perf_counter() - _t0
        return [float(margin)]


# ─────────────────────────────────────────────────────────────────────────────
# MuJoCo CasADi callbacks (require mujoco + casadi)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import mujoco as mj
    _MJ_AVAILABLE = True
except ImportError:
    _MJ_AVAILABLE = False

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger("grasp_planner_3d")
if not log.handlers:
    log.addHandler(logging.NullHandler())


def _get_actuated_indices(model) -> list[int]:
    """Return the qpos address for each actuated joint, in actuator order."""
    return [model.jnt_qposadr[model.actuator_trnid[i, 0]]
            for i in range(model.nu)]


class _FingerPointDistCallback3D(ca.Callback):
    """
    (q[nu], p[3]) → mj_geomDistance(finger_geom, dummy_sphere_at_p)

    Returns surface-to-surface signed distance:
        0  => finger surface is touching the sphere at p
        >0 => gap
        <0 => penetration
    """
    def __init__(self, name, model, data_cb,
                 finger_gid, cp_gid, cp_mocap_id, act_idx, cutoff=5):
        ca.Callback.__init__(self)
        self.model = model
        self.data  = data_cb
        self.finger_gid  = finger_gid
        self.cp_gid      = cp_gid
        self.cp_mocap_id = cp_mocap_id
        self.act_idx     = act_idx
        self.cutoff      = cutoff
        self._n_calls = 0
        self._t_total = 0.0
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return (ca.Sparsity.dense(len(self.act_idx), 1) if i == 0
                else ca.Sparsity.dense(3, 1))       # p is 3D

    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        import time as _time
        _t0 = _time.perf_counter()
        q_act = np.array(arg[0]).flatten()
        p     = np.array(arg[1]).flatten()
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val
        self.data.mocap_pos[self.cp_mocap_id] = p[:3]   # full 3D
        mj.mj_forward(self.model, self.data)
        dist = mj.mj_geomDistance(self.model, self.data,
                                   self.finger_gid, self.cp_gid,
                                   self.cutoff, np.zeros(6))
        self._n_calls += 1
        self._t_total += _time.perf_counter() - _t0
        return [dist]


class _SDFCallback3D(ca.Callback):
    """
    (q[nu], p[3]) → signed distance from 3D point p to object geom surface.

    Handles object rotation by transforming p into the geom's local frame
    before applying the analytical box SDF.  Works even when the object has
    a freejoint (i.e. can translate and rotate).

    Sign convention (same as _box_sdf_3d):
        < 0  point is inside the box (on-surface target: sdf == −2*r_cp)
          0  point is on the surface
        > 0  point is outside the box
    """
    def __init__(self, name, model, data_cb, geom_id, act_idx):
        ca.Callback.__init__(self)
        self.model   = model
        self.data    = data_cb
        self.geom_id = geom_id
        self.act_idx = act_idx
        self._n_calls = 0
        self._t_total = 0.0
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return (ca.Sparsity.dense(len(self.act_idx), 1) if i == 0
                else ca.Sparsity.dense(3, 1))

    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        import time as _time
        _t0 = _time.perf_counter()
        q_act = np.array(arg[0]).flatten()
        point = np.array(arg[1]).flatten()
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val
        mj.mj_forward(self.model, self.data)

        # Object geom pose
        gid = self.geom_id
        obj_pos = self.data.geom_xpos[gid].copy()          # (3,) world frame
        obj_mat = self.data.geom_xmat[gid].reshape(3, 3)   # rotation world←geom
        geom_size = self.model.geom_size[gid]

        geom_type = self.model.geom_type[gid]
        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            # Transform point to geom-local frame (handles object rotation)
            p_local = obj_mat.T @ (point - obj_pos)
            dist = _box_sdf_3d(p_local, np.zeros(3),
                               geom_size[0], geom_size[1], geom_size[2])
        elif geom_type == mj.mjtGeom.mjGEOM_SPHERE:
            dist = float(np.linalg.norm(point - obj_pos) - geom_size[0])
        elif geom_type == mj.mjtGeom.mjGEOM_CYLINDER:
            # Cylinder with half-height = geom_size[1], radius = geom_size[0]
            p_local = obj_mat.T @ (point - obj_pos)
            r_xy = float(np.linalg.norm(p_local[:2])) - geom_size[0]
            r_z  = abs(p_local[2]) - geom_size[1]
            dist = float(np.sqrt(max(r_xy, 0)**2 + max(r_z, 0)**2)
                         + min(max(r_xy, r_z), 0.0))
        else:
            # Fallback: distance from geom center
            dist = float(np.linalg.norm(point - obj_pos))

        self._n_calls += 1
        self._t_total += _time.perf_counter() - _t0
        return [dist]


class _NonPenetrationCallback3D(ca.Callback):
    """
    (q[nu],) → mj_geomDistance(finger_geom, obj_geom)

    Positive → gap (no penetration), zero → touching, negative → penetrating.
    Identical to the 2D version — mj_geomDistance is always 3D internally.
    """
    def __init__(self, name, model, data_cb, finger_gid, obj_gid, act_idx,
                 cutoff=0.5):
        ca.Callback.__init__(self)
        self.model      = model
        self.data       = data_cb
        self.finger_gid = finger_gid
        self.obj_gid    = obj_gid
        self.act_idx    = act_idx
        self.cutoff     = cutoff
        self._n_calls = 0
        self._t_total = 0.0
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i): return ca.Sparsity.dense(len(self.act_idx), 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        import time as _time
        _t0 = _time.perf_counter()
        q_act = np.array(arg[0]).flatten()
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val
        mj.mj_forward(self.model, self.data)
        dist = mj.mj_geomDistance(self.model, self.data,
                                   self.finger_gid, self.obj_gid,
                                   self.cutoff, None)
        self._n_calls += 1
        self._t_total += _time.perf_counter() - _t0
        return [dist]


class _AnalyticalSDFCallback3D(ca.Callback):
    """
    (p[3],) → signed distance from point p to the object geom surface.

    The object pose is captured ONCE at construction time (no MuJoCo FK
    during eval).  This makes the callback ~200× faster than _SDFCallback3D
    and reduces the FD perturbation count from 26 to 3.

    Works only for a fixed-pose object (set qpos then mj_forward before
    constructing this callback).
    """
    def __init__(self, name: str, model, data_cb, geom_id: int):
        ca.Callback.__init__(self)
        # Freeze the object geometry in world frame
        self.obj_pos = data_cb.geom_xpos[geom_id].copy()
        self.obj_mat = data_cb.geom_xmat[geom_id].reshape(3, 3).copy()
        gs = model.geom_size[geom_id]
        self.hx, self.hy, self.hz = float(gs[0]), float(gs[1]), float(gs[2])
        self.geom_type = model.geom_type[geom_id]
        self._n_calls = 0
        self._t_total = 0.0
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1   # only p[3]
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):  return ca.Sparsity.dense(3, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        import time as _time
        _t0 = _time.perf_counter()
        p = np.array(arg[0]).flatten()

        if self.geom_type == 6:   # mjGEOM_BOX = 6
            p_local = self.obj_mat.T @ (p - self.obj_pos)
            dist = _box_sdf_3d(p_local, np.zeros(3),
                               self.hx, self.hy, self.hz)
        elif self.geom_type == 2:  # mjGEOM_SPHERE = 2
            dist = float(np.linalg.norm(p - self.obj_pos) - self.hx)
        else:
            # Cylinder: hx=radius, hy=half-height
            p_local = self.obj_mat.T @ (p - self.obj_pos)
            r_xy = float(np.linalg.norm(p_local[:2])) - self.hx
            r_z  = abs(float(p_local[2])) - self.hy
            dist = float(np.sqrt(max(r_xy, 0)**2 + max(r_z, 0)**2)
                         + min(max(r_xy, r_z), 0.0))

        self._n_calls += 1
        self._t_total += _time.perf_counter() - _t0
        return [dist]


# ─────────────────────────────────────────────────────────────────────────────
# GraspConfig3D
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraspConfig3D:
    """Configuration for the 3D grasp planner (Kinova Gen3 + LEAP hand)."""

    # Cost weights
    w_ik:      float = 0.4     # IK residual (usually kept hard via constraint)
    w_reg:     float = 0.3    # joint regularisation toward q_ref
    w_surface: float = 0.3     # soft surface SDF penalty (when on_object=False)
    q_scale:   float = 1.0
    sdf_scale: float = 1.0

    # Constraint flags
    joint_limits:           bool = False
    on_object:              bool = True   # p1,p2 hard-constrained on surface
    ik_constraint:          bool = True   # finger must touch contact point
    penetration_constraint: bool = False   # finger gap == -1e-4 (redundant with IK; enable if needed)
    max_iter:               int  = 120

    # Geometry names — must match scene_pick_place_pos.xml
    obj_geom:    str = 'obj_box_geom'
    obj_body:    str = 'obj_box'
    thumb_geom:  str = 'leap_th_tip'     # thumb distal tip mesh geom
    index_geom:  str = 'leap_if_tip'     # index finger distal tip mesh geom
    middle_geom: str = 'leap_mf_tip'     # middle finger distal tip mesh geom
    ring_geom:   str = 'leap_rf_tip'     # ring finger distal tip mesh geom
    cp1_geom:    str = 'cp1'             # contact marker sphere — thumb
    cp2_geom:    str = 'cp2'             # contact marker sphere — index
    cp3_geom:    str = 'cp3'             # contact marker sphere — middle
    cp4_geom:    str = 'cp4'             # contact marker sphere — ring
    cp1_body:    str = 'cp1_body'
    cp2_body:    str = 'cp2_body'
    cp3_body:    str = 'cp3_body'
    cp4_body:    str = 'cp4_body'


# 4-finger contact-point seeds: (d_thumb, d_index, d_middle, d_ring).
# Each d_i is element-wise multiplied by (hx, hy, hz) and added to the object
# centre to produce the warm-start contact point.
# S = lateral spread factor — index and ring straddle middle by S * half-size.
_S = 0.5

_FACE_SEEDS_4F_UNIT = [
    # Thumb -Y, I/M/R on +Y face spread in X
    (np.array([ 0.0, -1.0,  0.0]), np.array([ _S,  1.0, 0.0]),
     np.array([ 0.0,  1.0,  0.0]), np.array([-_S,  1.0, 0.0])),
    # Thumb +Y, I/M/R on -Y face
    (np.array([ 0.0,  1.0,  0.0]), np.array([-_S, -1.0, 0.0]),
     np.array([ 0.0, -1.0,  0.0]), np.array([ _S, -1.0, 0.0])),
    # Thumb -X, I/M/R on +X face spread in Y
    (np.array([-1.0,  0.0,  0.0]), np.array([ 1.0,  _S, 0.0]),
     np.array([ 1.0,  0.0,  0.0]), np.array([ 1.0, -_S, 0.0])),
    # Thumb +X, I/M/R on -X face
    (np.array([ 1.0,  0.0,  0.0]), np.array([-1.0, -_S, 0.0]),
     np.array([-1.0,  0.0,  0.0]), np.array([-1.0,  _S, 0.0])),
    # Thumb +Z (top), I/M/R on -Y face spread in X
    (np.array([ 0.0,  0.0,  1.0]), np.array([ _S, -1.0, 0.0]),
     np.array([ 0.0, -1.0,  0.0]), np.array([-_S, -1.0, 0.0])),
    # Thumb -Z (bottom), I/M/R on +Y face spread in X
    (np.array([ 0.0,  0.0, -1.0]), np.array([ _S,  1.0, 0.0]),
     np.array([ 0.0,  1.0,  0.0]), np.array([-_S,  1.0, 0.0])),
]


# ─────────────────────────────────────────────────────────────────────────────
# GraspPlanner3D
# ─────────────────────────────────────────────────────────────────────────────

class GraspPlanner3D:
    """
    3D grasp contact-point solver for Kinova Gen3 + LEAP hand.

    Decision variables: q[23], p1[3], p2[3]  (NO gamma / wrench feasibility).

    Parameters
    ----------
    model   : mj.MjModel  (from scene_pick_place_pos.xml)
    data    : mj.MjData
    cfg     : GraspConfig3D  (optional)
    logger  : logging.Logger (optional)
    log_dir : str            (optional — enables per-solve IPOPT log files)
    """

    def __init__(self, model, data,
                 cfg: GraspConfig3D | None = None,
                 logger=None,
                 log_dir: str | None = None):
        self.model   = model
        self.data    = data
        self.cfg     = cfg or GraspConfig3D()
        self.log     = logger or log
        self.log_dir = log_dir

        if log_dir and logger is None:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(
                os.path.join(log_dir, f"grasp3d_py_{ts}.log"))
            fh.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
            self.log.addHandler(fh)

        c = self.cfg
        self._obj_gid    = self._require_geom(c.obj_geom)
        self._thumb_gid  = self._require_geom(c.thumb_geom)
        self._index_gid  = self._require_geom(c.index_geom)
        self._middle_gid = self._require_geom(c.middle_geom)
        self._ring_gid   = self._require_geom(c.ring_geom)
        self._cp1_gid    = self._require_geom(c.cp1_geom)
        self._cp2_gid    = self._require_geom(c.cp2_geom)
        self._cp3_gid    = self._require_geom(c.cp3_geom)
        self._cp4_gid    = self._require_geom(c.cp4_geom)

        for bname in (c.cp1_body, c.cp2_body, c.cp3_body, c.cp4_body):
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bname)
            if bid == -1:
                raise ValueError(f"GraspPlanner3D: body '{bname}' not found.")
        cp1_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, c.cp1_body)
        cp2_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, c.cp2_body)
        cp3_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, c.cp3_body)
        cp4_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, c.cp4_body)
        self._cp1_mocap = model.body_mocapid[cp1_bid]
        self._cp2_mocap = model.body_mocapid[cp2_bid]
        self._cp3_mocap = model.body_mocapid[cp3_bid]
        self._cp4_mocap = model.body_mocapid[cp4_bid]

        self._act_idx = _get_actuated_indices(model)
        gs = model.geom_size[self._obj_gid]
        self._obj_hx = float(gs[0])
        self._obj_hy = float(gs[1])
        self._obj_hz = float(gs[2])

    # ── public API ────────────────────────────────────────────────────────────

    def solve(self,
              q_ref:    np.ndarray,
              obj_pos:  np.ndarray,
              p1_init:  np.ndarray | None = None,
              p2_init:  np.ndarray | None = None,
              p3_init:  np.ndarray | None = None,
              p4_init:  np.ndarray | None = None) -> dict:
        """
        Run 3D grasp optimisation (synchronous / blocking).

        Jointly optimises (q[23], p1[3], p2[3], p3[3], p4[3]) — all 4 fingers.

        Parameters
        ----------
        q_ref   : (nu,) warm-start joint angles (all 23 actuated DOF).
        obj_pos : (3,)  object center in world frame.
        p1_init : (3,)  optional warm-start for thumb contact point.
        p2_init : (3,)  optional warm-start for index contact point.
        p3_init : (3,)  optional warm-start for middle finger contact point.
        p4_init : (3,)  optional warm-start for ring finger contact point.

        Returns
        -------
        dict — keys: success, q, p1, p2, p3, p4, cost, iterations, status
        """
        cfg        = self.cfg
        model      = self.model
        act_idx    = self._act_idx
        obj_center = np.asarray(obj_pos, dtype=float)
        hx, hy, hz = self._obj_hx, self._obj_hy, self._obj_hz

        data_cb = mj.MjData(model)
        data_cb.qpos[:] = self.data.qpos[:]
        data_cb.qvel[:] = self.data.qvel[:]
        mj.mj_forward(model, data_cb)

        uid = id(data_cb)

        # ── callbacks ────────────────────────────────────────────────────────
        # IK callbacks: (q[23], p[3]) → mj_geomDistance — MuJoCo FK required
        ik_thumb_cb  = _FingerPointDistCallback3D(
            f'gp3_ik_th_{uid}', model, data_cb,
            self._thumb_gid,  self._cp1_gid, self._cp1_mocap, act_idx)
        ik_index_cb  = _FingerPointDistCallback3D(
            f'gp3_ik_if_{uid}', model, data_cb,
            self._index_gid,  self._cp2_gid, self._cp2_mocap, act_idx)
        ik_middle_cb = _FingerPointDistCallback3D(
            f'gp3_ik_mf_{uid}', model, data_cb,
            self._middle_gid, self._cp3_gid, self._cp3_mocap, act_idx)
        ik_ring_cb   = _FingerPointDistCallback3D(
            f'gp3_ik_rf_{uid}', model, data_cb,
            self._ring_gid,   self._cp4_gid, self._cp4_mocap, act_idx)
        # SDF callback: (p[3],) → box SDF — analytical, no MuJoCo, FD on 3 vars
        sdf_cb = _AnalyticalSDFCallback3D(
            f'gp3_sdf_{uid}', model, data_cb, self._obj_gid)
        # Non-penetration callbacks: only instantiated when explicitly requested
        if cfg.penetration_constraint:
            nonpen_thumb_cb  = _NonPenetrationCallback3D(
                f'gp3_np_th_{uid}', model, data_cb,
                self._thumb_gid,  self._obj_gid, act_idx)
            nonpen_index_cb  = _NonPenetrationCallback3D(
                f'gp3_np_if_{uid}', model, data_cb,
                self._index_gid,  self._obj_gid, act_idx)
            nonpen_middle_cb = _NonPenetrationCallback3D(
                f'gp3_np_mf_{uid}', model, data_cb,
                self._middle_gid, self._obj_gid, act_idx)
            nonpen_ring_cb   = _NonPenetrationCallback3D(
                f'gp3_np_rf_{uid}', model, data_cb,
                self._ring_gid,   self._obj_gid, act_idx)

        # ── optimisation problem ─────────────────────────────────────────────
        opti = ca.Opti()
        q    = opti.variable(model.nu)   # (23,) arm + LEAP joints
        p1   = opti.variable(3)          # (3,) thumb contact point
        p2   = opti.variable(3)          # (3,) index contact point
        p3   = opti.variable(3)          # (3,) middle finger contact point
        p4   = opti.variable(3)          # (3,) ring finger contact point

        d_thumb  = ik_thumb_cb(q,  p1)
        d_index  = ik_index_cb(q,  p2)
        d_middle = ik_middle_cb(q, p3)
        d_ring   = ik_ring_cb(q,   p4)
        sdf1 = sdf_cb(p1)
        sdf2 = sdf_cb(p2)
        sdf3 = sdf_cb(p3)
        sdf4 = sdf_cb(p4)
        if cfg.penetration_constraint:
            gap_thumb  = nonpen_thumb_cb(q)   # noqa: F821
            gap_index  = nonpen_index_cb(q)   # noqa: F821
            gap_middle = nonpen_middle_cb(q)  # noqa: F821
            gap_ring   = nonpen_ring_cb(q)    # noqa: F821

        cost = cfg.w_ik * (d_thumb**2 + d_index**2 + d_middle**2 + d_ring**2)
        if not cfg.on_object:
            cost += cfg.w_surface * ((sdf1 / cfg.sdf_scale)**2 +
                                     (sdf2 / cfg.sdf_scale)**2 +
                                     (sdf3 / cfg.sdf_scale)**2 +
                                     (sdf4 / cfg.sdf_scale)**2)
        cost += cfg.w_reg * ca.sumsqr((q - q_ref) / cfg.q_scale)
        opti.minimize(cost)

        # Joint limits
        if cfg.joint_limits:
            for i in range(model.nu):
                jid = model.actuator_trnid[i, 0]
                if model.jnt_limited[jid]:
                    opti.subject_to(opti.bounded(
                        model.jnt_range[jid, 0], q[i], model.jnt_range[jid, 1]))

        # Contact points on object surface
        _r_cp = 1e-4
        if cfg.on_object:
            opti.subject_to(sdf1 == -2 * _r_cp)
            opti.subject_to(sdf2 == -2 * _r_cp)
            opti.subject_to(sdf3 == -2 * _r_cp)
            opti.subject_to(sdf4 == -2 * _r_cp)

        # IK: each fingertip touches its contact point
        if cfg.ik_constraint:
            opti.subject_to(d_thumb  == 0)
            opti.subject_to(d_index  == 0)
            opti.subject_to(d_middle == 0)
            opti.subject_to(d_ring   == 0)

        # Non-penetration: disabled by default (redundant with IK + SDF)
        if cfg.penetration_constraint:
            opti.subject_to(gap_thumb  == -1e-4)
            opti.subject_to(gap_index  == -1e-4)
            opti.subject_to(gap_middle == -1e-4)
            opti.subject_to(gap_ring   == -1e-4)

        # Contact point bounding box (3D) — prevents contacts flying to infinity
        margin = max(hx, hy, hz)
        for p in (p1, p2, p3, p4):
            for i, h in enumerate([hx, hy, hz]):
                opti.subject_to(opti.bounded(
                    obj_center[i] - h - margin, p[i],
                    obj_center[i] + h + margin))

        # ── warm-start ───────────────────────────────────────────────────────
        def _seed(init, default):
            return np.asarray(init, float) if init is not None else default

        p1_seed = _seed(p1_init, obj_center + np.array([ 0.0, -hy,  0.0]))
        p2_seed = _seed(p2_init, obj_center + np.array([ hx*_S,  hy, 0.0]))
        p3_seed = _seed(p3_init, obj_center + np.array([ 0.0,    hy, 0.0]))
        p4_seed = _seed(p4_init, obj_center + np.array([-hx*_S,  hy, 0.0]))

        opti.set_initial(q,  q_ref)
        opti.set_initial(p1, p1_seed)
        opti.set_initial(p2, p2_seed)
        opti.set_initial(p3, p3_seed)
        opti.set_initial(p4, p4_seed)

        # ── IPOPT options ────────────────────────────────────────────────────
        ipopt_opts: dict = {
            'jacobian_approximation': 'finite-difference-values',
            'hessian_approximation':  'limited-memory',
            'max_iter':               cfg.max_iter,
            'sb':                     'no',
            'tol':                    1e-6,
            'dual_inf_tol':           1.0,
            'constr_viol_tol':        1e-8,
            'print_level':            0,
            # Acceptable convergence — FD/L-BFGS can't tighten dual_inf below ~6e-2,
            # so primary tol (1e-6) is unachievable. Loosen acceptable criteria to
            # physically meaningful robotics tolerances (1mm constraint satisfaction).
            'acceptable_tol':             0.1,    # dual_inf ~6e-2 < 0.1 -> NLP condition met
            'acceptable_iter':            5,      # trigger after 5 consecutive "good" iters
            'acceptable_constr_viol_tol': 1e-3,  # 1mm; inf_pr ~1.67e-4 << 1e-3
        }
        if self.log_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ipopt_opts['output_file']      = os.path.join(
                self.log_dir, f"grasp3d_ipopt_{ts}.log")
            ipopt_opts['file_print_level'] = 5

        opti.solver('ipopt', {'ipopt': ipopt_opts, 'print_time': False})

        # ── solve ────────────────────────────────────────────────────────────
        def _profile_log():
            cbs = [
                ('ik_thumb', ik_thumb_cb), ('ik_index', ik_index_cb),
                ('sdf',      sdf_cb),
            ]
            if cfg.penetration_constraint:
                cbs += [('np_thumb', nonpen_thumb_cb), ('np_index', nonpen_index_cb)]  # noqa: F821
            parts = [
                f"{nm}: {cb._n_calls}calls {cb._t_total:.1f}s"
                f" ({1e3*cb._t_total/max(cb._n_calls,1):.1f}ms/call)"
                for nm, cb in cbs
            ]
            self.log.info("[profile3d] " + " | ".join(parts))

        try:
            sol = opti.solve()
            _profile_log()
            return {
                'success':    True,
                'q':          sol.value(q),
                'p1':         sol.value(p1),
                'p2':         sol.value(p2),
                'p3':         sol.value(p3),
                'p4':         sol.value(p4),
                'cost':       float(sol.value(opti.f)),
                'iterations': sol.stats()['iter_count'],
                'status':     'converged',
            }
        except Exception as e:
            self.log.warning(f"GraspPlanner3D.solve: {e}")
            _profile_log()
            try:
                try:    _cost = float(opti.debug.value(opti.f))
                except: _cost = None
                try:    _iters = opti.stats().get('iter_count')
                except: _iters = None
                return {
                    'success':    False,
                    'q':          opti.debug.value(q),
                    'p1':         opti.debug.value(p1),
                    'p2':         opti.debug.value(p2),
                    'p3':         opti.debug.value(p3),
                    'p4':         opti.debug.value(p4),
                    'cost':       _cost,
                    'iterations': _iters,
                    'status':     'best-effort',
                }
            except Exception as e2:
                self.log.error(f"GraspPlanner3D debug extraction: {e2}")
                return {'success': False, 'q': None,
                        'p1': None, 'p2': None, 'p3': None, 'p4': None,
                        'cost': None, 'iterations': None, 'status': 'failed'}

    def verify(self, result: dict) -> dict:
        """Post-solve sanity check: IK residuals, gap, SDF values."""
        if result.get('q') is None:
            return {}
        model  = self.model
        data_v = mj.MjData(model)
        data_v.qpos[:] = self.data.qpos[:]
        for idx, val in zip(self._act_idx, result['q']):
            data_v.qpos[idx] = val
        data_v.mocap_pos[self._cp1_mocap] = result['p1']
        data_v.mocap_pos[self._cp2_mocap] = result['p2']
        data_v.mocap_pos[self._cp3_mocap] = result['p3']
        data_v.mocap_pos[self._cp4_mocap] = result['p4']
        mj.mj_forward(model, data_v)

        ik_t  = mj.mj_geomDistance(model, data_v,
                                    self._thumb_gid,  self._cp1_gid, 0.5, None)
        ik_i  = mj.mj_geomDistance(model, data_v,
                                    self._index_gid,  self._cp2_gid, 0.5, None)
        ik_m  = mj.mj_geomDistance(model, data_v,
                                    self._middle_gid, self._cp3_gid, 0.5, None)
        ik_r  = mj.mj_geomDistance(model, data_v,
                                    self._ring_gid,   self._cp4_gid, 0.5, None)
        gap_t = mj.mj_geomDistance(model, data_v,
                                    self._thumb_gid,  self._obj_gid, 0.5, None)
        gap_i = mj.mj_geomDistance(model, data_v,
                                    self._index_gid,  self._obj_gid, 0.5, None)
        gap_m = mj.mj_geomDistance(model, data_v,
                                    self._middle_gid, self._obj_gid, 0.5, None)
        gap_r = mj.mj_geomDistance(model, data_v,
                                    self._ring_gid,   self._obj_gid, 0.5, None)

        obj_pos = data_v.geom_xpos[self._obj_gid].copy()
        obj_mat = data_v.geom_xmat[self._obj_gid].reshape(3, 3)
        def _sdf3(p):
            p_local = obj_mat.T @ (np.asarray(p, float) - obj_pos)
            return _box_sdf_3d(p_local, np.zeros(3),
                               self._obj_hx, self._obj_hy, self._obj_hz)

        s1 = _sdf3(result['p1'])
        s2 = _sdf3(result['p2'])
        s3 = _sdf3(result['p3'])
        s4 = _sdf3(result['p4'])

        info = {
            'ik_thumb_mm':   ik_t  * 1000,
            'ik_index_mm':   ik_i  * 1000,
            'ik_middle_mm':  ik_m  * 1000,
            'ik_ring_mm':    ik_r  * 1000,
            'gap_thumb_mm':  gap_t * 1000,
            'gap_index_mm':  gap_i * 1000,
            'gap_middle_mm': gap_m * 1000,
            'gap_ring_mm':   gap_r * 1000,
            'sdf_p1_mm':     s1    * 1000,
            'sdf_p2_mm':     s2    * 1000,
            'sdf_p3_mm':     s3    * 1000,
            'sdf_p4_mm':     s4    * 1000,
        }
        self.log.info(
            f"[verify3d] IK=({ik_t*1e3:.2f},{ik_i*1e3:.2f},"
            f"{ik_m*1e3:.2f},{ik_r*1e3:.2f})mm "
            f"GAP=({gap_t*1e3:+.2f},{gap_i*1e3:+.2f},"
            f"{gap_m*1e3:+.2f},{gap_r*1e3:+.2f})mm")
        return info

    # ── private ───────────────────────────────────────────────────────────────

    def _require_geom(self, name: str) -> int:
        gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
        if gid == -1:
            raise ValueError(
                f"GraspPlanner3D: geom '{name}' not found. "
                f"Check GraspConfig3D geometry name fields.")
        return gid


# ─────────────────────────────────────────────────────────────────────────────
# Multi-start wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MultiStartGraspPlanner3D:
    """
    Runs GraspPlanner3D from each of 6 canonical face-pair seeds and returns
    the best feasible result ranked by: converged > best-effort, then cost.

    Parameters
    ----------
    model, data, cfg, logger, log_dir : forwarded to GraspPlanner3D
    """

    def __init__(self, model, data,
                 cfg: GraspConfig3D | None = None,
                 logger=None, log_dir: str | None = None):
        self._planner  = GraspPlanner3D(model, data, cfg, logger, log_dir)
        self._obj_hx   = self._planner._obj_hx
        self._obj_hy   = self._planner._obj_hy
        self._obj_hz   = self._planner._obj_hz

    def solve(self, q_ref: np.ndarray, obj_pos: np.ndarray,
              max_seeds: int | None = None) -> dict:
        """Try face-pair seeds (up to max_seeds), return best result."""
        c  = np.asarray(obj_pos, float)
        hx = self._obj_hx
        hy = self._obj_hy
        hz = self._obj_hz

        seeds = _FACE_SEEDS_4F_UNIT[:max_seeds] if max_seeds else _FACE_SEEDS_4F_UNIT
        results = []
        for d_th, d_if, d_mf, d_rf in seeds:
            scale = np.array([hx, hy, hz])
            p1_init = c + d_th * scale
            p2_init = c + d_if * scale
            p3_init = c + d_mf * scale
            p4_init = c + d_rf * scale
            r = self._planner.solve(q_ref, obj_pos,
                                    p1_init=p1_init, p2_init=p2_init,
                                    p3_init=p3_init, p4_init=p4_init)
            results.append(r)

        def _rank(r):
            s = {'converged': 0, 'best-effort': 1, 'failed': 2}[r['status']]
            c = r['cost'] if r['cost'] is not None else 1e9
            return (s, c)

        results.sort(key=_rank)
        best = results[0]
        best['all_results'] = results
        return best

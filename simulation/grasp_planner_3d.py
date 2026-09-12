"""
grasp_planner_3d.py
===================
3D grasp contact-point solver — IPOPT formulation (Kinova Gen3 + LEAP hand).

Seeding (MultiStartGraspPlanner3D)
-----------------------------------
    Both seed sources ray from the SAME origin, _ray_origin_local (the mesh's
    volumetric centroid) -- see that function for why the geom frame origin is
    the wrong point on a YCB scan (it sits at the object's base).

    _seed_pair generates one candidate per call:
      1. Sphere-trace a random direction FROM THAT ORIGIN to surface point p1s.
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
    from grasp_control import object_sdf as _object_sdf
    _CIK_AVAILABLE = True
except ImportError:
    _CIK_AVAILABLE = False
    _object_sdf = None

try:
    from grasp_control import object_uv_atlas as _object_uv_atlas
    _UV_ATLAS_AVAILABLE = _object_uv_atlas._XATLAS_AVAILABLE
except ImportError:
    _object_uv_atlas = None
    _UV_ATLAS_AVAILABLE = False

_GEOM_TYPE_MESH = 7   # mj.mjtGeom.mjGEOM_MESH

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
    'print_time':            False,
    'qpsol':                 'osqp',
    # error_on_fail=False: with O(500) inequality constraints on 23 DOFs the
    # linearised QP subproblem is often primal-infeasible at the initial
    # (DLS warm-start) point; OSQP finds the minimum-constraint-violation
    # direction and lets SQP continue, instead of aborting right there.
    # polish=True: more accurate QP subproblem solutions.
    # (This block was previously out of sync with constrained_ik._SQP_SOLVER_OPTS
    # despite the "copied verbatim" comment above — error_on_fail=True/polish=False/
    # max_iter=500/missing tol_du measurably degraded convergence, see
    # RECOMMENDER_CONVERGENCE_FINDINGS.md-adjacent staged-convergence testing.)
    'qpsol_options':         {'error_on_fail': False,
                              'osqp': {'verbose': False, 'polish': True}},
    'max_iter':              800,
    # tol_du is a KKT-residual (dual) tolerance that scales with the objective
    # weight — see constrained_ik._SQP_SOLVER_OPTS's tuning note (co-tuned with
    # tip_weight=100 there). Kept identical here for the same reason.
    'tol_du':                1e-2,
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


def _symbolic_contact_frame_ca(n_in_sym, smooth_blend: bool = False):
    """
    Build right-handed (n_in, t1, t2) contact frame as a CasADi MX expression.

    Mirrors _build_contact_frame_3d but operates on symbolic MX vectors so the
    frame is part of the NLP's symbolic graph.  CasADi can autodiff through it.

    Parameters
    ----------
    n_in_sym : (3,) CasADi MX — inward contact normal (need not be unit-length;
                normalisation is applied internally).
    smooth_blend : bool — False (default, UNCHANGED): the blend weight is
                ca.fabs(n[0]) - 0.9. ca.fabs is C0 but not C1 at n[0]==0 (its
                derivative jumps from -1 to +1), so the exact Hessian of this
                expression is undefined there and enormous/discontinuous in a
                neighborhood of it -- exactly the kind of term
                hessian_approximation='exact' cannot integrate (see the IPOTP
                dual-indeterminacy investigation: this was the concrete,
                findable non-twice-differentiable candidate, alongside the
                grad_norm-style ca.norm_2 calls elsewhere in this file, which
                are smooth away from the origin and only singular exactly AT
                a zero vector -- a measure-zero, physically unreachable point
                for a nonzero normal/tangent, unlike this fabs kink which sits
                on n[0]==0, a plane the normal can and does cross).
                True: replace ca.fabs(n[0]) with the smooth surrogate
                sqrt(n[0]**2 + eps_abs) (eps_abs ~1e-6) -- C-infinity
                everywhere, agrees with |n[0]| to O(eps_abs) away from 0, and
                still feeds the same tanh blend. Opt-in and off by default so
                no existing call site's numerics change; use this only to
                test whether removing the kink lets hessian_approximation=
                'exact' actually run (nonzero Lagrangian Hessian evals) and/or
                changes the dual-multiplier non-uniqueness reported at the
                antipodal pinch.
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
    if smooth_blend:
        _abs_n0 = ca.sqrt(n[0]**2 + 1e-6)   # C-infinity surrogate for ca.fabs(n[0])
    else:
        _abs_n0 = ca.fabs(n[0])
    w = _abs_n0 - 0.9
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


# ── Mesh (YCB) object SDF support ───────────────────────────────────────────
# geom_type == 7 (mjGEOM_MESH) objects have no analytic form, so their SDF is
# a precomputed table (grasp_control.object_sdf) queried through a small set
# of module-level CasADi Functions built once per object shape and cached
# here by body id. conservative=False everywhere below: these are used to
# PLACE contacts, not to keep the arm clear of the object (see object_sdf's
# casadi_fn docstring — the conservative offset biases a projected contact
# ~1mm outside the true surface).
# Keyed by (id(model), body_id), NOT body_id alone: body_id is just an integer
# index into ONE model's body array, and a freshly-built MjModel (e.g. a new
# object attached via ycb_grasp.scene.build) reliably reuses the same small
# integer indices as any other freshly-built model with the same scene
# structure. Keying by body_id alone silently served a DIFFERENT object's
# cached SDF table across separate MjModel instances that happened to attach
# their object at the same body index — measured: two different YCB objects
# in the same process solved to bit-identical (wrong) results. id(model) is a
# stable Python identity, unique per live MjModel instance, so this cannot
# collide across scenes. object_sdf.load_or_bake's own on-DISK cache is
# already correctly content-hash-keyed (grasp_control/object_sdf.py's
# _cache_key, over the hull vertices) — this in-process layer only wraps that
# in CasADi Functions, so re-keying here costs a re-wrap, not a re-bake, when
# the same object recurs across separate model instances.
_MESH_SDF_CACHE: dict[tuple[int, int], dict] = {}


def _mesh_sdf_entry(model, body_id: int) -> dict:
    """Lazily bake/load and cache the CasADi SDF machinery for one object body."""
    cache_key = (id(model), body_id)
    entry = _MESH_SDF_CACHE.get(cache_key)
    if entry is not None:
        return entry
    if _object_sdf is None:
        raise RuntimeError("object_sdf module unavailable — cannot handle mesh geoms")
    table, hulls = _object_sdf.load_or_bake(model, body_id)
    # Unique per (model, body) — see the cache-key comment above; a plain
    # body_id-based name can collide across separate MjModel instances the
    # same way the old cache key did.
    _tag = f"gp3_sdf_{id(model)}_{body_id}"
    fn = _object_sdf.casadi_fn(table, name=_tag, conservative=False)
    # load_or_bake discards the raw hull vertices after baking (only keeps the
    # half-space (A,b) form) — re-extract them the same way
    # object_sdf.body_hull_halfspaces does, for _minor_axis_local's SVD (the
    # table/hulls themselves don't carry a vertex array).
    _hulls_v, _verts = _object_sdf.body_hull_halfspaces(model, body_id)
    _grad_fn = _object_sdf.casadi_grad_fn(fn, name=f"{_tag}_grad")
    # 3x3 Hessian of fn (grad_fn differentiated again), for the local-quadratic
    # mesh contact (_mesh_quadratic_contact_ca) and seed curvature gating
    # (_mesh_surface_kappa_max_np). MUST be cached here, not rebuilt per call
    # (_sdf_hessian_np used to do exactly that): fn is a B-spline over the
    # full SDF lattice, and ca.jacobian of its gradient graph is expensive to
    # CONSTRUCT (not just evaluate) — rebuilding it ~O(seed candidates x
    # Picard stages) times per solve (up to a few hundred) turned a
    # sub-second op into a multi-minute stall. Built once per (model, body).
    _hess_x = ca.MX.sym("x", 3)
    _hessian_fn = ca.Function(f"{_tag}_hess", [_hess_x],
                              [ca.jacobian(_grad_fn(_hess_x), _hess_x)])
    # Dense VISUAL mesh vertices, for the local surface fit
    # (_mesh_local_surface_fit_np). The collision hull above is a convex
    # decomposition -- 466 verts for 036_wood_block, i.e. a handful of large
    # triangles per box face, far too coarse for a local least-squares fit
    # (measured: 0-3 verts within 50mm of a mid-face seed). The visual mesh is
    # the actual scan: 8194 verts for the same object. body_visual_mesh folds
    # geom_pos/geom_quat the same way body_hull_halfspaces does, so both live
    # in the same body frame as the SDF table.
    _vis_verts = None
    _vis_normals = None
    _vol_centroid = None
    try:
        from grasp_control import object_uv_atlas as _oua
        _vv, _vf = _oua.body_visual_mesh(model, body_id)
        _vis_verts = np.asarray(_vv, float)
        # TRUE volumetric centroid of the closed visual mesh, by signed-tetra
        # decomposition about the origin. This is the shared SEED RAY ORIGIN
        # (see _ray_origin_local) -- deliberately NOT a vertex mean, which is a
        # vertex-DENSITY average and lands wherever the scan/decomposition
        # happened to put more vertices: measured 17.6mm off the true centroid
        # on 036_wood_block's hull (466 verts unevenly split across faces).
        # Cross-checked against MuJoCo's own body_ipos (which it computes from
        # the same geometry): agrees to 0.1mm on the block, 0.0mm on
        # 017_orange, 0.6mm on 065-a_cups.
        _vfi = np.asarray(_vf, int)
        _a, _b, _cc = (_vis_verts[_vfi[:, 0]], _vis_verts[_vfi[:, 1]],
                       _vis_verts[_vfi[:, 2]])
        _tv = np.einsum("ij,ij->i", _a, np.cross(_b, _cc)) / 6.0
        _tot = float(_tv.sum())
        if abs(_tot) > 1e-12:
            _vol_centroid = (((_a + _b + _cc) / 4.0) * _tv[:, None]).sum(0) / _tot
        # Area-weighted per-vertex normals, accumulated from the faces. Used by
        # the local surface fit to reject vertices facing the other way -- the
        # inner surface of a thin shell (a cup wall is ~2-3mm) otherwise lands
        # inside the fit's distance band and corrupts the quadratic.
        _vf = np.asarray(_vf, int)
        _fn = np.cross(_vis_verts[_vf[:, 1]] - _vis_verts[_vf[:, 0]],
                       _vis_verts[_vf[:, 2]] - _vis_verts[_vf[:, 0]])
        _vis_normals = np.zeros_like(_vis_verts)
        for _k in range(3):
            np.add.at(_vis_normals, _vf[:, _k], _fn)
        _nn = np.linalg.norm(_vis_normals, axis=1, keepdims=True)
        _vis_normals = _vis_normals / np.maximum(_nn, 1e-12)
    except Exception:
        pass          # visual mesh is optional; the fit falls back to the SDF
    entry = dict(
        table=table,
        verts=_verts,
        visual_verts=_vis_verts,
        visual_normals=_vis_normals,
        vol_centroid=_vol_centroid,
        fn=fn,
        grad_fn=_grad_fn,
        normal_fn=_object_sdf.casadi_normal_fn(fn, name=f"{_tag}_normal"),
        # Full-length projector (k=5, object_sdf's own default) for seeding —
        # _seed_pair/_march_sdf_np start from points a bbox-radius or more
        # away from the surface, same regime object_sdf.surface_project was
        # designed for.
        project_fn=_object_sdf.surface_project(fn, k=5, name=f"{_tag}_proj"),
        # Short projector (k=2) for the in-NLP tangent-plane reprojection: that
        # seed already starts near the surface (it's a small tangent offset
        # from an on-surface anchor), so a short unroll suffices — see
        # object_sdf.surface_project's docstring on why more iters doesn't
        # strictly help anyway (oscillates at creases past k~5).
        project_fn_short=_object_sdf.surface_project(fn, k=2, name=f"{_tag}_proj2"),
        hessian_fn=_hessian_fn,
    )
    _MESH_SDF_CACHE[cache_key] = entry
    return entry


def _mesh_sdf_np(point, entry: dict) -> float:
    """Signed distance to a mesh object (numpy scalar), via the cached CasADi Function."""
    return float(entry["fn"](np.asarray(point, float)))


def _mesh_normal_np(point, entry: dict) -> np.ndarray:
    """Outward unit surface normal at/near a mesh object surface (numpy)."""
    return np.asarray(entry["normal_fn"](np.asarray(point, float))).flatten()


def _mesh_project_np(point, entry: dict) -> np.ndarray:
    """Project a free world-frame point onto the mesh surface (numpy)."""
    return np.asarray(entry["project_fn"](np.asarray(point, float))).flatten()


def _geom_sdf_np(point, geom_type: int, center, mat, size, mesh_entry: dict | None = None) -> float:
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
    elif geom_type == _GEOM_TYPE_MESH and mesh_entry is not None:
        return _mesh_sdf_np(p_l, mesh_entry)   # table is baked in the object BODY frame
    return float(np.linalg.norm(p - c))


def _geom_normal_np(point, geom_type: int, center, mat, size, mesh_entry: dict | None = None) -> np.ndarray:
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
    elif geom_type == _GEOM_TYPE_MESH and mesh_entry is not None:
        # Table queried in object BODY frame (p_l, matching constrained_ik's
        # R_obj.T @ (p_world - c_obj) convention for the same table); the
        # gradient there is a body-frame direction, so rotate back to world.
        n_l = _mesh_normal_np(p_l, mesh_entry)
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


def _contact_friction(model, obj_gid: int, *tip_gids: int) -> tuple[float, float]:
    """(mu, mu_t) at an object-vs-fingertip contact, combined the way MuJoCo's own
    contact solver actually does it: ELEMENTWISE MAX of geom_friction across the two
    geoms in contact (confirmed empirically against mj_forward's d.contact[i].friction
    — MuJoCo does NOT average or take the object's value alone).

    model.geom_friction is [sliding, torsional, rolling] per geom; this returns just
    the two components _embed_wrench_cone_ca/build_W_ca use: sliding (index 0, "mu")
    and torsional (index 1, "mu_t" — the soft-finger spin-friction coefficient, see
    scripts/3D_minimum_NCF_soft.py). Previously mu was read from the OBJECT geom only
    (model.geom_friction[obj_gid][0]), ignoring the fingertip pad's own friction
    entirely — not wrong when the pad happens to be softer/equal, but not the actual
    contact value MuJoCo's physics uses, and mu_t had no source at all (soft-finger
    torsion wasn't modeled). tip_gids: one or more fingertip geoms actually touching
    the object (e.g. thumb_gid, index_gid) — max'd in along with the object's own.
    """
    fric = model.geom_friction[obj_gid].copy()
    for gid in tip_gids:
        fric = np.maximum(fric, model.geom_friction[gid])
    return float(fric[0]), float(fric[1])


def _project_to_surface_np(point, geom_type, center, mat, size, iters=12, mesh_entry=None):
    """Move a world-frame point to the nearest surface point.

    BOX: exact closed form — clamp to box in local frame, snap dominant axis to
    its face.  One step, always exact, works from corners and edges where the
    SDF gradient is diagonal and a single Newton step falls short.

    MESH: object_sdf.surface_project's unit-gradient-step iteration (not
    Newton — Newton divides by ‖grad d‖², which blows up at the creases where
    convex hulls meet; see object_sdf.py's surface_project docstring).

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

    if geom_type == _GEOM_TYPE_MESH and mesh_entry is not None:
        p_l = R.T @ (p - c)
        return c + R @ _mesh_project_np(p_l, mesh_entry)

    for _ in range(iters):
        s = _geom_sdf_np(p, geom_type, c, mat, size)
        if abs(s) < 1e-9:
            break
        p = p - s * _geom_normal_np(p, geom_type, c, mat, size)
    return p


def _ray_scan_span_np(mesh_entry: dict, origin_l) -> float:
    """How far a crossing scan must reach from origin_l to leave the mesh behind.

    The farthest vertex from the scan origin, with margin. Callers must NOT pass
    their own bbox_r here: _fixed_antipodal_seed/_seed_pair's bbox_r is
    max(geom_size)*2.5, and for a MESH geom_size is not the half-extent -- on
    065-a_cups it evaluates to 26.4mm against an object whose far wall already
    sits at 27mm from the volumetric centroid. A scan that short simply misses
    the far wall: measured 0 crossings on a 45-degree ray and 1 on another,
    which silently fell back to nearest-surface projection and landed INSIDE the
    cup, i.e. exactly the bug the crossing scan exists to fix.
    """
    V = mesh_entry.get("verts")
    if V is None or len(V) == 0:
        return 0.25
    return float(np.max(np.linalg.norm(np.asarray(V, float) - np.asarray(origin_l, float),
                                       axis=1))) * 1.5


def _ray_surface_crossings_np(mesh_entry: dict, origin_l, dir_l,
                              span: float | None = None,
                              step: float = 2e-3, n_refine: int = 20):
    """Every point where a ray crosses the mesh SDF's zero level set, as sorted
    signed distances along dir_l from origin_l (all object-LOCAL).

    WHY THIS EXISTS. The seed sources used to pick surface points by projecting
    a far-away ray endpoint to the NEAREST surface, and by sphere-marching until
    the SDF turned positive. Both are correct only for a SOLID object. A cup is
    two thin shells with a void between them, and the void is POSITIVE (measured
    on 065-a_cups: +12.20mm at the volumetric centroid, -2.48mm inside the wall,
    +170mm far outside). So "march until sdf > 0" stops at the INNER wall, and
    "project to nearest surface" from a point near the rim lands on the inner
    wall too -- measured 6 of 14 seed contacts inside the cup.

    Enumerating sign changes instead makes the topology explicit: a solid object
    yields exactly 2 crossings, a cup 4 (outer, inner, inner, outer). The caller
    then picks by POSITION rather than hoping a first-hit rule lands right --
    see _outer_pair_t, which takes the first and last.

    Fixed step, not a sphere-march: the march's adaptive step is |sdf|-scaled,
    and inside a thin wall |sdf| is small but the wall is not, so it can stride
    past one. 065-a_cups' wall is 6.0mm thick with a minimum |SDF| of only
    2.75mm; a 2mm fixed step resolves it with margin. Each bracketed crossing is
    then bisected to n_refine digits, so the step size sets which walls are
    FOUND, not how precisely they are located.
    """
    if span is None:
        span = _ray_scan_span_np(mesh_entry, origin_l)
    ts = np.arange(-span, span + 0.5 * step, step)
    vals = np.array([float(mesh_entry["fn"](origin_l + t * dir_l)) for t in ts])
    out = []
    for i in range(len(ts) - 1):
        if np.sign(vals[i]) == np.sign(vals[i + 1]):
            continue
        lo, hi, s_lo = ts[i], ts[i + 1], np.sign(vals[i])
        for _ in range(n_refine):
            mid = 0.5 * (lo + hi)
            if np.sign(float(mesh_entry["fn"](origin_l + mid * dir_l))) == s_lo:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return np.asarray(out, float)


def _outer_pair_t(mesh_entry: dict, origin_l, dir_l, span: float | None = None):
    """(t_first, t_last) of the OUTER surface crossings along dir_l, or None when
    the ray finds fewer than two (a miss, or a numerically degenerate graze).

    First and last are the outer walls for any ray that fully traverses the
    object, whatever its interior topology -- that is the whole point of picking
    by position instead of by first hit. Returns None rather than guessing so
    the caller can fall back to the projection path unchanged.
    """
    xs = _ray_surface_crossings_np(mesh_entry, origin_l, dir_l, span)
    # An ODD count means the scan started or ended inside material -- a clipped
    # or degenerate ray, not a real traversal. Bail rather than pair a genuine
    # outer wall with a mid-object crossing.
    if len(xs) < 2 or len(xs) % 2 == 1:
        return None
    return float(xs[0]), float(xs[-1])


def _march_sdf_np(p_start, direction, geom_type, center, mat, size,
                  max_steps=80, mesh_entry=None):
    """Sphere-march from p_start along direction until exiting the object, then project to surface."""
    d_unit = np.asarray(direction, float)
    d_unit = d_unit / (np.linalg.norm(d_unit) + 1e-12)
    p = np.asarray(p_start, float)
    for _ in range(max_steps):
        sdf = _geom_sdf_np(p, geom_type, center, mat, size, mesh_entry=mesh_entry)
        if sdf > 1e-5:
            return _project_to_surface_np(p, geom_type, center, mat, size, mesh_entry=mesh_entry)
        step = max(abs(sdf) * 0.5, 1e-3)
        p = p + step * d_unit
    return _project_to_surface_np(p, geom_type, center, mat, size, mesh_entry=mesh_entry)


def _ray_origin_local(geom_type, mesh_entry=None) -> np.ndarray:
    """The point every seed source rays FROM, in the object's LOCAL frame.

    THE SINGLE DEFINITION OF "THE MIDDLE OF THE OBJECT" for seeding. Both seed
    sources (_fixed_antipodal_seed's fixed-axis ray and _seed_pair's random
    ray + antipodal march) call this, so they agree on where the object is.
    They previously did not: _fixed_antipodal_seed rayed through the hull
    vertex mean while _seed_pair rayed from the geom frame ORIGIN, which on a
    YCB scan is wherever the capture rig put it -- for 036_wood_block,
    017_orange and 065-a_cups alike that is the object's BASE (hull z-extent
    starts at ~0), i.e. a point on the table below the object, 112/44/63mm
    from the centroid respectively.

    Returns the mesh's TRUE VOLUMETRIC centroid (signed-tetra, cached on
    mesh_entry by _mesh_sdf_entry), falling back to the hull-vertex mean and
    then to the geom origin when no mesh is available. The volumetric centroid
    rather than a vertex mean because a vertex mean is a vertex-DENSITY
    average: on 036_wood_block's 466-vertex collision hull it sits 17.6mm off
    the true centroid, biased toward whichever faces the convex decomposition
    tessellated more finely. MuJoCo's body_ipos agrees with the volumetric
    value to 0.1mm, which is the independent check that this is the real thing.

    LOCAL frame, and a RAY ORIGIN ONLY. Every SDF / normal / projection call
    still takes the geom frame's own centre, since that is the frame those
    functions are defined in -- see _fixed_antipodal_seed's own note. Callers
    convert with: c_ray_world = center + obj_mat @ _ray_origin_local(...).

    Analytic primitives return the origin: their geom frame is already centred.
    """
    if geom_type != _GEOM_TYPE_MESH or mesh_entry is None:
        return np.zeros(3)
    vc = mesh_entry.get("vol_centroid")
    if vc is not None:
        return np.asarray(vc, float)
    V = mesh_entry.get("verts")
    if V is not None and len(V) >= 3:
        return np.asarray(V, float).mean(0)
    return np.zeros(3)


def _seed_pair(geom_type, size, center, obj_mat, bbox_r, rng,
               delta_max=np.deg2rad(45), mesh_entry=None,
               prefer_outer: bool = True):
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
    # Ray from the SHARED seed origin (_ray_origin_local: the mesh's volumetric
    # centroid), not from the geom frame origin. On every YCB object measured
    # the geom origin sits at the object's BASE -- a point on the table below
    # it -- so a random direction cast from there both starts outside the
    # object and biases every footprint upward. `c` itself stays the SDF frame
    # centre for all projection/normal calls below.
    c_ray = c + obj_mat @ _ray_origin_local(geom_type, mesh_entry)
    u = rng.standard_normal(3)
    u[2] *= 0.5                              # bias toward side faces, away from top/bottom
    u /= np.linalg.norm(u) + 1e-12
    # OUTER-surface selection for the first contact -- the ray's crossing
    # NEAREST its far endpoint, i.e. the outer wall on the side the ray points
    # toward, rather than whatever surface happens to be nearest that endpoint.
    _o_l = (_ray_origin_local(geom_type, mesh_entry)
            if (geom_type == _GEOM_TYPE_MESH and mesh_entry is not None) else None)
    _outer = None
    if prefer_outer and _o_l is not None:
        _u_l = np.asarray(obj_mat, float).T @ u
        _outer = _outer_pair_t(mesh_entry, _o_l, _u_l)
    if _outer is not None:
        _u_l = np.asarray(obj_mat, float).T @ u
        p1s = _project_to_surface_np(c + obj_mat @ (_o_l + _outer[1] * _u_l),
                                     geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
    else:
        p1s = _project_to_surface_np(c_ray + u * bbox_r, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
    n1_in = -_geom_normal_np(p1s, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)

    # Rotate march direction about world z only — prevents downward tilt that
    # exits through the bottom face and hits _reachable_contact rejection.
    ang  = rng.uniform(-delta_max, delta_max)
    ca_, sa_ = np.cos(ang), np.sin(ang)
    Rz = np.array([[ca_, -sa_, 0.0], [sa_, ca_, 0.0], [0.0, 0.0, 1.0]])
    d  = Rz @ n1_in
    delta = abs(ang)

    # Second contact: the FURTHEST crossing along the march direction, not the
    # first. _march_sdf_np stops the moment the SDF turns positive, which inside
    # a cup is the cavity -- so it returns the INNER wall. Scanning the whole ray
    # and taking the last crossing returns the far OUTER wall instead.
    _outer2 = None
    if prefer_outer and _o_l is not None:
        _d_l = np.asarray(obj_mat, float).T @ d
        _p1_l = np.asarray(obj_mat, float).T @ (p1s - c)
        _outer2 = _outer_pair_t(mesh_entry, _p1_l, _d_l)
    if _outer2 is not None:
        _d_l = np.asarray(obj_mat, float).T @ d
        _p1_l = np.asarray(obj_mat, float).T @ (p1s - c)
        p2s = _project_to_surface_np(c + obj_mat @ (_p1_l + _outer2[1] * _d_l),
                                     geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
    else:
        p2s = _march_sdf_np(p1s + 1e-3 * d, d, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
    n2_in = -_geom_normal_np(p2s, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)

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


def _surface_walk_np(p_start, d_tan, dist, geom_type, size, center, obj_mat,
                     mesh_entry=None, n_step=6):
    """Walk `dist` along the surface from p_start in tangent direction d_tan.

    A straight step of 45mm from a contact leaves a 36mm-radius sphere entirely,
    so stepping then projecting lands wherever the projection happens to point.
    Stepping in n_step increments and re-projecting each time keeps the walk ON
    the surface and re-derives the tangent as the normal turns -- a cheap
    geodesic march, which is what "45mm away along this face" has to mean on a
    curved object.
    """
    p = np.asarray(p_start, float).copy()
    d = np.asarray(d_tan, float).copy()
    step = float(dist) / max(int(n_step), 1)
    for _ in range(max(int(n_step), 1)):
        p_next = p + step * d
        p_next = _project_to_surface_np(p_next, geom_type, center, obj_mat, size,
                                        mesh_entry=mesh_entry)
        if not np.all(np.isfinite(p_next)):
            return None
        n_out = _geom_normal_np(p_next, geom_type, center, obj_mat, size,
                                mesh_entry=mesh_entry)
        if not np.all(np.isfinite(n_out)) or np.linalg.norm(n_out) < 1e-9:
            return None
        n_out = n_out / np.linalg.norm(n_out)
        d = d - np.dot(d, n_out) * n_out          # re-tangent against the new normal
        nd = np.linalg.norm(d)
        if nd < 1e-9:
            return None
        d = d / nd
        p = p_next
    return p


# Measured on the model's own rest pose: the middle fingertip sits at
# [-0.1, -45.4, 0.0] mm from the index fingertip in the PALM frame -- pure -y,
# and |mf-rf| is the same 45.4mm, i.e. the fingers are evenly pitched along the
# palm's -y axis. Handedness is already carried by the palm frame, so nothing
# needs to infer "right hand" from the thumb/index relationship.
_MF_FROM_IF_PALM = np.array([0.0, -1.0, 0.0])
_MF_PITCH_M = 0.0454


def _patch_point_np(frame, t0, t1, center_np, mat_np):
    """World point at patch coordinate (t0,t1), via the frame dict's own
    reconstruction identity -- the SAME one _run_stage uses symbolically:
        p_local(t) = seed_l + t0*axis0_l + t1*axis1_l + h(t)*n_l
        h(t)       = -(kappa0*t0^2 + kappa1*t1^2) / (2*grad_norm)
    """
    h = -(float(frame['kappa0']) * t0**2 + float(frame['kappa1']) * t1**2) \
        / (2.0 * float(frame['grad_norm']))
    p_l = (np.asarray(frame['seed_l'], float)
           + t0 * np.asarray(frame['axis0_l'], float)
           + t1 * np.asarray(frame['axis1_l'], float)
           + h * np.asarray(frame['n_l'], float))
    return np.asarray(center_np, float) + np.asarray(mat_np, float) @ p_l


def _seed_third_contact(seed, geom_type, size, center, obj_mat, rng,
                        mesh_entry=None, mf_dir_world=None, n_fan=5,
                        fan_half_deg=40.0, strategy='tangent',
                        palm_R=None, pitch_m=_MF_PITCH_M, fk_probe=None,
                        prefer_outer=True, patch_frame=None,
                        patch_offset_m=0.030):
    """Candidate THIRD contacts for a tripod grasp (cfg.n_contacts >= 3).

    strategy:
      'fan'       -- ORIGINAL. Fan perpendicular to the grasp axis at object
                     scale (reach = max(size)*1.5 from the pinch midpoint), then
                     nearest-surface projection. Kept as the ablation baseline.
                     Two known defects: it places candidates 47-59mm from the
                     index contact (measured on 017_orange) with nothing tying it
                     to the index's neighbourhood, and it uses projection rather
                     than the ray-crossing placement the PAIR seeder was migrated
                     to (_outer_pair_t), so on a non-convex object it can land on
                     an inner wall -- the exact failure the crossing scan exists
                     to prevent.
      'tangent'   -- (B) MEASURED NON-VIABLE, kept only to document the result.
                     Steps the rigid palm-frame finger base-pitch (45.4mm along
                     palm -y, verified identical at qpos0 and TS.home_qpos()) along
                     the surface from the index contact. The CONSTANT is real; the
                     failure is that a palm-frame direction maps to an arbitrary
                     WORLD direction -- at TS.home_qpos() the wrist is oriented so
                     R_palm @ [0,-1,0] = [-0.001, 0.023, -1.0], i.e. straight DOWN.
                     Every candidate landed ~21mm below the table-clearance gate:
                     '5 candidate(s) -> no viable candidate' on every seed of every
                     object.
      'kinematic' -- (D) MEASURED NON-VIABLE, for a deeper reason worth recording.
                     Asks where the middle fingertip IS once the hand is posed for
                     the pinch. Its premise does not hold: there is no pose in this
                     pipeline where that finger is near the object.
                       - at q_ref / rest:            |mf - if| ~ 176-185mm
                       - after one DLS call:         thumb residual 89mm, index 64mm
                                                     (the pinch itself has not converged)
                       - at a SOLVED, LIFTING n=2 grasp:
                             SDF(middle tip) = +97.7 / +146.4 / +53.3 mm
                             on orange / lemon / wood_block
                     The n=2 NLP never asks the middle finger to curl, so it stays
                     extended into free space. A kinematic seeder therefore reads a
                     position that does not exist, and no amount of fixing the IK
                     call changes that.

    CONSEQUENCE: the third contact cannot be seeded from kinematics at all. It has
    to be chosen GEOMETRICALLY on the object -- off the grasp axis, above the
    table, curvature-acceptable -- and the NLP then curls the finger to it. That is
    what 'fan' does, and it is why 'fan' is the only strategy that reliably yields
    candidates.

    fk_probe: callable(seed) -> world position of the middle fingertip at the
        pinch pose, or None. Required by 'kinematic'; ignored otherwise.

    Returns a list of dicts, each the seed extended with 'p3'/'p3s'/'n3_in',
    ready for the SAME gates the pair passes through. Empty list if nothing
    lands on the surface.
    """
    p1s = np.asarray(seed['p1s'], float)
    p2s = np.asarray(seed['p2s'], float)
    ax  = p2s - p1s
    ax_n = float(np.linalg.norm(ax))
    if ax_n < 1e-9:
        return []
    ax = ax / ax_n
    mid = 0.5 * (p1s + p2s)

    def _finish(p3s, tag):
        if p3s is None or not np.all(np.isfinite(p3s)):
            return None
        n3_in = -_geom_normal_np(p3s, geom_type, center, obj_mat, size,
                                 mesh_entry=mesh_entry)
        if not np.all(np.isfinite(n3_in)) or np.linalg.norm(n3_in) < 1e-9:
            return None
        cand = dict(seed)
        cand['p3']    = np.asarray(p3s, float).copy()
        cand['p3s']   = np.asarray(p3s, float)
        cand['n3_in'] = n3_in
        cand['fan_deg'] = float(tag)
        return cand

    out = []

    if strategy == 'kinematic':
        # (D) Where the middle finger actually lands when the hand holds this
        # pinch. Cannot propose a contact the arm can't reach, which is the
        # failure mode the fan's DLS ranking only screens for after the fact.
        p_mf = None if fk_probe is None else fk_probe(seed)
        if p_mf is None:
            return []
        p3s = _project_to_surface_np(np.asarray(p_mf, float), geom_type, center,
                                     obj_mat, size, mesh_entry=mesh_entry)
        c = _finish(p3s, 0.0)
        return [c] if c is not None else []

    if strategy == 'patch_offset':
        # Walk a FIXED distance from the index contact in the index patch's OWN
        # coordinates. The patch axes are the surface's principal-curvature
        # directions, so this is defined entirely by object geometry -- no palm
        # frame, hence none of the world-orientation dependence that made
        # 'tangent' propose sub-table contacts, and no reliance on the middle
        # finger being anywhere in particular, which is what 'kinematic'
        # wrongly assumed.
        #
        # The offset is CLAMPED to the patch bounds: the shared-patch branch of
        # _run_stage confines contact 3 to exactly this region, so a seed outside
        # it would be discarded the same way the fan's is.
        if patch_frame is None:
            return []
        _lo0, _hi0 = float(patch_frame['t_lo_0']), float(patch_frame['t_hi_0'])
        _lo1, _hi1 = float(patch_frame['t_lo_1']), float(patch_frame['t_hi_1'])
        _d = float(patch_offset_m)
        # Fan over DIRECTIONS IN PATCH COORDINATES (not world), so the DLS rank
        # has alternatives without any of them leaving the patch.
        for a in np.linspace(0.0, 2.0 * np.pi, max(int(n_fan), 1), endpoint=False):
            _t0 = float(np.clip(_d * np.cos(a), _lo0, _hi0))
            _t1 = float(np.clip(_d * np.sin(a), _lo1, _hi1))
            if abs(_t0) < 1e-6 and abs(_t1) < 1e-6:
                continue
            p3s = _patch_point_np(patch_frame, _t0, _t1, center, obj_mat)
            c = _finish(p3s, float(np.rad2deg(a)))
            if c is not None:
                c['patch_t'] = (_t0, _t1)
                out.append(c)
        return out

    if strategy == 'tangent':
        # (B) Palm -y, projected onto the index contact's tangent plane, walked
        # along the surface. A small fan is retained ONLY so the DLS ranking has
        # alternatives when the nominal direction is blocked -- +/-15deg, not the
        # +/-40deg the old fan used, which is wide enough to leave the face.
        if palm_R is None:
            return []
        n2 = np.asarray(seed['n2_in'], float)
        n2 = n2 / (np.linalg.norm(n2) + 1e-12)
        d0 = np.asarray(palm_R, float) @ _MF_FROM_IF_PALM
        d_tan = d0 - np.dot(d0, n2) * n2
        if np.linalg.norm(d_tan) < 1e-9:
            return []
        d_tan = d_tan / np.linalg.norm(d_tan)
        _side = np.cross(n2, d_tan)
        for a_deg in np.linspace(-15.0, 15.0, max(int(n_fan), 1)):
            a = np.deg2rad(a_deg)
            d_a = np.cos(a) * d_tan + np.sin(a) * _side
            d_a = d_a - np.dot(d_a, n2) * n2
            nd = np.linalg.norm(d_a)
            if nd < 1e-9:
                continue
            p3s = _surface_walk_np(p2s, d_a / nd, float(pitch_m), geom_type, size,
                                   center, obj_mat, mesh_entry=mesh_entry)
            c = _finish(p3s, a_deg)
            if c is not None:
                out.append(c)
        return out

    # ---- 'fan': the original, unchanged apart from optional crossing placement ----
    ref = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ax, ref); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(ax, u)
    if mf_dir_world is not None:
        b = np.asarray(mf_dir_world, float)
        b = b - np.dot(b, ax) * ax
        if np.linalg.norm(b) > 1e-9:
            u = b / np.linalg.norm(b)
            v = np.cross(ax, u)
            fan = np.linspace(-np.deg2rad(fan_half_deg), np.deg2rad(fan_half_deg),
                              max(int(n_fan), 1))
        else:
            fan = np.linspace(-np.pi, np.pi, max(int(n_fan), 1), endpoint=False)
    else:
        fan = np.linspace(-np.pi, np.pi, max(int(n_fan), 1), endpoint=False)
    reach = max(float(np.max(np.asarray(size, float)[:3])), ax_n) * 1.5
    for a in fan:
        d_hat = np.cos(a) * u + np.sin(a) * v
        try:
            p3s = _project_to_surface_np(mid + reach * d_hat, geom_type, center,
                                         obj_mat, size, mesh_entry=mesh_entry)
        except Exception:
            continue
        c = _finish(p3s, float(np.rad2deg(a)))
        if c is not None:
            out.append(c)
    return out


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


def _minor_axis_local(geom_type, size, mesh_entry=None):
    """Unit vector in the OBJECT LOCAL frame along the object's minor principal
    axis (thinnest direction) — the natural two-finger pinch axis, matching
    benchmarks/ycb_grasp/ik_demo.py's pinch_targets_from (SVD on hull
    vertices). A random-direction antipodal search (_seed_pair) can pick an
    awkward, hard-to-reach pinch geometry; this gives MultiStartGraspPlanner3D
    a well-conditioned seed to try FIRST, before the randomized ones.

    BOX (geom_type 6): shortest half-extent axis — exact, no SVD needed.
    MESH: SVD of the body-frame hull vertices baked into mesh_entry's table
    (same vertices object_sdf.body_hull_halfspaces already extracted).
    CYLINDER/SPHERE: no well-defined "minor axis" (axisymmetric); falls back
    to local x — _seed_pair's randomized seeds are relied on for those shapes.
    """
    if geom_type == 6:   # BOX — shortest half-extent
        ax = int(np.argmin(np.asarray(size, float)[:3]))
        v = np.zeros(3); v[ax] = 1.0
        return v
    if geom_type == _GEOM_TYPE_MESH and mesh_entry is not None:
        V = mesh_entry.get("verts")
        if V is not None and len(V) >= 3:
            # Centre the SVD on the SHARED ray origin (volumetric centroid),
            # not on the vertex mean. The vertex mean is a vertex-DENSITY
            # average (17.6mm off on 036_wood_block's hull), and centring the
            # SVD there tilts the principal axes toward whichever faces the
            # convex decomposition tessellated more finely -- the same bias
            # this change removes from the ray origin, applied to the ray
            # DIRECTION. The vertex set itself is still what is decomposed;
            # only the centre it is measured about changes.
            c = _ray_origin_local(geom_type, mesh_entry)
            _, _, Vt = np.linalg.svd(V - c, full_matrices=False)
            return Vt[2] / (np.linalg.norm(Vt[2]) + 1e-12)
    return np.array([1.0, 0.0, 0.0])


def _fixed_antipodal_seed(geom_type, size, center, obj_mat, local_axis, mesh_entry=None,
                          prefer_outer: bool = True):
    """
    One deterministic, perfectly antipodal seed pair along a fixed axis
    (given in the OBJECT's local frame) through the object's hull CENTROID
    (see the centroid note below) — zero
    angular jitter, unlike _seed_pair's randomized march direction.

    Tried FIRST by MultiStartGraspPlanner3D.solve(), ahead of the randomized
    seeds, as a well-conditioned starting point for every object — local_axis
    is normally _minor_axis_local's result (the natural pinch axis), not an
    arbitrary guess.

    local_axis : (3,) unit vector in the object's local frame, e.g. [1,0,0].

    The ray is cast through the mesh's HULL CENTROID, not through the geom
    frame's origin. Those coincide for the primitive shapes but NOT for YCB
    meshes: the scans are authored with the origin wherever the capture rig
    put it, commonly at the object's base. 036_wood_block is the clear case --
    its MJCF carries <inertial pos="... 0.1027">, i.e. the true centre of mass
    sits 103mm ABOVE the body origin on a 207mm-tall block. Raying through the
    origin there exits at the bottom RIM, so both seed contacts land on an
    edge (measured kappa_max 536 against the seed_kappa_max_reject limit of
    40), the whole minor-axis pair is discarded by solve()'s curvature gate,
    and the solve falls back to _seed_pair's randomized search -- which starts
    high on the object and, because each Picard stage re-seeds from the
    previous stage's solution, ratchets upward to the top edge (measured:
    every configuration converged to z~205mm on a 207.7mm block, regardless of
    which cost term or collision constraint was ablated). Centroid-raying
    removes that whole failure chain at its source.

    The centroid is used ONLY as the ray origin. Every SDF/normal call still
    takes the geom frame's own centre, since that is the frame those functions
    are defined in.

    Returns the same dict schema as _seed_pair (offsets=(0,0), delta_deg=0
    since both contacts land exactly on the surface with no jitter).
    """
    c = np.asarray(center, float)
    d_world = obj_mat @ np.asarray(local_axis, float)
    d_world /= np.linalg.norm(d_world) + 1e-12
    bbox_r = float(np.max(size)) * 2.5

    # Ray origin: the SHARED seed origin in WORLD (_ray_origin_local -- the
    # mesh's volumetric centroid, or the geom centre for primitives, where the
    # two coincide). Shared with _seed_pair so both sources agree on where the
    # object is; this used to be an inline hull-VERTEX mean, 17.6mm off the
    # true centroid on 036_wood_block.
    c_ray = c + obj_mat @ _ray_origin_local(geom_type, mesh_entry)

    # OUTER-surface selection. Projecting the two far endpoints to the nearest
    # surface lands on an INNER wall for a hollow object (see
    # _ray_surface_crossings_np). Taking the ray's first and last zero crossings
    # gives the two outer walls directly. Falls back to projection for analytic
    # primitives (no SDF callable), and whenever the ray finds fewer than two
    # crossings.
    _outer = None
    if prefer_outer and geom_type == _GEOM_TYPE_MESH and mesh_entry is not None:
        _o_l = _ray_origin_local(geom_type, mesh_entry)
        _d_l = np.asarray(obj_mat, float).T @ d_world
        _outer = _outer_pair_t(mesh_entry, _o_l, _d_l)
    if _outer is not None:
        _o_l = _ray_origin_local(geom_type, mesh_entry)
        _d_l = np.asarray(obj_mat, float).T @ d_world
        # Polished onto the zero level set so the contact is exactly where the
        # surrogate will be fitted, as the projection path already guaranteed.
        p1s = _project_to_surface_np(c + obj_mat @ (_o_l + _outer[1] * _d_l),
                                     geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
        p2s = _project_to_surface_np(c + obj_mat @ (_o_l + _outer[0] * _d_l),
                                     geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
    else:
        p1s = _project_to_surface_np(c_ray + d_world * bbox_r, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
        p2s = _project_to_surface_np(c_ray - d_world * bbox_r, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
    n1_in = -_geom_normal_np(p1s, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
    n2_in = -_geom_normal_np(p2s, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)

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


def _chart_pair_seeds(uv_atlas: dict, center, obj_mat, geom_type, size, mesh_entry=None,
                      max_down_deg: float = 60.0, top_k: int = 5):
    """Ranked list of chart-aware antipodal seed pairs for a MESH object,
    replacing _seed_pair's random-direction march for the use_uv_atlas_contact
    path — see the session's seeding-rethink notes (chart assignment is fixed
    for the WHOLE solve at seed time, see _mesh_uv_local_contact_ca /
    GraspPlanner3D.solve's chart-assignment comment, so a good seed here is
    the only lever available; nothing downstream can escape a bad chart
    choice).

    Filters the atlas's big (>= min_area_frac) charts by
    object_uv_atlas.filter_bottom_facing_charts (drops charts resting against
    the table at the object's CURRENT placement — gravity-based, see that
    function's docstring), then scores every chart PAIR by
    combo_score = -||n_i + n_j|| (small resultant norm = well-opposed normals
    = force-closure-friendly; reduces to maximizing -n_i.n_j for pairs, see
    chart_score_debug.py's docstring for the N-finger generalization this
    matches). Returns the top_k pairs, best (most negative... i.e. highest
    score) first, each as a p1s/p2s/n1_in/n2_in seed dict (same schema as
    _seed_pair/_fixed_antipodal_seed) — the chart CENTROID (object-local),
    projected onto the true SDF surface, is used as that chart's contact
    point (no further in-chart search here; _mesh_uv_local_contact_ca's local
    neighborhood is what actually explores within the chart during the NLP).

    Returns [] if uv_atlas is None or fewer than 2 charts survive filtering
    (caller falls back to the existing _seed_pair random search).
    """
    if uv_atlas is None:
        return []
    atlas = uv_atlas["atlas"]
    big = uv_atlas["big_chart_ids"]
    chart_normal = uv_atlas["chart_normal"]
    chart_centroid = uv_atlas["chart_centroid"]

    c = np.asarray(center, float)
    up_local = obj_mat.T @ np.array([0.0, 0.0, 1.0])
    charts = _object_uv_atlas.filter_bottom_facing_charts(
        big, chart_normal, up_local, max_down_deg=max_down_deg)
    if len(charts) < 2:
        return []

    scored = []
    for i in range(len(charts)):
        for j in range(i + 1, len(charts)):
            ci, cj = int(charts[i]), int(charts[j])
            s = chart_normal[ci] + chart_normal[cj]
            scored.append((-float(np.linalg.norm(s)), ci, cj))
    scored.sort(key=lambda t: -t[0])   # highest score (closest to 0) first

    seeds = []
    for score, ci, cj in scored[:top_k]:
        p1s_l = chart_centroid[ci]
        p2s_l = chart_centroid[cj]
        p1s = _project_to_surface_np(c + obj_mat @ p1s_l, geom_type, c, obj_mat, size,
                                     mesh_entry=mesh_entry)
        p2s = _project_to_surface_np(c + obj_mat @ p2s_l, geom_type, c, obj_mat, size,
                                     mesh_entry=mesh_entry)
        n1_in = -_geom_normal_np(p1s, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
        n2_in = -_geom_normal_np(p2s, geom_type, c, obj_mat, size, mesh_entry=mesh_entry)
        seeds.append({
            'p1': p1s.copy(), 'p2': p2s.copy(),
            'p1s': p1s, 'p2s': p2s,
            'n1_in': n1_in, 'n2_in': n2_in,
            'offsets': (0.0, 0.0), 'delta_deg': 0.0,
            'chart_pair': (ci, cj),
            'antipodal_score': score,   # -||n_ci + n_cj|| — see docstring
        })
    return seeds


def _reachable_contact(p, ground_z, r_tip, z_margin=0.002):
    """Returns False if the contact is too close to the support surface for the
    fingertip to reach.

    `r_tip` is whatever the caller decides the tip needs underneath it. Passing
    cfg.r_thumb/r_index (the ISOTROPIC bounding-sphere radius, 19.4mm on the
    LEAP tip) is the conservative reading and bans the bottom 21.4mm of every
    object -- 71% of a 30mm 009_gelatin_box. The pad's honest extent along the
    contact direction is 10.8mm (see _tip_support_along), and the NLP's own
    ground-collision constraint (ground_clearance_m) enforces real clearance on
    the final pose anyway, so the SEED gate does not have to be the binding
    one. cfg.seed_ground_clearance_m replaces r_tip here when set.
    """
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


def _tangent_basis_np(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to unit normal n."""
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = ref - np.dot(ref, n) * n
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n, t1)
    return t1, t2


def _mesh_tangent_contact_ca(opti, seed_world: np.ndarray, seed_normal_out: np.ndarray,
                             center_np, mat_np, mesh_entry: dict, t_bound: float = 0.05):
    """Mesh contact as a 2-DOF tangent-plane offset from a seed point, reprojected onto
    the surface via object_sdf's short (k=2) unit-gradient-step projector.

    Replaces the free-R^3-point + SDF=0 equality used for analytic shapes: the returned
    expression `p_world` is on (near) the surface BY CONSTRUCTION, so no new equality
    constraint is added for mesh objects — only the 2 tangential DOF are free, bounded to
    keep the offset within the projector's basin of good convergence (t_bound, metres).

    Parameters
    ----------
    seed_world      : (3,) world-frame anchor point (already on the surface, e.g. from
                       _seed_pair), around which the tangent frame is built.
    seed_normal_out : (3,) OUTWARD unit normal at the seed, defines the tangent plane.

    Returns (t_var, p_world) — the new opti.variable(2) and the CasADi expression for the
    resulting world-frame contact point.
    """
    t1, t2 = _tangent_basis_np(seed_normal_out)
    seed_l = mat_np.T @ (np.asarray(seed_world, float) - np.asarray(center_np, float))
    t1_l   = mat_np.T @ t1
    t2_l   = mat_np.T @ t2

    t_var = opti.variable(2)
    opti.subject_to(opti.bounded(-t_bound, t_var[0], t_bound))
    opti.subject_to(opti.bounded(-t_bound, t_var[1], t_bound))
    opti.set_initial(t_var, np.zeros(2))

    p_tangent_l = ca.DM(seed_l) + t_var[0] * ca.DM(t1_l) + t_var[1] * ca.DM(t2_l)
    p_surf_l    = mesh_entry["project_fn_short"](p_tangent_l)
    p_world     = ca.DM(center_np) + ca.DM(mat_np) @ p_surf_l
    return t_var, p_world


def _mesh_uv_local_contact_ca(opti, seed_world: np.ndarray, center_np, mat_np,
                              mesh_entry: dict, uv_atlas: dict, chart_id: int,
                              rings: int = 2):
    """Mesh contact as a 2-DOF offset in a UV-ATLAS-DERIVED local plane, bounded
    by that local neighborhood's ACTUAL boundary polygon (not an arbitrary
    Euclidean box like _mesh_tangent_contact_ca's t_bound) — see
    grasp_control.object_uv_atlas module docstring for the full design
    rationale (why a local neighborhood, not the whole chart, in one solve;
    how GraspPlanner3D's existing Picard relinearization loop re-centers this
    between stages to reach the whole chart across a multi-stage solve).

    Same call signature/frame convention as _mesh_tangent_contact_ca (seed_world
    is WORLD frame; center_np/mat_np are the object's world pose, used to
    convert to/from the object-local frame the uv_atlas and mesh_entry SDF
    both operate in — object_uv_atlas.body_visual_mesh folds geom_pos/geom_quat
    the same way object_sdf.body_hull_halfspaces does, so the two agree on
    what "local frame" means).

    chart_id : which atlas chart this contact is assigned to — decided ONCE at
        seed time (object_uv_atlas.nearest_big_chart) and held fixed for the
        whole solve; only the local neighborhood WITHIN this chart is rebuilt
        between Picard stages (see caller in GraspPlanner3D.solve).

    Returns (uv_var, p_world) — the new opti.variable(2) and the CasADi
    expression for the resulting world-frame contact point (matches
    _mesh_tangent_contact_ca's return contract exactly).
    """
    seed_l = mat_np.T @ (np.asarray(seed_world, float) - np.asarray(center_np, float))
    nb = _object_uv_atlas.local_neighborhood(uv_atlas, chart_id, seed_l, rings=rings)
    centroid, (t1, t2) = nb["plane_centroid"], nb["plane_basis"]
    bound_A, bound_b = nb["bound_A"], nb["bound_b"]

    uv_var = opti.variable(2)
    # Linear inequality bound from the local neighborhood's actual convex-hull
    # boundary (in-plane coords) — replaces _mesh_tangent_contact_ca's fixed
    # +/-t_bound box with the real local extent of usable surface.
    opti.subject_to(ca.DM(bound_A) @ uv_var <= ca.DM(bound_b))
    opti.set_initial(uv_var, nb["seed_uv"])

    p_flat_l = ca.DM(centroid) + uv_var[0] * ca.DM(t1) + uv_var[1] * ca.DM(t2)
    p_surf_l = mesh_entry["project_fn_short"](p_flat_l)
    p_world  = ca.DM(center_np) + ca.DM(mat_np) @ p_surf_l
    return uv_var, p_world


def _sdf_hessian_np(mesh_entry: dict, p_local: np.ndarray) -> np.ndarray:
    """3x3 Hessian of the object-local SDF at p_local, numpy — evaluates the
    CasADi Function cached as mesh_entry['hessian_fn'] (built once in
    _mesh_sdf_entry). MUST use the cached Function: constructing
    ca.jacobian(grad_fn(x), x) from scratch is expensive (differentiating a
    B-spline interpolant graph twice), and this is called O(seed candidates x
    Picard stages) times per solve — a previous per-call-rebuild version of
    this function turned a sub-second op into a multi-minute stall.
    """
    return np.asarray(mesh_entry["hessian_fn"](np.asarray(p_local, float))).reshape(3, 3)


def _mesh_surface_kappa_max_np(mesh_entry: dict, p_local: np.ndarray) -> float:
    """Largest-magnitude principal curvature of the SDF's zero level set at
    p_local (object-local), via the Hessian restricted to the tangent plane.

    Used to steer seed SAMPLING away from edges/corners (see _seed_pair's
    caller in solve()'s seed-generation loop): a seed placed right at a near-
    corner point forces _mesh_quadratic_contact_ca's per-axis SDF-comparison
    bound down to near-zero along the edge-approaching direction (verified:
    kappa up to ~190 on a box corner collapses that axis's bound below 1mm),
    which is the geometrically CORRECT answer for that representation but a
    bad place to have seeded a grasp attempt in the first place — flat
    regions give more usable tangential search room under ANY contact
    representation, not just the quadratic one, so this is a seeding
    improvement independent of which representation is active.
    """
    grad_l = np.asarray(mesh_entry["grad_fn"](p_local), float).reshape(3)
    gnorm = float(np.linalg.norm(grad_l)) + 1e-9
    n_hat = grad_l / gnorm
    t1_l, t2_l = _tangent_basis_np(n_hat)
    H = _sdf_hessian_np(mesh_entry, p_local)
    T = np.stack([t1_l, t2_l], axis=1)
    H_tt = T.T @ H @ T
    return float(np.max(np.abs(np.linalg.eigvalsh(H_tt))))


def _principal_curvature_axes_np(H_tt: np.ndarray, t1_l: np.ndarray, t2_l: np.ndarray):
    """Eigendecompose H_tt (2x2, SDF Hessian restricted to an arbitrary tangent
    basis) and return the two principal-curvature directions in OBJECT-LOCAL
    R^3 (unit vectors) plus their curvatures, sorted ascending by |curvature|
    (axis 0 = flattest direction, axis 1 = most-curved direction).

    Aligning the free variables to these directions (rather than the arbitrary
    _tangent_basis_np pair) is what makes an anisotropic per-axis bound
    possible: a seed near an edge has one nearly-flat direction (along the
    edge) and one sharply-curved direction (toward the edge) -- bounding both
    by the worst eigenvalue (as an isotropic bound must) throws away all the
    safe search room in the flat direction. See module docstring context.
    """
    eigvals, eigvecs = np.linalg.eigh(H_tt)     # ascending by value; eigh assumes symmetric
    order = np.argsort(np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    T = np.stack([t1_l, t2_l], axis=1)          # 3x2, maps 2D tangent coords -> R^3
    axes_l = T @ eigvecs                        # 3x2, columns are unit (T orthonormal, eigvecs orthonormal)
    return axes_l[:, 0], axes_l[:, 1], float(eigvals[0]), float(eigvals[1])


def _sdf_axis_bound_np(mesh_entry: dict, seed_l: np.ndarray, axis_l: np.ndarray,
                       t_bound_max: float, tol: float = 5e-4, n_steps: int = 8,
                       n_refine: int = 6) -> float:
    """How far along axis_l (unit, object-local) can a tangent offset go before
    the TRUE SDF value at that point departs from its seed-time value by more
    than `tol` (metres) -- i.e. before the seed's tangent-plane approximation
    departs from the real surface by more than tol. Directly measures
    surrogate validity instead of inferring it from a curvature magnitude, so
    it can't be fooled by a curvature estimate that over- or under-predicts
    actual model error (e.g. near a knot of the SDF's B-spline interpolant, or
    higher-order terms the quadratic drops).

    NOT a binary search: |fn(seed + t*axis)| is not monotonic in t in general
    -- near a corner, walking toward one adjacent face can dip |fn| back down
    before it rises again on the far side, and a binary search's implicit
    monotonicity assumption would silently accept that far side as "in
    tolerance" despite having crossed a face it shouldn't have. A forward
    march that stops at the FIRST violation has no such blind spot.

    Compares against fn(seed_l) rather than an absolute-zero baseline: seed_l
    is the previous Picard stage's solved point re-used as this stage's seed,
    not always a freshly SDF-projected point (see solve()'s relinearization
    loop), so it can sit a little off the true surface itself -- anchoring to
    its own value keeps the bound measuring "how far can the quadratic model
    depart from what the seed already sees" rather than being spuriously
    collapsed to 0 whenever the seed carries a small pre-existing residual.

    Two-phase: a coarse forward march (n_steps, size t_bound_max/n_steps)
    finds the first step that VIOLATES tolerance -- preserving the "stop at
    first violation, no monotonicity assumed across the whole range" property
    above, since each coarse step only trusts what it directly measures.
    Within that one bracketing [t_ok, t_violate] interval (width exactly one
    coarse step), a bounded bisection refines the boundary -- this recovers
    resolution the coarse grid alone cannot reach: a true bound of, say, 5mm
    sitting inside a first coarse step of 6.25mm (t_bound_max/n_steps with
    n_steps=8, t_bound_max=50mm) would otherwise be reported as exactly 0mm,
    needlessly discarding real, valid search room (confirmed on
    036_wood_block: a seed with a genuine ~5mm safe bound reported 0mm from
    the coarse march alone, collapsing that axis's trust region to a
    degenerate point for no reason -- verified the SDF value only exceeds
    tolerance between 5mm and 6.25mm, not before). Bisection is safe WITHIN
    this one bracket even though the full function isn't globally monotonic:
    the bracket is small (one coarse step), and the search still only trusts
    values it measures, converging to the true crossing point in that
    interval rather than assuming one.

    n_steps forward march + n_refine bisection steps, cheap: n_steps +
    n_refine calls to mesh_entry['fn'] (a CasADi Function, already fast) --
    run once per contact per Picard stage, the same frequency _sdf_hessian_np
    already runs at.
    """
    f0 = float(mesh_entry["fn"](seed_l))

    def _in_tol(t: float) -> bool:
        return abs(float(mesh_entry["fn"](seed_l + t * axis_l)) - f0) <= tol

    step = t_bound_max / n_steps
    t_ok, t_bad = 0.0, None
    for i in range(1, n_steps + 1):
        t = i * step
        if not _in_tol(t):
            t_bad = t
            break
        t_ok = t
    if t_bad is None:
        return t_ok   # every coarse step stayed in tolerance -- flat out to t_bound_max

    lo, hi = t_ok, t_bad   # _in_tol(lo) True (or lo==0, trivially true), _in_tol(hi) False
    for _ in range(n_refine):
        mid = 0.5 * (lo + hi)
        if _in_tol(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _mesh_local_surface_fit_np(mesh_entry: dict, seed_l: np.ndarray,
                               t1_l: np.ndarray, t2_l: np.ndarray, n_l: np.ndarray,
                               radius: float = 0.04, band: float = 0.004,
                               band_inward: float | None = None,
                               normal_agree_min: float = 0.5,
                               min_pts: int = 12,
                               quad_gain_min: float = 0.5,
                               grad_norm: float = 1.0):
    """Local surface curvature at seed_l fitted DIRECTLY to nearby mesh
    vertices, with a plane-vs-quadratic model-selection test.

    Returns (kappa0, kappa1, axis0_l, axis1_l, info) in the same convention as
    _principal_curvature_axes_np, or None when the mesh cannot support a fit
    (too few vertices in range) and the caller should fall back to the SDF
    Hessian.

    WHY NOT THE SDF HESSIAN. An SDF encodes distance to the WHOLE shape, so its
    second derivative is a global quantity: near any feature it reports
    curvature that belongs to that feature, not to the local patch. Measured on
    036_wood_block, mid-face on a flat 104x206mm side: the SDF Hessian gives
    kappa=(+0.002, -12.330), the -12.3 coming from the vertical corner ~50mm
    away bleeding into the field. That fake curvature makes the paraboloid
    surrogate bend away from a genuinely flat face immediately, so
    _sdf_axis_bound_np correctly caps the trust region at 4.6mm -- on a face
    with ~50mm of usable surface in that direction. Mesh vertices ARE the
    surface, so a fit to them cannot be contaminated by geometry that is not
    in the sample.

    WHY THE MODEL-SELECTION TEST. A raw mesh fit is not trustworthy either. The
    YCB meshes are scans: the block's "flat" face carries ~1mm of bimodal
    structure (tessellation/scan relief, NOT random noise -- it does not average
    out, plane RMS stays 0.72-0.79mm from n=64 to n=479 samples). A quadratic
    fitted to that absorbs the offset, giving kappa that scales as ~1/r with the
    fit radius: measured -35.5, -15.5, -6.1, -2.4 at r=15,25,40,60mm, i.e. no
    converged value at any radius. So curvature is accepted only when the
    quadratic beats the plane by quad_gain_min in RMS residual -- a real model
    comparison rather than a tuned curvature threshold. The separation is wide:
    on 017_orange the quadratic improves RMS by 91% at EVERY radius with kappa
    stable at -32.2 across a 4x sweep (real curvature), while on the block's
    face it manages only 15-34% (fitting relief). Below the gate, kappa is
    returned as exactly 0 -- a planar patch, whose validity bound is then set
    by _sdf_axis_bound_np's direct SDF search rather than by a fake curvature.

    grad_norm : ||grad f|| at the seed. Needed ONLY for the height-Hessian ->
        SDF-Hessian conversion at the end (H_tt = -||grad f|| * W): n_l arrives
        unit-length, so the gradient's magnitude cannot be recovered from it and
        must be passed in. Defaults to 1.0, which is very nearly right for a
        well-conditioned SDF (measured 0.994-1.001 on the benchmark objects) but
        is not assumed.

    radius : tangent-plane sampling radius (m). Sized to span a useful fraction
        of a face rather than a few mm, since the whole point is a patch that
        covers the graspable region.
    band   : max |offset along the normal| (m) for an included vertex. Keeps
        points that have wrapped onto an ADJACENT face out of the fit -- without
        it a seed near an edge mixes two faces into one quadratic.

        For a THIN-SHELL object this band must also exclude the far side of the
        shell. A YCB cup's wall is only ~2-3mm thick, so a symmetric 4mm band
        centred on the outer surface reaches straight through it and pulls the
        INNER surface into the same fit -- two roughly parallel sheets whose
        least-squares quadratic is meaningless. The band is therefore applied
        ASYMMETRICALLY: vertices are kept from band_out on the outward side to
        band_in on the inward side, with band_in defaulting to a fraction of
        band so a thin wall's inner surface falls outside it. Vertices are also
        rejected when their own surface normal disagrees with the seed's, which
        catches the inner surface even when it sits within the distance band.
    """
    V = mesh_entry.get("visual_verts")
    if V is None or len(V) < min_pts:
        return None
    # Cap the sampling radius by the OBJECT's own size. A quadratic can only
    # describe a patch that subtends a modest angle: on 065-a_cups (outer radius
    # ~29mm) the default 40mm radius wraps most of the way around the wall and
    # the fit collapses (gain 0.25, kappa drifting), while r=10-20mm recovers a
    # stable kappa=(0, -45.3) at gain 0.95-0.97 -- correct for a cylinder, and
    # consistent across a 2x radius sweep. The default was tuned on the wood
    # block's 104mm face and is simply too large for small objects, so scale it
    # to a fraction of the object's smallest extent.
    _ext = np.asarray(V, float).max(0) - np.asarray(V, float).min(0)
    radius = float(min(radius, 0.35 * float(np.min(_ext))))
    d = np.asarray(V, float) - np.asarray(seed_l, float)
    u = d @ t1_l
    v = d @ t2_l
    w = d @ n_l
    # Asymmetric band (see the `band` note): generous outward, tight inward, so a
    # thin shell's far surface is excluded rather than fitted alongside the near
    # one. band_in defaults to a third of band -- below a YCB cup's ~2-3mm wall.
    band_in = band if band_inward is None else band_inward
    m = ((u * u + v * v) <= radius * radius) & (w <= band) & (w >= -band_in)
    # Normal agreement: reject vertices whose own outward normal opposes the
    # seed's. On a thin wall the inner surface faces the other way, so this
    # removes it even where it falls inside the distance band; on a solid object
    # it is a no-op for anything the band already admits.
    nrm = mesh_entry.get("visual_normals")
    if nrm is not None and len(nrm) == len(V):
        m &= (np.asarray(nrm, float) @ np.asarray(n_l, float)) >= normal_agree_min
    n_sel = int(m.sum())
    if n_sel < min_pts:
        return None
    U, Vv, W = u[m], v[m], w[m]

    ones = np.ones(n_sel)
    A_pl = np.stack([U, Vv, ones], axis=1)
    c_pl, *_ = np.linalg.lstsq(A_pl, W, rcond=None)
    rms_pl = float(np.sqrt(((W - A_pl @ c_pl) ** 2).mean()))

    A_q = np.stack([U * U, U * Vv, Vv * Vv, U, Vv, ones], axis=1)
    c_q, *_ = np.linalg.lstsq(A_q, W, rcond=None)
    rms_q = float(np.sqrt(((W - A_q @ c_q) ** 2).mean()))

    gain = 1.0 - (rms_q / rms_pl) if rms_pl > 1e-12 else 0.0
    info = dict(n=n_sel, rms_plane=rms_pl, rms_quad=rms_q, gain=gain,
                radius=radius, planar=bool(gain < quad_gain_min))

    if gain < quad_gain_min:
        # Planar: keep the seed's own tangent axes, zero curvature. Return
        # order matches _principal_curvature_axes_np: (axis0, axis1, k0, k1).
        return (np.asarray(t1_l, float), np.asarray(t2_l, float), 0.0, 0.0, info)

    # w = a u^2 + b uv + c v^2 + ... -> the HEIGHT Hessian in (t1,t2) is
    # W = [[2a, b], [b, 2c]], i.e. the second derivative of the surface's height
    # ABOVE ITS OWN TANGENT PLANE, measured along +n_l (outward).
    #
    # That is the OPPOSITE SIGN CONVENTION to the SDF Hessian this function must
    # return, and it must be converted, not returned raw. The caller
    # (_mesh_quadratic_contact_ca) consumes kappa only through
    #     h(t) = -(kappa0*t0^2 + kappa1*t1^2) / (2*grad_norm)
    # which is the second-order implicit-function solution of f(seed + t.axis +
    # h*n) = 0, so its kappa must be the SDF's tangential Hessian H_tt. Equating
    # the two expressions for the same height gives
    #     W = -H_tt / ||grad f||        =>      H_tt = -||grad f|| * W
    # Verified numerically on all three benchmark objects (SDF H_tt eigenvalues
    # vs -||grad f||*W eigenvalues agree in sign and magnitude to a few percent;
    # the residual difference is the SDF-vs-mesh disagreement this fit exists to
    # correct, not a convention mismatch).
    #
    # Returning W RAW was a sign bug: on 017_orange it gave kappa=(-30.3,-31.2)
    # where the true SDF convention needs +31, so h came out POSITIVE and the
    # paraboloid bulged OUTWARD on a convex sphere -- the surrogate curving away
    # from the object instead of hugging it, sitting ~0.93mm outside the true
    # surface at the trust-region edge. Visible directly in the seed-quadratic
    # visualizer as a patch bowing the wrong way.
    W_tt = np.array([[2.0 * c_q[0], c_q[1]],
                     [c_q[1], 2.0 * c_q[2]]], float)
    H_tt = -float(grad_norm) * W_tt
    return _principal_curvature_axes_np(H_tt, t1_l, t2_l) + (info,)


def _patch_max_sdf_err_np(mesh_entry: dict, seed_l, axis0_l, axis1_l, n_l,
                          kappa0, kappa1, grad_norm,
                          t_lo_0, t_hi_0, t_lo_1, t_hi_1, n=5) -> float:
    """Max |SDF - SDF(seed)| over the whole paraboloid patch, not just its axes.

    Reconstructs the same surface the surrogate hands the NLP --
    p_l(t) = seed_l + t0*axis0_l + t1*axis1_l + h(t)*n_l, h as in
    _mesh_quadratic_contact_ca -- on an n x n grid over the bound rectangle and
    returns the worst departure from the seed's own SDF value. Anchored to
    fn(seed_l) for the same reason _sdf_axis_bound_np is: the seed may itself
    carry a small residual from the previous Picard stage, and an absolute-zero
    baseline would charge that residual to the patch.

    n=5 (25 evaluations) rather than a fine grid: this runs inside a bisection,
    per contact, per Picard stage. The grid includes all four corners and both
    centre-lines, which is where the error actually concentrates -- the axis
    searches already cover the centre-lines exactly, so the corners are what
    this adds.
    """
    f0 = float(mesh_entry["fn"](seed_l))
    T0 = np.linspace(t_lo_0, t_hi_0, n)
    T1 = np.linspace(t_lo_1, t_hi_1, n)
    worst = 0.0
    for a in T0:
        for b in T1:
            h = -(kappa0 * a * a + kappa1 * b * b) / (2.0 * grad_norm)
            p = seed_l + a * axis0_l + b * axis1_l + h * n_l
            e = abs(float(mesh_entry["fn"](p)) - f0)
            if e > worst:
                worst = e
    return worst


def _shrink_patch_to_tol(mesh_entry: dict, seed_l, axis0_l, axis1_l, n_l,
                         kappa0, kappa1, grad_norm, bounds,
                         tol: float, n_refine: int = 6, n_grid: int = 5):
    """Scale the four axis bounds down uniformly until the WHOLE patch is within
    `tol`, and return the scaled (t_lo_0, t_hi_0, t_lo_1, t_hi_1).

    The axis searches bound the centre-lines; this bounds the area they span.
    Returns the input unchanged (scale 1.0) when the patch already honours the
    tolerance -- but that is not the typical case. Because the axis searches
    deliberately walk out until the CENTRE-LINES hit the tolerance, the corners
    start over it, so the shrink engages on curved and planar patches alike
    (measured: 4 of 6 017_orange contacts enter the bisection, not just the
    large 036_wood_block faces where t_bound_max is what stopped the search).

    Bisection on the scale factor is sound here in a way a bisection along a
    single axis would not be: error grows monotonically with the patch's SIZE
    (a smaller rectangle is a subset of a larger one, so its max cannot exceed
    the larger one's), even though error along any one axis is not monotonic in
    t. Shrinking can only ever remove sample points.
    """
    t_lo_0, t_hi_0, t_lo_1, t_hi_1 = bounds

    def _err(scale: float) -> float:
        return _patch_max_sdf_err_np(
            mesh_entry, seed_l, axis0_l, axis1_l, n_l, kappa0, kappa1, grad_norm,
            t_lo_0 * scale, t_hi_0 * scale, t_lo_1 * scale, t_hi_1 * scale,
            n=n_grid)

    if _err(1.0) <= tol:
        return t_lo_0, t_hi_0, t_lo_1, t_hi_1

    lo, hi = 0.0, 1.0          # _err(0) == 0 <= tol by construction; _err(1) > tol
    for _ in range(n_refine):
        mid = 0.5 * (lo + hi)
        if _err(mid) <= tol:
            lo = mid
        else:
            hi = mid
    return t_lo_0 * lo, t_hi_0 * lo, t_lo_1 * lo, t_hi_1 * lo


def _mesh_quadratic_contact_ca(opti, seed_world: np.ndarray, seed_normal_out: np.ndarray,
                               center_np, mat_np, mesh_entry: dict,
                               t_bound_max: float = 0.05, sdf_err_tol: float = 5e-4,
                               mesh_fit: bool = False, mesh_fit_radius: float = 0.04,
                               mesh_fit_quad_gain_min: float = 0.5):
    """Mesh contact as a 2-DOF offset along the two PRINCIPAL CURVATURE AXES at
    the seed, placed on a LOCAL QUADRATIC (paraboloid) surrogate of the
    surface fit from the SDF's own gradient + Hessian -- no per-candidate SDF
    projection call.

        h(t1,t2) = -(t1,t2) H_tt' (t1,t2)^T / (2 ||grad f(seed)||)
        p(t1,t2) = seed + t1*axis0_hat + t2*axis1_hat + h(t1,t2)*n_hat

    derived from the second-order Taylor expansion of the SDF f about the
    seed, imposing f=0 and keeping only the leading (quadratic) term of the
    implicit relationship this defines for h(t1,t2). H_tt' here is H_tt
    re-expressed in the principal-axis basis, i.e. diagonal: H_tt' =
    diag(kappa0, kappa1).

    This is exactly differentiable (a degree-2 polynomial in t1,t2) with no
    oscillating-residual failure mode near creases, UNLIKE surface_project --
    but unlike surface_project it has NO correction mechanism if (t1,t2)
    strays past where the quadratic model is accurate, so each axis's bound is
    sized independently by _sdf_axis_bound_np: a direct binary search on the
    TRUE SDF value along that axis, not a curvature-magnitude proxy. This
    matters specifically near an edge/corner, where curvature is anisotropic
    -- one axis (along the edge) can stay near t_bound_max while the other
    (toward the edge) shrinks sharply; an isotropic bound would needlessly
    shrink both and strand the optimizer near a poor seed. This bound is
    load-bearing, not a tuning knob -- see module docstring context.

    Same call signature/frame convention as _mesh_tangent_contact_ca
    (seed_world is WORLD frame; center_np/mat_np are the object's world pose).
    Return contract is EXTENDED (4-tuple, not 2): (t_var, p_world, bounds, frame)
    where bounds = (t_bound_0, t_bound_1) in metres (needed by the Picard loop
    to detect a solution pinned at its trust region -- see solve()'s
    relinearization loop) and frame is a dict of the object-local paraboloid
    parameters (seed_l, axis0_l, axis1_l, n_l, kappa0, kappa1, grad_norm,
    t_bound_0, t_bound_1) letting a caller reconstruct/draw p_local(t1,t2) =
    seed_l + t1*axis0_l + t2*axis1_l + h(t1,t2)*n_l post-hoc (e.g. for the
    quadratic-fit visualizer, analogous to plot_uv_path.py for the UV atlas).

    sdf_err_tol : max allowed |SDF value| (metres) at the edge of the bound
        along each axis -- the model-validity tolerance _sdf_axis_bound_np
        searches for. Independent of t_bound_max, which is only a hard cap
        for directions where the surface stays flat well past a useful range.
    """
    t1, t2 = _tangent_basis_np(seed_normal_out)
    seed_l = mat_np.T @ (np.asarray(seed_world, float) - np.asarray(center_np, float))
    t1_l   = mat_np.T @ t1
    t2_l   = mat_np.T @ t2

    grad_l  = np.asarray(mesh_entry["grad_fn"](seed_l), float).reshape(3)
    # Floor, not just an epsilon against exact zero: ||grad f|| sags near a
    # crease (object_sdf.surface_project's own docstring measures it down to
    # ~0.01 there) rather than only hitting exactly 0. h = -(...)/(2*grad_norm)
    # amplifies inversely with this floor, so an un-floored near-zero value
    # would blow h up by ~100x right where the seed is least trustworthy —
    # floor it at the same order of magnitude object_sdf already treats as
    # "crease regime" so h stays bounded even there (the SDF-comparison bound
    # search below still independently catches an unsafe result; this just
    # keeps h itself from being numerically wild going in).
    grad_norm = max(float(np.linalg.norm(grad_l)), 1e-2)
    # True (unfloored) unit normal -- needed by the mesh fit below to measure
    # each sampled vertex's height above the seed's tangent plane. Distinct
    # from grad_norm, which is a denominator and therefore floored.
    n_l_unit = grad_l / (float(np.linalg.norm(grad_l)) + 1e-9)
    H       = _sdf_hessian_np(mesh_entry, seed_l)
    T       = np.stack([t1_l, t2_l], axis=1)          # 3x2
    H_tt    = T.T @ H @ T                              # 2x2, curvature restricted to tangent plane

    axis0_l, axis1_l, kappa0, kappa1 = _principal_curvature_axes_np(H_tt, t1_l, t2_l)

    # Prefer curvature fitted directly to the MESH over the SDF Hessian's --
    # the SDF's second derivative is contaminated by geometry that is not local
    # (a corner 50mm away shows up as kappa=-12.3 on a flat face), which
    # collapses the trust region on exactly the large flat regions a grasp
    # wants. _mesh_local_surface_fit_np returns None when the mesh is too
    # sparse to fit, in which case the SDF Hessian above stands.
    if mesh_fit:
        _fit = _mesh_local_surface_fit_np(
            mesh_entry, seed_l, t1_l, t2_l, n_l_unit,
            radius=mesh_fit_radius, quad_gain_min=mesh_fit_quad_gain_min,
            # TRUE (unfloored) magnitude -- this scales a curvature conversion,
            # not a denominator, so the grad_norm floor used for h() would
            # distort it near a crease rather than protect it.
            grad_norm=float(np.linalg.norm(grad_l)))
        if _fit is not None:
            axis0_l, axis1_l, kappa0, kappa1, _fit_info = _fit

    # Per-axis bound via direct SDF comparison, not a shared curvature-derived
    # radius -- lets a flat direction (e.g. along a nearby edge) keep nearly
    # all of t_bound_max while a sharply-curved direction (toward the edge)
    # shrinks on its own.
    # ASYMMETRIC per-axis, per-SIDE bounds. Each of the four directions
    # (+/-axis0, +/-axis1) gets its own search, and each becomes that side's
    # box limit directly.
    #
    # This previously took min(+side, -side) and applied it symmetrically, on
    # the reasoning that a shared box must never overstate safety on the looser
    # side. That is true but throws away most of the patch: the tighter side is
    # usually tight because the seed happens to sit near ONE face edge, and
    # collapsing both sides to that distance discards the entire rest of the
    # face. Measured on 036_wood_block mid-face (planar fit, so the surrogate is
    # exact): +axis0 is good to 31.5mm and -axis0 to 4.6mm, and the symmetric
    # rule gave [-4.6, +4.6] -- 9mm of a 104mm-wide face, with 27mm of verified-
    # good surface on the positive side simply discarded. Asymmetric bounds give
    # [-4.6, +31.5], which is the actual measured validity region and lets a
    # contact traverse the face it is standing on.
    #
    # Nothing about the surrogate requires symmetry: h(t) is evaluated at
    # whatever t the optimizer picks, and each side's bound is an independent
    # measurement of how far the model stays within sdf_err_tol in THAT
    # direction. The old symmetric form was a conservative simplification, not a
    # correctness requirement.
    t_lo_0 = -_sdf_axis_bound_np(mesh_entry, seed_l, -axis0_l, t_bound_max, tol=sdf_err_tol)
    t_hi_0 = _sdf_axis_bound_np(mesh_entry, seed_l, axis0_l, t_bound_max, tol=sdf_err_tol)
    t_lo_1 = -_sdf_axis_bound_np(mesh_entry, seed_l, -axis1_l, t_bound_max, tol=sdf_err_tol)
    t_hi_1 = _sdf_axis_bound_np(mesh_entry, seed_l, axis1_l, t_bound_max, tol=sdf_err_tol)

    # PATCH-WIDE shrink. The four searches above each walk ONE axis, so they
    # bound the error along the rectangle's two centre-lines and say nothing
    # about its interior or corners -- and the corner is where a paraboloid
    # departs worst from the real surface. At a 0.5mm tolerance with a 50mm cap
    # the gap was invisible (measured 0.63mm patch error against a 0.5mm
    # bound). At 4mm with a 100mm cap it is not: 036_wood_block measured
    # 8.78mm of true patch error against a 4mm tolerance, and the patch visibly
    # hung off the side of the block it was fitted to.
    #
    # So the axis bounds are treated as an upper bracket and scaled DOWN
    # uniformly until the whole patch honours sdf_err_tol. Uniform, rather than
    # shrinking whichever axis looks guilty: the asymmetry between the four
    # sides is a real measurement (see the note above) and scaling preserves
    # its shape, while singling out one side would distort it by an amount no
    # measurement justifies.
    #
    # Costs one 25-point grid evaluation when the patch already honours the
    # tolerance and n_refine+1 of them when it does not -- measured at 1-3ms per
    # contact either way, against a multi-second NLP per seed. Note the shrink
    # is NOT a rare path at a 4mm tolerance: most contacts on every object
    # measured (017_orange included) do enter the bisection, because the axis
    # searches walk out until the CENTRE-LINES reach the tolerance, which
    # leaves the corners over it by construction.
    t_lo_0, t_hi_0, t_lo_1, t_hi_1 = _shrink_patch_to_tol(
        mesh_entry, seed_l, axis0_l, axis1_l, n_l_unit,
        kappa0, kappa1, grad_norm, (t_lo_0, t_hi_0, t_lo_1, t_hi_1),
        tol=sdf_err_tol)

    # Reported bound stays the SYMMETRIC half-width (the smaller side), since
    # that is what the Picard loop's pinned-at-trust-region test and the
    # edge-margin cost both interpret as "how much room this axis has". Callers
    # wanting the true asymmetric interval read t_lo_*/t_hi_* from the frame.
    t_bound_0 = min(-t_lo_0, t_hi_0)
    t_bound_1 = min(-t_lo_1, t_hi_1)

    t_var = opti.variable(2)
    opti.subject_to(opti.bounded(t_lo_0, t_var[0], t_hi_0))
    opti.subject_to(opti.bounded(t_lo_1, t_var[1], t_hi_1))
    opti.set_initial(t_var, np.zeros(2))

    # h(t1,t2): leading-order implicit-function solution of f(seed + t.axis_hat
    # + h*n_hat) = 0 for h, from the SDF's 2nd-order Taylor expansion at seed.
    # Diagonal because t_var is expressed directly in the eigenbasis. Uses the
    # FLOORED grad_norm (the Taylor-expansion denominator, where a near-zero
    # true value would blow h up) -- n_l below is unit-length regardless,
    # normalized by the true (unfloored) gradient magnitude, since it's a
    # direction, not a denominator, and flooring it would make it non-unit.
    h = -(kappa0 * t_var[0]**2 + kappa1 * t_var[1]**2) / (2.0 * grad_norm)
    n_l = grad_l / (float(np.linalg.norm(grad_l)) + 1e-9)
    p_surf_l = ca.DM(seed_l) + t_var[0] * ca.DM(axis0_l) + t_var[1] * ca.DM(axis1_l) + h * ca.DM(n_l)
    p_world  = ca.DM(center_np) + ca.DM(mat_np) @ p_surf_l
    # Everything a caller needs to independently RECONSTRUCT and DRAW this
    # paraboloid patch post-hoc (e.g. the local-quadratic visualizer) without
    # re-deriving it from mesh_entry -- object-local frame throughout, matching
    # how obj_center/obj_mat are already saved by _save_iter_npz. p_local(t) =
    # seed_l + t[0]*axis0_l + t[1]*axis1_l + h(t)*n_l reproduces p_surf_l above
    # exactly for any (t0,t1), including OUTSIDE (t_bound_0,t_bound_1) (useful
    # for a viz that wants to show a little of the invalid region too).
    frame = dict(seed_l=seed_l, axis0_l=axis0_l, axis1_l=axis1_l, n_l=n_l,
                kappa0=kappa0, kappa1=kappa1, grad_norm=grad_norm,
                t_bound_0=t_bound_0, t_bound_1=t_bound_1,
                t_lo_0=t_lo_0, t_hi_0=t_hi_0, t_lo_1=t_lo_1, t_hi_1=t_hi_1)
    return t_var, p_world, (t_bound_0, t_bound_1), frame


def _quadratic_inward_normal_ca(t_var, frame: dict, mat_np):
    """INWARD unit normal of the local paraboloid at (t0,t1), as a CasADi MX
    expression in WORLD coordinates -- the closed-form counterpart of
    _sym_inward_normal_ca for a mesh contact under use_quadratic_contact.

    The patch _mesh_quadratic_contact_ca builds is

        p(t) = seed + t0*a0 + t1*a1 + h(t)*n,
        h(t) = -(kappa0*t0^2 + kappa1*t1^2) / (2*grad_norm)

    so its tangents and hence its normal are available in CLOSED FORM:

        dp/dt0 = a0 - (kappa0*t0/grad_norm) * n
        dp/dt1 = a1 - (kappa1*t1/grad_norm) * n
        n(t)   = normalize(dp/dt0 x dp/dt1)

    Every coefficient (a0, a1, n, kappa0/1, grad_norm) is a numpy constant
    frozen at seed time, so this is a low-degree polynomial in t_var --
    exactly differentiable, no SDF call, no mesh lookup.

    WHY THIS MATTERS. The wrench-cone LP and GWS both build their friction
    cones on a contact frame [n_in | t1 | t2]. For a mesh that frame was
    always a FROZEN parameter, i.e. the normal at the seed, held constant
    while the optimizer moved the contact across the patch -- which is
    inconsistent with the very surrogate being used for POSITION: kappa is
    precisely the statement that the normal tilts as the contact moves, and
    freezing the frame discards it. Measured normal error against the true
    SDF over a patch: 017_orange 9.22deg median (frozen) vs 2.02deg
    (analytic); 065-a_cups 8.77 vs 5.02. On a PLANAR patch (kappa=0) the two
    coincide exactly, which is the correct degenerate case.

    Sign: returns the INWARD normal, matching _sym_inward_normal_ca and
    _build_contact_frame_3d's convention (R[:,0] = inward). frame['n_l'] is
    the OUTWARD SDF gradient direction, so the cross product is oriented
    against it and then negated.

    mat_np : object world rotation; the frame's vectors are object-local, and
        the wrench machinery works in world, so the result is rotated out.
    """
    a0 = ca.DM(np.asarray(frame["axis0_l"], float))
    a1 = ca.DM(np.asarray(frame["axis1_l"], float))
    n_l = np.asarray(frame["n_l"], float)
    n_dm = ca.DM(n_l)
    k0 = float(frame["kappa0"]); k1 = float(frame["kappa1"])
    gn = float(frame["grad_norm"])

    dp0 = a0 - (k0 * t_var[0] / gn) * n_dm
    dp1 = a1 - (k1 * t_var[1] / gn) * n_dm
    nc = ca.cross(dp0, dp1)
    # Orient against the seed's OUTWARD normal. a0,a1,n_l are right-handed up
    # to the eigen-decomposition's arbitrary axis signs, so the raw cross
    # product's orientation is not guaranteed -- fix it with the numpy-side
    # sign, which is a constant (not a function of t_var) and so keeps the
    # expression smooth.
    _sgn = float(np.sign(np.dot(np.cross(np.asarray(frame["axis0_l"], float),
                                         np.asarray(frame["axis1_l"], float)), n_l)))
    if _sgn == 0.0:
        _sgn = 1.0
    n_out_sym = (_sgn * nc) / (ca.norm_2(nc) + 1e-12)
    n_in_local = -n_out_sym
    return ca.DM(np.asarray(mat_np, float)) @ n_in_local


def _friction_cone_verts(mu: float) -> np.ndarray:
    """5-vertex linearized Coulomb cone in the LOCAL contact frame [n_in|t1|t2]
    (origin + 4 unit-normal-force edges at ±mu tangential). Shared by the
    embedded wrench-cone LP and the GWS primitive-wrench matrix so both use
    the exact same cone geometry. Point-contact-with-friction (PCwF): pure
    force generators, no torsional/spin component about the contact normal —
    see _soft_finger_torque_ca / build_W_ca's mu_t argument for the
    soft-finger extension (scripts/3D_minimum_NCF_soft.py)."""
    return np.array([[0.0, 0.0, 0.0],
                     [1.0, 0.0, -mu],
                     [1.0, -mu, 0.0],
                     [1.0,  mu, 0.0],
                     [1.0, 0.0,  mu]])


def _soft_finger_torque_ca(n_O, mu_t: float):
    """(tau_plus, tau_minus): the ± free-vector spin-moment terms ADDED to the
    p×f torque of the 2 soft-finger vertices (build_W_ca's f_soft columns),
    each carrying normal force ncf=1 (unit squeeze, matching the rest of
    _friction_cone_verts' unit-force convention) — so the moment is exactly
    mu_t*n_O, no ncf factor needed here.

    n_O : (3,) contact INWARD normal, expressed in the OBJECT BODY frame
        (R_ow @ R_param[:,0] — R_param's column 0 is n_in by the
        [n_in|t1|t2] convention built by _build_contact_frame_3d /
        _symbolic_contact_frame_ca). A free vector: does NOT depend on
        contact position (no lever arm), matching
        scripts/3D_minimum_NCF_soft.py's t5_b/t6_b (mu_t*ncf*n_b term, added
        directly to the cross-product torque, not multiplied through it).
    """
    tau = mu_t * n_O
    return tau, -tau


def build_W_ca(p1, p2, R1_param, R2_param, obj_center_np, obj_R_np, mu,
               mu_t: float = 0.0, extra_contacts=None):
    """Primitive wrench matrix W (6 x n*s), n contacts x s cone edges each.

    extra_contacts: optional [(p, R_param), ...] appended AFTER (p1,R1),(p2,R2).
    None (default) reproduces the 2-contact matrix byte-for-byte -- the n=2 path
    is the measured one (12/15 lifts) and must not move. A third contact off the
    grasp axis is the point of passing it: a 2-contact W is structurally
    rank-5-of-6 (no moment arm about the line through the two contacts), which is
    what project_grasp_axis_torque exists to paper over and what the soft-finger
    columns tried, and measurably failed, to fix by adding near-duplicate columns.
    s=5 (PCwF) when mu_t<=0 (default — unchanged from before); s=7 (soft
    finger) when mu_t>0, adding 2 pure-normal-force+spin-moment generators per
    contact (scripts/3D_minimum_NCF_soft.py's soft-finger linearization) so W
    can express torque about the grasp axis — a PCwF W is structurally rank-
    deficient there (2-contact pinch, zero moment arm on that axis for any
    tangential force), a finding from the rubiks-cube GWS-only test session.

    Column j of contact i is the object-body-frame wrench [τ;f] produced by a
    unit-normal-force generator along that contact's j-th generator (cone edge
    or soft-finger torsion vertex). Shares _friction_cone_verts/R_param/
    obj-frame convention with _embed_wrench_cone_ca's per-corner _wrench_sum
    so the GWS block and the existing wrench-feasibility LP agree on what a
    "primitive wrench" is.

    Returns a (6, n*(5 or 7)) CasADi MX expression, symbolic in the contact
    positions (and in the R_params if those are themselves expressions, e.g.
    under symbolic_normals).
    """
    verts_c = _friction_cone_verts(mu)
    obj_c   = ca.DM(obj_center_np)
    R_ow    = ca.DM(obj_R_np.T)   # world → object body-frame rotation

    def _col(p, f, extra_tau=None):
        p_O = R_ow @ (p - obj_c)
        f_O = R_ow @ f
        tau = ca.cross(p_O, f_O)
        if extra_tau is not None:
            tau = tau + extra_tau
        return ca.vertcat(tau, f_O)

    cols = []
    _contacts = [(p1, R1_param), (p2, R2_param)]
    if extra_contacts:
        _contacts.extend(extra_contacts)
    for p, R_param in _contacts:
        forces = [R_param @ ca.DM(v) for v in verts_c]
        cols.extend(_col(p, f) for f in forces)
        if mu_t > 0.0:
            n_O = R_ow @ R_param[:, 0]   # inward normal, object body frame
            tau_plus, tau_minus = _soft_finger_torque_ca(n_O, mu_t)
            f_soft = R_param @ ca.DM([1.0, 0.0, 0.0])   # unit normal force, both vertices
            cols.append(_col(p, f_soft, extra_tau=tau_plus))
            cols.append(_col(p, f_soft, extra_tau=tau_minus))
    return ca.horzcat(*cols)


def _embed_gws_ca(opti, W, alpha_reg: float = 0.0):
    """Add the min-weight (FRoGGeR) LP as NLP decision variables/constraints.

        max_{alpha,beta}  beta - alpha_reg * ||alpha||^2
        s.t.  W @ alpha == 0
              sum(alpha) == 1
              alpha >= beta * 1

    Single-level: alpha/beta are opti.variable()s in the SAME Opti problem as
    q/p1/p2/W's own p1/p2-dependence, not a nested LP differentiated through —
    IPOPT's KKT system couples everything and gives exact gradients. At
    convergence beta == the min-weight metric's optimal value and alpha is the
    closure witness (see module-level GWS brief).

    alpha is left free (no alpha >= 0): if the current contact geometry is not
    yet in force closure, beta < 0 with some alpha_j < 0 is the CORRECT value,
    and must stay smoothly climbable rather than becoming infeasible.

    alpha_reg (proximal/Tikhonov term, 0.0 = off, byte-identical to the plain
    LP embedding): adds -alpha_reg*||alpha||^2, making the embedded problem a
    strictly convex QP in alpha instead of an LP.

    MEASURED, and it is NOT the fix for the dual indeterminacy -- keep it off
    unless you have a reason. The hypothesis it was added to test was that at
    a symmetric antipodal pinch the LP's optimal set is a FACE (several alpha_j
    tying at beta), making the primal degenerate. That hypothesis is FALSE for
    this geometry: solving the LP under random 1e-9 cost tilts returns
    alpha = 0.1*ones to machine precision every time, i.e. the optimum is the
    single VERTEX where all 10 components tie at beta = 1/n_cols exactly. The
    primal is already unique, so there is nothing for a proximal term to
    disambiguate -- measured on 017_orange, eps=1e-6 shifted beta by -1.8%
    (0.09676 -> 0.09505) and eps=1e-4 pushed the solve to
    Maximum_Iterations_Exceeded while biasing an already-unique alpha.

    The real mechanism is DUAL, and structural: W is 6 x (n*5) with rank 5,
    not 6, for a 2-contact pinch (measured singular values
    4.0/4.0/2.83/0.144/0.144/0.0). Its left-null direction is exactly torque
    about the GRASP AXIS. Because W^T u = 0 for that u, the multiplier lambda
    of the 6-row equality W@alpha == 0 is determined only up to lambda + c*u
    for any scalar c -- non-unique along exactly one direction, by geometry,
    not by conditioning. No alpha-side regularization touches that; the fixes
    that would are (a) a third non-collinear contact, giving W a real moment
    arm about every axis (see build_W_ca's extra_contacts and cfg.n_contacts),
    or (b) dropping the dependent row / regularizing on the LAMBDA side.
    See also build_W_ca's "structurally rank-5-of-6" note and
    for_gws_recommender's soft-finger discussion, which record the same rank
    deficiency from the primal side.

    Kept (default off) because it is a cheap, already-wired knob for probing
    the alpha side, and alpha_reg is returned as a separate cost term rather
    than folded in silently so a caller comparing beta across settings sees
    both contributions.

    Returns (alpha, beta, cost_alpha_reg) — cost_alpha_reg is alpha_reg *
    sumsqr(alpha) (a ca.DM(0.0) sentinel when alpha_reg == 0.0), for the
    caller to add into its own minimize() call alongside -w_gws*beta.
    """
    n_cols = int(W.shape[1])
    alpha  = opti.variable(n_cols)
    beta   = opti.variable()
    opti.subject_to(W @ alpha == ca.DM.zeros(6, 1))
    opti.subject_to(ca.sum1(alpha) == 1.0)
    opti.subject_to(alpha - beta * ca.DM.ones(n_cols, 1) >= 0)
    cost_alpha_reg = (alpha_reg * ca.sumsqr(alpha)) if alpha_reg > 0.0 else ca.DM(0.0)
    return alpha, beta, cost_alpha_reg


# ca.det/ca.chol have no AD rule for MX (CasADi raises "'eval' not defined for
# class Determinant" from IPOPT's gradient pass — confirmed empirically, not
# just undocumented). SX supports both with proper derivative rules, so the
# logdet is built once as a small SX-graph Function and called from the MX
# problem — CasADi differentiates through an embedded Function call fine, this
# only avoids doing linear algebra directly in MX. M is SPD by construction
# (W W^T + delta*I, delta>0), so a Cholesky-based logdet (2*sum(log(diag(L))))
# is valid and only needs the diagonal of L, not a full logm/eig.
_x_sx = ca.SX.sym('_gws_logdet_x', 6, 6)
_logdet_sx_fn = ca.Function(
    'gws_span_logdet6',
    [_x_sx],
    [2 * ca.sum1(ca.log(ca.diag(ca.chol(_x_sx))))])


def _gws_span_logdet_ca(W, delta: float):
    """logdet(W W^T + delta*I) — smooth barrier pushing all 6 singular values
    of W up together (full-rank / well-spanned wrench hull), unlike a hard
    rank or sigma_min constraint. delta keeps it defined when rank(W) < 6
    (e.g. collinear contacts, coplanar normals — see module-level GWS brief's
    "critical caveat").
    """
    M = W @ W.T + delta * ca.DM.eye(6)
    return _logdet_sx_fn(M)


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

    verts_c = _friction_cone_verts(mu)
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
class CostWeights:
    """Dimensionless NLP cost-term priorities (each term normalized to ≈1 at its
    reference: d_ref=5mm, g_ref=task_load_N, y_ref=g_ref)."""
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


@dataclass
class SlackTolerances:
    """Wrench-cone slack tolerances (see _embed_wrench_cone_ca / min_gamma_for_accel_lp
    slack_penalty mode). A converged NLP no longer certifies exact wrench resistance by
    itself — wrench_ok additionally requires max_slack_norm below these thresholds.
    Absolute units: N for force rows, N*m for torque rows."""
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


@dataclass
class AlignmentConfig:
    """Grasp-axis and fingerpad orientation cost terms."""
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


@dataclass
class GWSConfig:
    """Grasp-Wrench-Space (GWS) min-weight quality objective (FRoGGeR-style relaxation
    of the Ferrari-Canny epsilon metric — see the module-level GWS implementation
    brief). Embeds the min-weight LP (alpha, beta) as NLP variables alongside q/p1/p2
    so IPOPT can steer contact placement toward deep, well-conditioned force closure,
    not just measure it post-hoc. ADDITIVE to AlignmentConfig.w_align, not a
    replacement — both default to 0.0 (off); existing callers (kinova_leap_pick_place.py
    etc.) that set w_align are unaffected until they explicitly opt into w_gws/w_span.
        -w_gws  * beta                              (maximize the min-weight margin)
        -w_span * logdet(W W^T + gws_span_delta*I)   (push toward a full-rank, well-
                                                        spanned wrench hull; beta alone
                                                        can be positive but nearly
                                                        meaningless if W is rank-deficient
                                                        — see the brief's "critical caveat")
    0.0 = off for either term."""
    w_gws:              float = 0.0
    w_span:              float = 0.0
    gws_span_delta:      float = 1e-6   # keeps logdet finite when rank(W) < 6

    # Soft-finger contact model for W (build_W_ca / GWS only — NOT the task-specific
    # wrench_constraint LP, which stays PCwF). False (default): point-contact-with-
    # friction, 5 generators/contact — matches all prior behavior exactly. True: adds
    # 2 torsional-spin generators/contact (7 total), letting a 2-contact pinch's W
    # express torque about the grasp axis, which PCwF is STRUCTURALLY rank-deficient
    # in (a flat-face antipodal grasp measured rank(W)=5/6 with PCwF this session —
    # the two soft-finger generators are exactly what's missing). mu_t is read from
    # model geom_friction (index 1, torsional) via _contact_friction — real per-object/
    # per-fingertip values already authored in the MJCF, combined the way MuJoCo's own
    # contact solver does it (elementwise max), not a hand-picked constant.
    gws_soft_finger:    bool = False

    # Proximal (Tikhonov) regularization on the min-weight LP's alpha: adds
    # gws_alpha_reg * ||alpha||^2 to the cost, turning
    #     max_{alpha,beta} beta  s.t. W@alpha==0, sum(alpha)==1, alpha>=beta*1
    # from an LP (whose optimal face is a FACE, not a vertex, whenever
    # multiple alpha_j tie at beta -- e.g. any symmetric antipodal pinch with
    # a 5-vertex polyhedral cone) into a strictly convex QP. Strictly convex
    # -> unique primal alpha* -> unique KKT multipliers, which is the direct
    # fix for IPOPT dual non-uniqueness/non-convergence traced to this block
    # (see the dual-indeterminacy investigation notes). 0.0 = off (byte-
    # identical to the pre-existing LP embedding). ~1e-6 is the recommended
    # starting point: small enough that its bias on beta is second-order and
    # independently quantifiable by comparing beta at reg=0 vs reg=1e-6 on a
    # solve that DOES converge with reg=0 (e.g. an asymmetric seed).
    gws_alpha_reg:      float = 0.0

    # Ablation switch for the non-C2 term hunt: passes smooth_blend=True to
    # every _symbolic_contact_frame_ca call feeding the GWS contact frame(s),
    # replacing that function's ca.fabs(n[0]) blend weight (C0, not C1, at
    # n[0]==0) with a C-infinity sqrt(n[0]**2+eps) surrogate. False (default)
    # is byte-identical to prior behavior.
    #
    # MEASURED: this is NOT where the trouble is, and it changes nothing on
    # either object tested. On 036_wood_block the result is bit-identical to
    # baseline (beta=0.09915393739443955, 167 iters, same contact gaps) even
    # though the flag verifiably reaches all 8 frame builds -- because the
    # solved normals sit at |n[0]| = 0.14/0.22, which is 68-76 tanh widths
    # from the kink, where the blend is saturated flat (|dalpha/dn0| = 0.00 to
    # machine precision).
    #
    # What the same measurement DID surface: the blend's tanh(w/0.01) has
    # width 0.01, and 017_orange's solved normals land at |n[0]| = 0.9073 and
    # 0.9395 -- 0.7 and 4.0 widths from the 0.9 switch point, where
    # |dalpha/dn0| ~ 31 and |d2alpha/dn0^2| ~ 3.8e3. So the orange's contact
    # TANGENT BASIS spins violently under infinitesimal normal change, and
    # that basis multiplies straight into W. It is a smooth term, which is
    # exactly why smoothing it further cannot help. Note also that beta is
    # EXACTLY invariant (0.100000000000 across 0-90 deg) to rotating the
    # tangent basis about n -- a pure gauge freedom -- so this ill-conditioned
    # dial moves the constraint Jacobian while leaving the objective flat.
    # Widening the transition (or gauge-fixing the tangent basis) is the
    # lead worth pulling here, not smoothing the abs.
    gws_smooth_frame:   bool = False


@dataclass
class UVAtlasConfig:
    """Mesh contact parameterization. use_uv_atlas_contact=False (default) —
    _mesh_tangent_contact_ca: 2-DOF offset in a Euclidean tangent plane at the seed,
    bounded by a fixed +/-t_bound Euclidean box unrelated to the object's real
    geometry. True — _mesh_uv_local_contact_ca: 2-DOF offset in a plane fit to the
    seed's actual local mesh neighborhood (a fixed-size ring of triangles within a
    large, precomputed UV-atlas CHART — see grasp_control.object_uv_atlas module
    docstring), bounded by that neighborhood's real boundary polygon. Re-centered
    between Picard relinearization stages (n_normal_relinearize) the same way the
    existing loop already re-centers the frozen surface normal for curved analytic
    shapes — lets a multi-stage solve walk across a whole chart via a sequence of
    small, cheap local linearizations, rather than being permanently confined to one
    Euclidean neighborhood of the seed for the whole solve. Requires xatlas (pip
    install xatlas); falls back to the tangent-plane scheme with a log warning if
    unavailable."""
    use_uv_atlas_contact: bool = False
    uv_atlas_rings:       int  = 2      # face-adjacency hops per local neighborhood
    uv_atlas_min_chart_frac: float = _object_uv_atlas.DEFAULT_MIN_CHART_AREA_FRAC \
                                     if _object_uv_atlas is not None else 0.01

    # Third mesh-contact ablation arm: _mesh_quadratic_contact_ca. Independent
    # of use_uv_atlas_contact (checked first in _run_stage if both are somehow
    # set) — places the contact on a local paraboloid fit from the SDF's own
    # gradient+Hessian at the seed instead of projecting a tangent-plane point
    # through object_sdf.surface_project. No chart/xatlas machinery involved,
    # so this does NOT interact with the chart-pair antipodal seeding gated on
    # use_uv_atlas_contact elsewhere (_seed_pair, last_chart_rank_table).
    use_quadratic_contact: bool = False
    quadratic_t_bound_max:  float = 0.10    # metres, cap even where the surface stays flat
    quadratic_sdf_err_tol:  float = 4e-3    # metres, max surrogate-vs-true-SDF gap per axis
    # Fit the local patch's curvature to the VISUAL MESH vertices around the
    # seed instead of to the SDF Hessian. The SDF's second derivative is a
    # global quantity and reports curvature belonging to nearby features rather
    # than to the patch (measured: kappa=-12.3 mid-face on a flat side of
    # 036_wood_block, from the corner 50mm away), which collapses the trust
    # region to a few mm on large flat faces. See _mesh_local_surface_fit_np.
    quadratic_mesh_fit:          bool  = False
    quadratic_mesh_fit_radius:   float = 0.04   # m, tangent sampling radius
    # Minimum RMS-residual improvement of the quadratic over a plane before any
    # curvature is accepted; below this the patch is treated as planar
    # (kappa=0). Measured separation is wide -- 0.91 on 017_orange, 0.15-0.34
    # on the wood block's flat face -- so this is a real model comparison, not
    # a tuned threshold.
    quadratic_mesh_fit_gain_min: float = 0.5
    # DIAGNOSTIC: pull contacts toward the object's COM plane. Blunt and
    # object-specific (wrong for a mug rim or bottle neck) -- exists to test
    # whether a mid-height grasp makes 036_wood_block liftable at all. Needs a
    # LARGE weight to register against the IK term. See the _run_stage comment.
    w_contact_height:            float = 0.0
    # If a stage's solved (t1,t2) sits within this fraction of its own
    # per-axis bound, the Picard loop treats it as PINNED (trust region ran
    # out, not a converged interior optimum) and keeps relinearizing even if
    # the usual position/normal-mismatch convergence check would pass — see
    # solve()'s relinearization loop.
    quadratic_pin_frac:     float = 0.9

    # Edge-avoidance cost for the quadratic-contact path (use_quadratic_contact
    # only): penalizes the paraboloid's own height function h(t1,t2) = -(kappa0*t1^2
    # + kappa1*t2^2)/(2*grad_norm) at the SOLVED (t1,t2) -- kappa0/kappa1/grad_norm
    # are the seed's frozen curvature (numpy floats, fixed for this Picard stage,
    # from _mesh_quadratic_contact_ca's `frame` dict), but t1/t2 are the live NLP
    # variables, so h(t_var) IS differentiable in t_var even though its coefficients
    # aren't. On a flat seed (kappa0=kappa1=0) this is identically zero -- free
    # movement in any tangential direction. Near an edge (one or both curvatures
    # large), moving toward the high-curvature direction costs quadratically more
    # than moving along the flat direction, so the optimizer is steered toward the
    # face and away from the edge FROM WITHIN the same solve -- unlike a penalty on
    # kappa0/kappa1 alone (which are constants at a fixed seed and only bias which
    # stage/seed looks best after the fact, not what a single stage's optimizer
    # does). 0.0 = off.
    # Edge-MARGIN penalty (replaces the earlier curvature-based w_edge_curvature;
    # see the _run_stage comment for why curvature was the wrong signal). Penalizes
    # a contact that comes within edge_margin_m of a trust-region bound that was set
    # by a MEASURED SDF divergence -- i.e. a real surface boundary -- and ignores
    # bounds sitting at quadratic_t_bound_max, which only mean "flat as far as the
    # search looked".
    w_edge_margin:          float = 0.0
    # How much surface to keep in reserve between the contact and a measured edge.
    # Deliberately small relative to a face: the trust region on a flat face is
    # meant to span most of that face, so this only trims the last few mm.
    # NAMED DISTINCTLY from edge_margin_m (the BOX keep-out band, a hard
    # constraint in _sym_geom_surface_con) -- different mechanism, different
    # units of meaning, and one shadowing the other silently would be a
    # nasty bug.
    edge_margin_sdf_m:      float = 0.005


@dataclass
class CollisionConfig:
    """Object/floor/arm clearance and collision-constraint configuration."""
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


@dataclass
class RegularizationConfig:
    """Posture-regularization target for the NLP's tie-breaker cost term."""
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


@dataclass
class MultiStartConfig:
    """Multi-start seeding / Picard-relinearization / restart-diversification knobs."""
    # extra Picard iterations (n total solves);
    # n_normal_relinearize can be zero for box, search stays within the seed faces
    # n_normal_relinearize = 2 for curved surfaces, since normals genuinely rotate there
    n_seeds:             int  = 5    # seed pairs per multi-start (each from _seed_pair)
    n_normal_relinearize: int  = 1
    verbose_profile:      bool = False

    # DIRECTIONAL fingertip radius for the IK offset, refrozen each Picard stage
    # from the previous stage's q and that stage's contact normal (see
    # GraspPlanner3D._tip_support_along). The default isotropic r_tip is
    # max||V - site|| over every direction -- a bounding sphere around an
    # elongated pad -- so the IK target contact + r_tip*n_out sits several mm
    # proud of the surface: measured 5.9mm (thumb) / 4.8mm (index) of pure
    # geometric slack on 036_wood_block against an IK residual of 0.02-0.05mm,
    # i.e. essentially the ENTIRE observed pre-squeeze gap. Using the support
    # distance along the actual contact normal removes that slack without making
    # the (non-smooth) support function a symbolic function of q -- it is a
    # per-stage constant, exactly like the frozen normal itself.
    # False = prior behavior (isotropic r_tip everywhere).
    directional_r_tip:    bool = False

    # Extra safety margin (m) ADDED to the directional radius. The support
    # distance targets a nominally ZERO gap, but the pad-vs-object contact is
    # mesh-vs-mesh and the stage's normal is frozen, so aiming at exactly zero
    # risks landing on the PENETRATING side -- the asymmetric failure
    # _tip_radius's own comment warns about (penetration is unrecoverable for
    # the squeeze; a small gap is not). A ~1mm cushion keeps the target on the
    # safe side while still closing the bulk of the 5-6mm isotropic overshoot.
    directional_r_tip_margin_m: float = 0.001

    # Antipodal-march jitter (deg): _seed_pair marches from contact 1 along its inward
    # normal rotated by up to +/- this angle to find contact 2. Large values (the old 45)
    # let the march exit through an ADJACENT box face ~half the time; since each contact
    # is then HARD-PINNED to its seed face, the solve returns a non-opposite grasp that
    # w_align cannot rescue. A smaller angle keeps contact 2 on the opposing face while
    # still allowing some obliqueness.
    seed_march_jitter_deg: float = 15.0

    # Mesh-object seed curvature gate: reject a _seed_pair draw if either
    # contact's largest-magnitude principal curvature (SDF Hessian restricted
    # to the tangent plane, see _mesh_surface_kappa_max_np) exceeds this —
    # i.e. resample rather than accept a seed sitting at/near an edge or
    # corner. Measured on 036_wood_block: genuinely flat face interior ~0-3,
    # near-edge ~20-190. 40 rejects clear edges/corners while keeping mildly
    # curved (rounded-edge) regions available. 0 disables the check (accept
    # any surface point, the pre-refactor behaviour). No-op for analytic
    # primitives (box/sphere/cylinder), which have exact closed-form surfaces.
    # Over-generation factor for DLS-IK seed ranking. 1 = off (take the first
    # n_seeds that pass the geometric gates, the historical behavior). k > 1
    # generates k*n_seeds candidates and keeps the n_seeds with the smallest
    # damped-least-squares fingertip residual, i.e. the ones the ARM can
    # actually reach. The geometric gates (_reachable_contact, seed_kappa_max_
    # reject) screen the SURFACE; this screens the KINEMATICS, and measurement
    # puts the dominant contact error there -- planned contacts sit sub-mm from
    # the surface while fingertip geoms stop 3-9mm short. Costs one DLS solve
    # per candidate (milliseconds) against a multi-second NLP per seed.
    seed_dls_rank_pool: int = 1
    seed_kappa_max_reject: float = 150.0

    # Vertical room the SEED gate requires under a contact, in metres. None
    # (default) = cfg.r_thumb/r_index, the fingertip's isotropic bounding-sphere
    # radius. A float overrides it with a flat floor, which is the less
    # conservative reading: the bounding sphere is set by the tip's LONG axis
    # (16.7mm, pointing away from the contact), while the pad only extends
    # 10.8mm along the direction that actually touches the object. The seed gate
    # only has to pick a plausible STARTING point -- the NLP's ground-collision
    # constraint (ground_clearance_m) is what keeps the final pose off the
    # table -- so an over-tight seed floor rejects reachable grasps on thin
    # objects for no safety gain (009_gelatin_box: 120/120 seeds rejected at the
    # bounding-sphere floor).
    seed_ground_clearance_m: float | None = None

    # True (default) -- seed contacts are placed on the object's OUTER surface by
    # taking the first and last zero crossings of the seed ray, instead of the
    # nearest-surface projection / first-positive-SDF march that preceded it.
    # Those older rules are correct only for a solid object: a cup's cavity reads
    # as OUTSIDE (positive SDF), so both land on the INNER wall -- measured 6 of
    # 14 seed contacts inside 065-a_cups. No effect on solid objects, where the
    # ray has exactly two crossings and first/last are what the old rules already
    # returned. Set False to allow inner-surface seeds; note the rest of the
    # pipeline does not yet support internal (expanding) grasps -- the squeeze
    # phase drives fingers in the closing direction, which unloads an inner-wall
    # contact rather than pressing it.
    seed_prefer_outer_surface: bool = True

    # q_ref (arm-pose) restart perturbation — matches ablate_ik.py's RESTART_SIGMA
    # pattern: seed 0 uses q_ref exactly as given (the operator's actual retargeted
    # pose), each subsequent seed perturbs it by Gaussian noise (arm/hand split, since
    # the arm's 7 DOF and the hand's redundant DOF have very different sensible scales)
    # before running that seed's DLS-warm-start + NLP solve. This is a STAND-IN for
    # eventually sampling from a real distribution around the operator's pose; for now
    # it only exists to give MultiStartGraspPlanner3D the same "diversify the STARTING
    # arm configuration across restarts" mechanism ConstrainedIKSolver's sqp-ms4 backend
    # already has (ablate_ik.py's RESTART_SIGMA=(0.25, 0.35) rad) — GraspPlanner3D
    # previously only diversified the CONTACT seed, never the arm pose, across its
    # n_seeds budget. 0.0 = off (seed 0 behavior only, i.e. unchanged from before).
    qref_restart_sigma_arm:  float = 0.0
    qref_restart_sigma_hand: float = 0.0


@dataclass
class SolverBackendConfig:
    """NLP backend selection (SQP+OSQP vs IPOPT) and contact-frame differentiability."""
    # use_slsqp=True  → SQP+OSQP (default; linear face-pin + wrench constraints exact)
    # use_slsqp=False → IPOPT (interior-point; use for non-box geometries)
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
    # Same idea for a MESH contact under use_quadratic_contact: build the
    # wrench/GWS contact frame from the paraboloid's OWN analytic normal
    # (_quadratic_inward_normal_ca) instead of freezing the seed's. The
    # surrogate already carries the curvature that says how the normal tilts
    # across the patch, so freezing the frame contradicts the model used for
    # position. Separate flag from symbolic_normals because it applies to a
    # different geometry class and changes NLP conditioning independently.
    quadratic_symbolic_normals: bool = False
    # Per-consumer ablation switches for quadratic_symbolic_normals. The master
    # flag turns the symbolic normal ON; each of these can turn it back OFF for
    # ONE consumer, so the four consumers (wrench frame, IK target, w_align,
    # orient_weight) can be attributed independently. None = follow the master
    # flag. These exist because all four went in together and regressed the
    # tabletop benchmark 6/8 -> 3/8 full cycles; the master flag alone cannot
    # say which one is responsible.
    quad_sym_normals_frame:  bool | None = None   # wrench/GWS contact frame
    quad_sym_normals_iktgt:  bool | None = None   # IK target p + r*n(t)
    quad_sym_normals_align:  bool | None = None   # w_align grasp-axis cost
    quad_sym_normals_orient: bool | None = None   # orient_weight pad-axis cost
    smooth_sdf: bool  = True
    slsqp_alpha: float = 400.0  # smooth SDF alpha (collision avoidance SDF only)


@dataclass
class TaskWrenchConfig:
    """Mass-scaled task-wrench budget for the embedded wrench-cone LP and the
    post-solve gamma certificate."""
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


@dataclass
class GeometryNames:
    """Model geometry/site names (must match scene XML)."""
    obj_geom:    str = 'obj_red_box_geom'
    obj_body:    str = 'obj_red_box'
    thumb_site:  str = 'leap_th_ds_tip'
    index_site:  str = 'leap_if_ds_tip'
    # Third load-bearing contact (cfg.n_contacts >= 3). The middle finger already had a
    # GEOM here (middle_geom, used for verify()'s diagnostic gaps and for the collision
    # tier) but no SITE -- and a site is what the IK cost, the DLS reachability ranking
    # and the FK callbacks all target. Resolved only when n_contacts >= 3.
    middle_site: str = 'leap_mf_ds_tip'
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


# Flat legacy field name -> (nested group attr name, field name on that group).
# Built once from the group dataclasses' own field lists (not hand-maintained) so it
# can never drift out of sync with them.
def _build_flat_field_map() -> dict[str, tuple[str, str]]:
    import dataclasses as _dc
    groups = {
        'cost': CostWeights, 'slack': SlackTolerances, 'align': AlignmentConfig,
        'gws': GWSConfig, 'uv_atlas': UVAtlasConfig, 'collision': CollisionConfig,
        'reg': RegularizationConfig, 'multistart': MultiStartConfig,
        'backend': SolverBackendConfig, 'task_wrench': TaskWrenchConfig,
        'names': GeometryNames,
    }
    flat_map: dict[str, tuple[str, str]] = {}
    for group_attr, group_cls in groups.items():
        for f in _dc.fields(group_cls):
            flat_map[f.name] = (group_attr, f.name)
    return flat_map


_GRASP_CFG_FLAT_FIELDS = _build_flat_field_map()
_GRASP_CFG_GROUPS = {v[0] for v in _GRASP_CFG_FLAT_FIELDS.values()}


class GraspConfig3D:
    """Configuration for the 3D grasp planner (Kinova Gen3 + LEAP hand).

    Grouped into nested sub-dataclasses by concern (cost, slack, alignment, GWS,
    UV-atlas contacts, collision, regularization, multi-start, solver backend, task
    wrench, geometry names) — see each group's own docstring. Two equivalent
    construction styles:

        # nested (new code)
        GraspConfig3D(gws=GWSConfig(w_gws=5.0, w_span=1.0))

        # flat (legacy — every existing call site in the repo uses this; still
        # fully supported, dispatched to the right group automatically)
        GraspConfig3D(w_gws=5.0, w_span=1.0)

    Not a @dataclass itself (a custom __init__ is required to accept the flat-kwarg
    legacy form), but every nested group IS a plain @dataclass, so
    dataclasses.fields()/replace() etc. work fine on cfg.cost, cfg.gws, and so on.
    """

    # Explicit slots for every group, purely so IDEs/static analysis see the
    # attributes; __init__ is what actually populates them.
    cost: CostWeights
    slack: SlackTolerances
    align: AlignmentConfig
    gws: GWSConfig
    uv_atlas: UVAtlasConfig
    collision: CollisionConfig
    reg: RegularizationConfig
    multistart: MultiStartConfig
    backend: SolverBackendConfig
    task_wrench: TaskWrenchConfig
    names: GeometryNames

    # Fields that were never grouped (kept top-level: solve()-wide flags/knobs that
    # don't belong to any one concern, plus the four per-finger radii GraspPlanner3D
    # itself measures and writes back in __init__).
    joint_limits:      bool  = True
    wrench_constraint: bool  = True
    max_iter:          int   = 50
    fixed_contacts:    bool  = False
    # Number of LOAD-BEARING contacts the NLP solves for. 2 = the antipodal pinch
    # (thumb+index) every measurement in this repo was taken on. 3+ is the tripod
    # work-in-progress: a third contact OFF the grasp axis supplies the moment arm a
    # 2-contact pinch structurally lacks (its W is rank-5-of-6, which is why
    # project_grasp_axis_torque exists and why the soft-finger columns were tried).
    # Introduced as an EXPLICIT field rather than inferred from the seed count so the
    # n=2 path's behaviour cannot change as a side effect of how seeding happens to
    # be configured.
    n_contacts:        int   = 2
    # Third-contact seeding strategy: 'fan' (original, ablation baseline),
    # 'tangent' (palm-pitch walked along the index's surface) or 'kinematic'
    # (surface under the middle fingertip at the pinch pose). See
    # _seed_third_contact. Only consulted when n_contacts >= 3.
    c3_seed_strategy:  str   = 'patch_offset'
    # Give contact 3 its OWN quadratic patch instead of reconstructing it inside
    # contact 2's. FALSE (shared) is the MEASURED-BETTER default; True is kept
    # only so the comparison can be re-run.
    #
    # The argument FOR an own patch was that the index patch is too small to hold
    # two fingertips: 17-24mm half-extent on rounded objects (median 19.7mm)
    # against a 45.4mm finger pitch. That argument is WRONG, in two ways.
    #
    # 1. The 45.4mm "finger pitch" is the rigid base-mount spacing measured at
    #    qpos0. Fingers curl INDEPENDENTLY, so their tips need not sit one
    #    base-pitch apart on the surface -- the shared patch's own solves place
    #    them 36-90mm apart, which is what they actually want.
    # 2. Measured head-to-head, seed 0, three objects, middle-fingertip gap to
    #    its assigned contact (pad radius removed, so ~0 = the finger reaches it):
    #        shared: -0.1 / -0.8 / +0.7 mm   <- converges on all three
    #        own:    +5.8 / -0.5 / +56.7 mm  <- 036_wood_block misses by 57mm
    #    and on the block the own-patch cost blew up 0.55 -> 37.25 with beta
    #    going NEGATIVE. That reproduces the 183mm failure this file's
    #    shared-patch comment already recorded from an earlier attempt.
    #
    # Sharing guarantees the two contacts are adjacent and mutually reachable.
    # An independent patch can be fitted anywhere the seeder points, including a
    # face the hand would have to re-approach entirely.
    c3_own_patch:      bool  = False
    # Offset distance (m) for c3_seed_strategy='patch_offset': how far from the
    # index contact to place the third contact, measured in the index patch's own
    # (t0,t1) coordinates and clamped to that patch's measured bounds.
    #
    # 15mm, MEASURED. Must stay strictly INSIDE the patch, not at its edge: the
    # half-extent is ~20mm on rounded objects, so a 30mm request CLAMPS to the
    # boundary and pins contact 3 against it. Measured on 017_orange that gives
    # beta = -0.230 (the only negative beta in the sweep) because a pinned contact
    # contributes a near-duplicate wrench column. On 036_wood_block, whose patch is
    # +/-97mm and never clamps, 30mm is harmless -- exactly the pattern clamping
    # predicts. Per-object, seed 0, orange/lemon/block:
    #     15mm  cost 4.15/4.98/0.34  beta .039/.033/.041  all 3 fingers within 1.5mm
    #     30mm  cost 7.31/6.70/0.56  beta -.230/.043/.043
    c3_patch_offset_m: float = 0.015
    r_thumb:  float | None = None
    r_index:  float | None = None
    r_middle: float | None = None
    r_ring:   float | None = None

    def __init__(self, **kwargs):
        # Pull out any nested-group kwargs first (e.g. gws=GWSConfig(...)).
        for group_attr, group_cls in (
            ('cost', CostWeights), ('slack', SlackTolerances),
            ('align', AlignmentConfig), ('gws', GWSConfig),
            ('uv_atlas', UVAtlasConfig), ('collision', CollisionConfig),
            ('reg', RegularizationConfig), ('multistart', MultiStartConfig),
            ('backend', SolverBackendConfig), ('task_wrench', TaskWrenchConfig),
            ('names', GeometryNames),
        ):
            setattr(self, group_attr, kwargs.pop(group_attr, None) or group_cls())

        # Top-level (never grouped) fields.
        self.joint_limits      = kwargs.pop('joint_limits', True)
        self.wrench_constraint = kwargs.pop('wrench_constraint', True)
        self.max_iter           = kwargs.pop('max_iter', 50)
        self.fixed_contacts     = kwargs.pop('fixed_contacts', False)
        self.n_contacts         = kwargs.pop('n_contacts', 2)
        self.c3_seed_strategy   = kwargs.pop('c3_seed_strategy', 'patch_offset')
        self.c3_own_patch       = kwargs.pop('c3_own_patch', False)
        self.c3_patch_offset_m  = kwargs.pop('c3_patch_offset_m', 0.015)
        self.r_thumb  = kwargs.pop('r_thumb', None)
        self.r_index  = kwargs.pop('r_index', None)
        self.r_middle = kwargs.pop('r_middle', None)
        self.r_ring   = kwargs.pop('r_ring', None)

        # Everything left is either a flat legacy field name (dispatch into its
        # group) or an error.
        unknown = []
        for key, value in kwargs.items():
            target = _GRASP_CFG_FLAT_FIELDS.get(key)
            if target is None:
                unknown.append(key)
                continue
            group_attr, field_name = target
            setattr(getattr(self, group_attr), field_name, value)
        if unknown:
            raise TypeError(
                f"GraspConfig3D() got unexpected keyword argument(s): {unknown!r}")

    def __repr__(self):
        groups = ', '.join(f'{g}={getattr(self, g)!r}' for g in sorted(_GRASP_CFG_GROUPS))
        top = ', '.join(f'{k}={getattr(self, k)!r}' for k in
                        ('joint_limits', 'wrench_constraint', 'max_iter', 'fixed_contacts',
                         'n_contacts', 'c3_seed_strategy', 'c3_own_patch', 'c3_patch_offset_m',
                         'r_thumb', 'r_index', 'r_middle', 'r_ring'))
        return f'GraspConfig3D({groups}, {top})'

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails, i.e. `name` isn't an
        # instance attribute already set in __init__ — so this is exactly the flat
        # legacy READ path (cfg.w_gws -> cfg.gws.w_gws), never shadowing a group.
        target = _GRASP_CFG_FLAT_FIELDS.get(name)
        if target is None:
            raise AttributeError(
                f"'GraspConfig3D' object has no attribute {name!r}")
        group_attr, field_name = target
        return getattr(object.__getattribute__(self, group_attr), field_name)

    def __setattr__(self, name, value):
        # Group attributes and top-level fields go through normally. A flat legacy
        # NAME (e.g. cfg.w_gws = 5.0 after construction) is redirected into its
        # group so mutation-after-construction keeps working exactly like the old
        # flat dataclass.
        if name in _GRASP_CFG_GROUPS or name in (
            'joint_limits', 'wrench_constraint', 'max_iter', 'fixed_contacts', 'n_contacts', 'c3_seed_strategy', 'c3_own_patch', 'c3_patch_offset_m',
            'r_thumb', 'r_index', 'r_middle', 'r_ring',
        ):
            object.__setattr__(self, name, value)
            return
        target = _GRASP_CFG_FLAT_FIELDS.get(name)
        if target is None:
            object.__setattr__(self, name, value)
            return
        group_attr, field_name = target
        setattr(object.__getattribute__(self, group_attr), field_name, value)


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
        self._obj_bid    = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, c.obj_body)
        self._thumb_sid  = self._require_site(c.thumb_site)
        self._index_sid  = self._require_site(c.index_site)
        # Third contact's site. REQUIRED once n_contacts >= 3 (a missing site would
        # otherwise surface as a confusing failure deep in the IK cost). Resolved even
        # at n=2, where nothing in the NLP targets it, for ONE reason: r_middle is
        # measured from it below, and the site-less fallback (geom_rbound) over-reports
        # the LEAP middle pad by 4.2mm (23.68 vs the true 19.47mm max-vertex distance).
        # _has_c3 and every other tripod branch gate on n_contacts, NOT on this being
        # non-None, so resolving it early cannot switch the tripod on.
        self._middle_sid = (self._optional_site(c.middle_site)
                            if int(c.n_contacts) < 3
                            else self._require_site(c.middle_site))
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
        # Pass the SITE: without it a mesh tip falls through to geom_rbound, which is
        # the bounding sphere about the mesh FRAME ORIGIN, not the pad extent from the
        # site. Measured on the LEAP middle tip: rbound 23.68mm vs 19.47mm true. The
        # inflated value fed _r3_ik (the tripod's third IK target = p + r*n), pushing
        # that target ~4mm off the surface. index and middle share mesh dataid 13, so
        # the corrected r_middle equals r_index exactly.
        c.r_middle = _tip_radius(self._middle_gid, self._middle_sid)
        c.r_ring   = _tip_radius(self._ring_gid)
        self.log.info(
            f"[tip_radius] r_thumb={c.r_thumb*1e3:.1f}mm  r_index={c.r_index*1e3:.1f}mm  "
            f"r_middle={c.r_middle*1e3:.1f}mm  r_ring={c.r_ring*1e3:.1f}mm  "
            f"(geom types: thumb={int(model.geom_type[self._thumb_gid])}  "
            f"index={int(model.geom_type[self._index_gid])})")

        # Tip mesh vertices expressed in the tip SITE's own frame, cached once —
        # the raw material for the DIRECTIONAL tip radius (_tip_support_along).
        # Site-local (not geom-local) because the support query needs to rotate
        # with the site frame the IK actually targets. None for a non-mesh tip
        # or a tip whose geom carries no mesh data, in which case the caller
        # falls back to the isotropic cfg.r_* value.
        def _tip_verts_site_local(gid, sid):
            if int(model.geom_type[gid]) != 7:      # not mjGEOM_MESH
                return None
            did = int(model.geom_dataid[gid])
            if did < 0:
                return None
            vadr = int(model.mesh_vertadr[did])
            vnum = int(model.mesh_vertnum[did])
            V = model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3)   # mesh/geom local
            _d0 = mj.MjData(model)
            mj.mj_forward(model, _d0)
            gpos, gmat = _d0.geom_xpos[gid], _d0.geom_xmat[gid].reshape(3, 3)
            spos, smat = _d0.site_xpos[sid], _d0.site_xmat[sid].reshape(3, 3)
            # geom-local vertex -> world (at qpos0) -> site-local
            V_world = gpos + V @ gmat.T
            return (V_world - spos) @ smat
        self._thumb_verts_sl = _tip_verts_site_local(self._thumb_gid, self._thumb_sid)
        self._index_verts_sl = _tip_verts_site_local(self._index_gid, self._index_sid)

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
        self._obj_geom_type = int(model.geom_type[self._obj_gid])
        self._mesh_entry    = None
        self._uv_atlas      = None   # dict: atlas, big_chart_ids, area_frac
        if self._obj_geom_type == _GEOM_TYPE_MESH:
            # geom_size is meaningless for a mesh geom — every analytic-shape
            # consumer of hx/hy/hz (bounding-box constraint, seed bbox_r
            # radius) instead gets object_sdf's baked bounding radius, a
            # conservative stand-in for "half-extent" in every axis.
            self._mesh_entry = _mesh_sdf_entry(model, self._obj_bid)
            _rb = float(self._mesh_entry["table"]["rbound"])
            self._obj_hx = self._obj_hy = self._obj_hz = _rb
            self._obj_size = np.array([_rb, _rb, _rb])
            if c.use_uv_atlas_contact:
                if not _UV_ATLAS_AVAILABLE:
                    self.log.warning(
                        "[uv_atlas] use_uv_atlas_contact=True but xatlas is not "
                        "installed — falling back to the tangent-plane contact scheme.")
                else:
                    _atlas, _big, _area_frac = _object_uv_atlas.load_or_build(
                        model, self._obj_bid, min_area_frac=c.uv_atlas_min_chart_frac)
                    # Per-chart normal/centroid — cheap (O(triangles), no disk cache
                    # needed), used by the antipodal chart-pair/triple seeding
                    # heuristic and the bottom-facing filter (see
                    # object_uv_atlas.chart_normals_centroids /
                    # filter_bottom_facing_charts docstrings).
                    _chart_n, _chart_c = _object_uv_atlas.chart_normals_centroids(_atlas)
                    self._uv_atlas = dict(atlas=_atlas, big_chart_ids=_big, area_frac=_area_frac,
                                          chart_normal=_chart_n, chart_centroid=_chart_c)
                    self.log.info(
                        f"[uv_atlas] {_atlas['n_charts']} charts, {len(_big)} usable "
                        f"(>{c.uv_atlas_min_chart_frac*100:.0f}% area, "
                        f"{_area_frac[_big].sum()*100:.1f}% of surface)")
        else:
            self._obj_hx        = float(gs[0])
            self._obj_hy        = float(gs[1])
            self._obj_hz        = float(gs[2])
            self._obj_size      = model.geom_size[self._obj_gid].copy()

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
        # Separate scratch MjData for _tip_support_along -- _dls_data's qpos is
        # live state for the DLS seeding path, not safe to stomp mid-solve.
        self._tip_data = mj.MjData(model)

    def _tip_support_along(self, which: str, q: np.ndarray, n_out: np.ndarray,
                           r_fallback: float) -> float:
        """DIRECTIONAL tip radius: how far the pad surface actually extends from
        the tip site ALONG n_out (world, unit, pointing out of the object), with
        the tip oriented as it is at configuration `q`.

        The isotropic cfg.r_thumb/r_index is max||V - site|| over ALL directions
        -- a bounding sphere. A LEAP fingertip is an elongated pad, so the vertex
        facing the object is several mm closer than the farthest one anywhere,
        and an IK target of contact + r_iso*n_out parks the pad that much short
        of the surface (measured on 036_wood_block: 5.9mm thumb / 4.8mm index of
        pure geometric slack, against an IK residual of only 0.02-0.05mm -- i.e.
        essentially ALL of the observed pre-squeeze gap, none of it solver
        error). The support function max_i <V_i, n_out> is the honest answer for
        one direction.

        Evaluated in NUMPY at a FIXED q -- the previous Picard stage's solution --
        never symbolically inside the NLP. That is the whole point: a support
        function is non-smooth (the argmax vertex switches), so making it a
        function of the decision variable q would degrade convergence, which is
        exactly why _tip_radius's own comment declined to do this. Frozen per
        stage it is just a constant, structurally identical to how the contact
        normal (_d1_lp/_d2_lp) and the paraboloid curvature are already refrozen
        each stage.

        Falls back to r_fallback (the isotropic radius) for a non-mesh tip.
        """
        verts_sl = (self._thumb_verts_sl if which == 'thumb' else self._index_verts_sl)
        if verts_sl is None:
            return float(r_fallback)
        sid = self._thumb_sid if which == 'thumb' else self._index_sid
        d = self._tip_data
        d.qpos[:len(q)] = q
        mj.mj_forward(self.model, d)
        smat = d.site_xmat[sid].reshape(3, 3)
        # Site-local vertices -> world directions, then support along n_out.
        n_site = smat.T @ np.asarray(n_out, float)      # n_out in the site frame
        return float(np.max(verts_sl @ n_site))

    # ── public API ─────────────────────────────────────────────────────────────

    def solve(self,
              q_ref:   np.ndarray,
              obj_pos: np.ndarray,
              p1_init: np.ndarray | None = None,
              p2_init: np.ndarray | None = None,
              d1:      np.ndarray | None = None,
              d2:      np.ndarray | None = None,
              p3_init: np.ndarray | None = None,
              d3:      np.ndarray | None = None,
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
        p3_init : (3,)  optional THIRD contact seed (middle finger), used only
                  when cfg.n_contacts >= 3. Produced by _seed_third_contact.
                  None keeps the solve a 2-contact pinch, unchanged.
        d3      : outward face direction for p3, same convention as d1/d2.
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

        if self._obj_geom_type == _GEOM_TYPE_MESH:
            # object_sdf's table is baked in the BODY frame (per-hull geom_pos/
            # geom_quat offsets are folded in at bake time — see object_sdf.
            # body_hull_halfspaces), so queries must use the body's own world
            # pose, not any individual hull geom's (which may not even equal
            # the body's for an off-center hull).
            obj_center_np = data_cb.xpos[self._obj_bid].copy()
            obj_R_np      = data_cb.xmat[self._obj_bid].reshape(3, 3).copy()
        else:
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
        _mu_raw, _mu_t_raw = _contact_friction(model, self._obj_gid, self._thumb_gid, self._index_gid)
        _mu     = round(1 * _mu_raw, 3)
        _mu_t   = round(1 * _mu_t_raw, 3)
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
        # THIRD contact seed. No geometric default: unlike p1/p2 there is no sensible
        # fallback position for a tripod's third finger, so absence simply means "solve
        # the 2-contact problem" rather than "invent a contact".
        p3_seed = (np.asarray(p3_init, float) if p3_init is not None else None)

        # UV-atlas CHART assignment — decided ONCE here from the initial seed,
        # held fixed for the whole solve (only the LOCAL neighborhood within
        # the chart is rebuilt between Picard relinearization stages, in
        # _run_stage below — see _mesh_uv_local_contact_ca / object_uv_atlas
        # module docstring). obj_center_np/obj_R_np are already computed above
        # this point in solve() (used by _d1_ws/_d2_ws just below).
        _p1_chart_id = _p2_chart_id = None
        if cfg.use_uv_atlas_contact and self._uv_atlas is not None:
            _atlas = self._uv_atlas["atlas"]
            _big_ids = self._uv_atlas["big_chart_ids"]
            _p1_l = obj_R_np.T @ (p1_seed - obj_center_np)
            _p2_l = obj_R_np.T @ (p2_seed - obj_center_np)
            _p1_chart_id = _object_uv_atlas.nearest_big_chart(_atlas, _big_ids, _p1_l)
            _p2_chart_id = _object_uv_atlas.nearest_big_chart(_atlas, _big_ids, _p2_l)

        # ── DLS warm-start IK ───────────────────────────────────────────────
        # Target the SAME offset point the main NLP cost optimizes for
        # (contact + r_tip*outward_normal, not the bare contact point) — using
        # the bare contact as the DLS target left the warm start short by
        # ~r_tip (a few mm, the fingerpad radius) in the approach direction,
        # before SQP/IPOPT even starts. ablate_ik.py's DLS call and its
        # ConstrainedIKSolver refine stage share this exact target for the
        # same reason (benchmarks/ycb_grasp/ablate_ik.py:94,108).
        _t_ws = time.perf_counter()
        self._dls_data.qpos[:n_act] = q_ref
        if obj_qpos_snap is not None:
            self._dls_data.qpos[n_act:] = obj_qpos_snap
        _d1_ws = (np.asarray(d1, float) if d1 is not None
                  else _geom_normal_np(p1_seed, geom_type, obj_center_np, obj_R_np, geom_size,
                                       mesh_entry=self._mesh_entry))
        _d2_ws = (np.asarray(d2, float) if d2 is not None
                  else _geom_normal_np(p2_seed, geom_type, obj_center_np, obj_R_np, geom_size,
                                       mesh_entry=self._mesh_entry))
        _dls_tgt1 = p1_seed + cfg.r_thumb * _d1_ws
        _dls_tgt2 = p2_seed + cfg.r_index * _d2_ws
        q_dls = self._dls_ik.solve(
            self.model, self._dls_data,
            [self._thumb_sid, self._index_sid],
            [_dls_tgt1, _dls_tgt2],
            q_bias=q_ref, null_gain=0.3)
        mj.mj_kinematics(self.model, self._dls_data)

        _err_th = float(np.linalg.norm(self._dls_data.site_xpos[self._thumb_sid] - _dls_tgt1))
        _err_if = float(np.linalg.norm(self._dls_data.site_xpos[self._index_sid] - _dls_tgt2))
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
                                          obj_R_np, geom_size,
                                          mesh_entry=self._mesh_entry) - _ar)
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
                       update_normals_in_callback: bool = False,
                       r1_override: float | None = None,
                       r2_override: float | None = None,
                       p3_ws: np.ndarray | None = None,
                       d3_lp: np.ndarray | None = None,
                       r3_override: float | None = None) -> dict:
            # p3_ws/d3_lp: the THIRD load-bearing contact (cfg.n_contacts >= 3).
            # None at n=2 -- every branch below that touches contact 3 is gated on
            # `_has_c3`, so the 2-contact problem built here is bit-identical to what
            # it was before the tripod work (verified: 014_lemon gamma_min and both
            # tip gaps unchanged to full precision).

            _t_stage_start = time.perf_counter()
            _uid = id(q_ws)
            # ONE predicate for the whole stage: a third contact exists iff the config
            # asks for it AND a seed was actually supplied. Keeping both conditions here
            # means no downstream branch can half-enable the tripod.
            _has_c3 = (int(cfg.n_contacts) >= 3 and p3_ws is not None
                       and self._middle_sid is not None)

            # ── FK callbacks (analytic Jacobians via mj_jacSite) ───────────
            thumb_cb = _SitePositionCallbackAnalytic(
                f'gp3_th_{_uid}', model, self._thumb_sid, n_act, obj_qpos_snap)
            index_cb = _SitePositionCallbackAnalytic(
                f'gp3_if_{_uid}', model, self._index_sid, n_act, obj_qpos_snap)
            # Third contact's FK, built only when the tripod is actually active so the
            # n=2 NLP gains no callback and no variables.
            middle_cb = (_SitePositionCallbackAnalytic(
                f'gp3_mf_{_uid}', model, self._middle_sid, n_act, obj_qpos_snap)
                if _has_c3 else None)

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
            _is_mesh = (geom_type == _GEOM_TYPE_MESH and self._mesh_entry is not None)
            _t1_var = _t2_var = None   # set below iff a mesh 2-DOF contact var is built
            _t3_var = _p3 = _t3_bounds = _t3_frame = None   # third contact (n_contacts>=3)
            _t1_bounds = _t2_bounds = None   # set below iff use_quadratic_contact (per-axis trust region)
            _t1_frame = _t2_frame = None     # set below iff use_quadratic_contact (paraboloid params, for viz)
            if cfg.fixed_contacts:
                # Staging/diagnostic mode: p1/p2 are CONSTANTS at the seed value, not
                # decision variables — same NLP shape as ConstrainedIKSolver's
                # fixed-target IK (q only). No surface constraint needed (nothing to
                # constrain); section 2 below is skipped for these via _is_mesh-style
                # gating (see include_surface branch).
                _p1 = ca.DM(np.asarray(p1_ws, float))
                _p2 = ca.DM(np.asarray(p2_ws, float))
            elif _is_mesh and _p1_chart_id is not None:
                # UV-atlas local-neighborhood parameterization: p1/p2 become 2-DOF
                # expressions (offset in a plane fit to the seed's local mesh
                # neighborhood WITHIN its assigned chart, bounded by that
                # neighborhood's real boundary — not an arbitrary Euclidean box),
                # reprojected onto the surface. Re-centered on p1_ws/p2_ws every
                # call, so the Picard loop's existing between-stage re-seeding
                # (see the caller in solve()) walks this across the whole chart
                # over multiple stages — see _mesh_uv_local_contact_ca /
                # grasp_control.object_uv_atlas module docstring. Chart id itself
                # is fixed for the whole solve (decided once, above, from the
                # ORIGINAL seed) — only the local neighborhood within it changes.
                _t1_var, _p1 = _mesh_uv_local_contact_ca(
                    _opti, p1_ws, obj_center_np, obj_R_np, self._mesh_entry,
                    self._uv_atlas["atlas"], _p1_chart_id, rings=cfg.uv_atlas_rings)
                _t2_var, _p2 = _mesh_uv_local_contact_ca(
                    _opti, p2_ws, obj_center_np, obj_R_np, self._mesh_entry,
                    self._uv_atlas["atlas"], _p2_chart_id, rings=cfg.uv_atlas_rings)
            elif _is_mesh and cfg.use_quadratic_contact:
                # Local-quadratic (paraboloid) parameterization: p1/p2 become
                # 2-DOF expressions placed directly on a surface model fit
                # from the SDF's gradient+Hessian at the seed — no per-
                # candidate surface_project call, no mesh/chart lookup. See
                # _mesh_quadratic_contact_ca. Re-centered every call the same
                # way _mesh_tangent_contact_ca is, so the Picard loop's
                # between-stage re-seeding applies unchanged.
                _n1_seed_out = -np.asarray(d1_lp, float) if d1_lp is not None else \
                    _geom_normal_np(p1_ws, geom_type, obj_center_np, obj_R_np, geom_size,
                                    mesh_entry=self._mesh_entry)
                _n2_seed_out = -np.asarray(d2_lp, float) if d2_lp is not None else \
                    _geom_normal_np(p2_ws, geom_type, obj_center_np, obj_R_np, geom_size,
                                    mesh_entry=self._mesh_entry)
                _t1_var, _p1, _t1_bounds, _t1_frame = _mesh_quadratic_contact_ca(
                    _opti, p1_ws, _n1_seed_out, obj_center_np, obj_R_np, self._mesh_entry,
                    t_bound_max=cfg.quadratic_t_bound_max,
                    sdf_err_tol=cfg.quadratic_sdf_err_tol,
                    mesh_fit=cfg.quadratic_mesh_fit,
                    mesh_fit_radius=cfg.quadratic_mesh_fit_radius,
                    mesh_fit_quad_gain_min=cfg.quadratic_mesh_fit_gain_min)
                _t2_var, _p2, _t2_bounds, _t2_frame = _mesh_quadratic_contact_ca(
                    _opti, p2_ws, _n2_seed_out, obj_center_np, obj_R_np, self._mesh_entry,
                    t_bound_max=cfg.quadratic_t_bound_max,
                    sdf_err_tol=cfg.quadratic_sdf_err_tol,
                    mesh_fit=cfg.quadratic_mesh_fit,
                    mesh_fit_radius=cfg.quadratic_mesh_fit_radius,
                    mesh_fit_quad_gain_min=cfg.quadratic_mesh_fit_gain_min)
                # THIRD contact SHARES contact 2's patch (index + middle on one
                # paraboloid, each with its own 2-DOF coordinate inside the SAME trust
                # region), reconstructed via the identity _mesh_quadratic_contact_ca's
                # frame dict documents:
                #     p_local(t) = seed_l + t0*axis0_l + t1*axis1_l + h(t)*n_l
                #
                # An earlier version fitted an INDEPENDENT third patch from a fan seed.
                # It failed badly -- measured 183mm between the middle fingertip and its
                # assigned contact on 036_wood_block -- because an independent patch can
                # land on a face the hand would have to re-approach entirely. Sharing
                # guarantees the two contacts are adjacent and mutually reachable, and a
                # standalone test confirmed the optimizer then slides BOTH to reachable
                # spots inside the bounds (index/middle 10-24mm apart, every finger
                # converging to its pad radius). The rest-pose fingertip separation
                # (178-219mm) that motivated separate patches was the wrong measurement:
                # what matters is whether both fingers can curl onto NEARBY contacts,
                # not how far apart they hang when extended.
                if _has_c3 and cfg.c3_own_patch:
                    # OWN PATCH, fitted at contact 3's own seed exactly as contacts 1
                    # and 2 are above. The shared-patch alternative (the else branch)
                    # is structurally unable to hold two fingertips at natural
                    # spacing: measured across 15 accepted seeds on 5 objects, the
                    # index patch half-extent is 17-24mm on rounded objects (median
                    # 19.7mm) against a 45.4mm middle-to-index finger pitch. Only 33%
                    # of seeds reach 45mm and every one of those is a flat face whose
                    # bound is quadratic_t_bound_max CLIPPING rather than a measured
                    # trust region.
                    #
                    # An independent patch was tried once before and measured 183mm
                    # off on 036_wood_block. That attempt paired it with the FAN
                    # seeder, which places candidates at object scale with nothing
                    # tying them to contact 2's neighbourhood, so the patch could land
                    # on a face the hand would have to re-approach. The tangent/
                    # kinematic seeders exist to remove that failure mode -- if this
                    # branch still fails under them, the independent patch itself is
                    # the problem and the finding is genuine rather than confounded.
                    # EXACTLY the _n1_seed_out/_n2_seed_out convention (:3531): d*_lp is
                    # INWARD and is negated once to give the outward seed normal, while
                    # the _geom_normal_np fallback is ALREADY outward and must not be
                    # negated. An earlier version negated the whole expression and then
                    # negated again -- a no-op that left the d3_lp path pointing INWARD,
                    # which fits the third paraboloid facing into the object.
                    _n3_seed_out = -np.asarray(d3_lp, float) if d3_lp is not None else \
                        _geom_normal_np(p3_ws, geom_type, obj_center_np, obj_R_np,
                                        geom_size, mesh_entry=self._mesh_entry)
                    _t3_var, _p3, _t3_bounds, _t3_frame = _mesh_quadratic_contact_ca(
                        _opti, np.asarray(p3_ws, float), _n3_seed_out,
                        obj_center_np, obj_R_np, self._mesh_entry,
                        t_bound_max=cfg.quadratic_t_bound_max,
                        sdf_err_tol=cfg.quadratic_sdf_err_tol,
                        mesh_fit=cfg.quadratic_mesh_fit,
                        mesh_fit_radius=cfg.quadratic_mesh_fit_radius,
                        mesh_fit_quad_gain_min=cfg.quadratic_mesh_fit_gain_min)
                elif _has_c3:
                    # SHARED patch (ablation baseline). Contact 3 is reconstructed
                    # inside contact 2's paraboloid via the identity its frame dict
                    # documents: p_local(t) = seed_l + t0*axis0_l + t1*axis1_l + h(t)*n_l.
                    # NOTE this DISCARDS the third seed entirely -- p3_ws never enters,
                    # and _t3_var starts at the middle of contact 2's patch -- which is
                    # why seed-to-final measured 64/51/190mm on orange/lemon/block.
                    _t3_var = _opti.variable(2)
                    _opti.subject_to(_opti.bounded(_t2_frame['t_lo_0'], _t3_var[0],
                                                   _t2_frame['t_hi_0']))
                    _opti.subject_to(_opti.bounded(_t2_frame['t_lo_1'], _t3_var[1],
                                                   _t2_frame['t_hi_1']))
                    # Start OFF contact 2's own initial point so the two don't begin
                    # coincident (identical contacts give a degenerate wrench matrix).
                    _opti.set_initial(_t3_var, np.array([0.5 * _t2_frame['t_hi_0'],
                                                         0.5 * _t2_frame['t_hi_1']]))
                    _h3 = -(_t2_frame['kappa0'] * _t3_var[0]**2
                            + _t2_frame['kappa1'] * _t3_var[1]**2) / (2.0 * _t2_frame['grad_norm'])
                    _p3_l = (ca.DM(_t2_frame['seed_l'])
                             + _t3_var[0] * ca.DM(_t2_frame['axis0_l'])
                             + _t3_var[1] * ca.DM(_t2_frame['axis1_l'])
                             + _h3 * ca.DM(_t2_frame['n_l']))
                    _p3 = ca.DM(obj_center_np) + ca.DM(obj_R_np) @ _p3_l
                    _t3_bounds = _t2_bounds
                    _t3_frame  = _t2_frame
            elif _is_mesh:
                # Tangent-plane parameterization: p1/p2 become 2-DOF expressions
                # (offset in the local tangent plane at the seed, reprojected
                # onto the surface) instead of free R^3 points + an SDF=0
                # equality — see _mesh_tangent_contact_ca. Surface adherence is
                # structural, so no equality constraint is added in section 2
                # below for mesh objects.
                _n1_seed_out = -np.asarray(d1_lp, float) if d1_lp is not None else \
                    _geom_normal_np(p1_ws, geom_type, obj_center_np, obj_R_np, geom_size,
                                    mesh_entry=self._mesh_entry)
                _n2_seed_out = -np.asarray(d2_lp, float) if d2_lp is not None else \
                    _geom_normal_np(p2_ws, geom_type, obj_center_np, obj_R_np, geom_size,
                                    mesh_entry=self._mesh_entry)
                _t1_var, _p1 = _mesh_tangent_contact_ca(
                    _opti, p1_ws, _n1_seed_out, obj_center_np, obj_R_np, self._mesh_entry)
                _t2_var, _p2 = _mesh_tangent_contact_ca(
                    _opti, p2_ws, _n2_seed_out, obj_center_np, obj_R_np, self._mesh_entry)
            else:
                _p1 = _opti.variable(3)
                _p2 = _opti.variable(3)

            # SYMBOLIC inward normals for the geometry cost terms, when the
            # quadratic surrogate can supply them. The paraboloid gives n(t) in
            # closed form (_quadratic_inward_normal_ca), so a cost that means
            # "align something with the contact normal" should track the normal
            # AT THE CONTACT THE SOLVER IS CHOOSING, not the seed's. Using the
            # frozen seed normal makes the cost pull toward a direction the
            # solve has already moved away from -- the same inconsistency the
            # wrench frame had before quadratic_symbolic_normals. Falls back to
            # the frozen -d*_lp whenever the surrogate isn't active.
            _n1_in_sym_cost = _n2_in_sym_cost = None
            if (cfg.quadratic_symbolic_normals and _is_mesh
                    and cfg.use_quadratic_contact
                    and _t1_frame is not None and _t2_frame is not None
                    and _t1_var is not None and _t2_var is not None):
                _n1_in_sym_cost = _quadratic_inward_normal_ca(_t1_var, _t1_frame, obj_R_np)
                _n2_in_sym_cost = _quadratic_inward_normal_ca(_t2_var, _t2_frame, obj_R_np)

            # Per-consumer ablation: each sub-flag can veto the symbolic normal
            # for ONE consumer while the others keep it (None = follow master).
            def _sym_pair(sub):
                if _n1_in_sym_cost is None:
                    return None, None
                if sub is False:
                    return None, None
                return _n1_in_sym_cost, _n2_in_sym_cost

            _n1_ik_s, _n2_ik_s = _sym_pair(cfg.quad_sym_normals_iktgt)
            _n1_al_s, _n2_al_s = _sym_pair(cfg.quad_sym_normals_align)
            _n1_or_s, _n2_or_s = _sym_pair(cfg.quad_sym_normals_orient)

            _tp1   = thumb_cb(_q)
            _tp2   = index_cb(_q)
            # IK cost: fingertip center should be at contact point + r_tip * outward_normal.
            # Without the offset the tip sphere embeds r_tip mm into the object surface.
            # r*_override (when set) is the DIRECTIONAL support distance along
            # this stage's frozen normal, from the previous stage's q -- see
            # _tip_support_along. It replaces the isotropic bounding-sphere
            # radius HERE ONLY: this is the target that decides how far off the
            # surface the pad parks, and the isotropic value overshoots it by
            # several mm on an elongated pad. The ground-clearance constraint in
            # section 5b deliberately keeps using cfg.r_* -- that one wants a
            # true bounding sphere (the tip can approach the floor from any
            # direction, not just along the contact normal).
            _r1_ik = float(cfg.r_thumb if r1_override is None else r1_override)
            _r2_ik = float(cfg.r_index if r2_override is None else r2_override)
            # Offset direction: the paraboloid's OWN outward normal n(t) when the
            # surrogate supplies it symbolically, else the frozen seed direction.
            #
            # This is the term the frozen normal costs most. The trust region
            # bounds the patch's POSITION error (|SDF| <= sdf_err_tol, measured
            # 0.4-0.7mm), but says nothing about how far the NORMAL has rotated
            # getting there -- and on a curved patch those decouple sharply:
            # 017_orange holds 0.41mm of surface error across a 13mm patch while
            # its normal turns 15.7deg. The IK target is p + r*n, so that
            # rotation is multiplied by the pad radius (~19mm) before it reaches
            # the target: 5.3mm of target displacement from a patch that is
            # itself accurate to 0.4mm, a ~13x amplification of an error the
            # trust region correctly reports as negligible. Using n(t) removes
            # the term rather than bounding it. Identically zero on a planar
            # patch (kappa=0, measured 0.00mm on 036_wood_block), so this only
            # bites on curved objects -- which is where the tip gaps are.
            _n1_out_ik = (-_n1_ik_s if _n1_ik_s is not None
                          else ca.DM(np.asarray(d1_lp, float)))
            _n2_out_ik = (-_n2_ik_s if _n2_ik_s is not None
                          else ca.DM(np.asarray(d2_lp, float)))
            _tp1_tgt = _p1 + _r1_ik * _n1_out_ik
            _tp2_tgt = _p2 + _r2_ik * _n2_out_ik
            _d1_sq = ca.sumsqr(_tp1 - _tp1_tgt)   # m²
            _d2_sq = ca.sumsqr(_tp2 - _tp2_tgt)   # m²
            # Third contact's IK term, same construction as 1 and 2: the fingertip
            # CENTER targets contact + r_tip*outward_normal, not the bare contact.
            _d3_sq = None
            if _has_c3 and _p3 is not None and middle_cb is not None:
                _tp3 = middle_cb(_q)
                _r3_ik = float(cfg.r_middle if cfg.r_middle is not None else cfg.r_index)
                _n3_out_ik = ca.DM(np.asarray(
                    d3_lp if d3_lp is not None else _n3_seed_out, float))
                _tp3_tgt = _p3 + _r3_ik * _n3_out_ik
                _d3_sq = ca.sumsqr(_tp3 - _tp3_tgt)   # m²

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
            # Averaged over the contacts present, so w_ik keeps its calibrated meaning:
            # at n=2 this is exactly the historical 0.5*(d1+d2), and a third contact does
            # not inflate the IK term relative to reg/align/gws (which would silently
            # re-tune every other weight).
            if _d3_sq is not None:
                _cost_ik = (_d1_sq + _d2_sq + _d3_sq) / (3.0 * _d_ref**2)
            else:
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
            # n=2 ONLY. The term is (p2-p1)-relational -- it asks that THE grasp axis
            # align with contact 1's inward normal -- and a tripod has no single grasp
            # axis, so there is nothing here to generalize. Deliberately left OFF for
            # n!=2 rather than replaced by a guessed n-contact analogue: for three
            # non-collinear contacts OPPOSITION is not the right objective anyway
            # (force closure there means the normals SPAN the origin, which is exactly
            # what the FRoGGeR min-weight beta measures and w_gws already optimizes).
            # If beta turns out not to carry it, add an n>2 strategy here then -- see
            # the caveat that beta measured unreliable at n=2, which was attributed to
            # the rank-5-of-6 degeneracy a third contact is supposed to remove.
            # Keeping the gate explicit preserves the measured n=2 path bit-identically
            # (12/15 lifts across 3 seeds x 5 objects at max_iter=80, n_seeds=3).
            if cfg.w_align > 0.0 and d1_lp is not None and cfg.n_contacts == 2:
                _n1_in_al = (_n1_al_s if _n1_al_s is not None
                             else ca.DM(-np.asarray(d1_lp, float)
                                        / (np.linalg.norm(d1_lp) + 1e-12)))
                _dp = _p2 - _p1
                _g_hat = _dp / (ca.norm_2(_dp) + 1e-9)
                _cost_align = ca.sumsqr(_g_hat - _n1_in_al)
                _cost = _cost + cfg.w_align * _cost_align

            # ── Fingerpad-normal alignment (orient_weight) ────────────────────
            # Penalize each tip's pad axis (R_tip(q) @ pad_axis, world) deviating from that
            # contact's INWARD surface normal, so the pad meets the face flush. Same term as
            # ConstrainedIKSolver's orient_weight (‖R_tip@pad_axis − n_in‖² per contact). The
            # inward normals are the paraboloid's SYMBOLIC n(t) under
            # quadratic_symbolic_normals, else the frozen seed directions (-d1_lp thumb,
            # -d2_lp index). In the symbolic case the cost couples the tip rotation to the
            # contact the solver is choosing, which is also what pins the directional pad
            # radius r_par: r_par is a support function of the pad-vs-normal angle, so a
            # cost that holds that angle steady holds r_par steady. Sentinel-zero when off so the log helper can
            # always evaluate it.
            _cost_orient = ca.DM(0.0)
            if (cfg.orient_weight > 0.0 and thumb_axis_cb is not None
                    and d1_lp is not None and d2_lp is not None):
                _n1_in_or = (_n1_or_s if _n1_or_s is not None
                             else ca.DM(-np.asarray(d1_lp, float)
                                        / (np.linalg.norm(d1_lp) + 1e-12)))
                _n2_in_or = (_n2_or_s if _n2_or_s is not None
                             else ca.DM(-np.asarray(d2_lp, float)
                                        / (np.linalg.norm(d2_lp) + 1e-12)))
                _e_th = thumb_axis_cb(_q) - _n1_in_or
                _e_if = index_axis_cb(_q) - _n2_in_or
                _cost_orient = ca.dot(_e_th, _e_th) + ca.dot(_e_if, _e_if)
                _cost = _cost + cfg.orient_weight * _cost_orient

            # Edge margin is now a HARD constraint (tightened tangential face bounds in
            # _sym_geom_surface_con via cfg.edge_margin_m) — no cost term needed. BOX
            # only, though (see _sym_geom_surface_con) — mesh contacts get no hard
            # edge keep-out at all, hence the soft curvature penalty below.

            # ── Contact-height / COM-proximity penalty (DIAGNOSTIC) ────────────
            # Pull contacts toward the object's centroid plane. This is a blunt
            # object-specific heuristic, NOT a general edge-avoidance rule -- it
            # encodes "grasp near the middle", which is wrong for objects you
            # should grasp high (a mug by its rim, a bottle by its neck). It
            # exists to answer one question the investigation has not yet
            # settled: does a mid-height grasp actually make 036_wood_block
            # liftable? Everything so far shows contacts are driven to the top
            # edge (IK cost gradient d(cost)/dz ~= -579 at the stage-1 seed,
            # against -0.4 for alignment), but not that fixing that is
            # sufficient. If the block still fails with contacts forced to the
            # COM plane, edge-avoidance is the wrong direction entirely and no
            # amount of principled regularization will help.
            #
            # Weight has to be LARGE to register: the IK term dominates every
            # geometric term by ~1400x, which is why the old curvature penalty
            # at w=100 moved the solution 0mm.
            if cfg.w_contact_height > 0.0 and _is_mesh:
                # COM world height: xipos is the body's inertial (COM) frame
                # origin, which for these YCB bodies is genuinely offset from
                # the body origin (036_wood_block: +103mm on a 207mm block).
                _com_w_z = float(data_cb.xipos[self._obj_bid][2])
                _cost_height = ca.DM(0.0)
                for _pv in (_p1, _p2):
                    if _pv is None:
                        continue
                    # world-frame height of the contact vs the object's COM
                    _dz = _pv[2] - float(_com_w_z)
                    _cost_height = _cost_height + _dz**2
                _cost = _cost + cfg.w_contact_height * _cost_height

            # ── Edge-margin penalty (quadratic mesh contact only) ──────────────
            # Keep each contact a margin away from where the surface actually
            # RUNS OUT, measured by the trust-region bound search rather than by
            # local curvature.
            #
            # This replaces an earlier curvature-based version that penalized the
            # paraboloid height h(t_var) = -(kappa0*t0^2 + kappa1*t1^2)/(2|grad|).
            # Curvature is the wrong signal for "near an edge" because it
            # conflates two independent things: a genuinely ROUND object (apple,
            # bottle shoulder) has large kappa everywhere and would be penalized
            # for its own true shape, while a FLAT-faced object has kappa ~ 0
            # across the whole face and gets no penalty at all -- right up to the
            # edge it is about to fall off. Measured on 036_wood_block: the
            # curvature term contributed ~1e-6 to the vertical cost gradient
            # (against ~-579 from the IK term) at every stage until the contact
            # was ALREADY on the top edge, i.e. it could only object after the
            # fact, never steer. Ablating it changed the solved contact by 0mm.
            #
            # _sdf_axis_bound_np measures the right thing directly: how far along
            # each principal axis the tangent offset can go before the TRUE SDF
            # departs from its seed value by more than quadratic_sdf_err_tol.
            # That is a statement about the surface's EXTENT, independent of how
            # curved it is, so it fires on a flat face near its boundary and
            # stays quiet in the middle of a round one.
            #
            # Only axes whose bound came from a MEASURED divergence are
            # penalized. An axis that ran the whole search without ever exceeding
            # tolerance is returned at exactly t_bound_max -- that is the hard
            # cap, meaning "flat as far as we looked", NOT an edge. Penalizing
            # proximity to that cap would push contacts away from the middle of
            # large flat faces for no reason (the case this term exists to allow).
            #
            # Shape: a one-sided quadratic hinge, zero until |t| passes
            # (bound - margin), then growing as the square of the excess. NOT a
            # barrier: the box constraint on t_var is already hard (and Ipopt
            # applies its own barrier to it), the margin is a robustness
            # heuristic rather than a physical limit, and a finite price lets the
            # solver still take a near-edge contact when that is the only option
            # instead of turning the problem infeasible. Zero gradient in the
            # interior also means it perturbs only the grasps that need it.
            if cfg.w_edge_margin > 0.0:
                _cost_edge = ca.DM(0.0)
                _margin = float(cfg.edge_margin_sdf_m)
                _cap = float(cfg.quadratic_t_bound_max)
                for _t_var, _frame in ((_t1_var, _t1_frame), (_t2_var, _t2_frame)):
                    if _t_var is None or _frame is None:
                        continue
                    for _i, _key in ((0, 't_bound_0'), (1, 't_bound_1')):
                        _tb = float(_frame[_key])
                        if _tb >= _cap - 1e-9:
                            continue          # capped => flat, no edge found
                        _safe = max(_tb - _margin, 0.0)
                        _excess = ca.fmax(0.0, ca.fabs(_t_var[_i]) - _safe)
                        _cost_edge = _cost_edge + _excess**2
                _cost = _cost + cfg.w_edge_margin * _cost_edge

            # ── 1. Joint limits (vectorized) ──────────────────────────────
            if cfg.joint_limits:
                _opti.subject_to(_opti.bounded(
                    ca.DM(self._lo_vec), _q, ca.DM(self._hi_vec)))

            # ── 2. Surface constraints ────────────────────────────────────────
            # Mesh contacts are on-surface BY CONSTRUCTION (tangent-plane +
            # reprojection above) — no equality constraint needed or added.
            # fixed_contacts: p1/p2 are constants, nothing to constrain either.
            if include_surface and not _is_mesh and not cfg.fixed_contacts:
                _Rt_dm = ca.DM(obj_R_np.T)
                _c_dm  = ca.DM(obj_center_np)
                for _p, _d_lp in ((_p1, d1_lp), (_p2, d2_lp)):
                    _pl = _Rt_dm @ (_p - _c_dm)   # local frame
                    _sym_geom_surface_con(_opti, _p, _d_lp,
                                            geom_type, obj_center_np, obj_R_np, geom_size,
                                            edge_margin=cfg.edge_margin_m)


            # Bounding box: prevents p1/p2 from flying to infinity (mesh contacts
            # are already surface-pinned, so this is a loose safety net only).
            # fixed_contacts: p1/p2 are constants — a constraint on a constant is a
            # useless no-op check, skip it.
            if not cfg.fixed_contacts:
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
            _R1_expr     = None   # set below iff a contact frame is actually needed
            _R2_expr     = None
            _R3_expr     = None   # third contact's frame (n_contacts >= 3 only)
            # Contact frame [n_in|t1|t2] is needed by BOTH the wrench-cone LP
            # (gamma/y/s) and the GWS min-weight LP (alpha/beta) — they share the
            # same _friction_cone_verts/frame convention (see build_W_ca's
            # docstring). Build it whenever EITHER wants it, not just when
            # wrench_constraint=True: previously GWS silently no-opted (with only
            # a log warning) whenever wrench_constraint=False, since _R1_expr was
            # never assigned outside that flag's block — GWS was supposed to be
            # independently selectable (see GraspConfig3D.w_gws's "ADDITIVE...
            # both default to 0.0" docstring) but structurally wasn't.
            _need_contact_frame = cfg.wrench_constraint or cfg.w_gws > 0.0 or cfg.w_span > 0.0
            if _need_contact_frame:
                # MESH under use_quadratic_contact: the paraboloid surrogate
                # supplies the normal in CLOSED FORM (see
                # _quadratic_inward_normal_ca), so the frame can track the
                # contact the same way it does for an analytic sphere/cylinder
                # -- no SDF call, no mesh lookup, exactly differentiable.
                # Gated on its own flag because it makes the frame a function
                # of the decision variables, which changes the NLP's
                # conditioning (see cfg.symbolic_normals' own note about
                # L-BFGS seeing curvature pairs from both movement AND frame
                # rotation).
                use_quad_sym = (cfg.quadratic_symbolic_normals
                                and cfg.quad_sym_normals_frame is not False
                                and _is_mesh and cfg.use_quadratic_contact
                                and _t1_frame is not None and _t2_frame is not None
                                and _t1_var is not None and _t2_var is not None)
                use_sym_normals = (cfg.symbolic_normals and
                                   geom_type in (2, 5))  # sphere / cylinder only
                _smooth_frame = cfg.gws_smooth_frame
                if use_quad_sym:
                    _n1_in_sym = _quadratic_inward_normal_ca(_t1_var, _t1_frame, obj_R_np)
                    _n2_in_sym = _quadratic_inward_normal_ca(_t2_var, _t2_frame, obj_R_np)
                    _R1_expr = _symbolic_contact_frame_ca(_n1_in_sym, smooth_blend=_smooth_frame)
                    _R2_expr = _symbolic_contact_frame_ca(_n2_in_sym, smooth_blend=_smooth_frame)
                    # Contact 3 lives on contact 2's patch (_t3_frame IS _t2_frame),
                    # so its normal comes from the SAME paraboloid evaluated at the
                    # third contact's own coordinate -- not a copy of contact 2's.
                    if _has_c3 and _t3_var is not None and _t3_frame is not None:
                        _R3_expr = _symbolic_contact_frame_ca(
                            _quadratic_inward_normal_ca(_t3_var, _t3_frame, obj_R_np),
                            smooth_blend=_smooth_frame)
                elif use_sym_normals:
                    # Contact frame built as a CasADi MX expression of _p1/_p2.
                    # CasADi re-evaluates this at every eval_f / eval_grad_f call,
                    # so the frame tracks the current contact position throughout
                    # the solve.  L-BFGS sees inconsistent curvature pairs because
                    # ∇f changes both from x movement AND frame rotation.
                    _c_dm  = ca.DM(obj_center_np)
                    _Rt_dm = ca.DM(obj_R_np.T)
                    _n1_in_sym = _sym_inward_normal_ca(_p1, geom_type, _c_dm, _Rt_dm, geom_size)
                    _n2_in_sym = _sym_inward_normal_ca(_p2, geom_type, _c_dm, _Rt_dm, geom_size)
                    _R1_expr = _symbolic_contact_frame_ca(_n1_in_sym, smooth_blend=_smooth_frame)
                    _R2_expr = _symbolic_contact_frame_ca(_n2_in_sym, smooth_blend=_smooth_frame)
                    if _has_c3 and _p3 is not None:
                        _R3_expr = _symbolic_contact_frame_ca(
                            _sym_inward_normal_ca(_p3, geom_type, _c_dm, _Rt_dm, geom_size),
                            smooth_blend=_smooth_frame)
                else:
                    # Default: contact frames as 3×3 parameter — frozen per NLP solve,
                    # updated between Picard iterations by the outer loop.
                    _n1_in = -(d1_lp if d1_lp is not None else
                                _geom_normal_np(p1_ws, geom_type, obj_center_np, obj_R_np, geom_size,
                                                mesh_entry=self._mesh_entry))
                    _n2_in = -(d2_lp if d2_lp is not None else
                                _geom_normal_np(p2_ws, geom_type, obj_center_np, obj_R_np, geom_size,
                                                mesh_entry=self._mesh_entry))
                    _R1_param = _opti.parameter(3, 3)
                    _R2_param = _opti.parameter(3, 3)
                    _opti.set_value(_R1_param, np.column_stack(_build_contact_frame_3d(_n1_in)))
                    _opti.set_value(_R2_param, np.column_stack(_build_contact_frame_3d(_n2_in)))
                    _R1_expr = _R1_param
                    _R2_expr = _R2_param
                    if _has_c3 and p3_ws is not None:
                        _n3_in = -(d3_lp if d3_lp is not None else
                                   _geom_normal_np(p3_ws, geom_type, obj_center_np,
                                                   obj_R_np, geom_size,
                                                   mesh_entry=self._mesh_entry))
                        _R3_param = _opti.parameter(3, 3)
                        _opti.set_value(_R3_param,
                                        np.column_stack(_build_contact_frame_3d(_n3_in)))
                        _R3_expr = _R3_param

            if cfg.wrench_constraint:
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

            # ── GWS min-weight / span quality objective ─────────────────────
            # Additive to the existing wrench-cone LP above, not a replacement
            # (see module-level GWS brief + cfg.w_gws/w_span docstring). Only
            # built when actually weighted, since it adds a bilinear equality
            # (W depends on p1/p2, alpha is a new variable multiplying W) that
            # would otherwise cost solver time for zero objective benefit.
            _cost_gws  = ca.DM(0.0)
            _cost_span = ca.DM(0.0)
            _gws_alpha = _gws_beta = _gws_W = None
            if (cfg.w_gws > 0.0 or cfg.w_span > 0.0) and _R1_expr is None:
                self.log.warning(
                    "[gws] w_gws/w_span > 0 but wrench_constraint=False — no contact "
                    "frame (_R1_expr) available, GWS terms skipped this stage.")
            if (cfg.w_gws > 0.0 or cfg.w_span > 0.0) and _R1_expr is not None:
                _gws_mu_t = _mu_t if cfg.gws_soft_finger else 0.0
                # EVERY load-bearing contact enters W. Omitting contact 3 made beta
                # the min-weight of the PINCH ALONE: the tripod could not show the
                # rank-6 improvement it exists for, and -w_gws*beta gave the solver
                # no gradient pulling contact 3 anywhere useful -- which is half of
                # why it collapsed onto contact 2. Reported three-finger betas from
                # before this (-4.56, -4.87) were 2-contact numbers.
                _gws_extra = ([(_p3, _R3_expr)]
                              if (_has_c3 and _p3 is not None and _R3_expr is not None)
                              else None)
                _gws_W = build_W_ca(_p1, _p2, _R1_expr, _R2_expr,
                                    obj_center_np, obj_R_np, _mu, mu_t=_gws_mu_t,
                                    extra_contacts=_gws_extra)
                if _has_c3 and _gws_extra is None:
                    self.log.warning(
                        "[gws] n_contacts>=3 but contact 3 has no frame/position this "
                        "stage — beta is the 2-contact min-weight, not the tripod's.")
                _gws_alpha, _gws_beta, _cost_gws_reg = _embed_gws_ca(
                    _opti, _gws_W, alpha_reg=cfg.gws_alpha_reg)
                # Uniform witness + beta<0 (not yet necessarily in closure) is a
                # neutral, always-valid start — mirrors the gamma/y cold-start
                # convention just above (no LP pre-solve to warm-start from).
                _opti.set_initial(_gws_alpha, np.ones(_gws_W.shape[1]) / _gws_W.shape[1])
                _opti.set_initial(_gws_beta, -1e-3)
                if cfg.w_gws > 0.0:
                    _cost_gws = -_gws_beta   # maximize beta = minimize -beta
                if cfg.w_span > 0.0:
                    _cost_span = -_gws_span_logdet_ca(_gws_W, cfg.gws_span_delta)
                # cost_gws_reg (alpha_reg*||alpha||^2) is added unconditionally
                # whenever the embedding was built at all (w_gws>0 or w_span>0),
                # NOT gated on w_gws — it's a property of the alpha/beta QP
                # itself (unique primal -> unique duals), independent of
                # whether beta is currently in the objective or only a
                # constraint-side variable.
                _cost = (_cost + cfg.w_gws * _cost_gws + cfg.w_span * _cost_span
                          + _cost_gws_reg)


            # Finalize cost (gamma + y regularizer + wrench-cone slack — all normalized)
            # Sentinel zero expressions let the log helper always evaluate all terms.
            #
            # When GWS is active (w_gws/w_span > 0), gamma/y/slack are dropped from the
            # COST (effective weights zeroed below) but stay as NLP CONSTRAINTS — the
            # embedded wrench-cone LP still gates feasibility (a corner must be
            # resistable, budget-bounded by gamma_max), it just no longer competes with
            # beta/logdet for the contact-placement gradient (they'd otherwise pull in
            # overlapping directions — a deeper/better-conditioned wrench hull generally
            # also needs less gamma — inflating tuning difficulty for no benefit). The
            # task-specific wrench budget is still independently certified post-solve by
            # verify()'s min_gamma_for_accel_lp_hard, unaffected by this gating.
            _gws_active = _gws_W is not None   # actually built this stage, not just weighted
            _w_gamma_eff = 0.0 if _gws_active else cfg.w_gamma
            _w_y_eff     = 0.0 if _gws_active else cfg.w_y
            _w_slack_eff = 0.0 if _gws_active else cfg.w_slack

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
                              + _w_gamma_eff * _cost_gamma
                              + _w_y_eff     * _cost_y)
                if _w_slack_eff and _s_list:
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
                    _opti_cost = _opti_cost + _w_slack_eff * _cost_slack
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

            # ── Per-term VERTICAL gradient (diagnostic; PFF_GRAD_Z=1) ─────────
            # A gradient NORM says how hard a term pushes, not WHICH WAY. To
            # answer "what drives contacts up the object?" what is needed is
            # the signed derivative of each term w.r.t. the contact's world
            # HEIGHT: d(cost_i)/dz < 0 means term i is lowered by moving the
            # contact UP, i.e. that term prefers a higher contact.
            #
            # Reported both RAW (d(cost_i)/dz with the configured weight
            # divided back out) and WEIGHTED (w_i * d(cost_i)/dz). The raw
            # number says what the term intrinsically prefers, independent of
            # how it happens to be weighted in this config; the weighted one
            # says what actually moves this solve. A term can be intrinsically
            # height-hungry yet irrelevant because its weight is small, or
            # nearly height-neutral yet dominant because its weight is large --
            # only reporting both separates those.
            _grad_z_exprs = {}
            _grad_z_pending = {}
            if os.environ.get("PFF_GRAD_Z") and _is_mesh and _t1_var is not None:
                _z_dirs = []
                for _pv, _tv in ((_p1, _t1_var), (_p2, _t2_var)):
                    if _tv is None:
                        continue
                    # d(contact world z)/d(t_var) -- the 2 tangent DOFs are the
                    # only way this stage can move the contact at all.
                    _z_dirs.append((_pv[2], _tv))
                _named = dict(_grad_terms)
                _named['align']  = (cfg.w_align * locals()['_cost_align']
                                    if cfg.w_align > 0.0 and '_cost_align' in locals()
                                    else ca.DM(0.0))
                _named['orient'] = (cfg.orient_weight * locals()['_cost_orient']
                                    if cfg.orient_weight > 0.0 and '_cost_orient' in locals()
                                    else ca.DM(0.0))
                _named['edge']   = (cfg.w_edge_margin * locals()['_cost_edge']
                                    if cfg.w_edge_margin > 0.0 and '_cost_edge' in locals()
                                    else ca.DM(0.0))
                _wts = {'ik': cfg.w_ik, 'reg': cfg.w_reg, 'gamma': cfg.w_gamma,
                        'y': cfg.w_y, 'slack': (cfg.w_slack or 0.0),
                        'align': cfg.w_align, 'orient': cfg.orient_weight,
                        'edge': cfg.w_edge_margin}
                for _nm, _term in _named.items():
                    _acc = ca.DM(0.0)
                    for _zexpr, _tv in _z_dirs:
                        _gt = ca.gradient(_term, _tv)      # d(term)/d(t_var), 2x1
                        _gz = ca.gradient(_zexpr, _tv)     # d(z)/d(t_var),    2x1
                        _den = ca.dot(_gz, _gz) + 1e-12
                        # least-squares projection: the component of the term's
                        # tangent-space gradient that lies along the direction
                        # which actually changes height
                        _acc = _acc + ca.dot(_gt, _gz) / _den
                    _grad_z_exprs[_nm] = _acc
                    _w = float(_wts.get(_nm, 0.0) or 0.0)
                    _grad_z_exprs[_nm + '_raw'] = _acc / _w if _w > 1e-12 else ca.DM(0.0)
                # Evaluated at the seed AFTER the solve, via _opti.debug.value
                # (the expressions depend on q and the wrench variables too, not
                # on t_var alone, so they cannot be lambdified over t_var by
                # itself -- debug.value resolves the whole variable vector).
                # Stashed for the caller to log once the stage has run.
                _grad_z_pending = dict(_grad_z_exprs)

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
            if not _is_mesh and not cfg.fixed_contacts:
                _opti.set_initial(_p1, p1_ws)
                _opti.set_initial(_p2, p2_ws)
            # else (_is_mesh): _p1/_p2 are expressions of _t1_var/_t2_var, already
            # initialized to (0,0) inside _mesh_tangent_contact_ca (i.e. "start
            # exactly at the seed anchor"). (fixed_contacts): _p1/_p2 are DM
            # constants — set_initial on a non-variable raises, so skip entirely.

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
                            # Raw (u,v) decision-variable trajectory — mesh contacts only
                            # (_mesh_tangent_contact_ca / _mesh_uv_local_contact_ca both name
                            # their 2-DOF variable _t1_var/_t2_var; ground truth, cheaper and
                            # more accurate than re-deriving uv from the recorded p1/p2 world
                            # points post-hoc, which would need a triangle re-lookup).
                            if _is_mesh and _t1_var is not None:
                                _rec['uv1'] = np.asarray(_opti.debug.value(_t1_var), float).flatten()
                                _rec['uv2'] = np.asarray(_opti.debug.value(_t2_var), float).flatten()
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
                                               obj_center_np, obj_R_np, geom_size,
                                               mesh_entry=self._mesh_entry)
                        _n2a = _geom_normal_np(_p2v, geom_type,
                                               obj_center_np, obj_R_np, geom_size,
                                               mesh_entry=self._mesh_entry)
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
                    _n1o = _geom_normal_np(_p1v, geom_type, obj_center_np, obj_R_np, geom_size,
                                           mesh_entry=self._mesh_entry)
                    _n2o = _geom_normal_np(_p2v, geom_type, obj_center_np, obj_R_np, geom_size,
                                           mesh_entry=self._mesh_entry)
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
                    # (u,v) trajectory in the local-neighborhood/tangent-plane frame —
                    # see _opti_cb's uv1/uv2 recording note. Frame is per-STAGE (Picard
                    # relinearization rebuilds it each stage — see GraspConfig3D.
                    # use_uv_atlas_contact docstring), so uv1/uv2 are only directly
                    # comparable WITHIN one saved file (one stage), not across stages.
                    if 'uv1' in _iter_rec[0]:
                        _out['uv1'] = np.stack([r['uv1'] for r in _iter_rec])
                        _out['uv2'] = np.stack([r['uv2'] for r in _iter_rec])
                    # Local-quadratic paraboloid params (use_quadratic_contact
                    # only) — one frame per contact per STAGE (same per-stage
                    # scoping as uv1/uv2 above: Picard relinearization rebuilds
                    # the whole paraboloid fit at each stage's new seed, so
                    # these are only valid for reconstructing THIS file's
                    # p1/p2 trajectory, not across stages). Flattened with a
                    # 'quad1_'/'quad2_' prefix per frame key rather than one
                    # nested dict, since np.savez only stores flat arrays.
                    if _t1_frame is not None:
                        for _k, _v in _t1_frame.items():
                            _out[f'quad1_{_k}'] = np.asarray(_v, float)
                    if _t2_frame is not None:
                        for _k, _v in _t2_frame.items():
                            _out[f'quad2_{_k}'] = np.asarray(_v, float)
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

            def _eval_grad_z(value_fn):
                """Per-term d(cost)/d(contact world z) at the solution, or {}.
                Negative => that term is REDUCED by moving the contact UP, i.e.
                it prefers a higher contact. See the _grad_z_exprs comment."""
                if not _grad_z_pending:
                    return {}
                out = {}
                for _nm, _ex in _grad_z_pending.items():
                    try:
                        out[_nm] = float(np.asarray(value_fn(_ex)).squeeze())
                    except Exception:
                        pass
                return out

            def _quad_pinned(value_fn) -> bool:
                """True if either mesh contact's solved (t1,t2) sits within
                cfg.quadratic_pin_frac of its own per-axis SDF-derived bound —
                i.e. the local-quadratic trust region ran out before the
                solver was done pushing that contact, not a freely-converged
                interior optimum. Only meaningful under use_quadratic_contact
                (both _t*_bounds are None otherwise, always returns False).
                Consumed by the Picard relinearization loop below to force
                another stage (fresh seed -> fresh bound) even when position/
                normal-mismatch convergence alone would call it done.
                """
                if _t1_bounds is None and _t2_bounds is None:
                    return False
                frac = float(cfg.quadratic_pin_frac)
                # A bound of ~0 means that axis was given NO freedom to begin
                # with (the seed itself was already at the edge of what
                # _sdf_axis_bound_np considers valid) — not an optimizer that
                # pushed against and used up a real trust region. Comparing
                # |tv| >= frac*bound there is degenerate (0 >= 0 is always
                # true), which would mark every such stage pinned forever and
                # defeat the Picard loop's early-exit unconditionally. Treat
                # a near-zero bound as "nothing to be pinned against" instead.
                _bound_eps = 1e-6
                for var, bounds in ((_t1_var, _t1_bounds), (_t2_var, _t2_bounds)):
                    if var is None or bounds is None:
                        continue
                    tv = np.asarray(value_fn(var), float).reshape(2)
                    for i in range(2):
                        if bounds[i] > _bound_eps and abs(tv[i]) >= frac * bounds[i]:
                            return True
                return False

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
                    # Third contact's solved position (None at n=2). The Picard refresh
                    # in solve() reads this to re-freeze contact 3's normal between
                    # stages, and downstream consumers (verify, the wrench layer) need
                    # it to see the tripod at all.
                    'p3':         (_sol.value(_p3) if _p3 is not None else None),
                    'cost':       float(_sol.value(_opti.f)),
                    'iterations': _sol.stats()['iter_count'],
                    'status':     'converged',
                    'return_status': _sol.stats().get('return_status'),
                    'gamma_nlp':     float(_sol.value(_gamma_lp)) if _gamma_lp is not None else None,
                    'slack_norms':   ([float(np.linalg.norm(_sol.value(sk))) for sk in _s_list]
                                       if _s_list else None),
                    'max_slack_norm': (float(max(np.linalg.norm(_sol.value(sk)) for sk in _s_list))
                                        if _s_list else None),
                    'n1_frozen':     _n1_in.tolist() if _n1_in is not None else None,
                    'n2_frozen':     _n2_in.tolist() if _n2_in is not None else None,
                    'stability_last20': _stab,
                    'gws_beta':      float(_sol.value(_gws_beta)) if _gws_beta is not None else None,
                    'quad_pinned':   _quad_pinned(_sol.value),
                    'grad_z':        _eval_grad_z(_sol.value),
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
                        'p3':         (_opti.debug.value(_p3) if _p3 is not None else None),
                        'cost':       _opti.debug.value(_opti.f) if _st else None,
                        'iterations': (_st or {}).get('iter_count'),
                        'status':     'best-effort',
                        'return_status': (_st or {}).get('return_status'),
                        'gamma_nlp':     float(_opti.debug.value(_gamma_lp)) if _gamma_lp is not None else None,
                        'slack_norms':   ([float(np.linalg.norm(_opti.debug.value(sk))) for sk in _s_list]
                                           if _s_list else None),
                        'max_slack_norm': (float(max(np.linalg.norm(_opti.debug.value(sk)) for sk in _s_list))
                                            if _s_list else None),
                        'n1_frozen':     _n1_in.tolist() if _n1_in is not None else None,
                        'n2_frozen':     _n2_in.tolist() if _n2_in is not None else None,
                        'stability_last20': _stab,
                        'gws_beta':      (float(_opti.debug.value(_gws_beta))
                                           if _gws_beta is not None else None),
                        'quad_pinned':   _quad_pinned(_opti.debug.value),
                        'grad_z':        _eval_grad_z(_opti.debug.value),
                    }
                except Exception as _e2:
                    self.log.error(f"GraspPlanner3D debug extraction: {_e2}")
                    _save_iter_npz('failed')
                    return {'success': False, 'q': None, 'p1': None, 'p2': None,
                            'p3': None,
                            'cost': None, 'iterations': None, 'status': 'failed',
                            'return_status': None,
                            'gamma_nlp': None, 'slack_norms': None, 'max_slack_norm': None,
                            'gws_beta': None, 'quad_pinned': False}

        # ── Run optimisation ─────────────────────────────────────────────────
        # Outer re-linearisation loop.  Contact normals are frozen per NLP solve
        # (opti.parameter — safe for L-BFGS), then updated between solves.
        # Early-exit when the contact positions have converged (position shift
        # < tol_p) or the normal mismatch is negligible (< tol_deg).
        _d1_lp = (np.asarray(d1, float) if d1 is not None
                  else _geom_normal_np(p1_seed, geom_type, obj_center_np, obj_R_np, geom_size,
                                       mesh_entry=self._mesh_entry))
        _d2_lp = (np.asarray(d2, float) if d2 is not None
                  else _geom_normal_np(p2_seed, geom_type, obj_center_np, obj_R_np, geom_size,
                                       mesh_entry=self._mesh_entry))
        _p1_ws, _p2_ws, _q_ws = p1_seed, p2_seed, q_dls
        # THIRD contact (cfg.n_contacts >= 3). Bound from the seed dict that
        # _seed_third_contact produced; None at n=2 so _run_stage's _has_c3 gate stays
        # False and the 2-contact problem is built exactly as before. The frozen normal
        # is derived the same way _d1_lp/_d2_lp are when the seed doesn't carry one.
        _p3_ws = None
        _d3_lp = None
        if int(cfg.n_contacts) >= 3 and p3_seed is not None:
            _p3_ws = p3_seed
            _d3_lp = (np.asarray(d3, float) if d3 is not None
                      else _geom_normal_np(p3_seed, geom_type, obj_center_np, obj_R_np,
                                           geom_size, mesh_entry=self._mesh_entry))
        _tol_p_m   = 5e-4    # 0.5 mm position shift → converged
        _tol_deg   = 2.0     # 2° normal mismatch → normals are accurate enough
        _tol_r_m   = 5e-4    # 0.5 mm directional-r_tip shift → radius is self-consistent
        res = {}
        _best_res  = {}      # best stage result by cost (Picard has no descent guarantee)
        # Box (geom_type 6) needs NO Picard relinearization: the face-pin surface
        # constraint keeps each contact on its seed face, where the inward normal is
        # CONSTANT (a flat face doesn't rotate as the contact slides), so the frozen
        # normal set at seed time is already exact for the whole solve. Curved surfaces
        # (sphere/cylinder) still relinearize since their normals genuinely turn with p.
        _n_relin = 0 if geom_type == 6 else cfg.n_normal_relinearize
        for _ri in range(_n_relin + 1):
            # Directional tip radii, refrozen from the CURRENT warm-start q and
            # this stage's frozen normals (both are constants for the solve about
            # to run, so the support function never enters the NLP symbolically).
            # -d*_lp is the OUTWARD direction: _d*_lp is the object's outward
            # surface normal used as the IK offset direction, and the pad extends
            # from the site back toward the finger, i.e. along -n_out.
            _r1_ov = _r2_ov = None
            if cfg.directional_r_tip:
                _m = float(cfg.directional_r_tip_margin_m)
                _r1_ov = self._tip_support_along('thumb', _q_ws, -_d1_lp, cfg.r_thumb) + _m
                _r2_ov = self._tip_support_along('index', _q_ws, -_d2_lp, cfg.r_index) + _m
                self.log.info(
                    f"[S{_ri+1}|r_tip] directional thumb={_r1_ov*1e3:.2f}mm "
                    f"index={_r2_ov*1e3:.2f}mm  (isotropic {cfg.r_thumb*1e3:.2f}/"
                    f"{cfg.r_index*1e3:.2f}mm, margin {_m*1e3:.1f}mm)")
            res = _run_stage(_q_ws, _p1_ws, _p2_ws,
                             include_surface=True,
                             d1_lp=_d1_lp,
                             d2_lp=_d2_lp,
                             max_iter_override=cfg.max_iter,
                             stage_label=f'S{_ri+1}',
                             iter_callback=iter_callback,
                             update_normals_in_callback=update_normals_in_callback,
                             r1_override=_r1_ov, r2_override=_r2_ov,
                             p3_ws=_p3_ws, d3_lp=_d3_lp)
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
                _p1r, geom_type, obj_center_np, obj_R_np, geom_size, mesh_entry=self._mesh_entry)
            _n2_actual = _geom_normal_np(
                _p2r, geom_type, obj_center_np, obj_R_np, geom_size, mesh_entry=self._mesh_entry)
            _cos1 = float(np.clip(np.dot(_d1_lp, _n1_actual), -1.0, 1.0))
            _cos2 = float(np.clip(np.dot(_d2_lp, _n2_actual), -1.0, 1.0))
            _mismatch_deg = max(np.degrees(np.arccos(_cos1)),
                                np.degrees(np.arccos(_cos2)))

            # 3. Trust-region pin (use_quadratic_contact only): the solved
            #    (t1,t2) sat at/near its per-axis SDF-derived bound this
            #    stage. Small dp/mismatch alone can't distinguish "found an
            #    interior optimum" from "the local quadratic ran out of room
            #    and the wall stopped the search" — treat a pin as NOT
            #    converged even if position/normal checks pass, so the next
            #    stage re-seeds at the pinned point and rebuilds a fresh
            #    (generally larger, since it's moved off the tight direction)
            #    bound there. See GraspConfig3D.quadratic_pin_frac.
            _pinned = bool(res.get('quad_pinned'))

            # 4. Directional-r_tip self-consistency: the radius this stage USED
            #    was computed from the PREVIOUS stage's q, but the pad's
            #    orientation moved during the solve, so the radius its own answer
            #    implies can differ. Measured swings of ~8mm between S1 and S2
            #    (S1's radius comes off the crude DLS seed pose, where the finger
            #    is nowhere near its final orientation) -- converging on a stale
            #    radius reintroduces exactly the gap this feature removes, just
            #    with a different sign. Treat a material shift as NOT converged,
            #    the same way a trust-region pin is, so the next stage re-solves
            #    against the corrected radius. Fixed point: r used == r implied.
            _dr = 0.0
            if cfg.directional_r_tip and res.get('q') is not None and _r1_ov is not None:
                _m = float(cfg.directional_r_tip_margin_m)
                _qr = np.asarray(res['q'])
                _r1_new = self._tip_support_along('thumb', _qr, -_n1_actual, cfg.r_thumb) + _m
                _r2_new = self._tip_support_along('index', _qr, -_n2_actual, cfg.r_index) + _m
                _dr = max(abs(_r1_new - _r1_ov), abs(_r2_new - _r2_ov))

            self.log.info(
                f"[relinearize S{_ri+1}→S{_ri+2}] "
                f"dp={_dp*1e3:.2f}mm  mismatch={_mismatch_deg:.1f}°  pinned={_pinned}"
                + (("  gradz[" + " ".join(
                    f"{k}={v:+.3g}" for k, v in sorted((res.get('grad_z') or {}).items())
                    if abs(v) > 1e-9) + "]") if res.get('grad_z') else "")
                + (f"  dr_tip={_dr*1e3:.2f}mm" if cfg.directional_r_tip else ""))

            if (_dp < _tol_p_m and _mismatch_deg < _tol_deg and not _pinned
                    and _dr < _tol_r_m):
                self.log.info(
                    f"[relinearize] converged after S{_ri+1} "
                    f"(dp={_dp*1e3:.2f}mm, mismatch={_mismatch_deg:.1f}°, "
                    f"dr_tip={_dr*1e3:.2f}mm)")
                break

            # Update normals and warm-start for next solve
            _d1_lp = _n1_actual
            _d2_lp = _n2_actual
            _p1_ws = _p1r
            _p2_ws = _p2r
            # Third contact tracks the same way: re-read its surface normal at the
            # solved point so the next Picard stage freezes an accurate frame. No-op
            # under the standardized preset (n_normal_relinearize=0 -> single stage),
            # kept correct so enabling relinearization later does not silently pin
            # contact 3 at its seed while 1 and 2 move.
            if _p3_ws is not None and res.get('p3') is not None:
                _p3_ws = np.asarray(res['p3'], float)
                _d3_lp = _geom_normal_np(_p3_ws, geom_type, obj_center_np, obj_R_np,
                                         geom_size, mesh_entry=self._mesh_entry)
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
            _n1f  = _geom_normal_np(_p1f, geom_type, obj_center_np, obj_R_np, geom_size,
                                    mesh_entry=self._mesh_entry)
            _n2f  = _geom_normal_np(_p2f, geom_type, obj_center_np, obj_R_np, geom_size,
                                    mesh_entry=self._mesh_entry)
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

        if self._obj_geom_type == _GEOM_TYPE_MESH:
            obj_pos = data_v.xpos[self._obj_bid].copy()
            obj_mat = data_v.xmat[self._obj_bid].reshape(3, 3)
        else:
            obj_pos = data_v.geom_xpos[self._obj_gid].copy()
            obj_mat = data_v.geom_xmat[self._obj_gid].reshape(3, 3)

        # IK residual uses the same offset target as the NLP: p + r·n_out
        # Measuring ‖site − p‖ would always return ~r_tip regardless of convergence.
        _p1_np  = np.asarray(result['p1'], float)
        _p2_np  = np.asarray(result['p2'], float)
        _n1_out = _geom_normal_np(_p1_np, self._obj_geom_type, obj_pos, obj_mat, self._obj_size,
                                  mesh_entry=self._mesh_entry)
        _n2_out = _geom_normal_np(_p2_np, self._obj_geom_type, obj_pos, obj_mat, self._obj_size,
                                  mesh_entry=self._mesh_entry)
        _tgt1   = _p1_np + self.cfg.r_thumb * _n1_out
        _tgt2   = _p2_np + self.cfg.r_index * _n2_out
        ik_t = float(np.linalg.norm(data_v.site_xpos[self._thumb_sid] - _tgt1))
        ik_i = float(np.linalg.norm(data_v.site_xpos[self._index_sid] - _tgt2))

        # NOTE for CoACD (mesh) objects: self._obj_gid is only ONE representative
        # hull among the body's several collision hulls, so these mj_geomDistance
        # gaps can undercount penetration/overcount clearance against a different
        # hull than the one actually nearest a given fingertip. Diagnostic-only
        # (not used by solve/verify's pass/fail), so left as an approximation.
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
            return _geom_sdf_np(p, self._obj_geom_type, obj_pos, obj_mat, self._obj_size,
                                mesh_entry=self._mesh_entry)

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
                                          obj_pos, obj_mat, self._obj_size,
                                          mesh_entry=self._mesh_entry)
                n2_out = _geom_normal_np(p2_np, self._obj_geom_type,
                                          obj_pos, obj_mat, self._obj_size,
                                          mesh_entry=self._mesh_entry)
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
                _mu_v_raw, _ = _contact_friction(model, self._obj_gid, self._thumb_gid, self._index_gid)
                _mu_v    = round(0.8 * _mu_v_raw, 3)
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
            'gws_beta':             result.get('gws_beta'),
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
                f"Check GraspConfig3D thumb_site / index_site / middle_site fields "
                f"(set together by grasp_config_builder.load_finger_config).")
        return sid

    def _optional_site(self, name: str):
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, name)
        return int(sid) if sid != -1 else None


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
                 dashboard=None,
                 seed: int | None = None):
        """
        seed : int | None — the constant every solve() call resets its seed-generation
            RNG to (see the reset in solve(), which explains why a FIXED reset rather
            than a free-running stream is deliberate: continuity under sub-mm pose
            jitter, not just reproducibility). None (default) keeps the original
            hardcoded constant (_SEED_RNG_CONST=42) — existing callers are unaffected.
            Pass an explicit int to run the SAME object/pose through independent seed
            streams, e.g. for an ablation over NLP-seed randomization
            (environments/grasp_bench's RandomizationConfig.nlp_seed).
        """
        self._planner       = GraspPlanner3D(model, data, cfg, logger, log_dir,
                                             dashboard=dashboard)
        self._obj_hx        = self._planner._obj_hx
        self._obj_hy        = self._planner._obj_hy
        self._obj_hz        = self._planner._obj_hz
        self._obj_geom_type = self._planner._obj_geom_type
        self._obj_size      = self._planner._obj_size
        self._mesh_entry    = self._planner._mesh_entry
        self._fk_data       = mj.MjData(model)           # FK queries for seed generation
        self._seed_rng_const = _SEED_RNG_CONST if seed is None else int(seed)
        self._rng           = np.random.default_rng(self._seed_rng_const)
        self.last_chart_rank_table = []   # set by solve() when use_uv_atlas_contact chart-pair seeding runs
        self.last_seed_rank_table = []    # set by solve() when seed_dls_rank_pool > 1
        # Per-seed THIRD-contact fan ranking (cfg.n_contacts >= 3): one row per
        # candidate with its fan angle and middle-finger DLS residual. Empty at n=2.
        self.last_c3_rank_table = []
        # Fingertip effective radii (r_thumb/r_index/r_middle/r_ring) are
        # measured from model geometry inside GraspPlanner3D.__init__ above —
        # nothing left to do here.
        _pl = self._planner

        # Log friction and torque bounds that will be computed inline at solve time.
        _obj_gid = _pl._obj_gid
        _mu_obj, _mu_t_obj = _contact_friction(model, _obj_gid, _pl._thumb_gid, _pl._index_gid)
        _mu_init = round(0.8 * _mu_obj, 3)
        _pl.log.info(
            f"[friction] combined_mu={_mu_obj:.3f} (obj-vs-fingertip max)  "
            f"effective_mu={_mu_init:.3f} (0.8x safety margin)  "
            f"combined_mu_t={_mu_t_obj:.4f}")

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
        if geom_type == _GEOM_TYPE_MESH:
            # See GraspPlanner3D.solve: object_sdf's table is body-frame, so
            # pose must come from the body, not any one hull geom.
            obj_bid       = self._planner._obj_bid
            obj_center_np = self._planner.data.xpos[obj_bid].copy()
            obj_R_np      = self._planner.data.xmat[obj_bid].reshape(3, 3).copy()
        else:
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
        self._rng = np.random.default_rng(self._seed_rng_const)

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
        _mu_raw2, _ = _contact_friction(model, obj_gid, self._planner._thumb_gid, self._planner._index_gid)
        _mu    = round(0.8 * _mu_raw2, 3)
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
            return _geom_sdf_np(p, geom_type, obj_center_np, obj_R_np, geom_size,
                                mesh_entry=self._mesh_entry)

        _ground_z  = cfg.ground_z
        # Seed-gate floor: flat override when set, else the conservative
        # bounding-sphere radius (unknown which tip goes where).
        _r_tip_min = (float(cfg.seed_ground_clearance_m)
                      if cfg.seed_ground_clearance_m is not None
                      else min(cfg.r_thumb, cfg.r_index))

        _t0 = time.perf_counter()

        # ── Fixed canonical seed: perfectly antipodal pair along the object's
        # MINOR PRINCIPAL AXIS (thinnest direction — the natural two-finger
        # pinch axis, same choice as benchmarks/ycb_grasp/ik_demo.py's
        # pinch_targets_from), tried FIRST, ahead of the randomized seeds
        # below (counts against the n_seeds budget). _seed_pair's
        # random-direction search can land on an awkward, hard-to-reach pinch
        # geometry (measured 2-6x worse DLS residual than the minor-axis
        # choice on several YCB objects) — this gives every solve one
        # well-conditioned attempt before falling back to random exploration.
        def _gate_kappa_max(p_l) -> float:
            """Largest-magnitude principal curvature at p_l (object-local) using
            the SAME surface model the solve's surrogate will fit.

            With cfg.quadratic_mesh_fit on, _mesh_quadratic_contact_ca takes its
            curvature from _mesh_local_surface_fit_np (a fit to nearby mesh
            VERTICES), not from the SDF Hessian. Gating on the SDF Hessian while
            fitting the mesh judges seeds by a quantity the solve never uses, and
            the two disagree badly on smooth objects: an SDF encodes distance to
            the WHOLE shape, so scan relief and far-away features bleed into its
            second derivative. Measured on 017_orange (true curvature 1/36.6mm =
            27.6 everywhere, no edges anywhere), 60 random surface points:

                SDF Hessian   median kappa 49.3, 73% above the gate of 40
                mesh fit      median kappa 32.8,  3% above

            i.e. the SDF-Hessian gate rejected 73% of a smooth sphere as
            "too curved to seed on". Gating on the fit admits 97% of it while
            still rejecting 036_wood_block's genuine edges (measured 190-536,
            far above any plausible threshold under either model).

            Falls back to the SDF Hessian exactly where the surrogate does --
            when the fit returns None (mesh too sparse to support one) -- so the
            gate and the patch never disagree about which model is in force.
            Costs ~1.0ms vs ~0.13ms per contact; paid per candidate seed, against
            a multi-second NLP per accepted seed.
            """
            me = self._planner._mesh_entry
            if cfg.quadratic_mesh_fit:
                grad_l = np.asarray(me["grad_fn"](p_l), float).reshape(3)
                gnorm = float(np.linalg.norm(grad_l))
                n_l = grad_l / (gnorm + 1e-9)
                t1_l, t2_l = _tangent_basis_np(n_l)
                _fit = _mesh_local_surface_fit_np(
                    me, p_l, t1_l, t2_l, n_l,
                    radius=cfg.quadratic_mesh_fit_radius,
                    quad_gain_min=cfg.quadratic_mesh_fit_gain_min,
                    grad_norm=gnorm)
                if _fit is not None:
                    # (axis0, axis1, kappa0, kappa1, info) -- the order
                    # _principal_curvature_axes_np returns, matching
                    # _mesh_quadratic_contact_ca's own unpacking.
                    _, _, _k0, _k1, _ = _fit
                    return max(abs(_k0), abs(_k1))
            return _mesh_surface_kappa_max_np(me, p_l)

        def _seed_kappa_ok(s) -> bool:
            """True unless this seed pair fails the mesh curvature gate (see
            seed_kappa_max_reject docstring) — shared by every seed source
            (fixed minor-axis, chart-pair, random _seed_pair) so a near-
            edge/corner point can't slip through whichever source happens to
            fill the n_seeds budget first."""
            if not (geom_type == _GEOM_TYPE_MESH and self._planner._mesh_entry is not None
                    and cfg.seed_kappa_max_reject > 0):
                return True
            p1s_l = obj_R_np.T @ (s['p1s'] - obj_center_np)
            p2s_l = obj_R_np.T @ (s['p2s'] - obj_center_np)
            k1 = _gate_kappa_max(p1s_l)
            k2 = _gate_kappa_max(p2s_l)
            return max(k1, k2) <= cfg.seed_kappa_max_reject

        def _dls_residual(s) -> float:
            """Cheap reachability score for one seed pair: the worst fingertip
            residual a damped-least-squares IK leaves when asked to put both
            tips on this pair's contacts. Milliseconds, no NLP.

            This is the ONLY seed screen that knows about the ARM. The other
            two (_reachable_contact, _seed_kappa_ok) are geometric -- above the
            table, not on an edge -- and a seed can pass both while sitting
            where the arm simply cannot bring a fingertip. That is the measured
            failure mode: planned contacts land sub-millimetre from the true
            surface (|SDF| 0.00-0.65mm) while the fingertip GEOM still stops
            3-9mm away, because the residual is in the kinematics, not the
            surface model.

            Mirrors the chart-pair path's scoring exactly (same SpatialIKSolver,
            same r_thumb/r_index target offsets along the inward normal, same
            q_bias/null_gain), so the two rankings are comparable. Finger
            assignment must already have been applied -- the targets depend on
            which contact is the thumb.
            """
            _d = self._planner._dls_data
            _d.qpos[:] = self._planner.data.qpos[:]
            _d.qpos[act_idx] = np.asarray(q_ref, float)[:len(act_idx)]
            _t1 = s['p1s'] + cfg.r_thumb * (-s['n1_in'])
            _t2 = s['p2s'] + cfg.r_index * (-s['n2_in'])
            self._planner._dls_ik.solve(
                model, _d, [self._planner._thumb_sid, self._planner._index_sid],
                [_t1, _t2], q_bias=q_ref, null_gain=0.3)
            mj.mj_kinematics(model, _d)
            return float(max(
                np.linalg.norm(_d.site_xpos[self._planner._thumb_sid] - _t1),
                np.linalg.norm(_d.site_xpos[self._planner._index_sid] - _t2)))

        seeds, attempts, rejected = [], 0, 0
        # Per-seed REJECT record, so a paired diagnostic can draw the seeds the
        # gates threw away rather than only the survivors (the rejects are the
        # whole story on objects where the gate refuses everything -- see
        # plot_seed_quadratic's --show-rejected). solve() stops at the first
        # n_seeds ACCEPTED, so these are the only rejects that ever materialise.
        self.last_seed_reject_table = []
        _axis_local = _minor_axis_local(geom_type, geom_size, mesh_entry=self._mesh_entry)
        _fs = _fixed_antipodal_seed(geom_type, geom_size, c, obj_R_np, _axis_local,
                                    prefer_outer=cfg.seed_prefer_outer_surface,
                                    mesh_entry=self._mesh_entry)
        if (_reachable_contact(_fs['p1s'], _ground_z, _r_tip_min) and
                _reachable_contact(_fs['p2s'], _ground_z, _r_tip_min) and
                _seed_kappa_ok(_fs)):
            _assign_seed_by_finger(_fs, _live_th, _live_if)
            seeds.append(_fs)
        else:
            self.last_seed_reject_table.append(
                dict(kind='minor-axis', why='unreachable or too-curved',
                     p1s=np.asarray(_fs['p1s'], float).copy(),
                     p2s=np.asarray(_fs['p2s'], float).copy(),
                     n1_in=np.asarray(_fs['n1_in'], float).copy(),
                     n2_in=np.asarray(_fs['n2_in'], float).copy()))
            log.debug(f"[seed_gen] minor-axis seed (local axis {_axis_local.tolist()}) "
                      f"unreachable or too-curved — skipped")

        # ── Chart-aware antipodal seeds (mesh + use_uv_atlas_contact only) ──
        # Ranked by chart-normal antipodality (see _chart_pair_seeds docstring
        # / the session's seeding-rethink notes) — tried NEXT, ahead of
        # _seed_pair's random-direction search, since chart assignment is
        # frozen for the whole solve at seed time (nothing downstream can
        # move a contact to a different chart), making a good chart choice
        # here strictly more valuable than positional randomization within an
        # arbitrarily-chosen chart. Falls back silently to [] (leaving the
        # random loop below as the only source) when the object isn't a mesh,
        # use_uv_atlas_contact is off, or fewer than 2 charts survive the
        # bottom-facing filter.
        #
        # Pure chart-normal antipodality has NO reachability awareness — measured
        # this session on 036_wood_block: the top-scored pair (its two TALL side
        # faces) is geometrically excellent (wrench beta consistently positive)
        # but sits far from the object's base height, which the arm/hand cannot
        # reach (ik cost stayed 2-3 orders of magnitude above a converged solve,
        # err 60-130mm). Re-rank the geometric candidates by a cheap DLS-IK
        # residual (same SpatialIKSolver/target-offset convention _run_stage's
        # warm-start uses — see its "DLS warm-start IK" comment) before
        # accepting: request MORE geometric candidates than the seed budget
        # (over-generate), then keep only the ones DLS can actually reach,
        # best-residual first.
        if cfg.use_uv_atlas_contact and self._planner._uv_atlas is not None:
            _chart_cands = _chart_pair_seeds(
                self._planner._uv_atlas, c, obj_R_np, geom_type, geom_size,
                mesh_entry=self._mesh_entry, top_k=max(n_seeds * 3, 8))

            _dls_data2 = self._planner._dls_data
            _scored_cands = []
            for _cs in _chart_cands:
                if not (_reachable_contact(_cs['p1s'], _ground_z, _r_tip_min) and
                        _reachable_contact(_cs['p2s'], _ground_z, _r_tip_min)):
                    continue
                # Same curvature gate as the fixed minor-axis and random
                # _seed_pair sources (_seed_kappa_ok, defined above) — chart-
                # pair candidates are otherwise ranked only by antipodality +
                # DLS-IK residual, neither of which screens for a centroid
                # sitting at/near a mesh edge or corner. Checked here (not
                # only had it been left to use_quadratic_contact's own
                # per-stage trust region) because these are the HIGHEST-
                # priority seeds (tried before the random fallback), so an
                # ungated near-corner chart-pair seed would be the most
                # likely one to actually win the seed budget.
                if not _seed_kappa_ok(_cs):
                    continue
                # Finger assignment (thumb vs index) BEFORE scoring, not after —
                # _assign_seed_by_finger can swap which physical contact is p1
                # vs p2, which changes the DLS targets _run_stage will actually
                # solve later. Scoring pre-swap measured a DLS problem that
                # wasn't the one actually used, giving a ranking inconsistent
                # with the real per-seed [dls_ws] residuals logged downstream
                # (caught this session: candidates re-ordered after the swap).
                _assign_seed_by_finger(_cs, _live_th, _live_if)
                _dls_data2.qpos[:] = self._planner.data.qpos[:]
                _dls_data2.qpos[act_idx] = np.asarray(q_ref, float)[:len(act_idx)]
                _tgt1 = _cs['p1s'] + cfg.r_thumb * (-_cs['n1_in'])
                _tgt2 = _cs['p2s'] + cfg.r_index * (-_cs['n2_in'])
                _q_dls_c = self._planner._dls_ik.solve(
                    model, _dls_data2,
                    [self._planner._thumb_sid, self._planner._index_sid],
                    [_tgt1, _tgt2], q_bias=q_ref, null_gain=0.3)
                mj.mj_kinematics(model, _dls_data2)
                _e1 = np.linalg.norm(_dls_data2.site_xpos[self._planner._thumb_sid] - _tgt1)
                _e2 = np.linalg.norm(_dls_data2.site_xpos[self._planner._index_sid] - _tgt2)
                _dls_res = float(max(_e1, _e2))
                _scored_cands.append((_dls_res, _cs))

            _scored_cands.sort(key=lambda t: t[0])
            # Full ranked table (chart_pair, antipodal_score, dls_res_mm) for every
            # SCORED candidate (reachable ones only — see the _reachable_contact
            # skip above), stashed on self for diagnostics/plotting (e.g.
            # benchmarks/ycb_grasp/plot_uv_path.py's rank annotations) — not part
            # of the res dict returned by solve() to avoid touching that schema.
            self.last_chart_rank_table = [
                dict(chart_pair=_cs.get('chart_pair'),
                    antipodal_score=_cs.get('antipodal_score'),
                    dls_res_mm=_dls_res * 1e3,
                    accepted=False)
                for _dls_res, _cs in _scored_cands
            ]
            _n_before = len(seeds)
            for _rank, (_dls_res, _cs) in enumerate(_scored_cands):
                if len(seeds) >= n_seeds:
                    break
                seeds.append(_cs)
                self.last_chart_rank_table[_rank]['accepted'] = True
                log.debug(f"[seed_gen] chart-pair {_cs.get('chart_pair')} "
                         f"dls_res={_dls_res*1e3:.1f}mm")
            log.info(f"[seed_gen] {len(seeds) - _n_before}/{len(_chart_cands)} "
                    f"chart-pair seeds accepted (best DLS residual "
                    f"{_scored_cands[0][0]*1e3:.1f}mm)"
                    if _scored_cands else
                    "[seed_gen] 0 chart-pair candidates reachable/scored")

        # ── Random seeds ──────────────────────────────────────────────────
        # When seed_dls_rank_pool > 1, OVER-GENERATE and rank by DLS-IK
        # residual instead of taking the first n_seeds that pass the geometric
        # gates. The gates say a seed is on a sane piece of surface; they say
        # nothing about whether the ARM can reach it, and that is where the
        # measured error actually lives (see _dls_residual's docstring:
        # sub-mm surrogate error, 3-9mm fingertip gaps). Ranking costs one DLS
        # solve per candidate -- milliseconds against a multi-second NLP -- and
        # is the same screen the chart-pair path already applies, which was
        # previously unavailable here because it was gated behind
        # use_uv_atlas_contact.
        _pool_mult = max(int(cfg.seed_dls_rank_pool), 1)
        _rank_random = _pool_mult > 1
        _target = n_seeds * _pool_mult if _rank_random else n_seeds
        _pool = []
        while len(seeds) + len(_pool) < _target and attempts < max_attempts:
            attempts += 1
            s = _seed_pair(geom_type, geom_size, c, obj_R_np, bbox_r, self._rng,
                           prefer_outer=cfg.seed_prefer_outer_surface,
                           delta_max=np.deg2rad(cfg.seed_march_jitter_deg),
                           mesh_entry=self._mesh_entry)
            # # Hemisphere check — n1 and n2 must point into opposing hemispheres
            # if float(np.dot(s['n1_in'], s['n2_in'])) >= 0:
            #     rejected += 1
            #     continue
            # Reachability check — reject contacts below the table or facing downward
            n1_out = -s['n1_in'];  n2_out = -s['n2_in']
            if (not _reachable_contact(s['p1s'], _ground_z, _r_tip_min) or
                    not _reachable_contact(s['p2s'], _ground_z, _r_tip_min)):
                rejected += 1
                self.last_seed_reject_table.append(
                    dict(kind='random', why='unreachable (too near floor)',
                         p1s=np.asarray(s['p1s'], float).copy(),
                         p2s=np.asarray(s['p2s'], float).copy(),
                         n1_in=np.asarray(s['n1_in'], float).copy(),
                         n2_in=np.asarray(s['n2_in'], float).copy()))
                continue
            # Curvature check (mesh only) — reject seeds landing at/near an
            # edge or corner of the SDF's zero level set, where the surface
            # genuinely can't be well-approximated locally regardless of which
            # contact representation is used downstream (analytic primitives
            # have exact closed-form surfaces with no such collapse, so this
            # is skipped for them). See _mesh_surface_kappa_max_np's docstring
            # for how this was found: a near-corner seed left the
            # use_quadratic_contact trust region correctly but uselessly
            # small (<1mm) along the edge-approaching axis, which a better
            # seed avoids by construction rather than needing a larger box.
            if not _seed_kappa_ok(s):
                rejected += 1
                self.last_seed_reject_table.append(
                    dict(kind='random',
                         why=f'kappa > {cfg.seed_kappa_max_reject:.0f}',
                         p1s=np.asarray(s['p1s'], float).copy(),
                         p2s=np.asarray(s['p2s'], float).copy(),
                         n1_in=np.asarray(s['n1_in'], float).copy(),
                         n2_in=np.asarray(s['n2_in'], float).copy()))
                continue
            _assign_seed_by_finger(s, _live_th, _live_if)
            _pool.append(s)

        if _rank_random and _pool:
            # Best-reachable first. Seeds already accepted above (minor-axis,
            # chart-pair) keep their priority -- they have their own rationale
            # for going first and the chart-pair ones are already DLS-ranked.
            _scored = sorted(((_dls_residual(_s), _i, _s)
                              for _i, _s in enumerate(_pool)), key=lambda t: t[:2])
            log.info(f"[seed_gen] DLS-ranked {len(_pool)} random candidates; "
                     f"residuals {_scored[0][0]*1e3:.1f}..{_scored[-1][0]*1e3:.1f}mm, "
                     f"keeping best {max(n_seeds - len(seeds), 0)}")
            self.last_seed_rank_table = [
                dict(dls_res_mm=_r * 1e3, accepted=(_k < max(n_seeds - len(seeds), 0)))
                for _k, (_r, _i, _s) in enumerate(_scored)
            ]
            for _r, _i, _s in _scored:
                if len(seeds) >= n_seeds:
                    break
                seeds.append(_s)
        else:
            seeds.extend(_pool[:max(n_seeds - len(seeds), 0)])

        if len(seeds) < n_seeds:
            log.warning(
                f"[seed_gen] only {len(seeds)}/{n_seeds} valid seeds after "
                f"{attempts} attempts ({rejected} rejected)")
        # Accepted seeds, same schema as the reject table -- the paired seed figure
        # draws both from ONE solve instead of re-deriving them in a second process
        # with its own RNG (which could not be guaranteed to match).
        # p3s/n3_in are recorded WHEN PRESENT. They are attached to the seed dict
        # later (the third-contact fan runs per seed, after this table is first
        # built), so a row carries them only once that seed has been through the
        # fan -- which is exactly the condition under which the figure should draw
        # a middle-finger seed. Absent at n=2, and the figure draws two contacts.
        self.last_seed_accept_table = [
            dict(kind=_s.get('kind', 'random'), why='accepted',
                 p1s=np.asarray(_s['p1s'], float).copy(),
                 p2s=np.asarray(_s['p2s'], float).copy(),
                 n1_in=np.asarray(_s['n1_in'], float).copy(),
                 n2_in=np.asarray(_s['n2_in'], float).copy(),
                 **({'p3s': np.asarray(_s['p3s'], float).copy(),
                     'n3_in': np.asarray(_s['n3_in'], float).copy()}
                    if _s.get('p3s') is not None else {}))
            for _s in seeds]
        log.info(
            f"[seed_gen] {len(seeds)} seeds in {(time.perf_counter()-_t0)*1e3:.1f}ms "
            f"({attempts} attempts, {rejected} rejected, "
            f"{len(self.last_seed_reject_table)} recorded)")

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
                                              geom_type, c, obj_R_np, geom_size,
                                              mesh_entry=self._mesh_entry)
                _wp2 = _project_to_surface_np(np.asarray(warm_contacts[1], float),
                                              geom_type, c, obj_R_np, geom_size,
                                              mesh_entry=self._mesh_entry)
                _wn1 = -_geom_normal_np(_wp1, geom_type, c, obj_R_np, geom_size,
                                        mesh_entry=self._mesh_entry)
                _wn2 = -_geom_normal_np(_wp2, geom_type, c, obj_R_np, geom_size,
                                        mesh_entry=self._mesh_entry)
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

        # Per-seed q_ref restart perturbation (see cfg.qref_restart_sigma_arm/hand's
        # docstring). Drawn from self._rng — the SAME fixed-per-solve RNG stream used
        # for contact seeding above, so a re-solve on a static/near-static pose
        # reproduces the same perturbations (the determinism this class already
        # guarantees for contact seeds, extended to q_ref restarts). Seed 0 (index 0,
        # including the warm-start seed prepended above when present) always gets the
        # operator's UNPERTURBED q_ref — only later seeds explore around it.
        _q_ref_base  = np.asarray(q_ref, float)
        _n_arm       = self._planner._n_arm_joints
        _sigma_arm   = float(cfg.qref_restart_sigma_arm)
        _sigma_hand  = float(cfg.qref_restart_sigma_hand)
        _lo_vec, _hi_vec = self._planner._lo_vec, self._planner._hi_vec

        def _perturbed_q_ref(seed_idx: int) -> np.ndarray:
            if seed_idx == 0 or (_sigma_arm <= 0.0 and _sigma_hand <= 0.0):
                return _q_ref_base
            q0 = _q_ref_base.copy()
            q0[:_n_arm]  += self._rng.normal(0.0, _sigma_arm, _n_arm)
            q0[_n_arm:]  += self._rng.normal(0.0, _sigma_hand, len(q0) - _n_arm)
            return np.clip(q0, _lo_vec, _hi_vec)

        results = []
        for i, seed in enumerate(seeds):
            if dash is not None:
                dash.push({'type': 'active', 'label': f'grasp3d seed {i+1}/{n_seeds}'})
            _q_ref_seed = _perturbed_q_ref(i)

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

            # ── THIRD contact (cfg.n_contacts >= 3): fan -> gate -> DLS-rank ──
            # Attached per seed, because the fan is built around THIS seed's grasp
            # axis. Same gates the pair already passed (_reachable_contact,
            # _seed_kappa_ok), then ranked by the SAME DLS-IK reachability screen the
            # chart-pair path uses -- the only seed screen that knows about the arm.
            # A seed that yields no viable third contact simply stays a 2-contact
            # pinch rather than failing: the tripod is an upgrade, not a precondition.
            if int(cfg.n_contacts) >= 3 and self._planner._middle_sid is not None:
                try:
                    _mf_dir = None
                    _R_palm = None
                    _palm_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'leap_palm')
                    if _palm_bid >= 0:
                        # Palm-FRAME prior (measured from the model's own rest pose:
                        # the middle fingertip lies at [0.692,-0.722,0.002] from the
                        # pinch midpoint, i.e. ~45 deg in the palm's xy-plane). Mapped
                        # through the LIVE palm rotation so the bias follows the hand
                        # instead of being a world-frame constant.
                        _R_palm = self._planner.data.xmat[_palm_bid].reshape(3, 3)
                        _mf_dir = _R_palm @ np.array([0.692, -0.722, 0.002])
                    # fk_probe for the 'kinematic' strategy: pose the hand for THIS
                    # pinch with the same DLS solve the ranking below uses, then read
                    # where the middle fingertip actually ended up. Returns None if
                    # the site is unavailable, which makes the strategy yield no
                    # candidate rather than inventing one.
                    def _mf_fk_probe(_sd):
                        """Where the middle fingertip lands when the hand poses for THIS
                        pinch AND is actually asked to bring that finger to the object.

                        TWO-PHASE, because a 2-target solve does not answer the question.
                        An earlier version solved DLS for thumb+index ONLY and read the
                        middle site out of the result -- but nothing drove that site
                        anywhere: it sat wherever null_gain's pull toward q_ref left it.
                        Measured consequence: on 017_orange it landed 7-15mm from the
                        index contact (looked correct, was luck -- that home pose happens
                        to leave the middle finger there), while on 014_lemon all three
                        seeds put it on unrelated parts of the surface and on
                        036_wood_block it drifted to the THUMB side of the block.

                        Phase 1 poses the hand for the pinch. Phase 2 re-solves with a
                        THIRD target for the middle site: the index contact displaced
                        along the index's own tangent plane, in the direction the middle
                        tip ALREADY lies at the phase-1 pose. That direction is read from
                        the live kinematics rather than a palm-frame constant -- the
                        constant ([0,-1,0], the rigid finger base-mount offset) is genuine
                        but maps to world DOWN at TS.home_qpos(), which is what made the
                        'tangent' strategy propose sub-table contacts on every seed.
                        """
                        _sid3 = self._planner._middle_sid
                        if _sid3 is None:
                            return None
                        _dp = self._planner._dls_data
                        _dp.qpos[:] = self._planner.data.qpos[:]
                        _dp.qpos[act_idx] = np.asarray(q_ref, float)[:len(act_idx)]
                        _ta = _sd['p1s'] + cfg.r_thumb * (-np.asarray(_sd['n1_in'], float))
                        _tb = _sd['p2s'] + cfg.r_index * (-np.asarray(_sd['n2_in'], float))
                        # Phase 1: pinch pose (thumb + index only).
                        self._planner._dls_ik.solve(
                            model, _dp,
                            [self._planner._thumb_sid, self._planner._index_sid],
                            [_ta, _tb], q_bias=q_ref, null_gain=0.3)
                        mj.mj_kinematics(model, _dp)
                        _mf_now = _dp.site_xpos[_sid3].copy()
                        _if_now = _dp.site_xpos[self._planner._index_sid].copy()
                        # Direction index -> middle AT THIS POSE, flattened onto the index
                        # contact's tangent plane so the target stays on the same face.
                        _n2 = np.asarray(_sd['n2_in'], float)
                        _n2 = _n2 / (np.linalg.norm(_n2) + 1e-12)
                        _dir = _mf_now - _if_now
                        _dir = _dir - np.dot(_dir, _n2) * _n2
                        _nd = float(np.linalg.norm(_dir))
                        if _nd < 1e-6:
                            return _mf_now          # degenerate: fall back to phase 1
                        _dir = _dir / _nd
                        # Phase 2: ask for the middle tip one finger-pitch along that
                        # direction from the index contact, on the surface.
                        _r_mf3 = float(cfg.r_middle if cfg.r_middle is not None
                                       else cfg.r_index)
                        _p3_guess = _project_to_surface_np(
                            np.asarray(_sd['p2s'], float) + _MF_PITCH_M * _dir,
                            geom_type, obj_center_np, obj_R_np, geom_size,
                            mesh_entry=self._mesh_entry)
                        if not np.all(np.isfinite(_p3_guess)):
                            return _mf_now
                        _n3_guess = _geom_normal_np(_p3_guess, geom_type, obj_center_np,
                                                    obj_R_np, geom_size,
                                                    mesh_entry=self._mesh_entry)
                        if not np.all(np.isfinite(_n3_guess)) or \
                                np.linalg.norm(_n3_guess) < 1e-9:
                            return _mf_now
                        _n3_guess = _n3_guess / np.linalg.norm(_n3_guess)
                        _tc = _p3_guess + _r_mf3 * _n3_guess
                        _dp.qpos[:] = self._planner.data.qpos[:]
                        _dp.qpos[act_idx] = np.asarray(q_ref, float)[:len(act_idx)]
                        self._planner._dls_ik.solve(
                            model, _dp,
                            [self._planner._thumb_sid, self._planner._index_sid, _sid3],
                            [_ta, _tb, _tc], q_bias=q_ref, null_gain=0.3)
                        mj.mj_kinematics(model, _dp)
                        return _dp.site_xpos[_sid3].copy()

                    _c3_strategy = str(getattr(cfg, 'c3_seed_strategy', 'fan'))
                    # For 'patch_offset': fit contact 2's patch HERE, at seed time,
                    # with the same call _run_stage makes later. The seeder then
                    # traverses that patch's own (t0,t1) coordinates, so the offset
                    # is expressed in the surface's principal-curvature frame and
                    # is clamped to the exact bounds the NLP will confine it to.
                    _c3_patch_frame = None
                    if (_c3_strategy == 'patch_offset'
                            and geom_type == _GEOM_TYPE_MESH
                            and self._mesh_entry is not None):
                        try:
                            _n2_out_seed = -np.asarray(seed['n2_in'], float)
                            _throwaway = ca.Opti()
                            _, _, _, _c3_patch_frame = _mesh_quadratic_contact_ca(
                                _throwaway, np.asarray(seed['p2s'], float),
                                _n2_out_seed, obj_center_np, obj_R_np,
                                self._mesh_entry,
                                t_bound_max=cfg.quadratic_t_bound_max,
                                sdf_err_tol=cfg.quadratic_sdf_err_tol,
                                mesh_fit=cfg.quadratic_mesh_fit,
                                mesh_fit_radius=cfg.quadratic_mesh_fit_radius,
                                mesh_fit_quad_gain_min=cfg.quadratic_mesh_fit_gain_min)
                        except Exception as _pe:
                            log.debug(f"[seed {i+1}] patch fit for c3 seeding failed: {_pe}")
                            _c3_patch_frame = None
                    _c3_cands = _seed_third_contact(
                        seed, geom_type, geom_size, obj_center_np, obj_R_np, self._rng,
                        mesh_entry=self._mesh_entry, mf_dir_world=_mf_dir,
                        n_fan=5, fan_half_deg=40.0,
                        strategy=_c3_strategy,
                        palm_R=(_R_palm if _palm_bid >= 0 else None),
                        fk_probe=_mf_fk_probe,
                        prefer_outer=cfg.seed_prefer_outer_surface,
                        patch_frame=_c3_patch_frame,
                        patch_offset_m=float(getattr(cfg, 'c3_patch_offset_m', 0.030)))
                    log.debug(f"[seed {i+1}] c3 strategy={_c3_strategy} "
                              f"-> {len(_c3_cands)} candidate(s)")
                    _c3_scored = []
                    for _c3 in _c3_cands:
                        if not _reachable_contact(_c3['p3s'], _ground_z, _r_tip_min):
                            continue
                        _dls_data3 = self._planner._dls_data
                        _dls_data3.qpos[:] = self._planner.data.qpos[:]
                        _dls_data3.qpos[act_idx] = np.asarray(q_ref, float)[:len(act_idx)]
                        _r_mf = float(cfg.r_middle if cfg.r_middle is not None
                                      else cfg.r_index)
                        _t1 = _c3['p1s'] + cfg.r_thumb * (-_c3['n1_in'])
                        _t2 = _c3['p2s'] + cfg.r_index * (-_c3['n2_in'])
                        _t3 = _c3['p3s'] + _r_mf * (-_c3['n3_in'])
                        self._planner._dls_ik.solve(
                            model, _dls_data3,
                            [self._planner._thumb_sid, self._planner._index_sid,
                             self._planner._middle_sid],
                            [_t1, _t2, _t3], q_bias=q_ref, null_gain=0.3)
                        mj.mj_kinematics(model, _dls_data3)
                        _e3 = float(np.linalg.norm(
                            _dls_data3.site_xpos[self._planner._middle_sid] - _t3))
                        _c3_scored.append((_e3, _c3))
                    _c3_scored.sort(key=lambda t: t[0])
                    self.last_c3_rank_table = [
                        dict(fan_deg=_c['fan_deg'], mf_dls_res_mm=_e * 1e3,
                             accepted=(_k == 0))
                        for _k, (_e, _c) in enumerate(_c3_scored)]
                    if _c3_scored:
                        _best_c3 = _c3_scored[0][1]
                        seed['p3']    = _best_c3['p3s'].copy()
                        seed['p3s']   = _best_c3['p3s']
                        seed['n3_in'] = _best_c3['n3_in']
                        # Back-fill the already-built accept row for THIS seed so
                        # the seed figure can draw the middle contact on the shared
                        # patch. Matched by p1s identity, not index: the accept
                        # table is built from `seeds` in order, but a seed skipped
                        # by the gamma pre-check never reaches here.
                        for _row in (self.last_seed_accept_table or []):
                            if np.allclose(_row.get('p1s'), seed['p1s'], atol=1e-12):
                                _row['p3s']   = np.asarray(_best_c3['p3s'], float).copy()
                                _row['n3_in'] = np.asarray(_best_c3['n3_in'], float).copy()
                                break
                        log.info(f"[seed {i+1}] third contact: fan={_best_c3['fan_deg']:+.0f}deg "
                                 f"mf_dls={_c3_scored[0][0]*1e3:.1f}mm "
                                 f"({len(_c3_scored)}/{len(_c3_cands)} candidates viable)")
                    else:
                        log.info(f"[seed {i+1}] third contact: no viable candidate "
                                 f"of {len(_c3_cands)} -- staying 2-contact")
                except Exception as _e_c3:
                    log.warning(f"[seed {i+1}] third-contact seeding failed: {_e_c3}")

            # ── Run NLP (warm-started from the pre-check LP's γ and cone y's) ──
            r = self._planner.solve(_q_ref_seed, obj_pos,
                                    p1_init=seed['p1'],
                                    p2_init=seed['p2'],
                                    d1=-seed['n1_in'],   # outward normal
                                    d2=-seed['n2_in'],
                                    p3_init=seed.get('p3'),
                                    d3=(-seed['n3_in'] if seed.get('n3_in') is not None
                                        else None),
                                    gamma_init=g_pre,
                                    y_by_corner_init=y_by_corner_pre)

            # ── Post-solve diagnostics ────────────────────────────────────────
            sdf_p1 = sdf_p2 = float('nan')
            ik_th  = ik_if  = float('nan')
            if r.get('p1') is not None:
                _p1f = np.asarray(r['p1'])
                _p2f = np.asarray(r['p2'])
                sdf_p1 = _geom_sdf_np(_p1f, geom_type, obj_center_np, obj_R_np, geom_size,
                                      mesh_entry=self._mesh_entry)
                sdf_p2 = _geom_sdf_np(_p2f, geom_type, obj_center_np, obj_R_np, geom_size,
                                      mesh_entry=self._mesh_entry)
                if r.get('q') is not None:
                    self._fk_data.qpos[act_idx] = np.asarray(r['q'], float)[:len(act_idx)]
                    mj.mj_kinematics(model, self._fk_data)
                    # Use offset target p + r·n_out to match the NLP objective.
                    _n1_out = _geom_normal_np(_p1f, geom_type, obj_center_np, obj_R_np, geom_size,
                                              mesh_entry=self._mesh_entry)
                    _n2_out = _geom_normal_np(_p2f, geom_type, obj_center_np, obj_R_np, geom_size,
                                              mesh_entry=self._mesh_entry)
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

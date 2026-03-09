"""
grasp_planner.py
================
Self-contained grasp contact-point solver extracted from
contact_points_feasibility_single_stage_v2.py.

All solver logic is verbatim from the working script.
Two public symbols are exported:

    GraspConfig   – mirrors the original Config dataclass
    GraspPlanner  – class with .solve(), .verify(), .show_in_viewer()

Typical usage
-------------
    from grasp_planner import GraspConfig, GraspPlanner

    planner = GraspPlanner(model, data)
    result  = planner.solve(q_ref, obj_pos_2d)   # runs IPOPT, returns dict

    if result["q"] is not None:
        planner.show_in_viewer(model, data, result)
        viewer.sync()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime

import casadi as ca
import mujoco as mj
import numpy as np

# Module-level logger.  If the caller doesn't configure it we emit nothing.
log = logging.getLogger("grasp_planner")
if not log.handlers:
    log.addHandler(logging.NullHandler())


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraspConfig:
    """Mirrors the original Config dataclass.  All defaults are identical."""

    # Cost weights
    w_ik:        float = 0.0
    w_antipodal: float = 1.0
    w_reg:       float = 0.0
    w_proximity: float = 0.0
    w_surface:   float = 0.0

    q_scale:   float = 1.0
    p_scale:   float = 1.0
    sdf_scale: float = 1.0

    joint_limits:           bool = False
    on_object:              bool = True    # hard-constrain p1,p2 on surface
    ik_constraint:          bool = True    # hard-constrain finger to touch p
    penetration_constraint: bool = True    # gap == -1e-4 (just touching)
    max_iter:               int  = 500

    # Geometry / body names – must match your XML
    obj_geom:   str = 'obj1_geom'
    obj_body:   str = 'obj1'
    thumb_geom: str = 'right_thumb_distal_geom'
    index_geom: str = 'right_index_distal_geom'
    cp1_geom:   str = 'cp1'        # dummy mocap sphere – thumb
    cp2_geom:   str = 'cp2'        # dummy mocap sphere – index
    cp1_body:   str = 'cp1_body'
    cp2_body:   str = 'cp2_body'
    site_p1:    str = 'contact_p1' # visualisation site – thumb target
    site_p2:    str = 'contact_p2' # visualisation site – index target


# ─────────────────────────────────────────────────────────────────────────────
# Utilities  (verbatim from original)
# ─────────────────────────────────────────────────────────────────────────────

def _get_actuated_indices(model: mj.MjModel) -> list[int]:
    return [model.jnt_qposadr[model.actuator_trnid[i, 0]]
            for i in range(model.nu)]


def _box_sdf_2d(point, obj_pos, half_x, half_y) -> float:
    d  = np.asarray(point) - np.asarray(obj_pos)
    dx = abs(d[0]) - half_x
    dy = abs(d[1]) - half_y
    return float(np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)
                 + min(max(dx, dy), 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# CasADi callbacks  (verbatim from original, renamed with _ prefix)
# ─────────────────────────────────────────────────────────────────────────────

class _FingerPointDistCallback(ca.Callback):
    """
    (q, p) → mj_geomDistance(finger_geom, dummy_sphere_at_p)

    Returns surface-to-surface signed distance:
        0  => finger surface is touching p
        >0 => gap
        <0 => penetration
    """
    def __init__(self, name, model, data_cb,
                 finger_gid, cp_gid, cp_mocap_id, act_idx, cutoff=5):
        ca.Callback.__init__(self)
        self.model = model; self.data = data_cb
        self.finger_gid = finger_gid; self.cp_gid = cp_gid
        self.cp_mocap_id = cp_mocap_id; self.act_idx = act_idx
        self.cutoff = cutoff
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return (ca.Sparsity.dense(len(self.act_idx), 1) if i == 0
                else ca.Sparsity.dense(2, 1))

    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        q_act = np.array(arg[0]).flatten()
        p     = np.array(arg[1]).flatten()
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val
        self.data.mocap_pos[self.cp_mocap_id] = [p[0], p[1], 0.0]
        mj.mj_forward(self.model, self.data)
        dist = mj.mj_geomDistance(self.model, self.data,
                                   self.finger_gid, self.cp_gid,
                                   self.cutoff, np.zeros(6))
        return [dist]


class _SDFCallback(ca.Callback):
    """(q, point) → SDF of point relative to the object geom."""

    def __init__(self, name, model, data_cb, geom_id, act_idx):
        ca.Callback.__init__(self)
        self.model = model; self.data = data_cb
        self.geom_id = geom_id; self.act_idx = act_idx
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return (ca.Sparsity.dense(len(self.act_idx), 1) if i == 0
                else ca.Sparsity.dense(2, 1))

    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        q_act = np.array(arg[0]).flatten()
        point = np.array(arg[1]).flatten()
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val
        mj.mj_forward(self.model, self.data)
        obj_pos   = self.data.geom_xpos[self.geom_id][:2]
        geom_type = self.model.geom_type[self.geom_id]
        geom_size = self.model.geom_size[self.geom_id]
        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            dist = _box_sdf_2d(point, obj_pos, geom_size[0], geom_size[1])
        else:
            dist = float(np.linalg.norm(point - obj_pos) - geom_size[0])
        return [dist]


class _NonPenetrationCallback(ca.Callback):
    """
    (q,) → mj_geomDistance(finger_geom, obj_geom)  (no dummy sphere)

        >0 => gap (no penetration)
         0 => touching
        <0 => penetrating
    """
    def __init__(self, name, model, data_cb, finger_gid, obj_gid, act_idx,
                 cutoff=0.5):
        ca.Callback.__init__(self)
        self.model = model; self.data = data_cb
        self.finger_gid = finger_gid; self.obj_gid = obj_gid
        self.act_idx = act_idx; self.cutoff = cutoff
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i): return ca.Sparsity.dense(len(self.act_idx), 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        q_act = np.array(arg[0]).flatten()
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val
        mj.mj_forward(self.model, self.data)
        dist = mj.mj_geomDistance(self.model, self.data,
                                   self.finger_gid, self.obj_gid,
                                   self.cutoff, None)
        return [dist]


# ─────────────────────────────────────────────────────────────────────────────
# GraspPlanner
# ─────────────────────────────────────────────────────────────────────────────

class GraspPlanner:
    """
    Wraps the single-stage grasp feasibility solver so it can be imported and
    called from any script without the viewer boilerplate.

    Parameters
    ----------
    model   : mj.MjModel – shared with the rest of the simulation.
    data    : mj.MjData  – shared with the simulation.  The planner reads
              the current qpos/qvel snapshot when solve() is called but uses
              its own internal MjData for all IPOPT evaluations, so the live
              simulation state is never corrupted.
    cfg     : GraspConfig  (optional – defaults used if omitted)
    logger  : logging.Logger (optional)
    log_dir : str – directory for per-solve IPOPT log files (optional)
    """

    def __init__(self,
                 model:   mj.MjModel,
                 data:    mj.MjData,
                 cfg:     GraspConfig | None = None,
                 logger:  logging.Logger | None = None,
                 log_dir: str | None = None):

        self.model   = model
        self.data    = data
        self.cfg     = cfg or GraspConfig()
        self.log     = logger or log
        self.log_dir = log_dir

        # Cache IDs once — raises immediately if XML names are wrong
        c = self.cfg
        self._obj_gid   = self._require_geom(c.obj_geom)
        self._thumb_gid = self._require_geom(c.thumb_geom)
        self._index_gid = self._require_geom(c.index_geom)
        self._cp1_gid   = self._require_geom(c.cp1_geom)
        self._cp2_gid   = self._require_geom(c.cp2_geom)

        cp1_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, c.cp1_body)
        cp2_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, c.cp2_body)
        if cp1_bid == -1 or cp2_bid == -1:
            raise ValueError(
                f"GraspPlanner: cp1_body='{c.cp1_body}' or "
                f"cp2_body='{c.cp2_body}' not found in model.")
        self._cp1_mocap = model.body_mocapid[cp1_bid]
        self._cp2_mocap = model.body_mocapid[cp2_bid]

        self._act_idx = _get_actuated_indices(model)
        self._obj_hx  = model.geom_size[self._obj_gid][0]
        self._obj_hy  = model.geom_size[self._obj_gid][1]

    # ── public API ────────────────────────────────────────────────────────────

    def solve(self,
              q_ref:   np.ndarray,
              obj_pos: np.ndarray) -> dict:
        """
        Run the grasp optimisation (synchronous / blocking).

        Parameters
        ----------
        q_ref   : (nu,) actuated joint angles used as warm-start and
                  regularisation reference.
                  → Pass the output of the ApproachController.
        obj_pos : (2,) object XY position in world frame.
                  → data.xpos[obj_body_id][:2]

        Returns
        -------
        dict
            success    : bool
            q          : (nu,) optimised joint angles  | None on failure
            p1         : (2,)  thumb contact point     | None on failure
            p2         : (2,)  index contact point     | None on failure
            cost       : float | None
            iterations : int   | None
            status     : 'converged' | 'best-effort' | 'failed'
        """
        cfg        = self.cfg
        model      = self.model
        act_idx    = self._act_idx
        obj_center = np.asarray(obj_pos, dtype=float)

        # Isolated MjData – solver never touches the live simulation copy
        data_cb = mj.MjData(model)
        data_cb.qpos[:] = self.data.qpos[:]
        data_cb.qvel[:] = self.data.qvel[:]
        mj.mj_forward(model, data_cb)

        # ── callbacks ────────────────────────────────────────────────────────
        ik_thumb_cb = _FingerPointDistCallback(
            'gp_ik_thumb', model, data_cb,
            self._thumb_gid, self._cp1_gid, self._cp1_mocap, act_idx)
        ik_index_cb = _FingerPointDistCallback(
            'gp_ik_index', model, data_cb,
            self._index_gid, self._cp2_gid, self._cp2_mocap, act_idx)
        sdf_cb = _SDFCallback(
            'gp_sdf', model, data_cb, self._obj_gid, act_idx)
        nonpen_thumb_cb = _NonPenetrationCallback(
            'gp_np_thumb', model, data_cb,
            self._thumb_gid, self._obj_gid, act_idx)
        nonpen_index_cb = _NonPenetrationCallback(
            'gp_np_index', model, data_cb,
            self._index_gid, self._obj_gid, act_idx)

        # ── optimisation problem ─────────────────────────────────────────────
        opti = ca.Opti()
        q  = opti.variable(model.nu)
        p1 = opti.variable(2)
        p2 = opti.variable(2)

        d_thumb   = ik_thumb_cb(q, p1)
        d_index   = ik_index_cb(q, p2)
        sdf1      = sdf_cb(q, p1)
        sdf2      = sdf_cb(q, p2)
        gap_thumb = nonpen_thumb_cb(q)
        gap_index = nonpen_index_cb(q)

        # cost (identical to original solve())
        cost = cfg.w_ik * (d_thumb**2 + d_index**2)
        if not cfg.on_object:
            cost += cfg.w_surface * ((sdf1 / cfg.sdf_scale)**2 +
                                     (sdf2 / cfg.sdf_scale)**2)
        d1 = p1 - obj_center
        d2 = p2 - obj_center
        cost += cfg.w_antipodal * ca.fmax(0, ca.dot(d1, d2))**2
        cost += cfg.w_reg       * ca.sumsqr((q - q_ref) / cfg.q_scale)
        cost += cfg.w_proximity * (ca.sumsqr((p1 - obj_center) / cfg.p_scale) +
                                   ca.sumsqr((p2 - obj_center) / cfg.p_scale))
        opti.minimize(cost)

        # constraints (identical to original solve())
        if cfg.joint_limits:
            for i in range(model.nu):
                jid = model.actuator_trnid[i, 0]
                if model.jnt_limited[jid]:
                    opti.subject_to(opti.bounded(
                        model.jnt_range[jid, 0], q[i], model.jnt_range[jid, 1]))

        if cfg.on_object:
            opti.subject_to(sdf1 == 0)
            opti.subject_to(sdf2 == 0)

        if cfg.ik_constraint:
            opti.subject_to(d_thumb == 0)
            opti.subject_to(d_index == 0)

        if cfg.penetration_constraint:
            opti.subject_to(gap_thumb == -1e-4)
            opti.subject_to(gap_index == -1e-4)

        # warm-start
        opti.set_initial(q,  q_ref)
        opti.set_initial(p1, obj_center + np.array([-self._obj_hx, 0.0]))
        opti.set_initial(p2, obj_center + np.array([+self._obj_hx, 0.0]))

        # IPOPT options
        ipopt_opts: dict = {
            'jacobian_approximation': 'finite-difference-values',
            'hessian_approximation':  'limited-memory',
            'max_iter':               cfg.max_iter,
            'sb':                     'no',
            'tol':                    1e-6,
            'dual_inf_tol':           1.0,
            'constr_viol_tol':        1e-8,
            'print_level':            0,
        }
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ipopt_opts['output_file']      = os.path.join(
                self.log_dir, f"grasp_ipopt_{ts}.log")
            ipopt_opts['file_print_level'] = 5

        opti.solver('ipopt', {'ipopt': ipopt_opts, 'print_time': False})

        # ── solve ────────────────────────────────────────────────────────────
        try:
            sol = opti.solve()
            return {
                'success':    True,
                'q':          sol.value(q),
                'p1':         sol.value(p1),
                'p2':         sol.value(p2),
                'cost':       float(sol.value(opti.f)),
                'iterations': sol.stats()['iter_count'],
                'status':     'converged',
            }
        except Exception as e:
            self.log.warning(f"GraspPlanner.solve: {e}")
            try:
                return {
                    'success':    False,
                    'q':          opti.debug.value(q),
                    'p1':         opti.debug.value(p1),
                    'p2':         opti.debug.value(p2),
                    'cost':       None,
                    'iterations': None,
                    'status':     'best-effort',
                }
            except Exception as e2:
                self.log.error(f"GraspPlanner.solve debug extraction: {e2}")
                return {
                    'success': False, 'q': None, 'p1': None, 'p2': None,
                    'cost': None, 'iterations': None, 'status': 'failed',
                }

    def verify(self, result: dict) -> dict:
        """
        Post-solve sanity check.  Logs and returns IK errors, non-penetration
        gaps, and SDF values.  Safe to call on any result dict.
        """
        if result.get('q') is None:
            self.log.warning("GraspPlanner.verify: result has no q, skipping.")
            return {}

        model  = self.model
        data_v = mj.MjData(model)
        data_v.qpos[:]                    = self.data.qpos[:]
        data_v.qpos[self._act_idx]        = result['q']
        data_v.mocap_pos[self._cp1_mocap] = [result['p1'][0], result['p1'][1], 0.0]
        data_v.mocap_pos[self._cp2_mocap] = [result['p2'][0], result['p2'][1], 0.0]
        mj.mj_forward(model, data_v)

        ik_t  = mj.mj_geomDistance(model, data_v,
                                    self._thumb_gid, self._cp1_gid, 0.5, None)
        ik_i  = mj.mj_geomDistance(model, data_v,
                                    self._index_gid, self._cp2_gid, 0.5, None)
        gap_t = mj.mj_geomDistance(model, data_v,
                                    self._thumb_gid, self._obj_gid,  0.5, None)
        gap_i = mj.mj_geomDistance(model, data_v,
                                    self._index_gid, self._obj_gid,  0.5, None)
        obj_p = data_v.geom_xpos[self._obj_gid][:2]
        s1    = _box_sdf_2d(result['p1'], obj_p, self._obj_hx, self._obj_hy)
        s2    = _box_sdf_2d(result['p2'], obj_p, self._obj_hx, self._obj_hy)

        info = {
            'ik_thumb_mm':  ik_t  * 1000,
            'ik_index_mm':  ik_i  * 1000,
            'gap_thumb_mm': gap_t * 1000,
            'gap_index_mm': gap_i * 1000,
            'sdf_p1_mm':    s1    * 1000,
            'sdf_p2_mm':    s2    * 1000,
        }
        self.log.info(
            f"[GraspPlanner.verify] "
            f"IK=({ik_t*1e3:.2f},{ik_i*1e3:.2f})mm  "
            f"GAP=({gap_t*1e3:+.2f},{gap_i*1e3:+.2f})mm  "
            f"SDF=({s1*1e3:.2f},{s2*1e3:.2f})mm"
        )
        return info

    def show_in_viewer(self,
                       model:  mj.MjModel,
                       data:   mj.MjData,
                       result: dict) -> None:
        """
        Write the solver result into *data* so the passive viewer shows the
        proposed grasp.  Call viewer.sync() immediately after this.
        """
        if result.get('q') is None:
            return

        data.qpos[self._act_idx]           = result['q']
        data.mocap_pos[self._cp1_mocap]    = [result['p1'][0], result['p1'][1], 0.0]
        data.mocap_pos[self._cp2_mocap]    = [result['p2'][0], result['p2'][1], 0.0]
        mj.mj_forward(model, data)

        for site_name, pt in [(self.cfg.site_p1, result['p1']),
                               (self.cfg.site_p2, result['p2'])]:
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
            if sid != -1:
                model.site_pos[sid] = [pt[0], pt[1], 0.0]

    # ── private ───────────────────────────────────────────────────────────────

    def _require_geom(self, name: str) -> int:
        gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
        if gid == -1:
            raise ValueError(
                f"GraspPlanner: geom '{name}' not found in model.  "
                f"Check GraspConfig geometry name fields.")
        return gid
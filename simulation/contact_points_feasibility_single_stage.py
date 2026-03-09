"""
Single-Stage Grasp Feasibility Optimization
============================================

Variables:
    q       -- actuated joint angles (8 DOF)
    p1, p2  -- 2D contact points on object surface

IK cost uses mj_geomDistance(finger_last_link_geom, dummy_sphere_at_p):
    - Surface-to-surface distance, driven to zero
    - No capsule radius math, no normals, no FingertipCallback
"""

import casadi as ca
from casadi import Callback
import numpy as np
import mujoco as mj
from dataclasses import dataclass
import logging
import sys
import os
from datetime import datetime
from numpy.linalg import norm


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logger():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path    = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    logger = logging.getLogger("grasp_opt")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    if hasattr(ch.stream, 'errors'):
        ch.stream = open(ch.stream.fileno(), mode='w',
                         encoding='utf-8', errors='replace', closefd=False)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger, log_path


log = logging.getLogger("grasp_opt")


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    # Cost weights
    w_ik:        float = 0.0   # finger surface to p (mj_geomDistance)
    w_antipodal: float = 0.0
    w_reg:       float = 0.0
    w_proximity: float = 1.0
    w_surface:   float = 0.0

    q_scale:   float = 1.0
    p_scale:   float = 1.0
    sdf_scale: float = 1.0

    joint_limits:           bool  = False
    on_object:              bool  = True
    ik_constraint:           bool  = True
    penetration_constraint: bool  = True
    max_iter:               int   = 500
    print_level:            int   = 5

    # Contact analysis: run every N iterations (0 = disabled during solve)
    contact_analysis_every: int   = 10

    # Geometry / body names (match XML)
    obj_geom:   str = 'obj1_geom'
    obj_body:   str = 'obj1'
    thumb_geom: str = 'right_thumb_distal_geom'
    index_geom: str = 'right_index_distal_geom'
    cp1_geom:   str = 'cp1'    # dummy sphere for thumb distance query
    cp2_geom:   str = 'cp2'    # dummy sphere for index distance query
    cp1_body:   str = 'cp1_body'
    cp2_body:   str = 'cp2_body'
    site_p1:    str = 'contact_p1'
    site_p2:    str = 'contact_p2'


# ============================================================
# UTILITIES
# ============================================================

def get_actuated_indices(model):
    return [model.jnt_qposadr[model.actuator_trnid[i, 0]]
            for i in range(model.nu)]


def box_sdf_2d(point, obj_pos, half_x, half_y):
    """2D signed distance from point to axis-aligned box."""
    d  = point - obj_pos
    dx = abs(d[0]) - half_x
    dy = abs(d[1]) - half_y
    return float(np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)
                 + min(max(dx, dy), 0.0))


# ============================================================
# MUJOCO CALLBACKS
# ============================================================

class FingerPointDistCallback(ca.Callback):
    """
    CasADi callback: (q (nu,), p (2,)) -> surface-to-surface signed distance (1,)

    Places a tiny mocap dummy sphere at p, runs mj_forward, then calls
    mj_geomDistance(finger_geom, dummy_geom).

    Returns the distance between the SURFACE of the finger geom and point p:
        dist = 0   =>  finger surface is touching p  (contact)
        dist > 0   =>  gap
        dist < 0   =>  penetration

    No capsule radius, no normals required.
    """

    def __init__(self, name, model, data_cb,
                 finger_gid, cp_gid, cp_mocap_id, act_idx,
                 cutoff=5):
        ca.Callback.__init__(self)
        self.model       = model
        self.data        = data_cb
        self.finger_gid  = finger_gid
        self.cp_gid      = cp_gid
        self.cp_mocap_id = cp_mocap_id
        self.act_idx     = act_idx
        self.cutoff      = cutoff
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

        # Set joint angles
        for idx, val in zip(self.act_idx, q_act):
            self.data.qpos[idx] = val

        # Move dummy sphere to p — mocap bodies are not overwritten by mj_forward
        self.data.mocap_pos[self.cp_mocap_id] = [p[0], p[1], 0.0]

        mj.mj_forward(self.model, self.data)

        # Surface-to-surface signed distance
        fromto = np.zeros(6, dtype=np.float64)
        dist = mj.mj_geomDistance(self.model, self.data,
                                   self.finger_gid, self.cp_gid,
                                   self.cutoff, fromto)
        log.debug(f"  eval {self.name()}: finger = {self.finger_gid}, cp = {self.cp_gid}, q_act={np.round(q_act,3)} p={np.round(p,4)} dist={dist*1000:.2f}mm")

        return [dist]


class SDFCallback(ca.Callback):
    """
    CasADi callback: (q (nu,), point (2,)) -> SDF of point to object geom (1,).
    Used to constrain p1, p2 to lie on the object surface.
    """

    def __init__(self, name, model, data_cb, geom_id, act_idx):
        Callback.__init__(self)
        self.model   = model
        self.data    = data_cb
        self.geom_id = geom_id
        self.act_idx = act_idx
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
            dist = box_sdf_2d(point, obj_pos, geom_size[0], geom_size[1])
        elif geom_type == mj.mjtGeom.mjGEOM_SPHERE:
            dist = float(np.linalg.norm(point - obj_pos) - geom_size[0])
        else:
            dist = float(np.linalg.norm(point - obj_pos) - geom_size[0])

        return [dist]


class NonPenetrationCallback(ca.Callback):
    """
    CasADi callback: (q (nu,),) -> mj_geomDistance(finger_geom, obj_geom) (1,)

    Directly measures the signed gap between a finger link geom and the object
    geom — with NO dummy sphere involved.

        dist > 0  =>  gap between surfaces (no penetration)
        dist = 0  =>  surfaces just touching
        dist < 0  =>  penetration (violation)

    Use as an inequality constraint:  dist >= 0
    or as a soft penalty:             w * ca.fmin(0, dist)**2
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
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(len(self.act_idx), 1)

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
        log.debug(f"  eval {self.name()}: "
                  f"finger={self.finger_gid} obj={self.obj_gid} "
                  f"dist={dist*1000:.3f}mm")
        return [dist]


# ============================================================
# PER-ITERATION CONTACT ANALYSIS (lightweight)
# ============================================================

def contact_analysis_iter(model, q_iter, p1_iter, p2_iter,
                           q_snapshot, cfg: Config, iteration: int):
    """
    Run a full contact analysis snapshot at a given solver iteration.
    Logs results to both terminal and log file via the module-level logger.

    Parameters
    ----------
    model       : MjModel
    q_iter      : ndarray  -- actuated joint angles at this iteration
    p1_iter     : ndarray  -- thumb contact point (2D)
    p2_iter     : ndarray  -- index contact point (2D)
    q_snapshot  : ndarray  -- full qpos vector used as base (non-actuated DOFs)
    cfg         : Config
    iteration   : int      -- current solver iteration number
    """
    obj_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.obj_geom)
    thumb_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.thumb_geom)
    index_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.index_geom)
    cp1_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.cp1_geom)
    cp2_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.cp2_geom)
    cp1_mocap = model.body_mocapid[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.cp1_body)]
    cp2_mocap = model.body_mocapid[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.cp2_body)]
    obj_half_x = model.geom_size[obj_gid][0]
    obj_half_y = model.geom_size[obj_gid][1]
    act_idx    = get_actuated_indices(model)

    # Build a fresh MjData for this snapshot — never touches the viewer's data
    data_i = mj.MjData(model)
    data_i.qpos[:]            = q_snapshot
    data_i.qpos[act_idx]      = q_iter
    data_i.mocap_pos[cp1_mocap] = [p1_iter[0], p1_iter[1], 0.0]
    data_i.mocap_pos[cp2_mocap] = [p2_iter[0], p2_iter[1], 0.0]
    data_i.qvel[:]             = 0.0
    mj.mj_forward(model, data_i)

    # --- IK distances (finger surface -> target point) ---
    d_thumb = mj.mj_geomDistance(model, data_i, thumb_gid, cp1_gid, 0.5, None)
    d_index = mj.mj_geomDistance(model, data_i, index_gid, cp2_gid, 0.5, None)

    # --- SDF errors ---
    obj_pos = data_i.geom_xpos[obj_gid][:2]
    sdf_p1  = box_sdf_2d(p1_iter, obj_pos, obj_half_x, obj_half_y)
    sdf_p2  = box_sdf_2d(p2_iter, obj_pos, obj_half_x, obj_half_y)

    # --- MuJoCo contact detection ---
    relevant_pairs = {
        (thumb_gid, obj_gid): "thumb-obj",
        (obj_gid, thumb_gid): "thumb-obj",
        (index_gid, obj_gid): "index-obj",
        (obj_gid, index_gid): "index-obj",
    }

    contact_summary = []
    for ci in range(data_i.ncon):
        con    = data_i.contact[ci]
        g1, g2 = con.geom[0], con.geom[1]
        label  = relevant_pairs.get((g1, g2))
        if label is None:
            continue
        frame  = con.frame.reshape(3, 3)
        normal = frame[0, :2]
        force_6d = np.zeros(6)
        mj.mj_contactForce(model, data_i, ci, force_6d)
        fn = force_6d[0]
        ft = np.linalg.norm(force_6d[1:3])
        contact_summary.append({
            'label':  label,
            'pos':    con.pos[:2].copy(),
            'normal': normal.copy(),
            'dist_mm': con.dist * 1000,
            'fn':     fn,
            'ft':     ft,
        })

    # --- Emit to log (and therefore terminal via StreamHandler) ---
    sep = "-" * 50
    log.info(sep)
    log.info(f"ITER {iteration:4d}  CONTACT ANALYSIS")
    log.info(f"  IK  thumb->p1 : {d_thumb*1000:+8.3f} mm  (0=contact)")
    log.info(f"  IK  index->p2 : {d_index*1000:+8.3f} mm  (0=contact)")
    log.info(f"  SDF p1        : {sdf_p1*1000:+8.3f} mm  (0=on surface)")
    log.info(f"  SDF p2        : {sdf_p2*1000:+8.3f} mm  (0=on surface)")
    log.info(f"  MuJoCo contacts detected: {data_i.ncon}  "
             f"(relevant: {len(contact_summary)})")

    if contact_summary:
        for c in contact_summary:
            log.info(f"    [{c['label']}]  pos={np.round(c['pos'],4)}  "
                     f"normal={np.round(c['normal'],3)}  "
                     f"pen={c['dist_mm']:.3f}mm  "
                     f"fn={c['fn']:.4f}N  ft={c['ft']:.4f}N")
    else:
        log.info("    (no relevant finger-object contacts in MuJoCo yet)")

    log.info(sep)

    return {
        'd_thumb': d_thumb,
        'd_index': d_index,
        'sdf_p1':  sdf_p1,
        'sdf_p2':  sdf_p2,
        'contacts': contact_summary,
    }


# ============================================================
# SOLVER
# ============================================================

def solve(model, data, object_state, q_ref, cfg: Config):

    obj_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.obj_geom)
    thumb_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.thumb_geom)
    index_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.index_geom)
    cp1_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.cp1_geom)
    cp2_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.cp2_geom)
    cp1_bid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.cp1_body)
    cp2_bid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.cp2_body)

    cp1_mocap_id = model.body_mocapid[cp1_bid]
    cp2_mocap_id = model.body_mocapid[cp2_bid]

    obj_half_x = model.geom_size[obj_gid][0]
    obj_half_y = model.geom_size[obj_gid][1]
    obj_center = object_state['position']
    act_idx    = get_actuated_indices(model)

    log.info(f"  obj half_extents: x={obj_half_x:.3f}  y={obj_half_y:.3f}")
    log.info(f"  cp1_gid={cp1_gid}  cp1_mocap_id={cp1_mocap_id}")
    log.info(f"  cp2_gid={cp2_gid}  cp2_mocap_id={cp2_mocap_id}")

    # Dedicated MjData for callbacks — never shared with viewer
    data_cb = mj.MjData(model)
    data_cb.qpos[:] = data.qpos[:]
    data_cb.qvel[:] = data.qvel[:]
    mj.mj_forward(model, data_cb)

    # Snapshot of full qpos for per-iteration contact analysis
    q_snapshot_cb = data.qpos.copy()

    # ---- Callbacks ----
    ik_thumb_cb = FingerPointDistCallback(
        'ik_thumb', model, data_cb,
        thumb_gid, cp1_gid, cp1_mocap_id, act_idx)

    ik_index_cb = FingerPointDistCallback(
        'ik_index', model, data_cb,
        index_gid, cp2_gid, cp2_mocap_id, act_idx)

    sdf_cb = SDFCallback('sdf', model, data_cb, obj_gid, act_idx)

    # Non-penetration callbacks: finger geom <-> object geom directly
    nonpen_thumb_cb = NonPenetrationCallback(
        'nonpen_thumb', model, data_cb, thumb_gid, obj_gid, act_idx)
    nonpen_index_cb = NonPenetrationCallback(
        'nonpen_index', model, data_cb, index_gid, obj_gid, act_idx)



    opti = ca.Opti()

    # ---- Decision variables ----
    q  = opti.variable(model.nu)   # joint angles
    p1 = opti.variable(2)          # contact point thumb
    p2 = opti.variable(2)          # contact point index

    # ---- IK: surface-to-surface distance via mj_geomDistance ----
    d_thumb = ik_thumb_cb(q, p1)   # 0 = finger surface touching p1
    d_index = ik_index_cb(q, p2)

    cost = cfg.w_ik * (d_thumb**2 + d_index**2)

    # ---- Surface: p1, p2 on object surface ----
    sdf1 = sdf_cb(q, p1)
    sdf2 = sdf_cb(q, p2)

    # ---- Non-penetration: finger geom vs object geom (direct, no dummy) ----
    # gap > 0 => clear, gap = 0 => touching, gap < 0 => penetrating
    gap_thumb = nonpen_thumb_cb(q)
    gap_index = nonpen_index_cb(q)

    if not cfg.on_object:
        cost += cfg.w_surface * ((sdf1 / cfg.sdf_scale)**2 +
                                 (sdf2 / cfg.sdf_scale)**2)

    # ---- Antipodal ----
    d1 = p1 - obj_center
    d2 = p2 - obj_center
    cost += cfg.w_antipodal * ca.fmax(0, ca.dot(d1, d2))**2

    # ---- Regularization ----
    cost += cfg.w_reg * ca.sumsqr((q - q_ref) / cfg.q_scale)

    # ---- Proximity: prevent p drifting ----
    cost += cfg.w_proximity * (ca.sumsqr((p1 - obj_center) / cfg.p_scale) +
                               ca.sumsqr((p2 - obj_center) / cfg.p_scale))

    opti.minimize(cost)

    # ---- Joint limits ----
    if cfg.joint_limits:
        log.info("Joint limits:")
        for i in range(model.nu):
            joint_id = model.actuator_trnid[i, 0]
            if model.jnt_limited[joint_id]:
                lo = model.jnt_range[joint_id, 0]
                hi = model.jnt_range[joint_id, 1]
                opti.subject_to(opti.bounded(lo, q[i], hi))
                log.info(f"  act[{i}] [{lo:.3f}, {hi:.3f}]")
            else:
                log.info(f"  act[{i}] no limit")

    # ---- On-surface constraint ----
    if cfg.on_object:
        opti.subject_to(sdf1 == 0)
        opti.subject_to(sdf2 == 0)

    if cfg.ik_constraint:
        opti.subject_to(d_thumb == 0)
        opti.subject_to(d_index == 0)

    if cfg.penetration_constraint:
        opti.subject_to(gap_thumb == -1e-4)
        opti.subject_to(gap_index == -1e-4)

    # ---- Initial values ----
    p1_init = obj_center + np.array([-obj_half_x, 0.0])
    p2_init = obj_center + np.array([+obj_half_x, 0.0])

    opti.set_initial(q,  q_ref)
    opti.set_initial(p1, p1_init)
    opti.set_initial(p2, p2_init)

    log.info(f"  q_init  = {np.round(q_ref, 3)}")
    log.info(f"  p1_init = {np.round(p1_init, 4)}")
    log.info(f"  p2_init = {np.round(p2_init, 4)}")

    # ---- Iteration callback ----
    def iteration_callback(i):
        try:
            q_val        = opti.debug.value(q)
            p1_val       = opti.debug.value(p1)
            p2_val       = opti.debug.value(p2)
            d_thumb_val  = float(opti.debug.value(d_thumb))
            d_index_val  = float(opti.debug.value(d_index))
            sdf1_val     = float(opti.debug.value(sdf1))
            sdf2_val     = float(opti.debug.value(sdf2))
            gap_thumb_val = float(opti.debug.value(gap_thumb))
            gap_index_val = float(opti.debug.value(gap_index))
            cost_val     = float(opti.debug.value(cost))

            rec = {
                'iter':           i,
                'cost_total':     cost_val,
                'cost_ik':        cfg.w_ik * (d_thumb_val**2 + d_index_val**2),
                'cost_ik_thumb':  cfg.w_ik * d_thumb_val**2,
                'cost_ik_index':  cfg.w_ik * d_index_val**2,
                'd_thumb_mm':        d_thumb_val * 1000,    # finger surface -> p1 (m)
                'd_index_mm':        d_index_val * 1000,
                'gap_thumb_mm':   gap_thumb_val * 1000,  # finger geom -> obj geom (mm)
                'gap_index_mm':   gap_index_val * 1000,
                'sdf1_mm':           sdf1_val * 1000,
                'sdf2_mm':           sdf2_val * 1000,
                'q':              q_val.tolist(),
                'p1':             p1_val.tolist(),
                'p2':             p2_val.tolist(),
            }
            log.debug(rec)

            # ---- Per-iteration contact analysis ----
            every = cfg.contact_analysis_every
            if every > 0 and (i % every == 0):
                contact_analysis_iter(
                    model,
                    q_val,
                    np.array(p1_val),
                    np.array(p2_val),
                    q_snapshot_cb,
                    cfg,
                    i,
                )

        except Exception:
            pass

    opti.callback(iteration_callback)

    # ---- IPOPT options ----
    ipopt_log = next((h.baseFilename for h in log.handlers
                      if isinstance(h, logging.FileHandler)), None)
    ipopt_opts = {
        'jacobian_approximation': 'finite-difference-values',
        'hessian_approximation':  'limited-memory',
        'max_iter':               cfg.max_iter,
        'sb':                     'no',
        'tol':         1e-6,
        'dual_inf_tol': 1,
        'constr_viol_tol': 1e-8,
    }

    if ipopt_log:
        ipopt_opts['output_file']      = ipopt_log
        ipopt_opts['file_print_level'] = 5
        ipopt_opts['print_level']      = 0

    opti.solver('ipopt', {'ipopt': ipopt_opts, 'print_time': True})

    try:
        sol = opti.solve()
        return {
            'success':    True,
            'q':          sol.value(q),
            'p1':         sol.value(p1),
            'p2':         sol.value(p2),
            'cost':       sol.value(opti.f),
            'iterations': sol.stats()['iter_count'],
        }
    except Exception as e:
        log.warning(f"Solver note: {e}")
        try:
            return {
                'success':    False,
                'q':          opti.debug.value(q),
                'p1':         opti.debug.value(p1),
                'p2':         opti.debug.value(p2),
                'cost':       None,
                'iterations': None,
            }
        except Exception as e2:
            log.error(f"Debug extraction failed: {e2}")
            return {'success': False, 'q': None, 'p1': None, 'p2': None}


# ============================================================
# VERIFICATION
# ============================================================

def verify(model, result, q_snapshot, cfg: Config):
    """
    Post-solve: report mj_geomDistance(finger, dummy_at_p) and SDF errors.
    """
    obj_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.obj_geom)
    thumb_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.thumb_geom)
    index_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.index_geom)
    cp1_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.cp1_geom)
    cp2_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.cp2_geom)
    cp1_mocap = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.cp1_body)]
    cp2_mocap = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.cp2_body)]
    obj_half_x = model.geom_size[obj_gid][0]
    obj_half_y = model.geom_size[obj_gid][1]
    act_idx    = get_actuated_indices(model)

    p1 = result['p1']
    p2 = result['p2']

    data_v = mj.MjData(model)
    data_v.qpos[:]       = q_snapshot
    data_v.qpos[act_idx] = result['q']
    data_v.mocap_pos[cp1_mocap] = [p1[0], p1[1], 0.0]
    data_v.mocap_pos[cp2_mocap] = [p2[0], p2[1], 0.0]
    mj.mj_forward(model, data_v)

    dist_thumb = mj.mj_geomDistance(model, data_v, thumb_gid, cp1_gid, 0.5, None)
    dist_index = mj.mj_geomDistance(model, data_v, index_gid, cp2_gid, 0.5, None)
    gap_thumb  = mj.mj_geomDistance(model, data_v, thumb_gid, obj_gid,  0.5, None)
    gap_index  = mj.mj_geomDistance(model, data_v, index_gid, obj_gid,  0.5, None)

    obj_pos = data_v.geom_xpos[obj_gid][:2]
    sdf_p1  = box_sdf_2d(p1, obj_pos, obj_half_x, obj_half_y)
    sdf_p2  = box_sdf_2d(p2, obj_pos, obj_half_x, obj_half_y)

    log.info(f"\n{'='*55}")
    log.info("VERIFICATION")
    log.info(f"{'='*55}")
    log.info(f"  p1 (thumb target): {np.round(p1, 4)}")
    log.info(f"  p2 (index target): {np.round(p2, 4)}")
    log.info(f"  IK thumb->p1 (dummy sphere): {dist_thumb*1000:.3f} mm  (0 = contact)")
    log.info(f"  IK index->p2 (dummy sphere): {dist_index*1000:.3f} mm  (0 = contact)")
    log.info(f"  GAP thumb-obj  (direct geom): {gap_thumb*1000:+.3f} mm  (>=0 = no penetration)")
    log.info(f"  GAP index-obj  (direct geom): {gap_index*1000:+.3f} mm  (>=0 = no penetration)")
    log.info(f"  SDF p1: {sdf_p1*1000:.3f} mm   SDF p2: {sdf_p2*1000:.3f} mm  (0 = on surface)")

    return data_v



# ============================================================
# CONTACT FORCES
# ============================================================

def compute_contact_forces(model, data_viz, cfg: Config, result):
    obj_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.obj_geom)
    thumb_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.thumb_geom)
    index_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.index_geom)

    data_c = mj.MjData(model)
    data_c.qpos[:] = data_viz.qpos[:]
    data_c.qvel[:] = 0.0
    mj.mj_forward(model, data_c)

    log.info("")
    log.info("=" * 55)
    log.info("CONTACT ANALYSIS")
    log.info("=" * 55)
    log.info(f"  Total contacts detected: {data_c.ncon}")

    if data_c.ncon == 0:
        log.info("  No contacts detected.")
        return

    relevant_pairs = {
        (thumb_gid, obj_gid): "thumb-object",
        (obj_gid, thumb_gid): "thumb-object",
        (index_gid, obj_gid): "index-object",
        (obj_gid, index_gid): "index-object",
    }

    found_relevant = False
    for i in range(data_c.ncon):
        con    = data_c.contact[i]
        g1, g2 = con.geom[0], con.geom[1]
        label  = relevant_pairs.get((g1, g2))
        pos    = con.pos[:2]
        dist   = con.dist
        frame  = con.frame.reshape(3, 3)
        normal = frame[0, :2]

        force_6d = np.zeros(6)
        mj.mj_contactForce(model, data_c, i, force_6d)
        fn = force_6d[0]
        ft = np.linalg.norm(force_6d[1:3])

        if label is not None:
            found_relevant = True
            log.info(f"  [{label}]")
            log.info(f"    Contact pos:    {np.round(pos, 4)}")
            log.info(f"    Normal:         {np.round(normal, 4)}")
            log.info(f"    Penetration:    {dist*1000:.3f} mm  "
                     f"({'penetrating' if dist < 0 else 'gap'})")
            log.info(f"    Normal force:   {fn:.4f} N")
            log.info(f"    Friction force: {ft:.4f} N")
        else:
            g1n = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1) or str(g1)
            g2n = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2) or str(g2)
            log.info(f"  [other] {g1n} <-> {g2n} | "
                     f"dist={dist*1000:.2f}mm | fn={fn:.3f}N")

    if not found_relevant:
        log.info("  No thumb-object or index-object contacts found.")


# ============================================================
# GLOBAL SEARCH
# ============================================================

def solve_global(model, data, object_state, cfg: Config):
    obj_gid    = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.obj_geom)
    obj_half_x = model.geom_size[obj_gid][0]
    obj_half_y = model.geom_size[obj_gid][1]
    act_idx    = get_actuated_indices(model)

    cp1_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.cp1_geom)
    cp2_gid   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.cp2_geom)
    cp1_mocap = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.cp1_body)]
    cp2_mocap = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.cp2_body)]
    thumb_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.thumb_geom)
    index_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, cfg.index_geom)

    base_y_vals = np.linspace(-0.3, 0.0, 3)
    curl_vals   = np.linspace(-0.5, -1.5, 3)
    q_base      = np.array([data.qpos[i] for i in act_idx])

    q_inits = []
    for by in base_y_vals:
        for curl in curl_vals:
            q = q_base.copy()
            q[1] = by;    q[2] = curl;   q[3] = curl;  q[4] = curl
            q[5] = -curl; q[6] = -curl;  q[7] = -curl
            q_inits.append(q)

    log.info(f"Global search: {len(q_inits)} initial conditions...")

    best_result = None
    best_cost   = np.inf
    results_all = []

    for k, q_init in enumerate(q_inits):
        try:
            result = solve(model, data, object_state, q_init, cfg)
        except Exception as e:
            log.warning(f"  [{k+1:2d}/{len(q_inits)}] solve() raised exception: {e} — skipping")
            continue

        if result['q'] is None:
            continue

        p1 = result['p1']
        p2 = result['p2']

        data_v = mj.MjData(model)
        data_v.qpos[:]       = data.qpos.copy()
        data_v.qpos[act_idx] = result['q']
        data_v.mocap_pos[cp1_mocap] = [p1[0], p1[1], 0.0]
        data_v.mocap_pos[cp2_mocap] = [p2[0], p2[1], 0.0]
        mj.mj_forward(model, data_v)

        d_thumb = mj.mj_geomDistance(model, data_v, thumb_gid, cp1_gid, 0.5, None)
        d_index = mj.mj_geomDistance(model, data_v, index_gid, cp2_gid, 0.5, None)

        obj_pos = data_v.geom_xpos[obj_gid][:2]
        sdf1    = box_sdf_2d(p1, obj_pos, obj_half_x, obj_half_y)
        sdf2    = box_sdf_2d(p2, obj_pos, obj_half_x, obj_half_y)
        ik_err  = abs(float(d_thumb)) + abs(float(d_index))

        results_all.append({
            'k': k, 'q_init': q_init, 'result': result,
            'ik_thumb': d_thumb, 'ik_index': d_index, 'ik_err': ik_err, 'sdf1': sdf1, 'sdf2': sdf2,
        })

        cost_str  = f"{result['cost']:.4f}" if result['cost'] is not None else "N/A"
        iters_str = str(result['iterations']) if result['iterations'] is not None else "N/A"
        log.info(f"  [{k+1:2d}/{len(q_inits)}] "
                 f"iters={iters_str}  "
                 f"cost={cost_str}  "
                 f"d_thumb={d_thumb*1000:.1f}mm  "
                 f"d_index={d_index*1000:.1f}mm  "
                 f"sdf=({sdf1*1000:.1f},{sdf2*1000:.1f})mm  "
                 f"{'converged' if result['success'] else 'best-effort'}")

        if ik_err < best_cost:
            best_cost   = ik_err
            best_result = result

    # log.info(f"\nGlobal search results (sorted by IK error):")
    # for r in sorted(results_all, key=lambda x: x['ik_err'])[:5]:
    #     log.info(f"  [{r['k']+1}] ik_err={r['ik_err']*1000:.1f}mm  "
    #              f"cost={r['result']['cost']:.4f}  "
    #              f"q_init={np.round(r['q_init'], 2)}")

    if best_result is None:
        log.error("All initial conditions failed.")
    else:
        log.info(f"\nBest: ik_err={best_cost*1000:.1f}mm")
    return best_result


# ============================================================
# MAIN
# ============================================================

def main():
    global log
    log, log_path = setup_logger()

    log.info("=" * 55)
    log.info("SINGLE-STAGE GRASP OPTIMIZATION")
    log.info("=" * 55)

    cfg   = Config()
    model = mj.MjModel.from_xml_path("models/planar_two_finger_manipulator.xml")
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)

    obj_bid      = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.obj_body)
    obj_pos      = data.xpos[obj_bid][:2].copy()
    object_state = {"position": obj_pos}
    log.info(f"Object @ {np.round(obj_pos, 4)}")

    act_idx    = get_actuated_indices(model)
    q_ref      = np.array([data.qpos[i] for i in act_idx])
    q_snapshot = data.qpos.copy()
    log.info(f"q_ref = {np.round(q_ref, 3)}")

    import mujoco.viewer
    import threading

    data.qpos[:] = q_snapshot
    mj.mj_forward(model, data)

    log.info("=" * 55)
    log.info("Controls:  SPACE=solve  R=reset  G=global search  Close=quit")
    log.info(f"Per-iteration contact analysis every {cfg.contact_analysis_every} iters "
             f"({'disabled' if cfg.contact_analysis_every == 0 else 'enabled'})")
    log.info("=" * 55)

    solve_requested  = threading.Event()
    reset_requested  = threading.Event()
    global_requested = threading.Event()

    def key_callback(keycode):
        if keycode == 32:
            log.info("SPACE -- queuing solve...")
            solve_requested.set()
        elif keycode == ord("R"):
            log.info("R -- queuing reset...")
            reset_requested.set()
        elif keycode == ord('G'):
            log.info("G -- global search...")
            global_requested.set()

    with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback) as viewer:

        viewer.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
        viewer.sync()

        while viewer.is_running():

            # ---- RESET ----
            if reset_requested.is_set():
                reset_requested.clear()
                data.qpos[:] = q_snapshot
                mj.mj_forward(model, data)
                try:
                    sid1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, cfg.site_p1)
                    sid2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, cfg.site_p2)
                    model.site_pos[sid1] = [0.0, 0.0, -1.0]
                    model.site_pos[sid2] = [0.0, 0.0, -1.0]
                except Exception:
                    pass
                viewer.sync()
                log.info("Reset to keyframe.")

            # ---- SOLVE ----
            if solve_requested.is_set():
                solve_requested.clear()

                q_current        = np.array([data.qpos[i] for i in act_idx])
                q_snap_now       = data.qpos.copy()
                mj.mj_forward(model, data)
                obj_pos_now      = data.xpos[obj_bid][:2].copy()
                object_state_now = {"position": obj_pos_now}

                log.info(f"Solving from q = {np.round(q_current, 3)}")
                result = solve(model, data, object_state_now, q_current, cfg)

                if result["q"] is None:
                    log.error("Solver failed.")
                else:
                    status = "converged" if result["success"] else "best-effort"
                    iters  = result['iterations'] if result['iterations'] is not None else "N/A"
                    cost   = f"{result['cost']:.5f}" if result['cost'] is not None else "N/A"
                    log.info(f"  {status} | {iters} iters | cost={cost}")
                    log.info(f"  q  = {np.round(result['q'], 4)}")
                    log.info(f"  p1 = {np.round(result['p1'], 4)}")
                    log.info(f"  p2 = {np.round(result['p2'], 4)}")

                    data.qpos[:]       = q_snap_now
                    data.qpos[act_idx] = result["q"]
                    mj.mj_forward(model, data)

                    p1, p2 = result["p1"], result["p2"]
                    try:
                        sid1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, cfg.site_p1)
                        sid2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, cfg.site_p2)
                        model.site_pos[sid1] = [p1[0], p1[1], 0.0]
                        model.site_pos[sid2] = [p2[0], p2[1], 0.0]
                    except Exception as e:
                        log.warning(f"Contact sites: {e}")

                    viewer.sync()
                    data_v = verify(model, result, q_snap_now, cfg)
                    compute_contact_forces(model, data_v, cfg, result)
                    log.info("Press SPACE to re-solve, R to reset.")

            # ---- GLOBAL ----
            if global_requested.is_set():
                global_requested.clear()

                q_snap_now       = data.qpos.copy()
                mj.mj_forward(model, data)
                obj_pos_now      = data.xpos[obj_bid][:2].copy()
                object_state_now = {"position": obj_pos_now}

                log.info("=" * 55)
                log.info("GLOBAL SEARCH")
                log.info("=" * 55)
                result = solve_global(model, data, object_state_now, cfg)

                if result is None:
                    log.error("Global search failed.")
                else:
                    status = "converged" if result["success"] else "best-effort"
                    log.info(f"  {status} | cost={result['cost']:.5f}")
                    log.info(f"  q  = {np.round(result['q'], 3)}")
                    log.info(f"  p1 = {np.round(result['p1'], 4)}")
                    log.info(f"  p2 = {np.round(result['p2'], 4)}")

                    data.qpos[:]       = q_snap_now
                    data.qpos[act_idx] = result["q"]
                    mj.mj_forward(model, data)

                    p1, p2 = result["p1"], result["p2"]
                    try:
                        sid1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, cfg.site_p1)
                        sid2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, cfg.site_p2)
                        model.site_pos[sid1] = [p1[0], p1[1], 0.0]
                        model.site_pos[sid2] = [p2[0], p2[1], 0.0]
                    except Exception as e:
                        log.warning(f"Contact sites: {e}")

                    viewer.sync()
                    data_v = verify(model, result, q_snap_now, cfg)
                    compute_contact_forces(model, data_v, cfg, result)
                    log.info("Global search done.")

            viewer.sync()

    log.info("Viewer closed.")



if __name__ == "__main__":
    main()
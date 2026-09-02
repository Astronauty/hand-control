"""
Compare IK convergence with DLS warm-start across three solver variants:

  A (baseline):  fmax SDF  + IPOPT + jacobian_approximation="finite-difference-values"
  B (analytic):  softplus SDF + IPOPT/exact + analytic FK Jacobians
  C (sqp):       softplus SDF + sqpmethod/qpOASES + analytic FK Jacobians

Variant A is the current production configuration.  Variant B smooths the SDF
kinks so CasADi's chain-rule AD is valid — but L-BFGS still needs more
iterations to handle constraint activation events.  Variant C uses SQP, which
re-linearizes constraints at every step (QP subproblem), which is better suited
to exact Jacobians with inequality-heavy NLPs.

Expected FK/iter reduction for B and C: ~8-12× vs A (1 eval + 1 Jacobian call
per geom vs N_ROBOT+1=24 FD perturbations per geom).

Usage:
    cd /home/aipexws5/daniel/hand-control
    python benchmarks/test_softplus_convergence.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import casadi as ca
import mujoco as mj

import grasp_control.constrained_ik as cik_mod
from grasp_control.constrained_ik import (
    ConstrainedIKSolver, _softplus,
    _SitePositionJacCallback, _SiteAxisJacCallback, _GeomPositionJacCallback,
    check_analytic_jacobians,
)
from grasp_control.ik import SpatialIKSolver
from kinova_common.constants import FINGER_SET, FINGER_CODE, FINGER_TIP_SITES, GEN3_XML
from kinova_leap_pick_place import _randomize_objects

# ---------------------------------------------------------------------------
# Softplus SDF variants (replace fmax → softplus in outside term only)
# ---------------------------------------------------------------------------

def _softplus_box(p_arm, arm_radius, box_center, box_R, half_extents):
    p_local = box_R.T @ (p_arm - box_center)
    q       = ca.fabs(p_local) - half_extents
    outside = ca.sqrt(ca.sumsqr(_softplus(q)) + 1e-12)
    inside  = ca.fmin(ca.fmax(ca.fmax(q[0], q[1]), q[2]), 0)
    return outside + inside - arm_radius


def _softplus_cyl(p_arm, arm_radius, cyl_center, cyl_R, cyl_radius, cyl_halfheight):
    p       = cyl_R.T @ (p_arm - cyl_center)
    radial  = ca.sqrt(p[0]*p[0] + p[1]*p[1] + 1e-12)
    dr      = radial - cyl_radius
    dz      = ca.fabs(p[2]) - cyl_halfheight
    outside = ca.sqrt(_softplus(dr)**2 + _softplus(dz)**2 + 1e-12)
    inside  = ca.fmin(ca.fmax(dr, dz), 0)
    return outside + inside - arm_radius


# ---------------------------------------------------------------------------
# SQP solver options (variant C)
# OSQP is used instead of qpOASES because with 497 inequality constraints and
# 23 DOFs the QP subproblem is often primal-infeasible at the starting point
# (many collision constraints violated simultaneously).  qpOASES fails hard in
# that case; OSQP (ADMM) finds the minimum-constraint-violation direction and
# lets sqpmethod continue.
# ---------------------------------------------------------------------------

_SQP_SOLVER_OPTS = {
    'print_time':            False,
    'qpsol':                 'osqp',
    # error_on_fail=False: let sqpmethod continue when the QP subproblem is
    # infeasible at the starting point (497 constraints, 23 DOFs — the
    # linearised constraints can be overdetermined when many are simultaneously
    # violated).  OSQP returns the minimum-violation direction in that case.
    'qpsol_options':         {'error_on_fail': False,
                              'osqp': {'verbose': False, 'polish': True}},
    'max_iter':              500,
    'hessian_approximation': 'limited-memory',
    'lbfgs_memory':          20,
    'convexify_strategy':    'regularize',
    'print_iteration':       False,
    'print_header':          False,
    'print_status':          False,
}


# ---------------------------------------------------------------------------
# Analytic-Jacobian FK callback wrappers
#
# These replace the enable_fd=True callbacks for variant B.  Each wraps a
# companion JacCallback (which calls mj_jacSite / mj_jac) and exposes it via
# has_jacobian()/get_jacobian() so CasADi uses the analytic Jacobian in its
# chain-rule differentiation of the NLP instead of FD-perturbing the callback.
#
# get_jacobian() must return a ca.Function (ca.Callback is a subclass) that
# takes (all_inputs..., all_outputs...) and returns (J,) where J = d(out)/d(in).
# For our 1-in-1-out callbacks: inputs=(q,), outputs=(p,), J shape=(3, n_robot).
# The companion callbacks already implement exactly this signature.
#
# self._jac_cb is held as an instance variable to prevent GC before IPOPT is
# done with it — CasADi holds a raw pointer to the Python object.
# ---------------------------------------------------------------------------

class _SitePositionCallbackAnalytic(ca.Callback):
    def __init__(self, name, model, site_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._sid   = site_id
        self._n     = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self._jac_cb = _SitePositionJacCallback(name + "_J", model, site_id, n_robot, obj_qpos)
        self.construct(name, {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, 1)
    def has_jacobian(self):        return True
    def get_jacobian(self, name, inames, onames, opts): return self._jac_cb

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        return [self._data.site_xpos[self._sid].copy()]


class _SiteAxisCallbackAnalytic(ca.Callback):
    def __init__(self, name, model, site_id, local_axis, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model     = model
        self._data      = mj.MjData(model)
        self._sid       = site_id
        self._localaxis = np.asarray(local_axis, dtype=float).flatten()
        self._n         = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self._jac_cb = _SiteAxisJacCallback(name + "_J", model, site_id, local_axis, n_robot, obj_qpos)
        self.construct(name, {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, 1)
    def has_jacobian(self):        return True
    def get_jacobian(self, name, inames, onames, opts): return self._jac_cb

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        R = self._data.site_xmat[self._sid].reshape(3, 3)
        return [R @ self._localaxis]


class _GeomPositionCallbackAnalytic(ca.Callback):
    def __init__(self, name, model, geom_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._gid   = geom_id
        self._n     = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self._jac_cb = _GeomPositionJacCallback(name + "_J", model, geom_id, n_robot, obj_qpos)
        self.construct(name, {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, 1)
    def has_jacobian(self):        return True
    def get_jacobian(self, name, inames, onames, opts): return self._jac_cb

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        return [self._data.geom_xpos[self._gid].copy()]


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

MODEL_PATH = 'models/scene_pick_place.xml'
N_ROBOT    = 23

OBJECT_DEFS = [
    ({'index': 'obj_red_box_c2',        'thumb': 'obj_red_box_c1'},        'obj_red_box'),
    ({'index': 'obj_red_sphere_c2',     'thumb': 'obj_red_sphere_c1'},     'obj_red_sphere'),
    ({'index': 'obj_blue_cylinder_c2',  'thumb': 'obj_blue_cylinder_c1'},  'obj_blue_cylinder'),
    ({'index': 'obj_blue_capsule_c2',   'thumb': 'obj_blue_capsule_c1'},   'obj_blue_capsule'),
    ({'index': 'obj_green_box_c2',      'thumb': 'obj_green_box_c1'},      'obj_green_box'),
    ({'index': 'obj_green_cylinder_c2', 'thumb': 'obj_green_cylinder_c1'}, 'obj_green_cylinder'),
]

OBJ_GEOM_NAMES = [
    'obj_red_box_geom', 'obj_red_sphere_geom',
    'obj_blue_cylinder_geom', 'obj_blue_capsule_geom',
    'obj_green_box_geom', 'obj_green_cylinder_geom',
    'floor',
]

PREGRASP_OFFSET = 0.02


def build_scene():
    model = mj.MjModel.from_xml_path(MODEL_PATH)
    data  = mj.MjData(model)
    for j in (0, 2, 4, 6):
        model.jnt_range[j] = [-np.pi, np.pi]
    _randomize_objects(model, data, np.random.default_rng(42))
    mj.mj_forward(model, data)

    home_model = mj.MjModel.from_xml_path(GEN3_XML)
    HOME_ARM   = home_model.key('home').qpos[:7].copy()
    Q_BIAS     = np.zeros(N_ROBOT)
    Q_BIAS[:7]    = HOME_ARM
    Q_BIAS[11:15] = [1.2, 0.0, 0.5, 0.5]
    Q_BIAS[15:19] = [1.2, 0.0, 0.5, 0.5]

    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
            for f in FINGER_SET]

    objects = []
    for contact_sites, body_name in OBJECT_DEFS:
        obj = {
            'id_S':    [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, contact_sites[f])
                        for f in FINGER_SET],
            'id_body': mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name),
            'id_geom': mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, body_name + '_geom'),
            'body_name': body_name,
        }
        obj['p_S_W']      = [data.site_xpos[sid].copy() for sid in obj['id_S']]
        obj['inward_S_W'] = [data.site_xmat[sid].reshape(3, 3)[:, 0].copy()
                             for sid in obj['id_S']]
        objects.append(obj)

    _geom_contype0  = model.geom_contype.copy()
    active_prefixes = tuple(f'leap_{FINGER_CODE[f]}_' for f in FINGER_CODE)
    robot_geom_names = []
    for gi in range(model.ngeom):
        gname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gi)
        if not gname or _geom_contype0[gi] == 0:
            continue
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ''
        if any(bname.startswith(p) for p in active_prefixes) or bname in ('leap_palm', 'bracelet_link'):
            robot_geom_names.append(gname)

    active_finger_geoms = {g for g in robot_geom_names
                           if any(g.startswith(f'leap_{FINGER_CODE[f]}_') for f in FINGER_SET)}

    def _active_obj_clearance(g):
        if '_ds_' in g or g.endswith('_tip'):   return -1.0
        if g.startswith(('leap_if_md', 'leap_th_px')): return -0.010
        return 0.002

    active_clearance_by_geom = {g: _active_obj_clearance(g) for g in active_finger_geoms}
    return model, data, Q_BIAS, id_C, objects, robot_geom_names, active_clearance_by_geom


# ---------------------------------------------------------------------------
# Single IK run — patches module-level names for the analytic variant
# ---------------------------------------------------------------------------

def run_one(model, data, Q_BIAS, id_C, obj,
            robot_geom_names, active_clearance_by_geom,
            dls_ik, constrained_ik, variant):
    """
    variant: 'fd'       — fmax SDF, IPOPT, jacobian_approximation=finite-difference-values (baseline)
             'analytic' — softplus SDF, IPOPT, analytic FK Jacobians, jacobian_approximation=exact
             'sqp'      — softplus SDF, sqpmethod/qpOASES, analytic FK Jacobians
    """
    # --- save originals (whole dicts so sqp can swap cleanly) ---
    orig_box         = cik_mod._sphere_box_distance
    orig_cyl         = cik_mod._sphere_cylinder_distance
    orig_site        = cik_mod._SitePositionCallback
    orig_axis        = cik_mod._SiteAxisCallback
    orig_geom        = cik_mod._GeomPositionCallback
    orig_batched     = cik_mod._BatchedGeomPositionCallback
    orig_solver_name = constrained_ik._solver_name
    orig_solver_opts = constrained_ik._solver_opts

    try:
        if variant in ('analytic', 'sqp'):
            cik_mod._sphere_box_distance      = _softplus_box
            cik_mod._sphere_cylinder_distance = _softplus_cyl
            cik_mod._SitePositionCallback     = _SitePositionCallbackAnalytic
            cik_mod._SiteAxisCallback         = _SiteAxisCallbackAnalytic
            cik_mod._GeomPositionCallback     = _GeomPositionCallbackAnalytic
            cik_mod._BatchedGeomPositionCallback = cik_mod._BatchedGeomPositionCallbackAnalytic

        if variant == 'analytic':
            constrained_ik._solver_name = 'ipopt'
            constrained_ik._solver_opts = {
                **orig_solver_opts,
                'ipopt': {**orig_solver_opts['ipopt'], 'jacobian_approximation': 'exact'},
            }
        elif variant == 'sqp':
            constrained_ik._solver_name = 'sqpmethod'
            constrained_ik._solver_opts = _SQP_SOLVER_OPTS

        ik_data = mj.MjData(model)
        mj.mj_resetData(model, ik_data)
        ik_data.qpos[:N_ROBOT] = Q_BIAS
        ik_data.qpos[N_ROBOT:] = data.qpos[N_ROBOT:]
        mj.mj_forward(model, ik_data)

        t0     = time.time()
        q_dls  = dls_ik.solve(model, ik_data, id_C, obj['p_S_W'],
                               q_bias=Q_BIAS, null_gain=0.3)
        dls_ms = (time.time() - t0) * 1e3

        t0 = time.time()
        constrained_ik.solve(ik_data, id_C, obj['p_S_W'],
                             q_bias=Q_BIAS, q_init=q_dls,
                             reduced_clearance_geoms=active_clearance_by_geom,
                             inward_dirs=obj['inward_S_W'])
        solve_ms = (time.time() - t0) * 1e3

    finally:
        cik_mod._sphere_box_distance      = orig_box
        cik_mod._sphere_cylinder_distance = orig_cyl
        cik_mod._SitePositionCallback     = orig_site
        cik_mod._SiteAxisCallback         = orig_axis
        cik_mod._GeomPositionCallback     = orig_geom
        cik_mod._BatchedGeomPositionCallback = orig_batched
        constrained_ik._solver_name       = orig_solver_name
        constrained_ik._solver_opts       = orig_solver_opts

    m      = constrained_ik.last_metrics
    errs   = m.get('site_err_mm', [float('nan')] * len(id_C))
    max_mm = max(errs) if errs else float('nan')
    iters  = m.get('iters', 0)
    total_fk = m.get('total_fk_calls', 0)
    fk_per_iter = (total_fk // iters) if isinstance(iters, int) and iters > 0 else 0
    return {
        'dls_ms':       dls_ms,
        'solve_ms':     solve_ms,
        'iters':        iters,
        'status':       m.get('status', '?'),
        'site_err_mm':  errs,
        'max_mm':       max_mm,
        'total_fk':     total_fk,
        'fk_per_iter':  fk_per_iter,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building scene...")
    model, data, Q_BIAS, id_C, objects, robot_geom_names, active_clearance_by_geom = build_scene()
    print(f"N_ROBOT={N_ROBOT}  FINGER_SET={FINGER_SET}  "
          f"n_arm_geoms={len(robot_geom_names)}  n_obj_geoms={len(OBJ_GEOM_NAMES)}")

    # --- validate analytic Jacobians before benchmarking ---
    print("\n--- Analytic Jacobian validation (check_analytic_jacobians) ---")
    site_ids = id_C
    geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g) for g in robot_geom_names[:5]]
    pad_ok = check_analytic_jacobians(model, N_ROBOT, site_ids, geom_ids,
                                      pad_axis=(-1., 0., 0.), eps=1e-6, atol=1e-3)
    if not pad_ok:
        print("[WARNING] Analytic Jacobian check FAILED — results for 'analytic' variant may be wrong")
    else:
        print("[OK] Analytic Jacobians validated")

    dls_ik = SpatialIKSolver(n_robot=N_ROBOT)
    constrained_ik = ConstrainedIKSolver(
        model, N_ROBOT,
        arm_geom_names=robot_geom_names,
        obj_geom_names=OBJ_GEOM_NAMES,
        clearance=0.005,
        posture_weight=0.0005,
        pad_axis=(-1.0, 0.0, 0.0),
        orient_weight=0.01,
        max_iter=500,
        verbose=False,
    )

    VARIANTS = [
        ('fd',      'fmax+FD (baseline)'),
        ('analytic','softplus+exact'),
        ('sqp',     'softplus+SQP'),
    ]

    print(f"\n{'Object':<22} {'Variant':<22} {'DLS':>6} {'Solve':>7} {'Iters':>6}  "
          f"{'Status':<26}  {'MaxTip':>9}  FK/iter")
    print("-" * 112)

    results = {}
    for obj in objects:
        bname = obj['body_name']
        row   = {}
        for vkey, vlabel in VARIANTS:
            r = run_one(model, data, Q_BIAS, id_C, obj, robot_geom_names,
                        active_clearance_by_geom, dls_ik, constrained_ik, vkey)
            row[vkey] = r
            print(f"  {bname:<20} {vlabel:<22} {r['dls_ms']:>5.0f}ms {r['solve_ms']:>6.0f}ms "
                  f"{str(r['iters']):>6}  {r['status']:<26}  {r['max_mm']:>8.2f}mm  "
                  f"{r['fk_per_iter']:>6}")
        results[bname] = row

    # --- 3-way summary ---
    def _cmp(base, cand, label):
        """Return (Δiters, Δms, note) comparing cand against base.
        Note is based on wall time (solve_ms), not iteration count — SQP's
        iterations are cheaper than FD's so raw iteration count is misleading."""
        bi = base['iters'];  ci = cand['iters']
        bt = base['solve_ms']; ct = cand['solve_ms']
        di = (ci - bi) if isinstance(bi, int) and isinstance(ci, int) else '?'
        dt = ct - bt
        if dt < -200:
            note = f'{label} faster'
        elif dt > 200:
            note = f'{label} SLOWER'
        else:
            note = 'similar'
        return di, dt, note

    print(f"\n{'--- Summary vs FD baseline ---':}")
    print(f"{'Object':<22}  "
          f"{'[analytic]':^28}  "
          f"{'[sqp]':^28}  "
          f"{'FK/iter':>20}")
    print(f"  {'':20}  {'Δiters':>7} {'Δms':>7} {'note':<12}  "
          f"{'Δiters':>7} {'Δms':>7} {'note':<12}  "
          f"{'FD':>6} {'analytic':>8} {'SQP':>6}")
    print("-" * 110)

    an_wins = 0; sqp_wins = 0
    for bname, row in results.items():
        if 'fd' not in row:
            continue
        fd = row['fd']

        an_di, an_dt, an_note   = '?', float('nan'), ''
        sqp_di, sqp_dt, sqp_note = '?', float('nan'), ''

        if 'analytic' in row:
            an_di, an_dt, an_note = _cmp(fd, row['analytic'], 'analytic')
            if an_dt < -200:
                an_wins += 1
        if 'sqp' in row:
            sqp_di, sqp_dt, sqp_note = _cmp(fd, row['sqp'], 'sqp')
            if sqp_dt < -200:
                sqp_wins += 1

        fpi_fd  = fd['fk_per_iter']
        fpi_an  = row.get('analytic', {}).get('fk_per_iter', '?')
        fpi_sqp = row.get('sqp', {}).get('fk_per_iter', '?')

        print(f"  {bname:<20}  "
              f"{str(an_di):>7} {an_dt:>7.0f} {an_note:<12}  "
              f"{str(sqp_di):>7} {sqp_dt:>7.0f} {sqp_note:<12}  "
              f"{fpi_fd:>6} {str(fpi_an):>8} {str(fpi_sqp):>6}")

    n = len(results)
    print(f"\n  analytic faster in {an_wins}/{n} cases")
    print(f"  sqp     faster in {sqp_wins}/{n} cases")


if __name__ == '__main__':
    main()

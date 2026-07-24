"""ConstrainedIK: SQP (production default) vs IPOPT, across all sample seeds.

The live app defaults to --ik-solver sqp (configure_sqp: sqpmethod/OSQP + softplus SDF +
analytic Jacobians), NOT IPOPT — which the earlier sweeps used. This compares the two on
reach rate AND solve time (target: few seconds), holding a NONZERO posture weight (per
the requirement to keep some regularization even though zero posture lowers tip error).

Reports per solver × posture setting: samples reaching <5mm (no collision), mean/median
site error, and mean/max solve time.

Usage:
    python simulation/test_cik_sqp_vs_ipopt.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'grasp_control'))
sys.path.insert(0, _HERE)

import grasp_control.constrained_ik as cik_mod            # noqa: E402
from grasp_control.constrained_ik import (ConstrainedIKSolver,  # noqa: E402
                                          configure_sqp)
from grasp_planner_3d import _geom_normal_np              # noqa: E402

_SCENE = os.path.join(os.path.dirname(_HERE), 'models', 'scene_pick_place.xml')
_N_ROBOT = 23
_FINGER_SET = ['index', 'thumb']
_TIP_SITES = {'index': 'leap_if_ds_tip', 'thumb': 'leap_th_ds_tip'}
_TIP_GEOMS = {'index': 'leap_if_tip', 'thumb': 'leap_th_tip'}
_FINGER_CODE = {'index': 'if', 'thumb': 'th', 'middle': 'mf', 'ring': 'rf'}
_BOX = 'obj_red_box'

# configure_sqp monkeypatches module-level names; keep pristine copies to restore
# so we can build IPOPT solvers again after any SQP solver was configured.
_PRISTINE = {n: getattr(cik_mod, n) for n in (
    '_sphere_box_distance', '_sphere_cylinder_distance', '_SitePositionCallback',
    '_SiteAxisCallback', '_GeomPositionCallback', '_BatchedGeomPositionCallback')}


def _restore_ipopt_module():
    for n, v in _PRISTINE.items():
        setattr(cik_mod, n, v)


def _pad_offset(model, f):
    d0 = mj.MjData(model)
    mj.mj_kinematics(model, d0)
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, _TIP_GEOMS[f])
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _TIP_SITES[f])
    mid = model.geom_dataid[gid]
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts_W = (d0.geom_xmat[gid].reshape(3, 3) @ model.mesh_vert[adr:adr + num].T).T \
              + d0.geom_xpos[gid]
    pad_dir_W = -d0.site_xmat[sid].reshape(3, 3)[:, 0]
    return float(np.max((verts_W - d0.site_xpos[sid]) @ pad_dir_W))


def _robot_geoms(model):
    out = []
    prefixes = tuple(f'leap_{c}_' for c in _FINGER_CODE.values())
    for gi in range(model.ngeom):
        nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gi)
        if not nm or model.geom_contype[gi] == 0:
            continue
        b = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ''
        if any(b.startswith(p) for p in prefixes) or b in ('leap_palm', 'bracelet_link'):
            out.append(nm)
    return out


def _solve(model, data, q_seed, id_C, targets, rgeoms, active,
           use_sqp, w_arm, w_hand, mi):
    posture = np.r_[np.full(7, w_arm), np.full(_N_ROBOT - 7, w_hand)]
    if use_sqp:
        # configure_sqp patches the module; build the solver AFTER patching so it picks
        # up analytic-Jacobian callbacks.
        solver = ConstrainedIKSolver(
            model, _N_ROBOT, arm_geom_names=rgeoms,
            obj_geom_names=[_BOX + '_geom', 'floor'], clearance=0.005,
            posture_weight=posture, pad_axis=(-1.0, 0.0, 0.0),
            tip_weight=1000.0, orient_weight=0.0, max_iter=mi)
        configure_sqp(solver)
    else:
        _restore_ipopt_module()
        solver = ConstrainedIKSolver(
            model, _N_ROBOT, arm_geom_names=rgeoms,
            obj_geom_names=[_BOX + '_geom', 'floor'], clearance=0.005,
            posture_weight=posture, pad_axis=(-1.0, 0.0, 0.0),
            tip_weight=1000.0, orient_weight=0.0, max_iter=mi)
        solver._solver_opts['ipopt']['tol'] = 1e-2
        solver._solver_opts['ipopt']['acceptable_tol'] = 1e-1
        solver._solver_opts['ipopt']['acceptable_iter'] = 5
    d = mj.MjData(model)
    d.qpos[:] = data.qpos[:]
    mj.mj_forward(model, d)
    t0 = time.time()
    solver.solve(d, id_C, targets, q_bias=q_seed, q_init=q_seed,
                 reduced_clearance_geoms=active, inward_dirs=None)
    dt = time.time() - t0
    m = solver.last_metrics
    return max(m.get('site_err_mm', [1e9, 1e9])), m.get('min_slack_mm'), dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-iter', type=int, default=300)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(_SCENE)
    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    rgeoms = _robot_geoms(model)
    pad = {f: _pad_offset(model, f) for f in _FINGER_SET}
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _TIP_SITES[f]) for f in _FINGER_SET]
    active = {g: (-1.0 if ('_ds_' in g or g.endswith('_tip')) else 0.002)
              for g in rgeoms
              if any(g.startswith(f'leap_{_FINGER_CODE[f]}_') for f in _FINGER_SET)}

    cases = []
    for s in samples:
        data = mj.MjData(model)
        oq = np.asarray(s['obj_qpos'], float)
        jadr = model.jnt_qposadr[model.body_jntadr[
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _BOX)]]
        data.qpos[jadr:jadr + 7] = oq[:7]
        for i, idx in enumerate(act_idx):
            data.qpos[idx] = s['q_seed'][i]
        mj.mj_forward(model, data)
        ogid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, _BOX + '_geom')
        obj_pos = data.geom_xpos[ogid].copy()
        obj_mat = data.geom_xmat[ogid].reshape(3, 3)
        gsize = model.geom_size[ogid]
        gtype = model.geom_type[ogid]
        c1 = data.site_xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _BOX + '_c1')].copy()
        c2 = data.site_xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _BOX + '_c2')].copy()
        contact = {'thumb': c1, 'index': c2}
        p_S = [contact[f] for f in _FINGER_SET]
        n_in = [-_geom_normal_np(p, gtype, obj_pos, obj_mat, gsize) for p in p_S]
        tgt = [p + pad[f] * ni for f, p, ni in zip(_FINGER_SET, p_S, n_in)]
        q_seed = np.array([data.qpos[i] for i in act_idx])
        cases.append((data, q_seed, tgt))

    # (label, use_sqp, w_arm, w_hand). All posture weights NONZERO.
    rows = [
        ('SQP  arm1e-5 hand1e-4 (prod)', True,  1e-5, 1e-4),
        ('SQP  arm1e-6 hand1e-4',        True,  1e-6, 1e-4),
        ('SQP  arm1e-4 hand1e-4',        True,  1e-4, 1e-4),
        ('IPOPT arm1e-5 hand1e-4 tol1e-2', False, 1e-5, 1e-4),
    ]
    print(f"\nSQP vs IPOPT over {len(samples)} seeds  (tip_w=1000, orient=0, "
          f"max_iter={args.max_iter}, NONZERO posture)")
    print("=" * 84)
    print(f"{'setting':<32} {'reach<5mm':>10} {'med_err':>8} {'mean_t':>8} {'max_t':>7}")
    print("-" * 84)
    for lbl, sqp, wa, wh in rows:
        errs, times, reach = [], [], 0
        for (data, q_seed, tgt) in cases:
            err, slack, dt = _solve(model, data, q_seed, id_C, tgt, rgeoms, active,
                                    sqp, wa, wh, args.max_iter)
            errs.append(err); times.append(dt)
            if err < 5.0 and (slack is None or slack > -0.5):
                reach += 1
        errs = np.array(errs); times = np.array(times)
        print(f"{lbl:<32} {reach:>7}/{len(samples)} {np.median(errs):>7.1f} "
              f"{np.mean(times):>7.2f}s {np.max(times):>6.2f}s")
    print("=" * 84)
    print("Target: high reach<5mm AND max solve time within a few seconds. Posture kept")
    print("nonzero on every row (arm >= 1e-6, hand = 1e-4).\n")


if __name__ == '__main__':
    main()

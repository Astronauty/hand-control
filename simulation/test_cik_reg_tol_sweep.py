"""ConstrainedIK convergence sweep: regularization weights + constraint tolerances.

Follows test_cik_allsamples.py, which found tol=1e-2 lifts reachability from 5/12 to
9/12. This sweeps two further axes on top of that baseline to push it higher:

  A. Posture (regularization) weight — how hard the solver pulls q toward q_seed.
     Looser arm posture lets the arm null-space serve the tips (reach further); too loose
     drifts the pose. Swept as (w_arm, w_hand); production is (1e-5, 1e-4).

  B. Constraint tolerance — constr_viol_tol / acceptable_constr_viol_tol. Looser lets
     IPOPT accept a sub-mm collision/joint violation and EXIT instead of iterating to
     tight feasibility it can't reach, trading a tiny (acceptable) violation for a
     converged-enough grasp.

Baseline held fixed: tip_weight=1000, orient_weight=0, tol=1e-2, max_iter=500.
Reports, per setting, how many of the 12 sample seeds reach <5mm site error with no
meaningful collision (slack > -0.5mm).

Usage:
    python simulation/test_cik_reg_tol_sweep.py
    python simulation/test_cik_reg_tol_sweep.py --axis posture   # only posture rows
    python simulation/test_cik_reg_tol_sweep.py --axis ctol      # only constr-tol rows
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'grasp_control'))
sys.path.insert(0, _HERE)

from grasp_control.constrained_ik import ConstrainedIKSolver  # noqa: E402
from grasp_planner_3d import _geom_normal_np                  # noqa: E402

_SCENE = os.path.join(os.path.dirname(_HERE), 'models', 'scene_pick_place.xml')
_N_ROBOT = 23
_FINGER_SET = ['index', 'thumb']
_TIP_SITES = {'index': 'leap_if_ds_tip', 'thumb': 'leap_th_ds_tip'}
_TIP_GEOMS = {'index': 'leap_if_tip', 'thumb': 'leap_th_tip'}
_FINGER_CODE = {'index': 'if', 'thumb': 'th', 'middle': 'mf', 'ring': 'rf'}
_BOX = 'obj_red_box'


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


def _solve(model, data, q_seed, id_C, targets, n_in, rgeoms, active,
           w_arm, w_hand, ctol, tol=1e-2, tip_w=1000.0, mi=500):
    posture = np.r_[np.full(7, w_arm), np.full(_N_ROBOT - 7, w_hand)]
    solver = ConstrainedIKSolver(
        model, _N_ROBOT, arm_geom_names=rgeoms,
        obj_geom_names=[_BOX + '_geom', 'floor'], clearance=0.005,
        posture_weight=posture, pad_axis=(-1.0, 0.0, 0.0),
        tip_weight=tip_w, orient_weight=0.0, max_iter=mi)
    o = solver._solver_opts['ipopt']
    o['tol'] = tol
    o['acceptable_tol'] = tol * 10
    o['acceptable_iter'] = 5
    o['constr_viol_tol'] = ctol
    o['acceptable_constr_viol_tol'] = ctol
    d = mj.MjData(model)
    d.qpos[:] = data.qpos[:]
    mj.mj_forward(model, d)
    solver.solve(d, id_C, targets, q_bias=q_seed, q_init=q_seed,
                 reduced_clearance_geoms=active, inward_dirs=None)
    m = solver.last_metrics
    return max(m.get('site_err_mm', [1e9, 1e9])), m.get('min_slack_mm')


def _load_samples():
    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--axis', choices=['posture', 'ctol', 'both'], default='both')
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(_SCENE)
    samples = _load_samples()
    rgeoms = _robot_geoms(model)
    pad = {f: _pad_offset(model, f) for f in _FINGER_SET}
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _TIP_SITES[f]) for f in _FINGER_SET]
    active = {g: (-1.0 if ('_ds_' in g or g.endswith('_tip')) else 0.002)
              for g in rgeoms
              if any(g.startswith(f'leap_{_FINGER_CODE[f]}_') for f in _FINGER_SET)}

    # (label, w_arm, w_hand, constr_viol_tol). Baseline = production posture + tight ctol.
    rows = []
    if args.axis in ('posture', 'both'):
        rows += [
            ('BASE arm1e-5 hand1e-4 ctol1e-6', 1e-5, 1e-4, 1e-6),
            ('arm1e-6 (looser arm)',           1e-6, 1e-4, 1e-6),
            ('arm1e-4 (tighter arm)',          1e-4, 1e-4, 1e-6),
            ('arm1e-5 hand1e-5 (loose hand)',  1e-5, 1e-5, 1e-6),
            ('arm1e-5 hand1e-3 (tight hand)',  1e-5, 1e-3, 1e-6),
        ]
    if args.axis in ('ctol', 'both'):
        rows += [
            ('ctol1e-4',  1e-5, 1e-4, 1e-4),
            ('ctol1e-3',  1e-5, 1e-4, 1e-3),
            ('arm1e-6 + ctol1e-4', 1e-6, 1e-4, 1e-4),
        ]

    # Precompute per-sample geometry once.
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
        cases.append((data, q_seed, tgt, n_in))

    print(f"\nConstrainedIK reg/tolerance sweep over {len(samples)} seeds "
          f"(tip_w=1000, orient=0, tol=1e-2, max_iter=500)")
    print("=" * 78)
    print(f"{'setting':<34} {'reach<5mm':>10} {'mean_err':>9} {'median':>8}")
    print("-" * 78)
    for lbl, wa, wh, ct in rows:
        errs = []
        reach = 0
        for (data, q_seed, tgt, n_in) in cases:
            err, slack = _solve(model, data, q_seed, id_C, tgt, n_in, rgeoms, active,
                                wa, wh, ct)
            errs.append(err)
            if err < 5.0 and (slack is None or slack > -0.5):
                reach += 1
        errs = np.array(errs)
        # cap wild misses for a meaningful mean
        capped = np.clip(errs, 0, 60)
        print(f"{lbl:<34} {reach:>7}/{len(samples)} {np.mean(capped):>8.1f} "
              f"{np.median(errs):>8.1f}")
    print("=" * 78)
    print("reach<5mm = samples landing tips within 5mm with no collision. Higher is")
    print("better; median err shows the typical case. Compare against the tol=1e-2")
    print("baseline (9/12 from test_cik_allsamples).\n")


if __name__ == '__main__':
    main()

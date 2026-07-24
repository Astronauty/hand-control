"""Comprehensive SQP vs IPOPT tuning study for ConstrainedIK on the site-reaching task.

Goal: sub-5mm tip error, high convergence rate, <5s solve, on the 12 samples.jsonl seeds.
Cost weights MAY differ per method (each tuned to its own best). Two-stage design to stay
under ~10 min total:

  STAGE 1 (coarse screen, 4-sample subset): sweep the key knobs cheaply to find promising
    regions.
    - SQP: softplus alpha {200, 500, 1500} × tol_du {1e-2, 1e-1, 1} × tip_w {500, 1000}
    - IPOPT: tol {1e-2, 1e-3} × acceptable_tol {1e-1, 1e-2} × tip_w {500, 1000}
  STAGE 2 (confirm, all 12 samples): the top-N configs per method by stage-1 reach rate.

Metrics per config: reach<5mm (no collision), median site err, mean/max solve time.

Usage:
    python simulation/study_ik_solvers.py                 # full 2-stage
    python simulation/study_ik_solvers.py --quick         # fewer configs
"""
import argparse
import functools
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
                                          configure_sqp, _softplus)
from grasp_planner_3d import _geom_normal_np              # noqa: E402

_SCENE = os.path.join(os.path.dirname(_HERE), 'models', 'scene_pick_place.xml')
_N_ROBOT = 23
_FINGER_SET = ['index', 'thumb']
_TIP_SITES = {'index': 'leap_if_ds_tip', 'thumb': 'leap_th_ds_tip'}
_TIP_GEOMS = {'index': 'leap_if_tip', 'thumb': 'leap_th_tip'}
_FINGER_CODE = {'index': 'if', 'thumb': 'th', 'middle': 'mf', 'ring': 'rf'}
_BOX = 'obj_red_box'

_PRISTINE = {n: getattr(cik_mod, n) for n in (
    '_sphere_box_distance', '_sphere_cylinder_distance', '_SitePositionCallback',
    '_SiteAxisCallback', '_GeomPositionCallback', '_BatchedGeomPositionCallback',
    '_softplus_sphere_box_distance', '_softplus_sphere_cylinder_distance')}


def _restore_module():
    for n, v in _PRISTINE.items():
        setattr(cik_mod, n, v)


def _set_softplus_alpha(alpha):
    """Rebuild the softplus SDFs to use a specific alpha (default hardcoded 500)."""
    sp = functools.partial(_softplus, alpha=alpha)

    def box(p_arm, arm_radius, box_center, box_R, half_extents):
        import casadi as ca
        p_local = box_R.T @ (p_arm - box_center)
        q = ca.fabs(p_local) - half_extents
        outside = ca.sqrt(ca.sumsqr(sp(q)) + 1e-12)
        inside = ca.fmin(ca.fmax(ca.fmax(q[0], q[1]), q[2]), 0)
        return outside + inside - arm_radius

    def cyl(p_arm, arm_radius, cyl_center, cyl_R, cyl_radius, cyl_hh):
        import casadi as ca
        p = cyl_R.T @ (p_arm - cyl_center)
        radial = ca.sqrt(p[0] * p[0] + p[1] * p[1] + 1e-12)
        dr = radial - cyl_radius
        dz = ca.fabs(p[2]) - cyl_hh
        outside = ca.sqrt(sp(dr) ** 2 + sp(dz) ** 2 + 1e-12)
        inside = ca.fmin(ca.fmax(dr, dz), 0)
        return outside + inside - arm_radius

    cik_mod._softplus_sphere_box_distance = box
    cik_mod._softplus_sphere_cylinder_distance = cyl


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


def _mk_cases(model, samples):
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    pad = {f: _pad_offset(model, f) for f in _FINGER_SET}
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
    return cases, act_idx


def _run_config(model, cases, id_C, rgeoms, active, cfg):
    """cfg keys: method, tip_w, w_arm, w_hand, max_iter, + method-specific tuning."""
    errs, times, reach = [], [], 0
    for (data, q_seed, tgt) in cases:
        posture = np.r_[np.full(7, cfg['w_arm']), np.full(_N_ROBOT - 7, cfg['w_hand'])]
        if cfg['method'] == 'sqp':
            _set_softplus_alpha(cfg['alpha'])
            solver = ConstrainedIKSolver(
                model, _N_ROBOT, arm_geom_names=rgeoms,
                obj_geom_names=[_BOX + '_geom', 'floor'], clearance=0.005,
                posture_weight=posture, pad_axis=(-1.0, 0.0, 0.0),
                tip_weight=cfg['tip_w'], orient_weight=0.0, max_iter=cfg['max_iter'])
            configure_sqp(solver)
            solver._solver_opts['tol_du'] = cfg['tol_du']
        else:
            _restore_module()
            solver = ConstrainedIKSolver(
                model, _N_ROBOT, arm_geom_names=rgeoms,
                obj_geom_names=[_BOX + '_geom', 'floor'], clearance=0.005,
                posture_weight=posture, pad_axis=(-1.0, 0.0, 0.0),
                tip_weight=cfg['tip_w'], orient_weight=0.0, max_iter=cfg['max_iter'])
            o = solver._solver_opts['ipopt']
            o['tol'] = cfg['tol']
            o['acceptable_tol'] = cfg['acc_tol']
            o['acceptable_iter'] = cfg.get('acc_iter', 8)
        d = mj.MjData(model)
        d.qpos[:] = data.qpos[:]
        mj.mj_forward(model, d)
        t0 = time.time()
        solver.solve(d, id_C, tgt, q_bias=q_seed, q_init=q_seed,
                     reduced_clearance_geoms=active, inward_dirs=None)
        times.append(time.time() - t0)
        m = solver.last_metrics
        err = max(m.get('site_err_mm', [1e9, 1e9]))
        slack = m.get('min_slack_mm')
        errs.append(err)
        if err < 5.0 and (slack is None or slack > -0.5):
            reach += 1
    errs = np.array(errs); times = np.array(times)
    return {'reach': reach, 'n': len(cases), 'med_err': float(np.median(errs)),
            'mean_t': float(np.mean(times)), 'max_t': float(np.max(times))}


def _sqp_grid(quick):
    alphas = [200.0, 1500.0] if quick else [200.0, 500.0, 1500.0]
    tol_dus = [1e-1, 1.0] if quick else [1e-2, 1e-1, 1.0]
    tips = [1000.0] if quick else [500.0, 1000.0]
    out = []
    for a in alphas:
        for td in tol_dus:
            for tw in tips:
                out.append({'method': 'sqp', 'alpha': a, 'tol_du': td, 'tip_w': tw,
                            'w_arm': 1e-5, 'w_hand': 1e-4, 'max_iter': 400,
                            'label': f'SQP a{a:.0f} tdu{td:g} tip{tw:.0f}'})
    return out


def _ipopt_grid(quick):
    tols = [1e-2] if quick else [1e-2, 1e-3]
    acc = [1e-2] if quick else [1e-1, 1e-2]
    tips = [1000.0] if quick else [500.0, 1000.0]
    out = []
    for t in tols:
        for at in acc:
            for tw in tips:
                out.append({'method': 'ipopt', 'tol': t, 'acc_tol': at, 'tip_w': tw,
                            'w_arm': 1e-5, 'w_hand': 1e-4, 'max_iter': 300,
                            'label': f'IPOPT tol{t:g} acc{at:g} tip{tw:.0f}'})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--stage1-samples', type=int, default=4)
    ap.add_argument('--top', type=int, default=3)
    args = ap.parse_args()

    t_start = time.time()
    model = mj.MjModel.from_xml_path(_SCENE)
    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    rgeoms = _robot_geoms(model)
    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _TIP_SITES[f]) for f in _FINGER_SET]
    active = {g: (-1.0 if ('_ds_' in g or g.endswith('_tip')) else 0.002)
              for g in rgeoms
              if any(g.startswith(f'leap_{_FINGER_CODE[f]}_') for f in _FINGER_SET)}

    all_cases, _ = _mk_cases(model, samples)
    # Stage-1 subset spread across the distinct object poses.
    sub_idx = list(range(0, len(samples), max(1, len(samples) // args.stage1_samples)))[:args.stage1_samples]
    sub_cases = [all_cases[i] for i in sub_idx]

    grid = _sqp_grid(args.quick) + _ipopt_grid(args.quick)
    print(f"\n=== STAGE 1: coarse screen ({len(grid)} configs × {len(sub_cases)} samples) ===")
    print(f"{'config':<28} {'reach':>7} {'med_err':>8} {'mean_t':>7} {'max_t':>6}")
    print("-" * 62)
    for cfg in grid:
        r = _run_config(model, sub_cases, id_C, rgeoms, active, cfg)
        cfg['_reach'] = r['reach']; cfg['_med'] = r['med_err']
        print(f"{cfg['label']:<28} {r['reach']:>4}/{r['n']:<2} {r['med_err']:>7.1f} "
              f"{r['mean_t']:>6.2f}s {r['max_t']:>5.2f}s  [{time.time()-t_start:.0f}s]")

    # Rank: reach desc, then median err asc. Top-N per method.
    def rank(c):
        return (-c['_reach'], c['_med'])
    sqp_top = sorted([c for c in grid if c['method'] == 'sqp'], key=rank)[:args.top]
    ipo_top = sorted([c for c in grid if c['method'] == 'ipopt'], key=rank)[:args.top]

    print(f"\n=== STAGE 2: confirm top-{args.top} per method on ALL {len(samples)} samples ===")
    print(f"{'config':<28} {'reach':>7} {'med_err':>8} {'mean_t':>7} {'max_t':>6}")
    print("-" * 62)
    best = None
    for cfg in sqp_top + ipo_top:
        r = _run_config(model, all_cases, id_C, rgeoms, active, cfg)
        flag = ' <5s' if r['max_t'] < 5.0 else ' SLOW'
        print(f"{cfg['label']:<28} {r['reach']:>4}/{r['n']:<2} {r['med_err']:>7.1f} "
              f"{r['mean_t']:>6.2f}s {r['max_t']:>5.2f}s{flag}  [{time.time()-t_start:.0f}s]")
        score = (r['reach'], -r['med_err'])
        if best is None or score > best[0]:
            best = (score, cfg['label'], r)
    print("=" * 62)
    if best:
        _, lbl, r = best
        print(f"BEST: {lbl}  -> {r['reach']}/{r['n']} reach<5mm, med {r['med_err']:.1f}mm, "
              f"max {r['max_t']:.2f}s")
    print(f"total study time: {time.time()-t_start:.0f}s\n")


if __name__ == '__main__':
    main()

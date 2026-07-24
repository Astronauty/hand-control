"""Cube-size sweep: reach rate vs box half-extent, contacts at face centers.

Hypothesis: smaller cubes ease the IK collision constraints (less volume for the fingers
to avoid) but tighten precision (the fingertips must land on a smaller face). This sweeps
the box half-extent with the contact sites always at the +/-x face centers, using the
winning SQP config (alpha=1500, tol_du=0.01, tip_w=500), across all 12 sample seeds.

For each size, the box geom is resized at runtime and the c1/c2 sites are moved to
+/-half_extent so they track the face centers. The box is relocated to each sample's
recorded obj_qpos; the arm starts at the recorded q_seed.

Usage:
    python simulation/test_cube_size_sweep.py
    python simulation/test_cube_size_sweep.py --sizes 0.02,0.03,0.04,0.05,0.06
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


def _set_softplus_alpha(alpha):
    sp = functools.partial(_softplus, alpha=alpha)
    import casadi as ca

    def box(p_arm, arm_radius, box_center, box_R, half_extents):
        p_local = box_R.T @ (p_arm - box_center)
        q = ca.fabs(p_local) - half_extents
        outside = ca.sqrt(ca.sumsqr(sp(q)) + 1e-12)
        inside = ca.fmin(ca.fmax(ca.fmax(q[0], q[1]), q[2]), 0)
        return outside + inside - arm_radius

    def cyl(p_arm, arm_radius, cyl_center, cyl_R, cyl_radius, cyl_hh):
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


def _resize_box(model, half):
    """Set the box half-extents and move c1/c2 sites to +/-half on x (face centers)."""
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, _BOX + '_geom')
    model.geom_size[gid] = np.array([half, half, half], float)
    s1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _BOX + '_c1')
    s2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _BOX + '_c2')
    model.site_pos[s1] = np.array([-half, 0.0, 0.0])
    model.site_pos[s2] = np.array([half, 0.0, 0.0])
    # rest z so the box sits on the floor (spawn qpos overwritten per sample anyway)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', default='0.020,0.030,0.040,0.050,0.060',
                    help="comma-separated half-extents (m)")
    ap.add_argument('--alpha', type=float, default=1500.0)
    ap.add_argument('--tol-du', type=float, default=0.01)
    ap.add_argument('--tip-w', type=float, default=500.0)
    ap.add_argument('--float-height', type=float, default=None,
                    help="if set, place the box CENTER at this z (m) for every size so "
                         "the face centers sit at a common height — isolates the precision/"
                         "span effect from the approach-height confound. Default: rest the "
                         "box bottom on the table (center z = half-extent).")
    args = ap.parse_args()
    sizes = [float(x) for x in args.sizes.split(',')]

    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    t0 = time.time()
    _hstr = (f"CENTER floated to z={args.float_height:.3f}m"
             if args.float_height is not None else "bottom on table")
    print(f"\nCube-size sweep — SQP (alpha={args.alpha:.0f}, tol_du={args.tol_du:g}, "
          f"tip_w={args.tip_w:.0f}), contacts at face centers, {len(samples)} seeds  "
          f"[{_hstr}]")
    print("=" * 74)
    print(f"{'half(cm)':>8} {'full(cm)':>8} {'reach<5mm':>10} {'med_err':>8} "
          f"{'mean_t':>7} {'max_t':>6} {'n_penetrate':>11}")
    print("-" * 74)

    _set_softplus_alpha(args.alpha)
    for half in sizes:
        # Fresh model per size so geom_size/site_pos edits are clean.
        model = mj.MjModel.from_xml_path(_SCENE)
        _resize_box(model, half)
        rgeoms = _robot_geoms(model)
        pad = {f: _pad_offset(model, f) for f in _FINGER_SET}
        act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
        id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _TIP_SITES[f]) for f in _FINGER_SET]
        active = {g: (-1.0 if ('_ds_' in g or g.endswith('_tip')) else 0.002)
                  for g in rgeoms
                  if any(g.startswith(f'leap_{_FINGER_CODE[f]}_') for f in _FINGER_SET)}

        errs, times, reach, penet = [], [], 0, 0
        for s in samples:
            data = mj.MjData(model)
            oq = np.asarray(s['obj_qpos'], float)
            jadr = model.jnt_qposadr[model.body_jntadr[
                mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _BOX)]]
            data.qpos[jadr:jadr + 7] = oq[:7]
            # z placement: rest bottom on the table (center = half) OR float the center
            # to a common height so face-center contacts sit at the same z for all sizes.
            data.qpos[jadr + 2] = (args.float_height if args.float_height is not None
                                   else half)
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
            posture = np.r_[np.full(7, 1e-5), np.full(_N_ROBOT - 7, 1e-4)]
            solver = ConstrainedIKSolver(
                model, _N_ROBOT, arm_geom_names=rgeoms,
                obj_geom_names=[_BOX + '_geom', 'floor'], clearance=0.005,
                posture_weight=posture, pad_axis=(-1.0, 0.0, 0.0),
                tip_weight=args.tip_w, orient_weight=0.0, max_iter=400)
            configure_sqp(solver)
            solver._solver_opts['tol_du'] = args.tol_du
            d = mj.MjData(model)
            d.qpos[:] = data.qpos[:]
            mj.mj_forward(model, d)
            t = time.time()
            solver.solve(d, id_C, tgt, q_bias=q_seed, q_init=q_seed,
                         reduced_clearance_geoms=active, inward_dirs=None)
            times.append(time.time() - t)
            m = solver.last_metrics
            err = max(m.get('site_err_mm', [1e9, 1e9]))
            slack = m.get('min_slack_mm')
            errs.append(err)
            if slack is not None and slack < -0.5:
                penet += 1
            if err < 5.0 and (slack is None or slack > -0.5):
                reach += 1
        errs = np.array(errs); times = np.array(times)
        print(f"{half*100:>8.1f} {half*200:>8.1f} {reach:>7}/{len(samples)} "
              f"{np.median(errs):>7.1f} {np.mean(times):>6.2f}s {np.max(times):>5.2f}s "
              f"{penet:>11}  [{time.time()-t0:.0f}s]")
    print("=" * 74)
    print("Hypothesis check: smaller cube -> fewer collisions (n_penetrate down) but")
    print("tighter precision (med_err up / reach down). Watch both columns move.\n")


if __name__ == '__main__':
    main()

"""Contact-height sweep: does letting the antipodal contacts sit HIGHER on the face
improve reach, across cube sizes, on the ground (bottom-on-table) env?

Setup: contacts on the +/-x faces of a box resting on the table. Instead of pinning them
to the face CENTER (z = box center), raise them by a height offset dz toward the top of
the face. Rationale from the size study: bottom-on-table small cubes fail because fingers
jam into the table; approaching a higher contact point lets the fingers clear the table.
This sweeps dz (as a fraction of the half-extent, so it scales with the box) for several
cube sizes, using the winning SQP config, over the 12 sample seeds.

Contacts stay on the face plane (x = +/-half, inward normal +/-x); only their z rises.
dz is clamped so the contact stays on the face (|z_local| <= half).

Usage:
    python simulation/test_contact_height_sweep.py
    python simulation/test_contact_height_sweep.py --sizes 0.03,0.04,0.05,0.06
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
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, _BOX + '_geom')
    model.geom_size[gid] = np.array([half, half, half], float)


def _solve_once(model, data, half, dz_local, rgeoms, active, id_C, pad,
                tip_w, alpha, tol_du):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _BOX)
    ogid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, _BOX + '_geom')
    obj_pos = data.geom_xpos[ogid].copy()
    obj_mat = data.geom_xmat[ogid].reshape(3, 3)
    gsize = model.geom_size[ogid]
    gtype = model.geom_type[ogid]
    R_WO = data.xmat[bid].reshape(3, 3)
    # Contacts on +/-x faces, raised by dz_local along the object's +z, clamped to face.
    dz = float(np.clip(dz_local, -half, half))
    c_thumb = obj_pos + R_WO @ np.array([-half, 0.0, dz])   # -x face -> c1 (thumb)
    c_index = obj_pos + R_WO @ np.array([+half, 0.0, dz])   # +x face -> c2 (index)
    contact = {'thumb': c_thumb, 'index': c_index}
    p_S = [contact[f] for f in _FINGER_SET]
    n_in = [-_geom_normal_np(p, gtype, obj_pos, obj_mat, gsize) for p in p_S]
    tgt = [p + pad[f] * ni for f, p, ni in zip(_FINGER_SET, p_S, n_in)]
    q_seed = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                       for i in range(model.nu)])
    posture = np.r_[np.full(7, 1e-5), np.full(_N_ROBOT - 7, 1e-4)]
    solver = ConstrainedIKSolver(
        model, _N_ROBOT, arm_geom_names=rgeoms,
        obj_geom_names=[_BOX + '_geom', 'floor'], clearance=0.005,
        posture_weight=posture, pad_axis=(-1.0, 0.0, 0.0),
        tip_weight=tip_w, orient_weight=0.0, max_iter=400)
    configure_sqp(solver)
    solver._solver_opts['tol_du'] = tol_du
    d = mj.MjData(model)
    d.qpos[:] = data.qpos[:]
    mj.mj_forward(model, d)
    t = time.time()
    solver.solve(d, id_C, tgt, q_bias=q_seed, q_init=q_seed,
                 reduced_clearance_geoms=active, inward_dirs=None)
    dt = time.time() - t
    m = solver.last_metrics
    return max(m.get('site_err_mm', [1e9, 1e9])), m.get('min_slack_mm'), dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', default='0.030,0.040,0.050,0.060')
    ap.add_argument('--alpha', type=float, default=1500.0)
    ap.add_argument('--tol-du', type=float, default=0.01)
    ap.add_argument('--tip-w', type=float, default=500.0)
    args = ap.parse_args()
    sizes = [float(x) for x in args.sizes.split(',')]
    # Height offsets as a FRACTION of the half-extent (scales with the box):
    #   0.0 = face center (baseline), up to ~0.9 = near the top of the face.
    dz_fracs = [0.0, 0.3, 0.6, 0.9]

    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    t0 = time.time()
    _set_softplus_alpha(args.alpha)
    print(f"\nContact-HEIGHT sweep — SQP (a={args.alpha:.0f}, tdu={args.tol_du:g}, "
          f"tip={args.tip_w:.0f}), box on table, {len(samples)} seeds")
    print("Contacts on +/-x faces, raised by dz = frac * half-extent toward the top.")
    print("=" * 78)
    print(f"{'full(cm)':>8} | " + " ".join(f"dz={f:.1f}h".rjust(11) for f in dz_fracs)
          + "   (reach<5mm / med_err_mm)")
    print("-" * 78)

    for half in sizes:
        model = mj.MjModel.from_xml_path(_SCENE)
        _resize_box(model, half)
        rgeoms = _robot_geoms(model)
        pad = {f: _pad_offset(model, f) for f in _FINGER_SET}
        act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
        id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _TIP_SITES[f]) for f in _FINGER_SET]
        active = {g: (-1.0 if ('_ds_' in g or g.endswith('_tip')) else 0.002)
                  for g in rgeoms
                  if any(g.startswith(f'leap_{_FINGER_CODE[f]}_') for f in _FINGER_SET)}

        cells = []
        for frac in dz_fracs:
            errs, reach = [], 0
            for s in samples:
                data = mj.MjData(model)
                oq = np.asarray(s['obj_qpos'], float)
                jadr = model.jnt_qposadr[model.body_jntadr[
                    mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _BOX)]]
                data.qpos[jadr:jadr + 7] = oq[:7]
                data.qpos[jadr + 2] = half   # rest bottom on table
                for i, idx in enumerate(act_idx):
                    data.qpos[idx] = s['q_seed'][i]
                mj.mj_forward(model, data)
                err, slack, _ = _solve_once(model, data, half, frac * half, rgeoms,
                                            active, id_C, pad, args.tip_w, args.alpha,
                                            args.tol_du)
                errs.append(err)
                if err < 5.0 and (slack is None or slack > -0.5):
                    reach += 1
            cells.append(f"{reach:>2}/12 {np.median(errs):>5.1f}")
        print(f"{half*200:>8.1f} | " + " ".join(c.rjust(11) for c in cells)
              + f"   [{time.time()-t0:.0f}s]")
    print("=" * 78)
    print("Each cell: reach<5mm / median site error (mm). dz=0.0h is the face-center")
    print("baseline. If reach rises with dz (esp. for larger cubes), a higher contact")
    print("clears the table / better matches finger approach — the contact-search")
    print("recommender should be allowed to pick higher points, not just face centers.\n")


if __name__ == '__main__':
    main()

"""ConstrainedIK reachability sweep over ALL samples.jsonl seeds + weight/tolerance vary.

Follows up test_cik_baseline_sites.py, which found sample-0's known-good sites are NOT
reachable to sub-mm. Question: is that sample-specific, or universal? And can relaxed
weights / solver tolerances land the tips on the sites to an ACCEPTABLE (few-mm) error?

For each recorded operator pose (q_seed + obj_qpos), relocate the box there, target the
box's c1/c2 sites (thumb<-c1, index<-c2, pad-offset backed off), and report the best
site error achieved across a few weight/tolerance settings.

Usage:
    python simulation/test_cik_allsamples.py
    python simulation/test_cik_allsamples.py --tol-relax     # add relaxed-tolerance rows
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
_BOX = 'obj_red_box'   # the only object in the scene; relocated to each sample's pose


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


def _relax_tol(solver, tol, acc_tol, acc_iter):
    """Loosen IPOPT tolerances post-construction (accept slight IK error / earlier exit)."""
    o = solver._solver_opts['ipopt']
    o['tol'] = tol
    o['acceptable_tol'] = acc_tol
    o['acceptable_iter'] = acc_iter
    o['acceptable_constr_viol_tol'] = 1e-4


def _solve_one(model, data, q_seed, id_C, ik_targets, n_in, rgeoms, active,
               tip_w, orient_w, mi, tol=None):
    POSTURE = np.r_[np.full(7, 0.1e-4), np.full(_N_ROBOT - 7, 0.1e-3)]
    solver = ConstrainedIKSolver(
        model, _N_ROBOT, arm_geom_names=rgeoms,
        obj_geom_names=[_BOX + '_geom', 'floor'], clearance=0.005,
        posture_weight=POSTURE, pad_axis=(-1.0, 0.0, 0.0),
        tip_weight=tip_w, orient_weight=orient_w, max_iter=mi)
    if tol is not None:
        _relax_tol(solver, tol, tol * 10, 5)
    d = mj.MjData(model)
    d.qpos[:] = data.qpos[:]
    mj.mj_forward(model, d)
    solver.solve(d, id_C, ik_targets, q_bias=q_seed, q_init=q_seed,
                 reduced_clearance_geoms=active,
                 inward_dirs=(n_in if orient_w > 0 else None))
    m = solver.last_metrics
    se = m.get('site_err_mm', [1e9, 1e9])
    return max(se), m.get('min_slack_mm'), str(m.get('status', '?'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tol-relax', action='store_true')
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

    # (label, tip_w, orient_w, max_iter, tol)  — tol=None keeps default tight tol
    settings = [
        ('tip100/or1',   100.0, 1.0, 500, None),
        ('tip1k/or0',   1000.0, 0.0, 500, None),
    ]
    if args.tol_relax:
        settings += [
            ('tip1k/or0/tol1e-2', 1000.0, 0.0, 500, 1e-2),
            ('tip1k/or0/tol1e-1', 1000.0, 0.0, 500, 1e-1),
        ]

    print(f"\nConstrainedIK reachability over {len(samples)} sample seeds "
          f"(box relocated to each sample's pose)")
    print(f"  robot geoms={len(rgeoms)}  pad_offset(mm)="
          f"{ {f: round(pad[f]*1e3,1) for f in _FINGER_SET} }")
    print("=" * 92)
    hdr = "  ".join(f"{lbl:>16}" for lbl, *_ in settings)
    print(f"{'sample':>6} {'orig_obj':>14}   best max-site-err (mm) per setting:")
    print(f"{'':>6} {'':>14}   {hdr}")
    print("-" * 92)

    reach_count = {lbl: 0 for lbl, *_ in settings}   # count samples reaching <5mm
    for si, s in enumerate(samples):
        data = mj.MjData(model)
        mj.mj_forward(model, data)
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
        ik_targets = [p + pad[f] * ni for f, p, ni in zip(_FINGER_SET, p_S, n_in)]
        q_seed = np.array([data.qpos[i] for i in act_idx])

        cells = []
        for lbl, tw, ow, mi, tol in settings:
            err, slack, st = _solve_one(model, data, q_seed, id_C, ik_targets, n_in,
                                        rgeoms, active, tw, ow, mi, tol)
            flag = '*' if (err < 5.0 and (slack is None or slack > -0.5)) else ' '
            if err < 5.0 and (slack is None or slack > -0.5):
                reach_count[lbl] += 1
            cells.append(f"{err:6.1f}{flag}")
        print(f"{si:>6} {s['object']:>14}   " + "  ".join(f"{c:>16}" for c in cells))

    print("-" * 92)
    print("samples reaching <5mm (with non-penetrating slack), per setting:")
    for lbl in reach_count:
        print(f"   {lbl:>22}: {reach_count[lbl]}/{len(samples)}")
    print("\n* = site error < 5mm and no collision. If most samples reach the sites, the")
    print("  sites are generally reachable and sample-0 was just a hard pose. If few do,")
    print("  the fixed c1/c2 sites are broadly unreachable and the recommender SHOULD")
    print("  search for reachable contacts (its current design).\n")


if __name__ == '__main__':
    main()

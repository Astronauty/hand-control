"""ConstrainedIK baseline on the predefined grasp sites — IK feasibility check.

Before layering the wrench cone on the recommender NLP, establish that a collision-aware
IK solution putting the fingertips ON the known-good sites actually EXISTS from the live
near-box seed. If ConstrainedIK can't reach the sites either, the contact drift we saw in
the full NLP isn't the wrench cone's fault.

Uses the EXACT production convention (kinova_leap_pick_place.py):
  FINGER_SET = [index, thumb];  thumb<-c1, index<-c2
  tip sites  = leap_{if,th}_ds_tip
  targets    = site aimed at contact backed off by the fingerpad surface offset along
               the inward normal (ik_targets = p - PAD_OFFSET*n)
  weights    = tip_weight, orient_weight, posture split (arm vs hand)

Sweeps max_iter and tip_weight to find settings that land the tips on the sites to sub-mm.

Usage:
    python simulation/test_cik_baseline_sites.py --sample 0
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


def _pad_offset(model, f):
    """Fingerpad surface offset along the pad axis (mirrors kinova_leap_pick_place)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--object', default='obj_red_box')
    ap.add_argument('--sample', type=int, default=0)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(_SCENE)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    obj = args.object

    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        s = [json.loads(l) for l in f if l.strip()][args.sample]
    oq = np.asarray(s['obj_qpos'], float)
    jadr = model.jnt_qposadr[model.body_jntadr[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj)]]
    data.qpos[jadr:jadr + 7] = oq[:7]
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    for i, idx in enumerate(act_idx):
        data.qpos[idx] = s['q_seed'][i]
    mj.mj_forward(model, data)

    ogid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, obj + '_geom')
    obj_pos = data.geom_xpos[ogid].copy()
    obj_mat = data.geom_xmat[ogid].reshape(3, 3)
    gsize = model.geom_size[ogid]
    gtype = model.geom_type[ogid]

    # thumb<-c1, index<-c2 (FINGER_SET order = [index, thumb])
    c1 = data.site_xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, obj + '_c1')].copy()
    c2 = data.site_xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, obj + '_c2')].copy()
    contact = {'thumb': c1, 'index': c2}
    p_S = [contact[f] for f in _FINGER_SET]                       # [index=c2, thumb=c1]
    n_out = [_geom_normal_np(p, gtype, obj_pos, obj_mat, gsize) for p in p_S]
    n_in = [-n for n in n_out]
    pad = {f: _pad_offset(model, f) for f in _FINGER_SET}
    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _TIP_SITES[f]) for f in _FINGER_SET]
    ik_targets = [p + pad[f] * ni for f, p, ni in zip(_FINGER_SET, p_S, n_in)]

    q_seed = np.array([data.qpos[i] for i in act_idx])
    rgeoms = _robot_geoms(model)
    print(f"\nConstrainedIK baseline on sites — {obj}, sample {args.sample}")
    print(f"  FINGER_SET={_FINGER_SET}  pad_offset(mm)="
          f"{ {f: round(pad[f]*1e3,1) for f in _FINGER_SET} }")
    print(f"  robot geoms constrained: {len(rgeoms)}")
    print("=" * 80)
    print(f"{'tip_w':>6} {'orient_w':>8} {'iter':>5} {'status':>26} "
          f"{'site_err(mm)':>18} {'min_slack_mm':>12}")
    print("-" * 80)

    POSTURE = np.r_[np.full(7, 0.1e-4), np.full(_N_ROBOT - 7, 0.1e-3)]
    # Active grasping fingers get reduced clearance vs the object so they can reach the
    # surface (mirrors _active_clearance_by_geom); disable the contact-tier tips.
    active = {g: (-1.0 if ('_ds_' in g or g.endswith('_tip')) else 0.002)
              for g in rgeoms
              if any(g.startswith(f'leap_{_FINGER_CODE[f]}_') for f in _FINGER_SET)}

    for tip_w, orient_w, mi in [(100.0, 1.0, 200), (100.0, 1.0, 500),
                                (1000.0, 1.0, 500), (1000.0, 0.0, 500),
                                (5000.0, 0.0, 800)]:
        solver = ConstrainedIKSolver(
            model, _N_ROBOT, arm_geom_names=rgeoms,
            obj_geom_names=[obj + '_geom', 'floor'], clearance=0.005,
            posture_weight=POSTURE, pad_axis=(-1.0, 0.0, 0.0),
            tip_weight=tip_w, orient_weight=orient_w, max_iter=mi)
        d = mj.MjData(model)
        d.qpos[:] = data.qpos[:]
        mj.mj_forward(model, d)
        q = solver.solve(d, id_C, ik_targets, q_bias=q_seed, q_init=q_seed,
                         reduced_clearance_geoms=active,
                         inward_dirs=(n_in if orient_w > 0 else None))
        m = solver.last_metrics
        se = m.get('site_err_mm', [])
        se_s = "[" + ",".join(f"{e:.2f}" for e in se) + "]"
        ms = m.get('min_slack_mm')
        ms_s = f"{ms:.2f}" if ms is not None else '-'
        st = str(m.get('status', '?'))[:26]
        print(f"{tip_w:>6.0f} {orient_w:>8.1f} {mi:>5} {st:>26} {se_s:>18} {ms_s:>12}")
    print("=" * 80)
    print("If any row reaches sub-mm site error with a non-negative slack, a collision-")
    print("feasible IK to the sites EXISTS — the full NLP's contact drift is then the")
    print("wrench cone pulling contacts off the (reachable) sites, not an IK limitation.\n")


if __name__ == '__main__':
    main()

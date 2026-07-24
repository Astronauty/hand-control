"""Run the current (no-slack) 3D_minimum_NCF LP on the predefined grasp sites.

The scene defines known-good grasp sites on each object (obj_red_box_c1/c2 at the two
opposing box-face centers). This feeds those exact contacts to the standalone hard-
equality LP (min_gamma_for_accel_lp in scripts/3D_minimum_NCF.py) and reports:
  - whether it finds a feasible gamma_min, and
  - per-corner detail: gamma per corner, and which corner (if any) is infeasible
    (the LP returns None on the FIRST infeasible corner, hiding how many fail — we
    instead check every corner so the failure mode is visible).

Contacts can be posed from a real near-box operator sample (samples.jsonl) so the
object/hand geometry matches live conditions, or left at the scene spawn pose.

Usage:
    python simulation/test_nonslack_lp_sites.py                 # scene spawn pose
    python simulation/test_nonslack_lp_sites.py --sample 0      # real near-box pose
"""
import argparse
import importlib
import os
import sys

import numpy as np
import mujoco as mj
from scipy.optimize import linprog

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'scripts'))

from grasp_planner_3d import (GraspConfig3D, _geom_normal_np,  # noqa: E402
                              _build_contact_frame_3d)

_ncf = importlib.import_module('3D_minimum_NCF')
min_gamma_for_accel_lp = _ncf.min_gamma_for_accel_lp
WrenchCheck = _ncf.WrenchCheck

_SCENE = os.path.join(os.path.dirname(_HERE), 'models', 'scene_pick_place.xml')
_DIM = {0: 'Tx', 1: 'Ty', 2: 'Tz', 3: 'Fx', 4: 'Fy', 5: 'Fz'}


def _pose_from_sample(model, data, obj_body, sample_idx):
    import json
    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    s = samples[sample_idx]
    oq = np.asarray(s['obj_qpos'], float)
    jadr = model.jnt_qposadr[model.body_jntadr[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body)]]
    data.qpos[jadr:jadr + 7] = oq[:7]
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    for i, idx in enumerate(act_idx):
        data.qpos[idx] = s['q_seed'][i]
    mj.mj_forward(model, data)
    print(f"[sample {sample_idx}] posed obj={obj_body} + arm at recorded q_seed "
          f"(orig object was {s.get('object')})")


def _build_lp_geometry(model, data, cfg, obj_body):
    """Return the LP inputs (accels/torques, pos, R, mu) for the two predefined sites,
    in object body frame — matching verify()'s construction exactly."""
    bid  = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body)
    ogid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, obj_body + '_geom')
    obj_pos = data.geom_xpos[ogid].copy()
    obj_mat = data.geom_xmat[ogid].reshape(3, 3)
    gsize = model.geom_size[ogid]
    gtype = model.geom_type[ogid]

    p1 = data.site_xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, obj_body + '_c1')].copy()
    p2 = data.site_xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, obj_body + '_c2')].copy()

    n1_out = _geom_normal_np(p1, gtype, obj_pos, obj_mat, gsize)
    n2_out = _geom_normal_np(p2, gtype, obj_pos, obj_mat, gsize)
    _, t1a, t2a = _build_contact_frame_3d(-n1_out)
    _, t1b, t2b = _build_contact_frame_3d(-n2_out)
    R1 = np.column_stack([-n1_out, t1a, t2a])
    R2 = np.column_stack([-n2_out, t1b, t2b])

    mass  = float(model.body_mass[bid])
    inert = model.body_inertia[bid]
    aab   = cfg.ang_accel_budget_xyz
    ab    = cfg.accel_budget_xyz
    mu    = round(0.8 * float(model.geom_friction[ogid][0]), 3)
    R_WO  = data.xmat[bid].reshape(3, 3)
    g_O   = R_WO.T @ model.opt.gravity
    accel = tuple(ab[i] + abs(g_O[i]) for i in range(3))

    p1_O = (R_WO.T @ (p1 - obj_pos)).reshape(3, 1)
    p2_O = (R_WO.T @ (p2 - obj_pos)).reshape(3, 1)
    R1_O = R_WO.T @ R1
    R2_O = R_WO.T @ R2

    task_F = [mass * accel[i] for i in range(3)]
    task_T = [float(inert[i]) * aab[i] for i in range(3)]
    return dict(p1=p1, p2=p2, pos=[p1_O, p2_O], R=[R1_O, R2_O], mu=mu,
                task_F=task_F, task_T=task_T, obj_pos=obj_pos)


def _per_corner_diag(geo):
    """Replicate the LP's per-corner loop but check EVERY corner (don't stop at the
    first infeasible one) so we can report how many fail and the worst directions."""
    Fx, Fy, Fz = geo['task_F']
    Tx, Ty, Tz = geo['task_T']
    mu = geo['mu']
    wc = WrenchCheck(2, geo['pos'], geo['R'], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0],
                     [mu, mu])
    nverts = wc.nverts
    V1 = np.vstack([wc.single_wrench_cone(1.0, geo['pos'][i], geo['R'][i], 1.0, mu)
                    for i in range(2)])
    w_tan = np.zeros(6)   # tan forces are zero here
    nvar = 2 * nverts + 1
    c = np.zeros(nvar); c[-1] = 1.0
    A_eq = np.hstack([V1.T, -w_tan.reshape(6, 1)])
    A_ub = np.zeros((2, nvar))
    for k in range(2):
        A_ub[k, k * nverts:(k + 1) * nverts] = 1
        A_ub[k, -1] = -1
    b_ub = np.zeros(2)
    bounds = [(0, None)] * nvar

    corners = set()
    for tx in (-Tx, Tx):
        for ty in (-Ty, Ty):
            for tz in (-Tz, Tz):
                for fx in (-Fx, Fx):
                    for fy in (-Fy, Fy):
                        for fz in (-Fz, Fz):
                            cc = (tx, ty, tz, fx, fy, fz)
                            if any(cc):
                                corners.add(cc)

    results = []
    for corner in corners:
        res = linprog(c=c, A_eq=A_eq, b_eq=np.array(corner), A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds)
        results.append((corner, res.success, (res.x[-1] if res.success else None)))
    return results, len(corners)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--object', default='obj_red_box')
    ap.add_argument('--sample', type=int, default=None,
                    help="samples.jsonl index for a near-box pose (default: scene spawn)")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(_SCENE)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    obj = args.object

    if args.sample is not None:
        _pose_from_sample(model, data, obj, args.sample)

    cfg = GraspConfig3D(obj_geom=obj + '_geom', obj_body=obj)
    geo = _build_lp_geometry(model, data, cfg, obj)

    print("\n" + "=" * 72)
    print(f"NON-SLACK LP (scripts/3D_minimum_NCF.py) on predefined sites — {obj}")
    print("=" * 72)
    print(f"  contacts (world): c1={np.round(geo['p1'], 3)}  c2={np.round(geo['p2'], 3)}")
    print(f"  friction mu (LP): {geo['mu']}")
    print(f"  task force  (N):  Fx={geo['task_F'][0]:.3f}  Fy={geo['task_F'][1]:.3f}  "
          f"Fz={geo['task_F'][2]:.3f}")
    print(f"  task torque (Nm): Tx={geo['task_T'][0]:.4f}  Ty={geo['task_T'][1]:.4f}  "
          f"Tz={geo['task_T'][2]:.4f}")

    # Headline: the actual function the recommender calls.
    g = min_gamma_for_accel_lp(
        geo['task_F'][0], geo['task_F'][1], geo['task_F'][2],
        geo['task_T'][0], geo['task_T'][1], geo['task_T'][2],
        n=2, pos=geo['pos'], R=geo['R'], ncf=[1.0, 1.0],
        tan_y=[0.0, 0.0], tan_z=[0.0, 0.0], mu=[geo['mu'], geo['mu']])
    print("-" * 72)
    print(f"  min_gamma_for_accel_lp() -> {g}   "
          f"({'FEASIBLE' if g is not None else 'INFEASIBLE'})")

    # Diagnostic: check every corner (the function stops at the first failure).
    results, n_corners = _per_corner_diag(geo)
    feas = [r for r in results if r[1]]
    infeas = [r for r in results if not r[1]]
    print("-" * 72)
    print(f"  per-corner check: {len(feas)}/{n_corners} feasible, "
          f"{len(infeas)} infeasible")
    if feas:
        gammas = [r[2] for r in feas]
        worst = max(feas, key=lambda r: r[2])
        print(f"  feasible-corner gamma: min={min(gammas):.3f}  max={max(gammas):.3f}")
        print(f"  hardest feasible corner (Tx,Ty,Tz,Fx,Fy,Fz):")
        print(f"    {tuple(round(v, 3) for v in worst[0])}  -> gamma={worst[2]:.3f}")
    if infeas:
        print(f"  example INFEASIBLE corners (grasp cannot resist these at ANY gamma):")
        for corner, _, _ in infeas[:4]:
            labeled = ", ".join(f"{_DIM[i]}={corner[i]:+.3f}" for i in range(6))
            print(f"    {labeled}")
    print("=" * 72)
    if g is None:
        print("The hard-equality LP declares this known-good grasp INFEASIBLE because at")
        print("least one corner wrench can't be exactly balanced by the friction cones.")
        print("A slack-relaxed LP would instead return a (possibly large) gamma + residual.")
    print()


if __name__ == '__main__':
    main()

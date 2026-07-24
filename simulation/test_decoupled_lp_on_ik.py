"""Decoupled approach: run IK-only (contacts free, wrench OFF), then wrench-check the
CONVERGED contacts with the standalone LP.

Rationale: test_recommender_contact_lift showed the IK-only solve converges 11-12/12 and
lifts the contacts ~18mm up the face on its own. The decoupled plan is: let IK find
reachable contacts, THEN certify wrench feasibility as a separate gamma-only LP on those
contacts (removing the 630 cone variables from the NLP).

This tests whether the IK-converged contacts are actually good for internal forces:
  - gamma  : min internal-force scale to resist the task wrench. LOW = efficient grasp,
             HIGH = marginal (needs lots of squeeze), None = geometrically infeasible.
             This is exactly the "are these contacts suitable for internal forces?" metric.
  - span_margin : force-closure geometry check (positive = achievable with mu). Catches
             contacts whose normals drifted out of opposition.
  - dz    : how far the contacts lifted from the face center.

CORRECT NORMALS: for each converged contact p, n_out = outward geom normal at p (via
_geom_normal_np). The LP wants contact frames with col0 = OUTWARD normal (force pushing
ON the object), expressed in the OBJECT body frame — exactly verify()'s construction. We
also project the grasp-axis component out of the torque budget (the antipodal-pinch
singularity fix), matching the live GRASP path and the projection added to the NLP.

Usage:
    python simulation/test_decoupled_lp_on_ik.py
    python simulation/test_decoupled_lp_on_ik.py --sizes 0.04,0.05,0.06 --accel 0.5
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'scripts'))

from grasp_planner_3d import (GraspConfig3D, GraspPlanner3D,  # noqa: E402
                              _fixed_antipodal_seed, _geom_normal_np,
                              _build_contact_frame_3d, _span_margin)

_ncf = importlib.import_module('3D_minimum_NCF')
min_gamma_for_accel_lp = _ncf.min_gamma_for_accel_lp

_SCENE = os.path.join(os.path.dirname(_HERE), 'models', 'scene_pick_place.xml')
_BOX = 'obj_red_box'


def _resize_box(model, half):
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, _BOX + '_geom')
    model.geom_size[gid] = np.array([half, half, half], float)
    for nm, x in ((_BOX + '_c1', -half), (_BOX + '_c2', half)):
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, nm)
        model.site_pos[sid] = np.array([x, 0.0, 0.0])


def _lp_gamma_on_contacts(model, data, cfg, p1_w, p2_w, accel, ang_accel,
                          moment_ref_mode=False):
    """Standalone LP on two world-frame contacts. Builds OUTWARD-normal contact frames
    in the object body frame (verify()'s recipe) and projects grasp-axis torque out.
    Returns (gamma, span_margin_deg)."""
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _BOX)
    ogid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, _BOX + '_geom')
    obj_pos = data.geom_xpos[ogid].copy()
    obj_mat = data.geom_xmat[ogid].reshape(3, 3)
    gsize = model.geom_size[ogid]
    gtype = model.geom_type[ogid]

    # OUTWARD normals at the converged contacts.
    n1_out = _geom_normal_np(p1_w, gtype, obj_pos, obj_mat, gsize)
    n2_out = _geom_normal_np(p2_w, gtype, obj_pos, obj_mat, gsize)
    # Contact frame col0 = OUTWARD normal (force on the object). _build_contact_frame_3d
    # takes the FIRST column vector; pass n_out so col0 = n_out.
    _, t1a, t2a = _build_contact_frame_3d(n1_out)
    _, t1b, t2b = _build_contact_frame_3d(n2_out)
    R1 = np.column_stack([n1_out, t1a, t2a])
    R2 = np.column_stack([n2_out, t1b, t2b])

    mass = float(model.body_mass[bid])
    inert = model.body_inertia[bid]
    mu = round(0.8 * float(model.geom_friction[ogid][0]), 3)
    R_WO = data.xmat[bid].reshape(3, 3)
    g_O = R_WO.T @ model.opt.gravity

    p1_O = (R_WO.T @ (p1_w - obj_pos)).reshape(3, 1)
    p2_O = (R_WO.T @ (p2_w - obj_pos)).reshape(3, 1)
    R1_O = R_WO.T @ R1
    R2_O = R_WO.T @ R2

    # Grasp-axis torque projection (antipodal singularity fix).
    ga = (p2_O - p1_O).flatten()
    ga = ga / (np.linalg.norm(ga) + 1e-12)
    T = np.array([float(inert[i]) * ang_accel[i] for i in range(3)])
    T = T - np.dot(T, ga) * ga
    T = np.abs(T)

    if moment_ref_mode:
        # Grasp-axis reference mode: gravity is passed SEPARATELY as grav_force so its
        # re-datum moment is honestly added; the accel budget stays a pure force box (no
        # |g| folded in). Reference = grasp midpoint (object frame).
        _mref = (0.5 * (p1_O + p2_O)).reshape(3)
        _grav = mass * g_O                      # gravity force vector (object frame)
        g = min_gamma_for_accel_lp(
            mass * accel[0], mass * accel[1], mass * accel[2],
            T[0], T[1], T[2], n=2, pos=[p1_O, p2_O], R=[R1_O, R2_O],
            ncf=[1.0, 1.0], tan_y=[0.0, 0.0], tan_z=[0.0, 0.0], mu=[mu, mu],
            moment_ref=_mref, grav_force=_grav,
            project_grasp_axis_moment=True)
    else:
        # Default CoM-referenced mode: gravity folded into the linear accel budget.
        accel_incl_g = tuple(accel[i] + abs(g_O[i]) for i in range(3))
        g = min_gamma_for_accel_lp(
            mass * accel_incl_g[0], mass * accel_incl_g[1], mass * accel_incl_g[2],
            T[0], T[1], T[2], n=2, pos=[p1_O, p2_O], R=[R1_O, R2_O],
            ncf=[1.0, 1.0], tan_y=[0.0, 0.0], tan_z=[0.0, 0.0], mu=[mu, mu])
    span = np.degrees(_span_margin(n1_out, n2_out, mu))
    return g, span


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', default='0.040,0.050,0.060')
    ap.add_argument('--accel', type=float, default=0.5, help="linear accel budget (m/s^2)")
    ap.add_argument('--ang-accel', type=float, default=1.0, help="angular budget (rad/s^2)")
    ap.add_argument('--w-align', type=float, default=0.0,
                    help="grasp-axis alignment cost weight (keeps contacts antipodal)")
    ap.add_argument('--moment-ref', action='store_true',
                    help="reference the task wrench about the grasp midpoint (not the "
                         "CoM), so raised contacts stay wrench-feasible; gravity re-datumed")
    args = ap.parse_args()
    sizes = [float(x) for x in args.sizes.split(',')]
    accel = (args.accel,) * 3
    ang = (args.ang_accel,) * 3

    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    t0 = time.time()
    print(f"\nDecoupled IK->LP — IK-only solve (wrench OFF) then LP wrench-check on the")
    print(f"  CONVERGED contacts.  accel={args.accel} ang={args.ang_accel}, "
          f"{len(samples)} seeds")
    print("=" * 90)
    print(f"{'full(cm)':>8} {'ik_conv':>8} {'lp_feasible':>12} {'gamma: med':>11} "
          f"{'p90':>7} {'max':>7} {'span_deg':>9} {'dz(mm)':>8}")
    print("-" * 90)

    for half in sizes:
        model = mj.MjModel.from_xml_path(_SCENE)
        _resize_box(model, half)
        act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _BOX)

        n_ikconv, gammas, spans, dzs, n_feas = 0, [], [], [], 0
        for s in samples:
            data = mj.MjData(model)
            oq = np.asarray(s['obj_qpos'], float)
            jadr = model.jnt_qposadr[model.body_jntadr[bid]]
            data.qpos[jadr:jadr + 7] = oq[:7]
            data.qpos[jadr + 2] = half
            for i, idx in enumerate(act_idx):
                data.qpos[idx] = s['q_seed'][i]
            mj.mj_forward(model, data)

            cfg = GraspConfig3D(obj_geom=_BOX + '_geom', obj_body=_BOX, max_iter=200,
                                wrench_constraint=False, w_align=args.w_align)
            planner = GraspPlanner3D(model, mj.MjData(model), cfg)
            planner.data.qpos[:] = data.qpos[:]
            mj.mj_forward(model, planner.data)
            q_ref = np.array([data.qpos[i] for i in act_idx])
            obj_pos = data.xpos[bid].copy()
            R_WO = data.xmat[bid].reshape(3, 3)
            seed = _fixed_antipodal_seed(planner._obj_geom_type, planner._obj_size,
                                         obj_pos, R_WO, [1.0, 0.0, 0.0])
            res = planner.solve(q_ref, obj_pos, p1_init=seed['p1'], p2_init=seed['p2'],
                                d1=-seed['n1_in'], d2=-seed['n2_in'])
            if res.get('p1') is None:
                continue
            if res.get('status') == 'converged':
                n_ikconv += 1
            p1w = np.asarray(res['p1'], float)
            p2w = np.asarray(res['p2'], float)
            # contact lift (object frame z above center)
            p1O = R_WO.T @ (p1w - obj_pos); p2O = R_WO.T @ (p2w - obj_pos)
            dzs.append(0.5 * (p1O[2] + p2O[2]) * 1e3)
            # LP wrench-check on the CONVERGED contacts.
            g, span = _lp_gamma_on_contacts(model, data, cfg, p1w, p2w, accel, ang,
                                            moment_ref_mode=args.moment_ref)
            spans.append(span)
            if g is not None:
                n_feas += 1
                gammas.append(g)
        gammas = np.array(gammas)
        _med = np.median(gammas) if len(gammas) else float('nan')
        _p90 = np.percentile(gammas, 90) if len(gammas) else float('nan')
        _mx = np.max(gammas) if len(gammas) else float('nan')
        print(f"{half*200:>8.1f} {n_ikconv:>6}/{len(samples):<2} "
              f"{n_feas:>7}/{len(samples):<3} {_med:>11.2f} {_p90:>7.2f} {_mx:>7.2f} "
              f"{np.median(spans):>9.1f} {np.mean(dzs):>8.1f}  [{time.time()-t0:.0f}s]")
    print("=" * 90)
    print("gamma = min internal-force scale (LOW is a good grasp; None = infeasible).")
    print("lp_feasible = contacts that CAN resist the wrench. span_deg > 0 = force-closure")
    print("geometry OK. If ik_conv is high but lp_feasible/gamma are poor, the IK-converged")
    print("contacts are reachable but NOT good for internal forces (your concern).\n")


if __name__ == '__main__':
    main()

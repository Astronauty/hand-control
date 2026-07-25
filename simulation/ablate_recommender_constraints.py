"""Constraint ablation for the contact_aware_teleop grasp recommender.

The recommender NLP has been returning best-effort / 0-of-N-seeds-converged on the
box (γ_min=—, WF=NO). This script toggles each independently-gated constraint in
GraspConfig3D on/off and reports, per configuration, how many seeds converge, the
iteration counts, and wrench feasibility — so we can attribute the non-convergence
to a specific constraint (or rule constraints out).

Constraints toggled (each maps to a real gate in GraspPlanner3D._run_stage):
  joint_limits      -> cfg.joint_limits          (opti bounds on q)
  wrench            -> cfg.wrench_constraint      (embedded wrench-cone LP)
  arm_collision     -> cfg.arm_geom_names         (Tier-1 palm+wrist SDFs; [] = off)
  legacy_col        -> cfg.col_constraint         (mid/ring fingertip SDFs; only
                                                    active when arm_geom_names empty)
  ground            -> cfg.col_use_ground         (fingertip-vs-floor planes)

Surface face-pin (include_surface) is NOT toggled: it defines the grasp (contacts on
the object surface) and is always on in the production solve. Removing it changes the
problem into "any two points", not a grasp, so it isn't a meaningful ablation target.

Usage:
    python simulation/ablate_recommender_constraints.py            # default: red box
    python simulation/ablate_recommender_constraints.py --object obj_green_box
    python simulation/ablate_recommender_constraints.py --seeds 5 --repeats 3
    python simulation/ablate_recommender_constraints.py --full     # power-set sweep
"""
import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from grasp_planner_3d import GraspConfig3D, MultiStartGraspPlanner3D  # noqa: E402

_SCENE = os.path.join(os.path.dirname(_HERE), 'models', 'scene_pick_place.xml')


def _tier1_arm_geoms(model):
    """Palm + wrist collision geoms — the Tier-1 subset the live recommender uses."""
    contype0 = model.geom_contype.copy()
    out = []
    for gi in range(model.ngeom):
        g = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gi)
        if not g or contype0[gi] == 0:
            continue
        b = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ''
        if b in ('leap_palm', 'bracelet_link'):
            out.append(g)
    return out


def _run_config(model, base_data, obj_body, obj_geom, q_ref, arm_geoms, flags,
                seeds, repeats, max_iter, accel=None, ang_accel=None):
    """Build a planner for one flag combination and solve `repeats` times.

    flags: dict with keys joint_limits, wrench, arm_collision, legacy_col, ground.
    accel/ang_accel: optional per-axis budget overrides (scalar -> isotropic tuple).
        The task wrench the grasp must resist is mass*(accel + |gravity|) linear and
        inertia*ang_accel angular, so lowering these shrinks the wrench cone the NLP
        must satisfy. ang_accel=0 gives a true zero torque budget; accel=0 still leaves
        a gravity-only linear force (the object must at least not be dropped).
    Returns aggregate stats across repeats × seeds.
    """
    def _tup(v, default):
        if v is None:
            return default
        return (float(v),) * 3 if np.isscalar(v) else tuple(v)

    cfg = GraspConfig3D(
        obj_geom=obj_geom, obj_body=obj_body, max_iter=max_iter,
        joint_limits=flags['joint_limits'],
        wrench_constraint=flags['wrench'],
        col_use_ground=flags['ground'],
        arm_geom_names=(list(arm_geoms) if flags['arm_collision'] else []),
        accel_budget_xyz=_tup(accel, (0.5, 0.5, 0.5)),
        ang_accel_budget_xyz=_tup(ang_accel, (1.0, 1.0, 1.0)),
    )
    planner = MultiStartGraspPlanner3D(model, mj.MjData(model), cfg)
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body)
    obj_pos = base_data.xpos[bid].copy()

    n_conv = n_seeds_total = n_wrench_ok = 0
    iters_conv, iters_all, solve_ms = [], [], []
    best_cost = None
    for _ in range(repeats):
        planner._planner.data.qpos[:] = base_data.qpos[:]
        mj.mj_forward(model, planner._planner.data)
        t0 = time.time()
        res = planner.solve(np.asarray(q_ref, float), obj_pos, max_seeds=seeds)
        solve_ms.append((time.time() - t0) * 1e3)
        for r in res.get('all_results', []):
            n_seeds_total += 1
            it = r.get('iterations')
            if it is not None:
                iters_all.append(it)
            if r.get('status') == 'converged':
                n_conv += 1
                if it is not None:
                    iters_conv.append(it)
            if r.get('wrench_ok'):
                n_wrench_ok += 1
        c = res.get('cost')
        if c is not None and (best_cost is None or c < best_cost):
            best_cost = c
    return {
        'conv': n_conv, 'total': n_seeds_total, 'wrench_ok': n_wrench_ok,
        'iters_conv_mean': (np.mean(iters_conv) if iters_conv else None),
        'iters_all_mean': (np.mean(iters_all) if iters_all else None),
        'solve_ms_mean': np.mean(solve_ms), 'best_cost': best_cost,
    }


# Ablation configurations. "ALL ON" is the current production setting (Tier-1 arm
# collision, wrench, joint limits, ground). Each subsequent row drops exactly one
# constraint so the delta vs ALL-ON isolates that constraint's effect on convergence.
def _default_configs():
    all_on = {'joint_limits': True, 'wrench': True, 'arm_collision': True,
              'legacy_col': True, 'ground': True}
    rows = [('ALL ON (production)', dict(all_on))]
    for key, label in [('arm_collision', '- arm collision (Tier-1)'),
                       ('wrench', '- wrench cone'),
                       ('joint_limits', '- joint limits'),
                       ('ground', '- ground plane'),
                       ('legacy_col', '- legacy mid/ring col')]:
        f = dict(all_on); f[key] = False
        rows.append((label, f))
    # Minimal: only the surface face-pin (everything else off) — the floor of the
    # problem; if THIS doesn't converge the issue is the grasp geometry / seeds, not
    # any add-on constraint.
    rows.append(('MINIMAL (surface only)',
                 {'joint_limits': False, 'wrench': False, 'arm_collision': False,
                  'legacy_col': False, 'ground': False}))
    return rows


def _full_configs():
    """Power set over the 3 heaviest constraints (wrench, arm_collision, joint_limits);
    ground+legacy_col held ON. 8 rows."""
    import itertools
    rows = []
    for wr, arm, jl in itertools.product([True, False], repeat=3):
        label = (f"wrench={'Y' if wr else 'N'} arm={'Y' if arm else 'N'} "
                 f"jlim={'Y' if jl else 'N'}")
        rows.append((label, {'joint_limits': jl, 'wrench': wr, 'arm_collision': arm,
                             'legacy_col': True, 'ground': True}))
    return rows


def _run_accel_sweep(model, data, obj_body, obj_geom, q_ref, arm_geoms,
                     seeds, repeats, max_iter):
    """All constraints ON (production Tier-1); sweep the accel/ang-accel budgets that
    scale the task wrench the cone must resist, from the production default down to
    zero. accel=0 leaves a gravity-only linear force; ang_accel=0 gives a true zero
    torque budget. If convergence recovers as the budget shrinks, the wrench cone is
    over-demanding (task too hard) rather than structurally malformed."""
    all_on = {'joint_limits': True, 'wrench': True, 'arm_collision': True,
              'legacy_col': True, 'ground': True}
    # (accel m/s², ang_accel rad/s²). Default is (0.5, 1.0). Scale both toward zero,
    # plus a linear-only and torque-only row to separate the two contributions.
    rows = [
        ('production (0.5, 1.0)',   0.5,  1.0),
        ('half     (0.25, 0.5)',    0.25, 0.5),
        ('tenth    (0.05, 0.1)',    0.05, 0.1),
        ('lin-only (0.0, 1.0)',     0.0,  1.0),   # zero torque budget only
        ('rot-only (0.5, 0.0)',     0.5,  0.0),   # gravity linear, zero torque
        ('ZERO     (0.0, 0.0)',     0.0,  0.0),   # gravity-only, no rotation
    ]
    print(f"\nAccel-budget sweep — object={obj_body}  seeds={seeds}  repeats={repeats}"
          f"  max_iter={max_iter}  (all constraints ON, Tier-1)")
    print("The wrench cone must resist mass*(accel+|g|) linear and inertia*ang_accel "
          "angular.")
    print("=" * 92)
    print(f"{'accel budget (lin,ang)':<26} {'conv':>9} {'wrench_ok':>10} {'it(conv)':>9} "
          f"{'it(all)':>8} {'ms':>7} {'best_cost':>10}")
    print("-" * 92)
    for label, a, aa in rows:
        s = _run_config(model, data, obj_body, obj_geom, q_ref, arm_geoms, all_on,
                        seeds, repeats, max_iter, accel=a, ang_accel=aa)
        conv_str = f"{s['conv']}/{s['total']}"
        wok_str = f"{s['wrench_ok']}/{s['total']}"
        itc = f"{s['iters_conv_mean']:.0f}" if s['iters_conv_mean'] is not None else '-'
        ita = f"{s['iters_all_mean']:.0f}" if s['iters_all_mean'] is not None else '-'
        bc = f"{s['best_cost']:.3g}" if s['best_cost'] is not None else '-'
        print(f"{label:<26} {conv_str:>9} {wok_str:>10} {itc:>9} {ita:>8} "
              f"{s['solve_ms_mean']:>7.0f} {bc:>10}")
    print("=" * 92)
    print("Read: if conv climbs as the budget shrinks, the wrench task is too demanding")
    print("for the geometry (over-constrained), not malformed. If even ZERO (gravity-only,")
    print("no rotation) stays 0/N while --ablation '- wrench cone' converged, the cone")
    print("MACHINERY itself is the problem, independent of how hard the task is.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--object', default='obj_red_box',
                    help="object body name (default: obj_red_box)")
    ap.add_argument('--seeds', type=int, default=3, help="seeds per solve (default 3)")
    ap.add_argument('--repeats', type=int, default=3,
                    help="solves per config, averaged (default 3)")
    ap.add_argument('--max-iter', type=int, default=120,
                    help="per-stage NLP iteration cap (default 120)")
    ap.add_argument('--full', action='store_true',
                    help="power-set sweep over wrench/arm/joint-limit instead of one-drop")
    ap.add_argument('--accel-sweep', action='store_true',
                    help="hold all constraints ON and sweep the accel/ang-accel budgets "
                         "(the task wrench the cone must resist) down to zero")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(_SCENE)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    obj_body = args.object
    obj_geom = obj_body + '_geom'
    if mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body) < 0:
        sys.exit(f"object body '{obj_body}' not found in scene")

    # q_ref: the model's home actuated pose — the same starting point the live
    # recommender seeds its regularizer from before the operator moves. (The live
    # solve uses the operator's near-box pose; this is a fixed, reproducible stand-in.)
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    q_ref = np.array([data.qpos[i] for i in act_idx])

    arm_geoms = _tier1_arm_geoms(model)

    if args.accel_sweep:
        _run_accel_sweep(model, data, obj_body, obj_geom, q_ref, arm_geoms,
                         args.seeds, args.repeats, args.max_iter)
        return

    configs = _full_configs() if args.full else _default_configs()

    print(f"\nAblation — object={obj_body}  seeds={args.seeds}  repeats={args.repeats}"
          f"  max_iter={args.max_iter}  (Tier-1 arm geoms: {len(arm_geoms)})")
    print("=" * 92)
    print(f"{'config':<26} {'conv':>9} {'wrench_ok':>10} {'it(conv)':>9} "
          f"{'it(all)':>8} {'ms':>7} {'best_cost':>10}")
    print("-" * 92)
    for label, flags in configs:
        s = _run_config(model, data, obj_body, obj_geom, q_ref, arm_geoms, flags,
                        args.seeds, args.repeats, args.max_iter)
        conv_str = f"{s['conv']}/{s['total']}"
        wok_str = f"{s['wrench_ok']}/{s['total']}"
        itc = f"{s['iters_conv_mean']:.0f}" if s['iters_conv_mean'] is not None else '-'
        ita = f"{s['iters_all_mean']:.0f}" if s['iters_all_mean'] is not None else '-'
        bc = f"{s['best_cost']:.3g}" if s['best_cost'] is not None else '-'
        print(f"{label:<26} {conv_str:>9} {wok_str:>10} {itc:>9} {ita:>8} "
              f"{s['solve_ms_mean']:>7.0f} {bc:>10}")
    print("=" * 92)
    print("Read: conv = seeds whose NLP reached 'converged' (not best-effort). If ALL-ON")
    print("is 0/N and a one-drop row jumps to N/N, that dropped constraint is the cause.")
    print("If MINIMAL (surface only) is also 0/N, the geometry/seeds are the problem, not")
    print("any add-on constraint.\n")


if __name__ == '__main__':
    main()

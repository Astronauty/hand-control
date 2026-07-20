"""Offline sweep of the collision-IK cost weights against a recorded pose dataset.

Goal
----
Find the ConstrainedIKSolver weights that minimize the gap between the grasp
recommender's contacts and where the collision-refined IK actually lands the
fingertips — i.e. tighten the "recommender NLP -> collision IK" tip-error delta
you watch on the live dashboard.

The dataset is produced by kinova_leap_pick_place.py --mode contact_aware_teleop
--record-samples PATH: press R while teleoping to append the live pose + object
+ recommender candidate. Because the recommendation is weight-independent, it's
captured once at record time and reused for every weight combination here — the
sweep only re-runs the collision IK.

This module deliberately reconstructs the SAME solver setup the sim uses (robot
collision-geom list, object geom list, per-finger pad-surface offsets, tiered
active-finger clearance mask, shape-aware inward normals). Keep these derivations
in sync with kinova_leap_pick_place.py if that file's setup changes.

Usage
-----
    uv run python tune_ik_weights.py samples.jsonl
    uv run python tune_ik_weights.py samples.jsonl --tip 20 60 100 200 \
        --posture-arm 1e-5 1e-4 --posture-hand 1e-5 1e-3 1e-2 \
        --orient 0.0 1.0 --clearance 0.005 --top 15
    uv run python tune_ik_weights.py samples.jsonl --csv results.csv

Objective (lower = better): mean over samples of the site->contact tip-error
INCREASE from the recommender-q pose to the collision-IK pose, with a large fixed
penalty added per sample that fails to converge or violates the requested
clearance — so the winning weights are accurate AND robust, not just good on the
easy poses.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import sys
import time

import numpy as np
import mujoco as mj


@contextlib.contextmanager
def _suppress_stdout():
    """Silence the ConstrainedIKSolver's unconditional per-solve diagnostics so the
    sweep's own progress lines stay readable. stderr (real errors) passes through."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield

# Same import shim the sim uses so _geom_normal_np resolves.
sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/simulation')
from grasp_planner_3d import _geom_normal_np                       # noqa: E402
from grasp_control import ConstrainedIKSolver                       # noqa: E402
from grasp_control.constrained_ik import configure_sqp             # noqa: E402

# --- Constants that MUST match kinova_leap_pick_place.py ---------------------
MODEL_PATH  = 'models/scene_pick_place.xml'
N_ROBOT     = 23                          # 7 arm + 16 LEAP hand
FINGER_SET  = ["index", "thumb"]          # order the sim solves in
FINGER_CODE = {"index": "if", "middle": "mf", "ring": "rf", "thumb": "th"}
FINGER_TIP_SITES = {"index": "leap_if_ds_tip", "middle": "leap_mf_ds_tip",
                    "ring": "leap_rf_ds_tip", "thumb": "leap_th_ds_tip"}
_OBJ_GEOM_NAMES = [
    'obj_red_box_geom', 'obj_red_sphere_geom',
    'obj_blue_cylinder_geom', 'obj_blue_capsule_geom',
    'obj_green_box_geom', 'obj_green_cylinder_geom',
    'floor',
]
ANOBJ_DISABLE = -1.0
# Per-sample penalty (mm-equivalent) added to the objective when the IK fails to
# converge or violates the requested clearance. Large vs a typical gap (~10-40mm)
# so any robustness failure dominates a small accuracy win.
FAIL_PENALTY_MM = 500.0


def _build_solver_setup(model):
    """Reconstruct the sim's IK-relevant model derivations. Returns a dict of the
    reusable pieces (geom-name lists, tip-site ids, pad offsets, clearance mask)."""
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)

    id_C = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
            for f in FINGER_SET]
    _TIP_GEOM_IDS = {f: mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM,
                                      f'leap_{FINGER_CODE[f]}_tip')
                     for f in FINGER_SET}

    # Robot collision geoms: every named collision geom on the four LEAP fingers +
    # palm + wrist (compiled contype != 0). The recorded dataset is from a normal run
    # (no --viz-only), so contype is intact and we read it directly off the model.
    _active_body_prefixes = tuple(f'leap_{code}_' for code in FINGER_CODE.values())
    robot_geom_names = []
    for gi in range(model.ngeom):
        gname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gi)
        if not gname or model.geom_contype[gi] == 0:
            continue
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ''
        if (any(bname.startswith(p) for p in _active_body_prefixes)
                or bname in ('leap_palm', 'bracelet_link')):
            robot_geom_names.append(gname)

    # Per-finger pad-surface offset: distance from the tip SITE to the fingerpad
    # SURFACE along the pad normal (-x of the site frame). Identical to the sim's
    # _pad_surface_offset.
    def _pad_surface_offset(f):
        gid = _TIP_GEOM_IDS[f]
        sid = id_C[FINGER_SET.index(f)]
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts_W = (data.geom_xmat[gid].reshape(3, 3)
                   @ model.mesh_vert[adr:adr + num].T).T + data.geom_xpos[gid]
        pad_dir_W = -data.site_xmat[sid].reshape(3, 3)[:, 0]
        return float(np.max((verts_W - data.site_xpos[sid]) @ pad_dir_W))

    pad_offset = {f: _pad_surface_offset(f) for f in FINGER_SET}

    # Tiered active-finger clearance mask vs the target object (matches the sim).
    active_finger_geoms = {g for g in robot_geom_names
                           if any(g.startswith(f'leap_{FINGER_CODE[f]}_')
                                  for f in FINGER_SET)}

    def _active_obj_clearance(g):
        if '_ds_' in g or g.endswith('_tip'):
            return ANOBJ_DISABLE
        if g.startswith(('leap_if_md', 'leap_th_px')):
            return -0.010
        return 0.002

    active_clearance_by_geom = {g: _active_obj_clearance(g)
                                for g in active_finger_geoms}

    return {
        'id_C': id_C,
        'robot_geom_names': robot_geom_names,
        'pad_offset': pad_offset,
        'active_clearance_by_geom': active_clearance_by_geom,
    }


def _obj_ids(model, obj_name):
    return {
        'id_body': mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_name),
        'id_geom': mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, obj_name + '_geom'),
    }


def load_samples(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _prep_sample(model, setup, sample, act_idx):
    """Turn a raw JSON sample into everything solve() needs, once (weight-independent):
    the seed pose q_seed, the object qpos, the pad-backed-off ik_targets, the raw
    contacts p_S_W, the inward normals, and the recommender-q full-robot pose. Returns
    None if the object isn't in this model."""
    obj_name = sample['object']
    try:
        oids = _obj_ids(model, obj_name)
    except Exception:
        return None

    q_seed   = np.asarray(sample['q_seed'], float)
    obj_qpos = np.asarray(sample['obj_qpos'], float)
    rec_q    = np.asarray(sample['rec_q'], float)
    p1       = np.asarray(sample['rec_p1'], float)   # thumb
    p2       = np.asarray(sample['rec_p2'], float)   # index

    # Pose a scratch MjData at the SEED so object geom poses (for the normals) reflect
    # the recorded scene.
    d = mj.MjData(model)
    d.qpos[:N_ROBOT] = q_seed
    d.qpos[N_ROBOT:N_ROBOT + len(obj_qpos)] = obj_qpos
    mj.mj_forward(model, d)

    # Contacts in FINGER_SET order (index<-p2, thumb<-p1), matching _run_ik_recommended.
    by_finger = {'thumb': p1, 'index': p2}
    p_S_W = [np.asarray(by_finger[f], float).copy() for f in FINGER_SET]

    gid   = oids['id_geom']
    gtype = int(model.geom_type[gid])
    c     = d.geom_xpos[gid].copy()
    R     = d.geom_xmat[gid].reshape(3, 3).copy()
    size  = model.geom_size[gid]
    n1_in = -_geom_normal_np(p1, gtype, c, R, size)
    n2_in = -_geom_normal_np(p2, gtype, c, R, size)
    n_by_finger = {'thumb': n1_in, 'index': n2_in}
    inward_S_W  = [np.asarray(n_by_finger[f], float).copy() for f in FINGER_SET]

    pad = setup['pad_offset']
    ik_targets = [p - pad[f] * n
                  for f, p, n in zip(FINGER_SET, p_S_W, inward_S_W)]

    # Recommender-q as a full-robot warm-start over the seed (actuated indices only).
    q_warm = q_seed.copy()
    for i, idx in enumerate(act_idx):
        q_warm[idx] = rec_q[i]

    # Recommender's OWN site->contact tip error (the "green" baseline the gap is measured
    # against). FK at q_warm with the object posed.
    d.qpos[:N_ROBOT] = q_warm
    mj.mj_forward(model, d)
    nlp_err_mm = [float(np.linalg.norm(d.site_xpos[s] - p) * 1e3)
                  for s, p in zip(setup['id_C'], p_S_W)]

    return {
        'obj_name':   obj_name,
        'q_seed':     q_seed,
        'obj_qpos':   obj_qpos,
        'ik_targets': ik_targets,
        'p_S_W':      p_S_W,
        'inward_S_W': inward_S_W,
        'q_warm':     q_warm,
        'nlp_err_mm': nlp_err_mm,
    }


def _make_solver(model, setup, tip_w, posture_arm, posture_hand, orient_w, clearance):
    posture_vec = np.r_[np.full(7, posture_arm),
                        np.full(N_ROBOT - 7, posture_hand)]
    solver = ConstrainedIKSolver(
        model, N_ROBOT,
        arm_geom_names=setup['robot_geom_names'],
        obj_geom_names=_OBJ_GEOM_NAMES,
        clearance=clearance,
        posture_weight=posture_vec,
        pad_axis=(-1.0, 0.0, 0.0),
        tip_weight=tip_w,
        orient_weight=orient_w,
        max_iter=800,
    )
    configure_sqp(solver)
    return solver


def _eval_weights(model, setup, prepared, weights):
    """Run the collision IK for every prepared sample under one weight tuple; return
    per-sample records and the aggregate objective."""
    tip_w, posture_arm, posture_hand, orient_w, clearance = weights
    with _suppress_stdout():
        solver = _make_solver(model, setup, tip_w, posture_arm, posture_hand,
                              orient_w, clearance)
    id_C = setup['id_C']

    data = mj.MjData(model)
    per_sample = []
    for p in prepared:
        data.qpos[:] = 0.0
        data.qpos[:N_ROBOT] = p['q_seed']
        data.qpos[N_ROBOT:N_ROBOT + len(p['obj_qpos'])] = p['obj_qpos']
        mj.mj_forward(model, data)

        failed = False
        try:
            with _suppress_stdout():
                q_sol = solver.solve(
                    data, id_C, p['ik_targets'],
                    q_bias=p['q_seed'], q_init=p['q_warm'],
                    reduced_clearance_geoms=setup['active_clearance_by_geom'],
                    inward_dirs=p['inward_S_W'])
        except Exception:
            failed = True
            q_sol = None

        # NOTE: on solver failure ConstrainedIKSolver.solve() returns the best iterate
        # (not None) and records success=False in last_metrics — so read that, not q_sol.
        if q_sol is None or not solver.last_metrics.get('success', False):
            per_sample.append({'obj': p['obj_name'], 'failed': True,
                               'ik_err_mm': None, 'gap_mm': None,
                               'min_slack_mm': solver.last_metrics.get('min_slack_mm')})
            continue

        # Achieved site->contact tip error (the "orange" collision-IK number).
        d2 = mj.MjData(model)
        d2.qpos[N_ROBOT:N_ROBOT + len(p['obj_qpos'])] = p['obj_qpos']
        d2.qpos[:N_ROBOT] = q_sol
        mj.mj_forward(model, d2)
        ik_err_mm = [float(np.linalg.norm(d2.site_xpos[s] - c) * 1e3)
                     for s, c in zip(id_C, p['p_S_W'])]
        # Gap = how much the collision IK gave up vs the recommender, per finger, meaned.
        gap = [ie - ne for ie, ne in zip(ik_err_mm, p['nlp_err_mm'])]
        gap_mm = float(np.mean(gap))
        # Clearance robustness: solver stashes the tightest slack in last_metrics.
        min_slack_mm = solver.last_metrics.get('min_slack_mm')
        clear_viol = (min_slack_mm is not None and min_slack_mm < -1.0)  # >1mm intrusion

        per_sample.append({
            'obj': p['obj_name'],
            'failed': failed or clear_viol,
            'ik_err_mm': ik_err_mm,
            'gap_mm': gap_mm,
            'min_slack_mm': min_slack_mm,
        })

    # Objective: mean gap, with a fixed penalty per failed/violating sample.
    n = len(per_sample)
    obj_terms = []
    n_fail = 0
    for r in per_sample:
        if r['failed'] or r['gap_mm'] is None:
            obj_terms.append(FAIL_PENALTY_MM)
            n_fail += 1
        else:
            obj_terms.append(r['gap_mm'])
    objective = float(np.mean(obj_terms)) if n else float('inf')
    valid_gaps = [r['gap_mm'] for r in per_sample
                  if not r['failed'] and r['gap_mm'] is not None]
    return {
        'weights': weights,
        'objective': objective,
        'n': n,
        'n_fail': n_fail,
        'mean_gap_ok': float(np.mean(valid_gaps)) if valid_gaps else None,
        'max_gap_ok': float(np.max(valid_gaps)) if valid_gaps else None,
        'per_sample': per_sample,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', help='JSONL file from --record-samples')
    ap.add_argument('--tip', type=float, nargs='+', default=[20.0, 60.0, 100.0, 200.0],
                    help='tip_weight grid')
    ap.add_argument('--posture-arm', type=float, nargs='+', default=[1e-5, 1e-4],
                    help='POSTURE_W_ARM grid (weight on arm joints 1..7)')
    ap.add_argument('--posture-hand', type=float, nargs='+', default=[1e-5, 1e-3, 1e-2],
                    help='POSTURE_W_HAND grid (weight on the 16 LEAP joints)')
    ap.add_argument('--orient', type=float, nargs='+', default=[0.0, 1.0],
                    help='orient_weight grid')
    ap.add_argument('--clearance', type=float, nargs='+', default=[0.005],
                    help='clearance grid (m)')
    ap.add_argument('--top', type=int, default=10, help='how many best rows to print')
    ap.add_argument('--csv', default=None, help='optional path to write full results CSV')
    args = ap.parse_args()

    print(f"[tune] loading model {MODEL_PATH} ...")
    model = mj.MjModel.from_xml_path(MODEL_PATH)
    setup = _build_solver_setup(model)
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    print(f"[tune] {len(setup['robot_geom_names'])} robot geoms, "
          f"pad offsets " + ", ".join(f"{f}={setup['pad_offset'][f]*1e3:.1f}mm"
                                       for f in FINGER_SET))

    raw = load_samples(args.dataset)
    prepared = [pr for pr in (_prep_sample(model, setup, s, act_idx) for s in raw)
                if pr is not None]
    n_drop = len(raw) - len(prepared)
    print(f"[tune] {len(prepared)} usable samples"
          + (f" ({n_drop} dropped: object not in model)" if n_drop else ""))
    if not prepared:
        print("[tune] no usable samples — nothing to sweep."); return
    # Baseline recommender tip error, for context.
    base = np.mean([np.mean(p['nlp_err_mm']) for p in prepared])
    print(f"[tune] mean recommender-q tip error across samples: {base:.1f} mm "
          f"(the 'green' baseline the gap is measured from)\n")

    grid = list(itertools.product(args.tip, args.posture_arm, args.posture_hand,
                                  args.orient, args.clearance))
    print(f"[tune] sweeping {len(grid)} weight combinations "
          f"x {len(prepared)} samples = {len(grid)*len(prepared)} IK solves ...\n")

    results = []
    t0 = time.time()
    for gi, w in enumerate(grid, 1):
        res = _eval_weights(model, setup, prepared, w)
        results.append(res)
        tw, pa, ph, ow, cl = w
        print(f"  [{gi:>3}/{len(grid)}] tip={tw:<6g} pArm={pa:<7g} pHand={ph:<7g} "
              f"orient={ow:<4g} clr={cl*1e3:.0f}mm  ->  "
              f"obj={res['objective']:7.2f}  "
              f"meanGap={('%.1f' % res['mean_gap_ok']) if res['mean_gap_ok'] is not None else '  n/a':>6}mm  "
              f"fails={res['n_fail']}/{res['n']}")
    dt = time.time() - t0

    results.sort(key=lambda r: r['objective'])
    print(f"\n[tune] done in {dt:.1f}s. Top {min(args.top, len(results))} by objective "
          f"(mean gap + {FAIL_PENALTY_MM:.0f}mm/fail):\n")
    hdr = (f"{'rank':>4}  {'tip':>6} {'pArm':>8} {'pHand':>8} {'orient':>6} {'clr_mm':>6}  "
           f"{'objective':>9}  {'meanGap':>8}  {'maxGap':>7}  {'fails':>7}")
    print(hdr); print('-' * len(hdr))
    for rank, r in enumerate(results[:args.top], 1):
        tw, pa, ph, ow, cl = r['weights']
        mg = '%.1f' % r['mean_gap_ok'] if r['mean_gap_ok'] is not None else 'n/a'
        xg = '%.1f' % r['max_gap_ok'] if r['max_gap_ok'] is not None else 'n/a'
        print(f"{rank:>4}  {tw:>6g} {pa:>8g} {ph:>8g} {ow:>6g} {cl*1e3:>6.0f}  "
              f"{r['objective']:>9.2f}  {mg:>8}  {xg:>7}  {r['n_fail']:>3}/{r['n']:<3}")

    best = results[0]
    tw, pa, ph, ow, cl = best['weights']
    print(f"\n[tune] BEST -> set in kinova_leap_pick_place.py:")
    print(f"         tip_weight={tw:g}")
    print(f"         POSTURE_W_ARM={pa:g}")
    print(f"         POSTURE_W_HAND={ph:g}")
    print(f"         orient_weight={ow:g}")
    print(f"         clearance={cl:g}")

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['tip', 'posture_arm', 'posture_hand', 'orient', 'clearance_m',
                         'objective', 'mean_gap_ok_mm', 'max_gap_ok_mm', 'n_fail', 'n'])
            for r in results:
                tw, pa, ph, ow, cl = r['weights']
                wr.writerow([tw, pa, ph, ow, cl, r['objective'],
                             r['mean_gap_ok'], r['max_gap_ok'], r['n_fail'], r['n']])
        print(f"\n[tune] full results -> {args.csv}")


if __name__ == '__main__':
    main()

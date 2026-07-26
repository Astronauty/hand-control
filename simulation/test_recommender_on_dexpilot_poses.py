"""Benchmark the grasp recommender's IK convergence on REAL teleoped hand poses.

Samples random frames from the latest dexpilot pose_trace logs (logs/dexpilot_*/
pose_trace.npz, key 'q_robot' = 23-DOF actuated joints per frame) and warm-starts the
recommender NLP (MultiStartGraspPlanner3D) from each, exactly as the live contact_aware_
teleop path does (_fire_recommender -> planner.solve(q_snap, obj_pos)). Replicates the live
recommender config INCLUDING the finger-link collision constraints just added to
_get_cat_planner, so this exercises the current code.

Reports, per sampled pose and in aggregate:
  - solve time (ms)
  - convergence (did it return contacts p1/p2, and the NLP status)
  - IK accuracy: recommender's own tip-site error at its q vs its recommended contacts (mm)
  - collision: exact mj_geomDistance of the recommender's RAW q finger links vs the object
               (worst signed distance, mm; negative = interpenetration)
  - wrench feasibility (verify()'s datum gamma certificate)

Usage:
  python simulation/test_recommender_on_dexpilot_poses.py [--n 20] [--seeds 3] [--log DIR]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from grasp_planner_3d import (GraspConfig3D, MultiStartGraspPlanner3D,  # noqa: E402
                              _geom_normal_np)

SCENE_XML = os.path.join(_ROOT, 'models', 'scene_pick_place.xml')
N_ROBOT = 23

# Object 0 = red box (prox_idx=0 in every recent dexpilot log).
OBJ_NAME = 'obj_red_box'
OBJ_GEOM = 'obj_red_box_geom'

# Disturbance budgets — must match the live NCF_* defaults the recommender uses.
NCF_ACCEL_BUDGET_XYZ = (20.0, 20.0, 20.0)   # match the live top-level budget
NCF_ANG_ACCEL_BUDGET = (0.5, 0.5, 0.5)

FINGER_CODES_ACTIVE = ('if', 'th')   # FINGER_SET = index + thumb
ANOBJ_DISABLE = -1.0


def build_robot_geom_names(model):
    """Replicate _robot_geom_names: named, collision (contype!=0), on a leap finger / palm /
    wrist body."""
    codes = ('if', 'th', 'mf', 'rf')
    prefixes = tuple(f'leap_{c}_' for c in codes)
    names = []
    for gi in range(model.ngeom):
        n = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gi)
        if not n or model.geom_contype[gi] == 0:
            continue
        bn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ''
        if any(bn.startswith(p) for p in prefixes) or bn in ('leap_palm', 'bracelet_link'):
            names.append(n)
    return names


def active_obj_clearance(g):
    """Same tiers as kinova_leap_pick_place._active_obj_clearance."""
    if '_ds_' in g or g.endswith('_tip'):
        return ANOBJ_DISABLE
    if g.startswith(('leap_if_md', 'leap_th_px')):
        return -0.010
    return 0.002


def build_live_config(model, accel_budget=None, ang_budget=None, edge_margin_m=0.015):
    """Reproduce _get_cat_planner's live GraspConfig3D EXACTLY (single source of truth for
    the recommender config across all tests): palm/wrist + active-finger links (tiered obj
    clearance) + non-active middle/ring (ground-only), orient_weight, ground_clearance_m.

    accel_budget/ang_budget/edge_margin_m override the disturbance budget and edge tolerance
    for sweeping — defaults match live."""
    accel_budget = tuple(accel_budget) if accel_budget is not None else NCF_ACCEL_BUDGET_XYZ
    ang_budget = tuple(ang_budget) if ang_budget is not None else NCF_ANG_ACCEL_BUDGET
    robot_geoms = build_robot_geom_names(model)

    rec_arm_geoms = [
        g for g in robot_geoms
        if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY,
                          model.geom_bodyid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g)]) or '')
        in ('leap_palm', 'bracelet_link')
    ]
    active_finger_geoms = {g for g in robot_geoms
                           if any(g.startswith(f'leap_{c}_') for c in FINGER_CODES_ACTIVE)}
    rec_finger_geoms = sorted(g for g in active_finger_geoms
                              if not ('_ds_' in g or g.endswith('_tip')))
    # Non-active fingers (middle/ring): ACTIVE OBJECT constraint at +1mm on the bounding-
    # sphere surface (they don't grasp, so they must stay OFF the target — a positive clearance
    # keeps their curled links/tips out of the box instead of being pressed in during the
    # squeeze). ALL mf/rf collision geoms included, incl. the ds/tip contact tier. The floor
    # test still applies. Matches _get_cat_planner (_REC_NONACTIVE_OBJ_CLR).
    REC_NONACTIVE_OBJ_CLR = 0.001
    rec_nonactive_geoms = sorted(
        g for g in robot_geoms
        if g.startswith('leap_mf_') or g.startswith('leap_rf_'))
    arm_geom_names = list(rec_arm_geoms) + rec_finger_geoms + rec_nonactive_geoms
    obj_clearance = {g: active_obj_clearance(g) for g in rec_finger_geoms}
    obj_clearance.update({g: REC_NONACTIVE_OBJ_CLR for g in rec_nonactive_geoms})

    print(f"[cfg] recommender collision geoms: {len(rec_arm_geoms)} palm/wrist + "
          f"{len(rec_finger_geoms)} active + {len(rec_nonactive_geoms)} non-active "
          f"(mf/rf, obj +{REC_NONACTIVE_OBJ_CLR*1e3:.0f}mm) = {len(arm_geom_names)} total")

    cfg = GraspConfig3D(
        obj_geom=OBJ_GEOM, obj_body=OBJ_NAME,
        max_iter=120, arm_geom_names=arm_geom_names,
        obj_clearance_by_geom=obj_clearance,
        w_align=10.0, orient_weight=2.0, edge_margin_m=edge_margin_m,
        ground_clearance_m=0.010,
        wrench_constraint=False, datum_gamma=True,
        accel_budget_xyz=accel_budget,
        ang_accel_budget_xyz=ang_budget,
    )
    return cfg, active_finger_geoms


def latest_logs(explicit=None, k=None):
    if explicit:
        files = sorted(glob.glob(os.path.join(explicit, '**', 'pose_trace.npz'), recursive=True))
        if not files and os.path.isfile(explicit):
            files = [explicit]
    else:
        files = sorted(glob.glob(os.path.join(_ROOT, 'logs', 'dexpilot_*', 'pose_trace.npz')),
                       key=os.path.getmtime, reverse=True)
    if k:
        files = files[:k]
    return files


def _is_contact_tier(g):
    return ('_ds_' in g or g.endswith('_tip'))


def audit_penetration(model, q_full, obj_qpos, obj_gid, active_finger_geoms, act_idx):
    """Exact mj_geomDistance of active-finger geoms vs the object at q_full (23-DOF actuated,
    expanded over a fresh data). Returns worst signed distance (m) separately for:
      - CONSTRAINED links (bs/md/px — the ones the recommender now bounds): this is the
        number that validates the finger constraints. Should not go far negative.
      - CONTACT-tier geoms (ds/tip): expected to touch, NOT constrained on purpose; reported
        only for context.
    Returns (con_geom, con_dist, contact_geom, contact_dist)."""
    d = mj.MjData(model)
    for i, adr in enumerate(act_idx):
        d.qpos[adr] = q_full[i]
    d.qpos[N_ROBOT:N_ROBOT + len(obj_qpos)] = obj_qpos
    mj.mj_forward(model, d)
    ft = np.zeros(6)
    con_g, con_d = None, 1e9
    ct_g, ct_d = None, 1e9
    for g in active_finger_geoms:
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g)
        if gid < 0:
            continue
        lb = (np.linalg.norm(d.geom_xpos[obj_gid] - d.geom_xpos[gid])
              - model.geom_rbound[gid] - model.geom_rbound[obj_gid])
        dist = max(mj.mj_geomDistance(model, d, gid, obj_gid, 1.0, ft), lb)
        if _is_contact_tier(g):
            if dist < ct_d:
                ct_d, ct_g = dist, g
        else:
            if dist < con_d:
                con_d, con_g = dist, g
    return con_g, con_d, ct_g, ct_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20, help='number of poses to sample')
    ap.add_argument('--seeds', type=int, default=3, help='recommender seeds per solve (live=3)')
    ap.add_argument('--log', type=str, default=None, help='specific log dir/file (default: newest)')
    ap.add_argument('--logs', type=int, default=3, help='how many recent log dirs to pool frames from')
    ap.add_argument('--rng', type=int, default=0, help='sampling seed')
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(SCENE_XML)
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    obj_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, OBJ_GEOM)
    thumb_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_th_ds_tip')
    index_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_if_ds_tip')

    cfg, active_finger_geoms = build_live_config(model)
    planner = MultiStartGraspPlanner3D(model, mj.MjData(model), cfg)

    files = latest_logs(args.log, args.logs)
    if not files:
        print("No dexpilot pose_trace.npz logs found.")
        return
    print(f"[data] pooling frames from {len(files)} log(s):")
    for f in files:
        print("        ", os.path.relpath(f, _ROOT))

    # Pool (q_robot, obj_qpos) frames across the selected logs.
    frames = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        q = d['q_robot']
        oq = d['obj_qpos']
        for i in range(len(q)):
            frames.append((q[i].copy(), oq[i].copy()))
    print(f"[data] {len(frames)} total frames; sampling {args.n}")

    rng = np.random.default_rng(args.rng)
    idxs = rng.choice(len(frames), size=min(args.n, len(frames)), replace=False)

    # Sync the planner's scene to the (static) object pose once; refreshed per solve below.
    results = []
    print()
    print(f"{'#':>3} {'solve_ms':>8} {'conv':>5} {'ik_th':>6} {'ik_if':>6} "
          f"{'link_pen':>8} {'tip_pen':>7} {'wf':>3} {'status':>12}  worst_link")
    print('-' * 96)

    for k, fi in enumerate(idxs):
        q_ref, obj_qpos = frames[fi]
        obj_pos = obj_qpos[:3]

        # Sync planner data to this frame's object pose (collision/surface geometry).
        planner._planner.data.qpos[:] = 0.0
        for i, adr in enumerate(act_idx):
            planner._planner.data.qpos[adr] = q_ref[i]
        planner._planner.data.qpos[N_ROBOT:N_ROBOT + len(obj_qpos)] = obj_qpos
        mj.mj_forward(model, planner._planner.data)

        t0 = time.perf_counter()
        try:
            res = planner.solve(q_ref, obj_pos, max_seeds=args.seeds)
        except Exception as e:
            print(f"{k:>3} {'--':>8} {'ERR':>5}  solve raised: {e}")
            results.append(dict(conv=False, err=True))
            continue
        solve_ms = (time.perf_counter() - t0) * 1e3

        conv = res.get('p1') is not None and res.get('p2') is not None
        status = str(res.get('status'))
        if not conv:
            print(f"{k:>3} {solve_ms:>8.0f} {'NO':>5} {'--':>6} {'--':>6} {'--':>7} "
                  f"{'--':>3} {status:>12}")
            results.append(dict(conv=False, solve_ms=solve_ms, status=status))
            continue

        p1 = np.asarray(res['p1'], float)   # thumb contact
        p2 = np.asarray(res['p2'], float)   # index contact
        q_nlp = np.asarray(res['q'], float)

        # IK accuracy: recommender's own tip-site error vs the target it ACTUALLY optimizes,
        # which is contact + r_tip*outward_normal (grasp_planner_3d.py:1261) — NOT the bare
        # contact. The tip SITE sits at the tip-sphere centre, so it legitimately stands
        # r_tip (~23.8mm) proud of the contact; measuring site-vs-bare-contact reports that
        # fixed offset as if it were error (the spurious "24mm plateau"). Measure against the
        # offset target so this is true convergence error (sub-mm when the NLP solves well).
        # outward normals at the two contacts, from the object geometry the planner used
        obj_c = planner._planner.data.geom_xpos[obj_gid].copy()
        obj_R = planner._planner.data.geom_xmat[obj_gid].reshape(3, 3).copy()
        obj_sz = model.geom_size[obj_gid]
        obj_gt = int(model.geom_type[obj_gid])
        n1 = _geom_normal_np(p1, obj_gt, obj_c, obj_R, obj_sz)
        n2 = _geom_normal_np(p2, obj_gt, obj_c, obj_R, obj_sz)
        dchk = mj.MjData(model)
        for i, adr in enumerate(act_idx):
            dchk.qpos[adr] = q_nlp[i]
        dchk.qpos[N_ROBOT:N_ROBOT + len(obj_qpos)] = obj_qpos
        mj.mj_forward(model, dchk)
        r_th = float(planner._planner.cfg.r_thumb)
        r_if = float(planner._planner.cfg.r_index)
        ik_th = float(np.linalg.norm(dchk.site_xpos[thumb_sid] - (p1 + r_th * n1)) * 1e3)
        ik_if = float(np.linalg.norm(dchk.site_xpos[index_sid] - (p2 + r_if * n2)) * 1e3)

        # Collision: exact worst penetration of the RAW recommender q, split into the
        # CONSTRAINED links (validates the finger constraints) and the CONTACT tier (ds/tip,
        # expected to touch, not constrained).
        con_g, con_d, ct_g, ct_d = audit_penetration(
            model, q_nlp, obj_qpos, obj_gid, active_finger_geoms, act_idx)
        pen_mm = con_d * 1e3        # constrained-link worst penetration (the metric that matters)
        ct_mm = ct_d * 1e3          # contact-tier worst (context only)

        # Wrench feasibility (datum gamma certificate).
        try:
            vinfo = planner._planner.verify(res) or {}
            wf = bool(vinfo.get('wrench_feasible', False))
        except Exception:
            wf = False

        flag = ' PEN' if pen_mm < -2.0 else ''
        print(f"{k:>3} {solve_ms:>8.0f} {'YES':>5} {ik_th:>6.1f} {ik_if:>6.1f} "
              f"{pen_mm:>8.1f} {ct_mm:>7.1f} {('Y' if wf else 'n'):>3} {status:>12}  "
              f"{con_g}{flag}")
        results.append(dict(conv=True, solve_ms=solve_ms, ik_th=ik_th, ik_if=ik_if,
                            pen_mm=pen_mm, ct_mm=ct_mm, wf=wf, status=status))

    # ── Aggregate ──────────────────────────────────────────────────────────
    conv = [r for r in results if r.get('conv')]
    print()
    print("=" * 60)
    print(f"SUMMARY over {len(results)} sampled poses ({args.seeds} seeds each)")
    print(f"  converged (returned contacts) : {len(conv)}/{len(results)} "
          f"({100*len(conv)/max(len(results),1):.0f}%)")
    if conv:
        def stats(key):
            v = np.array([r[key] for r in conv], float)
            return v.mean(), np.median(v), v.min(), v.max()
        for label, key, unit in [('solve time', 'solve_ms', 'ms'),
                                 ('IK err thumb', 'ik_th', 'mm'),
                                 ('IK err index', 'ik_if', 'mm'),
                                 ('link penetration', 'pen_mm', 'mm'),
                                 ('tip penetration', 'ct_mm', 'mm')]:
            m, md, lo, hi = stats(key)
            print(f"  {label:<20}: mean={m:7.1f}  median={md:7.1f}  "
                  f"min={lo:7.1f}  max={hi:7.1f}  {unit}")
        n_wf = sum(1 for r in conv if r['wf'])
        n_pen = sum(1 for r in conv if r['pen_mm'] < -2.0)
        n_reach = sum(1 for r in conv if max(r['ik_th'], r['ik_if']) < 15.0)
        print(f"  wrench-feasible (datum γ)   : {n_wf}/{len(conv)}")
        print(f"  IK-reachable (<15mm both)   : {n_reach}/{len(conv)}")
        print(f"  penetrating (<-2mm exact)   : {n_pen}/{len(conv)}")


if __name__ == '__main__':
    main()

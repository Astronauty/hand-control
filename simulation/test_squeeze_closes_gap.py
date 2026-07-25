"""Physics test: when the fingertips start with a PAD GAP off the object (the consequence of
using the MAX pad offset for r_tip — the pad hovers slightly proud rather than penetrating),
does the grasp squeeze CLOSE the gap cleanly, or does it SHOVE the object?

This drives the REAL GraspController (same internal_force_torques + slip_correction + softened
PD + force ramp as the live GRASP loop in kinova_leap_pick_place.py) headlessly. For a sweep of
initial pad gaps, it:
  1. poses the two grasping fingers so their pads sit `gap` mm off two opposing box faces,
  2. enables the squeeze with the live ramp (SQUEEZE_RAMP_S),
  3. steps physics for a fixed settle time,
  4. measures how far the OBJECT moved (translation + rotation) — the "shove".

A small, monotonically-growing object displacement with gap == "squeeze closes it gently".
A large jump at some gap == "the pad gap caused an unbalanced shove" (the failure the ramp
guards against). This directly tests the 'max pad offset is safe because the squeeze removes
the deviation' assumption.

Usage:
  python simulation/test_squeeze_closes_gap.py [--gaps 0,2,4,6,8] [--settle 2.0]
"""
import argparse
import os
import sys

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'grasp_control'))

from grasp_control import GraspController  # noqa: E402
from grasp_control import ConstrainedIKSolver  # noqa: E402
from grasp_planner_3d import MultiStartGraspPlanner3D, _geom_normal_np  # noqa: E402
import test_recommender_on_dexpilot_poses as base  # noqa: E402

SCENE_XML = os.path.join(_ROOT, 'models', 'scene_pick_place.xml')
N_ROBOT = 23
OBJ_BODY = 'obj_red_box'
OBJ_GEOM = 'obj_red_box_geom'
THUMB_SITE = 'leap_th_ds_tip'
INDEX_SITE = 'leap_if_ds_tip'

# Live constants (kinova_leap_pick_place.py).
SQUEEZE_RAMP_S = 1.0
SQUEEZE_PD_SCALE = 2.0
GAMMA = 5.0             # raw gamma scale for the null-space squeeze (pre-safety-factor)
SETTLE_DEFAULT = 2.0


def make_provider(model, obj_bid, contacts_O):
    """Provider(data) -> [(p_W, R_W_inward)] tracking the object body via stored object-local
    contact points + inward normals (object-local), mirroring the live _make_provider."""
    def provider(data):
        p_WoO = data.xpos[obj_bid].copy()
        R_WO = data.xmat[obj_bid].reshape(3, 3).copy()
        out = []
        for p_O, n_in_O in contacts_O:
            p_W = p_WoO + R_WO @ p_O
            n_W = R_WO @ n_in_O
            # Build a frame whose col0 is the inward normal (the convention compute() uses).
            a = np.array([0, 0, 1.0]) if abs(n_W[2]) < 0.9 else np.array([1.0, 0, 0])
            t1 = np.cross(n_W, a); t1 /= np.linalg.norm(t1) + 1e-12
            t2 = np.cross(n_W, t1)
            out.append((p_W, np.column_stack([n_W, t1, t2])))
        return out
    return provider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gaps', type=str, default='0,2,4,6,8',
                    help='comma-separated initial pad gaps in mm')
    ap.add_argument('--settle', type=float, default=SETTLE_DEFAULT,
                    help='sim seconds to step under squeeze')
    args = ap.parse_args()
    gaps_mm = [float(x) for x in args.gaps.split(',')]

    model = mj.MjModel.from_xml_path(SCENE_XML)
    obj_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, OBJ_BODY)
    obj_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, OBJ_GEOM)
    th_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, THUMB_SITE)
    if_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, INDEX_SITE)
    id_C = [if_sid, th_sid]   # FINGER_SET order = [index, thumb]
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]

    # Build the live recommender and grab ONE real reachable grasp pose from a dexpilot frame.
    # This gives a valid two-finger pinch q (arm approached, pads near the box) — the naive
    # jacobian-IK from qpos0 cannot reach the box, so we must seed from a real solve.
    cfg, _ = base.build_live_config(model)
    planner = MultiStartGraspPlanner3D(model, mj.MjData(model), cfg)
    files = base.latest_logs(None, 3)
    d = np.load(files[0], allow_pickle=True)
    q_ref = d['q_robot'][len(d['q_robot']) // 2].copy()
    obj_qpos_frame = d['obj_qpos'][len(d['q_robot']) // 2].copy()
    planner._planner.data.qpos[:] = 0.0
    for i, adr in enumerate(act_idx):
        planner._planner.data.qpos[adr] = q_ref[i]
    planner._planner.data.qpos[N_ROBOT:N_ROBOT + len(obj_qpos_frame)] = obj_qpos_frame
    mj.mj_forward(model, planner._planner.data)
    res = planner.solve(q_ref, obj_qpos_frame[:3], max_seeds=5)
    if res.get('p1') is None:
        print("recommender did not converge on the seed frame; aborting.")
        return
    q_grasp = np.asarray(q_ref, float).copy()
    for i, adr in enumerate(act_idx):
        q_grasp[adr] = res['q'][i]
    p1_rec = np.asarray(res['p1'], float)   # thumb contact (world)
    p2_rec = np.asarray(res['p2'], float)   # index contact (world)

    obj_hx = float(model.geom_size[obj_gid][0])
    obj_qpos0 = obj_qpos_frame.copy()
    print(f"[seed] recommended grasp from dexpilot frame; obj at {np.round(obj_qpos0[:3],3)}")

    # PD gains: order-of-magnitude match to the live Kp/Kd (exact gains are built in main()).
    Kp = np.r_[np.full(7, 300.0), np.full(16, 5.0)]
    Kd = np.r_[np.full(7, 30.0), np.full(16, 0.3)]

    # Object-LOCAL contacts derived from the recommended WORLD contacts, so the provider
    # tracks the box. Inward normal = from contact toward object centre (box geometry).
    R_WO0 = mj.MjData(model)
    R_WO0.qpos[N_ROBOT:N_ROBOT + len(obj_qpos0)] = obj_qpos0
    for i, adr in enumerate(act_idx):
        R_WO0.qpos[adr] = q_grasp[i]
    mj.mj_forward(model, R_WO0)
    p_WoO0 = R_WO0.xpos[obj_bid].copy()
    R_WO0m = R_WO0.xmat[obj_bid].reshape(3, 3).copy()
    obj_c = R_WO0.geom_xpos[obj_gid].copy()
    obj_R = R_WO0.geom_xmat[obj_gid].reshape(3, 3).copy()
    obj_sz = model.geom_size[obj_gid]
    obj_gt = int(model.geom_type[obj_gid])
    contacts_O = []
    for p_rec in (p2_rec, p1_rec):   # id_C order = [index, thumb] = [p2, p1]
        n_out = _geom_normal_np(p_rec, obj_gt, obj_c, obj_R, obj_sz)
        p_O = R_WO0m.T @ (p_rec - p_WoO0)
        n_in_O = R_WO0m.T @ (-n_out)
        contacts_O.append((p_O, n_in_O))
    # Pinch axis (world) — direction between the two recommended contacts.
    pinch_W = (p2_rec - p1_rec); pinch_W /= np.linalg.norm(pinch_W) + 1e-12

    print(f"gamma={GAMMA}  ramp={SQUEEZE_RAMP_S}s  settle={args.settle}s  "
          f"mass={model.body_mass[obj_bid]:.3f}kg")
    print(f"{'gap_mm':>7} {'obj_dxyz_mm':>12} {'obj_drot_deg':>12} {'settled?':>9}  verdict")
    print('-' * 62)

    dt = model.opt.timestep
    n_steps = int(args.settle / dt)

    for gap in gaps_mm:
        data = mj.MjData(model)
        # Robot at the recommended grasp pose; object nudged `gap` mm along +pinch so the pads
        # start `gap` off the faces (the max-pad-offset "hover" the squeeze must close).
        for i, adr in enumerate(act_idx):
            data.qpos[adr] = q_grasp[i]
        data.qpos[N_ROBOT:N_ROBOT + len(obj_qpos0)] = obj_qpos0
        data.qpos[N_ROBOT:N_ROBOT + 3] += (gap * 1e-3) * pinch_W
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)

        obj_p_start = data.xpos[obj_bid].copy()
        obj_R_start = data.xmat[obj_bid].reshape(3, 3).copy()

        # Build the controller with a provider tracking the object.
        gc = GraspController(
            model, N_ROBOT, tip_site_ids=id_C, obj_site_ids=None,
            obj_body_id=obj_bid, kp=Kp, kd=Kd, gamma=GAMMA,
            squeeze_pd_scale=SQUEEZE_PD_SCALE, support_weight=True,
            pad_offsets=[0.0, 0.0],
            obj_contact_provider=make_provider(model, obj_bid, contacts_O))
        q_hold = data.qpos[:N_ROBOT].copy()
        gc.set_target(q_hold)
        gc.set_squeeze(True)

        # Step physics under the squeeze with the live ramp.
        for k in range(n_steps):
            data.qvel[:N_ROBOT] = 0.0
            kp_eff, kd_eff = gc.effective_gains()
            tau = np.zeros(model.nv)
            tau[:N_ROBOT] = (kp_eff * (q_hold - data.qpos[:N_ROBOT])
                             + kd_eff * (0 - data.qvel[:N_ROBOT]))
            ramp = min(1.0, (k + 1) * dt / SQUEEZE_RAMP_S)
            tau[:N_ROBOT] += gc.internal_force_torques(data, scale=ramp)
            tau[:N_ROBOT] += gc.slip_correction_torques(data)
            data.qfrc_applied[:] = tau
            data.qfrc_applied[:N_ROBOT] += data.qfrc_bias[:N_ROBOT]
            mj.mj_step(model, data)

        obj_p_end = data.xpos[obj_bid].copy()
        obj_R_end = data.xmat[obj_bid].reshape(3, 3).copy()
        d_trans = np.linalg.norm(obj_p_end - obj_p_start) * 1e3
        dR = obj_R_start.T @ obj_R_end
        d_rot = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        # Settled? object nearly at rest at the end (not still flying).
        v_obj = float(np.linalg.norm(data.cvel[obj_bid]))
        settled = 'yes' if v_obj < 0.02 else f'no({v_obj:.2f})'

        verdict = ('gentle' if d_trans < 5 else
                   'MODERATE' if d_trans < 20 else '** SHOVE **')
        print(f"{gap:>7.1f} {d_trans:>12.1f} {d_rot:>12.1f} {settled:>9}  {verdict}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Contact-stability probe (test #1 from the teleop-vs-auto slip investigation).

Loads a GRASP-phase frame from a pose_trace.npz, sets the full MuJoCo state to it,
FREEZES the arm+fingers (robot qpos held rigid — qvel pinned to 0 every step, exactly
as the live GRASP loop holds them), leaves the box FREE, and steps physics forward.

Question it answers: with the robot held perfectly still, does the box tilt away on
its OWN (the contact configuration is self-unstable), or does it stay put (the failure
needs the teleop motion to drive it)?

Run the same probe on:
  - a TELE frame just before tilt runs away  -> expect divergence if contact-config unstable
  - an AUTO frame mid-hold (stable in the log) -> expect it to stay put (control case)

If TELE diverges rigid but AUTO holds rigid, the contact CONFIGURATION differs in
stability under the SAME contact model — supporting 'the settings work for auto but
fall apart in tele'. If TELE also holds when rigid, the failure is motion-driven, not
a self-unstable contact.

Usage:
  python simulation/probe_contact_stability.py <pose_trace.npz> [--t SIMTIME] [--hold-s 2.0]
"""
import argparse
import os
import sys

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

SCENE_XML = os.path.join(_ROOT, 'models', 'scene_pick_place.xml')
N_ROBOT = 23   # 7 arm + 16 finger DOF; box freejoint follows in qpos[23:30], qvel[23:29]
BOX_QPOS = slice(23, 30)   # pos(3) + quat(4)
BOX_QVEL = slice(23, 29)   # lin(3) + ang(3)


def quat_ang(q1, q2):
    dot = abs(float(np.clip(np.dot(q1, q2), -1, 1)))
    return np.degrees(2 * np.arccos(dot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace', help='path to a pose_trace.npz')
    ap.add_argument('--t', type=float, default=None,
                    help='sim-time (s) of the GRASP frame to probe; default = first '
                         'GRASP frame after the phase settles ~1s in')
    ap.add_argument('--hold-s', type=float, default=2.0,
                    help='sim-time (s) to hold the frozen robot and watch the box')
    ap.add_argument('--kp', type=float, default=500.0,
                    help='robot position-hold stiffness (rigid freeze); high = stiffer')
    ap.add_argument('--replay-traj', action='store_true',
                    help="Instead of freezing the robot, drive the ARM through the logged "
                         "trajectory forward from the chosen frame (reproducing the real "
                         "disturbance), while holding the FINGERS at --finger-kp. Use with "
                         "--finger-kp to A/B soft vs rigid fingers on the SAME arm motion.")
    ap.add_argument('--finger-kp', type=float, default=None,
                    help="Finger-joint position-hold stiffness for --replay-traj. Default "
                         "None = use a soft hold (~live squeezed compliance). A high value "
                         "(e.g. 500) rigidly pins the fingers at the grasp config. The A/B "
                         "that tests whether finger compliance is the failure mechanism.")
    args = ap.parse_args()

    z = np.load(args.trace, allow_pickle=True)
    t = z['t']
    ph = z['control_phase'] if 'control_phase' in z.files else None
    qpos_all = z['qpos']

    # Choose the frame: nearest GRASP frame to --t, or ~1s into GRASP if unset.
    if ph is not None:
        g = np.where(ph == 'GRASP')[0]
        if len(g) == 0:
            sys.exit('No GRASP rows in this trace.')
    else:
        g = np.arange(len(t))
    if args.t is not None:
        i = g[int(np.argmin(np.abs(t[g] - args.t)))]
    else:
        i = g[min(30, len(g) - 1)]

    model = mj.MjModel.from_xml_path(SCENE_XML)
    data = mj.MjData(model)
    assert model.nq == qpos_all.shape[1], (
        f'trace qpos width {qpos_all.shape[1]} != model.nq {model.nq}')

    # Set full state to the logged frame; velocities zero (clean quasi-static start,
    # as the live GRASP hold pins qvel each step anyway).
    q_frozen = qpos_all[i].copy()
    data.qpos[:] = q_frozen
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)

    box_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'obj_red_box')
    q0_box = data.qpos[BOX_QPOS][3:7].copy()   # box orientation datum
    p0_box = data.qpos[BOX_QPOS][:3].copy()
    ncon0 = int(data.ncon)

    dt = model.opt.timestep

    if args.replay_traj:
        # Drive the ARM through the logged trajectory forward from frame i (reproducing the
        # real disturbance), while holding the FINGERS at --finger-kp. A/B soft vs rigid
        # fingers on the SAME arm motion to test whether finger compliance is the cause.
        arm_kp = args.kp
        arm_kd = 2.0 * np.sqrt(arm_kp)
        # Soft finger default ~ the live squeezed compliance (Kp[7:]=1.2 * SQUEEZE_PD_SCALE=5
        # = 6.0); a high value rigidly pins the fingers at the grasp config.
        fkp = args.finger_kp if args.finger_kp is not None else 6.0
        fkd = 2.0 * np.sqrt(fkp)
        # Sim-time span available in the trajectory from frame i.
        traj = qpos_all[i:]
        traj_t = t[i:] - t[i]
        span = float(traj_t[-1])
        n = int(min(args.hold_s, span) / dt)
        finger_hold = q_frozen[7:23].copy()   # fingers pinned at the grasp config
        print(f"[probe] {os.path.basename(os.path.dirname(args.trace))}  frame idx={i} "
              f"t={t[i]:.3f}s  phase={ph[i] if ph is not None else '?'}")
        print(f"[probe] REPLAY-TRAJ: arm follows logged trajectory (kp={arm_kp}); "
              f"fingers held at kp={fkp:.1f} "
              f"({'RIGID' if fkp >= 100 else 'soft'}); box free; {n*dt:.2f}s\n")
        print(" t     box_tilt  box_dz(mm)  box_xy(mm)  ncon  arm_track_err")
        for k in range(n):
            th = k * dt
            # Interpolate the logged ARM config at this sim-time.
            arm_tgt = np.array([np.interp(th, traj_t, traj[:, j]) for j in range(7)])
            data.qvel[:7] = 0.0            # quasi-static arm (as live loop pins it)
            tau = np.zeros(model.nv)
            tau[:7] = arm_kp * (arm_tgt - data.qpos[:7]) - arm_kd * data.qvel[:7]
            tau[7:N_ROBOT] = (fkp * (finger_hold - data.qpos[7:N_ROBOT])
                              - fkd * data.qvel[7:N_ROBOT])
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[:N_ROBOT] = tau[:N_ROBOT] + data.qfrc_bias[:N_ROBOT]
            mj.mj_step(model, data)
            if k % max(1, n // 20) == 0 or k == n - 1:
                tilt = quat_ang(q0_box, data.qpos[BOX_QPOS][3:7])
                dz = (data.qpos[BOX_QPOS][2] - p0_box[2]) * 1e3
                xy = np.linalg.norm(data.qpos[BOX_QPOS][:2] - p0_box[:2]) * 1e3
                aerr = np.linalg.norm(arm_tgt - data.qpos[:7])
                print(f"{th:6.2f} {tilt:9.1f} {dz:11.1f} {xy:11.1f}  {int(data.ncon):4d}  {aerr:12.4f}")
        tilt_f = quat_ang(q0_box, data.qpos[BOX_QPOS][3:7])
        ncon_f = int(data.ncon)
        ejected = tilt_f > 20 or ncon_f == 0
        print(f"\n[verdict] fingers kp={fkp:.1f} -> final tilt={tilt_f:.1f}deg  ncon={ncon_f}"
              f"  ->  {'EJECTED/tilted' if ejected else 'HELD'}")
        return

    n = int(args.hold_s / dt)
    print(f"[probe] {os.path.basename(os.path.dirname(args.trace))}  frame idx={i} "
          f"t={t[i]:.3f}s  phase={ph[i] if ph is not None else '?'}")
    print(f"[probe] start: box_z={p0_box[2]:.4f}  ncon={ncon0}  "
          f"holding robot RIGID (kp={args.kp}) for {args.hold_s:.1f}s, box free\n")
    print(" t_hold  box_tilt  box_dz(mm)  box_xy_drift(mm)  ncon")

    kd = 2.0 * np.sqrt(args.kp)   # critically-damped hold on the frozen robot
    for k in range(n):
        # RIGID robot hold: PD toward the frozen config + pin robot velocity to 0 each
        # step (exactly the live GRASP quasi-static hold). The box DOFs are untouched —
        # they evolve purely under contact + gravity.
        data.qvel[:N_ROBOT] = 0.0
        tau = np.zeros(model.nv)
        tau[:N_ROBOT] = (args.kp * (q_frozen[:N_ROBOT] - data.qpos[:N_ROBOT])
                         - kd * data.qvel[:N_ROBOT])
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[:N_ROBOT] = tau[:N_ROBOT] + data.qfrc_bias[:N_ROBOT]
        mj.mj_step(model, data)

        if k % max(1, n // 20) == 0 or k == n - 1:
            tilt = quat_ang(q0_box, data.qpos[BOX_QPOS][3:7])
            dz = (data.qpos[BOX_QPOS][2] - p0_box[2]) * 1e3
            xy = np.linalg.norm(data.qpos[BOX_QPOS][:2] - p0_box[:2]) * 1e3
            print(f"{k*dt:7.2f} {tilt:9.1f} {dz:11.1f} {xy:16.1f}  {int(data.ncon):4d}")

    tilt_f = quat_ang(q0_box, data.qpos[BOX_QPOS][3:7])
    ncon_f = int(data.ncon)
    verdict = ("CONTACT SELF-UNSTABLE (box tilted away with robot frozen)"
               if (tilt_f > 20 or ncon_f == 0)
               else "CONTACT STABLE when robot held rigid (failure is motion-driven)")
    print(f"\n[verdict] final tilt={tilt_f:.1f}deg  ncon={ncon_f}  ->  {verdict}")


if __name__ == '__main__':
    main()

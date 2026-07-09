#!/usr/bin/env python3
# All variable naming notation follows https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html
"""Minimal demo of grasp_control.GraspController: solve the grasp IK for one object
in models/scene_pick_place.xml, replay to the grasp config kinematically, then let
Enter toggle a pure internal (pinching) force f_c = null(G) @ gamma (w_des = 0).

No RRT, no dashboard, no object randomization — the point is to isolate and verify
the internal-force grasp controller. Run from the repo root:

    python grasp_controller_demo.py --obj 0            # red box
    python grasp_controller_demo.py --obj 3 --gamma 10 # blue capsule, harder squeeze

Keys:  Enter — HOLD -> SQUEEZE -> RELEASED -> HOLD ...   |   Q/Esc — quit
"""
import argparse
import queue
import time

import numpy as np
import mujoco as mj
import mujoco.viewer  # noqa: F401

from grasp_control import (SpatialIKSolver, ConstrainedIKSolver, GraspController)
from grasp_control.constrained_ik import configure_sqp


FINGER_TIP_SITES = {
    "index":  "leap_if_ds_tip",
    "middle": "leap_mf_ds_tip",
    "ring":   "leap_rf_ds_tip",
    "thumb":  "leap_th_ds_tip",
}
FINGER_CODE = {"index": "if", "middle": "mf", "ring": "rf", "thumb": "th"}
FINGER_SET = ["index", "thumb"]

GEN3_XML = 'mujoco_menagerie/kinova_gen3/gen3.xml'

# finger -> contact-site mapping per object, same table as kinova_leap_pick_place.py.
OBJECT_DEFS = [
    ({'index': 'obj_red_box_c2',        'thumb': 'obj_red_box_c1'},        'obj_red_box'),
    ({'index': 'obj_red_sphere_c2',     'thumb': 'obj_red_sphere_c1'},     'obj_red_sphere'),
    ({'index': 'obj_blue_cylinder_c2',  'thumb': 'obj_blue_cylinder_c1'},  'obj_blue_cylinder'),
    ({'index': 'obj_blue_capsule_c2',   'thumb': 'obj_blue_capsule_c1'},   'obj_blue_capsule'),
    ({'index': 'obj_green_box_c2',      'thumb': 'obj_green_box_c1'},      'obj_green_box'),
    ({'index': 'obj_green_cylinder_c2', 'thumb': 'obj_green_cylinder_c1'}, 'obj_green_cylinder'),
]

N_ROBOT = 23
APPROACH_STEPS = 300   # kinematic replay resolution Q_BIAS -> q_target
PRINT_EVERY = 250      # SQUEEZE verification printout period (sim steps)


def make_key_callback(key_queue):
    """GLFW key callback for the MuJoCo passive viewer: Enter / Q / Esc only."""
    _MAP = {
        257: 'enter',   # GLFW_KEY_ENTER (main keyboard)
        335: 'enter',   # GLFW_KEY_KP_ENTER (numpad)
        81:  'quit',    # Q
        256: 'quit',    # Escape
    }
    def _cb(keycode):
        event = _MAP.get(keycode)
        if event:
            key_queue.put(event)
    return _cb


def _build_q_bias(model):
    """Arm home pose from gen3.xml's "home" keyframe + middle/ring fingers curled
    out of the way (same rationale as kinova_leap_pick_place.py: the null-space
    pull toward this pose gives a forward approach and keeps unused fingers off
    the floor)."""
    home_arm = mj.MjModel.from_xml_path(GEN3_XML).key('home').qpos[:7].copy()
    q_bias = np.zeros(N_ROBOT)
    q_bias[:7]    = home_arm
    q_bias[11:15] = [1.2, 0.0, 0.5, 0.5]  # leap_mf_*: curl out of the way
    q_bias[15:19] = [1.2, 0.0, 0.5, 0.5]  # leap_rf_*: curl out of the way
    return q_bias


def _build_constrained_ik(model):
    """ConstrainedIKSolver over every collision geom on the four fingers + palm +
    wrist vs all objects + floor, with the tiered active-finger clearances from
    kinova_leap_pick_place.py (contact tier disabled, adjacent -10mm, proximal
    +2mm; floor always fully constrained)."""
    active_body_prefixes = tuple(f'leap_{code}_' for code in FINGER_CODE.values())
    robot_geom_names = []
    for gi in range(model.ngeom):
        gname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gi)
        if not gname or model.geom_contype[gi] == 0:
            continue   # skip unnamed / visual-only geoms
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ''
        if (any(bname.startswith(p) for p in active_body_prefixes)
                or bname in ('leap_palm', 'bracelet_link')):
            robot_geom_names.append(gname)

    obj_geom_names = [body + '_geom' for _, body in OBJECT_DEFS] + ['floor']
    solver = ConstrainedIKSolver(
        model, N_ROBOT,
        arm_geom_names=robot_geom_names,
        obj_geom_names=obj_geom_names,
        clearance=0.005,
        posture_weight=0.0005,
        pad_axis=(-1.0, 0.0, 0.0),  # LEAP fingerpad normal in the fingertip-site frame
        orient_weight=0.01,         # align each fingerpad with the contact inward normal
        max_iter=500,
    )
    configure_sqp(solver)

    active_finger_geoms = {g for g in robot_geom_names
                           if any(g.startswith(f'leap_{FINGER_CODE[f]}_') for f in FINGER_SET)}

    def _active_obj_clearance(g):
        if '_ds_' in g or g.endswith('_tip'):
            return -1.0        # contact tier: must reach the surface (disabled)
        if g.startswith(('leap_if_md', 'leap_th_px')):
            return -0.010      # adjacent tier: bounded sphere slack
        return 0.002           # proximal tier: stay off the surface

    active_clearance_by_geom = {g: _active_obj_clearance(g) for g in active_finger_geoms}
    print(f"[IK] {len(robot_geom_names)} robot geoms x {len(obj_geom_names)} objects, "
          f"{len(active_finger_geoms)} active-finger geoms with tiered clearance")
    return solver, active_clearance_by_geom


def _solve_grasp_ik(model, obj, q_bias, constrained_ik, active_clearance_by_geom,
                    tip_site_ids, dls_only=False):
    """Two-stage grasp IK: DLS warm start -> collision-aware SQP refinement.
    Fingertips driven onto the object's contact sites."""
    ik_data = mj.MjData(model)
    ik_data.qpos[:N_ROBOT] = q_bias
    mj.mj_forward(model, ik_data)

    t0 = time.time()
    dls_ik = SpatialIKSolver(n_robot=N_ROBOT)
    q_dls = dls_ik.solve(model, ik_data, tip_site_ids, obj['p_S_W'],
                         q_bias=q_bias, null_gain=0.3)
    dls_ms = (time.time() - t0) * 1e3
    if dls_only:
        print(f"[IK] DLS only: {dls_ms:.0f}ms (WARNING: penetration unbounded — "
              f"physics handoff may pop)")
        q_target = q_dls
    else:
        t0 = time.time()
        q_target = constrained_ik.solve(ik_data, tip_site_ids, obj['p_S_W'],
                                        q_bias=q_bias, q_init=q_dls,
                                        reduced_clearance_geoms=active_clearance_by_geom,
                                        inward_dirs=obj['inward_S_W'])
        print(f"[IK] DLS {dls_ms:.0f}ms + SQP {(time.time() - t0) * 1e3:.0f}ms")

    # Tip-error audit at the solution.
    chk = mj.MjData(model)
    chk.qpos[:N_ROBOT] = q_target
    mj.mj_forward(model, chk)
    errs = [f"{np.linalg.norm(chk.site_xpos[s] - t) * 1e3:.1f} mm"
            for s, t in zip(tip_site_ids, obj['p_S_W'])]
    print(f"[IK] tip errors = {errs}")
    return q_target


def _measured_tip_forces(model, data, tip_geom_ids, obj_geom_id):
    """Sum of contact-normal force magnitudes between each fingertip geom and the
    object geom, one entry per finger (order matches tip_geom_ids)."""
    totals = [0.0] * len(tip_geom_ids)
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for k, tg in enumerate(tip_geom_ids):
            if {c.geom1, c.geom2} == {tg, obj_geom_id}:
                mj.mj_contactForce(model, data, i, f6)
                totals[k] += abs(f6[0])   # contact-frame normal component
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GraspController demo: IK grasp + Enter-key internal force.")
    parser.add_argument('--obj', type=int, default=0, choices=range(len(OBJECT_DEFS)),
                        help="object index: " + ", ".join(
                            f"{i}={body}" for i, (_, body) in enumerate(OBJECT_DEFS)))
    parser.add_argument('--gamma', type=float, default=5.0,
                        help="internal squeeze force scale (null-space weight); "
                             "per-contact force ~ gamma/sqrt(2) for a 2-contact pinch")
    parser.add_argument('--dls-only', action='store_true',
                        help="skip the collision-aware SQP refinement (debug only)")
    parser.add_argument('--arm-kp', type=float, default=40.0,
                        help="arm joint PD stiffness, Nm/rad (fingers stay at 0.8)")
    parser.add_argument('--arm-kd', type=float, default=4.0,
                        help="arm joint PD damping, Nm·s/rad — inert during the "
                             "quasi-static hold (qvel is zeroed each step) but keep "
                             "within the explicit-damping limit dt < 2I/Kd at the wrist")
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path('models/scene_pick_place.xml')
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    contact_sites, body_name = OBJECT_DEFS[args.obj]
    tip_site_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
                    for f in FINGER_SET]
    tip_geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f'leap_{FINGER_CODE[f]}_tip')
                    for f in FINGER_SET]
    obj = {
        'id_S':    [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, contact_sites[f])
                    for f in FINGER_SET],
        'id_body': mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name),
        'id_geom': mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, body_name + '_geom'),
    }
    obj['p_S_W'] = [data.site_xpos[sid].copy() for sid in obj['id_S']]
    # Inward surface normal at each contact site (site x-axis, scene XML convention) —
    # used by the constrained IK's fingerpad orientation cost.
    obj['inward_S_W'] = [data.site_xmat[sid].reshape(3, 3)[:, 0].copy()
                         for sid in obj['id_S']]

    Q_BIAS = _build_q_bias(model)
    constrained_ik, active_clearance_by_geom = (None, None)
    if not args.dls_only:
        constrained_ik, active_clearance_by_geom = _build_constrained_ik(model)

    print(f"[Demo] solving grasp IK for {body_name} ...")
    q_target = _solve_grasp_ik(model, obj, Q_BIAS, constrained_ik,
                               active_clearance_by_geom, tip_site_ids,
                               dls_only=args.dls_only)

    # PD gains: arm sized for Gen3 forcerange (stiffness selectable via --arm-kp),
    # fingers small like the LEAP actuators.
    Kp = np.concatenate([np.full(7, args.arm_kp), np.full(16, 0.8)])
    Kd = np.concatenate([np.full(7, args.arm_kd), np.full(16, 0.05)])
    ctrl = GraspController(model, N_ROBOT,
                           tip_site_ids=tip_site_ids,
                           obj_site_ids=obj['id_S'],
                           obj_body_id=obj['id_body'],
                           kp=Kp, kd=Kd, gamma=args.gamma)

    # Release config: arm stays at the grasp arm config, all fingers back to bias.
    q_release = q_target.copy()
    q_release[7:] = Q_BIAS[7:]

    # Start at Q_BIAS so the first PD error is zero (all-zero qpos would point the
    # arm straight up and explode against the elbow-bent target).
    mj.mj_resetData(model, data)
    data.qpos[:N_ROBOT] = Q_BIAS
    mj.mj_forward(model, data)

    approach_path = np.linspace(Q_BIAS, q_target, APPROACH_STEPS)
    phase = 'APPROACH'
    approach_i = 0
    step_i = 0
    p_obj_at_squeeze = None

    keys = queue.Queue()
    print(f"[Demo] gamma={args.gamma:.1f} -> expected per-contact force "
          f"~{args.gamma / np.sqrt(2):.2f} N")
    print("[Demo] Enter: HOLD -> SQUEEZE -> RELEASED -> HOLD ...  |  Q/Esc: quit")

    with mj.viewer.launch_passive(model, data, key_callback=make_key_callback(keys)) as viewer:
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        running = True
        while viewer.is_running() and running:
            step_start = time.time()

            while not keys.empty():
                key = keys.get_nowait()
                if key == 'quit':
                    running = False
                elif key == 'enter' and phase == 'HOLD':
                    phase = 'SQUEEZE'
                    ctrl.set_squeeze(True)
                    p_obj_at_squeeze = data.xpos[obj['id_body']].copy()
                    viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                    print(f"\r\n[Demo] -> SQUEEZE (internal force on, gamma={args.gamma:.1f})")
                elif key == 'enter' and phase == 'SQUEEZE':
                    phase = 'RELEASED'
                    ctrl.set_squeeze(False)
                    ctrl.set_target(q_release)
                    viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                    print("\r\n[Demo] -> RELEASED (fingers opening to bias posture)")
                elif key == 'enter' and phase == 'RELEASED':
                    phase = 'HOLD'
                    ctrl.set_target(q_target)
                    print("\r\n[Demo] -> HOLD (back at grasp config; Enter to squeeze)")

            if phase == 'APPROACH':
                # Kinematic replay: qpos overwrite + mj_forward, no mj_step —
                # sidesteps arm->finger inertial coupling QACC explosions.
                data.qpos[:N_ROBOT] = approach_path[approach_i]
                data.qvel[:N_ROBOT] = 0.0
                mj.mj_forward(model, data)
                approach_i += 1
                if approach_i >= APPROACH_STEPS:
                    phase = 'HOLD'
                    ctrl.set_target(q_target)
                    print("\r\n[Demo] -> HOLD (at grasp config; press Enter to squeeze)")
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
                continue

            # Physics phases (HOLD / SQUEEZE / RELEASED): quasi-static hold —
            # zero qvel each step (explicit qfrc_applied damping has dt < 2I/Kd
            # marginal at the wrist; without this the hold diverges -> BADQACC).
            data.qvel[:N_ROBOT] = 0.0
            data.qfrc_applied[:] = ctrl.compute(data)
            mj.mj_step(model, data)

            step_i += 1
            if phase == 'SQUEEZE' and step_i % PRINT_EVERY == 0:
                f_cmd = [np.linalg.norm(ctrl.last_f_c[3*k:3*k+3])
                         for k in range(len(FINGER_SET))]
                f_meas = _measured_tip_forces(model, data, tip_geom_ids, obj['id_geom'])
                adr = model.body_dofadr[obj['id_body']]
                v_obj = data.qvel[adr:adr+3]
                drift = np.linalg.norm(data.xpos[obj['id_body']] - p_obj_at_squeeze)
                print(f"[squeeze] cmd |f_c| = "
                      + ", ".join(f"{f}:{v:.2f}N" for f, v in zip(FINGER_SET, f_cmd))
                      + "  |  meas normal = "
                      + ", ".join(f"{f}:{v:.2f}N" for f, v in zip(FINGER_SET, f_meas))
                      + f"  |  obj |v|={np.linalg.norm(v_obj)*1e3:.2f}mm/s"
                      + f" drift={drift*1e3:.2f}mm")

            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

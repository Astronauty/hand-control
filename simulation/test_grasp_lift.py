"""Integration test: recommend -> approach -> squeeze -> lift.

Drives the FULL grasp pipeline headless from the sim start pose:
  1. Recommend a grasp for the red box (MultiStartGraspPlanner3D, live config).
  2. Commit the recommender's own q as the grasp pose (single-solve, matching the live
     _commit_recommended_pose path) and settle the arm there via PD.
  3. Solve gamma at the grasp (solve_gamma_live, same datum LP the controller uses).
  4. Enable the REAL GraspController squeeze (internal_force_torques + slip_correction +
     softened PD + force ramp), settle.
  5. Jog the palm +z (the live wrist-jog: DLS from a 6D palm velocity into q_grasp_hold[:7]),
     lifting the box.
  6. Report whether the box lifted and stayed held (didn't slip out).

Sweeps the three stability levers the operator flagged:
  --pd-scale  : SQUEEZE_PD_SCALE (finger PD softening while squeezing)
  --gamma-mult: GAMMA_SAFETY_FACTOR (squeeze force scale)
  --lift      : jog distance in +z (m)

Usage:
  python simulation/test_grasp_lift.py [--pd-scale 2.0] [--gamma-mult 5.0] [--lift 0.08]
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

from grasp_control import GraspController, ConstrainedIKSolver  # noqa: E402
from grasp_planner_3d import MultiStartGraspPlanner3D, _geom_normal_np  # noqa: E402
from scripts.rrt_planner import RRTPlanner  # noqa: E402
import test_recommender_on_dexpilot_poses as base  # noqa: E402

OBJ_BODIES = ['obj_red_box', 'obj_red_sphere', 'obj_blue_cylinder',
              'obj_blue_capsule', 'obj_green_box', 'obj_green_cylinder']

import importlib.util as _ilu

SCENE_XML = base.SCENE_XML
N_ROBOT = 23
N_FINGERS = 2

# Live constants (kinova_leap_pick_place.py).
Kp = np.concatenate([np.full(7, 40.0), np.full(16, 1.2)])
Kd = np.concatenate([np.full(7, 4.0),  np.full(16, 0.075)])
SQUEEZE_RAMP_S = 1.0
GAMMA_FALLBACK = 250.0
NCF_ACCEL_BUDGET_XYZ = (20.0, 20.0, 20.0)
NCF_ANG_ACCEL_BUDGET = (1.0, 1.0, 1.0)
JOG_VEL = 0.2
JOG_SING_EPS = 0.02
JOG_LAM_MAX = 0.05

OBJ_BODY = 'obj_red_box'
OBJ_GEOM = 'obj_red_box_geom'


def load_gamma_solver():
    """Load solve_gamma_live from the main script without running main() (it's guarded by
    __name__ == '__main__'). NCF_DATUM_MODE is set inside main(), so we default it to True
    (the live value) rather than reading it from module scope."""
    src_path = os.path.join(_ROOT, 'kinova_leap_pick_place.py')
    spec = _ilu.spec_from_file_location('klpp_mod', src_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve_gamma_live, True


def pad_offset(model, tip, site):
    d0 = mj.MjData(model); mj.mj_forward(model, d0)
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, tip)
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site)
    mid = model.geom_dataid[gid]
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    vW = (d0.geom_xmat[gid].reshape(3, 3) @ model.mesh_vert[adr:adr + num].T).T + d0.geom_xpos[gid]
    pad_dir = -d0.site_xmat[sid].reshape(3, 3)[:, 0]
    return float(np.max((vW - d0.site_xpos[sid]) @ pad_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pd-scale', type=float, default=5.0)   # live SQUEEZE_PD_SCALE
    ap.add_argument('--gamma-mult', type=float, default=5.0)
    ap.add_argument('--accel', type=float, default=20.0,
                    help='linear accel budget per axis (m/s^2), used for BOTH the recommender '
                         'config and the gamma solve (kept consistent, matching live)')
    ap.add_argument('--edge', type=float, default=0.015, help='edge_margin_m (m)')
    ap.add_argument('--lift', type=float, default=0.08, help='+z jog distance (m)')
    ap.add_argument('--jog-vel', type=float, default=0.2,
                    help='lift speed (m/s); lower = slower lift (live "pick up" is gradual, '
                         'which gives the capped slip-correction time to drift)')
    ap.add_argument('--slip-kp', type=float, default=200.0,
                    help='slip_correction spring stiffness (live default 200)')
    ap.add_argument('--slip-fmax', type=float, default=10.0,
                    help='slip_correction per-finger force cap N (live default 10)')
    ap.add_argument('--settle', type=float, default=1.5, help='settle time per phase (s)')
    ap.add_argument('--hold', type=float, default=2.0, help='hold time at top after lift (s)')
    args = ap.parse_args()
    accel_budget = (args.accel, args.accel, args.accel)

    model = mj.MjModel.from_xml_path(SCENE_XML)
    dt = model.opt.timestep
    obj_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, OBJ_BODY)
    obj_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, OBJ_GEOM)
    th_s = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_th_ds_tip')
    if_s = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'leap_if_ds_tip')
    id_C = [if_s, th_s]                       # FINGER_SET order [index, thumb]
    palm_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'leap_palm')
    act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]

    solve_gamma_live, NCF_DATUM_MODE = load_gamma_solver()

    _PAD = {'index': pad_offset(model, 'leap_if_tip', 'leap_if_ds_tip'),
            'thumb': pad_offset(model, 'leap_th_tip', 'leap_th_ds_tip')}

    # ── 1. Recommend at the SIM START pose ──────────────────────────────────────
    cfg, _ = base.build_live_config(model, accel_budget=accel_budget,
                                    edge_margin_m=args.edge)
    planner = MultiStartGraspPlanner3D(model, mj.MjData(model), cfg)
    # sim start: home robot (qpos0) + object at its spawn pose.
    d0 = mj.MjData(model)
    mj.mj_forward(model, d0)
    obj_qpos0 = d0.qpos[N_ROBOT:].copy()
    q_ref = d0.qpos[:N_ROBOT].copy()          # actuated-order == qpos order for first 23
    q_ref_act = np.array([d0.qpos[a] for a in act_idx])
    planner._planner.data.qpos[:] = d0.qpos[:]
    mj.mj_forward(model, planner._planner.data)
    obj_pos = d0.xpos[obj_bid].copy()
    res = planner.solve(q_ref_act, obj_pos, max_seeds=5)
    if res.get('p1') is None:
        print("recommender did not converge at sim start — abort"); return
    p1_rec = np.asarray(res['p1'], float)     # thumb
    p2_rec = np.asarray(res['p2'], float)     # index
    print(f"[1] recommendation: thumb={np.round(p1_rec,3)} index={np.round(p2_rec,3)} "
          f"cost={res.get('cost'):.3f}")

    # Committed grasp pose = recommender's own q (single-solve path).
    q_grasp = q_ref.copy()
    for i, a in enumerate(act_idx):
        q_grasp[a] = res['q'][i]

    # Object-local contacts + inward frames for the provider (col0 = inward normal).
    obj_c = planner._planner.data.geom_xpos[obj_gid].copy()
    obj_R = planner._planner.data.geom_xmat[obj_gid].reshape(3, 3).copy()
    obj_sz = model.geom_size[obj_gid]; obj_gt = int(model.geom_type[obj_gid])
    p_WoO = d0.xpos[obj_bid].copy(); R_WO = d0.xmat[obj_bid].reshape(3, 3).copy()
    rec_local = []
    for p_rec in (p2_rec, p1_rec):            # id_C order [index, thumb]
        n_out = _geom_normal_np(p_rec, obj_gt, obj_c, obj_R, obj_sz)
        n_in = -n_out
        a = np.array([0, 0, 1.0]) if abs(n_in[2]) < 0.9 else np.array([1.0, 0, 0])
        t1 = np.cross(n_in, a); t1 /= np.linalg.norm(t1) + 1e-12
        t2 = np.cross(n_in, t1)
        R_W = np.column_stack([n_in, t1, t2])
        rec_local.append((R_WO.T @ (p_rec - p_WoO), R_WO.T @ R_W))

    # ── 2. Approach: plan the REAL RRT to q_grasp and replay it (matches live) ───
    # The live pipeline reaches the grasp via a collision-free RRT path — NOT a straight-line
    # q-interpolation, which drives the arm through the table/box and stalls on contact. Build
    # the same RRTPlanner the main script uses, plan to q_grasp with the active fingers given
    # 0mm clearance vs the target box (touch allowed, matching _run_rrt), and replay the path.
    robot_geoms = base.build_robot_geom_names(model)
    active_finger_geoms = {g for g in robot_geoms
                           if any(g.startswith(f'leap_{c}_') for c in ('if', 'th'))}
    rrt = RRTPlanner(model, robot_geoms, OBJ_BODIES, extra_obj_geom_names=['floor'],
                     n_robot=N_ROBOT, n_plan=7, clearance=0.005)
    rrt._data.qpos[N_ROBOT:] = obj_qpos0.copy()   # planner sees the live object pose
    q_start = d0.qpos[:N_ROBOT].copy()
    active_skip = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g) for g in active_finger_geoms]
    pair_clearance = {(g, obj_gid): 0.0 for g in active_skip}
    q_goal = rrt.rebranch(q_start, q_grasp)
    path = rrt.plan(q_start, q_goal, pair_clearance=pair_clearance)
    if path is None:
        n_interp = 120
        path = [q_start + t * (q_goal - q_start) for t in np.linspace(0, 1, n_interp)]
        print(f"[2a] RRT failed -> linear fallback ({n_interp} steps)")
    else:
        print(f"[2a] RRT path: {len(path)} waypoints")

    data = mj.MjData(model)
    data.qpos[:] = d0.qpos[:]
    mj.mj_forward(model, data)
    # Replay KINEMATICALLY (set qpos per waypoint), matching the live REACH phase — the live
    # code deliberately does NOT PD-track intermediate waypoints (arm->finger inertial
    # coupling on large arm moves causes QACC explosions); it sets qpos directly, then hands
    # only the FINAL pose to physics PD.
    for q_wp in path:
        data.qpos[:N_ROBOT] = q_wp
        data.qvel[:N_ROBOT] = 0.0
        mj.mj_forward(model, data)
    # Final waypoint: hand off to physics. Do NOT settle unsqueezed first — the pads start
    # ~mm off the faces, and an unsqueezed settle lets light contact + gravity nudge the box
    # several mm out from between the fingers before any grip forms (measured: box drifts
    # ~9mm, index pad gap 0.4mm -> 9.2mm, and the softened finger PD can't recover). The live
    # code squeezes from the grasp handoff (squeeze_on at REACH->GRASP) with the ramp easing
    # force in. So we go straight to the squeeze phase below.
    q_hold = q_grasp.copy()
    data.qpos[:N_ROBOT] = q_grasp
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)
    _q_err = np.linalg.norm(data.qpos[:7] - q_grasp[:7])
    print(f"[2b] handed off at grasp (arm err {_q_err:.4f} rad); squeezing immediately")
    box_z_pregrasp = data.xpos[obj_bid][2]
    _ft = np.zeros(6)
    pad_gap = [mj.mj_geomDistance(model, data, mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g),
                                  obj_gid, 0.1, _ft) * 1e3 for g in ('leap_if_tip', 'leap_th_tip')]
    print(f"[2] approached; box_z={box_z_pregrasp:.4f}  pad-gap(if,th)={np.round(pad_gap,1)}mm")

    # ── 3. Solve gamma at the grasp ─────────────────────────────────────────────
    R_WO_l = data.xmat[obj_bid].reshape(3, 3)
    _p_O = [p_O.copy() for (p_O, _R) in rec_local]
    _R_in = [R_O.copy() for (_p, R_O) in rec_local]
    _mu = [float(model.geom_friction[obj_gid, 0])] * N_FINGERS
    _mass = float(model.body_mass[obj_bid])
    _g_O = R_WO_l.T @ model.opt.gravity
    _accel_box = tuple(accel_budget) if NCF_DATUM_MODE else \
        tuple(accel_budget[i] + abs(_g_O[i]) for i in range(3))
    _inertia = model.body_inertia[obj_bid]
    _gamma = solve_gamma_live(_p_O, _R_in, _mu, _mass, _accel_box,
                              tuple(np.abs(NCF_ANG_ACCEL_BUDGET)), _inertia,
                              grav_O=(_g_O if NCF_DATUM_MODE else None))
    if _gamma is None or not np.isfinite(_gamma) or _gamma <= 0:
        gamma_raw, gamma_live = GAMMA_FALLBACK, GAMMA_FALLBACK
        print(f"[3] gamma LP infeasible -> fallback {GAMMA_FALLBACK}")
    else:
        gamma_raw = float(_gamma); gamma_live = gamma_raw * args.gamma_mult
        print(f"[3] gamma raw={gamma_raw:.2f} x{args.gamma_mult} = {gamma_live:.2f} "
              f"(~{gamma_live/np.sqrt(2):.2f} N/contact, mass={_mass:.3f}kg mu={_mu[0]:.1f})")

    # ── 4. Squeeze (real GraspController + ramp) ─────────────────────────────────
    def provider(d):
        pW = d.xpos[obj_bid]; RW = d.xmat[obj_bid].reshape(3, 3)
        return [(pW + RW @ p_O, RW @ R_O) for (p_O, R_O) in rec_local]
    gc = GraspController(model, N_ROBOT, tip_site_ids=id_C, obj_site_ids=None,
                         obj_body_id=obj_bid, kp=Kp, kd=Kd, gamma=gamma_live,
                         squeeze_pd_scale=args.pd_scale, support_weight=True,
                         pad_offsets=[_PAD['index'], _PAD['thumb']],
                         obj_contact_provider=provider)
    gc.set_target(q_hold); gc.set_squeeze(True)
    squeeze_steps = 0
    for _ in range(int(args.settle / dt)):
        data.qvel[:N_ROBOT] = 0.0
        kp_eff, kd_eff = gc.effective_gains()
        tau = np.zeros(model.nv)
        tau[:N_ROBOT] = kp_eff * (q_hold - data.qpos[:N_ROBOT]) + kd_eff * (0 - data.qvel[:N_ROBOT])
        squeeze_steps += 1
        ramp = min(1.0, squeeze_steps * dt / SQUEEZE_RAMP_S)
        tau[:N_ROBOT] += gc.internal_force_torques(data, scale=ramp)
        tau[:N_ROBOT] += gc.slip_correction_torques(data, kp=args.slip_kp, f_max=args.slip_fmax)
        data.qfrc_applied[:] = tau
        data.qfrc_applied[:N_ROBOT] += data.qfrc_bias[:N_ROBOT]
        mj.mj_step(model, data)
    box_z_squeezed = data.xpos[obj_bid][2]

    # Contact-force diagnostic: how much grip actually formed between the fingertips and box.
    def _finger_box_contacts(d):
        if_g = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'leap_if_tip')
        th_g = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'leap_th_tip')
        # any collision geom on the two active fingers vs the box
        fdict = {'index': [], 'thumb': []}
        for ci in range(d.ncon):
            con = d.contact[ci]
            g1, g2 = con.geom1, con.geom2
            other = None
            if g1 == obj_gid: other = g2
            elif g2 == obj_gid: other = g1
            if other is None:
                continue
            nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, other) or ''
            f6 = np.zeros(6); mj.mj_contactForce(model, d, ci, f6)
            fn = abs(f6[0])   # normal force
            if nm.startswith('leap_if_'):
                fdict['index'].append(fn)
            elif nm.startswith('leap_th_'):
                fdict['thumb'].append(fn)
        return fdict
    _fc = _finger_box_contacts(data)
    _if_f = sum(_fc['index']); _th_f = sum(_fc['thumb'])
    print(f"[4] squeezed; box_z={box_z_squeezed:.4f} (Δ={(box_z_squeezed-box_z_pregrasp)*1e3:+.1f}mm)  "
          f"contact normal force: index={_if_f:.1f}N ({len(_fc['index'])}ct) "
          f"thumb={_th_f:.1f}N ({len(_fc['thumb'])}ct)")

    # ── 5. Jog palm +z (lift) ────────────────────────────────────────────────────
    jog_vel = args.jog_vel
    n_lift = int((args.lift / jog_vel) / dt)     # steps to travel `lift` at jog_vel
    # Track the box's height RELATIVE to the palm during the lift — this separates the two
    # failure modes: box-slips-through-fingers (relative drops) vs arm-sags-with-box (relative
    # constant, absolute drops). Live shows slow slip during a gradual lift.
    palm_z0 = data.xpos[palm_bid][2]; box_z0 = data.xpos[obj_bid][2]
    rel0 = box_z0 - palm_z0
    for _ in range(n_lift):
        Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, Jp, Jr, palm_bid)
        J6 = np.vstack([Jp[:, :7], Jr[:, :7]])
        v6 = np.array([0.0, 0.0, jog_vel, 0.0, 0.0, 0.0])   # +z linear
        sig = np.linalg.svd(J6, compute_uv=False)[-1]
        lam2 = 0.0 if sig >= JOG_SING_EPS else (1 - (sig / JOG_SING_EPS) ** 2) * JOG_LAM_MAX ** 2
        qdot = J6.T @ np.linalg.solve(J6 @ J6.T + lam2 * np.eye(6), v6)
        q_hold[:7] += qdot * dt
        data.qvel[:N_ROBOT] = 0.0; data.qvel[:7] = qdot
        kp_eff, kd_eff = gc.effective_gains()
        tau = np.zeros(model.nv)
        tau[:N_ROBOT] = (kp_eff * (q_hold - data.qpos[:N_ROBOT])
                         + kd_eff * (np.r_[qdot, np.zeros(16)] - data.qvel[:N_ROBOT]))
        tau[:N_ROBOT] += gc.internal_force_torques(data, scale=1.0)
        tau[:N_ROBOT] += gc.slip_correction_torques(data, kp=args.slip_kp, f_max=args.slip_fmax)
        data.qfrc_applied[:] = tau
        data.qfrc_applied[:N_ROBOT] += data.qfrc_bias[:N_ROBOT]
        mj.mj_step(model, data)
    box_z_top = data.xpos[obj_bid][2]
    rel_top = box_z_top - data.xpos[palm_bid][2]
    print(f"[5b] lift @ {jog_vel} m/s: box-rel-palm drift during lift = "
          f"{(rel_top - rel0)*1e3:+.1f}mm (neg = box slipped DOWN through fingers)")
    # HOLD at the top for `args.hold` s — the real test is whether the grip holds the box up
    # over time, not just the instant after the jog. Track the box height; a slipping grasp
    # sags/drops during the hold.
    # Per-finger contact-count / patch diagnostic: a fingertip that touches the box at a
    # single point rolls/slides under load regardless of normal force. Compare against the
    # live trace (which showed thumb=1ct). geom ids for the two active tips.
    _if_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'leap_if_tip')
    _th_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'leap_th_tip')
    def _tip_ncon(gid):
        pts = [data.contact[ci].pos for ci in range(data.ncon)
               if obj_gid in (data.contact[ci].geom1, data.contact[ci].geom2)
               and gid in (data.contact[ci].geom1, data.contact[ci].geom2)]
        if not pts:
            return 0, 0.0
        pts = np.array(pts)
        spread = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) > 1 else 0.0
        return len(pts), spread
    _ncon_if, _ncon_th, _spread_if, _spread_th = [], [], [], []
    z_trace = []
    for _ in range(int(args.hold / dt)):
        data.qvel[:N_ROBOT] = 0.0
        kp_eff, kd_eff = gc.effective_gains()
        tau = np.zeros(model.nv)
        tau[:N_ROBOT] = kp_eff * (q_hold - data.qpos[:N_ROBOT]) + kd_eff * (0 - data.qvel[:N_ROBOT])
        tau[:N_ROBOT] += gc.internal_force_torques(data, scale=1.0)
        tau[:N_ROBOT] += gc.slip_correction_torques(data, kp=args.slip_kp, f_max=args.slip_fmax)
        data.qfrc_applied[:] = tau
        data.qfrc_applied[:N_ROBOT] += data.qfrc_bias[:N_ROBOT]
        mj.mj_step(model, data)
        z_trace.append(data.xpos[obj_bid][2])
        nc_if, sp_if = _tip_ncon(_if_gid); nc_th, sp_th = _tip_ncon(_th_gid)
        _ncon_if.append(nc_if); _ncon_th.append(nc_th)
        _spread_if.append(sp_if); _spread_th.append(sp_th)
    z_trace = np.array(z_trace)
    print(f"[patch] hold contact count/patch:  index={np.mean(_ncon_if):.1f}ct "
          f"(spread {np.mean(_spread_if)*1e3:.1f}mm)  thumb={np.mean(_ncon_th):.1f}ct "
          f"(spread {np.mean(_spread_th)*1e3:.1f}mm)")

    box_z_final = data.xpos[obj_bid][2]
    lifted = (box_z_final - box_z_pregrasp)
    sag = (box_z_top - box_z_final)   # how much it dropped DURING the hold
    held = all(mj.mj_geomDistance(model, data, mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g),
                                  obj_gid, 0.2, _ft) < 0.01 for g in ('leap_if_tip', 'leap_th_tip'))
    # Held ABOVE ground throughout: min box height over the hold stayed well above rest.
    min_hold_z = float(z_trace.min()) if len(z_trace) else box_z_final
    stayed_up = (min_hold_z - box_z_pregrasp) > 0.5 * args.lift
    print(f"[5] jogged +{args.lift*1e3:.0f}mm; box_z_top={box_z_top:.4f}")
    print(f"[6] held {args.hold:.1f}s: box_z_final={box_z_final:.4f}  lifted={lifted*1e3:+.1f}mm  "
          f"sag_during_hold={sag*1e3:+.1f}mm  min_z_during_hold={(min_hold_z-box_z_pregrasp)*1e3:+.1f}mm  "
          f"held={held} stayed_up={stayed_up}")
    print()
    verdict = ('LIFTED & HELD' if lifted > 0.5 * args.lift and held and stayed_up else
               'PARTIAL/SLIPPED' if lifted > 0.01 else 'FAILED (box not lifted)')
    print(f"  accel={args.accel} pd_scale={args.pd_scale} gamma_mult={args.gamma_mult} "
          f"edge={args.edge*1e3:.0f}mm: {verdict} "
          f"(rose {lifted*1e3:+.1f}mm of {args.lift*1e3:.0f}mm, sag {sag*1e3:+.1f}mm over {args.hold:.0f}s)")


if __name__ == '__main__':
    main()

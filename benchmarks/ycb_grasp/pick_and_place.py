"""Tabletop pick-and-place benchmark: plan a grasp on a YCB object resting on
the table, execute it with grasp_control.GraspController, then transport the
object to the bin and release it.

The tabletop counterpart of pick_from_floor.py, and deliberately the same
pipeline: MultiStartGraspPlanner3D -> recommended contacts -> GraspController
squeeze -> resolved-rate DLS jog. Two things differ, both because the scene
does:

  * The run starts from the robot's HOME pose, not a random arm configuration.
    pick_from_floor randomizes q0 to exercise planning-plus-execution from an
    arbitrary posture; here the point is to characterize the GRASP on each
    object, so the arm always starts where the real system starts. (Note the
    home pose is table_scene.HOME_ARM, re-solved for the +90 deg yawed base --
    ik_demo.home_bias() aims the arm along +x, away from the objects.)

  * Fingertip clearance is measured against the TABLE TOP, not z=0, so
    ground_clearance_m is applied at table_scene.TABLE_TOP_Z.

After the lift, the object is carried to the bin authored in
scene_pick_place.xml and released; success is the scene's own 3D containment
test (table_scene.in_bin), the same one the teleop pipeline's arrival check
uses.

MODES
  --mode autonomous (default)
      Runs the full plan/execute/transport cycle headlessly (or with --view),
      writing artifacts per object. This is the benchmark.
  --mode scene-only
      Builds and settles the scene, prints what was loaded, and exits. For
      checking object placement/meshing without paying for a solve.
  --teleop-cmd
      Prints the kinova_leap_pick_place.py command that opens this same scene
      under teleop, and exits. The teleop entry point owns its own modes
      (contact_aware_teleop, dexpilot, anyteleop, ...); this benchmark does not
      reimplement or modify them -- the eventual integration is that teleop's
      CONTACT-PLANNING step calls the same planner this script exercises, while
      the dexpilot and anyteleop baselines stay untouched.

Every run writes into out/tabletop/<--out-tag>/<object>/ (see out_paths.py --
artifacts are grouped by ENVIRONMENT, and a tagged sweep nests under it rather
than creating a sibling top-level folder):
  seed<N>.png                  final-pose render
  seed<N>.mp4                  the whole run
  seed<N>_quadratic_path.png   per-contact local-quadratic fit + wrench summary
  seed<N>_mesh_fit.png         plane-vs-quadratic mesh fit at the solved contact

    python benchmarks/ycb_grasp/pick_and_place.py --object 036_wood_block
    python benchmarks/ycb_grasp/pick_and_place.py --object 065-a_cups --seed 2
    python benchmarks/ycb_grasp/pick_and_place.py --mode scene-only --object 065-j_cups
    python benchmarks/ycb_grasp/pick_and_place.py --teleop-cmd --object 017_orange
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from grasp_control import GraspController                                       # noqa: E402
from kinova_common.constants import FINGER_CODE, FINGER_SET, FINGER_TIP_SITES   # noqa: E402
from kinova_common.grasp_plots import write_grasp_plots                         # noqa: E402
from kinova_common.wrench import solve_gamma_live                               # noqa: E402
from simulation.grasp_config_builder import for_ablation_default                # noqa: E402
from simulation.grasp_planner_3d import MultiStartGraspPlanner3D                # noqa: E402
from ycb_grasp import out_paths as OP                                           # noqa: E402
from ycb_grasp import table_scene as TS                                         # noqa: E402
from ycb_grasp.ik_demo import clearance_by_geom, render, robot_geom_names       # noqa: E402
from ycb_grasp.pick_from_floor import (CONTACT_GAP_TOL_M, JOG_LAM_MAX,          # noqa: E402
                                       JOG_SING_EPS, PRINT_EVERY, SQUEEZE_RAMP_S,
                                       VideoRecorder, _gamma_stability_ceiling,
                                       _measured_tip_forces, _pad_surface_offset,
                                       _tip_gaps_mm, local_contact_frame,
                                       make_object_contact_provider,
                                       recommended_inward_normals)

N_ROBOT = TS.N_ROBOT
DEFAULT_COL_CLEARANCE_M = 0.002
APPROACH_STEPS = 300
LIFT_DISTANCE_M = 0.12          # clear the table before traversing
LIFT_SPEED_MPS = 0.06           # was 0.02 -- see JOG_RAMP_S on why this is safe
TRANSPORT_SPEED_MPS = 0.15      # lateral carry to the bin, faster than the lift
RELEASE_SETTLE_STEPS = 400

# Cartesian ACCELERATION slew limit on the commanded palm twist, adopted from
# the teleop stack (kinova_leap_pick_place.NCF_ACCEL_BUDGET_XYZ). This is not
# merely a smoothing nicety -- it is what makes the no-slip guarantee true.
#
# solve_gamma_live sizes the squeeze force for a disturbance BOX expressed as an
# acceleration budget, converting it to a force box via the object's own mass
# (f = m*a). That guarantee is only valid if the executed motion actually stays
# inside the box. The jog previously applied its full commanded twist on step 0,
# so the palm went from rest to the commanded speed in a single 2 ms timestep --
# an unbounded acceleration, i.e. a disturbance far outside the box gamma was
# solved for, at the exact moment the grasp is most fragile.
#
# Clamping the per-step CHANGE in commanded velocity to ACCEL_BUDGET*dt makes
# the budget "ENFORCED by construction" in the teleop comment's words, so the
# executed acceleration cannot exceed what the squeeze force was sized for --
# regardless of the cruise speed. That decoupling is what makes raising the
# speeds safe: peak speed sets how long the carry takes, the slew limit sets
# what the grasp has to survive.
JOG_ACCEL_BUDGET_MPS2 = 20.0    # teleop's NCF_ACCEL_BUDGET_XYZ (m/s^2)
JOG_VEL_MAX_MPS = 0.3           # teleop's JOG_VEL peak-speed cap (m/s)


def _obj_hull_geom_ids(model, body_name):
    """Every collision-hull geom id of one object. Concave YCB objects are
    V-HACD decompositions (065-a_cups: 35 hulls), and a gap/contact test against
    one arbitrary hull is not a test against the object."""
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    return [g for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid and model.geom_group[g] == 3
            and (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0)]


def _jog_steps(distance_m, speed_mps, dt):
    """Steps needed to travel distance_m at speed_mps under the slew limiter.

    The limiter spends v/a seconds ramping up and the same again ramping down,
    losing v^2/a of distance against a square velocity profile, so a naive
    distance/speed/dt stops the phase short -- the lift would clear less than
    LIFT_DISTANCE_M and the carry would land short of the bin.
    """
    v = min(max(speed_mps, 1e-9), JOG_VEL_MAX_MPS)
    n_ramp = int(np.ceil(v / JOG_ACCEL_BUDGET_MPS2 / dt))
    lost = v * v / JOG_ACCEL_BUDGET_MPS2          # ramp-up + ramp-down shortfall
    n_flat = max(int((distance_m + lost) / v / dt), 1)
    return n_flat + 2 * n_ramp


def _jog_to(model, data, ctrl, q_cmd, v6_fn, n_steps, _sync, palm_bid,
            obj_bid=None, tip_geom_ids=None, obj_gid=None, label="jog"):
    """Resolved-rate DLS jog driven by a per-step world-frame palm twist.

    Same singularity-robust pattern as pick_from_floor._run_lift_jog (JOG_SING_EPS
    / JOG_LAM_MAX identical), generalized to take the twist as a callable so the
    LIFT (straight up) and TRANSPORT (lateral toward the bin) phases share one
    implementation. Returns the updated joint command.
    """
    n = model.nv
    dt = model.opt.timestep
    contact_lost = {f: False for f in FINGER_SET}
    # Slew-limited velocity command, exactly as the teleop GRASP branch drives
    # it (kinova_leap_pick_place.py ~:5449): clip the TARGET to the peak-speed
    # cap, then move the COMMAND toward it by at most ACCEL*dt per step, and
    # ramp back to zero over the tail so the phase ends without a jolt either.
    dv_max = JOG_ACCEL_BUDGET_MPS2 * dt
    v_cmd = np.zeros(6)
    n_stop = int(np.ceil(JOG_VEL_MAX_MPS / JOG_ACCEL_BUDGET_MPS2 / dt))
    for i in range(n_steps):
        v_tgt = np.asarray(v6_fn(i), float).copy()
        v_tgt[:3] = np.clip(v_tgt[:3], -JOG_VEL_MAX_MPS, JOG_VEL_MAX_MPS)
        if i >= n_steps - n_stop:
            v_tgt[:] = 0.0                    # decelerate into the phase end
        v_cmd = v_cmd + np.clip(v_tgt - v_cmd, -dv_max, dv_max)
        v6 = v_cmd
        Jp = np.zeros((3, n))
        Jr = np.zeros((3, n))
        mj.mj_jacBody(model, data, Jp, Jr, palm_bid)
        J6 = np.vstack([Jp[:, :7], Jr[:, :7]])
        sigma_min = np.linalg.svd(J6, compute_uv=False)[-1]
        lam2 = (0.0 if sigma_min >= JOG_SING_EPS
                else (1.0 - (sigma_min / JOG_SING_EPS) ** 2) * JOG_LAM_MAX ** 2)
        qdot = J6.T @ np.linalg.solve(J6 @ J6.T + lam2 * np.eye(6), v6)
        q_cmd[:7] += qdot * dt

        data.qvel[:N_ROBOT] = 0.0
        data.qvel[:7] = qdot
        kp, kd = ctrl.effective_gains()
        tau = np.zeros(n)
        tau[:N_ROBOT] = (kp * (q_cmd - data.qpos[:N_ROBOT])
                         + kd * (np.r_[qdot, np.zeros(N_ROBOT - 7)] - data.qvel[:N_ROBOT])
                         + data.qfrc_bias[:N_ROBOT]
                         + ctrl.internal_force_torques(data, scale=1.0))
        data.qfrc_applied[:] = tau
        mj.mj_step(model, data)
        _sync()

        if tip_geom_ids is not None and obj_gid is not None:
            f = _measured_tip_forces(model, data, tip_geom_ids, obj_gid)
            for fname, force in zip(FINGER_SET, f):
                if force <= 1e-6:
                    contact_lost[fname] = True
        if i % PRINT_EVERY == 0 and obj_bid is not None:
            print(f"[{label}] t={i * dt:.2f}s obj_z={data.xpos[obj_bid][2]:.3f} "
                  f"palm_z={data.xpos[palm_bid][2]:.3f}")
    return q_cmd, contact_lost


def run_pick_place(object_id, seed, n_seeds=3, n_relin=3, gws=True, w_gws=5.0,
                   w_span=1.0, view=False, out_dir=None, do_transport=True,
                   max_iter=int(os.environ.get("PFF_MAXITER", 200)),
                   w_edge_margin=0.0, directional_r_tip=True,
                   mesh_fit=True, sdf_err_tol=None, quadratic_path=False,
                   quad_sym_normals=False, seed_rank_pool=1,
                   impratio=None, gamma_override=None,
                   squeeze_pd_scale=0.25, finger_kp=0.8, finger_kd=0.05,
                   lift_speed=LIFT_SPEED_MPS, transport_speed=TRANSPORT_SPEED_MPS,
                   contact_profile="stock"):
    """Plan + execute one grasp on one object, then carry it to the bin."""
    rng = np.random.default_rng(seed)
    t_build = time.time()
    model, data, info = TS.build([object_id], impratio=impratio)
    body_name = next(iter(info))
    obj_bid = info[body_name]["bid"]
    # Contact profile BEFORE the settle, so the object settles under the same
    # contact model the grasp is then measured with. contact_tuning's numbers are
    # fingertip-scoped on purpose: MuJoCo combines an unpaired contact's solref by
    # taking the MIN of the two geoms, so softening the OBJECT does nothing while
    # the fingertip sits at the stock 0.004 (= 2 timesteps at dt=2ms), which is the
    # release-fling mechanism that module documents.
    if contact_profile == "tuned":
        from ycb_grasp import contact_tuning as _ct
        _applied = _ct.apply_to_model(model, object_id=object_id)
        print(f"[contact] tuned profile: {_applied}")
    else:
        print(f"[contact] stock profile: impratio={model.opt.impratio:g} "
              f"noslip_iterations={model.opt.noslip_iterations} "
              f"timestep={model.opt.timestep:g}")
    TS.settle(model, data)
    pos, quat = TS.object_pose(model, data, body_name, info)
    # The compiled qpos0 must match the settled pose, or mj_resetData would put
    # the object back at its pre-settle spawn height.
    adr = info[body_name]["qadr"]
    model.qpos0[adr:adr + 3] = pos
    model.qpos0[adr + 3:adr + 7] = quat
    print(f"[scene] {object_id} -> {body_name} settled at "
          f"{np.round(pos, 3).tolist()} ({(time.time() - t_build):.1f}s)")

    q_home = TS.home_qpos()
    rgeoms = robot_geom_names(model)
    obj_geom0 = TS.hull_geoms(model, body_name)[0]
    obj_gids = _obj_hull_geom_ids(model, body_name)

    cfg_kw = dict(n_seeds=n_seeds, max_iter=max_iter, arm_geom_names=rgeoms,
                  obj_clearance_by_geom=clearance_by_geom(rgeoms),
                  col_clearance_m=DEFAULT_COL_CLEARANCE_M,
                  use_quadratic_contact=True,
                  quadratic_mesh_fit=mesh_fit,
                  w_edge_margin=w_edge_margin,
                  directional_r_tip=directional_r_tip,
                  w_align=float(os.environ.get("PFF_ALIGN_W", 10.0)),
                  orient_weight=float(os.environ.get("PFF_ORIENT_W", 2.0)),
                  # Fingertips must clear the TABLE, not the floor: the object
                  # rests on the table top, so a floor-relative clearance would
                  # permit contacts driven straight through the table surface.
                  ground_z=TS.TABLE_TOP_Z,
                  ground_clearance_m=0.006)
    if n_relin is not None:
        cfg_kw["n_normal_relinearize"] = n_relin
    if seed_rank_pool and seed_rank_pool > 1:
        # Over-generate random seeds and keep the ones a DLS-IK can actually
        # reach -- see GraspConfig3D.seed_dls_rank_pool.
        cfg_kw["seed_dls_rank_pool"] = int(seed_rank_pool)
    if quad_sym_normals:
        # Build the wrench/GWS contact frame from the paraboloid's own
        # analytic normal instead of freezing the seed's -- see
        # GraspConfig3D.quadratic_symbolic_normals.
        cfg_kw["quadratic_symbolic_normals"] = True
        # Per-consumer ablation. PFF_SYM_<CONSUMER>=0 turns the symbolic normal
        # back off for exactly one of the four consumers while the master flag
        # keeps it on for the rest, so the 6/8 -> 3/8 regression can be
        # attributed. Unset = follow the master flag.
        for _env, _key in (("PFF_SYM_FRAME",  "quad_sym_normals_frame"),
                           ("PFF_SYM_IKTGT",  "quad_sym_normals_iktgt"),
                           ("PFF_SYM_ALIGN",  "quad_sym_normals_align"),
                           ("PFF_SYM_ORIENT", "quad_sym_normals_orient")):
            _v = os.environ.get(_env)
            if _v is not None:
                cfg_kw[_key] = (_v not in ("0", "false", "False", ""))
    if sdf_err_tol is not None:
        # Trust-region tolerance for the local-quadratic surrogate: how far the
        # paraboloid may depart from the true SDF along each axis before that
        # axis's bound stops. Larger = more surface per patch, at more model
        # error. See GraspConfig3D.quadratic_sdf_err_tol.
        cfg_kw["quadratic_sdf_err_tol"] = sdf_err_tol
    if gws:
        cfg_kw["wrench_constraint"] = False
        cfg_kw["w_gws"] = w_gws
        cfg_kw["w_span"] = w_span
    cfg = for_ablation_default(obj_geom=obj_geom0, obj_body=body_name, **cfg_kw)

    log_dir = None
    if out_dir is not None:
        log_dir = str(Path(out_dir) / f"_quad_log_{object_id}_seed{seed}")
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir, exist_ok=True)

    # The seed drives the PLANNER's multi-start RNG, not the start pose. This
    # benchmark always launches from HOME (that is the point -- it characterizes
    # the grasp, not IK from an arbitrary posture), so the only legitimate source
    # of run-to-run variation is the planner's own restart jitter
    # (qref_restart_sigma_*). Leaving this unset made every --seed produce a
    # byte-identical trajectory, which silently turned a 3-seed sweep into one
    # sample repeated three times.
    planner = MultiStartGraspPlanner3D(model, data, cfg, log_dir=log_dir, seed=seed)
    t0 = time.time()
    res = planner.solve(q_home, np.asarray(pos, float), max_seeds=n_seeds)
    t_solve = time.time() - t0
    print(f"[plan] status={res.get('status')} rs={res.get('return_status')} "
          f"iterations={res.get('iterations')}  ({t_solve * 1e3:.0f}ms)")
    result = dict(object=object_id, seed=seed, t_solve_s=t_solve,
                  status=res.get("status"), phase_log=[])
    if res.get("q") is None or res.get("p1") is None:
        print("[plan] FAILED — no feasible grasp found.")
        result["phase_log"].append("plan_failed")
        return res, result

    verify_info = planner._planner.verify(res)
    result["gamma_min"] = verify_info.get("gamma_min")
    result["wrench_feasible"] = verify_info.get("wrench_feasible")
    print(f"[plan] wrench_feasible={result['wrench_feasible']} "
          f"gamma_min={result['gamma_min']}")

    q_target = np.zeros(N_ROBOT)
    q_target[:len(res["q"])] = res["q"]
    q_target[len(res["q"]):] = q_home[len(res["q"]):]

    tip_site_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
                    for f in FINGER_SET]
    tip_geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"leap_{FINGER_CODE[f]}_tip")
                    for f in FINGER_SET]

    data.qpos[:N_ROBOT] = q_target
    mj.mj_forward(model, data)

    if out_dir is not None:
        _write_plots(model, data, res, verify_info, log_dir, object_id, seed,
                     body_name, obj_bid, pos, out_dir,
                     quadratic_path=quadratic_path, n_relin=n_relin)
    shutil.rmtree(log_dir, ignore_errors=True)

    p_WoO = data.xpos[obj_bid].copy()
    R_WO = data.xmat[obj_bid].reshape(3, 3).copy()
    n1_in, n2_in = recommended_inward_normals(
        model, data, planner._planner._obj_gid, planner._planner._mesh_entry,
        np.asarray(res["p1"], float), np.asarray(res["p2"], float))
    by_p = {"thumb": res["p1"], "index": res["p2"]}
    by_n = {"thumb": n1_in, "index": n2_in}
    rec_local = [local_contact_frame(np.asarray(by_p[f], float),
                                     np.asarray(by_n[f], float), p_WoO, R_WO)
                 for f in FINGER_SET]
    pad_offset = {f: _pad_surface_offset(model, data, f, tip_site_ids[i], tip_geom_ids[i])
                  for i, f in enumerate(FINGER_SET)}

    if os.environ.get("PFF_CONTACT_TRACE"):
        # The disturbance LP's feasibility is decided by the contact GEOMETRY,
        # not by the object's mass -- two near-parallel inward normals cannot
        # resist a transverse wrench at any gamma, and the LP then returns None
        # and the caller silently substitutes GAMMA_FALLBACK. Dump the geometry
        # that actually went in, so an "infeasible" is diagnosable.
        _n1, _n2 = np.asarray(n1_in, float), np.asarray(n2_in, float)
        print(f"[contact] p1={np.round(res['p1'], 4).tolist()} "
              f"p2={np.round(res['p2'], 4).tolist()}")
        print(f"[contact] n1_in={np.round(_n1, 3).tolist()} "
              f"n2_in={np.round(_n2, 3).tolist()} "
              f"n1.n2={float(_n1 @ _n2):+.3f} "
              f"span={np.linalg.norm(np.asarray(res['p2'], float) - np.asarray(res['p1'], float)) * 1000:.1f}mm")

    gamma_live = _solve_gamma(model, data, obj_bid, R_WO, rec_local,
                              planner._planner._obj_gid, tip_geom_ids,
                              gamma_override)
    result["gamma"] = gamma_live

    Kp = np.concatenate([np.full(7, 40.0), np.full(16, finger_kp)])
    Kd = np.concatenate([np.full(7, 4.0), np.full(16, finger_kd)])
    ctrl = GraspController(
        model, N_ROBOT, tip_site_ids=tip_site_ids, obj_site_ids=None,
        obj_body_id=obj_bid, kp=Kp, kd=Kd,
        gamma=gamma_live, squeeze_pd_scale=squeeze_pd_scale, support_weight=True,
        pad_offsets=[pad_offset[f] for f in FINGER_SET],
        obj_contact_provider=make_object_contact_provider(rec_local, obj_bid))

    # Start from HOME (not a random configuration -- see the module docstring).
    mj.mj_resetData(model, data)
    data.qpos[:N_ROBOT] = q_home
    mj.mj_forward(model, data)
    ctrl.set_target(q_target)
    obj_gid = planner._planner._obj_gid

    viewer_cm = mj.viewer.launch_passive(model, data) if view else None
    viewer = viewer_cm.__enter__() if viewer_cm is not None else None
    if viewer is not None:
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # Frame the WHOLE task, not just the pick. The camera is static, so aiming
    # it at the object's spawn position (lookat=pos, dist=0.9) let the object
    # leave frame during transport -- the block is carried 0.70 m but that
    # framing only covers ~0.37 m from the lookat point, so the clip looked like
    # it stopped after the lift. Centre on the midpoint of the carry and pull
    # back far enough to hold both ends plus the lift height.
    _cam_lookat = np.array([0.5 * (pos[0] + TS.BIN_CENTER[0]),
                            0.5 * (pos[1] + TS.BIN_CENTER[1]),
                            pos[2] + 0.5 * LIFT_DISTANCE_M])
    _span = float(np.linalg.norm(np.asarray(TS.BIN_CENTER) - np.asarray(pos)[:2]))
    _cam_dist = max(0.9, 1.4 * _span + LIFT_DISTANCE_M)
    recorder = (VideoRecorder(str(Path(out_dir) / f"seed{seed}.mp4"),
                              lookat=_cam_lookat, dist=_cam_dist, elev=-25)
                if out_dir is not None else None)
    VIDEO_STRIDE = 4
    frame_i = [0]

    def _sync():
        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)
        if recorder is not None and frame_i[0] % VIDEO_STRIDE == 0:
            recorder.capture(model, data)
        frame_i[0] += 1

    palm_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "leap_palm")
    try:
        # APPROACH: kinematic replay home -> planned pose.
        path = np.linspace(q_home, q_target, APPROACH_STEPS)
        for i in range(APPROACH_STEPS):
            data.qpos[:N_ROBOT] = path[i]
            data.qvel[:N_ROBOT] = 0.0
            mj.mj_forward(model, data)
            _sync()
        result["phase_log"].append("approach_done")

        # HOLD: quasi-static settle at the planned pose.
        for _ in range(200):
            data.qvel[:N_ROBOT] = 0.0
            data.qfrc_applied[:] = ctrl.compute(data)
            mj.mj_step(model, data)
            _sync()
        result["phase_log"].append("hold_settled")

        gaps = _tip_gaps_mm(model, data, tip_geom_ids, obj_gid, obj_geom_ids=obj_gids)
        result["tip_gaps_mm"] = dict(zip(FINGER_SET, gaps))
        if any(g > CONTACT_GAP_TOL_M * 1000 for g in gaps):
            print(f"[exec] ABORT before SQUEEZE — gap too large: {result['tip_gaps_mm']}")
            result["phase_log"].append("squeeze_aborted_no_contact")
            return res, result
        print(f"[exec] gap check OK: {result['tip_gaps_mm']}")

        # SQUEEZE: ramp the internal force in.
        ctrl.set_squeeze(True)
        n_ramp = max(int(SQUEEZE_RAMP_S / model.opt.timestep), 1)
        for i in range(n_ramp * 4):
            scale = min(1.0, i / n_ramp)
            data.qvel[:N_ROBOT] = 0.0
            data.qfrc_applied[:] = ctrl.compute(data)
            if scale < 1.0:
                kp, kd = ctrl.effective_gains()
                data.qfrc_applied[:N_ROBOT] = (
                    kp * (q_target - data.qpos[:N_ROBOT])
                    + kd * (0 - data.qvel[:N_ROBOT])
                    + data.qfrc_bias[:N_ROBOT]
                    + ctrl.internal_force_torques(data, scale=scale))
            mj.mj_step(model, data)
            _sync()
            # PFF_SQUEEZE_TRACE=1 samples the gap and the measured force through
            # the ramp. The design assumption (see pick_from_floor's
            # CONTACT_GAP_TOL_M comment) is that the squeeze CLOSES the gap the
            # IK deliberately leaves; this is how you check that it actually does
            # on a given object rather than assuming it.
            if os.environ.get("PFF_SQUEEZE_TRACE") and i % 100 == 0:
                _g = _tip_gaps_mm(model, data, tip_geom_ids, obj_gid, obj_gids)
                _f = _measured_tip_forces(model, data, tip_geom_ids, obj_gid)
                print(f"[squeeze] i={i:4d} scale={scale:.2f} "
                      f"gap={np.round(_g, 2).tolist()} f={np.round(_f, 2).tolist()}")
        f_meas = _measured_tip_forces(model, data, tip_geom_ids, obj_gid)
        result["squeeze_forces_N"] = dict(zip(FINGER_SET, np.round(f_meas, 3).tolist()))
        result["phase_log"].append("squeeze_done")
        print(f"[squeeze] final={result['squeeze_forces_N']}")

        # Holding gains for everything that follows (see effective_gains).
        ctrl.set_transporting(True)
        obj_z0 = float(data.xpos[obj_bid][2])
        q_cmd = q_target.copy()

        # LIFT: straight up, clear of the table.
        dt = model.opt.timestep
        n_lift = _jog_steps(LIFT_DISTANCE_M, lift_speed, dt)
        q_cmd, lost_lift = _jog_to(
            model, data, ctrl, q_cmd, lambda i: np.array([0, 0, lift_speed, 0, 0, 0.]),
            n_lift, _sync, palm_bid, obj_bid, tip_geom_ids, obj_gid, label="lift")
        result["lift_obj_dz_mm"] = (float(data.xpos[obj_bid][2]) - obj_z0) * 1000
        result["lift_contact_lost"] = lost_lift
        result["phase_log"].append("lift_done")
        print(f"[lift] object rose {result['lift_obj_dz_mm']:.1f}mm  lost={lost_lift}")

        if do_transport:
            # TRANSPORT: lateral toward the bin centre, holding height.
            delta = TS.BIN_CENTER - data.xpos[obj_bid][:2]
            dist = float(np.linalg.norm(delta))
            n_tr = _jog_steps(dist, transport_speed, dt)
            dirn = delta / max(dist, 1e-9)
            q_cmd, lost_tr = _jog_to(
                model, data, ctrl, q_cmd,
                lambda i: np.array([dirn[0] * transport_speed,
                                    dirn[1] * transport_speed, 0, 0, 0, 0.]),
                n_tr, _sync, palm_bid, obj_bid, tip_geom_ids, obj_gid, label="transport")
            result["transport_contact_lost"] = lost_tr
            result["phase_log"].append("transport_done")

            # RELEASE: stop squeezing and let the object settle into the bin.
            ctrl.set_squeeze(False)
            ctrl.set_transporting(False)
            # PEAK RELEASE VELOCITY — the fling metric. contact_tuning.py scores
            # every contact setting on BOTH grasp force AND this number, because
            # the two trade off along the same axis and optimizing either alone
            # picks a setting that fails the other. Sampled over the whole settle
            # (not just the final state) since the fling is a transient: the object
            # is pumped over ~50ms and may well have hit something by step 400.
            _peak_v = 0.0
            _peak_w = 0.0
            for _ in range(RELEASE_SETTLE_STEPS):
                data.qvel[:N_ROBOT] = 0.0
                data.qfrc_applied[:] = ctrl.compute(data)
                mj.mj_step(model, data)
                _vw = data.cvel[obj_bid]          # spatial velocity, [ang(3); lin(3)]
                _peak_w = max(_peak_w, float(np.linalg.norm(_vw[:3])))
                _peak_v = max(_peak_v, float(np.linalg.norm(_vw[3:])))
                _sync()
            result["release_peak_speed_mps"] = round(_peak_v, 3)
            result["release_peak_spin_radps"] = round(_peak_w, 2)
            print(f"[release] peak |v|={_peak_v:.3f} m/s  |w|={_peak_w:.1f} rad/s")
            result["in_bin"] = TS.in_bin(model, data, obj_bid)
            result["object_final_xyz"] = np.round(data.xpos[obj_bid], 4).tolist()
            result["phase_log"].append("release_done")
            print(f"[place] in_bin={result['in_bin']} "
                  f"obj={result['object_final_xyz']}")

        if out_dir is not None:
            try:
                render(model, data, str(Path(out_dir) / f"seed{seed}.png"),
                       lookat=pos, dist=0.9, elev=-25)
            except Exception as e:
                print(f"[exec] render failed: {e}")
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)
        if recorder is not None:
            recorder.close()

    return res, result


def _solve_gamma(model, data, obj_bid, R_WO, rec_local, obj_gid, tip_geom_ids,
                 override):
    """Datum-mode internal force for a hold/transport disturbance budget, clamped
    to the simulator's stability ceiling. Same formulation as pick_from_floor --
    see its comment for why Task-B (datum) and not Task-A (CoM)."""
    if override is not None:
        return float(override)
    ACCEL = (0.5, 0.5, 0.5)
    ANG = (0.1, 0.1, 0.1)
    FALLBACK = 2.0
    g_O = R_WO.T @ model.opt.gravity
    mu = [float(model.geom_friction[obj_gid, 0])] * len(FINGER_SET)
    gamma = solve_gamma_live([p for p, _ in rec_local], [R for _, R in rec_local],
                             mu, float(model.body_mass[obj_bid]), ACCEL, ANG,
                             model.body_inertia[obj_bid], grav_O=g_O)
    if gamma is None or not np.isfinite(gamma) or gamma <= 0.0:
        print(f"[plan] solve_gamma_live infeasible -> gamma={FALLBACK}")
        gamma = FALLBACK
    ceiling = _gamma_stability_ceiling(model, obj_bid, tip_geom_ids)
    if gamma > ceiling:
        print(f"[plan] clamping gamma {gamma:.2f} -> {ceiling:.2f} (stability ceiling)")
        gamma = ceiling
    else:
        print(f"[plan] gamma={gamma:.2f} (ceiling {ceiling:.2f})")
    return gamma


def _write_plots(model, data, res, verify_info, log_dir, object_id, seed,
                 body_name, obj_bid, pos, out_dir, quadratic_path=False,
                 n_relin=None):
    """Thin shim onto kinova_common.grasp_plots.write_grasp_plots.

    The implementation moved there so the live teleop recommender can emit the
    SAME figures; this signature is preserved for the call site below.
    """
    return write_grasp_plots(model, data, res, verify_info, log_dir, object_id,
                             seed, body_name, obj_bid, pos, out_dir,
                             quadratic_path=quadratic_path, n_relin=n_relin)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", default="036_wood_block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--n-relin", type=int, default=3)
    ap.add_argument("--mode", choices=["autonomous", "scene-only"],
                    default="autonomous")
    ap.add_argument("--teleop-cmd", action="store_true",
                    help="print the kinova_leap_pick_place.py command that opens this "
                         "same scene under teleop, then exit (does not run teleop)")
    ap.add_argument("--teleop-mode", default="contact_aware_teleop",
                    help="--mode to put in the printed teleop command. The dexpilot and "
                         "anyteleop baselines are untouched by this benchmark.")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--no-transport", dest="do_transport", action="store_false",
                    help="stop after the lift; skip carrying to the bin")
    ap.add_argument("--no-mesh-fit", dest="mesh_fit", action="store_false",
                    help="use the SDF-Hessian curvature instead of the mesh fit")
    ap.add_argument("--w-edge-margin", type=float, default=0.0)
    ap.add_argument("--seed-rank-pool", type=int, default=1,
                    help="generate this many times n_seeds random candidates and "
                         "keep the best by DLS-IK fingertip residual (1 = off)")
    ap.add_argument("--quad-sym-normals", action="store_true",
                    help="build the contact frame from the paraboloid's analytic "
                         "normal instead of freezing the seed's "
                         "(GraspConfig3D.quadratic_symbolic_normals)")
    ap.add_argument("--quadratic-path", action="store_true",
                    help="write the Picard-trajectory figure instead of the "
                         "per-contact grasp figure (useful when tuning "
                         "--n-relin; near-empty at --n-relin 0)")
    ap.add_argument("--sdf-err-tol", type=float, default=None,
                    help="metres; max surrogate-vs-true-SDF gap that sizes each "
                         "trust-region axis (default: GraspConfig3D's 5e-4)")
    ap.add_argument("--impratio", type=float, default=None,
                    help="override the scene's contact impratio (scene XML sets 100; "
                         "the floor-pick benchmark measured 20 as best)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="override the solved internal-force scale")
    ap.add_argument("--squeeze-pd-scale", type=float, default=0.25,
                    help="finger PD multiplier DURING the squeeze ramp. Lower lets the "
                         "internal-force term win against the finger PD; too low and the "
                         "measured force falls short of the commanded gamma.")
    ap.add_argument("--contact-profile", choices=["stock", "tuned"], default="stock",
                    help="stock (default): whatever the scene XML compiles to "
                         "(impratio=100, noslip_iterations=0, fingertip "
                         "solref=[0.004,1.0]). tuned: apply contact_tuning.py's "
                         "measured settings (fingertip solref=[0.02,2.0], "
                         "noslip_iterations=5). NOTE contact_tuning pairs noslip=5 "
                         "with gamma=10.0 and squeeze_pd_scale=1.0 -- pass those "
                         "via --gamma/--squeeze-pd-scale; noslip alone removes the "
                         "tangential compliance a heavy object leans on.")
    ap.add_argument("--finger-kp", type=float, default=0.8)
    ap.add_argument("--finger-kd", type=float, default=0.05)
    ap.add_argument("--lift-speed", type=float, default=LIFT_SPEED_MPS,
                    help=f"vertical lift speed m/s (default {LIFT_SPEED_MPS}). The jog "
                         "eases in/out over JOG_RAMP_S, so this is the cruise speed.")
    ap.add_argument("--transport-speed", type=float, default=TRANSPORT_SPEED_MPS,
                    help=f"lateral carry speed m/s (default {TRANSPORT_SPEED_MPS})")
    OP.add_out_args(ap, OP.TABLETOP)
    args = ap.parse_args()

    if args.teleop_cmd:
        print("# Same scene under the existing teleop entry point:")
        print(f"python kinova_leap_pick_place.py --scene pick_place "
              f"--mode {args.teleop_mode}")
        print("#")
        print("# Object selection there comes from models/scene_objects.json's "
              "'pick_place' list.")
        print(f"# To load only {args.object}, edit that list (or pass the entry point's "
              "own --objects flag).")
        print("# The dexpilot / anyteleop baseline modes are unmodified by this "
              "benchmark.")
        return

    if args.mode == "scene-only":
        model, data, info = TS.build([args.object], impratio=args.impratio)
        TS.settle(model, data)
        bn = next(iter(info))
        p, q = TS.object_pose(model, data, bn, info)
        V = TS.hull_vertices(model, bn)
        ext = (V.max(0) - V.min(0)) * 1000
        print(f"[scene-only] {args.object} -> {bn}")
        print(f"  mass       {model.body_mass[info[bn]['bid']]:.3f} kg")
        print(f"  extents    {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm")
        print(f"  hulls      {len(TS.hull_geoms(model, bn))}")
        print(f"  settled at {np.round(p, 4).tolist()} (table top {TS.TABLE_TOP_Z})")
        print(f"  impratio   {model.opt.impratio}")
        return

    out_dir = OP.resolve_out(args, OP.TABLETOP) / args.object
    out_dir.mkdir(parents=True, exist_ok=True)
    _, result = run_pick_place(
        args.object, args.seed, n_seeds=args.n_seeds, n_relin=args.n_relin,
        view=args.view, out_dir=str(out_dir), do_transport=args.do_transport,
        w_edge_margin=args.w_edge_margin, mesh_fit=args.mesh_fit,
        sdf_err_tol=args.sdf_err_tol, quadratic_path=args.quadratic_path,
        quad_sym_normals=args.quad_sym_normals,
        seed_rank_pool=args.seed_rank_pool,
        impratio=args.impratio, gamma_override=args.gamma,
        squeeze_pd_scale=args.squeeze_pd_scale,
        finger_kp=args.finger_kp, finger_kd=args.finger_kd,
        lift_speed=args.lift_speed, transport_speed=args.transport_speed,
        contact_profile=args.contact_profile)
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

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
from kinova_common.constants import (FINGER_CODE, FINGER_SET,                    # noqa: E402
                                     FINGER_TIP_SITES, SLOT_ROLES,
                                     finger_joint_slices)
from kinova_common.grasp_plots import write_grasp_plots                         # noqa: E402
from kinova_common.wrench import solve_gamma_live                               # noqa: E402
from simulation.epsilon_metric import (epsilon_quality,                        # noqa: E402
                                       epsilon_subspace)
from simulation.grasp_config_builder import (parse_fingers as _parse_fingers,    # noqa: E402
                                             for_gws_recommender,
                                             for_ablation_default,
                                             load_seed_config)                  # noqa: E402
from simulation.grasp_planner_3d import (MultiStartGraspPlanner3D,   # noqa: E402
                                         _geom_normal_np)                # noqa: E402
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

# ── What counts as a successful LIFT ─────────────────────────────────────────
# result["lift_ok"] is the single boolean to aggregate over. It exists because
# the fields it replaces each answer only half the question, and reading either
# alone counts real failures as successes:
#
#   lift_obj_dz_mm      the object's z-displacement, and NOTHING about whether
#                       the hand still has it. Measured on 061_foam_brick:
#                       +119mm with lift_contact_lost={'thumb': True} -- the
#                       object was carried on ONE finger and scored as a lift.
#                       It also goes NEGATIVE when the object is knocked off the
#                       table (-103.4mm on 036_wood_block) while still logging
#                       phase "lift_done".
#   lift_contact_lost   sticky per-finger flag, set the first time a tip's
#                       measured force touches <=1e-6 N at ANY sampled step and
#                       never cleared. A tip can momentarily unload mid-jog and
#                       re-seat, which is not a failed grasp, so this alone is
#                       too strict to gate on.
#   phase_log           "lift_done" is appended unconditionally after the jog
#                       returns. It records that the jog RAN, not that it
#                       worked, so it is not a success signal at all (the same
#                       class as the known release_done != grasped issue).
#
# lift_ok therefore requires BOTH: the object actually went up, and the hand is
# still holding it WHEN THE LIFT ENDS. Transient mid-jog unloading is tolerated
# (the sticky flag is reported separately as lift_contact_lost for anyone who
# wants the strict reading); a one-finger carry at the end is not.
LIFT_OK_MIN_FRAC = 0.80         # of LIFT_DISTANCE_M -- "reached most of the
                                # commanded travel", not merely "moved at all"
LIFT_OK_MIN_FORCE_N = 1e-3      # per fingertip at the final step. Above the
                                # 1e-6 noise floor _jog_to trips on, so a tip
                                # that is merely grazing does not pass.

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

# Disturbance budget for the grasp NLP's gamma certificate, shared with the teleop
# recommender (kinova_leap_pick_place.NCF_ACCEL_BUDGET_XYZ / NCF_ANG_ACCEL_BUDGET)
# now that both use the same solver preset. This REPLACES the former per-script
# (0.25, 0.25, 0.25) / (0.5, 0.5, 0.5) budget: gamma is solved per object either
# way, but a ~80x larger budget yields much larger gamma, so gamma_min values are
# not comparable across this change. It matches JOG_ACCEL_BUDGET_MPS2 above, which
# is what the jog's slew limiter actually enforces -- so the certificate is now
# sized for the motion this script really executes.
NCF_ACCEL_BUDGET_XYZ = (JOG_ACCEL_BUDGET_MPS2,) * 3
NCF_ANG_ACCEL_BUDGET = (1.0, 1.0, 1.0)
JOG_VEL_MAX_MPS = 0.3           # teleop's JOG_VEL peak-speed cap (m/s)

# The shaky protocol's own lift height, imported rather than restated so the
# lift_ok threshold and the executed trajectory cannot drift apart.
from ycb_grasp.shaky_pickup import LIFT_HEIGHT_M as SHAKY_LIFT_M   # noqa: E402


def _fmt_eps(v):
    """Epsilon in the paper's units (x1e3), or a marker when it is undefined.

    None is NOT formatted as 0: at n=2 the wrench set is rank-deficient and
    epsilon does not exist, which is a different statement from "epsilon is zero"
    (a grasp exactly on the closure boundary). See simulation/epsilon_metric.py.
    """
    return "--" if v is None else f"{1e3 * v:.2f}e-3"


def _fingers_for_object(obj_id, fingers):
    """Slot-ordered finger list for ONE object.

    An explicit --fingers wins (operator intent beats the table). Otherwise the
    per_object map in models/grasp_finger_config.json decides, falling back to
    its default list. Without this the benchmark fell back to the module-level
    SLOT_ROLES, which is resolved at IMPORT from the config default and cannot
    see per_object -- so every object ran at thumb+index and the tripod
    assignments were silently ignored. Mirrors _rec_fingers_for in
    kinova_leap_pick_place.py so plan and execution agree in both harnesses.
    """
    parsed = _parse_fingers(fingers)
    if parsed:
        return list(parsed)
    import json as _json
    from simulation.grasp_config_builder import FINGER_CONFIG_PATH as _FCP
    try:
        raw = _json.loads(Path(_FCP).read_text())
    except (OSError, ValueError):
        return list(SLOT_ROLES)
    roles = ((raw.get('per_object') or {}).get(obj_id)
             or raw.get('fingers') or SLOT_ROLES)
    return [r for r in roles if isinstance(r, str)]


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
            obj_bid=None, tip_geom_ids=None, obj_gid=None, label="jog",
            finger_set=None, obj_geom_ids=None):
    """Resolved-rate DLS jog driven by a per-step world-frame palm twist.

    Same singularity-robust pattern as pick_from_floor._run_lift_jog (JOG_SING_EPS
    / JOG_LAM_MAX identical), generalized to take the twist as a callable so the
    LIFT (straight up) and TRANSPORT (lateral toward the bin) phases share one
    implementation. Returns the updated joint command.
    """
    n = model.nv
    dt = model.opt.timestep
    # This run's fingers, not the import-time default: under --fingers thumb,middle
    # the module global still says ['index','thumb'], so contact loss would be
    # attributed to a finger that is not grasping.
    _fset = list(finger_set) if finger_set else list(FINGER_SET)
    contact_lost = {f: False for f in _fset}
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
            f = _measured_tip_forces(model, data, tip_geom_ids, obj_gid,
                                     obj_geom_ids=obj_geom_ids)
            for fname, force in zip(_fset, f):
                if force <= 1e-6:
                    contact_lost[fname] = True
        if i % PRINT_EVERY == 0 and obj_bid is not None:
            print(f"[{label}] t={i * dt:.2f}s obj_z={data.xpos[obj_bid][2]:.3f} "
                  f"palm_z={data.xpos[palm_bid][2]:.3f}")
    return q_cmd, contact_lost


def run_pick_place(object_id, seed, n_seeds=None, n_relin=None, gws=True, w_gws=None,
                   w_span=1.0, view=False, out_dir=None, do_transport=True,
                   # None DEFERS to the preset (which measures 800 as the cap
                   # that converges 7/7; 80 gave 4/7). Passing a number here
                   # overrides it, and passing the old hardcoded 200 silently
                   # defeated the preset entirely -- solves reported iters=200
                   # while the preset asked for 800.
                   max_iter=(int(os.environ["PFF_MAXITER"])
                             if os.environ.get("PFF_MAXITER") else None),
                   w_edge_margin=0.0, directional_r_tip=True,
                   mesh_fit=True, sdf_err_tol=None, bound_inset=None,
                   bound_keep_frac=None,
                   quad_sym_normals=False, seed_rank_pool=1, backend=None,
                   impratio=None, gamma_override=None,
                   squeeze_pd_scale=0.25, finger_kp=0.8, finger_kd=0.05,
                   gamma_ref=1.0,
                   lift_speed=LIFT_SPEED_MPS, transport_speed=TRANSPORT_SPEED_MPS,
                   contact_profile="stock", fingers=None, force_execute=False,
                   release_open_frac=0.5,
                   lift_mode="standard", plan_override=None,
                   nullspace_tracking=False, gap_tol_m=None,
                   sep_hard=False, min_sep_mm=12.0,
                   squeeze_pd_per_finger=False, contact_gated_alloc=False,
                   force_feedback_ki=0.0, advance_q_target=0.0):
    """Plan + execute one grasp on one object, then carry it to the bin.

    lift_mode : "standard" (default) runs this benchmark's own 12 cm lift, scored
        by `lift_ok`. "shaky" substitutes FRoGGeR's execution test -- 10 cm in 1 s,
        hold 1.5 s, 3 mm sinusoid in all axes -- and additionally reports
        `pick_success` under their failure criteria. Only the LIFT changes; every
        earlier phase is identical, so the two modes are comparable up to the lift.

    plan_override : optional callable supplying (cfg, q_start) for an ALTERNATE
        planner arm, so it can be executed and scored down this same path. Called
        with keywords (body_name, model, data, info, cfg_kw, fingers, seed, pos,
        q_home); `cfg_kw` carries this benchmark's assembled knobs for the arm to
        extend. Return q_start=None to keep the HOME start. None (default) uses
        `for_gws_recommender` from HOME, which is what every existing tabletop
        result was measured with -- that path is unchanged.
    """
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

    # n_seeds=None defers to the preset (which sets 1 -- one seed reaches the
    # NLP, the DLS rank pool does the selecting). Only pass it as an override
    # when the caller asked for a specific count.
    cfg_kw = dict(arm_geom_names=rgeoms,
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
    if max_iter is not None:
        cfg_kw["max_iter"] = max_iter
    if n_seeds is not None:
        cfg_kw["n_seeds"] = n_seeds
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
    if backend is not None:
        # Solver BACKEND only -- the NLP is identical either way. use_slsqp appears
        # in exactly two places in grasp_planner_3d.py: which plugin Opti gets
        # (:3929) and a log label (:4098). Every cost term, constraint, bound and
        # seed gate -- and the smooth-SDF alpha, despite its slsqp_alpha name, which
        # is gated on cfg.smooth_sdf not on the backend -- is built BEFORE that
        # branch. So this flag swaps the solver on one fixed problem.
        cfg_kw["use_slsqp"] = (backend == "sqp")
    if sdf_err_tol is not None:
        # Trust-region tolerance for the local-quadratic surrogate: how far the
        # paraboloid may depart from the true SDF along each axis before that
        # axis's bound stops. Larger = more surface per patch, at more model
        # error. See GraspConfig3D.quadratic_sdf_err_tol.
        cfg_kw["quadratic_sdf_err_tol"] = sdf_err_tol
    if bound_inset is not None:
        # Constant keep-back from every trust-region bound. Independent of
        # sdf_err_tol above: that one asks "is the surrogate still accurate
        # here", which stays TRUE right up to a box's crease, so it alone
        # cannot keep a contact off a sharp edge. See
        # GraspConfig3D.quadratic_bound_inset.
        cfg_kw["quadratic_bound_inset"] = bound_inset
    if bound_keep_frac is not None:
        # Floor on what the inset may leave -- see
        # GraspConfig3D.quadratic_bound_keep_frac.
        cfg_kw["quadratic_bound_keep_frac"] = bound_keep_frac
    # --gws is now the DEFAULT architecture (see for_gws_recommender below), so
    # the flag only overrides the term WEIGHTS; wrench_constraint=False comes
    # from the shared preset either way.
    if gws:
        # w_gws=None (the default) DEFERS to the preset, which picks the value
        # matching its own beta normalization (beta*n_cols vs raw beta). Forcing
        # 5.0 here would silently re-apply the raw-beta weight on top of the
        # scaled term, a ~14x over-weighting. Pass a number to mean it literally.
        if w_gws is not None:
            cfg_kw["w_gws"] = w_gws
        cfg_kw["w_span"] = w_span
        # PFF_GWS_ALPHA_REG: proximal (Tikhonov) regularization on the
        # embedded min-weight LP's alpha (see GWSConfig.gws_alpha_reg) --
        # turns that LP into a strictly convex QP so its primal/dual are
        # unique at a symmetric antipodal pinch. Unset = 0.0 (off, prior
        # behavior). PFF_GWS_SMOOTH_FRAME=1: swap _symbolic_contact_frame_ca's
        # ca.fabs kink for a smooth surrogate (GWSConfig.gws_smooth_frame).
        # Both are ablation knobs for the IPOPT dual-indeterminacy
        # investigation, not defaults -- see their GWSConfig docstrings.
        _alpha_reg = os.environ.get("PFF_GWS_ALPHA_REG")
        if _alpha_reg is not None:
            cfg_kw["gws_alpha_reg"] = float(_alpha_reg)
        if os.environ.get("PFF_GWS_SMOOTH_FRAME"):
            cfg_kw["gws_smooth_frame"] = True
        # PFF_GWS_SOFT_FINGER=1: re-probe the soft-finger
        # W (GWSConfig.gws_soft_finger). Off by default and MEASURED HARMFUL for
        # beta -- see for_gws_recommender's note. Exposed because that
        # abandonment was decided on RAW beta, which is not comparable across
        # the column-count change the flag causes (10 -> 14 cols moves the
        # arithmetic ceiling 1/n_cols from 0.100 to 0.0714), and because the
        # flag IS the exact fix for the rank-5-of-6 dual indeterminacy: it
        # takes rank(W) 5 -> 6 and cond(W) inf -> 40 at the authored mu_t=0.05.
        if os.environ.get("PFF_GWS_SOFT_FINGER"):
            cfg_kw["gws_soft_finger"] = True
        # PFF_GWS_BETA_SCALE_NCOLS=1: weight beta*n_cols instead of raw beta so
        # w_gws means the same thing across column-count changes (soft-finger
        # 10->14, n_contacts 2->3). PFF_GWS_W overrides w_gws itself, which you
        # generally want alongside the scaling since it multiplies the effective
        # weight by n_cols (w_gws=5.0 raw ~= 0.5 scaled at 10 cols).
        if os.environ.get("PFF_GWS_BETA_SCALE_NCOLS"):
            cfg_kw["gws_beta_scale_ncols"] = True
        _w_gws_env = os.environ.get("PFF_GWS_W")
        if _w_gws_env is not None:
            cfg_kw["w_gws"] = float(_w_gws_env)
    # PFF_QUAD_TANGENT_FRAME=0 reverts the quadratic contact frame's TANGENT
    # basis to the pre-change _symbolic_contact_frame_ca reconstruction (the
    # tanh(|n0|-0.9)/0.01 blend). Default is the paraboloid's own dp0/dp1 --
    # see GraspConfig3D.quad_tangent_frame. Outside the `if gws` block because
    # the frame is used by the wrench cone too, not just the GWS terms.
    _qtf = os.environ.get("PFF_QUAD_TANGENT_FRAME")
    if _qtf is not None:
        cfg_kw["quad_tangent_frame"] = (_qtf not in ("0", "false", "False", ""))
    # Seed/surrogate settings from models/grasp_seed_config.json, applied as
    # DEFAULTS (setdefault, not update) so every explicit CLI flag and PFF_* env
    # override set above still wins. Precedence: file -> per-object -> CLI/env.
    for _k, _v in load_seed_config(object_id).items():
        cfg_kw.setdefault(_k, _v)
    # STANDARDIZED CONFIG: the same preset the teleop recommender uses
    # (kinova_leap_pick_place.py _get_cat_planner) -- IK-only NLP
    # (wrench_constraint=False) + datum-gamma certificate + the FRoGGeR
    # min-weight objective + soft-finger W + quadratic contacts.
    #
    # Everything in cfg_kw is passed through as an OVERRIDE, so this benchmark
    # KEEPS its own tuned seeding//solver knobs that the preset would otherwise
    # dictate: max_iter (200, not the live loop's 120), directional_r_tip,
    # seed_ground_clearance_m and the rest of grasp_seed_config.json, plus every
    # PFF_* env and CLI override. Only the architecture is shared.
    #
    # NOTE the disturbance budget now comes from the preset: 20 m/s^2 / 1 rad/s^2
    # (the teleop carry budget) instead of this script's former 0.25 / 0.5. gamma
    # is solved per object either way, but against a budget ~80x larger, so
    # reported gamma_min values are NOT comparable to runs from before this change.
    cfg_kw.pop("obj_geom", None)
    cfg_kw.setdefault("obj_geom", obj_geom0)
    # fingers: ordered role list, e.g. ['thumb','middle']. None = whatever
    # models/grasp_finger_config.json holds for this object (or its default list).
    # ORDER MATTERS: slot 1 anchors the antipodal seed march. The preset applies it
    # with setdefault, so every CLI/env override above still wins.
    # PLAN OVERRIDE: let a caller supply the config and the start pose, so an
    # ALTERNATE PLANNER ARM can be executed down this same path. Everything after
    # the solve -- gamma, squeeze, lift, scoring -- is then shared by construction,
    # which is the property a between-arm execution comparison needs. Without this
    # the preset and the HOME start are hardcoded, so only `ours` was executable
    # and the FRoGGeR arm stopped at the planner.
    #
    # The hook takes the assembled cfg_kw so an arm inherits this benchmark's tuned
    # seeding/solver knobs and overrides only what its formulation requires.
    if plan_override is not None:
        # PARSED slot list, not the raw comma-string `fingers` arrives as. The
        # override needs the same ordered role list `_SLOTS` is built from below;
        # handing it the string made `FINGER_TIP_SITES[r]` iterate CHARACTERS
        # (KeyError: 't').
        _ov = plan_override(
            body_name=body_name, model=model, data=data, info=info,
            cfg_kw=dict(cfg_kw),
            fingers=_fingers_for_object(object_id, fingers),
            seed=seed, pos=pos, q_home=q_home)
        # An arm may return a THIRD element: a result it already solved. FRoGGeR's
        # synthesis loop draws seeds until one yields a feasible grasp, so the
        # accepted grasp belongs to a specific attempt; re-solving from a re-drawn
        # seed does NOT reproduce it (measured: contacts 345-484 mm from the hand
        # where the accepted attempt had them on the object). Executing the
        # accepted result directly is the only faithful reading of their protocol.
        _pre_res = None
        if isinstance(_ov, tuple) and len(_ov) == 3:
            cfg, q_start, _pre_res = _ov
        else:
            cfg, q_start = _ov
    else:
        _pre_res = None
        cfg = for_gws_recommender(body_name,
                                  cfg_kw.pop("arm_geom_names"),
                                  cfg_kw.pop("obj_clearance_by_geom"),
                                  accel_budget_xyz=NCF_ACCEL_BUDGET_XYZ,
                                  ang_accel_budget_xyz=NCF_ANG_ACCEL_BUDGET,
                                  # RESOLVED per object, not the raw --fingers.
                                  # Passing `fingers` (None when the flag is
                                  # absent) let the PLANNER fall back to the
                                  # config's DEFAULT list while the executor had
                                  # already resolved per_object, so a tripod
                                  # object planned 2 contacts and execution then
                                  # died binding 3 slots to them.
                                  fingers=_fingers_for_object(object_id, fingers),
                                  # Contact separation. Defaults OFF, so an
                                  # untouched run is bit-identical to before
                                  # (verified: 014_lemon n=2 and 036_wood_block
                                  # n=3 reproduce gamma/beta/forces to every
                                  # digit). sep_hard_mode is left at its 'ball'
                                  # default and is NOT exposed here -- see the
                                  # GraspConfig3D field for the measurement that
                                  # settled it.
                                  sep_hard=bool(sep_hard),
                                  contact_min_sep_m=float(min_sep_mm) * 1e-3,
                                  **cfg_kw)
        q_start = None

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
    #
    # A plan_override arm may supply its own start pose: FRoGGeR's (7a) has no IK
    # or alignment term, so nothing in its objective prefers opposed contacts --
    # that preference lives entirely in its heuristic sampler, which is why the
    # paper describes the sampler as part of the method. Starting it from HOME
    # would handicap it for a reason unrelated to the formulation under test.
    planner = MultiStartGraspPlanner3D(model, data, cfg, log_dir=log_dir, seed=seed)
    t0 = time.time()
    if _pre_res is not None:
        res = _pre_res                      # the attempt synthesis accepted
    else:
        res = planner.solve(np.asarray(q_home if q_start is None else q_start, float),
                            np.asarray(pos, float), max_seeds=cfg.n_seeds)
    t_solve = time.time() - t0
    print(f"[plan] status={res.get('status')} rs={res.get('return_status')} "
          f"iterations={res.get('iterations')}  ({t_solve * 1e3:.0f}ms)")
    # lift_ok defaults FALSE so every abort path (plan failure, wrench-infeasible
    # contacts, squeeze aborted) reports the same field as a run that lifted and
    # dropped. An aggregate over lift_ok then needs no .get() fallback and cannot
    # silently count a missing key as anything.
    result = dict(object=object_id, seed=seed, t_solve_s=t_solve,
                  status=res.get("status"), phase_log=[], lift_ok=False)
    if res.get("q") is None or res.get("p1") is None:
        print("[plan] FAILED — no feasible grasp found.")
        result["phase_log"].append("plan_failed")
        return res, result

    # Pairwise contact separations, printed for EVERY run at n>=3. The n=3 work
    # established that a collapsed contact is invisible in every other number the
    # plan reports -- it still certifies wrench_feasible, because a doubled
    # contact is not an infeasible one -- and at n=4 three fingertips share one
    # quadratic patch, so this is the measurement that says whether the patch was
    # large enough. Also prints what the SEEDER proposed, so a collapse caused by
    # the seeder can be told apart from one caused by the NLP.
    _sol_pts = [(_k, np.asarray(res[_k], float).reshape(3))
                for _k in ("p1", "p2", "p3", "p4") if res.get(_k) is not None]
    if len(_sol_pts) >= 3:
        _sep = "  ".join(
            f"{_a[0][-1]}-{_b[0][-1]} {np.linalg.norm(_a[1] - _b[1]) * 1e3:5.1f}mm"
            for _i, _a in enumerate(_sol_pts) for _b in _sol_pts[_i + 1:])
        print(f"[contacts] solved separations: {_sep}")
        _c4t = getattr(planner, "last_c4_rank_table", None)
        if _c4t:
            _acc = next((r for r in _c4t if r.get("accepted")), _c4t[0])
            print(f"[contacts] c4 seed: {len(_c4t)} ranked, accepted "
                  f"fan={_acc.get('fan_deg'):+.0f}deg "
                  f"sep_c2={_acc.get('sep2_mm'):.1f}mm "
                  f"sep_c3={_acc.get('sep3_mm'):.1f}mm "
                  f"rf_dls={_acc.get('rf_dls_res_mm'):.1f}mm")
        elif len(_sol_pts) < 4:
            print("[contacts] c4 seed: NO viable candidate -- ran as a "
                  f"{len(_sol_pts)}-contact grasp")
        _pd = getattr(planner, "last_c4_patch_diag", None)
        if _pd:
            print(f"[contacts] c4 patch: half-extents "
                  f"{'/'.join(f'{h:.1f}' for h in _pd['half_extents_mm'])}mm  "
                  f"min={_pd['min_half_mm']:.1f}mm  fan_r={_pd['fan_r_mm']:.1f}mm  "
                  f"max_chord={_pd['max_chord_mm']:.1f}mm  "
                  f"min_sep={_pd['min_sep_mm']:.0f}mm")

    verify_info = planner._planner.verify(res)
    result["gamma_min"] = verify_info.get("gamma_min")
    result["wrench_feasible"] = verify_info.get("wrench_feasible")
    result["gws_beta"] = verify_info.get("gws_beta")
    print(f"[plan] wrench_feasible={result['wrench_feasible']} "
          f"gamma_min={result['gamma_min']}")

    q_target = np.zeros(N_ROBOT)
    q_target[:len(res["q"])] = res["q"]
    q_target[len(res["q"]):] = q_home[len(res["q"]):]

    # THIS RUN's fingers, derived from --fingers (slot order) and reversed into
    # FINGER_SET order, exactly as the teleop entry point does. Everything below --
    # tip sites/geoms, pad offsets, contact-loss bookkeeping, reported gaps -- is
    # built from _FSET so the squeeze monitors the fingers the NLP actually planned
    # for. Reading the module global here was the bug: it is resolved at import from
    # the config default and never sees --fingers.
    _SLOTS = _fingers_for_object(object_id, fingers)
    _FSET = list(reversed(_SLOTS))
    print(f"[fingers] slots={_SLOTS}  monitored order={_FSET}")
    tip_site_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
                    for f in _FSET]
    tip_geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"leap_{FINGER_CODE[f]}_tip")
                    for f in _FSET]

    data.qpos[:N_ROBOT] = q_target
    mj.mj_forward(model, data)
    p_WoO_pre = data.xpos[obj_bid].copy()
    R_WO_pre  = data.xmat[obj_bid].reshape(3, 3).copy()

    if out_dir is not None:
        # planner= also writes the PAIRED seed figure from the tables this very
        # solve recorded as it gated (last_seed_accept_table/last_seed_reject_table).
        _write_plots(model, data, res, verify_info, log_dir, object_id, seed,
                     body_name, obj_bid, pos, out_dir, n_relin=n_relin,
                     planner=planner)
    shutil.rmtree(log_dir, ignore_errors=True)

    if os.environ.get("PFF_GEOM_TRACE"):
        # WHERE the NLP put the contacts vs WHERE the fingertips ended up. The
        # n=3 failure mode is one contact thrown far off, so the question is
        # whether the NLP moved the CONTACT POINT somewhere unreachable (a
        # placement problem) or placed it sensibly and the IK failed to track it
        # (a kinematics problem). Printing both, plus the per-contact IK residual,
        # separates those.
        _pts = [res.get("p1"), res.get("p2"), res.get("p3"), res.get("p4")]
        for _slot, _p in zip(_SLOTS, _pts):
            if _p is None:
                continue
            _p = np.asarray(_p, float).reshape(3)
            _sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[_slot])
            _tip = data.site_xpos[_sid]
            # contact in the OBJECT frame too: tells us if it even lies on the object
            _pO = R_WO_pre.T @ (_p - p_WoO_pre) if R_WO_pre is not None else _p
            print(f"[geom] {_slot:6s} contact_W={np.round(_p, 4).tolist()} "
                  f"tip_W={np.round(_tip, 4).tolist()} "
                  f"|tip-contact|={np.linalg.norm(_tip - _p) * 1000:7.2f}mm "
                  f"contact_objframe={np.round(_pO, 4).tolist()}")

    if os.environ.get("PFF_ALIGN_TRACE"):
        # Angle between each fingerpad normal (site -x, see pad_axis) and the contact
        # INWARD normal at the solved pose: how flush orient_weight actually got the pad.
        for _f, _sid in zip(_FSET, tip_site_ids):
            _R = data.site_xmat[_sid].reshape(3, 3)
            _pad_W = _R @ np.array([-1.0, 0.0, 0.0])
            print(f"[align] {_f}: pad_normal_W={np.round(_pad_W, 3).tolist()}")

    p_WoO = data.xpos[obj_bid].copy()
    R_WO = data.xmat[obj_bid].reshape(3, 3).copy()
    n1_in, n2_in = recommended_inward_normals(
        model, data, planner._planner._obj_gid, planner._planner._mesh_entry,
        np.asarray(res["p1"], float), np.asarray(res["p2"], float))
    # Slot -> finger by POSITION, over HOWEVER MANY contacts this solve returned.
    # Name-keyed dicts ({"thumb": p1, "index": p2}) KeyError as soon as the run's
    # fingers differ from that pair; zipping _SLOTS against the contact list binds
    # each NLP contact to whichever finger actually serves it, at n=2 or n=3.
    # recommended_inward_normals only covers p1/p2 (it is the 2-contact helper), so
    # contact 3's inward normal is taken from the same _geom_normal_np it wraps.
    _slot_pts = [np.asarray(res["p1"], float), np.asarray(res["p2"], float)]
    _slot_nrm = [n1_in, n2_in]
    # Contacts 3 and 4 are bound the same way, in slot order. Written as a loop so
    # a fifth slot would not need a third hand-written copy of this block.
    for _key, _min_slots in (("p3", 3), ("p4", 4)):
        _p_exec = res.get(_key)
        if _p_exec is None or len(_SLOTS) < _min_slots:
            continue
        _p_exec = np.asarray(_p_exec, float).reshape(3)
        _n_out = _geom_normal_np(
            _p_exec, int(model.geom_type[planner._planner._obj_gid]),
            data.xpos[obj_bid].copy(), data.xmat[obj_bid].reshape(3, 3).copy(),
            model.geom_size[planner._planner._obj_gid].copy(),
            mesh_entry=planner._planner._mesh_entry)
        _slot_pts.append(_p_exec)
        _slot_nrm.append(-np.asarray(_n_out, float))
    # DEGRADE, don't fail, when the solve returned FEWER contacts than --fingers
    # named. This used to raise, and that was right while the only way to get
    # here was a planner/executor disagreement -- a bug. It is no longer the only
    # way: the third- and fourth-contact seeders are both documented to return
    # nothing rather than force an unreachable contact ("the tripod is an
    # upgrade, not a precondition"), so a four-finger request on an object whose
    # patch cannot separate a fourth contact legitimately yields three. Raising
    # there threw away a perfectly good tripod AND the measurement of why the
    # fourth contact was dropped -- the cells that most need to be looked at.
    #
    # Trailing slots are released: _SLOTS is positional and slot k is served by
    # the k-th contact, so dropping the TAIL is the only reduction that keeps
    # every remaining binding correct.
    if len(_slot_pts) < len(_SLOTS):
        print(f"[fingers] solve returned {len(_slot_pts)} contacts for "
              f"{len(_SLOTS)} named fingers ({','.join(_SLOTS)}) -- executing as "
              f"a {len(_slot_pts)}-contact grasp with "
              f"{','.join(_SLOTS[:len(_slot_pts)])}")
        _SLOTS = _SLOTS[:len(_slot_pts)]
        _FSET = list(reversed(_SLOTS))
        tip_site_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE,
                                      FINGER_TIP_SITES[f]) for f in _FSET]
        tip_geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM,
                                      f"leap_{FINGER_CODE[f]}_tip") for f in _FSET]
    elif len(_slot_pts) > len(_SLOTS):
        _slot_pts = _slot_pts[:len(_SLOTS)]
        _slot_nrm = _slot_nrm[:len(_SLOTS)]
    result["n_contacts_executed"] = len(_slot_pts)
    by_p = {r: v for r, v in zip(_SLOTS, _slot_pts)}
    by_n = {r: v for r, v in zip(_SLOTS, _slot_nrm)}
    rec_local = [local_contact_frame(np.asarray(by_p[f], float),
                                     np.asarray(by_n[f], float), p_WoO, R_WO)
                 for f in _FSET]
    pad_offset = {f: _pad_surface_offset(model, data, f, tip_site_ids[i], tip_geom_ids[i])
                  for i, f in enumerate(_FSET)}

    if os.environ.get("PFF_CONTACT_TRACE"):
        # The disturbance LP's feasibility is decided by the contact GEOMETRY,
        # not by the object's mass -- two near-parallel inward normals cannot
        # resist a transverse wrench at any gamma, and the LP then returns None
        # and the run then ABORTS (the GAMMA_FALLBACK substitution was removed --
        # see _solve_gamma). Dump the geometry that actually went in, so an
        # "infeasible" is diagnosable.
        _n1, _n2 = np.asarray(n1_in, float), np.asarray(n2_in, float)
        print(f"[contact] p1={np.round(res['p1'], 4).tolist()} "
              f"p2={np.round(res['p2'], 4).tolist()}")
        print(f"[contact] n1_in={np.round(_n1, 3).tolist()} "
              f"n2_in={np.round(_n2, 3).tolist()} "
              f"n1.n2={float(_n1 @ _n2):+.3f} "
              f"span={np.linalg.norm(np.asarray(res['p2'], float) - np.asarray(res['p1'], float)) * 1000:.1f}mm")

    # FERRARI-CANNY EPSILON, the quality column FRoGGeR's Table I reports next to
    # the normalized min-weight metric. Scored from `rec_local` -- the SAME contact
    # geometry the gamma certificate is solved on, in the same object frame with the
    # same inward-normal convention -- so the two numbers cannot describe different
    # grasps. At a FIXED gamma=1.0 by design: epsilon scales linearly in gamma, so
    # scoring at each run's own solved gamma would confound grasp geometry with how
    # hard that run decided to squeeze. See simulation/epsilon_metric.py.
    #
    # Expect `degenerate` at n=2: a pinch's wrench set is rank-5-of-6 and contains
    # no 6-ball. That is a structural property of two contacts, not a failure, and
    # it is why the paper's epsilon column is a four-finger number.
    try:
        _eps = epsilon_quality([p for p, _ in rec_local],
                              [R for _, R in rec_local],
                              [float(model.geom_friction[
                                  planner._planner._obj_gid, 0])] * len(rec_local),
                              gamma=1.0)
        result["epsilon"] = _eps.get("epsilon")
        result["epsilon_degenerate"] = _eps.get("degenerate")
        result["epsilon_force_sub"] = epsilon_subspace(
            [p for p, _ in rec_local], [R for _, R in rec_local],
            [float(model.geom_friction[planner._planner._obj_gid, 0])] * len(rec_local),
            gamma=1.0, which="force")
        print(f"[plan] epsilon={_fmt_eps(result['epsilon'])} "
              f"(degenerate={result['epsilon_degenerate']}, "
              f"force-subspace={_fmt_eps(result['epsilon_force_sub'])})")
    except Exception as _e:
        print(f"[plan] epsilon scoring failed: {_e}")

    gamma_live = _solve_gamma(model, data, obj_bid, R_WO, rec_local,
                              planner._planner._obj_gid, tip_geom_ids,
                              gamma_override)
    result["gamma"] = gamma_live
    if gamma_live is None and force_execute:
        # This return is UPSTREAM of the VideoRecorder's construction, so an
        # infeasible gamma produced no .mp4 at all -- the one failure mode with no
        # visual record. Substitute a nominal gamma purely so the run reaches the
        # recorder; the result still carries wrench_infeasible=True.
        print("[exec] wrench-infeasible contacts but --force-execute set: "
              "continuing at a NOMINAL gamma (result is still a failure)")
        result["wrench_infeasible"] = True
        result["gamma_substituted"] = True
        gamma_live = 1.0
        result["gamma"] = gamma_live
    if gamma_live is None:
        # Wrench-infeasible contact geometry: there is no grasp to execute. Report
        # it as a planning outcome rather than squeezing at a fabricated gamma.
        result["phase_log"] = ["gamma_infeasible_no_grasp"]
        result["wrench_infeasible"] = True
        print("[exec] ABORT before SQUEEZE -- wrench-infeasible contacts "
              "(see [contact] trace: PFF_CONTACT_TRACE=1)")
        return res, result

    Kp = np.concatenate([np.full(7, 40.0), np.full(16, finger_kp)])
    Kd = np.concatenate([np.full(7, 4.0), np.full(16, finger_kd)])
    ctrl = GraspController(
        model, N_ROBOT, tip_site_ids=tip_site_ids, obj_site_ids=None,
        obj_body_id=obj_bid, kp=Kp, kd=Kd,
        gamma=gamma_live, squeeze_pd_scale=squeeze_pd_scale, support_weight=True,
        squeeze_pd_per_finger=squeeze_pd_per_finger,
        tip_geom_ids=tip_geom_ids, obj_geom_ids=obj_gids,
        contact_gated_alloc=contact_gated_alloc,
        force_feedback_ki=force_feedback_ki,
        advance_q_target=advance_q_target,
        # FRoGGeR eq. (18)'s tracking projector; off unless asked for. See
        # GraspController.nullspace_tracking.
        nullspace_tracking=nullspace_tracking,
        pad_offsets=[pad_offset[f] for f in _FSET],
        # Cone-constrained gamma: solve null-space weights so EVERY contact is
        # compressive and in-cone, not just the sign-anchor contact. mu comes from
        # the live model (same source _solve_gamma uses) rather than a default, and
        # carries the planner's own 0.8x safety margin via cone_margin.
        cone_mu=float(model.geom_friction[planner._planner._obj_gid, 0]),
        cone_margin=0.2, cone_f_min=0.5,
        # Keep the finger PD's authority against the squeeze CONSTANT as gamma
        # moves (see GraspController.effective_gains). finger_kp/kd were tuned
        # when solved gamma was ~0.5; without this a budget change silently
        # rescales the PD/squeeze ratio, and the tuning no longer means anything.
        #
        # Defaults to 1.0 (ON) because gamma now VARIES ACROSS OBJECTS in this
        # benchmark -- measured: lemon ~1.0, orange ~1.8, tennis ball ~2.4,
        # gelatin box ~4.8, wood block clamped at 12.0. One absolute finger gain
        # cannot be right across a 12x spread; at the top of it the block's
        # squeeze overwhelmed the PD and the grasp failed on every seed. The
        # light objects sit near gamma_ref so their behaviour barely moves.
        gamma_ref=gamma_ref,
        # Finger-gain slices for THIS run's fingers. The default is hardcoded to
        # index+thumb, so a tripod's middle finger kept full stiff gains while the
        # other two were softened to close -- it never took part in the
        # CLOSING/HOLDING switch that makes the 2-contact grasp work.
        active_joint_slices=finger_joint_slices(model, _FSET),
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
    # VIEW FROM THE FAR SIDE OF THE TABLE. The robot base is mounted at
    # TS.BASE_POS=(0.0,-0.15), i.e. the -y edge, so the default azimuth 135 looks
    # from behind/beside the arm and the arm's own links occlude the fingers at
    # the moment of grasp. Looking back toward the base from +y puts the hand
    # between the camera and the arm, so the contacts stay visible.
    _cam_azim = -70
    recorder = (VideoRecorder(str(Path(out_dir) / f"seed{seed}.mp4"),
                              lookat=_cam_lookat, dist=_cam_dist,
                              azim=_cam_azim, elev=-25)
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
        # PLANNED GRASP POSE: the hand where the NLP put it, before any squeeze
        # moves the object. Same far-side camera as the video.
        #
        # FRAME THE HAND, NOT THE OBJECT ORIGIN. lookat=pos aimed at the free
        # joint's origin, which for YCB meshes sits at the object's BASE, not
        # its centroid. At the planned pose the hand is above the object, so the
        # taller the object the further the palm rode above the top of the
        # frame -- the 036_wood_block renders clipped the hand off entirely at
        # every seed while the lemon/ball (short, origin near the contact) were
        # fine. Centre on the palm/object midpoint and widen dist with the
        # vertical span so the whole hand is in frame for any object height.
        if out_dir is not None:
            try:
                _p_palm = data.xpos[palm_bid].copy()
                _mid = 0.5 * (np.asarray(pos, float) + _p_palm)
                _vspan = float(abs(_p_palm[2] - pos[2]))
                _pdist = max(0.7, 2.2 * _vspan + 0.35)
                render(model, data, str(Path(out_dir) / f"seed{seed}_planned.png"),
                       lookat=_mid, dist=_pdist, azim=_cam_azim, elev=-25)
            except Exception as _e:
                print(f"[exec] planned-pose render failed: {_e}")

        # HOLD: quasi-static settle at the planned pose.
        for _ in range(200):
            data.qvel[:N_ROBOT] = 0.0
            data.qfrc_applied[:] = ctrl.compute(data)
            mj.mj_step(model, data)
            _sync()
        result["phase_log"].append("hold_settled")
        # Diagnostic hook: a probe may be attached as pick_and_place._PROBE_DUMP
        # to inspect per-finger contact state at the phase boundaries. Absent in
        # normal runs, so this costs one attribute lookup.
        _pd = globals().get("_PROBE_DUMP")
        if _pd is not None:
            _pd("hold", model, data, tip_geom_ids, obj_gids, _FSET, ctrl)

        gaps = _tip_gaps_mm(model, data, tip_geom_ids, obj_gid, obj_geom_ids=obj_gids)
        result["tip_gaps_mm"] = dict(zip(_FSET, gaps))
        # PER-ARM gap tolerance. The 8 mm default is sized for THIS solver's r_tip
        # convention, where the IK target is the MAX tip-mesh offset and the
        # "safe" solution deliberately leaves up to ~4-5 mm for the squeeze to
        # close. A FRoGGeR-style arm parks the pad further out by construction:
        # its (7d) pins a FIXED body-frame point (site + pad_offset*pad_axis) to
        # the surface, and wherever the contact is off that axis the real geom
        # surface lies beyond it -- measured 8-11 mm on this fingertip, whose
        # off-axis angle at the solution is 11-25 deg.
        #
        # Raising the gate for that arm is NOT weakening the check: the squeeze
        # still has to close the gap and the post-lift force test still has to
        # pass, so a grasp that is merely far away still fails, just later and
        # with a measured reason instead of being refused a chance. The gate's
        # purpose -- catching an IK that converged nowhere near the object -- is
        # preserved, since the failures it was written for are hundreds of mm out.
        _gap_tol_mm = (gap_tol_m if gap_tol_m is not None
                       else CONTACT_GAP_TOL_M) * 1000
        if any(g > _gap_tol_mm for g in gaps):
            result["gap_check_failed"] = True
            if not force_execute:
                print(f"[exec] ABORT before SQUEEZE — gap too large: "
                      f"{result['tip_gaps_mm']} (tol {_gap_tol_mm:.1f} mm)")
                result["gap_tol_mm"] = _gap_tol_mm
                result["phase_log"].append("squeeze_aborted_no_contact")
                return res, result
            # --force-execute: carry on so the recorded video SHOWS the failure
            # (fingers closing on nothing, object shoved) instead of the clip
            # ending at the abort with a frozen pre-squeeze pose. The outcome is
            # still reported as a failure -- see squeeze_forces_N / lift_*.
            print(f"[exec] gap too large but --force-execute set, continuing: "
                  f"{result['tip_gaps_mm']}")
            result["phase_log"].append("gap_check_failed_forced")
        print(f"[exec] gap check OK: {result['tip_gaps_mm']}")

        # SQUEEZE: ramp the internal force in.
        ctrl.set_squeeze(True)
        n_ramp = max(int(SQUEEZE_RAMP_S / model.opt.timestep), 1)
        _f_peak = np.zeros(len(tip_geom_ids))
        _n_force_samples = 0
        _cmd_sum = np.zeros(len(tip_geom_ids))
        _mes_sum = np.zeros(len(tip_geom_ids))
        _n_track = 0
        _n_all_loaded = 0
        # PFF_DRIFT_TRACE=1 decomposes the squeeze: how big the internal-force
        # torque is against the PD torque that is supposed to hold the planned
        # posture, and how far each fingertip actually travels from where the
        # planner put it -- measured in the PALM frame, so arm motion does not
        # contaminate it. This is the diagnostic for "the tips do not stay near
        # the planned grasp": a large tau_int/tau_pd ratio means the PD has no
        # authority to resist the squeeze, and the drift is the consequence.
        _drift = os.environ.get("PFF_DRIFT_TRACE")
        if _drift:
            _R_pw0 = data.xmat[palm_bid].reshape(3, 3).T
            _p_w0  = data.xpos[palm_bid].copy()
            _tip_P0 = np.array([_R_pw0 @ (data.site_xpos[s] - _p_w0)
                                for s in tip_site_ids])
            print(f"[drift] planned tips (palm frame, mm):\n"
                  f"        {dict(zip(_FSET, np.round(_tip_P0 * 1000, 1).tolist()))}")
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
            if _drift and i % 100 == 0:
                _kp, _kd = ctrl.effective_gains()
                _t_pd  = _kp * (q_target - data.qpos[:N_ROBOT])
                _t_int = ctrl.internal_force_torques(data, scale=min(1.0, i / n_ramp))
                _R_pw = data.xmat[palm_bid].reshape(3, 3).T
                _p_w  = data.xpos[palm_bid]
                _tip_P = np.array([_R_pw @ (data.site_xpos[s] - _p_w)
                                   for s in tip_site_ids])
                _dr = np.linalg.norm(_tip_P - _tip_P0, axis=1) * 1000
                _rows = []
                for _k, (_lo, _hi) in enumerate(ctrl.active_joint_slices):
                    _rows.append(f"{_FSET[_k]}: |pd|={np.abs(_t_pd[_lo:_hi]).max():.3f} "
                                 f"|int|={np.abs(_t_int[_lo:_hi]).max():.3f} "
                                 f"drift={_dr[_k]:.1f}mm")
                _f = _measured_tip_forces(model, data, tip_geom_ids, obj_gid,
                                             obj_geom_ids=obj_gids)
                _op = data.xpos[obj_bid]
                _fc = ctrl.last_f_c_W
                print(f"[drift] i={i:4d} scale={min(1.0, i / n_ramp):.2f}  "
                      + "  ".join(_rows) + f"  fn={np.round(_f, 2).tolist()}"
                      + f"  obj={np.round(_op, 3).tolist()}"
                      + f"  |f_c|={np.round(np.linalg.norm(_fc, axis=1), 2).tolist()}")
            _f_now = _measured_tip_forces(model, data, tip_geom_ids, obj_gid,
                                          obj_geom_ids=obj_gids)
            _f_peak = np.maximum(_f_peak, _f_now)
            # FORCE-TRACKING ACCURACY: what the allocator COMMANDED vs what the
            # contacts actually delivered, sampled over the whole ramp. The
            # controller is open-loop in force -- it applies f_c and never checks
            # -- so without this a -18% error on a seated finger and a -100%
            # error on a stalled one are indistinguishable downstream.
            # Compared on the NORMAL component, which is what f_c[3k] is in the
            # contact frame and what _measured_tip_forces reports.
            _fc = getattr(ctrl, 'last_f_c', None)
            if _fc is not None and len(_fc) >= 3 * len(_f_now):
                _cmd_now = np.abs([float(_fc[3 * _k]) for _k in range(len(_f_now))])
                _cmd_sum += _cmd_now
                _mes_sum += np.asarray(_f_now, float)
                _n_track += 1
            _n_force_samples += 1
            _n_all_loaded += int(all(v > 0.0 for v in _f_now))
            if os.environ.get("PFF_SQUEEZE_TRACE") and i % 100 == 0:
                _g = _tip_gaps_mm(model, data, tip_geom_ids, obj_gid, obj_gids)
                _f = _measured_tip_forces(model, data, tip_geom_ids, obj_gid,
                                             obj_geom_ids=obj_gids)
                print(f"[squeeze] i={i:4d} scale={scale:.2f} "
                      f"gap={np.round(_g, 2).tolist()} f={np.round(_f, 2).tolist()}")
        f_meas = _measured_tip_forces(model, data, tip_geom_ids, obj_gid,
                                      obj_geom_ids=obj_gids)
        # PEAK over the whole ramp alongside the final instant. The reported
        # value was a single sample at the end of the squeeze, which cannot tell
        # "this finger never touched" from "this finger was loaded and then the
        # object shifted". Measured on the three rounded objects at n=3, the
        # zero-force finger is zero in 1007 of 1007 samples, so the distinction
        # did not change those verdicts -- but the metric should not depend on
        # that being true. squeeze_forces_peak_N is what a force-closure check
        # should read; squeeze_forces_N stays the end-of-ramp state.
        # Mean over the ramp, per finger, plus the aggregate ratio. Reported even
        # when open-loop, so every arm is comparable on the same number.
        if _n_track:
            _cmd_mean = _cmd_sum / _n_track
            _mes_mean = _mes_sum / _n_track
            result["force_cmd_mean_N"]  = dict(zip(_FSET, np.round(_cmd_mean, 3).tolist()))
            result["force_meas_mean_N"] = dict(zip(_FSET, np.round(_mes_mean, 3).tolist()))
            result["force_track_ratio"] = round(
                float(_mes_mean.sum() / max(_cmd_mean.sum(), 1e-9)), 4)
            # Worst per-finger relative error -- the aggregate can look healthy
            # while one finger delivers nothing.
            _rel = [(_mes_mean[i] - _cmd_mean[i]) / _cmd_mean[i]
                    for i in range(len(_cmd_mean)) if _cmd_mean[i] > 1e-6]
            result["force_track_worst_rel"] = round(float(min(_rel)), 4) if _rel else None
        result["squeeze_forces_peak_N"] = dict(
            zip(_FSET, np.round(_f_peak, 3).tolist()))
        result["squeeze_all_loaded_frac"] = round(float(_n_all_loaded)
                                                  / max(_n_force_samples, 1), 4)
        # _FSET, not the global: these are the MEASURED per-fingertip forces and
        # mislabelling them would report the middle finger's load under 'index'.
        result["squeeze_forces_N"] = dict(zip(_FSET, np.round(f_meas, 3).tolist()))
        result["phase_log"].append("squeeze_done")
        # Diagnostic hook: a probe may be attached as pick_and_place._PROBE_DUMP
        # to inspect per-finger contact state at the phase boundaries. Absent in
        # normal runs, so this costs one attribute lookup.
        _pd = globals().get("_PROBE_DUMP")
        if _pd is not None:
            _pd("squeeze", model, data, tip_geom_ids, obj_gids, _FSET, ctrl)
        print(f"[squeeze] final={result['squeeze_forces_N']}")

        # Holding gains for everything that follows (see effective_gains).
        ctrl.set_transporting(True)
        obj_z0 = float(data.xpos[obj_bid][2])
        q_cmd = q_target.copy()

        # LIFT: straight up, clear of the table.
        dt = model.opt.timestep
        if lift_mode == "shaky":
            # FRoGGeR's execution test (their Sec. IV), which is what produces the
            # paper's "% pick success" column. Replaces ONLY the lift jog: approach,
            # settle, gap gate and squeeze ramp above are unchanged, because that
            # sequence is measured and tuned (SOLVER_STATE sec 6) and re-deriving it
            # would inject differences unrelated to the grasp under test.
            #
            # Scored by THEIR criteria (>30 deg rotation, >7.5 cm deviation) AND by
            # this benchmark's own lift_ok below. Both are kept because they answer
            # different questions and neither subsumes the other: their test can pass
            # on a grasp that never really loaded the fingers (the object simply did
            # not move much), which is the phantom-success mode lift_ok's force gate
            # exists to catch.
            from ycb_grasp import shaky_pickup as SP
            _sh = SP.run_shaky_pickup(
                model, data, ctrl, q_cmd, _jog_to, _sync, palm_bid, obj_bid,
                tip_geom_ids=tip_geom_ids, obj_gid=obj_gid, finger_set=_FSET,
                synth_time_s=float(result.get("t_solve_s", 0.0)))
            q_cmd = _sh.pop("q_cmd", q_cmd)
            lost_lift = _sh.pop("contact_lost", {})
            result["pick_success"] = _sh["success"]
            result["pick_fail_reason"] = _sh["fail_reason"]
            result["pick_max_rot_deg"] = round(_sh["max_rot_deg"], 2)
            result["pick_max_dev_m"] = round(_sh["max_dev_m"], 4)
            print(f"[shaky] success={_sh['success']} reason={_sh['fail_reason']} "
                  f"max_rot={_sh['max_rot_deg']:.1f}deg "
                  f"max_dev={1e3 * _sh['max_dev_m']:.1f}mm")
        else:
            n_lift = _jog_steps(LIFT_DISTANCE_M, lift_speed, dt)
            q_cmd, lost_lift = _jog_to(
                model, data, ctrl, q_cmd, lambda i: np.array([0, 0, lift_speed, 0, 0, 0.]),
                n_lift, _sync, palm_bid, obj_bid, tip_geom_ids, obj_gid, label="lift", finger_set=_FSET, obj_geom_ids=obj_gids)
        result["lift_obj_dz_mm"] = (float(data.xpos[obj_bid][2]) - obj_z0) * 1000
        result["lift_contact_lost"] = lost_lift
        # SUCCESS TEST (see LIFT_OK_MIN_FRAC above). Both halves are required:
        #   rose   -- travelled at least LIFT_OK_MIN_FRAC of the COMMANDED lift.
        #             Also excludes the negative-dz case (object knocked off the
        #             table), which the phase log alone reports as "lift_done".
        #   held   -- every fingertip still carrying load AT THE FINAL STEP.
        #             Sampled fresh here rather than read from lost_lift, which
        #             is sticky: it latches on the first momentary unload and so
        #             cannot say whether the grasp RECOVERED. Both numbers are
        #             kept in the result so either reading stays available.
        _f_end = (_measured_tip_forces(model, data, tip_geom_ids, obj_gid,
                                             obj_geom_ids=obj_gids)
                  if (tip_geom_ids is not None and obj_gid is not None) else [])
        result["lift_final_forces_N"] = dict(zip(_FSET, np.round(_f_end, 4).tolist()))
        # Against the distance THIS mode actually commanded. The shaky protocol
        # lifts 10 cm (their number), not LIFT_DISTANCE_M's 12 cm, and scoring it
        # against the larger figure would fail grasps that tracked the commanded
        # trajectory perfectly.
        _cmd_lift_m = (SHAKY_LIFT_M if lift_mode == "shaky" else LIFT_DISTANCE_M)
        _rose = result["lift_obj_dz_mm"] >= LIFT_OK_MIN_FRAC * _cmd_lift_m * 1000
        _held = bool(_f_end) and all(f > LIFT_OK_MIN_FORCE_N for f in _f_end)
        result["lift_ok"] = bool(_rose and _held)
        result["phase_log"].append("lift_done")
        print(f"[lift] object rose {result['lift_obj_dz_mm']:.1f}mm  lost={lost_lift}  "
              f"final_forces={result['lift_final_forces_N']}  "
              f"lift_ok={result['lift_ok']}"
              + ("" if result["lift_ok"] else
                 f"  ({'did not rise' if not _rose else 'not held at end'})"))

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
                n_tr, _sync, palm_bid, obj_bid, tip_geom_ids, obj_gid, label="transport", finger_set=_FSET, obj_geom_ids=obj_gids)
            result["transport_contact_lost"] = lost_tr
            result["phase_log"].append("transport_done")

            # RELEASE: stop squeezing AND actively open the fingers.
            ctrl.set_squeeze(False)
            ctrl.set_transporting(False)
            # Dropping the squeeze is not the same as letting go. The PD setpoint
            # is still q_target -- the GRASP pose -- so the fingers hold their
            # closed shape and the object is released only if gravity can pull it
            # free. That works for a lemon and fails for anything the hand can
            # wedge: measured on 025_mug 2-finger, seeds 1 and 2 finish at
            # z = 0.805 / 0.810 against a bin floor of 0.630, i.e. ~180mm up,
            # still on the hand, and `release_done` is appended regardless.
            #
            # Blend the GRASPING fingers' joints toward q_home rather than adding
            # a constant offset: the joint sign that opens a finger is NOT shared
            # across fingers (measured: +0.3 rad on every grasping joint EXTENDS
            # the index, tip-to-palm 117.9 -> 122.2mm, while CURLING the thumb,
            # 148.2 -> 129.9mm), so a uniform delta closes half the hand. q_home
            # holds index and thumb at 0.0 -- the open pose -- so interpolating
            # toward it opens each finger in its own correct direction.
            # Only the fingers named by --fingers move; the arm keeps its
            # transport pose so the object is dropped where it was carried.
            if release_open_frac > 0.0:
                _q_open = np.array(q_target, float)
                _f = float(np.clip(release_open_frac, 0.0, 1.0))
                for _lo, _hi in finger_joint_slices(model, _FSET):
                    _q_open[_lo:_hi] = ((1.0 - _f) * np.asarray(q_target, float)[_lo:_hi]
                                        + _f * np.asarray(q_home, float)[_lo:_hi])
                ctrl.set_target(_q_open)
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
                # Final-state render, same far-side viewpoint as the video.
                render(model, data, str(Path(out_dir) / f"seed{seed}.png"),
                       lookat=pos, dist=0.9, azim=_cam_azim, elev=-25)
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
    # SHARED BUDGET: the same disturbance box the NLP's gamma certificate is solved
    # against (NCF_ACCEL_BUDGET_XYZ / NCF_ANG_ACCEL_BUDGET, == the teleop stack's),
    # not a second hardcoded copy. These were (0.5,0.5,0.5)/(0.1,0.1,0.1) while the
    # certificate used the cfg budget, so planning and execution sized gamma for
    # DIFFERENT tasks -- measured on 014_lemon: certificate gamma_min=1.07 at
    # 20 m/s^2 vs commanded gamma=0.13 at 0.5 m/s^2, an 8x disagreement on the same
    # contacts via the same LP. The jog's slew limiter clamps executed acceleration
    # to JOG_ACCEL_BUDGET_MPS2, which NCF_ACCEL_BUDGET_XYZ is derived from, so this
    # budget is the one actually ENFORCED during the carry.
    ACCEL = NCF_ACCEL_BUDGET_XYZ
    ANG = NCF_ANG_ACCEL_BUDGET
    g_O = R_WO.T @ model.opt.gravity
    # One friction entry per CONTACT. Taken from rec_local, which this function
    # already receives and which IS the per-run contact list -- _FSET is a local
    # of run_pick_place and is NOT in scope here (referencing it raised NameError
    # and broke the default thumb+index path), and len(FINGER_SET) is the
    # import-time default (always 2) which under-sizes this at n_contacts=3.
    mu = [float(model.geom_friction[obj_gid, 0])] * len(rec_local)
    gamma = solve_gamma_live([p for p, _ in rec_local], [R for _, R in rec_local],
                             mu, float(model.body_mass[obj_bid]), ACCEL, ANG,
                             model.body_inertia[obj_bid], grav_O=g_O)
    if gamma is None or not np.isfinite(gamma) or gamma <= 0.0:
        # NO FALLBACK. An infeasible LP means the contact geometry cannot resist
        # the disturbance box at ANY squeeze force -- substituting a constant here
        # (formerly FALLBACK=2.0) converted "no feasible grasp" into "squeeze
        # anyway at a made-up force", which is how an 80-degree-splay grasp on
        # 014_lemon (n1.n2=+0.166) still reached the squeeze phase. Return None and
        # let the caller abort.
        print("[plan] solve_gamma_live INFEASIBLE for this contact geometry "
              "-- no gamma can resist the disturbance box; aborting the grasp.")
        return None
    ceiling = _gamma_stability_ceiling(model, obj_bid, tip_geom_ids)
    if gamma > ceiling:
        print(f"[plan] clamping gamma {gamma:.2f} -> {ceiling:.2f} (stability ceiling)")
        gamma = ceiling
    else:
        print(f"[plan] gamma={gamma:.2f} (ceiling {ceiling:.2f})")
    return gamma


def _write_plots(model, data, res, verify_info, log_dir, object_id, seed,
                 body_name, obj_bid, pos, out_dir, n_relin=None, planner=None):
    """Thin shim onto kinova_common.grasp_plots.write_grasp_plots.

    The implementation moved there so the live teleop recommender can emit the
    SAME figures; this signature is preserved for the call site below.
    """
    return write_grasp_plots(model, data, res, verify_info, log_dir, object_id,
                             seed, body_name, obj_bid, pos, out_dir,
                             n_relin=n_relin, planner=planner)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", default="036_wood_block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--backend", choices=["ipopt", "sqp"], default=None,
                    help="NLP solver backend (GraspConfig3D.use_slsqp). Same cost "
                         "function and constraints either way -- only the solver "
                         "plugin changes. Default: the preset's own choice (ipopt).")
    # DEFAULT None = defer to the preset. grasp_config_builder.for_gws_recommender
    # setdefaults n_normal_relinearize=0 (a SINGLE stage, coherent with
    # quadratic_symbolic_normals supplying the normal in closed form), and an
    # explicit value here overrides that setdefault. This used to default to 3,
    # which silently made every benchmark run a 4-stage Picard solve
    # (range(n_relin+1)) rather than the single stage the preset documents --
    # and the grasp-contacts figure then captioned it "3 Picard stages", wrong
    # on both counts. Pass --n-relin explicitly to opt back into
    # relinearization.
    ap.add_argument("--n-relin", type=int, default=None,
                    help="Picard relinearization stages; the solve runs N+1 stages and "
                         "may break early on convergence. Default: the preset's own "
                         "choice (0, i.e. a single stage).")
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
    ap.add_argument("--sdf-err-tol", type=float, default=None,
                    help="metres; max surrogate-vs-true-SDF gap that sizes each "
                         "trust-region axis (default: GraspConfig3D's 5e-4)")
    ap.add_argument("--bound-inset", type=float, default=None,
                    help="metres; constant shaved off EVERY side of each trust-region "
                         "rectangle after the SDF search sizes it. Keeps contacts off "
                         "sharp edges, which the SDF-error criterion cannot see (a "
                         "planar surrogate matches a box face right up to the crease). "
                         "(default: GraspConfig3D.quadratic_bound_inset, 10mm)")
    ap.add_argument("--bound-keep-frac", type=float, default=None,
                    help="fraction of each side the inset may never consume, guaranteeing "
                         "a usable patch (default: GraspConfig3D's 0.5)")
    ap.add_argument("--impratio", type=float, default=None,
                    help="override the scene's contact impratio (scene XML sets 100; "
                         "the floor-pick benchmark measured 20 as best)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="override the solved internal-force scale")
    ap.add_argument("--squeeze-pd-scale", type=float, default=0.25,
                    help="finger PD multiplier DURING the squeeze ramp. Lower lets the "
                         "internal-force term win against the finger PD; too low and the "
                         "measured force falls short of the commanded gamma.")
    ap.add_argument("--release-open-frac", type=float, default=0.5,
                    help="how far to open the GRASPING fingers toward q_home at "
                         "release, 0..1 (0 = legacy: hold the grasp pose and let "
                         "gravity do it). Dropping the squeeze alone leaves the PD "
                         "holding the closed shape, which wedges wide objects -- "
                         "025_mug finished ~180mm above the bin floor, still on "
                         "the hand. 0.5 is measured best: it unwedges without the "
                         "extra release impulse a full open imparts (1.0 costs "
                         "014_lemon its in_bin).")
    ap.add_argument("--force-execute", action="store_true",
                    help="run the squeeze/lift even when the pre-squeeze gap check "
                         "fails or gamma is infeasible, so the FAILURE is visible in "
                         "the recorded video instead of the clip ending at the abort. "
                         "Diagnostic only -- the run is still reported as failed.")
    ap.add_argument("--contact-gated-alloc", action="store_true",
                    help="build the grasp map G only from fingers ACTUALLY in "
                         "contact, so the internal force is split over the grasp "
                         "that exists rather than the one the plan hoped for")
    ap.add_argument("--force-feedback-ki", type=float, default=0.0,
                    help="integral gain on the per-finger force error "
                         "(commanded minus measured normal). 0 = open loop, "
                         "which is what the controller has always been")
    ap.add_argument("--advance-q-target", type=float, default=0.0,
                    help="re-datum a SEATED finger's PD target to where it "
                         "actually is, so tracking stops pulling it back off the "
                         "surface it just reached (any value > 0 enables)")
    ap.add_argument("--squeeze-pd-per-finger", action="store_true",
                    help="scale the squeeze-phase finger PD PER FINGER, by that "
                         "finger's own |tau_int|/|tau_pd| ratio, instead of one "
                         "global --squeeze-pd-scale. A finger already winning "
                         "keeps full tracking authority; only one losing the "
                         "standoff is softened, and only by its shortfall")
    ap.add_argument("--sep-hard", action="store_true",
                    help="require every pair of contacts to be at least "
                         "--min-sep-mm apart, as a HARD NLP constraint. Without "
                         "it nothing keeps two contacts apart and they collapse "
                         "onto one point -- measured 15/15 cells at n=3 and 3/3 "
                         "four-contact cells at n=4, all still certifying "
                         "wrench_feasible because a doubled contact is not an "
                         "infeasible one. Can make the solve infeasible on a "
                         "patch too small to hold the contacts that far apart")
    ap.add_argument("--min-sep-mm", type=float, default=12.0,
                    help="separation floor (mm) for --sep-hard. Default 12 = one "
                         "LEAP pad extent, i.e. two pads just touching")
    ap.add_argument("--fingers", default=None,
                    help="comma-separated fingers to grasp with, IN SLOT ORDER, e.g. "
                         "'thumb,middle' or 'thumb,index,middle'. Slot 1 anchors the "
                         "grasp (the antipodal seed marches from it), so the order is "
                         "not interchangeable. Default: models/grasp_finger_config.json.")
    ap.add_argument("--contact-profile", choices=["stock", "tuned"], default="stock",
                    help="stock (default): whatever the scene XML compiles to "
                         "(impratio=100, noslip_iterations=0, fingertip "
                         "solref=[0.004,1.0]). tuned: apply contact_tuning.py's "
                         "measured settings (fingertip solref=[0.02,2.0], "
                         "noslip_iterations=5). NOTE contact_tuning pairs noslip=5 "
                         "with gamma=10.0 and squeeze_pd_scale=1.0 -- pass those "
                         "via --gamma/--squeeze-pd-scale; noslip alone removes the "
                         "tangential compliance a heavy object leans on.")
    ap.add_argument("--no-directional-r-tip", dest="directional_r_tip",
                    action="store_false",
                    help="use the ISOTROPIC fingertip radius for the IK target "
                         "(max||V-site||, ~19.5mm) instead of the support distance "
                         "along the contact normal (~9.95mm + margin). The isotropic "
                         "value is a bounding sphere around an elongated pad, so it "
                         "parks the tip ~9.5mm proud of the surface; this flag exists "
                         "to A/B that slack against the directional default.")
    ap.add_argument("--gamma-ref", type=float, default=1.0,
                    help="gamma that --finger-kp/--finger-kd were tuned at. The finger "
                         "PD gains are scaled by gamma/gamma_ref, so the PD keeps "
                         "constant authority against the squeeze as gamma changes (e.g. "
                         "after a disturbance-budget change). Pass 0 for the legacy "
                         "absolute-gain behaviour.")
    ap.add_argument("--finger-kp", type=float, default=0.8)
    ap.add_argument("--finger-kd", type=float, default=0.05)
    ap.add_argument("--lift-speed", type=float, default=LIFT_SPEED_MPS,
                    help=f"vertical lift speed m/s (default {LIFT_SPEED_MPS}). The jog "
                         "eases in/out over JOG_RAMP_S, so this is the cruise speed.")
    ap.add_argument("--transport-speed", type=float, default=TRANSPORT_SPEED_MPS,
                    help=f"lateral carry speed m/s (default {TRANSPORT_SPEED_MPS})")
    ap.add_argument("--nullspace-tracking", action="store_true",
                    help="apply FRoGGeR's eq. (18) projector to the tracking torque "
                         "while squeezing, so tracking cannot move the fingertips "
                         "across the object surface. Their Allegro has 16 hand DOFs "
                         "against a rank-12 Jh; our 2-contact LEAP has 8 against "
                         "rank 6, so the projector discards ~55%% of finger tracking "
                         "authority here (measured) -- expect a weaker hold, not "
                         "just a cleaner one.")
    ap.add_argument("--lift-mode", choices=["standard", "shaky"], default="standard",
                    help="standard (default): this benchmark's 12cm lift, scored by "
                         "lift_ok. shaky: FRoGGeR's execution test (Sec. IV) -- 10cm "
                         "in 1s, hold 1.5s, 3mm sinusoid in all axes from t+0.25s -- "
                         "additionally scored by their failure criteria (>30deg "
                         "rotation, >7.5cm deviation). This is what produces the "
                         "paper's '%% pick success' column.")
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
        fingers=args.fingers, force_execute=args.force_execute,
        sep_hard=args.sep_hard, min_sep_mm=args.min_sep_mm,
        squeeze_pd_per_finger=args.squeeze_pd_per_finger,
        contact_gated_alloc=args.contact_gated_alloc,
        force_feedback_ki=args.force_feedback_ki,
        advance_q_target=args.advance_q_target,
        release_open_frac=args.release_open_frac,
        w_edge_margin=args.w_edge_margin, mesh_fit=args.mesh_fit,
        directional_r_tip=args.directional_r_tip,
        sdf_err_tol=args.sdf_err_tol, bound_inset=args.bound_inset,
        bound_keep_frac=args.bound_keep_frac,
        backend=args.backend,
        quad_sym_normals=args.quad_sym_normals,
        seed_rank_pool=args.seed_rank_pool,
        impratio=args.impratio, gamma_override=args.gamma,
        squeeze_pd_scale=args.squeeze_pd_scale,
        # --gamma-ref 0 opts back out to absolute gains.
        gamma_ref=(args.gamma_ref or None),
        finger_kp=args.finger_kp, finger_kd=args.finger_kd,
        lift_speed=args.lift_speed, transport_speed=args.transport_speed,
        lift_mode=args.lift_mode,
        nullspace_tracking=args.nullspace_tracking,
        contact_profile=args.contact_profile)
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

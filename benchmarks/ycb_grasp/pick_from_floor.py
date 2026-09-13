"""End-to-end floor-pick test: place a YCB object resting on the floor, start the
robot from a RANDOM joint configuration, plan a grasp with MultiStartGraspPlanner3D
(the same NLP solver benchmarks/ycb_grasp/ablate_grasp.py exercises), replay the arm
to the planned config, then hand off to grasp_control.GraspController for the
internal (pinching) squeeze -- the same controller kinova_leap_pick_place.py and
grasp_controller_demo.py use, just fed the NLP's recommended contacts instead of
either authored contact sites or a teleop operator.

Unlike ik_demo.place_objects (which deliberately keeps objects FLOATING, well clear
of the floor, because the planner's own floor-clearance gating was found to fail
right at the floor), this rests the object ON the floor -- ground_clearance_m is
set explicitly on the GraspConfig3D so the planner keeps both fingertips off the
table while still finding a valid pinch. The object is placed at a computed resting
z, then SETTLED under gravity (robot frozen at home_bias) before planning, since
MuJoCo's soft contact lets it sink a further ~1-2mm once physics actually steps --
planning against the pre-settle pose left that much real gap at the fingertips.

Execution runs APPROACH (kinematic replay, random q0 -> planned q) -> HOLD (PD
settle) -> a contact-gap sanity gate (the planner's own IK target deliberately
leaves a few mm of clearance rather than touching -- see grasp_planner_3d.py's
_tip_radius comment -- so a small gap here is normal, not a failure) -> SQUEEZE
(grasp_control.GraspController's internal pinching force, ramped up, gently closing
that gap under contact dynamics) -> LIFT (a slow 100mm vertical jog via
resolved-rate DLS IK, same singularity-robust pattern kinova_leap_pick_place.py's
arrow-key jog uses, validating grasp stability by checking the object tracks the
palm's motion and neither fingertip's contact force drops to zero). Fully
automatic in both headless and --view modes.

Every run writes three artifacts into out/<object>/: the final-pose PNG, an MP4
of the whole run, and the local-quadratic contact-fit iso views.

Status: 017_orange lifts reliably (3/3 seeds tested, object tracking the palm to
within 4-7mm of a 100mm rise, contact never lost). 036_wood_block does NOT lift
on any seed tested: the grasp is simply not strong enough to carry it.

Measured, with the floor's collision DISABLED so the grasp alone bears the load
(the check to run -- an object still standing on the floor reads plausible
contact forces while the floor silently carries the weight, which is how an
earlier read of this mistook the failure for the block rotating out of the
pinch): the 0.729kg block (7.15N) falls straight through the fingers within
~0.5s at BOTH gamma=5.7 and gamma=12, i.e. at 3N and at 7N of measured normal
force. At the moment of release the TANGENTIAL force is only ~1.8-2.3N per
contact against a 7.15N weight -- roughly half what is needed -- even though
mu=1.0 and 7N of normal force should permit up to 14N. The friction-cone
utilization |ft|/(mu*fn) sits at 0.19-0.71, never saturating, so this is not
cone-limited sliding: the contacts never develop the tangential force required
before the object accelerates away. Raising gamma scales normal force cleanly
(3.3N -> 7.4N) without fixing it, and impratio 20/50/100 makes no difference.

Two things previously suspected and RULED OUT, recorded so they are not re-tried:
finger PD stiffness (sweeping Kp 0.8->20 monotonically REDUCES grip force, since
the stiffened PD fights the squeeze -- what squeeze_pd_scale exists to prevent),
and the object tipping about the grasp axis (the rotation seen in the lift trace
happens after the block is already sliding down, and the block never left the
floor during it).

    python benchmarks/ycb_grasp/pick_from_floor.py                       # default: 017_orange, seed 0
    python benchmarks/ycb_grasp/pick_from_floor.py --object 036_wood_block --seed 3
    python benchmarks/ycb_grasp/pick_from_floor.py --view                # interactive viewer
    python benchmarks/ycb_grasp/pick_from_floor.py --no-lift             # skip the lift-jog check
    # the settings the orange results above were measured with:
    python benchmarks/ycb_grasp/pick_from_floor.py --object 017_orange --seed 4 \
        --n-seeds 3 --gws --n-relin 3 --w-edge-curvature 100 --directional-r-tip
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import mujoco as mj
import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from grasp_control import GraspController                                       # noqa: E402
from grasp_control import object_uv_atlas as oua                                # noqa: E402
from kinova_common.constants import FINGER_CODE, FINGER_SET, FINGER_TIP_SITES   # noqa: E402
from kinova_common.wrench import solve_gamma_live                               # noqa: E402
from kinova_common.video import H264Writer                                      # noqa: E402
from simulation.grasp_config_builder import for_ablation_default                # noqa: E402
from simulation.grasp_planner_3d import MultiStartGraspPlanner3D, _geom_normal_np  # noqa: E402
from ycb_grasp import out_paths as OP                                           # noqa: E402
from ycb_grasp import plot_quadratic_path as QP                                 # noqa: E402
from ycb_grasp import scene as S, workspace as W                                # noqa: E402
from ycb_grasp.ik_demo import (clearance_by_geom, home_bias, object_hull_verts,  # noqa: E402
                               quat_mat, render, robot_geom_names)
from ycb_grasp.workspace import _arm_bounds                                     # noqa: E402

N_ROBOT = 23
DEFAULT_COL_CLEARANCE_M = 0.002
FLOOR_Z = 0.0
REST_MARGIN_M = 0.001     # lowest hull vertex sits this far above the floor
APPROACH_STEPS = 300      # kinematic replay resolution, random q0 -> planned q
SQUEEZE_RAMP_S = 0.5      # internal_force_torques' scale ramp 0->1 (see grasp_controller.py)
CONTACT_GAP_TOL_M = 0.008  # gap the squeeze is expected to gently close on its own --
                           # see grasp_planner_3d.py's own r_tip sizing comment: the IK
                           # target is DELIBERATELY the MAX (not mean) tip-mesh offset, so
                           # the "safe" solution leaves a real gap up to ~4-5mm rather than
                           # touching (mean offset would instead have ~half the contacts
                           # PENETRATING). Squeezing across a gap this small is the intended
                           # design, not a bug -- this tolerance is a hard-failure GATE for
                           # a much larger, genuinely-broken gap (e.g. IK didn't converge
                           # near the object at all), not a "must already be touching" check.

# LIFT jog: same singularity-robust resolved-rate DLS pattern as
# kinova_leap_pick_place.py's arrow-key palm jog (JOG_SING_EPS/JOG_LAM_MAX
# match that file's tuned values exactly).
LIFT_DISTANCE_M = 0.10
LIFT_SPEED_MPS = 0.02      # slow jog -- "validate stability," not a fast lift
JOG_SING_EPS = 0.02        # rad*m onset of DLS damping near a Jacobian singularity
JOG_LAM_MAX = 0.05         # peak damping at the singularity

VIDEO_FPS = 30
VIDEO_W, VIDEO_H = 960, 720
PRINT_EVERY = 250


def random_arm_q0(model, rng, obj_body_id, obj_clearance_m=0.05, max_tries=500):
    """A random FULL-RANGE arm configuration (hand at a light curl, matching
    home_bias's finger posture) that is collision-free against itself, the
    floor, and the placed object -- rejection sampling on mj_forward's own
    collision pass (data.ncon / data.contact after mj_forward already reflects
    narrow-phase collisions; no separate mj_collision call needed).

    obj_clearance_m is deliberately generous (5cm): this only needs to avoid
    the arm spawning INSIDE the object, not to approach it -- the grasp plan
    and approach replay handle actually reaching the object.
    """
    lo, hi = _arm_bounds(model)
    hand_bias = home_bias()[7:]
    data = mj.MjData(model)
    robot_prefixes = ("leap_", "bracelet_link", "half_arm", "forearm",
                      "spherical_wrist", "shoulder_link", "base_link")

    def _robot_geom(gi):
        bn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ""
        return bn.startswith(robot_prefixes)

    for _ in range(max_tries):
        q0 = np.zeros(N_ROBOT)
        q0[:7] = rng.uniform(lo, hi)
        q0[7:] = hand_bias
        data.qpos[:N_ROBOT] = q0
        mj.mj_forward(model, data)
        # Reject only contacts that INVOLVE the robot (self-collision, or
        # robot-vs-object/floor) -- the object resting on the floor is a
        # real, permanent contact after settle_object_on_floor and must not
        # itself count as a rejection (an "any contact" check silently
        # rejected every single sample once the object was actually
        # touching the floor, since ncon was never 0 again).
        if any(_robot_geom(data.contact[i].geom1) or _robot_geom(data.contact[i].geom2)
              for i in range(data.ncon)):
            continue
        # Keep every robot geom comfortably clear of the object's own body
        # origin as a cheap proxy (avoids spawning the arm through the
        # object without needing a full geom-vs-geom distance pass here).
        obj_p = data.xpos[obj_body_id]
        clear = True
        for gi in range(model.ngeom):
            if model.geom_contype[gi] == 0 or not _robot_geom(gi):
                continue
            if np.linalg.norm(data.geom_xpos[gi] - obj_p) < obj_clearance_m:
                clear = False
                break
        if clear:
            return q0
    raise RuntimeError(f"no collision-free random arm config found in {max_tries} tries")


def place_object_on_floor(obj_id, ws, rng, x_min=0.30, r_max=0.72, tries=4000):
    """Sample a resting pose for obj_id: reachable xy (from the palm workspace
    cloud, ignoring its z entirely -- floor placement doesn't need the palm
    envelope's height band), random yaw-only orientation (a full random quat
    would rest most YCB meshes on an arbitrary, often-unstable face; yaw-only
    keeps the object "upright" the way it would actually sit after being set
    down, and matches how the YCB meshes are authored -- roughly canonical
    up-axis), z solved from the hull's own lowest vertex so the object starts
    just above the floor -- close enough that settle_object_on_floor's short
    gravity-settle (called by the caller right after scene.build) converges
    quickly to the object's true resting pose.

    Returns (pos(3), quat(4))."""
    V = object_hull_verts(obj_id)
    for _ in range(tries):
        xy = W.sample_positions(ws, 1, rng, margin=0.01)[0][:2]
        if xy[0] < x_min or np.linalg.norm(xy) > r_max:
            continue
        yaw = rng.uniform(-np.pi, np.pi)
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        z = FLOOR_Z - (V @ quat_mat(quat).T)[:, 2].min() + REST_MARGIN_M
        return np.array([xy[0], xy[1], z]), quat
    raise RuntimeError(f"no valid floor placement for {obj_id} in {tries} tries")


def _pad_surface_offset(model, data, finger, tip_site_id, tip_geom_id):
    """Distance from the fingertip SITE (tip-mesh centroid) to the fingerpad
    SURFACE along the pad normal (-x of the site frame) -- same derivation as
    kinova_leap_pick_place.py's _pad_surface_offset, needed so the squeeze's
    slip-correction anchor sits on the pad surface, not the mesh centroid."""
    mid = model.geom_dataid[tip_geom_id]
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts_W = (data.geom_xmat[tip_geom_id].reshape(3, 3) @ model.mesh_vert[adr:adr + num].T).T \
              + data.geom_xpos[tip_geom_id]
    pad_dir_W = -data.site_xmat[tip_site_id].reshape(3, 3)[:, 0]
    return float(np.max((verts_W - data.site_xpos[tip_site_id]) @ pad_dir_W))


def recommended_inward_normals(model, data, obj_gid, mesh_entry, p1, p2):
    """Outward->inward surface normals at p1 (thumb)/p2 (index) for the
    object's live pose, via the planner's own shape-aware _geom_normal_np --
    same call verify() makes internally (see grasp_planner_3d.py's verify()),
    exposed here for the controller's contact-frame provider."""
    gtype = int(model.geom_type[obj_gid])
    # MESH pose comes from the BODY, not the collision hull: object_sdf's table is
    # BODY-frame, and a CoACD hull's origin is offset from the body origin (measured
    # [-10.8, +21.0, +26.2] mm on 014_lemon). Feeding the hull frame evaluates the SDF
    # ~35mm from the true point and returns normals off the wrong part of the surface:
    # the same committed contacts read n1.n2 = +0.368 (splayed, LP-infeasible) in the
    # hull frame vs -0.801 (a true pinch, gamma_min 1.07) in the body frame.
    # Matches the convention already used by grasp_planner_3d.verify(),
    # MultiStartGraspPlanner3D.solve, plot_seed_quadratic and ablate_grasp.
    if mesh_entry is not None:
        c = data.xpos[model.geom_bodyid[obj_gid]].copy()
        R = data.xmat[model.geom_bodyid[obj_gid]].reshape(3, 3).copy()
    else:
        c = data.geom_xpos[obj_gid].copy()
        R = data.geom_xmat[obj_gid].reshape(3, 3).copy()
    size = model.geom_size[obj_gid]
    n1_out = _geom_normal_np(p1, gtype, c, R, size, mesh_entry=mesh_entry)
    n2_out = _geom_normal_np(p2, gtype, c, R, size, mesh_entry=mesh_entry)
    return -n1_out, -n2_out


def local_contact_frame(p_W, n_in_W, p_WoO, R_WO):
    """Object-LOCAL (p_O, R_O) for one contact -- col0 of R_O is the inward
    normal, matching the site convention grasp_control.GraspController
    expects from obj_contact_provider. Tangent axes are arbitrary (only col0
    is used, by the grasp map + slip-correction anchor)."""
    n = n_in_W / (np.linalg.norm(n_in_W) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref); t1 /= (np.linalg.norm(t1) + 1e-12)
    t2 = np.cross(n, t1)
    R_W = np.column_stack([n, t1, t2])
    p_O = R_WO.T @ (p_W - p_WoO)
    R_O = R_WO.T @ R_W
    return p_O, R_O


def make_object_contact_provider(rec_local, obj_body_id):
    """Live (p_W, R_W_inward) provider tracking the object body's CURRENT
    pose from the stored object-local frames -- same pattern as
    kinova_leap_pick_place.py's _make_provider, needed because the NLP's
    recommended contacts have no MuJoCo sites of their own."""
    def _provider(data):
        p_WoO = data.xpos[obj_body_id]
        R_WO = data.xmat[obj_body_id].reshape(3, 3)
        return [(p_WoO + R_WO @ p_O, R_WO @ R_O) for (p_O, R_O) in rec_local]
    return _provider


class VideoRecorder:
    """Fixed external camera -> MP4, one frame per call to capture(). Uses the
    same offscreen mj.Renderer + camera convention as ik_demo.render/
    plot_quadratic_path._render_hand_rgb (not the interactive viewer, which
    can't be captured this way), so the video's framing matches the repo's
    existing static renders. The writer encodes BGR; MuJoCo's Renderer returns
    RGB, hence the channel-swap in capture(). H264Writer emits H.264 rather than
    cv2's mp4v so the clips preview in browsers and desktop viewers."""
    def __init__(self, path, lookat, dist=0.6, azim=135, elev=-55,
                w=VIDEO_W, h=VIDEO_H, fps=VIDEO_FPS, groups=(0, 1, 2, 5)):
        self.path = str(path)
        self.w, self.h = w, h
        self._writer = H264Writer(self.path, fps, (w, h))
        self._opt = mj.MjvOption()
        mj.mjv_defaultOption(self._opt)
        for g in range(6):
            self._opt.geomgroup[g] = 1 if g in groups else 0
        self._cam = mj.MjvCamera()
        mj.mjv_defaultCamera(self._cam)
        self._cam.lookat[:] = lookat
        self._cam.distance, self._cam.azimuth, self._cam.elevation = dist, azim, elev
        self._renderer = None
        self._model = None

    def capture(self, model, data):
        if self._renderer is None or self._model is not model:
            self._renderer = mj.Renderer(model, self.h, self.w)
            self._model = model
        self._renderer.update_scene(data, camera=self._cam, scene_option=self._opt)
        rgb = self._renderer.render()
        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        self._writer.release()
        if self._renderer is not None:
            del self._renderer
            self._renderer = None


def _measured_tip_forces(model, data, tip_geom_ids, obj_geom_id):
    totals = [0.0] * len(tip_geom_ids)
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for k, tg in enumerate(tip_geom_ids):
            if {c.geom1, c.geom2} == {tg, obj_geom_id}:
                mj.mj_contactForce(model, data, i, f6)
                totals[k] += abs(f6[0])
    return totals


def _run_lift_jog(model, data, ctrl, obj_bid, tip_geom_ids, obj_gid, q_target, _sync,
                  palm_body="leap_palm", distance_m=LIFT_DISTANCE_M, speed_mps=LIFT_SPEED_MPS):
    """Slowly jog the palm straight up by distance_m via resolved-rate DLS IK
    (orientation held, world-frame +z linear velocity only) -- the SAME
    singularity-robust Jacobian-transpose pattern kinova_leap_pick_place.py's
    arrow-key jog uses (JOG_SING_EPS/JOG_LAM_MAX match that file exactly),
    with the squeeze controller still active throughout. This is the grasp's
    stability validation: does the object track the palm's motion (a good
    grasp) or slip/stay behind (a bad one)?

    Returns a dict with per-step-sampled diagnostics: whether each finger's
    contact force EVER dropped to zero during the jog (a dropped grasp),
    the final object-to-palm-motion tracking error, and the achieved lift
    height.
    """
    # Switch the finger joints from CLOSING gains to HOLDING gains: the squeeze
    # has converged and the grasp is about to bear the object's weight, which
    # back-drives the softened joints and bleeds normal force away (measured on
    # 036_wood_block seed 2: fn 8.20 -> 6.71N over ~200ms, object falling while
    # both contacts were still present). See GraspController.effective_gains.
    ctrl.set_transporting(True)

    palm_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, palm_body)
    n = model.nv
    dt = model.opt.timestep
    n_steps = max(int(distance_m / speed_mps / dt), 1)
    v6 = np.array([0.0, 0.0, speed_mps, 0.0, 0.0, 0.0])

    palm_z0 = float(data.xpos[palm_bid][2])
    obj_z0 = float(data.xpos[obj_bid][2])
    q_lift = q_target.copy()
    contact_ever_lost = {f: False for f in FINGER_SET}

    print(f"[exec] -> LIFT ({distance_m * 1000:.0f}mm @ {speed_mps * 1000:.0f}mm/s, "
         f"{n_steps} steps)")
    for i in range(n_steps):
        Jp = np.zeros((3, n))
        Jr = np.zeros((3, n))
        mj.mj_jacBody(model, data, Jp, Jr, palm_bid)
        J6 = np.vstack([Jp[:, :7], Jr[:, :7]])
        sigma_min = np.linalg.svd(J6, compute_uv=False)[-1]
        lam2 = (0.0 if sigma_min >= JOG_SING_EPS
               else (1.0 - (sigma_min / JOG_SING_EPS) ** 2) * JOG_LAM_MAX ** 2)
        qdot_jog = J6.T @ np.linalg.solve(J6 @ J6.T + lam2 * np.eye(6), v6)
        q_lift[:7] += qdot_jog * dt

        # Feedforward the commanded joint rate into qvel (not zeroed) so the
        # PD tracks a MOVING target smoothly -- zeroing qvel here (the
        # quasi-static convention every other phase uses) would need a huge
        # position error each step to produce any motion at all, since qvel
        # is rebuilt from a single qacc step and immediately discarded.
        data.qvel[:N_ROBOT] = 0.0
        data.qvel[:7] = qdot_jog
        kp, kd = ctrl.effective_gains()
        tau = np.zeros(n)
        tau[:N_ROBOT] = (kp * (q_lift - data.qpos[:N_ROBOT])
                        + kd * (np.r_[qdot_jog, np.zeros(N_ROBOT - 7)] - data.qvel[:N_ROBOT])
                        + data.qfrc_bias[:N_ROBOT]
                        + ctrl.internal_force_torques(data, scale=1.0))
        data.qfrc_applied[:] = tau
        mj.mj_step(model, data)
        _sync()

        f_meas = _measured_tip_forces(model, data, tip_geom_ids, obj_gid)
        for f, force in zip(FINGER_SET, f_meas):
            if force <= 1e-6:
                contact_ever_lost[f] = True
        if i % PRINT_EVERY == 0:
            palm_dz = float(data.xpos[palm_bid][2]) - palm_z0
            obj_dz = float(data.xpos[obj_bid][2]) - obj_z0
            print(f"[lift] t={i * dt:.2f}s palm_dz={palm_dz * 1000:.1f}mm "
                 f"obj_dz={obj_dz * 1000:.1f}mm  f_meas={dict(zip(FINGER_SET, np.round(f_meas, 2)))}")

    palm_dz_final = float(data.xpos[palm_bid][2]) - palm_z0
    obj_dz_final = float(data.xpos[obj_bid][2]) - obj_z0
    tracking_err_mm = abs(palm_dz_final - obj_dz_final) * 1000
    print(f"[lift] done: palm moved {palm_dz_final * 1000:.1f}mm, object moved "
         f"{obj_dz_final * 1000:.1f}mm (tracking error {tracking_err_mm:.1f}mm), "
         f"contact_ever_lost={contact_ever_lost}")
    return dict(
        lift_palm_dz_mm=palm_dz_final * 1000,
        lift_obj_dz_mm=obj_dz_final * 1000,
        lift_tracking_err_mm=tracking_err_mm,
        lift_contact_ever_lost=contact_ever_lost,
    )


def settle_object_on_floor(model, data, obj_bid, n_steps=400):
    """Step physics with the robot frozen at home_bias() until the object
    stops moving, then return its TRUE settled (pos, quat).

    place_object_on_floor computes z from the hull's raw lowest vertex, but
    MuJoCo's contact is soft: even an "exact" placement sinks ~1-2mm further
    once physics actually steps (confirmed empirically -- this sink alone
    was enough to reopen a multi-mm gap at a fingertip planned against the
    pre-settle pose, since the grasp was never replanned against where the
    object actually ends up). Planning against the SETTLED pose instead
    removes that gap at the source rather than compensating for it later.
    """
    q_home = home_bias()
    for _ in range(n_steps):
        data.qpos[:N_ROBOT] = q_home
        data.qvel[:N_ROBOT] = 0.0
        mj.mj_step(model, data)
    mj.mj_forward(model, data)
    adr = model.jnt_qposadr[model.body_jntadr[obj_bid]]
    pos = data.qpos[adr:adr + 3].copy()
    quat = data.qpos[adr + 3:adr + 7].copy()
    return pos, quat


def _tip_gaps_mm(model, data, tip_geom_ids, obj_geom_id, obj_geom_ids=None):
    """True fingertip-geom-to-object gap (mm) via mj_geomDistance, one per
    finger (tip_geom_ids order) -- the mechanical reality check the NLP's
    site-based IK cost doesn't itself guarantee: a mesh object's IK residual
    can converge to a few mm at the fingertip SITE while the actual tip MESH
    (a different, offset piece of geometry) still sits a real gap away from
    the object's collision hull. Squeezing a finger that isn't actually
    touching sends the grasp map an unbalanced (non-antipodal) force, which
    the arm PD cannot resist -- confirmed empirically (an 8.9mm untouched
    index-finger gap here launched the object ~700mm on one seed's squeeze).

    obj_geom_ids : all of the object's collision hulls, when it has more than
        one. A concave YCB object is a V-HACD decomposition (065-a_cups: 35
        hulls), and the distance to any SINGLE hull is not the distance to the
        object -- a fingertip touching the cup's rim can be tens of mm from
        whichever hull happens to be named <body>_geom, which would abort the
        squeeze on a perfectly good grasp. The true gap is the MINIMUM over
        hulls. Defaults to [obj_geom_id] (the single-hull objects behave exactly
        as before).
    """
    gids = list(obj_geom_ids) if obj_geom_ids else [obj_geom_id]
    return [min(mj.mj_geomDistance(model, data, tg, og, 0.1, None) for og in gids) * 1000
            for tg in tip_geom_ids]


# Internal force (N) validated as stable at the reference contact softness
# below.
#
# This was 2.0 while the scene ran MuJoCo's default pyramidal cone at
# impratio=1, where gamma=5 and 8 diverged within the squeeze ramp. That
# instability was an artifact of the soft TANGENTIAL constraints, not a real
# limit: with scene.build now setting cone=elliptic / impratio=10 (see its
# comment), a re-measured sweep on 036_wood_block held cleanly at gamma =
# 2/4/6/8/12 N -- object drift 0.1-0.6mm throughout, with measured contact force
# tracking commanded gamma linearly (0.95 N -> 6.03 N) rather than blowing up.
# 12 N was the largest tested; the ceiling is set there rather than higher
# because nothing above it has been validated.
GAMMA_STABILITY_REF_N = 12.0
# Contact time constant (s) the reference value was measured at -- the softer
# side of the tip/object pair in that scene (the object's 0.02s).
GAMMA_STABILITY_REF_TAU = 0.02


def _gamma_stability_ceiling(model, obj_body_id, tip_geom_ids):
    """Largest internal-force scale (N) safe to command in THIS scene.

    solve_gamma_live answers a STATICS question -- the internal force needed for
    force closure under a disturbance budget -- and knows nothing about the
    simulator. Commanding its answer directly is what launches the object; the
    grasp is "feasible" and the integrator still cannot hold it.

    What actually sets the limit is the contact solver, not the grasp. MuJoCo's
    soft contact is a spring-damper of time constant tau = solref[0], and the
    explicit integrator resolves it only while the contact force stays small
    relative to what that spring can absorb per step. Stiffness goes as 1/tau^2,
    so the tolerable force scales the same way: halving tau (a stiffer contact)
    quarters the headroom.

    Deliberately NOT scaled by object mass. The obvious derivation (force f on
    mass m moves the contact f/m * tau^2, so f_max ~ m*dt/tau^2) predicts a 15x
    lower ceiling for the orange than the block -- 0.12 N, which measurement
    contradicts outright: gamma=2 grips the orange stably. The ceiling is a
    property of the contact model and timestep, which both objects share, so the
    mass term is spurious and was dropped rather than kept as a plausible-looking
    formula that the data does not support.

    tau is the SOFTER (larger solref[0]) of the object and fingertip pair, since
    that side dominates the combined contact's response timescale.
    """
    obj_gids = [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == obj_body_id]
    tau_obj = max(float(model.geom_solref[g][0]) for g in obj_gids)
    tau_tip = max(float(model.geom_solref[g][0]) for g in tip_geom_ids)
    tau = max(tau_obj, tau_tip)          # softer contact governs
    # Stiffness ~ 1/tau^2, and the timestep sets how much of it a step resolves.
    dt_ref = 0.002                       # timestep the reference value was measured at
    scale = (tau / GAMMA_STABILITY_REF_TAU) ** 2 * (float(model.opt.timestep) / dt_ref)
    return GAMMA_STABILITY_REF_N * scale


def run_pick(object_id, seed, n_seeds=1, n_relin=None, gws=False, w_gws=5.0,
            w_span=1.0, use_quadratic=True, view=False, render_path=None,
            video_path=None, quad_plot_path=None, do_lift=True, max_iter=200,
            w_edge_margin=0.0, directional_r_tip=False,
            directional_r_tip_margin_m=0.001):
    rng = np.random.default_rng(seed)

    base = mj.MjModel.from_xml_path(str(REPO / "models" / "scene_kinova_leap.xml"))
    ws = W.load_or_build(base, n=200_000, seed=0)

    pos, quat = place_object_on_floor(object_id, ws, rng)
    model, data, info = S.build([(object_id, pos, quat)])
    body_name = next(iter(info))
    obj_bid = info[body_name]["bid"]
    mj.mj_forward(model, data)

    # Settle under gravity BEFORE planning -- the computed placement sinks a
    # further ~1-2mm once physics actually steps (soft contact), which is
    # enough to reopen a real fingertip-to-object gap if the grasp is
    # planned against the pre-settle pose instead. Re-read the object's
    # TRUE resting (pos, quat) and plan against that.
    pos, quat = settle_object_on_floor(model, data, obj_bid)
    # Object's compiled qpos0 (what mj_resetData later restores) must match
    # the settled pose too, or resetting for the random-q0 start would put
    # the object back at the pre-settle spawn height.
    adr = model.jnt_qposadr[model.body_jntadr[obj_bid]]
    model.qpos0[adr:adr + 3] = pos
    model.qpos0[adr + 3:adr + 7] = quat

    print(f"[pick] {object_id}  seed={seed}  settled pos={np.round(pos, 3).tolist()}  "
         f"quat={np.round(quat, 3).tolist()}")

    q0 = random_arm_q0(model, rng, obj_bid)
    print(f"[pick] random arm q0 (deg) = {np.round(np.degrees(q0[:7]), 1).tolist()}")

    rgeoms = robot_geom_names(model)
    obj_clr = clearance_by_geom(rgeoms)
    obj_geom0 = S.hull_geoms(model, body_name)[0]

    cfg_kw = dict(n_seeds=n_seeds, max_iter=max_iter, arm_geom_names=rgeoms,
                 obj_clearance_by_geom=obj_clr, col_clearance_m=DEFAULT_COL_CLEARANCE_M,
                 use_quadratic_contact=use_quadratic,
                 w_edge_margin=w_edge_margin,
                 directional_r_tip=directional_r_tip,
                 directional_r_tip_margin_m=directional_r_tip_margin_m,
                 # Grasp-axis + fingerpad alignment, matching the production
                 # preset (GraspConfigBuilder.for_teleop_recommender). Without
                 # these BOTH default to 0.0, and nothing in the cost asks the
                 # two contact normals to OPPOSE each other -- the solver is free
                 # to return a pair whose normals are near-perpendicular, which
                 # it routinely did (measured n1.n2 of +0.128, -0.608, -0.938
                 # across three wood-block seeds where -1.0 is a true pinch).
                 # That is not a small quality loss, it is the difference between
                 # a liftable grasp and an impossible one: gamma required to hold
                 # the block goes 6.9N at n1.n2=-1.0, 12.7N at -0.966, 203N at
                 # -0.866, and INFEASIBLE at any finite force past about -0.7.
                 # orient_weight matters as much as w_align here -- w_align alone
                 # measured WORSE than nothing on 2 of 3 seeds (it aligns the
                 # grasp AXIS, but the pad can still meet the surface obliquely);
                 # the pair together is what produced feasible grasps on all
                 # three. Higher is not better: w_align=30 regressed them again.
                 w_align=10.0, orient_weight=2.0,
                 ground_clearance_m=0.006)   # keep fingertips off the table explicitly
    if n_relin is not None:
        cfg_kw["n_normal_relinearize"] = n_relin
    if gws:
        cfg_kw["wrench_constraint"] = False
        cfg_kw["w_gws"] = w_gws
        cfg_kw["w_span"] = w_span
    cfg = for_ablation_default(obj_geom=obj_geom0, obj_body=body_name, **cfg_kw)

    # log_dir wired only when a quadratic-path plot was actually requested --
    # grasp3d_iter_*.npz per-stage recording is otherwise dead weight.
    log_dir = None
    if quad_plot_path is not None:
        log_dir = str(Path(quad_plot_path).parent / f"_quad_log_{object_id}_seed{seed}")
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    planner = MultiStartGraspPlanner3D(model, data, cfg, log_dir=log_dir)
    q_ref = home_bias()   # NLP regularization target -- not the sim's start state
    t0 = time.time()
    res = planner.solve(q_ref, np.asarray(pos, float), max_seeds=n_seeds)
    print(f"[plan] status={res.get('status')} rs={res.get('return_status')} "
         f"iterations={res.get('iterations')}  ({(time.time() - t0) * 1e3:.0f}ms)")
    if res.get("q") is None or res.get("p1") is None:
        print("[plan] FAILED — no feasible grasp found; nothing to execute.")
        return res, None

    verify_info = planner._planner.verify(res)
    print(f"[plan] wrench_feasible={verify_info.get('wrench_feasible')} "
         f"gamma_min={verify_info.get('gamma_min')}")

    q_target = np.zeros(N_ROBOT)
    q_target[:len(res["q"])] = res["q"]
    # Unsolved DOFs (e.g. non-grasping fingers) stay at the home bias posture.
    q_target[len(res["q"]):] = q_ref[len(res["q"]):]

    tip_site_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
                    for f in FINGER_SET]
    tip_geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"leap_{FINGER_CODE[f]}_tip")
                    for f in FINGER_SET]

    data.qpos[:N_ROBOT] = q_target
    mj.mj_forward(model, data)

    # Quad-plot generated HERE (planned pose already forwarded into `data`,
    # verify_info already computed) so it can carry the real wrench summary
    # and hand render instead of "unavailable" placeholders -- data gets
    # mj_resetData'd below anyway before the random-q0 execution replay, so
    # this temporary use of the planned pose doesn't leak into execution.
    if quad_plot_path is not None:
        try:
            stages = QP._iter_trace_quadratic_stages(log_dir, res=res)
            if os.environ.get("PFF_DEBUG_STAGES"):
                for si, stage in enumerate(stages):
                    for ci, frame in stage["contact"].items():
                        print(f"[debug] stage {si+1} contact {ci}: "
                             f"t_bound=({frame['t_bound_0']:.5f}, {frame['t_bound_1']:.5f}) "
                             f"kappa=({frame['kappa0']:.2f}, {frame['kappa1']:.2f}) "
                             f"seed_l={np.round(frame['seed_l'], 4).tolist()} "
                             f"n_l={np.round(frame['n_l'], 4).tolist()}")
            if stages and any(stage["contact"] for stage in stages):
                bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
                V, F = oua.body_visual_mesh(model, bid)
                hand_rgb = QP._render_hand_rgb(model, data, lookat=pos, dist=0.45, elev=-55)
                QP.plot_quadratic_path(V, F, stages, object_id, Path(quad_plot_path),
                                       hand_rgb=hand_rgb, verify_info=verify_info)
                print(f"[plan] quadratic contact path -> {quad_plot_path} "
                     f"({len(stages)} Picard stage(s))")
            else:
                print("[plan] no quad_* frame data in the trace -- "
                     "use_quadratic_contact must be True for this plot")
        except Exception as e:
            print(f"[plan] quadratic-path plot failed: {e}")
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    p_WoO = data.xpos[obj_bid].copy()
    R_WO = data.xmat[obj_bid].reshape(3, 3).copy()
    n1_in, n2_in = recommended_inward_normals(
        model, data, planner._planner._obj_gid, planner._planner._mesh_entry,
        np.asarray(res["p1"], float), np.asarray(res["p2"], float))
    by_finger_p = {"thumb": res["p1"], "index": res["p2"]}
    by_finger_n = {"thumb": n1_in, "index": n2_in}
    rec_local = [local_contact_frame(np.asarray(by_finger_p[f], float),
                                     np.asarray(by_finger_n[f], float), p_WoO, R_WO)
                for f in FINGER_SET]

    pad_offset = {f: _pad_surface_offset(model, data, f, tip_site_ids[i], tip_geom_ids[i])
                 for i, f in enumerate(FINGER_SET)}

    # Minimum internal-force scale for a modest hold/lift disturbance budget.
    #
    # DATUM / "Task B" mode (grav_O passed): the disturbance is referenced about
    # the GRASP MIDPOINT and gravity's grasp-axis moment is projected out. This
    # is the hold/transport formulation, which is what a lift actually is.
    #
    # The CoM / "Task A" mode used previously here is the wrong question for this
    # task, and demonstrably so: it reports INFEASIBLE even for a perfect
    # antipodal pinch straight through the object's CoM with zero disturbance
    # budget, because a 2-contact pinch cannot resist a lateral force box
    # referenced about the CoM (a pinch is structurally rank-deficient about its
    # own grasp axis -- see build_W_ca's docstring). Every wood-block grasp
    # returned None from it and silently fell back to GAMMA_FALLBACK=2.0, which
    # is roughly a THIRD of the force actually needed -- hence a hand that lifts
    # away while the block stays on the floor.
    #
    # In datum mode the answer is physically interpretable: at zero disturbance
    # it converges to weight/(2*mu) -- measured 5.96N against a predicted 5.96N
    # for the 0.729kg block at mu=0.6 -- i.e. exactly the friction needed for two
    # fingers to carry the object's weight. The 6-19N figures this returns were
    # never anomalous; they are what holding this block costs.
    ACCEL_BUDGET_XYZ = (0.5, 0.5, 0.5)      # m/s^2, PURE accel budget (gravity is
                                            # passed separately in datum mode, NOT
                                            # folded in here)
    ANG_ACCEL_BUDGET = (0.1, 0.1, 0.1)      # rad/s^2, principal-frame angular budget
    GAMMA_FALLBACK = 2.0
    obj_mass = float(model.body_mass[obj_bid])
    obj_inertia = model.body_inertia[obj_bid]
    g_O = R_WO.T @ model.opt.gravity
    mu_c = [float(model.geom_friction[planner._planner._obj_gid, 0])] * len(FINGER_SET)
    p_O_list = [p_O for p_O, _R_O in rec_local]
    R_O_list = [R_O for _p_O, R_O in rec_local]
    gamma_live = solve_gamma_live(p_O_list, R_O_list, mu_c, obj_mass,
                                  ACCEL_BUDGET_XYZ, ANG_ACCEL_BUDGET, obj_inertia,
                                  grav_O=g_O)
    if gamma_live is None or not np.isfinite(gamma_live) or gamma_live <= 0.0:
        print(f"[plan] solve_gamma_live infeasible for this disturbance budget "
             f"-> falling back to gamma={GAMMA_FALLBACK}")
        gamma_live = GAMMA_FALLBACK
    else:
        print(f"[plan] gamma_live={gamma_live:.2f} (min internal-force scale for "
             f"accel={list(ACCEL_BUDGET_XYZ)} m/s^2 + gravity, datum mode)")
    gamma_max = _gamma_stability_ceiling(model, obj_bid, tip_geom_ids)
    if gamma_live > gamma_max:
        print(f"[plan] clamping gamma {gamma_live:.2f} -> {gamma_max:.2f} "
             f"(simulator stability ceiling)")
        gamma_live = gamma_max
    else:
        print(f"[plan] gamma {gamma_live:.2f} is within the stability ceiling "
             f"({gamma_max:.2f})")

    Kp = np.concatenate([np.full(7, 40.0), np.full(16, 0.8)])
    Kd = np.concatenate([np.full(7, 4.0), np.full(16, 0.05)])
    ctrl = GraspController(
        model, N_ROBOT, tip_site_ids=tip_site_ids, obj_site_ids=None,
        obj_body_id=obj_bid, kp=Kp, kd=Kd,
        gamma=gamma_live, squeeze_pd_scale=0.25, support_weight=True,
        pad_offsets=[pad_offset[f] for f in FINGER_SET],
        obj_contact_provider=make_object_contact_provider(rec_local, obj_bid))

    # Reset to the RANDOM start config (q0), not q_target -- the whole point is
    # to test planning + execution FROM an arbitrary starting posture, not to
    # teleport straight to the answer. mj_resetData restores qpos0, which for
    # the object's free joint is exactly the resting pose baked in at compile
    # time by scene.build([(object_id, pos, quat)]) -- the object doesn't move
    # during planning, so its qpos needs no separate re-seeding here.
    mj.mj_resetData(model, data)
    data.qpos[:N_ROBOT] = q0
    mj.mj_forward(model, data)

    approach_path = np.linspace(q0, q_target, APPROACH_STEPS)
    ctrl.set_target(q_target)
    obj_gid = planner._planner._obj_gid
    result = dict(plan=res, verify=verify_info, phase_log=[])

    viewer_cm = mj.viewer.launch_passive(model, data) if view else None
    viewer = viewer_cm.__enter__() if viewer_cm is not None else None
    if viewer is not None:
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    recorder = (VideoRecorder(video_path, lookat=pos, dist=0.6)
               if video_path is not None else None)
    # One frame per PHYSICS step would be gratuitously long/slow to encode
    # (APPROACH alone is 300 steps, HOLD/SQUEEZE/LIFT hundreds more) -- video
    # is meant for visual review, not frame-accurate analysis, so only every
    # VIDEO_STRIDE-th step is captured. Sized so a ~2000-step run (roughly
    # this pipeline's full APPROACH+HOLD+SQUEEZE+LIFT length) becomes a
    # ~15-25s clip at VIDEO_FPS rather than a multi-minute one.
    VIDEO_STRIDE = 4
    _frame_i = 0

    def _sync():
        nonlocal _frame_i
        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)
        if recorder is not None and _frame_i % VIDEO_STRIDE == 0:
            recorder.capture(model, data)
        _frame_i += 1

    try:
        # APPROACH: kinematic replay (qpos overwrite + mj_forward, no mj_step) --
        # sidesteps arm->finger inertial coupling QACC blowups, same as
        # grasp_controller_demo.py's APPROACH phase.
        for i in range(APPROACH_STEPS):
            data.qpos[:N_ROBOT] = approach_path[i]
            data.qvel[:N_ROBOT] = 0.0
            mj.mj_forward(model, data)
            _sync()
        result["phase_log"].append("approach_done")
        print("[exec] APPROACH complete -> HOLD (settling)")

        # HOLD: quasi-static PD settle at q_target before squeezing.
        for _ in range(200):
            data.qvel[:N_ROBOT] = 0.0
            data.qfrc_applied[:] = ctrl.compute(data)
            mj.mj_step(model, data)
            _sync()
        result["phase_log"].append("hold_settled")

        # SANITY GATE: the planner's r_tip is DELIBERATELY sized so the IK
        # target leaves a small (up to ~4-5mm), non-penetrating gap rather
        # than touching -- see grasp_planner_3d.py's _tip_radius comment;
        # the squeeze phase below is EXPECTED to close it gently. This gate
        # only catches a much larger, genuinely-broken gap (e.g. the mesh
        # collision proxy let the IK converge nowhere near the real object).
        gaps_mm = _tip_gaps_mm(model, data, tip_geom_ids, obj_gid)
        gap_by_finger = dict(zip(FINGER_SET, gaps_mm))
        result["tip_gaps_mm"] = gap_by_finger
        untouched = {f: g for f, g in gap_by_finger.items() if g > CONTACT_GAP_TOL_M * 1000}
        if untouched:
            print(f"[exec] ABORT before SQUEEZE — gap too large to close gently "
                 f"(tol={CONTACT_GAP_TOL_M * 1000:.1f}mm): {gap_by_finger}")
            result["phase_log"].append("squeeze_aborted_no_contact")
            result["measured_tip_forces_N"] = dict(zip(FINGER_SET, [0.0] * len(FINGER_SET)))
            result["object_drift_mm"] = float(np.linalg.norm(data.xpos[obj_bid] - p_WoO)) * 1e3
            if render_path is not None:
                try:
                    render(model, data, render_path, lookat=pos, dist=0.5, elev=-55)
                    print(f"[exec] render -> {render_path}")
                except Exception as e:
                    print(f"[exec] render failed: {e}")
            return res, result
        print(f"[exec] gap check OK (<= {CONTACT_GAP_TOL_M * 1000:.1f}mm), squeeze will close it: {gap_by_finger}")

        # SQUEEZE: internal (pinching) force, ramped 0->1 over SQUEEZE_RAMP_S so
        # the pair of contact forces doesn't arrive as an unbalanced shove while
        # a finger is still closing the last gap (see internal_force_torques'
        # own docstring).
        ctrl.set_squeeze(True)
        if viewer is not None:
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        print(f"[exec] -> SQUEEZE (ramping over {SQUEEZE_RAMP_S}s)")
        n_ramp = max(int(SQUEEZE_RAMP_S / model.opt.timestep), 1)
        n_squeeze_steps = n_ramp * 4
        for i in range(n_squeeze_steps):
            scale = min(1.0, i / n_ramp)
            data.qvel[:N_ROBOT] = 0.0
            data.qfrc_applied[:] = ctrl.compute(data)
            if scale < 1.0:
                # compute() applies internal_force_torques(data) at scale=1;
                # redo the squeeze term at the ramped scale instead.
                kp, kd = ctrl.effective_gains()
                data.qfrc_applied[:N_ROBOT] = (kp * (q_target - data.qpos[:N_ROBOT])
                                               + kd * (0 - data.qvel[:N_ROBOT])
                                               + data.qfrc_bias[:N_ROBOT]
                                               + ctrl.internal_force_torques(data, scale=scale))
            mj.mj_step(model, data)
            _sync()
            if i % PRINT_EVERY == 0:
                f_meas = _measured_tip_forces(model, data, tip_geom_ids, obj_gid)
                print(f"[squeeze] t={i * model.opt.timestep:.2f}s scale={scale:.2f} "
                     f"meas_normal={dict(zip(FINGER_SET, np.round(f_meas, 2)))}")
        result["phase_log"].append("squeeze_done")

        f_meas = _measured_tip_forces(model, data, tip_geom_ids, obj_gid)
        drift_mm = float(np.linalg.norm(data.xpos[obj_bid] - p_WoO)) * 1e3
        print(f"[squeeze] final meas_normal={dict(zip(FINGER_SET, np.round(f_meas, 2)))}  "
             f"obj_drift={drift_mm:.2f}mm")
        result["measured_tip_forces_N"] = dict(zip(FINGER_SET, f_meas))
        result["object_drift_mm"] = drift_mm

        if do_lift:
            lift_result = _run_lift_jog(model, data, ctrl, obj_bid, tip_geom_ids, obj_gid,
                                        q_target, _sync)
            result.update(lift_result)
            result["phase_log"].append("lift_done")

        if render_path is not None:
            try:
                render(model, data, render_path, lookat=pos, dist=0.5, elev=-55)
                print(f"[exec] render -> {render_path}")
            except Exception as e:
                print(f"[exec] render failed: {e}")
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)
        if recorder is not None:
            recorder.close()
            print(f"[exec] video -> {video_path}")

    return res, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="017_orange")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--n-relin", type=int, default=None)
    ap.add_argument("--gws", action="store_true")
    ap.add_argument("--w-gws", type=float, default=5.0)
    ap.add_argument("--w-span", type=float, default=1.0)
    ap.add_argument("--tangent-plane", dest="use_quadratic", action="store_false",
                    help="use the tangent-plane mesh contact model instead of the "
                         "local-quadratic (SDF) one")
    ap.add_argument("--view", action="store_true", help="interactive viewer")
    # --render/--video/--quad-plot used to gate these; every run now writes all
    # three into out/<object>/, so the flags are gone rather than kept as no-ops.
    ap.add_argument("--no-lift", dest="do_lift", action="store_false",
                    help="skip the post-squeeze vertical lift-jog stability check")
    ap.add_argument("--w-edge-margin", type=float, default=0.0,
                    help="penalize a contact that comes within edge_margin_sdf_m of a "
                         "trust-region bound set by a MEASURED SDF divergence (a real "
                         "surface boundary), ignoring bounds at quadratic_t_bound_max "
                         "which only mean 'flat as far as the search looked'. Replaces "
                         "the old curvature-based penalty, which was blind on flat faces "
                         "and mis-fired on genuinely round objects (0.0 = off)")
    ap.add_argument("--directional-r-tip", action="store_true",
                    help="size the IK fingertip offset by the pad's support distance along "
                         "the contact normal (refrozen per Picard stage) instead of the "
                         "isotropic bounding-sphere radius -- removes several mm of "
                         "pre-squeeze gap on an elongated pad")
    ap.add_argument("--r-tip-margin-mm", type=float, default=1.0,
                    help="safety cushion added to the directional radius (mm), keeping the "
                         "target on the non-penetrating side")
    OP.add_out_args(ap, OP.FLOOR)
    args = ap.parse_args()

    # One subfolder per YCB object, so runs group by object rather than piling
    # every objectxseed artifact into one flat directory. All three artifacts
    # (final-pose PNG, run MP4, quadratic-path iso views) are ALWAYS produced --
    # a run you cannot look at afterwards is not worth much, and the flags that
    # used to gate them only ever saved a few seconds.
    out_dir = OP.resolve_out(args, OP.FLOOR) / args.object
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}"
    render_path = str(out_dir / f"{tag}.png")
    video_path = str(out_dir / f"{tag}.mp4")
    quad_plot_path = str(out_dir / f"{tag}_quadratic_path.png")

    run_pick(args.object, args.seed, n_seeds=args.n_seeds, n_relin=args.n_relin,
            gws=args.gws, w_gws=args.w_gws, w_span=args.w_span,
            use_quadratic=args.use_quadratic, view=args.view, render_path=render_path,
            video_path=video_path, quad_plot_path=quad_plot_path, do_lift=args.do_lift,
            w_edge_margin=args.w_edge_margin,
            directional_r_tip=args.directional_r_tip,
            directional_r_tip_margin_m=args.r_tip_margin_mm / 1000.0)


if __name__ == "__main__":
    main()

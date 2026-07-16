"""Kinova Gen3 arm wrist-position + palm-orientation tracking for DexPilot.

Maps the human wrist camera-space position to a 3D target in the robot workspace
and the human palm orientation to a desired approach direction on the robot.
Both are solved together via DLS IK with an optional orientation constraint.

Coordinate conventions
  Camera space  (raw[0:3] from /hand/joint_angles):
    [0] image x — right is positive (mirror of robot frame)
    [1] image y — down  is positive
  MediaPipe world landmarks space:
    x — right  y — up  z — toward camera
  Robot world frame:
    x — forward from robot base   z — up

R_cam_robot maps MediaPipe WORLD landmark vectors → robot world vectors.
The default assumes the camera is mounted facing the workspace from in front:
  cam +x (right)        → robot +y  (robot's left as seen from behind)
  cam +y (up)           → robot +z  (world up)
  cam +z (toward camera)→ robot -x  (camera behind workspace → cam-toward = robot-backward)
Adjust this 3×3 rotation matrix if your camera is mounted differently.

Orientation tracking is DELTA-based (see step()): only the human wrist's change
in orientation since startup is applied to the robot's home approach axis, so
the arm holds its home pose until you actually rotate your wrist, and an
imperfect R_cam_robot only skews the mapping of *rotations*, not an absolute pose.
"""
from __future__ import annotations

import json
import os

import numpy as np
import mujoco as mj

from grasp_control import SpatialIKSolver

_CALIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "calibration")
_R_CORRECT_PATH = os.path.join(_CALIB_DIR, "orientation_correction.json")


_TELEOP_CONFIG_PATH = os.path.join(_CALIB_DIR, "teleop_config.json")


def load_teleop_config(path: str | None = None) -> dict:
    """Load DexPilot teleop tunables (calibration/teleop_config.json).

    Returns a dict of DexPilotController kwargs derived from the config's
    `position` block: `position_mode`, `abs_scale`, `world_from_board` (3x3 list
    -> np.ndarray). Missing file or missing keys fall back to sane defaults
    (relative mode, abs_scale 1.0, identity remap), so teleop still runs without
    the file. `_comment*` keys are ignored. The caller may override `position_mode`
    afterwards (e.g. from the --position-mode CLI flag).
    """
    path = path or _TELEOP_CONFIG_PATH
    pos = {}
    if os.path.exists(path):
        with open(path) as f:
            pos = (json.load(f) or {}).get("position", {}) or {}
        print(f"[teleop] loaded config from {path}")
    else:
        print(f"[teleop] no {os.path.basename(path)} — using position defaults "
              f"(relative, abs_scale=1.0).")

    wfb = pos.get("world_from_board")
    return {
        "position_mode":    pos.get("mode", "relative"),
        "abs_scale":        float(pos.get("abs_scale", 1.0)),
        "world_from_board": (np.asarray(wfb, float) if wfb is not None
                             else np.eye(3)),
    }


def load_camera_calibration(
    extrinsics_path: str | None = None,
    intrinsics_path: str | None = None,
    world_from_board: np.ndarray | None = None,
) -> dict:
    """Build calibrated DexPilotArmController kwargs from ChArUco calibration.

    Produces `R_cam_robot`, `scale_x`, `scale_z` from the measured camera
    intrinsics + extrinsics, replacing the hardcoded `_DEFAULT_R_CAM_ROBOT` and
    guessed pixel gains. Feed the result to DexPilotArmController via **kwargs
    (or DexPilotController's arm_kwargs).

    Assumptions / scope (matches how the controller consumes these):
      * The board frame IS the robot world frame (identity world<-board) unless
        you pass `world_from_board`. So R_cam_robot = R_world_cam directly.
      * MediaPipe world landmarks are expressed in camera-aligned axes, so the
        extrinsic camera->world ROTATION applies to them; that fixes ORIENTATION
        mapping (the high-confidence win).
      * Position stays delta-based on NORMALISED image coords. scale converts a
        full-frame normalised delta -> metres at the workspace depth:
            metres = Δn * pixels_per_frame * Z / f   =>  scale = (W or H) * Z / f
        with Z = board_distance_m. This is a principled replacement for the
        arbitrary 0.3/0.2, valid at roughly the calibrated working depth — it is
        NOT full 3D metric positioning (no live per-frame depth).

    Args:
        extrinsics_path: path to camera_extrinsics.json (default: calibration/).
        intrinsics_path: path to camera_intrinsics.json (default: calibration/).
        world_from_board: optional 3x3 rotation mapping board axes -> robot world
                          axes. Default identity (board == robot world frame).
    Returns:
        dict with keys R_cam_robot (3x3), scale_x (float), scale_z (float).
    """
    extrinsics_path = extrinsics_path or os.path.join(_CALIB_DIR, "camera_extrinsics.json")
    intrinsics_path = intrinsics_path or os.path.join(_CALIB_DIR, "camera_intrinsics.json")

    with open(extrinsics_path) as f:
        extr = json.load(f)
    with open(intrinsics_path) as f:
        intr = json.load(f)

    R_world_cam = np.asarray(extr["R_world_cam"], dtype=float)   # cam axes -> world axes
    Z = float(extr["board_distance_m"])                          # workspace depth [m]

    R_wb = np.eye(3) if world_from_board is None else np.asarray(world_from_board, float)
    # camera axes -> board(world) axes -> robot world axes. NOTE: orientation
    # tracking no longer uses R_cam_robot (it uses press-8 palm-local alignment,
    # which is camera-mounting-independent). R_cam_robot is retained only for
    # backward compatibility / any position-side use.
    R_cam_robot = R_wb @ R_world_cam

    W, H = intr["image_size"]                                    # pixels
    K = np.asarray(intr["camera_matrix"], dtype=float)
    fx, fy = K[0, 0], K[1, 1]
    scale_x = float(W) * Z / fx    # normalised-x delta -> metres at depth Z
    scale_z = float(H) * Z / fy    # normalised-y delta -> metres at depth Z

    return {"R_cam_robot": R_cam_robot, "scale_x": scale_x, "scale_z": scale_z}

# Pinch-site local +z points world +X (forward) at HOME_ARM (verified via FK).
# This is the "approach direction" axis we align with the human palm normal.
_PALM_LOCAL_AXIS = np.array([0.0, 0.0, 1.0])

# Default camera → robot world rotation — a hand-guessed fallback used only when
# no ChArUco calibration is available. Prefer load_camera_calibration() above,
# which supplies a measured R_cam_robot (and pixel→metre scales) from
# calibration/camera_extrinsics.json + camera_intrinsics.json.
# cam +x → robot +y,  cam +y → robot +z,  cam +z → robot -x
_DEFAULT_R_CAM_ROBOT = np.array([
    [ 0,  0, -1],   # robot x = -cam z
    [ 1,  0,  0],   # robot y = +cam x
    [ 0,  1,  0],   # robot z = +cam y
], dtype=float)


class DexPilotArmController:
    """Drives the Kinova arm to track the human wrist via DLS IK + EMA.

    Position: delta-mapped from the first received camera wrist position.
    Orientation: palm normal from world landmarks → pinch_site approach axis.

    Args:
        model:        MuJoCo model (composite arm + hand).
        n_arm:        Number of arm DOF (default 7).
        palm_site:    Site on bracelet_link to drive ('pinch_site').
        q_bias:       n_arm-element null-space bias (arm HOME_ARM pose).
        alpha:        EMA factor for output smoothing (0=max smooth, 1=raw).
        scale_x:      Camera image Δx → robot Δx scale [m].
        scale_z:      Camera image Δy → robot Δz scale [m].
        scale_depth:  Monocular hand-depth Δ [m] → robot Δy gain (dimensionless;
                      depth already metric). Modest default — the monocular
                      estimate is noisy; raise once it proves stable, or drop in
                      a RealSense depth source. 0.0 freezes robot-Y (old behaviour).
        absolute:     If True, cam[:3] is a metric wrist position in the ChArUco
                      board frame; motion maps absolutely (×abs_scale) from the
                      press-8 zero instead of image deltas. Requires the publisher
                      to be in absolute mode (calibration present).
        world_from_board: 3×3 axis remap board→robot world (default identity).
                      Only skews a delta, so calibrate empirically if an axis
                      comes out mirrored/rotated.
        abs_scale:    Gain on absolute board motion (1.0 = 1:1 metric; raise to
                      1.5-2.0 to amplify a comfortable hand range into the robot
                      workspace).
        R_cam_robot:  3×3 rotation mapping MediaPipe world → robot world.
                      Pass None to disable palm-orientation tracking.
    """

    def __init__(
        self,
        model: mj.MjModel,
        n_arm: int = 7,
        palm_site: str = 'pinch_site',
        q_bias: np.ndarray | None = None,
        alpha: float = 0.3,
        scale_x: float = 0.3,
        scale_z: float = 0.2,
        scale_depth: float = 0.5,
        position_mode: str = "relative",
        world_from_board: np.ndarray | None = None,
        abs_scale: float = 1.0,
        full_orientation: bool = True,
        R_align: np.ndarray | None = None,
        R_cam_robot: np.ndarray | None = _DEFAULT_R_CAM_ROBOT,
        identity_orientation: bool = False,
    ) -> None:
        self._model       = model
        self._n_arm       = n_arm
        self._q_bias      = q_bias if q_bias is not None else np.zeros(n_arm)
        self._alpha       = alpha
        self._scale_x     = scale_x
        self._scale_z     = scale_z
        self._scale_depth = scale_depth
        # Position mode (see _camera_to_world). Both consume cam[:3] = the metric
        # wrist position in the ChArUco board frame (published in absolute mode):
        #   "relative": press-8 re-zeroable — robot tracks abs_scale × (board
        #     displacement from the press-8 board_ref), anchored at _home_site.
        #   "absolute": true absolute — the board position maps to a FIXED robot
        #     world position (board origin -> robot BASE origin (0,0,0)), scaled by
        #     abs_scale. No press-8 re-zero; the workspace is physically pinned.
        # "legacy" (delta image pixels) remains available for the no-calibration
        # fallback. abs_scale applies to BOTH relative and absolute.
        if position_mode not in ("relative", "absolute", "legacy"):
            raise ValueError(f"position_mode must be relative|absolute|legacy, "
                             f"got {position_mode!r}")
        self._position_mode = position_mode
        self._Rwb        = (np.eye(3) if world_from_board is None
                            else np.asarray(world_from_board, float))
        self._abs_scale  = abs_scale
        # Full 3-DOF wrist orientation tracking (vs legacy approach-axis-only,
        # which ignored roll about the axis — the reason wrist roll didn't
        # register). Delta-based off the press-8 reference either way.
        self._full_orientation = full_orientation
        # Constant world-side correction R_des = R_correct @ R_mp_to_robot @ palm_R.
        # Reset to IDENTITY now that the MediaPipe left-handedness is fixed at the
        # source (see human_palm_frame_robot_aligned) — the earlier measured
        # R_correct was compensating for that broken frame and is no longer valid.
        # Re-measure with teleop/diagnose_frame.py 'c' if a residual board-yaw
        # offset remains, and pass R_align= to set it.
        # Load a saved orientation correction if present (from a prior multi-pose
        # calibration), else identity — or an explicit R_align override.
        #
        # identity_orientation: force R_correct = I and disable ALL calibration
        # (press-8 auto-calib AND the saved orientation_correction.json). This is
        # the DIRECT mapping R_des = R_mp_to_robot @ palm_R — the world->wrist
        # rotation shown in the MediaPipe overlay becomes the exact target in
        # MuJoCo, with no offset. Meaningful now that the arm palm frame is built
        # from the STABLE image landmarks (the saved correction was fit against the
        # old flip-prone world-landmark frame and is no longer valid).
        self._identity_orientation = identity_orientation
        if identity_orientation:
            self._R_correct = np.eye(3)
            self._has_full_correction = True   # block press-8 from overwriting I
            print("[arm] identity orientation: direct hand->wrist mapping, "
                  "no calibration offset.")
        elif R_align is not None:
            self._R_correct = np.asarray(R_align, float)
            self._has_full_correction = True
        elif os.path.exists(_R_CORRECT_PATH):
            with open(_R_CORRECT_PATH) as f:
                self._R_correct = np.array(json.load(f)["R_correct"], float)
            self._has_full_correction = True
            print(f"[arm] loaded saved orientation correction from {_R_CORRECT_PATH}")
        else:
            self._R_correct = np.eye(3)
            self._has_full_correction = False
        self._R_cam_robot = R_cam_robot

        self._site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, palm_site)
        # n_arm only: pinch_site depends solely on the 7 arm joints.
        # n_robot=23 caused a shape mismatch (q_bias[7] vs q[23]) in the
        # null-space projection inside SpatialIKSolver.solve().
        # adaptive_damping: singularity-robust IK for live teleop — the arm gets
        # driven to arbitrary human-tracked poses that can pass through singular
        # configurations, where fixed small damping made dq explode ("whipping").
        # w0=0.02: measured 6-DOF pinch_site manipulability is ~0.043 median /
        # 0.065 home / 0.11 max, dropping toward 0 at singularities. 0.02 sits
        # below normal operation (so damping stays low, tracking crisp) but above
        # the ill-conditioned region (p10~0.006), so damping ramps as you near a
        # singularity. lambda_max=0.15 bounds dq firmly in the singular limit.
        self._ik      = SpatialIKSolver(n_robot=n_arm, adaptive_damping=True,
                                        lambda_max=0.15, w0=0.02)

        # Scratch MjData so the IK iterations don't affect the main sim state
        self._scratch = mj.MjData(model)

        self._q_arm_prev: np.ndarray | None = None
        self._cam_home:   np.ndarray | None = None   # first received camera wrist xy
        self._board_ref:  np.ndarray | None = None   # hand board-pos at press-8 (absolute mode)
        # When True, the next step() captures R_correct so the hand's CURRENT
        # orientation maps to the robot's home wrist (auto-calibrate at press-8).
        self._orient_calib_pending = False
        self._last_raw: np.ndarray | None = None   # R_mp_to_robot @ palm_R (uncorrected)
        self._calib_pairs: list = []               # (raw_i, R_site_i) for multi-pose solve
        self._home_site:  np.ndarray | None = None   # pinch_site world pos at startup
        self._palm_R_home: np.ndarray | None = None  # human palm frame (robot world) at startup
        self._axis_home:   np.ndarray | None = None  # pinch_site approach axis (world) at startup
        self._R_site_home: np.ndarray | None = None  # full pinch_site rotation (world) at startup
        # Last IK target pose (world) — the frame the IK cost drives pinch_site
        # toward. Exposed via target_frame() for visualisation/debugging.
        self._tgt_pos:  np.ndarray | None = None
        self._tgt_R:    np.ndarray | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_home(self, data: mj.MjData) -> None:
        """Snapshot the current pinch_site position as the arm home reference.

        Call once after the simulation is reset to the home/bias pose.
        """
        mj.mj_forward(self._model, data)
        self._home_site = data.site_xpos[self._site_id].copy()
        # Robot pinch_site orientation (world) at HOME_ARM. Full-rotation tracking
        # is delta-based off this reference so the arm holds its home pose until
        # the human wrist rotates. (_axis_home kept for the legacy axis-only path.)
        R_site = data.site_xmat[self._site_id].reshape(3, 3).copy()
        self._R_site_home = R_site
        self._axis_home = R_site @ _PALM_LOCAL_AXIS
        self._scratch.qpos[:] = data.qpos[:]

    # ------------------------------------------------------------------
    # Camera-to-world mapping
    # ------------------------------------------------------------------

    def _camera_to_world(self, cam: np.ndarray) -> np.ndarray:
        """Map camera-space wrist position to a robot world-frame position target.

        Three modes (self._position_mode):
          * "relative": cam[:3] is a metric wrist position in the ChArUco board
            frame. On the first frame after press-8 we snapshot it as board_ref;
            thereafter the robot tracks the metric board-frame motion (×abs_scale)
            from _home_site. Re-zeroable at press-8 — the robot's current wrist
            pose ↔ the hand's current board position — then moves from there.
          * "absolute": TRUE absolute. cam[:3] (board frame) maps to a FIXED robot
            world position: board ORIGIN -> robot BASE origin (0,0,0), the whole
            position scaled by abs_scale. No press-8 anchor, no re-zero — the
            workspace is physically pinned to the board.
          * "legacy": cam = [image_x, image_y, depth_m]; motion is relative to the
            first received frame, mapped through per-axis pixel scales.
        Position mapping in ALL modes is INDEPENDENT of hand orientation.
        """
        if self._home_site is None:
            raise RuntimeError(
                "init_home() must be called before DexPilotArmController.step()")

        if self._position_mode == "relative":
            if self._board_ref is None:
                self._board_ref = cam[:3].copy()   # hand board-pos at press-8
            delta_board = cam[:3] - self._board_ref
            # Map the board/world-frame displacement into the robot frame by a
            # FIXED world->robot rotation (_Rwb, default identity for board ==
            # robot world). Must NOT depend on hand orientation (coupling it made
            # the same real motion map to different robot directions).
            return self._home_site + self._abs_scale * (self._Rwb @ delta_board)

        if self._position_mode == "absolute":
            # Board ORIGIN -> robot BASE origin (0,0,0). The full board-frame
            # position (remapped to robot axes by _Rwb) is scaled by abs_scale.
            # No _home_site, no _board_ref: the robot wrist goal is a fixed
            # function of where the hand is over the board. abs_scale>1 amplifies
            # reach about the board origin.
            return self._abs_scale * (self._Rwb @ cam[:3])

        # "legacy": cam = [image_x, image_y, depth_m]; depth from monocular hand-size in
        # the publisher (raw[2]). Delta-based like x/y: only motion relative to
        # the startup depth drives robot-Y, so absolute-depth bias doesn't matter.
        if self._cam_home is None:
            self._cam_home = cam[:3].copy()

        delta = cam[:3] - self._cam_home
        # Depth (metres) → robot-Y. Sign: hand moving TOWARD the camera makes the
        # estimated depth SMALLER, which should push the arm toward the workspace
        # (robot +X is forward, but Y here is the controller's depth axis) — hence
        # negate so closer hand = positive Y motion. scale_depth kept modest until
        # the monocular signal proves stable; tune or replace with RealSense.
        return self._home_site + np.array([
            -delta[0] * self._scale_x,   # image x → robot x (negated: mirror)
            -delta[2] * self._scale_depth,  # depth (m) → robot y (was frozen)
            -delta[1] * self._scale_z,   # image y (down+) → robot z (negate: up+)
        ])

    # ------------------------------------------------------------------
    # Control step
    # ------------------------------------------------------------------

    def step(
        self,
        cam_wrist: np.ndarray,
        data: mj.MjData,
        palm_R: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve the arm IK and return smoothed 7-DOF arm joint angles.

        Args:
            cam_wrist: wrist position in camera space (raw[0:3] from ROS message).
            data:      Current simulation state (arm qpos used as IK warm-start).
            palm_R:    (3,3) human palm rotation from DexPilotRetargeter.human_palm_frame().
                       Column [:, 2] is the palm normal in MediaPipe world coords.
                       When provided and R_cam_robot is set, adds a palm-orientation
                       constraint so the pinch_site approach axis tracks the palm normal.
        Returns:
            q_arm: (n_arm,) joint angles in radians.
        """
        if self._home_site is None:
            self.init_home(data)

        pos_target = self._camera_to_world(cam_wrist)

        # ABSOLUTE orientation, DIRECT (no press-8 offset). palm_R here is the
        # ROBOT-ALIGNED palm frame (human_palm_frame_robot_aligned): its axis
        # roles match pinch_site (X=palm-normal, Y=toward-thumb, Z=along-fingers),
        # so an identical physical orientation gives an identical matrix. We only
        # need to bring it from MediaPipe world axes into robot world axes:
        #   R_des = R_mp_to_robot @ palm_R,   R_mp_to_robot = R_cam_robot @ R_mp_to_cv
        # R_mp_to_cv=diag([1,-1,-1]) converts MediaPipe-world (x-right,y-up,
        # z-toward-cam) -> OpenCV-cam, the frame R_cam_robot expects. No R_align,
        # no home anchor: palm-down -> wrist-down by construction. Wrist snaps to
        # match the hand orientation when tracking starts (accepted).
        orientation = None
        if palm_R is not None and self._R_cam_robot is not None:
            R_mp_to_cv = np.diag([1.0, -1.0, -1.0])
            R_mp_to_robot = self._R_cam_robot @ R_mp_to_cv
            raw = R_mp_to_robot @ palm_R
            # AUTO-CALIBRATE at press-8: on the first tracked frame, define the
            # constant correction so the operator's CURRENT hand orientation maps
            # to the robot's HOME wrist orientation. Because palm_R tracks the hand
            # rigidly (verified), this one alignment is correct for ALL later
            # poses — and it avoids the circular "match the moving wrist" problem
            # (the home frame is fixed, not chasing the hand).
            # Skip auto-calib if a full (multi-pose/saved) correction exists —
            # the single-point press-8 fit would clobber it and reintroduce the
            # wrong relative rotations.
            if (self._orient_calib_pending and self._R_site_home is not None
                    and not self._has_full_correction):
                self._R_correct = self._R_site_home @ raw.T
                self._orient_calib_pending = False
                print("[arm] orientation auto-calibrated at press-8 "
                      "(current hand pose -> robot home wrist).")
            elif self._orient_calib_pending:
                self._orient_calib_pending = False  # consumed; keep full correction
            # Store the UNCORRECTED mapping for multi-pose calibration.
            self._last_raw = raw.copy()
            # R_des = R_correct @ (R_mp_to_robot @ palm_R)
            R_des = self._R_correct @ raw

            if self._full_orientation:
                orientation = R_des
            else:
                target_axis = R_des @ _PALM_LOCAL_AXIS
                norm = np.linalg.norm(target_axis)
                if norm > 1e-6:
                    orientation = (_PALM_LOCAL_AXIS, target_axis / norm)

        # Record the IK target pose for visualisation. Orientation is the full
        # 3x3 R_des when full_orientation; else fall back to the current site
        # rotation (axis-only mode doesn't constrain a full frame).
        self._tgt_pos = pos_target.copy()
        if isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
            self._tgt_R = orientation.copy()
        else:
            self._tgt_R = None

        # Seed the scratch state from the current arm configuration
        self._scratch.qpos[:self._n_arm] = data.qpos[:self._n_arm].copy()
        mj.mj_forward(self._model, self._scratch)

        q_full = self._ik.solve(
            self._model, self._scratch,
            [self._site_id], [pos_target],
            orientations=[orientation],
            q_bias=self._q_bias,
            null_gain=0.3,
        )
        q_arm = q_full[:self._n_arm].copy()

        # EMA smoothing
        if self._q_arm_prev is not None:
            q_arm = self._alpha * q_arm + (1.0 - self._alpha) * self._q_arm_prev
        self._q_arm_prev = q_arm.copy()
        return q_arm

    def target_frame(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the last IK target pose (pos, R) in world coords, or None.

        This is exactly the frame the IK cost drives pinch_site toward — draw it
        alongside pinch_site to see position + orientation tracking error. R is
        None if orientation tracking is off / axis-only (no full target frame).
        """
        if self._tgt_pos is None:
            return None
        R = self._tgt_R if self._tgt_R is not None else np.eye(3)
        return self._tgt_pos, R

    def request_orientation_calib(self) -> None:
        """Arm the auto-calibration: the next tracked frame maps the operator's
        current hand orientation to the robot's home wrist. Called from press-8
        so calibration happens against the FIXED home frame (not the live wrist,
        which would chase the hand)."""
        self._orient_calib_pending = True

    def calibrate_orientation(self, data: mj.MjData) -> None:
        """Capture the constant orientation correction by aligning the current
        target to the robot's ACTUAL wrist orientation.

        Call while the operator holds their hand to visually MATCH the robot's
        current pinch_site orientation. We solve for R_correct so that the
        current mapped target equals the real wrist frame:
            R_correct_new @ (R_mp_to_robot @ palm_R) = R_site_current
        Since the current target is R_des = R_correct_old @ R_mp_to_robot @ palm_R,
        we get R_correct_new = R_site_current @ (R_mp_to_robot @ palm_R)^T
                             = R_site_current @ (R_correct_old^T @ R_des)^T. Using
        the stored R_des (self._tgt_R) directly:
            raw = R_correct_old^T @ R_des      (= R_mp_to_robot @ palm_R)
            R_correct_new = R_site_current @ raw^T
        Because palm_R tracks the hand RIGIDLY, this single alignment makes the
        target match the wrist for ALL subsequent poses.
        """
        if self._tgt_R is None:
            print("[arm] calibrate_orientation: no target yet. Press 8 to START "
                  "tracking first (the target only updates while tracking is "
                  "active), then hold your hand to match the wrist and press 9.")
            return
        mj.mj_forward(self._model, data)
        R_site = data.site_xmat[self._site_id].reshape(3, 3).copy()
        raw = self._R_correct.T @ self._tgt_R          # R_mp_to_robot @ palm_R
        self._R_correct = R_site @ raw.T
        print("[arm] orientation calibrated to current wrist. R_correct =")
        print(np.array2string(self._R_correct, precision=4))

    # -- Multi-pose orientation calibration --------------------------------
    # Solves the FULL rotation mapping M (not just the home) from several
    # (hand, wrist) pairs: hold your hand to match the robot at each posed
    # orientation, capture, then solve M with M @ raw_i = R_site_i for all i.
    # This fixes wrong RELATIVE rotations that a single home-alignment can't.

    def capture_calib_pose(self, data: mj.MjData) -> int:
        """Record one (uncorrected-mapping, actual-wrist) pair. Returns count."""
        if self._last_raw is None:
            print("[arm] capture: no hand mapping yet — is a hand tracked?")
            return len(self._calib_pairs)
        mj.mj_forward(self._model, data)
        R_site = data.site_xmat[self._site_id].reshape(3, 3).copy()
        self._calib_pairs.append((self._last_raw.copy(), R_site))
        print(f"[arm] captured calibration pose {len(self._calib_pairs)} "
              f"(hand matched to current wrist).")
        return len(self._calib_pairs)

    def solve_calib(self) -> None:
        """Solve R_correct from all captured pairs (least-squares, orthonormalised).
        Needs >=2 poses; more poses -> better full-rotation fit."""
        if len(self._calib_pairs) < 2:
            print(f"[arm] solve: need >=2 poses, have {len(self._calib_pairs)}.")
            return
        A_raw = np.hstack([p[0] for p in self._calib_pairs])   # 3 x 3N
        A_site = np.hstack([p[1] for p in self._calib_pairs])
        M = A_site @ np.linalg.pinv(A_raw)
        U, _, Vt = np.linalg.svd(M)
        self._R_correct = U @ Vt                                # nearest rotation
        det = float(np.linalg.det(self._R_correct))
        print(f"[arm] SOLVED R_correct from {len(self._calib_pairs)} poses "
              f"(det={det:+.2f}). Per-pose residuals (0=perfect, >1=bad):")
        errs = []
        for i, (raw, site) in enumerate(self._calib_pairs):
            err = float(np.linalg.norm(self._R_correct @ raw - site))
            errs.append(err)
            flag = "  <-- BAD, re-capture" if err > 1.0 else ""
            print(f"       pose {i+1}: {err:.3f}{flag}")
        mean_err = float(np.mean(errs))
        if mean_err < 0.3:
            verdict = "GOOD fit — a single rotation maps hand->wrist."
        elif mean_err < 0.8:
            verdict = "ROUGH — re-capture the BAD poses (held more accurately)."
        else:
            verdict = ("POOR — no single rotation fits. Likely bad pose-matching, "
                       "OR the hand->wrist map isn't a single rotation (deeper).")
        print(f"[arm] mean residual {mean_err:.3f} -> {verdict}")
        self._has_full_correction = True   # don't let press-8 clobber it
        # Persist so it loads next launch.
        with open(_R_CORRECT_PATH, "w") as f:
            json.dump({"R_correct": self._R_correct.tolist(),
                       "mean_residual": mean_err}, f, indent=2)
        print(f"[arm] saved -> {_R_CORRECT_PATH}")
        print(np.array2string(self._R_correct, precision=4))

    def clear_calib(self) -> None:
        self._calib_pairs = []
        print("[arm] cleared captured calibration poses.")

    def reset(self) -> None:
        """Reset delta-tracking state (e.g., when hand tracking is lost).

        Clears both the position and orientation startup references so the next
        received frame re-snapshots home; axis_home (from init_home) is kept.
        """
        self._q_arm_prev  = None
        self._cam_home    = None
        self._board_ref   = None   # re-snap the absolute board zero on press-8
        self._palm_R_home = None
        self._R_align     = np.eye(3)   # re-capture press-8 orientation alignment

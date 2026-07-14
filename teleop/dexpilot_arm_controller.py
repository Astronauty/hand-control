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
        absolute: bool = False,
        world_from_board: np.ndarray | None = None,
        abs_scale: float = 1.0,
        full_orientation: bool = True,
        R_align: np.ndarray | None = None,
        R_cam_robot: np.ndarray | None = _DEFAULT_R_CAM_ROBOT,
    ) -> None:
        self._model       = model
        self._n_arm       = n_arm
        self._q_bias      = q_bias if q_bias is not None else np.zeros(n_arm)
        self._alpha       = alpha
        self._scale_x     = scale_x
        self._scale_z     = scale_z
        self._scale_depth = scale_depth
        # Absolute board-anchored positioning (see _camera_to_world). When True,
        # cam[:3] is a metric wrist position in the ChArUco board frame (published
        # by ui/mediapipe_joint_angles.py in absolute mode), and motion is mapped
        # 1:1×abs_scale from the press-8 zero rather than from image deltas.
        self._absolute   = absolute
        self._Rwb        = (np.eye(3) if world_from_board is None
                            else np.asarray(world_from_board, float))
        self._abs_scale  = abs_scale
        # Full 3-DOF wrist orientation tracking (vs legacy approach-axis-only,
        # which ignored roll about the axis — the reason wrist roll didn't
        # register). Delta-based off the press-8 reference either way.
        self._full_orientation = full_orientation
        # Constant palm->pinch_site convention offset for ABSOLUTE orientation.
        # The MediaPipe palm frame (X=along-hand, Z=palm-normal) and the robot
        # pinch_site frame use different axis roles; R_align (right-multiplied on
        # palm_R) reconciles them. Default identity — tune live by watching the
        # target vs pinch_site triads until 'palm flat' maps to the intended pose.
        self._R_align = np.eye(3) if R_align is None else np.asarray(R_align, float)
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
        self._home_site:  np.ndarray | None = None   # pinch_site world pos at startup
        self._palm_R_home: np.ndarray | None = None  # human palm frame (robot world) at startup
        self._axis_home:   np.ndarray | None = None  # pinch_site approach axis (world) at startup
        self._R_site_home: np.ndarray | None = None  # full pinch_site rotation (world) at startup
        self._R_board_to_robot: np.ndarray | None = None  # press-8 board->robot frame map
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

        Two modes:
          * absolute: cam[:3] is a metric wrist position in the ChArUco board
            frame. On the first frame after press-8 we snapshot it as board_ref;
            thereafter the robot tracks the metric board-frame motion (×abs_scale)
            from _home_site. This aligns the MuJoCo and board frames at press-8 —
            the robot's current wrist pose ↔ the hand's current board position —
            then moves absolutely from there.
          * delta (legacy): cam = [image_x, image_y, depth_m]; motion is relative
            to the first received frame, mapped through per-axis pixel scales.
        """
        if self._home_site is None:
            raise RuntimeError(
                "init_home() must be called before DexPilotArmController.step()")

        if self._absolute:
            if self._board_ref is None:
                self._board_ref = cam[:3].copy()   # hand board-pos at press-8
            delta_board = cam[:3] - self._board_ref
            # Map the board-frame displacement into the robot frame via the
            # press-8 correspondence (falls back to _Rwb until it's captured on
            # the first oriented frame). This is what makes hand-forward ->
            # robot-forward regardless of how the board is yawed on the table.
            R_map = (self._R_board_to_robot
                     if self._R_board_to_robot is not None else self._Rwb)
            return self._home_site + self._abs_scale * (R_map @ delta_board)

        # cam = [image_x, image_y, depth_m]; depth from monocular hand-size in
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

        # PRESS-8 EMPIRICAL ALIGNMENT (camera-mounting / board-yaw independent).
        #
        # On the first frame after press-8 we snapshot the human palm frame
        # (palm_R_home) and pair it with the robot pinch_site frame at home
        # (R_site_home) to define ONE correspondence used for BOTH position and
        # orientation:
        #     R_board_to_robot = R_site_home @ palm_R_home^T
        # This maps a displacement/rotation expressed relative to your hand's
        # press-8 pose into the robot's frame relative to its home pose. Because
        # it's derived from the actual press-8 snapshot, it needs no measurement
        # of how the board sits vs the robot base and is immune to camera
        # mounting / MediaPipe-vs-OpenCV / board-yaw (all of which previously made
        # motion map to the wrong axes — the "depth goes toward camera" and
        # "rotation about wrong axis" bugs).
        if (palm_R is not None and self._palm_R_home is None
                and self._R_site_home is not None):
            self._palm_R_home = palm_R.copy()
            self._R_board_to_robot = self._R_site_home @ self._palm_R_home.T
            # Capture R_align so the ABSOLUTE orientation target equals the
            # robot's current wrist pose at press-8 (no jump), while remaining
            # absolute afterwards. Solve R_des(palm_R_home)=R_site_home for
            # R_align in R_des = R_mp_to_robot @ palm_R @ R_align:
            #   R_align = palm_R_home^T @ R_mp_to_robot^T @ R_site_home
            if self._R_cam_robot is not None:
                R_mp_to_cv = np.diag([1.0, -1.0, -1.0])
                R_mp_to_robot = self._R_cam_robot @ R_mp_to_cv
                self._R_align = (self._palm_R_home.T @ R_mp_to_robot.T
                                 @ self._R_site_home)

        pos_target = self._camera_to_world(cam_wrist)

        # ABSOLUTE orientation: the robot wrist mirrors the hand's ACTUAL
        # orientation in the world, not a delta from a home pose. palm_R is in
        # MediaPipe world axes (x-right, y-up, z-toward-cam); map it to robot
        # world axes:
        #   R_mp_to_robot = R_cam_robot @ R_mp_to_cv
        # where R_mp_to_cv=diag([1,-1,-1]) converts MediaPipe-world -> OpenCV-cam
        # (the frame R_cam_robot=WORLD_FROM_BOARD@R_world_cam expects). R_align is
        # a constant palm->pinch_site convention offset (default identity; tune
        # live so 'palm flat' maps to the intended wrist pose).
        #   R_des = R_mp_to_robot @ palm_R @ R_align
        # No home anchor -> the wrist snaps to match the hand orientation at
        # press-8 (accepted; makes the MuJoCo target actually mirror the cam hand).
        orientation = None
        if palm_R is not None and self._R_cam_robot is not None:
            R_mp_to_cv = np.diag([1.0, -1.0, -1.0])
            R_mp_to_robot = self._R_cam_robot @ R_mp_to_cv
            R_des = R_mp_to_robot @ palm_R @ self._R_align

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

    def reset(self) -> None:
        """Reset delta-tracking state (e.g., when hand tracking is lost).

        Clears both the position and orientation startup references so the next
        received frame re-snapshots home; axis_home (from init_home) is kept.
        """
        self._q_arm_prev  = None
        self._cam_home    = None
        self._board_ref   = None   # re-snap the absolute board zero on press-8
        self._palm_R_home = None
        self._R_board_to_robot = None   # re-derive press-8 frame map next frame
        self._R_align     = np.eye(3)   # re-capture press-8 orientation alignment

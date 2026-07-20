"""Top-level DexPilot teleoperation controller.

Connects ROS hand-tracking, DexPilot kinematic retargeting, and Gen3 arm IK
into a single step() call that returns a 23-DOF joint target each frame.

Typical usage inside a MuJoCo viewer loop:

    ctrl = DexPilotController(model, q_bias=Q_BIAS)
    ctrl.init_ros()
    ctrl.init_home(data)          # snapshot home-pose wrist position

    while viewer.is_running():
        ctrl.spin()               # process one ROS message
        q = ctrl.step(model, data)
        if q is not None:
            data.qpos[:N_ROBOT] = q
            data.qvel[:N_ROBOT] = 0.0
            mj.mj_forward(model, data)
        viewer.sync()
"""
from __future__ import annotations

import numpy as np
import mujoco as mj

from teleop.ros_interface            import ROSInterface
from teleop.dexpilot_retargeter      import DexPilotRetargeter
from teleop.dexpilot_arm_controller  import DexPilotArmController


class DexPilotController:
    """Integrates ROS hand tracking, DexPilot retargeting, and arm IK.

    Args:
        model:       MuJoCo model (composite arm + LEAP hand).
        q_bias:      Full 23-DOF bias; first n_arm entries used for arm IK.
        n_arm:       Arm DOF count (default 7).
        n_hand:      Hand DOF count (default 16).
        hand_alpha:  EMA factor applied to retargeted hand joints (0=max smooth).
        **arm_kwargs: Forwarded to DexPilotArmController (scale_x, scale_z, alpha…).
    """

    def __init__(
        self,
        model: mj.MjModel,
        q_bias: np.ndarray | None = None,
        n_arm: int = 7,
        n_hand: int = 16,
        hand_alpha: float = 0.3,
        debug: bool = False,
        eps: float | None = None,
        hand_tracking: bool = True,
        **arm_kwargs,
    ) -> None:
        self._n_arm   = n_arm
        self._n_hand  = n_hand
        self._n_robot = n_arm + n_hand
        self._hand_alpha = hand_alpha
        # When False, the LEAP fingers hold a fixed OPEN pose (q_bias hand joints)
        # instead of DexPilot retargeting — easier to read the hand orientation
        # during orientation debugging (fingers don't curl together).
        self._hand_tracking = hand_tracking
        self._hand_bias = (q_bias[n_arm:n_arm + n_hand].copy()
                           if q_bias is not None else np.zeros(n_hand))

        arm_bias = q_bias[:n_arm] if q_bias is not None else None

        self._ros    = ROSInterface()
        self._retarg = DexPilotRetargeter(model, n_arm=n_arm, debug=debug, eps=eps)
        self._arm    = DexPilotArmController(
            model,
            n_arm=n_arm,
            q_bias=arm_bias,
            **arm_kwargs,
        )

        self._q_hand_prev: np.ndarray | None = None
        self._active = False   # gated: tracking starts only after start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_ros(self) -> None:
        """Connect to ROS 2. Call before the simulation loop."""
        self._ros.init()

    def init_home(self, data: mj.MjData) -> None:
        """Snapshot the home wrist site position for delta-based arm tracking.

        Call once after the simulation is reset to the home/bias pose.
        """
        self._arm.init_home(data)

    def shutdown(self) -> None:
        self._ros.shutdown()

    # ------------------------------------------------------------------
    # Per-frame interface
    # ------------------------------------------------------------------

    def spin(self) -> None:
        """Process one pending ROS message (non-blocking, 1 ms timeout)."""
        self._ros.spin_once()

    @property
    def active(self) -> bool:
        """True once tracking has been started (via start())."""
        return self._active

    def start(self, data: mj.MjData) -> None:
        """(Re)zero tracking to the robot's current pose and enable tracking.

        Call on a key press. Snapshots the robot's current pinch_site position
        and approach axis as the new home, and clears the human-side startup
        references so the NEXT received frame captures your current hand pose /
        wrist orientation as home. Because tracking is delta-based, the robot
        holds still at the press instant and its wrist orientation is treated as
        matching yours; it then follows your movement/rotation from there.
        """
        self._arm.init_home(data)   # re-capture robot home pos + approach axis
        self._arm.reset()           # clear cam_home + palm_R_home -> re-snap next frame
        # Auto-calibrate orientation: the hand pose held at press-8 maps to the
        # robot's home wrist. Fixes the circular "match the moving wrist" problem.
        self._arm.request_orientation_calib()
        self._retarg.reset()
        self._q_hand_prev = None
        self._active = True

    def stop(self) -> None:
        """Disable tracking; the arm holds its last pose until start() again."""
        self._active = False

    def step(self, model: mj.MjModel, data: mj.MjData) -> np.ndarray | None:
        """Compute a 23-DOF joint target from the latest hand message.

        Returns None until start() has been called AND the first extended
        message (≥120 floats) has arrived.
        The message format expected (from ui/mediapipe_joint_angles.py):
          [0:3]   wrist position in camera space
          [57:120] 21 MediaPipe world landmarks × 3 coords (metres)
        """
        if not self._active:
            return None

        raw = self._ros.raw_msg
        if raw is None or len(raw) < 120:
            return None

        # --- Landmark blocks ---
        # World landmarks (metric, full 3D): fingertip POSITIONS and the palm FRAME
        # for finger retargeting. Image landmarks (raw[120:183], extended layout):
        # used ONLY for the ARM palm-orientation below (stable near edge-on).
        #
        # NOTE: fingertips deliberately use WORLD, not image, landmarks. Image
        # landmarks foreshorten finger extension into pseudodepth, which fails when
        # the fingers point toward/along the camera axis (verified: targets lose
        # their along-finger component and the optimiser curls the fingers into a
        # fist). The world 3D head captures that extension correctly.
        world_lm = np.array(raw[57:120], dtype=float).reshape(21, 3)
        image_lm = (np.array(raw[120:183], dtype=float).reshape(21, 3)
                    if len(raw) >= 183 else None)

        # --- Hand retargeting (16 DOF) — world landmarks (see note above) ---
        if self._hand_tracking:
            q_hand = self._retarg.retarget(world_lm)
            # EMA smoothing on hand joints
            if self._q_hand_prev is not None:
                q_hand = (self._hand_alpha * q_hand
                          + (1.0 - self._hand_alpha) * self._q_hand_prev)
            self._q_hand_prev = q_hand.copy()
        else:
            # Hold a fixed OPEN hand (bias pose) — no finger curling.
            q_hand = self._hand_bias

        # --- Arm: position + palm-orientation tracking (7 DOF) ---
        # Use the ROBOT-ALIGNED palm frame so an identical physical hand
        # orientation maps to an identical robot wrist orientation (no press-8
        # offset). Distinct from the finger-retargeting palm frame above.
        #
        # The arm palm frame is built from IMAGE landmarks (image_lm above), NOT the
        # world landmarks. The world-landmark depth flips the palm-NORMAL sign near
        # edge-on poses (verified via the dual-normal overlay); the image-landmark
        # frame stays stable. Falls back to world landmarks if the publisher didn't
        # send the image block (old message layout).
        cam_wrist      = np.array(raw[0:3], dtype=float)
        if image_lm is not None:
            # IMAGE landmarks live in a different BASIS than world landmarks:
            # image = {x-right, y-DOWN, z-pseudodepth}, world = {x-right, y-UP,
            # z-toward-cam} (what the downstream R_mp_to_cv=diag([1,-1,-1]) expects).
            #
            # DO NOT negate the input coordinates — negating a coordinate and then
            # taking cross products REFLECTS the frame (det -1), which an SVD
            # "nearest rotation" then silently repairs by flipping an ARBITRARY
            # axis, producing an inconsistent single-axis inversion (rotating +Y
            # showed as -Y in MuJoCo). Instead build the palm frame in the image
            # basis (already orthonormal — pure cross products + normalisation) and
            # re-express it in the world basis with a proper-rotation change of
            # basis C. Image->world is diag([1,-1,-1]) (det +1, the only Y-flip
            # variant that is a rotation not a reflection).
            _C            = np.diag([1.0, -1.0, -1.0])   # image basis -> world basis
            palm_R, _     = self._retarg.human_palm_frame_robot_aligned(image_lm)
            palm_R        = _C @ palm_R
        else:
            palm_R, _     = self._retarg.human_palm_frame_robot_aligned(world_lm)
        q_arm          = self._arm.step(cam_wrist, data, palm_R=palm_R)

        return np.concatenate([q_arm, q_hand])

    @property
    def retargeter(self) -> DexPilotRetargeter:
        """The finger-retargeting stage — exposed so a live tuner can mutate its
        tunable constants (BETA/GAMMA/EPS/ETA1/ETA2/S1_GAIN/S2_GAIN) per frame."""
        return self._retarg

    def poll_retarget_config(self) -> bool:
        """Hot-reload the retargeting constants from calibration/retarget_config.json
        when you edit + save it. Cheap (mtime check); call once per frame. Returns
        True on a reload. This is the text-entry tuning path: edit the JSON in your
        editor and the live retargeter picks it up."""
        return self._retarg.poll_config()

    def target_frame(self):
        """Last IK target pose (pos, R) in world coords — the frame the arm IK
        drives pinch_site toward. Passthrough to the arm controller for viz."""
        return self._arm.target_frame()

    def calibrate_orientation(self, data: mj.MjData) -> None:
        """Capture the constant orientation correction against the robot's actual
        wrist. Hold your hand to match the robot wrist, then call this."""
        self._arm.calibrate_orientation(data)

    def capture_calib_pose(self, data: mj.MjData) -> int:
        """Multi-pose calib: record one (hand, wrist) pair."""
        return self._arm.capture_calib_pose(data)

    def solve_calib(self) -> None:
        """Multi-pose calib: solve the full rotation correction from captures."""
        self._arm.solve_calib()

    def clear_calib(self) -> None:
        self._arm.clear_calib()

    def reset(self) -> None:
        """Reset all transient state (call when hand tracking is lost)."""
        self._retarg.reset()
        self._arm.reset()
        self._q_hand_prev = None

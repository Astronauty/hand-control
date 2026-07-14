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
        **arm_kwargs,
    ) -> None:
        self._n_arm   = n_arm
        self._n_hand  = n_hand
        self._n_robot = n_arm + n_hand
        self._hand_alpha = hand_alpha

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

        # --- Hand retargeting (16 DOF) ---
        world_lm = np.array(raw[57:120], dtype=float).reshape(21, 3)
        q_hand   = self._retarg.retarget(world_lm)

        # EMA smoothing on hand joints
        if self._q_hand_prev is not None:
            q_hand = (self._hand_alpha * q_hand
                      + (1.0 - self._hand_alpha) * self._q_hand_prev)
        self._q_hand_prev = q_hand.copy()

        # --- Arm: position + palm-orientation tracking (7 DOF) ---
        cam_wrist      = np.array(raw[0:3], dtype=float)
        palm_R, _      = self._retarg.human_palm_frame(world_lm)  # (3,3) world-from-palm
        q_arm          = self._arm.step(cam_wrist, data, palm_R=palm_R)

        return np.concatenate([q_arm, q_hand])

    def target_frame(self):
        """Last IK target pose (pos, R) in world coords — the frame the arm IK
        drives pinch_site toward. Passthrough to the arm controller for viz."""
        return self._arm.target_frame()

    def reset(self) -> None:
        """Reset all transient state (call when hand tracking is lost)."""
        self._retarg.reset()
        self._arm.reset()
        self._q_hand_prev = None

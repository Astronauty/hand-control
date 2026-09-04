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

import threading

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
        retargeter: str = "dexpilot",
        pinch_debounce: bool = True,
        output_ema: bool = True,
        **arm_kwargs,
    ) -> None:
        self._n_arm   = n_arm
        self._n_hand  = n_hand
        self._n_robot = n_arm + n_hand
        # Independent noisy-input filter toggles (see --pinch-debounce / --output-ema).
        #   pinch_debounce: median + hysteresis + N-frame on the pinch DECISION, in the
        #     retargeter (passed through below). False → raw per-finger threshold.
        #   output_ema: EMA smoothing on the 16 hand joints here. False → hand_alpha=1.0
        #     (use the raw solved q_hand each frame, no smoothing).
        self._output_ema = bool(output_ema)
        self._hand_alpha = hand_alpha if self._output_ema else 1.0
        # When False, the LEAP fingers hold a fixed OPEN pose (q_bias hand joints)
        # instead of DexPilot retargeting — easier to read the hand orientation
        # during orientation debugging (fingers don't curl together).
        self._hand_tracking = hand_tracking
        self._hand_bias = (q_bias[n_arm:n_arm + n_hand].copy()
                           if q_bias is not None else np.zeros(n_hand))

        arm_bias = q_bias[:n_arm] if q_bias is not None else None

        self._ros    = ROSInterface()
        # Finger-retargeting backend. Default 'dexpilot' keeps the original hand-rolled
        # retargeter (and this import path) untouched; 'anyteleop' lazily loads the
        # separable dex-retargeting backend (see anyteleop/). Guarded so a missing
        # anyteleop package never affects the default path.
        if str(retargeter).lower() == "dexpilot":
            self._retarg = DexPilotRetargeter(model, n_arm=n_arm, debug=debug, eps=eps,
                                              pinch_debounce=pinch_debounce)
        else:
            from anyteleop.factory import make_retargeter
            self._retarg = make_retargeter(retargeter, model, n_arm=n_arm,
                                           debug=debug, eps=eps,
                                           pinch_debounce=pinch_debounce)
        self._arm    = DexPilotArmController(
            model,
            n_arm=n_arm,
            q_bias=arm_bias,
            **arm_kwargs,
        )

        self._q_hand_prev: np.ndarray | None = None
        self._active = False   # gated: tracking starts only after start()

        # --- Off-thread finger retargeting -------------------------------------
        # The DexPilot finger solve is a ~40 ms scipy SLSQP (see DexPilotRetargeter.
        # retarget); running it inline in step() every teleop loop iteration stalled
        # the pre-lock-in loop to ~25 Hz (the "laggy wrist when the recommender is
        # running" report — the NLP was NOT the cause). Mirror the arm IK's async
        # pattern: a daemon worker owns the SLSQP and solves from the LATEST posted
        # landmarks; step() posts the new landmarks and returns the most recently
        # solved q_hand (one-frame latency, standard for async retargeting). All
        # shared state below is guarded by _hand_lock; _hand_evt wakes the worker.
        # EMA smoothing stays on the control thread (against the last RETURNED q_hand)
        # so it's independent of solve cadence.
        self._hand_lock    = threading.Lock()
        self._hand_evt     = threading.Event()
        self._hand_pending: np.ndarray | None = None      # world_lm (21,3) to solve
        self._hand_result:  np.ndarray | None = None       # latest raw solved q_hand (16,)
        self._hand_stop    = threading.Event()
        self._hand_thread  = threading.Thread(
            target=self._hand_worker, name="dexpilot-finger-retarget", daemon=True)
        self._hand_thread.start()

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
        self._arm.close()   # stop the background IK worker thread
        self._hand_stop.set()          # stop the finger-retarget worker thread
        self._hand_evt.set()           # unblock its wait so it sees the stop flag
        if self._hand_thread.is_alive():
            self._hand_thread.join(timeout=1.0)
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

    @property
    def raw_msg(self) -> list | None:
        """Latest raw /hand/joint_angles payload (or None). Exposed so callers can
        visualise the world landmarks (raw[57:120]) without reaching into ._ros."""
        return self._ros.raw_msg

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
        # Drop any stale pre-reset finger solve so tracking re-zeros cleanly (the
        # worker re-solves from the next posted landmarks).
        with self._hand_lock:
            self._hand_pending = None
            self._hand_result = None
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
            # Post the latest landmarks to the finger worker (owns the SLSQP) and
            # take the most recently solved q_hand WITHOUT blocking — the ~40 ms solve
            # runs off-thread so step() stays cheap. One-frame latency; until the
            # worker has produced its first solution, hold the open bias pose.
            with self._hand_lock:
                self._hand_pending = world_lm
                q_solved = (None if self._hand_result is None
                            else self._hand_result.copy())
            self._hand_evt.set()
            if q_solved is None:
                q_hand = (self._hand_bias if self._q_hand_prev is None
                          else self._q_hand_prev)
            else:
                # EMA smoothing on hand joints (on the control thread, against the
                # last RETURNED q_hand — independent of the worker's solve cadence).
                if self._q_hand_prev is not None:
                    q_hand = (self._hand_alpha * q_solved
                              + (1.0 - self._hand_alpha) * self._q_hand_prev)
                else:
                    q_hand = q_solved
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

        # The arm IK now solves on a BACKGROUND thread (see DexPilotArmController)
        # and returns None until the worker has produced its first solution (the
        # first frame or two after press-8 / reset). Propagate that None — the
        # teleop loop already holds the last pose when step() returns None — rather
        # than concatenating it (np.concatenate([None, ...]) crashes on a 0-d array).
        if q_arm is None:
            return None
        return np.concatenate([q_arm, q_hand])

    def _hand_worker(self) -> None:
        """Background loop: retarget fingers from the latest posted landmarks, forever.

        Blocks on _hand_evt until step() posts world landmarks, then runs the SLSQP
        retarget and publishes q_hand under _hand_lock. Only ever solves the MOST
        RECENT landmarks — if several arrive while one solve is in flight, the stale
        ones are skipped (always chase the freshest hand pose, never a backlog).
        Exits when _hand_stop is set (see shutdown()). A bad solve is swallowed so a
        single failure can't kill the worker."""
        while not self._hand_stop.is_set():
            # Wait for landmarks; time out periodically so a set _hand_stop is noticed.
            if not self._hand_evt.wait(timeout=0.1):
                continue
            self._hand_evt.clear()
            with self._hand_lock:
                world_lm = self._hand_pending
                self._hand_pending = None
            if world_lm is None:
                continue
            try:
                q = self._retarg.retarget(world_lm)
            except Exception as e:   # noqa: BLE001 — a bad solve must not kill the worker
                print(f"[hand] finger retarget worker error (skipping): {e!r}")
                continue
            with self._hand_lock:
                self._hand_result = q.copy()

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

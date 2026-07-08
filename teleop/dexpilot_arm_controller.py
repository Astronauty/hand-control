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

import numpy as np
import mujoco as mj

from grasp_control import SpatialIKSolver

# Pinch-site local +z points world +X (forward) at HOME_ARM (verified via FK).
# This is the "approach direction" axis we align with the human palm normal.
_PALM_LOCAL_AXIS = np.array([0.0, 0.0, 1.0])

# Default camera → robot world rotation.
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
        R_cam_robot: np.ndarray | None = _DEFAULT_R_CAM_ROBOT,
    ) -> None:
        self._model       = model
        self._n_arm       = n_arm
        self._q_bias      = q_bias if q_bias is not None else np.zeros(n_arm)
        self._alpha       = alpha
        self._scale_x     = scale_x
        self._scale_z     = scale_z
        self._R_cam_robot = R_cam_robot

        self._site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, palm_site)
        # n_arm only: pinch_site depends solely on the 7 arm joints.
        # n_robot=23 caused a shape mismatch (q_bias[7] vs q[23]) in the
        # null-space projection inside SpatialIKSolver.solve().
        self._ik      = SpatialIKSolver(n_robot=n_arm)

        # Scratch MjData so the IK iterations don't affect the main sim state
        self._scratch = mj.MjData(model)

        self._q_arm_prev: np.ndarray | None = None
        self._cam_home:   np.ndarray | None = None   # first received camera wrist xy
        self._home_site:  np.ndarray | None = None   # pinch_site world pos at startup
        self._palm_R_home: np.ndarray | None = None  # human palm frame (robot world) at startup
        self._axis_home:   np.ndarray | None = None  # pinch_site approach axis (world) at startup

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_home(self, data: mj.MjData) -> None:
        """Snapshot the current pinch_site position as the arm home reference.

        Call once after the simulation is reset to the home/bias pose.
        """
        mj.mj_forward(self._model, data)
        self._home_site = data.site_xpos[self._site_id].copy()
        # Robot approach axis (world) at HOME_ARM: R_site @ local_axis.
        # Orientation tracking is delta-based off this reference so the arm
        # holds its home pose until the human wrist actually rotates.
        R_site = data.site_xmat[self._site_id].reshape(3, 3)
        self._axis_home = R_site @ _PALM_LOCAL_AXIS
        self._scratch.qpos[:] = data.qpos[:]

    # ------------------------------------------------------------------
    # Camera-to-world mapping
    # ------------------------------------------------------------------

    def _camera_to_world(self, cam: np.ndarray) -> np.ndarray:
        """Map camera-space wrist position to a robot world-frame position target.

        Delta-based: the first call records the home reference so the arm starts
        at its current pose and moves relative to the user's initial hand position.
        """
        if self._home_site is None:
            raise RuntimeError(
                "init_home() must be called before DexPilotArmController.step()")

        cam_xy = cam[:2]
        if self._cam_home is None:
            self._cam_home = cam_xy.copy()

        delta = cam_xy - self._cam_home
        return self._home_site + np.array([
            -delta[0] * self._scale_x,   # image x → robot x (negated: mirror)
             0.0,                          # robot y held at home (no depth yet)
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

        # Build a DELTA-based orientation target from the human palm rotation.
        #
        # Absolute mapping (old behaviour) forced pinch_site's approach axis to
        # point at R_cam_robot @ palm_normal regardless of the robot's startup
        # pose, so the arm twisted away from HOME_ARM the instant tracking began.
        #
        # Instead we snapshot the human palm frame at startup (palm_R_home) and
        # the robot approach axis at HOME_ARM (axis_home). Each frame we compute
        # the human's rotation SINCE startup, express it in the robot world via
        # R_cam_robot, and rotate axis_home by it. At t=0 the delta is identity,
        # so the target equals the current axis -> zero correction; the arm only
        # rotates as the human wrist rotates. This also makes tracking robust to
        # an imperfect R_cam_robot, since only the *change* in palm frame matters.
        orientation = None
        if (palm_R is not None and self._R_cam_robot is not None
                and self._axis_home is not None):
            if self._palm_R_home is None:
                self._palm_R_home = palm_R.copy()

            # Human rotation since startup, in the human world frame:
            #   R_delta_cam = palm_R @ palm_R_home^T
            R_delta_cam = palm_R @ self._palm_R_home.T
            # Same rotation expressed in the robot world frame (similarity xf):
            #   R_delta_robot = R_cam_robot @ R_delta_cam @ R_cam_robot^T
            R_delta_robot = self._R_cam_robot @ R_delta_cam @ self._R_cam_robot.T

            target_axis = R_delta_robot @ self._axis_home
            norm = np.linalg.norm(target_axis)
            if norm > 1e-6:
                orientation = (_PALM_LOCAL_AXIS, target_axis / norm)

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

    def reset(self) -> None:
        """Reset delta-tracking state (e.g., when hand tracking is lost).

        Clears both the position and orientation startup references so the next
        received frame re-snapshots home; axis_home (from init_home) is kept.
        """
        self._q_arm_prev  = None
        self._cam_home    = None
        self._palm_R_home = None

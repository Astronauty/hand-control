"""
teleop_manager.py
=================
TAMP-aware teleop pipeline.

Architecture
------------

  MediaPipe (ROS topic: /hand/joint_angles)
       │
       │  raw landmark data
       ▼
  parse_mediapipe_to_q()
       │
       │  q_teleop  ← what the human hand says to do
       ▼
  ApproachController.get_collision_free_joint_angles()   ← your existing node
       │
       │  q_safe    ← nearest collision-free joint config
       ▼
  MuJoCo simulation   (TRACKING state)
       │
       │  user presses G → GraspPlanner.solve() runs in background thread
       ▼
  GraspPlanner returns result  → MuJoCo viewer shows proposed grasp
       │
       │  user presses Y (accept) or N (reject) in viewer
       ▼
  GRASPING state: robot holds confirmed q / p1 / p2 as fixed reference


TAMP State Machine
------------------
  IDLE        waiting for calibration  (C key or ROS trigger)
  TRACKING    approach phase – ApproachController drives robot from MediaPipe
  CONFIRMING  GraspPlanner finished – viewer shows proposed grasp for review
  GRASPING    confirmed grasp held; MediaPipe input frozen

Key bindings (MuJoCo viewer must be focused)
--------------------------------------------
  C   calibrate wrist offset            (also triggered by ROS Empty topic)
  G   run GraspPlanner now              (only in TRACKING state)
  Y   accept proposed grasp             (only in CONFIRMING state)
  N   reject proposed grasp → TRACKING  (only in CONFIRMING state)
  R   reset to TRACKING from any state
"""

from __future__ import annotations

import threading
import time

import casadi as ca
from casadi import Callback
import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Empty, Float32MultiArray

# ── GraspPlanner import ───────────────────────────────────────────────────────
from grasp_planner import GraspConfig, GraspPlanner


# ─────────────────────────────────────────────────────────────────────────────
# Signed-distance CasADi callback  (verbatim from approach_controller.py)
# ─────────────────────────────────────────────────────────────────────────────

class SignedDistanceConstraint(Callback):
    """mj_geomDistance(geom1, geom2) as a CasADi Callback."""

    def __init__(self, name, model, data, geom_id1, geom_id2, opts={}):
        Callback.__init__(self)
        self.model    = model
        self.data     = data
        self.geom_id1 = geom_id1
        self.geom_id2 = geom_id2
        self.fromto   = np.zeros(6, dtype=np.float64)
        self.construct(name, opts)

    def get_n_in(self):  return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.model.nq, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(1, 1)

    def has_jacobian(self): return False

    def eval(self, arg):
        q = np.array(arg[0]).flatten()
        self.data.qpos[:self.model.nq] = q
        mujoco.mj_forward(self.model, self.data)
        dist = mujoco.mj_geomDistance(
            self.model, self.data,
            self.geom_id1, self.geom_id2,
            1000.0, self.fromto)
        return [dist]


# ─────────────────────────────────────────────────────────────────────────────
# ApproachController  (refactored from approach_controller.py)
#
# Changes from original:
#   • No longer a ROS Node or viewer — just a pure computation class.
#   • Uses a *separate* MjData copy so it never corrupts the simulation data.
#   • get_collision_free_joint_angles() returns the best-effort q on failure
#     instead of crashing.
# ─────────────────────────────────────────────────────────────────────────────

class ApproachController:
    """
    Given a reference q from MediaPipe teleop, returns the nearest q that
    keeps the finger distal geoms outside the object geom by at least
    `clearance` metres.

    Parameters
    ----------
    model     : mujoco.MjModel
    clearance : minimum signed distance between finger and object (m)
    max_iter  : IPOPT iteration limit (keep small for real-time use)
    """

    # Geom names that must exist in the XML
    _THUMB_GEOM = "right_thumb_distal_geom"
    _INDEX_GEOM = "right_index_distal_geom"
    _OBJ_GEOM   = "obj1_geom"

    def __init__(self,
                 model:     mujoco.MjModel,
                 clearance: float = 1e-1,
                 max_iter:  int   = 1000):
        self.model     = model
        self.clearance = clearance
        self.max_iter  = max_iter

        # Dedicated data copy — approach controller never shares state
        self._data = mujoco.MjData(model)
        mujoco.mj_forward(model, self._data)

        self._thumb_gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, self._THUMB_GEOM)
        self._index_gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, self._INDEX_GEOM)
        self._obj_gid   = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, self._OBJ_GEOM)

        call_opts = {"enable_fd": True}
        self._dist_index = SignedDistanceConstraint(
            "ac_dist_index", model, self._data,
            self._index_gid, self._obj_gid, call_opts)
        self._dist_thumb = SignedDistanceConstraint(
            "ac_dist_thumb", model, self._data,
            self._thumb_gid, self._obj_gid, call_opts)

        self._solver_opts = {
            "print_time": False,
            "ipopt": {
                "jacobian_approximation": "finite-difference-values",
                "hessian_approximation":  "limited-memory",
                "print_level": 0,
                "sb": "yes",
                "max_iter": self.max_iter,
            },
            "ad_weight_sp": 0,
        }

    def sync_state(self, live_data: mujoco.MjData) -> None:
        """Copy the live simulation qpos/qvel into this controller's data."""
        self._data.qpos[:] = live_data.qpos[:]
        self._data.qvel[:] = live_data.qvel[:]
        mujoco.mj_forward(self.model, self._data)

    def get_collision_free_joint_angles(self, q_ref: np.ndarray) -> np.ndarray:
        """
        Solve for the nearest q to q_ref that satisfies the collision
        clearance constraint.  Falls back to q_ref on failure.
        """
        nq   = self.model.nq
        opti = ca.Opti()
        q    = opti.variable(nq)

        cost = ca.bilin(ca.diag(np.ones(nq)), q - q_ref, q - q_ref)
        opti.minimize(cost)
        opti.subject_to(self._dist_index(q) >= self.clearance)
        # Note: original only constrained index; uncomment thumb if needed:
        # opti.subject_to(self._dist_thumb(q) >= self.clearance)

        opti.solver('ipopt', self._solver_opts)
        opti.set_initial(q, q_ref)

        try:
            sol = opti.solve()
            return sol.value(q)
        except Exception as e:
            try:
                return opti.debug.value(q)   # best-effort
            except Exception:
                return q_ref.copy()          # fallback: pass through unchanged


# ─────────────────────────────────────────────────────────────────────────────
# FK callback  (kept from original teleop_manager for joint tracking)
# ─────────────────────────────────────────────────────────────────────────────

class Optimization2DFKCallback(ca.Callback):
    """Forward-kinematics callback for optimization with finite differences."""

    def __init__(self, name, model, data, body_id, actuated_indices, opts={}):
        ca.Callback.__init__(self)
        self.model            = model
        self.data             = data
        self.body_id          = body_id
        self.actuated_indices = actuated_indices
        self.construct(name, opts)

    def get_n_in(self):  return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(len(self.actuated_indices), 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(2, 1)

    def has_jacobian(self): return False

    def eval(self, arg):
        q_actuated = np.array(arg[0]).flatten()
        q_full = self.data.qpos.copy()
        for idx, val in zip(self.actuated_indices, q_actuated):
            q_full[idx] = val
        self.data.qpos[:] = q_full
        mujoco.mj_kinematics(self.model, self.data)
        return [self.data.xpos[self.body_id][:2].copy()]


# ─────────────────────────────────────────────────────────────────────────────
# TAMP Node
# ─────────────────────────────────────────────────────────────────────────────

class TeleopTAMPNode(Node):
    """
    ROS 2 node implementing the full TAMP pipeline.

    Subscribes to MediaPipe joint angles, runs the ApproachController every
    step, and launches GraspPlanner in a background thread on demand.
    The MuJoCo passive viewer is the confirmation interface.
    """

    def __init__(self, model_path: str):
        super().__init__('teleop_tamp')

        # ── MuJoCo ───────────────────────────────────────────────────────────
        self.model    = mujoco.MjModel.from_xml_path(model_path)
        self.data     = mujoco.MjData(self.model)
        self.data_fk  = mujoco.MjData(self.model)  # FK opt uses this copy
        mujoco.mj_step(self.model, self.data)

        # ── TAMP state ────────────────────────────────────────────────────────
        # IDLE → TRACKING → CONFIRMING → GRASPING
        self._state      = "IDLE"
        self._state_lock = threading.Lock()

        # Calibration
        self._human_wrist_start: np.ndarray | None = None
        self._robot_wrist_start: np.ndarray | None = None
        self._virtual_offset:    np.ndarray | None = None

        # MediaPipe
        self._current_human_wrist: np.ndarray | None = None
        self._current_raw_msg = None

        # Confirmed grasp (set when user accepts)
        self.confirmed_q:  np.ndarray | None = None
        self.confirmed_p1: np.ndarray | None = None
        self.confirmed_p2: np.ndarray | None = None

        # Pending result (shown during CONFIRMING)
        self._pending_result: dict | None = None
        self._planning_thread: threading.Thread | None = None

        # ── sub-systems ───────────────────────────────────────────────────────
        self._actuated_indices = [
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ]

        self.approach_ctrl = ApproachController(
            self.model, clearance=1e-1, max_iter=1000)

        self.grasp_planner = GraspPlanner(
            self.model, self.data,
            cfg=GraspConfig(),
        )

        self._setup_fk_callbacks()

        # ── loop counters ─────────────────────────────────────────────────────
        self._opt_counter   = 0
        self._opt_frequency = 5   # run approach IK every N steps

        # ── ROS ───────────────────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, '/hand/joint_angles', self._hand_cb, 10)
        self.create_subscription(
            Empty, '/teleop/trigger_calibration', self._calibrate_cb, 10)
        self.target_pub = self.create_publisher(
            PoseStamped, '/robot/target_pose', 10)

        self.get_logger().info(
            "\nTeleopTAMPNode ready.\n"
            "Controls (MuJoCo viewer must be focused):\n"
            "  C  calibrate wrist offset\n"
            "  G  plan grasp now  (while TRACKING)\n"
            "  Y  accept proposed grasp  (while CONFIRMING)\n"
            "  N  reject proposed grasp  (while CONFIRMING)\n"
            "  R  reset to TRACKING\n"
        )

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new: str) -> None:
        with self._state_lock:
            old = self._state
            self._state = new
        self.get_logger().info(f"[TAMP] {old} → {new}")

    # ── viewer key callback ───────────────────────────────────────────────────

    def key_callback(self, keycode: int) -> None:
        """Register with launch_passive as key_callback=node.key_callback."""
        s = self.state

        if keycode == ord('C'):
            self._do_calibrate()

        elif keycode == ord('G') and s == "TRACKING":
            self._trigger_grasp_planning()

        elif keycode == ord('Y') and s == "CONFIRMING":
            self._accept_grasp()

        elif keycode == ord('N') and s == "CONFIRMING":
            self._reject_grasp()

        elif keycode == ord('R'):
            self._reset()

    # ── main control loop ─────────────────────────────────────────────────────

    def update_control(self, viewer) -> None:
        """Call this every tick inside the viewer loop."""
        mujoco.mj_step(self.model, self.data)

        s = self.state

        if s == "TRACKING":
            self._run_approach_step()

        elif s == "CONFIRMING":
            # Hold the proposed configuration so user can inspect it
            if self._pending_result and self._pending_result.get('q') is not None:
                self.grasp_planner.show_in_viewer(
                    self.model, self.data, self._pending_result)
            viewer.sync()

        elif s == "GRASPING":
            # Lock to the confirmed joint configuration
            if self.confirmed_q is not None:
                for i, idx in enumerate(self._actuated_indices):
                    self.data.qpos[idx] = self.confirmed_q[i]
                mujoco.mj_forward(self.model, self.data)

    # ── TRACKING step ─────────────────────────────────────────────────────────

    def _run_approach_step(self) -> None:
        if self._virtual_offset is None or self._current_human_wrist is None:
            return

        # 1. Compute target base from MediaPipe wrist position
        human_curr  = self._camera_to_robot(self._current_human_wrist)
        target_base = human_curr + self._virtual_offset

        # 2. Parse full q_teleop from MediaPipe finger angles
        q_teleop = self._parse_mediapipe_to_q(target_base)

        # 3. Run collision-avoidance IK at reduced frequency
        self._opt_counter += 1
        if self._opt_counter % self._opt_frequency == 0:
            self.approach_ctrl.sync_state(self.data)
            q_safe = self.approach_ctrl.get_collision_free_joint_angles(q_teleop)
        else:
            q_safe = q_teleop

        # 4. Apply to simulation
        for i, idx in enumerate(self._actuated_indices):
            self.data.qpos[idx] = q_safe[i]
        self.data.ctrl[:] = q_safe
        mujoco.mj_forward(self.model, self.data)

        # 5. Publish for any ROS visualisers
        self._publish_target(target_base)

    # ── grasp planning ────────────────────────────────────────────────────────

    def _trigger_grasp_planning(self) -> None:
        """Launch GraspPlanner in a daemon thread.  Non-blocking."""
        if self._planning_thread and self._planning_thread.is_alive():
            self.get_logger().warning(
                "Grasp planner already running – ignoring duplicate request.")
            return

        # Snapshot the current state so the thread has stable inputs
        q_snap = np.array([self.data.qpos[i] for i in self._actuated_indices])
        obj_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY,
            self.grasp_planner.cfg.obj_body)
        obj_pos = self.data.xpos[obj_bid][:2].copy()

        self.get_logger().info(
            f"[TAMP] GraspPlanner starting  "
            f"q={np.round(q_snap,3)}  obj_pos={np.round(obj_pos,4)}")

        def _plan():
            result = self.grasp_planner.solve(q_snap, obj_pos)
            iters  = result.get('iterations', 'N/A')
            cost   = f"{result['cost']:.5f}" if result['cost'] else 'N/A'
            self.get_logger().info(
                f"[TAMP] Planner done: {result['status']}  "
                f"iters={iters}  cost={cost}")
            self.grasp_planner.verify(result)

            self._pending_result = result
            self.state = "CONFIRMING"
            self.get_logger().info(
                "[TAMP] Proposed grasp shown in viewer.\n"
                "  Press Y to ACCEPT, N to REJECT.")

        self._planning_thread = threading.Thread(target=_plan, daemon=True)
        self._planning_thread.start()

    # ── confirmation actions ──────────────────────────────────────────────────

    def _accept_grasp(self) -> None:
        r = self._pending_result
        if r is None or r.get('q') is None:
            self.get_logger().warning("[TAMP] No valid result to accept.")
            return

        self.confirmed_q  = r['q'].copy()
        self.confirmed_p1 = r['p1'].copy()
        self.confirmed_p2 = r['p2'].copy()
        self.state = "GRASPING"
        self.get_logger().info(
            f"[TAMP] Grasp ACCEPTED.\n"
            f"  p1={np.round(self.confirmed_p1,4)}\n"
            f"  p2={np.round(self.confirmed_p2,4)}"
        )

    def _reject_grasp(self) -> None:
        self.get_logger().info("[TAMP] Grasp rejected.")
        self._pending_result = None
        self.state = "TRACKING"

    def _reset(self) -> None:
        self.get_logger().info("[TAMP] Manual reset.")
        self.confirmed_q  = None
        self.confirmed_p1 = None
        self.confirmed_p2 = None
        self._pending_result = None
        self.state = "TRACKING" if self._virtual_offset is not None else "IDLE"

    # ── calibration ───────────────────────────────────────────────────────────

    def _do_calibrate(self) -> None:
        if self._current_human_wrist is None:
            self.get_logger().warning("[TAMP] No hand detected, cannot calibrate.")
            return
        self._human_wrist_start = self._current_human_wrist.copy()
        self._robot_wrist_start = self._get_robot_wrist()
        human_scaled            = self._camera_to_robot(self._human_wrist_start)
        self._virtual_offset    = self._robot_wrist_start - human_scaled
        self.state = "TRACKING"
        self.get_logger().info(
            f"[TAMP] Calibrated.  offset={np.round(self._virtual_offset,4)}")

    def _calibrate_cb(self, msg) -> None:
        self._do_calibrate()

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _hand_cb(self, msg) -> None:
        self._current_raw_msg = msg
        if len(msg.data) >= 3:
            self._current_human_wrist = np.array(msg.data[0:3])

    # ── MediaPipe → q_teleop ──────────────────────────────────────────────────

    def _parse_mediapipe_to_q(self, target_base: np.ndarray) -> np.ndarray:
        """
        Parse the raw MediaPipe message into actuated joint angles.

        Message format (from original teleop_manager):
            [wrist(6 values) | thumb_CMC(3), thumb_MCP(3), thumb_IP(3) |
             index_MCP(3), index_PIP(3), index_DIP(3) | ...]
            Each joint: [Yaw(Z), Pitch(Y), Roll(X)]  in degrees.

        Returns q_teleop as (nu,) array, falling back to current q on error.
        """
        q_current = np.array(
            [self.data.qpos[i] for i in self._actuated_indices])

        if self._current_raw_msg is None:
            return q_current

        raw = self._current_raw_msg.data

        def get_flexion(joint_idx: int) -> float:
            start = 6 + joint_idx * 3   # skip wrist (6 values)
            if start >= len(raw):
                return 0.0
            return np.radians(float(raw[start]))  # index 0 = Z-rotation (flexion)

        # Thumb: CMC=0, MCP=1, IP=2
        th_mcp = get_flexion(1)
        th_pip = get_flexion(2)
        th_dip = get_flexion(2) * 0.5  # DIP approximated from IP

        # Index: MCP=3, PIP=4, DIP=5
        in_mcp = get_flexion(3)
        in_pip = get_flexion(4)
        in_dip = get_flexion(5)

        # XML actuator order:
        # [Th_base_x, Th_base_y, Index_MCP, Index_PIP, Index_DIP,
        #  Thumb_MCP, Thumb_PIP, Thumb_DIP]
        q = q_current.copy()
        if len(q) >= 8:
            q[0] =  target_base[0]   # base X
            q[1] =  target_base[1]   # base Y
            q[2] = -in_mcp
            q[3] = -in_pip
            q[4] = -in_dip
            q[5] =  th_mcp
            q[6] =  th_pip
            q[7] =  th_dip
        return q

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_robot_wrist(self) -> np.ndarray:
        return np.array([self.data.qpos[0], self.data.qpos[1], 0.0])

    def _camera_to_robot(self, camera_pos: np.ndarray) -> np.ndarray:
        """
        Map MediaPipe camera axes to robot planar axes.
        Camera: X (Right), Y (Down), Z (Depth)
        Robot:  X (Right), Y (Forward)
        """
        SCALE_X, SCALE_Y = 2.0, 2.5
        return np.array([
            camera_pos[0] * SCALE_X,
            camera_pos[2] * SCALE_Y,   # depth → forward
            0.0,
        ])

    def _publish_target(self, target: np.ndarray) -> None:
        msg = PoseStamped()
        msg.header.frame_id    = "world"
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.pose.position.x    = float(target[0])
        msg.pose.position.y    = float(target[1])
        msg.pose.position.z    = 0.0
        self.target_pub.publish(msg)

    def _setup_fk_callbacks(self) -> None:
        """Build FK callbacks for the joint-tracking cost (kept for future use)."""
        joint_bodies = {
            'thumb_mcp': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            "right_thumb_proximal_link"),
            'thumb_pip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            "right_thumb_medial_link"),
            'thumb_dip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            "right_thumb_distal_link"),
            'index_mcp': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            "right_index_proximal_link"),
            'index_pip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            "right_index_medial_link"),
            'index_dip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            "right_index_distal_link"),
        }
        call_opts = {"enable_fd": True}
        self.fk_callbacks = {
            name: Optimization2DFKCallback(
                f'fk_{name}', self.model, self.data_fk,
                bid, self._actuated_indices, call_opts)
            for name, bid in joint_bodies.items() if bid != -1
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = TeleopTAMPNode(r"models/planar_two_finger_manipulator.xml")

    with mujoco.viewer.launch_passive(
            node.model, node.data,
            key_callback=node.key_callback) as viewer:

        while viewer.is_running():
            rclpy.spin_once(node, timeout_sec=0.001)
            node.update_control(viewer)
            viewer.sync()
            time.sleep(node.model.opt.timestep)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
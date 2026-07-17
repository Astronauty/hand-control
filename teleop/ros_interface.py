"""Thin ROS 2 subscriber for /hand/joint_angles (Float32MultiArray).

Mirrors the _init_ros / on_hand_message / spin_ros_once pattern from
simulation/tamp_manager.py, extracted so DexPilotController can reuse it
without pulling in the entire TAMP stack.
"""
from __future__ import annotations

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


class ROSInterface:
    """Subscribes to /hand/joint_angles and stores the latest message."""

    def __init__(self) -> None:
        self._raw_msg: list | None = None
        self._current_wrist: np.ndarray | None = None
        self._retarget_params: list | None = None   # latest /hand/retarget_params
        self._ros_node = None

    def init(self) -> None:
        """Initialize ROS 2 and create the hand-angles subscription."""
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                "rclpy not available — cannot initialize ROS 2 interface")
        rclpy.init()
        self._ros_node = Node("dexpilot")
        self._ros_node.create_subscription(
            Float32MultiArray, "/hand/joint_angles",
            lambda msg: self._on_hand_message(list(msg.data)), 10)
        # Live finger-retargeting tunables from the MediaPipe window's sliders
        # (teleop/retarget_tuner.py). 8 floats: 7 params + trailing save flag.
        from teleop.retarget_tuner import RETARGET_PARAM_TOPIC
        self._ros_node.create_subscription(
            Float32MultiArray, RETARGET_PARAM_TOPIC,
            lambda msg: self._on_retarget_params(list(msg.data)), 10)

    def _on_hand_message(self, data_list: list) -> None:
        self._raw_msg = data_list
        if len(data_list) >= 3:
            self._current_wrist = np.array(data_list[0:3], float)

    def _on_retarget_params(self, data_list: list) -> None:
        # Store as-is; the controller validates length and applies it. Held until
        # consumed (consume_retarget_params) so a slower sim loop can't miss the
        # momentary save flag.
        self._retarget_params = data_list

    def consume_retarget_params(self) -> list | None:
        """Return the latest retarget-param message and clear it, so each message
        (incl. its momentary save flag) is applied exactly once."""
        p = self._retarget_params
        self._retarget_params = None
        return p

    def spin_once(self) -> None:
        """Process one pending ROS message (non-blocking, 1 ms timeout)."""
        if self._ros_node is not None:
            rclpy.spin_once(self._ros_node, timeout_sec=0.001)

    def shutdown(self) -> None:
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            try:
                rclpy.shutdown()
            except Exception:
                pass

    @property
    def raw_msg(self) -> list | None:
        return self._raw_msg

    @property
    def current_wrist(self) -> np.ndarray | None:
        return self._current_wrist

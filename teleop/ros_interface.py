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
        self._ros_node = None

    def init(self) -> None:
        """Initialize ROS 2 and create the hand-angles subscription."""
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                "rclpy not available — cannot initialize ROS 2 interface")
        rclpy.init()
        self._ros_node = Node("dexpilot")
        # BEST_EFFORT sensor QoS (must match the fusion publisher). Reliable QoS
        # here made this subscriber back-pressure the 60 Hz fusion publisher once
        # the teleop loop slowed post-press-8 — cascading upstream and freezing a
        # camera. Best-effort => we just drop stale frames; the producer never
        # blocks. See teleop/hand_message.sensor_qos for the full failure chain.
        try:
            from teleop.hand_message import sensor_qos   # package import (main app)
        except ImportError:
            from hand_message import sensor_qos           # same-dir import (scripts)
        self._ros_node.create_subscription(
            Float32MultiArray, "/hand/joint_angles",
            lambda msg: self._on_hand_message(list(msg.data)), sensor_qos())

    def _on_hand_message(self, data_list: list) -> None:
        self._raw_msg = data_list
        if len(data_list) >= 3:
            self._current_wrist = np.array(data_list[0:3], float)

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

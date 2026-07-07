from __future__ import annotations

import rclpy
from rclpy import Node
from std_msgs.msg import *
import logging



class DexPilot():
    def __init__(self, eps, beta):
        self._init_ros()
    
        # Params
        self.eps = eps
        self.beta = beta
        
        # Joint angles
        self.q_h # Human hand joint angles
        self.q_r # Robot hand joint angles

        # Task space vectors
         
    
    def _init_ros(self):
        rclpy.init()
        self._ros_mediapipe_sub = Node("mediapipe_sub")
        self._ros_node.create_subscription(
            Float32MultiArray, ""

        )
    
    def _extract_world_landmarks(self):
        """
        Compute the orientation of the palm frame via MCP landmarks of the human hand
        """

    def _switching_weight(self, d):
        if d > self.eps:
            return 1
        elif <= eps:
    
        
    def _distancing_function(self, d):
        if d > self.eps:
            return 1
        elif:
            return
    
        pass


def main():
    pass

if __name__ == "__main__":
    main()
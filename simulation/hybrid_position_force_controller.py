import numpy as np
from grasp_planning_utils import spatial_transforms


class HybridPositionForceGraspController:
    """Object-level hybrid position/force controller using spatial twists."""

    def __init__(self, Kp, Kd):
        """
        Initialize the PD controller with 6x6 gain matrices.
        Kp: 6x6 Proportional gain matrix (stiffness)
        Kd: 6x6 Derivative gain matrix (damping)
        """
        self.Kp = np.array(Kp, dtype=float)
        self.Kd = np.array(Kd, dtype=float)
        
    
    def compute_pose_error(self, pos_curr, quat_curr, pos_des, quat_des):
        """
        Computes the 6D twist error between two poses

        Args:
            pose_curr (_type_): _description_
            quat_curr (_type_): _description_
            pos_des (_type_): _description_
            quat_des (_type_): _description_
        """
        
        pose_err = pos_des - po
        
    def c

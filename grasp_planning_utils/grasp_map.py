
import numpy as np


class SpatialGraspPlanning():
    def __init__():
        pass
    
    def skew_symmetric(self, p):
        """Returns the 3x3 skew-symmetric matrix of a 3D position vector

        Args:
            p (_type_): _description_
        """
        p_hat = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], -p[0], 0]
            ])
    
    def adjoint_map(self, R, p):
        """Adjoint map for spatial transforms.
        [R       0]
        [p_hat R R]
        
        Args:
            R : 3x3 rotation matrix from contact frame to object frame.
            p : 3-dimensional position vector of the contact frame in the object frame.
        """
        
        R = np.array(R, dtype=float)
        p = np.array(p, dtype=float)
        
        g = np.block([
            [R                         , np.zeros((3,3))],
            [self.skew_symmetric(p) @ R,               R]
        ])
        
        return g
    
    def grasp_map(self, R, p)
        return 
    
    
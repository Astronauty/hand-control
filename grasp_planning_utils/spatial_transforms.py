import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm

class SpatialMath:
    """Utility class for rigid body transformations and spatial vectors."""
    
    @staticmethod
    def get_transform_matrix(pos, quat_scipy):
        """Converts position and quaternion [x, y, z, w] into a 4x4 SE(3) matrix."""
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat_scipy).as_matrix()
        T[:3, 3] = pos
        return T

    @staticmethod
    def extract_twist_from_se3(se3_matrix):
        """Extracts the 6D twist [v_x, v_y, v_z, w_x, w_y, w_z] from a 4x4 se(3) matrix."""
        v = se3_matrix[:3, 3]
        skew_w = se3_matrix[:3, :3]
        w = np.array([skew_w[2, 1], skew_w[0, 2], skew_w[1, 0]])
        return np.concatenate([v, w])

    @staticmethod
    def compute_error_twist(T_curr, T_des):
        """Computes the 6D spatial error twist V_err using the matrix logarithm."""
        T_curr_inv = np.linalg.inv(T_curr)
        T_err = T_des @ T_curr_inv
        
        se3_err = logm(T_err)
        se3_err = np.real(se3_err) # Strip tiny numerical imaginary artifacts
        
        return SpatialMath.extract_twist_from_se3(se3_err)

    @staticmethod
    def skew_symmetric(vec):
        """Returns the 3x3 skew-symmetric matrix of a 3D vector."""
        return np.array([
            [0,      -vec[2],  vec[1]],
            [vec[2],  0,      -vec[0]],
            [-vec[1], vec[0],  0     ]
        ])

    @staticmethod
    def adjoint_map(R_mat, p_vec):
        """
        Computes the 6x6 spatial adjoint map (wrench transformation).
        Maps a force-first wrench from a local frame to the base frame.
        """
        g = np.block([
            [R_mat, np.zeros((3, 3))],
            [SpatialMath.skew_symmetric(p_vec) @ R_mat, R_mat]
        ])
        return g
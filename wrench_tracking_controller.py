import numpy as np
import scipy.linalg
import mujoco as mj


def contact_force_solve(W_des, G, internal_force_mag):
    """
    Solve for the contact forces to achieve 

    Args:
        W_des (_type_): Desired object wrench
        G (_type_): Grasp map (maps contact forces in contact frame to obj wrenches in obj frame)
        internal_force_mag (_type_): _description_
    """
    if np.linalg.cond(G):
        print("Warning: large condition number in grasp map. Pseudoinverse for computing contact forces may be sensitive to noise.")
    
    f_c = np.linalg.pinv(G)*W_des + np.linalg.null_space(G).flatten()
    
    # Create a dense numpy array for the Jacobian (3 rows, nv columns)
    jac_index = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jac_index, None, site_id)
        

    
    
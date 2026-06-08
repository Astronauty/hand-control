# All variable naming notation follows https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.linalg
import math
import time

import mujoco as mj
from mujoco import viewer



def planar_hat_map(a):
    a_hat = np.array([
        [0, -a],
        [a, 0]
    ])
    return a_hat


def planar_grasp_map_PCWF(p_S1_O, p_S2_O, R_S1, R_S2):
    """Returns the planar grasp map G for two contacts, that maps contact forces in the contact frames to an
    overall wrench on the object. 

    Args:
        p_S1_O (_type_): Location of site 1 relative to obj center in obj frame
        p_S2_O (_type_): Location of site 2 relative to obj center in obj frame
        R_S1 (_type_): Rotation of site 1 relative to obj frame
        R_S2 (_type_): Rotation of site 2 relative to obj frame
    """
    
    # Grasp map for site 1 contact (maps contact forces in contact 1 frame to object wrench in object frame)
    G_1 = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-p_S1_O[1], p_S1_O[0]],
    ]) @ R_S1

    # Grasp map for site 2 contact (maps contact forces in contact 2 frame to object wrench in object frame)
    G_2 = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-p_S2_O[1], p_S2_O[0]],
    ]) @ R_S2

    # Form the overall grasp map (maps stacked contact forces [f1 f2]^T in respective local contact frames to
    # wrench in object frame)
    G = np.block([G_1, G_2])
    
    return G
    
if __name__ == "__main__":
    sns.set_theme(style="ticks")
    
    # Load in mujoco model
    model = mj.MjModel.from_xml_path('models/planar_two_finger_manipulator.xml') 
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    
    
    # Parse IDs of relevant sites and bodies
    id_S1_target = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'index_box_touch') # Location on object to touch w/ index finger
    id_S2_target = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'thumb_box_touch') # Location on object to touch w/ thumb
    # index_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'index_touch')
    # thumb_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'thumb_touch')
    id_C1_site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'index_touch')
    id_C2_site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'thumb_touch')
    id_O_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'obj1')
    
    
    # Extract locations 
    p_WoO_W = data.xpos[id_O_body][:2].copy() # Object location in world frame
    p_WoS1_W = data.site_xpos[id_S1_target][:2].copy() # Contact site 1 location in world frame
    p_WoS2_W = data.site_xpos[id_S2_target][:2].copy() # Contact site 2 location in world frame
    
    p_OS1_W = p_WoS1_W - p_WoO_W # Contact site 1 location relative to object in world frame
    p_OS2_W = p_WoS2_W - p_WoO_W # Contact site 2 location relative to object in world frame
    
    # # Define contact frames
    # p1_o = data.site_xpos[target_index_site_id][:2].copy()
    # p2_o = data.site_xpos[target_thumb_site_id][:2].copy()
    
    # Rotations (Planar 2x2)
    R_WS1 = data.site_xmat[id_S1_target].reshape(3, 3)[:2, :2]
    R_WS2 = data.site_xmat[id_S2_target].reshape(3, 3)[:2, :2]
    R_WO = data.xmat[id_O_body].reshape(3, 3)[:2, :2]
    
    # Transform positions and rotations to Object Frame (O)
    p_OS1_O = R_WO.T @ p_OS1_W
    p_OS2_O = R_WO.T @ p_OS2_W
    R_OS1 = R_WO.T @ R_WS1
    R_OS2 = R_WO.T @ R_WS2
    
    # Compute grasp map and nullspace forces
    G = planar_grasp_map_PCWF(p_OS1_O, p_OS2_O, R_OS1, R_OS2)
    gamma = 5.0
    G_null = scipy.linalg.null_space(G).flatten()
    f_C1_S1 = G_null[:2] * gamma # Force at contact 1 in Site 1 frame
    f_C2_S2 = G_null[2:] * gamma # Force at contact 2 in Site 2 frame
    
    # IK (find q_target to match up finger contact site with object contact site)
    q_target = data.qpos.copy()
    for _ in range(500):
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        
        p_WoC1_W = data.site_xpos[id_C1_site][:2]
        p_WoC2_W = data.site_xpos[id_C2_site][:2]
        
        err = np.concatenate([p_WoS1_W - p_WoC1_W, p_WoS2_W - p_WoC2_W])
        print("IK Err: ", err)
        if np.linalg.norm(err) < 1e-3: break
            
        J_p_C1_W = np.zeros((3, model.nv))
        J_p_C2_W = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, J_p_C1_W, None, id_C1_site)
        mj.mj_jacSite(model, data, J_p_C2_W, None, id_C2_site)
        
        J = np.vstack([J_p_C1_W[:2, :], J_p_C2_W[:2, :]])
        dq = (J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(4))) @ err
        q_target += 0.5 * dq
        data.qpos[:] = q_target
        
    print("q_target: ", q_target)


    # Simulation
    mj.mj_resetData(model, data)
    control_phase = 'REACH'
    # Kp, Kd = 50.0, 10.0
    Kp = 0.5
    Kd = 0.05
    
    with mj.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # print("q_err: ", q_target)
            if control_phase == 'REACH':
                tau_ctrl = Kp * (q_target - data.qpos) + Kd * (0 - data.qvel)
                if np.linalg.norm(q_target - data.qpos) < 0.05:
                    control_phase = 'GRASP'
                    
            data.qfrc_applied[:] = tau_ctrl + data.qfrc_bias
            mj.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
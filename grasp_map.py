# All variable naming notation follows https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.linalg
import math

import mujoco as mj
from mujoco import viewer




def planar_hat_map(a):
    a_hat = np.array([
        [0, -a],
        [a, 0]
    ])
    return a_hat


def planar_grasp_map_PCWF(ax, p1_o, p2_o, theta1, theta2, plot_grasp_map_nullspace=True):
    """Returns the planar grasp G map assuming 2 point contacts with friction.
    F_o = G f_c
        where Fo_o is the object wrench in the object frame,
              [f_c1 f_c2] is the stacked contact forces [f1_x f1_y f2_x f2_y]^T in the CONTACT FRAMES c1, c2

    Args:
        p1_o (3x1): Location of the first contact in object frame.
        p2_o (3x1): Location of second contact in object frame.
        theta1: rotation from contact frame 1 to object frame
        theta2: rotation from contact frame 2 to object frame

    Returns:
        G (3x4): Planar Grasp Map
    """
    x1, y1 = p1_o[0], p1_o[1]
    x2, y2 = p2_o[0], p2_o[1]
    
    # Wrenches applied by each object in the contact frame
    W1 = np.array([
        [1, 0],
        [0, 1],
        [-y1, x1]
    ])
    
    W2 = np.array([
        [1, 0],
        [0, 1],
        [-y2, x2]
    ])
    
    
    R1 = np.array([
        [np.cos(theta1), -np.sin(theta1)],
        [np.sin(theta1),  np.cos(theta1)]
    ])
    
    R2 = np.array([
        [np.cos(theta2), -np.sin(theta2)],
        [np.sin(theta2),  np.cos(theta2)]
    ])

    G1 = W1 @ R1
    G2 = W2 @ R2
    
    # G = np.block([
    #     [np.eye(2,2), np.eye(2,2)],
    #     [-y1, x1, -y2, x2]
    # ])
    
    G = np.block([G1, G2])
    
    # Visualize Grasp Map Nullspace
    if plot_grasp_map_nullspace:
        G_null = scipy.linalg.null_space(G).flatten()
        print(f"Nullspace of G:\n{G_null}")

        f1null_c1 = G_null[0:2].reshape(2,1)
        f2null_c2 = G_null[2:4].reshape(2,1)
        
        f1null_o = R1 @ f1null_c1
        f2null_o = R2 @ f2null_c2
        
        draw_null_force_arrow(ax, f1null_o, p1_o)
        draw_null_force_arrow(ax, f2null_o, p2_o)

    return G


def draw_frame(ax, x, y, theta, label):
    """
    Draws a local coordinate frame.
    rot: Rotation of the frame in radians.
    """
    axis_length = 0.25
    
    ax.plot(x, y, 'ko') # 'ko' means black circle
    
    # Calculate the new X-axis vector components
    # The X-axis starts at (1, 0) in local space
    dx_x = axis_length * math.cos(theta)
    dy_x = axis_length * math.sin(theta)
    
    # Calculate the new Y-axis vector components
    # The Y-axis starts at (0, 1) in local space, which is 90 degrees offset from X
    dx_y = -axis_length * math.sin(theta)
    dy_y = axis_length * math.cos(theta)

    # 3. Draw Local X-axis (Red arrow)
    ax.annotate('', 
                xy=(x + dx_x, y + dy_x), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="red", lw=2.5))

    # 4. Draw Local Y-axis (Green arrow)
    ax.annotate('', 
                xy=(x + dx_y, y + dy_y), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="green", lw=2.5))

    # Position the anchor in the local "bottom-left" quadrant
    offset = axis_length * 0.2
    label_dx = -offset * math.cos(theta) + offset * math.sin(theta)
    label_dy = -offset * math.sin(theta) - offset * math.cos(theta)
    
    # Dynamically align the text so the bounding box grows AWAY from the origin
    # If dx is positive (right of origin), align left (pushes text further right)
    h_align = 'left' if label_dx > 0 else 'right'
    v_align = 'bottom' if label_dy > 0 else 'top'
    
    # Optional: Catch near-zero values to perfectly center it on axes
    if abs(label_dx) < 1e-5: h_align = 'center'
    if abs(label_dy) < 1e-5: v_align = 'center'
    
    ax.text(x + label_dx, y + label_dy, 
            label, 
            color='black', 
            fontsize=18, 
            fontweight='bold',
            ha=h_align, 
            va=v_align)
 
def draw_box_object(ax, bl_corner, width, height, label='o'):
    rect = patches.Rectangle(
        bl_corner,          # Bottom-left corner at x=-1, y=2
        width,              # Width
        height,               # Height
        linewidth=2,     
        edgecolor='black', 
        facecolor='none' 
    )
    
    ax.add_patch(rect)
    
    ox = bl_corner[0] + (width / 2)
    oy = bl_corner[1] + (height / 2)

    # # Plot a dot at the origin
    # ax.plot(ox, oy, 'ko') # 'ko' means black circle
    
    draw_frame(ax, ox, oy, theta=0, label='o')

    pass

def draw_null_force_arrow(ax, gnull, end_o, length=0.6, color="blue", lw=2.5, normalize=True):
    """
    gnull: 2D force direction as np.array([fx, fy]) or np.array([[fx], [fy]])
    end_o: arrow tip location, np.array([x, y])
    """
    d = np.asarray(gnull, dtype=float).reshape(2)  # handles (2,) and (2,1)
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise ValueError("Zero-length direction vector from gnull.")

    if normalize:
        d = (d / n) * length

    end_o = np.asarray(end_o, dtype=float).reshape(2)
    start_o = end_o - d

    ax.annotate(
        "",
        xy=(end_o[0], end_o[1]),
        xytext=(start_o[0], start_o[1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
    )
if __name__ == "__main__":
    sns.set_theme(style="ticks")
    
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 3)

    # Draw object
    draw_box_object(ax, (-1, 0), 2, 2)
    
    # Define contact frames
    p1_o = np.array([-1, 1.5])
    p2_o = np.array([1, 1.0])
    
    theta1 = np.pi
    theta2 = 0
    
    
    # Draw grasp map contact frames
    draw_frame(ax, p1_o[0], p1_o[1], 0, label='c1')
    draw_frame(ax, p2_o[0], p2_o[1], np.pi, label='c2')
    
    G = planar_grasp_map_PCWF(ax, p1_o, p2_o, theta1, theta2, plot_grasp_map_nullspace=True)
    print(f"Grasp Map G: \n {G}")

    
    plt.show()
    

    # Solve for desired contact forces based on nullspace of grasp map
    # and scaling parameter gamma
    gamma = 5.0
    G_null = scipy.linalg.null_space(G).flatten()
    
    f_c1 = G_null[:2] * gamma
    f_c2 = G_null[2:] * gamma
    
    print("f_c1: ", f_c1)
    print("f_c2: ", f_c2)
    
    # Load in mujoco model
    model = mj.MjModel.from_xml_path('models/planar_two_finger_manipulator.xml') 
    data = mj.MjData(model)
    
    index_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'index_touch')
    thumb_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'thumb_id')
    
    jacp_index = np.zeros((3, model.nv)) # translational jacobian of the index finger
    jacr_index = np.zeros((3, model.nv)) # rotational jacobian of the index finger

    jacp_thumb = np.zeros((3, model.nv))
    jacr_thumb = np.zeros((3, model.nv))   
 
    
    # Form the jacobians
    mj.mj_jacSite(model, data, jacp_index, jacr_index, index_site_id)
    mj.mj_jacSite(model, data, jacp_thumb, jacr_thumb, thumb_site_id)
    
    print(jacp_index.shape)
    print(jacp_thumb.shape)
    print(f_c1.shape)
    print(f_c2.shape)
    
    tau_index = jacp_index[:2, :].T @ f_c1
    tau_thumb = jacp_thumb[:2, :].T @ f_c2
    
    print(tau_index.shape)
    
    # print("tau_index: ", tau_index)
    # print("tau_thumb: ", tau_thumb)
              
    
    
    
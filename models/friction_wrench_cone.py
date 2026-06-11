import mujoco
from mujoco import viewer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# Load model 
model = mujoco.MjModel.from_xml_path('models/planar_two_finger_manipulator.xml') 
data = mujoco.MjData(model)

# Set to contact keyframe 
mujoco.mj_resetDataKeyframe(model, data, model.keyframe("both_contacts").id)

# Stabilize
for _ in range(10000):
    mujoco.mj_step(model, data)

# Verify that both contacts are found, and
# Extract respective tangential friction coefficient of contact 
# (note: when geoms have equal priority, mujoco uses max friction coeff between the 2 geoms)
thumb_mu = None
index_mu = None
for i in range(data.ncon):
    contact = data.contact[i]
    contact_body1_id = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1])
    contact_body2_id = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2])

    # These only work if distal links are defined before obj1 in xml file 
    if (contact_body1_id == "right_thumb_distal_link" and contact_body2_id == "obj1"):
        thumb_mu = contact.mu
    elif (contact_body1_id == "right_index_distal_link" and contact_body2_id == "obj1"):
        index_mu = contact.mu
assert thumb_mu != None, "Contact between Thumb and Object NOT found :("
assert index_mu != None, "Contact between Index and Object NOT found :("

def calc_wrench(sensordata, mu_t):
    assert len(sensordata) == 13, "sensordata is not of right length (13)"
    if sensordata[0] == 0: # indicates that contact was not found 
        return 0

    '''
    sensordata: (dependent on definition in xml)
    [0]: found?
    [1-3]: contact force in contact frame
    [4-6]: contact pos in global frame 
    [7-9]: contact normal dir in global frame
    [10:12]: contact tangent dir in global frame 

    Contact Frame (mujoco convention)
    -> x: normal to contact
    -> y: tangent to contact

    In 2D global frame, only x (positive right) and y (positive up) are relevant
    '''

    # Extract data from sensor
    F_c = sensordata[1:4] # contact force in contact frame
    Fnormal_c = F_c[0] # normal contact force from finger to object 
    p_gc = sensordata[4:7] # contact position in global frame
    normal_g = sensordata[7:10] # contact normal direction in global frame 
    tangent_g = sensordata[10:13] # contact tangent direction in global frame  

    # 2D Friction cone edge vectors in contact frame
    f0_c = np.array([[0], [0]])
    f1_c = np.array([[Fnormal_c], [-mu_t*Fnormal_c]]) 
    f2_c = np.array([[Fnormal_c], [mu_t*Fnormal_c]])

    # 2D Friction cone edge UNIT vectors in contact frame 
    # unitf1_c = f1_c / np.linalg.norm(f1_c)
    # unitf2_c = f2_c / np.linalg.norm(f2_c)

    # 2D Friction cone edge vectors in body frame 
    R_gc =  np.array([normal_g[0:2], tangent_g[0:2]]) # contact frame ito global frame
    assert np.array_equal(np.round(data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj1")]).astype(int), np.eye(3).reshape(-1)), "Body frame == Global frame assumption broke down D:"
    # R_gc == R_bc
    f0_b = R_gc @ f0_c
    f1_b = R_gc @ f1_c
    f2_b = R_gc @ f2_c

    # 2D Contact position vector in body frame 
    p_gb = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj1")]
    p_bc = p_gc - p_gb
    p_bc = np.array([[p_bc[0]], [p_bc[1]]])

    # (Mz, fx, fy) Wrench cone edge vectors in body frame 
    F0_b = np.array([p_bc[0]*f0_b[1] - p_bc[1]*f0_b[0], f0_b[0], f0_b[1]])
    F1_b = np.array([p_bc[0]*f1_b[1] - p_bc[1]*f1_b[0], f1_b[0], f1_b[1]])
    F2_b = np.array([p_bc[0]*f2_b[1] - p_bc[1]*f2_b[0], f2_b[0], f2_b[1]])

    # Make more display friendly (Mz, fx, fy)
    F0_b = F0_b.reshape(3)
    F1_b = F1_b.reshape(3)
    F2_b = F2_b.reshape(3)

    return F1_b, F2_b, F0_b # These are wrench vertices/edges

# Slanted (non-symmetrical) Contact Test
test_slanted = False
slanted_idx_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj1")] + [1, -1, 0]
slanted_thumb_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj1")] + [-1, 1, 0]
slanted_idx_force = [1, 1, 0] # contact frame
slanted_thumb_force = [1, 1, 0] # contact frame 
idx_contact_normal = [-1, 0, 0]
thumb_contact_normal = [1, 0, 0]
idx_contact_tangent = [0, 1, 0]
thumb_contact_tangent = [0, -1, 0]
slanted_idx_contact = np.array([1,*slanted_idx_force, *slanted_idx_pos, *idx_contact_normal, *idx_contact_tangent]) # found, force, pos, normal, tangent
slanted_thumb_contact = np.array([1, *slanted_thumb_force, *slanted_thumb_pos, *thumb_contact_normal, *thumb_contact_tangent])
slanted_idx_wrench = calc_wrench(slanted_idx_contact, 1.0)
slanted_thumb_wrench = calc_wrench(slanted_thumb_contact, 1.0)

# Mujoco Simulation Test (if not test_slanted)
# Get sensor contact data and calculate respective wrench vertices/edges
# (note: visually, thumb is on the left)
thumb_contact = data.sensordata[0:13] # found, force, pos, normal, tangent
thumb_wrenches = calc_wrench(thumb_contact, thumb_mu) # in body frame 
assert thumb_wrenches != 0, "Sensor Contact between Thumb and Object NOT found :(" 
index_contact = data.sensordata[13:26] # found, force, pos, normal, tangent
index_wrenches = calc_wrench(index_contact, index_mu) # in body frame 
assert index_wrenches != 0, "Sensor Contact between Index and Object NOT found :("

# Make wrenches plot-friendly (fx, fy, Mz)
# Always put thumb first for legend to be plotted correctly
if test_slanted:
    print(f'slanted thumb wrench vertices/edges: {slanted_thumb_wrench}') # (Mz, fx, fy)
    print(f'slanted idx wrench vertices/edges: {slanted_idx_wrench}') # (Mz, fx, fy)
    thumb_wrenches = np.roll(slanted_thumb_wrench, -1, axis=1)
    index_wrenches = np.roll(slanted_idx_wrench, -1, axis=1)
    # all_wrenches = np.array([slanted_thumb_wrench[0], slanted_thumb_wrench[1], slanted_idx_wrench[0], slanted_idx_wrench[1]])
    all_wrenches = np.array([thumb_wrenches[0], thumb_wrenches[1], thumb_wrenches[2], index_wrenches[0], index_wrenches[1], index_wrenches[2]])
else:
    print(f'contact thumb wrench vertices/edges: {thumb_wrenches}') # (Mz, fx, fy)
    print(f'contact idx wrench vertices/edges: {index_wrenches}') # (Mz, fx, fy)
    thumb_wrenches = np.roll(thumb_wrenches, -1, axis=1)
    index_wrenches = np.roll(index_wrenches, -1, axis=1)
    all_wrenches = np.array([thumb_wrenches[0], thumb_wrenches[1], thumb_wrenches[2], index_wrenches[0], index_wrenches[1], index_wrenches[2]])
# all_wrenches = np.roll(all_wrenches, -1, axis=1) # (fx, fy, Mz) # impt to define axis!

# Get all possible sum of thumb_vertex + index_vertex (to approx Minkowski sum from convex hull)
vert_sums = []
for tw in thumb_wrenches:
    for iw in index_wrenches:
        vert_sums.append(tw+iw)
vert_sums = np.array(vert_sums)
hull = ConvexHull(vert_sums)
print(hull.equations)

# Plot wrenches on 3D axes
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i, val in enumerate(all_wrenches):
    if i < 2:
        ax.plot([0, val[0]], [0, val[1]], [0, val[2]], label=f'Fthumb_{i+1}') # * unpacks vector into individual arguments
    elif i > 2 and i < 5:
        ax.plot([0, val[0]], [0, val[1]], [0, val[2]], label=f'Findex_{i-1}')
ax.set_xlim([min(all_wrenches[:,0]), max(all_wrenches[:,0])])
ax.set_ylim([min(all_wrenches[:,1]), max(all_wrenches[:,1])])
ax.set_zlim([min(all_wrenches[:,2]), max(all_wrenches[:,2])])
ax.plot_trisurf(vert_sums[:,0], vert_sums[:,1], vert_sums[:,2],
                triangles=hull.simplices, alpha=0.3)
ax.set_xlabel('F_x in body frame\n(right +ve)', labelpad=20)
ax.set_ylabel('F_y in body frame\n(up +ve)', labelpad=20)
ax.set_zlabel('M_z in body frame\n(CCW +ve)', labelpad=20)
plt.legend()
plt.tight_layout()
plt.show()

# with viewer.launch_passive(model, data) as v:
#     v.sync()
#     while v.is_running():
#         mujoco.mj_step(model, data)
#         v.sync()
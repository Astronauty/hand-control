import time
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Empty
import casadi as ca
from casadi import Callback



class TeleopCalibrationNode(Node):
    def __init__(self, model_path):
        super().__init__('teleop_calibration')
        
        # 1. Load MuJoCo (The Robot)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        # 2. State
        self.state = "IDLE"  # IDLE -> TRACKING
        self.human_wrist_start = None  # Captured at 'C'
        self.robot_wrist_start = None  # Captured at 'C'
        self.virtual_offset = None     # The calculated link
        
        self.current_human_wrist = None
        
        # 3. ROS Setup
        self.create_subscription(Float32MultiArray, '/hand/joint_angles', self.hand_callback, 10)
        self.create_subscription(Empty, '/teleop/trigger_calibration', self.calibration_callback, 10)
        self.target_pub = self.create_publisher(PoseStamped, '/robot/target_pose', 10)
        
        print(">> Node Ready. Move robot to start pos, then press 'C' in MediaPipe.")

    def hand_callback(self, msg):
        """Store the latest human wrist position AND raw data."""
        self.current_raw_msg = msg  # <--- SAVE THIS for the angle extraction
        
        if len(msg.data) >= 3:
            self.current_human_wrist = np.array(msg.data[0:3])

    def get_robot_wrist_position(self):
        """Reads the CURRENT robot base position from MuJoCo joints."""
        # Your XML has slide joints 'Th_base_x' and 'Th_base_y'
        # qpos indices: 0 is x, 1 is y (based on your XML structure)
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        return np.array([x, y, 0.0])

    def calibration_callback(self, msg):
        """Triggered when user presses 'C'."""
        if self.current_human_wrist is None:
            print("Ignored: No human hand detected.")
            return

        # 1. Capture Snapshots
        self.human_wrist_start = self.current_human_wrist.copy()
        self.robot_wrist_start = self.get_robot_wrist_position()
        
        # 2. Calculate the "Virtual Link" (Offset)
        # Robot = Human + Offset  =>  Offset = Robot - Human
        # Note: We scale the Human input (Camera Z is depth) to Robot Y
        human_scaled = self.map_camera_to_robot_frame(self.human_wrist_start)
        self.virtual_offset = self.robot_wrist_start - human_scaled
        
        self.state = "TRACKING"
        print(f"\n[CALIBRATED] Offset established: {self.virtual_offset}")
        print(f"Human Start: {human_scaled} | Robot Start: {self.robot_wrist_start}\n")

    def map_camera_to_robot_frame(self, camera_pos):
        """
        Maps Camera axes to Robot axes.
        Camera: X (Right), Y (Down), Z (Depth)
        Robot (Planar): X (Right), Y (Up/Forward), Z (Height/Ignore)
        """
        # Tuning Scales
        SCALE_X = 2.0
        SCALE_Y = 2.5 # Depth needs more gain usually
        
        rx = camera_pos[0] * SCALE_X
        ry = camera_pos[2] * SCALE_Y # Map Depth (Z) to Planar Y
        rz = 0.0
        return np.array([rx, ry, rz])

    def update_control(self):
        """Main control loop."""
        # Update Physics
        mujoco.mj_step(self.model, self.data)
        
        if self.state == "TRACKING" and self.current_human_wrist is not None:
            # 1. Calculate Virtual Wrist (Base) Target
            human_curr = self.map_camera_to_robot_frame(self.current_human_wrist)
            target_base = human_curr + self.virtual_offset
            
            # 2. Apply FULL CONTROLS (Base + Fingers) <<--- CHANGED
            self.apply_teleop_controls(target_base)
            
            # 3. Visualize Target
            mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mocap')
            if mocap_id != -1:
                # Still useful to see where the "Goal" is vs where physics is
                self.data.mocap_pos[0][0] = target_base[0]
                self.data.mocap_pos[0][1] = target_base[1]

    def publish_target(self, target):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(target[0])
        msg.pose.position.y = float(target[1])
        msg.pose.position.z = 0.0
        self.target_pub.publish(msg)

    def apply_teleop_controls(self, target_base_pos):
        """
        Directly maps MediaPipe joint angles to MuJoCo actuators.
        Call this inside update_control() when state is TRACKING.
        """
        # 0. Safety Check
        if self.current_raw_msg is None:
            return

        # 1. Parse The Raw Message
        # Format: [Wrist(6) | Thumb_CMC(3), Thumb_MCP(3)... | Index_MCP(3)...]
        # Each joint has 3 angles: [Yaw(Z), Pitch(Y), Roll(X)]
        # We generally use 'Z' (Yaw) for flexion in this specific coordinate setup
        data = self.current_raw_msg.data
        
        # Helper to get angle in RADIANS for a specific joint
        # joint_index corresponding to JOINT_ORDER list
        def get_flexion(joint_idx):
            start_idx = 6 + (joint_idx * 3) # Skip wrist(6) + prev joints
            deg = data[start_idx] # Index 0 is Z-rotation (Flexion)
            return np.radians(deg)

        # 2. Extract Finger Angles (Indices from your JOINT_ORDER)
        # Thumb: MCP=1, IP=2 (XML has MCP, PIP, DIP? Mapping 3 joints)
        # Your MP script has: thumb_cmc(0), thumb_mcp(1), thumb_ip(2)
        th_mcp = get_flexion(1) 
        th_pip = get_flexion(2)
        th_dip = get_flexion(2) * 0.5 # Mimic DIP based on IP since MP only has 3 thumb pts
        
        # Index: MCP=3, PIP=4, DIP=5
        in_mcp = get_flexion(3)
        in_pip = get_flexion(4)
        in_dip = get_flexion(5)

        # 3. Apply to MuJoCo Actuators
        # XML Order: [Th_base_x, Th_base_y, Index_MCP, Index_PIP, Index_DIP, Thumb_MCP, Thumb_PIP, Thumb_DIP]
        
        ctrl = self.data.ctrl
        
        # A. Base Movement (Virtual Wrist)
        ctrl[0] = target_base_pos[0]
        ctrl[1] = target_base_pos[1]
        
        # B. Index Finger (Gain -1.0 often needed depending on axis definition)
        # Adjust signs here if fingers curl backwards!
        ctrl[2] = -in_mcp 
        ctrl[3] = -in_pip 
        ctrl[4] = -in_dip 

        # C. Thumb 
        ctrl[5] = th_mcp
        ctrl[6] = th_pip
        ctrl[7] = th_dip


class Optimization2DFKCallback(ca.Callback):
    """Forward kinematics callback for optimization with finite differences"""
    
    def __init__(self, name, model, data, body_id, actuated_indices, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.data = data
        self.body_id = body_id
        self.actuated_indices = actuated_indices
        self.nq = model.nq
        self.construct(name, opts)

    def get_n_in(self): 
        return 1
    
    def get_n_out(self): 
        return 1
    
    def get_sparsity_in(self, i): 
        return ca.Sparsity.dense(len(self.actuated_indices), 1)
    
    def get_sparsity_out(self, i): 
        return ca.Sparsity.dense(2, 1)

    def eval(self, arg):
        # Map optimizer input (actuated joints) to full qpos
        q_actuated = np.array(arg[0]).flatten()
        q_full = self.data.qpos.copy()
        
        for idx, val in zip(self.actuated_indices, q_actuated):
            q_full[idx] = val
        
        # Update MuJoCo kinematics
        self.data.qpos[:] = q_full
        mujoco.mj_kinematics(self.model, self.data)
        
        # Return 2D position
        pos_2d = self.data.xpos[self.body_id][:2].copy()
        return [pos_2d]

    def has_jacobian(self):
        return False


class TeleopOptimizationNode(Node):
    def __init__(self, model_path):
        super().__init__('teleop_optimization')
        
        # 1. Load MuJoCo (The Robot)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.data_opt = mujoco.MjData(self.model)  # Separate data for optimization
        mujoco.mj_step(self.model, self.data)
        
        # 2. State
        self.state = "IDLE"  # IDLE -> TRACKING
        self.human_wrist_start = None
        self.robot_wrist_start = None
        self.virtual_offset = None
        
        self.current_human_wrist = None
        self.current_raw_msg = None
        
        # 3. Setup Optimization Components
        self.setup_optimization()
        
        # 4. ROS Setup
        self.create_subscription(Float32MultiArray, '/hand/joint_angles', 
                                self.hand_callback, 10)
        self.create_subscription(Empty, '/teleop/trigger_calibration', 
                                self.calibration_callback, 10)
        self.target_pub = self.create_publisher(PoseStamped, '/robot/target_pose', 10)
        
        # 5. Optimization frequency control
        self.opt_counter = 0
        self.opt_frequency = 5  # Run optimization every N steps
        
        print(">> Optimization Node Ready. Move robot to start pos, then press 'C' in MediaPipe.")

    def setup_optimization(self):
        """Initialize optimization callbacks and identify actuated joints
        
        IMPORTANT: Your XML must have consistent joint axes for both fingers to curl
        in the same direction. The thumb joints should use axis="0 0 -1" (negative Z)
        while index joints use axis="0 0 1" (positive Z).
        """
        
        # Identify actuated joints (exclude free joints like the object)
        self.actuated_indices = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.actuated_indices.append(qpos_adr)
        
        print(f"Actuated joint indices: {self.actuated_indices}")
        
        # Use exact body names from your XML
        # Note: Your XML uses 'medial_link' for PIP joints, not 'middle_link'
        print("\nLoading joint bodies...")
        
        self.joint_bodies = {
            'thumb_mcp': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_thumb_proximal_link"),
            'thumb_pip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_thumb_medial_link"),
            'thumb_dip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_thumb_distal_link"),
            'index_mcp': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_index_proximal_link"),
            'index_pip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_index_medial_link"),
            'index_dip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_index_distal_link"),
        }
        
        # Verify all bodies found
        all_found = True
        for joint_name, body_id in self.joint_bodies.items():
            if body_id == -1:
                print(f"  ✗ {joint_name}: NOT FOUND")
                all_found = False
            else:
                print(f"  ✓ {joint_name}: body_id={body_id}")
        
        if not all_found:
            print("\n⚠ WARNING: Some bodies not found! Check body names in XML.")
        
        # Create FK callbacks for ALL tracked joints with enable_fd option
        call_opts = {"enable_fd": True}
        
        self.fk_callbacks = {}
        for joint_name, body_id in self.joint_bodies.items():
            if body_id != -1:  # Valid body ID
                self.fk_callbacks[joint_name] = Optimization2DFKCallback(
                    f'fk_{joint_name}', self.model, self.data_opt,
                    body_id, self.actuated_indices, call_opts
                )
        
        print(f"\nCreated {len(self.fk_callbacks)} FK callbacks")
        
        # Solver options (using finite differences like approach_controller.py)
        self.solver_opts = {
            "print_time": False,
            "ipopt": {
                "jacobian_approximation": "finite-difference-values",
                "hessian_approximation": "limited-memory",
                "print_level": 0,
                "sb": "yes",
                "max_iter": 100,  # Reduced for real-time performance
            },
            "ad_weight_sp": 0
        }

    def hand_callback(self, msg):
        """Store the latest human wrist position AND raw data."""
        self.current_raw_msg = msg
        
        if len(msg.data) >= 3:
            self.current_human_wrist = np.array(msg.data[0:3])

    def get_robot_wrist_position(self):
        """Reads the CURRENT robot base position from MuJoCo joints."""
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        return np.array([x, y, 0.0])

    def calibration_callback(self, msg):
        """Triggered when user presses 'C'."""
        if self.current_human_wrist is None:
            print("Ignored: No human hand detected.")
            return

        # Capture snapshots
        self.human_wrist_start = self.current_human_wrist.copy()
        self.robot_wrist_start = self.get_robot_wrist_position()
        
        # Calculate the "Virtual Link" (Offset)
        human_scaled = self.map_camera_to_robot_frame(self.human_wrist_start)
        self.virtual_offset = self.robot_wrist_start - human_scaled
        
        self.state = "TRACKING"
        print(f"\n[CALIBRATED] Offset established: {self.virtual_offset}")
        print(f"Human Start: {human_scaled} | Robot Start: {self.robot_wrist_start}\n")

    def map_camera_to_robot_frame(self, camera_pos):
        """
        Maps Camera axes to Robot axes.
        Camera: X (Right), Y (Down), Z (Depth)
        Robot (Planar): X (Right), Y (Up/Forward), Z (Height/Ignore)
        """
        SCALE_X = 2.0
        SCALE_Y = 2.5
        
        rx = camera_pos[0] * SCALE_X
        ry = camera_pos[2] * SCALE_Y  # Map Depth (Z) to Planar Y
        rz = 0.0
        return np.array([rx, ry, rz])

    def extract_joint_positions(self):
        """
        Extract ALL tracked joint positions from MediaPipe data.
        
        Returns:
            dict: {joint_name: [x, y] position} in robot frame, or None if data unavailable
        """
        if self.current_raw_msg is None:
            return None
        
        data = self.current_raw_msg.data
        
        # MediaPipe Hand Landmarks (21 points total):
        # 0: Wrist
        # 1-4: Thumb (CMC, MCP, IP, TIP)
        # 5-8: Index (MCP, PIP, DIP, TIP)
        # 9-12: Middle, 13-16: Ring, 17-20: Pinky
        
        # Expected message format from MediaPipe publisher:
        # [wrist_x, wrist_y, wrist_z,                    # 0-2
        #  thumb_cmc_x, thumb_cmc_y, thumb_cmc_z,        # 3-5
        #  thumb_mcp_x, thumb_mcp_y, thumb_mcp_z,        # 6-8
        #  thumb_ip_x, thumb_ip_y, thumb_ip_z,           # 9-11
        #  thumb_tip_x, thumb_tip_y, thumb_tip_z,        # 12-14
        #  index_mcp_x, index_mcp_y, index_mcp_z,        # 15-17
        #  index_pip_x, index_pip_y, index_pip_z,        # 18-20
        #  index_dip_x, index_dip_y, index_dip_z,        # 21-23
        #  index_tip_x, index_tip_y, index_tip_z,        # 24-26
        #  ...]
        
        # Minimum required: wrist + thumb (4 landmarks) + index (4 landmarks) = 9 landmarks * 3 = 27 values
        if len(data) < 27:
            print(f"Warning: MediaPipe data too short ({len(data)} values, need 27+)")
            return None
        
        joint_positions = {}
        
        # Helper function to extract and convert 3D position
        def get_position_2d(start_idx):
            pos_3d = np.array(data[start_idx:start_idx+3])
            pos_robot = self.map_camera_to_robot_frame(pos_3d)
            return pos_robot[:2]
        
        try:
            # Thumb joints (we track MCP, PIP/IP, DIP/TIP)
            # MediaPipe has: CMC(1), MCP(2), IP(3), TIP(4)
            # We map: MCP->MCP, IP->PIP, TIP->DIP
            joint_positions['thumb_mcp'] = get_position_2d(6)   # Thumb MCP
            joint_positions['thumb_pip'] = get_position_2d(9)   # Thumb IP -> our PIP
            joint_positions['thumb_dip'] = get_position_2d(12)  # Thumb TIP -> our DIP
            
            # Index joints
            # MediaPipe has: MCP(5), PIP(6), DIP(7), TIP(8)
            joint_positions['index_mcp'] = get_position_2d(15)  # Index MCP
            joint_positions['index_pip'] = get_position_2d(18)  # Index PIP
            joint_positions['index_dip'] = get_position_2d(21)  # Index DIP
            # Note: Using DIP position, not TIP, for better joint alignment
            
            return joint_positions
            
        except Exception as e:
            print(f"Error extracting joint positions: {e}")
            return None

    def optimize_joint_configuration(self, target_joints, target_base):
        """
        Solve optimization problem to find joint angles that:
        1. Match ALL joint positions from MediaPipe (not just fingertips)
        2. Move base toward target position
        3. Stay close to current configuration (for smoothness)
        
        Args:
            target_joints: dict of {joint_name: [x, y] target position}
            target_base: [x, y] target position for base
            
        Returns:
            Optimal joint angles or None if optimization fails
        """
        try:
            # Create optimization problem
            opti = ca.Opti()
            q_actuated = opti.variable(len(self.actuated_indices))
            
            # Build cost function: sum over ALL tracked joints
            total_cost = 0.0
            
            # Track each joint position
            joint_weights = {
                'thumb_mcp': 1.0,
                'thumb_pip': 1.0,
                'thumb_dip': 1.0,
                'index_mcp': 1.0,
                'index_pip': 1.0,
                'index_dip': 1.0,
            }
            
            for joint_name, target_pos in target_joints.items():
                if joint_name in self.fk_callbacks:
                    # Get predicted position from FK
                    p_joint = self.fk_callbacks[joint_name](q_actuated)
                    
                    # Add squared error to cost
                    weight = joint_weights.get(joint_name, 1.0)
                    total_cost += weight * ca.sumsqr(p_joint - target_pos)
            
            # Base position tracking (first 2 actuators are base x, y)
            # Lower weight since we care more about joint tracking
            cost_base = 0.3 * (ca.sumsqr(q_actuated[0] - target_base[0]) + 
                              ca.sumsqr(q_actuated[1] - target_base[1]))
            total_cost += cost_base
            
            # Regularization: stay close to current configuration (smoothness)
            q_current = np.array([self.data.qpos[i] for i in self.actuated_indices])
            cost_reg = 0.05 * ca.sumsqr(q_actuated - q_current)
            total_cost += cost_reg
            
            # Set objective
            opti.minimize(total_cost)
            
            # Joint limits (if specified in model)
            for i in range(self.model.nu):
                qpos_idx = self.actuated_indices[i]
                
                # Find corresponding joint
                jnt_id = None
                for j in range(self.model.njnt):
                    if self.model.jnt_qposadr[j] == qpos_idx:
                        jnt_id = j
                        break
                
                if jnt_id is not None and self.model.jnt_limited[jnt_id]:
                    q_min = self.model.jnt_range[jnt_id, 0]
                    q_max = self.model.jnt_range[jnt_id, 1]
                    opti.subject_to(opti.bounded(q_min, q_actuated[i], q_max))
            
            # Setup solver
            opti.solver('ipopt', self.solver_opts)
            
            # Warm start from current state
            opti.set_initial(q_actuated, q_current)
            
            # Solve
            sol = opti.solve()
            q_optimal = sol.value(q_actuated)
            
            # Optional: print cost for debugging
            if self.opt_counter % 100 == 0:
                print(f"Optimization cost: {sol.value(total_cost):.6f}")
            
            return q_optimal
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None

    def apply_optimization_control(self, target_base_pos):
        """
        Use optimization to find joint angles that match full MediaPipe joint configuration.
        Minimizes error across ALL tracked joints, not just fingertips.
        """
        # Extract all joint positions from MediaPipe
        target_joints = self.extract_joint_positions()
        
        if target_joints is None:
            print("No joint positions available from MediaPipe")
            return
        
        # Run optimization to match full configuration
        q_optimal = self.optimize_joint_configuration(target_joints, target_base_pos[:2])
        
        if q_optimal is not None:
            # CRITICAL: For real-time teleop, directly set joint positions
            # This gives instant response instead of waiting for actuators to settle
            for i, idx in enumerate(self.actuated_indices):
                self.data.qpos[idx] = q_optimal[i]
            
            # Update kinematics (no dynamics simulation)
            mujoco.mj_forward(self.model, self.data)
            
            # Also set ctrl so actuators track the position
            self.data.ctrl[:] = q_optimal
            
            # Optional: Print debug info
            if self.opt_counter % 50 == 0:
                print(f"\nTracking {len(target_joints)} joints:")
                for joint_name, target_pos in target_joints.items():
                    if joint_name in self.joint_bodies:
                        body_id = self.joint_bodies[joint_name]
                        actual_pos = self.data.xpos[body_id][:2]
                        error = np.linalg.norm(actual_pos - target_pos)
                        print(f"  {joint_name}: error = {error:.4f}m")
        else:
            # Fallback: maintain current position
            q_current = np.array([self.data.qpos[i] for i in self.actuated_indices])
            for i, idx in enumerate(self.actuated_indices):
                self.data.qpos[idx] = q_current[i]
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[:] = q_current

    def update_control(self):
        """Main control loop."""
        # Update Physics
        mujoco.mj_step(self.model, self.data)
        
        if self.state == "TRACKING" and self.current_human_wrist is not None:
            # Calculate target base position
            human_curr = self.map_camera_to_robot_frame(self.current_human_wrist)
            target_base = human_curr + self.virtual_offset
            
            # Run optimization-based control at reduced frequency
            self.opt_counter += 1
            if self.opt_counter % self.opt_frequency == 0:
                self.apply_optimization_control(target_base)
            
            # Visualize target
            mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mocap')
            if mocap_id != -1:
                self.data.mocap_pos[0][0] = target_base[0]
                self.data.mocap_pos[0][1] = target_base[1]
            
            # Publish target for visualization
            self.publish_target(target_base)

    def publish_target(self, target):
        """Publish target pose for ROS visualization"""
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(target[0])
        msg.pose.position.y = float(target[1])
        msg.pose.position.z = 0.0
        self.target_pub.publish(msg)


def main():
    rclpy.init()
    node = TeleopOptimizationNode(r"models\planar_two_finger_manipulator.xml")
    # node = TeleopCalibrationNode(r"..\models\planar_two_finger_manipulator.xml")
    
    with mujoco.viewer.launch_passive(node.model, node.data) as viewer:
        print("\nControls:")
        print("  - Press 'C' in MediaPipe to calibrate")
        print("  - Move your hand to control the robot")
        print("  - Close viewer to exit\n")
        
        while viewer.is_running():
            rclpy.spin_once(node, timeout_sec=0.001)
            node.update_control()
            viewer.sync()
            time.sleep(node.model.opt.timestep)
            
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

import time
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Empty

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

def main():
    rclpy.init()
    node = TeleopCalibrationNode(r"..\models\planar_two_finger_manipulator.xml")
    
    with mujoco.viewer.launch_passive(node.model, node.data) as viewer:
        while viewer.is_running():
            rclpy.spin_once(node, timeout_sec=0.001)
            node.update_control()
            viewer.sync()
            time.sleep(node.model.opt.timestep)
            
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
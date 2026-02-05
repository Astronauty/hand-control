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
        """Store the latest human wrist position (Camera Frame)."""
        if len(msg.data) >= 3:
            # XYZ from MediaPipe
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
        # Update Physics (Keep simulation running)
        mujoco.mj_step(self.model, self.data)
        
        if self.state == "TRACKING" and self.current_human_wrist is not None:
            # 1. Get current human pos in robot frame
            human_curr = self.map_camera_to_robot_frame(self.current_human_wrist)
            
            # 2. Apply the calibrated offset
            # Target = Current_Human + (Robot_Start - Human_Start)
            target = human_curr + self.virtual_offset
            
            # 3. Publish & Visualize
            self.publish_target(target)
            
            # Move 'mocap' marker in MuJoCo for visualization
            mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mocap')
            if mocap_id != -1:
                self.data.mocap_pos[0] = target

    def publish_target(self, target):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(target[0])
        msg.pose.position.y = float(target[1])
        msg.pose.position.z = 0.0
        self.target_pub.publish(msg)

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
#!/usr/bin/env python3
import sys
import os
import time
import math
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Empty

# MediaPipe Imports
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
from geometry_msgs.msg import PoseStamped

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

# --- Configuration & Constants ---
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
ANGLE_TEXT_COLOR = (255, 150, 150)

SNAPSHOT_DIR = "hand_snapshots"
CSV_PATH = os.path.join(SNAPSHOT_DIR, "hand_snapshots.csv")
MODEL_FILENAME = "hand_landmarker.task"

JOINT_ORDER = [
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
]

# --- Global State ---
DEBUG_FLAG = True
ros_node = None
joint_angles_pub = None
calibration_pub = None

# Task Phase Global State
current_phase = "IDLE"
last_landmarks = None
last_timestamp = None
contact_detected = False
fingertip_distance_buffer = deque(maxlen=20)
velocity_buffer = deque(maxlen=10)

# Thresholds
GRASP_CLOSURE_THRESHOLD = 0.05
VELOCITY_IDLE = 0.005
VELOCITY_APPROACH = 0.01
OPEN_HAND_THRESHOLD = 0.1


# --- Math & Geometry Helpers ---

def compute_local_frame(origin, point_x, point_y):
    """Compute a local 3D coordinate frame from 3 points."""
    x_axis = point_x - origin
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
    
    y_ref = point_y - origin
    z_axis = np.cross(x_axis, y_ref)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-10)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)
    
    return np.column_stack([x_axis, y_axis, z_axis])


def get_euler_angles(hand_landmarks):
    """Convert MediaPipe landmarks to joint Euler angles (ZYX convention)."""
    # Convert landmarks to numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    wrist = points[0]
    points = points - wrist  # Normalize to wrist at (0,0,0)
    
    euler_angles = {}
    
    def joint_euler(parent_idx, current_idx, child_idx, ref_idx):
        parent = points[parent_idx]
        current = points[current_idx]
        child = points[child_idx]
        ref_point = points[ref_idx]
        
        local_frame = compute_local_frame(current, child, ref_point)
        r = R.from_matrix(local_frame)
        return r.as_euler('ZYX', degrees=True)
    
    # Thumb (Reference: Index MCP)
    euler_angles['thumb_cmc'] = joint_euler(0, 1, 2, ref_idx=5)
    euler_angles['thumb_mcp'] = joint_euler(1, 2, 3, ref_idx=5)
    euler_angles['thumb_ip'] = joint_euler(2, 3, 4, ref_idx=5)
    
    # Fingers (References: Neighboring MCPs)
    euler_angles['index_mcp'] = joint_euler(0, 5, 6, ref_idx=9)
    euler_angles['index_pip'] = joint_euler(5, 6, 7, ref_idx=9)
    euler_angles['index_dip'] = joint_euler(6, 7, 8, ref_idx=9)
    
    euler_angles['middle_mcp'] = joint_euler(0, 9, 10, ref_idx=5)
    euler_angles['middle_pip'] = joint_euler(9, 10, 11, ref_idx=5)
    euler_angles['middle_dip'] = joint_euler(10, 11, 12, ref_idx=5)
    
    euler_angles['ring_mcp'] = joint_euler(0, 13, 14, ref_idx=9)
    euler_angles['ring_pip'] = joint_euler(13, 14, 15, ref_idx=9)
    euler_angles['ring_dip'] = joint_euler(14, 15, 16, ref_idx=9)
    
    euler_angles['pinky_mcp'] = joint_euler(0, 17, 18, ref_idx=13)
    euler_angles['pinky_pip'] = joint_euler(17, 18, 19, ref_idx=13)
    euler_angles['pinky_dip'] = joint_euler(18, 19, 20, ref_idx=13)
    
    return euler_angles

def get_wrist_pose(hand_landmarks):
    """
    Computes the 6DOF Pose (Position + Orientation) of the wrist in Camera Frame.
    """
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    # 1. Position: Wrist Landmark (Normalized 0-1)
    # Ideally, de-normalize this using depth if available, but for now we use relative Z
    wrist_pos = points[0] 
    
    # 2. Orientation: Derived from Wrist, Index, and Pinky (Palm Plane)
    index_mcp = points[5]
    pinky_mcp = points[17]
    
    # Compute rotation matrix where Z is normal to palm
    palm_rotation_matrix = compute_local_frame(wrist_pos, index_mcp, pinky_mcp)
    
    # Convert to Quaternion [x, y, z, w]
    r = R.from_matrix(palm_rotation_matrix)
    # quat = r.as_quat()
    
    return wrist_pos, r.as_euler('ZYX', degrees=True)



# --- Visualization ---

def draw_landmarks(image, hand_landmarks, handedness):
    
    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    h, w, _ = image.shape
    
    # Draw handedness label
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    cv2.putText(
        image, handedness,
        (int(min(xs) * w), int(min(ys) * h - MARGIN)),
        cv2.FONT_HERSHEY_DUPLEX, 0.8, HANDEDNESS_TEXT_COLOR, 1
    )

# --- Data Recording ---

def init_csv_if_needed(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return

    header = ["timestamp", "hand_index", "handedness"]
    for i in range(21):
        header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"]
    for joint in JOINT_ORDER:
        header += [f"{joint}_yaw", f"{joint}_pitch", f"{joint}_roll"]

    with open(path, "w") as f:
        f.write(",".join(header) + "\n")

def save_hand_data_to_csv(path, timestamp, hand_index, handedness, hand_landmarks, angles):
    row = [timestamp, hand_index, handedness]
    for lm in hand_landmarks:
        row += [lm.x, lm.y, lm.z]
    for joint in JOINT_ORDER:
        yaw, pitch, roll = angles[joint]
        row += [yaw, pitch, roll]
    
    with open(path, "a") as f:
        f.write(",".join(map(str, row)) + "\n")

# --- ROS 2 Integration ---

def init_ros():
    """Initialize ROS 2 node and publisher."""
    global ros_node, joint_angles_pub
    rclpy.init()
    ros_node = rclpy.create_node('hand_joint_publisher')
    
    joint_angles_pub = ros_node.create_publisher(Float32MultiArray, '/hand/joint_angles', 10)
    calibration_pub = ros_node.create_publisher(Empty, '/teleop/trigger_calibration', 10) # <--- TRIGGER

    ros_node.get_logger().info("ROS 2 Node Started. Publishing to /hand/joint_angles")
    ros_node.get_logger().info(" - Press 'C' to calibrate home position")


def trigger_calibration():
    """Publishes an Empty message to trigger calibration downstream."""
    if calibration_pub:
        msg = Empty()
        calibration_pub.publish(msg)
        print("[USER] Calibration Trigger Sent!")


def publish_hand_config(angles, wrist_pose):
    global joint_angles_pub, ros_node
    if not joint_angles_pub:
        return
        
    msg = Float32MultiArray()
    joint_config = []
    
    joint_config.extend(wrist_pose[0])  # Wrist Position
    joint_config.extend(wrist_pose[1])  # Wrist Orientation (Euler)

    for joint in JOINT_ORDER:
        yaw, pitch, roll = angles[joint]
        joint_config.extend([yaw, pitch, roll])
    
    msg.data = joint_config
    joint_angles_pub.publish(msg)
    
    # Allow ROS to process any callbacks (though we have none)
    # timeout_sec=0 ensures we don't block the video loop
    rclpy.spin_once(ros_node, timeout_sec=0)

# --- Main Application ---

def main():
    global last_timestamp, last_landmarks

    # 1. Setup Filesystem
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at: {model_path}")
        print("Please download 'hand_landmarker.task' and place it in the script directory.")
        return

    init_csv_if_needed(CSV_PATH)
    
    # 2. Setup ROS
    try:
        init_ros()
    except Exception as e:
        print(f"Failed to initialize ROS: {e}")
        return

    # 3. Setup MediaPipe Landmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Note: Use VIDEO mode for live streams to maintain tracking state (faster/smoother)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    last_frame = None
    last_hand_data = None
    
    # Timestamp for MediaPipe (must be monotonically increasing)
    start_time_ms = int(time.time() * 1000)

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Timestamp calculation
                curr_time_ms = int(time.time() * 1000)
                # MediaPipe requires relative timestamp from start of stream for VIDEO mode
                frame_timestamp_ms = curr_time_ms - start_time_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Process Frame
                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                annotated = frame.copy() # Work on BGR for display
                hand_data = []

                if result.hand_landmarks:
                    for i, hand_landmarks in enumerate(result.hand_landmarks):
                        # Safety check for handedness index
                        handedness = "Unknown"
                        if i < len(result.handedness):
                            handedness = result.handedness[i][0].category_name
                        
                        angles = get_euler_angles(hand_landmarks)
                        wrist_pose = get_wrist_pose(hand_landmarks)

                        # Publish to ROS
                        publish_hand_config(angles, wrist_pose)

                        # Draw
                        draw_landmarks(annotated, hand_landmarks, handedness)
                        
                        hand_data.append((hand_landmarks, handedness, angles))
                    
                    last_timestamp = time.time()
                    last_frame = annotated.copy()
                    last_hand_data = list(hand_data)

                # UI Overlay
                ui_text = "SPACE: Save Snapshot | Q: Quit" if DEBUG_FLAG else "Q: Quit"
                cv2.putText(annotated, "Press 'C' to Calibrate Home Position", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated, ui_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow("Hand Tracking (ROS 2 Humble)", annotated)

                # Input Handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == 32:  # SPACE
                    if DEBUG_FLAG and last_frame is not None and last_hand_data:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        img_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{ts}.png")
                        
                        cv2.imwrite(img_path, last_frame)
                        print(f"[INFO] Saved image: {img_path}")

                        for i, (lm, handed, ang) in enumerate(last_hand_data):
                            save_hand_data_to_csv(CSV_PATH, ts, i, handed, lm, ang)
                        print("[INFO] Data saved to CSV")

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if ros_node:
            ros_node.destroy_node()
        rclpy.shutdown()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
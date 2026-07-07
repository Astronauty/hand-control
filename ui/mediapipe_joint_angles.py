#!/usr/bin/env python3
import sys
import os
import time
import math
import argparse
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


# --- One Euro Filter ---
# Adaptive low-pass filter for noisy tracking signals (Casiez et al., 2012).
# At low velocity the cutoff is min_cutoff (heavy smoothing).
# At high velocity the cutoff rises as min_cutoff + beta*speed (less lag).
# Key parameters:
#   min_cutoff  (Hz) : smoothing at rest. Lower = smoother but laggier. ~1–2 Hz.
#   beta             : speed coefficient. Higher = less lag on fast moves. ~0.1–0.5.
#   d_cutoff    (Hz) : cutoff for the derivative estimate. Usually left at 1.0.

class OneEuroFilter:
    """One Euro Filter for a single scalar channel."""

    def __init__(self, freq: float = 30.0,
                 min_cutoff: float = 1.0,
                 beta: float = 0.2,
                 d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te  = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, t: float | None = None) -> float:
        # Update frequency from wall-clock timestamps if provided
        if t is not None and self._t_prev is not None:
            dt = t - self._t_prev
            if dt > 0:
                self.freq = 1.0 / dt
        if t is not None:
            self._t_prev = t

        if self._x_prev is None:
            self._x_prev = x
            return x

        # Derivative estimate → filter it → adaptive cutoff
        dx     = (x - self._x_prev) * self.freq
        a_d    = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter the signal
        a     = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev  = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev  = None
        self._dx_prev = 0.0
        self._t_prev  = None


class OneEuroFilterArray:
    """One Euro Filter applied independently to each element of a fixed-length array."""

    def __init__(self, n: int, freq: float = 30.0,
                 min_cutoff: float = 1.0,
                 beta: float = 0.2,
                 d_cutoff: float = 1.0):
        self._filters = [
            OneEuroFilter(freq, min_cutoff, beta, d_cutoff) for _ in range(n)
        ]

    def __call__(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        return np.array([f(float(v), t) for f, v in zip(self._filters, x)])

    def reset(self) -> None:
        for f in self._filters:
            f.reset()


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

def get_flexion_angles(world_landmarks):
    """Compute finger closure angles (degrees) for the index and thumb.

    Uses world landmarks (metric, wrist at origin).  All values increase as
    the hand closes from spread-open toward pinch.  No Euler-angle discontinuities.

    Returns 6 values:
        [index_mcp, index_pip, index_dip, thumb_spread, thumb_ip, thumb_ip*0.5]

    Index joints: inter-segment bend at MCP / PIP / DIP.
        0° = straight segment, ~70-90° = fully flexed.

    Thumb: the primary pinch motion is CMC adduction (the whole thumb sweeps
    across the palm), NOT MCP/IP flexion.  We capture this as the angle at the
    wrist between the direction to the index-MCP and the direction to the
    thumb-tip.  Large angle (60-100°) = spread open;  small angle (5-25°) = pinch.
    The ROM calibration handles the inverted direction automatically.

    Message layout (appended after the 51-value Euler block): indices 51-56.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])

    def angle_between(va, vb):
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))))

    def bend(p, j, c):
        return angle_between(pts[j] - pts[p], pts[c] - pts[j])

    idx_mcp = bend(0, 5, 6)    # wrist  → index-MCP  → index-PIP
    idx_pip = bend(5, 6, 7)    # index-MCP → PIP → DIP
    idx_dip = bend(6, 7, 8)    # index-PIP → DIP → tip

    # Thumb spread angle: angle at wrist between (wrist→index-MCP) and
    # (wrist→thumb-tip).  Large when spread apart, small when pinching.
    # This captures CMC adduction — the primary thumb closure mechanism.
    thumb_spread = angle_between(pts[5] - pts[0], pts[4] - pts[0])

    th_ip = bend(2, 3, 4)      # thumb-MCP → IP → tip  (secondary flexion)

    return [idx_mcp, idx_pip, idx_dip, thumb_spread, th_ip, th_ip * 0.5]


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



# --- Hand Selection ---

def _hand_bbox_area(hand_landmarks) -> float:
    """Return the bounding-box area of a hand in normalised image coordinates.

    Larger area == hand is closer to the camera (appears bigger in frame).
    Used to pick the dominant hand when multiple hands are detected.
    """
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


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
    # FIX: Add calibration_pub to global list
    global ros_node, joint_angles_pub, calibration_pub 
    
    rclpy.init()
    ros_node = rclpy.create_node('hand_joint_publisher')
    
    joint_angles_pub = ros_node.create_publisher(Float32MultiArray, '/hand/joint_angles', 10)
    calibration_pub = ros_node.create_publisher(Empty, '/teleop/trigger_calibration', 10)

    ros_node.get_logger().info("ROS 2 Node Started. Publishing to /hand/joint_angles")
    ros_node.get_logger().info(" - Press 'C' to calibrate home position")


def trigger_calibration():
    """Publishes an Empty message to trigger calibration downstream."""
    if calibration_pub:
        msg = Empty()
        calibration_pub.publish(msg)
        print("[USER] Calibration Trigger Sent!")


def publish_hand_config(angles, wrist_pose, flexion_angles=None, world_landmarks=None):
    global joint_angles_pub, ros_node
    if not joint_angles_pub:
        return

    msg = Float32MultiArray()
    joint_config = []

    joint_config.extend(wrist_pose[0])  # Wrist Position          [0:3]
    joint_config.extend(wrist_pose[1])  # Wrist Orientation (Euler)[3:6]

    for joint in JOINT_ORDER:
        yaw, pitch, roll = angles[joint]
        joint_config.extend([yaw, pitch, roll])               # [6:51]

    # Append inter-segment flexion angles (robust, no discontinuities) [51:57]
    # [idx_mcp, idx_pip, idx_dip, th_mcp, th_ip, th_ip*0.5]
    if flexion_angles is not None:
        joint_config.extend(flexion_angles)

    # Append world landmark positions: 21 landmarks × 3 coords = 63 floats [57:120]
    # Metric units (metres), wrist approximately at origin.
    if world_landmarks is not None:
        joint_config.extend(world_landmarks)

    msg.data = joint_config
    joint_angles_pub.publish(msg)
    
    # Allow ROS to process any callbacks (though we have none)
    # timeout_sec=0 ensures we don't block the video loop
    rclpy.spin_once(ros_node, timeout_sec=0)

# --- Main Application ---

_VIRTUAL_CAM_KEYWORDS = ("obs", "virtual", "dshow2", "ndi", "manycam", "xsplit", "camtwist")


def _get_camera_names_windows(max_index: int) -> list[str]:
    """Query DirectShow camera friendly names via PowerShell (Windows only). Order matches OpenCV indices."""
    import subprocess
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-PnpDevice -Class Camera -Status OK | Select-Object -ExpandProperty FriendlyName"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            names = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            # Pad to max_index+1 entries so list index matches camera index
            while len(names) <= max_index:
                names.append("")
            return names
    except Exception:
        pass
    return [""] * (max_index + 1)


def _probe_cameras(max_index: int = 4) -> list[int]:
    """Return indices of real cameras first, virtual cameras (OBS etc.) last."""
    import sys
    names = _get_camera_names_windows(max_index) if sys.platform == "win32" else [""] * (max_index + 1)
    real, virtual = [], []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        cap.release()
        name = names[i].lower() if i < len(names) else ""
        is_virtual = any(kw in name for kw in _VIRTUAL_CAM_KEYWORDS)
        if is_virtual:
            print(f"[INFO] Camera {i} ({names[i]}): virtual cam — deprioritised.")
            virtual.append(i)
        else:
            label = f" ({names[i]})" if names[i] else ""
            print(f"[INFO] Camera {i}{label}: real camera.")
            real.append(i)
    return real + virtual


def main():
    global last_timestamp, last_landmarks

    parser = argparse.ArgumentParser(description="Hand tracking publisher")
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera index to use (0=laptop built-in, 1=first USB camera, etc.). "
             "Omit to auto-select: prefers index 1+ (external webcam) if available.")
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="List available camera indices and exit.")
    args = parser.parse_args()

    if args.list_cameras:
        cams = _probe_cameras()
        print(f"Available cameras: {cams}")
        print("  0 is usually the laptop built-in camera.")
        print("  1+ are external/USB cameras (e.g. EMEET).")
        return

    # Auto-select camera: prefer external (index >= 1) for better index+thumb visibility
    if args.camera is not None:
        camera_index = args.camera
    else:
        available = _probe_cameras()
        if not available:
            print("ERROR: No cameras found.")
            return
        # Prefer first external camera; fall back to laptop if only one exists
        camera_index = next((i for i in available if i >= 1), available[0])
        print(f"[INFO] Available cameras: {available}  -- using index {camera_index}")
        if camera_index == 0:
            print("[INFO] Only built-in camera found. For better index+thumb tracking,")
            print("       plug in your EMEET webcam and set a side-view angle, then rerun.")

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
        num_hands=3,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {camera_index}.")
        print(f"       Run with --list-cameras to see available indices.")
        return
    print(f"[INFO] Camera {camera_index} opened.")

    # One Euro Filters — applied per-channel before publishing.
    # Wrist: [x, y, z, yaw, pitch, roll]  (image-space position + orientation)
    # Flexion: [idx_mcp, idx_pip, idx_dip, thumb_spread, thumb_ip, thumb_ip*0.5]
    #
    # Tuning guide:
    #   min_cutoff : smoothing at rest (Hz). Lower = smoother, more lag. ~1-2 Hz.
    #   beta       : speed adaptation. Higher = less lag on fast moves. ~0.1-0.4.
    #
    # Wrist gets slightly higher beta so base motion feels responsive.
    # Flexion gets a lower min_cutoff for heavier smoothing of finger jitter.
    filter_wrist    = OneEuroFilterArray(6,  freq=30.0, min_cutoff=1.0, beta=0.3)
    filter_flexion  = OneEuroFilterArray(6,  freq=30.0, min_cutoff=1.5, beta=0.2)
    filter_world_lm = OneEuroFilterArray(63, freq=30.0, min_cutoff=1.5, beta=0.2)
    last_frame = None
    last_hand_data = None
    show_debug = False  # toggled with 'D' — overlays flexion angles on video

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
                    # Pick the closest hand: largest bounding box in image space.
                    # Draw all detected hands so the operator can see what's tracked,
                    # but only publish and filter the selected one.
                    best_i = max(
                        range(len(result.hand_landmarks)),
                        key=lambda i: _hand_bbox_area(result.hand_landmarks[i]),
                    )

                    for i, lm in enumerate(result.hand_landmarks):
                        label = (result.handedness[i][0].category_name
                                 if i < len(result.handedness) else "?")
                        draw_landmarks(annotated, lm,
                                       label if i == best_i else f"{label} (ignored)")

                    # --- Process only the closest hand ---
                    hand_landmarks = result.hand_landmarks[best_i]
                    handedness = (result.handedness[best_i][0].category_name
                                  if best_i < len(result.handedness) else "Unknown")

                    world_lm = (result.hand_world_landmarks[best_i]
                                if result.hand_world_landmarks
                                and best_i < len(result.hand_world_landmarks)
                                else hand_landmarks)

                    angles = get_euler_angles(world_lm)
                    wrist_pos, wrist_euler = get_wrist_pose(hand_landmarks)
                    flexion = get_flexion_angles(world_lm)

                    # Apply One Euro Filter before publishing.
                    # t is wall-clock seconds — used to adapt the filter's
                    # internal frequency estimate to the actual frame rate.
                    t = curr_time_ms / 1000.0
                    wrist_state = filter_wrist(
                        np.concatenate([wrist_pos, wrist_euler]), t)
                    flexion_f = filter_flexion(np.array(flexion), t)

                    wrist_pose_f = (wrist_state[:3], wrist_state[3:])

                    world_flat = np.array(
                        [[lm.x, lm.y, lm.z] for lm in world_lm]).flatten()
                    world_flat_f = filter_world_lm(world_flat, t)

                    # Publish to ROS (Euler + flexion + world landmarks)
                    publish_hand_config(angles, wrist_pose_f,
                                        flexion_f.tolist(), world_flat_f.tolist())

                    if show_debug:
                        labels = ["idx_mcp", "idx_pip", "idx_dip",
                                  "th_spr ", "th_ip  ", "th_ip*.5"]
                        x0 = annotated.shape[1] - 190
                        for j, (lbl, val) in enumerate(zip(labels, flexion_f)):
                            cv2.putText(annotated,
                                        f"{lbl}:{val:5.1f}",
                                        (x0, 30 + j * 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 1)

                    hand_data.append((hand_landmarks, handedness, angles))

                    last_timestamp = time.time()
                    last_frame = annotated.copy()
                    last_hand_data = list(hand_data)
                else:
                    # Hand lost — reset filters so stale state doesn't corrupt
                    # the first frame when the hand reappears.
                    filter_wrist.reset()
                    filter_flexion.reset()
                    filter_world_lm.reset()

                # UI Overlay
                dbg_label = "[D:ON]" if show_debug else "D:Debug"
                ui_text = (f"{dbg_label} | SPACE:Snap | Q:Quit"
                           if DEBUG_FLAG else f"{dbg_label} | Q:Quit")
                cv2.putText(annotated, "In MuJoCo: C -> spread open, C -> pinch to calibrate",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(annotated, ui_text, (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow(f"Hand Tracking  [cam {camera_index}]", annotated)

                # Input Handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    trigger_calibration()
                elif key == ord("d"):
                    show_debug = not show_debug
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
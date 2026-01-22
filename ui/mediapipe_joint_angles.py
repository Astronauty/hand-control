import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import os
import time
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# Configure mediapipe options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

### Debug/Visualization config
# Visualization config
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

ANGLE_TEXT_COLOR = (255, 150, 150)  # red/pink for angles
ANGLE_FONT_SIZE = 0.25
ANGLE_FONT_THICKNESS = 1

SNAPSHOT_DIR = "hand_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
CSV_PATH = os.path.join(SNAPSHOT_DIR, "hand_snapshots.csv")

JOINT_ORDER = [
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
]

DEBUG_FLAG = True


### Utilities for getting joint angles from hand landmarks


def compute_local_frame(origin, point_x, point_y):

    x_axis = point_x - origin
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
    
    y_ref = point_y - origin
    z_axis = np.cross(x_axis, y_ref)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-10)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)
    
    return np.column_stack([x_axis, y_axis, z_axis])


def get_euler_angles(hand_landmarks):

    # Convert landmarks to numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    wrist = points[0]   # USE WRIST AS ORIGIN
    points = points - wrist  # Now wrist is at (0, 0, 0)
    
    euler_angles = {}
    
    def joint_euler(parent_idx, current_idx, child_idx, ref_idx):
        """Calculate Euler angles for a joint"""
        parent = points[parent_idx]
        current = points[current_idx]
        child = points[child_idx]
        ref_point = points[ref_idx]
        
        # Compute local coordinate frame
        local_frame = compute_local_frame(current, child, ref_point)
        
        # Convert to Euler angles (ZYX convention)
        r = R.from_matrix(local_frame)
        euler = r.as_euler('ZYX', degrees=True)
        
        return euler  # [yaw, pitch, roll]
    
    # Thumb (use index finger as reference)
    euler_angles['thumb_cmc'] = joint_euler(0, 1, 2, ref_idx=5)
    euler_angles['thumb_mcp'] = joint_euler(1, 2, 3, ref_idx=5)
    euler_angles['thumb_ip'] = joint_euler(2, 3, 4, ref_idx=5)
    
    # Index finger (use middle finger as reference)
    euler_angles['index_mcp'] = joint_euler(0, 5, 6, ref_idx=9)
    euler_angles['index_pip'] = joint_euler(5, 6, 7, ref_idx=9)
    euler_angles['index_dip'] = joint_euler(6, 7, 8, ref_idx=9)
    
    # Middle finger (use index finger as reference)
    euler_angles['middle_mcp'] = joint_euler(0, 9, 10, ref_idx=5)
    euler_angles['middle_pip'] = joint_euler(9, 10, 11, ref_idx=5)
    euler_angles['middle_dip'] = joint_euler(10, 11, 12, ref_idx=5)
    
    # Ring finger (use middle finger as reference)
    euler_angles['ring_mcp'] = joint_euler(0, 13, 14, ref_idx=9)
    euler_angles['ring_pip'] = joint_euler(13, 14, 15, ref_idx=9)
    euler_angles['ring_dip'] = joint_euler(14, 15, 16, ref_idx=9)
    
    # Pinky finger (use ring finger as reference)
    euler_angles['pinky_mcp'] = joint_euler(0, 17, 18, ref_idx=13)
    euler_angles['pinky_pip'] = joint_euler(17, 18, 19, ref_idx=13)
    euler_angles['pinky_dip'] = joint_euler(18, 19, 20, ref_idx=13)
    
    return euler_angles


########### Not used currently ###############

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)  # First point
    b = np.array(b)  # Vertex point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def get_finger_angles(hand_landmarks):
    """Extract finger joint angles from hand landmarks"""
    landmarks = [[lm.x, lm.y] for lm in hand_landmarks]
    
    angles = {}

    # Thumb angles
    angles['thumb_cmc'] = calculate_angle(landmarks[0], landmarks[1], landmarks[2])
    angles['thumb_mcp'] = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
    angles['thumb_ip'] = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
    
    # Index finger angles
    angles['index_mcp'] = calculate_angle(landmarks[0], landmarks[5], landmarks[6])
    angles['index_pip'] = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
    angles['index_dip'] = calculate_angle(landmarks[6], landmarks[7], landmarks[8])
    
    # Middle finger angles
    angles['middle_mcp'] = calculate_angle(landmarks[0], landmarks[9], landmarks[10])
    angles['middle_pip'] = calculate_angle(landmarks[9], landmarks[10], landmarks[11])
    angles['middle_dip'] = calculate_angle(landmarks[10], landmarks[11], landmarks[12])
    
    # Ring finger angles
    angles['ring_mcp'] = calculate_angle(landmarks[0], landmarks[13], landmarks[14])
    angles['ring_pip'] = calculate_angle(landmarks[13], landmarks[14], landmarks[15])
    angles['ring_dip'] = calculate_angle(landmarks[14], landmarks[15], landmarks[16])
    
    # Pinky finger angles
    angles['pinky_mcp'] = calculate_angle(landmarks[0], landmarks[17], landmarks[18])
    angles['pinky_pip'] = calculate_angle(landmarks[17], landmarks[18], landmarks[19])
    angles['pinky_dip'] = calculate_angle(landmarks[18], landmarks[19], landmarks[20])
    
    return angles



###### Annotate images with landmarks and angles #######


def draw_landmarks(image, hand_landmarks, handedness):
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in hand_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image, proto, solutions.hands.HAND_CONNECTIONS
    )

    h, w, _ = image.shape
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    cv2.putText(
        image, handedness,
        (int(min(xs) * w), int(min(ys) * h) - 10),
        cv2.FONT_HERSHEY_DUPLEX, 1, HANDEDNESS_TEXT_COLOR, 1
    )


def draw_angles(image, hand_landmarks, angles):
    h, w, _ = image.shape
    joint_pos = {
        "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3,
        "index_mcp": 5, "index_pip": 6, "index_dip": 7,
        "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11,
        "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15,
        "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19,
    }

    for joint, idx in joint_pos.items():
        yaw, pitch, roll = angles[joint]
        lm = hand_landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)

        cv2.putText(
            image,
            f"{yaw:.0f},{pitch:.0f},{roll:.0f}",
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            ANGLE_TEXT_COLOR,
            1,
        )



############  Saving utils ##############
    
def init_csv_if_needed(path):
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


###################  Main function  ###############################



init_csv_if_needed(CSV_PATH)

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)

last_frame = None
last_hand_data = None

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)
        annotated = rgb.copy()
        hand_data = []

        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[i][0].category_name
                angles = get_euler_angles(hand_landmarks)

                draw_landmarks(annotated, hand_landmarks, handedness)
                # draw_angles(annotated, hand_landmarks, angles)

                hand_data.append((hand_landmarks, handedness, angles))

            # 🔒 Freeze frame + data together
            last_frame = annotated.copy()
            last_hand_data = list(hand_data)

        display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        
        

        if DEBUG_FLAG == True:
            cv2.putText(display, "SPACE: Save | Q: Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display, "Q: Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Hand Tracking", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:   # SPACE

            if DEBUG_FLAG == True:
                if last_frame is None or not last_hand_data:
                    print("Nothing to save yet")
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                img_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{ts}.png")
                ok = cv2.imwrite(
                    img_path,
                    cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
                )

                print("Saved image:", img_path, "success:", ok)

                for i, (lm, handed, ang) in enumerate(last_hand_data):
                    save_hand_data_to_csv(
                        CSV_PATH, ts, i, handed, lm, ang
                    )

                print("Snapshot + CSV saved")

cap.release()
cv2.destroyAllWindows()

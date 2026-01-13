import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import os
import time
import numpy as np


# Configure mediapipe options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global buffer for latest annotated frame
latest_annotated_frame = None

### Visualization utils
# Visualization config
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        # Compute and print angles
        angles = get_finger_angles(hand_landmarks)
        print(f"Hand {idx} ({handedness[0].category_name}) angles:")
        for joint, angle in angles.items():
            print(f"  {joint}: {angle:.1f}°")
            
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Handedness label.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

### Utilities for getting joint angles from hand landmarks

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

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # Convert output image (RGB) and draw landmarks, store for display
    rgb = output_image.numpy_view()
    annotated = draw_landmarks_on_image(rgb, result)
    global latest_annotated_frame
    latest_annotated_frame = annotated
    


# Create a hand landmarker instance with the live stream mode:
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


# Use OpenCV’s VideoCapture to start capturing from the webcam.
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert OpenCV frame to numpy
        numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        
        # Track timestamps in ms
        frame_timestamp_ms = time.monotonic_ns() // 1_000_000
        
        # Run inference
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        # Display latest annotated frame
        if latest_annotated_frame is not None:
            display_bgr = cv2.cvtColor(latest_annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Hand Tracking', display_bgr)
        else:
            cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
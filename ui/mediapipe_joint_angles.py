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
from mediapipe.framework.formats import landmark_pb2
from geometry_msgs.msg import PoseStamped

# drawing_utils / drawing_styles live under mp.solutions (not mp.tasks.vision).
# draw_landmarks() expects the solutions HAND_CONNECTIONS frozenset format.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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


def _palm_frame_robot_aligned(wrist, idx_mcp, pky_mcp):
    """Palm frame whose axis ROLES match the robot pinch_site (and the arm
    controller's human_palm_frame_robot_aligned): X=palm normal (thumb×fingers),
    Y=toward thumb, Z=along fingers. Used ONLY for the overlay so the triad
    colours are consistent with the MuJoCo target triad."""
    z = 0.5 * (idx_mcp + pky_mcp) - wrist          # along fingers
    z = z / (np.linalg.norm(z) + 1e-10)
    thumb_dir = idx_mcp - pky_mcp
    x = np.cross(thumb_dir, z)                      # palm normal
    x = x / (np.linalg.norm(x) + 1e-10)
    y = np.cross(z, x)                              # toward thumb
    y = y / (np.linalg.norm(y) + 1e-10)
    return np.column_stack([x, y, z])


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


# Physical wrist->middle-MCP span of an adult hand [m]. Used as the known object
# size for monocular depth-from-apparent-size. Rough default; override per user
# by measuring landmark-0 to landmark-9 on your own hand.
_HAND_SPAN_M = 0.09

# Which hand to track/publish. MediaPipe's handedness label ("Right"/"Left")
# assumes a NON-mirrored image (this feed is not flipped). Set to "Left" to
# track the left hand instead.
_TRACK_HANDEDNESS = "Right"

# DISPLAY-ONLY 180deg rotation. When True, the shown window is rotated a
# half-turn (two 90deg turns) so the on-screen picture matches how your real arm
# looks to you (camera mounted upside-down / facing you). This is applied ONLY to
# the final displayed image, AFTER all landmark processing, overlays, and ROS
# publishing — so it does NOT affect the tracked coordinates or the robot mapping.
# The hand triad overlays rotate WITH the picture (still aligned to the hand); the
# text labels also rotate (a cosmetic tradeoff). Set False to disable.
_DISPLAY_ROTATE_180 = True

# Freeze the monocular depth in ABSOLUTE mode. estimate_wrist_depth() reads the
# apparent hand SIZE, which shrinks when you ROTATE your wrist (foreshortening) —
# misread as the hand moving in depth, so rotation caused the target to TRANSLATE.
# Since depth scales the WHOLE back-projected ray, that also skewed board-X/Y.
# Freezing depth to a constant makes the wrist position depend only on the (u,v)
# pixel — stable under rotation. Cost: no toward/away control on the depth axis
# (which was unreliable anyway). Set to None to restore live monocular depth.
_FROZEN_DEPTH_M = 0.5

# Board -> world axis remap. The camera looks DOWN at a flat board, so the board
# normal (+Z) points DOWN into the table while MuJoCo world +Z is UP. This proper
# rotation (det=+1) flips Y and Z so the published world frame is Z-up, matching
# MuJoCo. It is applied to the loaded extrinsic so BOTH the published wrist coords
# and the drawn board axes use the corrected frame. (A ~23° camera-tilt residual
# remains — the board alone can't sense gravity; lay the board under a truly
# vertical camera or add a gravity-align step to remove it. Fine for teleop.)
_WORLD_FROM_BOARD = np.diag([1.0, -1.0, -1.0])


def estimate_wrist_depth(hand_landmarks, fx: float, img_w: int) -> float:
    """Monocular depth of the hand from apparent size (single RGB camera).

    MediaPipe's wrist z is wrist-relative (~0), so it does NOT track distance to
    the camera. Apparent hand SIZE does: a physically constant span projects to
    fewer pixels as the hand recedes. With calibrated focal length fx:
        Z = fx * HAND_SPAN_M / span_pixels
    where span_pixels = (normalised wrist->middle-MCP distance) * img_w.

    Uses the wrist(0)->middle-MCP(9) segment: rigid, roughly in-plane, and less
    affected by finger flexion than the full bounding box. Returns metres; this
    is a noisy monocular estimate (best used as a DELTA, not absolute), and is
    the natural swap-in point for a RealSense depth reading later.
    """
    w = hand_landmarks[0]
    m = hand_landmarks[9]
    span_norm = float(np.hypot(m.x - w.x, m.y - w.y))
    span_px = span_norm * img_w
    if span_px < 1e-3:
        return 0.0
    return fx * _HAND_SPAN_M / span_px


def wrist_board_position(u_px, v_px, depth_m, K, dist, R_world_cam, t_cam_world):
    """Metric 3D wrist position in the fixed ChArUco board (world) frame.

    Back-projects the wrist PIXEL through the camera intrinsics at the estimated
    depth to a camera-frame point, then applies the (verified) extrinsic inverse
    to express it in board coordinates:

        p_cam   = [x_n*Z, y_n*Z, Z]          (x_n,y_n = undistorted normalised)
        p_board = R_world_cam @ (p_cam - t_cam_world)

    Depth Z is AXIAL (along the optical axis), matching estimate_wrist_depth's
    Z = fx*S/px — so the ray is NOT unit-normalised before scaling by depth.
    Returns a (3,) array in metres, board frame.
    """
    und = cv2.undistortPoints(np.array([[[u_px, v_px]]], dtype=float), K, dist)
    x_n, y_n = float(und[0, 0, 0]), float(und[0, 0, 1])
    p_cam = np.array([x_n * depth_m, y_n * depth_m, depth_m])
    return R_world_cam @ (p_cam - t_cam_world)


# ChArUco board geometry — must match the printed board (README: 5x7,
# DICT_5X5_100, 0.75 marker ratio, measured 45 mm square).
_BOARD_SQUARES = (5, 7)
_BOARD_SQUARE_M = 0.045
_BOARD_MARKER_RATIO = 0.75


def _make_charuco_detector():
    """Build the ChArUco detector for on-demand board re-calibration."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        _BOARD_SQUARES, _BOARD_SQUARE_M,
        _BOARD_SQUARE_M * _BOARD_MARKER_RATIO, d)
    return board, cv2.aruco.CharucoDetector(board)


def recalibrate_board(frame, board, detector, K, dist, extr_path):
    """Re-solve the board pose from the CURRENT frame (on-demand, key 'B').

    Fixed-rig re-calibration without leaving the teleop session: detect the
    board, solvePnP, and return fresh (R_world_cam_corrected, t_cam_world) with
    the Z-up remap already applied — plus persist raw extrinsics to disk so the
    next launch uses them. Returns None (pose unchanged) if the board isn't found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cc, ci, _, _ = detector.detectBoard(gray)
    if ci is None or len(ci) < 6:
        print(f"[recalib] board not detected ({0 if ci is None else len(ci)} corners) — pose unchanged.")
        return None
    obj_pts, img_pts = board.matchImagePoints(cc, ci)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        print("[recalib] solvePnP failed — pose unchanged.")
        return None
    R_cam_world, _ = cv2.Rodrigues(rvec)
    R_world_cam_raw = R_cam_world.T
    t_cam_world = tvec.ravel()
    # Persist raw extrinsics (same schema as charuco_calibration.py) so a restart
    # loads this pose; the Z-up remap is re-applied on load and here.
    import json as _json
    with open(extr_path, "w") as f:
        _json.dump({
            "R_cam_world": R_cam_world.tolist(),
            "t_cam_world": t_cam_world.tolist(),
            "R_world_cam": R_world_cam_raw.tolist(),
            "camera_pos_in_world": (-R_cam_world.T @ t_cam_world).tolist(),
            "board_distance_m": float(np.linalg.norm(t_cam_world)),
        }, f, indent=2)
    print(f"[recalib] board re-solved & saved ({len(ci)} corners, "
          f"dist {np.linalg.norm(t_cam_world)*100:.1f} cm).")
    return _WORLD_FROM_BOARD @ R_world_cam_raw, t_cam_world


# --- Visualization ---

# Deferred info-labels. When the display is rotated 180deg (_DISPLAY_ROTATE_180),
# corner text drawn before the rotation would come out upside-down. So draw_label
# BUFFERS its calls here and they are flushed by flush_labels() AFTER rotation, at
# rotation-corrected screen positions, keeping the text upright on a flipped image.
# When rotation is off the buffer is flushed as-is (positions unchanged).
_deferred_labels: list[tuple] = []


def draw_label(image, text, org, scale=0.55, thickness=1,
               color=(0, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, box=True):
    """Queue a label for upright rendering AFTER any display rotation.

    Buffered rather than drawn immediately so it can be composited after the
    180deg display flip (see flush_labels). `org` is the position in the UNROTATED
    image; flush_labels maps it to the rotated frame when needed. Style:
      box=True  -> white background box, black text (info readouts).
      box=False -> plain colored text, no box (hand-anchored labels).
    Works for ANY anchor (corner OR hand-following), since flush_labels remaps the
    position through the same 180deg transform."""
    _deferred_labels.append((text, org, scale, thickness, color, font, box))


def _render_label(image, text, org, scale, thickness, color, font, box):
    """Actually draw one label at `org` (optional white box + text)."""
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    if box:
        cv2.rectangle(image, (x - 3, y - th - 4), (x + tw + 3, y + base + 2),
                      (255, 255, 255), cv2.FILLED)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def flush_labels(image, rotated_180: bool):
    """Render all buffered labels UPRIGHT, then clear the buffer.

    If the image was rotated 180deg, a label anchored at (x, y) in the original
    frame must move to the mirrored screen position AND stay upright (not
    mirrored). We remap the anchor to (W-1-x-tw, H-1-y+th) and draw normally — the
    text glyphs themselves are never rotated, so they read correctly. This holds
    for any anchor, so hand-following labels track the (now-rotated) hand too."""
    h, w = image.shape[:2]
    for text, org, scale, thickness, color, font, box in _deferred_labels:
        if rotated_180:
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x, y = org
            # Map the anchor so the label lands at the same SCREEN position after a
            # 180deg flip: (x,y) -> (W-1-x-tw, H-1-y+th). Upright, not mirrored.
            org = (w - 1 - x - tw, h - 1 - y + th)
        _render_label(image, text, org, scale, thickness, color, font, box)
    _deferred_labels.clear()


def draw_board_axes(image, K, dist, R_cam_world, t_cam_world, axis_len=0.05):
    """Overlay the FIXED ChArUco board world frame (X=red, Y=green, Z=blue).

    drawFrameAxes needs the FORWARD extrinsic (board->camera): rvec from
    R_cam_world, tvec = t_cam_world. The axes stay glued to the board origin,
    perspective-correct, and serve as a live check that the calibration is valid.
    """
    rvec, _ = cv2.Rodrigues(R_cam_world)
    cv2.drawFrameAxes(image, K, dist, rvec, t_cam_world.reshape(3, 1), axis_len)


def draw_palm_frame(image, wrist_px, palm_R, length_px=40):
    """Overlay the MOVING palm frame as short 2D segments from the wrist pixel.

    palm_R columns are the palm-frame axes; we draw their image-plane (x,y)
    components from the wrist pixel. With the robot-aligned palm frame the colours
    mean the SAME roles as the MuJoCo target triad:
      red   = column 0 = palm normal
      green = column 1 = toward thumb
      blue  = column 2 = along fingers

    DEPTH COMPONENT: the (x,y) segment discards each axis' out-of-screen (z)
    component, which is exactly where the palm-normal flip happens (it points
    mostly INTO/OUT of the screen). To make the flip visible, draw a dot at each
    axis tip encoding its z:
      - dot radius grows with |z| (axis pointing more toward/away from camera),
      - FILLED  dot = z > 0 (toward camera / out of screen),
      - HOLLOW  dot = z < 0 (away from camera / into screen).
    A sign flip of the palm normal therefore shows as the red dot popping
    between filled and hollow at the instant the MuJoCo triad flips.
    """
    ox, oy = int(wrist_px[0]), int(wrist_px[1])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X,Y,Z
    for i, c in enumerate(colors):
        ax = palm_R[:, i]
        ex = int(ox + ax[0] * length_px)
        ey = int(oy + ax[1] * length_px)
        cv2.line(image, (ox, oy), (ex, ey), c, 2)
        # Depth cue at the tip: radius ~ |z|, fill = sign(z).
        z = float(ax[2])
        r = int(3 + abs(z) * 8)                 # 3px (in-plane) .. 11px (edge-on)
        thickness = cv2.FILLED if z > 0 else 2  # filled=out of screen, hollow=into
        cv2.circle(image, (ex, ey), r, c, thickness)


def draw_landmarks(image, hand_landmarks, handedness):

    # The Tasks API returns a plain list of landmarks, but mp.solutions'
    # draw_landmarks expects a NormalizedLandmarkList proto (with .landmark).
    # Convert before drawing.
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in hand_landmarks
    ])

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      image,
      proto,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    h, w, _ = image.shape

    # Handedness label — DEFERRED so it renders upright after any display
    # rotation (the hand SKELETON above is drawn immediately and rotates with the
    # picture; only the text needs to stay readable). box=False keeps the original
    # plain green DUPLEX style. Its hand-anchored position is remapped by
    # flush_labels through the same 180deg transform, so it still tracks the hand.
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    draw_label(
        image, handedness,
        (int(min(xs) * w), int(min(ys) * h - MARGIN)),
        scale=0.8, color=HANDEDNESS_TEXT_COLOR,
        font=cv2.FONT_HERSHEY_DUPLEX, box=False,
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


def publish_hand_config(angles, wrist_pose, flexion_angles=None,
                        world_landmarks=None, image_landmarks=None):
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
    # Metric units (metres), wrist approximately at origin. Used by finger
    # retargeting (needs metric 3D).
    if world_landmarks is not None:
        joint_config.extend(world_landmarks)

    # Append IMAGE landmark positions: 21 landmarks × 3 coords = 63 floats
    # [120:183]. Normalised image coords (x,y in [0,1], z = MediaPipe image
    # pseudo-depth). The arm's palm-frame NORMAL is built from these instead of
    # the world landmarks because the world-landmark depth flips the palm-normal
    # sign near edge-on poses, while the image-landmark frame stays stable
    # (verified via the dual-normal overlay). Requires world_landmarks present so
    # the [57:120] block keeps the layout fixed.
    if image_landmarks is not None:
        joint_config.extend(image_landmarks)

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

    # Load calibrated camera model for monocular depth AND absolute board-frame
    # wrist positioning. fx alone drives depth-from-hand-size; the full K + dist
    # + extrinsics (R_world_cam, t_cam_world) let us back-project the wrist pixel
    # into the fixed ChArUco board frame (see wrist_board_position). If intrinsics
    # are missing we fall back to an fx guess (depth-as-delta only); if extrinsics
    # are missing we publish legacy normalised wrist coords (delta teleop).
    import json as _json
    _calib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "calibration")
    _intr_path = os.path.join(_calib_dir, "camera_intrinsics.json")
    _extr_path = os.path.join(_calib_dir, "camera_extrinsics.json")
    _K = _dist = _R_world_cam = _t_cam_world = None
    _calib_ok = False   # True only when BOTH intrinsics + extrinsics loaded
    try:
        with open(_intr_path) as _f:
            _intr = _json.load(_f)
        _K = np.array(_intr["camera_matrix"], dtype=float)
        _dist = np.array(_intr["dist_coeffs"], dtype=float)
        _fx = float(_K[0, 0])
        print(f"[INFO] loaded fx={_fx:.1f} for depth estimation.")
        with open(_extr_path) as _f:
            _extr = _json.load(_f)
        # Fold the board->world remap into the extrinsic so the published frame
        # is Z-up (MuJoCo-consistent). _R_world_cam then maps cam -> WORLD axes.
        _R_world_cam = _WORLD_FROM_BOARD @ np.array(_extr["R_world_cam"], dtype=float)
        _t_cam_world = np.array(_extr["t_cam_world"], dtype=float)
        _calib_ok = True
        print("[INFO] loaded extrinsics — publishing ABSOLUTE Z-up world-frame wrist (metres).")
    except (FileNotFoundError, KeyError) as _e:
        if _K is None:
            _fx = 600.0
            print("[INFO] no intrinsics; using fx guess for depth (delta-only).")
        else:
            print("[INFO] no extrinsics; publishing LEGACY normalised wrist (delta teleop).")

    # Rolling window of the per-axis wrist-vs-board angles, to measure per-axis
    # NOISE (std). If the palm-NORMAL axis (camera-depth direction) is much
    # noisier than thumb/fingers, that's the monocular single-camera signature.
    _ang_hist = deque(maxlen=30)

    # ChArUco detector for on-demand board re-calibration (key 'B'). Needs
    # intrinsics; re-solving updates the world frame live without a restart.
    _board = _charuco_det = None
    if _K is not None:
        _board, _charuco_det = _make_charuco_detector()
        print("[INFO] press 'B' to re-solve the board pose (re-calibrate the world frame).")

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
    #
    # In ABSOLUTE mode the wrist x/y/z are metric board coords (~0.1-0.5 m), not
    # normalised [0,1], and the metric depth is noisy — so smooth harder (lower
    # min_cutoff) to keep absolute position from jittering. Euler channels are
    # degrees in both modes; a low cutoff is fine for them too.
    if _calib_ok:
        filter_wrist = OneEuroFilterArray(6, freq=30.0, min_cutoff=0.5, beta=0.15)
    else:
        filter_wrist = OneEuroFilterArray(6, freq=30.0, min_cutoff=1.0, beta=0.3)
    filter_flexion  = OneEuroFilterArray(6,  freq=30.0, min_cutoff=1.5, beta=0.2)
    filter_world_lm = OneEuroFilterArray(63, freq=30.0, min_cutoff=1.5, beta=0.2)
    # Image landmarks feed the arm palm-frame normal (more stable than world LM
    # near edge-on). Same smoothing profile as the world-landmark filter.
    filter_image_lm = OneEuroFilterArray(63, freq=30.0, min_cutoff=1.5, beta=0.2)
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

                # Fixed ChArUco board world axes (verifies calibration validity).
                if _calib_ok:
                    draw_board_axes(annotated, _K, _dist,
                                    _R_world_cam.T, _t_cam_world)

                # Track the RIGHT hand only. Among right-handed detections, pick
                # the closest (largest bounding box). The frame is NOT mirrored,
                # so MediaPipe's handedness labels are trustworthy. best_i is None
                # when no right hand is visible -> treated like "hand lost".
                best_i = None
                if result.hand_landmarks:
                    def _label(i):
                        return (result.handedness[i][0].category_name
                                if i < len(result.handedness) else "?")

                    _right_idxs = [i for i in range(len(result.hand_landmarks))
                                   if _label(i) == _TRACK_HANDEDNESS]
                    if _right_idxs:
                        best_i = max(_right_idxs,
                                     key=lambda i: _hand_bbox_area(result.hand_landmarks[i]))

                    for i, lm in enumerate(result.hand_landmarks):
                        label = _label(i)
                        draw_landmarks(annotated, lm,
                                       label if i == best_i else f"{label} (ignored)")

                if best_i is not None:
                    # --- Process only the selected right hand ---
                    hand_landmarks = result.hand_landmarks[best_i]
                    handedness = (result.handedness[best_i][0].category_name
                                  if best_i < len(result.handedness) else "Unknown")

                    world_lm = (result.hand_world_landmarks[best_i]
                                if result.hand_world_landmarks
                                and best_i < len(result.hand_world_landmarks)
                                else hand_landmarks)

                    angles = get_euler_angles(world_lm)
                    wrist_pos, wrist_euler = get_wrist_pose(hand_landmarks)
                    wrist_pos = np.array(wrist_pos, dtype=float)

                    fh, fw = frame.shape[0], frame.shape[1]
                    u_px = hand_landmarks[0].x * fw
                    v_px = hand_landmarks[0].y * fh

                    if _calib_ok:
                        # ABSOLUTE mode: publish the metric wrist position in the
                        # fixed board frame (metres). Depth is FROZEN (see
                        # _FROZEN_DEPTH_M) so wrist rotation — which changes
                        # apparent hand size — no longer translates the target.
                        # The wrist position then depends only on the (u,v) pixel.
                        depth_m = (_FROZEN_DEPTH_M if _FROZEN_DEPTH_M is not None
                                   else estimate_wrist_depth(hand_landmarks, _fx, fw))
                        wrist_pos = wrist_board_position(
                            u_px, v_px, depth_m, _K, _dist,
                            _R_world_cam, _t_cam_world)
                    else:
                        # LEGACY: normalised image x,y + monocular depth in z.
                        wrist_pos[2] = estimate_wrist_depth(hand_landmarks, _fx, fw)
                    flexion = get_flexion_angles(world_lm)

                    # Moving palm frame overlay (2D). Build with the SAME axis
                    # ROLES the arm controller uses (X=palm-normal, Y=toward-thumb,
                    # Z=along-fingers) so the triad colours mean the same thing as
                    # the MuJoCo target triad: red=normal, green=thumb, blue=fingers.
                    _pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                    _palm_R = _palm_frame_robot_aligned(_pts[0], _pts[5], _pts[17])
                    draw_palm_frame(annotated, (u_px, v_px), _palm_R)
                    # Numeric normal-depth readout: watch this cross zero — that
                    # is where cross(thumb_dir, fingers) degenerates and the
                    # palm-normal sign flips (the MuJoCo triad flip). |Nz| near 1
                    # = palm edge-on to the camera (worst case for roll).
                    draw_label(annotated,
                        f"palm normal z (img): {float(_palm_R[2, 0]):+.2f}  "
                        f"(|z|~1 edge-on; sign flip => normal flip)",
                        (10, fh - 88))
                    # SECOND normal built from WORLD landmarks — the frame the ARM
                    # actually consumes (raw[57:120] -> human_palm_frame_robot_aligned).
                    # The overlay above uses IMAGE landmarks; the arm uses WORLD
                    # landmarks (a different MediaPipe depth estimate). Draw the
                    # world-built normal as an extra dot so we can see which one
                    # flips: if WORLD flips while IMG stays stable => the flip is in
                    # MediaPipe's world-landmark depth (data; RealSense fixes it),
                    # NOT the frame construction and NOT the IK. Drawn as a hollow
                    # magenta ring at the SAME image-plane tip; its FILL encodes the
                    # world-normal's out-of-screen sign (filled=toward cam).
                    _wlm2 = np.array([[lm.x, lm.y, lm.z] for lm in world_lm])
                    _palm_w2 = _palm_frame_robot_aligned(_wlm2[0], _wlm2[5], _wlm2[17])
                    _wn = _palm_w2[:, 0]                       # world palm normal
                    _wex = int(u_px + _wn[0] * 40)
                    _wey = int(v_px + _wn[1] * 40)
                    _wthick = cv2.FILLED if _wn[2] > 0 else 2
                    cv2.circle(annotated, (_wex, _wey), 6, (255, 0, 255), _wthick)
                    draw_label(annotated,
                        f"palm normal z (WORLD/arm): {float(_wn[2]):+.2f}  "
                        f"(magenta dot; compare to img red)",
                        (10, fh - 110))

                    # --- Sanity readout: wrist pose in the BOARD (world) frame ---
                    # Verify the MediaPipe side is reasonable BEFORE debugging the
                    # MuJoCo mapping. Only meaningful when calibration is loaded.
                    if _calib_ok:
                        # Position (metres) in the board frame — per-axis distances
                        # from the board origin (already computed as wrist_pos).
                        _wp = wrist_pos
                        draw_label(annotated,
                            f"wrist pos (board m): X={_wp[0]:+.2f} Y={_wp[1]:+.2f} Z={_wp[2]:+.2f}",
                            (10, fh - 66))
                        # Orientation: build the palm frame from WORLD landmarks
                        # (a real 3D frame, in MediaPipe world axes), map it into
                        # the board frame, and report each axis' angle from the
                        # board axes (0deg = wrist axis aligned with that board axis).
                        _wlm = np.array([[lm.x, lm.y, lm.z] for lm in world_lm])
                        _palm_w = _palm_frame_robot_aligned(_wlm[0], _wlm[5], _wlm[17])
                        _R_mp_to_board = _R_world_cam @ np.diag([1.0, -1.0, -1.0])
                        _palm_board = _R_mp_to_board @ _palm_w   # cols in board axes
                        # Angle of each wrist axis from the SAME-named board axis
                        # (0=aligned, 90=perpendicular, 180=flipped). No abs, so a
                        # flipped axis shows as ~180 -> easy to spot.
                        _ang = [np.degrees(np.arccos(np.clip(_palm_board[i, i], -1, 1)))
                                for i in range(3)]
                        draw_label(annotated,
                            f"wrist ori vs board (deg): N={_ang[0]:4.0f} T={_ang[1]:4.0f} F={_ang[2]:4.0f}",
                            (10, fh - 44))
                        # Per-axis NOISE (std over the rolling window). Hold your
                        # hand STILL and read this: if N (palm-normal, the
                        # camera-depth axis) is much noisier than T/F, that's the
                        # monocular single-camera limitation (out-of-plane rotation
                        # poorly observed) — the likely residual floor.
                        _ang_hist.append(_ang)
                        if len(_ang_hist) >= 10:
                            _s = np.std(np.array(_ang_hist), axis=0)
                            draw_label(annotated,
                                f"ori NOISE std (deg): N={_s[0]:4.1f} T={_s[1]:4.1f} F={_s[2]:4.1f}  "
                                f"(N>>T,F => single-cam depth noise)",
                                (10, fh - 22))

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

                    # IMAGE landmarks (normalised) for the arm palm-frame normal.
                    image_flat = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
                    image_flat_f = filter_image_lm(image_flat, t)

                    # Publish to ROS (Euler + flexion + world LM + image LM)
                    publish_hand_config(angles, wrist_pose_f,
                                        flexion_f.tolist(), world_flat_f.tolist(),
                                        image_flat_f.tolist())

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
                    filter_image_lm.reset()

                # UI Overlay — deferred (drawn upright after any display rotation).
                dbg_label = "[D:ON]" if show_debug else "D:Debug"
                ui_text = (f"{dbg_label} | SPACE:Snap | Q:Quit"
                           if DEBUG_FLAG else f"{dbg_label} | Q:Quit")
                draw_label(annotated, "In MuJoCo: C -> spread open, C -> pinch to calibrate",
                           (10, 30))
                draw_label(annotated, ui_text, (10, 52))

                # Display-only 180deg rotation (see _DISPLAY_ROTATE_180). Rotate a
                # LOCAL copy right before showing so the tracked data, overlays'
                # math, and last_frame stay unrotated — this is purely cosmetic.
                # The HAND landmark/triad overlays rotate WITH the picture (they're
                # already baked into `annotated`); the info-LABELS are rendered
                # AFTER rotation by flush_labels so they stay upright and readable.
                _disp = (cv2.rotate(annotated, cv2.ROTATE_180)
                         if _DISPLAY_ROTATE_180 else annotated.copy())
                flush_labels(_disp, _DISPLAY_ROTATE_180)
                cv2.imshow(f"Hand Tracking  [cam {camera_index}]", _disp)

                # Input Handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    trigger_calibration()
                elif key == ord("d"):
                    show_debug = not show_debug
                elif key == ord("b"):
                    # On-demand board re-calibration (fixed rig, no restart).
                    if _charuco_det is not None:
                        _res = recalibrate_board(frame, _board, _charuco_det,
                                                 _K, _dist, _extr_path)
                        if _res is not None:
                            _R_world_cam, _t_cam_world = _res
                            _calib_ok = True
                            print("[recalib] world frame updated live.")
                    else:
                        print("[recalib] no intrinsics loaded — cannot re-solve board.")
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
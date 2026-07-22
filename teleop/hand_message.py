"""Shared builders for the /hand/joint_angles message (Float32MultiArray).

The single-camera publisher (ui/mediapipe_joint_angles.py) and the multi-camera
fusion node (teleop/hand_fusion_node.py) must emit the SAME 183-float layout so
everything downstream (DexPilotController / retargeter / arm IK) is unchanged.
The layout math is centralised here so the two producers can't drift.

Message layout (indices into Float32MultiArray.data):
    [0:3]     wrist position          (board/world metres, or legacy image x,y,depth)
    [3:6]     wrist orientation        (Euler ZYX degrees, palm frame)
    [6:51]    15 joints x 3            (Euler ZYX degrees; JOINT_ORDER)
    [51:57]   flexion angles           [idx_mcp, idx_pip, idx_dip, thumb_spread,
                                        thumb_ip, thumb_ip*0.5]  (degrees)
    [57:120]  21 world landmarks x 3   (metres, wrist ~origin; palm frame source)
    [120:183] 21 image landmarks x 3   (normalised; arm palm-normal source)

The consumer (teleop/ros_interface.py + dexpilot_controller.py) reads these
blocks by fixed offset, so producers MUST keep every block present and in order.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# Joint order for the [6:51] Euler block. Must match the downstream reader.
JOINT_ORDER = [
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
]

# MediaPipe hand skeleton bone connections (landmark index pairs). Used by the
# per-camera and fused-landmark visualisers to draw bones, not just dots.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm base
]

# Board -> world axis remap (must match ui/mediapipe_joint_angles.py). The camera
# looks DOWN at the board, so the board normal points into the table; this proper
# rotation (det=+1) flips Y and Z so the published world frame is Z-up like MuJoCo.
WORLD_FROM_BOARD = np.diag([1.0, -1.0, -1.0])


def compute_local_frame(origin, point_x, point_y):
    """Right-handed frame from 3 points: x toward point_x, z normal to the plane."""
    x_axis = point_x - origin
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
    y_ref = point_y - origin
    z_axis = np.cross(x_axis, y_ref)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-10)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)
    return np.column_stack([x_axis, y_axis, z_axis])


def get_euler_angles(points: np.ndarray) -> dict:
    """Per-joint Euler angles (ZYX degrees) from a (21,3) landmark array.

    `points` are wrist-relative-normalised internally. Mirrors the original
    get_euler_angles() in ui/mediapipe_joint_angles.py exactly.
    """
    points = np.asarray(points, float)
    wrist = points[0]
    points = points - wrist

    def joint_euler(parent_idx, current_idx, child_idx, ref_idx):
        current = points[current_idx]
        child = points[child_idx]
        ref_point = points[ref_idx]
        local_frame = compute_local_frame(current, child, ref_point)
        return R.from_matrix(local_frame).as_euler('ZYX', degrees=True)

    ea = {}
    ea['thumb_cmc'] = joint_euler(0, 1, 2, ref_idx=5)
    ea['thumb_mcp'] = joint_euler(1, 2, 3, ref_idx=5)
    ea['thumb_ip'] = joint_euler(2, 3, 4, ref_idx=5)
    ea['index_mcp'] = joint_euler(0, 5, 6, ref_idx=9)
    ea['index_pip'] = joint_euler(5, 6, 7, ref_idx=9)
    ea['index_dip'] = joint_euler(6, 7, 8, ref_idx=9)
    ea['middle_mcp'] = joint_euler(0, 9, 10, ref_idx=5)
    ea['middle_pip'] = joint_euler(9, 10, 11, ref_idx=5)
    ea['middle_dip'] = joint_euler(10, 11, 12, ref_idx=5)
    ea['ring_mcp'] = joint_euler(0, 13, 14, ref_idx=9)
    ea['ring_pip'] = joint_euler(13, 14, 15, ref_idx=9)
    ea['ring_dip'] = joint_euler(14, 15, 16, ref_idx=9)
    ea['pinky_mcp'] = joint_euler(0, 17, 18, ref_idx=13)
    ea['pinky_pip'] = joint_euler(17, 18, 19, ref_idx=13)
    ea['pinky_dip'] = joint_euler(18, 19, 20, ref_idx=13)
    return ea


def get_flexion_angles(points: np.ndarray) -> list:
    """6 flexion values from a (21,3) metric world-landmark array.

    [idx_mcp, idx_pip, idx_dip, thumb_spread, thumb_ip, thumb_ip*0.5], degrees.
    Mirrors get_flexion_angles() in ui/mediapipe_joint_angles.py.
    """
    pts = np.asarray(points, float)

    def angle_between(va, vb):
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(
            np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))))

    def bend(p, j, c):
        return angle_between(pts[j] - pts[p], pts[c] - pts[j])

    idx_mcp = bend(0, 5, 6)
    idx_pip = bend(5, 6, 7)
    idx_dip = bend(6, 7, 8)
    thumb_spread = angle_between(pts[5] - pts[0], pts[4] - pts[0])
    th_ip = bend(2, 3, 4)
    return [idx_mcp, idx_pip, idx_dip, thumb_spread, th_ip, th_ip * 0.5]


def get_wrist_orientation_euler(image_points: np.ndarray) -> np.ndarray:
    """Wrist orientation (Euler ZYX degrees) from the palm plane.

    Uses wrist(0), index-MCP(5), pinky-MCP(17) — matches get_wrist_pose() in
    ui/mediapipe_joint_angles.py (which builds the same palm frame).
    """
    pts = np.asarray(image_points, float)
    palm_R = compute_local_frame(pts[0], pts[5], pts[17])
    return R.from_matrix(palm_R).as_euler('ZYX', degrees=True)


def build_message(wrist_pos: np.ndarray,
                  wrist_euler: np.ndarray,
                  euler_angles: dict,
                  flexion: list,
                  world_lm: np.ndarray,
                  image_lm: np.ndarray) -> list:
    """Assemble the flat 183-float /hand/joint_angles payload.

    Args:
        wrist_pos:   (3,) wrist position.
        wrist_euler: (3,) wrist orientation Euler ZYX degrees.
        euler_angles: dict keyed by JOINT_ORDER -> (yaw, pitch, roll).
        flexion:     6 flexion values.
        world_lm:    (21,3) world landmarks (metres).
        image_lm:    (21,3) image landmarks (normalised).
    Returns:
        list[float] of length 183.
    """
    cfg: list[float] = []
    cfg.extend(np.asarray(wrist_pos, float).ravel().tolist())     # [0:3]
    cfg.extend(np.asarray(wrist_euler, float).ravel().tolist())   # [3:6]
    for joint in JOINT_ORDER:                                     # [6:51]
        yaw, pitch, roll = euler_angles[joint]
        cfg.extend([float(yaw), float(pitch), float(roll)])
    cfg.extend([float(v) for v in flexion])                       # [51:57]
    cfg.extend(np.asarray(world_lm, float).ravel().tolist())      # [57:120]
    cfg.extend(np.asarray(image_lm, float).ravel().tolist())      # [120:183]
    return cfg

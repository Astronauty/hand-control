"""Convert this repo's MediaPipe world landmarks into dex-retargeting's input frame.

dex-retargeting's optimizers expect human keypoints in the MANO convention (wrist-
centered, re-expressed through the hand's own estimated frame and an operator->MANO
change-of-basis). Its reference pipeline (example/vector_retargeting/
single_hand_detector.py) produces this as:

    kp = parse_keypoint_3d(mp_world_landmarks)     # (21,3)
    kp = kp - kp[0]                                 # wrist-center
    frame = estimate_frame_from_hand_points(kp)     # SVD hand frame
    joint_pos = kp @ frame @ operator2mano

Our teleop pipeline already carries the same 21 MediaPipe *world* landmarks in every
/hand/joint_angles message (raw[57:120]) — the exact analogue of parse_keypoint_3d's
output — so we port the two transform functions verbatim and reuse dex-retargeting's own
OPERATOR2MANO constant (imported, not copied) to guarantee an identical basis. The result
is the `joint_pos` array the retargeter's ref_value is indexed from.
"""
from __future__ import annotations

import numpy as np


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Wrist coordinate frame (MANO convention) from wrist-centered (21,3) landmarks.

    Ported verbatim from dex-retargeting's SingleHandDetector so the basis matches the
    library exactly. Uses landmarks [wrist(0), index_mcp(5), middle_mcp(9)].
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Vector from middle-MCP to wrist (palm long axis)
    x_vector = points[0] - points[2]

    # Palm-plane normal via SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    _u, _s, v = np.linalg.svd(points)
    normal = v[2, :]

    # Gram-Schmidt orthonormalize x against the normal, then z = x x normal
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # Fix handedness: pinky->index should align with +z in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal = normal * -1
        z = z * -1
    return np.stack([x, normal, z], axis=1)


def world_landmarks_to_mano(world_lm: np.ndarray, hand_type: str = "right") -> np.ndarray:
    """(21,3) MediaPipe world landmarks -> (21,3) MANO-convention keypoints.

    hand_type: "right" or "left" (selects OPERATOR2MANO_{RIGHT,LEFT}).
    """
    from dex_retargeting.constants import OPERATOR2MANO, HandType

    kp = np.asarray(world_lm, dtype=float).reshape(21, 3)
    kp = kp - kp[0:1, :]                         # wrist-center
    frame = estimate_frame_from_hand_points(kp)
    ht = HandType.right if str(hand_type).lower().startswith("r") else HandType.left
    return kp @ frame @ OPERATOR2MANO[ht]


def build_ref_value(joint_pos: np.ndarray, optimizer) -> np.ndarray:
    """Assemble the retargeter's ref_value from MANO keypoints + the optimizer's indices.

    POSITION optimizers consume absolute keypoints at the target human indices; VECTOR /
    DexPilot optimizers consume task-minus-origin vectors. `target_link_human_indices` is
    (2, N) for the vector/dexpilot case and (N,) for position — matching dex-retargeting's
    own example logic.
    """
    indices = optimizer.target_link_human_indices
    indices = np.asarray(indices)
    if indices.ndim == 2:                        # vector / dexpilot: task - origin
        origin, task = indices[0, :], indices[1, :]
        return joint_pos[task, :] - joint_pos[origin, :]
    return joint_pos[indices, :]                 # position: absolute

"""Build the VWJ optimization targets from human MANO hand keypoints.

Clean-room reimplementation of the reference-value construction in the method of
arXiv:2506.09384 (see vwj/__init__.py). Given the human hand's 21 MANO keypoints
(in the robot world frame) it produces, per frame:

  * ref_link_vec  (M, 3): the target vector for each of the M robot link-pairs, and
  * weights       (M,):   the per-pair cost weight, including the CONTINUOUS
                          (sigmoid) pinch weighting that is this paper's headline
                          change over DexPilot's discrete pinch switch.

The M = 3 * n_fingertip link-pairs, in this fixed order (matching the robot
link-pair list the optimizer is built with):
  [0]                      world -> thumb_tip        (global thumb position)
  [1 : 1+F]                wrist -> fingertip_i      (overall hand shape)
  [1+F : 1+2F-1]           thumb -> primary_i        (pinch; F-1 pairs, no thumb)
  [1+2F-1 : 1+3F-1]        dip_i -> fingertip_i      (fingertip orientation)
where F = n_fingertip (=4 for LEAP: thumb, index, middle, ring), and "primary" =
the non-thumb fingertips.

MANO keypoint indices (standard 21-point layout): wrist=0; tips at 4/8/12/16/20;
the DIP (direction base) joints at 3/7/11/15/19.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray, c: float = 0.0, w: float = 1.0) -> np.ndarray:
    """Logistic 1/(1+exp(w (x - c))) — the upstream (arXiv:2506.09384) convention.
    NOTE the sign: with w>0 this DECREASES as x grows (so the pinch term, called with
    the positive slope, is HIGH when the thumb-finger gap x is small and low when far);
    with w<0 it increases with x (used to fade the wrist-shape term while pinching)."""
    return 1.0 / (1.0 + np.exp(w * (np.asarray(x, float) - c)))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-9)


# Standard MANO 21-keypoint indices.
WRIST_INDEX = 0
# (tip, dip) per finger, thumb first then index/middle/ring (LEAP's primaries).
FINGERTIP_INDICES = np.array([4, 8, 12, 16], int)        # thumb, index, middle, ring
FINGERTIP_BASE_INDICES = np.array([3, 7, 11, 15], int)   # each finger's DIP joint


def build_ref_values(hand_kps_world: np.ndarray, weights: dict,
                     pinch_transition: float = 0.1, pinch_contact: float = 0.01,
                     pinch_slope: float = 10.0):
    """Assemble (ref_link_vec, ref_weights) from 21 MANO keypoints in world frame.

    hand_kps_world : (21, 3) human keypoints, already in the robot world frame
                     (i.e. scaled + placed; scaling is applied by the caller).
    weights        : dict with keys world_thumb, wrist_fingertip, thumb_primary,
                     fingertip_orientation (the per-group base weights).
    pinch_*        : continuous-pinch parameters (transition/contact thresholds, slope).

    Returns (ref_link_vec (M,3), ref_weights (M,)).
    """
    kps = np.asarray(hand_kps_world, float).reshape(21, 3)
    F = len(FINGERTIP_INDICES)
    M = 3 * F

    wrist = kps[WRIST_INDEX]
    tips = kps[FINGERTIP_INDICES]           # (F,3): thumb, index, middle, ring
    dips = kps[FINGERTIP_BASE_INDICES]      # (F,3)
    thumb_tip = tips[0]
    primary_tips = tips[1:]                  # (F-1, 3)

    # thumb<->primary distances drive the pinch sigmoids.
    thumb_primary_dist = np.linalg.norm(primary_tips - thumb_tip[None, :], axis=1)   # (F-1,)
    # pinch term weight RISES as the gap closes (w = +slope).
    w_thumb_primary = sigmoid(thumb_primary_dist, c=pinch_transition, w=pinch_slope)
    # wrist->fingertip (shape) weight FALLS while pinching (w = -slope), so during a
    # pinch the objective favours the pinch geometry over holding the global shape.
    # First entry is the thumb (keyed on the closest primary gap), then the primaries.
    w_wrist_fingertip = sigmoid(
        np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist]),
        c=pinch_transition, w=-pinch_slope)                                          # (F,)

    ref = np.zeros((M, 3))
    wt = np.zeros(M)

    # segment layout
    wf0, wf1 = 1, 1 + F                      # wrist->fingertip
    tp0, tp1 = wf1, wf1 + (F - 1)            # thumb->primary
    or0, or1 = tp1, tp1 + F                  # dip->tip orientation

    # [0] world -> thumb_tip : absolute thumb position.
    ref[0] = thumb_tip
    wt[0] = weights["world_thumb"]

    # [1:1+F] wrist -> fingertip_i.
    ref[wf0:wf1] = tips - wrist[None, :]
    wt[wf0:wf1] = weights["wrist_fingertip"] * w_wrist_fingertip

    # [tp0:tp1] thumb -> primary_i, with CONTINUOUS distance rescaling: the target
    # magnitude linearly maps [contact, transition] -> [0, transition], clamped
    # below contact (target 0 = touching) and passed through above transition. This
    # is the paper's smooth replacement for DexPilot's discrete distance clamp.
    rel = primary_tips - thumb_tip[None, :]
    rel_dist = np.linalg.norm(rel, axis=1)
    denom = max(pinch_transition - pinch_contact, 1e-9)
    scale = pinch_transition / denom
    rescaled = scale * (rel_dist - pinch_contact)
    rescaled[rel_dist < pinch_contact] = 0.0
    rescaled[rel_dist > pinch_transition] = rel_dist[rel_dist > pinch_transition]
    ref[tp0:tp1] = _normalize(rel) * rescaled[:, None]
    wt[tp0:tp1] = weights["thumb_primary"] * w_thumb_primary

    # [or0:or1] dip_i -> tip_i : fingertip orientation (distal-phalanx direction).
    ref[or0:or1] = tips - dips
    wt[or0:or1] = weights["fingertip_orientation"]

    return ref, wt


# The robot link-pair list the optimizer must be built with, in the SAME order as
# build_ref_values lays out ref_link_vec. Frame names are the MuJoCo sites.
def robot_link_pairs(tip_sites: list[str], dip_sites: list[str], wrist_site: str):
    """(origin, task) site-name pairs matching build_ref_values' vector order.
    tip_sites / dip_sites are [thumb, index, middle, ring] in that order."""
    thumb_tip = tip_sites[0]
    primaries = tip_sites[1:]
    pairs = [("world", thumb_tip)]                                  # [0]
    pairs += [(wrist_site, t) for t in tip_sites]                   # wrist->fingertip
    pairs += [(thumb_tip, t) for t in primaries]                   # thumb->primary
    pairs += [(d, t) for d, t in zip(dip_sites, tip_sites)]        # dip->tip
    return pairs

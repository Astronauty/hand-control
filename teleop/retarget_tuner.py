"""Shared plumbing for live-tuning DexPilot finger-retargeting constants.

The tuning UI lives as OpenCV trackbars ON the MediaPipe "Hand Tracking" window
(option 2: one window for camera + hand-tuning sliders), inside the MediaPipe
subprocess. That process does NOT import mujoco/the retargeter, so the slider
side is deliberately dependency-light: it only needs cv2 + the ranges below, and
it PUBLISHES the seven constants over ROS (/hand/retarget_params). The sim process
subscribes and writes them onto the live DexPilotRetargeter each frame (see
DexPilotController.apply_retarget_params), and owns the save-to-JSON.

Message /hand/retarget_params (Float32MultiArray), 8 floats:
    [BETA, GAMMA, EPS, ETA1, ETA2, S1_GAIN, S2_GAIN, save_flag]
The first seven are in PARAM_ORDER (== DexPilotRetargeter.TUNABLE). save_flag is a
momentary 1.0 for a single frame when the operator flips the SAVE trackbar; the
sim consumes it, persists, and the slider resets it to 0.

This module also keeps the original standalone RetargetTuner (its own OpenCV
window, sim-side) for headless/no-camera tuning, but the wired-in path is now
MediaPipeRetargetSliders + the ROS topic.
"""
from __future__ import annotations

# Canonical tunable order. MUST equal DexPilotRetargeter.TUNABLE; asserted where
# the retargeter is importable (sim side) so the two can't silently drift.
PARAM_ORDER = ('BETA', 'GAMMA', 'EPS', 'ETA1', 'ETA2', 'S1_GAIN', 'S2_GAIN')

# ROS topic carrying the 7 params + trailing save flag (8 floats).
RETARGET_PARAM_TOPIC = '/hand/retarget_params'

# Per-parameter slider range [lo, hi]. Chosen to bracket useful tuning values:
#   BETA    ~1.6 nominal; 0.5-3.0 spans "fingers barely open" -> "over-spread".
#   GAMMA   2.5e-3 nominal; regularisation, 0 (aggressive curl) .. 5e-2 (lazy).
#   EPS     pinch threshold [m]; 0.005 used in main, up to 5 cm (paper's 3 cm).
#   ETA1    S1 near-contact target [m]; ~0 .. 2 cm.
#   ETA2    S2 min inter-primary separation [m]; 0 .. 6 cm.
#   S1/S2   pinch cost gains; 0 .. 800 (paper: 200 / 400).
RANGES: dict[str, tuple[float, float]] = {
    'BETA':    (0.5,   3.0),
    'GAMMA':   (0.0,   5.0e-2),
    'EPS':     (0.0,   5.0e-2),
    'ETA1':    (0.0,   2.0e-2),
    'ETA2':    (0.0,   6.0e-2),
    'S1_GAIN': (0.0,   800.0),
    'S2_GAIN': (0.0,   800.0),
}
STEPS = 1000            # trackbar resolution (int positions 0..STEPS per param)
_SAVE_TRACK = "SAVE (0->1)"


def value_to_pos(v: float, lo: float, hi: float) -> int:
    """Map a constant value to an integer trackbar position in [0, STEPS]."""
    if hi <= lo:
        return 0
    frac = (float(v) - lo) / (hi - lo)
    return int(round(min(1.0, max(0.0, frac)) * STEPS))


def pos_to_value(pos: int, lo: float, hi: float) -> float:
    """Inverse of value_to_pos."""
    return lo + (hi - lo) * (pos / STEPS)


# Default seed values for the sliders when the subprocess has no better source.
# The sim's loaded retarget_config.json (or code defaults) is the real source of
# truth, but the MediaPipe process can't read it cheaply, so it starts sliders at
# these nominal values; the operator adjusts from there. Kept in sync with the
# DexPilotRetargeter class defaults by eye (they're documentation constants).
SEED_DEFAULTS: dict[str, float] = {
    'BETA': 1.6, 'GAMMA': 2.5e-3, 'EPS': 0.005,
    'ETA1': 1e-4, 'ETA2': 3e-2, 'S1_GAIN': 200.0, 'S2_GAIN': 400.0,
}


class MediaPipeRetargetSliders:
    """Trackbars attached to an EXISTING OpenCV window, publishing over ROS.

    Lives in the MediaPipe subprocess. Attaches one trackbar per tunable constant
    (plus a momentary SAVE track) to the given window name, then publish() reads
    them each frame and pushes an 8-float message on RETARGET_PARAM_TOPIC. No
    mujoco/retargeter import — the sim side applies the values.

    Args:
        window:  name of the already-created cv2 window (the camera feed window).
        node:    an rclpy Node to create the publisher on.
        seeds:   optional {name: value} initial slider positions (else SEED_DEFAULTS).
    """

    def __init__(self, window: str, node, seeds: dict[str, float] | None = None):
        import cv2
        from std_msgs.msg import Float32MultiArray
        self._cv2 = cv2
        self._Msg = Float32MultiArray
        self._win = window
        self._names = list(PARAM_ORDER)
        seeds = seeds or SEED_DEFAULTS
        for name in self._names:
            lo, hi = RANGES[name]
            pos = value_to_pos(seeds.get(name, SEED_DEFAULTS[name]), lo, hi)
            # No-op callback; publish() polls positions each frame.
            cv2.createTrackbar(name, self._win, pos, STEPS, lambda _v: None)
        cv2.createTrackbar(_SAVE_TRACK, self._win, 0, 1, lambda _v: None)
        self._pub = node.create_publisher(self._Msg, RETARGET_PARAM_TOPIC, 10)
        print(f"[retarget-sliders] attached to '{self._win}' — publishing "
              f"{RETARGET_PARAM_TOPIC} ({', '.join(self._names)}). Flip SAVE to persist.")

    def publish(self) -> None:
        """Read every trackbar and publish [7 params, save_flag]. Cheap; call once
        per camera frame (the MediaPipe loop already pumps cv2.waitKey)."""
        cv2 = self._cv2
        vals = []
        for name in self._names:
            lo, hi = RANGES[name]
            pos = cv2.getTrackbarPos(name, self._win)
            vals.append(pos_to_value(pos, lo, hi))
        save = 1.0 if cv2.getTrackbarPos(_SAVE_TRACK, self._win) else 0.0
        if save:
            cv2.setTrackbarPos(_SAVE_TRACK, self._win, 0)   # momentary
        msg = self._Msg()
        msg.data = [float(v) for v in vals] + [save]
        self._pub.publish(msg)


# ---------------------------------------------------------------------------
# Legacy standalone window (sim-side own window). No longer wired into the app —
# kept for headless / no-camera tuning. Requires the retargeter import.
# ---------------------------------------------------------------------------

class RetargetTuner:
    """OpenCV trackbar panel in its OWN window, bound to a live retargeter.

    Superseded by MediaPipeRetargetSliders (which folds the sliders into the
    camera window). Retained for a camera-less tuning session.
    """

    def __init__(self, retargeter, window: str = "DexPilot retarget tuning"):
        import cv2
        import numpy as np
        from teleop.dexpilot_retargeter import DexPilotRetargeter
        self._cv2 = cv2
        self._np = np
        self._rt = retargeter
        self._win = window
        self._names = list(DexPilotRetargeter.TUNABLE)
        self._ok = True
        try:
            cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._win, 460, 40 * (len(self._names) + 1) + 180)
            for name in self._names:
                lo, hi = RANGES[name]
                pos = value_to_pos(getattr(self._rt, name), lo, hi)
                cv2.createTrackbar(name, self._win, pos, STEPS, lambda _v: None)
            cv2.createTrackbar(_SAVE_TRACK, self._win, 0, 1, lambda _v: None)
            self._render()
            cv2.waitKey(1)
            print(f"[retarget-tuner] slider window '{self._win}' open.")
        except cv2.error as e:
            self._ok = False
            print(f"[retarget-tuner] could not open slider window ({e}); "
                  "tuning disabled.")

    def _render(self) -> None:
        cv2, np = self._cv2, self._np
        h = 40 * (len(self._names) + 1) + 180
        img = np.full((h, 460, 3), 40, np.uint8)
        cv2.putText(img, "current retargeting constants", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        y = 60
        for name in self._names:
            txt = f"{name:8s} = {float(getattr(self._rt, name)):.5g}"
            cv2.putText(img, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 120), 1, cv2.LINE_AA)
            y += 30
        cv2.imshow(self._win, img)

    def apply(self) -> None:
        if not self._ok:
            return
        cv2 = self._cv2
        for name in self._names:
            lo, hi = RANGES[name]
            pos = cv2.getTrackbarPos(name, self._win)
            setattr(self._rt, name, pos_to_value(pos, lo, hi))
        if cv2.getTrackbarPos(_SAVE_TRACK, self._win):
            self._rt.save_config()
            cv2.setTrackbarPos(_SAVE_TRACK, self._win, 0)
        self._render()
        cv2.waitKey(1)

    def close(self) -> None:
        if self._ok:
            try:
                self._cv2.destroyWindow(self._win)
            except self._cv2.error:
                pass

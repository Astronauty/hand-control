"""Live slider window for tuning DexPilot finger-retargeting constants.

Opens a small OpenCV trackbar window alongside the MuJoCo viewer. Each frame the
teleop loop calls apply(), which reads the sliders and writes the seven tunable
constants (BETA, GAMMA, EPS, ETA1, ETA2, S1_GAIN, S2_GAIN) straight onto the live
DexPilotRetargeter — so moving a slider re-shapes the very next retarget() solve.

OpenCV trackbars are integer-only, so each constant is mapped to an int position
over a chosen [lo, hi] range at a fixed resolution (`STEPS`). The mapping is
linear per-param; ranges are picked wide enough to bracket useful values without
wasting resolution. The window also holds a "SAVE (toggle 0->1)" track: flip it and
apply() persists the current values to calibration/retarget_config.json, then the
track auto-resets so it's a momentary button.

Why OpenCV and not tkinter/dearpygui: the MediaPipe publisher already brings up an
OpenCV window in this process' GUI stack, cv2 is guaranteed present in the runtime
env, and trackbars need no event loop of their own — apply() polls them, so there's
nothing to fight the sim loop for the main thread.

Usage (in the teleop branch of kinova_leap_pick_place.py):

    from teleop.retarget_tuner import RetargetTuner
    tuner = RetargetTuner(_dexpilot_ctrl.retargeter)   # after ctrl construction
    ...
    while viewer.is_running():
        ...
        tuner.apply()      # once per frame, cheap
    tuner.close()
"""
from __future__ import annotations

import cv2
import numpy as np

from teleop.dexpilot_retargeter import DexPilotRetargeter

# Per-parameter slider range [lo, hi]. Chosen to bracket useful tuning values:
#   BETA    ~1.6 nominal; 0.5-3.0 spans "fingers barely open" -> "over-spread".
#   GAMMA   2.5e-3 nominal; regularisation, 0 (aggressive curl) .. 5e-2 (lazy).
#   EPS     pinch threshold [m]; 0.005 used in main, up to 5 cm (paper's 3 cm).
#   ETA1    S1 near-contact target [m]; ~0 .. 2 cm.
#   ETA2    S2 min inter-primary separation [m]; 0 .. 6 cm.
#   S1/S2   pinch cost gains; 0 .. 800 (paper: 200 / 400).
_RANGES: dict[str, tuple[float, float]] = {
    'BETA':    (0.5,   3.0),
    'GAMMA':   (0.0,   5.0e-2),
    'EPS':     (0.0,   5.0e-2),
    'ETA1':    (0.0,   2.0e-2),
    'ETA2':    (0.0,   6.0e-2),
    'S1_GAIN': (0.0,   800.0),
    'S2_GAIN': (0.0,   800.0),
}
_STEPS = 1000          # trackbar resolution (int positions 0.._STEPS per param)
_WIN   = "DexPilot retarget tuning"
_SAVE_TRACK = "SAVE (0->1)"


class RetargetTuner:
    """OpenCV trackbar panel bound to a live DexPilotRetargeter instance."""

    def __init__(self, retargeter: DexPilotRetargeter,
                 window: str = _WIN) -> None:
        self._rt   = retargeter
        self._win  = window
        self._names = list(DexPilotRetargeter.TUNABLE)
        self._ok = True
        try:
            cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._win, 460, 40 * (len(self._names) + 1) + 180)
            for name in self._names:
                lo, hi = _RANGES[name]
                pos = self._value_to_pos(getattr(self._rt, name), lo, hi)
                # A no-op onChange: apply() polls positions itself each frame, so
                # the callback stays empty (cv2 requires one on some builds).
                cv2.createTrackbar(name, self._win, pos, _STEPS, lambda _v: None)
            cv2.createTrackbar(_SAVE_TRACK, self._win, 0, 1, lambda _v: None)
            self._render()   # paint an initial body so the window isn't blank
            cv2.waitKey(1)
            print(f"[retarget-tuner] slider window '{self._win}' open — "
                  f"tuning {', '.join(self._names)}. Flip SAVE to persist.")
        except cv2.error as e:
            # Headless / no HighGUI backend: degrade gracefully rather than take
            # down teleop. The retargeter still runs on its loaded constants.
            self._ok = False
            print(f"[retarget-tuner] could not open slider window ({e}); "
                  "tuning disabled, retargeting uses loaded/default constants.")

    @staticmethod
    def _value_to_pos(v: float, lo: float, hi: float) -> int:
        if hi <= lo:
            return 0
        frac = (float(v) - lo) / (hi - lo)
        return int(round(min(1.0, max(0.0, frac)) * _STEPS))

    @staticmethod
    def _pos_to_value(pos: int, lo: float, hi: float) -> float:
        return lo + (hi - lo) * (pos / _STEPS)

    def _render(self) -> None:
        """Draw the current values into the window body. A trackbar-only OpenCV
        window renders BLACK until something is imshow'n into it AND the HighGUI
        event loop is pumped (waitKey) — apply() does both each frame."""
        h = 40 * (len(self._names) + 1) + 180
        img = np.full((h, 460, 3), 40, np.uint8)   # dark-grey canvas
        cv2.putText(img, "current retargeting constants", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        y = 60
        for name in self._names:
            val = float(getattr(self._rt, name))
            txt = f"{name:8s} = {val:.5g}"
            cv2.putText(img, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 120), 1, cv2.LINE_AA)
            y += 30
        cv2.putText(img, "flip SAVE 0->1 to persist -> retarget_config.json",
                    (12, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (160, 160, 160), 1, cv2.LINE_AA)
        cv2.imshow(self._win, img)

    def apply(self) -> None:
        """Read every slider and write its value onto the retargeter. Cheap; call
        once per frame. Also repaints the window body and pumps the HighGUI event
        loop (without which the window renders black), and services SAVE."""
        if not self._ok:
            return
        for name in self._names:
            lo, hi = _RANGES[name]
            pos = cv2.getTrackbarPos(name, self._win)
            setattr(self._rt, name, self._pos_to_value(pos, lo, hi))
        if cv2.getTrackbarPos(_SAVE_TRACK, self._win):
            self._rt.save_config()
            cv2.setTrackbarPos(_SAVE_TRACK, self._win, 0)   # momentary
        self._render()
        cv2.waitKey(1)          # process paint/UI events — REQUIRED to draw

    def close(self) -> None:
        if self._ok:
            try:
                cv2.destroyWindow(self._win)
            except cv2.error:
                pass

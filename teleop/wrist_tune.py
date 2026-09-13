#!/usr/bin/env python3
"""
wrist_tune.py — live tuner for the arm's wrist-tracking responsiveness.

The point is to SEE and MEASURE the effect: sliders write the three tunables to
teleop/calibration/wrist_track_config.json, which the running app hot-reloads every
frame (like the retarget tuner). The app logs the live tracking error (|reference
wrist - actual pinch site|) + the active params to wrist_track_error.jsonl, and this
GUI plots that error live — so you move a slider, then watch the error trace respond
under your own hand motion. Every dialed value and its error is in the log for
offline analysis afterward.

Run alongside the app (which must be launched with --tune-wrist):

    # terminal 1 — the sim, in a baseline teleop mode
    ./start_teleop.sh sim dexpilot --tune-wrist        # (press 8 to start tracking)
    # terminal 2 — this tuner
    python3 teleop/wrist_tune.py

Layout
------
    top     live tracking-error trace (mm), mean over the visible window annotated
    bottom  JOG_VEL, WRIST_TRACK_GAIN, JOG_QDOT_MAX sliders + Apply / Reset-defaults

Tuning guide
------------
    JOG_VEL           peak wrist speed cap (m/s). Raise first — biggest lever on the
                      lag for large/fast motions. 0.6 default; try 0.8-1.0.
    WRIST_TRACK_GAIN  P-gain on position error. Raise to reach the speed cap at a
                      smaller error (briskier medium corrections). 12 default; try
                      16-20. Too high => jitter/buzz when holding still on noisy input.
    JOG_QDOT_MAX      final per-joint arm-rate cap (rad/s). Leave at 2.0 unless a fast
                      motion visibly clips (arm can't hold the wrist direction).

Watch the error trace: lower mean error under the same hand motion = better. If the
trace gets spiky/oscillatory when you raise a value, you've gone too far — back off.
"""
import argparse
import json
import os
import sys
from collections import deque

# Import the shared config/log contract so field names never drift from the app.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))
try:
    from teleop.wrist_track_tune import (CONFIG_PATH, LOG_DEFAULT, DEFAULTS,
                                         _env_seeded_defaults)
except ImportError:
    from wrist_track_tune import (CONFIG_PATH, LOG_DEFAULT, DEFAULTS,
                                  _env_seeded_defaults)

# Slider ranges: (min, max) per tunable. Chosen to bracket the useful tuning band.
RANGES = {
    "JOG_VEL":          (0.1, 1.5),
    "WRIST_TRACK_GAIN": (2.0, 30.0),
    "JOG_QDOT_MAX":     (1.0, 6.0),
    "TRACK_ACCEL":      (1.0, 30.0),   # m/s^2 accel-slew cap (higher = snappier, more overshoot)
    "TRACK_DAMP":       (0.0, 15.0),   # velocity damping (higher = less overshoot, slower)
}


def _read_config(path):
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        cfg = {}
    out = dict(_env_seeded_defaults())
    for k in DEFAULTS:
        if k in cfg:
            try:
                out[k] = float(cfg[k])
            except (TypeError, ValueError):
                pass
    return out


def _write_config(path, vals):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: float(vals[k]) for k in DEFAULTS}, f, indent=2)


def _tail_errors(log_path, seen_offset):
    """Read new JSONL lines since seen_offset. Returns (rows, new_offset)."""
    rows = []
    try:
        with open(log_path) as f:
            f.seek(seen_offset)
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            seen_offset = f.tell()
    except OSError:
        pass
    return rows, seen_offset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG_PATH, help="wrist_track_config.json to write")
    ap.add_argument("--log", default=LOG_DEFAULT, help="wrist_track_error.jsonl to read")
    ap.add_argument("--window", type=int, default=600,
                    help="error-trace points shown (rolling window)")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from matplotlib.animation import FuncAnimation

    vals = _read_config(args.config)

    fig = plt.figure(figsize=(11, 7))
    fig.suptitle("Wrist-tracking tuner — move a slider, watch the error trace "
                 "(app must run with --tune-wrist)", fontsize=11)
    ax_err = fig.add_axes([0.09, 0.42, 0.86, 0.48])
    ax_err.set_ylabel("tracking error (mm)")
    ax_err.set_xlabel("recent control steps")
    ax_err.grid(True, alpha=0.3)
    (line_err,) = ax_err.plot([], [], color="#e4572e", lw=1.3, label="|ref - actual| (mm)")
    mean_txt = ax_err.text(0.02, 0.92, "", transform=ax_err.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round", fc="#fff3", ec="#e4572e"))
    ax_err.legend(loc="upper right", fontsize=8)

    # One slider per tunable in DEFAULTS order (labels annotate the two that trade
    # speed vs overshoot). Stacked from the bottom up.
    _LABELS = {
        "JOG_VEL":          "JOG_VEL (m/s) — peak speed",
        "WRIST_TRACK_GAIN": "WRIST_TRACK_GAIN — P-gain",
        "JOG_QDOT_MAX":     "JOG_QDOT_MAX (rad/s)",
        "TRACK_ACCEL":      "TRACK_ACCEL (m/s²) — snappier↑",
        "TRACK_DAMP":       "TRACK_DAMP — less overshoot↑",
    }
    keys = list(DEFAULTS.keys())
    sliders = {}
    y0, dy = 0.05, 0.055
    for i, k in enumerate(reversed(keys)):
        sliders[k] = Slider(fig.add_axes([0.16, y0 + i * dy, 0.55, 0.028]),
                            _LABELS.get(k, k), *RANGES[k], valinit=vals[k])
    b_apply = Button(fig.add_axes([0.79, 0.20, 0.17, 0.05]), "Apply (write config)")
    b_reset = Button(fig.add_axes([0.79, 0.13, 0.17, 0.05]), "Reset to defaults")
    status = fig.text(0.16, 0.005, "", fontsize=8.5, color="#2e7d32")

    def _apply(_event=None):
        v = {k: sliders[k].val for k in keys}
        _write_config(args.config, v)
        status.set_text("applied -> " + os.path.relpath(args.config) + ":  "
                        + "  ".join(f"{k}={v[k]:.2f}" for k in keys)
                        + "   (app reloads next frame)")
        fig.canvas.draw_idle()

    def _reset(_event=None):
        for k in keys:
            sliders[k].set_val(DEFAULTS[k])
        _apply()

    b_apply.on_clicked(_apply)
    b_reset.on_clicked(_reset)
    # Write on slider release (mouse-up) for a hands-free feel; dragging doesn't spam.
    _slider_axes = {sliders[k].ax for k in keys}
    def _on_release(_evt):
        if _evt.inaxes in _slider_axes:
            _apply()
    fig.canvas.mpl_connect("button_release_event", _on_release)

    errs = deque(maxlen=args.window)
    # Start reading at the END of any pre-existing log so a stale file's offset can't
    # desync us; new rows the app appends after we launch are what we plot.
    try:
        state = {"offset": os.path.getsize(args.log)}
    except OSError:
        state = {"offset": 0}
    state["total"] = 0

    import numpy as _np

    def update(_frame):
        rows, state["offset"] = _tail_errors(args.log, state["offset"])
        for r in rows:
            if "err_mm" in r:
                errs.append(r["err_mm"]); state["total"] += 1
        if errs:
            ys = list(errs)
            line_err.set_data(range(len(ys)), ys)
            ax_err.set_xlim(0, max(len(ys), 10))
            ax_err.set_ylim(0, max(5.0, 1.15 * max(ys)))
            mean_txt.set_text(f"mean {(_np.mean(ys)):.1f} mm   "
                              f"p95 {(_np.percentile(ys, 95)):.1f} mm   "
                              f"n={len(ys)} (total {state['total']})")
        else:
            # Nothing logged yet — tell the user WHY the plot is blank instead of
            # leaving it silently empty (the usual cause: app not launched with
            # --tune-wrist, or press-8 not done so tracking isn't producing a target).
            _exists = os.path.exists(args.log)
            mean_txt.set_text(
                ("waiting for data… log file not created yet.\n"
                 "Is the sim running with --tune-wrist, and have you pressed 8 to track?"
                 if not _exists else
                 "log file exists but no rows yet — press 8 and move your hand."))
        fig.canvas.draw_idle()
        return line_err, mean_txt

    # Keep a reference on the figure so the animation isn't garbage-collected (a GC'd
    # FuncAnimation silently stops updating — the classic 'plot never moves' bug).
    fig._wrist_anim = FuncAnimation(fig, update, interval=200, cache_frame_data=False,
                                    save_count=1)
    print(f"[wrist-tune] writing {args.config}")
    print(f"[wrist-tune] reading {args.log}")
    print("[wrist-tune] launch the app with --tune-wrist, press 8 to track, then move the "
          "reference wrist and tune. Slider release (or Apply) writes the config.")
    print("[wrist-tune] if the plot stays blank, check that the sim is running WITH "
          "--tune-wrist and that you pressed 8 — the plot annotates the reason.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

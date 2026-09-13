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

    s_jv = Slider(fig.add_axes([0.13, 0.26, 0.6, 0.03]), "JOG_VEL (m/s)",
                  *RANGES["JOG_VEL"], valinit=vals["JOG_VEL"])
    s_gn = Slider(fig.add_axes([0.13, 0.19, 0.6, 0.03]), "WRIST_TRACK_GAIN",
                  *RANGES["WRIST_TRACK_GAIN"], valinit=vals["WRIST_TRACK_GAIN"])
    s_qd = Slider(fig.add_axes([0.13, 0.12, 0.6, 0.03]), "JOG_QDOT_MAX (rad/s)",
                  *RANGES["JOG_QDOT_MAX"], valinit=vals["JOG_QDOT_MAX"])
    b_apply = Button(fig.add_axes([0.79, 0.22, 0.16, 0.05]), "Apply (write config)")
    b_reset = Button(fig.add_axes([0.79, 0.14, 0.16, 0.05]), "Reset to defaults")
    status = fig.text(0.13, 0.06, "", fontsize=9, color="#2e7d32")

    def _apply(_event=None):
        v = {"JOG_VEL": s_jv.val, "WRIST_TRACK_GAIN": s_gn.val, "JOG_QDOT_MAX": s_qd.val}
        _write_config(args.config, v)
        status.set_text(f"applied -> {os.path.relpath(args.config)}: "
                        f"JOG_VEL={v['JOG_VEL']:.2f}  GAIN={v['WRIST_TRACK_GAIN']:.1f}  "
                        f"QDOT_MAX={v['JOG_QDOT_MAX']:.2f}  (app reloads on next frame)")
        fig.canvas.draw_idle()

    def _reset(_event=None):
        s_jv.set_val(DEFAULTS["JOG_VEL"])
        s_gn.set_val(DEFAULTS["WRIST_TRACK_GAIN"])
        s_qd.set_val(DEFAULTS["JOG_QDOT_MAX"])
        _apply()

    # Apply on release so dragging doesn't spam the file (write on mouse-up).
    for s in (s_jv, s_gn, s_qd):
        s.on_changed(lambda _v: None)  # live label only; write on Apply / release
    b_apply.on_clicked(_apply)
    b_reset.on_clicked(_reset)
    # Also write on slider release for a hands-free feel.
    def _on_release(_evt):
        if _evt.inaxes in (s_jv.ax, s_gn.ax, s_qd.ax):
            _apply()
    fig.canvas.mpl_connect("button_release_event", _on_release)

    errs = deque(maxlen=args.window)
    state = {"offset": 0}

    def update(_frame):
        rows, state["offset"] = _tail_errors(args.log, state["offset"])
        for r in rows:
            if "err_mm" in r:
                errs.append(r["err_mm"])
        if errs:
            ys = list(errs)
            line_err.set_data(range(len(ys)), ys)
            ax_err.set_xlim(0, max(len(ys), 10))
            ax_err.set_ylim(0, max(5.0, 1.15 * max(ys)))
            import numpy as _np
            mean_txt.set_text(f"mean {(_np.mean(ys)):.1f} mm   "
                              f"p95 {(_np.percentile(ys, 95)):.1f} mm   "
                              f"n={len(ys)}")
        return line_err, mean_txt

    anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    print(f"[wrist-tune] writing {args.config}")
    print(f"[wrist-tune] reading {args.log}")
    print("[wrist-tune] launch the app with --tune-wrist, press 8 to track, then move the "
          "reference wrist and tune. Slider release (or Apply) writes the config.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

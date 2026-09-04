#!/usr/bin/env python3
"""
hand_tune.py — live 3D hand skeleton with the pinch thresholds beside it.

The point is to SEE the effect: the skeleton is drawn live, and each finger
lights up the moment the debounced detector says it is pinched. Move the EPS
slider and fingers change colour under your hand in real time, so you can pinch
deliberately and watch exactly where the threshold catches and releases.

    python3 ui/hand_tune.py                 # live from /hand/joint_angles
    python3 ui/hand_tune.py --file hq_*.npz # scrub a recording instead

Layout
------
    left    live 3D skeleton; a pinched finger turns solid, its bone thickens
    right   thumb-to-finger distance traces with the enter/exit lines drawn
    bottom  EPS, enter/exit fractions, median window, and Apply

The enter/exit pair is hysteresis: enter below EPS*enter_frac, leave above
EPS*exit_frac. The gap between them stops a distance hovering near the
threshold from chattering the grasp on and off, which is why raising EPS alone
is rarely the right fix for a flickering finger.
"""

import argparse
import json
import os
import sys
from collections import deque

import numpy as np

CONFIG = "calibration/retarget_config.json"
THUMB_TIP = 4
FINGERS = [("index", 8), ("middle", 12), ("ring", 16)]

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
# Which finger each bone belongs to, so a pinched finger can be highlighted.
BONE_FINGER = ([None] * 4 + ["index"] * 4 + ["middle"] * 4
               + ["ring"] * 4 + [None] * 4 + [None] * 3)
COLORS = {"index": "#f3a712", "middle": "#4ea699", "ring": "#2e86ab",
          None: "#8d99ae"}
PINCH_COLOR = "#e4572e"


def load_config(path=CONFIG):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"EPS": 0.03, "PINCH_ENTER_FRAC": 1.0, "PINCH_EXIT_FRAC": 1.6,
                "PINCH_ENTER_N": 2.0, "PINCH_EXIT_N": 3.0, "PINCH_MEDIAN_N": 5.0}


class PinchState:
    """
    The retargeter's debounced detector, run incrementally for live use.

    Deliberately stateful rather than recomputed per frame: the whole point of
    the enter/exit counters is that the decision depends on history, so a
    stateless "is it below the line right now" display would show something the
    robot never sees.
    """

    def __init__(self, median_n=5):
        self.med = deque(maxlen=max(int(median_n), 1))
        self.on = False
        self.below = 0
        self.above = 0
        self.transitions = 0

    def update(self, d, eps, enter_frac, exit_frac, enter_n, exit_n):
        self.med.append(d)
        m = float(np.median(self.med))
        if m < eps * enter_frac:
            self.below += 1
            self.above = 0
        elif m > eps * exit_frac:
            self.above += 1
            self.below = 0
        else:
            self.below = self.above = 0        # dead band: hold
        if not self.on and self.below >= enter_n:
            self.on = True
            self.transitions += 1
        elif self.on and self.above >= exit_n:
            self.on = False
            self.transitions += 1
        return self.on, m


class LiveSource:
    def __init__(self, topic):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray
        rclpy.init()
        self._rclpy = rclpy
        self.node = Node("hand_tune")
        self.latest = None
        self.count = 0
        self.node.create_subscription(Float32MultiArray, topic, self._cb, 10)
        print(f"subscribed to {topic}")

    def _cb(self, msg):
        raw = np.asarray(msg.data, dtype=float)
        if len(raw) >= 120:
            self.latest = raw[57:120].reshape(21, 3)
            self.count += 1

    def next(self):
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        return self.latest

    def close(self):
        self.node.destroy_node()
        self._rclpy.shutdown()


class FileSource:
    def __init__(self, path):
        d = np.load(path)
        self.lm = d["raw"][:, 57:120].reshape(-1, 21, 3)
        self.i = 0
        self.count = 0
        print(f"{path}: {len(self.lm)} frames (looping)")

    def next(self):
        f = self.lm[self.i % len(self.lm)]
        self.i += 1
        self.count += 1
        return f

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="hq_*.npz from hand_quality.py")
    ap.add_argument("--topic", default="/hand/joint_angles")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--span", type=float, default=0.12,
                    help="half-width of the 3D view box, metres")
    ap.add_argument("--history", type=int, default=400)
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from matplotlib.animation import FuncAnimation

    cfg = load_config(args.config)
    src = FileSource(args.file) if args.file else LiveSource(args.topic)

    fig = plt.figure(figsize=(13, 7.5))
    fig.canvas.manager.set_window_title("hand pose + pinch tuning")
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.set_position([0.02, 0.30, 0.46, 0.66])
    axd = fig.add_axes([0.56, 0.30, 0.41, 0.66])

    s_eps = Slider(fig.add_axes([0.10, 0.20, 0.55, 0.03]),
                   "EPS (mm)", 5.0, 100.0, valinit=cfg["EPS"] * 1e3)
    s_ent = Slider(fig.add_axes([0.10, 0.155, 0.55, 0.03]),
                   "enter x", 0.5, 1.5, valinit=cfg.get("PINCH_ENTER_FRAC", 1.0))
    s_ext = Slider(fig.add_axes([0.10, 0.11, 0.55, 0.03]),
                   "exit x", 1.0, 3.0, valinit=cfg.get("PINCH_EXIT_FRAC", 1.6))
    s_med = Slider(fig.add_axes([0.10, 0.065, 0.55, 0.03]),
                   "median N", 1, 15, valinit=cfg.get("PINCH_MEDIAN_N", 5),
                   valstep=1)
    b_apply = Button(fig.add_axes([0.74, 0.15, 0.2, 0.05]), "Apply to config")
    b_reset = Button(fig.add_axes([0.74, 0.08, 0.2, 0.05]), "Reset counters")

    status = fig.text(0.02, 0.015, "waiting for hand data...", fontsize=10,
                      family="monospace")

    bone_lines = [ax3.plot([], [], [], lw=2.5,
                           color=COLORS[BONE_FINGER[k]])[0]
                  for k in range(len(BONES))]
    pts, = ax3.plot([], [], [], "o", ms=4, color="#22223b")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.set_title("live hand — a finger turns red when the detector says pinched")

    hist = {f[0]: deque(maxlen=args.history) for f in FINGERS}
    detect = {f[0]: PinchState(int(s_med.val)) for f in FINGERS}

    def reset_counters(_=None):
        for f in FINGERS:
            detect[f[0]] = PinchState(int(s_med.val))
    s_med.on_changed(lambda _: reset_counters())
    b_reset.on_clicked(reset_counters)

    def update(_frame):
        lm = src.next()
        if lm is None:
            return
        eps = s_eps.val / 1e3
        ent, ext = s_ent.val, s_ext.val
        en_n = cfg.get("PINCH_ENTER_N", 2)
        ex_n = cfg.get("PINCH_EXIT_N", 3)

        active = {}
        parts = []
        for name, tip in FINGERS:
            d = float(np.linalg.norm(lm[THUMB_TIP] - lm[tip]))
            hist[name].append(d)
            on, med = detect[name].update(d, eps, ent, ext, en_n, ex_n)
            active[name] = on
            parts.append(f"{name:<6} {d*1e3:5.1f}mm "
                         f"{'PINCH' if on else '  .  '} "
                         f"({detect[name].transitions} trans)")

        # --- 3D skeleton, coloured by live pinch state ----------------------
        for k, (a, b) in enumerate(BONES):
            ln = bone_lines[k]
            ln.set_data([lm[a, 0], lm[b, 0]], [lm[a, 1], lm[b, 1]])
            ln.set_3d_properties([lm[a, 2], lm[b, 2]])
            f = BONE_FINGER[k]
            if f is not None and active.get(f):
                ln.set_color(PINCH_COLOR)
                ln.set_linewidth(4.5)
            else:
                ln.set_color(COLORS[f])
                ln.set_linewidth(2.5)
        pts.set_data(lm[:, 0], lm[:, 1])
        pts.set_3d_properties(lm[:, 2])

        c = lm.mean(0)
        ax3.set_xlim(c[0] - args.span, c[0] + args.span)
        ax3.set_ylim(c[1] - args.span, c[1] + args.span)
        ax3.set_zlim(c[2] - args.span, c[2] + args.span)

        # --- distance traces with the two thresholds ------------------------
        axd.clear()
        for name, _ in FINGERS:
            h = np.array(hist[name]) * 1e3
            axd.plot(h, color=PINCH_COLOR if active[name] else COLORS[name],
                     lw=1.8 if active[name] else 1.0, label=f"thumb-{name}")
        axd.axhline(eps * ent * 1e3, color="k", ls="--", lw=1.2)
        axd.axhline(eps * ext * 1e3, color="k", ls=":", lw=1.2)
        axd.text(0.01, eps * ent * 1e3, " enter", va="bottom", fontsize=8,
                 transform=axd.get_yaxis_transform())
        axd.text(0.01, eps * ext * 1e3, " exit", va="bottom", fontsize=8,
                 transform=axd.get_yaxis_transform())
        axd.set_ylim(0, 160)
        axd.set_ylabel("thumb-to-finger (mm)")
        axd.set_xlabel(f"last {args.history} frames")
        axd.legend(loc="upper right", fontsize=8)
        axd.grid(alpha=0.25)

        status.set_text(f"frames {src.count}   |   " + "   ".join(parts))

    def apply(_):
        cfg["EPS"] = s_eps.val / 1e3
        cfg["PINCH_ENTER_FRAC"] = float(s_ent.val)
        cfg["PINCH_EXIT_FRAC"] = float(s_ext.val)
        cfg["PINCH_MEDIAN_N"] = float(int(s_med.val))
        os.makedirs(os.path.dirname(args.config) or ".", exist_ok=True)
        with open(args.config, "w") as f:
            json.dump(cfg, f, indent=1)
        print(f"wrote {args.config}  EPS={cfg['EPS']*1e3:.0f}mm "
              f"enter={cfg['PINCH_ENTER_FRAC']:.2f} "
              f"exit={cfg['PINCH_EXIT_FRAC']:.2f} "
              f"median_n={cfg['PINCH_MEDIAN_N']:.0f}")
        print("  dexpilot hot-reloads this file — no restart needed")
    b_apply.on_clicked(apply)

    anim = FuncAnimation(fig, update, interval=33, cache_frame_data=False)
    fig._anim = anim
    try:
        plt.show()
    finally:
        src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

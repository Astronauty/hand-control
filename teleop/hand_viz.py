#!/usr/bin/env python3
"""
hand_viz.py — live 3D view of whatever is on /hand/joint_angles.

Subscribes only. Nothing else runs: no MuJoCo, no retargeting, no IK. So if the
skeleton here looks like a hand in a sensible orientation, the publisher and the
message layout are correct and any remaining problem is downstream.

Usage
-----
    python3 teleop/hand_viz.py                      # world landmarks (raw[57:120])
    python3 teleop/hand_viz.py --show image         # image landmarks (raw[120:183])
    python3 teleop/hand_viz.py --show both          # side by side
    python3 teleop/hand_viz.py --absolute           # add the wrist back (raw[0:3])

    python3 teleop/hand_viz.py --print              # text only, no window (headless/SSH)

Reading it
----------
Hold your hand flat, palm down, fingers pointing away from you.

  * Fingers should extend along +Y in the OpenXR convention (Y-up), or along +Z
    in the MuJoCo convention (Z-up). Which one you see tells you directly which
    --frame the publisher should use.
  * The thumb should sit to one side, not merged into the fingers. If the thumb
    lands among the fingers, the XR_TO_MP remap is wrong.
  * Segment lengths should look plausible: ~7-9 cm wrist to knuckle, ~2-4 cm per
    phalanx. Uniformly short proximal phalanges mean the OpenXR metacarpals were
    kept instead of dropped.
  * The palm-frame determinant printed in the corner must be +1. A reflected
    frame (-1) is silently "repaired" by SVD downstream into an arbitrary axis
    flip — the failure that makes rotating +Y show up as -Y in MuJoCo.
"""

import argparse
import sys

import numpy as np

# MediaPipe's 21-landmark connectivity, used for the skeleton lines.
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # knuckle ridge
]
FINGER_COLORS = {
    "thumb":  "#e4572e",
    "index":  "#f3a712",
    "middle": "#4ea699",
    "ring":   "#2e86ab",
    "pinky":  "#a23b72",
    "palm":   "#8d99ae",
}
BONE_COLOR = (
    ["thumb"] * 4 + ["index"] * 4 + ["middle"] * 4
    + ["ring"] * 4 + ["pinky"] * 4 + ["palm"] * 3
)


def palm_frame(lm):
    """
    Right-handed frame from wrist / index-MCP / pinky-MCP, the same construction
    the arm controller uses. Returns (R, det). det must be +1: a reflected frame
    gets silently repaired by SVD downstream into an arbitrary single-axis flip.
    """
    v1 = lm[5] - lm[0]
    v2 = lm[17] - lm[0]
    n = np.cross(v1, v2)
    if np.linalg.norm(n) < 1e-9 or np.linalg.norm(v1) < 1e-9:
        return np.eye(3), 1.0
    n = n / np.linalg.norm(n)
    x = v1 / np.linalg.norm(v1)
    y = np.cross(n, x)
    R = np.column_stack([x, y, n])
    return R, float(np.linalg.det(R))


def describe(lm, wrist):
    """One-line health summary — the numbers worth eyeballing every frame."""
    R, det = palm_frame(lm)
    span = float(np.linalg.norm(lm.max(0) - lm.min(0)))
    # Wrist to middle-MCP: a real hand is roughly 8-10 cm here. Much shorter
    # usually means the OpenXR metacarpals were kept in the remap.
    palm_len = float(np.linalg.norm(lm[9] - lm[0]))
    thumb_gap = float(np.linalg.norm(lm[4] - lm[8]))
    # Which axis the fingers point along tells you the frame convention.
    finger_dir = lm[12] - lm[0]
    axis = "XYZ"[int(np.argmax(np.abs(finger_dir)))]
    sign = "+" if finger_dir[np.argmax(np.abs(finger_dir))] > 0 else "-"
    return (f"wrist=({wrist[0]:+.2f},{wrist[1]:+.2f},{wrist[2]:+.2f})m  "
            f"span={span*100:.0f}cm  palm={palm_len*100:.1f}cm  "
            f"thumb-index={thumb_gap*100:.1f}cm  "
            f"fingers->{sign}{axis}  det={det:+.2f}")


# ---------------------------------------------------------------------------
# ROS subscriber
# ---------------------------------------------------------------------------

class HandSubscriber:
    """Latest message only. Never blocks; the viewer polls whenever it redraws."""

    def __init__(self, topic):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray

        rclpy.init()
        self._rclpy = rclpy
        self.node = Node("hand_viz")
        self.latest = None
        self.count = 0
        self.node.create_subscription(
            Float32MultiArray, topic, self._cb, 10)
        self.node.get_logger().info(f"subscribed to {topic}")

    def _cb(self, msg):
        self.latest = np.asarray(msg.data, dtype=float)
        self.count += 1

    def poll(self):
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        return self.latest

    def close(self):
        self.node.destroy_node()
        self._rclpy.shutdown()


def unpack(raw, absolute):
    """
    raw -> (world_lm, image_lm, wrist). Layout per dexpilot_controller.step():
        [0:3] wrist | [3:57] unused | [57:120] world (wrist-rel) | [120:183] image
    Returns (None, None, None) if the message is too short to be usable.
    """
    if raw is None or len(raw) < 120:
        return None, None, None
    wrist = raw[0:3]
    world = raw[57:120].reshape(21, 3)
    image = raw[120:183].reshape(21, 3) if len(raw) >= 183 else None
    if absolute:
        world = world + wrist
        if image is not None:
            image = image + wrist
    return world, image, wrist


# ---------------------------------------------------------------------------
# Text mode
# ---------------------------------------------------------------------------

def run_print(sub, absolute, hz):
    import time
    print("waiting for /hand/joint_angles ...  (ctrl-c to stop)")
    last = 0.0
    seen = False
    try:
        while True:
            raw = sub.poll()
            now = time.time()
            if raw is not None and now - last >= 1.0 / hz:
                last = now
                if not seen:
                    print(f"first message: {len(raw)} floats "
                          f"({'image block present' if len(raw) >= 183 else 'no image block'})")
                    seen = True
                world, _, wrist = unpack(raw, absolute)
                if world is not None:
                    print(f"  {describe(world, wrist)}")
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\nstopped")


# ---------------------------------------------------------------------------
# 3D viewer
# ---------------------------------------------------------------------------

def run_viz(sub, show, absolute, span):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    panels = ["world"] if show in ("world", "image") else ["world", "image"]
    if show == "image":
        panels = ["image"]

    fig = plt.figure(figsize=(6 * len(panels), 6.5))
    fig.canvas.manager.set_window_title("hand pose — /hand/joint_angles")
    axes = {}
    for i, name in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        ax.set_title(f"{name} landmarks", fontsize=11)
        axes[name] = ax

    artists = {}
    for name, ax in axes.items():
        lines = [ax.plot([], [], [], lw=2.5,
                         color=FINGER_COLORS[BONE_COLOR[k]])[0]
                 for k in range(len(BONES))]
        pts = ax.plot([], [], [], "o", ms=3.5, color="#22223b")[0]
        artists[name] = (lines, pts)

    status = fig.text(0.5, 0.035, "waiting for messages ...", ha="center",
                      fontsize=9, family="monospace")
    hint = fig.text(0.5, 0.008,
                    "hold hand flat, palm down — note which axis the fingers "
                    "point along; det must be +1",
                    ha="center", fontsize=8, color="#6c757d")

    def setup_axes(ax, center):
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    for ax in axes.values():
        setup_axes(ax, np.zeros(3))

    def update(_frame):
        raw = sub.poll()
        world, image, wrist = unpack(raw, absolute)
        if world is None:
            return []

        data = {"world": world, "image": image}
        drawn = []
        for name, ax in axes.items():
            lm = data.get(name)
            lines, pts = artists[name]
            if lm is None:
                continue
            for k, (a, b) in enumerate(BONES):
                lines[k].set_data([lm[a, 0], lm[b, 0]], [lm[a, 1], lm[b, 1]])
                lines[k].set_3d_properties([lm[a, 2], lm[b, 2]])
                drawn.append(lines[k])
            pts.set_data(lm[:, 0], lm[:, 1])
            pts.set_3d_properties(lm[:, 2])
            drawn.append(pts)
            # Keep the hand centred as it moves; cheap and avoids it drifting
            # out of frame in absolute mode.
            setup_axes(ax, lm.mean(0))

        status.set_text(f"{len(raw)} floats | msgs {sub.count} | "
                        f"{describe(world, wrist)}")
        drawn.append(status)
        return drawn

    anim = FuncAnimation(fig, update, interval=33, blit=False,
                         cache_frame_data=False)
    fig._anim = anim          # keep a reference or GC kills the animation
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/hand/joint_angles")
    ap.add_argument("--show", choices=["world", "image", "both"], default="world",
                    help="which landmark block to draw")
    ap.add_argument("--absolute", action="store_true",
                    help="add raw[0:3] back so the hand moves in the world "
                         "instead of being pinned at the origin")
    ap.add_argument("--span", type=float, default=0.13,
                    help="half-width of the view box in metres")
    ap.add_argument("--print", dest="text", action="store_true",
                    help="text summary only, no window (headless or SSH)")
    ap.add_argument("--hz", type=float, default=4.0,
                    help="print rate in --print mode")
    args = ap.parse_args()

    sub = HandSubscriber(args.topic)
    try:
        if args.text:
            run_print(sub, args.absolute, args.hz)
        else:
            run_viz(sub, args.show, args.absolute, args.span)
    finally:
        sub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
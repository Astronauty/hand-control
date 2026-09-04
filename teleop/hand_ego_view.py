#!/usr/bin/env python3
"""
hand_ego_view.py — "what the tracker sees": the hand projected into the
headset's own camera frustum, with FoV bounds and range warnings.

Why not the real camera feed
----------------------------
You can't have one. HTC does not expose raw camera images on the Focus series —
passthrough is a compositor underlay/overlay the runtime draws, and an app can
never read those pixels. So there is no image stream to forward. (Raw camera
texture access existed on the original Focus and was removed.)

What this does instead is reconstruct the geometry: it takes the headset pose
and the hand landmarks, both of which ARE available, and renders the hand as it
falls in the headset's field of view. For debugging tracking dropouts that is
the information you actually want — whether the hand is near the FoV edge, too
close, too far, or outside the volume entirely — none of which needs pixels.

Requires the publisher to place the headset pose at raw[3:10] (it does; that
region is unused by the retargeter).

Usage
-----
    python3 ui/hand_ego_view.py
    python3 ui/hand_ego_view.py --fov 100 --trail 60
"""

import argparse
import sys
from collections import deque

import numpy as np

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
FINGER_COLORS = {
    "thumb": "#e4572e", "index": "#f3a712", "middle": "#4ea699",
    "ring": "#2e86ab", "pinky": "#a23b72", "palm": "#8d99ae",
}
BONE_COLOR = (["thumb"] * 4 + ["index"] * 4 + ["middle"] * 4
              + ["ring"] * 4 + ["pinky"] * 4 + ["palm"] * 3)

# Hand-tracking works roughly between these ranges on a headset-mounted sensor.
# Outside them the estimate degrades well before tracking is formally lost.
NEAR_M, FAR_M = 0.20, 1.00


def quat_to_R(q):
    """OpenXR quaternion (x,y,z,w) -> 3x3 rotation."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project(pts_world, head_p, head_q, fov_deg, aspect):
    """
    World points -> normalized image coords in [-1,1], plus per-point depth.

    OpenXR camera convention: the headset looks down its own -Z, +X right,
    +Y up. Depth is therefore -z in head-local coordinates.

    Returns (uv (N,2), depth (N,)). Points behind the camera get depth <= 0 and
    should not be drawn.
    """
    R = quat_to_R(head_q)
    local = (pts_world - head_p) @ R          # world -> head-local (R^T applied)
    depth = -local[:, 2]

    f = 1.0 / np.tan(np.radians(fov_deg) * 0.5)
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    u = (local[:, 0] / safe) * f / aspect
    v = (local[:, 1] / safe) * f
    return np.column_stack([u, v]), depth


class HandSubscriber:
    def __init__(self, topic):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray
        rclpy.init()
        self._rclpy = rclpy
        self.node = Node("hand_ego_view")
        self.latest = None
        self.count = 0
        self.node.create_subscription(Float32MultiArray, topic, self._cb, 10)
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


def unpack(raw):
    """raw -> (landmarks_world (21,3), head_p (3,), head_q (4,)) or Nones."""
    if raw is None or len(raw) < 120:
        return None, None, None
    wrist = raw[0:3]
    head_p = raw[3:6]
    head_q = raw[6:10]
    lm = raw[57:120].reshape(21, 3) + wrist       # wrist-relative -> absolute
    if np.allclose(head_q, 0.0):
        return lm, None, None                     # publisher not sending head pose
    return lm, head_p, head_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/hand/joint_angles")
    ap.add_argument("--fov", type=float, default=100.0,
                    help="horizontal FoV of the tracking cameras, degrees")
    ap.add_argument("--aspect", type=float, default=1.0)
    ap.add_argument("--trail", type=int, default=45,
                    help="frames of wrist trail to draw (0 disables)")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    sub = HandSubscriber(args.topic)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.canvas.manager.set_window_title("egocentric view — what the tracker sees")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_facecolor("#11131a")
    ax.set_xticks([]); ax.set_yticks([])

    # FoV bound, plus an inner margin where estimates start to degrade.
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, ec="#4a4e69", lw=2))
    ax.add_patch(Rectangle((-0.8, -0.8), 1.6, 1.6, fill=False, ec="#2f3247",
                           lw=1, ls="--"))
    ax.text(0, 1.06, "field of view", color="#4a4e69", ha="center", fontsize=8)
    ax.plot(0, 0, "+", color="#2f3247", ms=10)

    lines = [ax.plot([], [], lw=3, color=FINGER_COLORS[BONE_COLOR[k]],
                     solid_capstyle="round")[0] for k in range(len(BONES))]
    pts = ax.plot([], [], "o", ms=5, color="#e8e9f3")[0]
    trail_line, = ax.plot([], [], "-", lw=1, color="#5c6378", alpha=0.7)
    trail = deque(maxlen=max(args.trail, 1))

    status = ax.text(-1.3, -1.22, "", color="#c9cbd8", fontsize=9,
                     family="monospace")
    warn = ax.text(0, 0.88, "", color="#ff6b6b", ha="center", fontsize=12,
                   fontweight="bold")

    def update(_f):
        raw = sub.poll()
        lm, head_p, head_q = unpack(raw)
        if lm is None:
            status.set_text("waiting for /hand/joint_angles ...")
            return []
        if head_p is None:
            status.set_text("no headset pose in message — update the publisher "
                            "(it writes raw[3:10])")
            return [status]

        uv, depth = project(lm, head_p, head_q, args.fov, args.aspect)
        wrist_depth = depth[0]

        # Behind the camera: nothing sensible to draw.
        if wrist_depth <= 0:
            warn.set_text("HAND BEHIND HEADSET")
            for ln in lines:
                ln.set_data([], [])
            pts.set_data([], [])
            status.set_text(f"msgs {sub.count} | wrist {wrist_depth:+.2f} m")
            return lines + [pts, status, warn]

        for k, (a, b) in enumerate(BONES):
            if depth[a] > 0 and depth[b] > 0:
                lines[k].set_data([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]])
            else:
                lines[k].set_data([], [])
        vis = depth > 0
        pts.set_data(uv[vis, 0], uv[vis, 1])

        if args.trail:
            trail.append(uv[0].copy())
            t = np.array(trail)
            trail_line.set_data(t[:, 0], t[:, 1])

        # Warnings, in the order that matters for a tracking dropout.
        edge = float(np.max(np.abs(uv[vis]))) if vis.any() else 9.9
        msgs = []
        if edge > 1.0:
            msgs.append("OUTSIDE FoV")
        elif edge > 0.8:
            msgs.append("NEAR FoV EDGE")
        if wrist_depth < NEAR_M:
            msgs.append("TOO CLOSE")
        elif wrist_depth > FAR_M:
            msgs.append("TOO FAR")
        warn.set_text("  ".join(msgs))

        status.set_text(
            f"msgs {sub.count} | range {wrist_depth:.2f} m | "
            f"edge {edge*100:.0f}% of FoV | "
            f"head ({head_p[0]:+.2f},{head_p[1]:+.2f},{head_p[2]:+.2f}) m")
        return lines + [pts, trail_line, status, warn]

    anim = FuncAnimation(fig, update, interval=33, blit=False,
                         cache_frame_data=False)
    fig._anim = anim
    plt.tight_layout()
    try:
        plt.show()
    finally:
        sub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
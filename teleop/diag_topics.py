#!/usr/bin/env python3
"""Standalone diagnostic: subscribe to the multicam topics and print arrival
stats. Separates "landmark nodes not publishing" from "fusion not consuming".

Run it WHILE the pipeline is up (in another terminal):
    python teleop/diag_topics.py --cameras c0 c1

Every second it prints, per camera: messages/sec on the landmarks topic, how
many carried a detected hand, the newest message age, and preview msgs/sec.
If landmarks msgs/sec is ~30 and 'det' is high, the landmark nodes are fine and
the problem is downstream (fusion/viewer). If it's 0, that camera isn't
publishing (camera/MediaPipe/launch issue).
"""
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage


class Diag(Node):
    def __init__(self, names):
        super().__init__("diag_topics")
        self._names = names
        self._lm = {n: 0 for n in names}
        self._det = {n: 0 for n in names}
        self._last = {n: 0.0 for n in names}
        self._prev = {n: 0 for n in names}
        for n in names:
            self.create_subscription(
                Float32MultiArray, f"/hand/cam_{n}/landmarks",
                lambda m, nm=n: self._on_lm(nm, m), 10)
            self.create_subscription(
                CompressedImage, f"/hand/cam_{n}/preview",
                lambda m, nm=n: self._on_prev(nm), 2)
        self.create_timer(1.0, self._report)
        self._t0 = time.time()

    def _on_lm(self, name, msg):
        self._lm[name] += 1
        d = msg.data
        if len(d) >= 2 and float(d[1]) > 0.5:
            self._det[name] += 1
        if len(d) >= 1:
            self._last[name] = float(d[0])

    def _on_prev(self, name):
        self._prev[name] += 1

    def _report(self):
        now = time.time()
        parts = []
        for n in self._names:
            age = (now - self._last[n]) * 1e3 if self._last[n] else -1
            parts.append(
                f"{n}: lm={self._lm[n]}/s det={self._det[n]}/s "
                f"prev={self._prev[n]}/s age={age:.0f}ms")
            self._lm[n] = 0; self._det[n] = 0; self._prev[n] = 0
        print("  ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="+", required=True)
    args = ap.parse_args()
    rclpy.init()
    node = Diag(args.cameras)
    print(f"[diag] listening on {args.cameras}. Ctrl-C to stop.\n"
          f"  lm/s ~30 + det/s high  => landmark nodes OK (problem is downstream)\n"
          f"  lm/s = 0               => that camera isn't publishing\n")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

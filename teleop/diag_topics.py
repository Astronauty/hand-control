#!/usr/bin/env python3
"""Standalone diagnostic: subscribe to the multicam topics and print arrival
stats. Pinpoints WHICH stage of the chain is frozen when the camera streams
stall in dexpilot / contact-aware teleop.

Run it WHILE the pipeline is up (in another terminal):
    python teleop/diag_topics.py --cameras c0 c1

Every second it prints, per camera: messages/sec on the landmarks topic, how
many carried a detected hand, the newest message age, and preview msgs/sec — and
a final row for /hand/joint_angles (the fused output the teleop actually
consumes). Reading the whole chain (camera -> landmarks -> fusion -> joint_angles)
in one place makes the freeze unambiguous:

    lm/s = 0                 => that camera isn't publishing (camera read wedge /
                                MediaPipe stall). Watch the landmark node's
                                terminal for "camera frame FROZEN ... reopening".
    lm/s ~30, ja/s = 0       => landmark nodes fine, FUSION stalled or stale-
                                gating everything. Watch fusion for "waiting N/2".
    lm/s ~30, ja/s ~60       => whole pipeline healthy => the freeze is in the
                                teleop / dashboard CONSUMER, not the camera chain.

'age' is the time since the newest message on each topic; a climbing age is the
live signature of a freeze. Pass --warn-stale-ms N to also print a loud STALE
line the instant any topic's age crosses N ms, so you catch the exact freeze
moment instead of eyeballing the per-second rows.
"""
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage

# Key for the fused-output row (kept distinct from any camera name).
_JA = "joint_angles"


class Diag(Node):
    def __init__(self, names, warn_stale_ms: float = 0.0):
        super().__init__("diag_topics")
        self._names = names
        self._warn_stale_ms = warn_stale_ms
        self._lm = {n: 0 for n in names}
        self._det = {n: 0 for n in names}
        self._last = {n: 0.0 for n in names}
        self._prev = {n: 0 for n in names}
        # Match the publishers' BEST_EFFORT sensor QoS, or a reliable sub here
        # won't even connect to the (now best-effort) landmark/joint_angles pubs.
        from hand_message import sensor_qos
        for n in names:
            self.create_subscription(
                Float32MultiArray, f"/hand/cam_{n}/landmarks",
                lambda m, nm=n: self._on_lm(nm, m), sensor_qos())
            self.create_subscription(
                CompressedImage, f"/hand/cam_{n}/preview",
                lambda m, nm=n: self._on_prev(nm), sensor_qos())
        # Fused output the teleop consumes. The 183-float message carries NO
        # wall-clock stamp (data[0:3] is wrist position), so age it by LOCAL
        # receive time — which is what freeze detection wants anyway (we care
        # when WE last got a message, not when the hand was captured).
        self._ja = 0
        self._ja_last = 0.0
        self.create_subscription(
            Float32MultiArray, "/hand/joint_angles",
            lambda m: self._on_ja(), sensor_qos())
        # Tracks which topics are already flagged stale, so the STALE warning
        # fires once per freeze episode (on the crossing) rather than every tick.
        self._stale_warned: set[str] = set()
        self.create_timer(1.0, self._report)
        # Faster stale-crossing check so the STALE line lands near the freeze
        # instant, independent of the 1 Hz summary.
        if self._warn_stale_ms > 0:
            self.create_timer(0.1, self._check_stale)
        self._t0 = time.time()

    def _on_lm(self, name, msg):
        self._lm[name] += 1
        d = msg.data
        if len(d) >= 2 and float(d[1]) > 0.5:
            self._det[name] += 1
        # Prefer the camera's own capture stamp (data[0]) for age; it reflects
        # the true source freeze. Falls back handled in _report when never-rx.
        if len(d) >= 1:
            self._last[name] = float(d[0])

    def _on_prev(self, name):
        self._prev[name] += 1

    def _on_ja(self):
        self._ja += 1
        self._ja_last = time.time()

    def _age_ms(self, now, stamp):
        return (now - stamp) * 1e3 if stamp else -1.0

    def _check_stale(self):
        """Between summaries, flag any topic whose newest message crossed the
        stale threshold. Fires once per episode; clears when the topic recovers."""
        now = time.time()
        ages = {n: self._age_ms(now, self._last[n]) for n in self._names}
        ages[_JA] = self._age_ms(now, self._ja_last)
        for key, age in ages.items():
            stale = age < 0 or age > self._warn_stale_ms
            if stale and key not in self._stale_warned:
                self._stale_warned.add(key)
                label = "joint_angles" if key == _JA else key
                shown = "never-rx" if age < 0 else f"{age:.0f}ms"
                print(f"  !! STALE {label}: age={shown} "
                      f"(> {self._warn_stale_ms:.0f}ms) at t+{now - self._t0:.1f}s")
            elif not stale and key in self._stale_warned:
                self._stale_warned.discard(key)
                print(f"  ++ RECOVERED {'joint_angles' if key == _JA else key} "
                      f"at t+{now - self._t0:.1f}s")

    def _report(self):
        now = time.time()
        parts = []
        for n in self._names:
            age = self._age_ms(now, self._last[n])
            parts.append(
                f"{n}: lm={self._lm[n]}/s det={self._det[n]}/s "
                f"prev={self._prev[n]}/s age={age:.0f}ms")
            self._lm[n] = 0; self._det[n] = 0; self._prev[n] = 0
        ja_age = self._age_ms(now, self._ja_last)
        parts.append(f"joint_angles: ja={self._ja}/s age={ja_age:.0f}ms")
        self._ja = 0
        print("  ".join(parts))


def main():
    ap = argparse.ArgumentParser(
        description="Live per-stage arrival stats for the multicam teleop chain.")
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--warn-stale-ms", type=float, default=0.0,
                    help="Print a loud STALE line the instant any topic's newest "
                         "message is older than this (ms). 0 disables. A good "
                         "value is a few frame periods, e.g. 200.")
    args = ap.parse_args()
    rclpy.init()
    node = Diag(args.cameras, warn_stale_ms=args.warn_stale_ms)
    print(f"[diag] listening on {args.cameras} + /hand/joint_angles. Ctrl-C to stop.\n"
          f"  lm/s ~30 + det/s high, ja/s ~60  => chain OK (freeze is in the consumer)\n"
          f"  lm/s ~30 but ja/s = 0            => fusion stalled (watch 'waiting N/2')\n"
          f"  lm/s = 0                         => that camera isn't publishing (read wedge)\n")
    if args.warn_stale_ms > 0:
        print(f"[diag] will flag any topic older than {args.warn_stale_ms:.0f}ms.\n")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

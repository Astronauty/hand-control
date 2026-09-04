#!/usr/bin/env python3
"""
hand_quality.py — record and analyse the hand-tracking stream.

Two problems this exists to solve.

1. NOTHING LOGS THE LANDMARKS. The divergence CSV holds scalar summaries and
   pose_trace.npz holds the robot's qpos, but the hand data that drove them is
   discarded. Tracking quality therefore cannot be audited after a session, and
   a bad-tracking trial is indistinguishable from a bad-control trial.

2. THE RETARGETER'S CONSTANTS WERE TUNED ON MEDIAPIPE. calibration/
   retarget_config.json's EPS/ETA1/ETA2 are absolute distances in metres,
   calibrated against MediaPipe world landmarks — whose scale is model-derived
   and only approximately metric. OpenXR reports true metric joints from a
   different estimator. Applying the same thresholds to both is an untested
   assumption; the bone-length report below measures the actual scale so it can
   be checked rather than assumed.

Usage
-----
    python3 teleop/hand_quality.py --record 60        # capture 60 s, then report
    python3 teleop/hand_quality.py --analyse logs/hand_quality/hq_*.npz

Why bone length is the core metric
----------------------------------
A hand's bones do not change length. So the standard deviation of a measured
bone length across frames is pure estimator noise, needing no ground truth and
no special procedure — just move naturally. Coefficient of variation (std/mean)
makes it comparable across bones of different sizes, and directly comparable
between the webcam and headset pipelines.
"""

import argparse
import os
import sys
import time

import numpy as np

# MediaPipe 21-landmark skeleton. Each entry is a bone whose length is fixed in
# a real hand — the invariant the noise measurement rests on.
BONES = [
    ("wrist-thumb_cmc", 0, 1), ("thumb_cmc-mcp", 1, 2),
    ("thumb_mcp-ip", 2, 3), ("thumb_ip-tip", 3, 4),
    ("wrist-index_mcp", 0, 5), ("index_mcp-pip", 5, 6),
    ("index_pip-dip", 6, 7), ("index_dip-tip", 7, 8),
    ("wrist-middle_mcp", 0, 9), ("middle_mcp-pip", 9, 10),
    ("middle_pip-dip", 10, 11), ("middle_dip-tip", 11, 12),
    ("wrist-ring_mcp", 0, 13), ("ring_mcp-pip", 13, 14),
    ("ring_pip-dip", 14, 15), ("ring_dip-tip", 15, 16),
    ("wrist-pinky_mcp", 0, 17), ("pinky_mcp-pip", 17, 18),
    ("pinky_pip-dip", 18, 19), ("pinky_dip-tip", 19, 20),
]

# Rough adult-hand spans, for a sanity check on absolute scale. Not precise
# anatomy — wide enough that a real hand passes and a scale error fails.
PLAUSIBLE_M = {
    "wrist-middle_mcp": (0.07, 0.11),      # palm length
    "index_mcp-pip":    (0.03, 0.055),     # proximal phalanx
    "index_dip-tip":    (0.015, 0.030),    # distal phalanx
}


def record(topic, seconds, out_dir):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    frames, stamps = [], []

    rclpy.init()
    node = Node("hand_quality")

    def _cb(msg):
        raw = np.asarray(msg.data, dtype=np.float32)
        if len(raw) >= 120:
            frames.append(raw[:183].copy() if len(raw) >= 183 else raw[:120].copy())
            stamps.append(time.time())

    node.create_subscription(Float32MultiArray, topic, _cb, 10)
    print(f"recording {seconds:g}s from {topic} — move your hand naturally, "
          f"open and close, rotate the wrist")

    t0 = time.time()
    last = 0
    try:
        while time.time() - t0 < seconds:
            rclpy.spin_once(node, timeout_sec=0.05)
            n = len(frames)
            if n and n // 100 != last // 100:
                print(f"  {n} frames  ({n / max(time.time()-t0, 1e-9):.0f} Hz)")
            last = n
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not frames:
        sys.exit("no frames received — is the publisher running and a hand in view?")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"hq_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    # Ragged if the message length changed mid-run; pad to the widest.
    w = max(len(f) for f in frames)
    arr = np.zeros((len(frames), w), dtype=np.float32)
    for i, f in enumerate(frames):
        arr[i, :len(f)] = f
    np.savez_compressed(path, raw=arr, t_wall=np.array(stamps))
    print(f"saved {path}  ({len(frames)} frames)")
    return path


def analyse(path):
    d = np.load(path)
    raw = d["raw"]
    tw = d["t_wall"]
    n = len(raw)
    if n < 10:
        sys.exit("too few frames to analyse")

    # raw[0:3] wrist, raw[57:120] wrist-relative world landmarks.
    lm = raw[:, 57:120].reshape(n, 21, 3)
    wrist = raw[:, 0:3]
    dur = tw[-1] - tw[0]

    print(f"{path}")
    print(f"  {n} frames over {dur:.1f}s  ({n/max(dur,1e-9):.1f} Hz mean)")

    # --- rate stability -----------------------------------------------------
    dt = np.diff(tw)
    print(f"  frame interval: median {np.median(dt)*1e3:.1f} ms, "
          f"p95 {np.percentile(dt,95)*1e3:.1f} ms, max {dt.max()*1e3:.0f} ms")
    gaps = int((dt > 3 * np.median(dt)).sum())
    print(f"  dropouts (>3x median interval): {gaps} "
          f"({100.0*gaps/len(dt):.1f}% of intervals)")

    # --- bone lengths: the core noise metric --------------------------------
    # A real bone has constant length, so std across frames is pure estimator
    # noise. No ground truth or calibration procedure required.
    print("\n  bone length (mm)      mean    std     CV%    range")
    cvs = []
    for name, a, b in BONES:
        L = np.linalg.norm(lm[:, a] - lm[:, b], axis=1)
        L = L[L > 1e-6]
        if len(L) < 10:
            continue
        cv = 100.0 * L.std() / L.mean()
        cvs.append(cv)
        flag = ""
        if name in PLAUSIBLE_M:
            lo, hi = PLAUSIBLE_M[name]
            if not (lo <= L.mean() <= hi):
                flag = f"  <-- outside {lo*1e3:.0f}-{hi*1e3:.0f}mm, CHECK SCALE"
        print(f"  {name:<20} {L.mean()*1e3:7.1f} {L.std()*1e3:6.2f} "
              f"{cv:6.1f}  {L.min()*1e3:5.1f}-{L.max()*1e3:.1f}{flag}")

    print(f"\n  median bone CV: {np.median(cvs):.1f}%   "
          f"(lower is better; this is the headline noise number)")

    # --- jitter: frame-to-frame fingertip motion ----------------------------
    # High-percentile values are glitches rather than hand motion — a hand
    # cannot move 5 cm in one frame.
    tips = [4, 8, 12, 16, 20]
    step = np.linalg.norm(np.diff(lm[:, tips], axis=0), axis=2)
    print(f"\n  fingertip step/frame: median {np.median(step)*1e3:.2f} mm, "
          f"p99 {np.percentile(step,99)*1e3:.1f} mm, max {step.max()*1e3:.0f} mm")
    glitch = int((step > 0.05).sum())
    print(f"  implausible jumps (>50mm in one frame): {glitch}")

    # --- wrist travel: is the operator using the space? ---------------------
    span = wrist.max(0) - wrist.min(0)
    print(f"\n  wrist travel (m): x {span[0]:.2f}  y {span[1]:.2f}  z {span[2]:.2f}")

    # --- pinch distances vs the retargeter's thresholds ---------------------
    # EPS/ETA are absolute metres in retarget_config.json, calibrated against
    # MEDIAPIPE world landmarks. If the headset's scale differs, the pinch
    # detector fires at the wrong hand aperture — which is why these are worth
    # reading against the configured value rather than assuming.
    print("\n  thumb-to-finger distance (the retargeter's S1, mm):")
    for nm, tip in (("index", 8), ("middle", 12), ("ring", 16)):
        s1 = np.linalg.norm(lm[:, 4] - lm[:, tip], axis=1)
        print(f"    thumb-{nm:<7} min {s1.min()*1e3:5.1f}  "
              f"median {np.median(s1)*1e3:5.1f}  max {s1.max()*1e3:5.1f}")
    print("    compare against EPS in calibration/retarget_config.json "
          "(0.03 = 30mm).")
    print("    If your open-hand minimum never approaches EPS, pinch never")
    print("    triggers; if the median sits below it, it never releases.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/hand/joint_angles")
    ap.add_argument("--record", type=float, metavar="SECONDS")
    ap.add_argument("--analyse", metavar="NPZ")
    ap.add_argument("--out-dir", default="logs/hand_quality")
    args = ap.parse_args()

    if args.analyse:
        analyse(args.analyse)
    elif args.record:
        analyse(record(args.topic, args.record, args.out_dir))
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

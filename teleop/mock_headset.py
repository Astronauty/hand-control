#!/usr/bin/env python3
"""
mock_headset.py — stands in for the Focus Vision so the rest of the pipeline
can be tested with no hardware.

Speaks the identical wire format hand_sender.gd will send, so everything
downstream (vive_hand_publisher -> /hand/joint_angles -> DexPilotController ->
MuJoCo) is exercised for real. Only the headset is simulated.

Usage
-----
    # terminal 1
    python3 ui/vive_hand_publisher.py --hand right

    # terminal 2
    python3 ui/mock_headset.py --motion open_close

    # terminal 3
    python3 kinova_leap_pick_place.py --mode dexpilot
    # (press 8 in the viewer to start tracking)

Motions
-------
    static      hand held still, flat, palm down   — check the frame mapping
    open_close  fingers curl and extend at 0.3 Hz  — check finger retargeting
    pinch       thumb and index meet and part      — check DexPilot keyvectors
    wave        wrist translates in a circle       — check arm IK and workspace
    noise       static plus 3 mm jitter            — check filtering and slew

Notes
-----
The skeleton is a plausible right hand, not an anatomical model. It is for
verifying plumbing, index mapping, and axis conventions — not for judging
tracking accuracy, which needs the real device.
"""

import argparse
import math
import socket
import struct
import sys
import time

import numpy as np

MAGIC = 0x45564956
N_JOINTS = 26
HDR_FMT = "<IHHIQI"
HDR_SIZE = struct.calcsize(HDR_FMT)

# ---------------------------------------------------------------------------
# OpenXR hand joint indices (XrHandJointEXT)
# ---------------------------------------------------------------------------
PALM, WRIST = 0, 1
THUMB = [2, 3, 4, 5]                  # metacarpal, proximal, distal, tip
INDEX = [6, 7, 8, 9, 10]              # metacarpal, proximal, intermediate, distal, tip
MIDDLE = [11, 12, 13, 14, 15]
RING = [16, 17, 18, 19, 20]
PINKY = [21, 22, 23, 24, 25]

FINGERS = [INDEX, MIDDLE, RING, PINKY]

# Lateral offset of each finger's base from the wrist centre, and segment
# lengths, in metres. Roughly adult-hand proportioned.
FINGER_X = {6: -0.020, 11: 0.000, 16: 0.019, 21: 0.036}
FINGER_SEGS = {
    6:  [0.070, 0.040, 0.024, 0.020],   # index: metacarpal then 3 phalanges
    11: [0.072, 0.044, 0.026, 0.021],   # middle
    16: [0.068, 0.040, 0.025, 0.020],   # ring
    21: [0.064, 0.032, 0.019, 0.017],   # pinky
}
THUMB_SEGS = [0.045, 0.035, 0.030]


def build_hand(curl: float, pinch: float) -> np.ndarray:
    """
    Synthesize 26 joint poses in the OpenXR convention: right-handed, Y-up,
    -Z forward, metres, wrist at the origin of the returned frame.

    curl  0 = fingers straight, 1 = fully closed
    pinch 0 = thumb resting,    1 = thumb tip meeting index tip

    The hand lies in the XY plane with fingers extending along +Y (up), which
    after the receiver's Y-up -> Z-up conversion puts them along MuJoCo +Z.
    """
    p = np.zeros((N_JOINTS, 3))

    p[WRIST] = [0.0, 0.0, 0.0]
    p[PALM] = [0.005, 0.045, 0.0]

    for base in (6, 11, 16, 21):
        seg = FINGER_SEGS[base]
        x = FINGER_X[base]
        # Curl bends each successive joint further, as a real finger does.
        angles = [curl * a for a in (0.0, 0.9, 1.1, 0.9)]
        ang = 0.0
        y, z = 0.0, 0.0
        p[base] = [x, y, z]
        for k, (length, delta) in enumerate(zip(seg, angles)):
            ang += delta
            y += length * math.cos(ang)
            z += length * math.sin(ang)     # curl folds toward -Z... see below
            p[base + 1 + k] = [x, y, -z]    # palm faces -Z, so fingers fold that way

    # Thumb. The metacarpal is fixed at the base of the palm; the tip is
    # interpolated toward the *current* index tip as pinch goes 0 -> 1, with
    # the intervening joints spread along the way. Driving the tip from the
    # index tip's actual position is what makes the thumb-index keyvector —
    # the quantity DexPilot cares most about — genuinely shrink.
    base = np.array([-0.030, 0.020, -0.008])
    p[THUMB[0]] = base

    rest_tip = base + np.array([0.052, 0.062, -0.014])   # thumb out, hand open
    index_tip = p[INDEX[4]]
    tip = rest_tip + pinch * (index_tip - rest_tip)

    # Bow the chain slightly off the straight base->tip line so the joints
    # don't sit colinear, which would look degenerate in the viewer.
    bow = np.array([0.012, -0.004, -0.010])
    for k, frac in enumerate((0.42, 0.74, 1.0)):
        pos = base + frac * (tip - base)
        pos = pos + bow * math.sin(math.pi * frac)
        p[THUMB[1] + k] = pos

    return p


def pack(hand: int, seq: int, joints: np.ndarray, head: np.ndarray) -> bytes:
    """One 780-byte frame. Orientations are identity — the receiver only uses
    the wrist quaternion, and identity is a valid, checkable value."""
    body = np.zeros((N_JOINTS, 7), dtype=np.float32)
    body[:, 0:3] = joints
    body[:, 6] = 1.0                         # quaternion w
    head_pose = np.array([*head, 0, 0, 0, 1], dtype=np.float32)

    hdr = struct.pack(HDR_FMT, MAGIC, 1, hand, seq, time.time_ns(), 1)
    return hdr + body.ravel().astype("<f4").tobytes() \
        + head_pose.astype("<f4").tobytes()


def motion_state(name: str, t: float):
    """(curl, pinch, wrist_offset) for the chosen motion at time t seconds."""
    if name == "static":
        return 0.15, 0.0, np.zeros(3)

    if name == "open_close":
        c = 0.5 - 0.5 * math.cos(2 * math.pi * 0.3 * t)
        return c, 0.0, np.zeros(3)

    if name == "pinch":
        p = 0.5 - 0.5 * math.cos(2 * math.pi * 0.4 * t)
        return 0.25 + 0.35 * p, p, np.zeros(3)

    if name == "wave":
        r = 0.08
        off = np.array([r * math.cos(2 * math.pi * 0.2 * t),
                        0.0,
                        r * math.sin(2 * math.pi * 0.2 * t)])
        return 0.2, 0.0, off

    if name == "noise":
        return 0.15, 0.0, np.random.normal(0, 0.003, 3)

    raise ValueError(f"unknown motion {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9870)
    ap.add_argument("--hand", choices=["left", "right"], default="right")
    ap.add_argument("--motion",
                    choices=["static", "open_close", "pinch", "wave", "noise"],
                    default="open_close")
    ap.add_argument("--rate", type=float, default=72.0, help="frames per second")
    ap.add_argument("--wrist", type=float, nargs=3, default=[0.0, 1.15, -0.35],
                    metavar=("X", "Y", "Z"),
                    help="wrist position in OpenXR STAGE coords (Y-up, -Z fwd)")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="fraction of frames to send with tracked=0, to "
                         "exercise the receiver's loss handling")
    args = ap.parse_args()

    hand_id = 0 if args.hand == "left" else 1
    wrist0 = np.array(args.wrist)
    head = np.array([0.0, 1.55, 0.0])

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"connected to {args.host}:{args.port} — sending {args.motion} "
          f"for the {args.hand} hand at {args.rate:g} Hz (ctrl-c to stop)")

    period = 1.0 / args.rate
    t0 = time.monotonic()
    seq = 0
    sent = 0

    try:
        while True:
            t = time.monotonic() - t0
            curl, pinch, off = motion_state(args.motion, t)

            joints = build_hand(curl, pinch) + (wrist0 + off)
            seq += 1

            if args.dropout and np.random.random() < args.dropout:
                # tracked=0 frame: full size, contents ignored by the receiver
                body = np.zeros(N_JOINTS * 7 + 7, dtype=np.float32)
                pkt = struct.pack(HDR_FMT, MAGIC, 1, hand_id, seq,
                                  time.time_ns(), 0) + \
                    body.astype("<f4").tobytes()
            else:
                pkt = pack(hand_id, seq, joints, head)
                sent += 1

            sock.sendall(pkt)

            if seq % int(args.rate * 2) == 0:
                print(f"  t={t:6.1f}s  seq={seq}  tracked_sent={sent}  "
                      f"curl={curl:.2f} pinch={pinch:.2f}")

            time.sleep(max(0.0, period - (time.monotonic() - t0 - t)))
    except KeyboardInterrupt:
        print("\nstopped")
    except (BrokenPipeError, ConnectionResetError):
        print("\nreceiver closed the connection")
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

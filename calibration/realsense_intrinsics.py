#!/usr/bin/env python3
"""Read Intel RealSense factory color intrinsics via pyrealsense2 and write them
in the pipeline's camera_intrinsics_<name>.json format.

Why not the ChArUco `intrinsics` subcommand for a RealSense?
  charuco_calibration.py opens the camera through cv2.VideoCapture, but the teleop
  pipeline captures the RealSense COLOR stream through pyrealsense2 (see
  teleop/hand_landmark_node.py::_RealSenseCapture). If you board-calibrate through
  a different capture path — or, as happened here, at a different RESOLUTION than
  the pipeline streams — the K matrix (fx, fy, cx, cy) and distortion no longer
  match the pixels fusion actually receives, so every reprojected ray is
  mis-scaled and the fused skeleton is dragged off (steady, motion-independent
  hundreds-of-px reprojection error). No amount of extrinsics recalibration fixes
  a resolution/intrinsics mismatch.

RealSense colour is factory-calibrated. This tool asks the SDK for the color
stream's intrinsics at EXACTLY the resolution the pipeline uses (default
640x480 — matching hand_landmark_node's `width or 640, height or 480`), so the
written file is guaranteed consistent with capture. After running this, only the
EXTRINSICS need re-solving (they sit on top of these intrinsics).

Usage:
    # default: 640x480 (what run_multicam / --multicam launches the RS at)
    python calibration/realsense_intrinsics.py --name rs

    # if you launch the pipeline at another size, MATCH it here:
    python calibration/realsense_intrinsics.py --name rs --width 1280 --height 720

Then redo extrinsics for that camera against the shared board:
    python calibration/charuco_calibration.py extrinsics-all --cam c0:0 --cam c1:2 --cam rs:8 --realsense rs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_HERE, "camera_intrinsics_{name}.json")

# RealSense distortion models we can faithfully serialise into OpenCV's
# [k1, k2, p1, p2, k3] convention. Brown-Conrady (and its "inverse"/"modified"
# variants) use exactly that 5-coeff layout; other models (Kannala-Brandt
# fisheye, the F-Theta ones) don't map onto it, so we refuse rather than write
# coefficients the pipeline would misinterpret.
_BROWN_CONRADY = {
    "distortion.brown_conrady",
    "distortion.modified_brown_conrady",
    "distortion.inverse_brown_conrady",
    "distortion.none",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Write RealSense factory color intrinsics as "
                    "camera_intrinsics_<name>.json")
    ap.add_argument("--name", default="rs",
                    help="Camera name; writes camera_intrinsics_<name>.json "
                         "(must match the --multicam / fusion --cameras name).")
    ap.add_argument("--width", type=int, default=640,
                    help="Color width the PIPELINE streams at (default 640, "
                         "matching hand_landmark_node's RealSense default).")
    ap.add_argument("--height", type=int, default=480,
                    help="Color height the PIPELINE streams at (default 480).")
    ap.add_argument("--fps", type=int, default=0,
                    help="Color fps to request; 0 = let the SDK pick (intrinsics "
                         "are fps-independent, this only affects the probe stream).")
    ap.add_argument("--out", default=None,
                    help="Explicit output path (overrides --name).")
    args = ap.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        sys.exit("ERROR: pyrealsense2 not installed. It's the same dependency the "
                 "pipeline uses for --multicam-realsense; install it in this env.")

    out_path = args.out or DEFAULT_OUT.format(name=args.name)

    # Bring up the color stream at the requested size, then read the stream
    # profile's intrinsics. We start the pipeline (rather than querying the
    # sensor cold) so the SDK reports the intrinsics for the ACTIVE profile —
    # exactly what the teleop pipeline gets from the same enable_stream call.
    pipe = rs.pipeline()
    cfg = rs.config()
    fps_candidates = [args.fps] if args.fps else [30, 15, 60, 6]
    profile = None
    last_err = None
    for fps in fps_candidates:
        try:
            cfg.enable_stream(rs.stream.color, args.width, args.height,
                              rs.format.bgr8, fps)
            profile = pipe.start(cfg)
            break
        except Exception as e:            # this (w,h,fps) combo isn't supported
            last_err = e
            cfg = rs.config()             # fresh config for the next attempt
            profile = None
    if profile is None:
        sys.exit(f"ERROR: could not start RealSense color at "
                 f"{args.width}x{args.height} (bgr8). The device may not offer "
                 f"this size — check with `rs-enumerate-devices`. Last error: "
                 f"{last_err}")

    try:
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
    finally:
        pipe.stop()

    model = str(intr.model).lower()
    if model not in _BROWN_CONRADY:
        sys.exit(f"ERROR: this color stream uses distortion model '{intr.model}', "
                 f"which does not map to OpenCV's [k1,k2,p1,p2,k3]. The pipeline "
                 f"assumes Brown-Conrady. Aborting rather than writing coefficients "
                 f"the fusion node would misapply.")

    # rs intrinsics -> OpenCV convention.
    #   camera_matrix = [[fx,0,ppx],[0,fy,ppy],[0,0,1]]
    #   coeffs order in rs (Brown-Conrady) is [k1,k2,p1,p2,k3] — same as OpenCV.
    camera_matrix = [
        [float(intr.fx), 0.0,            float(intr.ppx)],
        [0.0,            float(intr.fy), float(intr.ppy)],
        [0.0,            0.0,            1.0],
    ]
    dist_coeffs = [float(c) for c in intr.coeffs]   # already [k1,k2,p1,p2,k3]

    data = {
        "image_size": [int(intr.width), int(intr.height)],
        "source": "realsense_sdk_factory",
        "distortion_model": str(intr.model),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[rs-intrinsics] color stream {intr.width}x{intr.height}, "
          f"model={intr.model}")
    print(f"[rs-intrinsics] fx={intr.fx:.2f} fy={intr.fy:.2f} "
          f"ppx={intr.ppx:.2f} ppy={intr.ppy:.2f}")
    print(f"[rs-intrinsics] dist(k1,k2,p1,p2,k3)={dist_coeffs}")
    print(f"[rs-intrinsics] wrote {out_path}")
    print(f"[rs-intrinsics] NOTE: these are valid ONLY at {intr.width}x{intr.height}. "
          f"Launch the RealSense at this size, then re-solve its EXTRINSICS.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Launch the full multi-camera hand-tracking pipeline in one command.

Spawns one hand_landmark_node per camera plus the hand_fusion_node, wiring their
names together. Configurable camera count: pass --cam <name>:<index> per physical
camera (2 or more). The fusion node publishes the SAME /hand/joint_angles message
the single-camera publisher does, so run your existing teleop (dexpilot /
contact-aware) against it unchanged.

Examples:
    # 3 cameras: names c0,c1,rs at OpenCV indices 1,2,3
    python teleop/run_multicam.py --cam c0:1 --cam c1:2 --cam rs:3

    # 2 cameras, show debug windows
    python teleop/run_multicam.py --cam c0:1 --cam c1:2 --show

Each <name> must have calibration files camera_intrinsics_<name>.json and
camera_extrinsics_<name>.json in calibration/ (see
calibration/charuco_calibration.py --name), all solved against the SAME board.

This is a plain process supervisor (no ROS launch package needed, matching how
the repo's other entrypoints run). Ctrl-C tears everything down.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_LANDMARK = os.path.join(_HERE, "hand_landmark_node.py")
_FUSION = os.path.join(_HERE, "hand_fusion_node.py")
_CALIB_DIR = os.path.join(os.path.dirname(_HERE), "calibration")


def _parse_cam(spec: str) -> tuple[str, int, tuple[int, int] | None]:
    """Parse a --cam <name>:<index>[:<W>x<H>] spec.

    Optional per-camera resolution lets each camera run at its own native size
    (triangulation does NOT require a shared resolution — each camera's intrinsics
    encode its own). Examples:
        c0:0            -> (c0, 0, None)     use default / --max-res
        c1:2:1280x960   -> (c1, 2, (1280,960))
    The resolution here MUST match that camera's calibrated intrinsics size.
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"--cam expects <name>:<index>[:<W>x<H>], got {spec!r}")
    name = parts[0]
    if not name:
        raise argparse.ArgumentTypeError(f"empty camera name in {spec!r}")
    try:
        idx = int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"index must be an int in {spec!r}")
    res = None
    if len(parts) >= 3 and parts[2]:
        try:
            w, h = parts[2].lower().split("x")
            res = (int(w), int(h))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"resolution must be <W>x<H> in {spec!r}")
    return name, idx, res


def _check_calibration(names: list[str]) -> None:
    """Warn (don't fail) if a camera's per-name calibration is missing.

    The fusion node falls back to the unsuffixed files, but for a real multi-cam
    rig every camera needs its OWN extrinsic in the shared frame, so flag it."""
    for n in names:
        intr = os.path.join(_CALIB_DIR, f"camera_intrinsics_{n}.json")
        extr = os.path.join(_CALIB_DIR, f"camera_extrinsics_{n}.json")
        missing = [os.path.basename(p) for p in (intr, extr)
                   if not os.path.exists(p)]
        if missing:
            print(f"[run] WARNING: camera '{n}' missing {missing} — it will fall "
                  f"back to the shared single-cam calibration, which is WRONG for "
                  f"triangulation. Run: python calibration/charuco_calibration.py "
                  f"intrinsics --camera <idx> --name {n}  (then extrinsics).")


def main():
    ap = argparse.ArgumentParser(
        description="Launch multi-camera hand-tracking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--cam", action="append", type=_parse_cam, required=True,
                    metavar="NAME:INDEX[:WxH]",
                    help="Camera name:OpenCV-index, optional :WxH resolution. "
                         "Repeat for each camera (>=2). Cameras may run at "
                         "DIFFERENT resolutions (each uses its own intrinsics).")
    ap.add_argument("--max-res", action="store_true",
                    help="Open EVERY camera at its highest supported resolution "
                         "(unless that camera specified an explicit :WxH). Best "
                         "landmark precision; calibrate each camera at the size it "
                         "actually opens (printed by its landmark node on launch).")
    ap.add_argument("--width", type=int, default=640,
                    help="Fallback width for cameras with no :WxH and no --max-res.")
    ap.add_argument("--height", type=int, default=480,
                    help="Fallback height for cameras with no :WxH and no --max-res.")
    ap.add_argument("--show", action="store_true",
                    help="Show a per-camera window: hand skeleton + the shared "
                         "world-frame axes projected through that camera (if all "
                         "cameras' axes land on the same spot, extrinsics agree).")
    ap.add_argument("--show-fused", action="store_true",
                    help="Show TWO windows: an orbitable 3D fused view (with camera "
                         "positions) and a grid of all camera views (raw image + "
                         "landmarks). Camera grid uses --preview-w sized previews.")
    ap.add_argument("--preview-w", type=int, default=640,
                    help="Per-camera preview width (px) in the camera grid window. "
                         "Raise for bigger camera views (e.g. 800, 960).")
    ap.add_argument("--preview-hz", type=float, default=15.0,
                    help="Camera preview publish rate (Hz); independent of tracking.")
    ap.add_argument("--sync-window", type=float, default=0.033)
    ap.add_argument("--vis-thresh", type=float, default=0.3)
    ap.add_argument("--min-views", type=int, default=2)
    args = ap.parse_args()

    if len(args.cam) < 2:
        ap.error("need at least 2 --cam entries for triangulation")

    names = [n for n, _, _ in args.cam]
    if len(set(names)) != len(names):
        ap.error(f"duplicate camera names: {names}")
    _check_calibration(names)

    procs: list[subprocess.Popen] = []

    def spawn(argv: list[str], label: str) -> None:
        print(f"[run] starting {label}: {' '.join(argv)}")
        procs.append(subprocess.Popen([sys.executable] + argv))

    try:
        # One landmark node per camera. Resolution precedence per camera:
        #   explicit :WxH  >  --max-res  >  global --width/--height fallback.
        for name, idx, res in args.cam:
            argv = [_LANDMARK, "--camera", str(idx), "--name", name]
            if res is not None:
                argv += ["--width", str(res[0]), "--height", str(res[1])]
            elif args.max_res:
                argv += ["--max-res"]
            else:
                argv += ["--width", str(args.width), "--height", str(args.height)]
            if args.show:
                argv.append("--show")
            argv += ["--preview-w", str(args.preview_w),
                     "--preview-hz", str(args.preview_hz)]
            spawn(argv, f"landmark cam '{name}' (index {idx})")
            time.sleep(1.0)   # stagger camera opens to avoid USB contention

        # Fusion node consuming all of them.
        fusion_argv = [_FUSION, "--cameras", *names,
                       "--sync-window", str(args.sync_window),
                       "--vis-thresh", str(args.vis_thresh),
                       "--min-views", str(args.min_views)]
        if args.show_fused:
            fusion_argv.append("--show")
        spawn(fusion_argv, "fusion node")

        print(f"\n[run] pipeline up: {len(names)} cameras -> /hand/joint_angles. "
              f"Run your teleop (dexpilot / contact-aware) as usual. Ctrl-C to stop.\n")

        # Supervise: if any child dies, tear the rest down.
        while True:
            time.sleep(0.5)
            dead = [p for p in procs if p.poll() is not None]
            if dead:
                print(f"[run] a child process exited (code {dead[0].returncode}); "
                      f"shutting the pipeline down.")
                break
    except KeyboardInterrupt:
        print("\n[run] Ctrl-C — stopping pipeline.")
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        # Give them a moment to clean up, then hard-kill any stragglers.
        deadline = time.time() + 3.0
        for p in procs:
            if p.poll() is None and time.time() < deadline:
                try:
                    p.wait(timeout=max(0.1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    pass
        for p in procs:
            if p.poll() is None:
                p.kill()
        print("[run] pipeline stopped.")


if __name__ == "__main__":
    main()

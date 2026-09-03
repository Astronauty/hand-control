#!/usr/bin/env python3
"""Validate that N cameras' extrinsics share ONE world frame — using the LIVE board.

Offline geometry can't catch a broken shared frame: reprojecting a world point
through a camera's own P and back is tautologically consistent, so pairwise
"re-triangulation error" is 0 even when two cameras were calibrated against the
board in different physical poses. The only ground truth is the physical board.

This tool shows the SAME fixed board to every camera in turn, solves its live
pose per camera (solvePnP), and transforms the observed board origin into each
camera's stored WORLD frame:

    p_k = -R_world_cam^k @ t_board_in_cam^k          (board origin in world, via cam k)

If the extrinsics truly share one frame, every camera maps the one physical board
origin to the SAME world point, so the p_k agree (spread ~ a couple cm at worst).
A camera whose extrinsic was solved against a different board pose lands its p_k
far from the others — that's the mis-calibrated view, named explicitly.

The board does NOT need to be at the original calibration spot, and cameras do
NOT need to see it simultaneously — just keep it FIXED while you cycle through
the cameras (SPACE to capture each; it re-uses the last capture per camera).

Usage:
    python calibration/assess_shared_frame.py \
        --cam c0:0 --cam c1:2 --cam rs:8 --realsense rs --square-mm <measured>
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Reuse the board, detectors, camera-open (incl. the pyrealsense2 path), and the
# same board->world remap the pipeline uses — so this measures the EXACT frame.
from charuco_calibration import (          # noqa: E402
    _make_board, _make_detectors, _make_window, _open_camera,
    _named, DEFAULT_INTRINSICS, DEFAULT_EXTRINSICS)

sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "teleop"))
from hand_message import WORLD_FROM_BOARD   # noqa: E402


def _parse_cam(spec: str):
    name, idx = spec.rsplit(":", 1)
    return name, int(idx)


def _load_world_cam(name: str):
    """Return (K, dist, R_world_cam, t_cam_world) for `name`, applying the
    board->world remap exactly as hand_fusion_node._load_camera /
    hand_landmark_node._load_extrinsic do, so p_k lands in the pipeline's world
    frame. t_cam_world is the stored world origin in this camera's frame."""
    intr_p = _named(DEFAULT_INTRINSICS, name)
    extr_p = _named(DEFAULT_EXTRINSICS, name)
    if not (os.path.exists(intr_p) and os.path.exists(extr_p)):
        sys.exit(f"[assess] missing calibration for '{name}': {intr_p} / {extr_p}")
    import json
    K = np.array(json.load(open(intr_p))["camera_matrix"], float)
    dist = np.array(json.load(open(intr_p))["dist_coeffs"], float)
    extr = json.load(open(extr_p))
    R_cam_board = np.array(extr["R_cam_world"], float)
    t_cam_world = np.array(extr["t_cam_world"], float)   # world origin in cam frame
    # R_cam_world = R_cam_board @ WFB^T  (world Z-up -> board -> cam). t is unchanged
    # by the WFB remap (world & board origins coincide; only axes are remapped).
    R_cam_world = R_cam_board @ WORLD_FROM_BOARD.T
    return K, dist, R_cam_world, t_cam_world


def _capture_board_pose(name, idx, K, dist, board, det, square_m, realsense, w, h):
    """Open the camera, live-preview board detection, and on SPACE solvePnP the
    board pose in THIS camera's frame. Returns (R_board_cam, t_board_cam) or None
    (S skip / Q quit)."""
    cap = _open_camera(idx, w, h, False, realsense)
    win = f"shared-frame: {name} — SPACE capture, S skip, Q quit"
    _make_window(win)
    pose = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = det.detectBoard(gray)
        vis = frame.copy()
        n = 0 if ci is None else len(ci)
        rvec = tvec = None
        if n >= 6:
            obj_pts, img_pts = board.matchImagePoints(cc, ci)
            ok_pnp, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
            if ok_pnp:
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, square_m * 2)
        cv2.putText(vis, f"{name}  corners: {n} (>=6; SPACE capture, S skip, Q quit)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release(); cv2.destroyWindow(win)
            raise KeyboardInterrupt
        if key == ord('s'):
            break
        if key == ord(' ') and rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            pose = (R, tvec.ravel())
            break
    cap.release(); cv2.destroyWindow(win)
    return pose


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate cameras share one world frame")
    ap.add_argument("--cam", action="append", type=_parse_cam, required=True,
                    metavar="NAME:INDEX", help="Repeat per camera, e.g. --cam c0:0")
    ap.add_argument("--realsense", action="append", default=[], metavar="NAME",
                    help="Mark a --cam NAME as a RealSense (pyrealsense2 color).")
    ap.add_argument("--square-mm", type=float, required=True,
                    help="MEASURED printed square size in mm (must match calib).")
    ap.add_argument("--rs-width", type=int, default=640)
    ap.add_argument("--rs-height", type=int, default=480)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    rs_names = set(args.realsense)
    square_m = args.square_mm / 1000.0
    board, adict = _make_board(square_m)
    _, det = _make_detectors(board, adict)

    print("[assess] Keep the board FIXED. Capture it from each camera in turn.")
    print("[assess] Each camera maps the ONE physical board origin into its stored")
    print("[assess] world frame; agreement = shared frame, disagreement = the culprit.\n")

    origins = {}    # name -> board origin in world, via that camera
    try:
        for name, idx in args.cam:
            is_rs = name in rs_names
            K, dist, R_world_cam, t_cam_world = _load_world_cam(name)
            w = args.rs_width if is_rs else args.width
            h = args.rs_height if is_rs else args.height
            print(f"[assess] ==== {name} ({'RealSense' if is_rs else f'index {idx}'}) ====")
            pose = _capture_board_pose(name, idx, K, dist, board, det, square_m,
                                       is_rs, w, h)
            if pose is None:
                print(f"[assess] {name}: skipped.")
                continue
            R_board_cam, t_board_cam = pose
            # Board origin expressed in world coords, THROUGH this camera's stored
            # extrinsic. The stored map is x_cam = R_world_cam @ x_world + t_cam_world,
            # so x_world = R_world_cam^T @ (x_cam - t_cam_world). The live board
            # origin sits at t_board_cam in the camera frame:
            p_world = R_world_cam.T @ (t_board_cam - t_cam_world)
            origins[name] = p_world
            print(f"[assess] {name}: board origin in world = {np.round(p_world, 3)} m\n")
    except KeyboardInterrupt:
        print("\n[assess] stopped early (Q).")

    if len(origins) < 2:
        sys.exit("[assess] need >= 2 captured cameras to compare.")

    names = list(origins)
    pts = np.array([origins[n] for n in names])
    centroid = pts.mean(axis=0)
    print("\n[assess] ===== SHARED-FRAME REPORT =====")
    print(f"[assess] {len(names)} cameras placed the board origin at:")
    devs = {}
    for n in names:
        d = np.linalg.norm(origins[n] - centroid)
        devs[n] = d
        print(f"[assess]   {n}: {np.round(origins[n], 3)} m   "
              f"({d*100:5.1f} cm from consensus)")
    spread = max(np.linalg.norm(origins[a] - origins[b])
                 for a in names for b in names)
    print(f"[assess] max pairwise spread: {spread*100:.1f} cm")
    worst = max(devs, key=devs.get)
    # A shared frame should agree to a couple cm (board detection + PnP noise).
    if spread < 0.03:
        print("[assess] VERDICT: cameras SHARE one world frame (spread < 3 cm). OK.")
    else:
        print(f"[assess] VERDICT: FRAME MISMATCH — spread {spread*100:.1f} cm >> a few cm.")
        print(f"[assess]   Outlier is '{worst}' ({devs[worst]*100:.1f} cm off consensus).")
        print(f"[assess]   Re-solve its extrinsic against the SAME fixed board as the "
              f"others (do not move the board between cameras).")


if __name__ == "__main__":
    main()

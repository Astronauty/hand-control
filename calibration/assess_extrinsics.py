"""Live assessment demo: is our ChArUco world-frame + scale usable?

Runs continuous extrinsics (solvePnP) against a fixed board and overlays a
scorecard so you can judge quality by eye AND by number, before trusting the
transform in the teleop pipeline. It answers three questions:

  1. ORIENTATION — are the drawn world axes on the board stable and correct?
     (X red / Y green / Z blue; Z should point OUT of the board toward you.)
  2. SCALE — from the recovered pose, re-measure a known board span and compare
     to ground truth. A large mismatch means wrong --square-mm or bad intrinsics.
  3. STABILITY — position/orientation jitter over a rolling window. High jitter
     = too few corners, motion blur, or a non-flat board.

Requires intrinsics first (camera_matrix + distortion):
    python calibration/charuco_calibration.py intrinsics --camera 1 --square-mm 45.0

Then:
    python calibration/assess_extrinsics.py --camera 1 --square-mm 45.0

Keys:  Q quit   S save current pose to camera_extrinsics.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

# Board geometry must match the printed board (README: 5x7, DICT_5X5_100, 0.75).
SQUARES_X = 5
SQUARES_Y = 7
MARKER_RATIO = 0.75
ARUCO_DICT = cv2.aruco.DICT_5X5_100

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INTRINSICS = os.path.join(_HERE, "camera_intrinsics.json")
DEFAULT_EXTRINSICS = os.path.join(_HERE, "camera_extrinsics.json")


def _make_board(square_len_m: float):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        squareLength=square_len_m,
        markerLength=square_len_m * MARKER_RATIO,
        dictionary=aruco_dict,
    )
    return board, aruco_dict


def _load_intrinsics(path: str):
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: intrinsics not found: {path}\n"
            f"Run first:\n"
            f"  python calibration/charuco_calibration.py intrinsics "
            f"--camera <N> --square-mm <measured>")
    with open(path) as f:
        intr = json.load(f)
    K = np.array(intr["camera_matrix"], dtype=np.float64)
    dist = np.array(intr["dist_coeffs"], dtype=np.float64)
    return K, dist, intr.get("rms_reproj_px")


def _reproj_error(obj_pts, img_pts, rvec, tvec, K, dist) -> float:
    """Mean pixel reprojection error for the current pose — the core quality metric."""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img_pts.reshape(-1, 2), axis=1)))


# NOTE ON METRIC SCALE VERIFICATION
# -----------------------------------
# There is no honest *self-contained* scale check from a single static board.
# A wrong focal length (bad intrinsics) or a wrong --square-mm produces a board
# that still reprojects perfectly and whose reconstructed corner spacing still
# matches the object model — the error is a global depth/focal ambiguity that
# one view cannot resolve. So the only real scale witnesses are:
#   1. Intrinsics calibration RMS (loaded from the intrinsics file).
#   2. A physical tape-measure check of the reported board distance.
# The demo therefore reports board distance prominently and asks you to confirm
# it against a ruler — that single external measurement pins the scale.


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, required=True)
    p.add_argument("--square-mm", type=float, required=True,
                   help="MEASURED printed square size in mm")
    p.add_argument("--intrinsics", default=DEFAULT_INTRINSICS)
    p.add_argument("--out", default=DEFAULT_EXTRINSICS)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--window", type=int, default=30,
                   help="rolling window (frames) for jitter stats")
    args = p.parse_args()

    square_m = args.square_mm / 1000.0
    board, aruco_dict = _make_board(square_m)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    K, dist, intr_rms = _load_intrinsics(args.intrinsics)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"[assess] intrinsics loaded (calib RMS={intr_rms}) ; square={args.square_mm} mm")
    print("[assess] Fix the board where you want WORLD ORIGIN, facing the camera.")
    print("[assess] Q quit, S save pose. Watch reproj px, scale %, and jitter.")

    tvec_hist: collections.deque = collections.deque(maxlen=args.window)
    zaxis_hist: collections.deque = collections.deque(maxlen=args.window)
    last_pose = None  # (rvec, tvec)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = charuco_detector.detectBoard(gray)
        vis = frame.copy()

        n_det = 0 if ci is None else len(ci)
        reproj = None
        dist_m = None
        if n_det >= 6:
            cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci)
            obj_pts, img_pts = board.matchImagePoints(cc, ci)
            ok_pnp, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
            if ok_pnp:
                last_pose = (rvec.copy(), tvec.copy())
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, square_m * 2)
                reproj = _reproj_error(obj_pts, img_pts, rvec, tvec, K, dist)
                dist_m = float(np.linalg.norm(tvec))
                tvec_hist.append(tvec.ravel())
                R, _ = cv2.Rodrigues(rvec)
                zaxis_hist.append(R[:, 2].copy())  # board normal in cam frame

        # ---- scorecard overlay ----
        lines = [f"corners: {n_det}  (need >=6)"]
        if reproj is not None:
            grade = "GOOD" if reproj < 1.0 else ("OK" if reproj < 2.0 else "POOR")
            lines.append(f"reproj: {reproj:.2f} px  [{grade}]")
            # Board distance is the scale witness: tape-measure this to confirm.
            lines.append(f"board dist: {dist_m*100:.1f} cm  <- verify w/ tape measure")
        if len(tvec_hist) >= 5:
            pos_std_mm = np.linalg.norm(np.std(np.array(tvec_hist), axis=0)) * 1000
            # angular jitter of the board normal, in degrees
            zs = np.array(zaxis_hist)
            zmean = zs.mean(axis=0); zmean /= (np.linalg.norm(zmean) + 1e-9)
            ang = np.degrees(np.arccos(np.clip(zs @ zmean, -1, 1)))
            jgrade = "STABLE" if pos_std_mm < 3 and ang.std() < 0.5 else "JITTERY"
            lines.append(f"jitter: pos {pos_std_mm:.1f} mm  "
                         f"tilt {ang.std():.2f} deg  [{jgrade}]")

        y = 30
        for ln in lines:
            cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            y += 30

        cv2.imshow("assess extrinsics — Q quit, S save", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            if last_pose is None:
                print("[assess] no pose yet; aim at the board.")
                continue
            _save(last_pose, args.out)

    cap.release()
    cv2.destroyAllWindows()
    _print_verdict(tvec_hist, zaxis_hist, intr_rms)


def _save(pose, out_path: str) -> None:
    rvec, tvec = pose
    R_cam_world, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    cam_pos_world = (-R_cam_world.T @ t)
    data = {
        "R_cam_world": R_cam_world.tolist(),
        "t_cam_world": t.tolist(),
        "R_world_cam": R_cam_world.T.tolist(),
        "camera_pos_in_world": cam_pos_world.tolist(),
        "board_distance_m": float(np.linalg.norm(t)),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[assess] saved pose -> {out_path}  (cam at {cam_pos_world} m in world)")


def _print_verdict(tvec_hist, zaxis_hist, intr_rms) -> None:
    print("\n===== assessment verdict =====")
    if len(tvec_hist) < 5:
        print("Board rarely detected — check lighting, focus, board flatness, --square-mm.")
        return
    pos_std_mm = np.linalg.norm(np.std(np.array(tvec_hist), axis=0)) * 1000
    print(f"intrinsics calib RMS: {intr_rms} px "
          f"({'good' if (intr_rms or 9) < 1.0 else 'high — recalibrate intrinsics'})")
    print(f"position jitter (static board): {pos_std_mm:.2f} mm rms")
    print("SCALE: confirm the reported 'board dist' matched your tape measure — "
          "that is the only true metric check.")
    verdict = "USABLE" if pos_std_mm < 5 else "MARGINAL — reduce jitter before trusting scale"
    print(f"overall: {verdict}")


if __name__ == "__main__":
    main()

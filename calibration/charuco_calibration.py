"""ChArUco camera calibration for the DexPilot teleop pipeline.

One printed ChArUco board does the whole job:
  1. `generate`   — render a letter-size board PNG to print at 100 % scale.
  2. `intrinsics` — wave the board around to solve the camera matrix + distortion.
  3. `extrinsics` — fix the board where you want world origin; solve T_cam_world.

The extrinsics step outputs the measured `R_cam_world` and metric scale that
replace the hardcoded `_DEFAULT_R_CAM_ROBOT` / `scale_x` / `scale_z` guesses in
teleop/dexpilot_arm_controller.py.

Requires OpenCV 4.7+ (uses the modern cv2.aruco CharucoBoard/Detector API);
the project's camera env ships opencv-contrib-python==4.11.

Typical flow (fixed camera):
    python calibration/charuco_calibration.py generate
    # print board.png at 100%, glue flat, MEASURE a square with calipers
    python calibration/charuco_calibration.py intrinsics --camera 1 --square-mm 35.0
    # fix board at desired world origin, facing camera
    python calibration/charuco_calibration.py extrinsics --camera 1 --square-mm 35.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Board definition — shared by all three subcommands so IDs/geometry match.
# 5x7 squares at 35 mm fits letter (8.5x11") with margin and stays resolvable
# at ~1 m. Marker length is 0.75 of the square (standard ChArUco ratio).
# ----------------------------------------------------------------------------
SQUARES_X = 5
SQUARES_Y = 7
DEFAULT_SQUARE_MM = 35.0
MARKER_RATIO = 0.75
ARUCO_DICT = cv2.aruco.DICT_5X5_100

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BOARD_PNG = os.path.join(_HERE, "board.png")
DEFAULT_INTRINSICS = os.path.join(_HERE, "camera_intrinsics.json")
DEFAULT_EXTRINSICS = os.path.join(_HERE, "camera_extrinsics.json")


def _named(default_path: str, name: str | None) -> str:
    """Insert a camera name before the extension for per-camera calibration files.

    Multi-camera triangulation (teleop/hand_fusion_node.py) loads one calibration
    per camera keyed by name: camera_intrinsics_<name>.json /
    camera_extrinsics_<name>.json. Passing --name to intrinsics/extrinsics routes
    output there instead of the shared single-camera file, so each camera in the
    rig gets its own calibration WITHOUT overwriting the others. All cameras
    solving extrinsics against the SAME fixed board land in ONE shared world frame
    (no separate stereo calibration needed). No --name -> unchanged single-cam file.
    """
    if not name:
        return default_path
    root, ext = os.path.splitext(default_path)
    return f"{root}_{name}{ext}"


def _parse_cam_spec(spec: str) -> tuple[str, int]:
    """Parse a --cam <name>:<index> spec for extrinsics-all (same format as
    teleop/run_multicam.py --cam, minus the optional resolution)."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--cam expects <name>:<index>, got {spec!r}")
    name, idx = spec.rsplit(":", 1)
    if not name:
        raise argparse.ArgumentTypeError(f"empty camera name in {spec!r}")
    try:
        return name, int(idx)
    except ValueError:
        raise argparse.ArgumentTypeError(f"index must be an int in {spec!r}")


def _make_board(square_len_m: float) -> tuple:
    """Build the ChArUco board and matching dictionary at a given square size.

    square_len_m matters only for metric outputs (intrinsics are scale-free,
    but pose/extrinsics are not). Pass the *measured* printed size, not nominal.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        squareLength=square_len_m,
        markerLength=square_len_m * MARKER_RATIO,
        dictionary=aruco_dict,
    )
    return board, aruco_dict


# ----------------------------------------------------------------------------
# generate
# ----------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> None:
    # Geometry is scale-free for the printed image; use nominal mm for sizing.
    square_m = args.square_mm / 1000.0
    board, _ = _make_board(square_m)

    # Render at print DPI so the PNG has a well-defined physical size.
    # pixels = inches * dpi ; 1 square = square_mm/25.4 inches.
    px_per_square = round((args.square_mm / 25.4) * args.dpi)
    img_w = px_per_square * SQUARES_X
    img_h = px_per_square * SQUARES_Y
    # Quiet-zone margin (10 % of a square) so the border markers detect cleanly.
    margin = round(px_per_square * 0.1)

    board_img = board.generateImage((img_w, img_h), marginSize=margin, borderBits=1)
    cv2.imwrite(args.out, board_img)

    total_w_mm = SQUARES_X * args.square_mm + 2 * (margin / args.dpi * 25.4)
    total_h_mm = SQUARES_Y * args.square_mm + 2 * (margin / args.dpi * 25.4)
    print(f"[generate] wrote {args.out}")
    print(f"[generate] {SQUARES_X}x{SQUARES_Y} board, nominal square={args.square_mm} mm, {args.dpi} DPI")
    print(f"[generate] printed size ~= {total_w_mm:.1f} x {total_h_mm:.1f} mm "
          f"({total_w_mm/25.4:.2f} x {total_h_mm/25.4:.2f} in)")
    print("[generate] PRINT AT 100% / ACTUAL SIZE (no 'fit to page').")
    print("[generate] Then MEASURE one black square edge with a ruler/calipers")
    print("[generate] and pass that value as --square-mm to the next steps.")
    print("[generate] Glue the print to something flat (foam board / clipboard).")


# ----------------------------------------------------------------------------
# shared capture helper
# ----------------------------------------------------------------------------
def _request_max_res(cap: cv2.VideoCapture) -> None:
    """Set the camera to its highest supported resolution (probe + readback).

    Mirrors hand_landmark_node._request_max_res so calibration happens at the
    SAME size the pipeline captures at under --max-res. UVC cameras clamp set()
    to a supported mode; MJPG first unlocks high-res modes some hide under raw."""
    ladder = [(3840, 2160), (2592, 1944), (2560, 1440), (1920, 1080),
              (1600, 1200), (1280, 960), (1280, 720), (1024, 768),
              (960, 720), (640, 480)]
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    best = (0, 0)
    for (w, h) in ladder:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if aw * ah > best[0] * best[1]:
            best = (aw, ah)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best[1])
    print(f"[camera] max-res -> {best[0]}x{best[1]}")


def _open_camera(index: int, width: int, height: int,
                 max_res: bool = False) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open camera index {index}. "
                 f"Check the index used by ui/mediapipe_joint_angles.py --list-cameras.")
    if max_res:
        _request_max_res(cap)
    else:
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] index {index} opened at {aw}x{ah} — intrinsics will be "
          f"valid ONLY at this size.")
    return cap


def _make_detectors(board, aruco_dict):
    """Create ArUco + ChArUco detectors (OpenCV 4.7+ API)."""
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    return aruco_detector, charuco_detector


def _make_window(name: str) -> None:
    """Create the display window up-front and force it in front of other apps.

    Without this the imshow window can open BEHIND the editor/terminal, so you
    never see the feed and (worse) end up typing SPACE/C/Q into the terminal
    where they do nothing — the keys only register when this window has focus.
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1280, 720)
    cv2.moveWindow(name, 60, 60)
    # WND_PROP_TOPMOST raises it above other windows on WMs that honor it.
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1.0)
    except Exception:
        pass  # older/limited builds — window still opens, just not forced top
    print(f"[window] opened '{name}'. If you don't see it, Alt-Tab to it and "
          f"CLICK it so it has focus before pressing keys.")


# ----------------------------------------------------------------------------
# intrinsics
# ----------------------------------------------------------------------------
def cmd_intrinsics(args: argparse.Namespace) -> None:
    square_m = args.square_mm / 1000.0
    board, aruco_dict = _make_board(square_m)
    _, charuco_detector = _make_detectors(board, aruco_dict)

    cap = _open_camera(args.camera, args.width, args.height, args.max_res)
    win = "intrinsics — SPACE capture, C calibrate, Q quit"
    _make_window(win)
    print("[intrinsics] SPACE = capture a view, C = calibrate, Q = quit.")
    print(f"[intrinsics] Collect {args.min_views}+ views: vary angle, tilt, distance.")
    print("[intrinsics] Fill the frame edges/corners across the set for good distortion.")

    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[intrinsics] frame grab failed; retrying.")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (w, h)

        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        vis = frame.copy()
        n_det = 0 if charuco_ids is None else len(charuco_ids)
        if n_det > 0:
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
        cv2.putText(vis, f"views: {len(all_corners)}  corners now: {n_det}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            # Need a decent number of corners for a useful view.
            if n_det >= 6:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                print(f"[intrinsics] captured view {len(all_corners)} ({n_det} corners)")
            else:
                print(f"[intrinsics] too few corners ({n_det}); reposition board.")
        if key == ord('c'):
            if len(all_corners) < args.min_views:
                print(f"[intrinsics] need >= {args.min_views} views, have {len(all_corners)}.")
                continue
            _run_intrinsics_calib(board, all_corners, all_ids, image_size, args.out)
            break

    cap.release()
    cv2.destroyAllWindows()


def _run_intrinsics_calib(board, all_corners, all_ids, image_size, out_path) -> None:
    # Build object/image point correspondences per view from the board model,
    # then calibrate with the generic cv2.calibrateCamera (stable across the
    # 4.x aruco API churn around calibrateCameraCharuco).
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 6:
            continue
        obj_points.append(obj_pts)
        img_points.append(img_pts)

    if len(obj_points) < 3:
        sys.exit("[intrinsics] not enough valid views after matching; recapture.")

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)

    print(f"[intrinsics] RMS reprojection error: {rms:.4f} px "
          f"({'good' if rms < 1.0 else 'high — recapture with more/varied views'})")
    print(f"[intrinsics] camera_matrix=\n{camera_matrix}")
    print(f"[intrinsics] dist_coeffs={dist_coeffs.ravel()}")

    data = {
        "image_size": list(image_size),
        "rms_reproj_px": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[intrinsics] wrote {out_path}")


# ----------------------------------------------------------------------------
# extrinsics
# ----------------------------------------------------------------------------
def _solve_one_extrinsic(camera: int, intrinsics_path: str, out_path: str,
                         square_mm: float, n_avg: int, width: int, height: int,
                         max_res: bool, title: str = "") -> bool:
    """Interactive extrinsic solve for ONE camera against the fixed board.

    Opens the camera, shows the live board detection + world axes, and on SPACE
    averages n_avg frames, solves the pose, saves to out_path, and returns True.
    Q (or window close) skips this camera and returns False. Shared by the single
    'extrinsics' command and the 'extrinsics-all' walkthrough.
    """
    if not os.path.exists(intrinsics_path):
        sys.exit(f"ERROR: intrinsics file not found: {intrinsics_path}. "
                 f"Run the 'intrinsics' subcommand first.")
    with open(intrinsics_path) as f:
        intr = json.load(f)
    camera_matrix = np.array(intr["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(intr["dist_coeffs"], dtype=np.float64)

    square_m = square_mm / 1000.0
    board, aruco_dict = _make_board(square_m)
    _, charuco_detector = _make_detectors(board, aruco_dict)

    cap = _open_camera(camera, width, height, max_res)
    win = f"extrinsics {title} — SPACE solve, S skip, Q quit".strip()
    _make_window(win)
    print(f"[extrinsics] {title} Fix the board at the WORLD ORIGIN, facing the camera.")
    print(f"[extrinsics] SPACE = average {n_avg} frames & solve, S = skip this "
          f"camera, Q = quit all.")

    solved = False
    quit_all = False
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        vis = frame.copy()
        n_det = 0 if charuco_ids is None else len(charuco_ids)
        if n_det >= 6:
            obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
            ok_pnp, rvec, tvec = cv2.solvePnP(obj_pts, img_pts,
                                              camera_matrix, dist_coeffs)
            if ok_pnp:
                cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs,
                                  rvec, tvec, square_m * 2)
        label = f"{title}  corners: {n_det}  (need >=6; SPACE solve, S skip, Q quit)"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit_all = True
            break
        if key == ord('s'):
            print(f"[extrinsics] {title} skipped.")
            break
        if key == ord(' '):
            pose = _average_pose(cap, charuco_detector, board,
                                 camera_matrix, dist_coeffs, n_avg)
            if pose is None:
                print("[extrinsics] board not stably detected; hold steady and retry.")
                continue
            _report_and_save_extrinsics(pose, out_path)
            solved = True
            break

    cap.release()
    cv2.destroyWindow(win)
    if quit_all:
        raise KeyboardInterrupt   # signal the caller to stop the whole walkthrough
    return solved


def cmd_extrinsics(args: argparse.Namespace) -> None:
    _solve_one_extrinsic(args.camera, args.intrinsics, args.out,
                         args.square_mm, args.n_avg, args.width, args.height,
                         args.max_res)


def cmd_extrinsics_all(args: argparse.Namespace) -> None:
    """Calibrate the extrinsics of EVERY camera in ONE command.

    Walks through each --cam <name>:<index> in turn against the SAME fixed board:
    the board must stay put for the whole run so all cameras share one world
    frame. For each camera it reads camera_intrinsics_<name>.json and writes
    camera_extrinsics_<name>.json. Press SPACE to solve a camera, S to skip it, Q
    to stop the walkthrough.
    """
    cams = args.cam
    print(f"[extrinsics-all] {len(cams)} cameras: "
          f"{', '.join(f'{n}:{i}' for n, i in cams)}")
    print("[extrinsics-all] KEEP THE BOARD FIXED at the world origin for the WHOLE "
          "run — that shared pose is what puts all cameras in one frame.")
    done, skipped = [], []
    try:
        for k, (name, idx) in enumerate(cams, 1):
            intr = _named(DEFAULT_INTRINSICS, name)
            out = _named(DEFAULT_EXTRINSICS, name)
            title = f"[{k}/{len(cams)}] {name}"
            print(f"\n[extrinsics-all] ==== {title} (camera index {idx}) ====")
            ok = _solve_one_extrinsic(idx, intr, out, args.square_mm, args.n_avg,
                                      args.width, args.height, args.max_res, title)
            (done if ok else skipped).append(name)
    except KeyboardInterrupt:
        print("\n[extrinsics-all] stopped early (Q).")
    cv2.destroyAllWindows()
    print(f"\n[extrinsics-all] done: {done or '(none)'}"
          f"   skipped: {skipped or '(none)'}")
    if done:
        print("[extrinsics-all] All solved cameras now share ONE world frame "
              "(same board pose). Validate with the fusion node's reproj log.")


def _average_pose(cap, charuco_detector, board, camera_matrix, dist_coeffs, n_avg):
    """Average board pose over n_avg frames to reduce single-frame PnP jitter.

    Returns (R_cam_board 3x3, t_cam_board 3) or None if too few solves.
    Translation is averaged directly; rotation via mean of rvecs (small-angle
    spread over a few static frames makes this adequate).
    """
    tvecs: list[np.ndarray] = []
    rvecs: list[np.ndarray] = []
    grabbed = 0
    while grabbed < n_avg:
        ok, frame = cap.read()
        if not ok:
            continue
        grabbed += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = charuco_detector.detectBoard(gray)
        if ci is None or len(ci) < 6:
            continue
        obj_pts, img_pts = board.matchImagePoints(cc, ci)
        ok_pnp, rvec, tvec = cv2.solvePnP(obj_pts, img_pts,
                                          camera_matrix, dist_coeffs)
        if ok_pnp:
            rvecs.append(rvec.ravel())
            tvecs.append(tvec.ravel())

    if len(tvecs) < max(3, n_avg // 3):
        return None
    t_mean = np.mean(tvecs, axis=0)
    r_mean = np.mean(rvecs, axis=0)
    R_mean, _ = cv2.Rodrigues(r_mean)
    print(f"[extrinsics] averaged {len(tvecs)}/{n_avg} valid frames; "
          f"tvec std = {np.std(tvecs, axis=0)} m")
    return R_mean, t_mean


def _report_and_save_extrinsics(pose, out_path) -> None:
    R_cam_board, t_cam_board = pose

    # solvePnP gives the board pose IN the camera frame: a board point p_b maps
    # to camera coords via p_c = R_cam_board @ p_b + t_cam_board. The board frame
    # IS our chosen world origin, so R_cam_world = R_cam_board, t_cam_world = t.
    # The camera position expressed in world coords is the usual inverse:
    #   p_cam_in_world = -R_cam_board^T @ t_cam_board
    cam_pos_world = (-R_cam_board.T @ t_cam_board)
    dist = float(np.linalg.norm(t_cam_board))

    data = {
        "R_cam_world": R_cam_board.tolist(),          # world axes -> camera axes
        "t_cam_world": t_cam_board.tolist(),          # world origin in camera frame [m]
        "R_world_cam": R_cam_board.T.tolist(),        # camera axes -> world axes
        "camera_pos_in_world": cam_pos_world.tolist(),
        "board_distance_m": dist,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[extrinsics] board (world origin) is {dist:.3f} m from the camera")
    print(f"[extrinsics] camera position in world frame [m]: {cam_pos_world}")
    print(f"[extrinsics] R_cam_world (maps world vectors -> camera vectors):")
    print(np.array_str(R_cam_board, precision=4, suppress_small=True))
    print(f"[extrinsics] wrote {out_path}")
    print()
    print("[extrinsics] NEXT: R_world_cam maps MediaPipe-camera vectors into your")
    print("[extrinsics] chosen world frame. Compose it with your world->robot")
    print("[extrinsics] convention to replace _DEFAULT_R_CAM_ROBOT and the")
    print("[extrinsics] scale_x/scale_z pixel gains in dexpilot_arm_controller.py.")


# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="render a printable ChArUco board PNG")
    g.add_argument("--out", default=DEFAULT_BOARD_PNG)
    g.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                   help="nominal printed square size for sizing the image")
    g.add_argument("--dpi", type=int, default=300)
    g.set_defaults(func=cmd_generate)

    i = sub.add_parser("intrinsics", help="live camera-matrix + distortion calibration")
    i.add_argument("--camera", type=int, required=True)
    i.add_argument("--name", default=None,
                   help="Per-camera name for multi-camera rigs. Routes output to "
                        "camera_intrinsics_<name>.json (matches the fusion node "
                        "and hand_landmark_node --name). Omit for single-camera.")
    i.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                   help="MEASURED printed square size in mm")
    i.add_argument("--out", default=None,
                   help="Explicit output path (overrides --name).")
    i.add_argument("--min-views", type=int, default=12)
    i.add_argument("--width", type=int, default=1280)
    i.add_argument("--height", type=int, default=720)
    i.add_argument("--max-res", action="store_true",
                   help="Calibrate at the camera's HIGHEST supported resolution "
                        "(matches run_multicam --max-res). Overrides --width/height. "
                        "The camera MUST then run the pipeline at this same size.")
    i.set_defaults(func=cmd_intrinsics)

    e = sub.add_parser("extrinsics", help="solve fixed-board camera->world pose")
    e.add_argument("--camera", type=int, required=True)
    e.add_argument("--name", default=None,
                   help="Per-camera name for multi-camera rigs. Reads "
                        "camera_intrinsics_<name>.json and writes "
                        "camera_extrinsics_<name>.json. Point EVERY camera at the "
                        "SAME fixed board so all extrinsics share one world frame. "
                        "Omit for single-camera.")
    e.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                   help="MEASURED printed square size in mm")
    e.add_argument("--intrinsics", default=None,
                   help="Explicit intrinsics path (overrides --name).")
    e.add_argument("--out", default=None,
                   help="Explicit output path (overrides --name).")
    e.add_argument("--n-avg", type=int, default=30)
    e.add_argument("--width", type=int, default=1280)
    e.add_argument("--height", type=int, default=720)
    e.add_argument("--max-res", action="store_true",
                   help="Open the camera at its HIGHEST supported resolution. Use "
                        "the SAME setting as the intrinsics step for this camera.")
    e.set_defaults(func=cmd_extrinsics)

    ea = sub.add_parser(
        "extrinsics-all",
        help="calibrate EVERY camera's extrinsics in one walkthrough (one board)")
    ea.add_argument("--cam", action="append", type=_parse_cam_spec, required=True,
                    metavar="NAME:INDEX",
                    help="Camera name:OpenCV-index. Repeat per camera, e.g. "
                         "--cam c0:0 --cam c1:2 --cam rs:4. Uses each camera's "
                         "camera_intrinsics_<name>.json; writes "
                         "camera_extrinsics_<name>.json. Keep the board FIXED.")
    ea.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                    help="MEASURED printed square size in mm")
    ea.add_argument("--n-avg", type=int, default=30)
    ea.add_argument("--width", type=int, default=1280)
    ea.add_argument("--height", type=int, default=720)
    ea.add_argument("--max-res", action="store_true",
                    help="Open each camera at its HIGHEST supported resolution "
                         "(match how you ran intrinsics for these cameras).")
    ea.set_defaults(func=cmd_extrinsics_all)

    args = p.parse_args()
    # Resolve per-camera paths from --name (explicit --out/--intrinsics win).
    if args.cmd == "intrinsics" and args.out is None:
        args.out = _named(DEFAULT_INTRINSICS, args.name)
    if args.cmd == "extrinsics":
        if args.intrinsics is None:
            args.intrinsics = _named(DEFAULT_INTRINSICS, args.name)
        if args.out is None:
            args.out = _named(DEFAULT_EXTRINSICS, args.name)
    args.func(args)


if __name__ == "__main__":
    main()

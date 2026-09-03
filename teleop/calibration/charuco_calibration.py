"""ChArUco camera calibration for the DexPilot teleop pipeline.

One printed ChArUco board does the whole job:
  1. `generate`   — render an A3-size board PNG to print at 100 % scale.
  2. `intrinsics` — wave the board around to solve the camera matrix + distortion.
  3. `extrinsics` — fix the board where you want world origin; solve T_cam_world.

The extrinsics step outputs the measured `R_cam_world` and metric scale that
replace the hardcoded `_DEFAULT_R_CAM_ROBOT` / `scale_x` / `scale_z` guesses in
teleop/dexpilot_arm_controller.py.

Requires OpenCV 4.7+ (uses the modern cv2.aruco CharucoBoard/Detector API);
the project's camera env ships opencv-contrib-python==4.11.

Typical flow (fixed camera):
    python calibration/charuco_calibration.py generate
    # or: generate --paper letter (smaller squares, same 5x7 grid, fits a normal printer)
    # print board.png at 100%, glue flat, MEASURE a square with calipers
    python calibration/charuco_calibration.py intrinsics --camera 1 --square-mm 50.0
    # fix board at desired world origin, facing camera
    python calibration/charuco_calibration.py extrinsics --camera 1 --square-mm 50.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Board definition — shared by all three subcommands so IDs/geometry match.
# 5x7 squares at 50 mm (current printed board) stays resolvable at ~1 m.
# Marker length is 0.75 of the square (standard ChArUco ratio).
# ----------------------------------------------------------------------------
SQUARES_X = 5
SQUARES_Y = 7
DEFAULT_SQUARE_MM = 50.0
MARKER_RATIO = 0.75
ARUCO_DICT = cv2.aruco.DICT_5X5_100

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BOARD_PNG = os.path.join(_HERE, "board.png")
DEFAULT_INTRINSICS = os.path.join(_HERE, "camera_intrinsics.json")
DEFAULT_EXTRINSICS = os.path.join(_HERE, "camera_extrinsics.json")

# Paper sizes (mm, portrait) for `generate --paper`. --square-mm still wins if
# passed explicitly; otherwise the square size is the largest that fits the
# SQUARES_X x SQUARES_Y grid on this sheet with PAPER_MARGIN_MM to spare.
PAPER_SIZES_MM = {
    "letter": (215.9, 279.4),
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
}
PAPER_MARGIN_MM = 10.0

# Extrinsic-solve stability: reject an averaged board pose whose per-frame
# translation scatter exceeds this — a jiggly board (hand-held, vibrating mount)
# bakes a noisy pose into the extrinsic. ~3 mm passes a genuinely-fixed board but
# flags a shaking one. Overridable via --pose-std-mm.
_POSE_STD_TOL_M = 0.003

# cornerSubPix refinement for ChArUco corners — raw detector corners jitter
# frame-to-frame; sub-pixel refinement is the single biggest reduction in that
# jitter (and in the drawn-axis wobble you see during the solve).
_SUBPIX_WIN = (5, 5)
_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


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
    if args.out is None:
        args.out = (DEFAULT_BOARD_PNG if not args.paper or args.paper == "a3"
                    else os.path.join(_HERE, f"board_{args.paper}.png"))

    square_mm = args.square_mm
    if args.paper and square_mm is None:
        paper_w, paper_h = PAPER_SIZES_MM[args.paper]
        usable_w = paper_w - 2 * PAPER_MARGIN_MM
        usable_h = paper_h - 2 * PAPER_MARGIN_MM
        square_mm = min(usable_w / SQUARES_X, usable_h / SQUARES_Y)
        print(f"[generate] --paper {args.paper}: sizing square to "
              f"{square_mm:.2f} mm to fill {paper_w:.0f}x{paper_h:.0f} mm sheet "
              f"({PAPER_MARGIN_MM:.0f} mm margin).")
    elif square_mm is None:
        square_mm = DEFAULT_SQUARE_MM
    args.square_mm = square_mm

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


class _RealSenseCapture:
    """cv2.VideoCapture-compatible shim over a RealSense COLOR stream.

    MUST match how the teleop pipeline captures the RealSense
    (teleop/hand_landmark_node.py::_RealSenseCapture): pyrealsense2 color at a
    fixed WxH in bgr8. Calibrating extrinsics through bare cv2.VideoCapture would
    open a DIFFERENT stream (or fail — the D435I color isn't reliably openable
    via V4L2), so the resulting pose would not match the pixels fusion receives.
    Exposes just the read()/get()/release()/isOpened() surface _open_camera's
    callers use.
    """

    def __init__(self, width: int, height: int, fps: int | None = None):
        import pyrealsense2 as rs
        self._rs = rs
        if fps is None:
            fps = self._best_fps(rs, width, height)
            if fps is None:
                sys.exit(f"ERROR: RealSense has no bgr8 color mode at "
                         f"{width}x{height}. Use a supported size (e.g. 640x480, "
                         f"1280x720) and calibrate intrinsics at that same size.")
        self._w, self._h = width, height
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipe = rs.pipeline()
        try:
            self._pipe.start(cfg)
        except Exception as e:
            sys.exit(f"ERROR: could not start RealSense color at "
                     f"{width}x{height}@{fps} ({e}). Is the device held by "
                     f"another process? (check: fuser /dev/video*)")
        self._opened = True
        print(f"[camera] RealSense color {width}x{height}@{fps} bgr8 started — "
              f"extrinsics will be valid ONLY at this size.")

    @staticmethod
    def _best_fps(rs, width: int, height: int):
        best = None
        try:
            for d in rs.context().query_devices():
                for s in d.query_sensors():
                    for p in s.get_stream_profiles():
                        if p.stream_type() != rs.stream.color:
                            continue
                        if p.format() != rs.format.bgr8:
                            continue
                        vp = p.as_video_stream_profile()
                        if vp.width() == width and vp.height() == height:
                            best = max(best or 0, p.fps())
        except Exception:
            return None
        return best

    def isOpened(self):
        return self._opened

    def read(self):
        try:
            frames = self._pipe.wait_for_frames(2000)
            color = frames.get_color_frame()
            if not color:
                return False, None
            return True, np.asanyarray(color.get_data())
        except Exception:
            return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        return 0.0

    def set(self, *_a, **_k):
        return True   # resolution fixed at construction

    def release(self):
        if self._opened:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._opened = False


def _open_camera(index: int, width: int, height: int,
                 max_res: bool = False, realsense: bool = False):
    if realsense:
        # RealSense: pyrealsense2 color at a FIXED size (max_res N/A — RS 1080p
        # color is 8 fps). Default 640x480 to match the pipeline unless the
        # caller overrode width/height. --camera index is ignored (SDK picks it).
        return _RealSenseCapture(width or 640, height or 480, fps=None)
    # Force the V4L2 backend explicitly (Linux) — the default lets OpenCV
    # cascade through other backends (e.g. its bundled Orbbec/obsensor probe)
    # after V4L2 fails, which is slow (extra ioctl round-trips) and prints a
    # spurious "obsensor ... Camera index out of range" error unrelated to
    # this rig (no Orbbec hardware here).
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open camera index {index}. "
                 f"Check the index used by teleop/ui.py --list-cameras.")
    if max_res:
        _request_max_res(cap)
    else:
        # Force MJPG BEFORE requesting the size. These UVC webcams only expose
        # their high-res modes (e.g. 1920x1080) under MJPG; the default YUYV
        # format tops out at 640x480, so a bare 1920x1080 request silently
        # CLAMPS to 640x480 YUYV — the exact trap that produced the squashed
        # c1/c2 intrinsics. (_request_max_res already does this; the fixed-size
        # path was missing it.)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Refuse to proceed if the camera substituted a different size than asked
    # for. Calibrating at a silently-clamped resolution bakes a mis-scaled K
    # (and a wrong image_size) into the file — invisible until every extrinsic
    # distance solved against it comes out wrong. max_res has no target to
    # check against (it takes whatever the ladder tops out at); realsense is
    # fixed at construction and never reaches here.
    if not max_res and width and height and (aw, ah) != (width, height):
        cap.release()
        sys.exit(
            f"ERROR: camera index {index} would not open at the requested "
            f"{width}x{height} — it clamped to {aw}x{ah}. That mode is not "
            f"available on this camera/pixel-format (check "
            f"v4l2-ctl --list-formats-ext). Calibrating at the clamped size "
            f"would bake a mis-scaled camera matrix into the file. Pass a "
            f"--width/--height the camera actually supports, or --max-res.")
    print(f"[camera] index {index} opened at {aw}x{ah} — intrinsics will be "
          f"valid ONLY at this size.")
    return cap


def _make_detectors(board, aruco_dict):
    """Create ArUco + ChArUco detectors (OpenCV 4.7+ API).

    Tuned (vs. cv2 defaults) for markers that are small in-frame or seen at an
    oblique angle — e.g. a board held far/tilted enough back to stay in view of
    several cameras at once. minMarkerPerimeterRate and polygonalApproxAccuracyRate
    are the two expensive ones: loosening either lets far more candidate shapes
    survive the cheap initial filter and reach the costly per-candidate decode
    step, so both are nudged rather than maxed out (~2x detectBoard() cost for
    a small additional nudge is not worth it; that call runs every preview frame
    in the live capture loop, so overly aggressive values here show up as visible
    lag, not just slower calibration solves):
      - minMarkerPerimeterRate 0.03 -> 0.02: don't discard small/far markers.
      - adaptiveThreshWinSize 3..23 step 10 -> 3..35 step 8: wider multi-scale
        search catches marker edges the coarse default steps skip over, without
        scanning as many levels as a finer step would.
      - perspectiveRemovePixelPerCell 4 -> 8: upsample more when unwarping each
        marker cell before bit-sampling, which helps decode a marker that only
        covers a small patch of the source image. Cheap — no per-candidate-count
        blowup, just a bigger fixed-size warp per candidate.
      - errorCorrectionRate 0.6 -> 0.8: tolerate more bit errors when matching
        a noisy/small marker against the dictionary. Cheap.
      - minOtsuStdDev 5.0 -> 3.0: don't reject low-contrast (small/far, or dim)
        marker regions during binarization. Cheap.
      - polygonalApproxAccuracyRate 0.03 -> 0.08: accept a less-precisely-square
        contour as a marker candidate — a marker seen at a grazing/oblique angle
        (camera close to the board's own plane) projects to a heavily
        foreshortened quad, and JPEG/blur noise on its edges makes the strict
        default reject it before bit-decoding is ever attempted. Cheap — unlike
        minMarkerPerimeterRate this doesn't blow up candidate count in practice
        (verified: same detectBoard() cost as the stricter 0.045 on real rig
        frames), so it's fine to lean on this one harder than the perimeter rate.
      - maxErroneousBitsInBorderRate 0.35 -> 0.5: tolerate noisier border bits,
        which perspective interpolation introduces more of at oblique angles.
        Cheap.
    CharucoParameters.minMarkers 2 -> 1: interpolate a ChArUco corner from just
    ONE neighboring detected marker instead of requiring two — at range or at
    an oblique angle, often only one of a corner's neighboring markers decodes.
    """
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.minMarkerPerimeterRate = 0.02
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 35
    aruco_params.adaptiveThreshWinSizeStep = 8
    aruco_params.perspectiveRemovePixelPerCell = 8
    aruco_params.errorCorrectionRate = 0.8
    aruco_params.minOtsuStdDev = 3.0
    aruco_params.polygonalApproxAccuracyRate = 0.08
    aruco_params.maxErroneousBitsInBorderRate = 0.5
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.minMarkers = 1
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, aruco_params)
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
def _intrinsics_capture_loop(camera: int, out_path: str, square_mm: float,
                             min_views: int, width: int, height: int,
                             max_res: bool, realsense: bool, fix_k3: bool,
                             title: str = "") -> bool:
    """Interactive intrinsics capture+solve for ONE camera. Returns True if a
    calibration was written, False if the user quit/skipped. Shared by the single
    `intrinsics` command and the `intrinsics-all` auto-discovery walkthrough.

    Q quits the WHOLE run (raises KeyboardInterrupt); S skips this camera.
    """
    square_m = square_mm / 1000.0
    board, aruco_dict = _make_board(square_m)
    _, charuco_detector = _make_detectors(board, aruco_dict)

    _w, _h, _mx = width, height, max_res
    if realsense:
        # RealSense capture is the pyrealsense2 stream size; 1080p color is 8 fps.
        # Fall back to the pipeline's 640x480 unless overridden. --max-res N/A.
        if (_w, _h) == (1280, 720):
            _w, _h = 640, 480
        _mx = False
    cap = _open_camera(camera, _w, _h, _mx, realsense)
    win = f"intrinsics {title} — SPACE capture, C solve, S skip, Q quit".strip()
    _make_window(win)
    print(f"[intrinsics] {title} SPACE = capture a view, C = calibrate, "
          f"S = skip, Q = quit.")
    print(f"[intrinsics] Collect {min_views}+ views: vary angle, tilt, distance; "
          f"fill the frame EDGES/CORNERS for good distortion.")

    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    solved = False
    quit_all = False

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
        cv2.putText(vis, f"{title}  views: {len(all_corners)}  corners: {n_det}"
                    "  (SPACE cap, C solve, S skip, Q quit)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit_all = True
            break
        if key == ord('s'):
            print(f"[intrinsics] {title} skipped.")
            break
        if key == ord(' '):
            if n_det >= 6:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                print(f"[intrinsics] captured view {len(all_corners)} ({n_det} corners)")
            else:
                print(f"[intrinsics] too few corners ({n_det}); reposition board.")
        if key == ord('c'):
            if len(all_corners) < min_views:
                print(f"[intrinsics] need >= {min_views} views, have {len(all_corners)}.")
                continue
            hw_id = _camera_hardware_id(camera, realsense)
            _run_intrinsics_calib(board, all_corners, all_ids, image_size,
                                  out_path, fix_k3=fix_k3, hardware_id=hw_id)
            solved = True
            break

    cap.release()
    cv2.destroyWindow(win)
    if quit_all:
        raise KeyboardInterrupt
    return solved


def cmd_intrinsics(args: argparse.Namespace) -> None:
    try:
        _intrinsics_capture_loop(
            args.camera, args.out, args.square_mm, args.min_views,
            args.width, args.height, args.max_res,
            getattr(args, "realsense", False), getattr(args, "fix_k3", False))
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


def cmd_intrinsics_all(args: argparse.Namespace) -> None:
    """Auto-discover all connected cameras and calibrate each one's intrinsics.

    Naming is STABLE across reruns: a camera whose hardware id already has a
    camera_intrinsics_<name>.json on disk keeps that same name (so re-running
    this after replugging cameras in a different USB order, or to redo just
    one camera, doesn't silently reassign c0/c1/c2 to different physical
    cameras and desync the extrinsics/fusion files that reference them by
    name). Only cameras with NO prior calibration on record get a fresh name:
    RealSense -> 'rs'/'rs2'/..., others -> the lowest unused c<N>.
    """
    try:
        from camera_identity import discover_cameras, calibrated_hardware_ids
    except Exception as e:
        sys.exit(f"[intrinsics-all] camera discovery unavailable: {e}")

    cams = discover_cameras()
    if not cams:
        sys.exit("[intrinsics-all] no color cameras discovered. Are they plugged "
                 "in? (RealSense color needs pyrealsense2 for capture, but is "
                 "still discovered as a UVC color node.)")

    # Reuse existing names by hardware id; only assign fresh ones for cameras
    # with no prior calibration file on record.
    existing = calibrated_hardware_ids(_HERE)              # name -> hw_id
    by_id = {hw: name for name, hw in existing.items()}    # hw_id -> name
    used_names = set(existing)

    def _next_free(prefix: str, n: list) -> str:
        while True:
            cand = prefix if n[0] == 0 else f"{prefix}{n[0]}"
            n[0] += 1
            if cand not in used_names:
                return cand

    rs_n, web_n = [0], [0]
    for c in cams:
        reused = by_id.get(c["id"]) if c["id"] else None
        if reused:
            c["name"] = reused
            c["reused"] = True
        elif c["is_realsense"]:
            c["name"] = _next_free("rs", rs_n)
            used_names.add(c["name"])
        else:
            c["name"] = _next_free("c", web_n)
            used_names.add(c["name"])

    print(f"[intrinsics-all] discovered {len(cams)} camera(s):")
    for c in cams:
        tag = "  [RealSense]" if c["is_realsense"] else ""
        tag += "  (reusing existing name)" if c.get("reused") else "  (new)"
        print(f"    {c['name']:>4}  index {c['index']}  {c['label']!r}"
              f"  id={c['id']}{tag}")
    print("[intrinsics-all] calibrating each in turn "
          "(SPACE capture, C solve, S skip, Q quit).")

    done, skipped, failed = [], [], []
    try:
        for k, c in enumerate(cams, 1):
            out = _named(DEFAULT_INTRINSICS, c["name"])
            title = f"[{k}/{len(cams)}] {c['name']}"
            print(f"\n[intrinsics-all] ==== {title} (index {c['index']}"
                  f"{', RealSense' if c['is_realsense'] else ''}) ====")
            try:
                ok = _intrinsics_capture_loop(
                    c["index"], out, args.square_mm, args.min_views,
                    args.width, args.height, args.max_res, c["is_realsense"],
                    args.fix_k3, title)
                (done if ok else skipped).append(c["name"])
            except SystemExit as e:
                # A per-camera failure (bad open, too few valid views, etc.)
                # calls sys.exit(), which — uncaught — would silently kill this
                # WHOLE multi-camera walkthrough and abandon every camera still
                # left in `cams`, without writing anything for them either.
                # Treat it the same as pressing 'S': log why, move on to the
                # next camera. Only 'Q' (KeyboardInterrupt, below) should stop
                # the whole batch.
                print(f"[intrinsics-all] {title} FAILED: {e} — skipping this "
                      f"camera, continuing with the rest.")
                failed.append(c["name"])
    except KeyboardInterrupt:
        print("\n[intrinsics-all] stopped early (Q).")
    cv2.destroyAllWindows()
    print(f"\n[intrinsics-all] done: {done or '(none)'}"
          f"   skipped: {skipped or '(none)'}"
          f"   failed: {failed or '(none)'}")
    if done:
        print("[intrinsics-all] NEXT: solve extrinsics against a FIXED board:")
        print("[intrinsics-all]   python calibration/charuco_calibration.py "
              "extrinsics-all " + " ".join(
                  f"--cam {c['name']}:{c['index']}" for c in cams
                  if c['name'] in done) + " --square-mm <measured>")


def _camera_hardware_id(index: int, realsense: bool) -> str | None:
    """Stable hardware identity of the camera being calibrated, for the JSON.

    Best-effort: returns None if identity can't be read (import guarded so the
    calibration tool still runs on platforms without the identity helper)."""
    try:
        from camera_identity import identity_for_index, label_for_index
    except Exception:
        return None
    hw = identity_for_index(index, realsense=realsense)
    if hw:
        print(f"[intrinsics] camera hardware id: {hw}  ({label_for_index(index)})")
    else:
        print("[intrinsics] hardware id unavailable (index-only binding).")
    return hw


def _radial_factor(dist, r):
    """OpenCV radial distortion factor 1 + k1 r^2 + k2 r^4 + k3 r^6 at radius r."""
    k1, k2 = float(dist[0]), float(dist[1])
    k3 = float(dist[4]) if len(dist) >= 5 else 0.0
    r2 = r * r
    return 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2


def _warn_if_distortion_overfit(dist) -> None:
    """Flag a non-monotonic / diverging radial model — the tell-tale of an
    overfit high-order term (usually k3) from insufficient frame-edge coverage.

    A healthy lens model's radial factor is monotonic out to the corner
    (r ~ 0.6-0.7 in normalised units). If it turns over or swings > ~10%, the
    correction bends the WRONG way at the edges, so triangulated points near the
    frame border smear — exactly the failure this guard exists to catch. We warn
    (not error) so an intentional strong-distortion calibration still saves.
    """
    rs = np.linspace(0.0, 0.7, 15)
    fac = np.array([_radial_factor(dist, r) for r in rs])
    swing = float(fac.max() - fac.min())
    # A well-behaved low/moderate-distortion model stays within a few % of 1.0
    # across the frame. Only flag genuine divergence: a large overall swing, or
    # the correction turning over enough to drop below 0.95 at the corner. (A
    # sub-1% ripple from float noise is NOT overfitting — don't warn on it.)
    if swing > 0.06 or fac[-1] < 0.95:
        print("[intrinsics] *** WARNING: distortion model looks OVERFIT ***")
        print(f"[intrinsics]   radial factor swings {swing*100:.1f}% over the frame "
              f"(corner={fac[-1]:.3f}); k3={float(dist[4]) if len(dist)>=5 else 0:.3f}")
        print("[intrinsics]   This bends the correction the wrong way at the EDGES "
              "(fused points near the border will smear).")
        print("[intrinsics]   Re-run with --fix-k3 (pins the 6th-order term to 0), "
              "or capture more views with the board pushed into the CORNERS.")


def _warn_if_intrinsics_geometry_bad(camera_matrix, image_size) -> None:
    """Flag a squashed/wrong-aspect intrinsic — the signature of calibrating on
    frames whose actual geometry didn't match the declared image_size (e.g. a
    camera streaming a scaled/letterboxed mode into a buffer of a different
    resolution). Two independent tells, both resolution-bound so focus/motion
    can't cause them:

      - fx != fy: real webcam pixels are square, so fx/fy should be ~1. A large
        ratio means one axis was scaled relative to the other before calibration.
        This corrupts the metric focal length, so every extrinsic distance solved
        against this K is off by the same factor — right board size, wrong range.
      - principal point far from the frame centre: cx,cy normally land within
        ~15% of (W/2, H/2). A cy sitting near H/4 is the vertical-squash smoking
        gun (the centre of a scaled-in image lands at a fraction of the buffer).

    Warn (not error) so an unusual-but-intentional sensor still saves; the RMS
    can be excellent (sharp corners) while the geometry is wrong, so this catches
    what the reprojection error and the distortion guard both miss.
    """
    fx, fy = float(camera_matrix[0][0]), float(camera_matrix[1][1])
    cx, cy = float(camera_matrix[0][2]), float(camera_matrix[1][2])
    W, H = int(image_size[0]), int(image_size[1])
    ratio = fx / fy if fy else float("inf")
    # 10% aspect and 15% principal-point tolerances pass every genuine square-
    # pixel calibration seen on this rig while flagging the ~1.44 squash we hit.
    aspect_bad = abs(ratio - 1.0) > 0.10
    pp_bad = (abs(cx - W / 2.0) > 0.15 * W) or (abs(cy - H / 2.0) > 0.15 * H)
    if not (aspect_bad or pp_bad):
        return
    print("[intrinsics] *** WARNING: intrinsic geometry looks WRONG ***")
    if aspect_bad:
        # Recover the stream height that WOULD make pixels square, so the message
        # points straight at the likely real capture size to recalibrate at.
        implied_h = H * (fy / fx) if fx else 0.0
        print(f"[intrinsics]   fx={fx:.0f} fy={fy:.0f} (fx/fy={ratio:.3f}); "
              f"square pixels need fx~=fy. This vertical field matches a stream "
              f"height of ~{implied_h:.0f}px, not the declared {H}.")
    if pp_bad:
        print(f"[intrinsics]   principal point (cx,cy)=({cx:.0f},{cy:.0f}) is far "
              f"from the frame centre ({W//2},{H//2}).")
    print("[intrinsics]   The camera likely streamed a SCALED/letterboxed mode, "
          "not a native {}x{} frame. Every extrinsic distance solved against this "
          "K will be off. Confirm the real capture size (v4l2-ctl "
          "--list-formats-ext) and recalibrate at a native, square-pixel mode."
          .format(W, H))


def _run_intrinsics_calib(board, all_corners, all_ids, image_size, out_path,
                          fix_k3: bool = False, hardware_id: str | None = None) -> None:
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

    # CALIB_FIX_K3 pins the unstable 6th-order radial term to 0. For a
    # low-distortion lens (RealSense color, most webcams) k3 is near zero and
    # only overfits without dense corner coverage — fixing it yields a stable,
    # monotonic model that behaves at the frame edges.
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=flags)

    print(f"[intrinsics] RMS reprojection error: {rms:.4f} px "
          f"({'good' if rms < 1.0 else 'high — recapture with more/varied views'})"
          + ("  [k3 fixed to 0]" if fix_k3 else ""))
    print(f"[intrinsics] camera_matrix=\n{camera_matrix}")
    print(f"[intrinsics] dist_coeffs={dist_coeffs.ravel()}")
    _warn_if_distortion_overfit(dist_coeffs.ravel())
    _warn_if_intrinsics_geometry_bad(camera_matrix, image_size)

    data = {
        "image_size": list(image_size),
        "rms_reproj_px": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
    }
    # Hardware identity binds this intrinsic to the physical camera (not a port or
    # OpenCV index), so the pipeline can auto-load it whatever port the camera
    # lands on, and warn if an index maps to a different camera. Omitted if unknown.
    if hardware_id:
        data["hardware_id"] = hardware_id
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[intrinsics] wrote {out_path}")


# ----------------------------------------------------------------------------
# extrinsics
# ----------------------------------------------------------------------------
def _solve_one_extrinsic(camera: int, intrinsics_path: str, out_path: str,
                         square_mm: float, n_avg: int, width: int, height: int,
                         max_res: bool, title: str = "",
                         realsense: bool = False,
                         std_tol_m: float = _POSE_STD_TOL_M) -> bool:
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

    cap = _open_camera(camera, width, height, max_res, realsense)
    win = f"extrinsics {title} — SPACE solve, S skip, Q quit".strip()
    _make_window(win)
    print(f"[extrinsics] {title} Fix the board at the WORLD ORIGIN, facing the camera.")
    print(f"[extrinsics] SPACE = average {n_avg} frames & solve, S = skip this "
          f"camera, Q = quit all.")

    solved = False
    quit_all = False
    stall_since = None
    warned_stall = False
    perf_read_ms: list[float] = []
    perf_detect_ms: list[float] = []
    perf_last_report = time.time()
    while True:
        t_read0 = time.time()
        ok, frame = cap.read()
        perf_read_ms.append((time.time() - t_read0) * 1000.0)
        if not ok:
            # A camera can wedge mid-session (USB glitch, driver hiccup) and
            # start returning ok=False on every read — without this, the loop
            # spins silently forever and the window just looks frozen with no
            # indication why. Surface it, then bail out with a clear error
            # instead of hanging indefinitely.
            now = time.time()
            if stall_since is None:
                stall_since = now
            elif not warned_stall and now - stall_since > 3.0:
                print(f"[extrinsics] {title} camera {camera} stopped returning "
                      f"frames — still retrying...")
                warned_stall = True
            elif now - stall_since > 15.0:
                sys.exit(f"[extrinsics] {title} camera {camera} produced no frames "
                         f"for 15s — likely disconnected or wedged. Check the "
                         f"cable/port and re-run.")
            continue
        stall_since = None
        warned_stall = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_det0 = time.time()
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        perf_detect_ms.append((time.time() - t_det0) * 1000.0)

        # Lightweight perf readout every ~2s — read() vs detectBoard() vs actual
        # achieved FPS, so a "laggy feed" report has concrete numbers to point
        # at (slow USB reads vs slow detection vs something outside this loop,
        # e.g. imshow/window compositing) instead of needing to be reproduced
        # blind after the fact.
        now = time.time()
        if now - perf_last_report > 2.0 and perf_read_ms:
            # perf_read_ms grows on every read ATTEMPT (success or failure);
            # perf_detect_ms only on successful ones that reached detectBoard —
            # the two are NOT the same length whenever reads are failing (e.g. a
            # stalling/timing-out camera), so they must be indexed independently.
            n_read = len(perf_read_ms)
            n_det = len(perf_detect_ms)
            fps = n_read / (now - perf_last_report)
            read_stats = (f"{sorted(perf_read_ms)[n_read//2]:.0f}/"
                         f"{max(perf_read_ms):.0f}")
            det_stats = (f"{sorted(perf_detect_ms)[n_det//2]:.0f}/"
                        f"{max(perf_detect_ms):.0f}") if n_det else "n/a (no successful reads)"
            print(f"[extrinsics] {title} perf: {fps:.1f} fps over {n_read} read "
                  f"attempts ({n_det} succeeded) | read ms (median/max) = "
                  f"{read_stats} | detect ms (median/max) = {det_stats}")
            perf_read_ms.clear()
            perf_detect_ms.clear()
            perf_last_report = now

        vis = frame.copy()
        n_det = 0 if charuco_ids is None else len(charuco_ids)
        live_dist_m = None
        if n_det >= 6:
            # Same refined SQPNP solve the SAVE path uses, so the drawn axes match
            # what gets averaged/stored (and jitter far less than the old default
            # iterative solvePnP on raw corners).
            sol = _solve_board_pose(gray, charuco_corners, charuco_ids, board,
                                    camera_matrix, dist_coeffs)
            if sol is not None:
                rvec, tvec = sol
                cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs,
                                  rvec, tvec, square_m * 2)
                live_dist_m = float(np.linalg.norm(tvec))
        label = f"{title}  corners: {n_det}  (need >=6; SPACE solve, S skip, Q quit)"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        if live_dist_m is not None:
            # Sanity-checkable BEFORE you commit a solve: a near-planar board
            # viewed at a marginal angle/distance can converge to a pose that
            # fits the 2D corners well (low reprojection error) but is at the
            # WRONG physical distance — a known planar-PnP ambiguity, not a
            # detection failure, so nothing else here would catch it. Eyeball
            # this against the real distance before pressing SPACE.
            dist_color = (0, 255, 255) if live_dist_m > 2.5 else (0, 255, 0)
            cv2.putText(vis, f"solved distance: {live_dist_m:.2f} m  "
                        f"(sanity-check against the real distance)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit_all = True
            break
        if key == ord('s'):
            print(f"[extrinsics] {title} skipped.")
            break
        if key == ord(' '):
            # The video window doesn't update at all while _average_pose grabs
            # its n_avg frames (no imshow in that loop) — on a camera with slow
            # or marginal reads this looks exactly like a freeze with zero
            # feedback. Print immediately so SPACE visibly did something.
            print(f"[extrinsics] {title} averaging {n_avg} frames...")
            pose = _average_pose(cap, charuco_detector, board,
                                 camera_matrix, dist_coeffs, n_avg, std_tol_m)
            if pose is None:
                # _average_pose already printed the specific reason (too few
                # solves, or unstable/jiggling). Stay in the loop to retry.
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
    # The extrinsic MUST be solved at the resolution this camera's intrinsics were
    # calibrated at (K is only valid there). The intrinsics file's image_size is
    # the source of truth, so auto-adopt it unless the user overrode --width/height
    # or asked for --max-res — otherwise a camera calibrated at, e.g., 1280x960
    # would be silently opened at the 1280x720 default and solved against a
    # mis-scaled K. (--realsense uses its own default via _open_camera.)
    w, h = args.width, args.height
    if not args.max_res and not args.realsense and (w, h) == (1280, 720):
        try:
            import json
            sz = json.load(open(args.intrinsics)).get("image_size")
            if sz:
                w, h = int(sz[0]), int(sz[1])
                if (w, h) != (1280, 720):
                    print(f"[extrinsics] using calibrated resolution {w}x{h} from "
                          f"{os.path.basename(args.intrinsics)} (pass --width/--height "
                          f"to override).")
        except Exception:
            pass
    _solve_one_extrinsic(args.camera, args.intrinsics, args.out,
                         args.square_mm, args.n_avg, w, h,
                         args.max_res, realsense=args.realsense,
                         std_tol_m=args.pose_std_mm / 1000.0)


def cmd_extrinsics_all(args: argparse.Namespace) -> None:
    """Calibrate the extrinsics of EVERY camera in ONE command.

    Walks through each --cam <name>:<index> in turn against the SAME fixed board:
    the board must stay put for the whole run so all cameras share one world
    frame. For each camera it reads camera_intrinsics_<name>.json and writes
    camera_extrinsics_<name>.json. Press SPACE to solve a camera, S to skip it, Q
    to stop the walkthrough.
    """
    if getattr(args, "auto", False):
        if args.cam:
            sys.exit("[extrinsics-all] --auto and --cam are mutually exclusive.")
        try:
            from camera_identity import match_calibrated_cameras
        except Exception as e:
            sys.exit(f"[extrinsics-all] --auto unavailable: {e}")
        cams, auto_rs = match_calibrated_cameras(_HERE)
        rs_names = set(args.realsense) | set(auto_rs)
        if len(cams) < 2:
            sys.exit(f"[extrinsics-all] --auto found only {len(cams)} calibrated "
                     f"camera(s) plugged in; need >= 2. Run intrinsics-all first, "
                     f"or plug the cameras in.")
    elif not args.cam:
        sys.exit("[extrinsics-all] pass --cam per camera, or --auto to discover.")
    else:
        cams = args.cam
        rs_names = set(args.realsense)
    print(f"[extrinsics-all] {len(cams)} cameras: "
          f"{', '.join(f'{n}:{i}' + ('(rs)' if n in rs_names else '') for n, i in cams)}")
    print("[extrinsics-all] KEEP THE BOARD FIXED at the world origin for the WHOLE "
          "run — that shared pose is what puts all cameras in one frame.")
    done, skipped, failed = [], [], []
    try:
        for k, (name, idx) in enumerate(cams, 1):
            intr = _named(DEFAULT_INTRINSICS, name)
            out = _named(DEFAULT_EXTRINSICS, name)
            title = f"[{k}/{len(cams)}] {name}"
            is_rs = name in rs_names
            # A RealSense captures via pyrealsense2 at its own (intrinsics-matched)
            # size; webcams use the shared --width/height. Both MUST equal the size
            # each camera's intrinsics were calibrated at, or the solved pose is
            # invalid (mis-scaled K).
            w = args.rs_width if is_rs else args.width
            h = args.rs_height if is_rs else args.height
            print(f"\n[extrinsics-all] ==== {title} "
                  f"({'RealSense color' if is_rs else f'camera index {idx}'}) ====")
            try:
                ok = _solve_one_extrinsic(idx, intr, out, args.square_mm, args.n_avg,
                                          w, h, args.max_res, title, realsense=is_rs,
                                          std_tol_m=args.pose_std_mm / 1000.0)
                (done if ok else skipped).append(name)
            except SystemExit as e:
                # Same failure mode as intrinsics-all: an uncaught sys.exit()
                # from a per-camera error (wedged camera, bad open, ...) would
                # otherwise kill the WHOLE walkthrough and silently abandon
                # every camera still left in `cams`. Skip just this one.
                print(f"[extrinsics-all] {title} FAILED: {e} — skipping this "
                      f"camera, continuing with the rest.")
                failed.append(name)
    except KeyboardInterrupt:
        print("\n[extrinsics-all] stopped early (Q).")
    cv2.destroyAllWindows()
    print(f"\n[extrinsics-all] done: {done or '(none)'}"
          f"   skipped: {skipped or '(none)'}"
          f"   failed: {failed or '(none)'}")
    if done:
        print("[extrinsics-all] All solved cameras now share ONE world frame "
              "(same board pose). Validate with the fusion node's reproj log.")


def cmd_stamp_id(args: argparse.Namespace) -> None:
    """Write the hardware id into an EXISTING intrinsics file — no recalibration.

    The intrinsic values (camera_matrix, dist_coeffs) are a property of the
    lens+sensor and don't change; this only records which physical camera they
    belong to, so run_multicam can auto-resolve the camera by identity on any
    port. Reads the id from the camera currently at --camera (pass --realsense for
    a RealSense so the SDK serial is used). Verifies the file's calibrated
    image_size matches the camera's current stream before stamping.
    """
    path = _named(DEFAULT_INTRINSICS, args.name)
    if not os.path.exists(path):
        sys.exit(f"[stamp-id] no intrinsics file at {path}. Calibrate first.")
    with open(path) as f:
        data = json.load(f)

    try:
        from camera_identity import identity_for_index, label_for_index
    except Exception as e:
        sys.exit(f"[stamp-id] camera_identity unavailable: {e}")

    hw = identity_for_index(args.camera, realsense=args.realsense)
    if not hw:
        sys.exit(f"[stamp-id] could not read a hardware id for camera "
                 f"{args.camera}. On a RealSense pass --realsense; otherwise this "
                 f"platform may not expose USB identity via sysfs.")

    # Sanity: confirm the camera at this index streams the size the file was
    # calibrated at, so we don't stamp an id onto a mismatched-resolution file.
    want = data.get("image_size")
    if want and not args.no_verify_size:
        _rs = args.realsense
        _w, _h = (int(want[0]), int(want[1]))
        cap = _open_camera(args.camera, _w, _h, False, _rs)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            gw, gh = frame.shape[1], frame.shape[0]
            if (gw, gh) != (_w, _h):
                print(f"[stamp-id] *** WARNING: camera {args.camera} streams "
                      f"{gw}x{gh} but {os.path.basename(path)} was calibrated at "
                      f"{_w}x{_h}. This may be the WRONG camera for this file. "
                      f"Re-run with --no-verify-size to stamp anyway. ***")
                sys.exit(1)

    prev = data.get("hardware_id")
    data["hardware_id"] = hw
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    label = label_for_index(args.camera)
    print(f"[stamp-id] {os.path.basename(path)}: hardware_id = {hw}  ({label})"
          + (f"   (was {prev})" if prev else ""))


# Reject a solve whose per-frame translation scatter exceeds this — a jiggly

def _solve_board_pose(gray, cc, ci, board, K, dist):
    """One refined board-pose solve. Refines the ChArUco corners to sub-pixel and
    uses SQPNP (stable on the near-planar board, unlike the default iterative
    solver which can flip to a mirror pose). Returns (rvec, tvec) or None."""
    cc = cv2.cornerSubPix(gray, cc.astype(np.float32), _SUBPIX_WIN, (-1, -1),
                          _SUBPIX_CRIT)
    obj_pts, img_pts = board.matchImagePoints(cc, ci)
    if obj_pts is None or len(obj_pts) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_SQPNP)
    return (rvec, tvec) if ok else None


def _average_pose(cap, charuco_detector, board, camera_matrix, dist_coeffs, n_avg,
                  std_tol_m: float = _POSE_STD_TOL_M):
    """Average board pose over n_avg frames to reduce single-frame PnP jitter.

    Returns (R_cam_board 3x3, t_cam_board 3), or None if too few solves OR the
    per-frame scatter is too large (a jiggly board — hold it steadier and retry).
    Corners are refined to sub-pixel and solved with SQPNP; translation is
    averaged directly, rotation via mean of rvecs (valid for the small spread of
    a static board — a large spread trips the stability gate below).
    """
    tvecs: list[np.ndarray] = []
    rvecs: list[np.ndarray] = []
    grabbed = 0
    stall_since = None
    warned_stall = False
    while grabbed < n_avg:
        ok, frame = cap.read()
        if not ok:
            # Same silent-hang risk as the preview loop: a wedged camera mid-
            # average would otherwise spin here forever with the window frozen
            # and no feedback. Surface it and give up cleanly instead.
            now = time.time()
            if stall_since is None:
                stall_since = now
            elif not warned_stall and now - stall_since > 3.0:
                print(f"[extrinsics] camera stopped returning frames mid-average "
                      f"— still retrying...")
                warned_stall = True
            elif now - stall_since > 15.0:
                print(f"[extrinsics] camera produced no frames for 15s during "
                      f"averaging — likely disconnected or wedged. Aborting "
                      f"this solve; check the cable/port and re-press SPACE.")
                return None
            continue
        stall_since = None
        warned_stall = False
        grabbed += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = charuco_detector.detectBoard(gray)
        if ci is None or len(ci) < 6:
            continue
        sol = _solve_board_pose(gray, cc, ci, board, camera_matrix, dist_coeffs)
        if sol is not None:
            rvec, tvec = sol
            rvecs.append(rvec.ravel())
            tvecs.append(tvec.ravel())

    if len(tvecs) < max(3, n_avg // 3):
        print(f"[extrinsics] only {len(tvecs)}/{n_avg} valid solves — board not "
              f"reliably detected; reposition and retry.")
        return None
    t_std = np.std(tvecs, axis=0)
    r_std = np.std(rvecs, axis=0)
    print(f"[extrinsics] averaged {len(tvecs)}/{n_avg} valid frames; "
          f"tvec std = {np.round(t_std*1000, 2)} mm, "
          f"rvec std = {np.round(np.degrees(r_std), 3)} deg")
    # Stability gate: a fixed board should solve to well under a mm of scatter.
    # High scatter = the board (or camera) is moving, or PnP is flipping between
    # poses — either way the averaged pose is unreliable. Refuse to save it.
    if float(np.linalg.norm(t_std)) > std_tol_m:
        print(f"[extrinsics] *** UNSTABLE: pose scatter {np.linalg.norm(t_std)*1000:.1f} mm "
              f"> {std_tol_m*1000:.1f} mm tolerance ***")
        print("[extrinsics]   The board or camera is JIGGLING. Fix the board rigidly "
              "(clamp it), steady the camera, and re-press SPACE. Loosen the gate "
              "with --pose-std-mm only if you understand the accuracy cost.")
        return None
    t_mean = np.mean(tvecs, axis=0)
    r_mean = np.mean(rvecs, axis=0)
    R_mean, _ = cv2.Rodrigues(r_mean)
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
    g.add_argument("--out", default=None,
                   help=f"output path (default: board.png, or board_<paper>.png "
                        f"when --paper is not a3)")
    g.add_argument("--square-mm", type=float, default=None,
                   help="nominal printed square size for sizing the image "
                        "(default: DEFAULT_SQUARE_MM, or largest that fits "
                        "--paper if given)")
    g.add_argument("--paper", choices=sorted(PAPER_SIZES_MM), default=None,
                   help="paper size to size the board for (letter, a4, a3); "
                        "default (no --paper) uses DEFAULT_SQUARE_MM on an "
                        "A3-sized board; ignored if --square-mm is also given")
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
    i.add_argument("--realsense", action="store_true",
                   help="Capture via pyrealsense2 COLOR (Intel RealSense) instead "
                        "of cv2 — the same stream the teleop pipeline uses. Solves "
                        "REAL distortion coeffs (the SDK's factory color intrinsics "
                        "report zero distortion, which leaves visible edge/radial "
                        "error). --camera index is ignored; set --width/--height to "
                        "the pipeline size (default 640x480).")
    i.add_argument("--fix-k3", action="store_true",
                   help="Pin the 6th-order radial distortion term k3 to 0 "
                        "(CALIB_FIX_K3). Use for low-distortion lenses (RealSense "
                        "color, most webcams) where k3 overfits without dense corner "
                        "coverage and makes the correction diverge at the frame "
                        "edges — the tool warns when it detects this.")
    i.set_defaults(func=cmd_intrinsics)

    ia = sub.add_parser(
        "intrinsics-all",
        help="auto-discover ALL connected cameras and calibrate each one's "
             "intrinsics (names auto-assigned c0/c1/rs; ids stamped)")
    ia.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                    help="MEASURED printed square size in mm")
    ia.add_argument("--min-views", type=int, default=12)
    ia.add_argument("--width", type=int, default=1280)
    ia.add_argument("--height", type=int, default=720)
    ia.add_argument("--max-res", action="store_true",
                    help="Calibrate each webcam at its highest supported resolution.")
    ia.add_argument("--fix-k3", action="store_true",
                    help="Pin k3=0 for all cameras (low-distortion lenses).")
    ia.set_defaults(func=cmd_intrinsics_all)

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
    e.add_argument("--realsense", action="store_true",
                   help="Capture via pyrealsense2 COLOR (Intel RealSense) instead "
                        "of cv2.VideoCapture — the same path the teleop pipeline "
                        "uses. --camera index is ignored; use --width/--height to "
                        "match this camera's intrinsics size (default 640x480).")
    e.add_argument("--pose-std-mm", type=float, default=_POSE_STD_TOL_M * 1000.0,
                   help="Reject a solve whose averaged board-pose scatter exceeds "
                        "this (mm) — guards against a jiggling board/camera. "
                        f"Default {_POSE_STD_TOL_M*1000:.0f} mm. Raise only if you "
                        "accept the accuracy cost.")
    e.set_defaults(func=cmd_extrinsics)

    ea = sub.add_parser(
        "extrinsics-all",
        help="calibrate EVERY camera's extrinsics in one walkthrough (one board)")
    ea.add_argument("--auto", action="store_true",
                    help="Auto-discover connected cameras and match each to its "
                         "camera_intrinsics_<name>.json by hardware id — no --cam "
                         "needed. RealSense auto-flagged. (Same discovery as "
                         "run_multicam --auto.) Keep the board FIXED for the run.")
    ea.add_argument("--cam", action="append", type=_parse_cam_spec, default=None,
                    metavar="NAME:INDEX",
                    help="Camera name:OpenCV-index. Repeat per camera, e.g. "
                         "--cam c0:0 --cam c1:2 --cam rs:4. Uses each camera's "
                         "camera_intrinsics_<name>.json; writes "
                         "camera_extrinsics_<name>.json. Omit and pass --auto to "
                         "discover automatically. Keep the board FIXED.")
    ea.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                    help="MEASURED printed square size in mm")
    ea.add_argument("--n-avg", type=int, default=30)
    ea.add_argument("--width", type=int, default=1280)
    ea.add_argument("--height", type=int, default=720)
    ea.add_argument("--max-res", action="store_true",
                    help="Open each camera at its HIGHEST supported resolution "
                         "(match how you ran intrinsics for these cameras).")
    ea.add_argument("--realsense", action="append", default=[], metavar="NAME",
                    help="Mark a --cam NAME as an Intel RealSense: capture its "
                         "COLOR via pyrealsense2 (same path as the teleop "
                         "pipeline), NOT cv2. Repeatable. RealSense cameras use "
                         "--rs-width/--rs-height (default 640x480) instead of "
                         "--width/--height, to match their intrinsics.")
    ea.add_argument("--rs-width", type=int, default=640,
                    help="Color width for --realsense cameras (default 640; must "
                         "equal their intrinsics' image_size width).")
    ea.add_argument("--rs-height", type=int, default=480,
                    help="Color height for --realsense cameras (default 480).")
    ea.add_argument("--pose-std-mm", type=float, default=_POSE_STD_TOL_M * 1000.0,
                    help="Reject a solve whose averaged board-pose scatter exceeds "
                         "this (mm) — guards against a jiggling board/camera. "
                         f"Default {_POSE_STD_TOL_M*1000:.0f} mm.")
    ea.set_defaults(func=cmd_extrinsics_all)

    s = sub.add_parser(
        "stamp-id",
        help="record the hardware id into an existing intrinsics file "
             "(no recalibration) so run_multicam can auto-resolve the camera")
    s.add_argument("--camera", type=int, required=True,
                   help="OpenCV index the physical camera is currently on.")
    s.add_argument("--name", required=True,
                   help="Camera name; stamps camera_intrinsics_<name>.json.")
    s.add_argument("--realsense", action="store_true",
                   help="Use the RealSense SDK serial (rs:<serial>) as the id.")
    s.add_argument("--no-verify-size", action="store_true",
                   help="Skip the resolution cross-check before stamping.")
    s.set_defaults(func=cmd_stamp_id)

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

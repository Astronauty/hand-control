#!/usr/bin/env python3
"""Per-camera hand-landmark node for the multi-camera teleop pipeline.

One instance per camera. Opens a single camera, runs ITS OWN MediaPipe
HandLandmarker (VIDEO mode — per-instance tracking state is required, so
instances are never shared), picks the one best RIGHT hand (matching the
single-camera publisher), and publishes the raw 2D image landmarks + per-landmark
visibility on a per-camera topic:

    /hand/cam_<name>/landmarks   (std_msgs/Float32MultiArray)

The fusion node (teleop/hand_fusion_node.py) subscribes to all such topics and
triangulates them into the shared world frame. This node does NOT publish
/hand/joint_angles — that stays the fusion node's job, preserving the 183-float
contract for downstream teleop.

Per-camera message layout (flat float array):
    [0]        stamp_s      capture time (seconds, monotonic-ish wall clock)
    [1]        detected     1.0 if a right hand was found this frame else 0.0
    [2]        cam_w        image width  (px)  — lets fusion denormalise uv
    [3]        cam_h        image height (px)
    [4:4+63]   uv_norm      21 landmarks x (x, y, z) NORMALISED image coords
                            (x,y in [0,1]); fusion multiplies x*cam_w, y*cam_h.
    [67:130]   vis          21 landmarks x (visibility, presence, 0) — we pack
                            visibility in slot 0 of each triple; slots 1,2 spare.
Length: 4 + 63 + 63 = 130 floats. When detected==0 the landmark blocks are zeros.

Usage:
    python teleop/hand_landmark_node.py --camera 1 --name c0
    python teleop/hand_landmark_node.py --camera 2 --name c1
    python teleop/hand_landmark_node.py --camera 3 --name rs   # RealSense as RGB
"""
from __future__ import annotations

import argparse
import json
import os
import time

# Quiet the OpenCV capture backend BEFORE importing cv2. Under --max-res the
# camera streams MJPG, whose bitstream occasionally has a few stray bytes before
# a restart marker; libjpeg-turbo recovers and returns a valid frame but prints a
# "Corrupt JPEG data: N extraneous bytes" line to stderr per event. Verified
# harmless (full 30 fps, low median reproj, only a rare per-landmark max blip that
# the One Euro filter absorbs), so suppress the console spam. OPENCV_LOG_LEVEL
# silences OpenCV's own logger; OPENCV_FFMPEG_LOGLEVEL quiets the FFmpeg backend.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # AV_LOG_QUIET

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage

import mediapipe as mp

from hand_message import HAND_CONNECTIONS, WORLD_FROM_BOARD, sensor_qos

# Model lives next to the original publisher.
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
_CALIB_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "calibration")

_TRACK_HANDEDNESS = "Right"   # matches teleop/ui.py
N_LM = 21


class _RealSenseCapture:
    """cv2.VideoCapture-compatible shim over an Intel RealSense COLOR stream.

    RealSense color can't be reliably opened at the desired resolution through bare
    V4L2/cv2.VideoCapture (multiple /dev/video nodes, stateful ioctls), so we drive
    it via pyrealsense2 instead. Exposes just the cv2.VideoCapture surface the node
    uses — isOpened(), read(), get(CAP_PROP_FRAME_WIDTH/HEIGHT), set() (no-op after
    construction), release() — so the capture loop, reopen, and preview code stay
    unchanged.

    Streams color at (width x height @ fps) in BGR (matching cv2's BGR convention),
    so downstream MediaPipe/preview code needs no format change. Default 640x480@30
    matches the USB webcams' framerate for clean fusion sync (the D435I's 1080p
    color caps at 8 fps — too slow for teleop).
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int | None = None,
                 retries: int = 8, retry_delay: float = 0.5):
        import pyrealsense2 as rs
        self._rs = rs
        # fps is RESOLUTION-DEPENDENT on the D435I: 640x480 does 30, but 1280x720
        # tops out at 15 and 1920x1080 at 8 — a hardcoded 30 makes start() raise
        # "Couldn't resolve requests" forever at those sizes. Query the device for
        # the highest bgr8 fps this exact WxH offers and use that (unless the caller
        # pinned one). No supported bgr8 mode at this size -> clear error.
        if fps is None:
            fps = self._best_fps(rs, width, height)
            if fps is None:
                raise RuntimeError(
                    f"RealSense has no bgr8 color mode at {width}x{height}. Pick a "
                    f"supported size (e.g. 640x480, 1280x720, 1920x1080) and "
                    f"calibrate its intrinsics at that size.")
        self._w, self._h, self._fps = width, height, fps
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        # start() also raises "Couldn't resolve requests" TRANSIENTLY while a prior
        # process is still releasing the device (RealSense handles don't free
        # instantly — a crashed/Ctrl-C'd run leaves it busy for a moment). Retry
        # with backoff so a mid-release device doesn't kill the whole pipeline.
        last_exc = None
        for _attempt in range(max(1, retries)):
            self._pipe = rs.pipeline()
            try:
                self._profile = self._pipe.start(cfg)
                self._opened = True
                print(f"[RealSense] color {width}x{height}@{fps} bgr8 started.")
                return
            except RuntimeError as e:
                last_exc = e
                time.sleep(retry_delay)
        raise RuntimeError(
            f"RealSense color {width}x{height}@{fps} start failed after "
            f"{retries} tries ({retry_delay}s apart): {last_exc}. Is the device "
            f"still held by another process? (check: fuser /dev/video*)")

    @staticmethod
    def _best_fps(rs, width: int, height: int) -> int | None:
        """Highest bgr8 color fps the device offers at exactly (width, height),
        or None if this size has no bgr8 color mode."""
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

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        """Return (ok, frame_bgr) like cv2.VideoCapture.read()."""
        try:
            frames = self._pipe.wait_for_frames(2000)   # ms timeout
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
        # Resolution is fixed at construction (the enabled stream); ignore cv2-style
        # set() calls so the shared open/reopen code path is a no-op here.
        return True

    def release(self):
        if self._opened:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._opened = False


def _load_extrinsic(name: str):
    """Load this camera's K, dist, and world->cam pose for the axis overlay.

    Returns (K, dist, rvec, tvec) ready for cv2.projectPoints, mapping the SHARED
    Z-up world frame -> this camera's pixels, or None if calibration is absent.
    Uses the same board->world remap (WORLD_FROM_BOARD) as the fusion node so the
    drawn axes are the exact world frame triangulation happens in.
    """
    intr_p = os.path.join(_CALIB_DIR, f"camera_intrinsics_{name}.json")
    extr_p = os.path.join(_CALIB_DIR, f"camera_extrinsics_{name}.json")
    if not os.path.exists(intr_p):
        intr_p = os.path.join(_CALIB_DIR, "camera_intrinsics.json")
    if not os.path.exists(extr_p):
        extr_p = os.path.join(_CALIB_DIR, "camera_extrinsics.json")
    if not (os.path.exists(intr_p) and os.path.exists(extr_p)):
        return None
    try:
        with open(intr_p) as f:
            intr = json.load(f)
        with open(extr_p) as f:
            extr = json.load(f)
        K = np.array(intr["camera_matrix"], float)
        dist = np.array(intr["dist_coeffs"], float)
        R_cam_board = np.array(extr["R_cam_world"], float)
        t = np.array(extr["t_cam_world"], float)
        # world (Z-up) -> board -> cam. R_cam_world = R_cam_board @ WFB^T (see
        # hand_fusion_node._load_camera). projectPoints wants a rotation VECTOR.
        R_cam_world = R_cam_board @ WORLD_FROM_BOARD.T
        rvec, _ = cv2.Rodrigues(R_cam_world)
        return K, dist, rvec, t
    except (KeyError, ValueError, OSError):
        return None


import contextlib


@contextlib.contextmanager
def _suppress_fd_stderr():
    """Temporarily redirect the OS-level stderr (fd 2) to /dev/null.

    The "Corrupt JPEG data" lines are printed by libjpeg-turbo straight to fd 2,
    below Python — so Python/logging redirection can't catch them, and the
    OPENCV_* env vars don't cover the V4L2 MJPG decode path. Redirecting fd 2
    around the frame grab silences them without hiding Python exceptions (those
    propagate normally; only C-level writes to fd 2 are dropped for the duration).
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def _hand_bbox_area(hand_landmarks) -> float:
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


class HandLandmarkNode(Node):
    """Captures one camera, detects the best right hand, publishes raw 2D LM."""

    def __init__(self, camera: int, name: str, width: int, height: int,
                 show: bool, max_res: bool = False,
                 preview_w: int = 640, preview_hz: float = 15.0,
                 realsense: bool = False):
        super().__init__(f"hand_landmark_{name}")
        self._name = name
        self._show = show
        self._realsense = realsense
        # BEST_EFFORT sensor QoS (see hand_message.sensor_qos): publish() must
        # NEVER block on a slow subscriber, or a wedged consumer back-pressures
        # this node's capture loop and freezes the camera (the press-8 freeze).
        self._pub = self.create_publisher(
            Float32MultiArray, f"/hand/cam_{name}/landmarks", sensor_qos())

        # Downscaled JPEG PREVIEW thumbnail on a SEPARATE, throttled topic — for
        # the combined viewer only. Kept small (thumb width) + low-rate so it never
        # competes with the full-res landmark stream or /hand/joint_angles.
        self._preview_pub = self.create_publisher(
            CompressedImage, f"/hand/cam_{name}/preview", sensor_qos())
        # Preview is now the PRIMARY camera view (tiled in its own window), so it
        # is larger + faster than a corner thumbnail. Still JPEG-compressed on a
        # throttled topic so it never competes with the full-res landmark stream.
        self._preview_w = preview_w
        self._preview_hz = preview_hz
        self._last_preview_t = 0.0

        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"hand_landmarker.task not found at {_MODEL_PATH}")

        if realsense:
            # RealSense COLOR via pyrealsense2 (see _RealSenseCapture). --camera is
            # ignored (the SDK picks the device); resolution comes from --width/height
            # (--max-res N/A — RS 1080p is only 8 fps). fps=None -> pick the highest
            # the device offers at that size (640x480:30, 1280x720:15, 1080p:8).
            self._cap = _RealSenseCapture(width or 640, height or 480, fps=None)
        else:
            self._cap = cv2.VideoCapture(camera)
            if not self._cap.isOpened():
                raise RuntimeError(f"could not open camera index {camera}")
            # Newest frame only: keeps read() from serving a queued stale buffer, both
            # in steady state and after a wedge (matched on the reopen path). Best-
            # effort — some backends ignore it.
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if max_res:
                width, height = self._request_max_res()
            elif width and height:
                # Set MJPG BEFORE the explicit size: many webcams expose their
                # higher modes (e.g. 1920x1080) ONLY under MJPG, and a raw request
                # silently clamps to a low mode like 640x480 (the c0 bug: intrinsics
                # 1080p but the stream fell back to 480p). --max-res already does
                # this in _request_max_res; mirror it for the explicit path. Harmless
                # for cameras whose requested mode is available raw.
                try:
                    self._cap.set(cv2.CAP_PROP_FOURCC,
                                  cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Remember open params so the capture loop can REOPEN the camera on a USB
        # dropout instead of exiting (a clean exit tore the whole pipeline down).
        self._camera_idx = camera
        self._cap_w, self._cap_h = width, height
        # Reopen escalation: a kernel-level UVC wedge (identical-frame freeze under
        # load) is often NOT cleared by immediately reopening the same busy /dev/videoN
        # — the reopened handle re-serves the same stale buffer and the freeze detector
        # re-trips forever. Count consecutive failed reopens so we can wait longer each
        # time (let the driver actually release the device) instead of tight-looping.
        self._reopen_fails = 0

        # ACTUAL capture size (the camera may clamp our request to a supported
        # mode). This is what gets published as cam_w/cam_h and what the camera's
        # intrinsics MUST have been calibrated at — the fusion node hard-errors on
        # a mismatch. So always calibrate at the size printed here.
        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f"[{name}] camera {camera} opened at {aw}x{ah} "
            f"-> /hand/cam_{name}/landmarks   "
            f"(calibrate intrinsics --name {name} at {aw}x{ah})")

        self._landmarker = self._make_landmarker()
        self._start_ms = int(time.time() * 1000)
        self._last_ts = -1   # strictly-monotonic timestamp guard for detect_for_video
        self._lost_streak = 0   # consecutive no-hand frames -> tracker reset trigger
        # Extrinsic for the world-axis overlay — needed by BOTH the --show window
        # and the preview thumbnail, so load it unconditionally.
        self._extr = _load_extrinsic(name)
        if self._extr is None:
            self.get_logger().warn(
                f"[{name}] no calibration found — world-axis overlay disabled.")
        if show:
            self._win = f"cam_{name}"
            cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)

    def _make_landmarker(self):
        """Build a fresh HandLandmarker (VIDEO mode).

        min_hand_detection_confidence is LOW (0.3): in VIDEO mode this gates
        RE-ACQUISITION after the hand leaves frame, and the default 0.5 made the
        hand slow/failing to re-detect on re-entry (the "detection bad only when my
        hand leaves frame" fault). Tracking/presence stay at 0.5 so in-frame
        tracking is unaffected. Rebuildable so a stuck tracker can be reset."""
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return HandLandmarker.create_from_options(options)

    def _reset_landmarker(self) -> None:
        """Recreate the landmarker to force a clean full-frame detection.

        MediaPipe VIDEO mode can get stuck in a 'tracking lost' state that won't
        re-detect a hand that has re-entered the frame. Recreating the landmarker
        (cheap) clears that state so the next frame runs fresh detection. Called
        after a sustained no-hand streak. _start_ms/_last_ts are reset so the new
        instance's timestamps stay monotonic from a fresh zero."""
        try:
            self._landmarker.close()
        except Exception:
            pass
        self._landmarker = self._make_landmarker()
        self._start_ms = int(time.time() * 1000)
        self._last_ts = -1
        self.get_logger().info(f"[{self._name}] tracker reset (re-detecting).")

    def _request_max_res(self) -> tuple[int, int]:
        """Probe a ladder of resolutions and return the highest the camera honours.

        UVC cameras clamp set(FRAME_WIDTH/HEIGHT) to their nearest supported mode,
        so we set each candidate high->low, read back the actual size, and take the
        largest pixel count that stuck. Tries MJPG first (many webcams only expose
        their top resolutions under MJPG, not raw YUY2).
        """
        ladder = [(3840, 2160), (2592, 1944), (2560, 1440), (1920, 1080),
                  (1600, 1200), (1280, 960), (1280, 720), (1024, 768),
                  (960, 720), (640, 480)]
        # Prefer MJPG to unlock high-res modes some cameras hide under raw formats.
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC,
                          cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        best = (0, 0)
        for (w, h) in ladder:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if aw * ah > best[0] * best[1]:
                best = (aw, ah)
        self.get_logger().info(f"[{self._name}] max-res probe -> {best[0]}x{best[1]}")
        return best

    def _reopen_camera(self) -> bool:
        """Reopen the camera after a dropout/wedge, restoring the resolution. Returns
        True on success. Keeps the pipeline alive across a transient USB glitch that
        would otherwise close the handle and exit the node (code 0).

        Hardened for the kernel-level UVC wedge behind a PERSISTENT identical-frame
        freeze: (1) sleep after release() with an escalating backoff so the driver
        actually frees the busy device before we grab it again — reopening instantly
        just re-acquires the same wedged handle; (2) force the V4L2 backend + MJPG so
        we don't silently fall back to a wrong mode; (3) CAP_PROP_BUFFERSIZE=1 so the
        reopened handle can't keep re-serving a queued stale buffer (the freeze)."""
        try:
            self._cap.release()
        except Exception:
            pass
        if self._realsense:
            # Rebuild the RealSense pipeline at the same resolution (SDK reopen).
            try:
                self._cap = _RealSenseCapture(self._cap_w or 640,
                                              self._cap_h or 480, fps=None)
            except Exception:
                self._reopen_fails += 1
                return False
            ok = self._cap.isOpened()
            if not ok:
                self._reopen_fails += 1
            return ok
        # Escalating settle delay: 1st reopen waits ~0.2 s, then grows (0.4, 0.6 …)
        # capped at 2 s, so a stubborn wedge gets progressively longer to clear
        # without the loop spinning on a dead handle.
        settle = min(0.2 * (self._reopen_fails + 1), 2.0)
        time.sleep(settle)
        # Explicit V4L2 backend: a plain VideoCapture(idx) may reattach to the same
        # wedged device node; requesting the backend forces a fresh V4L2 open path.
        self._cap = cv2.VideoCapture(self._camera_idx, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._reopen_fails += 1
            return False
        # Newest frame only — never re-serve a queued stale buffer (the freeze).
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Mirror the initial-open MJPG-before-size ordering so we don't clamp to a
        # low mode (the c0 1080p->480p fallback bug).
        if self._cap_w and self._cap_h:
            try:
                self._cap.set(cv2.CAP_PROP_FOURCC,
                              cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        if self._cap_w:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
        if self._cap_h:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
        # NOTE: do NOT reset _reopen_fails here. A UVC wedge reopens "successfully"
        # then immediately re-freezes; resetting on open-success would keep the
        # backoff pinned at its minimum forever. It's reset only when a genuinely
        # FRESH frame arrives (feed actually recovered — see spin_capture).
        return True

    def spin_capture(self) -> None:
        """Blocking capture+detect+publish loop (this node owns its camera)."""
        fail_streak = 0
        frozen_streak = 0
        last_sample = None
        # Loop on rclpy.ok() only — a closed camera is treated as a transient
        # dropout to recover from, NOT a reason to exit (exiting cleanly tore the
        # whole pipeline down via the supervisor).
        while rclpy.ok():
            if not self._cap.isOpened():
                if not self._reopen_camera():
                    time.sleep(0.2)
                    continue
            # Suppress libjpeg's fd-2 "Corrupt JPEG data" spam during the MJPG
            # decode (harmless; see _suppress_fd_stderr).
            with _suppress_fd_stderr():
                ok, frame = self._cap.read()
            if not ok:
                # A camera can hiccup transiently; don't spin silently forever.
                # Warn periodically and try REOPENING so the node self-recovers
                # (a silent dead camera stalled fusion at 1/2; a closed handle
                # used to exit the node and kill the pipeline).
                fail_streak += 1
                if fail_streak in (30, 150) or fail_streak % 300 == 0:
                    self.get_logger().warn(
                        f"[{self._name}] camera read failing "
                        f"({fail_streak} frames) — reopening.")
                    self._reopen_camera()
                time.sleep(0.005)
                continue
            if fail_streak:
                self.get_logger().info(f"[{self._name}] camera recovered.")
                fail_streak = 0

            # A wedged UVC driver can keep re-serving the SAME buffered frame
            # forever with ok=True — no read ever fails, so the fail_streak path
            # above never fires, but the feed is dead (this is exactly what a
            # "process is busy but the picture never moves" report looks like).
            # Cheap sparse-pixel fingerprint (tiny slice, not a full-frame hash —
            # negligible next to the MediaPipe inference already running every
            # frame) catches a byte-identical repeat; treat it like a read
            # failure and reuse the same reopen recovery.
            sample = frame[::37, ::41].tobytes()
            if last_sample is not None and sample == last_sample:
                frozen_streak += 1
                if frozen_streak in (30, 150) or frozen_streak % 300 == 0:
                    self.get_logger().warn(
                        f"[{self._name}] camera frame FROZEN (identical for "
                        f"{frozen_streak} reads) — reopening "
                        f"(attempt {self._reopen_fails + 1}, "
                        f"settle {min(0.2 * (self._reopen_fails + 1), 2.0):.1f}s).")
                    # Count each freeze-triggered reopen so the settle delay in
                    # _reopen_camera escalates — a UVC wedge reopens "successfully"
                    # but re-freezes, so open-success alone must NOT clear this.
                    self._reopen_fails += 1
                    self._reopen_camera()
                    last_sample = None
                time.sleep(0.005)
                continue
            if frozen_streak:
                self.get_logger().info(
                    f"[{self._name}] camera unfroze after {self._reopen_fails} "
                    f"reopen attempt(s).")
                frozen_streak = 0
            # A genuinely fresh frame arrived — the feed actually recovered, so the
            # escalating reopen backoff can reset.
            self._reopen_fails = 0
            last_sample = sample

            # MediaPipe VIDEO mode REQUIRES strictly increasing timestamps. Wall
            # ms can repeat within a millisecond (fast cameras) -> detect_for_video
            # raises and would kill this node ("worked then stopped"). Use a
            # strictly-monotonic ms counter instead.
            ts = int(time.time() * 1000) - self._start_ms
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self._landmarker.detect_for_video(mp_image, ts)
            except Exception as e:   # noqa: BLE001 — one bad frame must not kill the node
                self.get_logger().error(
                    f"[{self._name}] detect error (skipping frame): {e!r}")
                continue

            h, w = frame.shape[:2]
            best_i, hand_lm = self._pick_right_hand(result)
            # Log detection state transitions so a "doesn't recover after leaving
            # frame" fault is visible: shows whether MediaPipe re-detects the hand.
            detected = hand_lm is not None
            n_any = len(result.hand_landmarks) if result.hand_landmarks else 0
            if detected != getattr(self, "_was_detected", None):
                labels = ([result.handedness[i][0].category_name
                           for i in range(len(result.handedness))]
                          if result.handedness else [])
                self.get_logger().info(
                    f"[{self._name}] hand {'ACQUIRED' if detected else 'lost'} "
                    f"(any-hands: {n_any}, handedness: {labels})")
                self._was_detected = detected
            self._publish(hand_lm, w, h)
            self._maybe_publish_preview(frame, result, best_i)

            # Reset the tracker when MediaPipe detects NO hand for a sustained run,
            # to escape a stuck 'tracking lost' state that won't re-detect a hand
            # that has re-entered the frame (the "bad only after leaving frame"
            # fault). Keyed on any-hands==0 (a stuck tracker returns nothing). The
            # streak (~15 frames @30fps ≈ 0.5 s) avoids resetting on brief blinks;
            # reset once per stuck episode, then wait for re-acquisition.
            if n_any == 0:
                self._lost_streak += 1
                if self._lost_streak == 15:
                    self._reset_landmarker()
            else:
                self._lost_streak = 0

            if self._show:
                self._draw(frame, result, best_i)
                cv2.imshow(self._win, frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            # Let ROS process anything pending without blocking the camera.
            rclpy.spin_once(self, timeout_sec=0.0)

    def _pick_right_hand(self, result):
        """Return (index, landmarks) of the operator's hand, or (None, None).

        Handedness-TOLERANT. MediaPipe's Right/Left label flickers — near frame
        edges and after re-entry the SAME physical hand is often mislabelled
        'Left', so strict `== "Right"` filtering dropped valid detections and the
        fused view went black until a clean re-entry. Since this rig tracks ONE
        hand, we:
          * 0 hands  -> lost.
          * 1 hand   -> take it, WHATEVER the label says (the flicker case).
          * 2+ hands -> prefer the tracked handedness; if none match (both
            mislabelled), fall back to the largest hand (closest = operator).
        `_TRACK_HANDEDNESS` still biases the choice when it's reliable (multiple
        hands), but never rejects the only hand in view.
        """
        if not result.hand_landmarks:
            return None, None
        n = len(result.hand_landmarks)

        if n == 1:
            return 0, result.hand_landmarks[0]

        def label(i):
            return (result.handedness[i][0].category_name
                    if i < len(result.handedness) else "?")

        pref = [i for i in range(n) if label(i) == _TRACK_HANDEDNESS]
        pool = pref if pref else list(range(n))   # fall back to all if none match
        best = max(pool, key=lambda i: _hand_bbox_area(result.hand_landmarks[i]))
        return best, result.hand_landmarks[best]

    def _publish(self, hand_lm, w: int, h: int) -> None:
        data = [0.0] * (4 + 63 + 63)
        data[0] = time.time()
        data[2] = float(w)
        data[3] = float(h)
        if hand_lm is not None:
            data[1] = 1.0
            for j, lm in enumerate(hand_lm):
                base = 4 + j * 3
                data[base + 0] = float(lm.x)
                data[base + 1] = float(lm.y)
                data[base + 2] = float(lm.z)
                # MediaPipe Tasks landmarks expose .visibility / .presence; some
                # builds leave them 0. Default to 1.0 (fully visible) when the
                # model doesn't populate them, so triangulation isn't starved.
                vis = getattr(lm, "visibility", 0.0) or 0.0
                pres = getattr(lm, "presence", 0.0) or 0.0
                vbase = 67 + j * 3
                data[vbase + 0] = float(vis) if vis > 0.0 else 1.0
                data[vbase + 1] = float(pres)
        msg = Float32MultiArray()
        msg.data = data
        self._pub.publish(msg)

    def _maybe_publish_preview(self, frame, result, best_i) -> None:
        """Publish a small JPEG preview thumbnail, throttled, for the combined
        viewer. Downscaled + low-rate so it never competes with tracking. Draws
        the detected skeleton + shared world axes so the panel is informative."""
        now = time.time()
        if now - self._last_preview_t < 1.0 / self._preview_hz:
            return
        self._last_preview_t = now
        h, w = frame.shape[:2]
        scale = self._preview_w / float(w)
        small = cv2.resize(frame, (self._preview_w, max(1, int(h * scale))))
        # Draw skeleton + world axes on the small copy (cheap at thumb size).
        self._draw(small, result, best_i)
        ok, enc = cv2.imencode(".jpg", small,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        m = CompressedImage()
        m.format = "jpeg"
        m.data = enc.tobytes()
        self._preview_pub.publish(m)

    def _draw(self, frame, result, best_i) -> None:
        h, w = frame.shape[:2]
        # Shared world-frame axes (X=red, Y=green, Z=blue) projected through THIS
        # camera's extrinsic. If all cameras' overlays sit on the same physical
        # spot, the extrinsics agree — a live shared-frame calibration check.
        self._draw_world_axes(frame)
        if not result.hand_landmarks:
            return
        for i, lm in enumerate(result.hand_landmarks):
            tracked = (i == best_i)
            bone = (0, 220, 0) if tracked else (90, 90, 90)
            joint = (60, 255, 60) if tracked else (120, 120, 120)
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], bone, 2, cv2.LINE_AA)
            for (px, py) in pts:
                cv2.circle(frame, (px, py), 3, joint, -1, cv2.LINE_AA)

    def _draw_world_axes(self, frame, axis_len: float = 0.05) -> None:
        """Overlay the shared world-frame triad using cv2.projectPoints."""
        if self._extr is None:
            return
        K, dist, rvec, tvec = self._extr
        origin = np.zeros(3)
        axes = np.array([origin,
                         [axis_len, 0, 0],
                         [0, axis_len, 0],
                         [0, 0, axis_len]], float)
        try:
            proj, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
        except cv2.error:
            return
        proj = proj.reshape(-1, 2).astype(int)
        o = tuple(proj[0])
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]   # X red, Y green, Z blue
        for k in range(3):
            cv2.line(frame, o, tuple(proj[k + 1]), colors[k], 2, cv2.LINE_AA)
        cv2.putText(frame, "world", (o[0] + 4, o[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def destroy(self):
        self._cap.release()
        if self._show:
            cv2.destroyAllWindows()
        try:
            self._landmarker.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Per-camera hand landmark publisher")
    ap.add_argument("--camera", type=int, required=True,
                    help="OpenCV camera index for THIS node.")
    ap.add_argument("--name", type=str, required=True,
                    help="Short camera name; topic is /hand/cam_<name>/landmarks. "
                         "Must match the calibration --name for this camera.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max-res", action="store_true",
                    help="Open the camera at its HIGHEST supported resolution "
                         "(probed live). Overrides --width/--height. Remember to "
                         "calibrate this camera's intrinsics at the SAME size.")
    ap.add_argument("--realsense", action="store_true",
                    help="Capture from an Intel RealSense COLOR stream via "
                         "pyrealsense2 instead of cv2.VideoCapture (--camera is then "
                         "ignored; the SDK selects the device). Streams --width x "
                         "--height @ 30fps in BGR. Default 640x480 — the D435I's "
                         "1080p color is only 8fps, too slow for teleop. Calibrate "
                         "this camera's intrinsics at the SAME size.")
    ap.add_argument("--show", action="store_true",
                    help="Show a debug window with detected landmarks.")
    ap.add_argument("--preview-w", type=int, default=640,
                    help="Preview thumbnail width (px) for the combined camera "
                         "window. Larger = bigger camera view, slightly more data.")
    ap.add_argument("--preview-hz", type=float, default=15.0,
                    help="Preview publish rate (Hz). Independent of tracking.")
    args = ap.parse_args()

    rclpy.init()
    node = HandLandmarkNode(args.camera, args.name, args.width, args.height,
                            args.show, max_res=args.max_res,
                            preview_w=args.preview_w, preview_hz=args.preview_hz,
                            realsense=args.realsense)
    try:
        node.spin_capture()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        try:
            node.destroy_node()
        except Exception:
            pass
        # Idempotent: the supervisor may SIGINT us mid-shutdown, and calling
        # rclpy.shutdown() twice raises "rcl_shutdown already called".
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-camera hand-landmark fusion node.

Subscribes to N per-camera landmark topics (from teleop/hand_landmark_node.py),
soft-syncs the latest frame from each, triangulates the 21 MediaPipe landmarks
into the shared world frame (teleop/triangulation.py), and publishes the SAME
183-float /hand/joint_angles message the single-camera publisher emits — so all
downstream teleop (DexPilotController / retargeter / arm IK) is unchanged.

The number of cameras is configurable: pass --cameras <name...> with 2 or more
names. Each name must match:
  * the --name used by that camera's hand_landmark_node (topic /hand/cam_<name>/…)
  * the calibration files camera_intrinsics_<name>.json / camera_extrinsics_<name>.json
    in calibration/ (produced by charuco_calibration.py --name <name>).
For backward compatibility, a camera with no _<name> calibration file falls back
to the unsuffixed camera_intrinsics.json / camera_extrinsics.json.

World frame: every camera's extrinsic is solved against the SAME board at the
SAME origin, so all projection matrices share one world frame. We then apply the
board->world Z-up remap (WORLD_FROM_BOARD) so the published skeleton is Z-up like
MuJoCo — matching what the single-camera absolute path already publishes.

Usage:
    python teleop/hand_fusion_node.py --cameras c0 c1 rs
    python teleop/hand_fusion_node.py --cameras c0 c1            # 2-camera
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage

import cv2

from triangulation import (CameraModel, triangulate_landmarks,
                           reprojection_errors)
from hand_message import (WORLD_FROM_BOARD, get_euler_angles,
                          get_flexion_angles, get_wrist_orientation_euler,
                          build_message)

_CALIB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration")

N_LM = 21

# Landmark indices for the palm frame / wrist (shared with MediaPipe convention).
_LM_WRIST = 0
_LM_IDX_MCP = 5
_LM_PKY_MCP = 17


# --- One Euro filter (per-channel), same math as the single-camera publisher ---
import math


class _OneEuro:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.2, d_cutoff=1.0):
        self.freq, self.min_cutoff = freq, min_cutoff
        self.beta, self.d_cutoff = beta, d_cutoff
        self._x = None; self._dx = 0.0; self._t = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        if t is not None and self._t is not None and t - self._t > 0:
            self.freq = 1.0 / (t - self._t)
        if t is not None:
            self._t = t
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        a_d = self._alpha(self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff)
        self._x = a * x + (1 - a) * self._x
        return self._x

    def reset(self):
        self._x = None; self._dx = 0.0; self._t = None


class _OneEuroArray:
    def __init__(self, n, **kw):
        self._f = [_OneEuro(**kw) for _ in range(n)]

    def __call__(self, x, t=None):
        return np.array([f(float(v), t) for f, v in zip(self._f, x)])

    def reset(self):
        for f in self._f:
            f.reset()


def _load_camera(name: str) -> CameraModel:
    """Build a CameraModel for `name`, applying the board->world Z-up remap.

    Loads camera_intrinsics_<name>.json + camera_extrinsics_<name>.json, falling
    back to the unsuffixed files if the per-name ones are absent. The remap folds
    WORLD_FROM_BOARD into R (and t) so all cameras project the SAME Z-up world
    frame — consistent with the single-camera publisher's absolute path.
    """
    intr_path = os.path.join(_CALIB_DIR, f"camera_intrinsics_{name}.json")
    extr_path = os.path.join(_CALIB_DIR, f"camera_extrinsics_{name}.json")
    if not os.path.exists(intr_path):
        intr_path = os.path.join(_CALIB_DIR, "camera_intrinsics.json")
    if not os.path.exists(extr_path):
        extr_path = os.path.join(_CALIB_DIR, "camera_extrinsics.json")

    with open(intr_path) as f:
        intr = json.load(f)
    with open(extr_path) as f:
        extr = json.load(f)

    K = np.array(intr["camera_matrix"], float)
    dist = np.array(intr["dist_coeffs"], float)
    R_cam_board = np.array(extr["R_cam_world"], float)   # world(board) -> cam
    t_cam_board = np.array(extr["t_cam_world"], float)   # board origin in cam

    # We want P mapping the Z-up WORLD frame -> pixels. A board point relates to a
    # world point by p_board = WORLD_FROM_BOARD^T @ p_world (WORLD_FROM_BOARD maps
    # board->world). Since p_cam = R_cam_board @ p_board + t, substituting gives
    #   p_cam = (R_cam_board @ WFB^T) @ p_world + t.
    # So R_cam_world = R_cam_board @ WORLD_FROM_BOARD^T; t is unchanged (the world
    # origin coincides with the board origin, only axes are remapped).
    R_cam_world = R_cam_board @ WORLD_FROM_BOARD.T
    cam = CameraModel(K, dist, R_cam_world, t_cam_board, name)
    # Resolution the intrinsics were calibrated at. K (fx, cx, ...) is only valid
    # at THIS resolution — if the camera streams a different size at runtime, every
    # projected ray is mis-scaled (a steady, motion-independent triangulation
    # error). The fusion node asserts the runtime frame size matches this.
    cam.calib_size = tuple(intr.get("image_size", [None, None]))  # (w, h)
    return cam


class HandFusionNode(Node):
    def __init__(self, names: list[str], sync_window: float, vis_thresh: float,
                 min_views: int, reproj_warn: float, show: bool = False):
        super().__init__("hand_fusion")
        self._names = names
        self._sync_window = sync_window
        self._vis_thresh = vis_thresh
        self._min_views = min_views
        self._reproj_warn = reproj_warn

        # Optional 3D viewer of the fused skeleton. Created on the MAIN thread in
        # main() (all HighGUI + waitKey must stay off the ROS executor thread, or
        # it starves the camera subscriptions). Here we only record the request
        # and the latest render state the main-thread viewer will draw.
        self._show = show
        self._viewer = None
        self._view_state = (None, "starting…")

        self._cams = [_load_camera(n) for n in names]
        for cam in self._cams:
            self.get_logger().info(
                f"[fusion] camera '{cam.name}' loaded (centre in world = "
                f"{np.array2string(cam.C, precision=3)})")

        # Latest per-camera frame: (stamp, uv[21,2] pixels, vis[21], present).
        self._latest: dict[str, tuple] = {}
        # Latest decoded preview thumbnail per camera (for the combined viewer).
        self._preview: dict[str, np.ndarray] = {}
        # Reentrant group so subscription callbacks run CONCURRENTLY with the fuse
        # timer under the MultiThreadedExecutor — otherwise a slow triangulation in
        # the timer blocks _on_cam (default mutually-exclusive group), backing up
        # the queues and staling `_latest` ("waiting 0/2"). Dict writes/reads are
        # atomic under the GIL and the timer snapshots whole tuples, so no lock.
        from rclpy.callback_groups import ReentrantCallbackGroup
        cbg = ReentrantCallbackGroup()
        for n in names:
            self.create_subscription(
                Float32MultiArray, f"/hand/cam_{n}/landmarks",
                lambda msg, nm=n: self._on_cam(nm, msg), 10, callback_group=cbg)
            if show:
                self.create_subscription(
                    CompressedImage, f"/hand/cam_{n}/preview",
                    lambda msg, nm=n: self._on_preview(nm, msg), 2,
                    callback_group=cbg)

        self._pub = self.create_publisher(
            Float32MultiArray, "/hand/joint_angles", 10)

        # One Euro on the FUSED output (moved here from the single-cam publisher).
        self._f_world = _OneEuroArray(63, freq=30.0, min_cutoff=1.5, beta=0.2)
        self._f_wrist = _OneEuroArray(6, freq=30.0, min_cutoff=0.5, beta=0.15)
        # ABSOLUTE-skeleton filter for the 3D viewer. The published message uses
        # wrist-relative filtered points; the viewer shows the skeleton at its true
        # world pose, so it needs its own filter on the absolute points (otherwise
        # it displayed RAW jitter while the message was already smoothed).
        self._f_view = _OneEuroArray(63, freq=30.0, min_cutoff=1.2, beta=0.15)

        # Fuse + publish on a timer (decoupled from any single camera's rate).
        # Same reentrant group so it never blocks the subscription callbacks.
        self.create_timer(1.0 / 60.0, self._fuse_and_publish, callback_group=cbg)
        self._last_ok_log = 0.0
        # Per-camera arrival counter: a corrupted/dropped frame in the landmark
        # node means NO message published that tick, so a low arrival rate here is
        # the direct symptom of USB/MJPG frame loss. Compared against wall time in
        # the health log to report each camera's effective fps.
        self._rx_count = {n: 0 for n in names}
        self._rx_since = time.time()
        self.get_logger().info(
            f"[fusion] {len(names)} cameras {names}; publishing /hand/joint_angles "
            f"(sync window {sync_window*1e3:.0f} ms, min_views {min_views}).")

    def _on_preview(self, name: str, msg: CompressedImage) -> None:
        """Decode a camera's JPEG preview thumbnail for the combined viewer."""
        try:
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                self._preview[name] = img
        except Exception:
            pass

    def _on_cam(self, name: str, msg: Float32MultiArray) -> None:
        d = msg.data
        if len(d) < 130:
            return
        self._rx_count[name] = self._rx_count.get(name, 0) + 1
        # Stamp with OUR receive time, NOT the transported d[0]. d[0] was a full
        # Unix epoch (~1.78e9) crammed into a float32 (Float32MultiArray) — float32
        # has ~7 significant digits, so an epoch quantises to ±128 s of error. That
        # made frames read as tens of seconds stale/future at random -> intermittent
        # "waiting 0/2" even at 30 fps. Receive-time is the correct clock for a
        # staleness check anyway (we care when WE got it), and it's our own process
        # clock so cross-process float precision is irrelevant.
        stamp = time.time()
        present = float(d[1]) > 0.5
        w = float(d[2]) or 1.0
        h = float(d[3]) or 1.0
        # Guard: runtime frame size MUST match the intrinsics' calibration size,
        # else K is mis-scaled and every ray from this camera is wrong (steady
        # triangulation error). Warn once per camera; a mismatch is a setup error
        # (calibrate and capture at the same resolution).
        cam = next((c for c in self._cams if c.name == name), None)
        if cam is not None and getattr(cam, "calib_size", (None, None))[0]:
            cw, ch = cam.calib_size
            if (int(w), int(h)) != (int(cw), int(ch)):
                if name not in getattr(self, "_res_warned", set()):
                    self._res_warned = getattr(self, "_res_warned", set())
                    self._res_warned.add(name)
                    self.get_logger().error(
                        f"[fusion] camera '{name}' streams {int(w)}x{int(h)} but its "
                        f"intrinsics were calibrated at {int(cw)}x{int(ch)} — its "
                        f"projection is INVALID. Re-run intrinsics --name {name} at "
                        f"{int(w)}x{int(h)}, or launch that camera at {int(cw)}x{int(ch)}.")
        uv = np.zeros((N_LM, 2))
        vis = np.zeros(N_LM)
        if present:
            for j in range(N_LM):
                b = 4 + j * 3
                uv[j, 0] = d[b + 0] * w   # denormalise x
                uv[j, 1] = d[b + 1] * h   # denormalise y
                vis[j] = d[67 + j * 3]
        self._latest[name] = (stamp, uv, vis, present)

    def _synced_views(self):
        """Return per-camera (uv_undistorted, vis) for cameras whose latest frame
        is mutually PAIRWISE-synced and currently detecting a hand; None otherwise.

        Two free-running webcams drift in phase, so a fixed "fresh vs now" test can
        admit one ~0 ms-old frame and one ~sync_window-old frame — triangulating a
        moving hand from images that far apart smears the 3D point (large,
        MOTION-dependent reprojection error). The fix anchors on the MOST RECENT
        detecting camera and admits another camera only if its frame is within
        sync_window of THAT anchor (so the admitted set spans <= sync_window in
        time), not merely within sync_window of wall-clock now. `last_span` (the
        max timestamp spread of the admitted set) is returned for diagnostics.
        """
        now = time.time()
        uv_per_cam = [None] * len(self._cams)
        vis_per_cam = [None] * len(self._cams)
        self._last_span = 0.0

        # Admit a camera if its LATEST frame is detecting AND recent (within
        # `stale` of now). We do NOT hard-reject on pairwise span: high-res cameras
        # have unequal per-frame MediaPipe latency, so their publish timestamps for
        # the same instant drift apart — a strict pairwise window silently dropped
        # the slower camera and stuck fusion at 1/2 even with both hands acquired.
        # The span is still MEASURED (for the health log) so real desync is visible;
        # it just doesn't block. `stale` is generous (a few frames) so a briefly
        # slow camera isn't dropped.
        stale = max(self._sync_window * 4.0, 0.12)   # ~120 ms default admit window
        idx = {cam.name: i for i, cam in enumerate(self._cams)}
        admitted_ts = []
        n_fresh = 0
        for cam in self._cams:
            entry = self._latest.get(cam.name)
            if entry is None or not entry[3]:
                continue                     # no frame yet / not detecting
            stamp, uv, vis, _ = entry
            if now - stamp > stale:
                continue                     # this camera's frame is genuinely old
            i = idx[cam.name]
            uv_per_cam[i] = cam.undistort_pixels(uv)
            vis_per_cam[i] = vis
            admitted_ts.append(stamp)
            n_fresh += 1
        if len(admitted_ts) >= 2:
            self._last_span = max(admitted_ts) - min(admitted_ts)
        return uv_per_cam, vis_per_cam, n_fresh

    def _fuse_and_publish(self) -> None:
        # Only compute + publish here (this runs in the ROS executor thread).
        # The GUI is driven from the MAIN thread (see main()) so cv2.waitKey never
        # blocks the executor and starves the camera subscriptions — the bug that
        # made --show-fused sit on "waiting for 2 views" while data was flowing.
        # Guard the whole compute so one bad frame can't kill the timer (which
        # would freeze fusion entirely — "worked then stopped").
        try:
            pts_for_view, info = self._fuse_once()
        except Exception as e:   # noqa: BLE001
            self.get_logger().error(f"[fusion] fuse error (skipping frame): {e!r}")
            pts_for_view, info = None, "fuse error (see log)"
        # Hold the last good skeleton for a short grace window so the viewer
        # doesn't strobe to black on brief single-frame detection gaps (a camera's
        # 1-frame handedness flicker / miss). After the grace expires, show empty.
        now = time.time()
        if pts_for_view is not None:
            self._view_hold = (pts_for_view, now)
            self._view_state = (pts_for_view, info)
        else:
            held = getattr(self, "_view_hold", None)
            if held is not None and now - held[1] < 0.3:   # 300 ms grace
                self._view_state = (held[0], info + " (holding)")
            else:
                self._view_state = (None, info)

    def _reset_filters(self) -> None:
        """Clear One Euro state when the hand is lost so re-acquisition doesn't
        glitch: a stale _t_prev from seconds ago corrupts the filter's frequency
        estimate on the first frame back, causing a jump."""
        self._f_world.reset()
        self._f_wrist.reset()
        self._f_view.reset()

    def wants_viewer(self) -> bool:
        return self._show

    def render_view(self) -> bool:
        """Draw the two viewer windows from the latest state. MUST be called from
        the MAIN thread (HighGUI). Lazily creates the windows on first call.
        Returns False once EITHER window is closed by the user.

        Two windows by design: an orbitable 3D fused view, and a separate grid of
        all camera preview images (raw frame + landmark overlay). Splitting them
        keeps the camera grid at full preview resolution and the 3D view fast +
        mouse-orbitable."""
        if not self._show:
            return False
        if self._viewer is None:
            from skeleton_viewer import SkeletonViewer, CameraGridWindow
            # Static camera poses (position + look direction) for the 3D overlay:
            # C is the camera centre in world; the optical axis is row 2 of R^T
            # (camera +z in world). Passed once — they don't move.
            cam_poses = [(cam.name, cam.C, cam.R[2, :]) for cam in self._cams]
            self._viewer = SkeletonViewer(cam_poses=cam_poses,
                                          cam_order=[c.name for c in self._cams])
            self._cam_grid = CameraGridWindow([c.name for c in self._cams])
        pts, info = self._view_state
        alive = self._viewer.show(pts, info)
        alive = self._cam_grid.show(self._preview) and alive
        if not alive:
            self._viewer.close()
            self._cam_grid.close()
            self._viewer = None
            self._show = False
            return False
        return True

    def _fuse_once(self):
        """Do one triangulate+publish; return (fused_pts_or_None, info_str) so the
        timer can also drive the optional 3D viewer regardless of hand presence."""
        uv_per_cam, vis_per_cam, n_fresh = self._synced_views()
        if n_fresh < self._min_views:
            self._reset_filters()
            # Show per-camera state so a stuck 1/2 is diagnosable: which camera is
            # detecting, and how old its last frame is.
            now = time.time()
            parts = []
            for cam in self._cams:
                e = self._latest.get(cam.name)
                if e is None:
                    parts.append(f"{cam.name}:NEVER-RX")
                else:
                    age = (now - e[0]) * 1e3
                    parts.append(f"{cam.name}:{'det' if e[3] else 'nohand'}"
                                 f"/{age:.0f}ms")
            msg = f"waiting {n_fresh}/{self._min_views} [{' '.join(parts)}]"
            # ALSO print to console every 2 s so a stuck state is visible without
            # reading the viewer text. (The success path logs separately.)
            if now - getattr(self, "_last_wait_log", 0.0) > 2.0:
                self.get_logger().warn(f"[fusion] {msg}")
                self._last_wait_log = now
            return None, msg

        pts, ok = triangulate_landmarks(
            self._cams, uv_per_cam, vis_per_cam,
            n_landmarks=N_LM, vis_thresh=self._vis_thresh,
            min_views=self._min_views)

        # Require the landmarks the downstream math depends on: wrist + the two
        # MCPs that define the palm frame, plus the fingertips for pinch. If the
        # core palm landmarks are missing, skip this frame (hand effectively lost).
        core = [_LM_WRIST, _LM_IDX_MCP, _LM_PKY_MCP]
        if not all(ok[i] for i in core):
            self._reset_filters()
            return None, "hand lost (core landmarks missing)"

        # Fill any un-triangulated landmark by holding its last good value so the
        # 21x3 block stays complete (rare gaps shouldn't blank the whole hand).
        pts = self._fill_gaps(pts, ok)

        # --- Build the 183-float message from the fused 3D skeleton ---
        # World landmarks block: metric, but the downstream retargeter's palm
        # frame is built relative to the WRIST, so re-origin at the wrist to match
        # the single-camera "wrist ~ origin" convention for world landmarks.
        world_lm = pts - pts[_LM_WRIST]

        # Image landmarks block: the arm palm-normal is built from these. We don't
        # have a single canonical image here (N cameras); reuse the triangulated
        # 3D wrist-relative skeleton projected to a stable pseudo-image is
        # overkill — instead feed the metric wrist-relative points, which are more
        # stable than any single view's image LM (the whole reason for triangulating).
        #
        # BASIS: the single-camera publisher's image LM live in the IMAGE basis
        # {x-right, y-DOWN, z-pseudodepth}; the consumer (dexpilot_controller)
        # builds palm_R from them and then applies a fixed change-of-basis
        # C = diag([1,-1,-1]) (image -> world) to get the palm frame into world
        # axes. Our fused points are ALREADY in the world basis (y-UP), so handing
        # them over as-is makes the consumer double-apply C and FLIP the palm
        # orientation. Pre-apply C^-1 == C (it's an involution) here so the
        # consumer's C cancels back to world. Keeps image_lm's declared "image
        # basis" contract and leaves the single-cam path untouched. C is defined
        # locally (NOT reused from WORLD_FROM_BOARD) — they're numerically equal
        # today but are conceptually distinct change-of-basis matrices.
        _C_IMG_WORLD = np.diag([1.0, -1.0, -1.0])   # image basis <-> world basis (involution)
        image_lm = world_lm @ _C_IMG_WORLD          # world -> image basis (C^-1 == C)

        euler = get_euler_angles(pts)          # per-joint Euler from metric 3D
        flexion = get_flexion_angles(world_lm) # wrist-relative metric
        wrist_euler = get_wrist_orientation_euler(pts)  # palm plane from 3D

        # Wrist position: the triangulated wrist in the shared world frame (metres).
        wrist_pos = pts[_LM_WRIST].copy()

        t = time.time()
        wrist_state = self._f_wrist(
            np.concatenate([wrist_pos, wrist_euler]), t)
        world_state = self._f_world(world_lm.ravel(), t).reshape(N_LM, 3)

        cfg = build_message(
            wrist_state[:3], wrist_state[3:], euler, flexion,
            world_state, image_lm)
        msg = Float32MultiArray()
        msg.data = [float(v) for v in cfg]
        self._pub.publish(msg)

        # Periodic health log: reprojection error flags bad calibration.
        # `span` is the time spread of the fused frame set — if reproj is high but
        # span is a large fraction of sync_window, the cause is temporal desync
        # (tighten --sync-window), NOT calibration. High reproj with span~0ms is a
        # genuine calibration/extrinsics problem.
        if t - self._last_ok_log > 2.0:
            errs = reprojection_errors(self._cams, uv_per_cam, pts)
            med = float(np.nanmedian(errs))
            mx = float(np.nanmax(errs))
            span_ms = getattr(self, "_last_span", 0.0) * 1e3
            # Effective per-camera frame rate since the last log. MJPG corruption
            # in the landmark node drops frames -> this fps sags below the camera's
            # nominal rate. A camera far below its peers is the bandwidth-starved one.
            dt = max(1e-3, t - self._rx_since)
            fps = {n: self._rx_count[n] / dt for n in self._rx_count}
            fps_str = " ".join(f"{n}:{fps[n]:.0f}" for n in self._rx_count)
            for n in self._rx_count:
                self._rx_count[n] = 0
            self._rx_since = t
            if med > self._reproj_warn:
                cause = ("desync? tighten --sync-window" if span_ms > 8.0
                         else "check extrinsics/intrinsics")
                flag = f"  <-- HIGH ({cause})"
            else:
                flag = ""
            # max reproj matters for corruption: a corrupt frame yields a garbled
            # landmark that spikes the MAX while the median stays low. med low +
            # max spiking + one camera's fps sagging => USB/MJPG corruption is
            # hurting tracking (move that camera to its own bus, or lower its res).
            self.get_logger().info(
                f"[fusion] {n_fresh} views, {int(ok.sum())}/{N_LM} LM, "
                f"reproj med {med:.1f}px max {mx:.1f}px, span {span_ms:.0f}ms, "
                f"fps[{fps_str}]{flag}")
            self._last_ok_log = t
            self._view_info = (f"{n_fresh} views  {int(ok.sum())}/{N_LM} LM  "
                               f"reproj {med:.1f}px")

        # Absolute fused world points drive the 3D viewer (not the wrist-relative
        # world_lm the message uses) so the skeleton sits at its true world pose.
        # One-Euro-smoothed so the display isn't the raw (jittery) triangulation.
        pts_view = self._f_view(pts.ravel(), t).reshape(N_LM, 3)
        return pts_view, getattr(self, "_view_info", "")

    def _fill_gaps(self, pts: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """Hold last-good value for landmarks not triangulated this frame."""
        if not hasattr(self, "_last_pts"):
            self._last_pts = None
        out = pts.copy()
        if self._last_pts is not None:
            for j in range(N_LM):
                if not ok[j]:
                    out[j] = self._last_pts[j]
        else:
            # First frame with a gap: fill from the wrist so nothing is NaN.
            for j in range(N_LM):
                if not ok[j]:
                    out[j] = pts[_LM_WRIST] if ok[_LM_WRIST] else 0.0
        self._last_pts = out.copy()
        return out


def main():
    ap = argparse.ArgumentParser(description="Multi-camera hand fusion node")
    ap.add_argument("--cameras", nargs="+", required=True,
                    help="Camera names (>=2), matching hand_landmark_node --name "
                         "and calibration --name. E.g. --cameras c0 c1 rs")
    ap.add_argument("--sync-window", type=float, default=0.033,
                    help="Max staleness [s] for a camera frame to join a fused "
                         "solve (default 33 ms ~= one 30 fps frame).")
    ap.add_argument("--vis-thresh", type=float, default=0.3,
                    help="Per-view visibility above which a landmark is used.")
    ap.add_argument("--min-views", type=int, default=2,
                    help="Minimum simultaneous views to triangulate a landmark.")
    ap.add_argument("--reproj-warn", type=float, default=8.0,
                    help="Median reprojection error [px] above which to warn.")
    ap.add_argument("--show", action="store_true",
                    help="Open a 3D viewer of the fused skeleton on a black window "
                         "(orbit with the mouse; shows the world-frame axes).")
    args = ap.parse_args()

    if len(args.cameras) < 2:
        ap.error("need at least 2 cameras for triangulation")

    rclpy.init()
    node = HandFusionNode(args.cameras, args.sync_window, args.vis_thresh,
                          args.min_views, args.reproj_warn, show=args.show)

    if not node.wants_viewer():
        # Headless: plain spin on this thread.
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    # With the viewer: spin ROS in a BACKGROUND thread so the camera
    # subscriptions keep flowing, and run the GUI (cv2.waitKey) on the MAIN
    # thread. Doing the GUI on the executor thread starves the subscriptions —
    # that was the "waiting for 2 views" hang with --show-fused.
    import threading
    stop = threading.Event()

    # Use a MultiThreadedExecutor spinning CONTINUOUSLY so callbacks drain as fast
    # as they arrive. The old spin_once(timeout=0.05) processed only ONE callback
    # per call and could block 50 ms — with ~90 msgs/s across 4 subscriptions (2
    # landmarks@30Hz + 2 previews@15Hz) the queues backed up, `_latest` went stale,
    # and fusion showed "waiting 0/2" even with the hand detected. `executor.spin()`
    # in a background thread keeps `_latest` fresh; the GUI runs on the main thread.
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    def _spin():
        try:
            executor.spin()          # drains continuously until shutdown
        except Exception as e:       # noqa: BLE001 — resilience is the point
            node.get_logger().error(f"[fusion] executor error: {e!r}")

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()
    try:
        while rclpy.ok():
            if not node.render_view():   # window closed by user
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        try:
            executor.shutdown()
        except Exception:
            pass
        spin_thread.join(timeout=1.0)
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

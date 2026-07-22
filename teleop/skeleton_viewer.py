"""Minimal orbitable 3D skeleton viewer on a black OpenCV window (no extra deps).

Renders a set of 3D world-frame points (the triangulated hand) as a skeleton via
a simple pinhole projection you can orbit with the mouse or arrow keys. Kept
ROS-free so the fusion node can own one and feed it fused landmarks each frame.

Controls (while the window is focused):
    mouse drag        orbit azimuth / elevation
    scroll / +,-      zoom (dolly the virtual camera)
    r                 reset view
    q / ESC           request close (returns False from show())
"""
from __future__ import annotations

import numpy as np
import cv2

from hand_message import HAND_CONNECTIONS


class SkeletonViewer:
    def __init__(self, name: str = "multicam hand tracking",
                 size: tuple[int, int] = (720, 720),
                 f: float = 700.0,
                 cam_poses: list | None = None,
                 cam_order: list | None = None):
        self._name = name
        self._W, self._H = size           # 3D panel size
        self._f = f                       # virtual focal length (px)
        self._az = np.deg2rad(35.0)       # azimuth
        self._el = np.deg2rad(-20.0)      # elevation
        self._dist = 0.9                  # camera distance from target (m)
        self._target = np.array([0, 0, 0.4])   # look-at (hand sits ~0.4 m out)
        # Orbit pole / screen-up reference. Default Z-UP: world +Z points toward
        # the TOP of the screen (MuJoCo-consistent). Press 'z' to toggle Z-down.
        self._world_up = np.array([0.0, 0.0, 1.0])
        self._drag = None
        # Static physical-camera poses to render in 3D: [(name, C_world, axis)].
        self._cam_poses = cam_poses or []
        self._cam_order = cam_order or [p[0] for p in self._cam_poses]
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._name, self._W, self._H)
        cv2.setMouseCallback(self._name, self._on_mouse)

    # -- interaction --------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            dx, dy = x - self._drag[0], y - self._drag[1]
            self._az += dx * 0.01
            self._el = float(np.clip(self._el + dy * 0.01,
                                     -np.pi / 2 + 0.05, np.pi / 2 - 0.05))
            self._drag = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._dist *= 0.9 if flags > 0 else 1.1

    def _handle_key(self, key: int) -> bool:
        if key in (ord('q'), 27):
            return False
        if key == ord('r'):
            self._az, self._el, self._dist = np.deg2rad(35), np.deg2rad(-20), 0.9
            self._world_up = np.array([0.0, 0.0, 1.0])   # back to Z-up
        elif key == ord('z'):
            self._world_up = -self._world_up               # toggle Z-down / Z-up
        elif key in (ord('+'), ord('=')):
            self._dist *= 0.9
        elif key in (ord('-'), ord('_')):
            self._dist *= 1.1
        return True

    # -- projection ---------------------------------------------------------
    def _view_matrix(self):
        """Orbit camera pose: position on a sphere around target, looking in.

        The orbit pole is self._world_up. For the default Z-DOWN view it is
        [0,0,-1] (world +Z points toward the bottom of the screen), matching the
        physical setup where the camera looks down and +Z goes into the table."""
        pole = self._world_up / (np.linalg.norm(self._world_up) + 1e-9)
        # Build an orthonormal basis (e0, e1) spanning the plane perpendicular to
        # the pole, so azimuth sweeps around the pole and elevation tilts toward it.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, pole)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e0 = np.cross(pole, ref); e0 /= np.linalg.norm(e0) + 1e-9
        e1 = np.cross(pole, e0)
        ca, sa = np.cos(self._az), np.sin(self._az)
        ce, se = np.cos(self._el), np.sin(self._el)
        # camera position on the orbit sphere (world): azimuth in the e0/e1 plane,
        # elevation toward the pole.
        offset = self._dist * (ce * (ca * e0 + sa * e1) + se * pole)
        eye = self._target + offset
        fwd = self._target - eye
        fwd /= np.linalg.norm(fwd) + 1e-9
        # Screen-up follows the pole (so 'up' on screen tracks the world up axis).
        right = np.cross(fwd, pole); right /= np.linalg.norm(right) + 1e-9
        up = np.cross(right, fwd)
        R = np.vstack([right, up, fwd])   # world -> camera
        return R, eye

    def _project(self, pts_world: np.ndarray):
        """World points -> image pixels (+ a depth for painter's ordering)."""
        R, eye = self._view_matrix()
        cam = (R @ (pts_world - eye).T).T          # (N,3) in camera frame
        z = cam[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = self._f * cam[:, 0] / z + self._W / 2
            v = -self._f * cam[:, 1] / z + self._H / 2
        px = np.stack([u, v], axis=1)
        return px, z

    # -- render -------------------------------------------------------------
    def show(self, pts_world: np.ndarray | None, info: str = "") -> bool:
        """Draw the 3D fused view (orbitable). Camera previews live in a SEPARATE
        window (CameraGridWindow) so this stays fast and mouse-orbitable.
        pts_world is (21,3) or None. Returns False to close."""
        img = self._render_3d(pts_world, info)
        cv2.imshow(self._name, img)
        return self._handle_key(cv2.waitKey(1) & 0xFF)

    def _render_3d(self, pts_world, info):
        img = np.zeros((self._H, self._W, 3), np.uint8)
        self._draw_world_axes(img)
        self._draw_cameras(img)
        if pts_world is not None and np.all(np.isfinite(pts_world)):
            px, z = self._project(pts_world)
            ok = z > 1e-3
            # Dashed origin -> wrist guide with the wrist's world coordinates.
            self._draw_wrist_vector(img, pts_world[0])
            for a, b in HAND_CONNECTIONS:
                if ok[a] and ok[b]:
                    pa = tuple(px[a].astype(int)); pb = tuple(px[b].astype(int))
                    cv2.line(img, pa, pb, (0, 220, 0), 2, cv2.LINE_AA)
            for j in range(len(px)):
                if ok[j]:
                    p = tuple(px[j].astype(int))
                    tip = j in (0, 4, 8, 12, 16, 20)
                    cv2.circle(img, p, 5 if tip else 3,
                               (60, 200, 255) if tip else (60, 255, 60),
                               -1, cv2.LINE_AA)
        cv2.putText(img, "X red  Y green  Z blue   drag orbit  z flip up  r reset",
                    (10, self._H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        if info:
            cv2.putText(img, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)
        return img

    def _draw_cameras(self, img):
        """Render each physical camera as a marker + short optical-axis line and
        a label, at its calibrated world position — so you can see the rig geometry
        around the hand and confirm the extrinsics place cameras sensibly."""
        for name, C, axis in self._cam_poses:
            C = np.asarray(C, float)
            axis = np.asarray(axis, float)
            pts = np.vstack([C, C + 0.06 * axis])   # camera centre + a look stub
            px, z = self._project(pts)
            if not np.all(z > 1e-3):
                continue
            c = tuple(px[0].astype(int))
            tip = tuple(px[1].astype(int))
            cv2.line(img, c, tip, (255, 180, 60), 1, cv2.LINE_AA)   # optical axis
            cv2.drawMarker(img, c, (255, 180, 60), cv2.MARKER_SQUARE, 8, 1)
            cv2.putText(img, name, (c[0] + 6, c[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 120), 1,
                        cv2.LINE_AA)

    def _draw_world_axes(self, img, length: float = 0.1):
        origin = np.zeros(3)
        axes = np.array([origin, [length, 0, 0], [0, length, 0], [0, 0, length]])
        px, z = self._project(axes)
        if not np.all(z > 1e-3):
            return
        o = tuple(px[0].astype(int))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for k in range(3):
            cv2.line(img, o, tuple(px[k + 1].astype(int)), colors[k], 2, cv2.LINE_AA)

    def _dashed_line3d(self, img, a_world, b_world, color, dash_m: float = 0.02,
                       thickness: int = 1):
        """Draw a dashed 3D segment a->b by projecting evenly-spaced sub-points.

        Dashing in WORLD space (not screen space) so the dash length reads as a
        real distance and stays consistent as you orbit. Segments behind the
        camera (z<=0) are skipped."""
        a_world = np.asarray(a_world, float)
        b_world = np.asarray(b_world, float)
        L = float(np.linalg.norm(b_world - a_world))
        if L < 1e-6:
            return
        n = max(2, int(L / dash_m) * 2)                 # on/off pairs
        ts = np.linspace(0, 1, n)
        pts = a_world[None, :] + ts[:, None] * (b_world - a_world)[None, :]
        px, z = self._project(pts)
        for i in range(0, n - 1, 2):                    # draw every other span
            if z[i] > 1e-3 and z[i + 1] > 1e-3:
                cv2.line(img, tuple(px[i].astype(int)),
                         tuple(px[i + 1].astype(int)), color, thickness, cv2.LINE_AA)

    def _draw_wrist_vector(self, img, wrist_world):
        """Dashed guides from the world origin to the wrist, broken into X/Y/Z
        legs, with per-axis distance labels and the total range.

        Reads the wrist position off the world frame: an L-path origin -> (x,0,0)
        -> (x,y,0) -> (x,y,z)=wrist, each leg dashed in its axis colour (X red,
        Y green, Z blue), plus a faint straight dashed line origin->wrist."""
        w = np.asarray(wrist_world, float)
        ox, oy, oz = 0.0, 0.0, 0.0
        # Axis legs of the L-path (each along one world axis).
        legs = [
            (np.array([ox, oy, oz]),  np.array([w[0], oy,   oz]),   (0, 0, 255)),   # X
            (np.array([w[0], oy, oz]), np.array([w[0], w[1], oz]),   (0, 255, 0)),   # Y
            (np.array([w[0], w[1], oz]), np.array([w[0], w[1], w[2]]), (255, 0, 0)), # Z
        ]
        for a, b, c in legs:
            self._dashed_line3d(img, a, b, c, thickness=1)
        # Faint direct origin->wrist line.
        self._dashed_line3d(img, np.zeros(3), w, (150, 150, 150), thickness=1)

        # Per-axis midpoint labels (project the leg midpoint).
        def _label(mid_world, text, color):
            px, z = self._project(mid_world[None, :])
            if z[0] > 1e-3:
                p = tuple(px[0].astype(int))
                cv2.putText(img, text, (p[0] + 4, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        _label(np.array([w[0] / 2, 0, 0]), f"X {w[0]*100:+.1f}cm", (80, 80, 255))
        _label(np.array([w[0], w[1] / 2, 0]), f"Y {w[1]*100:+.1f}cm", (80, 255, 80))
        _label(np.array([w[0], w[1], w[2] / 2]), f"Z {w[2]*100:+.1f}cm", (255, 160, 80))

        # Total distance in the top-left info stack.
        dist = float(np.linalg.norm(w))
        cv2.putText(img, f"wrist range from origin: {dist*100:.1f} cm",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200),
                    1, cv2.LINE_AA)

    def close(self):
        try:
            cv2.destroyWindow(self._name)
        except cv2.error:
            pass


class CameraGridWindow:
    """Separate window tiling ALL camera preview images (raw frame + landmark
    overlay) into one auto-arranged grid. Kept independent of the 3D viewer so the
    3D view stays fast and mouse-orbitable, and so the camera views can be shown
    at full preview resolution."""

    def __init__(self, names: list, name: str = "camera views",
                 cell_w: int = 640):
        self._names = list(names)
        self._name = name
        self._cell_w = cell_w
        # Grid shape: near-square. 1->1x1, 2->1x2, 3->1x3, 4->2x2, ...
        n = len(self._names)
        self._cols = 1 if n <= 1 else (2 if n <= 4 else 3)
        self._rows = int(np.ceil(n / self._cols))
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)

    def show(self, previews: dict) -> bool:
        """Tile the latest preview per camera. Returns False if closed (q/ESC)."""
        cells = []
        for name in self._names:
            img = previews.get(name)
            if img is None:
                img = np.zeros((int(self._cell_w * 3 / 4), self._cell_w, 3),
                               np.uint8)
                cv2.putText(img, f"{name}: no preview yet",
                            (12, img.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1,
                            cv2.LINE_AA)
            else:
                # Scale each preview to a common cell width, keep aspect.
                h, w = img.shape[:2]
                s = self._cell_w / float(w)
                img = cv2.resize(img, (self._cell_w, max(1, int(h * s))))
                cv2.putText(img, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (60, 255, 60), 2, cv2.LINE_AA)
            cells.append(img)

        # Pad cells to a uniform height so hstack/vstack line up.
        ch = max(c.shape[0] for c in cells)
        cells = [self._pad_to(c, ch, self._cell_w) for c in cells]
        # Pad the grid to rows*cols with black cells, then tile.
        while len(cells) < self._rows * self._cols:
            cells.append(np.zeros((ch, self._cell_w, 3), np.uint8))
        rows = [np.hstack(cells[r * self._cols:(r + 1) * self._cols])
                for r in range(self._rows)]
        grid = np.vstack(rows)
        cv2.imshow(self._name, grid)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord('q'), 27)

    @staticmethod
    def _pad_to(img, h, w):
        out = np.zeros((h, w, 3), np.uint8)
        ih, iw = img.shape[:2]
        out[:min(ih, h), :min(iw, w)] = img[:min(ih, h), :min(iw, w)]
        return out

    def close(self):
        try:
            cv2.destroyWindow(self._name)
        except cv2.error:
            pass

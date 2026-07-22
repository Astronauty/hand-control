"""Offline sanity tests for teleop/triangulation.py (no ROS, no cameras).

Synthesises a known 3D hand skeleton and several calibrated cameras, projects
the points to each camera's pixels, then checks triangulate_landmarks recovers
the original 3D within tolerance — including with per-view occlusion and noise.

Run:  python teleop/test_triangulation.py
"""
from __future__ import annotations

import numpy as np

from triangulation import (CameraModel, triangulate_landmarks,
                           triangulate_point, reprojection_errors)


def _K(fx=480.0, fy=480.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def _look_at_camera(name, cam_pos_world, target=np.zeros(3),
                    up=np.array([0, 0, 1.0])):
    """Build a CameraModel positioned at cam_pos_world looking at target.

    Returns a CameraModel whose R_cam_world/t_cam_world follow the charuco
    convention (world axes -> camera axes; world origin in camera frame).
    """
    cam_pos_world = np.asarray(cam_pos_world, float)
    # Camera looks along +z (OpenCV convention): z_cam = target - pos.
    z = target - cam_pos_world
    z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    # Rows are the camera axes in world coords -> R_world_cam; transpose for
    # R_cam_world (world -> cam).
    R_world_cam = np.column_stack([x, y, z])
    R_cam_world = R_world_cam.T
    t_cam_world = -R_cam_world @ cam_pos_world   # world origin in camera frame
    return CameraModel(_K(), None, R_cam_world, t_cam_world, name)


def _project(cam: CameraModel, pts: np.ndarray) -> np.ndarray:
    Xh = np.hstack([pts, np.ones((len(pts), 1))])
    x = (cam.P @ Xh.T).T
    return x[:, :2] / x[:, 2:3]


def _synthetic_hand(n=21, seed=0):
    """A blob of points ~10 cm across, 0.4 m in front of the world origin."""
    rng = np.random.default_rng(seed)
    return np.array([0.0, 0.0, 0.4]) + rng.uniform(-0.05, 0.05, size=(n, 3))


def test_perfect_recovery():
    cams = [
        _look_at_camera("c0", [0.0, -0.6, 0.4]),
        _look_at_camera("c1", [0.5, -0.4, 0.4]),
        _look_at_camera("c2", [-0.5, -0.4, 0.4]),
    ]
    hand = _synthetic_hand()
    uv = [_project(c, hand) for c in cams]
    pts, ok = triangulate_landmarks(cams, uv)
    assert ok.all(), "all landmarks should triangulate"
    err = np.max(np.linalg.norm(pts - hand, axis=1))
    assert err < 1e-6, f"noise-free recovery should be exact, got {err:.2e} m"
    print(f"[perfect]   max 3D error = {err:.2e} m  (all {ok.sum()} LM ok)")


def test_noise_recovery():
    cams = [
        _look_at_camera("c0", [0.0, -0.6, 0.4]),
        _look_at_camera("c1", [0.5, -0.4, 0.4]),
        _look_at_camera("c2", [-0.5, -0.4, 0.4]),
    ]
    hand = _synthetic_hand(seed=1)
    rng = np.random.default_rng(42)
    uv = [_project(c, hand) + rng.normal(0, 0.5, size=(21, 2)) for c in cams]
    pts, ok = triangulate_landmarks(cams, uv)
    assert ok.all()
    err = np.median(np.linalg.norm(pts - hand, axis=1))
    # 0.5 px noise at ~0.5 m / f=480 -> sub-mm to low-mm; assert < 5 mm.
    assert err < 5e-3, f"median error under noise too high: {err*1e3:.2f} mm"
    print(f"[noise]     median 3D error = {err*1e3:.3f} mm  (0.5 px noise)")


def test_occlusion_visibility_gating():
    """Camera 2 is 'occluded' for the thumb tip: its uv is garbage but its
    visibility is low, so gating must drop it and still recover from c0+c1."""
    cams = [
        _look_at_camera("c0", [0.0, -0.6, 0.4]),
        _look_at_camera("c1", [0.5, -0.4, 0.4]),
        _look_at_camera("c2", [-0.5, -0.4, 0.4]),
    ]
    hand = _synthetic_hand(seed=2)
    uv = [_project(c, hand).copy() for c in cams]
    vis = [np.ones(21), np.ones(21), np.ones(21)]
    # Corrupt landmark 4 (thumb tip) in c2 and mark it low-visibility.
    uv[2][4] = np.array([10.0, 10.0])
    vis[2][4] = 0.05
    pts, ok = triangulate_landmarks(cams, uv, vis_per_cam=vis, vis_thresh=0.3)
    assert ok[4], "thumb tip should still triangulate from the 2 good views"
    err = np.linalg.norm(pts[4] - hand[4])
    assert err < 1e-6, f"gated recovery should be exact, got {err:.2e} m"
    print(f"[occlusion] thumb-tip error with 1 view gated out = {err:.2e} m")


def test_min_views_drops_single_view():
    cams = [_look_at_camera("c0", [0.0, -0.6, 0.4])]
    hand = _synthetic_hand(seed=3)
    uv = [_project(cams[0], hand)]
    pts, ok = triangulate_landmarks(cams, uv, min_views=2)
    assert not ok.any(), "a single view cannot triangulate any landmark"
    assert np.isnan(pts).all()
    print("[min_views] single view correctly yields zero triangulated LM")


def test_reprojection_health():
    cams = [
        _look_at_camera("c0", [0.0, -0.6, 0.4]),
        _look_at_camera("c1", [0.5, -0.4, 0.4]),
    ]
    hand = _synthetic_hand(seed=4)
    uv = [_project(c, hand) for c in cams]
    pts, ok = triangulate_landmarks(cams, uv)
    errs = reprojection_errors(cams, uv, pts)
    assert np.nanmax(errs) < 1e-3, "clean data should reproject to ~0 px"
    print(f"[reproj]    max reprojection error = {np.nanmax(errs):.2e} px")


if __name__ == "__main__":
    tests = [test_perfect_recovery, test_noise_recovery,
             test_occlusion_visibility_gating, test_min_views_drops_single_view,
             test_reprojection_health]
    for t in tests:
        t()
    print("\nAll triangulation tests passed.")

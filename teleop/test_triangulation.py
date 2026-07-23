"""Offline sanity tests for teleop/triangulation.py (no ROS, no cameras).

Synthesises a known 3D hand skeleton and several calibrated cameras, projects
the points to each camera's pixels, then checks triangulate_landmarks recovers
the original 3D within tolerance — including with per-view occlusion and noise.

Run:  python teleop/test_triangulation.py
"""
from __future__ import annotations

import numpy as np

from triangulation import (CameraModel, triangulate_landmarks,
                           triangulate_point, reprojection_errors,
                           per_camera_reproj)


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
    pts, ok, _ = triangulate_landmarks(cams, uv)
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
    pts, ok, _ = triangulate_landmarks(cams, uv)
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
    pts, ok, _ = triangulate_landmarks(cams, uv, vis_per_cam=vis, vis_thresh=0.3)
    assert ok[4], "thumb tip should still triangulate from the 2 good views"
    err = np.linalg.norm(pts[4] - hand[4])
    assert err < 1e-6, f"gated recovery should be exact, got {err:.2e} m"
    print(f"[occlusion] thumb-tip error with 1 view gated out = {err:.2e} m")


def test_min_views_drops_single_view():
    cams = [_look_at_camera("c0", [0.0, -0.6, 0.4])]
    hand = _synthetic_hand(seed=3)
    uv = [_project(cams[0], hand)]
    pts, ok, _ = triangulate_landmarks(cams, uv, min_views=2)
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
    pts, ok, _ = triangulate_landmarks(cams, uv)
    errs = reprojection_errors(cams, uv, pts)
    assert np.nanmax(errs) < 1e-3, "clean data should reproject to ~0 px"
    print(f"[reproj]    max reprojection error = {np.nanmax(errs):.2e} px")


def test_bad_view_gate_identifies_and_recovers():
    """A 3rd camera whose calibrated pose disagrees with reality (as when its
    extrinsics aren't in the shared board frame) reprojects far worse than the
    two consistent views. per_camera_reproj must single it out, and dropping it
    must restore the clean 3D — the auto-reject the fusion node relies on."""
    c0 = _look_at_camera("c0", [0.0, -0.6, 0.4])
    c1 = _look_at_camera("c1", [0.5, -0.4, 0.4])
    rs_true = _look_at_camera("rs", [-0.5, -0.4, 0.4])
    hand = _synthetic_hand(seed=7)
    # rs's PIXELS come from its true pose, but the model we triangulate with has
    # a corrupted pose (shifted + not looking where it thinks) — a stand-in for
    # extrinsics solved against a different board origin than c0/c1.
    uv = [_project(c0, hand), _project(c1, hand), _project(rs_true, hand)]
    rs_bad = _look_at_camera("rs", [-0.5, -0.1, 0.7])
    cams = [c0, c1, rs_bad]

    pts, ok, _ = triangulate_landmarks(cams, uv)
    per = per_camera_reproj(cams, uv, pts)
    worst = int(np.nanargmax(per))
    assert cams[worst].name == "rs", (
        f"the mis-calibrated view should be worst, got per-cam={per}")
    # DLT spreads a bad view's error across every camera, so the consistent
    # views are lifted too — the bad view need not be 5x worse, only the clear
    # max (and above a sane gate). The exact-recovery check below is the real
    # guarantee that dropping the RIGHT view fixes the solve.
    assert per[worst] > 40.0, f"bad view should breach the 40px gate, got {per}"
    assert per[worst] == max(p for p in per if np.isfinite(p))

    # Drop it and re-triangulate from the two consistent views -> exact recovery.
    uv2 = list(uv); uv2[worst] = None
    pts2, ok2, _ = triangulate_landmarks(cams, uv2)
    err_before = np.nanmedian(np.linalg.norm(pts - hand, axis=1))
    err_after = np.nanmedian(np.linalg.norm(pts2 - hand, axis=1))
    assert err_after < 1e-6 < err_before, (
        f"dropping the bad view should restore exact 3D: "
        f"before={err_before:.4f} m after={err_after:.2e} m")
    print(f"[gate]      bad-view reproj={per[worst]:.0f}px "
          f"(good ~{max(per[0], per[1]):.0f}px); "
          f"3D err {err_before*1e3:.1f}mm -> {err_after*1e3:.2e}mm after drop")


def test_per_landmark_occlusion_reject():
    """The real occlusion case: three cameras all report FULL visibility, but one
    has a fingertip occluded and hallucinates it elsewhere. Visibility can't flag
    it (MediaPipe hands don't populate it), so reject_px must drop that ONE view
    for that ONE landmark by reprojection consensus — keeping it for all others."""
    cams = [
        _look_at_camera("c0", [0.0, -0.6, 0.4]),
        _look_at_camera("c1", [0.5, -0.4, 0.4]),
        _look_at_camera("c2", [-0.5, -0.4, 0.4]),
    ]
    hand = _synthetic_hand(seed=11)
    uv = [_project(c, hand).copy() for c in cams]
    # c2 hallucinates the index fingertip (LM 8) ~40 px away — full visibility.
    TIP = 8
    uv[2][TIP] = uv[2][TIP] + np.array([40.0, -30.0])

    # WITHOUT rejection: the bad view drags the fused fingertip off.
    pts_off, ok_off, drop_off = triangulate_landmarks(cams, uv, reject_px=0.0)
    err_off = np.linalg.norm(pts_off[TIP] - hand[TIP])

    # WITH rejection: c2 is dropped for LM 8 only, recovered from c0+c1 exactly.
    pts, ok, dropped = triangulate_landmarks(cams, uv, reject_px=8.0)
    err_on = np.linalg.norm(pts[TIP] - hand[TIP])

    assert dropped[TIP] == 2, f"c2 should be rejected for the fingertip, got {dropped[TIP]}"
    assert (dropped == -1).sum() == 20, "only the occluded landmark should reject a view"
    assert err_on < 1e-6 < err_off, (
        f"rejection should restore the fingertip: off={err_off*1e3:.1f}mm "
        f"on={err_on*1e3:.2e}mm")
    # Every other landmark still uses all three views and stays exact.
    others = [i for i in range(21) if i != TIP]
    assert np.max(np.linalg.norm(pts[others] - hand[others], axis=1)) < 1e-6
    print(f"[occl-rej]  fingertip off={err_off*1e3:.1f}mm -> {err_on*1e3:.2e}mm; "
          f"dropped view {dropped[TIP]} for LM{TIP} only")


if __name__ == "__main__":
    tests = [test_perfect_recovery, test_noise_recovery,
             test_occlusion_visibility_gating, test_min_views_drops_single_view,
             test_reprojection_health, test_bad_view_gate_identifies_and_recovers,
             test_per_landmark_occlusion_reject]
    for t in tests:
        t()
    print("\nAll triangulation tests passed.")

"""Multi-view triangulation of MediaPipe hand landmarks (pure math, no ROS).

Given N calibrated cameras that each detect the SAME hand's 21 landmarks in 2D
image pixels, reconstruct one metric 21-landmark 3D skeleton in the shared world
frame. Kept ROS-free and MediaPipe-free so it is unit-testable offline against
recorded landmark streams — the only genuinely new logic in the multi-camera
pipeline, so it is de-risked first.

Coordinate frame
  All cameras' extrinsics are solved against the SAME ChArUco board at the SAME
  world origin (see calibration/charuco_calibration.py --name), so every camera's
  projection matrix maps the ONE shared world frame -> that camera's pixels:

      P_k = K_k @ [ R_cam_world_k | t_cam_world_k ]        (3x4)

  where R_cam_world_k, t_cam_world_k are exactly the values charuco extrinsics
  writes as "R_cam_world" / "t_cam_world" (world axes -> camera axes; world
  origin in camera frame). No separate stereo calibration is required.

Triangulation
  For a world point X (homogeneous), each camera contributes the projection
  x_k ~ P_k X. Two rows per view come from x_k × (P_k X) = 0:

      u_k * P_k[2] - P_k[0]
      v_k * P_k[1... -> see below]

  Stacking these rows over all views gives A X = 0, solved by SVD (the smallest
  singular vector). This is the linear DLT method (Hartley & Zisserman 12.2),
  which generalises cleanly from 2 to N views and lets us weight each view by the
  landmark's per-view visibility so an occluded camera's hallucinated 2D point
  does not poison the fit.
"""
from __future__ import annotations

import numpy as np


class CameraModel:
    """One calibrated camera: intrinsics + world->camera extrinsic -> P (3x4).

    Args:
        K:            (3,3) camera matrix (pixels).
        dist:         (k,) OpenCV distortion coeffs, or None. When present, 2D
                      landmarks are undistorted to a normalised ideal-pinhole ray
                      before triangulation (P is then built with K=I-effective via
                      the normalised points). We keep K in P and undistort pixels
                      to *pixel* coordinates on the ideal model instead — simpler
                      and matches how cv2.triangulatePoints expects inputs.
        R_cam_world:  (3,3) world axes -> camera axes (charuco "R_cam_world").
        t_cam_world:  (3,) world origin in camera frame (charuco "t_cam_world").
        name:         label for diagnostics.
    """

    def __init__(self, K, dist, R_cam_world, t_cam_world, name: str = ""):
        self.K = np.asarray(K, float).reshape(3, 3)
        self.dist = None if dist is None else np.asarray(dist, float).ravel()
        self.R = np.asarray(R_cam_world, float).reshape(3, 3)
        self.t = np.asarray(t_cam_world, float).ravel()
        self.name = name
        # Projection matrix P = K [R | t] mapping world (homogeneous) -> pixels.
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])   # 3x4
        self.P = self.K @ Rt                              # 3x4
        # Camera centre in world coords: C = -R^T t (R maps world->cam).
        self.C = -self.R.T @ self.t

    def undistort_pixels(self, uv: np.ndarray) -> np.ndarray:
        """Map raw pixel landmarks to ideal-pinhole pixels matching self.P.

        cv2.undistortPoints with P=K returns pixels on the distortion-free model,
        which is exactly the model self.P (built from K, no distortion) projects
        to. If no distortion is set, returns uv unchanged.

        Args:
            uv: (M,2) pixel coordinates.
        Returns:
            (M,2) undistorted pixel coordinates.
        """
        uv = np.asarray(uv, float).reshape(-1, 1, 2)
        if self.dist is None:
            return uv.reshape(-1, 2)
        import cv2
        out = cv2.undistortPoints(uv, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)


def _dlt_rows(P: np.ndarray, u: float, v: float) -> np.ndarray:
    """Two DLT rows for one view: from x × (P X) = 0 for x = (u, v, 1).

    Rows (Hartley & Zisserman, eq. 12.2):
        u * P[2,:] - P[0,:]
        v * P[2,:] - P[1,:]
    """
    return np.vstack([u * P[2, :] - P[0, :],
                      v * P[2, :] - P[1, :]])


def triangulate_point(cams: list[CameraModel],
                      uvs: list[np.ndarray],
                      weights: list[float] | None = None) -> np.ndarray | None:
    """Triangulate ONE world point from >=2 views by weighted linear DLT.

    Args:
        cams:    list of CameraModel, one per contributing view.
        uvs:     list of (2,) UNDISTORTED pixel coords aligned with cams.
        weights: optional per-view non-negative weights (e.g. visibility). Views
                 with weight <= 0 are dropped. None -> all weight 1.
    Returns:
        (3,) world point, or None if fewer than 2 usable views.
    """
    if weights is None:
        weights = [1.0] * len(cams)
    rows = []
    used = 0
    for cam, uv, w in zip(cams, uvs, weights):
        if w <= 0.0:
            continue
        u, v = float(uv[0]), float(uv[1])
        # Weight scales both rows; sqrt so the squared residual scales by w.
        rows.append(np.sqrt(w) * _dlt_rows(cam.P, u, v))
        used += 1
    if used < 2:
        return None
    A = np.vstack(rows)                 # (2*used, 4)
    # Smallest right singular vector of A is the homogeneous solution.
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None                     # point at infinity — degenerate
    return X[:3] / X[3]


def triangulate_landmarks(cams: list[CameraModel],
                          uv_per_cam: list[np.ndarray | None],
                          vis_per_cam: list[np.ndarray | None] | None = None,
                          n_landmarks: int = 21,
                          vis_thresh: float = 0.3,
                          min_views: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate all N landmarks across all cameras.

    Args:
        cams:        list of CameraModel (all in the shared world frame).
        uv_per_cam:  per-camera (n_landmarks, 2) UNDISTORTED pixels, or None if
                     that camera has no detection this frame.
        vis_per_cam: per-camera (n_landmarks,) visibility in [0,1], or None
                     (treated as all-1 for cameras that DID detect). A camera
                     whose uv is None contributes to no landmark.
        n_landmarks: expected landmark count (21 for MediaPipe hands).
        vis_thresh:  a view is used for a landmark only if its visibility exceeds
                     this (drops occluded/hallucinated points from the fit).
        min_views:   minimum usable views to accept a landmark.
    Returns:
        pts:  (n_landmarks, 3) world points; rows for un-triangulated landmarks
              are NaN.
        ok:   (n_landmarks,) bool — True where the landmark was triangulated.
    """
    if vis_per_cam is None:
        vis_per_cam = [None] * len(cams)

    pts = np.full((n_landmarks, 3), np.nan)
    ok = np.zeros(n_landmarks, bool)

    for j in range(n_landmarks):
        sub_cams, sub_uv, sub_w = [], [], []
        for cam, uv, vis in zip(cams, uv_per_cam, vis_per_cam):
            if uv is None:
                continue
            w = 1.0 if vis is None else float(vis[j])
            if w <= vis_thresh:
                continue
            sub_cams.append(cam)
            sub_uv.append(uv[j])
            sub_w.append(w)
        if len(sub_cams) < min_views:
            continue
        X = triangulate_point(sub_cams, sub_uv, sub_w)
        if X is not None:
            pts[j] = X
            ok[j] = True
    return pts, ok


def reprojection_errors(cams: list[CameraModel],
                        uv_per_cam: list[np.ndarray | None],
                        pts: np.ndarray) -> np.ndarray:
    """Mean per-landmark reprojection error [px] over the cameras that saw it.

    A calibration/triangulation health check: large values flag bad extrinsics
    or a mis-associated landmark. NaN where a landmark was not triangulated or
    no camera reprojects it.

    Args:
        cams:       list of CameraModel.
        uv_per_cam: per-camera (n,2) UNDISTORTED pixels (or None), same as fed to
                    triangulate_landmarks.
        pts:        (n,3) triangulated world points (NaN rows allowed).
    Returns:
        (n,) mean reprojection error in pixels (NaN where undefined).
    """
    n = pts.shape[0]
    errs = np.full(n, np.nan)
    for j in range(n):
        if not np.all(np.isfinite(pts[j])):
            continue
        Xh = np.append(pts[j], 1.0)
        per_view = []
        for cam, uv in zip(cams, uv_per_cam):
            if uv is None:
                continue
            x = cam.P @ Xh
            if abs(x[2]) < 1e-12:
                continue
            proj = x[:2] / x[2]
            per_view.append(np.linalg.norm(proj - uv[j]))
        if per_view:
            errs[j] = float(np.mean(per_view))
    return errs

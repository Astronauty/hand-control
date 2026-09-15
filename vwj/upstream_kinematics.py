"""MuJoCo-backed `robot_model` duck-type for the UPSTREAM VWJ optimizer.

The `vwj_upstream` mode runs Mingrui-Yu/retargeting's `VectorWristJointOptimizerV2`
*verbatim* (a local, un-committed clone under third_party/). That optimizer talks to a
`RobotAdaptor` wrapping a `robot_model`; upstream's is a pinocchio model built from a
URDF. No Gen3 URDF exists here and the installed pinocchio (2.7.0) can't load MJCF, so
this class provides the SAME small `robot_model` interface the optimizer calls, backed
by the existing MuJoCo Gen3+LEAP model via `vwj.kinematics.MujocoKinematics`.

Interface the optimizer + adaptor use (verified against the upstream source):
    robot_model:  dof, joint_limits (dof,2), get_joint_index(name),
                  get_frames_index(names), get_frame_pose_by_id(id)->4x4,
                  get_frame_space_jacobian_by_id(id)->(6,dof) LOCAL_WORLD_ALIGNED,
                  compute_forward_kinematics(qpos_dof), compute_jacobians(qpos_dof),
                  check_joint_dim(q).
    adaptor:      actuated_joints_name, actuated_joints_model_idx, doa,
                  forward_qpos, backward_qpos, backward_jacobian.

We present ONLY the 23 optimized joints as the whole model (dof == doa == 23), so the
adaptor's dof<->doa maps are the identity — the optimizer sees a 23-DOF robot directly.

The critical fidelity point: upstream's `get_frame_space_jacobian` uses pinocchio's
LOCAL_WORLD_ALIGNED reference frame, whose linear rows are the velocity of the frame
ORIGIN in world axes and angular rows are the WORLD angular velocity — exactly what
MuJoCo's `mj_jacSite` returns as (Jp, Jr). So the 6xdof spatial Jacobian is [Jp; Jr],
matching the upstream convention row-for-row (verified numerically in the module test).
"""
from __future__ import annotations

import numpy as np

from vwj.kinematics import MujocoKinematics


class MujocoRobotModel:
    """A pinocchio-`RobotPinocchio`-shaped facade over MujocoKinematics (23 DOF)."""

    def __init__(self, model, opt_joint_names: list[str], frame_names: list[str],
                 wrist_frame: str, site_names: list[str] | None = None,
                 wrist_site: str | None = None):
        # Reuse the validated MuJoCo FK/Jacobian adaptor. `frame_names` are the LOGICAL
        # names the optimizer references (e.g. "if_tip", "wrist"); `site_names` are the
        # matching MJCF sites MuJoCo actually computes (default: frame_names == sites).
        sites = list(site_names) if site_names is not None else list(frame_names)
        wsite = wrist_site if wrist_site is not None else wrist_frame
        self._kin = MujocoKinematics(model, opt_joint_names, sites, wsite)
        self._joint_names = list(opt_joint_names)
        self._frame_names = list(frame_names)
        self._n = self._kin.n_opt
        # Cached per-solve FK/Jacobian state (mirrors pinocchio's data workflow: call
        # compute_* once, then read frame poses/jacobians by id).
        self._pos = None            # (n_frame, 3) world site positions
        self._wrist_R = None        # (3,3) wrist site rotation
        self._Jp = None             # (n_frame, 3, n) translational jacobians
        self._Jr = None             # (n_frame, 3, n) rotational jacobians
        self._have_jac = False

    # -- shape / lookups -----------------------------------------------------------
    @property
    def dof(self) -> int:
        return self._n

    @property
    def joint_limits(self) -> np.ndarray:
        """(dof, 2) lower/upper. Non-finite (continuous joints) -> a wide finite range,
        matching how the clean-room path treats them (the solver needs finite bounds)."""
        lo = np.where(np.isfinite(self._kin.lo), self._kin.lo, -2.0 * np.pi)
        hi = np.where(np.isfinite(self._kin.hi), self._kin.hi, 2.0 * np.pi)
        return np.stack([lo, hi], axis=1)

    def check_joint_dim(self, q) -> None:
        assert len(q) == self._n, f"expected {self._n}-dof q, got {len(q)}"

    def get_joint_index(self, name: str) -> int:
        return self._joint_names.index(name)

    def get_frame_index(self, name: str) -> int:
        return self._frame_names.index(name)

    def get_frames_index(self, names: list[str]) -> list[int]:
        return [self._frame_names.index(n) for n in names]

    # -- FK / Jacobian (pinocchio-style two-step: compute, then read) --------------
    def compute_forward_kinematics(self, qpos_dof: np.ndarray) -> None:
        self._kin.forward(np.asarray(qpos_dof, float))
        self._pos = self._kin.site_positions()
        self._wrist_R = self._kin.data.site_xmat[
            self._kin.site_ids[self._kin.wrist_idx]].reshape(3, 3).copy()
        self._have_jac = False

    def compute_jacobians(self, qpos_dof: np.ndarray) -> None:
        self._kin.forward_with_jac(np.asarray(qpos_dof, float))
        self._pos = self._kin.site_positions()
        self._wrist_R = self._kin.data.site_xmat[
            self._kin.site_ids[self._kin.wrist_idx]].reshape(3, 3).copy()
        self._Jp, self._Jr = self._kin.site_jacobians()
        self._have_jac = True

    def get_frame_pose_by_id(self, frame_id: int) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = self._pos[frame_id]
        # Only the wrist frame's rotation is read by the optimizer; give every frame the
        # wrist rotation as a harmless placeholder (positions are what the link-vec terms
        # use; the wrist term reads pose[:3,:3] only at the wrist index).
        T[:3, :3] = self._wrist_R
        return T

    def get_frame_pose(self, frame_name: str) -> np.ndarray:
        return self.get_frame_pose_by_id(self._frame_names.index(frame_name))

    def get_frame_space_jacobian_by_id(self, frame_id: int) -> np.ndarray:
        """(6, dof) spatial Jacobian in the LOCAL_WORLD_ALIGNED convention: rows 0:3 are
        the frame-origin linear velocity in world axes (= MuJoCo Jp), rows 3:6 the world
        angular velocity (= MuJoCo Jr)."""
        if not self._have_jac:
            raise RuntimeError("call compute_jacobians before get_frame_space_jacobian")
        return np.concatenate([self._Jp[frame_id], self._Jr[frame_id]], axis=0)

    def get_frame_space_jacobian(self, frame_name: str) -> np.ndarray:
        return self.get_frame_space_jacobian_by_id(self._frame_names.index(frame_name))

"""`vwj_upstream` retargeter: drives Mingrui-Yu/retargeting's optimizer VERBATIM.

This is the A/B partner to the clean-room `vwj` mode. Where `VWJRetargeter` runs my own
reimplementation of the objective, this runs the UPSTREAM `VectorWristJointOptimizerV2`
(and its nlopt-SLSQP solver, torch-autograd cost, and quaternion helpers) UNCHANGED, from
a local, un-committed clone under third_party/mingrui_retargeting (its licence is absent,
so it is never vendored into this repo — see third_party/ in .gitignore).

The only substitution is the kinematics backend: upstream's pinocchio `RobotPinocchio`
is replaced by `vwj.upstream_kinematics.MujocoRobotModel`, which exposes the identical
`robot_model` interface backed by the SAME MuJoCo Gen3+LEAP model the simulator runs
(the installed pinocchio can't load MJCF and no Gen3 URDF exists). So `vwj` and
`vwj_upstream` share the FK and differ ONLY in the objective/solver code — the cleanest
possible isolation of "their optimizer vs. my reimplementation".

Exposes the same interface as `VWJRetargeter` (retarget(world_lm, wrist_pos, wrist_R),
reset, poll_config, human_palm_frame_robot_aligned) so it drops into the same controller
path with no changes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import mujoco as mj

from vwj.retargeter import (_ARM_JOINTS, _HAND_JOINTS, _TIP_SITES, _DIP_SITES,
                            _WRIST_SITE, _default_config, _CONFIG_PATH)
from vwj.refvalues import build_ref_values, robot_link_pairs
from vwj.upstream_kinematics import MujocoRobotModel

# Local clone paths (git submodule utils_python provides mr_utils). Never committed.
_UPSTREAM_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "third_party", "mingrui_retargeting")


def _ensure_upstream_on_path() -> None:
    """Put the upstream clone's src/ and its mr_utils submodule on sys.path (once)."""
    src = os.path.join(_UPSTREAM_ROOT, "src")
    utils = os.path.join(_UPSTREAM_ROOT, "third_party", "utils_python")
    if not os.path.isdir(src):
        raise RuntimeError(
            f"vwj_upstream: upstream clone not found at {_UPSTREAM_ROOT}. Clone it:\n"
            f"  git clone https://github.com/Mingrui-Yu/retargeting.git "
            f"{_UPSTREAM_ROOT}\n"
            f"  (cd {_UPSTREAM_ROOT} && git submodule update --init third_party/utils_python)")
    for p in (src, utils):
        if p not in sys.path:
            sys.path.insert(0, p)


# The upstream frame names (from the LEAP profile) mapped to THIS repo's MJCF sites, in
# the exact order build_ref_values / robot_link_pairs lay out the 12 link pairs.
_UPSTREAM_LINK_PAIRS = [
    ("world", "thumb_tip"),
    ("wrist", "thumb_tip"), ("wrist", "if_tip"),
    ("wrist", "mf_tip"), ("wrist", "rf_tip"),
    ("thumb_tip", "if_tip"), ("thumb_tip", "mf_tip"), ("thumb_tip", "rf_tip"),
    ("th_dip", "thumb_tip"), ("if_dip", "if_tip"),
    ("mf_dip", "mf_tip"), ("rf_dip", "rf_tip"),
]
# The optimizer only needs consistent NAMES for origin/task frames + the wrist. We map
# those logical names to MJCF sites for the MuJoCo robot_model.
_NAME2SITE = {
    "thumb_tip": _TIP_SITES[0], "if_tip": _TIP_SITES[1],
    "mf_tip": _TIP_SITES[2], "rf_tip": _TIP_SITES[3],
    "th_dip": _DIP_SITES[0], "if_dip": _DIP_SITES[1],
    "mf_dip": _DIP_SITES[2], "rf_dip": _DIP_SITES[3],
    "wrist": _WRIST_SITE,
}


class VWJUpstreamRetargeter:
    """Whole-arm-hand retargeter using the UPSTREAM optimizer verbatim (MuJoCo FK)."""

    whole_robot = True

    def __init__(self, model: mj.MjModel, n_arm: int = 7, hand_type: str = "right",
                 q_home: np.ndarray | None = None, debug: bool = False, **_ignored):
        _ensure_upstream_on_path()
        from retargeting.core.optimizers.vector_wrist_joint import VectorWristJointOptimizerV2
        from retargeting.core.kinematics.adaptor import RobotAdaptor

        self.model = model
        self.hand_type = hand_type
        self.debug = debug
        self._opt_joints = _ARM_JOINTS + _HAND_JOINTS

        # Frames the optimizer references: origins + tasks + wrist, deduped in the order
        # the upstream optimizer computes them (it dedupes internally, but our robot_model
        # returns positions by index, so we just need every referenced frame present).
        logical_frames = []
        for o, t in _UPSTREAM_LINK_PAIRS:
            for nm in (o, t):
                # _WRIST_SITE appears as an origin in the pairs; use the logical "wrist".
                nm = "wrist" if nm == _WRIST_SITE else nm
                if nm != "world" and nm not in logical_frames:
                    logical_frames.append(nm)
        if "wrist" not in logical_frames:
            logical_frames.append("wrist")
        self._logical_frames = logical_frames
        site_frames = [_NAME2SITE[n] for n in logical_frames]
        # wrist logical frame index -> its site is _WRIST_SITE.
        wrist_logical = "wrist"

        self.robot_model = MujocoRobotModel(
            model, self._opt_joints, logical_frames, wrist_logical,
            site_names=site_frames, wrist_site=_WRIST_SITE)
        # doa == dof == 23: the adaptor's maps are the identity (see upstream_kinematics).
        self.adaptor = RobotAdaptor(self.robot_model, actuated_joints_name=self._opt_joints)

        # Link-pair NAMES the optimizer optimizes over (its origin/task lists). "world" is
        # a fixed point at the origin — the optimizer has no such frame, so we emulate it
        # with a dedicated always-at-origin logical frame appended to the model.
        # Instead of a special-case, we exploit that build_ref_values puts the world->thumb
        # pair as an ABSOLUTE thumb target; upstream does the same via a "world" origin whose
        # position is 0. We add a synthetic "world" site by mapping it to a frame we force to
        # origin — simplest: give the optimizer origin name "world" and ensure our robot_model
        # returns 0 for it. We register it as an extra frame pinned at the origin.
        self._config = _default_config()
        self._load_config_values()

        targets = {
            "origin_links_name": [o if o != "world" else "__world__" for o, _ in _UPSTREAM_LINK_PAIRS],
            "task_links_name": [t for _, t in _UPSTREAM_LINK_PAIRS],
            "wrist_link_name": "wrist",
        }
        # Register the synthetic world frame in the robot_model (pinned at origin).
        self._install_world_frame()

        params = {"huber_delta": self._config["huber_delta"],
                  "solver_params": {"ftol_abs": 1e-5, "maxtime": -1.0}}
        overrides = self._config.get("joint_limit_overrides", [])
        self.optimizer = VectorWristJointOptimizerV2(
            self.adaptor, targets=targets, params=params,
            joint_limit_overrides=overrides, solver="nlopt")

        self._q_home = (np.zeros(self.robot_model.dof) if q_home is None
                        else np.asarray(q_home, float)[:self.robot_model.dof].copy())
        self._q_last = self._q_home.copy()
        self._cfg_mtime = None
        self.poll_config()   # pick up disk config if present

    # -- synthetic "world" origin frame -------------------------------------------
    def _install_world_frame(self):
        """The optimizer's world->thumb pair needs a "world" origin at position 0. Add a
        logical frame "__world__" to the robot_model that always returns the origin, and
        a zero Jacobian (a fixed point doesn't move with q)."""
        rm = self.robot_model
        if "__world__" in rm._frame_names:
            return
        rm._frame_names.append("__world__")
        self._world_idx = len(rm._frame_names) - 1

        # Wrap the pose/jacobian readers so __world__ returns origin / zeros.
        _orig_pose = rm.get_frame_pose_by_id
        _orig_jac = rm.get_frame_space_jacobian_by_id
        widx = self._world_idx
        n = rm.dof

        def pose_by_id(fid, _op=_orig_pose, _w=widx):
            if fid == _w:
                T = np.eye(4)
                return T
            return _op(fid)

        def jac_by_id(fid, _oj=_orig_jac, _w=widx, _n=n):
            if fid == _w:
                return np.zeros((6, _n))
            return _oj(fid)

        rm.get_frame_pose_by_id = pose_by_id
        rm.get_frame_space_jacobian_by_id = jac_by_id
        # get_frames_index will now resolve "__world__" too.

    # -- config -------------------------------------------------------------------
    def _load_config_values(self):
        c = self._config
        self.scale = float(c["human_hand_scale"])

    def _read_disk_config(self, path):
        import json
        cfg = _default_config()
        try:
            with open(path) as f:
                disk = json.load(f)
            for k, v in disk.items():
                if k == "weights" and isinstance(v, dict):
                    cfg["weights"].update(v)
                elif not k.startswith("_"):
                    cfg[k] = v
            self._cfg_mtime = os.path.getmtime(path)
        except (OSError, json.JSONDecodeError):
            pass
        return cfg

    def poll_config(self, path: str | None = None) -> bool:
        """Hot-reload weights from vwj_config.json (shared with the clean-room mode)."""
        path = path or _CONFIG_PATH
        try:
            mt = os.path.getmtime(path)
        except OSError:
            return False
        if self._cfg_mtime == mt:
            return False
        self._config = self._read_disk_config(path)
        self._load_config_values()
        # huber_delta / joint_limit_overrides are baked into the optimizer at construction;
        # weights + scale + thresholds are applied per-frame in retarget(), so a live edit
        # of those takes effect immediately. (A huber/override change needs a restart.)
        return True

    def reset(self, q_home: np.ndarray | None = None) -> None:
        if q_home is not None:
            self._q_home = np.asarray(q_home, float)[:self.robot_model.dof].copy()
        self._q_last = self._q_home.copy()

    # -- the solve ----------------------------------------------------------------
    def retarget(self, world_lm: np.ndarray, wrist_pos: np.ndarray,
                 wrist_R: np.ndarray, q_prev: np.ndarray | None = None) -> np.ndarray:
        """Same signature/behaviour as VWJRetargeter.retarget, but the optimize is the
        UPSTREAM VectorWristJointOptimizerV2.retarget()."""
        from anyteleop.landmarks import world_landmarks_to_mano

        c = self._config
        mano = world_landmarks_to_mano(world_lm, self.hand_type) * self.scale
        wrist_R = np.asarray(wrist_R, float)
        kps_world = np.asarray(wrist_pos, float)[None, :] + mano @ wrist_R.T

        # Build the link-vec targets + weights (identical math to upstream's
        # _build_ref_values, verified line-for-line for the non-ablation path).
        ref_vec, ref_w = build_ref_values(
            kps_world, c["weights"],
            pinch_transition=c["pinch_transition_threshold"],
            pinch_contact=c["pinch_contact_threshold"],
            pinch_slope=c["pinch_sigmoid_slope"])

        wrist_quat = np.empty(4)
        mj.mju_mat2Quat(wrist_quat, wrist_R.reshape(9))   # (w,x,y,z), upstream convention

        ref_values = {
            "links_vec": ref_vec,
            "wrist_quat": wrist_quat,
            "qpos_doa": self._q_home.copy(),
            "qpos_doa_last": (self._q_last if q_prev is None
                              else np.asarray(q_prev, float)).copy(),
            "weights": {
                "links_vec": ref_w,
                "wrist_rot": float(c["weights"]["wrist_rotation"]),
                "joint_pos": np.asarray(c["joint_position_weights"], float),
                "joint_vel": np.asarray(c["joint_velocity_weights"], float),
            },
        }
        try:
            q = np.asarray(self.optimizer.retarget(ref_values), float)
        except Exception:
            if self.debug:
                import traceback; traceback.print_exc()
            q = ref_values["qpos_doa_last"].copy()

        # Per-frame max-joint-speed clamp (as the clean-room path and upstream teleop do).
        max_speed = np.asarray(c["max_joint_speed"], float)
        dq = np.clip(q - self._q_last, -max_speed, max_speed)
        lo, hi = self.robot_model.joint_limits[:, 0], self.robot_model.joint_limits[:, 1]
        q = np.clip(self._q_last + dq, lo, hi)
        self._q_last = q.copy()
        return q

    def human_palm_frame_robot_aligned(self, lm: np.ndarray):
        from teleop.dexpilot_retargeter import DexPilotRetargeter
        if not hasattr(self, "_geom"):
            self._geom = DexPilotRetargeter(self.model, n_arm=7, load_config=False)
        return self._geom.human_palm_frame_robot_aligned(lm)

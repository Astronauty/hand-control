"""VWJ whole-arm-hand retargeter: per-frame NLopt-SLSQP solve (clean-room, arXiv:2506.09384).

Ties vwj.kinematics (MuJoCo FK/Jacobians) + vwj.objective (4-term cost) + vwj.refvalues
(human targets) into one solve that returns the FULL robot pose q (7 arm + 16 hand)
tracking the human hand — unlike the finger-only DexPilot/AnyTeleop retargeters.

Per frame:
  1. MANO keypoints (wrist-centered, hand frame) from the world landmarks.
  2. Placed into the ROBOT WORLD frame at the human wrist target: a rigid transform
     (wrist target position + orientation), with a human->robot hand-size scale.
  3. build_ref_values -> 12 target vectors + continuous pinch weights.
  4. NLopt SLSQP over the 23 joints, warm-started from the last solution, box-bounded
     by joint limits, with a per-frame max-joint-speed clamp on the result.

The wrist TARGET pose (pos+R) is supplied by the caller (the same wrist-following the
other baselines' arm IK uses) so all four conditions place the hand identically; VWJ
then owns how the whole arm+hand realizes it.
"""
from __future__ import annotations

import json
import os

import numpy as np
import mujoco as mj

from vwj.kinematics import MujocoKinematics
from vwj.objective import VWJObjective
from vwj.refvalues import build_ref_values, robot_link_pairs

_ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
_HAND_JOINTS = ["leap_if_mcp", "leap_if_rot", "leap_if_pip", "leap_if_dip",
                "leap_mf_mcp", "leap_mf_rot", "leap_mf_pip", "leap_mf_dip",
                "leap_rf_mcp", "leap_rf_rot", "leap_rf_pip", "leap_rf_dip",
                "leap_th_cmc", "leap_th_axl", "leap_th_mcp", "leap_th_ipl"]
# tip/DIP sites, thumb first then index/middle/ring (matches refvalues finger order).
# MANO keypoint indices for the pinch report (thumb, index, middle, ring tips) — the
# same indices refvalues.FINGERTIP_INDICES uses, and the same the AnyTeleop backend
# reports on, so the three backends' d_s1 signals are directly comparable.
_MANO_TIP_THUMB, _MANO_TIP_INDEX, _MANO_TIP_MIDDLE, _MANO_TIP_RING = 4, 8, 12, 16

_TIP_SITES = ["leap_th_ds_tip", "leap_if_ds_tip", "leap_mf_ds_tip", "leap_rf_ds_tip"]
_DIP_SITES = ["leap_th_ds_lower", "leap_if_ds_lower", "leap_mf_ds_lower", "leap_rf_ds_lower"]
_WRIST_SITE = "pinch_site"

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "teleop", "calibration", "vwj_config.json")


def _default_config() -> dict:
    """The upstream LEAP profile values (arXiv:2506.09384), verbatim as defaults."""
    return {
        "human_hand_scale": 1.5,
        "huber_delta": 0.02,
        "pinch_transition_threshold": 0.1,
        "pinch_contact_threshold": 0.01,
        "pinch_sigmoid_slope": 10.0,
        "weights": {
            "world_thumb": 10.0, "wrist_fingertip": 1.0, "thumb_primary": 10.0,
            "fingertip_orientation": 10.0, "wrist_rotation": 0.1,
        },
        # per-joint (7 arm + 16 hand) reg weights. REMAPPED from the upstream Paxini
        # order [side,mcp,pip,dip] to THIS repo's LEAP order [mcp,rot,pip,dip] per finger:
        # upstream's side-joint regs (0.5) land on the *_rot joints here, not *_mcp — else
        # the reg pulls the MCP flexion toward a curled home and the fingers (esp. the
        # middle) won't uncurl. Now: if_rot/mf_rot/rf_rot=0.5 (anti-splay), rf_dip=0.5,
        # th_mcp=0.1. (Indices: 8=if_rot,12=mf_rot,16=rf_rot,18=rf_dip,21=th_mcp.)
        "joint_position_weights": [0, 0, 1.0, 0, 0.5, 0, 0,
                                   0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0.1, 0],
        "joint_velocity_weights": [0.1] * 7 + [0.01] * 16,
        # per-frame joint-speed clamp (rad/step), from the LEAP profile.
        "max_joint_speed": [0.2] * 6 + [0.5] + [1.0] * 16,
        # joint-limit overrides (LEAP profile): clamp the PIP+DIP of index/middle/ring
        # to >=0 so those fingers can't hyperextend (which reads as sideways/curl).
        # Opt-vector indices 9,10=if_pip,if_dip; 13,14=mf_pip,mf_dip; 17,18=rf_pip,rf_dip.
        "joint_limit_overrides": [{"indices": [9, 10, 13, 14, 17, 18], "lower": 0.0}],
        "max_iter": 50,
    }


class VWJRetargeter:
    """Whole-arm-hand VWJ retargeter. retarget() returns a 23-vector (arm+hand)."""

    # Marks this as a WHOLE-ROBOT retargeter (owns the arm too) so the controller
    # drives it on the VWJ path — passing it the wrist target pose and applying its
    # full 23-DOF q directly — instead of the finger-only dexpilot/anyteleop path.
    whole_robot = True

    def __init__(self, model: mj.MjModel, n_arm: int = 7, hand_type: str = "right",
                 q_home: np.ndarray | None = None, debug: bool = False,
                 pinch_debounce: bool = True, **_ignored):
        self.model = model
        self.hand_type = hand_type
        self.debug = debug
        # Pinch distances the trial logger reads (mirror DexPilotRetargeter's attrs —
        # see _update_pinch). These MUST exist before the first retarget(): the trial
        # block reads them via getattr(rtg, 'last_d_s1_filt', rtg.last_d_s1), whose
        # default arg is evaluated EAGERLY, so a missing last_d_s1 raises even though
        # last_d_s1_filt is present.
        self.last_d_s1: list[float] = [float("inf")] * 3
        self.last_d_s1_filt: list[float] = [float("inf")] * 3
        self._pinch_debounce = bool(pinch_debounce)
        self._pinch_median_n = 5
        self._pinch_hist: list[list[float]] = [[], [], []]
        self.kin = MujocoKinematics(model, _ARM_JOINTS + _HAND_JOINTS,
                                    [_WRIST_SITE] + _TIP_SITES + _DIP_SITES, _WRIST_SITE)
        self.link_pairs = robot_link_pairs(_TIP_SITES, _DIP_SITES, _WRIST_SITE)
        self.n_opt = self.kin.n_opt
        self._q_home = (np.zeros(self.n_opt) if q_home is None
                        else np.asarray(q_home, float)[:self.n_opt].copy())
        self._q_last = self._q_home.copy()
        self._cfg_mtime = None
        self.load_config()
        self._build_objective()
        self._build_solver()

    # -- config (hot-reloadable, mirrors the other retargeters) --------------------
    def load_config(self, path: str | None = None) -> bool:
        path = path or _CONFIG_PATH
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
        self.cfg = cfg
        self.scale = float(cfg["human_hand_scale"])
        return True

    def poll_config(self, path: str | None = None) -> bool:
        path = path or _CONFIG_PATH
        try:
            mt = os.path.getmtime(path)
        except OSError:
            self._cfg_mtime = None
            return False
        if self._cfg_mtime == mt:
            return False
        self.load_config(path)
        self._build_objective()      # weights may have changed
        return True

    def _build_objective(self):
        c = self.cfg
        self.obj = VWJObjective(
            self.kin, self.link_pairs,
            huber_delta=c["huber_delta"], wrist_rot_weight=c["weights"]["wrist_rotation"],
            joint_pos_weight=np.asarray(c["joint_position_weights"], float),
            joint_vel_weight=np.asarray(c["joint_velocity_weights"], float))
        self._max_speed = np.asarray(c["max_joint_speed"], float)

    def _build_solver(self):
        import nlopt
        opt = nlopt.opt(nlopt.LD_SLSQP, self.n_opt)
        lo = np.where(np.isfinite(self.kin.lo), self.kin.lo, -2.0 * np.pi)
        hi = np.where(np.isfinite(self.kin.hi), self.kin.hi, 2.0 * np.pi)
        # Apply the LEAP profile's joint-limit overrides (e.g. PIP+DIP >= 0 so the
        # primary fingers can't hyperextend). Each entry raises `lower` (and/or lowers
        # `upper`) on the listed opt-vector indices — a faithful part of the method.
        for ov in self.cfg.get("joint_limit_overrides", []):
            idx = np.asarray(ov.get("indices", []), int)
            if idx.size == 0:
                continue
            if "lower" in ov:
                lo[idx] = np.maximum(lo[idx], float(ov["lower"]))
            if "upper" in ov:
                hi[idx] = np.minimum(hi[idx], float(ov["upper"]))
        opt.set_lower_bounds(lo.tolist())
        opt.set_upper_bounds(hi.tolist())
        opt.set_ftol_rel(1e-5)
        opt.set_maxeval(int(self.cfg.get("max_iter", 50)))
        self._opt = opt
        self._lo, self._hi = lo, hi

    def reset(self, q_home: np.ndarray | None = None) -> None:
        if q_home is not None:
            self._q_home = np.asarray(q_home, float)[:self.n_opt].copy()
        self._q_last = self._q_home.copy()
        self._pinch_hist = [[], [], []]
        self.last_d_s1 = [float("inf")] * 3
        self.last_d_s1_filt = [float("inf")] * 3

    # -- pinch reporting (trial-logger seam) ---------------------------------------
    def _update_pinch(self, mano: np.ndarray) -> None:
        """Index/middle/ring tip -> thumb distances, for the trial pinch signal.

        REPORTING ONLY — this never feeds the VWJ solve. VWJ has no discrete pinch
        detector: build_ref_values turns these same distances into CONTINUOUS sigmoid
        weights inside the objective, so there is no boolean to borrow. As the AnyTeleop
        backend does, we just report the raw distances and let the discrete decision be
        made downstream by DexPilotAttemptTrigger (min(d_s1) < eps).

        `mano` is the ALREADY-SCALED MANO keypoint array (metres, robot hand size), the
        same array build_ref_values consumes, so the reported gap matches the geometry
        the optimizer actually pinches with.

        With pinch_debounce on, an optional self-contained rolling median fills
        last_d_s1_filt; with it off, filt == raw.
        """
        kps = np.asarray(mano, float).reshape(21, 3)
        thumb = kps[_MANO_TIP_THUMB]
        d = [float(np.linalg.norm(kps[t] - thumb))
             for t in (_MANO_TIP_INDEX, _MANO_TIP_MIDDLE, _MANO_TIP_RING)]
        self.last_d_s1 = d
        if not self._pinch_debounce:
            self.last_d_s1_filt = list(d)
            return
        for i in range(3):
            hist = self._pinch_hist[i]
            hist.append(d[i])
            if len(hist) > self._pinch_median_n:
                del hist[0]
            self.last_d_s1_filt[i] = float(np.median(hist))

    # -- the solve -----------------------------------------------------------------
    def retarget(self, world_lm: np.ndarray, wrist_pos: np.ndarray,
                 wrist_R: np.ndarray, q_prev: np.ndarray | None = None) -> np.ndarray:
        """world_lm (21,3): this frame's MediaPipe world landmarks (raw[57:120]).
        wrist_pos (3,), wrist_R (3,3): the ROBOT wrist target pose (world frame) — the
        same wrist-following target the other baselines' arm IK uses.
        Returns q (23,): arm+hand joint targets."""
        from anyteleop.landmarks import world_landmarks_to_mano
        # 1. MANO keypoints (wrist-centered, hand frame), scaled to robot hand size.
        mano = world_landmarks_to_mano(world_lm, self.hand_type) * self.scale
        self._update_pinch(mano)
        # 2. place into robot world at the wrist target pose.
        kps_world = wrist_pos[None, :] + mano @ np.asarray(wrist_R, float).T
        # 3. human target vectors + continuous pinch weights.
        ref_vec, ref_w = build_ref_values(
            kps_world, self.cfg["weights"],
            pinch_transition=self.cfg["pinch_transition_threshold"],
            pinch_contact=self.cfg["pinch_contact_threshold"],
            pinch_slope=self.cfg["pinch_sigmoid_slope"])
        # wrist orientation target as wxyz quat.
        ref_wq = np.empty(4)
        mj.mju_mat2Quat(ref_wq, np.asarray(wrist_R, float).reshape(9))

        f = self.obj.get_objective(ref_vec, ref_w, ref_wq,
                                    q_rest=self._q_home, q_last=self._q_last)
        self._opt.set_min_objective(lambda x, g: f(x, g))
        x0 = np.clip(self._q_last if q_prev is None else np.asarray(q_prev, float),
                     self._lo, self._hi)
        try:
            q = np.asarray(self._opt.optimize(x0.tolist()), float)
        except Exception:
            if self.debug:
                import traceback; traceback.print_exc()
            q = x0
        # 4. per-frame max-joint-speed clamp (relative to the last solution).
        dq = np.clip(q - self._q_last, -self._max_speed, self._max_speed)
        q = np.clip(self._q_last + dq, self._lo, self._hi)
        self._q_last = q.copy()
        return q

    # arm-orientation geometry the app's marker/overlay expects — delegate to the
    # DexPilot palm-frame math (same as the AnyTeleop wrapper does).
    def human_palm_frame_robot_aligned(self, lm: np.ndarray):
        from teleop.dexpilot_retargeter import DexPilotRetargeter
        if not hasattr(self, "_geom"):
            self._geom = DexPilotRetargeter(self.model, n_arm=7, load_config=False)
        return self._geom.human_palm_frame_robot_aligned(lm)

"""AnyTeleopRetargeter — dex-retargeting LEAP finger backend, drop-in for DexPilotRetargeter.

Same public surface the teleop controller/worker depend on
(retarget / human_palm_frame_robot_aligned / poll_config / reset, plus the
last_d_s1[_filt] pinch attributes), so it slots into the existing seam
(teleop/dexpilot_controller.py:71/290) with no worker changes.

The finger SOLVE is delegated to dex-retargeting (AnyTeleop's published retargeting
library). Everything geometric and optimizer-INDEPENDENT — the robot-aligned palm frame
used for arm orientation, and the pinch-state debounce used by the trial logger — is
delegated to an internal DexPilotRetargeter instance, so no code is copied out of teleop/
and those behaviors stay identical across backends. The only thing that differs between
the DexPilot and AnyTeleop conditions is which optimizer produces the 16 finger angles.
"""
from __future__ import annotations

import os

import numpy as np

from anyteleop import config as _cfg
from anyteleop import landmarks as _lm
from anyteleop.joint_remap import build_remap, apply_remap

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# MANO-keypoint fingertip indices (dex-retargeting / MediaPipe convention):
_TIP_THUMB, _TIP_INDEX, _TIP_MIDDLE, _TIP_RING = 4, 8, 12, 16


class AnyTeleopRetargeter:
    """Retargets 21 MediaPipe world landmarks -> 16 LEAP joint angles via dex-retargeting."""

    def __init__(self, model, n_arm: int = 7, debug: bool = False,
                 eps: float | None = None, load_config: bool = True,
                 pinch_debounce: bool = True) -> None:
        self._model = model
        self._n_arm = n_arm
        self._n_hand = 16
        self.debug = debug

        # Internal DexPilotRetargeter: used ONLY for the optimizer-independent palm-frame
        # geometry the ARM needs (human_palm_frame*, pure MediaPipe->palm math). Its SLSQP
        # solve and its pinch debounce are NOT used — AnyTeleop's pinch reporting is
        # self-contained (see _update_pinch), so the AnyTeleop baseline carries none of this
        # repo's DexPilot pinch-detection logic.
        from teleop.dexpilot_retargeter import DexPilotRetargeter
        self._geom = DexPilotRetargeter(model, n_arm=n_arm, debug=False, eps=eps,
                                        load_config=load_config)

        # AnyTeleop-native pinch reporting state (raw distances; optional self-contained
        # median when --pinch-debounce is on). Independent of the DexPilot debounce class.
        self._pinch_debounce = bool(pinch_debounce)
        self._pinch_median_n = 5
        self._pinch_hist: list[list[float]] = [[], [], []]

        # Load our config (retargeting type, hand) and build the dex-retargeting solver.
        cfg = _cfg.load_config() if load_config else dict(_cfg._DEFAULTS)
        self._cfg_mtime = _cfg.file_mtime()
        self._hand_type = str(cfg.get("hand_type", "right"))
        self._out_alpha = float(cfg.get("output_alpha", 1.0))
        # dex-retargeting YAML overrides (None => keep the YAML's own value).
        self._scaling_factor = cfg.get("scaling_factor", None)
        self._low_pass_alpha = cfg.get("low_pass_alpha", None)
        self._build_solver(cfg.get("retargeting_type", "vector"))

        # Pinch distances the trial logger reads (mirror DexPilotRetargeter's attrs).
        self.last_d_s1 = [float("inf")] * 3
        self.last_d_s1_filt = [float("inf")] * 3
        self._q_prev: np.ndarray | None = None

    # -- solver construction --------------------------------------------------
    def _build_solver(self, retargeting_type: str) -> None:
        from dex_retargeting.retargeting_config import RetargetingConfig
        from dex_retargeting.constants import (RetargetingType, HandType, RobotName,
                                               get_default_config_path)

        RetargetingConfig.set_default_urdf_dir(_ASSETS_DIR)
        rtype = (RetargetingType.dexpilot
                 if str(retargeting_type).lower() == "dexpilot"
                 else RetargetingType.vector)
        htype = (HandType.right if self._hand_type.lower().startswith("r")
                 else HandType.left)
        cfg_path = get_default_config_path(RobotName.leap, rtype, htype)
        # Override the vendored YAML's scaling_factor / low_pass_alpha from our config
        # when set (None => keep the YAML value), so those knobs are tunable via
        # calibration/anyteleop_config.json without editing the asset files.
        override = {}
        if self._scaling_factor is not None:
            override["scaling_factor"] = float(self._scaling_factor)
        if self._low_pass_alpha is not None:
            override["low_pass_alpha"] = float(self._low_pass_alpha)
        self._rt = RetargetingConfig.load_from_file(
            cfg_path, override=override or None).build()
        self._rtype_name = rtype.name
        # dex-qpos-order -> sim _HAND_JOINT_NAMES-order permutation (derived + verified).
        self._perm = build_remap(self._rt.optimizer.robot)
        self._q_lo, self._q_hi = self._sim_hand_ranges()
        if self.debug:
            print(f"[anyteleop] built {rtype.name} retargeter for leap ({htype.name}); "
                  f"remap={self._perm.tolist()}")

    def _sim_hand_ranges(self):
        """(lo, hi) arrays for the 16 sim hand joints, in _HAND_JOINT_NAMES order.

        dex-retargeting clips to its URDF limits, which can round-differ from the MJCF by
        ~1e-3 rad at a bound; clamp to the sim's exact ranges so PD targets never exceed
        the model's joint limits.
        """
        import mujoco as _mj
        from anyteleop.joint_remap import SIM_HAND_JOINT_NAMES
        lo = np.empty(16); hi = np.empty(16)
        for i, jn in enumerate(SIM_HAND_JOINT_NAMES):
            jid = _mj.mj_name2id(self._model, _mj.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                lo[i], hi[i] = -np.inf, np.inf
            else:
                lo[i], hi[i] = self._model.jnt_range[jid]
        return lo, hi

    # -- main entry (DexPilotRetargeter-compatible) ---------------------------
    def retarget(self, world_lm: np.ndarray, q_prev: np.ndarray | None = None,
                 image_lm: np.ndarray | None = None) -> np.ndarray:
        """Map 21 MediaPipe world landmarks -> (16,) LEAP joint angles (sim order).

        Signature matches DexPilotRetargeter.retarget. image_lm is accepted for
        interface parity but not used by the dex-retargeting solve (fingers come from the
        world landmarks, the same input dex-retargeting's own pipeline uses).
        """
        world_lm = np.asarray(world_lm, dtype=float).reshape(21, 3)

        # MANO keypoints for the solver + pinch distances.
        joint_pos = _lm.world_landmarks_to_mano(world_lm, self._hand_type)
        self._update_pinch(joint_pos)

        ref_value = _lm.build_ref_value(joint_pos, self._rt.optimizer)
        dex_qpos = np.asarray(self._rt.retarget(ref_value), dtype=float)

        q = apply_remap(dex_qpos, self._perm)          # -> sim _HAND_JOINT_NAMES order
        q = np.clip(q, self._q_lo, self._q_hi)         # never exceed the MJCF joint limits

        # Optional extra EMA on top of dex-retargeting's own low-pass.
        if self._out_alpha < 1.0 and self._q_prev is not None:
            q = self._out_alpha * q + (1.0 - self._out_alpha) * self._q_prev
        self._q_prev = q
        return q

    def _update_pinch(self, joint_pos: np.ndarray) -> None:
        """Compute the index/middle/ring tip -> thumb distances for the trial pinch signal.

        AnyTeleop-native pinch reporting: RAW distances, NO debounce. dex-retargeting has no
        pinch detector of its own (its DexPilot optimizer folds a project/escape term into
        the SOLVE, and the vector type has no pinch concept at all), so this backend does not
        borrow this repo's DexPilot median+Schmitt+N-frame debounce. It just reports the raw
        thumb-to-finger distances; the discrete decision is made downstream by
        DexPilotAttemptTrigger (min(d_s1) < eps). last_d_s1_filt mirrors last_d_s1 (no
        filtering) so the fallback consumer stays consistent.

        When pinch_debounce is on, an OPTIONAL self-contained rolling median is applied to
        last_d_s1_filt only (a few lines here, still independent of the DexPilot class); with
        it off, filt == raw. Either way this touches only the reported signal, never the
        dex-retargeting finger solve."""
        thumb = joint_pos[_TIP_THUMB]
        d = [float(np.linalg.norm(joint_pos[t] - thumb))
             for t in (_TIP_INDEX, _TIP_MIDDLE, _TIP_RING)]
        self.last_d_s1 = d
        if not self._pinch_debounce:
            self.last_d_s1_filt = list(d)
            return
        # Optional self-contained median (independent of the DexPilot debounce class).
        for i in range(3):
            hist = self._pinch_hist[i]
            hist.append(d[i])
            if len(hist) > self._pinch_median_n:
                del hist[0]
            self.last_d_s1_filt[i] = float(np.median(hist))

    # -- delegated geometry (arm orientation) ---------------------------------
    def human_palm_frame_robot_aligned(self, lm: np.ndarray):
        return self._geom.human_palm_frame_robot_aligned(lm)

    def human_palm_frame(self, lm: np.ndarray):
        return self._geom.human_palm_frame(lm)

    # -- lifecycle / config ---------------------------------------------------
    def reset(self) -> None:
        self._q_prev = None
        self._rt.reset()
        self._geom.reset()          # palm-frame geom only
        self._pinch_hist = [[], [], []]   # clear the self-contained median window
        self.last_d_s1 = [float("inf")] * 3
        self.last_d_s1_filt = [float("inf")] * 3

    def poll_config(self, path: str | None = None) -> bool:
        """Hot-reload anyteleop_config.json; rebuild the solver if any solver-affecting
        field changed (type / hand / scaling_factor / low_pass_alpha). output_alpha is
        applied live without a rebuild."""
        mtime = _cfg.file_mtime(path)
        if mtime is None or mtime == self._cfg_mtime:
            return False
        self._cfg_mtime = mtime
        cfg = _cfg.load_config(path)
        new_type = str(cfg.get("retargeting_type", "vector"))
        new_hand = str(cfg.get("hand_type", "right"))
        new_scale = cfg.get("scaling_factor", None)
        new_lpa = cfg.get("low_pass_alpha", None)
        self._out_alpha = float(cfg.get("output_alpha", 1.0))
        rebuild = (new_type.lower() != self._rtype_name.lower()
                   or new_hand != self._hand_type
                   or new_scale != self._scaling_factor
                   or new_lpa != self._low_pass_alpha)
        if rebuild:
            self._hand_type = new_hand
            self._scaling_factor = new_scale
            self._low_pass_alpha = new_lpa
            self._build_solver(new_type)
            self._q_prev = None
        return True

    def tunables(self) -> dict:
        return {"retargeting_type": self._rtype_name, "hand_type": self._hand_type,
                "output_alpha": self._out_alpha,
                "scaling_factor": self._scaling_factor,
                "low_pass_alpha": self._low_pass_alpha}

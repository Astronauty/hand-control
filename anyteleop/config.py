"""Permissive JSON config loader for the AnyTeleop backend.

Mirrors the tolerant pattern of DexPilotRetargeter.load_config/poll_config: a missing
file or missing keys fall back to defaults, and malformed JSON is caught and ignored so
live editing never crashes the sim. Config lives beside the other calibration files at
teleop/calibration/anyteleop_config.json (calibration/ was consolidated into teleop/).
"""
from __future__ import annotations

import json
import os

# <repo>/teleop/calibration/anyteleop_config.json — next to retarget_config.json etc.
# (anyteleop/ is a sibling of teleop/ at the repo root; calibration/ now lives under teleop/.)
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "teleop", "calibration", "anyteleop_config.json",
)

_DEFAULTS = {
    # "vector" (real-time video retargeting AnyTeleop uses live) or "dexpilot"
    # (DexPilot-style pinch-aware optimizer). Both are LEAP-supported by dex-retargeting.
    "retargeting_type": "vector",
    "hand_type": "right",
    # EMA on the retargeter's internal low-pass (dex-retargeting also has its own
    # low_pass_alpha in the YAML; this is an OPTIONAL extra smoothing knob applied by our
    # wrapper on top, 1.0 = no extra smoothing).
    "output_alpha": 1.0,
    # dex-retargeting's OWN finger-retargeting knobs, overriding the vendored LEAP YAML
    # so they can be tuned without editing anyteleop/assets/.../leap_hand_right*.yml.
    # null => keep the YAML value (scaling_factor 1.6, low_pass_alpha 0.2).
    #   scaling_factor: robot-to-human hand-size gain (higher = more finger curl/reach).
    #   low_pass_alpha: smaller = smoother but laggier (0<=a<=1); <0 disables the filter.
    "scaling_factor": None,
    "low_pass_alpha": None,
}


def config_path() -> str:
    return _CONFIG_PATH


def load_config(path: str | None = None) -> dict:
    """Return the merged config (defaults <- file). Never raises on a bad/missing file."""
    cfg = dict(_DEFAULTS)
    p = path or _CONFIG_PATH
    try:
        with open(p, "r") as f:
            user = json.load(f)
        if isinstance(user, dict):
            for k in _DEFAULTS:
                if k in user:
                    cfg[k] = user[k]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return cfg


def file_mtime(path: str | None = None) -> float | None:
    p = path or _CONFIG_PATH
    try:
        return os.path.getmtime(p)
    except OSError:
        return None

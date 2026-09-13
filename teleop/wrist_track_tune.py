"""Live wrist-tracking responsiveness tuning: shared config + error log.

The dexpilot / contact_aware_teleop arm follows the operator's wrist with a
resolved-rate velocity loop (_solve_wrist_qdot in kinova_leap_pick_place.py):
    v = clip(WRIST_TRACK_GAIN * pos_error, ±JOG_VEL), accel-slewed, DLS-mapped
    to joint rates, hard-capped at JOG_QDOT_MAX.
All three are hot-tunable at runtime through teleop/calibration/wrist_track_config.json,
mirroring the retargeter's poll_config() idiom (mtime-gated, edit-and-save picks it
up live). The env vars TELEOP_JOG_VEL / TELEOP_TRACK_GAIN / TELEOP_QDOT_MAX seed the
STARTUP defaults; once a config file exists it wins and can be changed live.

This module is the shared piece: the running app imports WristTrackConfig to poll the
JSON and to append a per-sample error log; teleop/wrist_tune.py (the slider GUI) writes
the JSON and reads the log back to plot the live tracking error. Keeping both sinks
here means the field names never drift between writer and reader.

    log line (JSONL, one per logged control step):
      {"t": sim_time, "wall": wall_time, "err_mm": |p_ref-p_cur|*1e3,
       "err_xyz_mm": [ex,ey,ez], "speed_cmd": |v_cmd|, "sigma_min": ...,
       "JOG_VEL":.., "WRIST_TRACK_GAIN":.., "JOG_QDOT_MAX":..}
"""
from __future__ import annotations

import json
import os
import time

# calibration/ lives next to this file's parent's calibration dir (teleop/calibration).
_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "calibration", "wrist_track_config.json")
LOG_DEFAULT = os.path.join(_HERE, "calibration", "wrist_track_error.jsonl")

# The three live-tunable constants and their bake-in defaults (kept in sync with
# kinova_leap_pick_place.py's JOG_VEL / WRIST_TRACK_GAIN / JOG_QDOT_MAX). A missing
# key falls back to these; the env vars override these at startup only.
DEFAULTS = {
    # Kept in sync with kinova_leap_pick_place.py's JOG_VEL / WRIST_TRACK_GAIN / JOG_QDOT_MAX
    # constants (the app passes those into WristTrackConfig, so they win at runtime; these are
    # the fallback + the GUI 'Reset to defaults' target).
    "JOG_VEL":          0.4,   # m/s   peak wrist-tracking speed cap
    "WRIST_TRACK_GAIN": 9.0,   # 1/s   P-gain on wrist position error
    "JOG_QDOT_MAX":     3.0,   # rad/s final per-joint arm-rate cap
    # The two knobs that trade SPEED vs OVERSHOOT (raising the gain alone makes the loop
    # faster but underdamped -> overshoot). Tune these together with the gain:
    "TRACK_ACCEL":      5.0,   # m/s^2  Cartesian accel-slew cap on the velocity command
                               #   (default matches the historical NCF_ACCEL_BUDGET_XYZ=5).
                               #   Lower = gentler ramp (less overshoot, slower); higher =
                               #   snappier (more overshoot). Decoupled from the wrench LP's
                               #   budget so tuning feel never weakens the no-slip guarantee.
    "TRACK_DAMP":       0.0,   # 1/s    velocity-damping coefficient: subtracts TRACK_DAMP*v
                               #   from the commanded velocity each step, so the jog decays
                               #   instead of coasting into the target -> kills overshoot at a
                               #   given gain. 0 = off (original behaviour). Try 2-8 to damp.
}


def _env_seeded_defaults() -> dict:
    """DEFAULTS overlaid with the TELEOP_* env vars (startup seed), so the very first
    config write matches whatever the operator launched with."""
    out = dict(DEFAULTS)
    for key, env in (("JOG_VEL", "TELEOP_JOG_VEL"),
                     ("WRIST_TRACK_GAIN", "TELEOP_TRACK_GAIN"),
                     ("JOG_QDOT_MAX", "TELEOP_QDOT_MAX"),
                     ("TRACK_ACCEL", "TELEOP_TRACK_ACCEL"),
                     ("TRACK_DAMP", "TELEOP_TRACK_DAMP")):
        v = os.environ.get(env)
        if v is not None:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                pass
    return out


class WristTrackConfig:
    """Holds the three tunables, hot-reloads them from JSON, and appends the error log.

    Construct once in the app after the tracking constants are known. Call poll() every
    control iteration (cheap: stats the file, reloads only on mtime change). Call
    log_sample(...) each control step you want recorded (throttle at the call site).
    """

    KEYS = tuple(DEFAULTS.keys())   # every tunable; extend by adding to DEFAULTS

    def __init__(self, jog_vel: float, track_gain: float, qdot_max: float,
                 config_path: str | None = None, log_path: str | None = None,
                 logging: bool = False):
        # Seed all tunables from DEFAULTS, then override the three the app passes
        # explicitly (already env-seeded on its side). New knobs (TRACK_ACCEL /
        # TRACK_DAMP) take their DEFAULTS value until the config/env sets them.
        for k in self.KEYS:
            setattr(self, k, float(DEFAULTS[k]))
        self.JOG_VEL = float(jog_vel)
        self.WRIST_TRACK_GAIN = float(track_gain)
        self.JOG_QDOT_MAX = float(qdot_max)
        self.config_path = config_path or CONFIG_PATH
        self.log_path = log_path or LOG_DEFAULT
        self.logging = bool(logging)
        self._cfg_mtime = None
        self._log_fh = None
        # Seed the config file from the current (env-seeded) values if absent, so the
        # slider GUI has something to read and edits round-trip immediately.
        if not os.path.exists(self.config_path):
            try:
                self.save()
            except OSError:
                pass
        else:
            self.load()

    # -- config I/O ----------------------------------------------------------------
    def values(self) -> dict:
        return {k: float(getattr(self, k)) for k in self.KEYS}

    def load(self, path: str | None = None) -> bool:
        path = path or self.config_path
        try:
            with open(path) as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        for k in self.KEYS:
            if k in cfg:
                try:
                    setattr(self, k, float(cfg[k]))
                except (TypeError, ValueError):
                    pass
        try:
            self._cfg_mtime = os.path.getmtime(path)
        except OSError:
            pass
        return True

    def save(self, path: str | None = None) -> str:
        path = path or self.config_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.values(), f, indent=2)
        try:
            self._cfg_mtime = os.path.getmtime(path)
        except OSError:
            pass
        return path

    def poll(self, path: str | None = None) -> bool:
        """Hot-reload from JSON when it changed on disk. Cheap every-frame call; returns
        True on an actual reload. Missing file resets the watch so a later create loads."""
        path = path or self.config_path
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self._cfg_mtime = None
            return False
        if self._cfg_mtime == mtime:
            return False
        return self.load(path)

    # -- error log -----------------------------------------------------------------
    def log_sample(self, *, t_sim: float, err_xyz, speed_cmd: float,
                   sigma_min: float | None = None, ang_err=None,
                   ang_speed_cmd: float | None = None, tgt_z=None, cur_z=None) -> None:
        """Append one tracking-error sample. err_xyz is (p_ref - p_cur) in metres.
        Optional orientation diagnostics: ang_err (rotation-error vector, its norm ~=
        angular error in rad), ang_speed_cmd (|commanded angular velocity|), and the
        target vs actual wrist +Z axis in world (tgt_z/cur_z) — logged so a capture can
        show whether the TARGET orientation rotates (tgt_z moves) while the ACTUAL one
        does not (cur_z static), which separates 'command not produced' from 'command
        not executed'. No-op unless logging is enabled. Never raises into the loop."""
        if not self.logging:
            return
        try:
            import numpy as np
            e = np.asarray(err_xyz, float).reshape(3)
            row = {
                "t": round(float(t_sim), 4),
                "wall": round(time.time(), 4),
                "err_mm": round(float(np.linalg.norm(e)) * 1e3, 3),
                "err_xyz_mm": [round(float(x) * 1e3, 3) for x in e],
                "speed_cmd": round(float(speed_cmd), 4),
                "JOG_VEL": round(self.JOG_VEL, 4),
                "WRIST_TRACK_GAIN": round(self.WRIST_TRACK_GAIN, 4),
                "JOG_QDOT_MAX": round(self.JOG_QDOT_MAX, 4),
            }
            if sigma_min is not None:
                row["sigma_min"] = round(float(sigma_min), 5)
            if ang_err is not None:
                a = np.asarray(ang_err, float).reshape(3)
                row["ang_err_deg"] = round(float(np.degrees(np.linalg.norm(a))), 3)
            if ang_speed_cmd is not None:
                row["ang_speed_cmd"] = round(float(ang_speed_cmd), 5)
            if tgt_z is not None:
                row["tgt_z"] = [round(float(x), 4) for x in np.asarray(tgt_z, float).reshape(3)]
            if cur_z is not None:
                row["cur_z"] = [round(float(x), 4) for x in np.asarray(cur_z, float).reshape(3)]
            if self._log_fh is None:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                self._log_fh = open(self.log_path, "a")
            self._log_fh.write(json.dumps(row) + "\n")
            self._log_fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

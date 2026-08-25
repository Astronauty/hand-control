#!/usr/bin/env python3
"""
divergence_log.py — capture what happened BEFORE the arm snapped to zero config.

The failure is over by the time you notice it: MuJoCo's BADQACC handler resets
`data` the instant qacc diverges, so by the time you see the arm standing
straight up, the state that caused it is gone. Printing at the moment of the
snap therefore shows you a clean, reset state and tells you nothing.

This keeps a ring buffer of the last N control iterations and dumps it to .npz
when a reset is detected, so the record starts well before the divergence.

Usage
-----
Construct once, after `data` exists:

    from ui.divergence_log import DivergenceLogger
    _divlog = DivergenceLogger(model, out_dir='logs/divergence')

Call once per control iteration, right after mj_step / mj_forward:

    _divlog.sample(model, data,
                   target=_dp_target,          # the PD setpoint (N_ROBOT,)
                   qdot_cmd=_qdot_arm,         # commanded arm rate (7,)
                   sigma_min=_sigma_min,       # pinch-site Jacobian conditioning
                   wrist_tgt=_teleop_wrist_tgt,
                   n_msgs=...)                 # optional: publisher message count

It self-triggers on data.time going backwards (the reset fingerprint) and also
on qacc exceeding a threshold, which catches the divergence one step EARLIER
than the reset does.

Reading the dump
----------------
    python3 ui/divergence_log.py logs/divergence/div_0001.npz
"""

import os
import sys
import time
from collections import deque

import numpy as np

# MuJoCo resets when |qacc| exceeds a large internal bound. Trip well below it so
# the dump captures the run-up rather than the aftermath.
QACC_TRIGGER = 1e6
QVEL_TRIGGER = 1e3


class DivergenceLogger:

    def __init__(self, model, out_dir="logs/divergence", history=600,
                 n_robot=23, cooldown_s=2.0, verbose=True):
        self.out_dir = out_dir
        self.n_robot = n_robot
        self.buf = deque(maxlen=history)
        self.last_t = 0.0
        self.n_dumps = 0
        self.cooldown_s = cooldown_s
        self._last_dump_wall = 0.0
        self.verbose = verbose
        os.makedirs(out_dir, exist_ok=True)
        if verbose:
            print(f"[divlog] capturing {history} iterations of history -> {out_dir}/")

    def sample(self, *args, **kwargs):
        """Wrapper so a logging bug can never kill the control loop — and with
        it the GLFW viewer thread, which segfaults on the way down."""
        try:
            self._sample(*args, **kwargs)
        except Exception as e:
            if self.verbose:
                print(f"\r\n[divlog] sample failed, logging disabled: {e}")
            self.sample = lambda *a, **k: None

    def _sample(self, model, data, target=None, qdot_cmd=None, sigma_min=None,
                wrist_tgt=None, n_msgs=None, note="", contacts=None):
        n = self.n_robot
        row = {
            "t": float(data.time),
            "t_wall": time.time(),
            "qpos": data.qpos[:n].copy(),
            "qvel": data.qvel[:n].copy(),
            "qacc": data.qacc[:n].copy(),
            "qfrc_applied": data.qfrc_applied[:n].copy(),
            "qfrc_constraint": data.qfrc_constraint[:n].copy(),
            "ncon": int(data.ncon),
            # Peak magnitudes are what the trigger and the eyeball scan use.
            "qacc_max": float(np.abs(data.qacc[:n]).max()),
            "qvel_max": float(np.abs(data.qvel[:n]).max()),
            "sigma_min": float(sigma_min) if sigma_min is not None else np.nan,
            "n_msgs": int(n_msgs) if n_msgs is not None else -1,
            "note": note,
        }
        row["target"] = (np.asarray(target[:n]).copy() if target is not None
                         else np.full(n, np.nan))
        row["qdot_cmd"] = (np.asarray(qdot_cmd).copy() if qdot_cmd is not None
                           else np.full(7, np.nan))
        # Tracking error is the single most diagnostic scalar: a large, growing
        # position error with a small commanded rate means the PD is losing the
        # arm, not that the operator moved fast.
        row["err_max"] = (float(np.abs(row["target"][:7] - data.qpos[:7]).max())
                          if target is not None else np.nan)
        if wrist_tgt is not None:
            row["wrist_p"] = np.asarray(wrist_tgt[0]).copy()
        else:
            row["wrist_p"] = np.full(3, np.nan)

        # Contact pairs, stored FIXED-WIDTH so the column stacks cleanly. A
        # ragged per-row list is what broke the first version of this file.
        pairs = np.full((8, 2), -1, dtype=np.int32)
        for i in range(min(data.ncon, 8)):
            c = data.contact[i]
            pairs[i] = (int(c.geom1), int(c.geom2))
        row["contact_pairs"] = pairs

        self.buf.append(row)

        # Two triggers. The time jump is the reset itself (MuJoCo's BADQACC handler
        # or the viewer's own Reset, both out of band); the magnitude trigger fires
        # one step earlier, while the diverging state is still in `data`.
        reset = data.time < self.last_t - 1e-12
        diverging = (row["qacc_max"] > QACC_TRIGGER
                     or row["qvel_max"] > QVEL_TRIGGER)
        self.last_t = float(data.time)

        if (reset or diverging) and (
                time.time() - self._last_dump_wall > self.cooldown_s):
            self._dump("reset" if reset else "diverging")

    def _dump(self, reason):
        if not self.buf:
            return
        self._last_dump_wall = time.time()
        self.n_dumps += 1
        path = os.path.join(self.out_dir, f"div_{self.n_dumps:04d}_{reason}.npz")

        # Stack per-key. Some fields (contact pair lists) are ragged — their
        # length changes with data.ncon — so np.array() on the raw column raises
        # "inhomogeneous shape". Fall back to an object array for those rather
        # than letting the exception escape: a logger that kills the control loop
        # (and with it the GLFW viewer, hence a segfault) is far worse than a
        # slightly awkward field in the dump.
        out = {}
        for k in (k for k in self.buf[0] if k != "note"):
            col = [r[k] for r in self.buf]
            try:
                out[k] = np.array(col)
            except (ValueError, TypeError):
                arr = np.empty(len(col), dtype=object)
                arr[:] = col
                out[k] = arr
        out["notes"] = np.array([r["note"] for r in self.buf])
        out["reason"] = reason
        try:
            np.savez_compressed(path, **out)
        except Exception as e:
            # Never let a dump failure take down the sim.
            print(f"\r\n[divlog] could not save {path}: {e}")
            return

        rows = list(self.buf)
        peak = max(rows, key=lambda r: r["qacc_max"])
        i_peak = rows.index(peak)
        if self.verbose:
            print(f"\r\n[divlog] {reason.upper()} — saved {path} "
                  f"({len(rows)} iterations)")
            print(f"[divlog]   peak |qacc| {peak['qacc_max']:.3g} at row "
                  f"{i_peak}/{len(rows)}, {len(rows)-i_peak} iterations before the end")
            print(f"[divlog]   at peak: |qvel| {peak['qvel_max']:.3g}  "
                  f"sigma_min {peak['sigma_min']:.4f}  ncon {peak['ncon']}  "
                  f"max PD err {peak['err_max']:.3f} rad")
            print(f"[divlog]   inspect: python3 ui/divergence_log.py {path}")


# ---------------------------------------------------------------------------
# Offline inspection
# ---------------------------------------------------------------------------

def inspect(path):
    d = np.load(path, allow_pickle=True)
    t = d["t"]
    qacc = d["qacc_max"]
    qvel = d["qvel_max"]
    sig = d["sigma_min"]
    err = d["err_max"]
    ncon = d["ncon"]
    qdot = d["qdot_cmd"]
    n = len(t)

    print(f"{path}  —  {n} iterations, reason={d['reason']}")
    print()
    i_peak = int(np.nanargmax(qacc))

    # The 40 iterations before the peak are where the cause lives. Everything
    # after is the explosion itself, which looks the same whatever caused it.
    lo = max(0, i_peak - 40)
    print(f"{'i':>4} {'sim t':>8} {'|qacc|':>10} {'|qvel|':>9} "
          f"{'sig_min':>8} {'PDerr':>7} {'|qdot|':>7} {'ncon':>5}")
    for i in range(lo, min(n, i_peak + 6)):
        mark = "  <<< PEAK" if i == i_peak else ""
        qd = np.nanmax(np.abs(qdot[i])) if qdot.ndim > 1 else np.nan
        print(f"{i:>4} {t[i]:>8.3f} {qacc[i]:>10.3g} {qvel[i]:>9.3g} "
              f"{sig[i]:>8.4f} {err[i]:>7.3f} {qd:>7.3f} {ncon[i]:>5}{mark}")

    print()
    print("Reading it:")
    print(f"  sigma_min at peak      {sig[i_peak]:.4f}   "
          f"(<0.02 = at/near the zero-config singularity)")
    print(f"  PD error at peak       {err[i_peak]:.3f} rad   "
          f"(>1.0 = the arm is far from its setpoint before diverging)")
    print(f"  ncon at peak           {int(ncon[i_peak])}   "
          f"(a jump here = contact impulse, not control divergence)")
    _rise = i_peak - lo
    print(f"  build-up               {_rise} iterations from the window start")
    print(f"  qacc growth ratio      "
          f"{qacc[i_peak] / max(qacc[lo], 1e-9):.3g}x over that span")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    inspect(sys.argv[1])
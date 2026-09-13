#!/usr/bin/env python3
"""
wrist_tune_analyze.py — turn a wrist_track_error.jsonl log into a recommendation.

Groups the logged tracking-error samples by the (JOG_VEL, WRIST_TRACK_GAIN,
JOG_QDOT_MAX) setting that was active, and reports the error distribution for each,
so the setting with the lowest error UNDER COMPARABLE MOTION stands out. To make the
comparison fair it weights by commanded speed: a setting only tested while barely
moving looks artificially good, so rows report the mean/median error AND the mean
commanded speed, and the recommendation prefers low error at a comparable (non-tiny)
speed.

    python3 teleop/wrist_tune_analyze.py                       # default log
    python3 teleop/wrist_tune_analyze.py --log path/to.jsonl
    python3 teleop/wrist_tune_analyze.py --min-speed 0.05      # ignore near-static samples
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))
try:
    from teleop.wrist_track_tune import LOG_DEFAULT
except ImportError:
    from wrist_track_tune import LOG_DEFAULT


def _key(r):
    return (round(r.get("JOG_VEL", 0), 3),
            round(r.get("WRIST_TRACK_GAIN", 0), 2),
            round(r.get("JOG_QDOT_MAX", 0), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=LOG_DEFAULT)
    ap.add_argument("--min-speed", type=float, default=0.03,
                    help="ignore samples with commanded speed below this (m/s) so "
                         "near-static holds don't flatter a setting (default 0.03)")
    args = ap.parse_args()

    import numpy as np
    groups = {}
    n_total = n_kept = 0
    try:
        with open(args.log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_total += 1
                if r.get("speed_cmd", 0.0) < args.min_speed:
                    continue
                n_kept += 1
                groups.setdefault(_key(r), []).append(
                    (float(r.get("err_mm", 0.0)), float(r.get("speed_cmd", 0.0))))
    except OSError:
        print(f"[analyze] cannot read {args.log}"); sys.exit(1)

    if not groups:
        print(f"[analyze] no samples above --min-speed={args.min_speed} m/s in {args.log} "
              f"({n_total} rows total). Move the wrist more while tuning, or lower --min-speed.")
        sys.exit(0)

    print(f"[analyze] {args.log}: {n_total} samples, {n_kept} above {args.min_speed} m/s, "
          f"{len(groups)} distinct settings\n")
    header = (f"{'JOG_VEL':>8} {'GAIN':>6} {'QDOT':>5} | {'n':>5} {'mean_err':>9} "
              f"{'med_err':>8} {'p95_err':>8} {'mean_spd':>8}")
    print(header); print("-" * len(header))
    rows = []
    for k, samples in sorted(groups.items()):
        e = np.array([s[0] for s in samples])
        v = np.array([s[1] for s in samples])
        rec = dict(jog_vel=k[0], gain=k[1], qdot=k[2], n=len(e),
                   mean=float(e.mean()), med=float(np.median(e)),
                   p95=float(np.percentile(e, 95)), spd=float(v.mean()))
        rows.append(rec)
        print(f"{k[0]:>8.2f} {k[1]:>6.1f} {k[2]:>5.2f} | {len(e):>5} "
              f"{rec['mean']:>7.1f}mm {rec['med']:>6.1f}mm {rec['p95']:>6.1f}mm "
              f"{rec['spd']:>6.3f}m/s")

    # Recommendation: among settings with enough samples and a comparable (>= median)
    # commanded speed, pick the lowest mean error. This avoids crowning a setting that
    # was only ever tested while barely moving.
    good = [r for r in rows if r["n"] >= 20]
    if not good:
        good = rows
    med_spd = float(np.median([r["spd"] for r in good]))
    cand = [r for r in good if r["spd"] >= 0.8 * med_spd] or good
    best = min(cand, key=lambda r: r["mean"])
    print("\n[recommendation] lowest mean error at a comparable commanded speed:")
    print(f"  JOG_VEL={best['jog_vel']:.2f}  WRIST_TRACK_GAIN={best['gain']:.1f}  "
          f"JOG_QDOT_MAX={best['qdot']:.2f}")
    print(f"  -> mean {best['mean']:.1f}mm, median {best['med']:.1f}mm, p95 {best['p95']:.1f}mm "
          f"over {best['n']} samples at {best['spd']:.3f} m/s mean speed")
    print("\nSanity-check this against the table above: prefer a setting whose LOW error is not "
          "just from low speed, and whose error trace was smooth (not oscillatory) live.")


if __name__ == "__main__":
    main()

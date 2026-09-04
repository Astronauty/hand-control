#!/usr/bin/env python3
"""Headless retargeting comparison: DexPilot vs AnyTeleop (vector & dexpilot).

Retargeting is a pure function world_landmarks(21,3) -> 16 LEAP joint angles, independent
of any object or the live task, so it can be compared fully headless. This drives a
designed sweep of synthetic hand poses (open, graded pinches, fist, per-finger curls)
through all three backends and reports, per backend:

  * solve latency (ms/frame) — the retargeting cost, mean & p95
  * joint-limit saturation — fraction of DOFs pinned at a bound (a retarget struggling
    to reach the target hits limits)
  * pinch agreement — do the backends' pinch decisions agree on the same poses?
  * robot fingertip tracking — after retargeting, how close are the ROBOT fingertips to
    the (scaled) human fingertip targets, computed via MuJoCo FK on the sim LEAP model.
    This is the retargeting-accuracy metric: lower = the robot hand better reproduces the
    commanded human hand shape.
  * pairwise joint-trajectory divergence — RMS joint-angle difference between backends
    over the sweep (how differently they move the SAME hand).

No cameras, no ROS, no objects. Deterministic (fixed synthetic poses).

    python3 anyteleop/compare_retargeting.py                 # table to stdout
    python3 anyteleop/compare_retargeting.py --json out.json  # + machine-readable dump
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from anyteleop.joint_remap import SIM_HAND_JOINT_NAMES

SCENE = os.path.join(_HERE, "models", "scene_pick_place.xml")

# MediaPipe landmark indices
WRIST, IDX_MCP, MID_MCP, PKY_MCP = 0, 5, 9, 17
TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
# MCP (knuckle) of each finger, to build a straight or curled finger.
MCP = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
PIP = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
DIP = {"thumb": 3, "index": 7, "middle": 11, "ring": 15, "pinky": 19}


def _base_hand():
    """A flat, open right hand in MediaPipe world-landmark convention (metres).

    Wrist at origin; fingers extend along +y; the four finger MCPs spread along x; thumb
    off toward +x. Coordinates are rough but anatomically ordered so the retargeters see a
    plausible hand. Returns (21,3)."""
    lm = np.zeros((21, 3))
    # finger MCP x-offsets (index..pinky), fingers point +y
    mcp_x = {"index": -0.03, "middle": -0.01, "ring": 0.01, "pinky": 0.03}
    seg = 0.030   # phalanx length
    for f in ("index", "middle", "ring", "pinky"):
        x = mcp_x[f]
        lm[MCP[f]] = [x, 0.05, 0.0]
        lm[PIP[f]] = [x, 0.05 + seg, 0.0]
        lm[DIP[f]] = [x, 0.05 + 2 * seg, 0.0]
        lm[TIPS[f]] = [x, 0.05 + 3 * seg, 0.0]
    # thumb: out toward +x, angled
    lm[MCP["thumb"]] = [0.03, 0.02, 0.0]
    lm[PIP["thumb"]] = [0.055, 0.035, 0.0]
    lm[DIP["thumb"]] = [0.075, 0.05, 0.0]
    lm[TIPS["thumb"]] = [0.09, 0.062, 0.0]
    return lm


def _curl_finger(lm, finger, amount):
    """Curl one finger toward the palm by `amount` in [0,1] (0=straight, 1=full fist).

    Bends the PIP/DIP/tip progressively down in -z and back in -y around the MCP."""
    lm = lm.copy()
    mcp = lm[MCP[finger]]
    for j, k in enumerate((PIP[finger], DIP[finger], TIPS[finger]), start=1):
        rel = lm[k] - mcp
        # rotate the phalanx about x-axis (curl in the y-z plane) by up to ~120 deg
        ang = amount * np.deg2rad(40 * j)
        c, s = np.cos(ang), np.sin(ang)
        y, z = rel[1], rel[2]
        lm[k] = mcp + np.array([rel[0], c * y + s * z, -s * y + c * z])
    return lm


def _pinch(lm, finger, amount):
    """Bring `finger` tip toward the thumb tip by fraction `amount` (1 = touching)."""
    lm = lm.copy()
    thumb = lm[TIPS["thumb"]]
    lm[TIPS[finger]] = lm[TIPS[finger]] + amount * (thumb - lm[TIPS[finger]])
    return lm


def build_pose_sweep():
    """Return [(name, (21,3) landmarks)] covering the interesting retargeting regimes."""
    base = _base_hand()
    poses = [("open", base)]
    # graded index->thumb pinch (the S1 pinch the retargeters key on)
    for a in (0.3, 0.6, 0.85, 1.0):
        poses.append((f"pinch_index_{int(a*100)}", _pinch(base, "index", a)))
    # middle and ring pinches
    poses.append(("pinch_middle_100", _pinch(base, "middle", 1.0)))
    poses.append(("pinch_ring_100", _pinch(base, "ring", 1.0)))
    # multi-finger pinch (index+middle to thumb)
    p = _pinch(base, "index", 1.0); p = _pinch(p, "middle", 1.0)
    poses.append(("pinch_index_middle", p))
    # per-finger curls
    for f in ("index", "middle", "ring"):
        poses.append((f"curl_{f}", _curl_finger(base, f, 0.8)))
    # full fist (all four fingers curled)
    fist = base
    for f in ("index", "middle", "ring", "pinky"):
        fist = _curl_finger(fist, f, 1.0)
    poses.append(("fist", fist))
    # graded whole-hand close (all fingers curl together)
    for a in (0.25, 0.5, 0.75):
        h = base
        for f in ("index", "middle", "ring", "pinky"):
            h = _curl_finger(h, f, a)
        poses.append((f"close_{int(a*100)}", h))
    return poses


# ---------------------------------------------------------------------------
# Robot fingertip tracking. We reuse the DexPilotRetargeter's OWN palm-frame vector
# builders (_human_vectors / _robot_vectors) as the ground-truth tracking definition:
# both express the 4 palm->fingertip vectors [index, middle, ring, thumb] in a consistent
# palm frame (human and robot respectively). Comparing their DIRECTIONS (unit vectors)
# measures how well the robot hand reproduces the commanded human hand SHAPE, independent
# of the human/robot hand-size difference. This avoids re-deriving frames (and the
# convention-mismatch that a naive human-vs-robot frame comparison introduces).
# ---------------------------------------------------------------------------

def tip_direction_errors(dp_geom, lm, q_hand):
    """Mean & max angle (deg) between robot and human palm->tip directions.

    dp_geom: a DexPilotRetargeter (provides _human_vectors + _robot_vectors, both in
    their own palm frames). lm: (21,3) landmarks. q_hand: (16,) retargeted joints.
    """
    hv = dp_geom._human_vectors(lm)          # 10 dicts; [0:4] palm->[if,mf,rf,th]
    rv = dp_geom._robot_vectors(q_hand)      # 10 vectors, same ordering
    errs = []
    for i in range(4):
        h = np.asarray(hv[i]["r"], dtype=float)
        r = np.asarray(rv[i], dtype=float)
        hu = h / (np.linalg.norm(h) + 1e-9)
        ru = r / (np.linalg.norm(r) + 1e-9)
        errs.append(np.rad2deg(np.arccos(np.clip(hu @ ru, -1, 1))))
    return float(np.mean(errs)), float(np.max(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write full per-pose results to this JSON file")
    ap.add_argument("--eps", type=float, default=0.03, help="pinch threshold (m)")
    ap.add_argument("--replay", help="hq_*.npz recording (real hand motion) to replay "
                    "through every backend instead of the synthetic sweep. Uses the "
                    "raw[57:120] world landmarks; far more meaningful for the absolute "
                    "tip-tracking metric than the synthetic poses.")
    ap.add_argument("--replay-stride", type=int, default=10,
                    help="with --replay, use every Nth frame (default 10).")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(SCENE)
    if args.replay:
        d = np.load(args.replay)
        lm_frames = d["raw"][:, 57:120].reshape(-1, 21, 3)[::args.replay_stride]
        poses = [(f"f{ i*args.replay_stride:05d}", lm_frames[i])
                 for i in range(len(lm_frames))]
        print(f"[replay] {args.replay}: {len(poses)} frames "
              f"(every {args.replay_stride}th of {len(d['raw'])})")
    else:
        poses = build_pose_sweep()

    # Build the three backends.
    from teleop.dexpilot_retargeter import DexPilotRetargeter
    from anyteleop.retargeter import AnyTeleopRetargeter

    backends = {}
    dp = DexPilotRetargeter(model, eps=args.eps, load_config=True)
    backends["dexpilot"] = dp
    # dp doubles as the palm-frame vector source for the tracking metric (its
    # _human_vectors/_robot_vectors define palm->tip in consistent frames).
    geom = dp
    at_v = AnyTeleopRetargeter(model, eps=args.eps, load_config=False)
    at_v._build_solver("vector"); backends["anyteleop_vector"] = at_v
    at_d = AnyTeleopRetargeter(model, eps=args.eps, load_config=False)
    at_d._build_solver("dexpilot"); backends["anyteleop_dexpilot"] = at_d

    lo = np.array([model.jnt_range[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jn)][0]
                   for jn in SIM_HAND_JOINT_NAMES])
    hi = np.array([model.jnt_range[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jn)][1]
                   for jn in SIM_HAND_JOINT_NAMES])
    sat_tol = 0.02 * (hi - lo)   # "pinned" if within 2% of a bound

    results = {b: {} for b in backends}
    latency = {b: [] for b in backends}
    q_all = {b: {} for b in backends}

    for name, lm in poses:
        for b, r in backends.items():
            t0 = time.perf_counter()
            q = np.asarray(r.retarget(lm), dtype=float)
            latency[b].append((time.perf_counter() - t0) * 1e3)
            q_all[b][name] = q
            terr_mean, terr_max = tip_direction_errors(geom, lm, q)
            sat = float(np.mean((q <= lo + sat_tol) | (q >= hi - sat_tol)))
            d_s1 = list(getattr(r, "last_d_s1", [np.nan] * 3))
            results[b][name] = {
                "q": q.tolist(),
                "tip_dir_err_deg": terr_mean,
                "tip_dir_err_max_deg": terr_max,
                "saturation": sat,
                "d_s1_mm": [float(x * 1e3) for x in d_s1],
            }

    # -------- report --------
    _src = f"replay {os.path.basename(args.replay)}" if args.replay else "synthetic pose sweep"
    print("\n=== Retargeting comparison (%s, %d poses) ===\n" % (_src, len(poses)))
    print("Per-backend summary:")
    hdr = f"  {'backend':20s} {'lat_ms_mean':>11s} {'lat_ms_p95':>10s} " \
          f"{'tip_dir_err°':>12s} {'sat_frac':>8s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    summary = {}
    for b in backends:
        lat = np.array(latency[b])
        terr = np.array([results[b][n]["tip_dir_err_deg"] for n in results[b]])
        sat = np.array([results[b][n]["saturation"] for n in results[b]])
        summary[b] = {"lat_mean": float(lat.mean()), "lat_p95": float(np.percentile(lat, 95)),
                      "tip_dir_err_mean": float(terr.mean()), "sat_mean": float(sat.mean())}
        print(f"  {b:20s} {lat.mean():11.2f} {np.percentile(lat,95):10.2f} "
              f"{terr.mean():12.1f} {sat.mean():8.2f}")

    # Pairwise joint-trajectory divergence (RMS over all poses & joints).
    print("\nPairwise joint-trajectory RMS divergence (rad):")
    names = list(backends)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            diffs = [q_all[a][n] - q_all[b][n] for n, _ in poses]
            rms = float(np.sqrt(np.mean(np.square(diffs))))
            print(f"  {a:20s} vs {b:20s}  RMS={rms:.3f}")

    # Pinch d_s1 comes from the SHARED landmark-based debounce, so it is identical across
    # backends by construction; we report it as a cross-check.
    if not args.replay:
        print("\nPinch d_s1 (index tip->thumb, mm) per pose  [eps=%.0fmm]:" % (args.eps*1e3))
        print(f"  {'pose':22s} " + " ".join(f"{b:>18s}" for b in backends))
        for n, _ in poses:
            if "pinch" in n or "close" in n or n == "open":
                row = " ".join(f"{results[b][n]['d_s1_mm'][0]:18.1f}" for b in backends)
                print(f"  {n:22s} {row}")
    else:
        # Over a real recording, confirm the backends agree on d_s1 frame-by-frame.
        base = "dexpilot"
        maxdiff = 0.0
        for n, _ in poses:
            for b in backends:
                if b == base:
                    continue
                maxdiff = max(maxdiff, abs(results[b][n]["d_s1_mm"][0]
                                           - results[base][n]["d_s1_mm"][0]))
        print(f"\nPinch d_s1 max cross-backend disagreement over {len(poses)} replay "
              f"frames: {maxdiff:.3f} mm (should be ~0 — shared landmark debounce).")

    if args.json:
        out = {"summary": summary, "poses": [n for n, _ in poses],
               "per_pose": results,
               "latency_ms": {b: latency[b] for b in backends}}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.json}")

    print("\nNotes:")
    print("  * tip_dir_err° = angle between robot & human fingertip directions in the palm")
    print("    frame (retargeting-shape accuracy; lower is better).")
    print("  * sat_frac = mean fraction of the 16 hand DOFs pinned at a joint limit.")
    print("  * Task (pick/place) performance needs live hand input driving a grasp and")
    print("    cannot run headless — use the start_teleop.sh --trial-log sweep for that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

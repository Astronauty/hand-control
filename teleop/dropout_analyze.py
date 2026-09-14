#!/usr/bin/env python3
"""
dropout_analyze.py — find WHY the headset hand-tracking drops out, from a pose_trace.

Reads a logs/<run>/pose_trace.npz recorded with the headset-hand-input fields
(hand_wrist, hand_head, hand_lm, hand_age_s — see kinova_leap_pick_place's recorder),
detects the tracking-dropout episodes (where hand_age_s spikes = the publisher stopped
sending because the tracked hand had tracked=0), and characterizes the hand pose /
workspace position at each dropout ONSET so you can tell whether losses cluster at:
  * a workspace REGION / range   (hand_wrist far from its usual center) -> out-of-range/FOV
  * a hand SHAPE / pose          (hand_lm at a distinctive configuration) -> pose-specific
    (a closed fist / edge-on palm tracks worse than an open, camera-facing hand)

    python3 teleop/dropout_analyze.py                        # newest run under logs/
    python3 teleop/dropout_analyze.py logs/<run>/pose_trace.npz

Frames note: hand_wrist is in the MuJoCo frame (--frame mujoco) while hand_head is raw
OpenXR — they are NOT in the same frame, so a wrist-to-head distance is meaningless and
is deliberately NOT computed. Range is assessed from hand_wrist's own spread; pose from
the wrist-relative (frame-independent) landmarks.
"""
import glob
import os
import sys

import numpy as np

# staleness above this (s) counts as a live dropout (input not refreshing). The publisher
# runs ~90 Hz, so a healthy gap is ~0.011s; 0.05 is ~4 missed frames.
DROPOUT_AGE_S = 0.05


def _newest_trace():
    fs = sorted(glob.glob("logs/*/pose_trace.npz"), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


def _episodes(is_drop, t):
    """Contiguous runs of is_drop==True. Yields (i_start, i_end, dur_s)."""
    eps = []
    i = 0
    n = len(is_drop)
    while i < n:
        if is_drop[i]:
            j = i
            while j + 1 < n and is_drop[j + 1]:
                j += 1
            eps.append((i, j, float(t[j] - t[i])))
            i = j + 1
        else:
            i += 1
    return eps


def _hand_span(lm):
    """A scalar 'openness' proxy for a (21,3) wrist-relative hand: mean fingertip distance
    from the wrist. Larger = more open/extended hand; smaller = curled/fist. MediaPipe tip
    indices 4,8,12,16,20; wrist is index 0 (already the relative origin, ~0)."""
    tips = lm[[4, 8, 12, 16, 20]]
    return float(np.mean(np.linalg.norm(tips, axis=1)))


def _palm_normal(lm):
    """Unit palm normal from wrist-relative landmarks (wrist=0, index_mcp=5, pinky_mcp=17).
    Direction the palm faces. When this rotates edge-on to the cameras the Vive loses the
    hand even with the wrist centered and in front — the classic hand-tracking failure."""
    n = np.cross(lm[5] - lm[0], lm[17] - lm[0])
    L = np.linalg.norm(n)
    return n / L if L > 1e-9 else np.array([0.0, 0.0, 1.0])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else _newest_trace()
    if not path or not os.path.exists(path):
        print("no pose_trace.npz found (pass a path, or run with --trial-log first)")
        sys.exit(1)
    z = np.load(path, allow_pickle=True)
    if "hand_age_s" not in z:
        print(f"{path} has no headset-hand fields — record a run on the current build "
              f"(the hand_wrist/hand_head/hand_lm/hand_age_s logging).")
        sys.exit(1)

    t = np.asarray(z["t"], float)
    age = np.asarray(z["hand_age_s"], float)
    hw = np.asarray(z["hand_wrist"], float)          # (N,3) MuJoCo frame
    hlm = np.asarray(z["hand_lm"], float)            # (N,21,3) wrist-relative
    valid = ~np.isnan(hw[:, 0])                      # rows with a real hand message

    print(f"[dropout] {path}")
    print(f"  {len(t)} rows, {valid.sum()} with hand input, "
          f"sim-time span {t.max()-t.min():.0f}s")

    is_drop = age > DROPOUT_AGE_S
    frac = 100.0 * np.mean(is_drop)
    eps = _episodes(is_drop, t)
    long_eps = [e for e in eps if e[2] > 0.2]
    print(f"  DROPOUT: {frac:.0f}% of frames stale (>{DROPOUT_AGE_S}s), "
          f"{len(eps)} episodes ({len(long_eps)} longer than 0.2s)")
    if eps:
        durs = np.array([e[2] for e in eps])
        print(f"  episode duration: median {np.median(durs)*1e3:.0f}ms, "
              f"max {durs.max():.1f}s")
    if not long_eps:
        print("  no substantial dropouts to characterize. (Good — or move the hand more.)")
        return

    # Onset = the last GOOD row just before each long dropout: that hand pose/position is
    # what the tracker lost on.
    onset_idx = []
    for i0, i1, dur in long_eps:
        k = i0 - 1
        while k > 0 and not valid[k]:
            k -= 1
        if valid[k]:
            onset_idx.append(k)
    onset_idx = np.array(onset_idx, int)

    # --- RANGE: where in the workspace were the dropouts vs. all tracked motion? ---
    wc = hw[valid].mean(0)                            # center of tracked wrist motion
    r_all = np.linalg.norm(hw[valid] - wc, axis=1)
    r_drop = np.linalg.norm(hw[onset_idx] - wc, axis=1)
    print("\n  RANGE (wrist distance from the center of tracked motion):")
    print(f"    all tracked frames:   median {np.median(r_all)*100:.0f}cm  p90 {np.percentile(r_all,90)*100:.0f}cm")
    print(f"    at dropout onsets:    median {np.median(r_drop)*100:.0f}cm  p90 {np.percentile(r_drop,90)*100:.0f}cm")
    if np.median(r_drop) > np.percentile(r_all, 75):
        print("    -> dropouts happen FARTHER OUT than typical motion => OUT-OF-RANGE / FOV-edge is a factor.")
    else:
        print("    -> dropout positions are within the normal motion range => not primarily range.")

    # per-axis: which direction are the dropouts offset toward?
    off = hw[onset_idx].mean(0) - wc
    print(f"    dropout onsets are offset from center by (x,y,z) = "
          f"({off[0]*100:+.0f},{off[1]*100:+.0f},{off[2]*100:+.0f}) cm")

    # --- POSE: hand openness at dropout vs. overall (curled/fist tracks worse) ---
    span_all = np.array([_hand_span(hlm[i]) for i in np.where(valid)[0]])
    span_drop = np.array([_hand_span(hlm[i]) for i in onset_idx])
    print("\n  POSE (hand openness = mean fingertip distance from wrist):")
    print(f"    all tracked frames:   median {np.median(span_all)*100:.1f}cm")
    print(f"    at dropout onsets:    median {np.median(span_drop)*100:.1f}cm")
    if np.median(span_drop) < np.percentile(span_all, 30):
        print("    -> dropouts happen with a MORE CLOSED/curled hand => POSE (fist/grasp) is a factor.")
    elif np.median(span_drop) > np.percentile(span_all, 70):
        print("    -> dropouts happen with a MORE OPEN/extended hand.")
    else:
        print("    -> hand openness at dropouts is typical => pose-openness is not the main factor.")

    # --- ORIENTATION: palm facing at dropout vs. overall (edge-on palm = classic Vive loss).
    # This is often the trigger when the wrist is CENTERED and IN FRONT but rotated for a grasp:
    # position/openness look fine, but the palm has turned away from the cameras.
    pn_all = np.array([_palm_normal(hlm[i]) for i in np.where(valid)[0]])
    good = pn_all.mean(0); good /= (np.linalg.norm(good) + 1e-9)   # typical well-tracked facing
    ang_all = np.degrees(np.arccos(np.clip(pn_all @ good, -1, 1)))
    ang_drop = np.array([np.degrees(np.arccos(np.clip(_palm_normal(hlm[i]) @ good, -1, 1)))
                         for i in onset_idx])
    print("\n  ORIENTATION (palm-facing angle from the typical well-tracked direction):")
    print(f"    all tracked frames:   median {np.median(ang_all):.0f}deg  p90 {np.percentile(ang_all,90):.0f}deg")
    print(f"    at dropout onsets:    median {np.median(ang_drop):.0f}deg  (each: "
          f"{', '.join(f'{a:.0f}' for a in ang_drop)})")
    if np.median(ang_drop) > np.percentile(ang_all, 75):
        print("    -> dropouts happen with the palm ROTATED AWAY from typical (toward edge-on)")
        print("       => PALM ORIENTATION is the trigger. Keep the palm facing the headset")
        print("          cameras through the grasp; a wrist rotated edge-on loses tracking even")
        print("          when centered and in front. (abs_scale/reach changes won't help this.)")
    else:
        print("    -> palm orientation at dropouts is typical => not primarily orientation.")

    # --- SPEED: hand velocity just before dropout vs. overall (motion blur). ---
    def _spd(k, w=3):
        a = max(0, k - w)
        if a == k or not (valid[a] and valid[k]):
            return np.nan
        return float(np.linalg.norm(hw[k] - hw[a]) / max(t[k] - t[a], 1e-3))
    spd_all = np.array([_spd(k) for k in np.where(valid)[0]])
    spd_all = spd_all[~np.isnan(spd_all)]
    spd_drop = np.array([_spd(i) for i in onset_idx])
    spd_drop = spd_drop[~np.isnan(spd_drop)]
    print("\n  SPEED (wrist speed just before onset; motion blur):")
    print(f"    all tracked frames:   median {np.median(spd_all):.2f} m/s  p90 {np.percentile(spd_all,90):.2f}")
    if len(spd_drop):
        print(f"    at dropout onsets:    median {np.median(spd_drop):.2f} m/s")
        if np.median(spd_drop) > np.percentile(spd_all, 75):
            print("    -> dropouts follow FASTER motion => motion blur is a contributing factor "
                  "(move the hand more slowly through grasps).")
        else:
            print("    -> speed at dropouts is typical => not primarily motion blur.")

    print("\n  Cross-check against the console LOST/REGAINED logs and the wrist MP4 for the "
          "onset frames to confirm the trigger (edge-on palm, occlusion, reach to a corner).")


if __name__ == "__main__":
    main()

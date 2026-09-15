#!/usr/bin/env python3
"""Burn the trial phase + detected events onto a run's 3rd-person overview video.

For DEBUGGING the phase/event detection: every frame is annotated with what the state
machine thought was happening (from events.jsonl) and what the object trace says actually
happened (lift episodes, classified as attempt / drop / placement by
parse_trials_tables.lift_episodes). Disagreements between the two are the point — that is
how the attempt-inflation and missed-drop defects were found in the first place.

    python3 tools/overlay_phase_video.py logs/ICRA_experiments/dexpilot_20260914_180043
    python3 tools/overlay_phase_video.py logs/<run> --out /tmp/annotated.mp4

TIME MAPPING (the part that is easy to get wrong)
-------------------------------------------------
SceneRecorder.capture(data, data.time) throttles on SIM time at 1/30 s, and re-anchors on
a backward jump (every sequential-spawn reset snaps data.time to 0). So frame k corresponds
to CUMULATIVE sim time k/30 across the whole run, not to data.time directly and not to wall
time. Verified on these runs: cumulative sim time / frame count = 29.4-29.5 fps.

The per-trial traces each restart at their own data.time, so this walks the pose_trace
(which spans the whole run) to build the cumulative-sim axis, then locates each trial's
window inside it by matching the trial's event timestamps.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from parse_trials_tables import lift_episodes, method_from_dir  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
PHASE_COLOR = {
    'APPROACH':  (180, 180, 180),
    'PICK':      (80, 200, 255),     # amber
    'TRANSPORT': (90, 220, 120),     # green
    'PLACE':     (240, 160, 90),     # blue
}
KIND_COLOR = {'attempt': (80, 200, 255), 'drop': (70, 70, 255),
              'placed': (90, 220, 120)}


def _text(img, s, org, scale=0.7, color=WHITE, thick=2):
    """Text with a dark outline so it stays readable over any scene background."""
    cv2.putText(img, s, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def load_events(run_dir: Path):
    rows = []
    with open(run_dir / 'events.jsonl', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def cumulative_sim_axis(run_dir: Path):
    """Cumulative sim-time for each pose_trace row: sim time advances within a trial and
    resets to 0 between them, so sum the positive deltas. Returns (cum, t_raw, trial_bounds)
    where trial_bounds are the cumulative times at which each reset occurred."""
    z = np.load(run_dir / 'pose_trace.npz', allow_pickle=True)
    t = np.asarray(z['t'], float)
    dt = np.diff(t, prepend=t[0])
    dt[dt < 0] = 0.0                     # a reset contributes no elapsed time
    return np.cumsum(dt), t


def build_timeline(run_dir: Path, fps: float):
    """Per-frame annotation state.

    Returns a list indexed by video frame, each entry a dict describing what to draw.
    Events are placed on the cumulative-sim axis by walking the pose_trace: an event at
    (trial_id, t_raw) lands at the first trace row belonging to that trial whose raw t
    reaches t_raw."""
    cum, t_raw = cumulative_sim_axis(run_dir)
    rows = load_events(run_dir)

    # Trial spans on the cumulative axis: a trial's rows are those between its
    # trial_start and trial_end raw times, on the segment after the corresponding reset.
    resets = [0] + (np.where(np.diff(t_raw) < 0)[0] + 1).tolist() + [len(t_raw)]
    segments = [(resets[i], resets[i + 1]) for i in range(len(resets) - 1)]

    by_trial = defaultdict(list)
    for r in rows:
        if r.get('trial_id'):
            by_trial[r['trial_id']].append(r)

    def locate(seg, t_target):
        """Cumulative time of raw time `t_target` within trace segment `seg`."""
        a, b = seg
        if b <= a:
            return None
        k = int(np.searchsorted(t_raw[a:b], t_target))
        k = min(max(k, 0), b - a - 1)
        return float(cum[a + k])

    ann = []
    trials = sorted(by_trial)
    # Segment i holds trial i+1. Each --sequential-spawn reset snaps data.time to 0 and
    # starts a new segment, and trial_start fires ~1s LATER inside that same segment (the
    # reset-to-home + press-8 recalibration gap). So the k-th trial lives in the k-th
    # segment, NOT the (k+1)-th: an earlier cut skipped segment 0 on the assumption it was
    # a pre-trial stretch, which shifted every trial one segment late and annotated every
    # frame after trial 1 with the previous trial's object. Verified against the event log:
    # segment elapsed times (12.72, 10.66, 33.91, 5.82, 10.63) exceed the trial durations
    # (7.73, 9.68, 32.94, 5.01, 9.60) by exactly that setup gap. Any trailing segment past
    # the last trial is a post-run stub and is simply unused.
    for idx, tid in enumerate(trials):
        if idx >= len(segments):
            break
        seg = segments[idx]
        evs = by_trial[tid]
        start = next((e for e in evs if e.get('event') == 'trial_start'), None)
        end = next((e for e in evs if e.get('event') == 'trial_end'), None)
        obj = (start or {}).get('object', '?')
        outcome = (end or {}).get('outcome', 'running')
        phases = [(locate(seg, e['t']), e.get('phase'))
                  for e in evs if e.get('event') == 'phase_enter']
        phases = [p for p in phases if p[0] is not None]
        marks = []
        for e in evs:
            ev = e.get('event')
            if ev in ('pick_confirmed', 'arrival', 'drop', 'contact_violation'):
                c = locate(seg, e['t'])
                if c is not None:
                    marks.append((c, ev))
        lifts = lift_episodes(run_dir, tid) or {'episodes': []}
        eps = [(locate(seg, e['t0']), locate(seg, e['t1']), e['kind'], e['dxy_m'])
               for e in lifts['episodes']]
        eps = [e for e in eps if e[0] is not None and e[1] is not None]
        ann.append(dict(tid=tid, obj=obj, outcome=outcome, seg=seg,
                        t0=float(cum[seg[0]]), t1=float(cum[seg[1] - 1]),
                        phases=sorted(phases), marks=sorted(marks), eps=eps,
                        n_att=lifts.get('attempts'), n_drop=lifts.get('drops'),
                        n_lift=lifts.get('lifts')))
    return ann, float(cum[-1])


def annotate(run_dir: Path, out_path: Path, fps_override=None):
    run = run_dir.name
    src = run_dir / f'{run}_overview.mp4'
    if not src.exists():
        sys.exit(f'no overview video at {src}')
    cap = cv2.VideoCapture(str(src))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ann, cum_total = build_timeline(run_dir, fps)
    arm = method_from_dir(run_dir) or '?'
    # Sanity: the recorder throttles on sim time, so frames/cum_sim should be ~fps.
    eff = nframes / cum_total if cum_total > 0 else fps
    print(f'[overlay] {run}: {nframes} frames, cumulative sim {cum_total:.1f}s '
          f'-> {eff:.2f} frames/sim-s (recorder fps {fps:g})')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not vw.isOpened():
        sys.exit(f'cannot open writer for {out_path}')

    BAR_H, BAR_Y = 26, h - 46
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Frame k sits at cumulative sim time k/eff (eff measured, not assumed).
        ts = k / eff
        cur = next((a for a in ann if a['t0'] <= ts <= a['t1']), None)

        _text(frame, f'{run}   [{arm}]', (14, 30), 0.65)
        _text(frame, f'frame {k}/{nframes}   sim {ts:6.2f}s', (14, 58), 0.55)

        if cur is None:
            _text(frame, 'between trials', (14, 88), 0.6, (180, 180, 180))
        else:
            phase = 'APPROACH'
            for pt, pn in cur['phases']:
                if pt <= ts and pn:
                    phase = pn
            col = PHASE_COLOR.get(phase, WHITE)
            _text(frame, f"trial {cur['tid']}  {cur['obj']}  [{cur['outcome']}]",
                  (14, 88), 0.6)
            _text(frame, f'phase: {phase}', (14, 118), 0.8, col)

            # trace-derived classification for the episode covering this instant
            live = next((e for e in cur['eps'] if e[0] <= ts <= e[1]), None)
            if live is not None:
                _text(frame, f'LIFT: {live[2]}  ({live[3]*1000:.0f}mm travel)',
                      (14, 150), 0.75, KIND_COLOR.get(live[2], WHITE))
            _text(frame, f"trace: {cur['n_lift']} lifts = "
                         f"{cur['n_att']} attempt + {cur['n_drop']} drop + placement",
                  (14, h - 62), 0.5, (200, 200, 200))

            # event flashes: hold a marker on screen for ~0.5s after it fires
            for mt, mv in cur['marks']:
                if 0 <= ts - mt <= 0.5:
                    _text(frame, mv.upper(), (w - 360, 118), 0.9, (70, 200, 255))

            # timeline bar for this trial, with episode bands
            span = max(1e-6, cur['t1'] - cur['t0'])
            cv2.rectangle(frame, (14, BAR_Y), (w - 14, BAR_Y + BAR_H), (40, 40, 40), -1)
            for e0, e1, kind, _d in cur['eps']:
                x0 = int(14 + (w - 28) * (e0 - cur['t0']) / span)
                x1 = int(14 + (w - 28) * (e1 - cur['t0']) / span)
                cv2.rectangle(frame, (x0, BAR_Y), (max(x1, x0 + 2), BAR_Y + BAR_H),
                              KIND_COLOR.get(kind, WHITE), -1)
            xh = int(14 + (w - 28) * (ts - cur['t0']) / span)
            cv2.line(frame, (xh, BAR_Y - 4), (xh, BAR_Y + BAR_H + 4), WHITE, 2)

        vw.write(frame)
        k += 1

    cap.release()
    vw.release()
    print(f'[overlay] wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--fps', type=float, default=None,
                    help='override the recorder fps (default: read from the video)')
    a = ap.parse_args()
    rd = Path(a.run_dir)
    out = Path(a.out) if a.out else rd / f'{rd.name}_overlay.mp4'
    annotate(rd, out, a.fps)


if __name__ == '__main__':
    main()

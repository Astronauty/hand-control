#!/usr/bin/env python3
"""Plot the event/phase TRACES of one or more --trial-log runs as timeline swimlanes.

A visual companion to parse_trials_tables.py (which emits LaTeX summary tables). Each run
becomes one horizontal lane: phase segments (APPROACH/PICK/TRANSPORT/PLACE) render as
shaded background bands, and the discrete events (attempt_start, pick_confirmed, drop,
arrival, contact_violation, trial_end, solve) render as markers along the lane — so the
APPROACH→PICK→TRANSPORT→PLACE flow and where a trial dropped / arrived / was abandoned read
at a glance across many runs.

Consumes the same logs/<run>/events.jsonl the trial logger writes (see trial_logger.py):
one JSON object per line, each carrying `trial_id`, `event`, `t` (sim-time s) and `t_wall`
(wall-clock epoch s). Folder selection mirrors parse_trials_tables.py exactly — pass run
dirs (globs expanded); only dirs containing events.jsonl are kept.

CLOCK
-----
Default x-axis is WALL-CLOCK (`t_wall`, re-based to each run's first event = 0), matching
the experiment-clock convention (completion times are wall-based; sim runs slower than
real-time). `--clock sim` uses sim-time `t` instead — note that 'solve' events are logged
with sim t=0 (they run on background threads), so under --clock sim they are dropped with a
note; under --clock wall they place correctly by t_wall.

USAGE
-----
  python plot_event_traces.py logs/contact_aware_teleop_*/ logs/dexpilot_*/
  python plot_event_traces.py --per-trial logs/contact_aware_teleop_20260727_183402
  python plot_event_traces.py --clock sim --events phase_enter,drop,arrival logs/<run>/...
  python plot_event_traces.py logs/*/ --out traces.png     # headless save (else shows window)
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path


# ── events.jsonl parsing (same reader/tolerance as parse_trials_tables.py) ────────────────

def load_events(run_dir: Path) -> list[dict]:
    fp = run_dir / 'events.jsonl'
    rows = []
    with open(fp, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass   # tolerate a torn final line from an interrupted run
    return rows


# ── phase / event vocabulary and styling ──────────────────────────────────────────────────

# Ordered so the legend and colors are stable. Colors chosen to read in both themes and to
# match the semantic grouping (approach=cool, pick=amber, transport=green, place=blue).
PHASE_COLORS = {
    'APPROACH':  "#e0e19e",   # light blue
    'PICK':      "#db6b25",   # amber
    'TRANSPORT': "#44346c",   # green
    'PLACE':     "#6bd690",   # deeper blue
}
PHASE_ORDER = ['APPROACH', 'PICK', 'TRANSPORT', 'PLACE']

# Discrete events → (marker, color, label). 'phase_enter' is drawn as bands, not a marker,
# so it's excluded here. 'attempt_result' is a lightweight tick (many per trial).
EVENT_STYLE = {
    'trial_start':       ('|', '#000000', 'trial start'),
    'attempt_start':     ('^', '#e6550d', 'attempt start'),
    'attempt_result':    ('2', '#fdae6b', 'attempt released'),
    'pick_confirmed':    ('o', '#31a354', 'pick confirmed'),
    'drop':              ('v', '#de2d26', 'drop'),
    'arrival':           ('*', '#3182bd', 'arrival (placed)'),
    'contact_violation': ('x', '#756bb1', 'contact violation'),
    'trial_end':         ('s', '#252525', 'trial end'),
    'solve':             ('.', '#bdbdbd', 'solve'),
}


# ── per-run trace extraction ──────────────────────────────────────────────────────────────

def _event_time(ev: dict, clock: str):
    """Return the chosen-clock timestamp for one event, or None if unavailable.
    Solve events carry sim t=0 (background-thread), so under --clock sim they are dropped."""
    if clock == 'wall':
        return ev.get('t_wall')
    t = ev.get('t')
    # sim clock: solve events log t=0 and are not meaningfully placeable on the sim axis
    if ev.get('event') == 'solve':
        return None
    return t


def extract_lanes(run_dir: Path, clock: str, per_trial: bool):
    """Turn one run's events into one or more (label, t0, segments, markers) lanes.

    - t0: the clock zero for this run (first event's chosen-clock time), so every lane's
      times are re-based to run-relative seconds.
    - segments: list of (phase, t_start, t_end) for the shaded phase bands, built from
      consecutive phase_enter events; the last phase runs to the final event time.
    - markers: list of (t, event_name) for discrete events.
    per_trial splits into one lane per trial_id (trial 0 = the pre-trial/idle bucket).
    """
    rows = load_events(run_dir)
    if not rows:
        return []

    # Establish the run's clock zero from the earliest available chosen-clock timestamp.
    times = [_event_time(r, clock) for r in rows]
    valid = [t for t in times if t is not None]
    if not valid:
        return []
    t0 = min(valid)

    method = next((r.get('method') for r in rows if r.get('event') == 'trial_start'), None)
    obj    = next((r.get('object') for r in rows if r.get('event') == 'trial_start'), None)

    def _lane_from(rows_subset, label):
        phase_pts = []   # (t_rel, phase)
        markers = []
        last_t = None
        for r in rows_subset:
            t = _event_time(r, clock)
            if t is None:
                continue
            tr = t - t0
            last_t = tr if last_t is None else max(last_t, tr)
            ev = r.get('event')
            if ev == 'phase_enter':
                ph = r.get('phase')
                if ph:
                    phase_pts.append((tr, ph))
            elif ev in EVENT_STYLE:
                markers.append((tr, ev))
        # Build phase bands from consecutive phase_enter points; extend last to lane end.
        segments = []
        for i, (ts, ph) in enumerate(phase_pts):
            te = phase_pts[i + 1][0] if i + 1 < len(phase_pts) else (last_t if last_t is not None else ts)
            segments.append((ph, ts, max(te, ts)))
        return (label, segments, markers)

    base = run_dir.name
    tag = f"  [{method or '?'}·{obj or '?'}]" if (method or obj) else ''
    if not per_trial:
        return [_lane_from(rows, base + tag)]

    # One lane per trial_id, in first-seen order.
    by_tid = {}
    for r in rows:
        by_tid.setdefault(r.get('trial_id'), []).append(r)
    lanes = []
    for tid in sorted(by_tid, key=lambda x: (x is None, x)):
        lanes.append(_lane_from(by_tid[tid], f"{base}  t{tid}"))
    return lanes


# ── plotting ──────────────────────────────────────────────────────────────────────────────

def plot(all_lanes, clock: str, event_filter, out_path, width, height):
    import matplotlib
    if out_path is not None:
        matplotlib.use('Agg')   # headless save
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    n = len(all_lanes)
    fig_h = height if height else max(2.5, 0.5 * n + 1.5)
    fig_w = width if width else 14.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    lane_h = 0.6
    used_phases = set()
    used_events = set()
    for i, (label, segments, markers) in enumerate(all_lanes):
        y = n - 1 - i   # first lane on top
        # phase bands
        for ph, ts, te in segments:
            color = PHASE_COLORS.get(ph, '#dddddd')
            ax.barh(y, max(te - ts, 1e-6), left=ts, height=lane_h,
                    color=color, alpha=0.55, edgecolor='none', zorder=1)
            used_phases.add(ph)
        # event markers
        for tr, ev in markers:
            if event_filter and ev not in event_filter:
                continue
            marker, color, _ = EVENT_STYLE[ev]
            ax.plot(tr, y, marker=marker, color=color, markersize=8,
                    markeredgewidth=1.2, linestyle='none', zorder=3)
            used_events.add(ev)

    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for lab, _, _ in reversed(all_lanes)], fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(f"time since run start (s) — {'wall-clock' if clock == 'wall' else 'sim-time'}")
    ax.set_title(f"Event/phase traces  ({n} lane{'s' if n != 1 else ''}, "
                 f"{'wall' if clock == 'wall' else 'sim'} clock)")
    ax.grid(axis='x', linestyle=':', alpha=0.4)

    # Legends: phases (patches) + events (markers). Only show what actually appeared.
    phase_handles = [Patch(facecolor=PHASE_COLORS[p], alpha=0.55, label=p)
                     for p in PHASE_ORDER if p in used_phases]
    event_handles = [Line2D([0], [0], marker=EVENT_STYLE[e][0], color='none',
                            markerfacecolor=EVENT_STYLE[e][1], markeredgecolor=EVENT_STYLE[e][1],
                            markersize=8, label=EVENT_STYLE[e][2])
                     for e in EVENT_STYLE if e in used_events
                     and (not event_filter or e in event_filter)]
    leg1 = ax.legend(handles=phase_handles, title='phase', loc='upper left',
                     bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)
    if event_handles:
        ax.legend(handles=event_handles, title='event', loc='upper left',
                  bbox_to_anchor=(1.01, 0.5), fontsize=8, title_fontsize=8)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        print(f"wrote {out_path}")
    else:
        plt.show()


# ── main (folder selection mirrors parse_trials_tables.py) ────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+', help='logs/<run>/ directories (globs expanded)')
    ap.add_argument('--clock', choices=['wall', 'sim'], default='wall',
                    help="x-axis clock: wall (t_wall, default — the experiment clock) or "
                         "sim (data.time; solve events are omitted on this axis)")
    ap.add_argument('--per-trial', action='store_true',
                    help='one lane per (run, trial_id) instead of one lane per run')
    ap.add_argument('--events', default=None,
                    help='comma-separated event types to draw (default: all). '
                         'e.g. --events phase_enter,drop,arrival (phase_enter always drives '
                         'the bands regardless).')
    ap.add_argument('--out', default=None, metavar='PATH',
                    help='save the figure to PATH (.png/.pdf); if omitted, show a window')
    ap.add_argument('--width', type=float, default=None, help='figure width (inches)')
    ap.add_argument('--height', type=float, default=None, help='figure height (inches)')
    args = ap.parse_args()

    # Expand any globs the shell left unexpanded and keep only real dirs with events.jsonl.
    dirs = []
    for pat in args.run_dirs:
        for p in ([pat] if os.path.isdir(pat) else glob.glob(pat)):
            if (Path(p) / 'events.jsonl').is_file():
                dirs.append(p)
    if not dirs:
        sys.exit('no run dirs with events.jsonl found among: ' + ' '.join(args.run_dirs))

    event_filter = set(args.events.split(',')) if args.events else None

    all_lanes = []
    for d in sorted(dirs):
        all_lanes.extend(extract_lanes(Path(d), args.clock, args.per_trial))
    if not all_lanes:
        sys.exit('no plottable events found (empty logs, or --clock sim with only solve '
                 'events).')

    plot(all_lanes, args.clock, event_filter, args.out, args.width, args.height)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Parse --trial-log runs into the paper's completion-time and subtask-success tables.

Consumes one or more logs/<run>/ directories produced by
kinova_leap_pick_place.py --trial-log (see trial_logger.py for the log schema) and emits
two LaTeX tables:

  Table 1  Task completion time (s), mean ± std over N trials per object.
  Table 2  Conditional subtask success rates (successes/attempts), per object×method.

WHAT IS AND ISN'T COMPUTABLE FROM THE CURRENT LOGS
--------------------------------------------------
The ENTIRE analysis is restricted to VALID trials: a trial counts iff it reached a terminal
state (a trial_end — success or timeout — or a successful arrival) AND was not abandoned
(operator reset / target switch). A timeout that never grasped the object IS valid: it is a
genuine task failure and counts against end-to-end. Only abandoned trials and unterminated
trials (run killed mid-trial) are dropped. All metrics are computed directly from
events.jsonl (no trace files needed), over that valid set:
  * completion time     — trial_end.duration_s (WALL-CLOCK: the operator's real elapsed
                          time. Sim runs slower than real-time, so the sim-time span is
                          logged separately as duration_sim_s and is NOT what's reported),
                          over successful valid trials.
  * transport drops     — total number of 'drop' events (object fell out of the grasp during
                          carry) over the valid trials. Reported as a raw count with the
                          valid-trial count n; lower is better.
  * end-to-end success  — trial_end.outcome == 'success' (reached the place site), over the
                          valid trials.

NOT computable for teleop (EXCLUDED, matching the paper): collision-avoidance rate. The
inadvertent-contact counter (n_inadvertent_contacts) only runs during the autonomous RRT
plan/replay; in contact_aware_teleop the pre-lock-in approach is teleoperated and its
contacts are not counted, so the collision-avoidance subtask is intentionally omitted here.

USAGE
-----
  python3 parse_trials_tables.py logs/contact_aware_teleop_*/ [logs/dexpilot_*/ ...]
  python3 parse_trials_tables.py --objects obj_red_box obj_box_lowmu logs/<run>/...

Object columns are auto-discovered from the trials' object names (sorted); pass
--objects to fix an explicit order / subset. Each run dir may contain many trials; pass
as many run dirs as you like and they are pooled per (method, object).
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── events.jsonl parsing ────────────────────────────────────────────────────────────────

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


def group_trials(rows: list[dict]) -> dict[int, list[dict]]:
    """Group event rows by trial_id, keeping only trials that actually started
    (have a trial_start). The trial_id==0 pre-trial bucket (stray solves / the initial
    APPROACH marker before the first lock-in) is dropped: it never has a trial_start."""
    by_id: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_id[r.get('trial_id')].append(r)
    started = {}
    for tid, evs in by_id.items():
        if any(e.get('event') == 'trial_start' for e in evs):
            started[tid] = evs
    return started


def trial_summary(evs: list[dict]) -> dict | None:
    """Reduce one trial's events to the fields the tables need. Returns None if the
    trial neither ended (trial_end) nor reached a successful place (arrival).

    An 'arrival' event means the object was placed in the target (the state machine set
    outcome=SUCCESS). A bug left some runs writing 'arrival' but no 'trial_end' (the
    loop-top arrival path didn't call end_trial), so treat a logged arrival as an implicit
    successful completion: outcome='success', duration from trial_start→arrival (wall-clock
    if both carry t_wall, else sim-time). This recovers those trials without a re-run."""
    start = next((e for e in evs if e.get('event') == 'trial_start'), None)
    end   = next((e for e in evs if e.get('event') == 'trial_end'), None)
    if start is None:
        return None
    if end is None:
        arrival = next((e for e in evs if e.get('event') == 'arrival'), None)
        if arrival is None:
            return None   # trial neither ended nor placed — not scorable
        # Synthesize an end from the arrival (a successful place).
        if start.get('t_wall') is not None and arrival.get('t_wall') is not None:
            _dur = max(0.0, arrival['t_wall'] - start['t_wall'])
        else:
            _dur = max(0.0, arrival.get('t', 0.0) - start.get('t', 0.0))
        end = {'outcome': 'success', 'duration_s': round(_dur, 3),
               'n_attempts': arrival.get('attempt', 0), 'n_drops': 0}
    mass_kg = start.get('mass_kg')
    return {
        'method':       start.get('method', 'unknown'),
        'object':       start.get('object', 'unknown'),
        'outcome':      end.get('outcome'),
        'duration_s':   end.get('duration_s'),
        'pick_confirmed': any(e.get('event') == 'pick_confirmed' for e in evs),
        'n_attempts':   end.get('n_attempts', 0),
        # Count DROP EVENTS directly, not trial_end.n_drops. drop events are always logged,
        # whereas n_drops is only on trial_end — an arrival-only success (synthesized end
        # above) would otherwise report 0 drops even if the object was dropped and recovered
        # before the successful place. This counts drops-into-success correctly for all end
        # types (trial_end, timeout, and arrival-recovered).
        'n_drops':      sum(1 for e in evs if e.get('event') == 'drop'),
        # Physical properties, stamped onto trial_start by object_props_from_model()
        # (trial_logger.py). Absent (-> None) in older logs written before the props stamp.
        'mass_g':       (mass_kg * 1e3) if mass_kg is not None else None,
        'mu':           start.get('mu'),
        'izz_gcm2':     start.get('izz_gcm2'),
        'shape':        start.get('shape'),
    }


# ── table assembly ───────────────────────────────────────────────────────────────────────

def collect(run_dirs):
    """Pool VALID trials across run dirs into records keyed by (method, object).

    A trial is VALID iff it reached a terminal state (trial_end or a successful arrival) and
    was NOT abandoned (operator reset / target switch). A trial that timed out without ever
    confirming a pick IS valid — it is a genuine TASK FAILURE (the operator could not grasp
    the object within the time budget) and counts against end-to-end, not silently excluded.
    Only abandoned trials (operator gave up / switched target) and unterminated trials (run
    killed mid-trial — trial_summary returns None) are excluded. Each record:
      duration_s, n_drops(int), success(bool), + the object property fields.
    Everything is event-based (no trace files)."""
    records = defaultdict(list)
    for rd in run_dirs:
        rd = Path(rd)
        rows = load_events(rd)
        for tid, evs in group_trials(rows).items():
            s = trial_summary(evs)
            if s is None:
                continue   # trial never reached a terminal state (run killed mid-trial)
            if s['outcome'] == 'abandoned':
                continue   # excluded: operator reset / target switch (not a real attempt)
            records[(s['method'], s['object'])].append({
                'duration_s': s['duration_s'],
                'n_drops':   s.get('n_drops', 0) or 0,
                'success':   (s['outcome'] == 'success'),
                'mass_g':   s['mass_g'],
                'mu':       s['mu'],
                'izz_gcm2': s['izz_gcm2'],
                'shape':    s['shape'],
            })
    return records


def fmt_time_cell(durations):
    vals = [d for d in durations if d is not None and d > 0]
    if not vals:
        return r'$-$'
    m = float(np.mean(vals)); sd = float(np.std(vals))
    return f'${m:.1f} \\pm {sd:.1f}$'


def fmt_frac(num, den, bold=False):
    if den == 0:
        return r'$-$'
    s = f'{num}/{den}'
    return f'\\textbf{{{s}}}' if bold else s


# MuJoCo geom type ids -> human shape names (mirrors trial_logger.GEOM_TYPE_*; the
# properties table's Shape column. Kept local so this parser needs no mujoco import.)
_SHAPE_NAMES = {2: 'Sphere', 3: 'Capsule', 5: 'Cylinder', 6: 'Box'}


def _mode_prop(recs, key):
    """The object's property value pooled over its trials. Physical properties are a
    constant of the object, so all non-None values should agree; we take the most common
    (mode) to be robust to a stray older-format trial, and return None if never logged."""
    vals = [r[key] for r in recs if r.get(key) is not None]
    if not vals:
        return None
    # round floats before counting so trivially-different reprs collapse to one bucket
    from collections import Counter
    buckets = Counter(round(v, 6) if isinstance(v, float) else v for v in vals)
    return buckets.most_common(1)[0][0]


def build_object_properties_table(records, objects):
    """Emit the tab:object_properties table with mass / mu / I_zz auto-filled from the
    logged per-object physical properties (object_props_from_model). Compliance is NOT a
    native MuJoCo quantity, so it stays a measured placeholder column. Values are pooled
    per object across all methods (they're object constants, identical across methods)."""
    # pool every record for each object regardless of method
    by_obj = defaultdict(list)
    for (m, o), recs in records.items():
        by_obj[o].extend(recs)

    lines = [
        r'\begin{table}[t]', r'  \centering', r'  \begin{threeparttable}',
        r'    \caption{Physical properties of the manipulated objects '
        r'(mass, $\mu$, and $I_{zz}$ read from the simulation; compliance measured '
        r'separately).}',
        r'    \label{tab:object_properties}',
        r'    \begin{tabular}{lccccc}',
        r'      \toprule',
        r'      Object & Shape & Mass & $\mu$\tnote{a} & Compliance\tnote{b} '
        r'& $I_{zz}$\tnote{c} \\',
        r'             &       & (g)  & (--)           & (mm/N)              '
        r'& (g\,cm$^2$) \\',
        r'      \midrule',
    ]
    for i, o in enumerate(objects, start=1):
        recs = by_obj.get(o, [])
        mass = _mode_prop(recs, 'mass_g')
        mu   = _mode_prop(recs, 'mu')
        izz  = _mode_prop(recs, 'izz_gcm2')
        shape = _mode_prop(recs, 'shape')
        shape_s = _SHAPE_NAMES.get(shape, '--') if shape is not None else '--'
        mass_s = f'{mass:.0f}' if mass is not None else r'$-$'
        mu_s   = f'{mu:.2f}'   if mu   is not None else r'$-$'
        izz_s  = f'{izz:.2f}'  if izz  is not None else r'$-$'
        # object name is an internal id; escape underscores for LaTeX
        name_s = o.replace('_', r'\_')
        lines.append(f'      {name_s} & {shape_s} & {mass_s} & {mu_s} & XX.X '
                     f'& {izz_s} \\\\')
    lines += [
        r'      \bottomrule',
        r'    \end{tabular}',
        r'    \begin{tablenotes}[para,flushleft]',
        r'      \footnotesize',
        r'      \item[a] Tangential (sliding) friction coefficient of the object geom '
        r'(\texttt{geom\_friction[0]}), as simulated.',
        r'      \item[b] Normal surface displacement per unit applied load. NOT a native '
        r'simulation quantity here (rigid contacts); measured/estimated separately.',
        r'      \item[c] Principal moment of inertia about the (near-)vertical axis, '
        r'computed by MuJoCo from the object mesh and mass under \emph{uniform density}.',
        r'    \end{tablenotes}',
        r'  \end{threeparttable}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def build_time_table(records, methods, objects, method_labels):
    lines = [
        r'\begin{table}[t]', r'  \centering',
        r'  \caption{Task completion times (s) for teleoperated pick-and-place, '
        r'mean $\pm$ std over per-object trials. Lower is better.}',
        r'  \label{tab:completion_times}',
        r'  \begin{tabularx}{\columnwidth}{l' + 'X' * len(objects) + '}',
        r'    \toprule',
        r'    & \multicolumn{%d}{c}{Completion Time (s) $\downarrow$} \\' % len(objects),
        r'    \cmidrule(lr){2-%d}' % (len(objects) + 1),
        '    Method & ' + ' & '.join(objects) + r' \\',
        r'    \midrule',
    ]
    # completion time is conventionally reported over SUCCESSFUL trials only
    for m in methods:
        cells = []
        for o in objects:
            recs = records.get((m, o), [])
            durs = [r['duration_s'] for r in recs if r['success']]
            cells.append(fmt_time_cell(durs))
        lines.append(f'    {method_labels.get(m, m)} & ' + ' & '.join(cells) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabularx}', r'\end{table}']
    return '\n'.join(lines)


def build_success_table(records, methods, objects, method_labels):
    # `records` already holds ONLY valid trials (terminated & not abandoned; see collect), so
    # every count here is over the valid set. 'Successful pick' is trivially n/n and omitted.
    def stage_counts(m, o):
        recs = records.get((m, o), [])
        n_valid = len(recs)
        n_drops = sum(r['n_drops'] for r in recs)
        n_e2e   = sum(1 for r in recs if r['success'])
        return n_valid, n_drops, n_e2e

    lines = [
        r'\begin{table}[t]', r'  \centering', r'  \begin{threeparttable}',
        r'    \caption{Subtask outcomes over VALID trials (a trial is valid iff it reached a '
        r'terminal state and was not abandoned; a timeout without a successful pick counts as '
        r'a task failure). Transport drops is the total number of drop events over the valid '
        r'trials ($n$), lower is better; end-to-end is successes/valid, higher is better. '
        r'Collision-avoidance is omitted for teleop (not measured in the logs).}',
        r'    \label{tab:subtask_success}',
        r'    \begin{tabular}{ll' + 'c' * len(objects) + '}',
        r'      \toprule',
        r'      & & \multicolumn{%d}{c}{Subtask outcome (per object)} \\'
        % len(objects),
        r'      \cmidrule(lr){3-%d}' % (len(objects) + 2),
        '      Method & Subtask & ' + ' & '.join(objects) + r' \\',
        r'      \midrule',
    ]
    for mi, m in enumerate(methods):
        label = method_labels.get(m, m)
        stab_cells, e2e_cells = [], []
        for o in objects:
            n_valid, n_drops, n_e2e = stage_counts(m, o)
            # Transport drops = number of DROP events over the valid trials (n_valid).
            stab_cells.append(r'$-$' if n_valid == 0
                              else f'{n_drops}~({{\\scriptsize $n{{=}}{n_valid}$}})')
            e2e_cells.append(fmt_frac(n_e2e, n_valid, bold=(m != methods[0])))
        lines.append(r'      \multirow{2}{*}{%s}' % label)
        lines.append(r'        & Transport drops\tnote{a}        & '
                     + ' & '.join(stab_cells) + r' \\')
        lines.append(r'        & End-to-end\tnote{b}             & '
                     + ' & '.join(e2e_cells) + r' \\')
        if mi != len(methods) - 1:
            lines.append(r'      \midrule')
    lines += [
        r'      \bottomrule',
        r'    \end{tabular}',
        r'    \begin{tablenotes}[para,flushleft]',
        r'      \footnotesize',
        r'      \item[a] Total \texttt{drop} events (object fell out of the grasp during '
        r'carry) over the valid trials. $n$ is the valid-trial count; lower is better.',
        r'      \item[b] Trials reaching the place site (\texttt{outcome=success}) over the '
        r'valid trials. Valid = reached a terminal state (success or timeout) and not '
        r'abandoned; a timeout without a successful grasp counts as a task failure. Only '
        r'abandoned (operator reset / target switch) and unterminated trials are excluded.',
        r'    \end{tablenotes}',
        r'  \end{threeparttable}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


METHOD_LABELS = {
    'dexpilot': r'DexPilot \cite{handaDexPilotVisionBasedTeleoperation2020}',
    'contact_aware_teleop':
        r'\begin{tabular}[c]{@{}l@{}}Contact-Aware\\ Teleop (Ours)\end{tabular}',
}


def emit_tables(dirs, args, heading=None):
    """Parse `dirs`, pool trials per (method, object), and print both LaTeX tables to
    stdout (a per-(method,object) summary goes to stderr). `heading` prefixes both the
    stderr summary and the LaTeX output with a comment naming the scope (used by
    --per-run). Returns True if any completed trials were found."""
    records = collect(dirs)
    if not records:
        tag = f' for {heading}' if heading else ''
        print(f'# no completed trials found{tag}', file=sys.stderr)
        return False

    methods = sorted({m for (m, _o) in records}, key=lambda x: (x != 'dexpilot', x))
    objects = args.objects or sorted({o for (_m, o) in records})

    # Summary to stderr so stdout stays pure LaTeX (pipe-friendly).
    if heading:
        print(f'# === {heading} ===', file=sys.stderr)
    print('# VALID trials pooled per (method, object)  '
          '[valid = terminated & not abandoned]:', file=sys.stderr)
    for (m, o), recs in sorted(records.items()):
        n = len(recs)
        ndrops = sum(r['n_drops'] for r in recs)
        ns = sum(r['success'] for r in recs)
        print(f'#   {m:24} {o:16}  n_valid={n:3}  drops={ndrops}  success={ns}',
              file=sys.stderr)
    print('', file=sys.stderr)

    if heading:
        print(f'% ===== {heading} =====')
    print(build_object_properties_table(records, objects))
    print()
    print(build_time_table(records, methods, objects, METHOD_LABELS))
    print()
    print(build_success_table(records, methods, objects, METHOD_LABELS))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+', help='logs/<run>/ directories (globs expanded)')
    ap.add_argument('--objects', nargs='*', default=None,
                    help='explicit object column order/subset (default: auto, sorted)')
    ap.add_argument('--per-run', action='store_true',
                    help='emit a separate table pair per run dir (default: pool all dirs '
                         'into one table pair)')
    args = ap.parse_args()

    # Expand any globs the shell left unexpanded and keep only real dirs with events.jsonl.
    dirs = []
    for pat in args.run_dirs:
        for p in ([pat] if os.path.isdir(pat) else glob.glob(pat)):
            if (Path(p) / 'events.jsonl').is_file():
                dirs.append(p)
    if not dirs:
        sys.exit('no run dirs with events.jsonl found among: ' + ' '.join(args.run_dirs))

    if args.per_run:
        any_trials = False
        for i, d in enumerate(dirs):
            if i:
                print()   # blank line between per-run blocks on stdout
            any_trials |= emit_tables([d], args, heading=d)
        if not any_trials:
            sys.exit('no completed trials found in any of the given runs.')
    else:
        if not emit_tables(dirs, args):
            sys.exit('no completed trials found in the given runs.')


if __name__ == '__main__':
    main()

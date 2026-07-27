#!/usr/bin/env python3
"""Parse --trial-log runs into the paper's completion-time and subtask-success tables.

Consumes one or more logs/<run>/ directories produced by
kinova_leap_pick_place.py --trial-log (see trial_logger.py for the log schema) and emits
two LaTeX tables:

  Table 1  Task completion time (s), mean ± std over N trials per object.
  Table 2  Conditional subtask success rates (successes/attempts), per object×method.

WHAT IS AND ISN'T COMPUTABLE FROM THE CURRENT LOGS
--------------------------------------------------
Directly from events.jsonl:
  * completion time     — trial_end.duration_s
  * successful pick     — a 'pick_confirmed' event exists for the trial (the object was
                          held > LIFT_HEIGHT_M above rest continuously for DWELL_S ≥ the
                          pick-dwell threshold; that IS the "lifted clear for ≥X s" test).
  * end-to-end success  — trial_end.outcome == 'success' (reached the place site).

NOT in events.jsonl — computed OFFLINE here from the per-trial trace (trial_*.npz):
  * stable transport grasp — max_slip_mm in events.jsonl is ALWAYS 0.0 (trial_logger.py's
                          note_slip() is defined but never wired into the control loop), and
                          in-hand rotation is not logged as a scalar at all. Both are
                          reconstructed from the trace:
                            - in-hand slip: object position expressed in the HAND frame
                              (leap_palm), deviation from its value at pick_confirmed. Using
                              the hand frame (not world) removes the legitimate carry motion
                              so only true in-hand sliding is counted.
                            - in-hand rotation: object orientation relative to the palm
                              orientation (exact wrist frame via mj_forward on the logged
                              q_robot), geodesic angle from its pick-time value. Also
                              carry-reorientation-free.

NOT computable for teleop (EXCLUDED, matching the paper): collision-avoidance rate. The
inadvertent-contact counter (n_inadvertent_contacts) only runs during the autonomous RRT
plan/replay; in contact_aware_teleop the pre-lock-in approach is teleoperated and its
contacts are not counted, so the collision-avoidance subtask is intentionally omitted here.

USAGE
-----
  python3 parse_trials_tables.py logs/contact_aware_teleop_*/ [logs/dexpilot_*/ ...]
  python3 parse_trials_tables.py --slip-mm 20 --rot-deg 30 logs/<run>/...

Object columns are auto-discovered from the trials' object names (sorted); pass
--objects to fix an explicit order / subset. Each run dir may contain many trials; pass
as many run dirs as you like and they are pooled per (method, object).
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Trace reconstruction needs the scene model (for the leap_palm wrist frame + FK).
# Imported lazily so --help / pure-events parsing works without mujoco installed.
SCENE_XML = 'models/scene_pick_place.xml'
PALM_BODY = 'leap_palm'
N_ROBOT = 23   # q_robot columns in the trace (matches kinova_leap_pick_place.py)

# Phase code stored per trace row (kinova_leap_pick_place.py step_pick_or_transport sample):
#   0 = PICK, 1 = TRANSPORT. Transport rows are the object-carry segment.
PHASE_TRANSPORT = 1


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
    trial never ended (no trial_end — e.g. the run was killed mid-trial)."""
    start = next((e for e in evs if e.get('event') == 'trial_start'), None)
    end   = next((e for e in evs if e.get('event') == 'trial_end'), None)
    if start is None or end is None:
        return None
    mass_kg = start.get('mass_kg')
    return {
        'method':       start.get('method', 'unknown'),
        'object':       start.get('object', 'unknown'),
        'outcome':      end.get('outcome'),
        'duration_s':   end.get('duration_s'),
        'pick_confirmed': any(e.get('event') == 'pick_confirmed' for e in evs),
        'n_attempts':   end.get('n_attempts', 0),
        'n_drops':      end.get('n_drops', 0),
        # Physical properties, stamped onto trial_start by object_props_from_model()
        # (trial_logger.py). Absent (-> None) in older logs written before the props stamp.
        'mass_g':       (mass_kg * 1e3) if mass_kg is not None else None,
        'mu':           start.get('mu'),
        'izz_gcm2':     start.get('izz_gcm2'),
        'shape':        start.get('shape'),
    }


# ── per-trial trace: in-hand slip + rotation over transport ──────────────────────────────

class TraceFK:
    """Loads the scene model once and reconstructs the leap_palm (wrist/hand) frame per
    trace row from the logged q_robot+obj_qpos, so object pose can be expressed in the
    hand frame — the frame in which 'in-hand slip/rotation' is meaningful."""

    def __init__(self):
        import mujoco as mj   # lazy: only needed when a trace is actually processed
        self._mj = mj
        self._model = mj.MjModel.from_xml_path(SCENE_XML)
        self._data = mj.MjData(self._model)
        self._palm = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_BODY, PALM_BODY)
        if self._palm < 0:
            raise RuntimeError(f"body '{PALM_BODY}' not found in {SCENE_XML}")
        if self._model.nq != N_ROBOT + 7:
            raise RuntimeError(f"model nq={self._model.nq} != {N_ROBOT}+7; trace layout "
                               "assumption (q_robot[23] + obj_qpos[7]) is stale")

    def palm_pose(self, q_robot: np.ndarray, obj_qpos: np.ndarray):
        """Return (p_palm[3], R_palm[3,3]) in world for one reconstructed step."""
        self._data.qpos[:] = np.concatenate([q_robot, obj_qpos])
        self._mj.mj_forward(self._model, self._data)
        return (self._data.xpos[self._palm].copy(),
                self._data.xmat[self._palm].reshape(3, 3).copy())


def _quat_geodesic_deg(q, q_ref):
    """Angle (deg) of the rotation taking q_ref to q. MuJoCo quats are [w,x,y,z]."""
    def _mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([aw*bw - ax*bx - ay*by - az*bz,
                         aw*bx + ax*bw + ay*bz - az*by,
                         aw*by - ax*bz + ay*bw + az*bx,
                         aw*bz + ax*by - ay*bx + az*bw])
    q_ref_inv = np.array([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])
    rel = _mul(q, q_ref_inv)
    w = float(np.clip(abs(rel[0]), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(w))


def transport_slip_rotation(trace_path: Path, fk: TraceFK) -> tuple[float, float] | None:
    """Max in-hand slip (mm) and rotation (deg) over the TRANSPORT segment of one trial's
    trace, both referenced to the object's pose-in-hand at the first transport step
    (== pick_confirmed). Returns None if the trace has no transport rows (e.g. the trial
    was abandoned before a confirmed pick — nothing to score for transport stability)."""
    z = np.load(trace_path)
    if 'phase' not in z.files:
        return None
    tmask = z['phase'] == PHASE_TRANSPORT
    n = int(tmask.sum())
    if n == 0:
        return None
    idx = np.nonzero(tmask)[0]

    obj_pos  = z['obj_pos'];  obj_quat = z['obj_quat']
    q_robot  = z['q_robot'];  obj_qpos = z['obj_qpos']

    # Object pose expressed in the palm frame, per transport step. Reference = first row.
    rel_p = np.empty((n, 3))
    rel_ang_ref = None
    slip = np.empty(n)
    rot  = np.empty(n)
    p0_hand = None
    for k, i in enumerate(idx):
        p_palm, R_palm = fk.palm_pose(q_robot[i], obj_qpos[i])
        # object position in the palm frame (carry motion cancels out)
        p_in_hand = R_palm.T @ (obj_pos[i] - p_palm)
        # object orientation relative to palm: quat(R_palm^T) ∘ obj_quat, as an angle vs ref
        R_rel = R_palm.T @ _quat_to_R(obj_quat[i])
        q_rel = _R_to_quat(R_rel)
        if k == 0:
            p0_hand = p_in_hand
            rel_ang_ref = q_rel
        slip[k] = np.linalg.norm(p_in_hand - p0_hand) * 1e3   # mm
        rot[k]  = _quat_geodesic_deg(q_rel, rel_ang_ref)      # deg
    return float(slip.max()), float(rot.max())


def _quat_to_R(q):
    """MuJoCo [w,x,y,z] quat -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _R_to_quat(R):
    """Rotation matrix -> MuJoCo [w,x,y,z] quat (numerically stable branch)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def find_trace(run_dir: Path, trial_id: int) -> Path | None:
    """Per-trial trace filename is trial_<id:04d>_<method>_<object>_<outcome>.npz
    (TrialRunner._trace_filename). Match on the id prefix, which is unique per run."""
    hits = sorted(run_dir.glob(f'trial_{trial_id:04d}_*.npz'))
    return hits[0] if hits else None


# ── table assembly ───────────────────────────────────────────────────────────────────────

def collect(run_dirs, slip_mm, rot_deg, need_trace):
    """Pool trials across run dirs into records keyed by (method, object). Each record:
      duration_s, picked(bool), stable(bool|None), success(bool).
    stable is None when it can't be scored (no transport segment) — those trials still
    count in the denominators up through the pick stage."""
    fk = None
    records = defaultdict(list)
    for rd in run_dirs:
        rd = Path(rd)
        rows = load_events(rd)
        for tid, evs in group_trials(rows).items():
            s = trial_summary(evs)
            if s is None:
                continue   # trial never ended; not scorable
            picked   = s['pick_confirmed']
            success  = (s['outcome'] == 'success')
            stable = None
            if picked and need_trace:
                tp = find_trace(rd, tid)
                if tp is not None:
                    if fk is None:
                        fk = TraceFK()
                    sr = transport_slip_rotation(tp, fk)
                    if sr is not None:
                        max_slip, max_rot = sr
                        stable = (max_slip <= slip_mm) and (max_rot <= rot_deg)
            records[(s['method'], s['object'])].append({
                'duration_s': s['duration_s'],
                'picked':  picked,
                'stable':  stable,
                'success': success,
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


def build_success_table(records, methods, objects, method_labels, slip_mm, rot_deg):
    def stage_counts(m, o):
        recs = records.get((m, o), [])
        n_all   = len(recs)
        n_pick  = sum(1 for r in recs if r['picked'])
        # stable scored only over picked trials that HAVE a transport trace
        scored  = [r for r in recs if r['picked'] and r['stable'] is not None]
        n_stab_den = len(scored)
        n_stab  = sum(1 for r in scored if r['stable'])
        n_e2e   = sum(1 for r in recs if r['success'])
        return n_all, n_pick, n_stab_den, n_stab, n_e2e

    lines = [
        r'\begin{table}[t]', r'  \centering', r'  \begin{threeparttable}',
        r'    \caption{Subtask success (successes/attempts). Conditional: each subtask is '
        r'scored only over trials in which all preceding subtasks succeeded. '
        r'Collision-avoidance is omitted for teleop (not measured in the current logs). '
        f'Stable transport = in-hand slip $\\le {slip_mm:.0f}$\\,mm and rotation '
        f'$\\le {rot_deg:.0f}^\\circ$ over the carry.}}',
        r'    \label{tab:subtask_success}',
        r'    \begin{tabular}{ll' + 'c' * len(objects) + '}',
        r'      \toprule',
        r'      & & \multicolumn{%d}{c}{Success (successes/attempts) $\uparrow$} \\'
        % len(objects),
        r'      \cmidrule(lr){3-%d}' % (len(objects) + 2),
        '      Method & Subtask & ' + ' & '.join(objects) + r' \\',
        r'      \midrule',
    ]
    for mi, m in enumerate(methods):
        label = method_labels.get(m, m)
        pick_cells, stab_cells, e2e_cells = [], [], []
        for o in objects:
            n_all, n_pick, n_stab_den, n_stab, n_e2e = stage_counts(m, o)
            pick_cells.append(fmt_frac(n_pick, n_all))
            stab_cells.append(fmt_frac(n_stab, n_stab_den))
            e2e_cells.append(fmt_frac(n_e2e, n_all, bold=(m != methods[0])))
        lines.append(r'      \multirow{3}{*}{%s}' % label)
        lines.append(r'        & Successful pick\tnote{a}        & '
                     + ' & '.join(pick_cells) + r' \\')
        lines.append(r'        & Stable transport grasp\tnote{b} & '
                     + ' & '.join(stab_cells) + r' \\')
        lines.append(r'      \cmidrule(lr){2-%d}' % (len(objects) + 2))
        lines.append(r'        & End-to-end                      & '
                     + ' & '.join(e2e_cells) + r' \\')
        if mi != len(methods) - 1:
            lines.append(r'      \midrule')
    lines += [
        r'      \bottomrule',
        r'    \end{tabular}',
        r'    \begin{tablenotes}[para,flushleft]',
        r'      \footnotesize',
        r'      \item[a] Object lifted clear of the support surface and held for the '
        r'pick-dwell window (pick\_confirmed). Denominator: all trials.',
        f'      \\item[b] Object retained without in-hand slip exceeding {slip_mm:.0f}\\,mm '
        f'or rotation exceeding {rot_deg:.0f}$^\\circ$ over the transport trajectory. '
        r'Denominator: trials passing successful pick with a recorded transport segment.',
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
    records = collect(dirs, args.slip_mm, args.rot_deg, need_trace=not args.no_trace)
    if not records:
        tag = f' for {heading}' if heading else ''
        print(f'# no completed trials found{tag}', file=sys.stderr)
        return False

    methods = sorted({m for (m, _o) in records}, key=lambda x: (x != 'dexpilot', x))
    objects = args.objects or sorted({o for (_m, o) in records})

    # Summary to stderr so stdout stays pure LaTeX (pipe-friendly).
    if heading:
        print(f'# === {heading} ===', file=sys.stderr)
    print('# trials pooled per (method, object):', file=sys.stderr)
    for (m, o), recs in sorted(records.items()):
        n = len(recs); npick = sum(r['picked'] for r in recs)
        nstab_den = sum(1 for r in recs if r['picked'] and r['stable'] is not None)
        nstab = sum(1 for r in recs if r['stable'])
        ns = sum(r['success'] for r in recs)
        print(f'#   {m:24} {o:16}  N={n:3}  pick={npick}  '
              f'stable={nstab}/{nstab_den}  success={ns}', file=sys.stderr)
    print('', file=sys.stderr)

    if heading:
        print(f'% ===== {heading} =====')
    print(build_object_properties_table(records, objects))
    print()
    print(build_time_table(records, methods, objects, METHOD_LABELS))
    print()
    print(build_success_table(records, methods, objects, METHOD_LABELS,
                              args.slip_mm, args.rot_deg))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+', help='logs/<run>/ directories (globs expanded)')
    ap.add_argument('--slip-mm', type=float, default=20.0,
                    help='max in-hand slip (mm) for a stable transport grasp (default 20)')
    ap.add_argument('--rot-deg', type=float, default=30.0,
                    help='max in-hand rotation (deg) for a stable transport grasp (default 30)')
    ap.add_argument('--objects', nargs='*', default=None,
                    help='explicit object column order/subset (default: auto, sorted)')
    ap.add_argument('--no-trace', action='store_true',
                    help='skip the trace-based stable-transport stage (events.jsonl only)')
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

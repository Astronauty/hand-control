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
  * pick attempts      — total number of 'attempt_start' events (grasp tries) over the valid
                          trials; more = harder to grasp. Raw count with the valid-trial n.
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


def method_from_dir(run_dir: Path) -> str | None:
    """The retargeter arm, read from the RUN DIRECTORY name ('anyteleop_2026...' ->
    'anyteleop'). Returns None if the name carries no recognized arm.

    Why not trial_start.method: mode normalization collapses every baseline onto the
    dexpilot pipeline (anyteleop/vwj run it with a different retargeter), and the dexpilot
    start_trial call additionally hardcoded the literal 'dexpilot'. So EVERY trial in every
    baseline log — AnyTeleop runs included — recorded method='dexpilot', and keying on that
    field silently pooled the two arms into one column. The directory name is the only
    reliable arm signal in the existing logs. Both are now fixed at the source
    (kinova_leap_pick_place.py logs _RETARGETER), so newer logs agree with the folder;
    this stays as the authority so old and new runs parse the same way."""
    name = Path(run_dir).name.lower()
    for arm in ('contact_aware_w_anyteleop', 'contact_aware_w_vwj_upstream',
                'contact_aware_w_vwj', 'contact_aware_w_dexpilot',
                'contact_aware_teleop', 'contact_aware_autonomous',
                'vwj_upstream', 'anyteleop', 'dexpilot', 'vwj'):
        if name.startswith(arm):
            return arm
    return None


def _ev_t(e: dict) -> float | None:
    """Sim-time of an event (data.time). Sim-time is the right clock for WITHIN-trial
    durations: it is monotonic inside a trial and immune to the sim's real-time factor,
    which varies by several x between runs."""
    t = e.get('t')
    return float(t) if isinstance(t, (int, float)) else None


def read_release_peak_speed(run_dir: Path, trial_id: int) -> float | None:
    """Peak object speed in the RELEASE window, from the per-trial trace npz.

    The trace carries obj_linvel and an int phase code (1 == TRANSPORT, 0 otherwise; see
    the trace.sample call in kinova_leap_pick_place.py). The release window is the tail of
    the trace after transport ends — what separates a clean set-down from chatter or a
    roll-out. Returns None when the trial has no trace file (e.g. abandoned before any
    trace rows) or the trace predates these channels."""
    hits = sorted(Path(run_dir).glob(f'trial_{trial_id:04d}_*.npz'))
    if not hits:
        return None
    try:
        z = np.load(hits[0], allow_pickle=True)
        if 'obj_linvel' not in z or 'phase' not in z or 't' not in z:
            return None
        v = np.linalg.norm(np.asarray(z['obj_linvel'], float), axis=1)
        ph = np.asarray(z['phase'])
        t = np.asarray(z['t'], float)
        tr = np.where(ph == 1)[0]
        if not len(tr):
            return None
        t_end = t[tr[-1]]
        w = (t >= t_end) & (t <= t_end + 1.0)      # 1 s post-transport window
        return float(v[w].max()) if w.any() else None
    except Exception:
        return None


SETTLE_SPEED  = 0.05    # object speed (m/s) below which it counts as come-to-rest.
SETTLE_CAP_S  = 2.0     # how far past an episode to look for that rest. Measured: rest
                        # arrives within 0.25s at the median and 1.70s at p95, but 6 of 47
                        # episodes never settle before the trial ends, so the search is
                        # capped rather than unbounded.
HELD_GAP_M    = 0.08    # fingertip-midpoint-to-object distance below which the object is
                        # treated as STILL HELD, used only to merge an episode across a
                        # mid-carry dip. NOTE this proxy is weak: measured over these runs
                        # the held distribution (p75 65mm, p90 77mm) overlaps the resting
                        # one (p25 60mm, p50 109mm), so 80mm is the least-bad cut rather
                        # than a clean separation. It is deliberately used ONLY for merging,
                        # never to decide whether a grasp succeeded.
REGRIP_GAP_S  = 0.40    # a dip longer than this ends the episode even if the hand stayed
                        # near the object — a genuine re-grasp, not a clip on the bin.
MOVED_M = 0.05          # lateral travel above which a lift counts as a TRANSPORT rather
                        # than a stationary grasp try. Chosen from the measured episode
                        # distribution over the ICRA runs: displacements cluster below
                        # ~50mm (1, 0, 4, 12, 13, 32, 38, 41mm — the object lifted and set
                        # straight back down) and then form a continuum from ~130mm up.
                        # The arm RANKING is insensitive to this cut anywhere in 20-150mm;
                        # only the absolute split between the two rows shifts.


def lift_episodes(run_dir: Path, trial_id: int, lift_m: float = 0.015,
                  min_s: float = 0.15, moved_m: float = MOVED_M,
                  bin_x: tuple = (0.320, 0.680),
                  bin_y: tuple = (-0.080, 0.600)) -> dict | None:
    """Recompute grasp attempts and drops from the per-trial trace, which is the
    AUTHORITY for both. Returns {'attempts': int, 'drops': int, 'episodes': [...]}, or
    None when the trial has no usable trace.

    A lift episode is the object rising `lift_m` above the surface it was resting on and
    staying up for at least `min_s`. Each episode is then classified by how far the object
    actually TRAVELLED, which is what distinguishes the two failure modes:

      ATTEMPT — the object was grasped and lifted but went nowhere (lateral travel
                <= `moved_m`): a grasp try that did not turn into a carry.
      DROP    — the object was carried somewhere (travel > `moved_m`) and ended up back on
                a surface OUTSIDE the target bin: it was lost in transport.

    An episode that travels and ends INSIDE the bin is a placement, not a drop — without
    that test every successful trial's final carry would be counted as a loss.

    Classifying by displacement, rather than by which side of pick_confirmed an episode
    falls on, is what keeps the two counts independent. Earlier cuts of this function
    anchored on the state machine and made drops a deterministic function of attempts.

    Why recompute rather than trust the logged counts:
      * attempt_start fired on the raw grasp-trigger rising edge with no lift requirement
        and no debounce, so it counted contact chatter — measured median inter-attempt gap
        12ms, median attempt duration 4ms. One trial logged 119 attempts for 5 real lifts.
      * drop detection only ran in TRANSPORT and tested height_above_rest < LIFT_HEIGHT_M.
        height_above_rest is referenced to the TABLE top, so in these runs it never fell
        below +0.85 and the test could not fire at all. Worse, 17 of 20 trials had their
        real lift/lose cycles BEFORE pick_confirmed, i.e. before TRANSPORT was ever
        entered, so they were invisible by construction. The logger recorded 1 drop across
        20 trials; the traces show ~47 lift episodes, about half landing back on a surface.

    The surface baseline is the running minimum of object z over a trailing 2s window, so
    it adapts when the object is set down somewhere new (counter vs bin floor) and needs no
    rest-height constant."""
    hits = sorted(Path(run_dir).glob(f'trial_{trial_id:04d}_*.npz'))
    if not hits:
        return None
    try:
        z = np.load(hits[0], allow_pickle=True)
        if 'obj_pos' not in z or 't' not in z:
            return None
        t = np.asarray(z['t'], float)
        op = np.asarray(z['obj_pos'], float)
        oz = op[:, 2]
    except Exception:
        return None
    if len(t) < 2:
        return None
    # Optional channels: used to keep an episode open across a mid-carry dip, and to find
    # where the object actually came to rest. Absent in older traces -> the simpler
    # episode-end behaviour, which is what this function did originally.
    vel = (np.linalg.norm(np.asarray(z['obj_linvel'], float), axis=1)
           if 'obj_linvel' in z else None)
    hand = (0.5 * (np.asarray(z['p_thumb'], float) + np.asarray(z['p_index'], float))
            if ('p_thumb' in z and 'p_index' in z) else None)
    gap = np.linalg.norm(hand - op, axis=1) if hand is not None else None

    lifted = np.zeros(len(t), bool)
    for i in range(len(t)):
        w0 = np.searchsorted(t, t[i] - 2.0)
        base = oz[w0:i + 1].min() if i > w0 else oz[i]
        lifted[i] = (oz[i] - base) > lift_m

    def _settle_idx(j, limit):
        """Index where the object comes to REST after the episode ends.

        Landing must be judged where the object stops, not where it was the instant the
        lift ended. A golf ball released above the bin was still 100mm up and short of the
        rim at episode end; it travelled a further 131mm over 1.56s and settled on the bin
        floor. Judging at episode end scored that placement as a drop.

        `limit` is the first index of the NEXT lift episode: the search must never run past
        the moment the object is picked up again, or it samples a position from the middle
        of the following grasp. That bug read a mug's 824mm carry as ending outside the bin
        (it sampled 1.7s later, inside a re-grasp that began 0.4s after the carry ended)
        and gave a stationary 12mm attempt 518mm of 'settle travel'."""
        if vel is None:
            return j
        k = j
        t_cap = t[j] + SETTLE_CAP_S
        while k < min(limit, len(t)) and t[k] <= t_cap:
            if vel[k] < SETTLE_SPEED:
                k2 = min(int(np.searchsorted(t, t[k] + 0.1)), limit - 1)
                if k2 <= k or np.all(vel[k:k2 + 1] < SETTLE_SPEED):
                    return k
            k += 1
        # Never settled before the cap or the next pick-up (still rolling, re-grasped
        # immediately, or the trial ended): take the last frame in the allowed window.
        return max(j, min(int(np.searchsorted(t, t_cap)), limit - 1, len(t) - 1))

    eps, i = [], 0
    while i < len(lifted):
        if lifted[i]:
            j = i
            while j + 1 < len(lifted) and lifted[j + 1]:
                j += 1
            # MERGE across a momentary dip while the object is STILL HELD. The height
            # baseline is a trailing minimum, so an object that clips the bin wall mid-carry
            # dips under the lift threshold for a few frames and the episode closes there —
            # mid-transport, still gripped — and then gets classified by wherever it
            # happened to be. That is how a successful golf-ball placement was scored as a
            # drop: 626mm travelled, hand still 17mm from the object, outside the footprint
            # at that instant. If the hand is still on the object and the lift resumes
            # shortly, it is one carry, not two episodes.
            while gap is not None and j + 1 < len(lifted):
                nxt = j + 1
                while nxt < len(lifted) and not lifted[nxt]:
                    nxt += 1
                if nxt >= len(lifted) or (t[nxt] - t[j]) > REGRIP_GAP_S:
                    break
                if gap[j:nxt + 1].max() > HELD_GAP_M:
                    break        # hand let go during the dip -> a genuine episode boundary
                j = nxt
                while j + 1 < len(lifted) and lifted[j + 1]:
                    j += 1
            if t[j] - t[i] >= min_s:
                # First frame of the NEXT lift episode (or end of trace): the settle search
                # must stop there, never sampling a position from inside a later grasp.
                nxt_lift = j + 1
                while nxt_lift < len(lifted) and not lifted[nxt_lift]:
                    nxt_lift += 1
                s = _settle_idx(j, nxt_lift)
                dxy = float(np.linalg.norm(op[j, :2] - op[i, :2]))
                in_bin = bool(bin_x[0] <= op[s, 0] <= bin_x[1]
                              and bin_y[0] <= op[s, 1] <= bin_y[1])
                eps.append({'t0': float(t[i]), 't1': float(t[j]),
                            't_settle': float(t[s]),
                            'peak_m': float(oz[i:j + 1].max() - oz[i]),
                            'end_z': float(oz[s]), 'dxy_m': dxy, 'ended_in_bin': in_bin,
                            'settle_travel_m': float(
                                np.linalg.norm(op[s, :2] - op[j, :2])),
                            'kind': ('attempt' if dxy <= moved_m
                                     else ('placed' if in_bin else 'drop'))})
            i = j + 1
        else:
            i += 1
    return {'attempts': sum(1 for e in eps if e['kind'] == 'attempt'),
            'drops':    sum(1 for e in eps if e['kind'] == 'drop'),
            'lifts':    len(eps),
            'episodes': eps}


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
    # Subtask flags, from the phase machine's events:
    #   pickup    — a 'pick_confirmed' fired (object grasped + lifted past the pick threshold).
    #   transport — reached 'arrival' (carried to the place target) GIVEN it was picked. Drops
    #               happen between pick_confirmed and arrival, so a picked-but-dropped trial has
    #               picked=True, arrival=False -> transport failure.
    #   dropoff   — 'arrival.set_down' (actually placed into the target, not just hovering over
    #               it) GIVEN it arrived. In this state machine arrival => outcome success, so
    #               end-to-end success == pickup AND transport AND dropoff.
    picked   = any(e.get('event') == 'pick_confirmed' for e in evs)
    arrival  = next((e for e in evs if e.get('event') == 'arrival'), None)
    arrived  = arrival is not None
    # Drop-off = the object ended up in the target. The authoritative signal is the trial's
    # SUCCESS outcome (the state machine sets outcome=success exactly when the object is placed
    # in the target). arrival.set_down is a finer "set down vs hovering" flag, but it's absent
    # in older logs and in the arrival-recovery path — so key drop-off off the outcome, which
    # keeps it consistent with end-to-end (a successful trial always completed drop-off).
    succeeded = end.get('outcome') == 'success'
    # Retarget/solve latencies: log_solve rows carry {component, ms}. The retargeter latency
    # (dexpilot vs anyteleop) is the headline per-method cost; also expose ik/rrt if present.
    solve_ms = {}
    for e in evs:
        if e.get('event') == 'solve' or ('component' in e and 'ms' in e):
            comp = e.get('component'); ms = e.get('ms')
            if comp is not None and ms is not None:
                solve_ms.setdefault(comp, []).append(float(ms))
    # --- Subtask DURATIONS, in sim-time (see _ev_t) -------------------------------------
    # pick      : phase_enter(PICK) -> pick_confirmed   (grasp committed -> sustained lift)
    # transport : pick_confirmed    -> arrival          (carry)
    # Both are None when the trial never reached that stage, so they never fabricate a 0.
    _t_pick_enter = next((_ev_t(e) for e in evs
                          if e.get('event') == 'phase_enter' and e.get('phase') == 'PICK'),
                         None)
    _pick_ev = next((e for e in evs if e.get('event') == 'pick_confirmed'), None)
    _t_pick_ok = _ev_t(_pick_ev) if _pick_ev else None
    _t_arrival = _ev_t(arrival) if arrival else None
    pick_dur = (None if (_t_pick_enter is None or _t_pick_ok is None)
                else max(0.0, _t_pick_ok - _t_pick_enter))
    transport_dur = (None if (_t_pick_ok is None or _t_arrival is None)
                     else max(0.0, _t_arrival - _t_pick_ok))
    return {
        'method':       start.get('method', 'unknown'),
        'object':       start.get('object', 'unknown'),
        'outcome':      end.get('outcome'),
        'duration_s':   end.get('duration_s'),
        'duration_sim_s': end.get('duration_sim_s'),
        'pick_duration_s':      pick_dur,
        'transport_duration_s': transport_dur,
        'pick_confirmed': picked,
        # Subtask outcomes (see above). Conditional rates are computed in collect().
        'pickup':       picked,
        'transport':    (picked and (arrived or succeeded)),
        'dropoff':      succeeded,
        # Count attempt_start / drop EVENTS directly, not the trial_end.n_* fields. The
        # events are always logged, whereas n_attempts/n_drops are only on trial_end — an
        # arrival-only success (synthesized end above) would otherwise report 0. This counts
        # grasp attempts and drops-into-success correctly for all end types (trial_end,
        # timeout, and arrival-recovered).
        'n_attempts':   sum(1 for e in evs if e.get('event') == 'attempt_start'),
        'n_drops':      sum(1 for e in evs if e.get('event') == 'drop'),
        # Per-component solve latencies (ms) collected across the trial (list per component).
        'solve_ms':     solve_ms,
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

    A trial is VALID iff it reached a terminal state — trial_end (success, timeout, or
    abandoned) or a successful arrival. Every terminated trial that did not succeed is a
    TASK FAILURE and counts against end-to-end:
      * timeout   — could not grasp the object within the time budget.
      * abandoned — the operator ended a trial they could not finish (object rolled off the
                    table, or out of reach). A failure of the method, not an exclusion.
    Only unterminated trials (run killed mid-trial — trial_summary returns None) are
    dropped. Excluding either failure mode inflates every arm, and inflates the worse arm
    most, because the worse arm is precisely the one that fails more often. Each record:
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
            # NOTE: 'abandoned' trials are INCLUDED, as task failures. In these runs an
            # abandon means the object rolled off the table or could not be reached — the
            # operator ending a trial they cannot finish, which is a failure of the
            # method, not an experimental exclusion. Dropping them inflates every arm and
            # inflates the WORSE arm most: it removed 3 DexPilot trials (32 attempts and
            # never picked; 50 attempts, picked then lost) against 1 AnyTeleop trial, and
            # turned a 7/10 vs 9/10 result into a meaningless 100% vs 100%.
            # Only unterminated trials (run killed mid-trial) are dropped, above.
            # Per-trial retarget latency (mean ms) if the 'solve' component=='retarget' event
            # was logged; and per-trial grasp-hold force from trial_end (peak/mean N).
            retarget_ms = None
            comp = s.get('solve_ms', {})
            if 'retarget' in comp and comp['retarget']:
                retarget_ms = float(np.mean(comp['retarget']))
            _end = next((e for e in evs if e.get('event') == 'trial_end'), {})
            # Arm from the DIRECTORY, not trial_start.method — see method_from_dir.
            _arm = method_from_dir(rd) or s['method']
            _lift = lift_episodes(rd, tid)
            records[(_arm, s['object'])].append({
                'duration_s': s['duration_s'],
                'duration_sim_s': s.get('duration_sim_s'),
                'pick_duration_s': s.get('pick_duration_s'),
                'transport_duration_s': s.get('transport_duration_s'),
                'release_peak_mps': read_release_peak_speed(rd, tid),
                'timeout':   (s['outcome'] == 'timeout'),
                'abandoned': (s['outcome'] == 'abandoned'),
                # Trace-derived attempts/drops — the AUTHORITY (see lift_episodes). Kept
                # alongside the logged n_attempts/n_drops so the disagreement on runs
                # recorded before the logger fix is visible rather than silent.
                'attempts_trace': (_lift or {}).get('attempts'),
                'drops_trace':    (_lift or {}).get('drops'),
                'lifts_trace':    (_lift or {}).get('lifts'),
                'n_attempts': s.get('n_attempts', 0) or 0,
                'n_drops':   s.get('n_drops', 0) or 0,
                'success':   (s['outcome'] == 'success'),
                # Subtask flags (booleans) for the conditional success-rate table.
                'pickup':    bool(s.get('pickup')),
                'transport': bool(s.get('transport')),
                'dropoff':   bool(s.get('dropoff')),
                # Comparison scalars.
                'retarget_ms':      retarget_ms,
                'grip_force_peak_n': _end.get('grip_force_peak_n'),
                'grip_force_mean_n': _end.get('grip_force_mean_n'),
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
        n_att   = sum(r['n_attempts'] for r in recs)
        n_drops = sum(r['n_drops'] for r in recs)
        n_e2e   = sum(1 for r in recs if r['success'])
        return n_valid, n_att, n_drops, n_e2e

    lines = [
        r'\begin{table}[t]', r'  \centering', r'  \begin{threeparttable}',
        r'    \caption{Subtask outcomes over the $n$ VALID trials per object (a trial is valid '
        r'iff it reached a terminal state and was not abandoned; a timeout without a '
        r'successful pick counts as a task failure). Pick attempts and transport drops are '
        r'totals over those trials, lower is better; end-to-end is successes/$n$, higher is '
        r'better. Collision-avoidance is omitted for teleop (not measured in the logs).}',
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
        n_cells, att_cells, stab_cells, e2e_cells = [], [], [], []
        for o in objects:
            n_valid, n_att, n_drops, n_e2e = stage_counts(m, o)
            # n is a property of the (method, object) cell — reported once in its own row,
            # not repeated on every subtask. Attempts/drops are then bare totals.
            n_cells.append(str(n_valid))
            att_cells.append(r'$-$' if n_valid == 0 else str(n_att))
            stab_cells.append(r'$-$' if n_valid == 0 else str(n_drops))
            e2e_cells.append(fmt_frac(n_e2e, n_valid, bold=(m != methods[0])))
        lines.append(r'      \multirow{4}{*}{%s}' % label)
        lines.append(r'        & Valid trials ($n$)             & '
                     + ' & '.join(n_cells) + r' \\')
        lines.append(r'        & Pick attempts\tnote{a}          & '
                     + ' & '.join(att_cells) + r' \\')
        lines.append(r'        & Transport drops\tnote{b}        & '
                     + ' & '.join(stab_cells) + r' \\')
        lines.append(r'        & End-to-end\tnote{c}             & '
                     + ' & '.join(e2e_cells) + r' \\')
        if mi != len(methods) - 1:
            lines.append(r'      \midrule')
    lines += [
        r'      \bottomrule',
        r'    \end{tabular}',
        r'    \begin{tablenotes}[para,flushleft]',
        r'      \footnotesize',
        r'      \item[a] Total grasp attempts (\texttt{attempt\_start} events) over the '
        r'$n$ valid trials; more attempts means the object was harder to grasp. Lower is '
        r'better.',
        r'      \item[b] Total \texttt{drop} events (object fell out of the grasp during '
        r'carry) over the $n$ valid trials. Lower is better.',
        r'      \item[c] Trials reaching the place site (\texttt{outcome=success}) over the '
        r'valid trials. Valid = reached a terminal state (success or timeout) and not '
        r'abandoned; a timeout without a successful grasp counts as a task failure. Only '
        r'abandoned (operator reset / target switch) and unterminated trials are excluded.',
        r'    \end{tablenotes}',
        r'  \end{threeparttable}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def build_subtask_conditional_table(records, methods, objects, method_labels):
    """CONDITIONAL subtask success rates + comparison scalars, per method (pooled over
    objects). Unlike build_success_table (per-object raw counts), this decomposes the task
    into the pipeline the reviewer asked for:
      Pickup    = trials with a pick_confirmed / n_valid
      Transport = trials that reached arrival GIVEN picked  (drops fail here)
      Drop-off  = trials placed in the target GIVEN arrived
      End-to-end = Pickup x Transport x Drop-off (== overall success)
    plus the two comparison scalars: finger-retarget latency (ms) and grasp-hold force (N)."""
    def pooled(m):
        recs = [r for o in objects for r in records.get((m, o), [])]
        n = len(recs)
        n_pick = sum(1 for r in recs if r.get('pickup'))
        n_trans = sum(1 for r in recs if r.get('transport'))
        n_drop = sum(1 for r in recs if r.get('dropoff'))
        n_e2e = sum(1 for r in recs if r['success'])
        lat = [r['retarget_ms'] for r in recs if r.get('retarget_ms') is not None]
        gfp = [r['grip_force_peak_n'] for r in recs if r.get('grip_force_peak_n') is not None]
        gfm = [r['grip_force_mean_n'] for r in recs if r.get('grip_force_mean_n') is not None]
        return dict(n=n, n_pick=n_pick, n_trans=n_trans, n_drop=n_drop, n_e2e=n_e2e,
                    lat=lat, gfp=gfp, gfm=gfm)

    def frac(a, b):
        return r'$-$' if b == 0 else f'{a}/{b} ({100.0*a/b:.0f}\\%)'

    def stat(vals, unit):
        if not vals:
            return r'$-$'
        return f'${np.mean(vals):.1f} \\pm {np.std(vals):.1f}$ {unit}'

    lines = [
        r'\begin{table}[t]', r'  \centering', r'  \begin{threeparttable}',
        r'    \caption{Conditional subtask success (pooled over objects) and comparison '
        r'scalars per method. Transport is conditioned on a successful pick, drop-off on a '
        r'successful transport, so end-to-end $=$ pickup $\times$ transport $\times$ drop-off. '
        r'Retarget latency is the per-frame finger-retargeting solve time; grasp-hold force is '
        r'the finger$\to$object normal force during transport (the method''s ability to '
        r'maintain the grasp under load).}',
        r'    \label{tab:subtask_conditional}',
        r'    \begin{tabular}{l' + 'c' * len(methods) + '}',
        r'      \toprule',
        '      & ' + ' & '.join(method_labels.get(m, m) for m in methods) + r' \\',
        r'      \midrule',
    ]
    P = {m: pooled(m) for m in methods}
    rows = [
        ('Valid trials ($n$)', lambda p: str(p['n'])),
        ('Pickup success',     lambda p: frac(p['n_pick'], p['n'])),
        ('Transport success (| pick)', lambda p: frac(p['n_trans'], p['n_pick'])),
        ('Drop-off success (| transport)', lambda p: frac(p['n_drop'], p['n_trans'])),
        ('End-to-end success', lambda p: frac(p['n_e2e'], p['n'])),
        (r'Retarget latency\tnote{a}', lambda p: stat(p['lat'], 'ms')),
        (r'Grasp-hold force (peak)\tnote{b}', lambda p: stat(p['gfp'], 'N')),
        (r'Grasp-hold force (mean)\tnote{b}', lambda p: stat(p['gfm'], 'N')),
    ]
    for label, fn in rows:
        lines.append('      ' + label + ' & ' + ' & '.join(fn(P[m]) for m in methods) + r' \\')
    lines += [
        r'      \bottomrule',
        r'    \end{tabular}',
        r'    \begin{tablenotes}[para,flushleft]',
        r'      \footnotesize',
        r'      \item[a] Mean per-frame finger-retargeting solve time (\texttt{solve} events, '
        r'component \texttt{retarget}), over valid trials. Lower is better.',
        r'      \item[b] Total finger$\to$object normal force during the transport phase '
        r'(\texttt{trial\_end.grip\_force\_*}). A method that cannot command grip force '
        r'independent of finger position holds weakly and drops under load.',
        r'    \end{tablenotes}',
        r'  \end{threeparttable}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


METHOD_LABELS = {
    'dexpilot': r'DexPilot \cite{handaDexPilotVisionBasedTeleoperation2020}',
    'anyteleop': r'AnyTeleop \cite{qinAnyTeleopGeneralVisionBased2023}',
    'vwj': r'VWJ \cite{xinAnalyzingKeyObjectives2025}',
    'vwj_upstream': r'VWJ (upstream)',
    'contact_aware_teleop':
        r'\begin{tabular}[c]{@{}l@{}}Contact-Aware\\ Teleop (Ours)\end{tabular}',
    'contact_aware_w_dexpilot':
        r'\begin{tabular}[c]{@{}l@{}}Ours\\ + DexPilot\end{tabular}',
    'contact_aware_w_anyteleop':
        r'\begin{tabular}[c]{@{}l@{}}Ours\\ + AnyTeleop\end{tabular}',
    'contact_aware_w_vwj':
        r'\begin{tabular}[c]{@{}l@{}}Ours\\ + VWJ\end{tabular}',
}


def _median_iqr(vals, unit='', fmt='.1f'):
    """Median [Q1, Q3] — the distribution summary to report instead of mean +/- std.

    These durations are strongly right-skewed (one object needing 119 grasp attempts sits
    in the same cell as one needing 1), so a mean +/- std implies a symmetric spread that
    does not exist and hides the tail. The IQR shows it."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return r'$-$'
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f'${med:{fmt}}$ [{q1:{fmt}}, {q3:{fmt}}]{unit}'


def build_metrics_table(records, methods, objects, method_labels):
    """The per-method metric panel: everything the current logs actually support, with the
    unsupported metrics named explicitly rather than quietly omitted.

    Every rate's DENOMINATOR is stated in its row label, because a conditional rate without
    one is uninterpretable. Valid = terminated and not abandoned; a timeout that never
    grasped IS valid and counts as a failure (excluding timeouts inflates every arm, and
    inflates the worse arm most)."""
    def pooled(m):
        recs = [r for o in objects for r in records.get((m, o), [])]
        n = len(recs)
        succ = [r for r in recs if r['success']]
        picked = [r for r in recs if r.get('pickup')]
        return dict(
            n=n, recs=recs,
            n_e2e=len(succ), n_pick=len(picked),
            n_trans=sum(1 for r in recs if r.get('transport')),
            n_timeout=sum(1 for r in recs if r.get('timeout')),
            n_aband=sum(1 for r in recs if r.get('abandoned')),
            t_total=[r['duration_s'] for r in succ],
            t_pick=[r.get('pick_duration_s') for r in recs],
            t_trans=[r.get('transport_duration_s') for r in recs],
            attempts=[r.get('n_attempts', 0) for r in recs],
            drops=[r.get('n_drops', 0) for r in picked],
            attempts_tr=[r.get('attempts_trace') for r in recs],
            drops_tr=[r.get('drops_trace') for r in recs],
            lifts_tr=[r.get('lifts_trace') for r in recs],
            vrel=[r.get('release_peak_mps') for r in recs],
            lat=[r['retarget_ms'] for r in recs if r.get('retarget_ms') is not None],
            gfp=[r['grip_force_peak_n'] for r in recs
                 if r.get('grip_force_peak_n') not in (None, 0.0)],
        )

    def frac(a, b):
        return r'$-$' if not b else f'{a}/{b} ({100.0*a/b:.0f}\\%)'

    def permean(vals):
        v = [x for x in vals if x is not None]
        return r'$-$' if not v else f'${np.mean(v):.1f}$'

    P = {m: pooled(m) for m in methods}
    rows = [
        (r'\multicolumn{%d}{l}{\emph{Task}} \\' % (len(methods) + 1), None),
        ('Valid trials ($n$)', lambda p: str(p['n'])),
        ('End-to-end success (/ valid)', lambda p: frac(p['n_e2e'], p['n'])),
        ('  failures: timeout', lambda p: str(p['n_timeout'])),
        ('  failures: abandoned\\tnote{c}', lambda p: str(p['n_aband'])),
        ('Completion time (s), median [IQR]', lambda p: _median_iqr(p['t_total'])),
        (r'\midrule \multicolumn{%d}{l}{\emph{Pick}} \\' % (len(methods) + 1), None),
        ('Pick success (/ valid)', lambda p: frac(p['n_pick'], p['n'])),
        (r'Total lifts per trial\tnote{d}', lambda p: permean(p['lifts_tr'])),
        (r'\quad of which stationary (attempts)', lambda p: permean(p['attempts_tr'])),
        (r'\quad (attempts as logged, uncorrected)', lambda p: permean(p['attempts'])),
        ('Pick duration (s), median [IQR]', lambda p: _median_iqr(p['t_pick'])),
        (r'\midrule \multicolumn{%d}{l}{\emph{Transport}} \\' % (len(methods) + 1), None),
        ('Transport success (/ picked)', lambda p: frac(p['n_trans'], p['n_pick'])),
        (r'Drops per trial\tnote{d}', lambda p: permean(p['drops_tr'])),
        (r'\quad (as logged, uncorrected)', lambda p: permean(p['drops'])),
        ('Transport duration (s), median [IQR]', lambda p: _median_iqr(p['t_trans'])),
        ('Release peak speed (m/s), median [IQR]',
         lambda p: _median_iqr(p['vrel'], fmt='.3f')),
        (r'\midrule \multicolumn{%d}{l}{\emph{Physical / computation}} \\'
         % (len(methods) + 1), None),
        (r'Grasp-hold force, peak (N)\tnote{a}', lambda p: _median_iqr(p['gfp'])),
        (r'Retarget latency (ms)\tnote{b}', lambda p: _median_iqr(p['lat'], fmt='.2f')),
    ]
    lines = [
        r'\begin{table}[t]', r'  \centering', r'  \begin{threeparttable}',
        r'    \caption{Teleoperation baselines, pooled over objects. Distributions are '
        r'reported as median [Q1, Q3]: the completion-time spread is the result, and a '
        r'mean $\pm$ std would imply a symmetry these trials do not have. Every rate names '
        r'its denominator. A timeout without success counts as a failure, not an exclusion.}',
        r'    \label{tab:teleop_metrics}',
        r'    \begin{tabular}{l' + 'c' * len(methods) + '}',
        r'      \toprule',
        '      Metric & ' + ' & '.join(method_labels.get(m, m) for m in methods) + r' \\',
        r'      \midrule',
    ]
    for label, fn in rows:
        if fn is None:
            lines.append('      ' + label)
            continue
        lines.append('      ' + label + ' & '
                     + ' & '.join(fn(P[m]) for m in methods) + r' \\')
    lines += [
        r'      \bottomrule',
        r'    \end{tabular}',
        r'    \begin{tablenotes}[para,flushleft]',
        r'      \footnotesize',
        r'      \item[a] Finger$\to$object normal force during transport '
        r'(\texttt{trial\_end.grip\_force\_peak\_n}); trials reporting exactly 0 N are '
        r'excluded as unmeasured rather than averaged in as zero.',
        r'      \item[b] Per-frame retargeting solve time — the like-for-like planning cost. '
        r'Absent from runs recorded before the emitter guard was fixed.',
        r'      \item[c] Trials the operator ended without completing — the object rolled '
        r'off the table or could not be reached. Counted as task failures, not excluded.',
        r'      \item[e] Every lift of the object off its resting surface, decomposed as '
        r'lifts $=$ stationary attempts $+$ drops $+$ placements. A lift that travels '
        r'further than 50\,mm and ends outside the bin is a drop; one that travels and '
        r'ends inside it is the placement. The stationary row is small by construction: '
        r'most lifts in these runs became carries.',
        r'      \item[d] Recomputed from the per-trial object trace (a grasp attempt is a '
        r'sustained lift off the resting surface; a drop is a lift that ends back on one). '
        r'The logged rows beneath are shown for comparison: attempt events fired on the raw '
        r'grasp-trigger edge with no lift test or debounce (median 4\,ms duration, so they '
        r'counted contact chatter), and drop detection ran only in TRANSPORT against a '
        r'table-referenced height that could not fire once the object was in the bin.',
        r'    \end{tablenotes}',
        r'  \end{threeparttable}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# Metrics that were requested but CANNOT be produced from these logs. Printed to stderr so
# the gap is explicit in the run output rather than showing up as a silent dash in a cell.
UNAVAILABLE_METRICS = [
    ('Gap-gate abort rate',
     "attempt_result.outcome is 'released' for every attempt in these runs — the "
     "squeeze_aborted_no_contact outcome is emitted only by benchmarks/ycb_grasp/*, "
     "not by this teleop pipeline."),
    ('Certified squeeze gamma*, force-tracking error, excess force, '
     'certificate issuance rate',
     'no solve/certificate events: these come from the contact-aware grasp path, which '
     'the DexPilot/AnyTeleop baseline arms never execute.'),
    ('Per-grasp solve time, fingertip residual, one-time SDF setup cost',
     'same reason — no grasp_rec/ik solve events in baseline-only runs.'),
]


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
          '[valid = terminated; timeout/abandoned count as failures]:', file=sys.stderr)
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
    print(build_metrics_table(records, methods, objects, METHOD_LABELS))
    if getattr(args, 'per_object', False):
        # One metric panel per object. Same builder, single-object `objects` list, so every
        # rate keeps its denominator and the per-object n is visible (it is small — 2 trials
        # per object per arm in the ICRA runs — so read these as a breakdown of the pooled
        # table above, not as independently powered comparisons).
        for _o in objects:
            if not any((m, _o) in records for m in methods):
                continue
            print()
            print(f'% ----- per-object: {_o} -----')
            print(build_metrics_table(records, methods, [_o], METHOD_LABELS))
    print()
    print(build_time_table(records, methods, objects, METHOD_LABELS))
    print()
    print(build_success_table(records, methods, objects, METHOD_LABELS))
    print()
    print(build_subtask_conditional_table(records, methods, objects, METHOD_LABELS))
    print('# metrics NOT computable from these logs:', file=sys.stderr)
    for name, why in UNAVAILABLE_METRICS:
        print(f'#   {name}\n#     -> {why}', file=sys.stderr)
    print('', file=sys.stderr)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+', help='logs/<run>/ directories (globs expanded)')
    ap.add_argument('--objects', nargs='*', default=None,
                    help='explicit object column order/subset (default: auto, sorted)')
    ap.add_argument('--per-object', dest='per_object', action='store_true',
                    help='also emit one metric panel per object (breakdown of the pooled '
                         'table; per-object n is small, so treat it as diagnostic)')
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

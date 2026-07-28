#!/usr/bin/env python3
"""Trial/event logging and success-detection state machine for automated pick-and-place
benchmarking (kinova_leap_pick_place.py, --mode dexpilot | contact_aware_teleop).

Two correlated output streams per run, written under logs/<run>/:
  events.jsonl              — one line per state transition (sparse, human-readable).
                               Every event carries trial_id + t (sim-time, data.time)
                               so it can be joined against the per-step trace by time.
  trial_<id>_trace.npz      — one array set per trial, one row per mj_step (full
                               physics resolution): fingertip positions/forces, object
                               pose/velocity. Buffered in memory, written once at
                               trial_end (no mid-trial appends).

Design note: this module owns NO MuJoCo model/data references beyond what's passed
into each call — it is deliberately decoupled from kinova_leap_pick_place.py's control
loop so the state machine can be unit-tested standalone before wiring it in.

State machine (settled spec — see conversation record, not re-derived here):
  Approach:  n_inadvertent_contacts — counts contact EPISODES (not-touching->touching)
             between a hand geom and floor/non-target-object geom, penetration >0.5mm.
             Target-object contact is exempt. Counted through the RRT plan+replay part of
             PICK; stops at the first attempt_start (grasp attempt begins).
  Pick:      attempts are unbounded (no MAX_ATTEMPTS) — a trial ends only on arrival
             or on TRIAL_TIMEOUT_S. An attempt starts on a method-specific trigger
             rising edge (see AttemptTrigger below) and is confirmed after the object
             holds LIFT_HEIGHT_M above its rest height continuously for DWELL_S. A dip
             below threshold before DWELL_S elapses resets the dwell timer WITHOUT
             ending the attempt, as long as the trigger condition is still active.
  Transport: entered at pick_confirmed. Arrival is a SET-IN-PLACE check: object XY within
             the place-site marker footprint AND |v| < ARRIVAL_SPEED_M_S (settled), whether
             held clear over the target or set down into it -> trial success. Falling below
             LIFT_HEIGHT_M *outside* the footprint counts a drop (phase reverts to Pick,
             awaiting the next trigger edge); falling below it *inside* the footprint is a
             placement in progress (stay in Transport until it settles), not a drop.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ── Tunable constants (all settled in the design conversation) ────────────────────────
LIFT_HEIGHT_M      = 0.01  # object clearance above rest height counted as "lifted"
DWELL_S             = 1.0    # continuous sim-time above LIFT_HEIGHT_M to confirm a pick
CONTACT_PENETRATION_M = 0.0005  # min penetration depth to count as a real contact (not solver margin)
CONTACT_EPISODE_COOLDOWN_S = 0.5  # min sim-time gap between counted episodes — a
                                   # sustained rest/scrape cycles through many hand
                                   # sub-geom pairs (e.g. the palm's 10 collision
                                   # pieces) within milliseconds; without this, each
                                   # newly-engaging sub-geom pair counted as a fresh
                                   # episode, exploding one real touch into hundreds
                                   # (observed in a live dexpilot log).
ARRIVAL_SPEED_M_S  = 0.05   # object linear speed below which it's considered "settled"
TRIAL_TIMEOUT_S     = 60.0   # trial force-ends (outcome='timeout') past this sim-time
PINCH_EPS_M         = 0.03   # DexPilot: min(d_s1) below this = operator fingers pinched
                              # (matches DexPilotRetargeter.EPS; kept here as the
                              # trial-logger's own copy so this module has no import-time
                              # dependency on teleop/dexpilot_retargeter.py)


# ── Geometry helpers ────────────────────────────────────────────────────────────────────

# MuJoCo geom type ids (mirrors simulation/test_grasp_robustness.py's local constants —
# not imported from mujoco.mjtGeom to keep this module import-light for standalone tests).
GEOM_TYPE_SPHERE   = 2
GEOM_TYPE_CAPSULE  = 3
GEOM_TYPE_CYLINDER = 5
GEOM_TYPE_BOX      = 6


def object_props_from_model(model, body_id: int, geom_id: int) -> dict:
    """Ground-truth physical properties of a target object, read straight from the
    compiled MuJoCo model (no mj_* calls — pure array access, so this stays import-light).

    Stamped onto trial_start (see TrialRunner.start_trial) so each trial records the ACTUAL
    simulated physics of its object — not the XML intent — letting the sweep conditions be
    correlated with completion time / success / phase durations in parse_trials_tables.py.

      mass_kg   body_mass (post-compile; for a single-geom object == the geom's mass)
      mu        tangential (sliding) friction, geom_friction[:,0]
      mu_tors   torsional friction, geom_friction[:,1] (held fixed across the sweep)
      condim    contact dimensionality (6 for the pads/objects here)
      shape     geom type id (GEOM_TYPE_BOX etc. above) — feeds the table's Shape column
      size      geom_size (half-extents / radius, shape-dependent)
      izz_gcm2  principal moment about the body's local z under uniform density
                (MuJoCo computes body_inertia from mesh+mass), converted kg·m² -> g·cm².
                Matches the paper's I_zz footnote-c definition. NOTE: this is the principal
                z; for a non-z-symmetric object MuJoCo may reorder principal axes, so treat
                it as "principal moment near vertical" rather than exactly world-z.
    """
    return {
        'mass_kg':  float(model.body_mass[body_id]),
        'mu':       float(model.geom_friction[geom_id, 0]),
        'mu_tors':  float(model.geom_friction[geom_id, 1]),
        'condim':   int(model.geom_condim[geom_id]),
        'shape':    int(model.geom_type[geom_id]),
        'size':     model.geom_size[geom_id].tolist(),
        'izz_gcm2': float(model.body_inertia[body_id, 2] * 1e7),
    }


def rest_half_height(geom_type: int, geom_size: np.ndarray) -> float:
    """Object half-height at rest (z-offset from floor to body origin), shape-aware.
    Matches the convention already used in models/scene_pick_place.xml body `pos` z
    values (e.g. box pos.z == size[2], sphere pos.z == size[0])."""
    if geom_type == GEOM_TYPE_SPHERE:
        return float(geom_size[0])
    if geom_type == GEOM_TYPE_CYLINDER:
        return float(geom_size[1])
    if geom_type == GEOM_TYPE_CAPSULE:
        return float(geom_size[0] + geom_size[1])
    return float(geom_size[2])  # box (and fallback)


# ── Event logger (Stream 1: events.jsonl) ──────────────────────────────────────────────

class EventLogger:
    """Append-only JSONL event stream. One instance per run (spans many trials)."""

    def __init__(self, run_dir: Path, dashboard=None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.run_dir / 'events.jsonl', 'a', encoding='utf-8')
        # Optional live mirror: if set, every logged event is also pushed to the
        # dashboard as an 'event' message. Anything with a .push(dict) works; push()
        # is non-blocking and drops on a full queue, so it can never stall logging.
        self._dashboard = dashboard

    def log(self, trial_id: int, t: float, event: str, t_wall: float | None = None,
            **fields):
        # Drop None-valued fields so optional annotations (e.g. approach_sub, which is
        # None outside APPROACH; prev/prev_sub on the first marker) neither clutter the
        # JSONL nor render as literal "key=None" in the dashboard's inline event log.
        fields = {k: v for k, v in fields.items() if v is not None}
        # Every event carries BOTH clocks: `t` is sim-time (data.time) for joining against
        # the pose trace and for the physics-coupled state machine; `t_wall` is wall-clock
        # (time.time()), the operator's real elapsed time — the clock the experiment
        # metrics (durations) and the dashboard use, since sim runs slower than real-time.
        # t_wall defaults to now() when the caller doesn't supply one, which is correct for
        # main-thread events; solve events on background threads pass their own captured
        # wall time (and a sim-time t) explicitly.
        if t_wall is None:
            t_wall = time.time()
        row = {'trial_id': trial_id, 't': round(float(t), 6),
               't_wall': round(float(t_wall), 6), 'event': event, **fields}
        self._fh.write(json.dumps(row, default=_json_default) + '\n')
        self._fh.flush()   # trials are minutes apart; flush cost is negligible
        if self._dashboard is not None:
            self._dashboard.push({'type': 'event', **row})

    def log_solve(self, trial_id: int, component: str, ms: float,
                  t: float = 0.0, t_wall: float | None = None, **extra):
        """component: 'rrt' | 'ik_dls' | 'ik_ipopt' | 'grasp_rec'. extra: whatever
        solve-specific fields the call site already has (status, n_waypoints,
        gamma_min, ...) — passed through as-is, same pattern as the existing
        dash.push(...) calls this wraps.

        Solves run on BACKGROUND threads where data.time (sim-time) is not meaningfully
        readable, so `t` (sim-time) defaults to 0.0 and the real timestamp goes in
        t_wall (defaulting to now()). This keeps the invariant that `t` is always sim-time
        and `t_wall` is always wall-clock — previously solves put wall-clock in `t`,
        breaking joins against the sim-time trace. `ms` is a measured duration, unaffected.
        Pass an explicit sim-time `t` if the call site has a valid data.time snapshot."""
        self.log(trial_id, t, 'solve', t_wall=t_wall, component=component,
                 ms=round(float(ms), 3), **extra)

    def close(self):
        self._fh.close()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ── Per-step trace buffer (Stream 2: trial_<id>_trace.npz) ─────────────────────────────

class TraceBuffer:
    """Accumulates one row per mj_step for a single trial; written once at trial end.

    Call `sample(...)` every physics step while a trial is active. Field set is fixed
    per instance (whatever's passed on the first `sample` call) — extend by adding
    kwargs at the call site, no schema change needed here.
    """

    def __init__(self):
        self._rows: list[dict] = []

    def sample(self, **fields):
        self._rows.append(fields)

    def __len__(self):
        return len(self._rows)

    def save(self, path: Path):
        if not self._rows:
            return
        keys = self._rows[0].keys()
        arrays = {k: np.asarray([r[k] for r in self._rows]) for k in keys}
        np.savez_compressed(path, **arrays)
        self._rows.clear()


# ── Attempt trigger detectors (method-specific rising edges) ───────────────────────────

class DexPilotAttemptTrigger:
    """Rising edge: operator's real fingers pinch (min(d_s1) < PINCH_EPS_M) AND the hand
    is simultaneously in contact with the target object. Requires d_s1 from
    DexPilotRetargeter.retarget() (currently computed and discarded there — call site
    needs to thread it through; see wiring notes)."""

    def __init__(self, eps: float = PINCH_EPS_M):
        self.eps = eps
        self._was_pinched = False

    def update(self, d_s1, hand_touching_target: bool) -> bool:
        """d_s1: iterable of finger-to-thumb distances (metres), e.g. [d_if, d_mf, d_rf].
        Returns True exactly on the step the trigger fires (rising edge)."""
        is_pinched = (min(d_s1) < self.eps) and hand_touching_target
        fired = is_pinched and not self._was_pinched
        self._was_pinched = is_pinched
        return fired

    def reset(self):
        self._was_pinched = False


class ContactAwareAttemptTrigger:
    """Rising edge: squeeze_on False->True (the existing GraspController invocation at
    the operator's lock-in, contact_aware_teleop mode)."""

    def __init__(self):
        self._was_squeezing = False

    def update(self, squeeze_on: bool) -> bool:
        fired = squeeze_on and not self._was_squeezing
        self._was_squeezing = squeeze_on
        return fired

    def reset(self):
        self._was_squeezing = False


# ── Contact-episode counter (Approach subtask) ──────────────────────────────────────────

class ContactEpisodeCounter:
    """Counts contact EPISODES (not just steps-with-contact) between hand geoms and
    floor/non-target-object geoms, so a sustained scrape counts once. Target-object
    contact is exempt throughout (per settled spec: brushing the target during
    approach is not a violation).

    Debounced by a time-based cooldown (CONTACT_EPISODE_COOLDOWN_S), not by pair
    identity: a sustained rest/scrape cycles through many DIFFERENT hand-geom/other-
    geom pairs within milliseconds (e.g. the palm alone has ~10 collision sub-geoms,
    any of which may independently engage/disengage the floor each step as the contact
    manifold shifts by sub-millimeter amounts) — tracking "new episode = new exact
    pair" undercounts the debounce and lets one continuous touch explode into hundreds
    of counted episodes. A cooldown after any increment absorbs that burst regardless
    of which specific geoms are involved."""

    def __init__(self):
        self.count = 0
        self._last_episode_t: float | None = None

    def update(self, contacts, hand_gids: set[int], target_gid: int, t_now: float,
               penetration_thresh: float = CONTACT_PENETRATION_M,
               cooldown_s: float = CONTACT_EPISODE_COOLDOWN_S) -> int:
        """contacts: iterable of objects with .geom1, .geom2, .dist (MuJoCo mjContact-
        like; pass data.contact[:data.ncon]). t_now: sim-time (data.time). Returns 1 if
        a new episode was counted this call, else 0 (never more than 1 per call — the
        cooldown collapses however many violating pairs are simultaneously active into
        a single episode)."""
        violation = False
        for con in contacts:
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in hand_gids:
                other_g = g2
            elif g2 in hand_gids:
                other_g = g1
            else:
                continue
            if other_g == target_gid:
                continue   # exempt: target contact during approach is allowed
            if float(con.dist) > -penetration_thresh:
                continue   # broad-phase-only / solver-margin contact, not a real touch
            violation = True
            break
        if not violation:
            return 0
        if (self._last_episode_t is not None
                and t_now - self._last_episode_t < cooldown_s):
            return 0
        self._last_episode_t = t_now
        self.count += 1
        return 1

    def reset(self):
        self.count = 0
        self._last_episode_t = None


# ── Trial state machine (Pick + Transport) ──────────────────────────────────────────────

class ApproachSub:
    """Sub-label rendered under APPROACH (dash + logged on the phase_enter event), so a
    single phase name still conveys what the operator is doing before the grasp:
      TELEOP    — free teleop wrist-tracking, BEFORE lock-in (before L is pressed).
      PLANNING  — from lock-in (L) onward: BOTH the background RRT/IK plan AND the
                  kinematic waypoint replay toward the pregrasp pose fold into this one
                  sub-label (the operator sees the arm 'getting into position').
    """
    TELEOP   = 'teleop'
    PLANNING = 'planning'


class TrialPhase:
    """Standardized phase names — the SINGLE logged vocabulary (see from_control). The
    control loop keeps its own finer state strings ('PLAN'/'REACH'/'GRASP', plus the
    pre-lock-in teleop period); from_control() maps those onto this set so events.jsonl
    and the pose trace speak one language.

      APPROACH  — everything BEFORE the operator commits the grasp (first Enter). Carries
                  an ApproachSub sub-label: TELEOP (pre-lock-in teleop) or PLANNING
                  (RRT plan + kinematic replay, i.e. control 'PLAN'/'REACH').
      PICK      — grasp committed (control 'GRASP', entered on the first Enter): grasp
                  posture + squeeze + lift attempts all happen here.
      TRANSPORT — object successfully picked and lifted (entered at pick_confirmed).
      PLACE     — object placed; the trial finishes.
    """
    APPROACH  = 'APPROACH'
    PICK      = 'PICK'        # grasp committed (control 'GRASP'): posture + squeeze + lift
    TRANSPORT = 'TRANSPORT'   # entered at pick_confirmed
    PLACE     = 'PLACE'       # object placed; trial done

    @staticmethod
    def from_control(control_phase: str, teleop_active: bool) -> tuple[str, str | None]:
        """Map a control-loop state onto (phase, approach_sub) in the standardized
        vocabulary. approach_sub is None for any non-APPROACH phase.

        control_phase: the main loop's 'PLAN' | 'REACH' | 'GRASP' string.
        teleop_active: True while in pre-lock-in teleop (before L) — takes precedence
                       over control_phase, which sits at its 'REACH' init value then.

        TRANSPORT / PLACE are NOT produced here: they are owned by the success state
        machine (TrialRunner), which advances the trial's own `phase` past PICK on
        pick_confirmed / arrival. from_control only classifies the pre-grasp control
        states, so it never regresses a lifted object back to APPROACH/PICK."""
        if teleop_active:
            return TrialPhase.APPROACH, ApproachSub.TELEOP
        if control_phase in ('PLAN', 'REACH'):
            return TrialPhase.APPROACH, ApproachSub.PLANNING
        # 'GRASP' (first Enter committed the grasp) and any fallthrough → PICK
        return TrialPhase.PICK, None


class TrialOutcome:
    SUCCESS   = 'success'
    TIMEOUT   = 'timeout'
    ABANDONED = 'abandoned'   # operator switched target / went home mid-trial
    RUNNING   = None


@dataclass
class TrialState:
    """One trial's mutable state. Constructed fresh by TrialRunner.start_trial()."""
    trial_id: int
    method: str            # 'dexpilot' | 'contact_aware_teleop'
    object_name: str
    t_start: float                     # sim-time (data.time) at trial start
    t_start_wall: float = 0.0          # wall-clock (time.time()) at trial start — the
                                        # clock the reported completion time is measured in
                                        # (sim runs slower than real-time; the operator's
                                        # real elapsed time is what the experiment reports)
    phase: str = TrialPhase.APPROACH   # trials start at lock-in (control 'PLAN'), which
                                        # is APPROACH/planning; advances to PICK when the
                                        # operator commits the grasp (control 'GRASP')
    attempt_id: int = 0
    attempt_active: bool = False   # trigger condition currently engaged (pinch/squeeze)
    dwell_t0: float | None = None  # sim-time the current continuous lift began
    pick_confirmed: bool = False
    pick_logged: bool = False   # forward phase_enter PICK emitted for the current PICK entry
                                 # (reset on a drop so a re-pick logs PICK again)
    n_inadvertent_contacts: int = 0
    n_drops: int = 0
    max_slip_mm: float = 0.0       # diagnostic only, does not gate drop counting
    outcome: str | None = TrialOutcome.RUNNING
    t_end: float | None = None


class TrialRunner:
    """Drives one trial's phase/attempt/drop/arrival logic per physics step. Owns no
    MuJoCo objects directly — the caller passes in whatever's needed each call, keeping
    this testable without a live model/data pair.

    Usage (per trial):
        runner = TrialRunner(event_logger, trace_dir)
        state  = runner.start_trial(trial_id, method, object_name, t_now)
        ...
        for each mj_step:
            runner.step_approach(state, t_now, contacts, hand_gids, target_gid)   # PICK: pre-attempt
            ...
            runner.step_pick_or_transport(state, t_now, height_above_rest,
                                           trigger_fired, trigger_active, ...)
            runner.trace.sample(t=t_now, ...)                                    # every step
        if arrived_or_timed_out:   # arrival return value, or check_timeout(state, t_now)
            runner.end_trial(state, t_now)   # trace filename auto-built from state
    """

    def __init__(self, event_logger: EventLogger, trace_dir: Path):
        self.events = event_logger
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace = TraceBuffer()
        self.contact_counter = ContactEpisodeCounter()

    # -- lifecycle -----------------------------------------------------------------

    def start_trial(self, trial_id: int, method: str, object_name: str,
                     t_now: float, props: dict | None = None) -> TrialState:
        """props: optional physical-property snapshot of the target object (mass, mu,
        I_zz, ...), stamped onto the trial_start event so the object's ground-truth
        physics can be correlated with completion time / success / phase offline. Built
        with object_props_from_model() at the call site; None keeps the old schema."""
        self.contact_counter.reset()
        self.trace = TraceBuffer()
        t_start_wall = time.time()   # experiment clock anchor (see TrialState.t_start_wall)
        state = TrialState(trial_id=trial_id, method=method, object_name=object_name,
                            t_start=t_now, t_start_wall=t_start_wall)
        self.events.log(trial_id, t_now, 'trial_start', t_wall=t_start_wall,
                         method=method, object=object_name, **(props or {}))
        # No phase_enter here: a trial starts at lock-in (control 'PLAN') → APPROACH,
        # and the caller's marker site already emits that transition from control_phase
        # (teleop→planning→PICK). TrialState.phase defaults to APPROACH to match, so the
        # trial machine's own phase never regresses. Emitting here too would duplicate
        # the marker site's APPROACH row on the lock-in step.
        return state

    def check_timeout(self, state: TrialState, t_now: float) -> bool:
        """Call once per step; returns True (and marks outcome) if the trial has timed
        out. Caller is responsible for calling end_trial() immediately after."""
        if t_now - state.t_start >= TRIAL_TIMEOUT_S and state.outcome is None:
            state.outcome = TrialOutcome.TIMEOUT
            return True
        return False

    def end_trial(self, state: TrialState, t_now: float):
        # Guard against a caller passing a post-reset sim-time (data.time snapped back
        # to ~0 by an out-of-band mj_resetData): that would log a negative duration and
        # a t < t_start. Clamp t_now up to t_start so duration is never negative; the
        # normal end paths (arrival / timeout) already pass a valid monotonic time.
        if t_now < state.t_start:
            t_now = state.t_start
        state.t_end = t_now
        if state.outcome is None:
            state.outcome = TrialOutcome.TIMEOUT
        # Completion time is reported in WALL-CLOCK — the operator's real elapsed time,
        # the experiment's metric of interest. Sim runs slower than real-time, so the
        # sim-time span (duration_sim_s, kept for reference / trace joins) understates it.
        # Wall clock is monotonic across sim resets, so no clamp is needed here.
        t_end_wall = time.time()
        duration_wall_s = max(0.0, t_end_wall - state.t_start_wall)
        self.events.log(state.trial_id, t_now, 'trial_end', t_wall=t_end_wall,
                         outcome=state.outcome,
                         n_inadvertent_contacts=state.n_inadvertent_contacts,
                         n_drops=state.n_drops, n_attempts=state.attempt_id,
                         max_slip_mm=state.max_slip_mm,
                         duration_s=round(duration_wall_s, 3),
                         duration_sim_s=round(t_now - state.t_start, 3))
        trace_path = self.trace_dir / f'{self._trace_filename(state)}.npz'
        self.trace.save(trace_path)

    @staticmethod
    def _trace_filename(state: TrialState) -> str:
        """trial_<id>_<method>_<object>_<outcome>, auto-named from the trial's own
        state — self-describing without opening events.jsonl. Non-alnum characters in
        method/object_name are collapsed to '_' (both are internal identifiers already
        close to filesystem-safe, e.g. 'obj_red_box', 'contact_aware_teleop', but this
        guards against anything unexpected making it into either field)."""
        def _safe(s):
            return ''.join(c if c.isalnum() or c == '_' else '_' for c in s)
        return (f'trial_{state.trial_id:04d}_{_safe(state.method)}_'
                f'{_safe(state.object_name)}_{state.outcome}')

    def abandon_trial(self, state: TrialState, t_now: float):
        """Force-end a still-RUNNING trial because the operator switched target or went
        home before it reached arrival/timeout — e.g. Ctrl+0 or Ctrl+<other digit>
        mid-trial. Distinct outcome from timeout/success so abandoned trials can be
        excluded or inspected separately in analysis. No-op if already ended."""
        if state.outcome is not None:
            return
        state.outcome = TrialOutcome.ABANDONED
        self.end_trial(state, t_now)

    # -- PICK phase, pre-attempt (RRT plan + waypoint replay) ------------------------

    def set_phase(self, state: TrialState, t_now: float, phase: str):
        if state.phase != phase:
            state.phase = phase
            self.events.log(state.trial_id, t_now, 'phase_enter', phase=phase)

    def step_approach(self, state: TrialState, t_now: float, contacts,
                       hand_gids: set[int], target_gid: int):
        """Call every step during the PICK pre-attempt part (RRT plan + replay). Counts
        inadvertent-contact episodes (time-debounced — see ContactEpisodeCounter)."""
        new_episodes = self.contact_counter.update(contacts, hand_gids, target_gid,
                                                    t_now)
        if new_episodes:
            state.n_inadvertent_contacts += new_episodes
            self.events.log(state.trial_id, t_now, 'contact_violation',
                             count=state.n_inadvertent_contacts)

    # -- PICK phase (grasp attempts) / TRANSPORT -------------------------------------

    def step_pick_or_transport(self, state: TrialState, t_now: float,
                                trigger_fired: bool, trigger_active: bool,
                                height_above_rest: float,
                                place_xy_offset: float | None = None,
                                object_speed: float | None = None,
                                place_marker_half_extent: float = 0.15):
        """Call every step once REACH has ended (GRASP begins). Drives the attempt /
        dwell / confirm / transport / drop / arrival state machine.

        trigger_fired:  True on the step a method-specific attempt trigger rising edge
                         fires (DexPilotAttemptTrigger / ContactAwareAttemptTrigger).
        trigger_active: True for every step the grasp condition (pinch+contact, or
                         squeeze_on) remains engaged, independent of the edge.
        height_above_rest: object z - rest_half_height (metres); compare against
                         LIFT_HEIGHT_M for both the pick-dwell check and the
                         transport/drop check (same threshold, per settled spec).
        place_xy_offset: |xy_object - xy_place_site| (metres), or None if not yet
                         relevant (e.g. still in GRASP, not TRANSPORT).
        object_speed:    object linear speed (m/s), for the arrival settle check.

        Arrival is a SET-IN-PLACE check: success once the object is inside the place
        footprint and settled (speed < ARRIVAL_SPEED_M_S), whether it is held clear over
        the target OR set down into it. A descent below LIFT_HEIGHT_M inside the footprint
        is treated as a placement-in-progress (stay in TRANSPORT until it settles), NOT a
        drop; only a descent outside the footprint counts a drop and reverts to PICK.

        Returns True if the trial reached arrival (success) this step.
        """
        if state.phase not in (TrialPhase.PICK, TrialPhase.TRANSPORT):
            # Advance internal state to PICK SILENTLY (needed for the attempt logic below),
            # but do NOT log phase_enter here: this method is called every step from t=0 in
            # dexpilot, so logging on first-call would stamp PICK at APPROACH before any
            # grasp. The forward PICK is logged instead on the first attempt_start (below) —
            # the meaningful "grasp attempt begins" moment, correct for BOTH modes.
            state.phase = TrialPhase.PICK

        if state.phase == TrialPhase.PICK:
            if trigger_fired and not state.attempt_active:
                # Log the forward APPROACH→PICK entry on the FIRST attempt (idempotent via
                # set_phase; state.phase is already PICK so we force the log with a small
                # helper). Single owner of PICK logging for all modes — the control-loop
                # marker no longer logs PICK (it never could for dexpilot: no 'GRASP').
                if not state.pick_logged:
                    state.pick_logged = True
                    self.events.log(state.trial_id, t_now, 'phase_enter',
                                     phase=TrialPhase.PICK)
                state.attempt_id += 1
                state.attempt_active = True
                state.dwell_t0 = None
                self.events.log(state.trial_id, t_now, 'attempt_start',
                                 attempt=state.attempt_id)

            if not trigger_active and state.attempt_active:
                # grasp released before confirmation -> attempt ends, unconfirmed
                state.attempt_active = False
                state.dwell_t0 = None
                self.events.log(state.trial_id, t_now, 'attempt_result',
                                 attempt=state.attempt_id, outcome='released')

            if state.attempt_active:
                lifted = height_above_rest > LIFT_HEIGHT_M
                if lifted:
                    if state.dwell_t0 is None:
                        state.dwell_t0 = t_now
                    elif t_now - state.dwell_t0 >= DWELL_S:
                        state.pick_confirmed = True
                        self.events.log(state.trial_id, t_now, 'pick_confirmed',
                                         attempt=state.attempt_id,
                                         dwell_s=round(t_now - state.dwell_t0, 3))
                        self.set_phase(state, t_now, TrialPhase.TRANSPORT)
                else:
                    state.dwell_t0 = None   # reset dwell, SAME attempt continues

        elif state.phase == TrialPhase.TRANSPORT:
            # SET-IN-PLACE arrival. Two success paths, both requiring the object be inside
            # the place footprint and settled (speed < ARRIVAL_SPEED_M_S):
            #   (a) held-clear hover: settled while still lifted (height > LIFT_HEIGHT_M) —
            #       the operator hovers the object over the target and holds it still.
            #   (b) set down in target: the object is lowered/released INTO the target and
            #       comes to rest there (height <= LIFT_HEIGHT_M). This is how a human
            #       actually places — carry over, set down, let go — and is why teleop runs
            #       that clearly placed the box never scored under the old held-clear-only
            #       rule (the object was still moving whenever it was above the threshold,
            #       and the moment it descended it was counted as a drop instead).
            # A descent to/below the threshold OUTSIDE the footprint is still a drop.
            in_place = (place_xy_offset is not None
                        and place_xy_offset <= place_marker_half_extent)
            settled = (object_speed is not None and object_speed < ARRIVAL_SPEED_M_S)
            if in_place and settled:
                self.set_phase(state, t_now, TrialPhase.PLACE)   # object placed → finish
                self.events.log(state.trial_id, t_now, 'arrival',
                                 attempt=state.attempt_id,
                                 xy_offset_m=round(place_xy_offset, 4),
                                 set_down=bool(height_above_rest <= LIFT_HEIGHT_M))
                state.outcome = TrialOutcome.SUCCESS
                return True

            if height_above_rest <= LIFT_HEIGHT_M:
                # Descended out of the carry. If it came down INSIDE the footprint we do
                # NOT count a drop or revert to PICK — it is a placement in progress that
                # just hasn't settled yet (still moving as it's released); stay in TRANSPORT
                # so the settled-in-place check above fires once it comes to rest. Only a
                # descent OUTSIDE the footprint is a genuine drop.
                if in_place:
                    return False
                state.n_drops += 1
                self.events.log(state.trial_id, t_now, 'drop', count=state.n_drops)
                state.attempt_active = False
                state.dwell_t0 = None
                state.pick_confirmed = False
                # set_phase logs the TRANSPORT→PICK phase_enter itself; mark pick_logged so
                # the next attempt_start doesn't emit a second PICK for this same entry.
                self.set_phase(state, t_now, TrialPhase.PICK)
                state.pick_logged = True
                return False

        return False

    def note_slip(self, state: TrialState, slip_mm: float):
        """Diagnostic only (does not gate drop counting) — call during TRANSPORT with
        translational deviation of object-in-palm-frame vs. its pose at pick_confirmed."""
        if slip_mm > state.max_slip_mm:
            state.max_slip_mm = slip_mm

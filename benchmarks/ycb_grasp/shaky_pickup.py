"""FRoGGeR's "shaky pickup" execution test (their Sec. IV).

Their protocol, verbatim from the paper:

    "we generate a pick trajectory where the end-effector is lifted 10cm in 1s and
     then held for 1.5s. We add sinusoidal perturbations to this trajectory with
     amplitude 3mm and varying frequency in all spatial axes 0.25s after the pick
     begins until the end of the simulation. A pick fails if either (1) the object
     rotates by more than 30 degrees or if the object deviates from the pick
     trajectory by more than 7.5cm at any point; or (2) the total grasp synthesis
     time exceeds 1 minute."

Why this matters for the benchmark: `l_bar*` is a PROXY for pick success, and the
paper's own data shows it is a noisy one -- median 0.61 for successes against 0.47
for failures. Their headline claim is 78.8% pick success, not a metric value, so
nothing that stops at the planner can speak to it.

WHAT THIS SHARES WITH THE EXISTING BENCHMARK. The approach, settle, gap gate and
squeeze ramp are `pick_and_place`'s, unchanged -- that sequence is measured and
tuned (SOLVER_STATE sec 6), and re-deriving it would introduce differences that
have nothing to do with the grasp being tested. Only the LIFT is replaced, by the
paper's timed profile with sinusoidal perturbation.

WHAT DIFFERS FROM THEIR SETUP, recorded rather than glossed:
  * They perturb the end-effector TRAJECTORY in Drake; this perturbs the palm
    velocity command through the same resolved-rate DLS jog the rest of the
    benchmark uses, so the perturbation reaches the object through the arm's real
    dynamics rather than being imposed on it.
  * "varying frequency" is unspecified in the paper. Three incommensurate
    frequencies are used per axis (see SHAKE_FREQS_HZ) so the perturbation does not
    repeat within the hold.
  * Their failure test is against the commanded pick trajectory; ours is against
    the palm's own achieved motion, which is the same quantity when tracking is
    good and a fairer one when it is not.
"""
from __future__ import annotations

import numpy as np

# Their numbers.
LIFT_HEIGHT_M = 0.10          # "lifted 10cm"
LIFT_TIME_S = 1.0             # "in 1s"
HOLD_TIME_S = 1.5             # "then held for 1.5s"
SHAKE_AMPLITUDE_M = 0.003     # "amplitude 3mm"
SHAKE_START_S = 0.25          # "0.25s after the pick begins"
FAIL_ROT_DEG = 30.0           # "rotates by more than 30 degrees"
FAIL_DEV_M = 0.075            # "deviates ... by more than 7.5cm"
SYNTH_TIMEOUT_S = 60.0        # "synthesis time exceeds 1 minute"

# "varying frequency in all spatial axes" -- unspecified. Incommensurate so the
# three axes do not re-phase within the 2.5 s run.
SHAKE_FREQS_HZ = (7.0, 11.0, 13.0)


def _quat_angle_deg(qa, qb):
    """Absolute rotation angle between two unit quaternions, in degrees."""
    qa = np.asarray(qa, float)
    qb = np.asarray(qb, float)
    d = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def shake_velocity(t, lift_speed):
    """Commanded palm linear velocity at time t into the pick.

    Base profile is a constant-speed lift for LIFT_TIME_S then hold. The sinusoidal
    term is added as a VELOCITY, with amplitude scaled so the resulting position
    excursion is SHAKE_AMPLITUDE_M: for a*sin(2*pi*f*t) in position, the velocity
    amplitude is a*2*pi*f.
    """
    v = np.zeros(3)
    if t < LIFT_TIME_S:
        v[2] = lift_speed
    if t >= SHAKE_START_S:
        for ax, f in enumerate(SHAKE_FREQS_HZ):
            v[ax] += SHAKE_AMPLITUDE_M * 2.0 * np.pi * f * np.cos(
                2.0 * np.pi * f * (t - SHAKE_START_S))
    return v


def run_shaky_pickup(model, data, ctrl, q_cmd, jog_to, sync, palm_bid, obj_bid,
                     tip_geom_ids=None, obj_gid=None, finger_set=None,
                     synth_time_s=0.0):
    """Execute the lift-and-shake and score it by their failure criteria.

    Expects the caller to have already approached, settled, passed the gap gate and
    ramped the squeeze -- i.e. the hand is holding the object at the planned grasp.
    `jog_to` is `pick_and_place._jog_to`, passed in rather than imported so this
    module does not depend on that script's import-time globals.

    Returns a dict:
        success            bool, their criteria
        fail_reason        None | 'rotation' | 'deviation' | 'synthesis_timeout'
        max_rot_deg        peak object rotation from its pre-lift orientation
        max_dev_m          peak object deviation from the palm's own motion
        lift_achieved_m    how far the object actually rose
    """
    dt = float(model.opt.timestep)
    n_lift = int(round(LIFT_TIME_S / dt))
    n_hold = int(round(HOLD_TIME_S / dt))
    lift_speed = LIFT_HEIGHT_M / LIFT_TIME_S

    if synth_time_s > SYNTH_TIMEOUT_S:
        return dict(success=False, fail_reason='synthesis_timeout',
                    max_rot_deg=0.0, max_dev_m=0.0, lift_achieved_m=0.0)

    obj_p0 = data.xpos[obj_bid].copy()
    obj_q0 = data.xquat[obj_bid].copy()
    palm_p0 = data.xpos[palm_bid].copy()

    max_rot, max_dev = 0.0, 0.0
    fail = [None]

    # `_jog_to` drives the whole profile in one call: its v6_fn takes the STEP
    # INDEX, so the timed lift-then-hold and the sinusoid are both expressed there.
    # Running it step-by-step instead would restart its velocity ramp every step.
    def _v6(i):
        return np.concatenate([shake_velocity(i * dt, lift_speed), np.zeros(3)])

    # Their criteria are checked DURING the motion ("at any point"), so the
    # per-step monitoring rides on the sync callback the jog already calls.
    state = dict(max_rot=0.0, max_dev=0.0)

    def _sync_and_check():
        rot = _quat_angle_deg(obj_q0, data.xquat[obj_bid])
        palm_moved = data.xpos[palm_bid] - palm_p0
        obj_moved = data.xpos[obj_bid] - obj_p0
        dev = float(np.linalg.norm(obj_moved - palm_moved))
        state['max_rot'] = max(state['max_rot'], rot)
        state['max_dev'] = max(state['max_dev'], dev)
        if fail[0] is None:
            if rot > FAIL_ROT_DEG:
                fail[0] = 'rotation'
            elif dev > FAIL_DEV_M:
                fail[0] = 'deviation'
        if sync is not None:
            sync()

    q_cmd, contact_lost = jog_to(
        model, data, ctrl, q_cmd, _v6, n_lift + n_hold, _sync_and_check, palm_bid,
        obj_bid, tip_geom_ids, obj_gid, label='shake', finger_set=finger_set)
    max_rot, max_dev = state['max_rot'], state['max_dev']
    fail = fail[0]

    return dict(success=(fail is None), fail_reason=fail,
                max_rot_deg=max_rot, max_dev_m=max_dev,
                lift_achieved_m=float(data.xpos[obj_bid][2] - obj_p0[2]),
                contact_lost=contact_lost, q_cmd=q_cmd)

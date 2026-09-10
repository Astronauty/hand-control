"""Measured contact / solver / controller settings for the tabletop pick-and-place.

Every number here came from a sweep scored on BOTH criteria -- the grasp must
bear real force and lift the object, AND the release must not fling it. Scoring
either alone picks a setting that fails the other, because the two failure modes
trade off along the same axis.

--------------------------------------------------------------------------
WHY THE FINGERTIP, NOT THE OBJECT
--------------------------------------------------------------------------
MuJoCo combines an unpaired contact's solref by taking the MIN of the two geoms
(and friction by the MAX). The LEAP fingertips ship as solref=[0.004, 1.0] and
the YCB objects are [0.02, 1.0], so the ACTUAL contact ran at tau=0.004 -- two
timesteps at dt=0.002. That is stiff enough that the discrete integrator
re-energizes the contact each step, and it is the source of the release "fling":
the index contact was measured making and breaking on ALTERNATING 2 ms steps
(f = 0.85, 0.0, 0.99, 0.0, ...) while an object that entered the release at
0.03 m/s was pumped to 2.07 m/s and 26 rad/s over ~56 ms. Energy with no
physical source.

Because of the min() rule, softening the OBJECT does nothing while the fingertip
stays at 0.004 -- and softening every geom (an earlier mistake here) silently
softens the fingertip 5x too, which destroys the grasp and makes it look as
though the damping ratio was to blame. It was not: with fingertip-scoped edits,
grasp quality was 3/3 across every tau/zeta cell tested.

--------------------------------------------------------------------------
WHY noslip_iterations, AND WHY IT NEEDS A HIGHER GAMMA
--------------------------------------------------------------------------
noslip_iterations was 0 (disabled). It is MuJoCo's dedicated post-pass that
removes tangential constraint drift, and it is by far the strongest lever on the
release: 065-e_cups, the worst offender at 2.02 m/s / 55 rad/s, drops to 0.00.

It is not free. Removing tangential compliance also removes what the heavy
036_wood_block (7.15 N, ~15x the orange) was leaning on to carry its weight, and
the lift collapsed 96 mm -> -6 mm at the previously-tuned squeeze. Restoring the
NORMAL force compensates exactly: gamma 10.0 with squeeze_pd_scale 1.0 brings the
block back to 96 mm while keeping the clean 0.04 m/s release. Neither change
works without the other -- do not adopt one of them alone.

--------------------------------------------------------------------------
MEASURED RESULTS (tau=0.02, zeta=2.0, noslip=5, impratio=100, gamma=10,
                  squeeze_pd_scale=1.0)
--------------------------------------------------------------------------
    object            force     lift     peak |v| at release
    036_wood_block     5.7 N     96 mm      0.04 m/s   (was 1.39)
    065-e_cups         2.0 N    115 mm      0.07 m/s   (was 2.02)
    017_orange         6.2 N    115 mm      2.92 m/s   NOT FIXED

--------------------------------------------------------------------------
THE ORANGE IS A GENUINE EXCEPTION -- DO NOT TREAT IT AS TUNED
--------------------------------------------------------------------------
A smooth sphere in a two-finger pinch rolls out from between the pads as they
open, and no damping/noslip setting tested prevents rolling. condim=4 on the
object (dropping ROLLING friction while keeping torsional) is the only lever
that moved it -- spin 95.6 -> 30.8 rad/s, speed 2.92 -> 1.68 -- which is a large
improvement but still an order of magnitude above the 0.07 the other two reach.
condim=3 produces no contact at all. Recorded as OPEN below rather than papered
over with a setting that does not work.

Crucially, the orange's fling CONVERGES under timestep refinement:

    dt = 0.002   1.68 m/s        (with condim=4)
    dt = 0.001   1.60 m/s
    dt = 0.0005  1.58 m/s

A discretization artifact shrinks as dt -> 0; this does not. So unlike the
block's and cup's chatter, the orange's roll-off is what the MODEL actually
says happens -- a rigid sphere pinched between two pads, released, rolls. No
solver or contact-parameter setting will remove it, and further dt/solref/noslip
sweeps on this object are wasted effort. The fix has to change the GRASP or the
RELEASE (a third finger, asymmetric opening, or lowering the object to the bin
floor before opening), not the contact model.

--------------------------------------------------------------------------
A MEASUREMENT CAVEAT WORTH KNOWING
--------------------------------------------------------------------------
Re-running the SAME config is deterministic to the last digit (verified 3x).
But changing impratio perturbs how the object settles, which changes the plan,
which occasionally lands a fingertip gap just over the 8 mm CONTACT_GAP_TOL_M
gate -- producing a 0.0 N "failure" that is a PLANNER outcome, not a contact
one. When a cell reports zero force, check phase_log for
'squeeze_aborted_no_contact' before attributing it to the contact settings.
"""

# Applied to the LEAP fingertip geoms only (names ending in "_tip").
FINGERTIP_SOLREF = (0.02, 2.0)          # stock is (0.004, 1.0)

# mjOption.
NOSLIP_ITERATIONS = 5
IMPRATIO = 100.0
TIMESTEP = 0.002

# GraspController.
GAMMA = 10.0
SQUEEZE_PD_SCALE = 1.0

# Per-object deltas from the defaults above. Only what MEASUREMENT justified.
PER_OBJECT = {
    "017_orange": dict(
        obj_condim=4,
        note="OPEN: still flings at 1.68 m/s / 30.8 rad/s even with condim=4 "
             "(best found; stock condim=6 gives 2.92 m/s / 95.6 rad/s). A smooth "
             "sphere rolls out of a 2-finger pinch on release. Needs a geometric "
             "or release-strategy fix, not a contact-parameter one.",
    ),
}

OPEN_ISSUES = (
    "017_orange release fling unresolved (see PER_OBJECT).",
    "009_gelatin_box cannot be planned lying flat: fingertip r=19.4mm forces "
    "contacts >=21.4mm above the table and the box is 28mm tall, leaving a 6.6mm "
    "band; 120/120 seeds rejected. Plans fine standing upright. Geometric, not tunable.",
    "table_scene.in_bin tests the object body ORIGIN against z>=0.635 (the bin "
    "floor), so an object resting in the bin only passes if it is tall enough to "
    "lift its origin clear. Under-reports successes.",
)


def apply_to_model(model, object_id=None):
    """Apply the measured settings to a compiled model in place.

    Fingertip-scoped for solref (see the module docstring on the min() rule).
    Returns the dict of what was applied, for logging.
    """
    import mujoco as mj

    applied = {}
    model.opt.noslip_iterations = NOSLIP_ITERATIONS
    model.opt.impratio = IMPRATIO
    applied["noslip_iterations"] = NOSLIP_ITERATIONS
    applied["impratio"] = IMPRATIO

    tips = [g for g in range(model.ngeom)
            if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or "").endswith("_tip")]
    for g in tips:
        model.geom_solref[g][0] = FINGERTIP_SOLREF[0]
        model.geom_solref[g][1] = FINGERTIP_SOLREF[1]
    applied["fingertip_solref"] = FINGERTIP_SOLREF
    applied["n_fingertip_geoms"] = len(tips)

    per = PER_OBJECT.get(object_id or "", {})
    if "obj_condim" in per:
        applied["obj_condim"] = per["obj_condim"]
    return applied

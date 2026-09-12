"""Shared Kinova Gen3 + LEAP hand constants: fingertip contact sites, finger naming, and
the arm's home-pose XML. Extracted from kinova_leap_pick_place.py so benchmarks/other
callers don't need to import the pick-and-place entry-point script for these.
"""
import numpy as np

# Fingertip contact sites added in models/build_kinova_leap.py (_add_fingertip_sites),
# named "<finger-body>_tip" with the "leap_" attach() prefix.
FINGER_TIP_SITES = {
    "index":  "leap_if_ds_tip",
    "middle": "leap_mf_ds_tip",
    "ring":   "leap_rf_ds_tip",
    "thumb":  "leap_th_ds_tip",
}
# Collision-geom name prefix for each finger's link chain (bs/px/md/ds + tip), used to
# build the RRT's finger_geom_names list.
FINGER_CODE = {"index": "if", "middle": "mf", "ring": "rf", "thumb": "th"}

# Which fingers the grasp uses, derived from models/grasp_finger_config.json so the
# EXECUTOR and the PLANNER cannot disagree about it. Previously this was a literal
# ["index", "thumb"] duplicated in five files while the planner learned the finger
# identities from GraspConfig3D's site fields -- so re-pointing the planner at the
# middle finger left the controller still squeezing with the index.
#
# ORDER IS A SEPARATE CONVENTION FROM SLOT ORDER, and the difference is load-bearing.
# The planner's contact SLOTS are thumb-first: slot 1 = p1 = thumb, slot 2 = p2 = the
# opposing finger. FINGER_SET is the REVERSE, opposing-finger-first, because id_C is
# built in FINGER_SET order (kinova_leap_pick_place.py:624) and then zipped against
# per-finger forces (:1209), contact sites (:1645) and IK targets (:1825) -- all of
# which document the pairing as "[index, thumb]: index<-p2, thumb<-p1" (:1643).
# Reversing the slot list here is what preserves that mapping; deriving it in slot
# order would silently swap every one of those zips.
#
# _finger_set_from_config() falls back to the historical literal if the file is
# missing or unreadable, so a fresh checkout behaves exactly as before.
def _finger_set_from_config():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "models" / "grasp_finger_config.json"
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return ["index", "thumb"]
    pairings = {k: v for k, v in (raw.get("pairings") or {}).items()
                if not k.startswith("_")}
    roles = pairings.get(raw.get("default"))
    if not roles:
        return ["index", "thumb"]
    # slot order (thumb-first) -> FINGER_SET order (opposing-finger-first)
    return list(reversed(roles))


def _slot_roles_from_config(pairing=None):
    """The pairing's roles in PLANNER SLOT order (slot 1 first = p1 = thumb).

    FINGER_SET is this list reversed; see the ordering note above. Exposed so the
    controller can map an NLP slot to the finger that serves it WITHOUT keying on
    the role name -- the two dicts at kinova_leap_pick_place.py:1644/:1822 used to
    hardcode {'thumb': p1, 'index': p2}, which KeyErrors the moment a pairing does
    not contain an 'index'."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "models" / "grasp_finger_config.json"
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return ["thumb", "index"]
    pairings = {k: v for k, v in (raw.get("pairings") or {}).items()
                if not k.startswith("_")}
    roles = pairings.get(pairing or raw.get("default"))
    return list(roles) if roles else ["thumb", "index"]


SLOT_ROLES = _slot_roles_from_config()
FINGER_SET = _finger_set_from_config()

# Gen3 arm "home" pose — a natural elbow-bent reach-forward configuration. Read at
# runtime from gen3.xml's "home" keyframe (see GEN3_XML / HOME_ARM in main) rather than
# hardcoded, so it tracks the source model. Used as the IK null-space bias for the 7 arm
# joints: the null-space pull toward this pose produces a forward/lateral approach to
# tabletop objects (confirmed visually) without needing explicit orientation constraints
# on the fingertips — the orientation approach (IKSolver's (local_axis, world_target)
# tuple) was implemented and validated but caused joint-limit clipping instability on
# this 23-DOF redundant chain when combined with position constraints, preventing
# convergence.
GEN3_XML = 'mujoco_menagerie/kinova_gen3/gen3.xml'

# FINGERTIP_POINTING_AXIS is kept for reference; used by _approach_orientation below.
# Not currently used in the main IK loop (position-only + HOME_ARM bias suffices), but
# available if a caller wants to add per-site orientation control in future.
FINGERTIP_POINTING_AXIS = np.array([0.0, -1.0, 0.0])

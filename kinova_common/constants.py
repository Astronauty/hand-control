"""Shared Kinova Gen3 + LEAP hand constants: fingertip contact sites, finger naming, and
the arm's home-pose XML. Extracted from kinova_leap_pick_place.py so benchmarks/other
callers don't need to import the pick-and-place entry-point script for these.
"""
from pathlib import Path

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
_FINGER_CONFIG_PATH = (Path(__file__).resolve().parent.parent
                       / "models" / "grasp_finger_config.json")
_FINGER_FALLBACK = ["thumb", "index"]     # slot order; the dataclass default


def _slot_roles_from_config(obj_id=None, path=None):
    """The grasp's roles in PLANNER SLOT ORDER (slot 1 first = p1 = thumb).

    Reads the SAME schema `grasp_config_builder.load_finger_config` reads: a flat
    top-level `fingers` list, with `per_object[<ycb id>]` overriding it. A missing or
    unreadable file falls back to thumb+index, so a fresh checkout behaves as before.

    THIS USED TO READ A SCHEMA THE FILE NO LONGER HAS. It looked up
    `pairings[default]`, and `grasp_finger_config.json` carries neither key -- so
    every call silently hit the `["index","thumb"]` fallback. That went unnoticed
    because the fallback happens to equal the file's current default, but it meant a
    `per_object` entry (e.g. a tripod for one object) was read by the PLANNER, via
    load_finger_config, and ignored by the EXECUTOR, via this function -- the exact
    planner/executor disagreement the module comment below says this exists to prevent.

    obj_id : YCB id to honour a `per_object` override for. None = the file default.
    """
    import json
    p = Path(path) if path is not None else _FINGER_CONFIG_PATH
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return list(_FINGER_FALLBACK)
    roles = (raw.get("per_object") or {}).get(obj_id) if obj_id else None
    if not roles:
        roles = raw.get("fingers")
    roles = [r for r in (roles or []) if not str(r).startswith("_")]
    return list(roles) if roles else list(_FINGER_FALLBACK)


def _finger_set_from_config(obj_id=None, path=None):
    """SLOT_ROLES reversed: opposing-finger-first. See the ordering note above."""
    return list(reversed(_slot_roles_from_config(obj_id=obj_id, path=path)))


def resolve_fingers(fingers=None, obj_id=None):
    """(SLOT_ROLES, FINGER_SET) for one grasp, resolved AT CALL TIME.

    The module-level SLOT_ROLES/FINGER_SET below are import-time constants, which is
    why `--pairing`/`--fingers` historically steered the PLANNER only: the executor
    had already bound the default at import. A caller that knows the grasp it is
    about to execute should call this instead and thread the result through, so a
    non-default finger list reaches the controller too.

    fingers : explicit ordered role list (or 'a,b' string) in SLOT order; wins over
              the file. None = the file's per-object entry, else its default.
    """
    if fingers is not None:
        if isinstance(fingers, str):
            fingers = [t.strip() for t in fingers.split(",") if t.strip()]
        slots = [str(f) for f in fingers]
    else:
        slots = _slot_roles_from_config(obj_id=obj_id)
    return slots, list(reversed(slots))


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


def finger_joint_slices(model, fingers=None):
    """(start, stop) qpos slices for the GRASPING fingers' joints.

    GraspController gates two things on these slices: effective_gains() (the
    CLOSING-vs-HOLDING finger-gain switch) and slip_correction_torques(). Its
    default was hardcoded ((7, 11), (19, 23)) -- LEAP index and thumb -- which
    silently EXCLUDES any other finger: at --fingers thumb,index,middle the
    middle finger's joints (11..14) fell outside both slices, so it kept full
    stiff gains while index and thumb were softened to close, and it never
    participated in the squeeze/transport gain switch at all.

    Derived from the model's own joint names (leap_<code>_*), so it tracks the
    model rather than restating its layout.
    """
    import mujoco as mj
    fingers = list(fingers) if fingers else list(FINGER_SET)
    out = []
    for f in fingers:
        code = FINGER_CODE.get(f)
        if code is None:
            continue
        adrs = [model.jnt_qposadr[j] for j in range(model.njnt)
                if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or '')
                .startswith(f'leap_{code}_')]
        if adrs:
            out.append((int(min(adrs)), int(max(adrs)) + 1))
    return tuple(sorted(out))

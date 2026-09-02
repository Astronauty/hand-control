"""LEAP hand fingertip approach-orientation and collision-geom lookup helpers.
Extracted verbatim from kinova_leap_pick_place.py.
"""
import mujoco as mj

from kinova_common.constants import FINGER_CODE, FINGERTIP_POINTING_AXIS


def _approach_orientation(normal):
    """Return an IKSolver (local_axis, world_target) orientation spec that points a
    LEAP fingertip's tip axis inward along -normal. Not used in the current main()
    control loop (position-only IK + HOME_ARM bias gives sufficient approach geometry),
    but available as a building block for future tighter orientation control."""
    return (FINGERTIP_POINTING_AXIS, -normal)


def _finger_collision_geoms(model, finger):
    """All collision/tip geom names belonging to one LEAP finger's link chain."""
    code = FINGER_CODE[finger]
    prefix = f"leap_{code}_"
    names = []
    for i in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        if name and name.startswith(prefix) and ("collision" in name or name.endswith("_tip")):
            names.append(name)
    return names

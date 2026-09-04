"""Object-property randomization: size/color/rest-pose for MuJoCo primitive geoms
(sphere/box/cylinder/capsule), extracted and generalized from
kinova_leap_pick_place.py's _randomize_objects.

randomize_mesh_body (YCB mesh bodies: mass/friction, no live geom_size to scale) is
future Phase-4 work — see the refactor plan's open question on mesh size
randomization (MuJoCo can't live-scale mesh vertices without recompiling).
"""
import mujoco as mj
import numpy as np


def randomize_primitive(model, data, body_name: str, geom_name: str,
                        base_rgb, xy: np.ndarray, rng: np.random.Generator,
                        size_range: tuple[float, float] = (0.88, 1.12),
                        surface_z: float = 0.0) -> None:
    """Randomize one primitive object's size, color, and rest pose in place.

    Must be called after MjData creation and before the first mj_forward. Updates
    both data.qpos and model.qpos0 so mj_resetData preserves the randomized object
    position throughout any IK precomputation loop. No-ops if body_name/geom_name
    aren't present in the model (object commented out of the scene XML).

    surface_z: height of the surface the object rests ON (default 0.0 = floor). For the
    clear-the-table scenes the robot and objects sit on a table/counter, so this is the
    table-top height (TABLE_TOP_Z) — the object is placed at surface_z + its half-height.
    """
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    if bid < 0 or gid < 0:
        return

    s = float(rng.uniform(*size_range))

    # Size: scale geom_size and bounding sphere uniformly
    model.geom_size[gid]   *= s
    model.geom_rbound[gid] *= s

    # Color: fixed pure R/G/B (no jitter)
    model.geom_rgba[gid, :3] = base_rgb

    # Contact sites: scale the x offset (contact-normal direction) in body frame
    for sit_id in range(model.nsite):
        if model.site_bodyid[sit_id] == bid:
            model.site_pos[sit_id, 0] *= s

    # Position: derive rest height from scaled geom size, then write freejoint qpos
    sz    = model.geom_size[gid]          # already scaled
    gtype = model.geom_type[gid]
    if gtype == mj.mjtGeom.mjGEOM_SPHERE:
        z_rest = float(sz[0])
    elif gtype == mj.mjtGeom.mjGEOM_BOX:
        z_rest = float(sz[2])
    elif gtype == mj.mjtGeom.mjGEOM_CYLINDER:
        z_rest = float(sz[1])             # half-height; rests on flat face
    elif gtype == mj.mjtGeom.mjGEOM_CAPSULE:
        z_rest = float(sz[0] + sz[1])     # radius + half-cylinder-length
    else:
        z_rest = float(sz[0])

    jnt_adr = model.body_jntadr[bid]
    if jnt_adr < 0:
        return
    qadr = model.jnt_qposadr[jnt_adr]
    # Rest ON the surface (table/counter top at surface_z; 0.0 = floor).
    pos7 = np.array([xy[0], xy[1], surface_z + z_rest, 1.0, 0.0, 0.0, 0.0])
    data.qpos[qadr:qadr + 7]   = pos7
    model.qpos0[qadr:qadr + 7] = pos7    # preserve across mj_resetData calls

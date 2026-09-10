"""Per-scene object selection + YCB mesh-object attachment for the clear-table scenes.

Two responsibilities:
  1. load_scene_objects(scene): read the per-scene object list from
     models/scene_objects.json (keyed by scene). This is the DEFAULT list; the
     --objects CLI flag overrides it, and a missing file/entry falls back to the primitive
     obj_* bodies authored in the scene XML.
  2. attach_ycb_object(spec, obj_id, xy, surface_z, ...): attach a built YCB object
     (assets/ycb_mjcf/<id>/<id>.xml) into an editable MjSpec as an obj_<slug> body that
     satisfies the clear-table object contract (freejoint, obj_<slug>_geom collision geom,
     condim=6 + grasp-friction, placed on the table). Reuses the MjSpec attach pattern from
     benchmarks/ycb_grasp/scene.py.

Scope: mesh objects are for the plain-teleop dexpilot / anyteleop modes (grasp by hand
physics, no recommender). They carry NO _c1/_c2 grasp sites; the contact-aware recommender
needs those (handled separately) and stays on the primitive stand-ins.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import mujoco as mj

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO, "models", "scene_objects.json")  # beside the scene XMLs
_YCB_DIR = os.path.join(_REPO, "assets", "ycb_mjcf")

_TYPICAL_MAX = 5   # soft cap (warn, don't fail)


def config_path() -> str:
    return _CONFIG_PATH


def ycb_dir() -> str:
    return _YCB_DIR


def load_scene_objects(scene: str, path: str | None = None) -> list[dict] | None:
    """Return the per-scene object list (list of {id, xy, ...}) or None if unavailable.

    Permissive: a missing file, missing scene key, or malformed JSON returns None (caller
    falls back to the scene XML's primitive objects). Warns (does not fail) past
    _TYPICAL_MAX objects.
    """
    p = path or _CONFIG_PATH
    try:
        with open(p) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    objs = cfg.get(scene)
    if not isinstance(objs, list) or not objs:
        return None
    if len(objs) > _TYPICAL_MAX:
        print(f"[scene-objects] warning: {scene} lists {len(objs)} objects "
              f"(> typical {_TYPICAL_MAX}); the table may be crowded.")
    return objs


def slug(obj_id: str) -> str:
    """YCB id -> a valid MuJoCo body prefix: obj_<sanitized id> (e.g. 065-a_cups ->
    obj_065_a_cups)."""
    s = re.sub(r"[^0-9A-Za-z_]", "_", str(obj_id))
    return f"obj_{s}"


def _mjcf_path(obj_id: str) -> str:
    return os.path.join(_YCB_DIR, obj_id, f"{obj_id}.xml")


def object_available(obj_id: str) -> bool:
    return os.path.exists(_mjcf_path(obj_id))


def rest_half_height_from_geom(model, gid) -> float:
    """Half-height of a compiled geom for table placement (mesh uses the local AABB z)."""
    gt = model.geom_type[gid]
    if gt == mj.mjtGeom.mjGEOM_MESH:
        # geom_aabb is (center(3), halfsize(3)) in the geom's local frame.
        return float(model.geom_aabb[gid][5])
    sz = model.geom_size[gid]
    if gt == mj.mjtGeom.mjGEOM_SPHERE:
        return float(sz[0])
    if gt == mj.mjtGeom.mjGEOM_BOX:
        return float(sz[2])
    if gt == mj.mjtGeom.mjGEOM_CYLINDER:
        return float(sz[1])
    if gt == mj.mjtGeom.mjGEOM_CAPSULE:
        return float(sz[0] + sz[1])
    return float(sz[0])


def attach_ycb_object(spec, obj_id: str, xy, *, mass: float | None = None,
                      friction=(2.0, 0.05, 0.005), condim: int = 6,
                      quat=(1.0, 0.0, 0.0, 0.0),
                      solref=(0.004, 1.0)) -> str:
    """Attach a built YCB object into `spec` as an obj_<slug> body. Returns the body name.

    The built YCB mjcf (assets/ycb_mjcf/<id>/<id>.xml) already carries a <freejoint/>,
    <inertial> (published mass), and visual + collision mesh geoms. We rename the body to
    obj_<slug>, name its collision geom obj_<slug>_geom (the clear-table pipeline derives
    the geom name as <body>_geom), set condim + grasp-friction on the collision geoms, and
    place the body at `xy` (z is set by the caller after compile, via the freejoint qpos,
    since the AABB half-height needs the compiled model). Raises FileNotFoundError with the
    build instructions if the object isn't built.

    solref: contact solver reference (time_constant, damping_ratio) for the collision geoms.
      This is the object's CONTACT COMPLIANCE, and it is object-specific — not every YCB item
      is rigid. solref[0] must be >= ~2*timestep (0.004 s at dt=2ms), the stability floor:
        * RIGID objects (boxes, blocks, hard fruit) -> ~0.004 (as stiff as stable; no give,
          rests flush on the table).
        * COMPLIANT objects (orange, apple, tennis ball, soft toys) -> LARGER, ~0.008-0.02,
          so the contact yields a little (deforms) under load — physically realistic. Larger
          is always stable (it's above the floor). Set per object via the config's `solref`.
    """
    path = _mjcf_path(obj_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YCB object {obj_id!r} not built at {path}. Download + convert it first:\n"
            f"  (see README 'YCB object assets') scripts/build_ycb.py --only {obj_id}")

    o = mj.MjSpec.from_file(path)
    body = o.worldbody.first_body()
    name = slug(obj_id)
    body.name = name                               # rename to the obj_<slug> contract
    body.pos = [float(xy[0]), float(xy[1]), 0.0]   # z fixed post-compile
    body.quat = list(quat)
    if mass is not None:
        # Scale the inertial mass if overridden (keep the tensor's shape).
        try:
            body.mass = float(mass)
        except Exception:
            pass

    # Name geoms to the contract + set grasp contact params on the collision geoms.
    # Collision hulls are numbered by ENUMERATION ORDER, not by id(g): CPython
    # reuses the address of a freed temporary, so id(g) & 0xffff collided as soon
    # as an object had more than a couple of hulls and compile failed with
    # "repeated name". Concave YCB objects hit this immediately -- 065-a_cups
    # decomposes into 35 hulls. Visual geoms are indexed the same way, for the
    # same reason.
    col_i = 0
    vis_i = 0
    for g in body.geoms:
        is_col = (g.contype != 0 or g.conaffinity != 0) and g.group == 3
        if is_col:
            g.condim = condim
            g.friction = list(friction)
            # Per-object contact compliance (see the solref arg). Default 0.004 (rigid, the
            # stability floor); the config can soften it for compliant items so they deform a
            # little instead of resting perfectly rigid. MuJoCo's own default (0.02) was too
            # soft for EVERY object and let them sink ~30 mm into the table.
            g.solref = list(solref)
            if col_i == 0:
                g.name = f"{name}_geom"   # <body>_geom, the pipeline's derived name
            else:
                g.name = f"{name}_col_{col_i}"
            col_i += 1
        else:
            g.name = f"{name}_vis_{vis_i}"
            vis_i += 1

    # attach_body renames children in place; a frame lets us prefix cleanly.
    spec.worldbody.add_frame().attach_body(body, "", "")
    return name


def place_on_surface(model, data, body_name: str, surface_z: float, clearance: float = 5e-4):
    """After compile: set the object's freejoint qpos z so its collision geom rests ON the
    surface for its CURRENT orientation, leaving a hair of clearance (default 0.5 mm).

    Orientation-aware: rather than assuming the object sits on its unrotated AABB half-height
    (which is wrong the moment a quat is applied, and even for an upright mesh whose local
    frame origin isn't its centroid), this lifts the object, runs mj_forward to get the geom's
    WORLD pose, rotates the local AABB's 8 corners into the world, and drops the object so the
    lowest corner touches surface_z + clearance. That seats the mesh flush instead of dropping
    it from a gap (the old AABB-half method left YCB objects ~1-4 cm above the table, so they
    free-fell and skittered/tipped on landing — the "objects sliding by themselves" bug).
    Call before the first settling mj_forward/mj_step of the run.
    """
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    jadr = model.body_jntadr[bid]
    if jadr < 0:
        return
    qadr = model.jnt_qposadr[jadr]
    # EVERY collision hull of the body drives the rest height, not just the one
    # named <body>_geom. A convex decomposition splits a concave object into many
    # hulls (065-a_cups: 35), and the first one covers only part of the shape --
    # using it alone placed the cup ~60mm BELOW the table top, its rim poking
    # through, because that hull's AABB is not the object's extent.
    gids = [g for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid and model.geom_group[g] == 3
            and (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0)]
    if not gids:
        # No collision geoms — fall back to a safe gap above the surface.
        data.qpos[qadr + 2] = surface_z + 0.02
        model.qpos0[qadr + 2] = surface_z + 0.02
        return

    # Lift well clear, resolve world poses, find the lowest oriented-AABB corner
    # over all hulls.
    data.qpos[qadr + 2] = surface_z + 0.25
    mj.mj_forward(model, data)
    lowest = min(
        (data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3)
         @ (model.geom_aabb[gid][:3]
            + np.array([sx * model.geom_aabb[gid][3],
                        sy * model.geom_aabb[gid][4],
                        sz * model.geom_aabb[gid][5]])))[2]
        for gid in gids
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
    )
    # Drop so the lowest corner rests just above the surface.
    data.qpos[qadr + 2] -= (lowest - surface_z - clearance)
    model.qpos0[qadr + 2] = data.qpos[qadr + 2]

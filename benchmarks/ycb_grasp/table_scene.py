"""Compose the clear-the-table scene for the pick-and-place benchmark: Kinova
Gen3 + LEAP hand mounted ON a table, YCB mesh objects resting on the table top,
and the bin (bowl_base / bowl_wall_*) authored in scene_pick_place.xml as the
place target.

This is the tabletop analogue of ycb_grasp/scene.py (which floats objects over a
bare floor for the IK/grasp-solver benchmarks). The differences that matter:

  * The robot base is MOUNTED on the table (base_link raised to TABLE_TOP_Z and
    yawed +90 deg about world z, so the arm reaches across the table WIDTH).
    That yaw is why the arm home pose here is HOME_ARM below and NOT
    ik_demo.home_bias()'s Gen3 keyframe -- the latter is authored for an
    unrotated base and reaches the wrong way once the mount is yawed.
  * Objects REST on the table, so grasp planning must keep fingertips clear of
    the TABLE TOP, not of z=0. Callers pass ground_z=TABLE_TOP_Z into the
    planner config; the module-level constant is the single source of truth.
  * The scene already contains a bin and a place site, so the benchmark can run
    a full pick -> transport -> release cycle rather than just a lift.

The scene-building logic on this branch lives inline in
kinova_leap_pick_place.py's __main__ block (object pruning, YCB attachment, base
mounting, post-compile placement). It is extracted here rather than imported
because that block is not callable, and duplicated deliberately rather than
refactored: the teleop entry point is a working, tuned pipeline and this
benchmark should not be able to break it.
"""
from pathlib import Path

import mujoco as mj
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCENE_XML = REPO / "models" / "scene_pick_place.xml"

# Table-top height (m). MUST match kinova_leap_pick_place.TABLE_TOP_Z and the
# table body in scene_kinova_leap.xml (table pos.z 0 + top geom pos.z 0.6 +
# half-thickness 0.025). Objects rest at this height and the robot is mounted on
# it, so it is also the planner's ground plane for fingertip clearance.
TABLE_TOP_Z = 0.625

# Robot base mount: on the table top, moved back toward the near (-y) edge along
# the length centre line, yawed +90 deg CCW about world z so the arm's natural
# +x reach becomes +y (across the table width). Mirrors the loader in
# kinova_leap_pick_place.py -- see its comment for the reachability rationale.
BASE_POS = (0.0, -0.15, TABLE_TOP_Z)
_S45 = 0.7071067811865476
BASE_QUAT = (_S45, 0.0, 0.0, _S45)

# Arm home pose, re-solved for the YAWED base (see kinova_leap_pick_place.py's
# HOME_ARM comment). Using gen3.xml's own "home" keyframe here instead would aim
# the arm along +x, away from the objects.
HOME_ARM = np.array([0.0055, 0.1208, 3.1387, -2.4034, -0.0018, 0.9762, 1.5708])

N_ROBOT = 23

# Bin (place target) authored in scene_pick_place.xml. Base top at 0.635, wall
# top at 0.685; the arrival test is "object XY within +/-BIN_HALF of centre AND
# z between those two".
BIN_CENTER = np.array([0.40, 0.30])
BIN_HALF = 0.12
BIN_BASE_TOP_Z = 0.635
BIN_WALL_TOP_Z = 0.685


def home_qpos(hand_curl=True):
    """(N_ROBOT,) home configuration: HOME_ARM for the arm, a light curl on the
    non-grasping fingers (matching ik_demo.home_bias's hand posture so the
    planner's regularization target is consistent across both benchmarks)."""
    q = np.zeros(N_ROBOT)
    q[:7] = HOME_ARM
    if hand_curl:
        q[11:15] = [1.2, 0.0, 0.5, 0.5]
        q[15:19] = [1.2, 0.0, 0.5, 0.5]
    return q


def build(obj_ids, xys=None, friction=(2.0, 0.05, 0.005), condim=6,
          keep_primitives=False, impratio=None):
    """Compile the tabletop scene with the given YCB objects attached.

    obj_ids : list of YCB ids (e.g. ["036_wood_block"]). Attached as obj_<slug>
        bodies via environments.scene_objects.attach_ycb_object, the same path
        the teleop pipeline uses, so object naming/geom conventions match.
    xys     : optional list of (x, y) table-top positions, one per object. When
        None, positions come from models/scene_objects.json's "pick_place" entry
        where the id appears, else a default in front of the base.
    keep_primitives : the scene XML authors several primitive obj_* stand-ins
        (red box, sphere, ...). They are pruned by default so the only pickable
        bodies are the YCB meshes under test -- a stray primitive would otherwise
        sit on the table and can collide with the arm.
    impratio : override the scene's contact impratio. The scene XML sets 100;
        benchmarks/ycb_grasp/scene.py measured 20 as the best value on the lift
        pipeline (>=50 went non-monotonic and dropped a seed). Left as None
        (scene default) here so the discrepancy is an explicit experiment rather
        than a silent difference -- see the gain-tuning task.

    Returns (model, data, info) where info maps each object body name to its
    body id, qpos address, YCB id and collision-geom name.
    """
    import sys
    sys.path.insert(0, str(REPO))
    from environments import scene_objects as _so

    spec = mj.MjSpec.from_file(str(SCENE_XML))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    if impratio is not None:
        spec.option.impratio = float(impratio)

    # Default XY per object from the scene config, falling back to a spot in
    # front of the base.
    cfg = _so.load_scene_objects("pick_place") or []
    cfg_xy = {e["id"]: e.get("xy") for e in cfg if isinstance(e, dict) and "id" in e}

    # Prune the authored primitive pickables (and every keyframe, which is sized
    # for a specific nq = 23 + 7*N and is invalidated by any object change --
    # deleting a body while keys still reference the old layout corrupts the
    # keyframe registry on compile).
    if not keep_primitives:
        for k in list(spec.keys):
            spec.delete(k)
        for b in list(spec.bodies):
            nm = b.name or ""
            if nm.startswith("obj_") and b.name not in (f"{_so.slug(o)}" for o in obj_ids):
                spec.delete(b)

    placed = []
    for i, oid in enumerate(obj_ids):
        xy = (xys[i] if xys is not None and i < len(xys)
              else cfg_xy.get(oid) or (0.0, 0.32))
        bn = _so.attach_ycb_object(spec, oid, xy, friction=friction, condim=condim)
        placed.append((bn, oid))

    # Mount the robot on the table, yawed (see BASE_POS/BASE_QUAT).
    for b in spec.bodies:
        if b.name == "base_link":
            b.pos = list(BASE_POS)
            b.quat = list(BASE_QUAT)
            break

    model = spec.compile()
    data = mj.MjData(model)

    # Seat each object flush on the table top. Orientation-aware and needs the
    # compiled model, hence post-compile via the freejoint qpos.
    for bn, _ in placed:
        _so.place_on_surface(model, data, bn, TABLE_TOP_Z)

    info = {}
    for bn, oid in placed:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bn)
        if bid < 0:
            raise RuntimeError(f"attached body {bn!r} not found after compile")
        info[bn] = dict(bid=bid, obj_id=oid,
                        qadr=int(model.jnt_qposadr[model.body_jntadr[bid]]),
                        geom=f"{bn}_geom")

    # The grasp solver snapshots object state as data.qpos[N_ROBOT:], so the
    # first object free joint must start exactly there.
    if info:
        first = min(v["qadr"] for v in info.values())
        if first != N_ROBOT:
            raise RuntimeError(f"first object qpos adr {first} != N_ROBOT {N_ROBOT}")

    for j in (0, 2, 4, 6):          # Gen3 continuous joints compile to [0,0]
        model.jnt_range[j] = [-np.pi, np.pi]

    mj.mj_forward(model, data)
    return model, data, info


def settle(model, data, n_steps=400, q_hold=None):
    """Step physics with the robot frozen at its home pose until the objects stop
    moving. Same rationale as pick_from_floor.settle_object_on_floor: MuJoCo's
    soft contact lets an object sink ~1-2mm after placement, which is enough to
    reopen a real fingertip gap if the grasp is planned against the pre-settle
    pose."""
    q = home_qpos() if q_hold is None else q_hold
    for _ in range(n_steps):
        data.qpos[:N_ROBOT] = q
        data.qvel[:N_ROBOT] = 0.0
        mj.mj_step(model, data)
    mj.mj_forward(model, data)


def object_pose(model, data, body_name, info):
    """(pos, quat) of one attached object, read from its free joint."""
    adr = info[body_name]["qadr"]
    return data.qpos[adr:adr + 3].copy(), data.qpos[adr + 3:adr + 7].copy()


def in_bin(model, data, bid):
    """Is this object's body origin inside the bin? Same 3D containment test the
    teleop pipeline's arrival check uses: XY within +/-BIN_HALF of the bin centre
    and z between the bin's base top and wall top."""
    p = data.xpos[bid]
    return bool(np.all(np.abs(p[:2] - BIN_CENTER) <= BIN_HALF)
                and BIN_BASE_TOP_Z <= p[2] <= BIN_WALL_TOP_Z)


def hull_geoms(model, body_name):
    """Collision-hull geom names of one attached object (group 3)."""
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == bid and model.geom_group[g] == 3:
            n = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g)
            if n:
                out.append(n)
    return out


def hull_vertices(model, body_name):
    """All collision-hull vertices of an object, in its body frame."""
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_group[g] != 3:
            continue
        vid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[vid], model.mesh_vertnum[vid]
        R = np.zeros(9)
        mj.mju_quat2Mat(R, model.geom_quat[g])
        out.append(model.mesh_vert[a:a + n] @ R.reshape(3, 3).T + model.geom_pos[g])
    return np.vstack(out)

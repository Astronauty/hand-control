"""Compose a benchmark scene: Kinova Gen3 + LEAP hand + floating YCB objects.

Built from ``models/scene_kinova_leap.xml`` rather than ``scene_pick_place.xml``
-- the latter carries its own box, mocap markers and ten nq=30 keyframes, none of
which belong in a scene meant to isolate the IK solver.

Composition goes through ``MjSpec``, not text XML: an ``<include>`` may not sit
inside ``<worldbody>``, and each YCB file carries its own ``meshdir`` that only
resolves relative to its own directory.

Objects float. Nothing is stepped -- ``mj_forward`` only -- so a free joint is
just a pose to write, and no resting height or settling is involved.

Mass and inertia are already correct in the generated YCB MJCF (an explicit
``<inertial>`` with the published YCB mass), so nothing here touches them.
"""
from pathlib import Path

import mujoco as mj
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE_SCENE = REPO / "models" / "scene_kinova_leap.xml"
YCB_DIR = REPO / "assets" / "ycb_mjcf"

N_ROBOT = 23
GROUP_WORKSPACE = 4
GROUP_TARGET = 5


def object_ids():
    return sorted(p.name for p in YCB_DIR.iterdir()
                  if p.is_dir() and (p / f"{p.name}.xml").exists())


def _attach_object(spec, obj_id, pos, quat, prefix):
    o = mj.MjSpec.from_file(str(YCB_DIR / obj_id / f"{obj_id}.xml"))
    body = o.worldbody.first_body()
    body.pos = list(pos)
    body.quat = list(quat)

    # The YCB files leave geoms unnamed; ConstrainedIKSolver addresses object
    # geoms by name, so name them here (visual first, then one per hull).
    n_col = 0
    for g in body.geoms:
        if g.group == 3:
            g.name = f"col_{n_col}"
            n_col += 1
        else:
            g.name = "vis"

    name = f"{prefix}{body.name}"     # attach_body renames in place, so capture first
    spec.worldbody.add_frame().attach_body(body, prefix, "")
    return name


def _add_mesh_geom(spec, name, verts, faces, rgba, group):
    """Attach an inline mesh as a non-colliding decoration geom."""
    mesh = spec.add_mesh()
    mesh.name = name
    mesh.uservert = np.asarray(verts, dtype=np.float64).ravel().tolist()
    mesh.userface = np.asarray(faces, dtype=np.int32).ravel().tolist()
    body = spec.worldbody.add_body()
    body.name = f"{name}_body"
    g = body.add_geom()
    g.name = name
    g.type = mj.mjtGeom.mjGEOM_MESH
    g.meshname = name
    g.contype, g.conaffinity = 0, 0
    g.group = group
    g.rgba = list(rgba)
    g.mass, g.density = 0.0, 0.0


def build(placements, workspace=None, target_sites=()):
    """Compile a scene.

    placements   : [(obj_id, pos(3), quat(4))]
    workspace    : optional (verts, faces) shell, drawn transparent on group 4
    target_sites : optional [(name, pos(3), rgba)] markers, drawn on group 5

    Returns (model, data, info) where info maps each object to its body id and
    free-joint qpos address.
    """
    spec = mj.MjSpec.from_file(str(BASE_SCENE))
    # default offscreen buffer is 640x480; renders here are larger
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    body_names = []
    for i, (obj_id, pos, quat) in enumerate(placements):
        body_names.append(_attach_object(spec, obj_id, pos, quat, f"o{i}_"))

    if workspace is not None:
        _add_mesh_geom(spec, "ws_shell", workspace[0], workspace[1],
                       (0.20, 0.55, 0.95, 0.13), GROUP_WORKSPACE)

    for name, pos, rgba in target_sites:
        body = spec.worldbody.add_body()
        body.name = f"{name}_body"
        body.pos = list(pos)
        g = body.add_geom()
        g.name = name
        g.type = mj.mjtGeom.mjGEOM_SPHERE
        g.size = [0.008, 0, 0]
        g.contype, g.conaffinity = 0, 0
        g.group = GROUP_TARGET
        g.rgba = list(rgba)
        g.mass, g.density = 0.0, 0.0

    model = spec.compile()
    data = mj.MjData(model)

    info = {}
    for i, bn in enumerate(body_names):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bn)
        if bid < 0:
            raise RuntimeError(f"attached body {bn!r} not found after compile")
        qadr = model.jnt_qposadr[model.body_jntadr[bid]]
        info[bn] = dict(bid=bid, qadr=int(qadr), obj_id=placements[i][0])

    # The IK solver snapshots object state as data.qpos[N_ROBOT:], so the first
    # free joint must start exactly there or that slice silently misaligns.
    first = min(v["qadr"] for v in info.values()) if info else N_ROBOT
    if first != N_ROBOT:
        raise RuntimeError(f"first object qpos adr {first} != N_ROBOT {N_ROBOT}")

    for j in (0, 2, 4, 6):          # Gen3 continuous joints compile to [0,0]
        model.jnt_range[j] = [-np.pi, np.pi]

    mj.mj_forward(model, data)
    return model, data, info


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

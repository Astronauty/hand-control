#!/usr/bin/env python3
"""
scene_export.py — dump a MuJoCo scene's STATIC geometry for the headset.

The headset does not need pixels; it needs the scene. MuJoCo worlds are rigid
bodies, so everything visual splits cleanly in two:

    STATIC  (this script, run once)  geom types, sizes, colours, mesh vertices
    DYNAMIC (pose_stream.py, 72 Hz)  one pose per geom, ~1.4 KB/frame

Bake the static half into the Godot APK and stream the dynamic half, and you get
a true stereo view of the scene for about the same bandwidth as the hand data —
no video encoder, no 30-50 ms of transport latency, and real depth perception
instead of a flat panel floating in VR.

Usage
-----
    python3 teleop/scene_export.py --xml models/scene_pick_place.xml \\
                              --out godot_scene/

Writes:
    scene.json   geom table: type, size, rgba, and the stream index each geom
                 occupies, so the Godot side can match poses to nodes by order
    meshes/*.obj triangle meshes for anything not expressible as a primitive

Copy that directory into the Godot project. Godot imports .obj natively and
scene.json with a JSON parser, so the runtime scene is built procedurally at
startup with no manual asset wiring.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import mujoco as mj
except ImportError:
    sys.exit("mujoco not installed: pip install mujoco")

# MuJoCo geom types this exporter understands. Primitives map onto Godot's own
# mesh primitives (cheap, no asset files); MESH is written out as .obj.
GEOM_NAMES = {
    mj.mjtGeom.mjGEOM_PLANE: "plane",
    mj.mjtGeom.mjGEOM_SPHERE: "sphere",
    mj.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mj.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mj.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mj.mjtGeom.mjGEOM_BOX: "box",
    mj.mjtGeom.mjGEOM_MESH: "mesh",
}


def write_obj(path, verts, faces, normals=None):
    """Minimal OBJ writer — Godot imports these natively."""
    with open(path, "w") as f:
        f.write("# exported from MuJoCo\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for tri in faces:
            a, b, c = (int(i) + 1 for i in tri)
            if normals is not None:
                f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                f.write(f"f {a} {b} {c}\n")


def export(xml_path, out_dir, group_max, include_visual_only, exclude=()):
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    geoms = []
    written = {}
    skipped = 0

    for gid in range(model.ngeom):
        group = int(model.geom_group[gid])
        # Collision geoms (group 3 in this scene) are invisible in the viewer and
        # would just clutter the headset view, so honour the same group filter.
        if group > group_max:
            skipped += 1
            continue
        gtype = int(model.geom_type[gid])
        if gtype not in GEOM_NAMES:
            skipped += 1
            continue
        # Fully transparent geoms are sites/markers the viewer hides.
        rgba = model.geom_rgba[gid].tolist()
        if rgba[3] <= 0.0 and not include_visual_only:
            skipped += 1
            continue

        # Many geoms in a composite model are UNNAMED — the Kinova and LEAP visual
        # meshes come from Menagerie without geom names. Inventing a name here
        # and looking it up later fails, so carry the raw id as the fallback
        # handle and let the streamer resolve by name only when a name exists.
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid) or ""
        # Name-based exclusion. Group filtering is the documented way to
        # separate visual from collision geometry, but this scene authors its
        # collision boxes in the same groups as the visuals, so the only
        # reliable handle is the naming convention the app itself relies on
        # (_finger_collision_geoms matches "collision" in the geom name).
        if name and any(pat in name for pat in exclude):
            skipped += 1
            continue
        entry = {
            "index": len(geoms),          # position in the pose stream
            "name": name,
            "gid": gid,                   # fallback handle for unnamed geoms
            "type": GEOM_NAMES[gtype],
            "size": model.geom_size[gid].tolist(),
            "rgba": rgba,
            "group": group,
        }

        if gtype == mj.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[gid])
            mesh_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_MESH, mid) or f"mesh_{mid}"
            # .txt, not .obj, deliberately. Godot RECOGNISES .obj as an
            # importable resource, so it is excluded from the export preset's
            # "non-resource files" filter and never lands in the APK as raw
            # data — only a ~180-byte .import stub does. The contents are still
            # ordinary OBJ text; scene_view.gd parses them with FileAccess, so
            # only the extension needs to be something Godot ignores.
            fname = f"{mesh_name}.txt"
            if mid not in written:
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                verts = model.mesh_vert[va:va + vn]
                faces = model.mesh_face[fa:fa + fn]
                norms = (model.mesh_normal[va:va + vn]
                         if model.mesh_normal is not None
                         and len(model.mesh_normal) >= va + vn else None)
                write_obj(os.path.join(mesh_dir, fname), verts, faces, norms)
                written[mid] = fname
            entry["mesh"] = f"meshes/{written[mid]}"

        geoms.append(entry)

    n_unnamed = sum(1 for g in geoms if not g["name"])
    scene = {
        "source": os.path.basename(xml_path),
        # Resolving unnamed geoms by id is only valid while the live model has
        # the same geom count as the export. Pruning the scene (e.g. --object)
        # shifts every id, so the streamer checks this and warns.
        "model_ngeom": int(model.ngeom),
        # OpenXR/Godot are Y-up while MuJoCo is Z-up. Rather than rewriting every
        # vertex here, the Godot side applies this single basis change to the
        # scene root, so meshes and streamed poses stay in MuJoCo coordinates all
        # the way to the renderer and there is exactly one place to get it wrong.
        "up_axis": "z",
        "n_geoms": len(geoms),
        "stream_floats": len(geoms) * 7,
        "geoms": geoms,
    }
    with open(os.path.join(out_dir, "scene.json"), "w") as f:
        json.dump(scene, f, indent=1)

    kinds = {}
    for g in geoms:
        kinds[g["type"]] = kinds.get(g["type"], 0) + 1
    print(f"exported {len(geoms)} geoms ({skipped} skipped) -> {out_dir}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"  {len(written)} unique meshes written (.txt — see the note in "
          f"write_obj's caller for why not .obj)")
    print(f"  {n_unnamed} unnamed geoms (resolved by id; re-export if you "
          f"change --object or prune the scene)")
    print(f"  pose stream: {len(geoms)*7} floats = "
          f"{len(geoms)*7*4} bytes/frame "
          f"({len(geoms)*7*4*72/1024:.0f} KB/s at 72 Hz)")
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="models/scene_pick_place.xml")
    ap.add_argument("--out", default="godot_scene")
    ap.add_argument("--group-max", type=int, default=2,
                    help="highest geom group to export; 2 keeps visual meshes "
                         "and drops the group-3 collision boxes")
    ap.add_argument("--include-invisible", action="store_true",
                    help="also export geoms with alpha 0")
    ap.add_argument("--exclude", default="collision",
                    help="comma-separated substrings; any geom whose NAME "
                         "contains one is dropped. Default drops the collision "
                         "boxes, which are invisible in the MuJoCo viewer but "
                         "would render as a blocky shell over the robot.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pats = tuple(p.strip() for p in args.exclude.split(",") if p.strip())
    export(args.xml, args.out, args.group_max, args.include_invisible, pats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
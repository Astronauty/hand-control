import shutil
import tarfile
from pathlib import Path
import trimesh, coacd

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "assets/ycb_raw"
OUT = REPO / "assets/ycb_mjcf"

def extract_archives():
    """YCB downloads land as <name>_google_16k.tgz; unpack to <name>/google_16k/."""
    for tgz in sorted(RAW.rglob("*_google_16k.tgz")):
        name = tgz.name.removesuffix("_google_16k.tgz")
        if (RAW / name / "google_16k").is_dir():
            continue
        with tarfile.open(tgz) as tf:
            tf.extractall(RAW)
        print(f"extracted {name}")

def find_visual_obj(obj_dir):
    p = obj_dir / "google_16k" / "textured.obj"
    return p if p.exists() else None

def process(obj_dir):
    name = obj_dir.name                      # e.g. 006_mustard_bottle
    src = find_visual_obj(obj_dir)
    if src is None:
        print(f"skip {name}: no google_16k mesh")
        return

    dst = OUT / name
    (dst / "collision").mkdir(parents=True, exist_ok=True)

    # copy the visual assets MuJoCo will render
    for f in ["textured.obj", "textured.mtl", "texture_map.png"]:
        s = src.parent / f
        if s.exists():
            shutil.copy(s, dst / f)

    # CoACD convex decomposition -> one obj per hull
    mesh = trimesh.load(src, force="mesh")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces),
                            max_convex_hull=8)   # tune per object
    collision_files = []
    for i, (v, f) in enumerate(parts):
        fn = f"collision/part_{i}.obj"
        trimesh.Trimesh(v, f).export(dst / fn)
        collision_files.append(fn)

    write_mjcf(dst, name, collision_files)

def write_mjcf(dst, name, collision_files):
    # relative paths so it works on any clone
    assets = [f'    <mesh name="{name}_visual" file="textured.obj"/>']
    for i, cf in enumerate(collision_files):
        assets.append(f'    <mesh name="{name}_col_{i}" file="{cf}"/>')

    # type="2d" is required: the default "cube" mapping garbles UV-mapped scans.
    # MuJoCo ignores textured.mtl, so the texture must be declared here and
    # reached via a material -- geoms have no texture attribute of their own.
    if (dst / "texture_map.png").exists():
        assets.append(f'    <texture name="{name}_tex" type="2d" '
                      f'file="texture_map.png"/>')
        assets.append(f'    <material name="{name}_mat" '
                      f'texture="{name}_tex"/>')
        visual_mat = f'material="{name}_mat" '
    else:
        visual_mat = ""

    geoms = [f'      <geom type="mesh" mesh="{name}_visual" '
             f'{visual_mat}class="visual"/>']
    for i in range(len(collision_files)):
        geoms.append(f'      <geom type="mesh" mesh="{name}_col_{i}" '
                     f'class="collision"/>')

    xml = f"""<mujoco model="{name}">
  <compiler meshdir="." texturedir="."/>
  <default>
    <default class="visual">
      <geom group="2" contype="0" conaffinity="0"/>
    </default>
    <default class="collision">
      <geom group="3"/>
    </default>
  </default>
  <asset>
{chr(10).join(assets)}
  </asset>
  <worldbody>
    <body name="{name}" pos="0 0 0">
      <freejoint/>
{chr(10).join(geoms)}
    </body>
  </worldbody>
</mujoco>
"""
    (dst / f"{name}.xml").write_text(xml)
    print(f"wrote {name}")

if __name__ == "__main__":
    extract_archives()
    for d in sorted(RAW.iterdir()):
        if d.is_dir() and d.name != "data":   # "data" holds the downloaded .tgz files
            process(d)
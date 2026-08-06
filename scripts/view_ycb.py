"""View a YCB object, with its CoACD collision hulls visible.

    python scripts/view_ycb.py 006_mustard_bottle
    python scripts/view_ycb.py 035_power_drill --collision-only

Geom groups in these files: 2 = visual mesh, 3 = collision hulls.
In the viewer press 2 / 3 to toggle each group.
"""
import argparse
import tempfile
from pathlib import Path

import mujoco
import mujoco.viewer

from build_ycb import OUT

SCENE = """<mujoco>
  <include file="{xml}"/>
  <statistic center="0 0 0" extent="0.3"/>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="-0.3 0.3 0.8" dir="0.3 -0.3 -1" diffuse="0.5 0.5 0.5"/>
  </worldbody>
</mujoco>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("object", help="e.g. 006_mustard_bottle")
    ap.add_argument("--collision-only", action="store_true",
                    help="hide the visual mesh so only the hulls show")
    args = ap.parse_args()

    obj = OUT / args.object
    xml = obj / f"{args.object}.xml"
    if not xml.exists():
        raise SystemExit(f"no such object: {xml}\n"
                         f"available: {', '.join(p.name for p in sorted(OUT.iterdir())[:5])} ...")

    # include must be top level, and meshdir is relative to the scene file,
    # so the wrapper has to live in the object's own directory
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=obj, delete=False) as f:
        f.write(SCENE.format(xml=xml.name))
        scene = Path(f.name)
    try:
        m = mujoco.MjModel.from_xml_path(str(scene))
    finally:
        scene.unlink()

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    n_hull = sum(1 for g in range(m.ngeom) if m.geom_group[g] == 3)
    print(f"{args.object}: {n_hull} collision hulls")
    print("keys: 2 = visual mesh, 3 = collision hulls, Space = pause/run")

    with mujoco.viewer.launch_passive(m, d) as v:
        v.opt.geomgroup[2] = 0 if args.collision_only else 1
        v.opt.geomgroup[3] = 1          # hulls on
        v.sync()
        while v.is_running():
            # no floor and a freejoint, so hold it still rather than let it fall
            mujoco.mj_forward(m, d)
            v.sync()

if __name__ == "__main__":
    main()

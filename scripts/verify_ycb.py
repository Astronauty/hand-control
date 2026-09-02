"""Check every generated YCB MJCF loads and carries the mass it claims.

    python scripts/verify_ycb.py

Guards the two things the generator can silently get wrong: an MJCF that does not
compile, and a body whose mass is not the published figure (which is what happens
the moment an <inertial> goes missing and MuJoCo falls back to summing geoms).
"""
import json
import sys
from pathlib import Path

import mujoco

import ycb_masses

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assets/ycb_mjcf"


def main():
    report = json.loads((OUT / "fit_report.json").read_text())["objects"]
    failures, checked = [], 0

    for obj in sorted(p for p in OUT.iterdir() if p.is_dir()):
        name = obj.name
        xml = obj / f"{name}.xml"
        if not xml.exists():
            failures.append(f"{name}: no MJCF")
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(xml))
        except Exception as exc:
            failures.append(f"{name}: will not compile -- {exc}")
            continue
        checked += 1

        hulls = sum(1 for i in range(model.ngeom) if model.geom_group[i] == 3)
        on_disk = len(list((obj / "collision").glob("part_*.obj")))
        if hulls != on_disk:
            failures.append(f"{name}: {hulls} collision geoms vs {on_disk} part files")

        mass = float(model.body_mass[1])
        record = report.get(name, {})
        # The report stores mass rounded to 6 decimals (milligrams) while the MJCF
        # carries full precision, so the tolerance here is that rounding -- half an
        # ulp of 1e-6 -- and nothing looser. A published mass has to match exactly:
        # it is a literal from the table, and any drift means the <inertial> was
        # dropped and MuJoCo went back to summing geoms.
        if abs(mass - record.get("mass_kg", -1.0)) > 1e-6:
            failures.append(f"{name}: mass {mass:.9f} kg != report {record.get('mass_kg')}")
        if name in ycb_masses.MASS_G:
            published = ycb_masses.MASS_G[name] / 1000.0
            if abs(mass - published) > 1e-9:
                failures.append(f"{name}: mass {mass:.9f} kg != published {published:.6f}")
        if record.get("n_hulls") not in (None, hulls):
            failures.append(f"{name}: {hulls} hulls vs report {record['n_hulls']}")

    print(f"{checked} MJCFs compiled")
    for f in failures:
        print(f"  FAIL {f}")
    print("all checks passed" if not failures else f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

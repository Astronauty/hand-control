"""One-off: re-emit existing YCB MJCF with texture/material bindings.

The first build wrote <mesh> assets only, so objects rendered untextured grey.
This regenerates each XML from the collision hulls already on disk -- CoACD is
the expensive step and its output is unchanged, so there is no need to re-run it.

Safe to delete once every clone has been rebuilt with the fixed build_ycb.py.
"""
from build_ycb import OUT, write_mjcf

def main():
    patched = skipped = 0
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        hulls = sorted((d / "collision").glob("part_*.obj"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if not hulls:
            print(f"skip {name}: no collision hulls")
            skipped += 1
            continue
        write_mjcf(d, name, [f"collision/{h.name}" for h in hulls])
        patched += 1
    print(f"\npatched {patched}, skipped {skipped}")

if __name__ == "__main__":
    main()

"""Choose a finger count per object from MEASURED patch geometry.

The rule is the seeder's own feasibility test, hoisted out so it can be asked
BEFORE committing to a finger count rather than discovered after a failed fan:

    a slot-k contact can exist  <=>  max_chord >= contact_min_sep_m

where max_chord = 2 * min(c4_patch_offset_m, 0.8 * min_half_extent) is the
furthest two bearings can be placed apart on slot 2's shared quadratic patch,
and contact_min_sep_m is the separation the PADS need not to overlap (22 mm,
the LEAP pad's 30.1x22.2 mm footprint at the contact face).

Slots 3 and 4 both ride slot 2's patch, so the SAME patch has to host every
extra finger -- which is why one measurement decides both.

Reports, per object, the patch it measured and the finger list it would use.
Plan-only: no execution, no renders.

    python benchmarks/ycb_grasp/select_fingers.py --objects 003_cracker_box,017_orange
    python benchmarks/ycb_grasp/select_fingers.py --all
"""
import argparse, io, contextlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

# Everything the tabletop scene can place. Kept explicit rather than globbed off
# assets/: a mesh existing is not the same as the object being graspable here.
CANDIDATES = [
    "003_cracker_box", "004_sugar_box", "008_pudding_box", "009_gelatin_box",
    "010_potted_meat_can", "036_wood_block", "061_foam_brick",
    "002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can",
    "021_bleach_cleanser", "006_mustard_bottle",
    "013_apple", "014_lemon", "015_peach", "016_pear", "017_orange", "018_plum",
    "056_tennis_ball", "055_baseball", "054_softball",
    "025_mug", "065-a_cups", "024_bowl",
    "011_banana", "012_strawberry", "040_large_marker", "038_padlock",
]


def probe(object_id, seed=0):
    """Plan once, read the patch the fourth-contact seeder measured."""
    import simulation.grasp_planner_3d as GP
    cap = {}
    _orig = GP.MultiStartGraspPlanner3D.solve

    def solve(self, *a, **k):
        r = _orig(self, *a, **k)
        cap["res"] = r
        cap["diag"] = getattr(self, "last_c4_patch_diag", None)
        raise SystemExit(0)                     # stop before execution
    GP.MultiStartGraspPlanner3D.solve = solve
    try:
        import benchmarks.ycb_grasp.pick_and_place as PP
        with contextlib.redirect_stdout(io.StringIO()):
            PP.run_pick_place(object_id=object_id, seed=seed,
                              fingers="thumb,index,middle,ring",
                              do_transport=False, out_dir=None, sep_hard=True)
    except SystemExit:
        pass
    except Exception as e:
        return dict(object=object_id, error=f"{type(e).__name__}: {e}")
    finally:
        GP.MultiStartGraspPlanner3D.solve = _orig

    res, diag = cap.get("res"), cap.get("diag")
    if res is None:
        return dict(object=object_id, error="no plan")
    n = sum(1 for k in ("p1", "p2", "p3", "p4") if res.get(k) is not None)
    out = dict(object=object_id, n_contacts=n,
               fingers=["thumb", "index", "middle", "ring"][:n])
    if diag:
        out.update(min_half_mm=round(diag["min_half_mm"], 1),
                   max_chord_mm=round(diag["max_chord_mm"], 1),
                   min_sep_mm=round(diag["min_sep_mm"], 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=None,
                    help="comma-separated ids (default: a small set)")
    ap.add_argument("--all", action="store_true", help="every candidate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write results here")
    a = ap.parse_args()
    objs = (CANDIDATES if a.all else
            [s.strip() for s in a.objects.split(",")] if a.objects else
            CANDIDATES[:6])
    rows = []
    print(f"{'object':22s} {'n':>1s} {'min_half':>8s} {'max_chord':>9s} {'need':>5s}  fingers")
    for o in objs:
        r = probe(o, a.seed)
        rows.append(r)
        if "error" in r:
            print(f"{o:22s} {'-':>1s} {'-':>8s} {'-':>9s} {'-':>5s}  {r['error'][:40]}")
            continue
        print(f"{o:22s} {r['n_contacts']:>1d} "
              f"{r.get('min_half_mm', float('nan')):>8.1f} "
              f"{r.get('max_chord_mm', float('nan')):>9.1f} "
              f"{r.get('min_sep_mm', float('nan')):>5.0f}  "
              f"{','.join(r['fingers'])}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
        print("wrote", a.json)


if __name__ == "__main__":
    main()

"""Head-to-head of force-delivery strategies at n=4.

Two questions, reported separately because they are not the same question and a
strategy can win one while losing the other:

  ACCURACY  does the force the contacts actually deliver match what the
            allocator commanded? (force_track_ratio, and the WORST per-finger
            relative error -- an aggregate can look healthy while one finger
            delivers nothing, which is exactly the failure being chased.)
  SUCCESS   did the grasp work? (all fingers loaded, real lift.)

Never score a cell on displacement alone: a 115 mm lift with a finger at 0.00 N
is not a four-finger grasp.
"""
import argparse, re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "benchmarks/ycb_grasp/out/tabletop/default/force_arms"


def _f(text, name):
    m = re.search(rf"^\s+{re.escape(name)}: (.+)$", text, re.M)
    return m.group(1).strip() if m else None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse(path):
    t = path.read_text(errors="replace")
    arm, _, rest = path.stem.partition("__")
    obj, _, sd = rest.rpartition("_s")
    d = dict(arm=arm, object=obj, seed=int(sd) if sd.isdigit() else -1)
    d["ratio"] = _num(_f(t, "force_track_ratio"))
    d["worst"] = _num(_f(t, "force_track_worst_rel"))
    d["lift_ok"] = (_f(t, "lift_ok") == "True")
    d["dz"] = _num(_f(t, "lift_obj_dz_mm"))
    d["loaded_frac"] = _num(_f(t, "squeeze_all_loaded_frac"))
    v = _f(t, "squeeze_forces_N")
    d["forces"] = None
    if v:
        try:
            d["forces"] = eval(v, {"__builtins__": {}})
        except Exception:
            pass
    # A grasp counts as ALL-LOADED only if every finger carries real force.
    d["all_loaded"] = bool(d["forces"]) and all(x > 0.5 for x in d["forces"].values())
    # Success = force closure AND a real lift, per the repo's own criterion.
    d["success"] = bool(d["all_loaded"] and d["lift_ok"])
    m = re.search(r"^exit=(\d+)", t, re.M)
    d["exit"] = int(m.group(1)) if m else None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT))
    a = ap.parse_args()
    rows = [parse(p) for p in sorted(Path(a.dir).glob("*.log"))]
    if not rows:
        print(f"no logs in {a.dir}")
        return
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)

    hdr = (f"{'arm':11s} {'n':>3s} {'ratio':>7s} {'worst':>7s} {'|worst|<20%':>11s} "
           f"{'allLoaded':>9s} {'lift_ok':>7s} {'SUCCESS':>7s}")
    print(hdr); print("-" * len(hdr))
    order = sorted(by, key=lambda k: -sum(r["success"] for r in by[k]))
    for arm in order:
        rs = by[arm]
        rat = [r["ratio"] for r in rs if r["ratio"] is not None]
        wor = [r["worst"] for r in rs if r["worst"] is not None]
        good = sum(1 for w in wor if abs(w) < 0.20)
        print(f"{arm:11s} {len(rs):>3d} "
              f"{(sum(rat)/len(rat) if rat else float('nan')):>7.3f} "
              f"{(sum(wor)/len(wor) if wor else float('nan')):>7.3f} "
              f"{good:>5d}/{len(wor):<5d} "
              f"{sum(r['all_loaded'] for r in rs):>9d} "
              f"{sum(r['lift_ok'] for r in rs):>7d} "
              f"{sum(r['success'] for r in rs):>7d}")
    print("\nratio = delivered/commanded internal force, mean over the squeeze ramp")
    print("worst = worst per-finger relative error (-1.0 = a finger delivered nothing)")
    print("SUCCESS = every finger loaded AND a real lift")
    bad = [r for r in rows if r["exit"] not in (0, None)]
    if bad:
        print(f"\nnon-zero exits: {len(bad)} "
              f"({', '.join(sorted({r['arm'] for r in bad}))})")


if __name__ == "__main__":
    main()

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
import argparse, re, statistics
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

    hdr = (f"{'arm':11s} {'n':>3s} {'medRatio':>8s} {'IQR':>13s} {'worst':>7s} "
           f"{'|worst|<20%':>11s} {'allLoaded':>9s} {'lift_ok':>7s} {'SUCCESS':>7s}")
    print(hdr); print("-" * len(hdr))
    order = sorted(by, key=lambda k: -sum(r["success"] for r in by[k]))
    for arm in order:
        rs = by[arm]
        # MEDIAN, not mean. The ratio is a quotient and its denominator goes
        # near zero on the cells where the allocator commands almost nothing --
        # measured 15.6 on 025_mug and 3.7 on 056_tennis_ball, which drags a
        # mean of sane per-cell values (0.58-0.95) up to ~3.0 and makes every
        # arm look like it triples the commanded force. The IQR is printed so
        # the spread is visible rather than hidden behind one number.
        rat = sorted(r["ratio"] for r in rs if r["ratio"] is not None)
        wor = [r["worst"] for r in rs if r["worst"] is not None]
        good = sum(1 for w in wor if abs(w) < 0.20)
        _med = statistics.median(rat) if rat else float('nan')
        _iqr = (f"[{rat[len(rat)//4]:.2f},{rat[3*len(rat)//4]:.2f}]"
                if len(rat) >= 4 else "     -       ")
        print(f"{arm:11s} {len(rs):>3d} "
              f"{_med:>8.3f} {_iqr:>13s} "
              f"{(sum(wor)/len(wor) if wor else float('nan')):>7.3f} "
              f"{good:>5d}/{len(wor):<5d} "
              f"{sum(r['all_loaded'] for r in rs):>9d} "
              f"{sum(r['lift_ok'] for r in rs):>7d} "
              f"{sum(r['success'] for r in rs):>7d}")
    print("\nmedRatio = delivered/commanded internal force, MEDIAN over cells")
    print("  (mean is useless here: cells where the allocator commands ~0 N give")
    print("   ratios of 3.7-25.7 and dominate it)")
    print("worst = worst per-finger relative error (-1.0 = a finger delivered nothing)")
    print("SUCCESS = every finger loaded AND a real lift")
    bad = [r for r in rows if r["exit"] not in (0, None)]
    if bad:
        print(f"\nnon-zero exits: {len(bad)} "
              f"({', '.join(sorted({r['arm'] for r in bad}))})")


if __name__ == "__main__":
    main()

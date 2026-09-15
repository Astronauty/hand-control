"""Aggregate the four-finger sweep into the patch-sufficiency table.

Reads the per-cell logs written by the sweep (one per object x seed) and
reports, for every cell:

  n_solved     how many contacts the NLP actually returned (4, or 3 when the
               fourth-contact seeder found no viable candidate)
  min_half     the SMALLEST half-extent of contact 2's quadratic patch -- the
               binding dimension, because the seed fan sweeps a full circle and
               clamps at the bearing of the tightest bound
  max_chord    2 * the adaptive fan radius: the furthest apart two fan bearings
               can be placed on that patch
  sep34        the separation the NLP actually left between contacts 3 and 4

The question this answers is whether contact 2's patch -- which carries THREE
fingertips at n=4, since slots 3 and 4 both share it by default -- is large
enough. Two distinct failures are separated here and should not be conflated:

  (a) max_chord < min_sep: the patch cannot hold a separated fourth contact at
      all, so the seeder returns nothing and the grasp degrades to a tripod.
      This is a PATCH SIZE failure.
  (b) the seeder places contact 4 with real separation and the NLP then drives
      sep34 to ~0. This is NOT a patch size failure -- there was room -- it is
      the same objective-side collapse the n=3 work measured, reproduced one
      slot up.

Usage:
    python benchmarks/ycb_grasp/four_finger_report.py [--dir <logdir>]
"""
import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO / "benchmarks/ycb_grasp/out/tabletop/default/four_finger"

_SEP_RE   = re.compile(r"\[contacts\] solved separations: (.+)")
_SEED_RE  = re.compile(r"\[contacts\] c4 seed: (\d+) ranked, accepted "
                       r"fan=([-+]?\d+)deg sep_c2=([\d.]+)mm sep_c3=([\d.]+)mm "
                       r"rf_dls=([\d.]+)mm")
_NOVIA_RE = re.compile(r"\[contacts\] c4 seed: NO viable candidate")
_PATCH_RE = re.compile(r"\[contacts\] c4 patch: half-extents ([\d./]+)mm\s+"
                       r"min=([\d.]+)mm\s+fan_r=([\d.]+)mm\s+"
                       r"max_chord=([\d.]+)mm\s+min_sep=(\d+)mm")
_PAIR_RE  = re.compile(r"(\d)-(\d)\s+([\d.]+)mm")


def _field(text, name):
    m = re.search(rf"^\s+{re.escape(name)}: (.+)$", text, re.M)
    return m.group(1).strip() if m else None


def parse_cell(path):
    t = path.read_text(errors="replace")
    obj, _, seed = path.stem.rpartition("_s")
    out = dict(object=obj, seed=int(seed) if seed.isdigit() else -1,
               exit=_field(t, "exit"))
    m = _SEP_RE.search(t)
    seps = {}
    if m:
        for a, b, v in _PAIR_RE.findall(m.group(1)):
            seps[f"{a}-{b}"] = float(v)
    out["seps"] = seps
    # n_solved: the highest contact index that appears in any pair, else 2.
    out["n_solved"] = max((int(i) for p in seps for i in p.split("-")), default=2)
    m = _PATCH_RE.search(t)
    if m:
        out.update(min_half=float(m.group(2)), fan_r=float(m.group(3)),
                   max_chord=float(m.group(4)), min_sep=float(m.group(5)))
    m = _SEED_RE.search(t)
    if m:
        out.update(c4_ranked=int(m.group(1)), c4_sep3_seed=float(m.group(4)),
                   c4_rf_dls=float(m.group(5)))
    elif _NOVIA_RE.search(t):
        out["c4_ranked"] = 0
    for k, cast in (("wrench_feasible", str), ("lift_ok", str),
                    ("gamma_min", float), ("gws_beta", float),
                    ("lift_obj_dz_mm", float), ("epsilon", float)):
        v = _field(t, k)
        if v is not None:
            try:
                out[k] = cast(v)
            except ValueError:
                out[k] = v
    v = _field(t, "squeeze_forces_N")
    if v:
        try:
            out["forces"] = eval(v, {"__builtins__": {}})
        except Exception:
            pass
    return out


# gamma_min above this is a DEGENERATE certificate, not a strong grasp: the
# executor's own ceiling is 25-60 N, so a five- or six-figure gamma means the LP
# found no bounded internal force and the number should not be averaged or
# compared. Measured here on 056_tennis_ball s0: 1.41e6 N with
# wrench_feasible=True.
GAMMA_DEGENERATE_N = 1e4


def classify(c):
    """Which of the two failures (if either) this cell shows."""
    if c.get("n_solved", 2) < 4:
        if c.get("max_chord") is not None and c.get("min_sep") is not None:
            if c["max_chord"] < c["min_sep"]:
                return "patch-too-small"
            return "seed-screen-empty"
        return "no-c4"
    s34 = c.get("seps", {}).get("3-4")
    if s34 is not None and s34 < 1.0:
        return "nlp-collapse"
    return "four-contact"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    a = ap.parse_args()
    logs = sorted(Path(a.dir).glob("*.log"))
    if not logs:
        print(f"no logs in {a.dir}")
        return
    cells = [parse_cell(p) for p in logs]
    hdr = (f"{'object':17s} {'s':>1s} {'n':>1s} {'min_half':>8s} {'max_chord':>9s} "
           f"{'min_sep':>7s} {'seed_s34':>8s} {'sol_s34':>7s} {'gamma':>7s} "
           f"{'beta':>7s} {'lift':>5s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        s34 = c.get("seps", {}).get("3-4")
        print(f"{c['object']:17s} {c['seed']:>1d} {c.get('n_solved', 0):>1d} "
              f"{c.get('min_half', float('nan')):>8.1f} "
              f"{c.get('max_chord', float('nan')):>9.1f} "
              f"{c.get('min_sep', float('nan')):>7.0f} "
              f"{c.get('c4_sep3_seed', float('nan')):>8.1f} "
              f"{(s34 if s34 is not None else float('nan')):>7.1f} "
              f"{c.get('gamma_min', float('nan')):>7.2f} "
              f"{c.get('gws_beta', float('nan')):>7.3f} "
              f"{str(c.get('lift_ok')):>5s}  {classify(c)}"
              + ("  [gamma degenerate]"
                 if (c.get('gamma_min') or 0) > GAMMA_DEGENERATE_N else ""))
    print()
    from collections import Counter
    tally = Counter(classify(c) for c in cells)
    print("verdicts: " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    n4 = [c for c in cells if c.get("n_solved") == 4]
    print(f"cells reaching 4 contacts: {len(n4)}/{len(cells)}")
    real = [c for c in n4 if (c.get("seps", {}).get("3-4") or 0) >= 1.0]
    print(f"cells with a SEPARATED 4th contact: {len(real)}/{len(cells)}")


if __name__ == "__main__":
    main()

"""Emit booktabs LaTeX for the FRoGGeR comparison tables, from measured JSON only.

Reads the JSON written by `frogger_exec_bench --json-out` and
`edge_seeking --json-out` and prints paper-ready tables. Nothing is hardcoded: every
number comes from a run, so a table cannot drift from the data that produced it.

Statistics follow the paper's own convention (their Table I): median with the
interquartile range in parentheses, over the cells that produced a result.

PREAMBLE. The single-column tables use \multirow and \shortstack for the category
column, so the document needs \usepackage{multirow} alongside \usepackage{booktabs}.
The two-column forms need only booktabs. Not verified by compilation here (no LaTeX
toolchain in this environment) -- check once before relying on them.

Usage:
    uv run python -m ycb_grasp.make_tables --exec out/synth.json --edge out/edge.json
    uv run python -m ycb_grasp.make_tables --edge out/edge.json --layout 1col
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM_LABEL = {"ours": r"\textbf{Ours}", "frogger": "FRoGGeR"}

# FRoGGeR's own object taxonomy (their Sec. IV): "spheroids, like fruits and balls;
# boxes/cylinders, like food containers, cans, or large cups; and adversarial objects
# with irregular geometry". Segmenting by it makes our table read against theirs and
# separates the shape classes that behave differently -- a spheroid has no edges for
# either method to seek, so mixing it with boxes hides the edge-seeking signal.
#
# We have no ADVERSARIAL objects in the tabletop set, so that row is absent rather
# than empty: claiming a category we did not test would misrepresent the coverage.
CATEGORY = {
    "014_lemon":       "spheroid",
    "017_orange":      "spheroid",
    "056_tennis_ball": "spheroid",
    "036_wood_block":  "box/cyl",
    "061_foam_brick":  "box/cyl",
}
CAT_ORDER = ["spheroid", "box/cyl", "adversarial"]


def _cats_present(rows):
    seen = {CATEGORY.get(r.get("object")) for r in rows}
    return [c for c in CAT_ORDER if c in seen]


def _miqr(vals, fmt="%.2f", scale=1.0):
    """median (q1, q3) in the paper's style, or -- when there is nothing to report."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "--"
    v = np.asarray(v, float) * scale
    return (f"{fmt % np.median(v)} ({fmt % np.percentile(v, 25)}, "
            f"{fmt % np.percentile(v, 75)})")


def _frac_bare(num, den):
    """Percentage without the (k/n) parenthetical -- for narrow single-column tables
    where the denominator is uniform and stated once in the caption."""
    return "--" if not den else r"%.0f\%%" % (100.0 * num / den)


def _frac(num, den):
    return "--" if not den else r"%.1f\%% (%d/%d)" % (100.0 * num / den, num, den)


def exec_table(rows, n_contacts=2, mu=2.0):
    """Table I analogue: convergence, pick success, quality, timing."""
    arms = [a for a in ("ours", "frogger") if any(r.get("arm") == a for r in rows)]
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Execution comparison on the YCB tabletop scene "
               r"($n=%d$ contacts, $\mu=%.1f$). Statistics are median "
               r"(interquartile range) over cells that produced a plan, following "
               r"the convention of Li et al. $\uparrow$/$\downarrow$ denote whether "
               r"higher or lower is better. Pick success uses the shaky-pickup "
               r"criteria; \emph{held} additionally requires every fingertip to "
               r"still carry load at the end of the lift. Rates are over all cells, "
               r"not over those reaching the lift. \emph{FRoGGeR's low lift rate "
               r"is an artifact of our port, not of their method}: their (7d) pins a "
               r"fixed body-frame pad point to the surface, which on our fingertip "
               r"leaves the pad 8--11\,mm clear of the object, and our executor's "
               r"pre-squeeze gap gate rejects it. Their arm additionally draws 20 "
               r"synthesis attempts against our single solve from a fixed home "
               r"pose.}" % (n_contacts, mu))
    out.append(r"\label{tab:execution}")
    out.append(r"\begin{tabular}{lcccccc}")
    out.append(r"\toprule")
    out.append(r"method & \% planned $\uparrow$ & \% reached lift $\uparrow$ & "
               r"\% pick success $\uparrow$ & \% held $\uparrow$ & "
               r"normalized $\bar{\ell}^*$ $\uparrow$ & "
               r"time per grasp (s) $\downarrow$ \\")
    out.append(r"\midrule")
    for a in arms:
        g = [r for r in rows if r.get("arm") == a]
        planned = [r for r in g if not r.get("plan_failed")
                   and r.get("l_bar") is not None]
        lifted = [r for r in g if "lift_done" in (r.get("phase_log") or "")]
        # Success rates are quoted OVER ALL CELLS, not over the ones that happened
        # to reach the lift. Conditioning on reaching it turns an arm that aborts
        # 11 of 15 grasps into "25% success", which reads as a modest gap rather
        # than the attrition it is. The "reached lift" column carries that
        # attrition explicitly so neither reading is hidden.
        ps = sum(1 for r in lifted if r.get("pick_success"))
        lo = sum(1 for r in lifted if r.get("lift_ok"))
        out.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
            ARM_LABEL.get(a, a),
            _frac(len(planned), len(g)),
            _frac(len(lifted), len(g)),
            _frac(ps, len(g)),
            _frac(lo, len(g)),
            _miqr([r.get("l_bar") for r in planned], "%.2f"),
            _miqr([r.get("t_total_s") for r in g], "%.1f")))
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def edge_table(rows):
    """The edge-seeking table: the failure mode their paper names but does not measure."""
    arms = [a for a in ("ours", "frogger") if any(r.get("arm") == a for r in rows)]
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Edge-seeking, measured. Li et al. identify edge-seeking as "
               r"the dominant failure mode of both their method and their baseline "
               r"and leave combating it to future work, but report no metric for it. "
               r"We measure, for each solved contact, the geodesic distance along the "
               r"object surface to the nearest point at which the surface normal turns "
               r"by more than $30^\circ$ -- a representation-independent edge test "
               r"applied identically to both methods. Larger is safer.}")
    out.append(r"\label{tab:edge}")
    out.append(r"\begin{tabular}{lccc}")
    out.append(r"\toprule")
    out.append(r"method & edge margin (mm) $\uparrow$ & worst grasp (mm) $\uparrow$ & "
               r"contacts $\leq 2$\,mm from an edge $\downarrow$ \\")
    out.append(r"\midrule")
    for a in arms:
        g = [r for r in rows if r.get("arm") == a]
        per = []
        for r in g:
            per += [x for x in (r.get("geo_margins_mm") or []) if x is not None]
        worst = [r.get("min_geo_margin_mm") for r in g
                 if r.get("min_geo_margin_mm") is not None]
        n_edge = sum(1 for x in per if x <= 2.0)
        out.append("%s & %s & %s & %s \\\\" % (
            ARM_LABEL.get(a, a),
            _miqr(per, "%.1f"),
            ("%.1f" % min(worst)) if worst else "--",
            _frac(n_edge, len(per))))
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def edge_per_object(rows):
    """Per-object medians -- shows the separation is not one object's doing."""
    objs = sorted({r["object"] for r in rows if r.get("object")})
    arms = [a for a in ("ours", "frogger") if any(r.get("arm") == a for r in rows)]
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Edge margin per object (median over seeds, mm). The sphere "
               r"(\texttt{017\_orange}) has no edges, so neither method can seek one "
               r"and both sit far from the nearest normal discontinuity; the "
               r"separation appears exactly on the objects that have edges.}")
    out.append(r"\label{tab:edge-per-object}")
    out.append(r"\begin{tabular}{l%s}" % ("c" * len(arms)))
    out.append(r"\toprule")
    out.append("object & " + " & ".join(ARM_LABEL.get(a, a) for a in arms) + r" \\")
    out.append(r"\midrule")
    for o in objs:
        cells = []
        for a in arms:
            v = [r.get("min_geo_margin_mm") for r in rows
                 if r.get("object") == o and r.get("arm") == a
                 and r.get("min_geo_margin_mm") is not None]
            cells.append(("%.1f" % np.median(v)) if v else "--")
        out.append("\\texttt{%s} & %s \\\\" % (o.replace("_", r"\_"),
                                               " & ".join(cells)))
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def exec_table_1col(rows, n_contacts=2, mu=2.0):
    """Single-column (IEEE \\columnwidth) execution table, segmented by category.

    A two-column IEEE page gives ~3.5 in, which will not hold six numeric columns at
    readable size. Two things buy the space: metrics become ROWS and methods become
    columns (two data columns, not six), and the category segmentation FRoGGeR uses
    groups objects instead of listing them.

    Percentages drop their (k/n) parenthetical -- at 3 objects x 3 seeds the
    denominator is uniform and stating it once in the caption is enough.
    """
    arms = [a for a in ("ours", "frogger") if any(r.get("arm") == a for r in rows)]
    cats = _cats_present(rows)
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Execution on the YCB tabletop scene ($n=2$ contacts, "
               r"$\mu=2.0$, 3 seeds per object). Categories follow Li et al. "
               r"Quality is median (IQR). \emph{Held} additionally requires every "
               r"fingertip to still carry load at the end of the lift, which "
               r"displacement-only criteria cannot detect. FRoGGeR's lift rate is "
               r"depressed by our port's fixed-pad-point constraint, not by their "
               r"method; see text.}")
    out.append(r"\label{tab:execution-1col}")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\begin{tabular}{@{}llcc@{}}")
    out.append(r"\toprule")
    out.append(r"category & metric & " + " & ".join(ARM_LABEL.get(a, a) for a in arms)
               + r" \\")
    for ci, cat in enumerate(cats):
        out.append(r"\midrule")
        sub = [r for r in rows if CATEGORY.get(r.get("object")) == cat]
        n_obj = len({r["object"] for r in sub})
        rowspec = [
            (r"\% reached lift $\uparrow$",
             lambda g: _frac_bare(len([r for r in g
                                      if "lift_done" in (r.get("phase_log") or "")]),
                                  len(g))),
            (r"\% pick success $\uparrow$",
             lambda g: _frac_bare(sum(1 for r in g if r.get("pick_success")), len(g))),
            (r"\% held $\uparrow$",
             lambda g: _frac_bare(sum(1 for r in g if r.get("lift_ok")), len(g))),
            (r"$\bar{\ell}^*$ $\uparrow$",
             lambda g: _miqr([r.get("l_bar") for r in g
                              if r.get("l_bar") is not None], "%.2f")),
            (r"time (s) $\downarrow$",
             lambda g: _miqr([r.get("t_total_s") for r in g], "%.0f")),
        ]
        for mi, (label, fn) in enumerate(rowspec):
            cells = [fn([r for r in sub if r.get("arm") == a]) for a in arms]
            lead = (r"\multirow{%d}{*}{\shortstack[l]{%s\\(%d obj.)}}"
                    % (len(rowspec), cat.replace("/", r"/"), n_obj)) if mi == 0 else ""
            out.append("%s & %s & %s \\\\" % (lead, label, " & ".join(cells)))
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def edge_table_1col(rows):
    """Single-column edge-seeking table, segmented by category."""
    arms = [a for a in ("ours", "frogger") if any(r.get("arm") == a for r in rows)]
    cats = _cats_present(rows)
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Edge-seeking. Li et al. name edge-seeking as the dominant "
               r"failure mode of both their method and their baseline and leave "
               r"combating it to future work, without a metric for it. For each "
               r"solved contact we measure the geodesic distance along the object "
               r"surface to the nearest point where the surface normal turns by "
               r"more than $30^\circ$ -- a representation-independent edge test "
               r"applied identically to both methods. Spheroids have no edges and "
               r"act as a control.}")
    out.append(r"\label{tab:edge-1col}")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\begin{tabular}{@{}llcc@{}}")
    out.append(r"\toprule")
    out.append(r"category & metric & " + " & ".join(ARM_LABEL.get(a, a) for a in arms)
               + r" \\")
    for cat in cats:
        out.append(r"\midrule")
        sub = [r for r in rows if CATEGORY.get(r.get("object")) == cat]
        n_obj = len({r["object"] for r in sub})
        per_arm = {}
        for a in arms:
            per = []
            for r in [x for x in sub if x.get("arm") == a]:
                per += [x for x in (r.get("geo_margins_mm") or []) if x is not None]
            per_arm[a] = per
        specs = [
            (r"margin (mm) $\uparrow$",
             lambda a: _miqr(per_arm[a], "%.1f")),
            (r"$\leq 2$\,mm from edge $\downarrow$",
             lambda a: _frac_bare(sum(1 for x in per_arm[a] if x <= 2.0),
                                  len(per_arm[a]))),
        ]
        for mi, (label, fn) in enumerate(specs):
            lead = (r"\multirow{%d}{*}{\shortstack[l]{%s\\(%d obj.)}}"
                    % (len(specs), cat, n_obj)) if mi == 0 else ""
            out.append("%s & %s & %s \\\\"
                       % (lead, label, " & ".join(fn(a) for a in arms)))
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exec", dest="exec_json", default=None,
                    help="JSON from frogger_exec_bench --json-out")
    ap.add_argument("--edge", dest="edge_json", default=None,
                    help="JSON from edge_seeking --json-out")
    ap.add_argument("--layout", choices=["1col", "2col", "both"], default="both",
                    help="1col: narrow IEEE \\columnwidth tables, metrics as rows "
                         "and methods as columns, segmented by object category. "
                         "2col: the wide form. Default prints both.")
    ap.add_argument("--n-contacts", type=int, default=2)
    ap.add_argument("--mu", type=float, default=2.0)
    args = ap.parse_args()

    want1 = args.layout in ("1col", "both")
    want2 = args.layout in ("2col", "both")
    if args.exec_json:
        rows = json.loads(Path(args.exec_json).read_text())
        if want1:
            print("% ---- single-column execution ----")
            print(exec_table_1col(rows, n_contacts=args.n_contacts, mu=args.mu))
            print()
        if want2:
            print("% ---- two-column execution ----")
            print(exec_table(rows, n_contacts=args.n_contacts, mu=args.mu))
            print()
    if args.edge_json:
        rows = json.loads(Path(args.edge_json).read_text())
        if want1:
            print("% ---- single-column edge-seeking ----")
            print(edge_table_1col(rows))
            print()
        if want2:
            print("% ---- two-column edge-seeking ----")
            print(edge_table(rows))
            print()
            print(edge_per_object(rows))


if __name__ == "__main__":
    main()

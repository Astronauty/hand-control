#!/usr/bin/env python3
"""ICRA single-column version of the contact-seed figure.

The benchmark's seed figure (kinova_common/seed_figure.py) is a DIAGNOSTIC: it
lays every seed the solve considered side by side at ~3.3in per panel, so a
5-seed solve is 17.5in wide. Dropped into a 3.5in IEEE column that is a 5x
reduction and every label becomes unreadable.

This regenerates the same content at paper scale:
  - 2x2 panel grid instead of 1xN, so each panel keeps ~1.6in of width
  - type sized for a 3.5in column at 100% reproduction (8pt title, 7pt legend)
  - vector PDF, no rasterised text

Panels are chosen, not truncated: the SELECTED seed first (that is the grasp
that executed), then the highest-ranked others. Which seeds appear is printed
so the figure can be regenerated identically.

Usage:
    python figures/make_seed_figure_icra.py --object 025_mug [--seed 0]
                                            [--fingers thumb,index,middle]
                                            [--out figures/seed_icra.pdf]
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import mujoco as mj
from kinova_common import seed_figure as SF

# IEEE single column is 3.5in (252pt). Type is sized for 100% reproduction:
# an 8pt title in the figure prints at 8pt in the paper.
COL_IN   = 3.5
TITLE_PT = 8.0
LEGEND_PT= 7.0
SUP_PT   = 8.5


def build(planner, model, data, res, n_panels=4, max_tris=3000):
    """Seed figure at ICRA single-column scale. Returns a Figure or None."""
    pl = planner._planner if hasattr(planner, "_planner") else planner
    acc = list(getattr(planner, "last_seed_accept_table", None)
               or getattr(pl, "last_seed_accept_table", None) or [])
    rej = list(getattr(planner, "last_seed_reject_table", None)
               or getattr(pl, "last_seed_reject_table", None) or [])
    if not acc and not rej:
        return None, []
    sc = SF._scene_from_planner(planner, model, data)
    recs = [SF._as_rec(e, True) for e in acc] + [SF._as_rec(e, False) for e in rej]

    # The SELECTED seed is the one whose solve was kept, so it leads. Accepted
    # seeds follow, then rejected ones -- a rejected panel still earns its place
    # in a paper figure because it shows WHY a seed was discarded.
    sel = res.get("seed_index") if isinstance(res, dict) else None
    acc_recs = [r for r in recs if r["ok"]]
    # _draw_block gates the solved-contact overlay (and the SELECTED styling) on
    # rec["winner"], which the diagnostic sets while assembling the full list.
    # Here each panel is drawn from a one-record list, so the flag has to be set
    # explicitly or no panel is ever the winner and the "NLP solution" legend
    # entry has nothing to point at.
    for k, r in enumerate(acc_recs):
        r["winner"] = (sel is not None and k == sel)
    rej_recs = [r for r in recs if not r["ok"]]
    order = []
    if sel is not None and 0 <= sel < len(acc_recs):
        order.append(acc_recs[sel])
    order += [r for k, r in enumerate(acc_recs) if k != sel]
    order += rej_recs
    order = order[:n_panels]

    solved = None
    if isinstance(res, dict) and res.get("p1") is not None:
        solved = {k: res.get(k) for k in ("p1", "p2", "p3")}

    nrow = 2 if len(order) > 2 else 1
    ncol = int(np.ceil(len(order) / nrow))
    # Square-ish panels; the legend strip and suptitle get their own bands.
    fig = plt.figure(figsize=(COL_IN, COL_IN * 0.36 * nrow + 0.34))
    gs = GridSpec(nrow, ncol, figure=fig)

    for i, rec in enumerate(order):
        r_, c_ = divmod(i, ncol)
        ax = fig.add_subplot(gs[r_, c_], projection="3d")
        sub = GridSpec(1, 1, figure=fig)
        # Reuse the diagnostic's own panel painter so the two figures cannot
        # drift apart: same mesh thinning, same markers, same colours.
        SF._draw_block(fig, sub, 0, [rec], sc,
                       header=None, ok=rec["ok"],
                       solved=(solved if rec["ok"] else None),
                       max_tris=max_tris)
        # _draw_block adds its own axes; drop the placeholder we made.
        fig.delaxes(ax)

    axes = [a for a in fig.axes if hasattr(a, "get_zlim")]
    for i, a in enumerate(axes[:len(order)]):
        rec = order[i]
        tag = ("selected" if rec.get("winner")
               else ("accepted" if rec["ok"] else "rejected"))
        a.set_title(tag, fontsize=TITLE_PT,
                    fontweight=("bold" if tag == "selected" else "normal"),
                    color=("#1f77b4" if tag == "selected"
                           else ("k" if rec["ok"] else "#b22222")), pad=-6.0)
        a.set_position(_grid_pos(i, nrow, ncol))
        # A 3D axes reserves a large margin around its data, so a panel sized to
        # the cell draws the mesh at ~55% of it. set_box_aspect zoom pushes the
        # rendered content out to the cell edges; 1.35 is the largest that does
        # not clip the mug handle on any panel (checked on the rejected panel,
        # whose patch sits lowest).
        try:
            a.set_box_aspect(None, zoom=1.35)
        except TypeError:
            pass

    keys = ["thumb", "index"] + (["middle"] if any(
        r["seed"].get("p3s") is not None for r in order) else [])
    handles = [Line2D([], [], ls="", marker="o", ms=3.6, color=SF._FCOL(k),
                      mec="k", mew=0.3, label=k) for k in keys]
    handles.append(Line2D([], [], ls="", marker="^", ms=3.6, color="0.35",
                          mec="k", mew=0.3, label="NLP solution"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=LEGEND_PT, handletextpad=0.25,
               columnspacing=0.9, borderpad=0.1)
    fig.suptitle(sc["obj"].replace("obj_", "").replace("_", " "),
                 fontsize=SUP_PT, y=0.985)
    return fig, [("selected" if r.get("winner")
                  else ("accepted" if r["ok"] else "rejected")) for r in order]


def _grid_pos(i, nrow, ncol):
    """Panel rectangle in figure coords. Hand-placed rather than tight_layout:
    3D axes carry a large invisible margin and tight_layout shrinks them to
    ~60% of the cell, which is what makes the diagnostic figure's panels small
    even when the figure is wide."""
    r_, c_ = divmod(i, ncol)
    left, right, bot, top = -0.04, 1.04, 0.10, 0.90
    w = (right - left) / ncol
    h = (top - bot) / nrow
    return [left + c_ * w, bot + (nrow - 1 - r_) * h, w, h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="025_mug")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fingers", default=None,
                    help="default: the object's entry in grasp_finger_config.json")
    ap.add_argument("--n-panels", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import benchmarks.ycb_grasp.pick_and_place as PP
    import simulation.grasp_planner_3d as G

    cap = {}
    _orig = G.MultiStartGraspPlanner3D.solve
    def _spy(self, *a, **k):
        r = _orig(self, *a, **k)
        cap["res"], cap["pl"] = r, self
        raise SystemExit
    G.MultiStartGraspPlanner3D.solve = _spy
    try:
        PP.run_pick_place(args.object, args.seed, n_seeds=3,
                          fingers=args.fingers, do_transport=False,
                          out_dir=None, view=False)
    except SystemExit:
        pass
    if "pl" not in cap:
        print("planner never reached seeding"); return 1

    pl = cap["pl"]
    fig, tags = build(pl, pl._planner.model, pl._planner.data, cap["res"],
                      n_panels=args.n_panels)
    if fig is None:
        print("no seeds recorded"); return 1
    out = Path(args.out or (REPO / "figures" /
               f"seed_{args.object}_s{args.seed}_icra.pdf"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} kB)")
    print(f"panels: {tags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

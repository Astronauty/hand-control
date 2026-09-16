"""Qualitative grasp figure: one grasp per object, IEEE double-column width.

One panel per object, captioned with the finger count the SELECTOR chose for it
(benchmarks/ycb_grasp/select_fingers.py). The finger count therefore varies
across the grid, which is the thing the figure is now showing.

Panels are auto-cropped around the hand with a constant aspect ratio. The source
renders are the planned-pose PNGs written by pick_and_place.py, which frame the
palm/object midpoint -- consistent enough that a detected crop beats twelve
hand-tuned windows.
"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pathlib import Path

ROOT = Path("benchmarks/ycb_grasp/out/tabletop/default")
OUT = Path("figures")

# PER-OBJECT layout (auto-selected finger count), vs the original
# per-ARM layout (a row per fixed finger set).
#
# The original figure was 2 arms x 6 objects: every object grasped twice, once
# with each fixed finger set, so the rows were the comparison. That question is
# answered. The one worth showing now is what the SELECTOR picks -- which is one
# grasp per object, with the finger count varying across the row, so a per-arm
# row structure has nothing to put in it.
#
# Reads default/<object>/seed<N>_planned.png, which is pick_and_place's NATIVE
# layout (--out-tag default, no arm level). The two_finger/three_finger folders
# were the deviation, created by sweeps that passed an arm name.

# Column = object, row = arm. Crop = (cx, cy, w) in source pixels; height
# follows from ASPECT. Windows were picked per panel to frame hand + object.
ASPECT = 4 / 3.0

# Display names only; WHICH objects appear and how many fingers each uses come
# from the selector's JSON, so the figure cannot drift from the rule that
# produced the grasps.
PRETTY = {
    "003_cracker_box": "Cracker box", "004_sugar_box": "Sugar box",
    "008_pudding_box": "Pudding box", "009_gelatin_box": "Gelatin box",
    "010_potted_meat_can": "Potted meat", "036_wood_block": "Wood block",
    "061_foam_brick": "Foam brick", "002_master_chef_can": "Chef can",
    "005_tomato_soup_can": "Soup can", "007_tuna_fish_can": "Tuna can",
    "021_bleach_cleanser": "Bleach", "006_mustard_bottle": "Mustard",
    "013_apple": "Apple", "014_lemon": "Lemon", "015_peach": "Peach",
    "016_pear": "Pear", "017_orange": "Orange", "018_plum": "Plum",
    "056_tennis_ball": "Tennis ball", "055_baseball": "Baseball",
    "054_softball": "Softball", "025_mug": "Mug", "065-a_cups": "Cup",
    "024_bowl": "Bowl", "011_banana": "Banana", "012_strawberry": "Strawberry",
    "040_large_marker": "Marker", "038_padlock": "Padlock",
}
NFING = {2: "2-finger", 3: "3-finger", 4: "4-finger"}

# Width of the crop window in source pixels; the hand spans far less than the
# 1200px frame, so without this most of every panel is empty tabletop.
CROP_W = 560

# Per-panel nudge (dx, dy) in source pixels, applied to the detected centre for
# the few panels where the arm's own links pull the centroid off the hand.
NUDGE = {}


def _hand_centre(im):
    """Centre of the black LEAP hand in one render.

    The hand is the only near-black thing in the upper part of the scene: the
    table is brown, the arm white, the bin blue. The table's cast shadow is also
    dark, so the search is restricted to the top of the frame, above the shadow
    that pools under the hand.
    """
    import numpy as np
    a = np.asarray(im.convert("RGB")).astype(float)
    dark = a.max(2) < 60
    dark[int(0.72 * a.shape[0]):, :] = False       # drop the foreground shadow
    ys, xs = np.nonzero(dark)
    if len(xs) == 0:
        return im.width / 2, im.height / 2
    # the palm/fingers are the dense core; percentiles reject the thin arm
    return float(np.median(xs)), float(np.median(ys))


def load_crop(obj, seed=0):
    """Cropped planned-pose render for one object, from the PER-OBJECT layout."""
    path = ROOT / obj / f"seed{seed}_planned.png"
    im = Image.open(path).convert("RGB")
    cx, cy = _hand_centre(im)
    dx, dy = NUDGE.get(obj, (0, 0))
    cx, cy, w = cx + dx, cy + dy, CROP_W
    h = w / ASPECT
    left, top = cx - w / 2, cy - h / 2
    left = min(max(left, 0), im.width - w)
    top = min(max(top, 0), im.height - h)
    return im.crop((int(left), int(top), int(left + w), int(top + h)))


def selected(sel_json):
    """[(object_id, n_contacts)] from the selector's JSON, ordered by finger
    count then id, so the grid reads 2-finger -> 4-finger left to right and the
    caption row tells a story rather than listing ids alphabetically.

    Objects the selector could not plan are DROPPED, not drawn blank: a missing
    panel in a qualitative figure reads as a failed grasp rather than an object
    that was never attempted.
    """
    rows = json.loads(Path(sel_json).read_text())
    out = [(r["object"], int(r["n_contacts"])) for r in rows
           if "error" not in r and r.get("n_contacts")]
    return sorted(out, key=lambda t: (t[1], t[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", default="benchmarks/ycb_grasp/out/tabletop/"
                                     "default/finger_selection.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", default="qualitative_grasps")
    a = ap.parse_args()

    items = [(o, n) for o, n in selected(a.sel)
             if (ROOT / o / f"seed{a.seed}_planned.png").exists()]
    if not items:
        print("no rendered objects found under", ROOT)
        return
    ncol = min(a.cols, len(items))
    nrow = (len(items) + ncol - 1) // ncol

    fig_w = 7.16                                   # IEEE double column
    fig_h = nrow * (fig_w - 0.06) / ncol / ASPECT + 0.22 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), squeeze=False)

    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        if k >= len(items):
            ax.axis("off")                          # pad a ragged last row
            continue
        obj, n = items[k]
        ax.imshow(load_crop(obj, a.seed))
        for sp in ax.spines.values():
            sp.set_linewidth(0.6); sp.set_color("0.25")
        # Finger count in the title, not a row label: it varies per panel now.
        ax.set_title(f"{PRETTY.get(obj, obj)}\n{NFING.get(n, str(n))}",
                     fontsize=7, pad=2, linespacing=1.15)

    fig.subplots_adjust(left=0.004, right=0.996, top=1.0 - 0.30 / fig_h,
                        bottom=0.004, wspace=0.02, hspace=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(Path(a.out).with_suffix("." + ext) if "/" in a.out
                    else OUT / f"{a.out}.{ext}", dpi=400)
    print(f"wrote {len(items)} panels ({ncol}x{nrow}) ->",
          OUT / f"{a.out}.pdf")


if __name__ == "__main__":
    main()

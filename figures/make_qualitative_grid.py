"""Qualitative grasp figure: 2 arms x 6 objects, IEEE double-column width.

Row = finger set, column = object, so the two-finger and three-finger grasp on
the same object sit one above the other under an identical camera.

Panels are auto-cropped around the hand with a constant aspect ratio. The source
renders are the planned-pose PNGs written by pick_and_place.py, which frame the
palm/object midpoint -- consistent enough that a detected crop beats twelve
hand-tuned windows.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pathlib import Path

ROOT = Path("benchmarks/ycb_grasp/out/tabletop/default")
OUT = Path("figures")

# Column = object, row = arm. Crop = (cx, cy, w) in source pixels; height
# follows from ASPECT. Windows were picked per panel to frame hand + object.
ASPECT = 4 / 3.0

OBJECTS = [
    ("014_lemon", "Lemon"),
    ("017_orange", "Orange"),
    ("056_tennis_ball", "Tennis ball"),
    ("036_wood_block", "Wood block"),
    ("009_gelatin_box", "Gelatin box"),
    ("025_mug", "Mug"),
]
ARMS = [
    ("two_finger", "Two-finger"),
    ("three_finger", "Three-finger"),
]

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


def load_crop(arm, obj):
    path = ROOT / arm / obj / "seed0_planned.png"
    im = Image.open(path).convert("RGB")
    cx, cy = _hand_centre(im)
    dx, dy = NUDGE.get((arm, obj), (0, 0))
    cx, cy, w = cx + dx, cy + dy, CROP_W
    h = w / ASPECT
    left, top = cx - w / 2, cy - h / 2
    # keep the window inside the frame
    left = min(max(left, 0), im.width - w)
    top = min(max(top, 0), im.height - h)
    return im.crop((int(left), int(top), int(left + w), int(top + h)))


def main():
    ncol, nrow = len(OBJECTS), len(ARMS)
    # IEEE double column = 7.16 in
    fig_w = 7.16
    panel_w = fig_w / ncol
    fig_h = nrow * (fig_w - 0.42) / ncol / ASPECT + 0.20
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h))

    for r, (arm, arm_label) in enumerate(ARMS):
        for c, (obj, obj_label) in enumerate(OBJECTS):
            ax = axes[r, c]
            ax.imshow(load_crop(arm, obj))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6); s.set_color("0.25")
            if r == 0:
                ax.set_title(obj_label, fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(arm_label, fontsize=8, labelpad=4)

    # Reserve real space for the column titles and the row labels; with six
    # columns the panels are short, so a fraction-based top margin that worked
    # at three columns clips the titles.
    title_in, label_in = 0.20, 0.42
    fig.subplots_adjust(left=label_in / fig_w, right=0.999,
                        top=1.0 - title_in / fig_h, bottom=0.004,
                        wspace=0.02, hspace=0.02)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"qualitative_grasps.{ext}", dpi=400)
    print("wrote", OUT / "qualitative_grasps.pdf")


if __name__ == "__main__":
    main()

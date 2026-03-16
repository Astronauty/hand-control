"""
grasp_benchmark.py
==================
USAGE
-----
Run from your project root (where models/ lives):

    python3 grasp_benchmark.py

Or override paths / grid resolution:

    python3 grasp_benchmark.py --model models/planar_two_finger_manipulator.xml \
                                --out   results/grasp_benchmark.png \
                                --nx 3 --ny 3 --nc 3

Needs grasp_planner.py on the Python path (place it next to this script).

World convention (from XML)
---------------------------
  gravity = "0 -9.81 0"  →  Y is vertical,  -Y is DOWN toward ground
  Th_base_x  axis="1 0 0"  →  +X is right / -X is left   (lateral)
  Th_base_y  axis="0 1 0"  →  +Y is up    / -Y is DOWN toward object
  Object at pos="0 -0.4 0" →  sitting 0.4 m below hand home position

3-D initial-condition grid
--------------------------
  base_x ∈ linspace(-0.1,  0.1,  nx)  lateral offset  (m)
  base_y ∈ linspace(-0.38, -0.1, ny)  vertical height  (-0.38 ≈ obj level,
                                        -0.1 = well above)
  curl   ∈ linspace(-0.5, -1.5,  nc)  finger curl magnitude  (rad)

Total solves = nx × ny × nc  (default 3×3×3 = 27)

Output figures
--------------
  <out>_heatmaps.png   — per-base_x-slice heatmaps (6 metrics × nx slices)
  <out>_aggregate.png  — contact scatter, joint box plots, violations, timing
"""

from __future__ import annotations
import sys, os, time, logging, argparse
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── locate grasp_planner ──────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in [_here, "/mnt/user-data/outputs"]:
    if os.path.exists(os.path.join(_cand, "grasp_planner.py")):
        sys.path.insert(0, _cand)
        break
from grasp_planner import GraspConfig, GraspPlanner, _get_actuated_indices

import mujoco as mj

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/planar_two_finger_manipulator.xml")
ap.add_argument("--out",   default="grasp_benchmark.png",
                help="Base filename; _heatmaps / _aggregate suffixes added")
ap.add_argument("--nx", type=int, default=1, help="base_x grid points")
ap.add_argument("--ny", type=int, default=5, help="base_y grid points")
ap.add_argument("--nc", type=int, default=5, help="curl grid points")
args = ap.parse_args()

out_stem   = args.out.replace(".png", "")
out_heat   = out_stem + "_heatmaps.png"
out_aggr   = out_stem + "_aggregate.png"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bench")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"Loading: {args.model}")
model = mj.MjModel.from_xml_path(args.model)
data  = mj.MjData(model)
mj.mj_resetData(model, data)
mj.mj_resetDataKeyframe(model, data, 0)
mj.mj_forward(model, data)

cfg     = GraspConfig()
planner = GraspPlanner(model, data, cfg=cfg)
act_idx = _get_actuated_indices(model)
obj_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.obj_body)
obj_pos = data.xpos[obj_bid][:2].copy()   # (x, y) in world
obj_hx  = model.geom_size[planner._obj_gid][0]
obj_hy  = model.geom_size[planner._obj_gid][1]

log.info(f"Object centre : {np.round(obj_pos, 4)}  "
         f"(x={obj_pos[0]:.3f}, y={obj_pos[1]:.3f}  ← y<0 means below origin)")
log.info(f"Object extents: ±{obj_hx:.3f} m (X)  ±{obj_hy:.3f} m (Y)")

# ─────────────────────────────────────────────────────────────────────────────
# Grid definition
#
# base_y range: object is at y=-0.4. We approach from above, so the useful
# range is just above the object top face (-0.4+0.05 = -0.35) up to well
# above it (-0.1).  -0.38 gives the solver very little room (fingers fully
# extended barely reach), while -0.1 means the hand is far above and needs
# full curl to reach.
# ─────────────────────────────────────────────────────────────────────────────
BASE_X_VALS = np.linspace(-0.1,  0.1,  args.nx)   # lateral offset (m)
BASE_Y_VALS = np.linspace(-0.3, 0.1, args.ny)   # vertical: -0.38≈obj level → -0.1=high above
CURL_VALS   = np.linspace(-0.5, -1.5,  args.nc)   # finger curl (rad)

NX, NY, NC = len(BASE_X_VALS), len(BASE_Y_VALS), len(CURL_VALS)
N = NX * NY * NC
q_base = np.array([data.qpos[i] for i in act_idx])

log.info(f"\nGrid: base_x({NX}) × base_y({NY}) × curl({NC}) = {N} solves")
log.info(f"  base_x : {np.round(BASE_X_VALS,3)}  (lateral, +right/-left)")
log.info(f"  base_y : {np.round(BASE_Y_VALS,3)}  (vertical, more-neg=closer to obj)")
log.info(f"  curl   : {np.round(CURL_VALS,3)}  (finger curl, more-neg=more curled)\n")

# ─────────────────────────────────────────────────────────────────────────────
# Run all solves
# ─────────────────────────────────────────────────────────────────────────────
records: list[dict] = []

for k, (ix, iy, ic) in enumerate(product(range(NX), range(NY), range(NC))):
    bx   = BASE_X_VALS[ix]
    by   = BASE_Y_VALS[iy]
    curl = CURL_VALS[ic]

    q = q_base.copy()
    q[0] =  bx
    q[1] =  by
    q[2] =  curl;  q[3] = curl;  q[4] = curl
    q[5] = -curl;  q[6] = -curl; q[7] = -curl

    log.info(f"  [{k+1:2d}/{N}]  bx={bx:+.2f}  by={by:.2f}  curl={curl:.2f}")
    t0 = time.perf_counter()

    try:
        result = planner.solve(q, obj_pos)
    except Exception as e:
        log.warning(f"    solve() raised: {e}")
        result = dict(success=False, q=None, p1=None, p2=None,
                      cost=None, iterations=None, status='failed')

    elapsed = time.perf_counter() - t0

    # post-solve metrics
    ik_t = ik_i = gap_t = gap_i = sdf1 = sdf2 = np.nan
    if result['q'] is not None:
        info = planner.verify(result)
        ik_t  = info['ik_thumb_mm']
        ik_i  = info['ik_index_mm']
        gap_t = info['gap_thumb_mm']
        gap_i = info['gap_index_mm']
        sdf1  = info['sdf_p1_mm']
        sdf2  = info['sdf_p2_mm']

    rec = dict(
        k=k, ix=ix, iy=iy, ic=ic,
        base_x=bx, base_y=by, curl=curl,
        status=result['status'], success=result['success'],
        cost=result['cost'], iterations=result['iterations'],
        elapsed_s=elapsed,
        q_final=result['q'], p1=result['p1'], p2=result['p2'],
        ik_thumb_mm=ik_t,  ik_index_mm=ik_i,
        gap_thumb_mm=gap_t, gap_index_mm=gap_i,
        sdf_p1_mm=sdf1,    sdf_p2_mm=sdf2,
        ik_err_mm  = (abs(ik_t)+abs(ik_i))/2 if not np.isnan(ik_t) else np.nan,
        sdf_err_mm = max(abs(sdf1),abs(sdf2)) if not np.isnan(sdf1) else np.nan,
        gap_min_mm = min(gap_t, gap_i)         if not np.isnan(gap_t) else np.nan,
    )
    records.append(rec)

    log.info(f"    {rec['status']:12s}  iters={rec['iterations']}  "
             f"IK=({ik_t:.2f},{ik_i:.2f})mm  "
             f"SDF=({sdf1:.2f},{sdf2:.2f})mm  "
             f"GAP=({gap_t:.2f},{gap_i:.2f})mm  "
             f"t={elapsed:.1f}s")

log.info(f"\nAll {N} solves complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
STATUS_COLOR = {'converged': '#2ecc71', 'best-effort': '#f39c12', 'failed': '#e74c3c'}

def slice_grid(ix_val: int, key: str, fill=np.nan) -> np.ndarray:
    """(NY × NC) array for one base_x slice."""
    arr = np.full((NY, NC), fill, dtype=float)
    for r in records:
        if r['ix'] != ix_val:
            continue
        v = r[key]
        arr[r['iy'], r['ic']] = (
            v if (v is not None and not (isinstance(v, float) and np.isnan(v)))
            else fill
        )
    return arr


def outcome_slice(ix_val: int) -> np.ndarray:
    """0=converged  1=best-effort  2=failed"""
    arr = np.full((NY, NC), 2.0)
    for r in records:
        if r['ix'] != ix_val:
            continue
        arr[r['iy'], r['ic']] = (
            0 if r['status'] == 'converged' else
            1 if r['status'] == 'best-effort' else 2
        )
    return arr


def annotate(ax, arr, fmt="{:.2f}", nan_str="N/A", fontsize=8):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = nan_str if np.isnan(v) else fmt.format(v)
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=fontsize, color='white', fontweight='bold')


def style_slice_ax(ax, ylabel=True):
    ax.set_xticks(range(NC))
    ax.set_xticklabels([f"{v:.2f}" for v in CURL_VALS], fontsize=7)
    ax.set_xlabel("curl (rad)", fontsize=7)
    if ylabel:
        ax.set_yticks(range(NY))
        ax.set_yticklabels([f"{v:.2f}" for v in BASE_Y_VALS], fontsize=7)
        ax.set_ylabel("base_y (m)\n← more negative = closer to obj", fontsize=7)
    else:
        ax.set_yticks(range(NY))
        ax.set_yticklabels([])


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Heatmaps  (6 metrics × NX base_x slices)
# Layout: rows = metrics (6), cols = base_x slices (NX)
# ─────────────────────────────────────────────────────────────────────────────
METRICS = [
    ("outcome",    "Outcome",                        None,       None,   None,    "auto"),
    ("cost",       "Cost",                           "plasma",   None,   None,    "auto"),
    ("ik_err_mm",  "IK Error  avg(|dθ|,|dι|) [mm]", "RdYlGn_r", 0,     None,    "auto"),
    ("sdf_err_mm", "SDF Error  max(|s1|,|s2|) [mm]","RdYlGn_r", 0,     None,    "auto"),
    ("gap_min_mm", "Min Gap  (+ clear / − penet) [mm]","RdYlGn", None,  None,    "auto"),
    ("iterations", "IPOPT Iterations",               "YlOrRd",   None,  None,    "auto"),
]

fig1_h = max(10, 2.8 * len(METRICS))
fig1_w = max(14, 4.5 * NX)
fig1, axes1 = plt.subplots(
    len(METRICS), NX,
    figsize=(fig1_w, fig1_h),
    squeeze=False
)

fig1.suptitle(
    "GraspPlanner — Heatmaps per base_x slice\n"
    f"Rows of each heatmap = base_y  |  Cols = curl  |  "
    f"Object @ y={obj_pos[1]:.3f} m  (gravity −Y, so −y = down toward object)",
    fontsize=11, fontweight='bold', y=1.01
)

for row, (metric, title, cmap, vmin, vmax, asp) in enumerate(METRICS):
    for col in range(NX):
        ax = axes1[row, col]

        if metric == "outcome":
            arr  = outcome_slice(col)
            cmap_use = matplotlib.colors.ListedColormap(
                ['#2ecc71', '#f39c12', '#e74c3c'])
            im = ax.imshow(arr, cmap=cmap_use, vmin=0, vmax=2, aspect=asp)
            olabels = {0:'conv', 1:'best\neff', 2:'fail'}
            for i in range(NY):
                for j in range(NC):
                    ax.text(j, i, olabels[int(arr[i,j])],
                            ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')
            if row == len(METRICS)-1 and col == 0:
                legend_els = [
                    Patch(facecolor='#2ecc71', label='Converged'),
                    Patch(facecolor='#f39c12', label='Best-effort'),
                    Patch(facecolor='#e74c3c', label='Failed'),
                ]
                ax.legend(handles=legend_els, fontsize=6,
                          loc='lower center', bbox_to_anchor=(0.5, -0.7),
                          ncol=1, frameon=False)
        else:
            fill = 0 if metric == "iterations" else np.nan
            arr  = slice_grid(col, metric, fill=fill)

            # symmetric colormap for gap
            if metric == "gap_min_mm":
                vabs = np.nanmax(np.abs(arr)) if not np.all(np.isnan(arr)) else 1.0
                vmin_, vmax_ = -vabs, vabs
            else:
                vmin_ = vmin if vmin is not None else np.nanmin(arr) if not np.all(np.isnan(arr)) else 0
                vmax_ = vmax if vmax is not None else np.nanmax(arr) if not np.all(np.isnan(arr)) else 1

            im = ax.imshow(arr, cmap=cmap, vmin=vmin_, vmax=vmax_, aspect=asp)
            fmt = "{:.0f}" if metric == "iterations" else (
                  "{:.4f}" if metric == "sdf_err_mm" else "{:.2f}")
            mask = (metric != "iterations")
            annotate(ax, arr, fmt=fmt, nan_str="N/A" if mask else "0")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # column header: show base_x value only on top row
        if row == 0:
            direction = "right" if BASE_X_VALS[col] > 0 else ("left" if BASE_X_VALS[col] < 0 else "centre")
            ax.set_title(
                f"base_x = {BASE_X_VALS[col]:+.2f} m\n({direction})\n{title}",
                fontsize=8, fontweight='bold'
            )
        else:
            ax.set_title(title, fontsize=8, fontweight='bold')

        style_slice_ax(ax, ylabel=(col == 0))

fig1.tight_layout()
fig1.savefig(out_heat, dpi=150, bbox_inches='tight')
plt.close(fig1)
log.info(f"Heatmaps saved → {out_heat}")


# ─────────────────────────────────────────────────────────────────────────────
# Pair clustering helper
#
# Two (p1, p2) pairs are considered the "same grasp" if both thumb contact
# points are within `tol` of each other AND both index contact points are
# within `tol`.  We do a simple greedy merge rather than sklearn to keep
# the dependency list minimal.
# ─────────────────────────────────────────────────────────────────────────────
def cluster_pairs(valid_records: list[dict], tol: float = 5e-3):
    """
    Group valid (p1, p2) pairs by proximity.

    Returns a list of cluster dicts:
        p1_mean   : (2,) mean thumb contact position
        p2_mean   : (2,) mean index contact position
        count     : int  number of solves in this cluster
        mean_ik   : float  mean IK error across members (mm)
        members   : list of record dicts
        angle_deg : float  grasp axis angle (degrees from +X axis)
        span_m    : float  distance |p1 - p2| (m)
    """
    clusters: list[dict] = []
    for r in valid_records:
        p1, p2 = np.array(r['p1']), np.array(r['p2'])
        placed = False
        for cl in clusters:
            if (np.linalg.norm(p1 - cl['p1_mean']) < tol and
                    np.linalg.norm(p2 - cl['p2_mean']) < tol):
                cl['members'].append(r)
                n = len(cl['members'])
                # incremental mean update
                cl['p1_mean'] = cl['p1_mean'] + (p1 - cl['p1_mean']) / n
                cl['p2_mean'] = cl['p2_mean'] + (p2 - cl['p2_mean']) / n
                placed = True
                break
        if not placed:
            clusters.append(dict(
                p1_mean=p1.copy(), p2_mean=p2.copy(),
                members=[r],
            ))

    # compute derived fields
    for cl in clusters:
        cl['count']   = len(cl['members'])
        cl['mean_ik'] = float(np.mean([r['ik_err_mm'] for r in cl['members']]))
        dx = cl['p2_mean'][0] - cl['p1_mean'][0]
        dy = cl['p2_mean'][1] - cl['p1_mean'][1]
        cl['angle_deg'] = float(np.degrees(np.arctan2(dy, dx)))
        cl['span_m']    = float(np.linalg.norm(cl['p2_mean'] - cl['p1_mean']))

    clusters.sort(key=lambda c: -c['count'])   # most frequent first
    return clusters


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Aggregate diagnostics  (all N runs pooled)
# Layout: 4 rows × 2 cols
#   [7a paired grasp scatter]  [7b angle histogram + pair table]
#   [8  joint box plots — full width]
#   [9  constraint violations — full width]
#   [10 solve time — full width]
# ─────────────────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(22, 22))
fig2.suptitle(
    f"GraspPlanner — Aggregate Diagnostics  ({N} initial conditions: "
    f"base_x({NX}) × base_y({NY}) × curl({NC}))\n"
    f"Object @ (x={obj_pos[0]:.3f}, y={obj_pos[1]:.3f}) m  "
    f"[Y axis: +up / −down toward ground]",
    fontsize=12, fontweight='bold', y=1.00
)

gs2 = gridspec.GridSpec(4, 2, figure=fig2, hspace=0.60, wspace=0.35,
                         height_ratios=[1.4, 0.9, 1.2, 1.0])

# ── 7a. Paired contact-point visualisation ───────────────────────────────────
ax7a = fig2.add_subplot(gs2[0, 0])

# object face
rect7 = patches.Rectangle(
    (obj_pos[0] - obj_hx, obj_pos[1] - obj_hy),
    2*obj_hx, 2*obj_hy,
    linewidth=2, edgecolor='black', facecolor='#ecf0f1', zorder=1
)
ax7a.add_patch(rect7)
ax7a.plot(*obj_pos, 'k+', markersize=12, zorder=2)

valid = [r for r in records if r['p1'] is not None]
clusters = cluster_pairs(valid, tol=5e-3)

if clusters:
    ik_vals  = np.array([c['mean_ik'] for c in clusters])
    counts   = np.array([c['count']   for c in clusters])
    norm7    = Normalize(vmin=0, vmax=max(ik_vals.max(), 0.1))
    cmap7    = plt.cm.RdYlGn_r

    # line-width and marker-size scale with frequency (sqrt for visual balance)
    max_count = counts.max()

    for ci, cl in enumerate(clusters):
        freq_scale = np.sqrt(cl['count'] / max_count)   # 0..1
        lw   = 1.0 + 5.0 * freq_scale          # 1 px (rare) → 6 px (most frequent)
        ms   = 40  + 160 * freq_scale           # small → large marker

        col  = cmap7(norm7(cl['mean_ik']))
        p1m, p2m = cl['p1_mean'], cl['p2_mean']
        mid  = (p1m + p2m) / 2

        # grasp axis line
        ax7a.plot([p1m[0], p2m[0]], [p1m[1], p2m[1]],
                  color=col, linewidth=lw, alpha=0.85, zorder=3,
                  solid_capstyle='round')

        # thumb (circle) and index (triangle) markers
        ax7a.scatter(*p1m, color=col, marker='o', s=ms, zorder=4,
                     edgecolors='black', linewidths=0.8)
        ax7a.scatter(*p2m, color=col, marker='^', s=ms, zorder=4,
                     edgecolors='black', linewidths=0.8)

        # frequency badge at midpoint
        ax7a.text(mid[0], mid[1], str(cl['count']),
                  ha='center', va='center', fontsize=7,
                  fontweight='bold', color='white',
                  bbox=dict(boxstyle='round,pad=0.15', facecolor=col,
                            edgecolor='black', linewidth=0.5),
                  zorder=5)

    # colourbar
    sm7 = ScalarMappable(cmap=cmap7, norm=norm7)
    sm7.set_array([])
    plt.colorbar(sm7, ax=ax7a, fraction=0.046, pad=0.04,
                 label='Mean IK error (mm)')

    # legend for marker shapes
    ax7a.scatter([], [], color='grey', marker='o', s=60,
                 label='Thumb (p1)', edgecolors='k')
    ax7a.scatter([], [], color='grey', marker='^', s=60,
                 label='Index (p2)', edgecolors='k')
    ax7a.plot([], [], color='grey', linewidth=3,
              label='Grasp axis\n(width ∝ frequency)')
    ax7a.legend(fontsize=7, loc='upper right', framealpha=0.9)

mg = 0.06
ax7a.set_aspect('equal')
ax7a.set_xlim(obj_pos[0]-obj_hx-mg, obj_pos[0]+obj_hx+mg)
ax7a.set_ylim(obj_pos[1]-obj_hy-mg, obj_pos[1]+obj_hy+mg)
ax7a.set_xlabel("X (m) — lateral", fontsize=8)
ax7a.set_ylabel("Y (m) — vertical (+up/−down)", fontsize=8)
ax7a.set_title(
    f"7a. Paired Grasp Contact Points  (○=thumb  △=index)\n"
    f"Line width & marker size ∝ frequency  |  Badge = # solves in cluster\n"
    f"{len(valid)} valid solves → {len(clusters)} unique pairs  "
    f"(cluster tol = 5 mm)",
    fontsize=9, fontweight='bold'
)
ax7a.grid(True, alpha=0.3)

# ── 7b. Angle histogram + pair frequency table ───────────────────────────────
ax7b = fig2.add_subplot(gs2[0, 1])

if clusters:
    angles  = [c['angle_deg'] for c in clusters]
    weights = [c['count']     for c in clusters]   # weight by frequency

    # histogram of grasp-axis angles, frequency-weighted
    bins = np.linspace(-180, 180, 25)
    n_hist, bin_edges = np.histogram(angles, bins=bins, weights=weights)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_colors_h = [cmap7(norm7(clusters[
                        min(range(len(clusters)),
                            key=lambda i: abs(clusters[i]['angle_deg'] - bc))
                    ]['mean_ik']))
                    for bc in bin_centres]

    ax7b.bar(bin_centres, n_hist, width=np.diff(bin_edges),
             color=bar_colors_h, alpha=0.80, edgecolor='white', linewidth=0.4)

    # reference lines for cardinal directions
    for ang, lbl in [(0, '+X\n(right)'), (90, '+Y\n(up)'),
                     (-90, '−Y\n(down)'), (180, '−X\n(left)'),
                     (-180, '')]:
        ax7b.axvline(ang, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
        if lbl:
            ax7b.text(ang, ax7b.get_ylim()[1] if ax7b.get_ylim()[1] > 0 else 1,
                      lbl, ha='center', va='bottom', fontsize=6.5, color='gray')

    ax7b.set_xlabel("Grasp axis angle  (degrees from +X)", fontsize=8)
    ax7b.set_ylabel("Frequency-weighted count", fontsize=8)
    ax7b.set_xlim(-180, 180)
    ax7b.set_xticks(np.arange(-180, 181, 45))

    # ── pair frequency table (inset below histogram) ─────────────────────────
    # top-8 pairs by frequency
    top_n = min(8, len(clusters))
    col_labels = ['Rank', 'Count', '%', 'p1 (x,y) m', 'p2 (x,y) m',
                  'Span mm', 'Angle °', 'IK err mm']
    table_data = []
    total_valid = len(valid)
    for rank, cl in enumerate(clusters[:top_n], start=1):
        table_data.append([
            str(rank),
            str(cl['count']),
            f"{100*cl['count']/total_valid:.0f}%",
            f"({cl['p1_mean'][0]:.3f}, {cl['p1_mean'][1]:.3f})",
            f"({cl['p2_mean'][0]:.3f}, {cl['p2_mean'][1]:.3f})",
            f"{cl['span_m']*1000:.1f}",
            f"{cl['angle_deg']:.1f}",
            f"{cl['mean_ik']:.3f}",
        ])

    tbl = ax7b.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='bottom',
        bbox=[0.0, -0.90, 1.0, 0.80],   # [left, bottom, width, height] in axes coords
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')
        else:
            cell.set_facecolor('white')

    ax7b.set_title(
        "7b. Grasp Axis Angle Distribution\n"
        "(bar height = frequency-weighted count)\n"
        f"Top-{top_n} pairs by frequency (table below)",
        fontsize=9, fontweight='bold'
    )
else:
    ax7b.text(0.5, 0.5, 'No valid pairs', transform=ax7b.transAxes,
              ha='center', va='center', fontsize=12)
    ax7b.set_title("7b. Grasp Axis Angle Distribution", fontsize=9, fontweight='bold')

# ── 8. Joint angle box plots  (row 1, full width) ────────────────────────────
ax8 = fig2.add_subplot(gs2[1, :])

all_q = [r['q_final'] for r in records if r['q_final'] is not None]
if all_q:
    Q  = np.array(all_q)
    nu = Q.shape[1]
    bp = ax8.boxplot(Q, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db'); patch.set_alpha(0.6)
    jnames = ['base_x', 'base_y',
              'idx_mcp', 'idx_pip', 'idx_dip',
              'th_mcp',  'th_pip',  'th_dip'][:nu]
    ax8.set_xticks(range(1, nu+1))
    ax8.set_xticklabels(jnames, rotation=35, ha='right', fontsize=8)
    ax8.set_ylabel("Joint angle (rad)", fontsize=8)
    ax8.set_title(f"8. Final Joint Angle Distribution  ({len(all_q)} valid solves)",
                  fontsize=9, fontweight='bold')
    ax8.grid(True, axis='y', alpha=0.3)
    bx_inits = [r['base_x'] for r in records]
    by_inits = [r['base_y'] for r in records]
    ax8.axhline(np.mean(bx_inits), color='red',    linestyle=':', alpha=0.5,
                linewidth=1.2, label=f'mean init base_x = {np.mean(bx_inits):.2f}')
    ax8.axhline(np.mean(by_inits), color='orange', linestyle=':', alpha=0.5,
                linewidth=1.2, label=f'mean init base_y = {np.mean(by_inits):.2f}')
    ax8.legend(fontsize=7, loc='upper right')
else:
    ax8.text(0.5, 0.5, 'No valid solves', transform=ax8.transAxes,
             ha='center', va='center', fontsize=12)
    ax8.set_title("8. Final Joint Angles", fontsize=9, fontweight='bold')

# ── 9. Constraint violations  (row 2, full width) ────────────────────────────
ax9  = fig2.add_subplot(gs2[2, :])

bw   = 0.16
xpos = np.arange(N)
slice_size = NY * NC

def safe(vals):
    return [v if not np.isnan(v) else 0.0 for v in vals]

ik_t_v  = safe([r['ik_thumb_mm']  for r in records])
ik_i_v  = safe([r['ik_index_mm']  for r in records])
sdf1_v  = safe([r['sdf_p1_mm']    for r in records])
sdf2_v  = safe([r['sdf_p2_mm']    for r in records])
gap_t_v = [r['gap_thumb_mm'] for r in records]
gap_i_v = [r['gap_index_mm'] for r in records]

ax9.bar(xpos - 1.5*bw, ik_t_v,               width=bw, label='|IK thumb| (mm)',  color='#3498db', alpha=0.85)
ax9.bar(xpos - 0.5*bw, ik_i_v,               width=bw, label='|IK index| (mm)',  color='#2980b9', alpha=0.85)
ax9.bar(xpos + 0.5*bw, [abs(v) for v in sdf1_v], width=bw, label='|SDF p1| (mm)', color='#e67e22', alpha=0.85)
ax9.bar(xpos + 1.5*bw, [abs(v) for v in sdf2_v], width=bw, label='|SDF p2| (mm)', color='#d35400', alpha=0.85)

ax9b = ax9.twinx()
ax9b.plot(xpos, safe(gap_t_v), 'v--', color='#8e44ad', alpha=0.8,
          label='gap thumb (mm)', markersize=4)
ax9b.plot(xpos, safe(gap_i_v), '^--', color='#9b59b6', alpha=0.8,
          label='gap index (mm)', markersize=4)
ax9b.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax9b.set_ylabel("Gap (mm)  [+ = clear  |  − = penetrating]", fontsize=8)
ax9b.tick_params(axis='y', labelsize=7)

for ix in range(NX):
    ax9.axvline(ix * slice_size - 0.5, color='black', linewidth=1.2,
                linestyle='--', alpha=0.4)
    mid = ix * slice_size + slice_size / 2 - 0.5
    ax9.text(mid, ax9.get_ylim()[1] if ax9.get_ylim()[1] > 0 else 1,
             f"bx={BASE_X_VALS[ix]:+.2f}", ha='center', va='bottom',
             fontsize=8, fontweight='bold')

ax9.set_xticks(xpos)
ax9.set_xticklabels([f"#{r['k']+1}" for r in records], fontsize=5, rotation=45)
ax9.set_ylabel("Constraint violation magnitude (mm)", fontsize=8)
ax9.set_title("9. Per-Solve Constraint Violations  "
              "(dashed lines separate base_x slices)",
              fontsize=9, fontweight='bold')
ax9.grid(True, axis='y', alpha=0.3)
l1, b1 = ax9.get_legend_handles_labels()
l2, b2 = ax9b.get_legend_handles_labels()
ax9.legend(l1+l2, b1+b2, fontsize=7, loc='upper right', ncol=3, frameon=True)

# ── 10. Solve time  (row 3, full width) ──────────────────────────────────────
ax10 = fig2.add_subplot(gs2[3, :])

bar_colors = [STATUS_COLOR[r['status']] for r in records]
ax10.bar(xpos, [r['elapsed_s'] for r in records],
         color=bar_colors, alpha=0.85, width=0.7)

max_t = max(r['elapsed_s'] for r in records)
for ix in range(NX):
    ax10.axvline(ix * slice_size - 0.5, color='black', linewidth=1.2,
                 linestyle='--', alpha=0.4)
    mid = ix * slice_size + slice_size / 2 - 0.5
    ax10.text(mid, max_t * 1.05, f"bx={BASE_X_VALS[ix]:+.2f}",
              ha='center', va='bottom', fontsize=8, fontweight='bold')

ax10.set_xticks(xpos)
ax10.set_xticklabels([f"#{r['k']+1}\n{r['status'][:4]}" for r in records],
                     fontsize=5, rotation=45)
ax10.set_ylabel("Wall-clock time (s)", fontsize=8)
ax10.set_title("10. Solve Time per Initial Condition", fontsize=9, fontweight='bold')
ax10.grid(True, axis='y', alpha=0.3)
ax10.legend(handles=[
    Patch(facecolor='#2ecc71', label='Converged'),
    Patch(facecolor='#f39c12', label='Best-effort'),
    Patch(facecolor='#e74c3c', label='Failed'),
], fontsize=8, loc='upper left')

# ── Summary footer ────────────────────────────────────────────────────────────
n_conv  = sum(1 for r in records if r['status'] == 'converged')
n_best  = sum(1 for r in records if r['status'] == 'best-effort')
n_fail  = sum(1 for r in records if r['status'] == 'failed')
v_recs  = [r for r in records if r['q_final'] is not None]
mean_ik  = np.nanmean([r['ik_err_mm']  for r in v_recs]) if v_recs else float('nan')
mean_sdf = np.nanmean([r['sdf_err_mm'] for r in v_recs]) if v_recs else float('nan')
mean_t   = np.mean([r['elapsed_s'] for r in records])
total_t  = sum(r['elapsed_s'] for r in records)

fig2.text(0.5, -0.005,
    f"Converged: {n_conv}/{N}   Best-effort: {n_best}/{N}   Failed: {n_fail}/{N}   "
    f"|   Mean IK error: {mean_ik:.3f} mm   Mean SDF error: {mean_sdf:.4f} mm   "
    f"|   Mean solve time: {mean_t:.1f} s   Total: {total_t:.0f} s",
    ha='center', va='bottom', fontsize=9, style='italic',
    bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8)
)

fig2.tight_layout()
fig2.savefig(out_aggr, dpi=150, bbox_inches='tight')
plt.close(fig2)
log.info(f"Aggregate saved  → {out_aggr}")
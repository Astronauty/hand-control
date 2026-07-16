#!/usr/bin/env python3
"""
test_grasp_robustness.py
========================
Headless batch robustness test for MultiStartGraspPlanner3D.

Sweeps (one-at-a-time; all other params held at nominal):
  1. Object XY position grid    : 5×5 = 25 points spanning the reachable workspace
  2. Object yaw angle           : 0 … 60°
  3. Box half-extent (uniform)  : 0.020 … 0.040 m
  4. Friction coefficient μ     : 0.3 … 1.5

Outputs (in logs/<prefix>_<ts>/):
  results.csv          — one row per test case
  results.json         — full candidate dicts per test
  plots/               — convergence, IK error, gamma, solve-time, contact scatter

Usage
-----
  python test_grasp_robustness.py --nc 6 --max-iter 150 --log-prefix robustness_v1
"""

import argparse
import csv
import json
import logging
import os, sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import mujoco as mj

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))                   # for grasp_control
if str(_REPO_ROOT / 'simulation') not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / 'simulation')) 

from grasp_planner_3d import GraspConfig3D, MultiStartGraspPlanner3D

# ── Constants ─────────────────────────────────────────────────────────────────

GEN3_XML = 'mujoco_menagerie/kinova_gen3/gen3.xml'
SCENE_XML = 'models/scene_pick_place.xml'
N_ROBOT  = 23

# Nominal object parameters (match obj_red_box in scene_pick_place.xml)
NOM_POS  = np.array([0.38, 0.42, 0.040])
NOM_YAW  = 0.0
NOM_SIZE = 0.040
NOM_MU   = 4.0

# ── Sweeps definition ─────────────────────────────────────────────────────────

SWEEPS = [
    {
        'name':   'position',
        'label':  'Object XY position (m)',
        'values': [(x, y) for x in [0.20, 0.30, 0.40, 0.50, 0.60]
                           for y in [-0.05, 0.07, 0.22, 0.35, 0.48]],
        'fmt':    None,   # tuple — handled specially in the loop
    },
    {
        'name':   'yaw',
        'label':  'Yaw angle (deg)',
        'values': [0.0, 15.0, 30.0, 45.0, 60.0],
        'fmt':    '{:.0f}°',
    },
    {
        'name':   'size',
        'label':  'Box half-extent (m)',
        'values': [0.020, 0.025, 0.030, 0.035, 0.040],
        'fmt':    '{:.3f}',
    },
    {
        'name':   'mu',
        'label':  'Friction coefficient μ',
        'values': [0.3, 0.5, 0.7, 1.0, 1.5],
        'fmt':    '{:.1f}',
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_q_bias() -> np.ndarray:
    gen3 = mj.MjModel.from_xml_path(GEN3_XML)
    home = gen3.key('home').qpos[:7].copy()
    q = np.zeros(N_ROBOT)
    q[:7]    = home
    q[11:15] = [1.2, 0.0, 0.5, 0.5]   # mf: curl out of the way
    q[15:19] = [1.2, 0.0, 0.5, 0.5]   # rf: curl out of the way
    return q


def yaw_to_quat(yaw_deg: float) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) for a Z-axis rotation."""
    a = np.deg2rad(yaw_deg) / 2.0
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def set_object_pose(data, qpos_adr: int, pos: np.ndarray, quat: np.ndarray):
    data.qpos[qpos_adr:qpos_adr+3] = pos
    data.qpos[qpos_adr+3:qpos_adr+7] = quat


def set_object_size(model, ms_planner, obj_gid: int, hs: float):
    """Update geom size in the model AND the cached values in the planner."""
    model.geom_size[obj_gid] = [hs, hs, hs]
    p = ms_planner._planner
    p._obj_hx = hs
    p._obj_hy = hs
    p._obj_hz = hs
    p._obj_size[:] = hs
    ms_planner._obj_hx = hs
    ms_planner._obj_hy = hs
    ms_planner._obj_hz = hs
    ms_planner._obj_size[:] = hs


def set_friction(model, ms_planner, obj_gid: int, mu: float):
    """Update MuJoCo sliding friction AND the planner cfg."""
    model.geom_friction[obj_gid, 0] = mu
    ms_planner._planner.cfg.mu = mu


def nominal_params():
    return dict(pos=NOM_POS.copy(), yaw_deg=NOM_YAW, size=NOM_SIZE, mu=NOM_MU)


def build_test_case(sweep_name: str, value) -> dict:
    p = nominal_params()
    if sweep_name == 'position':
        x, y = value
        p['pos'] = np.array([x, y, NOM_POS[2]])
    elif sweep_name == 'yaw':
        p['yaw_deg'] = value
    elif sweep_name == 'size':
        p['size'] = value
    elif sweep_name == 'mu':
        p['mu'] = value
    return p


# ── Single test run ───────────────────────────────────────────────────────────

def run_one(model, data, ms_planner, q_bias, qpos_adr, obj_gid,
            pos, yaw_deg, size, mu, nc, max_iter, log) -> dict:
    """Configure one test case and return a metrics dict."""
    # Reset arm to home
    data.qpos[:N_ROBOT] = q_bias
    data.qvel[:] = 0.0

    # Apply test parameters
    set_object_size(model, ms_planner, obj_gid, size)
    set_friction(model, ms_planner, obj_gid, mu)
    set_object_pose(data, qpos_adr, pos, yaw_to_quat(yaw_deg))
    mj.mj_forward(model, data)

    obj_pos = data.geom_xpos[obj_gid].copy()

    t0 = time.perf_counter()
    try:
        result = ms_planner.solve(q_bias, obj_pos, max_seeds=nc)
    except Exception as exc:
        log.warning(f"  [EXCEPTION] {exc}")
        return {'status': 'error', 'error': str(exc),
                'solve_time_ms': round((time.perf_counter() - t0) * 1000, 1)}
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    all_cands = result.get('all_results', [result])
    best = result   # already sorted by MultiStartGraspPlanner3D

    vinfo = {}
    if best.get('q') is not None:
        try:
            vinfo = ms_planner._planner.verify(best) or {}
        except Exception as exc:
            log.warning(f"  [verify exception] {exc}")

    metrics = {
        'status':          best.get('status', 'failed'),
        'solve_time_ms':   elapsed_ms,
        'iterations':      best.get('iterations', -1),
        'cost':            best.get('cost'),
        'ik_thumb_mm':     vinfo.get('ik_thumb_mm'),
        'ik_index_mm':     vinfo.get('ik_index_mm'),
        'gap_thumb_mm':    vinfo.get('gap_thumb_mm'),
        'gap_index_mm':    vinfo.get('gap_index_mm'),
        'sdf_p1_mm':       vinfo.get('sdf_p1_mm'),
        'sdf_p2_mm':       vinfo.get('sdf_p2_mm'),
        'gamma_min':       vinfo.get('gamma_min'),
        'wrench_feasible': vinfo.get('wrench_feasible', False),
        'n_converged':     sum(1 for r in all_cands
                               if r.get('status') == 'converged'),
        'n_seeds_tried':   len(all_cands),
    }
    p1 = best.get('p1')
    p2 = best.get('p2')
    if p1 is not None:
        metrics.update(p1_x=float(p1[0]), p1_y=float(p1[1]), p1_z=float(p1[2]))
    if p2 is not None:
        metrics.update(p2_x=float(p2[0]), p2_y=float(p2[1]), p2_z=float(p2[2]))

    return metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

CMAP = plt.cm.viridis


def _fig_metric_vs_param(sweep_rows, sweep_meta, metric_key, ylabel, out_path,
                          hline=None):
    values  = [r['sweep_value'] for r in sweep_rows]
    metrics = [r.get(metric_key) for r in sweep_rows]
    valid   = [(v, m) for v, m in zip(values, metrics) if m is not None]
    if not valid:
        return

    xs, ys = zip(*valid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, 'o-', color='steelblue', linewidth=2, markersize=7)
    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', linewidth=1, alpha=0.6,
                   label=f'y={hline}')
        ax.legend(fontsize=9)
    ax.set_xlabel(sweep_meta['label'], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{ylabel} vs {sweep_meta['label']}", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_ik_vs_param(sweep_rows, sweep_meta, out_path):
    values  = [r['sweep_value'] for r in sweep_rows]
    ik_th   = [r.get('ik_thumb_mm') for r in sweep_rows]
    ik_if   = [r.get('ik_index_mm') for r in sweep_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    v_th = [(v, m) for v, m in zip(values, ik_th) if m is not None]
    v_if = [(v, m) for v, m in zip(values, ik_if) if m is not None]
    if v_th:
        xs, ys = zip(*v_th)
        ax.plot(xs, ys, 'o-', color='tomato', linewidth=2, markersize=7,
                label='Thumb IK (mm)')
    if v_if:
        xs, ys = zip(*v_if)
        ax.plot(xs, ys, 's--', color='steelblue', linewidth=2, markersize=7,
                label='Index IK (mm)')
    ax.set_xlabel(sweep_meta['label'], fontsize=11)
    ax.set_ylabel('IK error (mm)', fontsize=11)
    ax.set_title(f"IK error vs {sweep_meta['label']}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_position_grid(sweep_rows, out_path):
    """4-panel 2D scatter of key metrics across the 5×5 XY workspace grid."""
    from matplotlib.lines import Line2D

    xs = np.array([r['sweep_value'][0] for r in sweep_rows])
    ys = np.array([r['sweep_value'][1] for r in sweep_rows])

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle('Workspace coverage — 5x5 XY grid  (arm base at origin)',
                 fontsize=13, fontweight='bold')

    def _decorate(ax, title):
        ax.plot(0, 0, 'k^', markersize=11, zorder=6, clip_on=False)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('World X (m)', fontsize=10)
        ax.set_ylabel('World Y (m)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # ── Panel 1: Convergence ──────────────────────────────────────────────────
    ax = axes[0, 0]
    pt_colors = ['#2ecc71' if r.get('status') == 'converged' else '#e74c3c'
                 for r in sweep_rows]
    ax.scatter(xs, ys, c=pt_colors, s=150, edgecolors='white', linewidths=0.8, zorder=4)
    for r, x, y in zip(sweep_rows, xs, ys):
        lbl = 'OK' if r.get('status') == 'converged' else 'X'
        col = '#1e8449' if lbl == 'OK' else '#922b21'
        ax.annotate(lbl, (x, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=7, color=col)
    legend_els = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=10, label='Converged'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=10, label='Failed'),
        Line2D([0],[0], marker='^', color='k', markersize=10, label='Arm base'),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc='upper right')
    _decorate(ax, 'Convergence')

    # ── Panel 2: IK error ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    ik_vals = np.array([
        max(r.get('ik_thumb_mm') or 0, r.get('ik_index_mm') or 0) or np.nan
        for r in sweep_rows])
    sc = ax.scatter(xs, ys, c=ik_vals, cmap='viridis', s=150,
                    edgecolors='white', linewidths=0.8, zorder=4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('max(IK_thumb, IK_index) mm', fontsize=9)
    _decorate(ax, 'IK error (mm)')

    # ── Panel 3: gamma_min ────────────────────────────────────────────────────
    ax = axes[1, 0]
    gm_vals = np.array([
        r['gamma_min'] if r.get('gamma_min') is not None else np.nan
        for r in sweep_rows])
    sc = ax.scatter(xs, ys, c=gm_vals, cmap='plasma', s=150,
                    edgecolors='white', linewidths=0.8, zorder=4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('gamma_min', fontsize=9)
    _decorate(ax, 'Wrench quality (gamma_min)')

    # ── Panel 4: Solve time ───────────────────────────────────────────────────
    ax = axes[1, 1]
    t_vals = np.array([
        r['solve_time_ms'] if r.get('solve_time_ms') is not None else np.nan
        for r in sweep_rows])
    sc = ax.scatter(xs, ys, c=t_vals, cmap='coolwarm', s=150,
                    edgecolors='white', linewidths=0.8, zorder=4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Solve time (ms)', fontsize=9)
    _decorate(ax, 'Solve time (ms)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_position_contact_scatter(sweep_rows, out_path):
    """Workspace-level view: box outlines at each tested position + p1/p2 contacts."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_title('Contact points across workspace — position sweep', fontsize=12,
                 fontweight='bold')

    for r in sweep_rows:
        x, y = r['sweep_value']
        converged = r.get('status') == 'converged'
        box_ec = '#27ae60' if converged else '#c0392b'

        # Box outline at this test position
        hs = NOM_SIZE
        rect = plt.Rectangle((x - hs, y - hs), 2*hs, 2*hs,
                              fill=False, edgecolor=box_ec, linewidth=1.2, alpha=0.6)
        ax.add_patch(rect)

        if not converged:
            ax.plot(x, y, 'x', color='#c0392b', markersize=10, markeredgewidth=2,
                    zorder=4)
            continue

        p1x, p1y = r.get('p1_x'), r.get('p1_y')
        p2x, p2y = r.get('p2_x'), r.get('p2_y')
        if p1x is not None and p2x is not None:
            ax.plot([p1x, p2x], [p1y, p2y], '-', color='#95a5a6',
                    linewidth=0.9, alpha=0.6, zorder=2)
        if p1x is not None:
            ax.plot(p1x, p1y, 'o', color='#e74c3c', markersize=7, alpha=0.9,
                    markeredgecolor='#922b21', markeredgewidth=0.6, zorder=5)
        if p2x is not None:
            ax.plot(p2x, p2y, 's', color='#2980b9', markersize=7, alpha=0.9,
                    markeredgecolor='#1a5276', markeredgewidth=0.6, zorder=5)

    ax.plot(0, 0, 'k^', markersize=13, zorder=8, clip_on=False)

    legend_els = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=10, label='p1 — thumb contact'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#2980b9',
               markersize=10, label='p2 — index contact'),
        Patch(facecolor='none', edgecolor='#27ae60', linewidth=1.5,
              label='Box outline (converged)'),
        Patch(facecolor='none', edgecolor='#c0392b', linewidth=1.5,
              label='Box outline (failed)'),
        Line2D([0],[0], marker='^', color='k', markersize=11,
               label='Arm base'),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc='upper right')

    all_xs = [r['sweep_value'][0] for r in sweep_rows]
    all_ys = [r['sweep_value'][1] for r in sweep_rows]
    margin = 0.07
    ax.set_xlim(min(all_xs) - NOM_SIZE - margin, max(all_xs) + NOM_SIZE + margin)
    ax.set_ylim(min(all_ys) - NOM_SIZE - margin, max(all_ys) + NOM_SIZE + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('World X (m)', fontsize=11)
    ax.set_ylabel('World Y (m)', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_3d_contact_grid(all_rows, out_path):
    """
    2×2 grid of 3D subplots — one per sweep.
    Each subplot shows the box wireframe in its object-local frame and the
    thumb (circle) / index (square) contact points transformed into that frame,
    coloured by the sweep parameter.

    Local frame: origin at box centre, axes aligned to box faces.
    This removes position offsets and yaw rotations so contacts are always
    expressed as "where on THIS box did the finger land?"
    """
    from itertools import product as _iproduct

    def _box_edges(ax, hx, hy, hz, **kw):
        """Draw 12 wireframe edges of box [-hx..hx, -hy..hy, -hz..hz]."""
        corners = list(_iproduct((-hx, hx), (-hy, hy), (-hz, hz)))
        for i in range(8):
            for j in range(i + 1, 8):
                if sum(corners[i][k] != corners[j][k] for k in range(3)) == 1:
                    ax.plot3D([corners[i][0], corners[j][0]],
                              [corners[i][1], corners[j][1]],
                              [corners[i][2], corners[j][2]], **kw)

    def _row_pose(row):
        """Return (center_world, R_local_to_world) reconstructed from row metadata."""
        sname = row['sweep']
        sv    = row['sweep_value']
        if sname == 'position':
            center = np.array([sv[0], sv[1], NOM_POS[2]])
            yaw_deg = 0.0
        elif sname == 'yaw':
            center  = NOM_POS.copy()
            yaw_deg = float(sv)
        else:
            center  = NOM_POS.copy()
            yaw_deg = 0.0
        c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return center, R

    def _row_hs(row):
        return float(row['sweep_value']) if row['sweep'] == 'size' else NOM_SIZE

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle('Contact points in object-local frame — 3D (all sweeps)',
                 fontsize=13, fontweight='bold')

    cmaps = {'position': 'plasma', 'yaw': 'viridis',
             'size': 'coolwarm', 'mu': 'RdYlGn'}

    for si, sweep in enumerate(SWEEPS):
        sname  = sweep['name']
        ax = fig.add_subplot(2, 2, si + 1, projection='3d')
        rows = [r for r in all_rows if r['sweep'] == sname]

        # Build scalar colour values
        if sname == 'position':
            scalars = [np.hypot(r['sweep_value'][0], r['sweep_value'][1]) for r in rows]
            clabel  = 'Distance from arm base (m)'
        else:
            scalars = [float(r['sweep_value']) for r in rows]
            clabel  = sweep['label']

        vmin, vmax = min(scalars), max(scalars)
        if abs(vmax - vmin) < 1e-9:
            vmin -= 0.5; vmax += 0.5
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmaps.get(sname, 'viridis'))

        # Draw box wireframe(s) — one per unique size, centered at local origin
        drawn = set()
        for r in rows:
            hs = _row_hs(r)
            if hs not in drawn:
                _box_edges(ax, hs, hs, hs, color='#2c3e50', alpha=0.20,
                           linewidth=0.8, linestyle='--')
                drawn.add(hs)

        # Plot contact points transformed to local frame
        for r, sc in zip(rows, scalars):
            if r.get('p1_x') is None or r.get('p2_x') is None:
                continue
            center, R = _row_pose(r)
            col = cmap(norm(sc))

            p1w = np.array([r['p1_x'], r['p1_y'], r.get('p1_z', NOM_POS[2])])
            p2w = np.array([r['p2_x'], r['p2_y'], r.get('p2_z', NOM_POS[2])])
            p1l = R.T @ (p1w - center)
            p2l = R.T @ (p2w - center)

            ax.scatter(*p1l, color=col, s=70, marker='o',
                       edgecolors='white', linewidths=0.5, depthshade=True)
            ax.scatter(*p2l, color=col, s=70, marker='s',
                       edgecolors='white', linewidths=0.5, depthshade=True)
            ax.plot3D([p1l[0], p2l[0]], [p1l[1], p2l[1]], [p1l[2], p2l[2]],
                      color=col, alpha=0.35, linewidth=0.9)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.10, shrink=0.65)
        cb.set_label(clabel, fontsize=8)

        max_hs = max(_row_hs(r) for r in rows)
        lim = max_hs * 1.4
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel('X_local (m)', fontsize=8, labelpad=2)
        ax.set_ylabel('Y_local (m)', fontsize=8, labelpad=2)
        ax.set_zlabel('Z_local (m)', fontsize=8, labelpad=2)
        ax.set_title(sweep['label'], fontsize=10, pad=6)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=25, azim=-55)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_convergence_bar(all_rows, out_path):
    sweep_names  = [s['name']  for s in SWEEPS]
    sweep_labels = [s['label'] for s in SWEEPS]

    conv_rates = []
    for sname in sweep_names:
        rows = [r for r in all_rows if r['sweep'] == sname]
        if not rows:
            conv_rates.append(0.0)
            continue
        conv = sum(1 for r in rows if r.get('status') == 'converged')
        conv_rates.append(100.0 * conv / len(rows))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2ecc71' if c >= 80 else '#e67e22' if c >= 40 else '#e74c3c'
              for c in conv_rates]
    bars = ax.bar(sweep_labels, conv_rates, color=colors, edgecolor='white',
                  linewidth=1.2)
    ax.bar_label(bars, fmt='%.0f%%', fontsize=10, padding=3)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Convergence rate (%)', fontsize=11)
    ax.set_title('Convergence rate per sweep', fontsize=13)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_contact_scatter(sweep_rows, sweep_meta, nom_size, out_path):
    """Top-down (XY) scatter of contact points, colored by parameter value."""
    values = sorted(set(r['sweep_value'] for r in sweep_rows))
    cmap   = CMAP
    norm   = plt.Normalize(vmin=min(values), vmax=max(values))

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw nominal box outline (top-down, centred at nominal XY)
    bx, by = NOM_POS[0], NOM_POS[1]
    hs = nom_size
    rect = plt.Rectangle((bx - hs, by - hs), 2*hs, 2*hs,
                          fill=False, edgecolor='black', linewidth=2,
                          linestyle='-', label='Nominal box')
    ax.add_patch(rect)

    for r in sweep_rows:
        c = cmap(norm(r['sweep_value']))
        p1x = r.get('p1_x')
        p1y = r.get('p1_y')
        p2x = r.get('p2_x')
        p2y = r.get('p2_y')
        if p1x is not None:
            ax.plot(p1x, p1y, 'o', color=c, markersize=9, alpha=0.85,
                    markeredgecolor='white', markeredgewidth=0.6)
        if p2x is not None:
            ax.plot(p2x, p2y, 's', color=c, markersize=9, alpha=0.85,
                    markeredgecolor='white', markeredgewidth=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(sweep_meta['label'], fontsize=10)

    p1_patch = mpatches.Patch(color='gray', label='p1 thumb (circle)')
    p2_patch = mpatches.Patch(color='gray', label='p2 index (square)',
                               linestyle='--')
    ax.legend(handles=[rect, p1_patch, p2_patch], fontsize=8, loc='upper right')

    margin = 0.08
    ax.set_xlim(bx - hs - margin, bx + hs + margin)
    ax.set_ylim(by - hs - margin, by + hs + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('World X (m)', fontsize=11)
    ax.set_ylabel('World Y (m)', fontsize=11)
    ax.set_title(f'Contact points — sweep: {sweep_meta["label"]}', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_solve_time(sweep_rows, sweep_meta, out_path):
    _fig_metric_vs_param(sweep_rows, sweep_meta, 'solve_time_ms',
                          'Total solve time (ms)', out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Grasp planner robustness sweep')
    ap.add_argument('--nc',         type=int,   default=6,
                    help='Seeds per solve (default 6)')
    ap.add_argument('--max-iter',   type=int,   default=150,
                    help='Max IPOPT iterations per seed (default 150)')
    ap.add_argument('--log-prefix', type=str,   default='robustness',
                    help='Log directory prefix (default robustness)')
    args = ap.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(f'logs/{args.log_prefix}_{ts}')
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'plots').mkdir(exist_ok=True)

    import sys, io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace',
                                   line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(utf8_stdout),
            logging.FileHandler(log_dir / 'robustness.log', encoding='utf-8'),
        ])
    log = logging.getLogger('robustness')
    log.info(f"Output directory: {log_dir}")
    log.info(f"nc={args.nc}  max_iter={args.max_iter}")

    # ── Model + planner ───────────────────────────────────────────────────────
    model = mj.MjModel.from_xml_path(SCENE_XML)
    data  = mj.MjData(model)
    for j in (0, 2, 4, 6):
        model.jnt_range[j] = [-np.pi, np.pi]
    mj.mj_forward(model, data)

    q_bias = make_q_bias()

    obj_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'obj_red_box')
    obj_gid     = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'obj_red_box_geom')
    jnt_id      = model.body_jntadr[obj_body_id]
    qpos_adr    = int(model.jnt_qposadr[jnt_id])

    grasp_cfg = GraspConfig3D(
        obj_geom  = 'obj_red_box_geom',
        obj_body  = 'obj_red_box',
        max_iter  = args.max_iter,
    )
    ms_planner = MultiStartGraspPlanner3D(model, data, grasp_cfg,
                                          log_dir=str(log_dir))
    log.info(f"Planner ready — nominal size: "
             f"hx={ms_planner._obj_hx:.3f} hy={ms_planner._obj_hy:.3f} "
             f"hz={ms_planner._obj_hz:.3f}")

    # ── Sweep loop ────────────────────────────────────────────────────────────
    all_rows   = []
    json_store = []

    for sweep in SWEEPS:
        sname  = sweep['name']
        log.info(f"\n{'='*60}")
        log.info(f"SWEEP: {sweep['label']}")
        log.info(f"{'='*60}")

        sweep_rows = []
        for val in sweep['values']:
            tc  = build_test_case(sname, val)
            if sname == 'position':
                tag = '({:.2f},{:.2f})'.format(*val)
            else:
                tag = sweep['fmt'].format(val)
            log.info(f"  [{sname}={tag}]  pos={np.round(tc['pos'],3)}  "
                     f"yaw={tc['yaw_deg']:.1f}°  size={tc['size']:.3f}  μ={tc['mu']:.2f}")

            metrics = run_one(
                model, data, ms_planner, q_bias, qpos_adr, obj_gid,
                pos      = tc['pos'],
                yaw_deg  = tc['yaw_deg'],
                size     = tc['size'],
                mu       = tc['mu'],
                nc       = args.nc,
                max_iter = args.max_iter,
                log      = log,
            )

            # Restore nominal parameters before the next case
            set_object_size(model, ms_planner, obj_gid, NOM_SIZE)
            set_friction(model, ms_planner, obj_gid, NOM_MU)

            row = {
                'sweep':       sname,
                'sweep_label': sweep['label'],
                'sweep_value': val,
                **metrics,
            }
            sweep_rows.append(row)
            all_rows.append(row)

            ik_str = (f"IK=({metrics.get('ik_thumb_mm', float('nan')):.1f},"
                      f"{metrics.get('ik_index_mm', float('nan')):.1f})mm"
                      if metrics.get('ik_thumb_mm') is not None else 'IK=N/A')
            gm_str = (f"γ={metrics['gamma_min']:.3f}"
                      if metrics.get('gamma_min') is not None else 'γ=N/A')
            log.info(f"    → status={metrics.get('status',-1):12s}  "
                     f"t={metrics.get('solve_time_ms',0):.0f}ms  "
                     f"{ik_str}  {gm_str}  "
                     f"WF={'OK' if metrics.get('wrench_feasible') else 'NO'}")

            json_store.append({**row,
                               'p1': list(tc.get('pos', [])),
                               'p2': None})

        # ── Per-sweep plots ───────────────────────────────────────────────────
        pdir = log_dir / 'plots'
        if sname == 'position':
            _fig_position_grid(sweep_rows, pdir / 'position_grid.png')
            _fig_position_contact_scatter(sweep_rows, pdir / 'contact_points_position.png')
        else:
            _fig_ik_vs_param(sweep_rows, sweep,
                              pdir / f'ik_error_{sname}.png')
            _fig_metric_vs_param(sweep_rows, sweep, 'gamma_min', 'γ_min (NCF scale)',
                                  pdir / f'gamma_{sname}.png', hline=0.0)
            _fig_solve_time(sweep_rows, sweep,
                            pdir / f'solve_time_{sname}.png')
            _fig_contact_scatter(sweep_rows, sweep, NOM_SIZE,
                                  pdir / f'contact_points_{sname}.png')
        log.info(f"  Plots written for sweep '{sname}'")

    # ── Global convergence bar ────────────────────────────────────────────────
    _fig_convergence_bar(all_rows, log_dir / 'plots' / 'convergence_rate.png')

    # ── 3D contact grid (all sweeps, object-local frame) ─────────────────────
    _fig_3d_contact_grid(all_rows, log_dir / 'plots' / 'contact_3d_grid.png')

    # ── CSV ───────────────────────────────────────────────────────────────────
    if all_rows:
        csv_path = log_dir / 'results.csv'
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(all_rows)
        log.info(f"\nCSV written: {csv_path}  ({len(all_rows)} rows)")

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = log_dir / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_store, f, indent=2, default=str)
    log.info(f"JSON written: {json_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info('\n' + '='*76)
    log.info(f"{'SWEEP':<12} {'VALUE':>14}  {'STATUS':<14} {'TIME(ms)':>9} "
             f"{'IK_TH':>7} {'IK_IF':>7} {'γ_MIN':>7} {'WF':>4}")
    log.info('-'*76)
    for r in all_rows:
        ik_t = r.get('ik_thumb_mm')
        ik_i = r.get('ik_index_mm')
        gm   = r.get('gamma_min')
        sv   = r['sweep_value']
        sv_str = ('({:.2f},{:.2f})'.format(*sv) if isinstance(sv, (tuple, list))
                  else f'{sv:8.3f}')
        log.info(
            f"{r['sweep']:<12} {sv_str:>14}  "
            f"{r.get('status','?'):<14} {r.get('solve_time_ms',0):>9.0f} "
            f"{ik_t if ik_t is not None else float('nan'):>7.1f} "
            f"{ik_i if ik_i is not None else float('nan'):>7.1f} "
            f"{gm if gm is not None else float('nan'):>7.3f} "
            f"{'Y' if r.get('wrench_feasible') else 'N':>4}"
        )
    log.info('='*76)
    log.info(f"Done. Results in {log_dir}/")


if __name__ == '__main__':
    main()

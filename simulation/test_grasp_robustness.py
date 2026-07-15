#!/usr/bin/env python3
"""
test_grasp_robustness.py
========================
Headless batch robustness test for MultiStartGraspPlanner3D.

Sweeps (one-at-a-time; all other params held at nominal):
  1. Object X position  offset  : −0.05 … +0.05 m
  2. Object Y position  offset  : −0.04 … +0.04 m
  3. Object yaw angle           : 0 … 60°
  4. Box half-extent (uniform)  : 0.020 … 0.040 m
  5. Friction coefficient μ     : 0.3 … 1.5

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
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import mujoco as mj

from simulation.grasp_planner_3d import GraspConfig3D, MultiStartGraspPlanner3D

# ── Constants ─────────────────────────────────────────────────────────────────

GEN3_XML = 'mujoco_menagerie/kinova_gen3/gen3.xml'
SCENE_XML = 'models/scene_pick_place_pos.xml'
N_ROBOT  = 23

# Nominal object parameters
NOM_POS  = np.array([0.40, 0.22, 0.03])
NOM_YAW  = 0.0
NOM_SIZE = 0.030
NOM_MU   = 1.0

# ── Sweeps definition ─────────────────────────────────────────────────────────

SWEEPS = [
    {
        'name':   'pos_x',
        'label':  'X-position offset (m)',
        'values': [-0.050, -0.025, 0.000, +0.025, +0.050],
        'fmt':    '{:+.3f}',
    },
    {
        'name':   'pos_y',
        'label':  'Y-position offset (m)',
        'values': [-0.040, -0.020, 0.000, +0.020, +0.040],
        'fmt':    '{:+.3f}',
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


def build_test_case(sweep_name: str, value: float) -> dict:
    p = nominal_params()
    if sweep_name == 'pos_x':
        p['pos'][0] += value
    elif sweep_name == 'pos_y':
        p['pos'][1] += value
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

    obj_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'obj_box')
    obj_gid     = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'obj_box_geom')
    jnt_id      = model.body_jntadr[obj_body_id]
    qpos_adr    = int(model.jnt_qposadr[jnt_id])

    grasp_cfg = GraspConfig3D(
        obj_geom  = 'obj_box_geom',
        obj_body  = 'obj_box',
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
            tag = sweep['fmt'].format(val)
            log.info(f"  [{sname}={tag}]  pos={tc['pos']}  yaw={tc['yaw_deg']:.1f}°  "
                     f"size={tc['size']:.3f}  μ={tc['mu']:.2f}")

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
    log.info('\n' + '='*72)
    log.info(f"{'SWEEP':<12} {'VALUE':>8}  {'STATUS':<14} {'TIME(ms)':>9} "
             f"{'IK_TH':>7} {'IK_IF':>7} {'γ_MIN':>7} {'WF':>4}")
    log.info('-'*72)
    for r in all_rows:
        ik_t = r.get('ik_thumb_mm')
        ik_i = r.get('ik_index_mm')
        gm   = r.get('gamma_min')
        log.info(
            f"{r['sweep']:<12} {r['sweep_value']:>8.3f}  "
            f"{r.get('status','?'):<14} {r.get('solve_time_ms',0):>9.0f} "
            f"{ik_t if ik_t is not None else float('nan'):>7.1f} "
            f"{ik_i if ik_i is not None else float('nan'):>7.1f} "
            f"{gm if gm is not None else float('nan'):>7.3f} "
            f"{'Y' if r.get('wrench_feasible') else 'N':>4}"
        )
    log.info('='*72)
    log.info(f"Done. Results in {log_dir}/")


if __name__ == '__main__':
    main()

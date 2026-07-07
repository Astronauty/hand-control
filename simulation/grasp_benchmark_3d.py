"""
grasp_benchmark_3d.py — Benchmark GraspPlanner3D on scene_pick_place_pos.xml.

Usage
-----
python simulation/grasp_benchmark_3d.py [options]

Sweep modes (mutually exclusive flags)
---------------------------------------
  (default)     Sweep Kinova arm joint angles j1 × j2 (2-D grid).
                Add --nz / --j3-range to add j3 as a third axis (3-D grid).
  --sweep-obj   Sweep object position x × y × z instead.

The object (`obj_box`) is held FIXED at the nominal position during joint sweeps.
The arm warm-start q_ref has its j1/j2/j3 entries overridden at each grid point.

CLI options
-----------
--n1            Grid steps for j1 (default 4)
--n2            Grid steps for j2 (default 4)
--nz            Grid steps for j3 / obj-z (default 1 = no 3rd axis)
--j1-range DEG  Two floats: min max for j1 in degrees (default -30 30)
--j2-range DEG  Two floats: min max for j2 in degrees (default 15 45)
--j3-range DEG  Two floats: min max for j3 in degrees (default -20 20)
--obj-pos XYZ   Fixed object position when sweeping joints (default 0.42 0.25 0.03)
--obj-x-range   Two floats: min max for object X in metres (default 0.35 0.50)
--obj-y-range   Two floats: min max for object Y in metres (default 0.15 0.35)
--obj-z-range   Two floats: min max for object Z in metres (default 0.03 0.10)
--nc            Multi-start seeds (default 6)
--log-dir       Directory for IPOPT logs
--out-dir       Directory for results npz and plots
--no-plot       Skip matplotlib visualisation

Results saved as <out-dir>/results_3d.npz with keys:
  j1, j2, [j3], obj_x, obj_y, obj_z, status, p1, p2, q, cost, iterations

Plots:
  grid_success.png  — 2-D heat-map (j1/j2 or x/y) coloured by success
  contacts_3d.png   — 3-D scatter of contact points coloured by status
  cost_hist.png     — histogram of objective cost across all solves
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "simulation"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

try:
    import mujoco as mj
except ImportError as exc:
    sys.exit(f"MuJoCo not available: {exc}")

from grasp_planner_3d import GraspConfig3D, GraspPlanner3D, MultiStartGraspPlanner3D

# ── logging ───────────────────────────────────────────────────────────────────

def _make_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("bench3d")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                datefmt="%H:%M:%S")
        fh = logging.FileHandler(os.path.join(log_dir, "bench3d.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ── model helpers ─────────────────────────────────────────────────────────────

def _load_scene(xml_path: str):
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    return model, data


def _get_freejoint_qpos_addr(model, body_name: str) -> int:
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        raise ValueError(f"Body '{body_name}' not found.")
    jnt_id = model.body_jntadr[bid]
    if jnt_id == -1:
        raise ValueError(f"Body '{body_name}' has no joint.")
    if model.jnt_type[jnt_id] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Body '{body_name}' joint is not a freejoint.")
    return int(model.jnt_qposadr[jnt_id])


def _fix_object(data, qpos_addr: int, pos: np.ndarray):
    """Teleport the freejoint body; zero its velocity."""
    data.qpos[qpos_addr:qpos_addr + 3]     = pos
    data.qpos[qpos_addr + 3:qpos_addr + 7] = [1, 0, 0, 0]  # identity quaternion
    data.qvel[qpos_addr:qpos_addr + 6]     = 0.0


def _nominal_q(model) -> np.ndarray:
    """
    Neutral reaching pose for Kinova Gen3 (7 DOF) + LEAP hand (16 DOF).

    Actuator order assumed by kinova_leap_pos.xml:
      0..6   joint_1..joint_7  (Kinova arm)
      7..10  IF  mcp/rot/pip/dip
      11..14 MF  mcp/rot/pip/dip
      15..18 RF  mcp/rot/pip/dip
      19..22 TH  cmc/axl/mcp/ipl
    """
    q = np.zeros(model.nu)
    q[:7]  = np.deg2rad([0, 30, 0, -60, 0, 60, 0])   # arm
    curl   = np.deg2rad([20, 0, 30, 20])
    for s in range(7, 19, 4):
        q[s:s+4] = curl
    q[19:23] = np.deg2rad([30, 0, 20, 20])            # thumb
    return q


def _apply_arm_angles(q_base: np.ndarray, j1: float, j2: float,
                      j3: float | None = None) -> np.ndarray:
    """Return a copy of q_base with Kinova joints 0,1,(2) overridden."""
    q = q_base.copy()
    q[0] = np.deg2rad(j1)
    q[1] = np.deg2rad(j2)
    if j3 is not None:
        q[2] = np.deg2rad(j3)
    return q


# ── single-solve helper ───────────────────────────────────────────────────────

def _run_one(planner_cls, model, data, cfg, logger, log_dir,
             q_ref, obj_pos, nc: int) -> dict:
    if nc <= 1:
        pl = planner_cls(model, data, cfg, logger=logger, log_dir=log_dir)
        return pl.solve(q_ref, obj_pos)
    ms = MultiStartGraspPlanner3D(model, data, cfg, logger=logger, log_dir=log_dir)
    return ms.solve(q_ref, obj_pos, max_seeds=nc)


# ── benchmark loop: joint sweep ───────────────────────────────────────────────

def _sweep_joints(args, model, data, obj_qpos_addr, q_base, logger, log_dir, out_dir):
    """Sweep j1 × j2 × [j3], object fixed."""
    j1_vals = np.linspace(*args.j1_range, args.n1)
    j2_vals = np.linspace(*args.j2_range, args.n2)
    j3_vals = np.linspace(*args.j3_range, args.nz) if args.nz > 1 else [None]
    obj_pos = np.array(args.obj_pos)

    _fix_object(data, obj_qpos_addr, obj_pos)
    mj.mj_forward(model, data)

    cfg = GraspConfig3D(w_reg=0.01, on_object=True, ik_constraint=True,
                        penetration_constraint=False, max_iter=100)

    total = args.n1 * args.n2 * args.nz
    done  = 0
    records = []

    for j1 in j1_vals:
        for j2 in j2_vals:
            for j3 in j3_vals:
                done += 1
                q_ref = _apply_arm_angles(q_base, j1, j2, j3)
                label = (f"j1={j1:.1f}° j2={j2:.1f}°"
                         + (f" j3={j3:.1f}°" if j3 is not None else ""))
                logger.info(f"[{done:3d}/{total}] {label}")

                result = _run_one(GraspPlanner3D, model, data, cfg,
                                  logger, log_dir, q_ref, obj_pos, args.nc)

                records.append({
                    'j1': j1, 'j2': j2, 'j3': j3 if j3 is not None else np.nan,
                    'obj_x': obj_pos[0], 'obj_y': obj_pos[1], 'obj_z': obj_pos[2],
                    'status':     result['status'],
                    'success':    result['success'],
                    'cost':       result['cost'],
                    'iterations': result['iterations'],
                    'q':          result['q'],
                    'p1':         result['p1'],
                    'p2':         result['p2'],
                })
                logger.info(f"  => status={result['status']} "
                            f"cost={result['cost']} iter={result['iterations']}")

    return records, ('j1', j1_vals, 'j2', j2_vals, j3_vals)


# ── benchmark loop: object position sweep ────────────────────────────────────

def _sweep_obj(args, model, data, obj_qpos_addr, q_base, logger, log_dir, out_dir):
    """Sweep object x × y × z, arm q_ref held at nominal."""
    x_vals = np.linspace(*args.obj_x_range, args.n1)
    y_vals = np.linspace(*args.obj_y_range, args.n2)
    z_vals = np.linspace(*args.obj_z_range, args.nz) if args.nz > 1 else [args.obj_pos[2]]

    cfg = GraspConfig3D(w_reg=0.01, on_object=True, ik_constraint=True,
                        penetration_constraint=False, max_iter=100)

    total = args.n1 * args.n2 * args.nz
    done  = 0
    records = []

    for ox in x_vals:
        for oy in y_vals:
            for oz in z_vals:
                done += 1
                obj_pos = np.array([ox, oy, oz])
                logger.info(f"[{done:3d}/{total}] obj=({ox:.3f}, {oy:.3f}, {oz:.3f})")

                _fix_object(data, obj_qpos_addr, obj_pos)
                mj.mj_forward(model, data)

                result = _run_one(GraspPlanner3D, model, data, cfg,
                                  logger, log_dir, q_base, obj_pos, args.nc)

                records.append({
                    'j1': np.nan, 'j2': np.nan, 'j3': np.nan,
                    'obj_x': ox, 'obj_y': oy, 'obj_z': oz,
                    'status':     result['status'],
                    'success':    result['success'],
                    'cost':       result['cost'],
                    'iterations': result['iterations'],
                    'q':          result['q'],
                    'p1':         result['p1'],
                    'p2':         result['p2'],
                })
                logger.info(f"  => status={result['status']} "
                            f"cost={result['cost']} iter={result['iterations']}")

    return records, ('obj_x', x_vals, 'obj_y', y_vals, z_vals)


# ── main runner ───────────────────────────────────────────────────────────────

def run_benchmark(args):
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or os.path.join(_ROOT, "logs", f"bench3d_{ts}")
    out_dir = args.out_dir or log_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)

    logger = _make_logger(log_dir)
    mode   = "obj-position" if args.sweep_obj else "arm-joints"
    logger.info(f"=== Grasp Benchmark 3D  {ts}  mode={mode} ===")
    logger.info(f"Grid: n1={args.n1} n2={args.n2} nz={args.nz} nc={args.nc}")

    xml_path = (args.xml if args.xml
                else os.path.join(_ROOT, "models", "scene_pick_place_pos.xml"))
    logger.info(f"Scene: {xml_path}")
    model, data = _load_scene(xml_path)

    obj_qpos_addr = _get_freejoint_qpos_addr(model, 'obj_box')
    q_base        = _nominal_q(model)

    if args.sweep_obj:
        records, sweep_meta = _sweep_obj(
            args, model, data, obj_qpos_addr, q_base, logger, log_dir, out_dir)
    else:
        records, sweep_meta = _sweep_joints(
            args, model, data, obj_qpos_addr, q_base, logger, log_dir, out_dir)

    # ── save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(out_dir, "results_3d.npz")
    np.savez(
        save_path,
        j1         = np.array([r['j1']  for r in records]),
        j2         = np.array([r['j2']  for r in records]),
        j3         = np.array([r['j3']  for r in records]),
        obj_x      = np.array([r['obj_x'] for r in records]),
        obj_y      = np.array([r['obj_y'] for r in records]),
        obj_z      = np.array([r['obj_z'] for r in records]),
        status     = np.array([r['status']    for r in records]),
        success    = np.array([r['success']   for r in records]),
        cost       = np.array([r['cost'] if r['cost'] is not None else np.nan
                               for r in records]),
        iterations = np.array([r['iterations'] if r['iterations'] is not None else -1
                               for r in records]),
        q  = np.array([r['q']  if r['q']  is not None else np.zeros(model.nu)
                       for r in records]),
        p1 = np.array([r['p1'] if r['p1'] is not None else np.zeros(3)
                       for r in records]),
        p2 = np.array([r['p2'] if r['p2'] is not None else np.zeros(3)
                       for r in records]),
    )
    logger.info(f"Saved -> {save_path}")

    n_ok   = sum(1 for r in records if r['success'])
    n_be   = sum(1 for r in records if r['status'] == 'best-effort')
    n_fail = sum(1 for r in records if r['status'] == 'failed')
    total  = len(records)
    logger.info(f"Summary: converged={n_ok}  best-effort={n_be}  failed={n_fail} / {total}")

    if not args.no_plot:
        _plot(records, sweep_meta, args.sweep_obj, out_dir, logger)

    return records


# ── visualisation ─────────────────────────────────────────────────────────────

def _plot(records, sweep_meta, is_obj_sweep, out_dir, logger):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    status_code = {'converged': 0, 'best-effort': 1, 'failed': 2}
    colours     = ['#2ecc71', '#e67e22', '#e74c3c']
    labels      = ['converged', 'best-effort', 'failed']
    patches     = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(3)]

    axis1_name, axis1_vals, axis2_name, axis2_vals, third_vals = sweep_meta
    a1_key = 'j1' if not is_obj_sweep else 'obj_x'
    a2_key = 'j2' if not is_obj_sweep else 'obj_y'

    # ── plot 1: 2-D success heat-map (j1×j2 or obj_x×obj_y) ─────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in records:
        sc = status_code.get(r['status'], 2)
        ax.scatter(r[a1_key], r[a2_key], c=colours[sc], s=140,
                   marker='s', linewidths=0, alpha=0.85)
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    ax.set_xlabel(f"{axis1_name} ({'°' if not is_obj_sweep else 'm'})")
    ax.set_ylabel(f"{axis2_name} ({'°' if not is_obj_sweep else 'm'})")
    ax.set_title("GraspPlanner3D — success by "
                 + ("arm joint angles (j1, j2)" if not is_obj_sweep else "object position"))
    ax.set_aspect('auto')
    fig.tight_layout()
    p1_path = os.path.join(out_dir, "grid_success.png")
    fig.savefig(p1_path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved -> {p1_path}")

    # ── plot 2: 3-D scatter of contact points ─────────────────────────────────
    fig = plt.figure(figsize=(9, 7))
    ax3 = fig.add_subplot(111, projection='3d')
    for r in records:
        sc  = status_code.get(r['status'], 2)
        col = colours[sc]
        # Object centre marker
        ax3.scatter(r['obj_x'], r['obj_y'], r['obj_z'],
                    c='#95a5a6', s=30, alpha=0.4, marker='D')
        if r['p1'] is not None:
            ax3.scatter(*r['p1'], c=col, s=35, marker='o')   # thumb
        if r['p2'] is not None:
            ax3.scatter(*r['p2'], c=col, s=35, marker='^')   # index
    ax3.legend(handles=patches, loc='upper left', fontsize=9)
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Z (m)")
    ax3.set_title("Contact points 3D  ○=thumb  △=index")
    fig.tight_layout()
    p2_path = os.path.join(out_dir, "contacts_3d.png")
    fig.savefig(p2_path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved -> {p2_path}")

    # ── plot 3: cost histogram ─────────────────────────────────────────────────
    costs = [r['cost'] for r in records if r['cost'] is not None]
    if costs:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(costs, bins=max(10, len(costs)//3), color='steelblue', edgecolor='white')
        ax.set_xlabel("Objective cost")
        ax.set_ylabel("Count")
        ax.set_title("Cost distribution (converged + best-effort)")
        fig.tight_layout()
        p3_path = os.path.join(out_dir, "cost_hist.png")
        fig.savefig(p3_path, dpi=130)
        plt.close(fig)
        logger.info(f"Saved -> {p3_path}")

    # ── plot 4: iterations vs cost scatter ─────────────────────────────────────
    ok_recs = [r for r in records if r['cost'] is not None and r['iterations'] is not None]
    if ok_recs:
        fig, ax = plt.subplots(figsize=(5, 3))
        iters = [r['iterations'] for r in ok_recs]
        costs = [r['cost']      for r in ok_recs]
        cols  = [colours[status_code.get(r['status'], 2)] for r in ok_recs]
        ax.scatter(iters, costs, c=cols, alpha=0.7, edgecolors='none')
        ax.legend(handles=patches, fontsize=8)
        ax.set_xlabel("IPOPT iterations")
        ax.set_ylabel("Objective cost")
        ax.set_title("Iterations vs cost")
        fig.tight_layout()
        p4_path = os.path.join(out_dir, "iter_vs_cost.png")
        fig.savefig(p4_path, dpi=130)
        plt.close(fig)
        logger.info(f"Saved -> {p4_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _floats(s): return [float(x) for x in s.split()]

def main():
    p = argparse.ArgumentParser(description="3D Grasp Planner Benchmark")
    p.add_argument("--sweep-obj", action="store_true",
                   help="Sweep object position (x×y×z) instead of arm joint angles")

    # Grid size
    p.add_argument("--n1", type=int, default=4,
                   help="Steps along first axis — j1 or obj_x (default 4)")
    p.add_argument("--n2", type=int, default=4,
                   help="Steps along second axis — j2 or obj_y (default 4)")
    p.add_argument("--nz", type=int, default=1,
                   help="Steps along third axis — j3 or obj_z (default 1 = disabled)")

    # Joint-sweep ranges
    p.add_argument("--j1-range", type=_floats, default=[-30.0, 30.0], metavar="DEG",
                   help="j1 range in degrees: 'MIN MAX' (default '-30 30')")
    p.add_argument("--j2-range", type=_floats, default=[15.0, 45.0], metavar="DEG",
                   help="j2 range in degrees: 'MIN MAX' (default '15 45')")
    p.add_argument("--j3-range", type=_floats, default=[-20.0, 20.0], metavar="DEG",
                   help="j3 range in degrees: 'MIN MAX' (default '-20 20')")

    # Object position
    p.add_argument("--obj-pos", type=_floats, default=[0.42, 0.25, 0.03], metavar="XYZ",
                   help="Fixed object pos for joint sweeps (default '0.42 0.25 0.03')")
    p.add_argument("--obj-x-range", type=_floats, default=[0.35, 0.50], metavar="M",
                   help="Object X range in metres (default '0.35 0.50')")
    p.add_argument("--obj-y-range", type=_floats, default=[0.15, 0.35], metavar="M",
                   help="Object Y range in metres (default '0.15 0.35')")
    p.add_argument("--obj-z-range", type=_floats, default=[0.03, 0.10], metavar="M",
                   help="Object Z range in metres for --sweep-obj (default '0.03 0.10')")

    p.add_argument("--nc",      type=int, default=6,
                   help="Multi-start seeds per solve (1=single, 6=all face pairs)")
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--xml", type=str, default=None,
                   help="Path to scene XML (default: models/scene_pick_place_pos.xml)")

    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()


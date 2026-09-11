"""Shared writer for the grasp analysis figures (quadratic patch + contact movement).

Extracted verbatim from benchmarks/ycb_grasp/pick_and_place.py's _write_plots so the
LIVE TELEOP recommender can emit the same figures the tabletop benchmark does, rather
than the plots being reachable only from the benchmark entry point.

Nothing here is environment-specific: the inputs are plain MuJoCo handles (model,
data, body id) plus the planner's own returned `res` / `verify()` dict and the
`log_dir` it wrote its per-Picard-stage npz traces into. Both the floor and tabletop
benchmarks and teleop can therefore share one implementation.

WHY log_dir IS NOT OPTIONAL. The solve's returned dict carries NO patch data. The
paraboloid frame (seed_l/axis0_l/axis1_l/n_l, kappa0/1, grad_norm, and both the
symmetric t_bound_* and asymmetric t_lo_*/t_hi_* trust bounds) is flattened into
grasp3d_iter_<ts>.npz under 'quad1_'/'quad2_' prefixes by GraspPlanner3D._save_iter_npz
and exists nowhere else. These figures are readable ONLY from that trace.

PASS `res=`. MultiStartGraspPlanner3D runs up to n_seeds attempts that ALL log into
the same log_dir, and the winner is chosen by cost rank, not file order. Without
`res`, _iter_trace_quadratic_stages falls back to the LAST attempt and will happily
draw trust regions from a DISCARDED candidate as if they belonged to the returned
solve. With it, the attempt is matched by the exact pre-NLP seed point
(res['p1_seed']/['p2_seed']) — a sub-micron identity match, not a proximity heuristic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# plot_quadratic_path / plot_grasp_contacts live under benchmarks/. They are pure
# drawing + npz-reading modules with no benchmark scene coupling, so they are imported
# rather than moved (moving them would churn five other callers for no gain).
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO / "benchmarks"))

from grasp_control import object_uv_atlas as oua                    # noqa: E402
from simulation.grasp_planner_3d import _mesh_sdf_entry             # noqa: E402
from ycb_grasp import plot_grasp_contacts as PGC                    # noqa: E402
from ycb_grasp import plot_quadratic_path as QP                     # noqa: E402


def write_grasp_plots(model, data, res, verify_info, log_dir, object_id, seed,
                      body_name, obj_bid, pos, out_dir, quadratic_path=False,
                      n_relin=None, render_hand=True):
    """Per-contact grasp figure for this solve (and optionally the older
    Picard-trajectory view).

    Default is the CONTACT view (plot_grasp_contacts): one zoomed panel per
    solved contact, drawn in plot_seed_quadratic.py's grammar, which answers
    "what does this grasp look like on the object". quadratic_path asks for
    the trajectory view instead, which answers "how did the contact move
    across Picard stages" -- the right question while tuning the
    relinearization loop, and near-empty at n_relin=0 where there is only one
    stage to plot.

    render_hand=False skips the offscreen hand render for the trajectory view.
    Teleop calls it that way: the render needs the GL context that the viewer
    thread owns, and grabbing it from a background plot worker deadlocks.

    Returns the written Path, or None if there was nothing to draw. Never raises --
    a plot failure must not abort a benchmark run or a live teleop session.
    """
    try:
        stages = QP._iter_trace_quadratic_stages(log_dir, res=res)
        if not (stages and any(s["contact"] for s in stages)):
            return None
        V, F = oua.body_visual_mesh(model, obj_bid)
        if quadratic_path:
            hand_rgb = None
            if render_hand:
                hand_rgb = QP._render_hand_rgb(model, data, lookat=pos,
                                               dist=0.45, elev=-35)
            out = Path(out_dir) / f"seed{seed}_quadratic_path.png"
            QP.plot_quadratic_path(V, F, stages, object_id, out,
                                   hand_rgb=hand_rgb, verify_info=verify_info)
            print(f"[plan] quadratic path -> {out.name}")
            return out
        # LAST stage carrying contact frames = the returned solve's contacts
        # (_iter_trace_quadratic_stages already narrowed to the winning attempt).
        last = next(s for s in reversed(stages) if s["contact"])
        # True-SDF probe for the per-patch error bar, in the same object-local
        # frame the saved quad_* frames use.
        sdf_fn = None
        try:
            _me = _mesh_sdf_entry(model, obj_bid)
            sdf_fn = lambda p: float(_me["fn"](np.asarray(p, float)))   # noqa: E731
        except Exception:
            pass
        out = Path(out_dir) / f"seed{seed}_grasp_contacts.png"
        got = PGC.plot_grasp_contacts(
            V, F, last, object_id, out,
            sdf_fn=sdf_fn, verify_info=verify_info, n_relin=n_relin)
        if got is not None:
            print(f"[plan] grasp contacts -> {out.name}")
            return out
        return None
    except Exception as e:
        print(f"[plan] contact plot failed: {e}")
        return None

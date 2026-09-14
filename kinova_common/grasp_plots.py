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
from ycb_grasp import out_paths as OP                             # noqa: E402
from ycb_grasp import plot_grasp_contacts as PGC                    # noqa: E402
from ycb_grasp import plot_quadratic_path as QP                     # noqa: E402


def write_grasp_plots(model, data, res, verify_info, log_dir, object_id, seed,
                      body_name, obj_bid, pos, out_dir,
                      n_relin=None, render_hand=True, planner=None):
    """Per-contact grasp figure for this solve.

    The CONTACT view (plot_grasp_contacts): one zoomed panel per solved contact,
    drawn in plot_seed_quadratic.py's grammar, which answers "what does this grasp
    look like on the object".

    The Picard-TRAJECTORY view ("how did the contact move across Picard stages")
    was REMOVED along with its quadratic_path argument: the solver now runs a
    single stage (n_normal_relinearize=0 paired with quadratic_symbolic_normals --
    see grasp_config_builder.for_gws_recommender), so there is no inter-stage
    trajectory left to draw. plot_quadratic_path is still imported for
    _iter_trace_quadratic_stages, its shared trace reader.

    planner: when given, ALSO writes the paired seed figure (seed<N>_contact_seeds.pdf)
    from that planner's last_seed_accept_table / last_seed_reject_table -- every
    contact seed THIS solve considered, accepted and rejected, in the same grammar
    as plot_seed_quadratic. Paired by construction: same seeds, same gates, same
    RNG draw as the grasp being drawn beside it. Omit it to skip that figure.

    render_hand=False skips the offscreen hand render.
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
        # NOTE: the Picard-TRAJECTORY figure (seed<N>_quadratic_path) was removed --
        # it answered "how did the contact MOVE across Picard stages", which the
        # BUILDER DEFAULT makes vacuous: for_gws_recommender setdefaults
        # n_normal_relinearize=0 (a single stage) alongside
        # quadratic_symbolic_normals. But that is only a setdefault -- a caller
        # passing n_normal_relinearize explicitly still gets a multi-stage solve
        # (benchmarks/ycb_grasp/pick_and_place.py's --n-relin did exactly that),
        # so do NOT read this as "the solver cannot run multiple stages".
        # plot_quadratic_path is still imported for _iter_trace_quadratic_stages, the
        # shared trace reader that narrows a log_dir to the WINNING attempt.
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
        out = OP.fig_path(Path(out_dir) / f"seed{seed}_grasp_contacts.png")
        # The stage COUNT is no longer reported in the figure. The builder
        # default is a single stage (n_normal_relinearize=0), and len(stages)
        # counted stage records found in the log dir rather than stages this
        # solve ran -- it read "3 Picard stages" for a one-stage, 173-iteration
        # solve. n_relin stays in THIS function's signature because callers pass
        # it and plot_quadratic_path.py still uses the multi-stage trace.
        got = PGC.plot_grasp_contacts(
            V, F, last, object_id, out,
            sdf_fn=sdf_fn, verify_info=verify_info)
        if got is not None:
            print(f"[plan] grasp contacts -> {out.name}")
        _write_seed_fig(planner, model, data, out_dir, seed, res=res)
        return out if got is not None else None
    except Exception as e:
        print(f"[plan] contact plot failed: {e}")
        return None


def _write_seed_fig(planner, model, data, out_dir, seed, res=None):
    """Paired seed figure, best-effort: a failure here must never lose the grasp
    figure that was already written.

    res is forwarded so the accepted panel can overlay the SOLVED contacts --
    without it the seed figure and the grasp-contacts figure read as different
    grasps, because the NLP moves the contact well off its seed."""
    if planner is None:
        return None
    try:
        from kinova_common.seed_figure import write_seed_figure
        got = write_seed_figure(planner, model, data,
                                # "contact_seeds", not "seeds". The seed<N>
                                # prefix is the RNG seed -- ONE planning run --
                                # while the seeds this figure draws are CONTACT
                                # seeds, the candidate grasps tried within that
                                # run (cfg.n_seeds of them). "seed0_seeds.pdf"
                                # collided those two senses in one filename.
                                Path(out_dir) / f"seed{seed}_contact_seeds.png",
                                res=res)
        if got is not None:
            print(f"[plan] contact seeds -> {got.name}")
        return got
    except Exception as e:
        print(f"[plan] seed figure failed: {e}")
        return None

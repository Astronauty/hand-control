"""GraspConfigBuilder: the one shared place that assembles a GraspConfig3D from a
model + a named preset + overrides, replacing two independently-duplicated inline
builders found in the repo:

  - kinova_leap_pick_place.py's _get_cat_planner (~80 lines: hand-tuned production
    values + ad hoc tiered collision-geom construction)
  - benchmarks/ycb_grasp/ablate_grasp.py's main() cfg_kw dict pattern (generic
    collision geoms from ik_demo.robot_geom_names/clearance_by_geom + CLI-toggled
    ablation fields)

Both call sites previously hand-assembled GraspConfig3D(...) from scratch with no
shared code between them. GraspConfigBuilder.for_teleop_recommender(...) and
.for_ablation_default(...) are behavior-preserving extractions of those two exact
construction patterns — not a new design, just a shared home for an existing one.
"""
from __future__ import annotations

from simulation.grasp_planner_3d import GraspConfig3D


def for_ablation_default(obj_geom: str, obj_body: str,
                         arm_geom_names: list, obj_clearance_by_geom: dict,
                         n_seeds: int = 5, max_iter: int = 200,
                         col_clearance_m: float = 0.005,
                         **overrides) -> GraspConfig3D:
    """Matches benchmarks/ycb_grasp/ablate_grasp.py's main() cfg_kw base (before its
    CLI-toggled --gws/--uv-atlas/--restarts overrides are layered on): generic
    collision geoms from the caller's precomputed robot_geom_names/clearance_by_geom
    (e.g. ik_demo.robot_geom_names(model) / ik_demo.clearance_by_geom(names), computed
    ONCE outside any per-object/per-seed loop since they're robot-only and don't
    depend on which object is attached), no GWS/UV-atlas/restart perturbation.

    **overrides is passed straight through to GraspConfig3D (flat or nested kwargs
    both work — see GraspConfig3D's own docstring) — e.g.
    for_ablation_default(..., w_gws=5.0, w_span=1.0) to opt into GWS, matching
    ablate_grasp.py's --gws flag.
    """
    return GraspConfig3D(
        obj_geom=obj_geom, obj_body=obj_body,
        n_seeds=n_seeds, max_iter=max_iter,
        arm_geom_names=arm_geom_names, obj_clearance_by_geom=obj_clearance_by_geom,
        col_clearance_m=col_clearance_m,
        **overrides,
    )


def for_teleop_recommender(obj_name: str, arm_geom_names: list,
                           obj_clearance_by_geom: dict,
                           accel_budget_xyz: tuple, ang_accel_budget_xyz: tuple,
                           max_iter: int = 120,
                           **overrides) -> GraspConfig3D:
    """Matches kinova_leap_pick_place.py's _get_cat_planner preset exactly: the
    hand-tuned production values for the live NLP grasp recommender (DECOUPLED
    IK-only + datum-gamma architecture — see RAISED_CONTACT_WRENCH_FINDINGS.md
    sec 5 for why wrench_constraint=False + datum_gamma=True + w_align +
    edge_margin_m are used together here).

    arm_geom_names / obj_clearance_by_geom are the CALLER's pre-computed tiered
    collision-geom set (palm/wrist tier + active-finger links + non-active-finger
    positive-clearance tier — see _get_cat_planner's own extensive comments for why
    this specific tiering exists) — this builder does not recompute that tiering
    itself, since it depends on live scene state (which fingers are active, which
    geoms exist in this scene) that only the caller has at hand.

    **overrides is passed straight through to GraspConfig3D for anything this
    preset doesn't already fix (e.g. per-object edge_margin_m tweaks).
    """
    cfg_kw = dict(
        obj_geom=obj_name + '_geom', obj_body=obj_name,
        max_iter=max_iter, arm_geom_names=arm_geom_names,
        obj_clearance_by_geom=obj_clearance_by_geom,
        w_align=10.0, orient_weight=2.0, edge_margin_m=0.03,
        ground_clearance_m=0.010,
        wrench_constraint=False, datum_gamma=True,
        accel_budget_xyz=tuple(accel_budget_xyz),
        ang_accel_budget_xyz=tuple(ang_accel_budget_xyz),
    )
    cfg_kw.update(overrides)
    return GraspConfig3D(**cfg_kw)

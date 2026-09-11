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

import json
from pathlib import Path

from simulation.grasp_planner_3d import GraspConfig3D

# Seed/surrogate settings that live in a tunable file rather than in
# GraspConfig3D's dataclass defaults. See load_seed_config().
SEED_CONFIG_PATH = Path(__file__).resolve().parent.parent / "models" / "grasp_seed_config.json"


def load_seed_config(obj_id: str | None = None,
                     path: str | Path | None = None) -> dict:
    """GraspConfig3D kwargs from models/grasp_seed_config.json.

    Returns a flat dict suitable for **-splatting into for_ablation_default (or
    for assigning onto an already-built cfg). Keys beginning with '_' are
    comments and are dropped. When obj_id is given, that id's block under
    'per_object' is merged OVER the top-level block.

    These are DEFAULTS. Callers must apply them so that an explicit CLI flag or
    environment override still wins -- dict.setdefault, not dict.update -- which
    gives the intended precedence: file -> per-object -> CLI/env.

    A missing or unreadable file returns {} rather than raising: the file is a
    tuning convenience, and the dataclass defaults must remain sufficient on
    their own for anyone running from a fresh checkout.
    """
    p = Path(path) if path is not None else SEED_CONFIG_PATH
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    out = {k: v for k, v in raw.items()
           if not k.startswith("_") and k != "per_object"}
    if obj_id is not None:
        per = raw.get("per_object") or {}
        out.update({k: v for k, v in (per.get(obj_id) or {}).items()
                    if not k.startswith("_")})
    return out


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


def for_gws_recommender(obj_name: str, arm_geom_names: list,
                        obj_clearance_by_geom: dict,
                        accel_budget_xyz: tuple, ang_accel_budget_xyz: tuple,
                        max_iter: int = 80,
                        w_gws: float = 5.0, w_span: float = 1.0,
                        obj_id: str | None = None,
                        **overrides) -> GraspConfig3D:
    """for_teleop_recommender + the FRoGGeR min-weight (GWS) quality objective.

    Rationale for the combination (see GWSConfig / RAISED_CONTACT_WRENCH_FINDINGS
    sec 5):
      * The teleop preset's NLP is IK-only (wrench_constraint=False): nothing in
        its cost asks for a GOOD grasp, only a reachable, antipodal, off-edge one.
        -w_gws*beta is the first term that steers contact PLACEMENT toward deep
        force closure rather than measuring it post-hoc.
      * ADDITIVE, not a replacement: w_align/orient_weight/edge_margin_m stay on.
        GWSConfig's own docstring is explicit that the two are additive.
      * datum_gamma stays True and is UNAFFECTED -- it only selects verify()'s
        post-solve certificate (Task-B/datum vs legacy CoM), which runs after the
        NLP either way. alpha/beta and gamma are separate computations over the
        same W: alpha is normalized (sum(alpha)=1, unitless closure witness),
        gamma is the task-specific squeeze force in newtons.
      * gws_soft_finger stays FALSE (the dataclass default). The rank argument for
        turning it ON is correct but MEASURED HARMFUL, so it is recorded here to
        stop it being re-adopted: a 2-contact PCwF W is genuinely rank-5-of-6
        (singular values 4.0/4.0/2.83/0.116/0.116/0) and the soft-finger columns do
        restore rank 6 -- but only in a vestigial direction (the 6th singular value
        is 0.1 against 4.0), because the authored torsional friction mu_t=0.05 is
        40x smaller than mu=2.0.
        Meanwhile beta PAYS FULL PRICE for them. beta is the MINIMUM weight of a
        convex combination constrained to sum(alpha)=1 -- a FIXED budget -- so going
        from 10 to 14 columns lowers the achievable minimum for purely arithmetic
        reasons (10 columns share 1.0 -> <=0.100 each; 14 -> <=0.071). And the two
        added columns per contact carry the IDENTICAL force vector, differing only
        by the +/-mu_t*n spin term: measured 99.5% parallel (cos=0.995). So the
        budget is diluted across near-duplicates that add almost no capability.
        Measured on 014_lemon (seed 0, otherwise identical config):
            gws_soft_finger=True   beta=-7e-06  WF=False  IK=(9.81, 8.02)mm
            gws_soft_finger=False  beta=+0.056  WF=True   IK=(4.68, 0.29)mm  gamma=1.96
        and beta=-0.487 on 036_wood_block with it on. If torque about the grasp axis
        ever genuinely matters, the fix is a THIRD contact (a real moment arm), not a
        near-degenerate column pair.
      * quadratic contact parameterization is forced ON: the repo is migrating to
        YCB meshes permanently, and the quadratic path is both the on-surface-by-
        construction parameterization and the ONLY path that records the
        paraboloid frame the analysis figures are drawn from. quadratic_mesh_fit
        fits curvature to the visual mesh rather than the SDF Hessian (which
        reports curvature belonging to features up to 50mm away and collapses the
        trust region on flat faces).
    """
    cfg_kw = dict(
        w_gws=w_gws, w_span=w_span, gws_soft_finger=False,
        use_quadratic_contact=True, quadratic_mesh_fit=True,
    )
    # SEED GATES from models/grasp_seed_config.json, as DEFAULTS (setdefault, so any
    # caller override still wins). Without this the teleop path ran the dataclass
    # defaults while the benchmark layered the tuned file values in -- i.e. the two
    # "shared" configs had DIFFERENT seed gates (seed_ground_clearance_m None vs
    # 0.005, and no per-object block at all). The gates themselves already run inside
    # solve() (_reachable_contact / _seed_kappa_ok / _dls_residual); this only makes
    # both paths gate on the same numbers, matching plot_seed_quadratic, which reads
    # the same file so "this diagnostic cannot drift from the gate it is drawing".
    for _k, _v in load_seed_config(obj_id).items():
        cfg_kw.setdefault(_k, _v)
    # DLS-IK reachability screen ON by default. This is the ONLY seed screen that
    # knows about the ARM -- the other two are geometric (above the table, not on an
    # edge) and a seed can pass both while sitting where the arm cannot bring a
    # fingertip. That is the measured failure mode here: 014_lemon planned contacts
    # sit sub-mm from the true surface while the THUMB GEOM stops 9.92mm away, which
    # aborts the squeeze at the 8mm gap gate. Over-generates k*n_seeds candidates and
    # keeps the n_seeds with the smallest DLS fingertip residual; also populates
    # planner.last_seed_rank_table for the seed diagnostics.
    cfg_kw.setdefault('seed_dls_rank_pool', 3)
    # n_seeds=3 (was 5) and max_iter=80 (was 120/200), MEASURED on the five tabletop
    # objects, seed 0, execution through pick_and_place.py -- not a speed compromise:
    #   200/5 -> 3/5 objects squeeze+lift, solve 5.4s
    #    80/3 -> 4/5 objects squeeze+lift, solve ~2.3s
    # 036_wood_block FAILED at 200/5 (index gap 10.96mm vs the 8mm squeeze gate) and
    # LIFTS at 80/3 with the tightest contacts measured (0.32/0.60mm). Cutting the
    # budget IMPROVED the grasp, which fits the IPOPT iterate log: the objective
    # oscillates (60 -> 256 -> 110 -> 79 across consecutive iterations) because the
    # duals are indeterminate under the antipodal minimax symmetry, so grinding
    # further lands on a worse iterate as often as a better one.
    # The DLS rank pool is what makes n_seeds=3 safe -- seeds are ordered by arm
    # reachability first, so seeds 4-5 were the worst of the pool (dropping them was
    # bit-identical on 014_lemon and 056_tennis_ball).
    cfg_kw.setdefault('n_seeds', 3)
    # NO PICARD RELINEARIZATION. n_normal_relinearize=0 means the solve is a SINGLE
    # stage: the contact frame is never re-frozen from a re-read normal. This is only
    # coherent together with quadratic_symbolic_normals -- the paraboloid supplies the
    # normal in CLOSED FORM (_quadratic_inward_normal_ca) as a function of the contact
    # coords, so the frame tracks the contact WITHIN the stage instead of being pinned
    # to the seed's normal for the whole solve. With relinearization off and symbolic
    # normals off, the seed normal would be frozen end-to-end.
    #
    # CAUTION, recorded rather than buried: quadratic_symbolic_normals and its four
    # per-consumer switches "went in together and regressed the tabletop benchmark
    # 6/8 -> 3/8 full cycles" (see GraspConfig3D's own docstring). The four consumers
    # -- wrench/GWS frame, IK target, w_align, orient_weight -- can each be switched
    # back off individually (quad_sym_normals_{frame,iktgt,align,orient}=False) to
    # attribute a regression. Expect to ablate those before trusting this default.
    cfg_kw.setdefault('n_normal_relinearize', 0)
    cfg_kw.setdefault('quadratic_symbolic_normals', True)
    cfg_kw.update(overrides)
    return for_teleop_recommender(
        obj_name, arm_geom_names, obj_clearance_by_geom,
        accel_budget_xyz=accel_budget_xyz,
        ang_accel_budget_xyz=ang_accel_budget_xyz,
        max_iter=max_iter, **cfg_kw)

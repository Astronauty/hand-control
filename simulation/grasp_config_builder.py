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


FINGER_CONFIG_PATH = Path(__file__).resolve().parent.parent / "models" / "grasp_finger_config.json"

# Contact SLOTS are positional in the NLP. Slot 1 is the anchor/opposing contact,
# slot 2 the contact it pinches against, slot 3+ the off-axis load-bearing ones.
# The GeometryNames fields are still named thumb_*/index_*/middle_* for slots 1/2/3
# because every consumer in grasp_planner_3d.py reads them by those names -- this
# table is the ONE place that knows the mapping is positional, not anatomical.
_SLOT_FIELDS = (("thumb_site", "thumb_geom"),
                ("index_site", "index_geom"),
                ("middle_site", "middle_geom"))


def parse_fingers(spec):
    """'thumb,middle' (or a list) -> ['thumb', 'middle']. None stays None.

    Lives here rather than in argparse so the parsing and the role validation in
    load_finger_config() cannot drift apart."""
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = [t.strip() for t in spec.split(",") if t.strip()]
    return [str(t) for t in spec]


def load_finger_config(obj_id: str | None = None,
                       fingers=None,
                       path: str | Path | None = None) -> dict:
    """GraspConfig3D kwargs selecting WHICH FINGERS a grasp uses.

    Returns a flat dict of slot site/geom names plus a derived n_contacts, suitable
    for **-splatting into a builder. Same contract as load_seed_config(): '_' keys
    are comments, 'per_object' overrides the default for one YCB id, and a missing
    or unreadable file returns {} rather than raising -- the dataclass defaults
    (thumb+index) must remain sufficient on a fresh checkout.

    n_contacts is DERIVED from len(fingers), never passed independently, so the
    finger list and the contact count cannot disagree.

    fingers : ordered list of role names (or a 'a,b' string). None uses the
        per-object entry if present, else the file's own 'fingers' list.
        ORDER MATTERS -- see the file's own _comment_order: slot 1 anchors the
        antipodal seed march and w_align gates on slot-1<->slot-2 opposition.
    """
    from kinova_common.constants import FINGER_CODE, FINGER_TIP_SITES

    fingers = parse_fingers(fingers)
    if fingers is None:
        p = Path(path) if path is not None else FINGER_CONFIG_PATH
        try:
            raw = json.loads(p.read_text())
        except (OSError, ValueError):
            return {}
        per = raw.get("per_object") or {}
        fingers = per.get(obj_id) if obj_id else None
        if not fingers:
            fingers = raw.get("fingers")
        if not fingers:
            return {}
        fingers = parse_fingers(fingers)

    if not 2 <= len(fingers) <= len(_SLOT_FIELDS):
        raise ValueError(
            f"load_finger_config: {fingers!r} names {len(fingers)} fingers; the NLP "
            f"supports 2..{len(_SLOT_FIELDS)} contact slots.")
    if len(set(fingers)) != len(fingers):
        raise ValueError(
            f"load_finger_config: {fingers!r} repeats a finger; each contact slot "
            f"needs its own fingertip.")

    out: dict = {"n_contacts": len(fingers)}
    for slot, role in enumerate(fingers):
        if role not in FINGER_TIP_SITES or role not in FINGER_CODE:
            raise ValueError(
                f"load_finger_config: unknown finger {role!r} in {fingers!r}. "
                f"Known fingers: {sorted(FINGER_TIP_SITES)!r}.")
        site_field, geom_field = _SLOT_FIELDS[slot]
        out[site_field] = FINGER_TIP_SITES[role]
        out[geom_field] = f"leap_{FINGER_CODE[role]}_tip"
    # No metadata keys: this dict is **-splatted into GraspConfig3D, whose __init__
    # raises TypeError on anything it does not recognise.
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
                        fingers=None,
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
      * gws_soft_finger stays FALSE (the dataclass default), but the reasoning
        below has been PARTLY RETRACTED -- read the re-measurement before citing
        it as a reason not to try the flag.

        Structural facts (re-confirmed, these hold): a 2-contact PCwF W is
        rank-5-of-6, singular values 4.0/4.0/2.83/0.144/0.144/0.0, and its
        left-null direction is exactly TORQUE ABOUT THE GRASP AXIS. The
        soft-finger columns do restore rank 6. The authored mu_t=0.05 puts the
        6th singular value at 0.1 against 4.0, and the two added columns per
        contact are 99.5% parallel in force (cos=0.995) -- so as CAPABILITY the
        added direction is indeed vestigial.

        RETRACTED -- "beta PAYS FULL PRICE ... the budget is diluted": the
        arithmetic is right (10 cols -> beta <= 0.100, 14 -> <= 0.0714) but it
        makes RAW beta incomparable across the flag, because the flag itself
        moves the 1/n_cols ceiling. On an ideal antipodal pinch beta/ceiling ==
        1.0000 for BOTH models (beta*n_cols == 1.0 identically), i.e. the entire
        0.100 -> 0.0714 drop is the scale change with ZERO geometric
        degradation. Compare beta*n_cols, or rescale w_gws, when judging this.

        RETRACTED -- the measured numbers do not reproduce. The note cited
        014_lemon soft=True as beta=-7e-06/WF=False and 036_wood_block as
        beta=-0.487. Re-measured on the tabletop benchmark (seed 0, n_seeds=1,
        n_relin=3), reporting beta*n_cols alongside raw beta:
            014_lemon      off  beta=0.0797 (x n=0.797)  gamma=1.256  WF=T  LIFTED
            014_lemon      ON   beta=0.0636 (x n=0.890)  gamma=1.049  WF=T  LIFTED
            017_orange     off  beta=0.0968 (x n=0.968)  gamma=1.944  WF=T  LIFTED
            017_orange     ON   beta=0.0702 (x n=0.983)  gamma=1.904  WF=T  LIFTED
            036_wood_block off  beta=0.0992 (x n=0.992)  gamma=27.72  WF=T  squeeze aborted (gap)
            036_wood_block ON   beta=0.0692 (x n=0.968)  gamma=None   WF=F  gamma-infeasible
        No negative beta anywhere; on BOTH rounded objects the flag lifts the
        object and IMPROVES beta*n_cols and gamma. Two likely reasons the old
        numbers differ: they predate eaa622a (contact-patch SDF ray-intersection)
        and 22a7b0a (solver preset), and pick_and_place.py did not report
        gws_beta AT ALL until it was wired up, so whatever harness produced them
        is not the one above.

        WHAT DOES SURVIVE as a reason to leave it off: 036_wood_block goes
        WRENCH-INFEASIBLE with the flag on. DIAGNOSED, and the cause is a
        TRUST-REGION defect on flat faces that this flag merely EXPOSES -- not
        a defect in the soft-finger model:

          * It is NOT a frozen-frame problem. This preset already sets
            quadratic_symbolic_normals=True with n_normal_relinearize=0, so the
            GWS frame comes from _quadratic_inward_normal_ca in closed form and
            tracks the contact within the stage (verified by call-count
            instrumentation, 16 calls on the default config).
          * NOT an oversized trust region either. The patches on
            036_wood_block come back +/-42.3 x +/-95.3 mm and +/-43.1 x
            +/-96.9 mm, which look bigger than the object only if the extents
            are read off `--mode scene-only` (72 x 74 x 71 mm -- that is
            TS.hull_vertices, a WORLD-frame bbox of the settled TILTED object).
            True object-local extent is 101 x 102 x 206 mm, a tall post, so
            95.3 < the 103.2 mm half-extent: the bound is INSIDE the face.
            _sdf_axis_bound_np behaves correctly (SDF departure 0.00-0.02 mm
            over the full 100 mm march vs a 4.0 mm tolerance, because the face
            IS flat that far), and kappa=0 on every patch so they all already
            route through the planar SDF-search bound.
          * Resulting contact FACES (object-local), default config:
                pcwf: c1 on -y, c2 on +y  -> opposed, n1.n2 = -0.996 (5.2 deg)
                soft: c1 on -y, c2 on +z  -> PERPENDICULAR, n1.n2 = +0.164
                                             (99.4 deg splay)
            span_margin_final collapses 2.124 -> 0.479 rad. The orange, being
            round, is untouched (2.117 -> 2.142) -- its patch bound is a real
            curvature measurement, not a cap.
          * beta is evaluated on the extrapolated paraboloid and reported
            +0.0692, while the TRUE normals at those solved points give
            beta = -0.0 (not force closure).
          * solve_gamma_live then correctly REJECTS it: the 99 deg grasp is
            feasible to ~5 m/s^2 (gamma 7.43) but INFEASIBLE at the 20 m/s^2
            carry budget, where the 5 deg pinch gives gamma=25.5. The
            certificate is doing its job; the NLP handed it a bad grasp.
          * Note solve_gamma_live is PCwF and takes no mu_t, so the flag
            changes WHERE contacts are placed and then has them certified by a
            model with no spin capability. That asymmetry is intended (the
            certificate should not assume torsion the executor cannot deliver)
            but it means gws_soft_finger can only ever be a PLACEMENT prior.

        beta is near-identical in splay sensitivity under both models (both
        collapse to ~0 past 80 deg), so the flag does NOT make beta blind to
        splay -- the oversized patch is what lets the contact reach a splay
        that far out in the first place.

        061_foam_brick is NOT evidence about this flag: it is wrench-infeasible
        in BOTH arms (147 deg / 134 deg splay), and on the PCwF BASELINE both
        of its contacts land on the SAME top face (n1.n2 = +0.84) with
        span_margin = -0.359 (closure geometrically impossible) while beta
        reports +0.0996 -- i.e. the same beta/geometry disagreement shows up on
        the BASELINE, so it is not attributable to this flag.

        Root cause of the flat-face regression is still OPEN (frozen frames and
        oversized patch bounds are both ruled out above). The default stays
        FALSE because on flat-faced objects the flag reliably lands on the bad
        geometry, not because the mechanism is understood. The cheap guard,
        independent of cause: reject any solve with span_margin_final < 0
        before trusting beta -- see _embed_gws_ca's docstring.

        For the DUAL indeterminacy specifically (non-unique KKT multipliers on
        W@alpha==0, see _embed_gws_ca), this flag IS the direct 2-finger fix:
        rank 5 -> 6 kills the left-null space, taking cond(W) from inf to 40 at
        mu_t=0.05 (saturating at 27.8 for mu_t >= 0.1 -- past that the 0.144
        tangential directions are the weak ones, so raising mu_t buys nothing).
        A THIRD contact remains strictly better (a real moment arm rather than a
        mu_t-scaled vestigial one), but it is not the ONLY fix.
        Probe with PFF_GWS_SOFT_FINGER=1 on the tabletop benchmark.
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
    # WHICH FINGERS, from models/grasp_finger_config.json, as DEFAULTS (setdefault) on
    # the same file -> per-object -> CLI/env precedence the seed config uses. This sets
    # the slot site/geom names AND derives n_contacts from the pairing's length, so the
    # contact count can no longer disagree with the finger identities.
    #
    # MEASURED: slot 2 is bound to the index ONLY through index_site/index_geom --
    # every downstream consumer reads self._index_sid / cfg.r_index, never a literal
    # name -- so re-pointing those two fields genuinely moves the contact. Verified by
    # the committed contact p2 inverting which fingertip it sits on (19.0mm off the
    # index -> 16.1mm off the middle on 017_orange) while the grasp span |p1-p2| is
    # preserved (67.8 -> 68.0mm). See models/grasp_finger_config.json for the table.
    for _k, _v in load_finger_config(obj_id, fingers).items():
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


def for_frogger(obj_name: str, arm_geom_names: list,
                obj_clearance_by_geom: dict,
                accel_budget_xyz: tuple, ang_accel_budget_xyz: tuple,
                max_iter: int = 80,
                k_l: float = 0.3,
                sdf_normals: bool = True,
                sdf_surface: bool = True,
                fingers=None,
                finger_obj_geoms=None,
                **overrides) -> GraspConfig3D:
    """The FRoGGeR arm of the benchmark: beta as the SOLE objective.

    FRoGGeR (arXiv 2302.13687) solves

        max_q  l*(q)   s.t.  q_min <= q <= q_max,  l_bar*(q) >= k_l,
                             s(FK_i(q)) = 0,  sigma(o_A, o_B; q) >= d_j

    with l_bar* = m * beta, m = ncols(W), and k_l = 0.3 in their experiments. The
    two structural differences from `for_gws_recommender` are that beta is the ONLY
    objective (they have no IK/align/orient/reg terms competing for the contact
    placement gradient) and that it carries a HARD robustness floor.

    This preset reproduces both within our NLP, and deliberately does NOT attempt
    to reproduce their bilevel LP -- see docs/GWS_IMPROVEMENTS.md items 1-3. The
    benchmark is therefore a comparison of OBJECTIVE STRUCTURE at a shared
    embedding, and the embedding's own convergence (lp_gap) is a known common-mode
    limitation to report, not a difference between arms.

    What is held in common with our arm, so the comparison is about the
    formulation: scene, seeding pool, collision model, execution, and scoring.

    k_l : the normalized floor. Applied as `beta >= k_l / m`, since our `beta` is
        raw and `l_bar* = m * beta`. Passed through to `GraspConfig3D.gws_beta_min`
        so the NLP carries it as a hard constraint. **0.0 disables the floor**,
        which is the configuration to use when measuring how often the floor is
        what rejects a grasp.

    sdf_surface : use FRoGGeR's formulation (7a)-(7e) in full -- q as the ONLY
        decision variable, contacts as the fingertip forward kinematics FK_i(q),
        and (7d) `s(FK_i(q)) = 0` constraining those to the surface.
        **This is the faithful setting and the default.**

        An earlier version gave contacts their own free 3-vectors pinned by the
        same equality. That reproduces (7d)'s ALGEBRA but not its MEANING: a free
        contact reaches the hand only through an IK COST, so zeroing that cost to
        match "beta alone" let the optimizer place contacts the hand never reaches
        (measured 1211/1315/1259 mm fingertip-to-contact at l_bar* = 0.9993). Under
        FK contacts that is structurally impossible -- there is no separate contact
        to drift from. It is also why FRoGGeR needs no IK term in (7a).

        Set False to run their objective inside our patch, which isolates objective
        structure from the parameterization.

    finger_obj_geoms : geom names of the ACTIVE fingers' links, which (7e) allows
        to interpenetrate the target object slightly. Their clearance is set to
        `frogger_finger_obj_margin_m` (negative) unless the caller already gave
        that geom an explicit value. None/empty leaves every clearance positive,
        which is NOT the paper's (7e).

    sdf_normals : take contact normals from the object's SDF gradient rather than
        the quadratic patch, which is FRoGGeR's own formulation (n = -grad s(p))
        and is the better-conditioned estimator here: measured 1.80/2.64 deg
        against the patch's 4.80/4.92 deg on 056_tennis_ball/017_orange
        (FROGGER_BENCH sec 8.5). Under sdf_surface=True there is no patch to take
        a normal from and this is the only available source; it becomes an
        independent axis only when sdf_surface=False.

    NOTE the contact count. A 2-contact pinch has rank(W) = 5 of 6 and cannot
    satisfy l_bar* >= 0.3 meaningfully, so a faithful run needs `fingers` of
    length >= 3. The preset does not force this -- a 2-contact run is legitimate
    as a control -- but a k_l > 0 floor at n=2 will reject nearly everything, and
    that is a property of the configuration, not a result.
    """
    cfg_kw = dict(overrides)

    # beta alone. Every other cost weight is zeroed rather than merely reduced:
    # the point of the comparison is that FRoGGeR has no competing placement
    # objective, and w_ik dominates our gradient by ~3 orders (SOLVER_STATE sec 4).
    cfg_kw.setdefault('w_gws', 1.0)
    cfg_kw.setdefault('w_span', 0.0)     # not part of (7a); ours only
    cfg_kw.setdefault('w_ik', 0.0)
    cfg_kw.setdefault('w_align', 0.0)
    cfg_kw.setdefault('orient_weight', 0.0)
    cfg_kw.setdefault('w_edge_margin', 0.0)
    cfg_kw.setdefault('w_contact_height', 0.0)
    # (7a) is `maximize l*(q)` -- ONE term. w_reg is zeroed with the rest rather
    # than kept as a "harmless posture prior": under frogger_fk_contacts the arm's
    # redundant DOFs are already constrained by (7d) (every fingertip must lie on
    # the surface) and (7e), so a regularizer is not needed to keep q bounded, and
    # including one would make the objective not theirs.
    cfg_kw.setdefault('w_reg', 0.0)

    # beta*n_cols, so w_gws and k_l both keep one meaning across contact counts and
    # cone models (m goes 10 -> 15 from n=2 to n=3). FRoGGeR's l_bar* IS the
    # normalized quantity, so this is the faithful choice, not a convenience.
    cfg_kw.setdefault('gws_beta_scale_ncols', True)

    # The hard robustness floor (7c).
    cfg_kw.setdefault('gws_beta_min_normalized', float(k_l))

    if sdf_surface:
        # FRoGGeR (7a)-(7e) in full: q is the ONLY decision variable, contacts are
        # the fingertip FK, and (7d) constrains those to the surface. Disables the
        # patch, so the paraboloid-dependent machinery (symbolic normals,
        # trust-region edge hinge) has nothing to act on -- turned off explicitly
        # rather than relying on the branch not firing.
        cfg_kw.setdefault('frogger_fk_contacts', True)
        cfg_kw.setdefault('use_quadratic_contact', False)
        cfg_kw.setdefault('quadratic_symbolic_normals', False)
        # The wrench-cone LP (gamma/y/slack) is OUR machinery for a task-specific
        # squeeze force and appears nowhere in (7a)-(7e). Off, so the frogger NLP
        # carries only the variables the paper's does.
        cfg_kw.setdefault('wrench_constraint', False)
        # (7a) is `maximize l*(q)`, with l* a FUNCTION of q -- their inner LP is
        # solved to optimality at every outer iterate. Our embedding only reaches
        # the metric at NLP convergence, which the measured lp_gap shows it often
        # does not. Bilevel is the paper's structure, so it is the default here.
        cfg_kw.setdefault('frogger_bilevel_lp', True)

    # (7e): a NEGATIVE margin d_j on FINGER-OBJECT pairs, which the paper states
    # explicitly. Applied to the ACTIVE fingers' geoms against the target object
    # only; every other pair keeps its positive clearance. Without it a point
    # contact ON the surface and a finite pad sphere with non-negative clearance
    # are mutually unsatisfiable, so this is what makes (7d) and (7e) consistent
    # rather than a relaxation for convenience.
    _fo = cfg_kw.get('frogger_finger_obj_margin_m', -0.002)
    if _fo is not None and finger_obj_geoms:
        # OVERWRITE, not setdefault. clearance_by_geom() pre-populates the active
        # fingers' distal geoms with the DISABLE SENTINEL (-1.0), meaning "no object
        # constraint at all" -- our contact tier, which lets a fingertip pass
        # arbitrarily deep through the object. (7e) is a BOUNDED allowance
        # (`d_j < 0`, "a small amount of interpenetration"), so leaving the sentinel
        # in place would be strictly more permissive than the paper, not equal to
        # it. An explicit caller value still wins via cfg_kw.
        _clr = dict(obj_clearance_by_geom or {})
        for _g in finger_obj_geoms:
            _clr[_g] = float(_fo)
        obj_clearance_by_geom = _clr
    if sdf_normals or sdf_surface:
        cfg_kw.setdefault('gws_sdf_normals', True)

    return for_gws_recommender(
        obj_name, arm_geom_names, obj_clearance_by_geom,
        accel_budget_xyz=accel_budget_xyz,
        ang_accel_budget_xyz=ang_accel_budget_xyz,
        max_iter=max_iter, fingers=fingers, **cfg_kw)

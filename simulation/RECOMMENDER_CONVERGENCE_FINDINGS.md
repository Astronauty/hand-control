# Grasp recommender convergence investigation — findings

Investigation into why the `contact_aware_teleop` grasp recommender NLP
(`GraspPlanner3D` / `MultiStartGraspPlanner3D`) returns `best-effort` / 0-of-N seeds
converged, with `γ_min=—  WF=NO` on the dashboard.

Test scripts (all in `simulation/`):
- `ablate_recommender_constraints.py` — toggle each NLP constraint on/off
- `test_fixed_contacts_wrench.py` — pin contacts at the predefined sites, solve q+wrench
- `test_nonslack_lp_sites.py` — run the standalone LP on the predefined sites
- `test_lp_vs_nlp_weights.py` — LP ground truth + NLP weight sweep

All tests use `obj_red_box` (the only object currently uncommented in the scene) and,
where noted, a real near-box operator pose from `samples.jsonl` (`q_seed` + `obj_qpos`).

---

## Summary of what was ruled IN and OUT

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Contact SEARCH is failing | **Out** | Pinning contacts at the known-good sites still fails |
| Reachability (bad `q_ref`) | **Partial** | Home-pose `q_ref` fails; near-box `q_seed` fixes the wrench-OFF case (3/3, 19 iters) |
| Tier-1 arm collision | **Out** | Dropping it doesn't change convergence (still 0/N) |
| Task wrench too demanding | **Partial** | Only ZERO budget recovered 2/6; even 10× easier task stays 0/6 |
| Missing `3D_minimum_NCF_slack.py` | **Real, but not the convergence cause** | LP warm-start is deliberately disabled (see below); missing module only breaks `verify()`/γ reporting |
| Grasp-axis torque singularity | **Real, fixed** | Non-slack LP: 0/64 corners; zero the grasp-axis torque → feasible (γ≈0.8–8) |
| Wrench-cone numerical conditioning | **Out** | Wrench terms are <4 in the gradient; `gradN_ik` dominates ~1000:1 |
| IK objective over-weighted | **Out** | `w_ik` 0.80→0.008 (100×) changed nothing |
| Structural IK↔wrench-constraint coupling | **Likely** | See conclusion |

---

## Detailed findings

### 1. The missing slack LP (`3D_minimum_NCF_slack.py`)

`grasp_planner_3d.py` imports `min_gamma_for_accel_lp` from `3D_minimum_NCF_slack` in a
`try/except` (grasp_planner_3d.py:93-99). **That module does not exist in the repo** —
neither `3D_minimum_NCF.py` nor `3D_minimum_NCF_soft.py` has the required
`slack_penalty=`/`return_y=` signature. So `_NCF_AVAILABLE = False` and
`min_gamma_for_accel_lp = None`.

Consequences:
- `verify()`'s LP call is guarded (grasp_planner_3d.py:~1636) → **γ_min never computed →
  the `WF=NO  γ_min=—` dashboard symptom**. This is the direct cause of that display.
- The pre-check LP warm-start (grasp_planner_3d.py:~1894) is also skipped.

**BUT** the LP warm-start is deliberately disabled anyway — see the comment at
grasp_planner_3d.py:1267: *"Note: Do not warm-start with LP wrench solutions. It makes
it much worse."* So the missing module does **not** cause the convergence failure; it
only breaks γ reporting. Reconstructing it (as a single-LP slack formulation mirroring
`_embed_wrench_cone_ca`'s `use_slack=True` path) would restore reporting + a ground-truth
oracle, but is not expected to fix convergence.

### 2. Grasp-axis torque singularity (a real bug, now fixed)

The standalone non-slack LP declared the known-good pinned grasp INFEASIBLE with
**0 of 64 corners feasible**. Diagnostic (`test_nonslack_lp_sites.py` + probes):
- Force-only corners are feasible (γ≈0.87 — the side-pinch resists full gravity fine).
- **Every corner carrying any `Tx` (grasp-axis torque) is infeasible.** Both contacts lie
  on the object x-axis with forces spanning ±x, so `p × f` has an identically-zero
  x-component: the pinch geometrically cannot produce torque about the grasp axis.
- The `Tx` demand is only `inertia·ang_accel ≈ 1e-4 N·m` (negligible), but hard equality
  rejects it because the achievable value is *exactly* 0. Since all 64 corners include a
  ±Tx term, all 64 fail.

**Fix applied:** project the grasp-axis component out of the torque budget before building
the wrench corners — the same projection `solve_gamma_live` already does in the live
GRASP path (kinova_leap_pick_place.py:~2911-2915). Implemented in `_run_stage` just
before the `_embed_wrench_cone_ca` call (grasp_planner_3d.py). Verified: the projected
budget becomes `[0, 1.3e-4, 1.3e-4]` and the problem becomes feasible (γ_nlp finite ≈25
instead of None).

Also set `n_normal_relinearize = 0` for boxes (geom_type 6): the face-pin surface
constraint keeps each contact on its seed face, where the box normal is constant, so no
Picard relinearization is needed. (Curved geoms still relinearize.)

### 3. The projection fixed FEASIBILITY but NOT convergence

After the projection, the wrench-ON solve is feasible (γ finite) but STILL does not
converge — `best-effort` at both 120 and 800 iterations, γ_nlp wandering to ~25 vs the
LP's true ~0.8. So feasibility was necessary but not sufficient.

### 4. Cost/constraint scaling analysis (ruled out the conditioning hypothesis)

Instrumented one solve (near-box seed, contacts pinned, wrench ON):

Reference scales:
```
task force  = [0.06, 0.06, 2.52] N   g_ref = 2.52 N
task torque = [0, 1.3e-4, 1.3e-4] N·m   t_ref = 1.9e-4 N·m   (Tx zeroed ✓)
```
The wrench-balance constraint is already row-scaled by `ref6 = [t_ref×3, g_ref×3]`
(grasp_planner_3d.py:655), so the 13,000× force/torque spread is normalized inside the
constraint. That part is fine.

Per-term COST values (final iterate): `Jik=1.17 Jreg=0.69 Jgam=9.91 Jy=2.73 Jslack=0.96`
— reasonably balanced.

Per-term GRADIENT norms (the decisive measurement):
```
          gradN_ik   gradN_gamma  gradN_y  gradN_slack  gradN_total
i=0       1872       0.06         0.002    0            1872
i=100     2454       0.06         0.044    3.7          2454
i=200      259       0.06         0.044    2.7           259
```
**The wrench terms are numerically negligible in the gradient (<4). `gradN_ik` dominates
the total ~1000:1 and never settles** (still 259 at iter 200). The earlier hypothesis
(wrench cone ill-conditioned, swamping the solver) is WRONG — it's the opposite.

Lowering `w_ik` by 100× (0.80 → 0.008) did **not** help — all stayed best-effort at
γ≈24.9. So it is not a simple objective-weight imbalance.

### 5. Formulation differences vs. ConstrainedIKSolver

The `ConstrainedIKSolver` (grasp_control/constrained_ik.py) solves the SAME
fingertips-on-contacts IK and converges well in production, but is a different NLP:

| | ConstrainedIKSolver | GraspPlanner3D NLP |
|---|---|---|
| Decision vars | `q` only (23) | `q`, `p1`, `p2`, `γ`, 630 `y`, (slack) |
| Contacts | fixed numeric targets | decision variables (`p1/p2`) |
| Wrench cone | none | 63-corner embedded cone |
| Collision | exact SDF constraints (pruned) | Tier-1 SDF + surface pin |
| Jacobian | `finite-difference-values` (IPOPT FD) | analytic/AD via CasADi |
| `acceptable_*` | tight (tol 1e-3, viol 1e-5) | loose (obj_change 1e-2, dual disabled) |

Note grasp_planner_3d.py's IPOPT opts comment (lines 164-165): *"the dual residual is
non-convergent when the active set is degenerate (minimax γ with antipodal symmetry)."*
The author already knew the γ-minimax is dual-non-convergent and added `acceptable_*`
fallbacks — so "best-effort" is partly an expected soft exit for the degenerate active set.

A standalone ConstrainedIKSolver run on the same pinned setup ALSO hit
Maximum_Iterations_Exceeded (12mm site error) — but the standalone harness differs from
the live path (target backoff, collision set), so this is not conclusive; the live
ConstrainedIK converges in production. Worth a faithful apples-to-apples rerun.

---

## Conclusion & recommended next step

The convergence failure is a **structural coupling** between the fingertip IK and the
wrench-balance equalities, NOT numerical conditioning and NOT the wrench cone's own
gradient. Adding the 63-corner equality set changes the problem from "put fingertips on 2
points" (converges, 19 iters) to "put fingertips on 2 points that ALSO form a
wrench-balanced grasp for all corners" — and IPOPT's L-BFGS cannot satisfy the dominant
IK gradient while holding the wrench equalities from this seed.

This points at **Option B — decouple the wrench feasibility from the IK/pose solve**:
since the contacts are face-pinned (nearly static), solve wrench feasibility as a
separate LP that constrains only `γ` (a scalar), removing the 630 `y` vars and 63
equality blocks from the NLP. The NLP then does IK + a single γ bound and should
converge like the wrench-OFF case.

Rescaling (Option A) was investigated and ruled out — the gradient imbalance is
objective-vs-constraint coupling, not term weighting.

### Changes already applied (keep — physically correct, low-risk)
- Grasp-axis torque projection in `_run_stage` (grasp_planner_3d.py) — mirrors
  `solve_gamma_live`; makes the wrench feasibility well-posed for antipodal pinches.
- `n_normal_relinearize = 0` for boxes — justified by the face-pin keeping normals constant.

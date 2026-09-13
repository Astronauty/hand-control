# Improving our min-weight (`beta`) machinery

Findings from the 2026-09-13 instrumentation session, written as candidate work items.
Companion to [`FROGGER_BENCH.md`](FROGGER_BENCH.md) (the benchmark and its measurements)
and [`FROGGER_COMPARISON.md`](FROGGER_COMPARISON.md) (component-by-component comparison).

Nothing here is implemented. Items are ordered by expected value against effort.

**Symbols.** `beta` — the min-weight metric: the largest value such that some `alpha` with
`W alpha = 0`, `sum(alpha) = 1` has every component `>= beta`. `l_bar* = m * beta` with
`m = ncols(W)` is FRoGGeR's normalized form, in `[0,1]`. `beta_rep` — what our NLP reports
(`res['gws_beta']`). `beta_rep_converged` — the same LP re-solved to optimality on the same
`W`. `lp_gap = beta_rep_converged - beta_rep`. `beta_true` — the LP re-solved on
SDF-gradient normals at the same contact points. `delta = beta_rep - beta_true`.

---

## 0. The finding these items address

`alpha` and `beta` are `opti.variable()`s in the main NLP, with `W alpha = 0` as an ordinary
constraint, so `beta` equals the min-weight metric only at convergence of the whole NLP.
Measured over 18 solves (FROGGER_BENCH §8.4a):

| term | median abs | source |
|---|---|---|
| `-lp_gap` | **0.00740** | the NLP stopping short of its own LP's optimum |
| surrogate residual (`beta_rep_converged - beta_true`) | **0.00045** | patch normals vs SDF-gradient normals |

`lp_gap > 0` on **18/18** solves including all six converged ones. `||W alpha||`, which the
LP constrains to zero, reaches **1.0e-2** on `017_orange`: the equality itself is
unconverged, not merely the objective loose. A `beta` in that state is not the min-weight
metric of anything — it is a point that has not reached the LP's feasible set.

Re-solving the same `W` repairs three of the four sign disagreements in §8.2, so those
were non-convergence rather than surrogate error.

---

## 1. Constraint scaling on the LP rows (cheapest; do first)

`||W alpha|| = 1.0e-2` against an LP that constrains it to 0 suggests IPOPT is satisfying
the `W alpha = 0` rows to a loose *relative* tolerance while the solve is declared
acceptable. `W`'s entries are wrenches — torque rows carry a moment-arm factor, force rows
do not — so the rows are not commensurate with each other or with the geometric constraints
(metres, radians).

Scale the `W alpha = 0` rows so IPOPT's constraint tolerance binds on them at the same
absolute magnitude as on the geometric constraints, e.g. divide each row by its own norm, or
normalize by a reference wrench as `_embed_wrench_cone_ca` already does per corner
(`(w + s)/ref6 == w_k/ref6`).

A few lines, and it distinguishes *scaling* from *structure* as the cause of `lp_gap`. Run
the `beta_sweep` before and after and compare `resid_Walpha` and `lp_gap` columns.

## 2. Restore `rank(W) = 6`

This is a prerequisite for item 3 and matters independently.

At `n_contacts = 2` with the default PCwF cone, `rank(W) = 5 of 6`. Measured on canonical
antipodal geometry:

| configuration | `m` | `rank(W)` | `sigma_6` |
|---|---|---|---|
| PCwF, `mu_t = 0` (default) | 10 | **5** | 6.9e-34 |
| soft finger, `mu_t = 0.05` | 14 | 6 | 0.060 |
| 3 contacts, off-axis third | 15 | 6 | 0.035 |

`null(W^T) = span{[-1,0,0,0,0,0]}` — the torque about the grasp axis, i.e. exactly the
rank deficiency `project_grasp_axis_torque` exists to handle. Since `nu` enters the KKT
system only as `W^T nu`, `nu` and `nu + c*n` are indistinguishable for any `n` in that null
space (verified: residual 6.3e-32). **The dual is therefore non-unique at n=2.**

The tripod restores rank without a comparability caveat. `gws_soft_finger` also does, but
SOLVER_STATE records it as measured harmful on *raw* `beta`, which is not comparable across
`m: 10 -> 14` (the ceiling `1/m` moves 0.100 -> 0.0714); re-judge it with
`gws_beta_scale_ncols` on if it is revisited.

## 3. Bilevel LP with implicit-KKT gradients

FRoGGeR solves the inner LP to optimality at every outer iterate and obtains
`grad l*(q)` by implicit differentiation of its KKT system, exploiting block structure
(their Prop. 1 / eq. 6): 0.18 ms at `m = 16` on CPU.

**Their Prop. 1 requires unique primal AND dual optima.** Our primal is already a unique
vertex — on canonical geometry the active set has rank 11 against `m+1 = 11` needed, both
symmetric and asymmetric. The dual is not, per item 2. So item 2 is a hard prerequisite,
not an optimization.

Three differences from their setup if we implement it:

1. **Rank first** (item 2).
2. **Solver architecture.** They call the LP directly and hand the gradient to NLopt SLSQP.
   Under CasADi/IPOPT the equivalent is a `Callback` with a custom `get_jacobian` wrapping
   an LP solve and returning the Prop. 1 gradient.
3. **Our patch variables persist.** FRoGGeR differentiates `l*` with respect to `q` alone.
   Ours needs it with respect to `(q, t1, t2, t3)`, since `W` depends on the contacts
   through the patch. A chain-rule extension of `partial_q H3`, not a structural obstacle.

## 4. What `gws_alpha_reg` does NOT fix

Recorded so it is not re-tried on the wrong premise. Adding `-rho*||alpha||^2` makes the
objective strictly concave in `alpha`, adding `2*rho*I` to the `(alpha,alpha)` block of the
KKT Hessian (zero for an LP) and making that system nonsingular **in alpha**.

But the primal is already unique (item 3), and regularizing `alpha` does not change `W`, so
`null(W^T)` and hence `nu`'s indeterminacy are untouched. This matches SOLVER_STATE §11,
which records `alpha_reg` as biasing an already-unique `alpha` while the dual residual never
converges.

`gws_alpha_reg` is not the fix for `lp_gap`. Item 1 or item 2 is.

## 5. Separate normal source from parameterization

Keep `p(t)` on the quadratic patch — the 2-DOF parameterization is not implicated by any of
this — and take `n` from `casadi_normal_fn` evaluated at `p(t)`. Composable CasADi, still
smooth in `t`, still no surface equality constraint; costs one spline evaluation per contact
per iteration.

FROGGER_BENCH §8.5 measures the patch normal at 4.80/4.92 deg against the SDF gradient's
1.80/2.64 deg (vs the analytic normal of a fitted sphere). Note SOLVER_STATE §2's argument
against the SDF concerns its **Hessian**, not its **gradient**, and does not transfer.

Expected gain is small — the surrogate term is ~1/16 of the disagreement — so this is worth
doing as a third benchmark configuration separating formulation from normal source, not as a
fix for our own solver. Deferred until after Phase 2 by decision on 2026-09-13.

## 6. `beta` as an objective is systematically under-weighted

`w_gws = 5.0` sits against `w_ik = 0.70`, and the IK term dominates the gradient
(SOLVER_STATE §4: measured `d(cost)/dz = -579` for ik against `-0.4` for align). FRoGGeR has
`beta` as the sole objective. A term that is outvoted has no obligation to reach its own
optimum, which is a plausible contributor to `lp_gap` being nonzero even on converged
solves.

Compounding this: FROGGER_BENCH §8.4 establishes the surrogate error is **one-sided** — no
perturbation in 2000 random-direction trials raised `beta`. So a `-w_gws * beta` term is
biased low by construction, and its effective weight is below its nominal one.

Raising `w_gws` is not a clean experiment: SOLVER_STATE §11 notes stage selection is by
cost, so changing the scale can select a different stage. `gws_beta_scale_ncols` makes the
term budget-invariant and is the better-posed knob.

## 7. Measure edge-seeking

FRoGGeR names edge-seeking as "the most common failure mode of both methods" and lists
combating it as future work. We have `quadratic_bound_inset` (0.010 m, active),
`quadratic_bound_keep_frac` (0.5), and `w_edge_margin` (**0.0, off**), the last firing only
on divergence-limited bounds so a genuinely flat face is not penalized.

The contact-to-bound distance is already computed, so an edge-proximity distribution per arm
is nearly free. This would convert existing machinery into a measured result on a limitation
the paper explicitly leaves open, and carries an ablation: whether enabling `w_edge_margin`
helps.

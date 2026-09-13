# FRoGGeR and this solver: formulation differences

Companion to [`FROGGER_BENCH.md`](FROGGER_BENCH.md), which specifies the benchmark and
records the `beta` instrumentation results. This file compares the two formulations
component by component, separating what is established by measurement from what is
merely different, and lists improvements each suggests for the other.

Paper: [FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric][paper]
(Li, Culbertson, Ames, et al., IROS 2023). Reference implementation: [alberthli/frogger][code].

[paper]: https://arxiv.org/abs/2302.13687
[code]: https://github.com/alberthli/frogger

**Symbols.** `beta` — the min-weight metric: the largest value such that some `alpha` with
`W alpha = 0`, `sum(alpha) = 1` has every component `>= beta`. `l_bar*` — FRoGGeR's
normalized form, `l_bar* = m * beta` with `m = ncols(W)`, so it lies in `[0,1]` and is
comparable across contact counts. `beta_rep` — what our NLP reports (`res['gws_beta']`).
`beta_rep_converged` — our own LP re-solved to optimality on our own `W`.
`lp_gap = beta_rep_converged - beta_rep`. `beta_true` — the LP re-solved on SDF-gradient
normals at the same contact points. `delta = beta_rep - beta_true`.

---

## 1. Component comparison

| component | FRoGGeR | this solver | assessment |
|---|---|---|---|
| min-weight LP | bilevel; inner LP solved to optimality, gradients by implicit KKT differentiation | single-level; `alpha`/`beta` are NLP variables | **FRoGGeR better**, §2.1 |
| objective | `max l*(q)`, sole term | `-w_gws * beta` among ~10 weighted terms | **FRoGGeR better**, §2.2 |
| robustness floor | hard `l_bar* >= 0.3` | none in the NLP; post-solve `gamma` LP instead | **different**, §3.1 |
| contact parameterization | FK output + SDF surface equality `s(FK_i(q)) = 0` | 2-DOF coordinates on a fitted quadratic patch | **different**, §3.2 |
| surface curvature | finite-differenced SDF Hessian (their §III-C) | fit to mesh vertices with model selection | **ours better**, §4.1 |
| contact normals | `-grad s(p)` | analytic patch normal | **FRoGGeR better**, §2.3 |
| edge-seeking | named as the most common failure mode; future work | trust-region inset, keep-fraction, divergence-limited hinge | **ours, unproven**, §4.2 |
| mesh requirements | watertight; objects that failed Poisson reconstruction were dropped | thin-shell and non-watertight handled directly | **ours better**, §4.3 |
| wrench certification | `k_l = 0.3`, task-agnostic | task-specific disturbance box via `solve_gamma_live` | **different**, §3.3 |
| contacts | 4 (Allegro) | 2 default, 3 slots | **FRoGGeR better**, §2.4 |
| collision | Drake witness points, V-HACD decomposition | bounding sphere vs primitive, plus a baked SDF table | **FRoGGeR better**, §2.5 |

---

## 2. Where FRoGGeR is ahead

### 2.1 The LP is solved to optimality at every iterate

FRoGGeR's `l*(q)` is a genuine function of `q`: the inner LP is solved exactly and
`grad l*(q)` follows from implicit differentiation of its KKT system (their eqs. 4-5). Ours
embeds `alpha`/`beta` as `opti.variable()`s with `W alpha = 0` as an ordinary NLP
constraint, so `beta` equals the min-weight metric only at convergence of the whole NLP.

Measured over the 18-cell sweep (FROGGER_BENCH §8.4a): `lp_gap > 0` on **18/18** solves,
median `|lp_gap| = 0.00740` against a median surrogate residual of `0.00045` — a factor of
**16**. `||W alpha||`, which the LP constrains to zero, reaches `1.0e-2` on `017_orange`, so
the equality itself is unconverged rather than the objective merely loose. Three of the four
sign disagreements in §8.2 are repaired by re-solving the same `W`.

This is the single most consequential difference. It gates the hard constraint (§3.1).

### 2.2 `beta` as the sole objective

Ours carries `w_gws = 5.0` against `w_ik = 0.70`, and the IK term dominates the gradient
(SOLVER_STATE §4: measured `d(cost)/dz = -579` for ik against `-0.4` for align). A term that
is outvoted has no obligation to reach its own optimum, which is a plausible contributor to
`lp_gap` being nonzero even on the six converged solves.

### 2.3 Normal accuracy

FROGGER_BENCH §8.5, against the analytic normal of a sphere fitted to each object's own
visual vertices: SDF gradient 1.80 / 2.64 deg versus patch normal 4.80 / 4.92 deg on
`056_tennis_ball` / `017_orange`. The 2.3-3.0 deg difference is representation error and
favours the SDF gradient, consistent with the paraboloid being a second-order fit evaluated
at a trust-region bound (pinned on 9/9 measured stages, SOLVER_STATE §2).

Note this concerns the patch's *normal*, a derived quantity, not its role as a
parameterization (§3.2). The two are separable: `casadi_normal_fn` evaluated at `p(t)` would
keep the 2-DOF parameterization while taking normals from the SDF.

### 2.4 Contact count

FRoGGeR uses 4 fingers. A 2-contact pinch is rank-5-of-6 and cannot satisfy
`l_bar* >= 0.3`. FROGGER_BENCH §8.6 measures `beta`'s sensitivity as 60x lower at n=3
(`delta` -0.00005 vs -0.00312 on the same object and seed), so contact count affects
`beta`'s reliability more than normal accuracy does.

### 2.5 Collision fidelity

Drake witness points with V-HACD convex decomposition against our bounding-sphere proxy.
Ours is conservative, so it constrains the reachable set more than theirs. This should be
stated as a limitation of the comparison rather than engineered around.

---

## 3. Differences without a measured verdict

### 3.1 Robustness floor versus post-solve certificate

FRoGGeR refuses to return a grasp below `l_bar* = 0.3`, which makes "converged" mean
"certified" — their 99.4% is a feasibility rate, not a termination rate. We have no floor in
the NLP; `verify()` runs a task-specific `gamma` LP afterwards.

Which is better posed is untested. Their own data gives some reason for caution:
`l_bar*` is a noisy success predictor (0.61 (0.49, 0.67) for successes against
0.47 (0.39, 0.60) for failures), so the floor does not cleanly separate outcomes.

### 3.2 Patch parameterization

The quadratic patch gives each contact exactly 2 DOF on a surface that is on-surface by
construction, which removes the surface equality constraint, supplies a trust region, and
keeps the NLP smooth. FRoGGeR instead carries `q` alone plus `nc` surface equalities.

Nothing measured here bears against the parameterization. The §2.3 finding concerns the
normal derived from it, and §8.4a shows the surrogate term is ~1/16 of the disagreement.

### 3.3 Task-specific wrench certification

`solve_gamma_live` certifies against an explicit disturbance box (`m*a` force box, `I*alpha`
torque box, gravity re-datumed about the contact centroid) and returns the internal-force
scale `gamma` needed. FRoGGeR's `k_l` is task-agnostic. A task-specific certificate is
arguably better posed for a known carry task, but the two have not been compared on the same
grasps.

---

## 4. Where this solver is ahead

### 4.1 Curvature from mesh vertices, with model selection

FRoGGeR §III-C estimates `grad^2 s` by finite-differencing SDF gradients along two random
directions at spacing `delta` (their eqs. 10-11), describing it as "coarse" and sensitive to
`delta`, for which they offer the heuristic `delta ~ 10x` mean mesh edge length.

An SDF encodes distance to the whole shape, so its second derivative is global: near any
feature it reports curvature belonging to that feature. Measured on `036_wood_block`, mid-face
on a flat side, comparing the SDF Hessian against our mesh-vertex fit (ground truth
`kappa = 0`):

| sample | SDF-Hessian kappa (1/m) | mesh-fit kappa (1/m) |
|---|---|---|
| 194 | (+0.129, **-3.277**) | (+0.000, +0.000) |
| 458 | (+1.210, **-1.862**) | (+0.000, +0.000) |
| 222 | (-1.088, **+9.051**) | (+0.000, +0.000) |
| 29 | (+0.498, **+1.857**) | (+0.000, +0.000) |
| 368 (near an edge) | (+10.955, +459.736) | (-0.056, +466.251) |
| 611 (near an edge) | (+0.062, +338.104) | (+0.045, +257.060) |

The SDF Hessian reports spurious curvature up to 9.05 1/m on flat faces; the mesh fit returns
exactly zero. Where curvature is real, both detect it. SOLVER_STATE §2 records the downstream
cost: fake curvature of -12.33 on a flat face, sourced from a corner ~50 mm away, collapsed
the trust region to 4.6 mm on a face with ~50 mm of usable surface.

Our fit adds a **plane-versus-quadratic model-selection test** with no FRoGGeR equivalent.
This matters specifically on scanned meshes: a raw quadratic fitted to the block's ~1 mm of
scan relief yields `kappa` scaling as `1/r` with fit radius (-35.5, -15.5, -6.1, -2.4 at
r = 15, 25, 40, 60 mm — no converged value at any radius). Curvature is accepted only when
the quadratic beats a plane by `quad_gain_min` in RMS residual (a parameter of
`_mesh_local_surface_fit_np`, default 0.5, not a `GraspConfig3D` field); below that,
`kappa` is returned as exactly 0. The separation is wide: `017_orange` improves RMS by 91% at every radius with
`kappa` stable at -32.2 across a 4x sweep, while the block's face manages 15-34%.

### 4.2 Edge-seeking machinery (exists; effect not yet measured)

FRoGGeR names edge-seeking as "the most common failure mode of both methods", attributes
it to the `epsilon` metric preferring large moment arms, notes it "yields unstable grasps in
practice", and lists combating it as future work.

We have three mechanisms:

| mechanism | default | kind |
|---|---|---|
| `quadratic_bound_inset` | **0.010 m (active)** | constant keep-back from every trust-region bound |
| `quadratic_bound_keep_frac` | 0.5 | floor on what the inset may consume |
| `w_edge_margin` | **0.0 (off)** | hinge on contacts within `edge_margin_sdf_m` of a *divergence-limited* bound |

`w_edge_margin`'s design point is that it fires only on bounds set by measured SDF
divergence — a real surface boundary — and ignores bounds at `quadratic_t_bound_max`, which
only mean "flat as far as the search looked". A genuinely flat face is therefore not
penalized.

**This is machinery aimed at a limitation FRoGGeR leaves open, not a demonstrated
improvement.** Only the inset is active in the measured configuration, and we have not
measured edge-seeking rates for either method. See §5.3.

### 4.3 Non-watertight and thin-shell meshes

FRoGGeR pruned its object set to watertight meshes via Poisson reconstruction and removed
objects that could not be reconstructed (transparency, thin walls), noting the method works
on non-watertight meshes "of adequate quality" but that the step isolates their results from
mesh quality.

We handle the thin-shell case directly. A cup is two shells around a void that reads as
outside the SDF (measured on `065-a_cups`: +12.20 mm at the volumetric centroid, -2.48 mm
inside the wall), so nearest-surface projection lands on the inner wall. `_outer_pair_t`
enumerates ray crossings and takes the first and last; the surface fit's band is asymmetric
so a 4 mm symmetric band does not reach through a 2-3 mm wall and mix both sheets into one
quadratic. Measured: `065-a_cups` 12 outer / 0 inner after the change, with `017_orange` and
`036_wood_block` unchanged at 14/14.

This widens the object set the benchmark can cover relative to their 43.

---

## 5. Candidate improvements

Ordered by expected value against effort. Items 5.1 and 5.2 are prerequisites for a faithful
FRoGGeR arm.

### 5.1 Converge the embedded LP

`lp_gap > 0` on 18/18 solves is the dominant error term and blocks any hard `beta`
constraint, which requires `beta` correct at every iterate. Options, cheapest first:

1. **`gws_alpha_reg > 0`.** Makes the embedded problem a strictly convex QP in `alpha`, so
   its primal and dual are unique. SOLVER_STATE records it as byte-identical to the plain LP
   at `alpha_reg = 0`, and `gws_alpha_reg` is already a config field. Test whether `lp_gap`
   shrinks.
2. **Raise `w_gws`, or normalize with `gws_beta_scale_ncols`.** If the LP is unconverged
   because its objective is outvoted (§2.2), weight is the lever. Note SOLVER_STATE's caveat
   that changing `w_gws` scale can select a different stage, so this is not a clean A/B.
3. **Bilevel, as the paper does.** Solve the LP exactly per iterate with implicit KKT
   gradients. Correct by construction, and the largest change.

### 5.2 Separate normal source from parameterization

Keep `p(t)` on the patch; take `n` from `casadi_normal_fn` evaluated at `p(t)`. Composable
CasADi, still smooth in `t`, still no surface constraint; costs one spline evaluation per
contact per iteration. Addresses §2.3 without giving up §3.2.

Note SOLVER_STATE §2's argument against the SDF concerns its **Hessian**, not its
**gradient**, and does not transfer. Expected gain is small — the surrogate term is ~1/16 of
the disagreement — so this is worth doing for the FRoGGeR arm and as a third configuration
that separates formulation from normal source, not as a fix for our own solver. Deferred
until after Phase 2 by decision on 2026-09-13.

### 5.3 Measure edge-seeking

FRoGGeR reports it as the dominant failure mode but gives no metric. A contact's distance to
the nearest divergence-limited trust-region bound is already computed, so an edge-proximity
distribution per arm is nearly free. This would convert §4.2 from machinery that exists into
a measured result, and it is the clearest place where this solver could contribute something
the paper explicitly wanted.

It also has a decision attached: `w_edge_margin` is off by default. If edge-seeking is
measurable on our arm, whether enabling it helps is an ablation worth running.

### 5.4 Contribute the curvature comparison

§4.1 is a concrete, measured criticism of a component the paper acknowledges as coarse. It
would strengthen the benchmark write-up to report SDF-Hessian versus mesh-fit curvature on
the shared object set, with the downstream trust-region consequence, rather than only citing
`beta` numbers.

### 5.5 Reconsider the robustness floor once `beta` is trustworthy

A hard `l_bar* >= k_l` is only meaningful once 5.1 lands. When it does, the choice between
their task-agnostic floor and our task-specific `gamma` certificate (§3.3) becomes testable
on the same grasps, and both can be reported.

---

## 6. Why their solve times are ~48x lower than this port's

Measured: the frogger configuration takes ~36-40 s per solve here, against their
reported median of 0.83 s total synthesis. That gap is an artifact of HOW the
collision gradient is obtained, not of the formulation.

### 6.1 Where the time goes

Instrumented on `017_orange` seed 0, n=2, one seed, `max_iter=80`:

| callback | count | total evaluations |
|---|---|---|
| exact geom-vs-object distance | 63 | **504,630** |
| min-weight LP | 1 | 589 |

63 callbacks is one per (arm geom, stage) pair; each was declared `enable_fd`, so
CasADi finite-differences it over all 23 joints -- 24 evaluations per gradient
request, ~334 requests each. `mj_geomDistance` itself is not the problem: measured
sub-microsecond per hull pair. The problem is asking for it half a million times.

### 6.2 What FRoGGeR does instead

Their eq. (8) computes the collision gradient ANALYTICALLY from the witness points
the distance query already returns:

```
sigma(o_A, o_B; q) = (-1)^(Ic+1) ||p_A - p_B||
grad_q sigma       = (-1)^Ic (J_B^T - J_A^T) n_AB
```

with `J_A`, `J_B` the Jacobians at the two witness points and `n_AB` the unit vector
between them. One distance query per constraint per iteration, then a Jacobian
product -- a 24x reduction in queries and no finite differencing at all.

They also cull distant pairs with Drake's broadphase and set those gradients to
zero, and handle the `p_A == p_B` degeneracy (where the direction is undefined) by
reusing the previous `n_AB`, initialized randomly.

### 6.3 This is implementable here

MuJoCo supplies both halves. `mj_geomDistance(..., fromto)` writes the witness
SEGMENT, i.e. both witness points, and `mj_jac(model, data, jacp, jacr, point, body)`
returns the Jacobian at an arbitrary point -- exactly `J_A` and `J_B`. So the
analytic form is one distance call, two `mj_jac` calls and a dot product, replacing
a 24-evaluation finite difference.

Until that lands, **do not compare this port's solve times to the paper's**: the
difference measures our gradient implementation, not their method. The `l_bar*`
values are unaffected, since the finite-differenced gradient converges to the same
place, only slower.

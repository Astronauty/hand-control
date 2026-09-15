# FRoGGeR as a benchmark arm

**Status: Phases 1-3 landed; results pending.** The benchmark harness is
`benchmarks/ycb_grasp/frogger_bench.py`. This file fixes the terms of the comparison and
records what the instrumentation established. Candidate improvements to our own `beta`
machinery are in [`GWS_IMPROVEMENTS.md`](GWS_IMPROVEMENTS.md).

Paper: [FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric][paper]
(Li, Culbertson, Ames, et al., IROS 2023). Reference implementation: [alberthli/frogger][code].

[paper]: https://arxiv.org/abs/2302.13687
[code]: https://github.com/alberthli/frogger

---

## 0a. Reproducibility caveat: numbers here depend on a dirty tree

**Every figure in this document was measured with uncommitted local changes present**,
notably in `simulation/grasp_config_builder.py` (a `w_gws` signature change and an
`n_seeds` change that predate this work). They are not reproducible from a commit hash
alone, and numbers taken at different points in the session are not necessarily
comparable to each other.

Measured instance: the `ours` configuration returns `l_bar* = 0.5877` on `017_orange`
seed 0 with those changes stashed, and `0.7476` with them applied. Both are correct for
their respective trees. Where a comparison spans that boundary it is noted.

Anything paper-bound should be re-measured against a committed tree.

## 0. Relationship to the existing solver

The min-weight metric is already implemented here. `_embed_gws_ca`
(`simulation/grasp_planner_3d.py`) is the paper's LP (2a)-(2d):

```
max_{alpha, beta}  beta    s.t.  W(q) alpha = 0,  sum(alpha) = 1,  alpha >= beta * 1
```

`docs/SOLVER_STATE.md` refers to it as the FRoGGeR min-weight metric. The benchmark is
therefore an ablation between two formulations sharing a metric, not an implementation of
an absent method.

A component-by-component comparison of the two formulations, including where each is ahead
and what improvements each suggests, is in [`FROGGER_COMPARISON.md`](FROGGER_COMPARISON.md).
This section lists only the axes the benchmark varies.

## 1. Points of difference

| | FRoGGeR (7a)-(7e) | this solver |
|---|---|---|
| decision variables | `q` | `q`, patch coordinates `t1/t2/t3`, `gamma`, cone coefficients `y`, slacks `s` |
| contacts | FK outputs, constrained by `s(FK_i(q)) = 0` | 2-DOF coordinates on a fitted quadratic patch with a trust region |
| normals | `-grad s(p)` from the object SDF | patch-symbolic (`quadratic_symbolic_normals`) or Picard-frozen |
| objective | `max l*(q)`, sole term | weighted sum of ~10 terms; `w_gws = 5.0`, `w_ik = 0.70` dominant |
| closure | hard constraint `l_bar* >= k_l = 0.3` | none; `beta` is free, weighted only through its cost term |
| collision | Drake witness points, `sigma >= d_j` | softplus bounding-sphere vs primitive, plus a baked SDF table |
| solver | NLopt SLSQP | IPOPT (SQP available via `--backend`) |
| contacts used | 4 (Allegro) | 2 by default; 3 slots exist |

The question under test is whether the leaner formulation produces better grasps on this
hardware, and under what conditions.

## 2. Two decisions

### 2.1 Complete the `n >= 3` wiring first

A 2-contact pinch has a rank-5-of-6 wrench matrix, which is the reason
`project_grasp_axis_torque` exists. It cannot satisfy `l_bar* >= 0.3` in a meaningful
sense, so a 2-contact FRoGGeR arm would not test the paper's claim. §8.5 gives a second,
measured reason: `beta`'s agreement with the true-normal value is 60x tighter at n=3.

### 2.2 The FRoGGeR arm uses true SDF normals

`beta` is computed from the wrench matrix the NLP assembles, whose normals come from the
quadratic patch rather than the object surface. §8 measures the consequence. The decisive
case is §8.2: on three solves, patch-derived `beta` reports non-closure for grasps that
both independent certificates accept. A hard `l_bar* >= k_l` constraint on that quantity
would reject them.

The paper's own formulation takes normals as `-grad s(p)`, so using the baked
`object_sdf` gradient is both the faithful reading and the defensible one. This makes
surface representation a second measurable axis rather than a confound: the patch exists
because SDF-Hessian curvature was measured to be unreliable (SOLVER_STATE §2: -12.33 on a
flat face, sourced from a corner 50 mm away), which bears directly on FRoGGeR's
finite-differenced mesh Hessian (their `delta ~ 10x` mean edge length). Ours is analytic
from the spline. The disagreement should be reported as a result.

Neither arm's `beta` is treated as a certificate. Both are scored by `_span_margin` (a
geometric test on true normals) and `solve_gamma_live` (the wrench LP), as `verify()`
does.

## 3. Shared components

Held constant across arms: object set, poses, friction, scene (`table_scene`); seeding
pool; collision model; execution (approach, hold, gap gate, squeeze ramp, lift,
transport); and scoring (`_span_margin`, `solve_gamma_live`, epsilon from
`composite_wrench_cone`, `phase_log` outcomes).

Varied: decision variables, objective, closure constraint, normal source.

## 4. Limitations to report

- Robot-side collision uses bounding spheres, not witness points. This is conservative
  relative to Drake and may make the FRoGGeR arm appear more constrained than in the paper.
- The LEAP hand exposes 3 NLP contact slots; the paper uses 4 Allegro fingers. Results are
  3-contact and should be labelled as such.

## 5. Phases

0. Design note.
1. **`n >= 3` wiring** — see §7.
2. **`frogger` planner mode** — DONE as a config preset, `grasp_config_builder.for_frogger`:
   `beta` as the sole objective (every other cost weight zeroed except a `w_reg` posture
   prior), `gws_beta_scale_ncols=True` so `w_gws` and `k_l` both mean one thing across
   contact counts, the hard floor `gws_beta_min_normalized` (FRoGGeR's (7c)), and
   `gws_sdf_normals` for their `n = -grad s(p)`.

   Contact parameterization is FRoGGeR's (7d): free world 3-vectors pinned by
   `s(p) = 0` (`sdf_surface_contact`), not our 2-DOF patch coordinates. Holding
   position on the patch would confine their objective to OUR trust region, and that
   bound is active — the solution sits on it 9/9 measured stages (SOLVER_STATE §2) —
   so it understates their formulation. Measured on `017_orange` seed 0: `l_bar*`
   0.869 on the patch against **0.999** with the surface equality, against a
   theoretical ceiling of 1.0. `--patch-normals` keeps the patch, isolating objective
   structure from parameterization.

   One departure from the paper remains, recorded rather than hidden: the LP embedding
   stays single-level (the bilevel rewrite needs `rank(W) = 6` first — see
   [`GWS_IMPROVEMENTS.md`](GWS_IMPROVEMENTS.md) items 2-3). `lp_gap` is therefore a
   COMMON-MODE limitation of both arms and is reported per solve.

   The OBB palm sampler is not built; both arms share the existing seeding pool, which
   keeps seeding out of the comparison.
3. **`benchmarks/ycb_grasp/frogger_bench.py`** — DONE (plan-only). `--arms ours,frogger`,
   N objects x M seeds, logging `l_bar*`, `l_bar_converged`, `lp_gap`, `delta`,
   `span_margin`, `gamma_min` and solve time. `--k-l 0` isolates the objective change from
   the floor; `--patch-normals` isolates objective structure from normal source.

   Execution scoring is NOT yet wired: the paper's shaky pickup (lift 10 cm in 1 s, hold
   1.5 s, 3 mm sinusoid in all axes from t+0.25 s; failure on >30 deg rotation, >7.5 cm
   deviation, or >60 s synthesis) remains to be added on top of `pick_and_place`'s
   execution path.
4. **Teleop setpoint path** — FRoGGeR as an alternate producer of `obj['q_target']`,
   reusing the existing `_commit_recommended_pose` -> `rebranch` -> `plan` -> GRASP-hold
   contract.

Phases 1-3 produce the result; Phase 4 is the demonstration.

## 6. Paper reference values

43 pruned watertight YCB objects, `mu = 0.7` (optimizer assumed 0.5), uniform density
150 kg/m^3, 20 trials each. Overall: 99.4% converged vs 44.8% baseline; 78.8% pick success
vs 58.0%; epsilon `4.6e-3` vs `2.0e-3`; normalized `l*` 0.58 vs 0.19; 0.21 s per solve,
3 solves, 0.83 s total vs 0.73 s / 50 / 13.8 s. They report `l_bar*` as a noisy predictor
of success: 0.61 (0.49, 0.67) for successes vs 0.47 (0.39, 0.60) for failures. Worth
replicating, as it bears on whether a `beta`-driven objective is well-posed.

This repository holds 85 objects in `assets/ycb_mjcf/`; the watertight-prune list must be
established before any cross-paper comparison is quoted.

---

## 7. Phase 1 (2026-09-13)

### 7.1 The `extra_contacts` call site

SOLVER_STATE §10 listed the `w_gws` call site as not passing `extra_contacts`, leaving
`beta` a 2-contact quantity at `n_contacts >= 3`. This was stale: commit `5fabede` passes
`_gws_extra` and warns when contact 3 has no frame. No code change required; the doc was
corrected.

### 7.2 `verify()` generalized to n contacts

`verify()` built `_pos_v` from `p1`/`p2` and passed `n=2` to the wrench LP, so at
`n_contacts >= 3` it certified the thumb+index pair while the NLP had solved a tripod.

It now covers every contact present, following `solve_gamma_live`'s conventions: the
moment reference is the contact centroid (reducing to the midpoint at n=2), and the
grasp-axis moment/torque projections are gated to `n == 2`. The gate is load-bearing: a
pinch cannot resist torque about the line through its contacts, so projecting states a real
limitation; a third off-axis contact removes that premise, so projecting at n>=3 would make
the certificate conservative against a capability the tripod has.

`verify()` now returns `n_contacts_verified`. A 3-contact `gamma_min` is not comparable to
a 2-contact one, the latter having had a disturbance component projected out. Measured on
`036_wood_block` seed 0: 28.387 N at n=2 (projected) against 38.855 N at n=3
(unprojected).

The n=2 path is unchanged bit-for-bit (`036_wood_block` seed 0, `gamma_min = 28.387`,
`beta` identical to all digits).

### 7.3 `beta` instrumentation

- **`simulation/beta_audit.py`** — rebuilds W from `_geom_normal_np` at the solve's own
  contact points and re-solves the min-weight LP in numpy. Imports the planner's
  `_friction_cone_verts` / `_build_contact_frame_3d` / `_span_margin`, so the cone geometry
  and frame convention cannot drift from the NLP's. Generalizes to n contacts. Read-only.
- **`benchmarks/ycb_grasp/beta_sweep.py`** — plan-only sweep reporting `beta` from both
  sources per solve, with `span_margin` and `gamma_min`.

LP validated against two independently documented facts:

| check | expected | measured |
|---|---|---|
| symmetric antipodal pinch | `alpha = 0.1*ones`, `beta = 1/n_cols` | `beta = 0.100000000000`, `alpha` uniform, `norm(W alpha) = 3.5e-17` |
| 76.3 deg splay at mu=0.6 | closure not achievable | `beta_true = -0.0`, `span_margin = -0.729` |

### 7.4 `constants.py` finger-set resolution

`_finger_set_from_config` and `_slot_roles_from_config` read a `pairings[default]` schema
that `models/grasp_finger_config.json` does not contain, so both always returned the
hardcoded fallback. The fallback equals the file's current default, which concealed the
fault; a `per_object` entry was honoured by the planner (through `load_finger_config`) and
ignored by the executor. Both now read the `fingers`/`per_object` schema. `resolve_fingers()`
was added for call-time resolution, since the module-level constants bind at import and are
why `--fingers` historically steered the planner only.

---

## 8. Reported `beta` against true-normal `beta`

`benchmarks/ycb_grasp/beta_sweep.py`, 6 objects x 3 seeds, plan-only, 80/3 GWS preset,
`thumb,index` (n=2), tabletop scene. `beta_true` is the same LP re-solved on
`_geom_normal_np` normals at each solve's own contact points; the normal source is the only
quantity that differs.

```
object             sd  status        beta_rep   beta_true      delta   span_marg   gamma_min
036_wood_block      0  converged     +0.08970    +0.09282   -0.00312     +2.1231      28.387
036_wood_block      1  converged     +0.08970    +0.09282   -0.00312     +2.1231      28.387
036_wood_block      2  converged     +0.08970    +0.09282   -0.00312     +2.1231      28.387
017_orange          0  best-effort   +0.08397    +0.09232   -0.00835     +2.0303       1.869
017_orange          1  best-effort   +0.08397    +0.09232   -0.00835     +2.0303       1.869
017_orange          2  best-effort   +0.08497    +0.08881   -0.00384     +1.9295       1.751
014_lemon           0  best-effort   +0.07630    +0.05733   +0.01897     +1.1977       1.098
014_lemon           1  best-effort   -0.00926    +0.05383   -0.06309     +1.2973       2.025
014_lemon           2  best-effort   +0.07729    +0.08559   -0.00829     +1.8432       1.089
056_tennis_ball     0  best-effort   -0.05220    +0.07913   -0.13132     +1.5650       2.375
056_tennis_ball     1  best-effort   -0.05220    +0.07913   -0.13132     +1.5650       2.375
056_tennis_ball     2  best-effort   -0.05220    +0.07913   -0.13132     +1.5650       2.375
009_gelatin_box     0  best-effort   +0.07438    -0.00000   +0.07438     +0.1626      INFEAS
009_gelatin_box     1  converged     +0.07912    +0.07945   -0.00033     +2.1342       4.039
009_gelatin_box     2  converged     -0.06319    -0.00000   -0.06319     +0.6033      INFEAS
061_foam_brick      0  converged     +0.08855    +0.09512   -0.00658     +2.1947       1.092
061_foam_brick      1  best-effort   +0.07948    +0.09338   -0.01390     +2.1963       1.127
061_foam_brick      2  converged     +0.08056    +0.09764   -0.01708     +2.1964       1.073
```

### 8.1 Direction of the error

15/18 solves have `delta <= 0`: the reported `beta` is below the true-normal value. The
optimistic case that `_embed_gws_ca`'s docstring is written around occurs once
(`009_gelatin_box` sd0).

A downward-biased `beta` is a weak objective rather than an unsatisfied certificate: the
solver ascends a surrogate that reads below the true value, so `w_gws` does not carry the
effective weight its tuning assumed.

### 8.2 Sign disagreements

| object | sd | `beta_rep` | `beta_true` | span_margin | wrench certificate |
|---|---|---|---|---|---|
| `014_lemon` | 1 | -0.00926 | +0.05383 | +1.2973 | feasible, gamma = 2.03 |
| `056_tennis_ball` | 1 | -0.05220 | +0.07913 | +1.5650 | feasible, gamma = 2.38 |
| `056_tennis_ball` | 2 | -0.05220 | +0.07913 | +1.5650 | feasible, gamma = 2.38 |
| `009_gelatin_box` | 0 | +0.07438 | -0.00000 | +0.1626 | INFEASIBLE |

The first three report non-closure for grasps both independent certificates accept. Under a
hard `l_bar* >= 0.3` constraint on patch-derived normals, these would be rejected as
infeasible. The fourth is the optimistic case: `beta = +0.074` on 62.4 deg splay contacts
whose wrench LP is infeasible.

### 8.3 Dependence on solver convergence

| status | n | median abs(delta) | max abs(delta) |
|---|---|---|---|
| `converged` | 7 | 0.00312 | 0.06319 |
| `best-effort` | 11 | 0.01897 | 0.13132 |

Every disagreement above 0.02 occurs on a best-effort solve. SOLVER_STATE §11 establishes
that these are IPOPT cycling under an indeterminate dual rather than truncation, so a
component of the disagreement is attributable to the iterate the cap lands on.

`009_gelatin_box` is the seed-starved object of SOLVER_STATE §9. Both of its failing cells
are geometric rather than numerical: 62.4 deg and 87.7 deg splay, against 175.4 deg on its
one good seed. `beta_true = -0.0` on both, in agreement with the geometry.

### 8.4 Mechanism of the downward bias

The bias is a property of the LP, not of the patch's particular errors. `beta` is the
optimal value of a maximization whose feasible set is determined by `W alpha = 0`.
Perturbing the normals rotates the primitive wrench columns; the optimum of the perturbed
problem is generically below that of the unperturbed one, and the true-normal configuration
is the reference point.

Measured on an antipodal pinch at mu=0.6, applying an equal-magnitude tilt in a uniformly
random direction to both normals (400 trials per row):

| tilt (deg) | mean `beta` | median | max | fraction below `beta_true` |
|---|---|---|---|---|
| 0 (reference) | 0.10000 | 0.10000 | 0.10000 | — |
| 1 | 0.09684 | 0.09677 | 0.09763 | 100.0% |
| 2 | 0.09363 | 0.09348 | 0.09501 | 100.0% |
| 5 | 0.08381 | 0.08335 | 0.08761 | 100.0% |
| 9 | 0.06934 | 0.06859 | 0.07679 | 100.0% |
| 15 | 0.04529 | 0.04396 | 0.05805 | 100.0% |

No perturbation in 2000 trials raised `beta`. The effect persists away from the arithmetic
ceiling, so it is not solely a boundary artifact of the symmetric optimum (5 deg tilt):

| baseline splay | `beta_true` | ceiling `1/n_cols` | mean `beta` | fraction below |
|---|---|---|---|---|
| 180 deg | 0.10000 | 0.1000 | 0.08376 | 100.0% |
| 160 deg | 0.07502 | 0.1000 | 0.05912 | 93.5% |
| 140 deg | 0.04477 | 0.1000 | 0.02854 | 90.5% |
| 125 deg | 0.01602 | 0.1000 | 0.00349 | 89.2% |

The mechanism accounts for the sign but not, on its own, the full magnitude. SOLVER_STATE
§2 reports a median normal error against the true SDF of 2.02 deg on `017_orange` under
analytic patch normals (9.22 deg frozen). Reconstructing that solve's geometry from its
`span_margin` (169.5 deg splay at mu = 2.0) and applying a 2.02 deg random tilt predicts
`delta = -0.0017` (mean over 600 trials), against a measured -0.0084. The rows above are at
mu = 0.6 and do not transfer directly; friction coefficient changes the cone aperture and
hence the sensitivity.

Two of the three candidate explanations for the residual have been tested and rejected.

**A systematic tilt does not account for it, and has the opposite sign.** A quadratic patch
on a locally convex surface tilts its normal in a consistent direction as the contact slides
outward, so both contacts of an antipodal pair tilt coherently rather than independently. At
2.02 deg, comparing the three models on the reconstructed `017_orange` geometry:

| model | mu = 0.6 | mu = 2.0 |
|---|---|---|
| random direction | -0.00562 | -0.00165 |
| coherent, toward antipodal | **+0.00497** | **+0.00144** |
| coherent, away from antipodal | -0.00513 | -0.00146 |

A coherent tilt toward antipodal raises `beta`. Curvature-driven correlated patch error
would therefore bias `beta` upward, not explain a measured -0.0084.

**Off-surface evaluation does not account for it.** The SDF gradient is a surface normal
only near the zero level set. Measured at the solved contacts, `|s|` has median 0.37 mm and
max 2.97 mm, and the largest disagreements do not track it: `056_tennis_ball` has the
largest `delta` (-0.131) with contacts 0.47-0.58 mm off-surface, while `009_gelatin_box` sd0
sits furthest off-surface (2.68/2.97 mm) with `delta` of the opposite sign.

The remaining candidate — that `beta_rep` is not the optimum of its own LP — is confirmed,
and it dominates. See §8.4a.

Two conditions concentrate the error at the solved contacts. First, the solution sits at a
trust-region bound on 9/9 measured stages (SOLVER_STATE §2), which is where a paraboloid
departs furthest from the fitted surface. Second, `beta` depends on the surrogate's
gradient rather than its position, and the trust region is sized by a position tolerance
(`sdf_err_tol`, 4 mm), which does not bound the normal error.

The optimistic cases are not excluded by this account. The bias is generic in the
perturbation direction, but a perturbation that happens to align the wrench columns more
favourably than the true geometry can raise `beta`; `009_gelatin_box` sd0 is such a case,
and it occurs where the true configuration is near-degenerate (62.4 deg splay,
`beta_true = -0.0`), so the reference value is at the bottom of its range.

### 8.4a The dominant term is a stopping artifact, not surrogate error

`alpha` and `beta` are IPOPT decision variables inside the main NLP, not a nested LP solved
to optimality. `audit_embedded_lp` re-solves the min-weight LP on the NLP's OWN wrench
matrix — patch normals unchanged — giving **`beta_rep_converged`**, and
**`lp_gap = beta_rep_converged - beta_rep`**. Since W is held fixed, `lp_gap` isolates
non-convergence from surrogate error. This decomposes `delta` exactly:

```
delta = (beta_rep - beta_rep_converged) + (beta_rep_converged - beta_true)
      =      -lp_gap                    +      surrogate residual
```

Over the 18-cell sweep:

| term | median abs | interpretation |
|---|---|---|
| `-lp_gap` | **0.00740** | the NLP stopping short of its own LP's optimum |
| surrogate residual | **0.00045** | patch normals vs SDF-gradient normals |

The stopping term is **16x** the surrogate term. `lp_gap > 0` on **18/18** solves, including
all six converged ones (median 0.00321; best-effort median 0.01099). The equality residual
`||W alpha||`, which the LP constrains to zero, reaches 1.0e-2 on `017_orange` — direct
evidence the constraint itself is unconverged rather than the objective merely being loose.

Re-solving repairs three of the four sign disagreements in §8.2:

| object | sd | `beta_rep` | `beta_rep_converged` | `beta_true` | resolved |
|---|---|---|---|---|---|
| `014_lemon` | 1 | -0.00926 | +0.06420 | +0.05383 | yes |
| `056_tennis_ball` | 0/1/2 | -0.05220 | +0.08213 | +0.07913 | yes |
| `009_gelatin_box` | 0 | +0.07438 | +0.08822 | -0.00000 | no |

The three false rejections were non-convergence, not the patch. The one false acceptance
(`009_gelatin_box` sd0) is genuine surrogate error: re-solving moves `beta` further from the
geometry, because the patch normals there describe a 62.4 deg splay configuration as
closable when it is not.

This reorders the conclusions. §8.1's downward bias is predominantly a stopping artifact;
§8.4's perturbation analysis explains the surrogate term, which is real but roughly 1/16 the
magnitude. The patch quadratic's role as a 2-DOF parameterization is not implicated.

### 8.5 Accuracy of the two normal estimators

`beta_true` is computed from `_geom_normal_np`, the gradient of the baked SDF B-spline.
`beta_rep` is computed from the quadratic patch's analytic normal. Both are estimators of
the same quantity, and the patch is NOT the noisier of the two by construction: its normal
field is a low-degree polynomial in `t` (`_quadratic_inward_normal_ca`), exactly
differentiable, and its curvature aggregates a neighbourhood of mesh vertices. Smoothness
and accuracy are independent properties; the relevant comparison is accuracy.

Measured against the analytic normal of a sphere least-squares fitted to each object's own
visual vertices. `056_tennis_ball` and `017_orange` are near-spherical, and the fit residual
bounds how much of each error is the object's true non-sphericity — a term common to both
estimators — rather than representation error:

| object | sphere fit residual (median / p95) | SDF-gradient normal | patch normal |
|---|---|---|---|
| `056_tennis_ball` | 0.115 / 0.560 mm | 1.80 deg | 4.80 deg |
| `017_orange` | 0.377 / 1.289 mm | 2.64 deg | 4.92 deg |

The difference, 2.3-3.0 deg, is representation error and favours the SDF gradient. This is
consistent with the paraboloid being a second-order fit evaluated at a trust-region bound
(SOLVER_STATE §2 records the solution as pinned to a bound on 9/9 stages), which is where a
second-order surrogate departs furthest from the surface it was fitted to.

Note that the two estimators read different surface representations: the patch is fitted to
mesh **visual vertices**, while `_geom_normal_np` evaluates the **baked SDF spline**. The
SDF gradient is not high-frequency noise — averaging it over a 5 mm neighbourhood, the
smoothing a quadratic fit performs, displaces it by a median of 0.03-0.17 deg (p90 <= 1.03
deg) across four objects — but that is a statement about its smoothness, and the table above
is the statement about its accuracy.

`beta_true` is therefore the better-conditioned of two estimates, not a ground truth. The
claims in §8.1-§8.3 should be read as "relative to the SDF-gradient normal", which is also
the quantity FRoGGeR's formulation uses.

### 8.6 Dependence on contact count

`036_wood_block` seed 0, varying only the finger list:

| fingers | n | `beta_rep` | `beta_true` | delta | `gamma_min` |
|---|---|---|---|---|---|
| `thumb,index` | 2 | +0.08970 | +0.09282 | -0.00312 | 28.387 (projected) |
| `thumb,index,middle` | 3 | +0.03660 | +0.03665 | -0.00005 | 38.855 (unprojected) |

Agreement is 60x tighter at n=3. With two contacts the wrench hull is rank-5-of-6, so the
LP optimum is unconstrained in the unspanned direction and a small normal error displaces
`beta` substantially. A third off-axis contact spans that direction, reducing the
sensitivity.

This localizes the unreliability to rank-deficient configurations. The formulation FRoGGeR
proposes (>= 3 contacts, hard `l_bar* >= k_l`) lies in the regime where `beta` is
well-behaved; the 2-contact default does not.

Raw `beta` is lower at n=3 because its ceiling is `1/n_cols` and n_cols goes 10 -> 15.
Compare `beta_true_scaled` (`beta * n_cols`) across contact counts. `gamma_min` is likewise
not comparable across n — see §7.2.

### 8.7 Determinism

Four full 18-cell sweeps. Runs 1, 3 and 4 are byte-identical (md5 `29edc6c1ad41` over the
table body). Run 2 differed in two cells (`056_tennis_ball` sd0, `061_foam_brick` sd2).

Run 2 was started while run 1 was still executing, so two sweeps shared the machine for its
duration. The two affected cells fall in the last two objects of the sweep order and are
both best-effort solves, the regime of §8.3. Re-running both cells in isolation, and again
in a different object grouping, reproduces the runs-1/3/4 values.

SOLVER_STATE §12's determinism claim holds, including across a multi-object sweep in one
process. The operational constraint is narrower: do not run two sweeps concurrently, and do
not modify planner source while one is in flight. Treat any overlapped run as void.

---

## 9. Benchmark results, plan-only (2026-09-13)

`benchmarks/ycb_grasp/frogger_bench.py`, 6 objects x 3 seeds, tripod
(`thumb,index,middle`), `k_l = 0.3`, 80/3 preset. Both arms share scene, settle,
seeding pool, collision model, solver backend and scoring; they differ in objective
structure, robustness floor, contact parameterization and normal source.

`l_bar* = n_cols * beta` (ceiling 1.0). `l_bar_converged` is the same LP re-solved to
optimality on the same `W`. `lp_gap = l_bar_converged - l_bar` in normalized units.

### 9.0 RETRACTION: the §9.2 numbers do not describe reachable grasps

**The `l_bar*` comparison in §9.2 is invalid as published and is retracted.** Found by
rendering the planned pose (`seed<N>_planned.png`), which is why the artifact work
mattered: the scored table alone did not reveal it.

`for_frogger` zeroes every cost term except `beta` and a small `w_reg`, including
**`w_ik = 0.0`**. `w_ik` is the term tying each FINGERTIP to its assigned CONTACT POINT.
With it at zero, nothing in the NLP connects the hand to the contacts, so the optimizer
places contacts that maximize `beta` on the object while the arm stays wherever `w_reg`
leaves it.

Measured on `017_orange` seed 0, tripod, fingertip-to-assigned-contact distance:

| arm | thumb | index | middle | reported `l_bar*` |
|---|---|---|---|---|
| ours | 19.1 mm | 17.4 mm | 20.6 mm | +0.5877 |
| frogger | **1211.5 mm** | **1314.7 mm** | **1259.4 mm** | +0.9993 |

The contacts themselves are well placed (verified on the object surface, 62-119 mm
apart). The grasp is simply not one any hand is performing: the render shows the hand out
of frame entirely while `beta` reports near-perfect closure.

So `l_bar* ~ 1.0` for the frogger arm measures *contact placement in isolation*, not a
grasp. It is unsurprising that an unconstrained-by-kinematics contact optimizer beats one
that must also satisfy reachability; that is not the comparison this benchmark intends.

**Why the error was possible.** FRoGGeR does not need an IK cost term because its
constraint (7d) is `s(FK_i(q)) = 0` — the fingertip FORWARD KINEMATICS are what lie on
the surface, so reachability is structural. Our NLP instead carries contact positions as
independent variables and *couples* them to the hand through `w_ik`. Dropping `w_ik` to
mimic "beta as the sole objective" removed that coupling, which their formulation never
had to state as a cost because it is built into their variables.

**FIXED** by `GraspConfig3D.frogger_fk_contacts`: contacts ARE the fingertip FK
expressions, `q` is the only decision variable, and (7d) constrains `FK_i(q) - r_tip*n`
to the surface. Drift is then structurally impossible rather than penalized.

Measured after the fix, `017_orange` seed 0, tripod:

| | tip->contact | `l_bar*` | `|s|` at the pad surface |
|---|---|---|---|
| before (free contacts, `w_ik = 0`) | 1211 / 1315 / 1259 mm | +0.9993 | — |
| after (FK contacts) | **0.0 / 0.0 / 0.0 mm** | **+0.5800** | 0.006 / 0.027 / 0.076 mm |

Zero by construction, and `l_bar*` falls to +0.5800 against ours at +0.5877 — the two are
now comparable quantities. (7d) verified across three objects: pad surfaces land within
~0.03 mm of the object, with one 1.3 mm outlier on `036_wood_block` where the single-step
pad-offset approximation is weakest. The planned-pose render shows a real tripod on the
object.

Every `l_bar*` in §9.2 predates this and remains retracted.

Unaffected by this: §8 in full (it concerns `beta` computed at contacts from a single
solver configuration, with no cross-arm claim), and the `lp_gap`/`resid_Walpha`
observations below, which are properties of the shared embedding.

### 9.1 Read `resid_Walpha` before `l_bar*`

The min-weight LP constrains `||W alpha|| = 0`. A solve that exits with that residual
nonzero has a reported `beta` that is **not a min-weight value at all** — it is an
infeasible iterate, and `l_bar*` computed from it is meaningless in either direction.

At a `1e-4` threshold, **9 of 36 solves are LP-infeasible** (ours 9/18 feasible,
frogger 11/18). The extreme cases: `056_tennis_ball` ours sd1/sd2 report
`l_bar* = -1.0813` with `||W alpha|| = 1.4e-2`, and `009_gelatin_box` frogger sd0
reports `l_bar* = +0.8959` with `||W alpha|| = 6.1e-1` while its converged value is
`-0.0000`. The wrench certificate independently rejects that last one (`wf=False`),
which is the certificate doing its job on a grasp the NLP should not have offered.

**This is common-mode**, a property of the shared single-level embedding
(GWS_IMPROVEMENTS.md §0-§1), not of either formulation. It affects both arms and is the
single largest threat to this table's validity. Report `l_bar_converged` alongside
`l_bar*`, and treat `resid_Walpha > 1e-4` rows as unscored.

### 9.2 Summary

| | ours | frogger |
|---|---|---|
| median `l_bar*` (LP-feasible rows only) | +0.5762 | **+1.0000** |
| median `l_bar_converged` (all rows) | +0.6871 | **+0.9983** |
| LP-feasible (`resid_Walpha <= 1e-4`) | 9/18 | 11/18 |
| wrench-feasible (`verify()`) | 16/18 | 17/18 |
| IPOPT converged | 8/18 | 10/18 |
| median `gamma_min` (N) | 1.863 (16) | **1.214** (17) |
| median solve time | 2.8 s | **2.0 s** |

**Retracted — see §9.0.** The frogger arm's fingertips are ~1.2 m from its own contacts,
so these `l_bar*` values describe contact placement without reachability, not grasps. The
solve-time and `gamma_min` columns are affected for the same reason: an arm that need not
reach its contacts has a smaller problem to solve. Retained only as a record of what was
run.

### 9.3 The parameterization matters, and by how much

The same benchmark with the frogger arm confined to OUR patch (`--patch-normals`,
preserved as `fb_patchpos`) gives median `l_bar* = +0.8723` against **+0.9990** under
FRoGGeR's own surface equality. Holding position on the patch understates their
formulation by ~0.13 in normalized units, because the trust region is an active bound
(SOLVER_STATE §2, pinned 9/9 stages).

Both configurations are worth reporting: the patch-position arm isolates objective
structure, the surface-equality arm tests the full formulation.

### 9.4 What is not yet measured

- **Execution.** Plan-only. The paper's shaky-pickup protocol is not wired, so nothing
  here speaks to pick success — which is FRoGGeR's headline claim and the quantity
  `l_bar*` is only a proxy for. Their own data has `l_bar*` as a noisy success
  predictor (0.61 vs 0.47 for successes vs failures).
- **The floor's contribution.** `--k-l 0` would separate the objective change from the
  hard constraint. The 17/18 vs 16/18 wrench-feasibility difference is the one number
  where the floor plausibly does visible work.
- **Fairness of the gamma comparison.** Lower `gamma_min` is better, but the two arms
  place contacts differently, so this compares grasps, not solvers, and is not
  normalized for contact separation.

---

## 10. Faithfulness work (2026-09-13, in progress)

The frogger arm reproduced FRoGGeR's algebra while retaining this repo's variable
structure and conveniences. Each departure is now either removed or recorded.

| (7a)-(7e) requires | was | now |
|---|---|---|
| `q` the only decision variable; contacts are `FK_i(q)` | free contact variables + IK cost | `frogger_fk_contacts` |
| (7a) is `max l*(q)`, one term | `w_gws` weighted, plus `w_reg = 0.03` | `w_reg = 0.0`; `beta` sole term |
| no `gamma`/`y`/slack variables | wrench-cone LP active | `wrench_constraint = False` |
| (7d) `s(FK_i(q)) = 0` | patch/trust-region, or free vector | equality at `FK - r_tip*n` |
| (7e) `d_j < 0` on finger-object pairs | positive clearance, or our disable sentinel | `frogger_finger_obj_margin_m = -0.002` |
| bilevel LP, `grad l*` by implicit KKT | single-level embedding | `frogger_bilevel_lp` |

### 10.1 The bilevel LP is implemented and its gradient verified

`_MinWeightLPCallback` solves the min-weight LP to optimality at every outer iterate and
supplies `d beta / d W_{ij} = -(nu_i * alpha_j)` from LP duality, `nu` being the multiplier
on the `W alpha = 0` rows. This is the quantity their eq. (6) produces; it is obtained from
duality rather than by assembling and pseudo-inverting `Omega`, which is identical wherever
the dual is unique.

Verified against central finite differences on the solved tripod geometry: 9 of 12 sampled
entries agree to ~1e-11. The 3 that do not are exactly the degenerate ones — 9 of 15
`alpha` components tie at `beta`, and there forward and backward differences themselves
disagree (-0.0299 vs -0.0340) with the analytic value equal to one of them. That is the
almost-everywhere differentiability the paper's Rademacher remark describes, not an error.

**`rank(W) = 6` holds on a real tripod without soft fingers** (measured on the solved
`017_orange` grasp: rank 6, `sigma_6 = 0.063`), so Prop. 1's unique-dual condition is met
by contact count alone at `n >= 3`.

### 10.2 Both faithfulness changes currently REGRESS the solve, and are not yet usable

Isolated on `017_orange` seed 0, tripod:

| configuration | `beta` |
|---|---|
| embedded LP, no (7e) — the previously reported working arm | +0.0405 |
| embedded LP + (7e) | +0.0172 |
| bilevel LP, no (7e) | +0.0503 |
| **bilevel LP + (7e)** | **-0.0000** |

The combination fails: the returned grasp puts all three contacts on the same side
(pairwise normal dots +0.914 / +0.966 / +0.931), which is genuinely non-closure, and
`verify()` rejects it. `Maximum_Iterations_Exceeded` persists at `max_iter = 400` and the
outcome is unchanged with `k_l = 0`, so it is neither an iteration budget nor the floor.

The callback itself is not at fault: CasADi differentiates it correctly (90/90 Jacobian
entries nonzero, norm 1.98) and its value matches `beta_audit` to 1e-9.

**Do not use `--arms frogger` for reported numbers until this is resolved.** The most
likely cause is that (7e)'s bounded negative margin and the bilevel gradient interact
through the seeding: with interpenetration permitted, a same-side seed is now feasible,
and `beta` alone — with `w_ik = 0`, `w_align = 0` — supplies no term that prefers opposed
contacts from a same-side start. FRoGGeR's own sampler addresses this by aligning the palm
with the object's OBB axes and setting the fingertip span from the bounding box, which this
benchmark has not implemented (§5, phase 2 note). That is the next thing to build.


---

## 11. Sampler wired; the frogger arm is still blocked (2026-09-13)

`frogger_bench` now seeds the frogger arm from the OBB sampler
(`simulation/obb_sampler.py`, their App. B-C) rather than from HOME, via
`_frogger_seed`. `ours` keeps HOME, which is what the tabletop benchmark has always
characterized. Seed provenance is recorded per row (`seed_source`, `seed_draw`,
`seed_seg_dist_mm`, `seed_straddles`).

### 11.1 The sampler works; it was not the blocker

The seed is good. On `017_orange` seed 0 the accepted draw straddles the object with
a fingertip-segment distance of **0.27 mm**, and the sampler straddles on 60-100% of
reachable draws across four objects. But the solve still returns `l_bar* = -0.0`.

Isolated with that seed held fixed:

| configuration | `beta` |
|---|---|
| embedded LP, no (7e) | +0.0549 |
| bilevel LP, no (7e) | +0.0407 |
| embedded LP + (7e) | +0.0198 |
| **bilevel LP + (7e)** | **-0.0000** |

Identical ordering to the HOME-seeded run in §10.2, so the seed was never the cause.
Each component degrades `beta` independently and the combination zeroes it.

### 11.2 What is ruled out

- **The seed** (§11.1, straddling at 0.27 mm).
- **The hard floor.** `beta = -0.00000` is identical at `k_l` = 0.3, 0.1 and **0.0**,
  and at margins -0.002 and -0.0005. A constraint that is off cannot be the binding
  one.
- **The LP callback.** Embedded in a standalone `Opti` problem it solves correctly
  (12 evaluations, `beta = 0.046154` recovered), and against finite differences its
  gradient matches to ~1e-11 away from degenerate vertices.
- **Iteration budget** (unchanged at `max_iter = 400`) and **contact count**
  (n=2 pairs behave as n=3).

### 11.3 What that leaves

The composition inside `_run_stage`, not any single component: most likely the
interaction between the FK-contact parameterization and (7e)'s negative clearance.
With `frogger_fk_contacts` the contacts ARE the fingertips, and (7e) simultaneously
permits those same fingertips to penetrate the object, so the surface equality (7d)
and the clearance constraint now act on the same geometry in opposite directions.
That is a real modelling question in their formulation as ported here, not a bug in
one of the pieces, and it is where the next session should start.

**`--arms frogger` remains unsuitable for reported numbers.** The `ours` arm is
unaffected throughout.


---

## 12. Root cause of the frogger regression: our sphere proxy, not their formulation

§11.3 guessed that the surface constraint and the interpenetration allowance conflict
"as a modelling question in their formulation". That was wrong. **FRoGGeR does not have
this conflict, because it never approximates the fingertip.**

### 12.1 The arithmetic

| quantity | value |
|---|---|
| fingertip PAD radius (what the surface constraint uses) | 19.4 mm |
| fingertip COLLISION sphere, `geom_rbound` | 23.8 mm |
| interpenetration we allowed | 2.0 mm |

The surface constraint pins the tip CENTRE 19.4 mm from the object. The collision
constraint, working on the 23.8 mm sphere, demands the centre stay at least
23.8 - 2.0 = 21.8 mm out. **Jointly infeasible by 2.4 mm**, so the solver satisfies
neither and `beta` collapses to exactly 0 -- which is why it was invariant to `k_l`
(0.3, 0.1, 0.0) and to the margin magnitude.

The 4.4 mm gap is a known repo defect in a new place: SOLVER_STATE §10 already records
`geom_rbound` over-reporting the LEAP pad, being the bounding sphere about the mesh
frame origin rather than the pad.

### 12.2 What FRoGGeR actually does (App. B-F)

- **Exact collision geometry.** Drake witness points on V-HACD convex decompositions --
  the true fingertip shape against the true object. "Pad touches the object" and
  "fingertip may sink in" therefore refer to the SAME surface and cannot contradict.
- **Interpenetration allowance: 3 mm** for fingertip/object pairs. We used 2 mm.
- **Every other pair: 1 mm** minimum safety margin. Ours are larger and per-geom.
- **Contact point: a FIXED point on each fingertip**, at 60 degrees tilted toward the
  palm from the very tip, chosen so the forward kinematics are fixed. We instead
  compute `site - r_tip * n_sdf`, which moves with the SDF normal.
- Constraint tolerances: joint 1e-2, surface contact 5e-4, collision 1e-3.

### 12.3 Consequence for the port

The bounding-sphere collision proxy is OUR approximation, adopted so CasADi can
differentiate the distance (see `constrained_ik`'s module docstring). It is
conservative and harmless when contacts are free variables held off the surface by an
IK cost. It becomes contradictory the moment the contacts ARE the fingertips and must
lie ON the surface, which is exactly what the faithful port does.

`constrained_ik` already uses `mj_geomDistance` (exact, guarded by a bounding-sphere
lower bound) rather than the sphere proxy, so an exact path exists in this repo.

---

## 13. n=2 sweep, both configurations (2026-09-13)

6 objects x 3 seeds, `thumb,index`, with the OBB sampler seeding the frogger
configuration and HOME seeding ours. **Dirty tree (§0a).**

| | ours | frogger |
|---|---|---|
| median `l_bar*` | **+0.8879** | +0.2968 |
| wrench-feasible | **16/18** | 10/18 |
| median `\|lp_gap\|` | 0.00090 | **0.00000** |
| median solve time | **2.4 s** | 34.5 s |

The frogger configuration is **worse on this benchmark as ported**, not better. Two
columns should not be read as method differences:

- **`lp_gap` = 0 exactly** is the bilevel LP doing what it is for: the inner problem is
  solved to optimality at every iterate, so there is no optimality gap to report. This
  is a genuine advantage of their formulation and the one clean result here.
- **34.5 s vs 2.4 s** measures our finite-differenced collision gradient, not their
  method. See FROGGER_COMPARISON §6: 504,630 distance evaluations against their
  analytic `grad sigma = (-1)^Ic (J_B^T - J_A^T) n_AB`, which needs one query per
  constraint per iteration.

### 13.1 The failures are in the solve, not the seed

7 of 18 frogger cells return exactly `l_bar* = 0`. Seed quality does not explain them:

| | cells | straddling seeds | median seed error |
|---|---|---|---|
| zero-score | 7 | **7/7** | 5.49 mm |
| nonzero | 11 | 11/11 | 13.44 mm |

Every seed straddles the object, and the FAILING cells have *better* seeds than the
succeeding ones. `056_tennis_ball` fails on all three seeds despite 2.3-3.6 mm seed
errors, while `036_wood_block` succeeds on all three from 39-53 mm errors.

So the remaining defect is in the NLP, not the sampler. Every failing cell is
`best-effort` / `Maximum_Iterations_Exceeded`, and the frogger configuration never
reaches `converged` on any cell, against 8/18 for ours -- consistent with the
finite-differenced gradient being too noisy for IPOPT to certify a solution, but that
is a hypothesis, not a measurement.

### 13.2 What this does and does not license

It does NOT license "our method beats FRoGGeR". The port carries a known
non-faithful gradient implementation (FROGGER_COMPARISON §6), never converges, and
was measured on a dirty tree. What it licenses is: **the port is not yet good enough
to benchmark against**, and the next step is the analytic collision gradient, which
addresses both the time and, plausibly, the convergence.

### 13.3 Re-run with SLSQP + analytic collision gradient: no net gain

The n=2 sweep repeated after switching the frogger configuration to the paper's
solver family (SQP, not IPOPT) and implementing their analytic collision gradient
(eq. 8). Same objects, seeds and sampler.

| | ours | frogger (IPOPT + FD) | frogger (SQP + analytic) |
|---|---|---|---|
| median `l_bar*` | +0.8879 | +0.2968 | **+0.0000** |
| wrench-feasible | 16/18 | 10/18 | **6/18** |
| median solve time | 2.4 s | 34.5 s | **5.3 s** |

**The single cell I tested improved; the sweep did not.** `017_orange` seed 0 went
0.7056 -> 0.9544, which is what prompted the re-run, but per cell: 6 improved, 8
regressed, 4 unchanged. Testing one cell before launching a 36-cell sweep was not
enough to justify the change.

The split is object-dependent rather than random:

| object | OBB edges (mm) | outcome |
|---|---|---|
| `017_orange` | 74 / 72 / 72 | improved 3/3 (0.95, 0.76, 0.84) |
| `056_tennis_ball` | 67 / 67 / 67 | improved 1/3, unchanged 2/3 |
| `014_lemon` | 65 / 55 / 54 | improved 2/3, regressed 1/3 |
| `036_wood_block` | 210 / 127 / 122 | regressed 3/3, all to zero |
| `009_gelatin_box` | 107 / 96 / 34 | regressed 2/3 |
| `061_foam_brick` | 80 / 66 / 66 | regressed 2/3, unchanged 1/3 |

The two elongated objects regress on every seed, and the roundest object improves on
every seed -- but `061_foam_brick` is round and regresses, so shape does not fully
explain it. Both speed results hold regardless: 5.3 s against 34.5 s, a 6.5x
reduction from the analytic gradient, and `lp_gap` stays exactly 0 from the bilevel
LP.

**Still not reportable.** The frogger configuration never reaches `converged` on any
of the 36 cells across both runs, under either solver. That invariance across a
backend change argues the remaining defect is in the problem as posed -- the
constraint set or its scaling -- rather than in the optimizer, which is where the
per-constraint tolerances we have NOT applied (Table III: joint 1e-2, surface 5e-4,
collision 1e-3, force closure 1e-5) become the obvious next suspect.

### 13.4 After the cone fix: best result so far, and the zeros are gone

Same sweep, after removing the origin vertex from the min-weight cone
(FROGGER_COMPARISON §7.1).

| | ours | frogger (v2: SQP + analytic) | frogger (v3: + cone fix) |
|---|---|---|---|
| median `l_bar*` | +0.8873 | +0.0000 | **+0.4047** |
| wrench-feasible | 16/18 | 6/18 | **11/18** |
| median solve time | 2.3 s | 5.3 s | **5.0 s** |

**Exact-zero cells: 11 -> 0.** That is the cone diagnosis confirmed on the sweep
rather than on one contrived pinch. Every cell that previously reported `beta = 0`
with no gradient now reports a real value.

Per cell, 12 improved and 3 regressed. `036_wood_block` recovered on all three seeds
(0.0 -> 0.30/0.45/0.37) and `017_orange` now sits at 0.97-1.00.

### 13.5 The remaining failures are now legible

7 cells report `l_bar*` well below -1 (to -31.8). That is not a new fault: these are
the same failing grasps, with `beta` finally reporting HOW FAR from closure they are
instead of clamping at zero. A large negative value is exactly the smoothly climbable
signal the relaxation is for -- the solver now has a gradient to descend, and did not
before.

All 7 have straddling seeds (2.3-23.3 mm), so the seed is again not the discriminator.
`009_gelatin_box` fails all three seeds; it is the flattest object in the set
(OBB 107 x 96 x 34 mm), where a fingertip pair has little depth to oppose across.

Still: the frogger configuration reaches `converged` on 0 of 18 cells, against 12/18
for ours. That has now survived a solver change, a gradient change and a cone fix,
which continues to point at the constraint set rather than the optimizer -- the
per-constraint tolerances (§7.4) remain the untested candidate.

### 13.6 Tolerance scaling: implemented, and the convergence label is a red herring

`frogger_tol_scaling` realizes Table III's per-constraint tolerances. CasADi's
`sqpmethod` exposes ONE primal tolerance, so each constraint is instead SCALED:
writing `g(x)/s = 0` under a single `tol_pr = T` gives it an effective tolerance
`T*s`. With `T = 1e-5` (the tightest, force closure) that is joint /1000,
collision /100, surface /50, force closure /1.

**The `converged` count was the wrong thing to chase.** Four measurements:

| probe | result |
|---|---|
| `max_iter` 20 / 80 / 200 / 800 | `l_bar*` 0.106 / 0.593 / 0.608 / **0.608** |
| `tol_pr` 1e-5 vs 1e-3 | 0.593 vs 0.587 |
| `tol_du` 1e-2 / 1e-1 / 1.0 | 0.608 / 0.608 / **0.608** |
| surface residual at the solution | **5.1e-12** against `tol_pr = 1e-5` |

`l_bar*` plateaus from 200 to 800 iterations, neither tolerance moves it, and the
constraints are satisfied seven orders of magnitude inside tolerance. **The solve
converges numerically; only the status label is wrong** -- CasADi's `sqpmethod`
keeps iterating without crediting a step, most plausibly a line-search artifact.

So `converged 0/18` was never evidence of a defect, and three of this session's
changes were pursued partly on the strength of it. The quantity that matters is
`l_bar*` and the wrench certificate, both of which are reported.

The scaling itself is a modest, mixed change on a 3-object probe:
`009_gelatin_box` -4.33 -> +0.34 and now wrench-feasible, `017_orange`
1.00 -> 0.94, `036_wood_block` +0.30 -> -1.47. Kept because it implements a stated
part of their method, not because it is a measured win.

### 13.7 Final n=2 state, and the tolerance scaling reverted

Full sweep with the cone fix and friction consistency, tolerance scaling OFF.

| | ours | frogger |
|---|---|---|
| median `l_bar*` | +0.8265 | +0.3370 |
| `l_bar* > 0` | 17/18 | 10/18 |
| wrench-feasible | 16/18 | 10/18 |
| median `\|lp_gap\|` | ~0.001 | **0.00000** |
| median solve time | 2.4 s | 5.1 s |

**Tolerance scaling (§13.6) was measured harmful and is now default OFF.** Ablated on
4 objects at seed 0:

| object | scaling ON | scaling OFF |
|---|---|---|
| `017_orange` | 0.5929 | 0.6664 |
| `036_wood_block` | **-265.09** | 0.2000 |
| `014_lemon` | -17.23 | -31.52 |
| `061_foam_brick` | -0.0087 | 0.2900 |
| median | **-8.6202** | **+0.2450** |

`tol OFF` and `both OFF` are identical, and `both ON` and `fric OFF` are identical, so
the friction change contributes nothing to the objective (expected -- it touches only
`verify()` and the gamma LP) and the scaling caused the whole regression.

**Why the approach was wrong.** Dividing a constraint by 50-1000 to emulate a
per-constraint tolerance also divides its GRADIENT by the same factor, so the scaled
constraints stop steering the solve relative to the unscaled ones. It is not a
tolerance change, it is a silent reweighting of the constraint Jacobian. Matching
Table III needs a solver that accepts per-constraint tolerances; CasADi's `sqpmethod`
exposes one, and this trick does not substitute for it.

### 13.8 Where the port stands

Working, verified, and faithful to the paper: the min-weight metric with a bilevel LP
and analytic KKT gradients (`lp_gap` exactly 0 on every cell); the 4-sided friction
cone; FK contacts with a fixed contact point; exact collision distance with their
eq. (8) analytic gradient; SLSQP; the OBB sampler; both collision margins; `k_l`.

The frogger configuration reaches `l_bar* > 0` on 10/18 cells and a median of +0.337,
against ours at 17/18 and +0.827. **That is not a method comparison**, for three
recorded reasons: our scene friction is mu = 2.0 against the paper's 0.7 (§7.2), which
changes which grasps close at all; solve times are not comparable until the remaining
FD paths are analytic (FROGGER_COMPARISON §6); and every number here was measured on a
dirty tree (§0a).

The 8 failing cells all have straddling seeds, so the seed is not the discriminator.
`009_gelatin_box` fails all three seeds and is the flattest object in the set.


### 13.9 Verified: both configurations build the friction cone at the same mu

Checked directly rather than by reading the code. Recomputing `l_bar*` from each
configuration's OWN solved contacts, at `mu = 2.0` and `mu = 1.6`, against what the
NLP reported (`017_orange` seed 0, n=2):

| config | reported | recomputed at mu=2.0 | recomputed at mu=1.6 | `mu` | `n_cols` |
|---|---|---|---|---|---|
| ours | 0.8069 | 0.9435 | 0.9297 | 2.000 | 12 |
| frogger | 0.9996 | **0.9996** | 0.9995 | 2.000 | 12 |

The cone friction is set in `_run_stage` at `_mu = round(1 * _mu_raw, 3)`, which is
shared by both configurations with no gating, so both build `W` at MuJoCo's combined
2.0. `frogger_raw_friction` affects only `verify()` and the post-solve gamma LP, never
the cone.

The frogger recomputation matches exactly. The `ours` gap of 0.137 is NOT friction --
recomputing at 1.6 moves it by 0.014, an order of magnitude too little. It is
`beta_delta`, the patch-versus-SDF normal difference of §8: `ours` builds `W` from
quadratic-patch normals while the audit uses the SDF gradient, and the frogger
configuration shows `beta_delta = -0.00000` precisely because it already uses SDF
normals.

So the two configurations differ in the NORMALS entering the cone, which is a
deliberate and documented axis of the comparison (§2.2), not in the friction
coefficient.

## 14. At the paper's friction (mu = 0.7)

`--mu 0.7` sets the object geom's sliding friction to the paper's simulation value.
`table_scene.build` already took `friction` as a parameter, so this is a CLI flag,
not a default change: every existing result at mu = 2.0 stands. Verified the value
reaches the contact -- MuJoCo combines by elementwise max and the LEAP fingertips are
0.5, so anything below 0.5 would be floored by the fingertip instead of the object.

| | ours mu=2.0 | ours mu=0.7 | frogger mu=2.0 | frogger mu=0.7 |
|---|---|---|---|---|
| median `l_bar*` | +0.8265 | +0.6606 | +0.3370 | **-7.4655** |
| `l_bar* > 0` | 17/18 | **18/18** | 10/18 | 7/18 |
| wrench-feasible | 16/18 | 15/18 | 10/18 | 7/18 |

**This is the first measurement in the paper's own regime**, and it does not flatter
the port: the frogger configuration's median falls off a cliff while ours degrades
gracefully.

### 14.1 The asymmetry is the interesting part

Lower friction should make force closure HARDER for both. Ours instead gains a
positive cell (17/18 -> 18/18) while its median drops modestly; the frogger
configuration loses three and its median goes deeply negative.

A plausible reading, not yet tested: our configuration carries `w_ik`, `w_align` and
the patch trust region, which keep contacts opposed and reachable regardless of what
`beta` is doing, so a harder friction regime degrades the METRIC without moving the
contacts much. The frogger configuration has only `beta` -- by construction, since
(7a) is a single term -- so when the friction regime makes `beta`'s landscape harder,
there is nothing else holding the solve in a good basin. If that is right, it is a
genuine finding about the two formulations rather than a port defect: FRoGGeR's
minimal objective is more exposed to the friction regime than a multi-term one.

Testing it means re-running the frogger configuration with `w_align` restored at
mu = 0.7 and seeing whether the cliff disappears. That deviates from (7a) on purpose,
as a diagnostic rather than a benchmark arm.

### 14.2 Which objects survive

At mu = 0.7 the frogger configuration stays positive on 7 cells: `017_orange` all
three seeds, `056_tennis_ball` two, `061_foam_brick` one, `014_lemon` one. It fails
every `036_wood_block` and `009_gelatin_box` seed -- the two non-round objects, which
is consistent with the mu = 2.0 pattern where the roundest object was also the most
reliable.

### 14.3 The collapse is geometric, and (7a) has no term to prevent it

FRoGGeR's objective is `maximize l*(q)` -- ONE term, confirmed against (7a)-(7e).
There is no alignment cost anywhere in their program, so the frogger configuration
correctly has none, and adding one would not be FRoGGeR.

Measuring what actually changes at mu = 0.7, the contact normals' dot product at the
solved grasp (-1 = perfectly opposed, +1 = same side), same seeds both runs:

| | median normal dot |
|---|---|
| mu = 2.0 | **-0.024** |
| mu = 0.7 | **+0.858** |

The contacts genuinely migrate to the SAME SIDE of the object. `036_wood_block` is
the clearest: -0.991 (well opposed) at mu = 2.0 against +0.874 at 0.7, with `l_bar*`
going +0.453 -> -8.167. So this is not the metric reporting the same grasp more
harshly; the solve converges somewhere geometrically worse.

**Why the seed does not save it.** Their sampler supplies opposition -- palm aligned
to an OBB axis, fingers pre-separated by that edge -- and it does straddle the object
on 60-100% of reachable draws (§11.1). But nothing in (7a) HOLDS the solve there. At
mu = 2.0 a wide friction cone means many contact pairs are in closure, so `beta`'s
landscape is broad and the solve stays near its opposed seed. At 0.7 the cone
narrows, that basin shrinks, and `beta` alone does not pull the contacts back.

Our own configuration does not have this problem because `w_ik` and `w_align` anchor
the contacts independently of `beta` -- which is exactly the multi-term structure
FRoGGeR dispenses with.

**This is a real property of the formulation under our conditions, not a port
defect,** but it is NOT yet a finding about the paper: they pair the minimal objective
with an Allegro hand, their own IK, and 20 sampler draws per object, any of which
could keep the solve in the opposed basin where ours drifts out. Reporting it as
"FRoGGeR fails at realistic friction" would overclaim.

The honest next test is sampler draws: we accept the FIRST straddling draw, while
they generate many candidates per object. If more draws recover the mu = 0.7 numbers,
the cause is our thin sampling rather than their objective.

## 15. At the paper's full protocol: mu = 0.7 with resampling

The harness previously drew seeds, picked one, solved ONCE and reported the result.
That is not FRoGGeR's protocol. They RESAMPLE until a grasp is feasible within a
60-second budget -- "we try to generate 20 feasible grasps", "a run converges if it
yields a feasible grasp in under 1 minute" -- and Table I reports a MEDIAN OF 3
SOLVES per feasible grasp (IQR 1-6). Their 99.4% convergence rate is a property of
that loop, so a single-attempt harness understates the method by construction.

`--max-attempts 20` implements it (`_frogger_synthesize`), accepting the first
attempt whose RETURNED grasp clears `l_bar* >= k_l`.

| frogger, mu = 0.7 | 1 attempt | up to 20 attempts |
|---|---|---|
| median `l_bar*` | **-7.4655** | **+0.3084** |
| wrench-feasible | 7/18 | **14/18** |
| median solve time | 5.5 s | 17.5 s |

Fourteen of eighteen cells converge, against ours at 14/18 wrench-feasible and a
median `l_bar*` of +0.6030. Individual recoveries are large: `036_wood_block` seed 2
went -63.44 to +0.3000 on attempt 4, `014_lemon` seed 0 -25.00 to +0.3341 on
attempt 6, `056_tennis_ball` seed 1 -15.62 to +0.7226 on attempt 2.

### 15.1 The attempt distribution matches the paper

| | this port | FRoGGeR (Table I) |
|---|---|---|
| attempts on success | median **2**, IQR 1-6 | median **3**, IQR 1-6 |
| convergence rate | 77.8% (14/18) | 99.4% |
| `l_bar*` over converged | 0.325 | 0.58 |

The attempt distribution is independent evidence that the port behaves like the
method: it is not a quantity that was tuned, and it lands on theirs.

Convergence and metric value are both below the paper's. Candidate causes, none
tested: their 43-object set is pruned to watertight meshes and excludes objects "too
large, small, or thin to reasonably grasp"; they use 4 Allegro fingers against our 2;
and our sampler accepts the first straddling draw rather than ranking many.

### 15.2 What still fails

`009_gelatin_box` exhausts all 20 attempts on every seed, and `036_wood_block` seed 0
does too. The gelatin box is the flattest object in the set (OBB 107 x 96 x 34 mm),
where a two-fingertip pinch has little depth to oppose across -- a plausible
geometric limit for `n = 2` rather than a port defect, and the n >= 3 path is the
test for that.

Note the cost: a failing cell now burns the full 60-second budget, which is why the
median solve time rose 5.5 s -> 17.5 s. That is the protocol working as specified,
not a regression.

## 16. Execution scoring: what is measured

### 16.1 Their rubric (Sec. IV), reproduced exactly

A binary PICK SUCCESS per trial. A pick fails if any of:

| criterion | threshold |
|---|---|
| object rotation from its pre-lift orientation | > 30 deg |
| object deviation from the pick trajectory, at any point | > 7.5 cm |
| total grasp synthesis time | > 60 s |

on a fixed trajectory: lift 10 cm in 1 s, hold 1.5 s, 3 mm sinusoidal perturbation
in all spatial axes from 0.25 s onward. They run 20 trials per object and report the
success RATE -- 78.8% overall (spheroid 95.3%, box/cylinder 81.6%, adversarial
63.0%). That is their whole execution rubric; everything else in Table I is
planning-side.

### 16.2 Trial count: one execution per planned grasp

Their 20 trials per object sample STOCHASTIC SYNTHESIS -- each trial re-runs the
sampler and solver. Our benchmark already varies that through its seeds, and our
execution is deterministic (same plan, same physics, same result), so repeating one
plan adds nothing. Each of the 18 object-seed cells is executed ONCE and the rate is
taken across cells. Comparable to their per-category rates once the object sets are
matched, which they are not yet (§15.1).

### 16.3 What is recorded beyond the binary

Their criteria are pass/fail and trajectory-relative, which admits a false pass: an
object that never moves BECAUSE IT WAS NEVER PICKED UP has zero deviation and zero
rotation. This repo has already produced exactly that failure mode -- a run reported
`release_done` with 0.0 N on both fingers, the object riding inside the hand.

So alongside their rubric:

| quantity | why |
|---|---|
| `success`, `fail_reason` | theirs, verbatim |
| peak rotation (deg), peak deviation (m) | continuous, not just pass/fail -- failing at 31 deg and at 120 deg are different results, and at 18 cells the continuous form carries far more than a binary |
| min squeeze force during the hold | the guard against the false pass above. A grasp with no force is not a grasp, whatever the trajectory says |
| contact lost during the jog | whether any fingertip force reached zero mid-motion |
| achieved lift height | separates "held but did not rise" from "rose and held" |

The primary reported number stays their binary rate; the rest are diagnostics that
make a failure legible rather than redefining success.

## 17. Object set: `009_gelatin_box` excluded (our deviation, not theirs)

`009_gelatin_box` is NOT on FRoGGeR's Table II exclusion list -- they kept it among
their 43. Dropping it here is our choice, recorded as such.

The reason: it exhausts all 20 synthesis attempts on every seed for the frogger
config at `n = 2`. It is the flattest object in the set (OBB 107 x 96 x 34 mm), and a
two-fingertip pinch has little depth to oppose across. FRoGGeR uses FOUR Allegro
fingers, so the object is not flat relative to THEIR hand -- which means its failure
here is plausibly an artifact of running their method at a contact count they never
tested, rather than a property of the method.

Effect on the mu = 0.7, resample-until-feasible results:

| | with gelatin (18 cells) | without (15 cells) |
|---|---|---|
| ours median `l_bar*` | +0.6030 | +0.6030 |
| ours wrench-feasible | 14/18 | 13/15 |
| frogger median `l_bar*` | +0.3084 | +0.3167 |
| frogger wrench-feasible | 14/18 | **14/15** |
| frogger convergence | 78% | **93%** |

93% against the paper's 99.4%, on a set of 5 objects against their 43. The frogger
config now has MORE feasible cells than ours (14/15 against 13/15) while ours keeps
the higher metric value.

**This does not license reading the exclusion as a fix.** One object was removed
because it failed, which moves a rate by construction; the honest statement is that
the port converges at 93% on objects a two-finger pinch can physically grasp, and
that the excluded object is the test case for the `n >= 3` path. It should return
when that path works.

---

## 9. Execution scoring (2026-09-14): pick success, epsilon, and a gap-gate wall

Qualification 1 of `FROGGER_STATUS.md` §1 is now closed: the paper's shaky pickup
runs against live grasps. Two new instruments, both additive:

- **`simulation/epsilon_metric.py`** — Ferrari-Canny `epsilon`, the classical quality
  FRoGGeR SCORES with, as distinct from `l_bar*`, the relaxation they OPTIMIZE.
  Verified against closed form on a 4-contact cross grasp (`eps = 2*mu*r` exactly,
  7/7 across mu, r and gamma); `tests/test_epsilon_metric.py`, 5/5 passing.
- **`benchmarks/ycb_grasp/frogger_exec_bench.py`** — plans with either arm and
  EXECUTES through `pick_and_place`'s measured approach/settle/gap-gate/squeeze path,
  with FRoGGeR's lift-and-shake substituted for the lift. `run_pick_place` grew a
  `plan_override` hook (cfg + start pose) because it previously hardcoded
  `for_gws_recommender` and a HOME start, so only `ours` was executable.

### 9.1 Pilot result, 2 objects x 3 seeds x 2 arms

`n = 2` (thumb+index), scene friction `mu = 2.0`, single solve, `k_l = 0.3`.

| | ours | frogger |
|---|---|---|
| reached the squeeze | **6/6** | 1/6 |
| pick success (of those that lifted) | 3/6 | 1/1 |
| `lift_ok` (of those that lifted) | **3/6** | 0/1 |
| median `l_bar*` | +0.945 | +0.398 |
| median wall-clock | 12.7 s | 3.7 s |

**The headline is the second row, and it is not a metric difference.** Five of six
frogger cells never reached the squeeze:

| cell | abort | fingertip gap (mm) |
|---|---|---|
| orange s0 | gap too large | index 8.54, thumb 11.56 |
| orange s2 | wrench-infeasible | -- (`l_bar*` = -10.30) |
| block s0 | gap too large | index 10.80, thumb 11.14 |
| block s1 | gap too large | index 8.15, thumb 8.87 |
| block s2 | gap too large | index 0.97, thumb 10.34 |

The gaps cluster at 8-11.5 mm, which is `frogger_pad_offset_m` (0.011). That is the
mechanism, not a coincidence. Their (7d) constrains `site + 11 mm * pad_axis` to the
surface — a FIXED body-frame point, which is the faithful reading of their sentence
and is already documented at `grasp_planner_3d.py:3705`. The executor's gap gate
measures the ACTUAL tip-geom surface, and the LEAP tip is a box of half-extents
~11 x 12 x 17 mm, so along any direction that is not the pad axis the real surface
sits further out than the 11 mm the constraint assumed. The solve is satisfied; the
hand is still a centimetre away.

`l_bar*` cannot see this. Orange s0 reports **+0.9996**, essentially the ceiling of
1.0, with the fingertips 8.5 and 11.6 mm off the object. A metric computed on
contact points that the hand does not reach is measuring a grasp nobody is holding —
the same class of defect as §7.1's 5th cone vertex, caught here only because the
grasp was executed.

### 9.2 Their criteria can score a one-finger carry as a success

`frogger | 017_orange | seed 1` passed the shaky pickup — object rose 97.8 mm,
rotated 2.5 deg, deviated 1.5 mm, all well inside their thresholds — while the index
finger's measured force at the final step was **0.0 N**. The object was carried on
the thumb alone.

Their criteria are displacement-only and cannot detect this: an object that tracks
the palm neither rotates nor deviates, however few fingers are actually loaded.
`lift_ok`'s force gate does detect it, which is why both are reported and why
`pick_success` alone should not be quoted from this harness. This is the
`release_done != grasped` failure class in a new place.

### 9.3 What this does and does not establish

Does: the execution path is wired, both arms run through one shared executor, and
the frogger arm's dominant failure mode on this hand is a REACHABILITY mismatch at
the pad-offset constraint rather than a grasp-quality deficit.

Does not: `n = 2` is not their configuration (four Allegro fingers), `mu = 2.0` is
not their friction, and 12 cells is a pilot. The 3/6 vs 1/1 success comparison rests
on one frogger cell that reached the lift and should not be quoted as a rate. The
6/6 vs 1/6 squeeze-reach difference is the result worth carrying forward.

### 9.4 The epsilon column is empty at n = 2, by construction

All 12 cells report `epsilon` degenerate. A two-contact pinch's wrench set is
rank-5-of-6 — it resists no torque about the line through its contacts — so it
contains no origin-centred 6-ball and Ferrari-Canny `epsilon` does not exist. This is
the same structural fact that gates `project_grasp_axis_torque` in the gamma
certificate, and it is why the paper's `epsilon` column is a FOUR-finger number.
Reporting 0.0 would have been wrong in a specific way: it reads as "closure, but
weak" rather than "not defined".

The force-subspace fallback is also uninformative at this scene's friction: it
saturates at exactly 1.0 for any `mu >= 1` (measured 0.667 / 1.000 / 1.000 / 1.000 at
mu = 0.5 / 1 / 2 / 4), because the normal direction is bounded by the normal force
while the tangential extent grows with mu. **`epsilon` becomes measurable only when
the `n >= 3` path works**, which makes it one more thing blocked on that path.

### 9.5 Running it

```bash
cd benchmarks
# the pilot above
uv run python -m ycb_grasp.frogger_exec_bench \
    --objects 017_orange,036_wood_block --seeds 0,1,2

# our own 12 cm lift instead of their shaky one
uv run python -m ycb_grasp.frogger_exec_bench --lift-mode standard
```

`--mu` is deliberately NOT plumbed here and exits with a message: `run_pick_place`
builds its own scene, so accepting the flag would have run at 2.0 while labelling the
output 0.7. Use the plan-only harness for the paper's friction regime.

---

## 10bis. Appendix B: their controller, read at last (2026-09-14)

§9 was written believing the paper specified no control law. **It does** -- App. B,
"Controller Implementation Details," which had not been read. Correcting that record
here, because it changes what our squeeze mechanism should be compared against.

### 10bis.1 What they actually do

Their control law (18):

```
tau = Jh^T R_BC F_C*  +  (I - Jh^T (Jh^T)^dagger) tau_joint
```

**Term 1 -- commanded contact forces**, from a QP solved online:

```
minimize_{F_C}  ||G F_C - (-^O w_des)||^2
s.t.  Lambda_i F_C,i <= 0                       (pyramidal friction cones)
      F_C,i^n >= F_min^n                        (minimum normal force)
      tau_lb - tau_joint,h <= J_i^T R_BC F_C,i <= tau_ub - tau_joint,h
```

with **`F_min^n = 1.0 N`** (0.25 N under 0.01 kg). The desired wrench is
`^O w_des = R_OB(^B w_grav + ^B w_err)`, gravity plus a PD **error wrench on the
OBJECT's pose**: `k_p,err = 50`, `k_d,err = 5`, `k_R,err = 50`, `k_omega,err = 5`.

**Term 2 -- tracking, null-space projected.** `tau_joint = tau_grav + tau_track`;
arm at `K_p,arm = 500 I`, `K_d,arm = diag(1,1,1,1,0.1,0.1,0.1)` via Drake's
differential IK; hand a **pure proportional** term
`tau_hand = -k_p,hand (q_h - q_h*)` with **`k_p,hand = 5`**. The projector's stated
purpose: applying these "does not change the contact positions between the hand and
object." They note it "does not affect the arm torques at all."

### 10bis.2 Consequences for this benchmark

- **Their squeeze is COMMANDED, not emergent.** An earlier reading here guessed the
  -3 mm interpenetration allowance was the force-generation mechanism. It is not:
  force comes from the QP, and `F_min^n = 1.0 N` is a floor they set explicitly.
- **Directly comparable to ours.** Their 1.0 N floor against our measured squeeze of
  1.62 / 1.61 N on `017_orange` -- the same order, a comparison we could not make
  before.
- **FROGGER_COMPARISON §3.3 needs qualifying.** It records theirs as task-agnostic
  and ours as task-specific. True of the PLANNER's `k_l` floor; NOT true of their
  CONTROLLER, whose `w_err` is a live task-specific wrench like our `gamma` LP.
- **Their hand gain is tiny and projected** (`k_p,hand = 5`), so tracking cannot
  fight the force command. Our `squeeze_pd_scale = 0.25` pursues the same end by
  weakening tracking globally instead of only where it conflicts.

### 10bis.3 The projector is implemented; it does not transfer cleanly

`GraspController(nullspace_tracking=True)` applies
`(I - Jh^T (Jh^T)^dagger)` to the tracking torque while squeezing, scoped to the
FINGER columns so the arm jog is untouched (their own note says the projection
leaves arm torques alone). Verified: the component of tracking torque lying in
`range(Jh^T)` drops from 4.46 to **5.9e-14**, the operator is idempotent to 4.0e-15,
and arm columns are bit-identical.

**But the DOF budget does not transfer.** Their Allegro has 16 hand DOFs against a
rank-12 `Jh` (4 contacts). Our 2-contact LEAP has **8 finger DOFs against rank 6**,
a null space of dimension 2, so the projector discards a measured **~55%** of finger
tracking authority (mean surviving fraction 0.449 over 500 random torques).

Measured end to end, `--lift-mode shaky`:

| | baseline | nullspace |
|---|---|---|
| `017_orange` s0 squeeze (N) | 1.622 / 1.605 | **1.696 / 1.693** |
| `017_orange` s0 shake deviation | **1.2 mm** | 2.2 mm |
| `017_orange` s0 pick / lift_ok | OK / True | OK / True |
| `036_wood_block` s0-s2 | 0/3 | 0/3 |

Squeeze force rises slightly (tracking no longer fights it, which is the projector's
purpose) and shake tracking degrades slightly (55% less authority). The block fails
3/3 either way -- that is a grasp failure, not a control-law one.

**Verdict: correct, faithful, and not a fix.** Worth keeping as an ablation and
reporting as a DOF-budget finding -- eq. (18) assumes a hand with DOFs to spare, and
a 2-contact pinch on an 8-DOF finger set does not have them. Default stays off.

### 10bis.4 The pad constant, corrected and honestly inconclusive

`frogger_pad_offset_m` was `0.011`, an ESTIMATE ("roughly one half-extent" of the tip
geom's bounding box). Measured against the 52-vertex tip meshes, the pad surface
along `pad_axis` is **9.96 mm** (thumb) / **9.95 mm** (index), site-to-geom-centre
0.05 mm. Corrected to **0.00995**.

The old value was wrong, so the correction stands on its own terms. It did **not**
improve execution. Plan-only A/B, `thumb,index`, same seeds (`max_iter` 80; repeated
at 400 with the same pattern, so this is not an iteration-budget artifact):

| cell | 0.011 | 0.00995 |
|---|---|---|
| `017_orange` s0/s1/s2 | +0.998 / +0.935 / +0.976 | +0.995 / +0.933 / +0.949 |
| `036_wood_block` s0 | -21.07 INFEAS | -1.74 INFEAS |
| `036_wood_block` s1 | **+0.431** (gamma 35.98) | **-14.73 INFEAS** |
| `036_wood_block` s2 | +0.302 (gamma 41.53) | +0.453 (gamma 23.99) |

wrench-feasible 4/6 -> 3/6. The orange is unmoved; the block swings in both
directions. Every cell exits `best-effort`/`Maximum_Iterations_Exceeded`, so the
solve is landing in different basins rather than responding smoothly to a 1 mm
change. **Report the constant as corrected, not as an improvement.**

### 10bis.5 Why the frogger cells abort wrench-infeasible

Traced on `036_wood_block` seed 1: `n1_in . n2_in = +0.989` -- the two contacts point
the SAME direction, 8.5 degrees apart, 125 mm apart on a 107 mm object.
`solve_gamma_live` is right to refuse it; no squeeze force holds two same-side
contacts.

This is §14.3's same-side collapse at `n = 2` (it was recorded at `n = 3`, dots
+0.914 / +0.966 / +0.931).

### 10bis.6 Their sampler DOES enforce opposition, and ours reproduces it

Checked against the paper (Sec. IV, App. C step 1, Fig. 4 caption), because an
earlier note here implied opposition was simply absent from their method. It is not:

> "(1) from the oriented bounding box of the object ... choose an axis with which to
> align the palm's **y-axis** up to sign and **use the width of this box edge to fix
> an initial guess for the separation of the hand's fingers**"

Their palm convention (App. C) is x = outward normal, z = toward the fingers, y =
right-handed -- so the y-axis IS the finger-spread direction. Aligning it to an OBB
axis and opening the fingers to that edge's width makes the pre-shape BRACKET the
object. Axis choice is weighted by side length (`a/(a+b+c)`), "motivated by
observations of preferred human grasps".

They state the mechanism is load-bearing:

> "the overall performance of both FRoGGeR and the baseline was **highly sensitive to
> the sampled initial conditions**. For instance, **if the initial width of the
> fingertips was not guided by object bounding boxes, both methods suffered in terms
> of runtime and grasp quality**, as enforcing surface constraints became harder."

So opposition lives in the SAMPLER by design, not in (7a). The method is sampler +
refinement as a unit, and criticizing (7a) for lacking an alignment term misreads it.

**Our port reproduces this correctly.** Measured, `thumb,index`, opposition as the
cosine between tip directions from the OBB CENTRE (-1 = perfectly opposed):

| cell | seed `tip_dot` | solved `n1.n2` |
|---|---|---|
| `017_orange` s0 | **-1.000** | **-1.000** (kept) |
| `017_orange` s1 | **-0.996** | +0.974 |
| `017_orange` s2 | **-0.914** | +0.138 |
| `036_wood_block` s0 | **-0.605** | +0.798 |
| `036_wood_block` s1 | **-0.758** | +0.989 |
| `036_wood_block` s2 | **-0.592** | +0.998 |

**Every seed starts opposed; the SOLVE destroys it on 5 of 6.** That relocates the
defect: it is not a seeding failure, and the sampler is not at fault. The solve walks
a well-opposed start to a same-side configuration, which is what needs explaining.

Note every one of these cells exits `best-effort` / `Maximum_Iterations_Exceeded`
(§10bis.4), so a plausible reading is that the iterate is simply mid-flight rather
than at any optimum of (7a) -- a hypothesis §10bis.4's `max_iter = 400` run does not
support, and which the next investigation should settle before any claim is made
about their objective.

**MEASUREMENT HAZARD, found the hard way.** Scoring opposition about
`data.xpos[bid]` gives entirely wrong answers: the YCB body origin sits at the
object's BASE, 40.4 mm below the OBB centre on `017_orange`. Doing so made every
seed read as same-side (`tip_dot` +0.12 to +0.28) when the same seeds are in fact
-0.91 to -1.00 about the true centre, and briefly produced the conclusion that the
sampler was broken. The sampler's own `_seg_dist` was right all along -- it scores
against the OBB centre. Use the OBB centre (or `vol_centroid`), never the body
origin.

---

## 11bis. Edge-seeking, measured (2026-09-14)

FRoGGeR names edge-seeking as the dominant failure mode of BOTH arms, attributes it
to the metric preferring large moment arms, states it "yields unstable grasps in
practice", and lists combating it as future work: "we hope to develop methods to
combat edge-seeking behavior." **They give no metric for it.**
`benchmarks/ycb_grasp/edge_seeking.py` supplies one.

### 11bis.1 Two metrics, and why the obvious one is not comparable

**Patch-bound margin** (`contact_edge_margins`): the distance from the solved contact
to the nearest trust-region bound that was set by MEASURED SDF divergence, using the
RAW (pre-inset) bound and skipping axes parked at `quadratic_t_bound_max` -- exactly
the test `w_edge_margin`'s own hinge applies. Exact, free, and reads data each solve
already records.

**It is not usable for the comparison.** The frogger arm sets
`frogger_fk_contacts=True`: its contacts are FK outputs pinned by (7d), with no local
parameterization and therefore no bounds. Measured: `quad1_frame` is None on 15/15
frogger solves. Reporting that as "no edge nearby" would credit their arm for a
measurement never taken.

**Geodesic margin** (`geodesic_edge_margin`), the comparable one: from the contact,
step outward along 16 tangent directions, reprojecting to the surface each step,
until the surface NORMAL turns more than 30 degrees from its value at the contact.
The distance when that first happens, minimized over directions, is the margin.
Asks the MESH, so it is identical for both arms regardless of representation.

### 11bis.2 Result: 5 objects x 3 seeds, n = 2, plan-only

| | ours | frogger |
|---|---|---|
| median geodesic margin | **12.0 mm** | 1.0 mm |
| IQR | (11.0, 15.5) | (1.0, 8.0) |
| worst grasp | 8.0 mm | 1.0 mm |
| **contacts within 2 mm of an edge** | **0 / 30** | **20 / 30** |

Per object (median over 3 seeds, mm):

| object | ours | frogger |
|---|---|---|
| `014_lemon` | 12.0 | 1.0 |
| `017_orange` | 18.0 | 17.0 |
| `036_wood_block` | 11.0 | 1.0 |
| `056_tennis_ball` | 15.0 | 1.0 |
| `061_foam_brick` | 11.0 | 1.0 |

Consistent on 4 of 5. `017_orange` is the exception and the informative one: a
sphere has no edges, so neither arm can seek one and both sit ~17-18 mm from the
nearest 30-degree normal turn. The separation appears exactly where edges exist.

1.0 mm is the search STEP, so frogger's contacts are on the edge, not near it.

### 11bis.3 What this does and does not establish

Does: on this object set, at n = 2, the frogger arm places 2/3 of its contacts within
2 mm of a measured surface boundary and ours places none. That is the behaviour the
paper describes and does not quantify, measured on a shared object set with one
representation-free metric.

Does not: it is a PLAN-ONLY property and does not by itself show edge-seeking causes
the execution failures. The mechanism is plausible (a contact on a crease has a
friction cone falling off two faces) but untested here; pairing these margins with
`frogger_exec_bench` outcomes is the test.

Nor does it isolate WHICH of our mechanisms is responsible. `quadratic_bound_inset`
(10 mm, active) is the obvious candidate and roughly matches the observed separation,
`w_edge_margin` is OFF by default, and the patch parameterization confines contacts
to a fitted trust region in the first place. An ablation over those three would
attribute it; this measurement only establishes that the difference exists.

---

## 12bis. Ferrari-Canny epsilon at n = 3 (2026-09-14)

§9.4 recorded epsilon as unmeasurable at n = 2 (a pinch's wrench set is rank-5-of-6,
so it contains no origin-centred 6-ball) and blocked on the n >= 3 path. That path
now runs: `--fingers thumb,index,middle` plans, `verify()` certifies, and
`epsilon_quality` returns `degenerate=False`. The column is unblocked.

**But the values say the tripod is not a real tripod.** Plan-only, `ours`, 5 objects
x 3 seeds, mu_opt = 0.8*2.0, gamma = 1.0:

| object | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `017_orange` | 2.84e-3 | 1.34e-3 | 1.63e-3 |
| `061_foam_brick` | 27.36e-3 | 1.45e-3 | 21.66e-3 |
| `036_wood_block` | **-1.1e-16** | **-1.1e-16** | **-1.1e-16** |
| `014_lemon` | **-3.6e-17** | **-3.6e-17** | **-3.6e-17** |
| `056_tennis_ball` | **-4.2e-17** | **-2.8e-17** | **-2.4e-17** |

Nine of fifteen cells are zero to machine precision: the origin lies exactly ON a
facet of the grasp wrench set, which is the boundary of force closure, not its
interior. `verify()` reports `wrench_feasible=True` on all fifteen.

The two are not in conflict. `verify()`'s `gamma` LP asks whether a specific
disturbance BOX can be resisted at some finite internal force; epsilon asks for the
largest ball in EVERY direction at gamma = 1. A wrench set that is a thin sliver --
wide where the task needs it, zero-thickness elsewhere -- passes the first and scores
zero on the second.

The sliver is the known n = 3 defect. §10bis measured contacts 2 and 3 at 4.0 mm
apart on `017_orange` (17.2 mm on one seed) and 25.8 mm on `036_wood_block`, against
68-90 mm from contact 1 -- so the "tripod" is geometrically a pinch with a doubled
finger, and a doubled contact adds no independent wrench directions. The three
objects reporting exact zeros are precisely the ones where the two contacts are
closest.

**Do not report an epsilon median from this.** A median over nine machine-precision
zeros ("-0.00e-3") describes the collapse, not the method. Epsilon is now instrumented
and correct -- it is the n = 3 CONTACT PLACEMENT that is not ready. Fixing the
collapse, not the metric, is what unblocks this column, and epsilon is now the
sharpest available test of whether a fix worked: it goes from ~0 to positive exactly
when the third contact becomes independent.

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

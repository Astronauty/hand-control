# FRoGGeR as a benchmark arm

**Status: design note plus Phase 1 results.** The benchmark harness itself is not yet
built. This file fixes the terms of the comparison before further code is written, and
records what Phase 1 established.

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
2. **`frogger` planner mode** — sole `beta` objective, SDF surface equality on free
   3-vectors in place of patch coordinates, `beta >= k_l / m` as a hard constraint, true
   SDF normals. Plus the OBB palm sampler (noisy alignment to OBB axes weighted by side
   length, palm 4 cm out, collision-free IK) as a seed generator alongside
   `_fixed_antipodal_seed`.
3. **`benchmarks/ycb_grasp/frogger_bench.py`** — `--planner {ours,frogger}`, N objects x M
   seeds, logging `l_bar*`, epsilon, `gamma_min`, span margin, solve time, solve count, and
   pick success. Includes the paper's shaky pickup: lift 10 cm in 1 s, hold 1.5 s, 3 mm
   sinusoidal perturbation in all axes from t+0.25 s; failure on >30 deg rotation, >7.5 cm
   deviation, or >60 s synthesis.
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

The residual factor of ~5 is unexplained. Candidate contributions, not yet separated: the
normal error at a trust-region bound exceeding the patch median quoted above; the
contribution of §8.3's best-effort iterate (`017_orange` is best-effort on all three
seeds); and a systematic rather than random tilt direction, which the random-direction
model above would understate. This should be resolved before `delta` is used
quantitatively rather than as a direction.

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

### 8.5 Dependence on contact count

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

### 8.6 Determinism

Four full 18-cell sweeps. Runs 1, 3 and 4 are byte-identical (md5 `29edc6c1ad41` over the
table body). Run 2 differed in two cells (`056_tennis_ball` sd0, `061_foam_brick` sd2).

Run 2 was started while run 1 was still executing, so two sweeps shared the machine for its
duration. The two affected cells fall in the last two objects of the sweep order and are
both best-effort solves, the regime of §8.3. Re-running both cells in isolation, and again
in a different object grouping, reproduces the runs-1/3/4 values.

SOLVER_STATE §12's determinism claim holds, including across a multi-object sweep in one
process. The operational constraint is narrower: do not run two sweeps concurrently, and do
not modify planner source while one is in flight. Treat any overlapped run as void.

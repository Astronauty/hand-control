# FRoGGeR as a benchmark arm

**Status: design note, Phase 0.** Nothing here is implemented yet. This file fixes what
the comparison *is* before any code is written, so the ablation stays honest about what
the two arms share and what they do not.

Paper: [FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric][paper]
(Li, Culbertson, Ames, et al., IROS 2023). Reference code: [alberthli/frogger][code].

[paper]: https://arxiv.org/abs/2302.13687
[code]: https://github.com/alberthli/frogger

---

## 0. The thing to be clear about first

**We already have the min-weight metric.** `_embed_gws_ca`
(`simulation/grasp_planner_3d.py`) is the paper's LP (2a)-(2d) verbatim:

```
max_{alpha, beta}  beta    s.t.  W(q) alpha = 0,  sum(alpha) = 1,  alpha >= beta * 1
```

`docs/SOLVER_STATE.md` already names it "the FRoGGeR min-weight metric". So this benchmark
is **an ablation of two formulations that share a metric**, not a port of a method we
lack. Describing it as a reimplementation would misstate the contribution.

## 1. What actually differs

| | FRoGGeR (7a-7e) | ours |
|---|---|---|
| decision variables | `q` alone | `q`, patch coords `t1/t2/t3`, `gamma`, cone coeffs `y`, slacks `s` |
| contacts | FK outputs, pinned by SDF equality `s(FK_i(q)) = 0` | 2-DOF coordinates on a locally-fitted quadratic patch + trust region |
| normals | `-grad s(p)` from the object SDF | patch-symbolic (`quadratic_symbolic_normals`) or Picard-frozen |
| objective | `max l*(q)`, sole term | weighted sum of ~10 terms; `w_gws = 5.0` is one of them, `w_ik = 0.70` dominates |
| closure | hard constraint `l_bar* >= k_l = 0.3` | none — `beta` is free, only pushed by its cost weight |
| collision | witness points via Drake, `sigma >= d_j`, `d_j < 0` for finger-object | softplus bounding-sphere vs primitive + baked SDF table |
| solver | NLopt SLSQP | IPOPT (SQP available via `--backend`) |
| fingers | 4 (Allegro) | 2 today; 3 slots exist, tripod is planner-side only |

The interesting question is not "can we run their code" but: **does the leaner
formulation beat the patch machinery on our hardware, and where?**

## 2. Two decisions, and why

### 2.1 Finish the `n >= 3` wiring first

A 2-contact pinch has a rank-5-of-6 wrench matrix (this is why
`project_grasp_axis_torque` exists) and therefore **cannot** satisfy `l_bar* >= 0.3` in
any meaningful sense. Benchmarking FRoGGeR as a pinch would test a claim the paper does
not make.

`SOLVER_STATE.md` §10 already lists the gap: `build_W_ca` accepts `extra_contacts`, but
the sole call site — the `w_gws`/`w_span` block — does not pass it, so `beta` is computed
from a 2-contact `W` even when the NLP solved a third contact. §4 argues that at `n > 2`
opposition is the wrong objective *because* `beta` measures the right thing, and that
argument does not hold until the third column is actually in `W`.

So Phase 1 is not benchmark scaffolding; it is the top open item in the solver doc, and
the benchmark is its forcing function.

### 2.2 The FRoGGeR arm uses true SDF normals

`SOLVER_STATE.md` is explicit that **`beta` is only as true as the patch its normals come
from**, with two measured cases: a reported `beta = +0.0692` that recomputes to `-0.0` on
true normals, and `061_foam_brick` at `beta = +0.0996` on a grasp where
`arccos(n1 . n2) = 76.3 deg` against a `61.4 deg` friction limit — force closure
*geometrically impossible*. The two disagree on 17% of solves.

FRoGGeR's entire objective is `beta`. Feeding it patch normals would have it optimize
exactly that fiction. The paper's own formulation takes normals as `-grad s(p)` from the
SDF, so using the baked `object_sdf` gradient is both the faithful reading and the
defensible one.

Consequence worth stating: this makes **surface representation a second measurable axis**,
not a confound. Our patch exists because SDF-Hessian curvature was measured to be garbage
(§2: `-12.33` on a flat face, sourced from a corner 50 mm away) — which is a real finding
against FRoGGeR's finite-differenced mesh Hessian (their `delta ~ 10x` mean edge length).
Ours is analytic from the spline. That disagreement is a result, and it should be reported
as one.

**Neither arm's `beta` is a certificate.** Both are scored by `_span_margin` (a pure
geometric test on true normals) and `solve_gamma_live` (the wrench LP), exactly as
`verify()` does today.

## 3. What the two arms share

Shared, so the comparison is about the formulation:

- object set, poses, friction, and scene (`table_scene`)
- seeding pool (both get the OBB palm sampler from Phase 2 *and* our existing seeds)
- collision model (bounding-sphere + SDF table — ours, for both)
- execution: approach, hold, gap gate, squeeze ramp, lift, transport
- scoring: `_span_margin`, `solve_gamma_live`, true epsilon from `composite_wrench_cone`,
  phase_log outcomes

Not shared (this *is* the experiment): decision variables, objective, closure constraint,
normal source.

## 4. Known limitations to report, not fix

- **Robot-side collision is bounding-sphere, not witness-point.** Conservative relative to
  Drake, so FRoGGeR may look more constrained on our arm than in the paper. Stated, not
  engineered around.
- **Our LEAP hand has 3 NLP contact slots; the paper uses 4 Allegro fingers.** Results are
  3-contact and should be labelled as such.
- **`FINGER_SET` is import-time** (§10), so today `--pairing` steers the planner only.
  Phase 1 fixes this; until it does, execution runs with a non-default pairing are not
  meaningful.

## 5. Phases

0. **This note.**
1. **`n >= 3` wiring** — `extra_contacts` into `build_W_ca` at the `w_gws` call site;
   `verify()` past hardcoded `n = 2`; `FINGER_SET` runtime rather than import-time.
2. **`frogger` planner mode** — sole `beta` objective, SDF surface equality on free
   3-vectors in place of patch coords, `beta >= k_l / m` hard constraint, true SDF
   normals. Plus the OBB palm sampler (noisy alignment to OBB axes weighted by side
   length, palm 4 cm out, collision-free IK) as a seed generator beside
   `_fixed_antipodal_seed`.
3. **`benchmarks/ycb_grasp/frogger_bench.py`** — `--planner {ours,frogger}`, N objects x M
   seeds, logging `l_bar*`, true epsilon, `gamma_min`, span margin, solve time, solve
   count, pick success. Adds the paper's **shaky pickup**: lift 10 cm in 1 s, hold 1.5 s,
   3 mm sinusoidal perturbation in all axes from t+0.25 s; fail on >30 deg rotation or
   >7.5 cm deviation, or >60 s synthesis.
4. **Teleop setpoint path** — FRoGGeR as an alternate producer of `obj['q_target']`,
   reusing the existing `_commit_recommended_pose` -> `rebranch` -> `plan` -> GRASP-hold
   contract unchanged.

Phases 1-3 produce the result; Phase 4 is the demo and can be pulled forward.

## 6. Paper reference numbers

43 pruned watertight YCB objects, `mu = 0.7` (optimizer assumed 0.5), uniform density
150 kg/m^3, 20 trials each. Overall: **99.4% converged** vs 44.8% baseline; **78.8% pick
success** vs 58.0%; epsilon `4.6e-3` vs `2.0e-3`; normalized `l*` 0.58 vs 0.19; **0.21 s
per solve, 3 solves, 0.83 s total** vs 0.73 s / 50 / 13.8 s. They note `l_bar*` is a
*noisy* predictor of success: 0.61 (0.49, 0.67) for successes vs 0.47 (0.39, 0.60) for
failures — worth replicating, since it bears directly on whether a `beta`-driven
objective is the right one.

Our object set is 85 in `assets/ycb_mjcf/`; the watertight-prune list needs to be
established before any cross-paper number is quoted.

---

## 7. Phase 1 progress (2026-09-13)

### 7.1 The `extra_contacts` gap was already closed

`SOLVER_STATE.md` §10 says the `w_gws` call site "does not pass" `extra_contacts`, so
`beta` is a 2-contact number even at `n_contacts >= 3`. **That is stale** — commit
`5fabede` ("third contact enters the GWS metric") fixed it, and the call site now passes
`_gws_extra` and warns when contact 3 has no frame. The doc needs updating, not the code.

### 7.2 `verify()` generalized to n contacts (done)

This was the real remaining gap. `verify()` built `_pos_v` from `p1`/`p2` only and passed
`n=2` to the wrench LP, so at `n_contacts >= 3` it certified **the thumb+index pair while
the NLP had solved a tripod** — a certificate for a different grasp than the one executed.

Generalized to every contact present, following `solve_gamma_live`'s conventions:
moment reference = contact **centroid** (reduces to the midpoint at n=2), and the
grasp-axis moment/torque projections gated to `n == 2`. That gate is the load-bearing
part: a pinch genuinely cannot resist torque about the line through its contacts, so
projecting is honest; a third contact off that axis removes the premise, so projecting at
n>=3 would make the certificate conservative against a capability the tripod has.

`verify()` now also returns **`n_contacts_verified`**.

> **Read `n_contacts_verified` before comparing `gamma_min` across runs.** A 3-contact
> `gamma_min` is a *different quantity* from a 2-contact one — strictly less projected.
> Measured on `036_wood_block` seed 0: pinch `gamma_min = 28.387 N` (projected), tripod
> `gamma_min = 38.855 N` (unprojected). The tripod is not "worse"; it is certified against
> a disturbance the pinch was excused from resisting.

Regression: the n=2 path is bit-identical (`036_wood_block` seed 0, `gamma_min = 28.387`,
`beta` unchanged to all digits).

### 7.3 `beta` instrumentation (done)

- **`simulation/beta_audit.py`** — rebuilds W from `_geom_normal_np` (the TRUE surface
  normal) at the solve's own contact points and re-solves the min-weight LP in numpy.
  Imports the planner's own `_friction_cone_verts` / `_build_contact_frame_3d` /
  `_span_margin` so it cannot drift from what the NLP means by a primitive wrench.
  Generalizes to n contacts. Read-only; nothing on the solve path.
- **`benchmarks/ycb_grasp/beta_sweep.py`** — plan-only sweep printing reported vs true
  `beta` per solve, with `span_margin` and `gamma_min` alongside.

LP validated against two facts the planner docstring states independently:

| check | expected | measured |
|---|---|---|
| symmetric antipodal pinch | `alpha = 0.1*ones`, `beta = 1/n_cols` exactly | `beta = 0.100000000000`, `alpha` all equal, `‖Wα‖ = 3.5e-17` |
| `061_foam_brick` splay 76.3 deg at mu=0.6 | closure impossible | `beta_true = -0.0`, `span_margin = -0.729` |

---

## 8. Measured: reported `beta` vs true-normal `beta` (2026-09-13)

`benchmarks/ycb_grasp/beta_sweep.py`, 6 objects x 3 seeds, plan-only, 80/3 GWS preset,
`thumb,index` (n=2), tabletop scene. `beta_true` is the same LP re-solved on
`_geom_normal_np` normals at the solve's OWN contact points, so the only thing that
differs is the surface the normals come from.

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
061_foam_brick      2  converged     +0.08981    +0.08972   +0.00009     +2.1298       1.143
```

### 8.1 The headline: the NLP is mostly PESSIMISTIC, not optimistic

**15/18 solves have `delta <= 0`** — the reported `beta` is *below* the true-normal value.
This is the opposite of the framing in `_embed_gws_ca`'s docstring, which is written around
the optimistic failure mode (reporting closure where none exists). That mode is real but
**rare: 1/18 here** (`009_gelatin_box` sd0). The dominant error is the NLP *understating*
its own grasp.

Why it matters for FRoGGeR: a pessimistic `beta` is a **weak objective**, not an unsafe
certificate. The solver is climbing a surrogate that reads lower than the truth, so it
keeps pushing on grasps that are already good. That is a direct argument for FRoGGeR's
formulation — a hard `l_bar* >= k_l` constraint on a *trustworthy* beta — over our current
"beta as one weighted cost term among ten".

### 8.2 Four sign flips: `beta` crosses zero differently from the geometry

| object | sd | `beta_rep` | `beta_true` | span_margin | wrench cert |
|---|---|---|---|---|---|
| `014_lemon` | 1 | **-0.00926** | +0.05383 | +1.2973 | feasible, γ=2.03 |
| `056_tennis_ball` | 1 | **-0.05220** | +0.07913 | +1.5650 | feasible, γ=2.38 |
| `056_tennis_ball` | 2 | **-0.05220** | +0.07913 | +1.5650 | feasible, γ=2.38 |
| `009_gelatin_box` | 0 | **+0.07438** | -0.00000 | +0.1626 | **INFEASIBLE** |

The first three report NON-closure (`beta < 0`) on grasps that are genuinely in closure by
both independent certificates. Under a FRoGGeR-style hard constraint `l_bar* >= 0.3` these
three would be **rejected as infeasible despite being good grasps** — so porting the hard
constraint onto patch-derived normals would not merely be unsafe, it would throw away
working grasps. This is the strongest evidence yet for §2.2's decision to give the FRoGGeR
arm true SDF normals.

The fourth is the classic optimistic case and reproduces the docstring's pattern exactly:
`beta = +0.074` on 62.4-deg-splay contacts whose wrench LP is infeasible.

### 8.3 `beta`'s error tracks solver convergence

| status | n | median abs(delta) | max abs(delta) |
|---|---|---|---|
| `converged` | 7 | **0.00312** | 0.06319 |
| `best-effort` | 11 | **0.01897** | 0.13132 |

A 6x median gap. Every large disagreement is a best-effort solve, and SOLVER_STATE §11
already establishes those are IPOPT cycling under a genuinely indeterminate dual (the
active set is degenerate under the antipodal minimax symmetry), *not* truncation. So
`beta`'s unreliability is substantially **an artifact of stopping mid-cycle**, which is a
different and more tractable diagnosis than "the patch is wrong".

`009_gelatin_box` is the known seed-starved object (SOLVER_STATE §9). Both its bad cells
are geometric, not numerical: 62.4 deg and 87.7 deg splay, vs 175.4 deg on the one good
seed. The audit agrees with the geometry (`beta_true = -0.0` on both).

### 8.4 Determinism: confirmed, after a false alarm

**The sweep IS deterministic.** Four full 18-cell runs: runs 1, 3 and 4 are byte-identical
(md5 `29edc6c1ad41` over the table body). Run 2 differed in exactly 2 cells
(`056_tennis_ball` sd0, `061_foam_brick` sd2) and was the outlier, not the rule.

Run 2 is explained by how it was launched, not by the solver: it was started while run 1
was **still executing**, so two 18-cell sweeps shared the machine (and its BLAS thread
pools) for run 2's entire duration. The two cells that moved are in the last two objects of
the sweep order — the ones still solving under contention. Both are `best-effort` solves,
i.e. exactly the IPOPT-cycling regime SOLVER_STATE §11 describes, where the iterate the cap
lands on is the least stable thing about the solve.

So SOLVER_STATE §12's determinism claim **holds**, including across a multi-object sweep in
one process. The operational rule is narrower than "one object per process":

> **Do not run two sweeps concurrently, and do not edit planner source while one is in
> flight.** Under contention, `best-effort` cells can land on a different iterate. Treat
> any run that overlapped another as void rather than as evidence.

An earlier version of this section claimed order-dependent intra-process state, on the
strength of `056_tennis_ball` sd0 reproducing `-0.05220` in isolation. That reproduction was
real but the inference was wrong — isolation reproduced the value because isolation is the
*uncontended* case, which is also what runs 1/3/4 were.

### 8.5 The tripod's `beta` is far more trustworthy

`036_wood_block` seed 0, same object and seed, only the finger list changed:

| fingers | n | `beta_rep` | `beta_true` | delta | `gamma_min` |
|---|---|---|---|---|---|
| `thumb,index` | 2 | +0.08970 | +0.09282 | -0.00312 | 28.387 (projected) |
| `thumb,index,middle` | 3 | +0.03660 | +0.03665 | **-0.00005** | 38.855 (unprojected) |

**A 60x tighter agreement at n=3.** With only two contacts the wrench hull is rank-5-of-6,
so the min-weight LP's optimum is soft in the unspanned direction and a small normal error
moves `beta` a lot. A third off-axis contact spans the sixth direction and pins the hull, so
the same normal error barely moves `beta`.

This matters for the FRoGGeR arm beyond faithfulness: it says `beta` is not uniformly
untrustworthy — **it is untrustworthy mainly where the grasp is rank-deficient.** The
formulation FRoGGeR actually proposes (>=3 contacts, hard `l_bar* >= k_l`) sits in the
regime where `beta` behaves, while our measured 2-contact default sits in the regime where
it does not. That is a point in the paper's favour that our own n=2 numbers would hide.

The absolute `beta` is LOWER at n=3 (+0.037 vs +0.090) purely because `beta`'s ceiling is
`1/n_cols` and n_cols goes 10 -> 15; compare `beta_true_scaled` (`beta * n_cols`) across
contact counts, not raw `beta`. Likewise `gamma_min` is not comparable here — see §7.2.

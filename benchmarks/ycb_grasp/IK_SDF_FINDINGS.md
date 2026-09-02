# Constrained-IK on YCB meshes: SDF integration, convergence fixes, backend ablation

Status as of this doc: SDF collision is wired into `ConstrainedIKSolver` and verified
correct; the "IK doesn't converge" problem that blocked any before/after comparison is
fixed; the solver-backend ablation (§5) has run its full sweep (107/108 solves — see
§5a for the one missing row) and points at `sqp-ms4` as the default, with one
structural failure (`003_cracker_box`) still to root-cause before calling that final.
This is the working design doc for that effort — see `benchmarks/ycb_grasp/` for the
code and the original plan at `the-current-kinova-pick-groovy-dahl.md` for the
scene/workspace half of the project.

---

## 1. The problem this solves

`ConstrainedIKSolver` (`grasp_control/constrained_ik.py`) has no branch for mesh geoms.
A mesh falls through to `_sphere_sphere_distance` and is modelled by its bounding
sphere — for `006_mustard_bottle`, r ≈ 12.6 cm around a 3×5×10 cm object. Measured
effect on the YCB demo (`benchmarks/ycb_grasp/ik_demo.py`): fingertip placement error
6–18× worse than with no object-collision constraint at all, i.e. the sphere proxy was
actively blocking reachable grasps rather than merely being imprecise.

`GraspPlanner3D`'s `_sym_geom_surface_con` (recommender side, not yet touched) has no
mesh branch either, and — worse — no `else`, so a mesh contact silently gets *no*
surface constraint.

## 2. Object SDF (`grasp_control/object_sdf.py`, 284 lines)

One precomputed signed-distance table per object **shape**, in the object's body frame.
A robot body's distance to the object is `table(R_obj^T (x_body − c_obj)) − r_body`, so
the table is pose-invariant (moving the object costs nothing — verified exactly
pose-invariant to 1e-16) and shared by every colliding body.

**Build**: `body_hull_halfspaces` reads exact half-space form `(A, b)` per collision
hull straight from `model.mesh_vert` (no OBJ/trimesh dependency, no frame-convention
risk — hull volume from the model matched the on-disk OBJ exactly). `bake` samples
`union_sdf_np` (min-over-hulls-of-max-over-faces) on a 64³ lattice over the hull bbox
+ 2 cm padding, measures the interpolant's optimism against ground truth on 4000
probes, and stores that as `safety_offset`.

**Query** (`casadi_fn`): a CasADi bspline interpolant, clamped into the grid box with
the distance travelled added back for far-field queries (unclamped, the interpolant
decays to 0 outside its box, which reads as "touching" at any distance and makes the
NLP infeasible everywhere). `conservative=True` (default) subtracts `safety_offset` —
correct for collision constraints, wrong for placing contacts, which need
`conservative=False` or every contact lands `safety_offset` outside the true surface
(measured: exactly reproduced a 0.64–0.89 mm bias before this was split out).

**Verified per object** (mustard/mug/banana — 7/44/5 hulls):
distance error p95 ≤ 0.74 mm below ground truth (conservative, by construction of the
offset); conservative fraction 99.9–100%; gradient-based normal error mean 1–5°
(vs. 10.67° mean / 103.56° max measured for a reduced-polytope alternative that was
tried and rejected — see the plan doc). `surface_project` (5-step **unit-direction**
walk toward the surface, not Newton) lands within ~0.9 mm max residual; Newton
diverges up to 354 mm at points where `‖∇d‖` collapses near concave creases (field is
not eikonal — verified this is a property of the geometry, not of max-of-planes vs.
true-Euclidean baking).

**Performance fix along the way**: naive `union_sdf_np` is O(points × total faces) —
~10¹⁰ ops for the 44-hull mug at 64³. A bounding-sphere prune looked obvious but is
**unsound**: it bounds true Euclidean distance, while the stored quantity (max over a
hull's half-spaces) is itself an underestimate near edges/vertices, so the sphere bound
can exceed the quantity it's meant to bound and prune the hull that actually holds the
minimum (measured 17.7 mm of corrupted output before this was caught). Fixed with a
cheap lower bound from a **subset of each hull's faces** (any subset's max is a valid
lower bound on the full max) — exact to 1e-17 and 2–4× faster.

## 3. `ConstrainedIKSolver` integration

New `sdf_bodies=(...)` constructor kwarg. `__init__` bakes/loads one table per named
body (cached to `assets/ycb_sdf/<hash-of-verts>.npz`, so remeshing invalidates
correctly) and **drops** any explicitly-listed geom belonging to that body — one
constraint per (arm geom, object), not per hull. `solve()`'s existing per-geom loop is
otherwise untouched: the object pose enters only as a constant transform
(`data.xpos`/`xmat`, refreshed per solve), matching how box/plane/cylinder objects
already work.

Pruned SDF pairs are tracked and rechecked separately from `pruned_pairs`
(`mj_geomDistance`-based) rather than jammed into the same list with a `gid2=-1`
sentinel — that sentinel would silently negative-index to the last geom in the model,
the same bug class the `_obj_gids` filter elsewhere in this file exists to prevent.

**Measured effect** (single IPOPT solve, before the convergence fix below — see §4 for
why these numbers were later found to be noise-dominated and superseded):
mustard 27.23→ ~1–4 mm tip error, mug constraint count 3195 pairs → 126 (885 pruned →
16), solve time 12.4 s → 2–3.5 s.

## 4. Convergence: the last IPOPT iterate is often much worse than the best one

Root cause found via `opti.debug` trace on a failing solve: constraint violation hits
0 by iteration ~10 (feasibility was never the problem), the objective descends cleanly
to near-converged by iteration 200, then **jumps** at iteration 250 (IPOPT watchdog/
restoration markers `w`/`H` in the log) and never recovers by the iteration cap.
Quantified by recording every iterate: **mug returned 34.73 mm, but the solver had
already found 0.52 mm at iteration 257** — a 67× gap, discarded because `solve()`
returned the *last* iterate on failure (the docstring claimed "best-iterate on
failure" — it did not do this).

**Fix** (`ConstrainedIKSolver.solve(..., track_best=True)`, default on): record `q` at
every `opti.callback`, then score objective + max constraint violation for all
recorded iterates in one batch and return the best **feasible** one (`_FEAS_TOL=1e-6`).
Only affects failed/capped solves — a converged solve's last iterate is already its
best. `last_metrics['best_iter']` is `None` when nothing beat the final point (a
genuinely converged solve), which is itself informative and is reported as such rather
than as NaN.

**Effect, three objects, single IPOPT run**: 4.17→0.73 mm (mustard), 34.73→0.52 mm
(mug), 1.97→0.05 mm (banana). This is the dominant fix in this whole effort — larger
than the SDF integration itself for the numbers that were being compared.

**S3 (exact NLP Jacobian), tried and reverted to default-off.** Hypothesis: FD-over-
NLP gradient noise (existing IPOPT config uses
`jacobian_approximation=finite-difference-values`, chosen because analytic box/
cylinder SDFs have kinks) was contributing to the late divergence, and the grid SDF is
C² (finite Hessians verified at all near-surface probes) so should tolerate exact
Jacobians. Measured on SDF-only constraint sets (no box/cylinder in play, so this was a
fair test): mustard 0.73→0.28 mm (better), mug 0.52→1.25 mm (worse), banana 0.05→0.49 mm
(worse). No consistent gain. **The divergence was about which iterate got returned,
not gradient quality** — kept as an opt-in `jacobian='exact'`, default stays `'auto'`
→ `'fd'`.

## 5. Backend ablation (`benchmarks/ycb_grasp/ablate_ik.py`)

Harness: same DLS warm start, same pinch targets, per object/seed, across arms.
Two fairness details the harness handles explicitly: `configure_sqp` swaps CasADi
module globals **process-wide**, so every arm calls `configure_ipopt()` first
(otherwise an "IPOPT" arm run after an SQP arm silently measures SQP internals — this
is exactly the failure mode the `_SQP_PATCHES`/`_PRISTINE_GLOBALS` refactor in
`constrained_ik.py` exists to make impossible to forget); and `_SQP_SOLVER_OPTS`
carries its own `max_iter=800` which is overridden per run so SQP doesn't get 2.7×
IPOPT's iteration budget for free.

Object set: 12 objects spanning 1–55 collision hulls (`003_cracker_box` … `024_bowl`).

### 5a. Full sweep: 12 objects × 3 seeds × {ipopt, sqp, sqp-ms4} — 107/108 solves

The 108th (`sqp-ms4`/`024_bowl`/seed 2) is missing — the run was killed by the harness's
background-task ceiling one solve short of finishing, before its own summary/CSV step
ran. Reconstructed from the console log (each row is unambiguous — object, seed, arm,
error, time, iterations, restarts used — so nothing here is estimated) and saved to
`benchmarks/ycb_grasp/out/ablate_ik_3arm_seeds3.csv`. Treat that CSV, not the console
log, as the source of truth going forward.

| arm | n | err median | err p90 | <1 mm | <5 mm | solve median | solve p90 |
|---|---|---|---|---|---|---|---|
| ipopt | 36 | 0.56 mm | 6.51 mm | 67% | 86% | 2405 ms | 2483 ms |
| sqp | 36 | 0.83 mm | **51.34 mm** | 50% | 58% | **158 ms** | 357 ms |
| **sqp-ms4** | 35 | **0.12 mm** | 6.75 mm | **83%** | 86% | **253 ms** | 1061 ms |

By hull count — single-shot SQP is *better* than IPOPT on complex objects and *worse*
on simple ones, the opposite of the naive expectation; `sqp-ms4` recovers the simple-
object failures without losing the complex-object win:

| hulls | ipopt | sqp | sqp-ms4 |
|---|---|---|---|
| 1–3 | 0.50 mm / 2383 ms | 13.86 mm / 120 ms | 0.22 mm / 262 ms |
| 4–20 | 0.53 mm / 2426 ms | 1.15 mm / 175 ms | 0.12 mm / 280 ms |
| 21+ | 0.69 mm / 2400 ms | **0.34 mm** / 148 ms | 0.12 mm / **157 ms** |

**Every SQP solve reports `Solve_Succeeded`**, including an 81.86 mm bowl result. SQP
is not failing to converge — it converges fast and reliably (28–182 iterations, always
"succeeded") to a genuine local minimum that is sometimes a bad one. `status` is not a
usable success signal for either backend; only the tip-error residual is.

`sqp-ms4` is the clear headline: **9.5× faster than IPOPT at the median, better median
accuracy, and higher `<1mm` rate**, at the cost of a p90 tail IPOPT doesn't have (IPOPT
never spikes past 6.51mm at p90; either arm has a bad-case tail somewhere).

**Restart usage** (n=35): 1 restart sufficed 51% of the time, 2→20%, 3→6%, all 4 used
23% — so the early-stop-at-1mm is doing real work, not just always burning the full
budget.

**Residual failures, even after 4 restarts** (5/35, >5 mm): `003_cracker_box` on 3 of
its 3 seeds (7.4, 29.6, 25.0 mm), `077_rubiks_cube` once (13.9 mm), `025_mug` once
(5.7 mm, borderline). **`003_cracker_box` failing on every seed, not intermittently, is
a structural signal, not restart bad luck** — it has only 1 collision hull (CoACD found
the box nearly convex already), which rules out "geometry is too complex" as the cause
and points instead at the pinch-target heuristic (`pinch_targets_from`'s minor-
principal-axis choice) or the DLS warm start behaving badly on a large, flat,
axis-dominant shape. Not yet root-caused — see §6.

## 6. Open items

- **Full sweep is done** (107/108 solves, §5a); numbers there are confirmed, not
  preliminary. One row (`sqp-ms4`/`024_bowl`/seed 2) never ran — the background process
  was killed one solve short — and would need a single targeted re-run to fill in, not
  a full sweep repeat.
- **Root-cause `003_cracker_box` failing all 3 seeds.** This is now the top item: it's
  the one result in §5a that looks like a bug rather than expected NLP non-convexity.
  Next step is comparing its DLS warm-start quality and pinch-target placement against
  a passing object of similar hull count (`013_apple`, `077_rubiks_cube` — also 1 hull,
  and only `077_rubiks_cube` failed, and only once) to isolate whether it's the target
  heuristic or the warm start.
- **`check_analytic_jacobians` has not yet been run against the SQP-mode analytic FK
  callbacks** on these YCB scenes. SQP's failure signature (fast, reports success,
  sometimes wrong) is also exactly what a subtly-wrong analytic Jacobian would produce.
  Should be ruled out before recommending `sqp-ms4` as a default — multi-start could be
  papering over a real bug rather than a benign local-minimum problem. Higher priority
  now that `003_cracker_box` shows a *consistent* failure, which a bad Jacobian could
  also explain and random restarts alone would not fix.
- `n_starts=4` and the restart σ (0.25/0.35 rad) were unvalidated guesses going in;
  §5a's restart-usage histogram (51% need only 1, 23% use all 4) suggests 4 is in a
  reasonable range, not obviously too small or wastefully large — but this wasn't tuned,
  only checked after the fact.
- GWS objective (`gws_nlp_implementation_brief.md`) work has not started; it depends on
  a converged, trustworthy IK baseline, which is the point of this doc.
- Recommender-side mesh support (`_sym_geom_surface_con`, `_geom_normal_np`) is
  unstarted — the SDF module's `surface_project`/`casadi_normal_fn` were built with that
  consumer in mind but nothing calls them yet.

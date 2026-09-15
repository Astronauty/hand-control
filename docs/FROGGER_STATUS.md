# FRoGGeR reimplementation: current state

**As of 2026-09-14.** What is implemented, what is verified, what is known to differ,
and what is not done. Read this first; the companions are longer and chronological.

- [`FROGGER_BENCH.md`](FROGGER_BENCH.md) — the benchmark, and the dated record of every
  measurement that produced the state below.
- [`FROGGER_COMPARISON.md`](FROGGER_COMPARISON.md) — component-by-component comparison
  of the two formulations.
- [`GWS_IMPROVEMENTS.md`](GWS_IMPROVEMENTS.md) — candidate work on our own `beta`
  machinery, which the port's instrumentation surfaced.

Paper: [arXiv 2302.13687][paper] (Li, Culbertson, Ames et al., IROS 2023).
Reference implementation: [alberthli/frogger][code].

[paper]: https://arxiv.org/abs/2302.13687
[code]: https://github.com/alberthli/frogger

**Symbols.** `beta` — the min-weight metric. `l_bar* = n_cols * beta` — FRoGGeR's
normalized form, in `[-1, 1]`, the quantity everything below reports. `k_l` — their
robustness floor, 0.3. `lp_gap` — the embedded LP's own optimality gap.

---

## 1. Can it be benchmarked against? Yes, with two qualifications

The port reproduces the paper's method component by component, each verified
numerically rather than by inspection. The strongest evidence is not a metric value
but the ATTEMPT DISTRIBUTION: their synthesis loop needs a median of 3 solves per
feasible grasp (IQR 1-6), and ours needs a median of 2 (IQR 1-6). That quantity was
not tuned for — it falls out of the loop — so landing on theirs indicates the
implementation behaves like the method.

**Qualification 1: execution scoring now exists; the frogger arm mostly does not
reach it.** `benchmarks/ycb_grasp/frogger_exec_bench.py` (2026-09-14) runs their shaky
pickup against live grasps through the shared executor. Pilot result, 2 objects x 3
seeds: `ours` reached the squeeze 6/6, `frogger` **1/6**, with five cells aborting at
the gap gate on fingertip gaps of 8-11.5 mm -- the `frogger_pad_offset_m` of 0.011.
Their (7d) pins a FIXED body-frame pad point to the surface while the gate measures the
real tip-geom surface, so the solve is satisfied with the hand a centimetre away, and
`l_bar*` reports +0.9996 on a grasp nobody is holding. See FROGGER_BENCH §9.

**Qualification 2: solve times are not comparable.** Ours 2.4 s, frogger 17.5 s, but
most of that is failing cells burning the full 60 s synthesis budget, which our
single-solve configuration never pays. A fair comparison needs the same budget on
both, or a per-successful-attempt figure.

---

## 2. Where it stands, at the paper's own protocol

`mu = 0.7`, resample-until-feasible (20 attempts, 60 s budget), `n = 2`
(thumb+index), 5 objects x 3 seeds. **Measured on a dirty tree** — see §6.

| | ours | frogger |
|---|---|---|
| median `l_bar*` | **+0.6030** | +0.3167 |
| wrench-feasible | 13/15 | **14/15** |
| median `\|lp_gap\|` | 0.00097 | **0.00000** |
| median solve time | **2.4 s** | 17.5 s |
| convergence (their criterion) | n/a | 93% |

Against the paper: convergence 93% vs 99.4%, `l_bar*` over converged 0.325 vs 0.58.

Ours reports the higher metric value; the frogger configuration has more feasible
cells and an exactly-zero optimality gap, the latter because its bilevel LP solves to
optimality at every iterate where our embedded one does not.

**Not a method comparison.** Our configuration's own `l_bar*` is UNDERSTATED by about
0.137 by its patch normals (FROGGER_BENCH §8), the two differ in normal source by
design, and `n = 2` is not the configuration the paper tests — they use four Allegro
fingers.

---

## 3. Implemented and verified

| component | paper | verification |
|---|---|---|
| min-weight metric | (2a)-(2d) | `beta = 1/n_cols` exactly on a symmetric antipodal pinch; `alpha` uniform; `\|\|W alpha\|\| = 3.5e-17` |
| bilevel LP + implicit-KKT gradient | Prop. 1, eq. (6) | matches central differences to ~1e-11 away from degenerate vertices; `lp_gap` exactly 0 on every cell |
| 4-sided friction cone | App. B-F | removing our 5th (origin) vertex took exact-zero cells from 11/18 to 0 |
| FK contacts + fixed contact point | (7d), App. B-F | tip-to-contact 0.0 mm by construction; pad surface within ~0.03 mm |
| exact collision + analytic gradient | (7e), eq. (8) | matches central differences to 1.9e-10; 34.5 s -> 5.3 s |
| robustness floor | (7c) | `beta >= k_l / n_cols`, cold start moved inside the floor |
| collision margins | App. B-F | 3 mm finger-object, 1 mm elsewhere |
| SLSQP solver family | Sec. IV | CasADi `sqpmethod` + OSQP |
| OBB heuristic sampler | App. B-C | straddles the object on 60-100% of reachable draws |
| resample-until-feasible | Sec. IV, Table I | median 2 attempts (IQR 1-6) against their 3 (IQR 1-6) |

Config surface: `GraspConfig3D.frogger_*`, 8 fields assembled by
`grasp_config_builder.for_frogger`. The five behavioural switches
(`fk_contacts`, `bilevel_lp`, `exact_collision`, `raw_friction`, `tol_scaling`) all
default False, so every non-frogger configuration is bit-identical; the three
numeric fields (`finger_obj_margin_m` -0.003, `pad_offset_m` 0.011, `tol_pr` 1e-5)
carry the paper's values but are inert unless their switch is on. `tol_scaling` is
False in the preset too — see §4.

---

## 4. Known differences, deliberate or unresolved

| difference | status |
|---|---|
| `n = 2` against their 4 fingers | the `n >= 3` path reaches `l_bar* = +0.078` and is not working |
| `009_gelatin_box` excluded | OUR deviation — they kept it. Fails all 20 attempts at `n = 2`; it is the flattest object (34 mm) and not flat relative to a 4-finger Allegro |
| per-constraint tolerances (Table III) | implemented, DEFAULT OFF, measured harmful: scaling a constraint also scales its gradient, which silently reweights the Jacobian (median `l_bar*` -8.62 against +0.25) |
| our collision model on the `ours` side | bounding spheres, not witness points — conservative |
| sampler draw ranking | we take the first straddling draw; they generate many candidates |
| object set | 5 objects against their pruned 43 |

---

## 5. Not done

1. **A pick-success RATE.** Execution scoring is wired (FROGGER_BENCH §9), but the
   frogger arm reaches the lift in 1/6 pilot cells, so there is no rate to quote yet.
   The pad-offset/gap-gate mismatch in §9.1 is the blocker, and it is a REACHABILITY
   defect, not a grasp-quality one. Note also that their displacement-only criteria
   scored a one-finger carry as a success (§9.2), so `pick_success` must be read
   alongside `lift_ok`.
2. **`n >= 3`.** The tripod sampler places three fingers and the solve leaves the floor
   (+0.078), but the result is not a good grasp. Blocks the gelatin box's return, any
   comparison at their contact count, AND the Ferrari-Canny `epsilon` column, which is
   undefined at n = 2 (a pinch's wrench set is rank-5-of-6, so it contains no 6-ball --
   FROGGER_BENCH §9.4).
3. **Remaining finite-differenced paths.** The collision gradient is analytic; other
   callbacks are not, which is part of the remaining solve-time gap.

---

## 6. Reproducibility

**Every number here was measured with uncommitted local changes present**, notably in
`simulation/grasp_config_builder.py` (a `w_gws` signature change and an `n_seeds`
change that predate this work). They move the `ours` baseline measurably — `l_bar*`
0.5877 with them stashed against 0.7476 applied on one cell — so figures taken at
different points are not necessarily comparable, and none are reproducible from a
commit hash alone. Anything paper-bound should be re-measured against a committed
tree.

Do not run two sweeps concurrently: under contention, best-effort cells can land on a
different iterate (FROGGER_BENCH §8.7).

## 7. Running it

```bash
cd benchmarks
# paper protocol: their friction, their resampling
uv run python -m ycb_grasp.frogger_bench --mu 0.7 --max-attempts 20 --seeds 0,1,2

# single solve at our own scene friction (the default, comparable to prior results)
uv run python -m ycb_grasp.frogger_bench --seeds 0

# one configuration only; --patch-normals isolates objective from parameterization
uv run python -m ycb_grasp.frogger_bench --arms frogger --patch-normals
```

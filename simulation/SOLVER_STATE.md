# Grasp solver — current state, and how each environment uses it

**Living document.** Unlike the `*_FINDINGS.md` files beside it (which record closed
investigations and should not be rewritten), this describes what the solver *currently
does* and is meant to be edited whenever that changes. If you change a default, a cost
term, a constraint, or the way an environment calls the planner, update this file in the
same commit.

Last verified against: `simulation/grasp_planner_3d.py`, `kinova_common/wrench.py`,
`grasp_control/grasp_controller.py`, `benchmarks/ycb_grasp/{pick_from_floor,pick_and_place}.py`,
`kinova_leap_pick_place.py` — 2026-09-11.

**Seed/surrogate settings live in `models/grasp_seed_config.json`**, not only in
`GraspConfig3D`'s dataclass defaults — `grasp_config_builder.load_seed_config()` reads it
and both `pick_and_place.py` and `plot_seed_quadratic.py` apply it as DEFAULTS, so an
explicit CLI flag or `PFF_*` env var still wins (file -> per-object -> CLI/env). Current
values: `seed_kappa_max_reject` 150, `quadratic_sdf_err_tol` 4 mm,
`quadratic_t_bound_max` 100 mm, `seed_ground_clearance_m` 5 mm,
`seed_prefer_outer_surface` true.

---

## 0. Pipeline at a glance

```
seed generation  ->  local surface fit  ->  NLP (per Picard stage)  ->  post-process gamma  ->  execute
   _seed_pair         quadratic patch       q, t1, t2, gamma, y        solve_gamma_live       squeeze + jog
```

The NLP's internal `gamma` is **not** what gets executed. See §5.

---

## 1. Seeding

Candidates are generated and inserted in priority order:

1. `_fixed_antipodal_seed` — deterministic, marches along the object's minor axis
   through the hull centroid. Inserted FIRST.
2. `_chart_pair_seeds` — mesh + `use_uv_atlas_contact` only, ranked by chart-normal
   antipodality.
3. `_seed_pair` — random march directions, drawn from `self._rng`, up to `max_attempts`.

All three place their contacts on the object's **outer** surface: the seed ray's zero
crossings are enumerated (`_ray_surface_crossings_np`) and the FIRST and LAST are taken
(`_outer_pair_t`). The scan sizes its own span from the mesh's vertex extent about the
ray origin (`_ray_scan_span_np`) — do NOT pass the seed helpers' `bbox_r`, which is
`max(geom_size)*2.5` and for a MESH is not the half-extent: on `065-a_cups` it evaluates
to 26.4 mm against a far wall already 27 mm from the volumetric centroid, so tilted rays
found 0-1 crossings, fell back to nearest-surface projection, and landed inside the cup
anyway. An ODD crossing count means a clipped or degenerate ray and is rejected rather
than paired. The older rules — project the far ray endpoint to the nearest surface,
and sphere-march until the SDF turns positive — are correct only for a SOLID object. A
cup is two thin shells around a void, and the void reads as OUTSIDE (measured on
`065-a_cups`: +12.20 mm at the volumetric centroid, −2.48 mm inside the wall), so both
rules landed on the INNER wall — 6 of 14 seed contacts. Measured after the change:
`065-a_cups` 12 outer / 0 inner, with `017_orange` and `036_wood_block` unchanged at
14/14 outer. A solid object has exactly two crossings, so first/last reproduce the old
behaviour exactly. Controlled by
`seed_prefer_outer_surface` (default True); inner-surface seeds are NOT supported
downstream, because the squeeze phase drives fingers in the closing direction and would
unload an inner-wall contact rather than press it.

Each candidate is then filtered by `_reachable_contact` (both contacts must sit at least
`ground_z + floor + 2mm`, where `floor` is `seed_ground_clearance_m` when set and the
fingertip's isotropic bounding-sphere radius otherwise) and `_seed_kappa_ok`
(`seed_kappa_max_reject`, evaluated on the MESH-FIT curvature when
`quadratic_mesh_fit` is on, so the gate and the surrogate judge the same surface).

**Consequence worth knowing:** because the deterministic seed goes in first and usually
wins the ranking, varying the planner's `seed` produces NO variation on rigid symmetric
objects (measured: wood block identical across seeds 0/1/2), and only mild variation on
spheres/cups. `seed` is plumbed through `MultiStartGraspPlanner3D(..., seed=)`; the
per-seed `q_ref` jitter (`qref_restart_sigma_arm/hand`) is 0.0 by default, so the RNG's
only live effect is `_seed_pair`'s march directions.

---

## 2. Local surface fit (mesh objects)

`_mesh_local_surface_fit_np` fits mesh **visual vertices** in a band around the seed and
runs a plane-vs-quadratic model-selection test (`quad_gain_min` on RMS residual). Below
the gain threshold it returns kappa=0 and the seed's own tangent axes.

This replaced fitting to the SDF Hessian. **An SDF's second derivative is a GLOBAL
quantity** — it reports curvature belonging to nearby features, not the local patch — and
it was producing fake curvature (-12.33 on a flat face, sourced from a corner 50 mm away)
that collapsed trust regions to 4.6 mm on a 104 mm face.

Patch parameterization, giving each contact exactly **2 decision variables**:

```
p(t) = seed + t0*axis0 + t1*axis1 + h(t)*n
h(t) = -(kappa0*t0^2 + kappa1*t1^2) / (2*grad_norm)
```

Mesh contacts are therefore **on-surface by construction** — no surface equality
constraint is added for them.

### Trust region (where the fit meets the optimizer)

Asymmetric per-side bounds from `_sdf_axis_bound_np`, which marches along each axis until
the paraboloid surrogate diverges from the true SDF by `sdf_err_tol`:

```python
t_lo_0 = -_sdf_axis_bound_np(mesh_entry, seed_l, -axis0_l, t_bound_max, tol=sdf_err_tol)
t_hi_0 =  _sdf_axis_bound_np(mesh_entry, seed_l,  axis0_l, t_bound_max, tol=sdf_err_tol)
opti.subject_to(opti.bounded(t_lo_0, t_var[0], t_hi_0))
```

An axis that reaches `t_bound_max` is flat/uncapped; one that stops short is
divergence-limited (an edge is near). `w_edge_margin`'s hinge fires only on
divergence-limited axes, so a genuinely flat face is not penalized.

Those four searches bound the rectangle's **centre-lines** only, and say nothing about
its corners — which is where a paraboloid departs worst. `_shrink_patch_to_tol` therefore
treats them as an upper bracket and scales all four down uniformly (bisection on a 5x5
grid over the real patch surface) until the WHOLE patch honours `sdf_err_tol`. Uniform,
so the measured four-way asymmetry keeps its shape. Without it, measured patch error ran
to **8.78 mm against a 4 mm tolerance** on `036_wood_block` and the patch visibly hung off
the face it was fitted to; after, every contact on orange/gelatin/block lands at or under
4.0 mm. Costs 1-3 ms per contact. Note the shrink is NOT a rare path: because the axis
searches walk out until the centre-lines reach the tolerance, the corners start over it,
so most contacts on every object measured enter the bisection.

**Measured caveat:** the solution sits AT a trust-region bound (`pinned=True`) 9/9 stages.
The Picard loop takes maximum-length steps, so the final contact is largely
seed + N x bound rather than an interior optimum. Directly relevant to seeding work.

---

## 3. NLP decision variables

| variable | size | notes |
|---|---|---|
| `_q` | n_act | arm + hand joints |
| `_t1_var`, `_t2_var` | 2 each | patch coords (mesh); `_p1`/`_p2` are free 3-vectors for primitives |
| `_gamma` | 1 | wrench-cone scale, bounded `[0, gamma_max=25]` |
| `_y1_k`, `_y2_k` | nverts per corner | cone-vertex coefficients |
| `_s_k` | 6 per corner | wrench slack (when enabled) |
| `alpha`, `beta` | n_cols, 1 | FRoGGeR min-weight LP (when `w_gws > 0`) |

---

## 4. Cost terms and constraints

Costs are normalized so each is ~1 at its reference level.

| term | default weight | expression |
|---|---|---|
| `ik` | **0.70** | `0.5*(d1^2+d2^2)/d_ref^2`, `d_ref = 5 mm` |
| `reg` | 0.03 | `\|\|(q - q_reg)/q_scale\|\|^2 / n_dof` |
| `gamma` | 0.15 | `gamma_lp / g_ref` (normalized by task load) |
| `y` | 0.6 | `sum \|\|y\|\|^2`, min-norm force distribution |
| `slack` | 1.0 | wrench-infeasibility penalty |
| `align` | 0.0 | `\|\|g_hat - n1_in\|\|^2`, grasp-axis opposition |
| `orient` | 0.0 | `\|\|R_tip*pad_axis - n_in\|\|^2` per contact |
| `gws` | 0.0 | `-beta` (see below) |
| `span` | 0.0 | `-logdet(W W^T + delta*I)` |
| `edge_margin` | 0.0 | one-sided hinge, divergence-limited axes only |
| `contact_height` | 0.0 | DIAGNOSTIC centroid-plane pull |

**`w_ik = 0.70` dominates.** Measured per-term vertical gradient at a solution:
`d(cost)/dz = -579` for ik, -0.4 for align, +9e-7 for edge. The IK term is what drives
contacts toward the top of an object, not any edge-seeking term.

### The FRoGGeR min-weight metric (`alpha`, `beta`)

```
max_{alpha,beta} beta   s.t.  W*alpha = 0,  sum(alpha) = 1,  alpha >= beta*1
```

`W` is the 6 x (2*s) primitive wrench matrix (s=5 polyhedral cone, s=7 with soft-finger
torsion when `mu_t > 0`); each column is the object-frame wrench `[tau; f]` from a unit
normal force along one cone generator. `alpha` is a convex combination that cancels to
zero net wrench; `beta` is the smallest weight in it. `beta > 0` iff the origin is
interior to the wrench hull, i.e. force closure, and its magnitude says how robustly.

Two deliberate choices:
- **`alpha` is NOT constrained non-negative.** Out of closure, `beta < 0` with some
  `alpha_j < 0` is the correct value and stays smoothly climbable instead of infeasible.
- **Embedded single-level**, not a nested LP differentiated through — IPOPT's KKT system
  couples it to q/p1/p2 and gives exact gradients.

### Constraints

1. Joint limits — `bounded(lo, q, hi)`
2. Surface — primitives only (mesh is on-surface by construction). Box gets a hard
   `edge_margin_m` keep-out; **mesh gets no hard edge keep-out at all**
3. Patch trust region (§2)
4. Wrench LP: `W*alpha = 0`, `sum(alpha) = 1`, `alpha >= beta`; `y >= 0`,
   `sum(y_i) <= gamma`; `(w + s)/ref6 == w_k/ref6` per corner
5. Collision — softplus sphere-vs-{box, cylinder, sphere} `>= ` per-geom clearance;
   sphere-plane ground for every arm geom and both fingertips

### Picard relinearization

```python
_n_relin = 0 if geom_type == 6 else cfg.n_normal_relinearize
```

Contact normals and the wrench frame are **frozen within a stage** (that is what keeps the
NLP smooth) and refreshed between stages. Boxes skip relinearization entirely.

---

## 5. Post-processing: the executed gamma

The NLP's internal `_gamma` is discarded. Execution re-solves from the *achieved* geometry:

```python
n1_in, n2_in = recommended_inward_normals(model, data, obj_gid, mesh_entry, p1, p2)
rec_local    = [local_contact_frame(p, n, p_WoO, R_WO) for f in FINGER_SET]
gamma_live   = solve_gamma_live(p_O, R_O, mu, mass, ACCEL, ANG, inertia, grav_O=g_O)
```

`solve_gamma_live` wraps `scripts/3D_minimum_NCF.min_gamma_for_accel_lp`. **gamma is a
SCALE ON THE FRICTION CONE**, not a function of acceleration: cone vertices scale linearly
(`V(gamma) = gamma*V(1)`), the accel/torque budget becomes a force box (`m*a`, `I*alpha`)
that is sign-expanded into 64 corners, and one LP per corner returns
`gamma = max_corner`. At zero disturbance it collapses to `weight/(2*mu)` — verified
5.96 N predicted vs 5.96 N returned for the 0.729 kg block at mu=0.6.

Infeasible on ANY corner returns `None`, and the caller substitutes `GAMMA_FALLBACK = 2.0`.
Then `_gamma_stability_ceiling` clamps for simulator stability (~12 N on the tabletop).

### KNOWN INCONSISTENCY — verify and execute ask different questions

| | planner verify (`grasp_planner_3d.py:4064`) | executor (`pick_*.py`) |
|---|---|---|
| friction | `0.8 * mu` (safety derate) | raw `mu` |
| linear accel | `cfg.accel_budget_xyz` = (0.25, 0.25, 0.25) | hardcoded (0.5, 0.5, 0.5) |
| angular accel | `cfg.ang_accel_budget_xyz` = (0.5, 0.5, 0.5) | hardcoded (0.1, 0.1, 0.1) |

So `wrench_feasible=True` with a given `gamma_min` does NOT bound the executed gamma, and
the executed gamma can come out below `gamma_min`. Teleop is a third setting again: it
sizes for **20 m/s^2** and enforces that with its slew limiter (§7).

Measured consequence: 7 of 12 planned cells in the tabletop matrix fell back to
`gamma = 2.0` because `solve_gamma_live` reported infeasible — while the planner had
stamped every one of them `wrench_feasible=True`.

---

## 6. Execution sequence (shared by both benchmarks)

```
approach (kinematic replay) -> hold/settle -> GAP GATE -> squeeze ramp
   -> set_transporting(True) -> lift jog -> [transport jog -> release]
```

- **Gap gate**: `CONTACT_GAP_TOL_M = 8 mm`, hard abort. The IK target is deliberately the
  MAX tip-mesh offset, so a "safe" solution leaves a real 4-5 mm gap that the squeeze is
  expected to close. Verified it does: 7.16 mm -> -0.6 mm by step 200, force appears,
  lift succeeds. Gaps over the gate are a PLANNER outcome, not a contact-tuning one.
- **Squeeze ramp**: `internal_force_torques(scale)` ramped over `SQUEEZE_RAMP_S`.
  Unramped full force launches the object (measured 35 N -> box thrown 400 mm).
- **`set_transporting(True)`**: switches finger PD from CLOSING to HOLDING gains. These
  are opposite requirements — soft closing gains let the internal-force term win
  (kp 0.8->20 dropped grip 6.03->1.28 N), but a soft holding gain is back-driven by the
  load and bleeds normal force. See `grasp_controller.effective_gains`.
- **Jogs**: resolved-rate DLS, `JOG_SING_EPS = 0.02`, `JOG_LAM_MAX = 0.05`.

---

## 7. How each environment differs

### `benchmarks/ycb_grasp/pick_from_floor.py` — FLOOR

| | |
|---|---|
| scene | `ycb_grasp/scene.py`, objects over a bare floor, base unrotated |
| start pose | **randomized `q0`** — exercises planning+execution from an arbitrary posture |
| ground plane | `z = 0` |
| object friction | mu = 0.6 (per `pick_from_floor`'s own gamma comment; not re-queried live) |
| gamma budget | `ACCEL_BUDGET_XYZ = (0.5,)*3`, `ANG_ACCEL_BUDGET = (0.1,)*3` |
| phases | approach -> hold -> squeeze -> lift. **No transport, no bin.** |
| artifacts | `out/floor/<tag>/<object>/` |

### `benchmarks/ycb_grasp/pick_and_place.py` — TABLETOP

| | |
|---|---|
| scene | `ycb_grasp/table_scene.py`, table + bin, base raised to `TABLE_TOP_Z` and **yawed +90 deg** |
| start pose | **always HOME** (`table_scene.HOME_ARM`) — the point is to characterize the GRASP, not IK from a random posture. `ik_demo.home_bias()` is WRONG here; it aims along +x, away from the objects |
| ground plane | `ground_z = TABLE_TOP_Z = 0.625` — fingertip clearance is against the TABLE, not z=0 |
| object friction | **mu = 2.0** (from `scene_objects.attach_ycb_object`) — much higher than the floor scene, so gamma floors are ~3x lower |
| scene impratio | 100 (XML) vs the floor benchmark's measured-best 20 — unresolved, overridable via `--impratio` |
| phases | approach -> hold -> squeeze -> lift -> transport -> release, scored by `TS.in_bin` |
| artifacts | `out/tabletop/<tag>/<object>/` |

Two scoring caveats specific to this env:
- `TS.in_bin` tests the object **body origin** against `z >= 0.635` (the bin floor), so an
  object resting IN the bin only passes if it is tall enough to lift its origin clear.
  Under-reports successes.
- `in_bin` alone scores a SCOOP as success — `065-j_cups` reached the bin twice with
  `squeeze_forces_N = 0.0/0.0` and contact lost on both fingers, riding inside the hand.
  Require real contact force before counting a grasp.

### `kinova_leap_pick_place.py` — TELEOP

Owns its own modes (`contact_aware_teleop`, `dexpilot`, `anyteleop`). The benchmarks do
not reimplement or modify them; the eventual integration is that teleop's
**contact-planning step** calls the same planner, with dexpilot/anyteleop untouched.

Differences that matter to the solver:

| | |
|---|---|
| gamma budget | `NCF_ACCEL_BUDGET_XYZ = (20, 20, 20) m/s^2` — **40x the benchmarks'** |
| why that is safe | the jog is slew-rate limited to exactly that budget, so it is *"ENFORCED by construction and gamma covers the true worst case at 1.0x margin"* |
| slew limiter | `jog_v += clip(v_lin - jog_v, -dv_max, dv_max)`, `dv_max = ACCEL*dt`; peak speed capped at `JOG_VEL = 0.3 m/s`; arm rate hard-clamped at `JOG_QDOT_MAX = 2.0 rad/s` |
| squeeze ramp | `SQUEEZE_RAMP_S = 0.1` (benchmarks use 0.5) |
| input smoothing | One-Euro filters on tracking input; EMA on IK output (`alpha = 0.3`, or 0.9 under `--physics`) |

`pick_and_place.py` has adopted teleop's acceleration limiter (`JOG_ACCEL_BUDGET_MPS2 = 20.0`,
`JOG_VEL_MAX_MPS = 0.3`) but still solves gamma for a (0.5, 0.5, 0.5) box — see §5.

---

## 8. Contact / solver settings

Stock model values, and the measured alternative (see
`benchmarks/ycb_grasp/contact_tuning.py` for the full derivation):

| | stock | tuned |
|---|---|---|
| fingertip `solref` | `[0.004, 1.0]` | `[0.02, 2.0]` |
| object `solref` | `[0.02, 1.0]` | unchanged |
| `noslip_iterations` | **0** | 5 |
| `impratio` | 100 | 100 |
| `timestep` | 0.002 | 0.002 |
| `gamma` | solved (often the 2.0 fallback) | 10.0 |
| `squeeze_pd_scale` | 0.25 | 1.0 |

**MuJoCo combines an unpaired contact's `solref` by taking the MIN of the two geoms** (and
friction by the MAX). The fingertip is stiffer than every object, so the effective contact
ran at tau = 0.004 s = **2 timesteps** — stiff enough that the integrator re-energizes it
each step. That is the release-"fling" mechanism. Because of the min() rule, softening the
OBJECT does nothing, and softening EVERY geom silently softens the fingertip too (an
earlier mistake that made it look as though the damping ratio destroyed the grasp — it did
not; grasp was 3/3 across every tau/zeta cell once scoped to fingertips).

`noslip_iterations` is the strongest lever on release quality but removes the tangential
compliance a heavy object leans on; it must be paired with a higher gamma. Neither change
works alone.

---

## 9. Open issues

- **Degenerate contact normals.** Seeds are emitted with `n1.n2 = +1.000` at 7.5 mm span
  (both contacts on the same spot, same face, same direction) and still pass the planner's
  verifier as `wrench_feasible=True`. This is why most cells fall back to `gamma = 2.0`.
- **`017_orange` release fling is not a contact-model artifact.** It CONVERGES under
  timestep refinement (1.68 / 1.60 / 1.58 m/s at dt = 2/1/0.5 ms), so it is what the model
  says happens when a rigid sphere pinched between two pads is released. Needs a grasp or
  release-strategy change, not a parameter.
- **`009_gelatin_box` lying flat is seed-starved, but no longer impossible.** The seed
  floor used to be the fingertip's isotropic bounding-sphere radius (19.4 mm), forcing
  contacts >= 21.4 mm above the table on a ~30 mm box — 120/120 seeds rejected. That
  radius is set by the tip's LONG axis, pointing away from the contact; the pad's honest
  extent along the contact direction is 10.8 mm (`_tip_support_along`), and the NLP's own
  `ground_clearance_m` constraint governs the FINAL pose regardless. With
  `seed_ground_clearance_m = 0.005` the box now yields seeds (measured 1/3 accepted, and
  1/8 over a larger draw). Still marginal: its volumetric centroid sits at 13.7 mm, so
  antipodal rays through the centroid put one contact near the table almost regardless of
  the floor. A side-approach seeding mode is the real fix.
- **`in_bin` origin-height bug** (§7).
- **Verify/execute gamma inconsistency** (§5).

## 10. Measurement hygiene

- Re-running the SAME config is deterministic to the last digit (verified 3x).
- But changing `impratio` perturbs settling -> changes the plan -> can push a fingertip gap
  just over the 8 mm gate, producing a 0.0 N "failure" that is a PLANNER outcome. Check
  `phase_log` for `squeeze_aborted_no_contact` before attributing a zero-force cell to
  contact settings.
- `PFF_SQUEEZE_TRACE=1` samples gap+force through the squeeze ramp;
  `PFF_CONTACT_TRACE=1` dumps contact geometry (`n1.n2`, span);
  `PFF_GRAD_Z=1` dumps per-term vertical cost gradients.

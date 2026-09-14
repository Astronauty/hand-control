# Grasp solver — current state, and how each environment uses it

**Living document.** Unlike the `*_FINDINGS.md` files beside it (which record closed
investigations and should not be rewritten), this describes what the solver *currently
does* and is meant to be edited whenever that changes. If you change a default, a cost
term, a constraint, or the way an environment calls the planner, update this file in the
same commit.

Last verified against: `simulation/grasp_planner_3d.py`, `simulation/grasp_config_builder.py`,
`kinova_common/{wrench,constants}.py`, `grasp_control/{grasp_controller,force_control}.py`,
`benchmarks/ycb_grasp/{pick_from_floor,pick_and_place}.py`, `kinova_leap_pick_place.py`
— 2026-09-14.

**Two JSON config files feed `GraspConfig3D`**, both read by `grasp_config_builder` and
both applied as DEFAULTS, so an explicit CLI flag or `PFF_*` env var still wins
(file -> per-object -> CLI/env):

- **`models/grasp_seed_config.json`** (`load_seed_config()`) — seed/surrogate settings.
  Current values: `seed_kappa_max_reject` 150, `quadratic_sdf_err_tol` 4 mm,
  `quadratic_t_bound_max` 100 mm, `seed_ground_clearance_m` 5 mm,
  `seed_prefer_outer_surface` true.
- **`models/grasp_finger_config.json`** (`load_finger_config()`) — WHICH FINGERS a grasp
  uses, by role. Named pairings resolve to the slot site/geom names `GraspConfig3D`
  already consumes: `thumb_index` (default), `thumb_middle`, `tripod`. **`n_contacts` is
  DERIVED from `len(roles)`, never passed independently** — that is the failure mode this
  replaces, where the contact COUNT and the finger IDENTITIES were two facts kept in sync
  by hand. Role names resolve through `constants.FINGER_TIP_SITES` / `FINGER_CODE`, so the
  table never repeats a site name and cannot drift from the model.

**Standardized GWS preset** (`for_gws_recommender`): `max_iter=80`, `n_seeds=3`,
`seed_dls_rank_pool=3`, `n_normal_relinearize=0` (single stage), `w_gws=5.0`, `w_span=1.0`.
The 80/3 budget is measured, and is NOT a speed-for-quality trade — see §11.

---

## 0. Pipeline at a glance

```
seed generation  ->  local surface fit  ->  NLP (per Picard stage)  ->  post-process gamma  ->  execute
   _seed_pair         quadratic patch       q, t1, t2, gamma, y        solve_gamma_live       squeeze + jog
   [+ _seed_third_contact fan]              [+ t3 on slot 2's patch]
```

The NLP's internal `gamma` is **not** what gets executed. See §5.

The bracketed steps run only at `n_contacts >= 3` (the `tripod` pairing). That path is
**planner-side only today** — the NLP solves for a third contact, but nothing downstream
consumes it. See §10.

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

### Reading the seed figure (`seed<N>_contact_seeds.pdf`)

Changed in `d9d2a16`, and the old behaviour invited a specific misreading:

- **Only the WINNING accepted panel is titled `accepted -- SELECTED`** (bold, blue).
  `solve()` runs a full NLP per accepted seed and keeps the cost-ranked best, so without
  this every accepted panel looked equally chosen.
- **Solution triangles are drawn ONLY on that winning panel.** `res['p1']/['p2']` are the
  winner's contacts; drawing them on a losing accepted panel paired one seed's patch with a
  different seed's solution, which reads as "the solution is off its patch" and is an
  artifact. A pre-`d9d2a16` figure showing triangles far from the circles on several panels
  is showing that artifact, not seed-to-solution drift.
- **The drawn rectangle is the frame the solve actually used** (`res['quad_frames']`), not a
  rebuild at that panel's own seed. The rebuild measured 24 mm off in y on `036_wood_block`
  seed 1, which made a solution strictly inside its bounds (t1 = +86.87 against +86.88)
  appear to sit outside its own trust region. The rebuild remains the fallback for records
  with no stage frame (rejected seeds never reached a stage).

**Consequence worth knowing:** because the deterministic seed goes in first and usually
wins the ranking, varying the planner's `seed` produces NO variation on rigid symmetric
objects (measured: wood block identical across seeds 0/1/2), and only mild variation on
spheres/cups. `seed` is plumbed through `MultiStartGraspPlanner3D(..., seed=)`; the
per-seed `q_ref` jitter (`qref_restart_sigma_arm/hand`) is 0.0 by default, so the RNG's
only live effect is `_seed_pair`'s march directions.

### The third contact (`n_contacts >= 3`)

`_seed_third_contact` is deliberately SEPARATE from `_seed_pair`: that function's antipodal
march is what every measured 2-contact result depends on, and it structurally cannot produce
an off-axis third point (it marches ALONG contact 1's inward normal).

**Why off-axis.** A 2-contact pinch has zero moment arm about the line through its contacts,
so its wrench matrix is rank-5-of-6 — the reason `project_grasp_axis_torque` exists, and the
reason the soft-finger columns were tried and measured harmful (they restore rank only at
singular value 0.1 against 4.0). A third contact perpendicular to the grasp axis supplies a
real moment arm. Measured on a sphere tripod vs the same-radius pinch, the internal force
needed drops by half or better (mu=1.6: gamma 1.03 -> 0.49).

Strategy is **fan, then let the solver move it**: offset from the grasp midpoint
perpendicular to the grasp axis, biased toward where the middle finger actually sits, fan
`+/- 40 deg` (5 candidates) around that bias, project each onto the surface. The bias is a
palm-FRAME prior measured from the model's rest pose (the middle fingertip lies at
`[0.692, -0.722, 0.002]` from the pinch midpoint, ~45 deg in the palm's xy-plane), mapped
through the LIVE palm rotation so it follows the hand rather than being a world constant.

Candidates go through the SAME gates the pair already passes (`_reachable_contact`,
`_seed_kappa_ok`), then are ranked by the same DLS-IK reachability screen the chart-pair path
uses — the only seed screen that knows about the arm — solving all three tips at once and
scoring the middle finger's residual. Rankings land in `last_c3_rank_table`.

**A seed that yields no viable third contact stays a 2-contact pinch rather than failing:**
the tripod is an upgrade, not a precondition.

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

**`quadratic_extent_clip` (default True, added `d9d2a16`) — a bound the SDF search
structurally cannot provide.** On a FLAT face a plane tracks the surface perfectly, so
nothing ever diverges: the search runs the full range and returns `t_bound_max`, which
reports the CAP, not the face. Measured on `036_wood_block`, the raw long-axis bounds came
back at exactly 100.0/100.0/99.6/99.2 mm against a 100 mm cap — saturated, not measured.
The NLP then legitimately walked contacts ~87-89 mm up the face to the block's top edge
(inside their bounds, `quad_pinned=True`), where the fingers cannot oppose each other.
The observed failure was the thumb contacting ALONE at 33 N and shoving the block off the
table; the same cell in a later sweep applied 68 N and launched it 1356 mm through the
floor, carrying the only negative `gws_beta` measured (-0.1937).

The clip additionally bounds each axis by how far the object's own face reaches along it,
using only vertices lying IN the patch plane (`|(v-seed).n| <= 2mm`) — projecting the whole
hull measures the object's bounding extent (+/-103 mm on that block), which is looser than
the 89 mm already present and clips nothing. It runs BETWEEN the corner shrink and
`bound_inset`, and only ever moves a side INWARD, so the inset takes its crease reserve out
of the already-clipped room rather than double-subtracting the same margin.

Measured effect on `036_wood_block` seed 2 (2-finger): `gws_beta` -0.1937 -> +0.0615,
lift dz -1356 mm -> ~0 mm. The grasp still fails, but as an ordinary failed grasp rather
than a physics blow-up.

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

### The third contact SHARES slot 2's patch

At `n_contacts >= 3` the third contact does **not** get a patch of its own. It gets its own
2-DOF coordinate `_t3_var` inside contact 2's paraboloid and trust region, reconstructed via
the same identity (`seed_l + t0*axis0_l + t1*axis1_l + h(t)*n_l`). It is initialized at half
the upper bound on each axis so it does not start coincident with contact 2 — identical
contacts give a degenerate wrench matrix.

An earlier version fitted an INDEPENDENT third patch from the fan seed and **failed badly**:
measured 183 mm between the middle fingertip and its assigned contact on `036_wood_block`,
because an independent patch can land on a face the hand would have to re-approach entirely.
Sharing guarantees the two contacts are adjacent and mutually reachable; a standalone test
confirmed the optimizer then slides BOTH to reachable spots inside the bounds (index/middle
10-24 mm apart, every finger converging to its pad radius).

**That 10-24 mm separation does NOT hold in the benchmark.** Measured 2026-09-14 over
5 objects x 3 seeds through full execution, contact 3 converges ONTO contact 2: the n=3
contacts figure's pairwise separations read `1-2 69mm  1-3 69mm  2-3 0mm` on `017_orange`
seed 0 — exactly coincident, with the index and middle panels rendering identically (same
curvature, same bounds, same SDF error). The half-bound initialization above prevents a
coincident START, not a coincident SOLUTION. Consequence: no off-axis moment arm, which is
the only reason a third contact exists. See §10 for the execution numbers.

The rest-pose fingertip separation (178-219 mm) that originally motivated separate patches
was the wrong measurement: what matters is whether both fingers can curl onto NEARBY
contacts, not how far apart they hang when extended.

---

## 3. NLP decision variables

| variable | size | notes |
|---|---|---|
| `_q` | n_act | arm + hand joints |
| `_t1_var`, `_t2_var` | 2 each | patch coords (mesh); `_p1`/`_p2` are free 3-vectors for primitives |
| `_t3_var` | 2 | third contact (`n_contacts >= 3`), coords on **slot 2's** patch (§2) |
| `_gamma` | 1 | wrench-cone scale, bounded `[0, gamma_max=25]` |
| `_y1_k`, `_y2_k` | nverts per corner | cone-vertex coefficients |
| `_s_k` | 6 per corner | wrench slack (when enabled) |
| `alpha`, `beta` | n_cols, 1 | FRoGGeR min-weight LP (when `w_gws > 0`) |

---

## 4. Cost terms and constraints

Costs are normalized so each is ~1 at its reference level.

| term | default weight | expression |
|---|---|---|
| `ik` | **0.70** | `0.5*(d1^2+d2^2)/d_ref^2`, `d_ref = 5 mm`; at `n_contacts >= 3`, `(d1^2+d2^2+d3^2)/(3*d_ref^2)` |
| `reg` | 0.03 | `\|\|(q - q_reg)/q_scale\|\|^2 / n_dof` |
| `gamma` | 0.15 | `gamma_lp / g_ref` (normalized by task load) |
| `y` | 0.6 | `sum \|\|y\|\|^2`, min-norm force distribution |
| `slack` | 1.0 | wrench-infeasibility penalty |
| `align` | 0.0 | `\|\|g_hat - n1_in\|\|^2`, grasp-axis opposition. **`n_contacts == 2` only** |
| `orient` | 0.0 | `\|\|R_tip*pad_axis - n_in\|\|^2` per contact |
| `gws` | 0.0 | `-beta` (see below) |
| `span` | 0.0 | `-logdet(W W^T + delta*I)` |
| `edge_margin` | 0.0 | one-sided hinge, divergence-limited axes only |
| `contact_height` | 0.0 | DIAGNOSTIC centroid-plane pull |

**`w_ik = 0.70` dominates.** Measured per-term vertical gradient at a solution:
`d(cost)/dz = -579` for ik, -0.4 for align, +9e-7 for edge. The IK term is what drives
contacts toward the top of an object, not any edge-seeking term.

The IK term **averages over the contacts present** so `w_ik` keeps its calibrated meaning:
at n=2 it is exactly the historical `0.5*(d1+d2)`, and a third contact does not inflate the
IK term relative to reg/align/gws (which would silently re-tune every other weight).

**`w_align` is switched OFF at `n_contacts != 2`**, not generalized. The term is
`(p2-p1)`-relational — it asks that THE grasp axis align with contact 1's inward normal —
and a tripod has no single grasp axis. For three non-collinear contacts, opposition is not
the right objective anyway: force closure there means the normals SPAN the origin, which is
exactly what the FRoGGeR min-weight `beta` measures and `w_gws` already optimizes. If `beta`
turns out not to carry it, an n>2 strategy goes here then. Keeping the gate explicit
preserves the measured n=2 path bit-identically (12/15 lifts, 3 seeds x 5 objects at 80/3).

### The FRoGGeR min-weight metric (`alpha`, `beta`)

```
max_{alpha,beta} beta   s.t.  W*alpha = 0,  sum(alpha) = 1,  alpha >= beta*1
```

`W` is the 6 x (n*s) primitive wrench matrix (s=5 polyhedral cone, s=7 with soft-finger
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

Note the standardized GWS preset sets `n_normal_relinearize=0`, i.e. a SINGLE stage — the
paraboloid supplies the normal symbolically instead (`quadratic_symbolic_normals`), so there
is nothing to re-freeze. The third contact tracks the same refresh path when
relinearization IS enabled, so turning it back on cannot silently pin contact 3 at its seed
while 1 and 2 move.

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

Infeasible on ANY corner returns `None`. What the caller then does **now differs by
environment** — see the table below. `_gamma_stability_ceiling` clamps the result for
simulator stability (~12 N on the tabletop).

`solve_gamma_live` generalizes to `n >= 2` contacts. The moment reference is the contact
CENTROID, which reduces to the midpoint at n=2 so the measured 2-contact path is unchanged.
The grasp-axis moment/torque **projections are applied only at n=2**: a two-contact pinch
cannot resist ANY torque about the line through its contacts, so projecting that component
out is honest. A third contact off the grasp axis is exactly what removes that premise, so
zeroing it at n>=3 would make the certificate CONSERVATIVE against a capability the tripod
actually has — wrong rather than merely unnecessary.

### Verify vs execute — mostly reconciled, one gap left

The tabletop executor now solves gamma against the **same disturbance box** the certificate
uses (`NCF_ACCEL_BUDGET_XYZ = (20,20,20) m/s^2`, `NCF_ANG_ACCEL_BUDGET = (1,1,1) rad/s^2`,
shared with the teleop stack), rather than a second hardcoded copy. Before that, planning
and execution sized gamma for DIFFERENT tasks — measured on `014_lemon`: certificate
`gamma_min = 1.07` at 20 m/s^2 vs commanded `gamma = 0.13` at 0.5 m/s^2, an **8x
disagreement on the same contacts via the same LP**. Because the budget changed by ~80x,
`gamma_min` values are NOT comparable to runs from before this change.

| | planner `verify()` | tabletop `pick_and_place.py` | floor `pick_from_floor.py` |
|---|---|---|---|
| friction | `0.8 * mu` (safety derate) | raw `mu` | raw `mu` |
| linear accel | `cfg.accel_budget_xyz` | `(20, 20, 20)` | `(0.5, 0.5, 0.5)` |
| angular accel | `cfg.ang_accel_budget_xyz` | `(1, 1, 1)` | `(0.1, 0.1, 0.1)` |
| LP infeasible | flags `wrench_feasible=False` | **aborts the grasp** | `GAMMA_FALLBACK = 2.0` |

**The remaining verify-side gap is the `0.8 * mu` derate.** (`verify()`'s hardcoded `n=2`
was the other half of this and is FIXED as of 2026-09-13 — it now certifies every contact
present, with the grasp-axis projections gated to `n == 2`; see §10. Note that this makes a
3-contact `gamma_min` incomparable to a 2-contact one — check `n_contacts_verified`.)

**The tabletop fallback was REMOVED, deliberately.** An infeasible LP means the contact
geometry cannot resist the disturbance box at ANY squeeze force; substituting a constant
converted "no feasible grasp" into "squeeze anyway at a made-up force". That is how an
80-degree-splay grasp on `014_lemon` (`n1.n2 = +0.166`) still reached the squeeze phase.
The floor benchmark still falls back to 2.0, and teleop falls back to `GAMMA_FALLBACK = 250.0`.

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

### The squeeze force: cone-constrained, not sign-anchored

`internal_force_torques` allocates `f_c = pinv(G) @ w_des + null(G) @ gamma`. **How `gamma`
is chosen changed in `daf4207`, and the old rule was only ever correct at n=2.**

`allocate()` (the fallback) orients each null-space basis vector using the FIRST non-None
`inward_dirs` entry and then breaks. With 2 antipodal contacts `null(G)` is 1-D, so that one
sign IS the answer. With 3 contacts the null space is 3-D and a uniform gamma over an
arbitrary SVD basis lands anywhere in it — **nothing holds the non-anchor contacts
compressive.** Measured on an asymmetric tripod: sign-anchor normal forces
`[1.213, 0.047, 0.325] N` vs cone-solve `[1.000, 1.000, 1.000] N`.

`solve_gamma_cone()` (default, `cone_gamma=True`) replaces it with an LP in `(gamma, t)`:

```
min  t   s.t.  f_k . n_k >= f_min      compressive, EVERY contact
               f_k . n_k <= t          t = peak normal force
               f_k . e_ki <= mu_eff * f_k . n_k    8-facet pyramid, per contact
```

The pyramid is **inscribed** (`mu_eff = mu * cos(pi/8) * (1 - margin)`), so feasibility
implies feasibility for the true circular cone, not merely for the approximation. The LP
returns the MINIMUM in-cone force (peak normal = `cone_f_min`); the caller's commanded
`gamma` then scales it up, so `cone_f_min` sets the shape and `gamma` sets the magnitude.

**Infeasible returns `None` rather than a fabricated gamma** — three near-parallel normals
genuinely cannot squeeze. The controller then falls back to the sign-anchor path, which is
sound at n=2 and is the only path when `cone_gamma=False`.

Regression at n=2: `017_orange` thumb+index unchanged through the switch (lift 119.11 mm,
contacts held), with squeeze forces RISING 1.20/1.19 -> 1.65/1.61 N as the cone solve
distributes load more evenly.

### `active_joint_slices` must match the run's fingers

`effective_gains()` and `slip_correction_torques()` both gate on these slices, and the
`GraspController` default is hardcoded `((7, 11), (19, 23))` — LEAP index and thumb. At
three fingers the middle finger's joints (11..14) fall outside both, so **it never
participates in the CLOSING/HOLDING switch at all**: it keeps full stiff gains, pinned at its
planned pre-contact posture, while index and thumb soften to 0.25 to let the squeeze close.

`constants.finger_joint_slices(model, fingers)` derives the slices from the model's own
`leap_<code>_*` joint names. thumb+index reproduces `((7, 11), (19, 23))` exactly.
**Only `pick_and_place.py` calls it** — see §9.

---

## 7. How each environment differs

### `benchmarks/ycb_grasp/pick_from_floor.py` — FLOOR

| | |
|---|---|
| scene | `ycb_grasp/scene.py`, objects over a bare floor, base unrotated |
| start pose | **randomized `q0`** — exercises planning+execution from an arbitrary posture |
| ground plane | `z = 0` |
| object friction | mu = 0.6 (per `pick_from_floor`'s own gamma comment; not re-queried live) |
| gamma budget | `ACCEL_BUDGET_XYZ = (0.5,)*3`, `ANG_ACCEL_BUDGET = (0.1,)*3` — **not** migrated to the shared 20 m/s^2 box the tabletop and teleop now share (§5) |
| LP infeasible | falls back to `GAMMA_FALLBACK = 2.0` (the tabletop aborts instead) |
| controller | `GraspController` **defaults**: `cone_mu = 0.7` (vs the scene's own 0.6) and the hardcoded `((7,11),(19,23))` gain slices — neither is passed (§9) |
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
| gamma budget | `(20,20,20) m/s^2` / `(1,1,1) rad/s^2`, shared with teleop and with the certificate (§5) |
| LP infeasible | **aborts the grasp** — no fallback constant |
| finger pairing | `--pairing {thumb_index,thumb_middle,tripod}`, default from `models/grasp_finger_config.json`. **Steers the PLANNER only** (§10) |
| controller | the only env that configures it: `cone_mu` from the LIVE geom friction, `cone_margin = 0.2`, `cone_f_min = 0.5`, and `finger_joint_slices(model, _FSET)` (§6) |
| phases | approach -> hold -> squeeze -> lift -> transport -> release, scored by `TS.in_bin` |
| artifacts | `out/tabletop/<tag>/<object>/`, incl. `seed<N>_planned.png` (planned pose before squeeze) |

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
`JOG_VEL_MAX_MPS = 0.3`) **and now also solves gamma against the same (20, 20, 20) box** —
so the certificate is sized for the motion the script really executes (§5). The floor
benchmark has not been migrated.

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
| `gamma` | solved per object (tabletop aborts if infeasible; floor still falls back to 2.0) | 10.0 |
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
  verifier as `wrench_feasible=True`. On the tabletop this no longer silently degrades into
  a made-up squeeze force — the run aborts instead (§5) — but the planner still certifies
  these grasps, so the root cause is unfixed.
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
- **Verify/execute gamma: the budget half is fixed, the friction half is not** (§5). The
  tabletop executor and the certificate now share one disturbance box, but `verify()` still
  derates friction to `0.8 * mu` and the floor benchmark still uses its own small budget.
- **The tripod is not wired past the NLP** (§10) — the planner solves a third contact that
  no consumer reads.
- **Only the tabletop configures the controller.** `pick_from_floor.py` and
  `kinova_leap_pick_place.py` construct `GraspController` without `cone_mu`/`cone_margin`/
  `cone_f_min` or `active_joint_slices`, so both silently take the defaults: `cone_mu = 0.7`
  and the hardcoded index+thumb gain slices (§6). Two consequences, unequal in severity:
  - The friction default is **benign today**: `mu_eff = 0.7*cos(pi/8)*0.8 = 0.517` against
    the floor scene's true 0.6, i.e. conservative. It is still a number that does not track
    its scene, and would become optimistic for any object with mu < 0.52.
  - The **gain-slice default is a real defect** for any run with more than two fingers, and
    it is the exact bug `daf4207` fixed for the tabletop without propagating. Neither env
    runs 3 fingers today, so it is latent rather than active.
- **`thumb_middle` is the weaker pinch.** All ten measured cells certify wrench-feasible,
  but `gamma_min` rises on every object vs `thumb_index`, and on `036_wood_block` by 4.9x
  (3.08 -> 14.93 N against a 25 N ceiling). Its "converged" solver status there is not a
  quality signal. `w_align` and the seed priors were tuned for `thumb_index` and need
  retuning before `thumb_middle` is worth preferring.

---

## 10. Tripod wiring status — planner-side only

`n_contacts >= 3` is a **work in progress**. What exists end-to-end inside the NLP:

| stage | third contact? |
|---|---|
| seeding (`_seed_third_contact` fan, gates, DLS rank) | yes |
| patch / trust region (shares slot 2's, §2) | yes |
| IK cost term (averaged over 3, §4) | yes |
| Picard normal refresh between stages | yes |
| solve result (`res['p3']`) | yes |
| seed figure (`▲` on the shared patch) | yes |

What does **not** see it yet:

- ~~**The NLP's own GWS metric.**~~ **DONE** (commit `5fabede`). The `w_gws` call site passes
  `_gws_extra` into `build_W_ca` and warns when `n_contacts>=3` but contact 3 has no frame
  this stage, so `beta` is the tripod's min-weight, not the pinch's. Reported three-finger
  betas from before that commit (-4.56, -4.87) were 2-contact numbers.
- ~~**`verify()`** — hardcoded `n=2`.~~ **DONE** (2026-09-13). Its wrench LP now covers every
  contact present, with the moment reference generalized to the contact CENTROID (which
  reduces to the midpoint at n=2) and the grasp-axis moment/torque projections gated to
  `n == 2` — honest for a pinch, but conservative against the very capability a third
  off-axis contact provides, so they self-disable at n>=3. Same convention as
  `solve_gamma_live` (§5).
  **`verify()` now returns `n_contacts_verified`, and you must read it before comparing
  `gamma_min` across runs:** a 3-contact `gamma_min` is a DIFFERENT quantity from a
  2-contact one, because the pinch's number had a disturbance component projected out.
  Measured on `036_wood_block` seed 0: pinch 28.387 N (projected) vs tripod 38.855 N
  (unprojected). The tripod is not worse — it is certified against more.
  The n=2 path is bit-identical after the change (verified: `036_wood_block` seed 0,
  `gamma_min = 28.387`, `beta` unchanged to all digits).
- ~~**Every executor.**~~ **PARTLY DONE (2026-09-14).** `pick_and_place.py` DOES read `p3`
  now: it binds contact 3, derives its inward normal from `_geom_normal_np`, and zips
  `_SLOTS` against the contact list so slots map to fingers positionally at n=2 or n=3.
  `pick_from_floor.py` and `kinova_leap_pick_place.py` are still 2-contact.
  **But three-finger execution does not WORK yet.** Measured 5 objects x 3 seeds, full
  execution, CLEAN TREE at `7827ec3`: 2-finger 9/14 true grasps (one cell excluded as a
  physics blow-up), 3-finger **1/15** (8/15 abort before squeeze). The n=3 arm is
  BIT-IDENTICAL between the dirty and clean trees (all 15 cells, beta and gamma_min).
  Every 3-finger plan certifies `wrench_feasible=True`, so this is not a metric failure.
  Two causes, both measured:
  (a) contact 3 COLLAPSES onto contact 2 — EXACTLY coincident, not merely close: the n=3
      contacts figure's pairwise separations read `1-2 69mm  1-3 69mm  2-3 0mm` on
      `017_orange` seed 0, supplying no off-axis moment arm;
  (b) every fingertip misses its target by 13-21 mm, giving 19-33 mm squeeze gaps against
      the 8 mm gate.
  This happens under BOTH patch branches: `c3_own_patch=True` still collapses (6.4 mm) and
  still aborts, so the shared trust region is NOT the cause — look at the seeding/objective.
  **The CONTROLLER is ruled out too, and measurably so.** On `017_orange` seed 0, every
  tripod config from effective kp 0.2 to 16 aborts with `squeeze_aborted_no_contact` — the
  fingertips never get close enough to squeeze, so gains are irrelevant to this failure. On
  the same object and seed the 2-contact grasp lifts 119 mm, and raising ITS gains actively
  breaks it (eff kp >= 4.0: contact lost, 0 N, no lift), exactly as `effective_gains`'
  docstring predicts. The soft-PD-plus-internal-force design is deliberate and correct; the
  tripod's problem is upstream in contact PLACEMENT (0/15 tripod solves under 2 mm
  worst-gap, vs 6/15 at n=2). The cone allocator (§6) and the gain-slice fix landed in the
  same commit and neither moved the tripod — they were prerequisites for it working at all,
  not the fix.
  On the CLEAN tree the two arms solve at comparable speed (2-finger 4.87 s, 3-finger
  5.58 s mean); the dirty tree's apparent "n=3 is 2x faster" was a property of that tree,
  not the formulation. Full table:
  `benchmarks/ycb_grasp/out/tabletop/clean_7827ec3/RESULTS_2v3_finger.md`.
- **`FINGER_SET` is import-time**, derived from the pairing file's `default`. So `--pairing`
  steers the PLANNER only; **plan-only sweeps are meaningful, execution runs with a
  non-default pairing are not.** `SLOT_ROLES` was added so the controller maps an NLP slot to
  a finger positionally rather than by role name — the old `{'thumb': p1, 'index': p2}` dicts
  KeyError'd on any pairing without an `index`.

Also measured while building this: `r_middle` must be taken from the middle **site**, not
`geom_rbound`, which is the bounding sphere about the mesh frame origin and over-reports the
LEAP middle pad by 4.2 mm (23.68 vs 19.47 mm). The inflated value fed the third IK target
(`p + r*n`) and pushed it ~4 mm off the surface. Index and middle share mesh dataid 13, so
the corrected `r_middle` equals `r_index` exactly.

---

## 11. The 80/3 preset — why a SMALLER budget helps

`max_iter` 120->80 and `n_seeds` 5->3, measured on the five tabletop objects (seed 0, full
execution). It improves BOTH quality and speed:

```
200/5  ->  3/5 objects squeeze+lift,  solve 5.4s
 80/3  ->  4/5 objects squeeze+lift,  solve ~2.3s
```

`036_wood_block` is the case that makes the point: at 200/5 it aborted with a 10.96 mm index
gap against the 8 mm squeeze gate; at 80/3 it lifts with the tightest contacts in the set
(0.32/0.60 mm) and is the only solve reaching `Solved_To_Acceptable_Level` (76 iters — it
stops BEFORE the cap rather than being truncated by it).

**Why.** The IPOPT log shows iterates cycling rather than settling — objective
60 -> 256 -> 110 -> 79 across consecutive iterations. Constraint violation is fine (7.4e-04);
what never converges is the DUAL residual (476 scaled, against an already-loosened
`acceptable_dual_inf_tol` of 1e3), because the active set is degenerate under the antipodal
minimax-gamma symmetry — the multipliers are genuinely indeterminate, so there is no dual
limit to converge to. Grinding past the point where the primal has settled lands on a worse
iterate about as often as a better one. So `best-effort` is reporting something true, not
masking a broken solve.

That degeneracy was probed directly and is **not tunable**: 7 IPOPT tolerance variants
(0/21 converged — none of those criteria ever fire, so only the final label could differ),
7 conditioning variants (`mu_strategy`, `nlp_scaling_method`, limited-memory history, exact
Hessian — 0/21; `hessian_approximation=exact` fails outright, 0 Lagrangian Hessian
evaluations, the constraints are not twice-differentiable here), and `max_iter=300` still
hits `Maximum_Iterations_Exceeded` every time.

`n_seeds` 5->3 is safe because of `seed_dls_rank_pool=3`: candidates are already ordered by
DLS-IK arm reachability, so seeds 4-5 were the worst of the pool. Dropping them was
bit-identical on `014_lemon` and `056_tennis_ball` at 40% less time.

**Backend: IPOPT kept.** SQP+OSQP was compared under an identical NLP (`use_slsqp` touches
only which plugin `Opti` gets, plus a log label; every cost term, constraint, bound and seed
gate is built before that branch). At 80/3 through full execution: IPOPT 4/5 lifts,
8.9-10.1 s wall; SQP 2/5 lifts, 5.3-5.7 s. SQP is ~1.8x faster and loses `036_wood_block`
(contacts degenerate to wrench-infeasible), `017_orange` and `056_tennis_ball`; it wins
`009_gelatin_box`, which IPOPT has never grasped. `--backend {ipopt,sqp}` keeps this
reproducible.

**Caveat: one seed per object.** Planner-side IK residuals mispredicted execution twice while
establishing this (SQP's apparent lemon advantage did not reproduce at five objects; its
best-in-table orange index residual of 0.90 mm still aborted at 19.2 mm), so these numbers
rest on `phase_log` outcomes, not on predicted gate margins. Multi-seed confirmation is not
done.

---

## 12. Measurement hygiene

- Re-running the SAME config is deterministic to the last digit (verified 3x).
- But changing `impratio` perturbs settling -> changes the plan -> can push a fingertip gap
  just over the 8 mm gate, producing a 0.0 N "failure" that is a PLANNER outcome. Check
  `phase_log` for `squeeze_aborted_no_contact` before attributing a zero-force cell to
  contact settings.
- `PFF_SQUEEZE_TRACE=1` samples gap+force through the squeeze ramp;
  `PFF_CONTACT_TRACE=1` dumps contact geometry (`n1.n2`, span);
  `PFF_GRAD_Z=1` dumps per-term vertical cost gradients.

# Wiring the autonomous IPOPT grasp solver into teleop (mesh objects + analysis plots)

Scope: make `--mode contact_aware_teleop` run the SAME NLP grasp recommender the
tabletop benchmark runs, on YCB MESH objects, and emit the same quadratic-patch /
contact-movement analysis figures.

Last verified against `kinova_leap_pick_place.py`, `simulation/grasp_planner_3d.py`,
`simulation/grasp_config_builder.py`, `benchmarks/ycb_grasp/pick_and_place.py` — 2026-09-10.

---

## 0. What already exists (do NOT rebuild)

The integration is ~80% done. `contact_aware_teleop` already:

- Instantiates `MultiStartGraspPlanner3D` per object (`_get_cat_planner`,
  kinova_leap_pick_place.py:2605) via the SHARED
  `grasp_config_builder.for_teleop_recommender`.
- Fires it on a background thread every `_REC_INTERVAL_S = 2.0` s for the nearest
  object (`_recommender_tick` / `_fire_recommender`), with `_REC_NC = 5` seeds.
- Gates candidates on `verify()`'s `wrench_feasible` (WF gate), applies display
  hysteresis, and shows p1/p2 live on the `rec1`/`rec2` mocap markers.
- Commits on `L` via `_setup_recommended_contact_frames` -> collision-IK -> RRT ->
  GRASP, then re-solves gamma with `solve_gamma_live` and squeezes.

`SOLVER_STATE.md` §7 already names this as the intended architecture: "teleop's
contact-planning step calls the same planner, with dexpilot/anyteleop untouched."

**So this is not an integration task. It is three specific gaps.**

### The commit path is SINGLE-SOLVE — this raises the stakes on mesh collision

`_run_ik_recommended_then_rrt` (kinova_leap_pick_place.py:1814) commits the
RECOMMENDER'S OWN `q` directly as the RRT goal. There is no second IK. The history
comment at :1819-1832 records why: a second `ConstrainedIKSolver` solve optimized a
DIFFERENT objective (posture bias = live `q_seed`) with a DIFFERENT tip offset
(`_PAD_OFFSET` 10 mm directional vs the recommender's `r_tip` 19 mm radial), so the
committed grasp landed visibly away from the displayed markers even on a static
object — the "grasp changes on lock-in" bug. It was dropped once the recommender's
NLP took on the finger-link collision constraints itself.

**Consequence for this work:** the NLP's `obj_clearance_by_geom` tiering is now the
ONLY thing keeping the committed pose out of the object. That was validated 20/20 on
PRIMITIVE boxes. Moving to mesh SDF constraints re-opens that validation — a mesh
penetration is committed straight to the RRT goal with nothing downstream to catch
it. Re-run the penetration check on meshes before trusting lock-in, and keep the
`ConstrainedIKSolver` O/I previews (`_fire_preview_ik`:1881) as the diagnostic.

### Scene: same XML, two independent builders

Teleop loads `models/scene_pick_place.xml` (:404) and `benchmarks/ycb_grasp/table_scene.py:33`
loads the SAME file, with matching `TABLE_TOP_Z = 0.625`, base mount and `N_ROBOT = 23`.
The duplication is deliberate (table_scene.py:21-27: the teleop entry point "is a
working, tuned pipeline and this benchmark should not be able to break it"). So the
two environments genuinely coincide — this plan does not need to reconcile scenes,
only object TYPE (primitive vs mesh). Note `table_scene.py:92-95` flags an unresolved
`impratio` discrepancy (XML sets 100, the benchmark measured 20 best); teleop has no
`--impratio` flag, so a like-for-like A/B needs one added or the XML value used on
both sides.

---

## 1. The three real gaps

### Gap A — mesh objects are hard-blocked

`kinova_leap_pick_place.py:673`:

```python
_siteless = [o['name'] for o in objects if not o['has_grasp_sites']]
if _needs_sites and _siteless:
    sys.exit("... Mesh/YCB objects are supported for plain teleop only ...")
```

and `kinova_leap_pick_place.py:2500`:

```python
_CAT_SUPPORTED = {'obj_red_box', 'obj_green_box', 'obj_box_lowmu', 'obj_box_heavy'}
```

**The `_c1/_c2` site requirement is stale for the recommender path.** Verified:
`_setup_recommended_contact_frames` (line 1502) OVERWRITES `p_S_W` / `inward_S_W`
from `rec['p1']`/`rec['p2']` and never reads `id_S`. Only the LEGACY non-recommender
`_run_ik` (line 1435) reads the sites. So the guard is broader than the code requires.

### Gap B — the config never asks for the quadratic patch

`for_teleop_recommender` does not set `use_quadratic_contact`, whose default is
`False` (grasp_planner_3d.py:1979). The tabletop benchmark sets it True
(pick_and_place.py:233). Without it the mesh contact falls back to the non-quadratic
parameterization, and **no paraboloid frame is recorded** — which is exactly the data
the plots need.

Also mesh-specific: `_recommended_inward_normals` (line 2801) calls
`_geom_normal_np(p, gtype, c, R, size)` WITHOUT `mesh_entry=`. For a mesh geom that
returns garbage. `benchmarks/ycb_grasp/pick_from_floor.py:213`
`recommended_inward_normals(model, data, obj_gid, mesh_entry, p1, p2)` is the correct
mesh-aware version and should be reused rather than re-derived.

### Gap C — no `log_dir`, therefore no plots

`MultiStartGraspPlanner3D(model, mj.MjData(model), cfg)` at line 2605 passes no
`log_dir`. The per-iteration `grasp3d_iter_<ts>.npz` files (`_save_iter_npz`,
grasp_planner_3d.py:3845) are written ONLY when `log_dir` is truthy, and those npz
files are the sole input to every analysis plot.

**This is the whole reason teleop has no figures.** The plotting stack is already
env-agnostic — it reads npz + a mesh, not a benchmark scene.

**Critical: the returned `res` dict carries NO patch data.** The paraboloid frame
(`seed_l, axis0_l, axis1_l, n_l, kappa0, kappa1, grad_norm`, and both the symmetric
`t_bound_*` and asymmetric `t_lo_*`/`t_hi_*` trust bounds) is flattened into the npz
under `quad1_*` / `quad2_*` prefixes (grasp_planner_3d.py:3891-3896) and nowhere else.
`log_dir` is therefore the SOLE channel for the quadratic-surface data — not an
optional debug extra.

**And it is not free.** grasp_planner_3d.py:3642-3658 gates the `gradN_*` series
behind `if self.log_dir:` with an explicit comment that they are "full-vector
reverse-mode AD evaluations (~110+ DOF) ... only computed when actually being logged
(diagnostic/visualization runs), NOT on every production solve (e.g.
test_grasp_recommender.py's live loop)". Teleop's recommender IS such a live loop, at
a 2 s cadence. This is the strongest argument for the gated `--rec-log-dir` design in
Phase 3 rather than logging unconditionally.

---

## 2. The plotting seam (already reusable)

`benchmarks/ycb_grasp/pick_and_place.py:538` `_write_plots(...)` takes only generic
MuJoCo handles:

```python
_write_plots(model, data, res, verify_info, log_dir, object_id, seed,
             body_name, obj_bid, pos, out_dir, quadratic_path=False, n_relin=None)
```

and internally does:

```python
stages = QP._iter_trace_quadratic_stages(log_dir, res=res)   # splits multi-seed attempts,
                                                             # matches the WINNING attempt
                                                             # by res['p1_seed']/['p2_seed']
V, F   = oua.body_visual_mesh(model, obj_bid)
PGC.plot_grasp_contacts(V, F, last, ...)      # per-contact zoomed patch view
QP.plot_quadratic_path(V, F, stages, ...)     # Picard trajectory across stages
```

Nothing in that path is tabletop-specific. Passing `res` matters: with `n_seeds=5`
several unrelated attempts log into one dir, and without `res` the reader falls back
to the LAST attempt, which is usually not the returned one.

**Plan: move `_write_plots` into a shared module and call it from both.**

---

## 3. Implementation plan

### Phase 1 — extract the plot writer (no behavior change)

1. Create `simulation/grasp_plot_writer.py` (or `kinova_common/grasp_plots.py`)
   holding `write_grasp_plots(...)`, moved verbatim from
   `benchmarks/ycb_grasp/pick_and_place.py:538`.
   - Imports `ycb_grasp.plot_quadratic_path` / `plot_grasp_contacts`. Those live
     under `benchmarks/`; either add the `benchmarks` path (teleop already does
     `sys.path.insert` for `simulation/`) or move the two plot modules alongside.
     Prefer the sys.path route first — moving them churns 5 other callers.
2. `pick_and_place.py` imports and delegates. Verify byte-identical output on
   `036_wood_block` before continuing.

### Phase 2 — mesh support in the recommender

3. **Narrow the site guard** (line 673) so it only fires for modes/paths that
   actually consume `id_S`. The recommender path does not. Concretely: keep the
   error for the legacy site-driven `_run_ik` selection path, drop it for
   `contact_aware_teleop` + `contact_aware_autonomous`, which always commit through
   `_setup_recommended_contact_frames`.
4. **Replace `_CAT_SUPPORTED` (line 2500) with a capability test**, not a name
   allowlist: an object is supported if its geom is a primitive the planner handles
   OR a mesh for which `_mesh_sdf_entry(model, bid)` bakes successfully.

   **BAKE ALL SDFs EAGERLY AT STARTUP — this is a hard requirement, not an
   optimization.** `_mesh_sdf_entry` (grasp_planner_3d.py:336) lazily bakes and
   caches via `object_sdf.load_or_bake`. Leaving it lazy puts the first bake inside
   the 2 s background recommender thread on first approach, which stalls that solve
   and (worse) makes the stall look like a planner failure. Bake every spawned
   object's SDF once, before the viewer opens, with a printed per-object progress
   line and timing:

   ```python
   # After compile, before viewer launch. Fail loudly per object; an object whose
   # SDF will not bake is simply marked unsupported rather than aborting the run.
   _cat_mesh_entry = {}    # obj_idx -> mesh_entry | None
   for _i, _o in enumerate(objects):
       _t0 = time.time()
       try:
           _cat_mesh_entry[_i] = _mesh_sdf_entry(model, _o['id_body'])
           print(f"[sdf] {_o['name']}: baked in {(time.time()-_t0)*1e3:.0f} ms")
       except Exception as _e:
           _cat_mesh_entry[_i] = None
           print(f"[sdf] {_o['name']}: UNSUPPORTED ({_e})")
   ```

   Then `_recommender_tick`'s `_supported` test and `_recommended_inward_normals`
   both read this cache — no bake ever happens on the render or solver thread.
   `load_or_bake` also persists to disk, so the cost is paid once per object across
   runs; the startup line should say which bakes were cache hits.
5. **Fix `_recommended_inward_normals`** (line 2801) to pass `mesh_entry=` for mesh
   geoms. Import `recommended_inward_normals` from the shared home rather than
   keeping teleop's primitive-only copy.
6. Confirm `id_geom` resolves for YCB bodies — it does:
   `environments/scene_objects.py:143` names the collision geom `<body>_geom`, the
   same convention `for_teleop_recommender` derives (`obj_name + '_geom'`).
   Caveat: a convex DECOMPOSITION splits a concave object into many geoms
   (scene_objects.py:178); `obj['id_geom']` grabs one. Meshes with multi-hull
   decomposition need the hull-set treatment `pick_and_place._obj_hull_geom_ids`
   already implements. Start with single-hull objects (`017_orange`, `014_lemon`,
   `036_wood_block`, `056_tennis_ball`).

### Phase 3 — turn on the quadratic patch + logging

7. Add to the `for_teleop_recommender` preset (or pass as overrides from
   `_get_cat_planner`), matching pick_and_place.py:233:
   ```python
   use_quadratic_contact=True,
   quadratic_mesh_fit=True,
   ground_z=TABLE_TOP_Z,        # teleop already overrides this global per --scene
   ```
   Keep the existing teleop-tuned `w_ik=5.0`, `w_align=10.0`, `edge_margin_m=0.03`,
   `wrench_constraint=False`, `datum_gamma=True` — those are load-bearing
   (RAISED_CONTACT_WRENCH_FINDINGS.md §5).
8. Add `--rec-log-dir DIR` (default off). When set, `_get_cat_planner` passes
   `log_dir=<DIR>/<object>/` to `MultiStartGraspPlanner3D`.

### Phase 3b — contact settings as a SEPARATE, mode-scoped profile

**Measured state of the world (verified live on `models/scene_pick_place.xml`):**

```
impratio          = 100.0      (from the XML)
noslip_iterations = 0          (MuJoCo default -- NOT set anywhere)
timestep          = 0.002
fingertip solref  = [0.004, 1.0]   on all 4 *_tip geoms   (stock)
```

So teleop today runs the **untuned** configuration. `contact_tuning.py` records the
tuned alternative (`FINGERTIP_SOLREF=(0.02,2.0)`, `NOSLIP_ITERATIONS=5`, `GAMMA=10.0`,
`SQUEEZE_PD_SCALE=1.0`) and even ships `apply_to_model(model, object_id=None)` (:120)
— but **that function has zero callers anywhere in the repo.** The findings were
recorded and never wired up, in the benchmark either.

**Why this matters here.** The stock fingertip `solref=[0.004,1.0]` is tau = 2
timesteps at dt=0.002. contact_tuning.py's docstring identifies that as the release
"fling" mechanism directly: the index contact was measured making and breaking on
ALTERNATING 2 ms steps (f = 0.85, 0.0, 0.99, 0.0, ...) while an object entering
release at 0.03 m/s was pumped to 2.07 m/s — energy with no physical source. And
because MuJoCo takes the **MIN** of the two geoms' solref, softening the OBJECT does
nothing while the fingertip stays at 0.004.

**Do NOT flip these globally.** Two hard constraints from the findings:

1. **`noslip` and `gamma` are a PAIR.** noslip=5 fixes the release (065-e_cups
   2.02 -> 0.00 m/s) but removes the tangential compliance the heavy 036_wood_block
   was leaning on; its lift collapsed 96 mm -> -6 mm. Restoring gamma 10.0 +
   squeeze_pd_scale 1.0 brings it back to 96 mm AND keeps the clean release.
   *"Neither change works without the other — do not adopt one of them alone."*
   Teleop's gamma is NOT a free constant: it comes from `solve_gamma_live` scaled by
   `GAMMA_SAFETY_FACTOR` (:4736). Forcing gamma=10.0 would override the very
   certificate this whole architecture exists to honour. **Resolve this before
   enabling noslip in teleop** — the benchmark's gamma=10.0 is a fixed override, and
   teleop's is a live certificate; they are not interchangeable.
2. **Changing `impratio` perturbs settling -> changes the plan.** It can land a
   fingertip gap just over the 8 mm `CONTACT_GAP_TOL_M` gate, producing a 0.0 N
   "failure" that is a PLANNER outcome. Always check `phase_log` for
   `squeeze_aborted_no_contact` before blaming contact settings.

**Design — a mode-scoped profile, so dexpilot/anyteleop are untouched:**

Add `--contact-profile {stock,tuned}` (default **`stock`**) to
`kinova_leap_pick_place.py`, applied AFTER `_spec.compile()` (i.e. after line 481,
where `model`/`data` exist) and **only when the mode is contact-aware**:

```python
_CONTACT_AWARE = args.mode in ('contact_aware_teleop', 'contact_aware_autonomous')
if args.contact_profile == 'tuned' and _CONTACT_AWARE:
    applied = contact_tuning.apply_to_model(model, object_id=<active ycb id>)
    print(f"[contact] tuned profile applied: {applied}")
elif args.contact_profile == 'tuned':
    sys.exit("--contact-profile tuned is only valid for the contact-aware modes; "
             "dexpilot/anyteleop are the untouched baselines.")
```

Refusing the flag outright on dexpilot/anyteleop is better than silently ignoring it
— it makes "the baselines are stock" a checked invariant rather than a convention.
Reuse `contact_tuning.apply_to_model` rather than re-deriving the numbers; this
finally gives that function its first caller.

Two gaps in `apply_to_model` to close while wiring it:
- It **reports** `obj_condim` in the returned dict but never applies it (:142-144).
  Either apply it to the object's collision geoms or rename the key so it cannot be
  mistaken for an applied setting.
- It does not set `GAMMA` / `SQUEEZE_PD_SCALE` (they are module constants for the
  caller to read). Given constraint (1) above, teleop should NOT blindly adopt
  `GAMMA=10.0` — see the validation gate below.

### Phase 3c — VALIDATE pick-and-drop before trusting either profile

Per the user's ask: if we start on global/stock settings, prove the basic behavior
first. This is a gate, not a nice-to-have — the release-fling failure mode is
invisible until you actually drop something.

Run each profile over the four single-hull objects and record, per object:
grasp force (N), lift height (mm), and **peak |v| at release** (m/s).

```
                     stock (today)        tuned (contact_tuning)
036_wood_block       ? / ? / ?            5.7 N / 96 mm / 0.04 m/s
056_tennis_ball      ? / ? / ?            ?
014_lemon            ? / ? / ?            ?
017_orange           ? / ? / ?            6.2 N / 115 mm / 2.92 m/s  KNOWN-BAD
```

Reference numbers in the right column are contact_tuning.py's tabletop measurements;
`056_tennis_ball` and `014_lemon` were not in that sweep, so they need measuring.
`017_orange` is a documented genuine exception — a smooth sphere rolls out of a
two-finger pinch, it CONVERGES under timestep refinement (1.68/1.60/1.58 m/s at
dt=2/1/0.5 ms), so it is what the model says happens and no contact setting fixes it.
**Do not tune against the orange**; record it and move on.

Only adopt `tuned` as teleop's default if it holds grasp force AND lift AND improves
release across the non-orange objects — and only once the gamma question in
constraint (1) is settled.

### Phase 4 — emit plots at lock-in, off the render thread

9. The recommender fires every 2 s. **Do not plot every solve** — matplotlib Agg at
   ~1-2 s/figure would starve the control loop and fill the disk.
   Plot on the COMMIT (`L`), which is the grasp that actually gets executed:
   - In `_fire_recommender`'s `_run()`, stash `res` + `verify_info` + the solve's
     `log_dir` snapshot on the candidate dict (`res` is needed for the winning-attempt
     match in §2).
   - On `L` lock-in, hand that to a **daemon worker thread** that calls
     `write_grasp_plots(...)` and writes to `<rec-log-dir>/<object>/lockin_<n>/`.
   - Because each solve reuses one `log_dir`, either clear it per solve (as
     pick_and_place.py:270 does with `shutil.rmtree`) or timestamp per solve.
     Timestamping is safer here — a background solve may overlap the plot worker.
10. Optional `--rec-plot-every-solve` for offline analysis runs, explicitly documented
    as not for live teleop.

---

## 4. How it gets invoked

Today (primitives, no plots):
```bash
python kinova_leap_pick_place.py --mode contact_aware_teleop --objects obj_red_box
```

After this work (YCB meshes + figures):
```bash
# scene_objects.json's "pick_place" list drives the YCB objects
python kinova_leap_pick_place.py \
    --scene pick_place \
    --mode contact_aware_teleop \
    --rec-log-dir out/teleop_rec/run1
```
Then in the viewer: `8` start tracking -> approach the object -> `L` lock in.
Each `L` writes `out/teleop_rec/run1/<object>/lockin_<n>/seed<k>_grasp_contacts.png`.

Autonomous A/B on the identical scene (no hand tracking needed — arrow-key jog):
```bash
python kinova_leap_pick_place.py --scene pick_place \
    --mode contact_aware_autonomous --recommender-grasp \
    --rec-log-dir out/teleop_rec/auto1
```

The benchmark's existing `--teleop-cmd` (pick_and_place.py:638) already prints the
matching teleop command; extend it to include `--rec-log-dir` once this lands.

Via the launcher (note `start_teleop.sh sim` defaults to `dexpilot`, unlike the Python
default, and always forces `--no-mediapipe`):
```bash
./start_teleop.sh sim contact_aware_teleop --rec-log-dir out/teleop_rec/run1
```

### Programmatic trigger (optional, if you want unattended capture)

The `L` lock-in is a GLFW key mapped in `kinova_common/keyboard.py:14-74` onto a
`queue.Queue`. That queue is ALREADY written from inside the loop for synthetic
events (`keys.put('enter')` at :4592 for the auto-squeeze deadline, `keys.put('reset')`
at :4544). ROS is inbound-only today — `teleop/ros_interface.py` subscribes to
`/hand/joint_angles` and nothing else. So a remote/scripted trigger is a small
addition: a new subscriber calling `keys.put('lock_in')` reuses the entire existing
handler at :3897 with no other change. Worth having if you want to batch-collect
figures across the object set without a human at the viewer.

---

## 5. Risks / things that will bite

- **First mesh SDF bake is slow — MEASURED at ~21 s PER OBJECT.** Confirmed on this
  machine: the five-object `pick_place` scene spends ~130 s in `_mesh_sdf_entry`
  before the viewer opens (gelatin 21.6 s, wood_block 21.2 s, orange 21.2 s, lemon
  20.2 s, tennis_ball 45.9 s). This is why the eager bake is a hard requirement, not
  an optimization — lazily inside the 2 s recommender thread it would have been
  catastrophic, and would have looked like a planner hang.

  **It is NOT amortized by the disk cache, and the cost is in ONE place.** Profiled
  per stage on `017_orange` with the npz already on disk:

  ```
  load_or_bake           19 ms     <- the disk cache works; the grid is NOT the cost
  casadi_fn           23634 ms     <- the B-spline interpolant over the SDF lattice
  casadi_grad_fn         41 ms
  hessian construct     114 ms
  ```

  So it is `object_sdf.casadi_fn` — constructing the CasADi B-spline over the full
  SDF grid — and it is paid on EVERY run regardless of the cache. A contact-aware
  teleop session on the default 5-object scene therefore starts ~2 minutes slower
  than a dexpilot one.

  (An earlier guess that `hessian_fn` was the culprit was wrong — it is 114 ms.
  Whatever is done about this has to target `casadi_fn`.)

  Worth fixing if it becomes a workflow irritant, but NOT in scope here and NOT to be
  guessed at: the fix is either a coarser lattice for the live recommender, caching
  the serialized CasADi function (`fn.save()`/`ca.Function.load`) next to the npz, or
  baking the objects in parallel across processes. Until then, iterate with a trimmed
  object set — `--objects` for primitives, or a shortened `models/scene_objects.json`
  list for meshes.
- **Solve time.** The teleop preset uses `max_iter=120` and 5 seeds against a
  primitive. Mesh SDF constraints are heavier; expect the 2 s cadence to be missed.
  `_recommender_tick` already guards with `_rec_idle`, so it degrades to "solve as
  fast as it can" rather than piling up threads — but the on-screen `solving` badge
  will be lit most of the time. Consider raising `_REC_INTERVAL_S` for meshes.
- **`n_seeds=5` + one `log_dir`** means several attempts' npz in one directory. Always
  pass `res=` to `_iter_trace_quadratic_stages` (see §2).
- **Trust-region pinning.** SOLVER_STATE.md §2 records `pinned=True` 9/9 stages — the
  Picard loop takes maximum-length steps. The contact-movement plot will therefore
  show seed + N x bound marches, not interior optima. That is a known, documented
  solver property, not a bug introduced here.
- **`009_gelatin_box` cannot be planned lying flat** (SOLVER_STATE.md §9): fingertip
  r = 19.4 mm vs a 28 mm box. It is in the default `pick_place` list — expect it to
  report unsupported/unreachable. Not a regression.
- **Don't touch dexpilot / anyteleop.** They are the paper's baselines.
- **`log_dir` is deleted by the benchmark after plotting** (pick_and_place.py:315
  `shutil.rmtree`). Teleop must NOT copy that: the plot worker runs async on a
  background thread, so a rmtree racing it would delete the npz mid-read. Timestamp
  per solve and clean up on exit instead.
- **Multi-hull V-HACD objects.** `pick_and_place._obj_hull_geom_ids` (:117) exists
  precisely because a concave YCB object compiles to several collision geoms; teleop's
  `obj['id_geom']` (line 658) resolves exactly one by `<body>_geom`. Concave objects
  (`025_mug`, the `065-*_cups`) need that hull-set treatment before they will work.
- **The `seed` argument.** `MultiStartGraspPlanner3D.solve()` RESETS `self._rng` to the
  constructor seed on every call, for pose-continuity. Combined with
  `_fixed_antipodal_seed` going in first (SOLVER_STATE.md §1), a static object returns
  the same candidate every 2 s — which is what makes the display hysteresis coherent.
  Don't "fix" that by reseeding per solve.

---

## 7. Contract this must preserve

`for_teleop_recommender`'s decoupled architecture is load-bearing and must survive:
`wrench_constraint=False` (IK-only NLP) + `datum_gamma=True` (gamma certified as a
post-solve LP in `verify()`, under the SAME formulation `solve_gamma_live` uses at
grasp time). That alignment is the whole point — it makes the recommender's
feasibility flag agree with the gate that actually admits the squeeze
(RAISED_CONTACT_WRENCH_FINDINGS.md §5). Adding `use_quadratic_contact=True` changes
the CONTACT PARAMETERIZATION only; it must not drag the wrench block back into the NLP.

Teleop's budget is also 40x the benchmarks' (`NCF_ACCEL_BUDGET_XYZ = 20 m/s^2` vs
0.5), which is safe only because the jog is slew-limited to exactly that budget
(SOLVER_STATE.md §7). Keep passing teleop's own budgets into the builder.

---

## 6. Suggested commit split

1. `refactor: shared grasp plot writer (no behavior change)`
2. `teleop recommender: eager SDF bake + mesh-aware normals + capability gate`
3. `teleop recommender: quadratic contact patch + optional --rec-log-dir`
4. `teleop recommender: write grasp analysis figures on lock-in`
5. `teleop: --contact-profile, contact-aware modes only (default stock)`
6. `docs: measured pick/drop validation table for both contact profiles`

Commits 1-4 are the integration; 5-6 are the contact-settings question, which is
independent and can land separately. Do 1-4 first: the analysis figures are what
make the contact validation in 6 legible.

---

## 9. Open question to settle during Phase 3b

`contact_tuning.py` pairs `noslip_iterations=5` with a FIXED `gamma=10.0`. Teleop
derives gamma from `solve_gamma_live` + `GAMMA_SAFETY_FACTOR` — a certificate against
its 20 m/s^2 budget, which the slew limiter then enforces by construction. Those are
two different contracts for the same symbol.

Overriding teleop's gamma to a fixed 10.0 would discard the certificate; leaving it
live while enabling noslip risks the 96 mm -> -6 mm lift collapse the findings
measured on the heavy block. The honest options are (a) keep `stock` as teleop's
default and treat `tuned` as a measurement profile only, or (b) re-derive the noslip
compensation as a floor ON the certified gamma (`gamma = max(gamma_live, gamma_floor)`)
and re-measure. **Recommend (a) until Phase 3c's table exists** — there is no basis
for changing teleop's force contract before the pick/drop numbers are in hand.

## Installation

This project uses [uv](https://docs.astral.sh/uv/). Create and sync the environment:
```bash
uv sync
```

Run any script through uv (no manual activation needed):
```bash
uv run python kinova_leap_pick_place.py
```
Or activate the venv directly: `source .venv/bin/activate`.

Pull MuJoCo Menagerie submodule models:
```bash
git submodule update --init --recursive
```

The `--dashboard` flag runs a pyqtgraph (Qt 6) GUI in a separate process. On Linux, Qt
6.5+ requires the `libxcb-cursor0` system library to load its xcb (X11) platform plugin;
without it the dashboard process fails to start silently (no window, no error in the main
terminal). Install it once:
```bash
sudo apt-get install libxcb-cursor0
```

---

## Kinova Gen3 + LEAP Hand Pick-and-Place

**Scene:** `models/scene_pick_place.xml` — 7-DOF Gen3 arm + 16-DOF LEAP hand, 6 randomised pickable objects.

`models/kinova_leap.xml` (torque-control, used above) and `models/kinova_leap_pos.xml` (position-servo, used by `kinova_leap_rrt_pos_test.py`) are generated, not hand-written — they merge the Gen3 arm and LEAP hand specs from `mujoco_menagerie/` via MuJoCo's `MjSpec` API. Regenerate them after updating either menagerie submodule or the mount pose/fingertip-site config in the script:

```bash
python models/build_kinova_leap.py
```

### Modes

#### Contact-aware autonomous grasp (default)
RRT path planning + constrained IK + internal-force grasp controller, planning to predefined per-object contact sites. Keyboard-driven target selection.

```bash
python kinova_leap_pick_place.py
# equivalently: --mode contact_aware_autonomous  (the old `rrt` is a deprecated alias)
```

#### Contact-aware teleoperation
Teleop the wrist (same DexPilot MediaPipe mapping) with MediaPipe-driven fingers, while an NLP grasp recommender continuously solves for the best 2-finger contacts on the object nearest your fingertips. The recommended thumb/index contacts are shown live as translucent spheres in the viewer (green = thumb, cyan = index) and refresh on a fixed interval as you move. Press **`L`** to lock them in and approach via RRT; then **`Enter`** to grasp, using the internal-force scale γ solved for that grasp geometry.

```bash
python kinova_leap_pick_place.py --mode contact_aware_teleop --camera 0
```

Requires a sourced ROS 2 environment (see [ROS 2 setup](#ros-2-setup) below). The recommender is currently validated on box-shaped objects; other shapes are recommended-but-unproven and are skipped until proven.

**Flow:** press **`8`** to start tracking (hold your hand at a neutral orientation — that instant is captured as home). Move near a box; watch the recommendation markers. **`L`** locks in → IK + RRT approach the recommended contacts. **`Enter`** commits to GRASP (γ re-solved on the committed geometry), **`Enter`** again toggles the internal-force squeeze, **`N`** releases and hands control back to teleop, **`Backspace`** resets. Pair with `--dashboard` to see the grasp-recommender solve statistics (see [CLI flags](#cli-flags)).

#### DexPilot teleoperation
Live MediaPipe hand retargeting via ROS 2, no grasp recommender — for wrist/finger retargeting and orientation calibration. Launches the MediaPipe publisher and MuJoCo viewer together.

```bash
python kinova_leap_pick_place.py --mode dexpilot --camera 0
```

Requires a sourced ROS 2 environment (see [ROS 2 setup](#ros-2-setup) below).

**Start tracking:** the robot holds its home pose until you press **`8`**. Hold your hand at a comfortable neutral orientation and press `8` — that instant is captured as home, so the robot's wrist orientation is treated as matching yours, and it then follows your movement and rotation relative to that pose. `Q`/`Esc` quits.

**Multi-camera (fused) input.** Both teleop modes take their hand pose from `/hand/joint_angles`, which by default the built-in single-camera publisher supplies. Pass `--multicam-auto` (or `--multicam` with explicit specs) to instead auto-launch the [multi-camera pipeline](teleop/MULTICAM.md) (triangulated, occlusion-robust) as a child process — no separate terminal:

```bash
# hands-off: discover + match all calibrated cameras by hardware id
python kinova_leap_pick_place.py --mode dexpilot \
    --multicam-auto --skeleton-view --camera-views

# or name each camera explicitly
python kinova_leap_pick_place.py --mode dexpilot \
    --multicam c0:0 --multicam c1:2 --multicam rs:8 --multicam-realsense rs \
    --skeleton-view --camera-views
```

`--multicam-auto` finds every calibrated camera currently plugged in (RealSense auto-flagged) — no indices or names to pass. With explicit `--multicam`, per-camera resolution is read from `camera_intrinsics_<name>.json`, so a bare `NAME:INDEX` is enough. Add `--recalibrate-extrinsics` to re-solve each camera's board pose interactively before teleop starts. See [CLI flags](#cli-flags) and [`teleop/MULTICAM.md`](teleop/MULTICAM.md).

---

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--mode {contact_aware_autonomous,contact_aware_teleop,dexpilot}` | `contact_aware_autonomous` | **contact_aware_autonomous**: autonomous RRT+IK grasp to predefined per-object contact sites (`rrt` is a deprecated alias). **contact_aware_teleop**: wrist+finger teleop with a live NLP grasp recommender, lock-in → RRT → grasp. **dexpilot**: live MediaPipe retargeting teleop via ROS 2, no recommender. |
| `--ik-solver {sqp,ipopt}` | `sqp` | IK solver backend (see below). |
| `--dashboard` | off | Launch a live pyqtgraph metrics dashboard (separate process): planning mode, proximity-based active object, scrolling fingertip→object distances, net hand→object wrench, per-finger contact normal forces, a combined RRT+IK planner solution log, and — in `contact_aware_teleop` — a **grasp-recommender panel** with per-solve statistics (status, seeds converged, solve time, γ_min, wrench feasibility, IK error) plus a rolling session summary. |
| `--viz-only` | off | Debug mode: disables arm/hand collision physics and never calls `mj_step`. REACH and GRASP phases hold their IK solution kinematically so you can inspect the IK/RRT result without dynamics interference. |
| `--seed N` | none | RNG seed for object randomization — the same seed reproduces the same layout (positions and sizes). Default: fresh entropy every run. Ignored with `--no-randomize`. |
| `--no-randomize` | off | Skip object randomization entirely: objects keep the positions, sizes, and colors authored in `models/scene_pick_place.xml`. |
| `--camera N` | auto | *(teleop modes only)* Camera index forwarded to the built-in single-camera MediaPipe publisher. Defaults to auto-select (prefers external/USB camera at index ≥ 1). Run `python teleop/ui.py --list-cameras` to see available indices. |
| `--multicam-auto` | off | *(teleop modes)* Hands-off multi-camera: discover connected cameras and match each to its calibration by hardware id — no specs needed. Uses every calibrated camera currently plugged in (≥ 2); RealSense auto-flagged. Equivalent to `run_multicam.py --auto`. Mutually exclusive with `--multicam`. |
| `--multicam NAME:INDEX[:WxH]` | none | *(teleop modes)* Auto-launch the [multi-camera pipeline](teleop/MULTICAM.md) (`teleop/run_multicam.py`) as a child process instead of the single-camera publisher, so `/hand/joint_angles` comes from the **fused** cameras. Repeat per camera (≥ 2), e.g. `--multicam c0:0 --multicam c1:2`. Implies `--no-mediapipe`. Resolution is read from each camera's `camera_intrinsics_<name>.json` when `:WxH` is omitted. Prefer `--multicam-auto` unless you need explicit indices. |
| `--multicam-realsense NAME` | none | *(with `--multicam`)* Mark a camera as an Intel RealSense — its node captures the COLOR stream via `pyrealsense2` (the `:INDEX` is then ignored; the SDK picks the device). Default 640×480 @ 30 fps (the D435I's 1080p color is only 8 fps). Repeatable. |
| `--multicam-max-res` | off | *(with `--multicam`)* Open each camera at its highest supported resolution (forwards `--max-res`). Prefer omitting it — resolutions come from the intrinsics files automatically. |
| `--recalibrate-extrinsics` | off | *(with `--multicam`)* Run the interactive extrinsics solve for each camera **before** teleop starts (fix the ChArUco board at the world origin, press **SPACE** per camera). Reuses each camera's calibrated resolution. Default reuses the saved extrinsics. |
| `--square-mm MM` | `50.0` | *(with `--recalibrate-extrinsics`)* MEASURED ChArUco square size in mm, forwarded to `charuco_calibration.py extrinsics`. Must match the board actually printed/mounted on the rig — measure a square with calipers, don't trust the nominal print size. Keep in sync with `DEFAULT_SQUARE_MM` in `teleop/calibration/charuco_calibration.py` if the board changes. |
| `--camera-views` | off | *(with `--multicam`)* Tile each camera's live feed + landmark overlay in a window (subscribes to `/hand/cam_<name>/preview`), like `run_multicam.py --show-fused`'s camera grid. |
| `--skeleton-view` | off | *(teleop modes)* Open a separate orbitable 3D window of the fused hand skeleton (from the world landmarks in `/hand/joint_angles`). |
| `--no-mediapipe` / `--external-hand` | off | *(teleop modes)* Do **not** spawn the built-in single-camera publisher — for when an external process already publishes `/hand/joint_angles`. Implied by `--multicam`. |

Flags can be combined, e.g.:
```bash
python kinova_leap_pick_place.py --mode contact_aware_teleop --dashboard --camera 0
python kinova_leap_pick_place.py --mode dexpilot --camera 1
python kinova_leap_pick_place.py --viz-only --dashboard
python kinova_leap_pick_place.py --ik-solver ipopt
python kinova_leap_pick_place.py --seed 42          # reproducible object layout

# multi-camera (fused) teleop — hands-off: discover + match calibrated cameras:
python kinova_leap_pick_place.py --mode dexpilot --multicam-auto \
    --skeleton-view --camera-views
```

---

### IK solver modes

The constrained IK problem has ~500 collision-avoidance inequality constraints (71 arm/hand geoms × 7 scene objects) on 23 DOFs. Two solver backends are available:

#### `sqp` (default) — sqpmethod + OSQP + softplus SDF + analytic Jacobians

Each IK iteration solves a quadratic programme (QP subproblem) over a linearisation of the constraints, using an L-BFGS Hessian approximation. Constraint Jacobians are computed analytically via `mj_jacSite` / `mj_jac` rather than by finite-differencing each DOF, so each iteration requires ~1 MuJoCo FK evaluation per callback instead of 24.

The signed distance functions (box, cylinder) use a softplus-smoothed `max(x, 0)` — i.e. `softplus(x) = max(x,0) + log(1+exp(−αx))/α` with α = 500 — so the gradient is C∞ everywhere and CasADi's chain-rule AD is valid. OSQP (ADMM) is used as the QP solver because with ~500 constraints on 23 DOFs the linearised QP is often primal-infeasible at the initial warm-start point; OSQP finds the minimum-constraint-violation step direction and lets SQP continue, whereas qpOASES fails hard.

Benchmarked against 6 object shapes with a DLS warm-start: **4/6 objects faster** in wall time vs the IPOPT baseline, with ~10× fewer FK evaluations per iteration (160–220 vs 2100–2200).

#### `ipopt` — IPOPT L-BFGS + finite-difference Jacobians

IPOPT's interior-point method with L-BFGS Hessian approximation. Constraint Jacobians are computed by IPOPT's own finite-difference perturbation (`jacobian_approximation = finite-difference-values`), which evaluates each callback at `q ± ε` for all 23 DOFs per iteration. This is slower per iteration but the FD averaging inadvertently smooths the kinks in the raw `max(x,0)` SDF, which helps L-BFGS accumulate a consistent curvature estimate. Useful as a fallback when SQP stalls (e.g. objects with complex concavities where the QP subproblem is repeatedly infeasible).

---

### Contact-aware autonomous mode controls

| Key | Action |
|---|---|
| `Ctrl+1` … `Ctrl+6` | Select grasp target (objects 1–6) |
| `Ctrl+0` | Return to home pose |
| `Enter` | Commit to GRASP / toggle internal-force squeeze |
| `N` | Release back to pregrasp |
| `← →` | Jog object x-position while grasping |
| `↑ ↓` | Jog object z-position (height) while grasping |
| `PgUp` / `PgDn` | Jog object y-position (depth) while grasping |
| `6` | Cycle IK config visualisation |
| `7` | Toggle IK collision bounding-sphere overlay |
| `Backspace` | Reset to home |
| `Q` / `Esc` | Quit |

### Contact-aware teleop mode controls

| Key | Action |
|---|---|
| `8` | Start / re-zero tracking (captures current hand orientation as home) |
| `L` | Lock in the recommended grasp contacts and approach via RRT |
| `Enter` | Commit to GRASP / toggle internal-force squeeze |
| `N` | Release and return to teleop control (re-arms the recommender) |
| `← →` / `↑ ↓` / `PgUp` `PgDn` | Jog the grasped object (x / z / y) after locking in and grasping |
| `Backspace` | Reset to home and re-arm the recommender |
| `P` | **Debug preview:** hold the robot at the recommender's own `q` (unconstrained NLP solution, no collision IK) |
| `O` | **Debug preview:** hold the robot at the collision-aware IK solution, warm-started from the recommender's `q` |
| `I` | **Debug preview:** same collision-aware IK, warm-started from a fresh DLS solve (A/B the warm-start against `O`) |
| `Q` / `Esc` | Quit |

The three preview keys (`P` / `O` / `I`) each hold the robot kinematically at one candidate pose for the *same* recommendation, so you can compare where the recommender wants the fingers versus what the collision-aware IK can actually reach, and how the two warm-starts differ. They are mutually exclusive; the recommender pauses re-solving while a preview is held. Lock-in itself uses the `O` variant (collision-aware IK warm-started from the recommender's `q`), and RRT plans to that refined solution.

The **grasp recommender** solves an NLP for two wrench-feasible contact points on the nearest supported object (currently box-shaped objects), visualising them as translucent spheres (green = thumb, cyan = index). Because the recommender's tip-reaching term is soft relative to its wrench/regularisation terms, its raw pose (`P`) does not exactly reach the contacts; the lock-in IK (`O`) refines this under collision constraints. Pair with `--dashboard` for per-solve statistics.

---

### ROS 2 setup

The teleop modes (`contact_aware_teleop`, `dexpilot`) need a sourced ROS 2 environment with CycloneDDS configured. The simplest path is to source the provided `setup.sh` — it sources ROS 2 Humble, pins CycloneDDS to loopback (`lo`), and fixes the `ROS_DOMAIN_ID`, all for a single-machine session where the MediaPipe publisher and the MuJoCo subscriber run on the same host:

```bash
source setup.sh
uv run python kinova_leap_pick_place.py --mode contact_aware_teleop --camera 0
```

Loopback is the right transport here — no dependence on `eno1`/`eno2`/wifi link state, lowest latency. If you need a physical interface instead (e.g. a publisher on another machine), set `CYCLONEDDS_URI` to a `state UP` interface (`ip link show` to list):

```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General></Domain></CycloneDDS>'
```

---

## 2D Environment

```bash
python internal_force_control.py
```

---

## YCB object assets

The [YCB object set](https://www.ycbbenchmarks.com/) provides scanned meshes of everyday
objects. They ship as textured **concave** meshes, which MuJoCo cannot use for contact
directly — a `<geom type="mesh">` collides against its *convex hull*, so a bowl becomes a
solid dome and a drill's trigger cavity fills in. `scripts/build_ycb.py` converts each
object into an MJCF with a convex-decomposed collision proxy, so concavities survive into
the physics.

Both `assets/ycb_raw/` (1.5 GB of downloads) and `assets/ycb_mjcf/` (584 MB of generated
output) are gitignored — regenerate them with the steps below.

### 1. Download

YCB is published in a public S3 bucket, so no AWS account or credentials are needed —
`--no-sign-request` fetches anonymously:

```bash
aws s3 cp --no-sign-request --recursive s3://ycb-benchmarks/ assets/ycb_raw/ \
  --exclude "*" --include "*google_16k*"
```

The `--exclude "*" --include "*google_16k*"` filter is the important part: every object is
published in `google_16k`, `google_64k`, and `google_512k` variants, and this pulls only
the 16k tier (the Google-scanner reconstruction decimated to ~16k faces — the sim-friendly
size). Without the filter the download is many times larger for no benefit here. Expect
~1.5 GB and a while to transfer.

That lands the tarballs at `assets/ycb_raw/data/google/<object>_google_16k.tgz`, though
the build step finds them anywhere under `assets/ycb_raw/`. Each archive already contains
the `<object>/google_16k/textured.obj` layout the converter looks for, so no manual
unpacking or reorganising is needed.

**84 objects, not 85.** `076_timer_google_16k.tgz` is a 45-byte stub *in the upstream
bucket* — not a failed download. It is skipped with a `no google_16k mesh` message; every
other object converts.

### 2. Convert

```bash
uv run python scripts/build_ycb.py
```

This extracts any not-yet-extracted archive in place, then per object:

1. Loads `google_16k/textured.obj` (the scan, with UVs).
2. **Seals the scan into a watertight solid** (`scripts/mesh_seal.py`) so that volume —
   and therefore mass and inertia — is defined at all.
3. Runs [CoACD](https://github.com/SarahWeiii/CoACD) convex decomposition at
   progressively finer concavity thresholds, **measuring against that solid after each
   one**, and keeps the first result whose falsely-solid volume is acceptable.
4. Writes `assets/ycb_mjcf/<object>/<object>.xml` with the visual mesh, the hulls, a
   texture/material binding, and an `<inertial>` carrying the **published YCB mass**,
   plus a fit record in `assets/ycb_mjcf/fit_report.json`.

The script is idempotent: objects already built at the current settings are skipped
(`--force` overrides), and the report is checkpointed after each object, so an
interrupted run resumes. A full 84-object run takes a few hours; `--jobs` sets how many
objects run at once, but CoACD is itself heavily threaded (~28 cores on one object), so
raising it past a handful mostly buys contention.

#### The hull count is an output, not a setting

The knob used to be `max_convex_hull=8`, applied uniformly. That fixes the *cost* of a
collision proxy rather than its *fidelity*, and across shapes the two are unrelated —
which is why near-convex objects came out right and cavities filled in. Auditing the
whole set built that way, overshoot outside the scan ranges from 0.33 mm to 33.9 mm
(median 4.07), and 57 of 84 objects sit beyond 3 mm:

| object | hulls | overshoot p99 |
|---|---|---|
| `058_golf_ball` | 1 (CoACD stopped early) | 0.33 mm |
| `013_apple` | 8 | 2.1 mm |
| `011_banana` | 4 (stopped early) | 3.9 mm |
| `048_hammer` | 5 (stopped early) | 8.0 mm |
| `065-a_cups` | 8 | 10.4 mm |
| `024_bowl` | 8 | 14.0 mm |
| `025_mug` | 8 | 17.4 mm |
| `029_plate` | 8 | 20.1 mm |
| `027_skillet` | 8 | 31.6 mm |
| `028_skillet_lid` | 8 | 33.9 mm |

53 objects hit the cap of 8 and were truncated mid-decomposition; 46 of those are the
ones beyond 3 mm. But the 31 that stopped early are not automatically fine either —
11 of them also miss, because CoACD's *own* default concavity threshold is no more
consistent across shapes than a hull count is: it stops a banana at 3.9 mm and a hammer
at 8.0 mm. Neither knob is a fidelity criterion. Measuring is.

Holding the *error* fixed instead lets the count land where the shape requires it: a
golf ball takes 1 hull, an apple 1, a banana 5, `065-a_cups` 36, a mug ~70.
`--max-hulls` is only a contact-cost backstop, and objects that hit it are reported as
`budget_limited` rather than silently degraded.

#### Sealing the scans, so that volume means something

The scans are **open surfaces**: `025_mug` arrives as 157 disconnected components with
~4.8k boundary edges, so "inside" is undefined, `trimesh` reports `is_watertight ==
False`, and `mesh.volume` is a meaningless number. Nothing volume-based — fit, mass, or
inertia — is computable until that is fixed. `scripts/mesh_seal.py` does it, and two
routes were measured against objects with published masses so the answer could be
checked:

| route | mug implied density | bowl | verdict |
|---|---|---|---|
| MeshFix surface repair (`pymeshfix`) | 6226 kg/m³ | 4407 | fragments; wrong per object |
| solid voxelisation + flood fill | 692 kg/m³ | 1637 | consistent |

MeshFix is exact on one clean shell, but these are not: run per connected component,
the mug's 36 surviving fragments seal to 19 cm³. Voxelisation — rasterise the shell,
close hairline cracks, flood-fill from outside, call whatever the flood cannot reach
interior — is immune to that, because a crack narrower than the pitch simply closes.
Its implied densities land where the materials say: golf ball 1107 (real ~1130), sponge
49 (foam ~50), mustard bottle 919 (~1000), cracker box 180 (~200).

Its known bias is that a shell rasterises about a voxel thicker than it is, so
thin-walled objects read heavy. That is bounded by the pitch, which makes it negligible
against the centimetre-scale volumes the fit metric weighs, and it cancels out of the
inertia tensor, which is rescaled to the published mass. Validated against analytic
solids: a box and a sphere come back within 2.4% on volume and 2.3% on inertia.

#### The common metric

Measured on the sealed solid, the gate is **falsely-solid volume** — space the hulls
claim is filled where the object is actually empty. That is the error that fills a cup
and turns a bowl into a dome. It has two limbs, because one threshold does not mean the
same thing on a convex object as on a hollow one:

- `false_solid_frac ≤ --max-false-solid` (default 0.05), as a fraction of the object's
  convex hull. Near-convex objects clear this at once, correctly: `013_apple` scores
  0.018 with a **single** hull, and no hull count beats it — 30 hulls only reach 0.014.
- `recovered ≥ 0.85` — of the false solid a plain convex hull carries, this much won
  back. Hollow objects live here: the banana's hull is 33% false solid and no
  achievable decomposition takes that to 5%.

The 0.05 is set just above a floor CoACD imposes, rather than picked for looks. CoACD
decomposes a **voxel remesh** of the scan at `preprocess_resolution=50` — voxels of
`extent/50`, which is 4.3 mm on the cracker box and 5.0 mm on the bleach cleanser — so
its hulls bulge about one such voxel outside the true surface no matter how many of them
it makes. A 0.02 target sat underneath that floor, and objects with 1.3–3.1 mm of actual
surface error churned the entire ladder and still reported as failures. Aiming below
~0.05 means raising CoACD's preprocess resolution first, which costs roughly 3× the
decomposition time.

`recovered` is the more consistent of the two, and among similar shapes it is tight —
`011_banana` at 0.858 and `065-a_cups` at 0.860 land on 2.98 mm and 2.45 mm of surface
error, `025_mug` at 0.893 on 2.99 mm.

**But do not read it as a bound on surface error.** Across the whole set, the objects
sitting at `recovered ≥ 0.85` span 0.22 mm to 15.97 mm of surface p99 — `027_skillet`
passes the volume gate at 17 hulls while still carrying a 16 mm local error. Volume
agreement and distance agreement are different questions, and holding one fixed leaves
the other loose.

Measured on the independent surface diagnostic, against the old fixed count:

| | median p99 | worst p99 | over 5 mm |
|---|---|---|---|
| fixed 8 hulls | 4.07 mm | 33.90 mm | 34 of 84 |
| volume gate | **2.31 mm** | **15.97 mm** | **11 of 84** |

Better on every aggregate — but **8 objects got worse**, and they are all cases where
volume is blind to features that are shallow and broad. `003_cracker_box` collapsed to a
single hull (a box *is* convex by volume) and went from 3.45 mm to 7.58 mm; `013_apple`
from 2.07 mm to 4.75 mm, its stem dimple costing almost no volume. If that matters for a
given object, `--max-false-solid` will not fix it — the fix is to require the surface
metric as well, which `scripts/mesh_fit.py` already computes and the report already
records as `overshoot_p99_mm` next to the volume numbers.

#### Mass and inertia

Every body carries an explicit `<inertial>` built from the **published YCB mass**
(`scripts/ycb_masses.py`, transcribed from Table II of the YCB paper) with the tensor
computed from the sealed solid and rescaled to that mass. YCB publishes mass and
dimensions but *not* inertia tensors, so uniform density is assumed for the tensor's
shape while the published figure sets its scale.

This is a fix, not a decoration. With no `<inertial>`, MuJoCo infers mass from the geoms
— counting the `contype=0` visual mesh *and* every overlapping collision hull.
`011_banana` came out at **0.379 kg against a published 0.066 kg**, nearly half of it
contributed by a mesh that exists only to be looked at, and the error moved every time
the hull count changed. Of the 84 objects:

- **64 published** — the table names the object directly.
- **15 assembly-share** — the table gives one figure for a multi-part set
  (`071_nine_hole_peg_test` 1435 g, `072_toy_airplane` 570 g, `073_lego_duplo` 523 g),
  split across the parts by sealed volume.
- **5 nominal-density** — no published mass exists (marbles are listed "N/A", the
  Rubik's cube postdates the table), so 1000 kg/m³ is used.

Each object's `mass_source` is in the report, so the assumed ones are never silently
mixed in with the measured ones.

Output per object:

```
assets/ycb_mjcf/
├── fit_report.json           # per-object: hull count, threshold, measured error
└── <object>/
    ├── <object>.xml          # generated MJCF
    ├── textured.obj          # visual mesh (copied)
    ├── texture_map.png       # texture (copied)
    ├── textured.mtl          # copied; MuJoCo does not read .mtl
    └── collision/part_*.obj  # generated convex hulls
```

Each `fit_report.json` record carries the settings it was built at, the CoACD threshold
that won (`"hull"` when a single convex hull sufficed), `false_solid_cm3` /
`false_solid_frac` / `recovered`, the sealed and convex-hull volumes, the diagnostic
`overshoot_p99_mm`, and `mass_kg` with its `mass_source`. Settings live on each record
rather than on the file, so rebuilding one object at a tighter tolerance does not
invalidate the other 83.

Each MJCF is standalone and drops into a scene with a top-level
`<include file="...xml"/>` (an `<include>` may not sit inside `<worldbody>`). Objects
carry a `<freejoint/>` and no floor, so they free-fall unless the scene provides one.

**Geom groups** separate the two representations — `group="2"` is the visual mesh
(`contype=0 conaffinity=0`, purely cosmetic), `group="3"` the collision hulls. In any
MuJoCo viewer press **`2`** / **`3`** to toggle each.

**Textures.** MuJoCo ignores the OBJ sidecar `textured.mtl`, so the generator emits a
`<texture type="2d">` (the default `"cube"` mapping garbles UV-mapped scans) plus a
`<material>`, and binds the material to the visual geom — geoms have no `texture`
attribute of their own, so the `geom → material → texture` chain is required. Collision
hulls stay unmaterialed.

### 3. Inspect

```bash
python -m mujoco.viewer --mjcf=assets/ycb_mjcf/006_mustard_bottle/006_mustard_bottle.xml
```

Press **Space** to pause (else the freejoint drops it out of frame), then **`3`** / **`2`**
to show the hulls and hide the visual mesh. Or use the helper, which holds the object
still and pre-sets the groups (run from `scripts/` — it imports `build_ycb`):

```bash
cd scripts
python view_ycb.py 024_bowl                    # both, toggle with 2 / 3
python view_ycb.py 024_bowl --collision-only   # hulls only
```

Worth checking the concave objects (`024_bowl`, `065-*_cups`, `035_power_drill`). To
grade them instead of eyeballing them, `--audit` re-measures whatever is on disk without
running CoACD at all:

```bash
uv run python scripts/build_ycb.py --audit                  # every built object
uv run python scripts/build_ycb.py --audit --only 024_bowl
```

It seals each object and re-measures both metrics, printing them worst-fit-first — the
fastest way to see whether a tighter tolerance is worth the extra hulls. To rebuild just
the ones that disappoint:

```bash
uv run python scripts/build_ycb.py --only 024_bowl 025_mug --max-false-solid 0.01 --force
```

### 4. Verify

```bash
uv run python scripts/verify_ycb.py
```

Compiles every MJCF in MuJoCo and checks each body's mass against both the report and
the published table, and each collision geom count against the hull files on disk. The
mass check is the one that matters: it fails the moment an `<inertial>` goes missing and
MuJoCo silently falls back to summing geoms.

---

## Camera calibration (ChArUco board)

Calibration produces, per camera, the intrinsics (`camera_matrix`, `dist_coeffs`) and the extrinsic pose (`R_cam_world`, `t_cam_world`) that place the camera in a fixed **world frame** anchored to the board. Single-camera teleop uses these for metric wrist positioning; multi-camera triangulation uses them to fuse views (see [`teleop/MULTICAM.md`](teleop/MULTICAM.md)).

### Board setup

Generate a printable ChArUco board PNG (5×7 squares, 50 mm nominal, sized for A3 paper):

```bash
python teleop/calibration/charuco_calibration.py generate
# or: generate --paper letter   (smaller squares, same 5x7 grid, fits a normal printer)
```

Writes `teleop/calibration/board.png` (~10.2 × 14.2 in at 300 DPI). Options: `--square-mm` (nominal square size, default 50 — `DEFAULT_SQUARE_MM` in `teleop/calibration/charuco_calibration.py`), `--paper {letter,a4,a3}` (sizes the square to fill that sheet instead), `--dpi` (default 300), `--out`.

1. Print at **100% / Actual size** — never "fit to page", which silently rescales and invalidates the metric calibration. The raw PNG has no DPI metadata, so viewers that assume 96 DPI size it wrong; if your viewer misbehaves, place the image on a letter-size 300 DPI canvas first or print from an application that lets you set the scale explicitly.
2. Glue or tape the print completely flat to something rigid (foam board, clipboard).
3. Measure one black square edge with calipers — printers drift a few percent — and pass the **measured** value as `--square-mm` to every step below.

### Single camera

```bash
python teleop/calibration/charuco_calibration.py intrinsics --camera 1 --square-mm 49.6
python teleop/calibration/charuco_calibration.py extrinsics --camera 1 --square-mm 49.6
```

`intrinsics`: wave the board across the frame; **SPACE** captures a view (collect 12+ at varied angle/tilt/distance), **C** calibrates. Aim for RMS < 1.0 px. `extrinsics`: fix the board at the desired world origin facing the camera; **SPACE** averages `--n-avg` frames and solves. Writes `teleop/calibration/camera_intrinsics.json` and `camera_extrinsics.json`.

### Multiple cameras (triangulation rig)

Setting up a rig on a new machine from scratch. Each camera gets its own named calibration files (`camera_intrinsics_<name>.json`, `camera_extrinsics_<name>.json`). Two rules make triangulation valid:

- **Same board pose for all extrinsics.** Every camera's extrinsic must be solved against the *same* fixed board placement — that shared pose is what puts all cameras in one world frame (no stereo calibration needed). Do not move the board between cameras.
- **Calibrate at the capture resolution.** Intrinsics are only valid at the resolution they were estimated at. Each camera's calibrated size is stored in its `camera_intrinsics_<name>.json` (`image_size`), which `run_multicam.py` and the `--multicam` launch **read back automatically** — so a camera streams at its calibrated resolution with no `:WxH` on the CLI. The fusion node hard-errors on a resolution mismatch.

**0. Find your camera indices.** OpenCV indices are assigned by USB enumeration and differ per machine/session:

```bash
python teleop/ui.py --list-cameras       # lists openable indices
python teleop/calibration/camera_identity.py                    # index -> hardware id + model label
```

Pick a stable `<name>` for each physical camera (e.g. `c0`, `c1`, `rs`) and note its current index. A RealSense exposes several `/dev/video*` nodes; only the **color** one is usable for tracking (`camera_identity.py` labels it; on Linux it is typically the highest-numbered readable node).

**1. Intrinsics.** Easiest is **auto-discovery** — one command finds every connected color camera, auto-names them (`c0`, `c1`, … and `rs` for a RealSense), and walks through calibrating each:

```bash
python teleop/calibration/charuco_calibration.py intrinsics-all --square-mm 49.6 --max-res
```

Per camera: **SPACE** captures a view (12+ at varied angle/tilt/distance, fill the edges), **C** solves, **S** skips, **Q** quits. It prints the `extrinsics-all` command to run next. Because each intrinsic is stamped with the camera's hardware id, the auto-assigned names are just labels — the *identity* is what binds them at launch (so enumeration order doesn't matter later).

Or calibrate cameras individually (independent of board placement; `--max-res` opens each at its highest mode):

```bash
python teleop/calibration/charuco_calibration.py intrinsics --camera 0 --name c0 --square-mm 49.6 --max-res
python teleop/calibration/charuco_calibration.py intrinsics --camera 2 --name c1 --square-mm 49.6 --max-res
```

Each run also **stamps the camera's hardware id** (USB `vendor:product:serial`, or RealSense SDK serial) into the intrinsics file — see *Camera identity* below.

**2. Extrinsics — all cameras in one command** (board fixed for the whole run). Fix the board at the world origin, then `--auto` discovers every calibrated camera and walks through each:

```bash
python teleop/calibration/charuco_calibration.py extrinsics-all --auto --square-mm 49.6
# or name them explicitly:
# ... extrinsics-all --cam c0:0 --cam c1:2 --cam rs:8 --square-mm 49.6
```

Per camera: **SPACE** solves and saves, **S** skips, **Q** stops the walkthrough. `--auto` uses the same discovery + hardware-id matching as `run_multicam --auto`, so it needs each camera's stamped intrinsics from step 1. (The single-camera `extrinsics --camera <idx> --name <name>` still works to redo just one.)

**3. Validate.** Launch the pipeline and watch the fusion node's health log — median reprojection error should be a few px; it warns above ~8 px, which flags a moved board, a resolution mismatch, or a bad intrinsic:

```bash
python teleop/run_multicam.py --auto --show-fused     # discover + match by id
# or name them: python teleop/run_multicam.py --cam c0 --cam c1 --show-fused
```

**4. Run teleop against the fused output.** Either launch `run_multicam.py` in one terminal and the teleop app in another, or let the app spawn the pipeline itself. `--multicam-auto` discovers and matches every calibrated camera by hardware id — fully hands-off:

```bash
# hands-off: discover + match all calibrated cameras
python kinova_leap_pick_place.py --mode dexpilot \
    --multicam-auto --skeleton-view --camera-views

# or name them explicitly
python kinova_leap_pick_place.py --mode dexpilot \
    --multicam c0 --multicam c1 --skeleton-view --camera-views
```

#### Camera identity (index-free launch)

Because intrinsics are a property of the lens+sensor — not the USB port — calibration stamps each camera's hardware id into its intrinsics file. At launch you can then discover everything automatically, or name cameras and omit indices:

```bash
python teleop/run_multicam.py --auto                   # discover + match all by id
python teleop/run_multicam.py --cam c0 --cam c1        # indices auto-resolved by id
```

`--auto` uses every calibrated camera currently plugged in (matched by id), flags any RealSense for SDK capture, and reports which are matched vs. calibrated-but-absent.

Passing an explicit index (`--cam c0:0`) still works and is *verified* against the stored id — a mismatch warns rather than silently applying the wrong intrinsic. Moving a camera to a different port never requires recalibration. (A RealSense still needs an explicit index — its color-stream index isn't derivable from the serial alone — but that index is verified against the serial.) If you ever have intrinsics that predate this stamping, add the id without recalibrating:

```bash
python teleop/calibration/charuco_calibration.py stamp-id --camera 0 --name c0
python teleop/calibration/charuco_calibration.py stamp-id --camera 8 --name rs --realsense
```

#### Intel RealSense

A RealSense participates as a plain RGB camera, but its color stream is captured through `pyrealsense2` (bundled in the env) rather than bare OpenCV. Pass `--multicam-realsense <name>` (in the app) or `--realsense <name>` (in `run_multicam.py`), and use `--realsense` on the calibration commands so they capture the same stream:

```bash
python teleop/calibration/charuco_calibration.py intrinsics --camera 8 --name rs --realsense --square-mm 49.6
python teleop/calibration/charuco_calibration.py extrinsics-all --cam c0:0 --cam c1:2 --cam rs:8 --realsense rs --square-mm 49.6
```

The D435I's 1080p color runs at only 8 fps, so RealSense capture defaults to 640×480 @ 30 fps — calibrate its intrinsics at the size you'll stream. (Factory SDK intrinsics report zero distortion; if fused landmarks warp near the frame edges, board-calibrate to recover the real coefficients.)

See [`teleop/MULTICAM.md`](teleop/MULTICAM.md) for the full multi-camera pipeline (per-camera resolution, the combined viewer, RealSense specifics, calibration validation, and the reprojection auto-reject safety net).

---

## MediaPipe publisher (standalone)

```bash
python teleop/ui.py               # auto-select camera
python teleop/ui.py --camera 1    # specify camera index
python teleop/ui.py --list-cameras
```

Publishes to `/hand/joint_angles` (`Float32MultiArray`, 183 floats):

| Slice | Content |
|---|---|
| `[0:3]` | Wrist position (board/world metres in absolute mode, or image x,y + monocular depth in legacy mode) |
| `[3:6]` | Wrist Euler angles (palm frame, ZYX degrees) |
| `[6:51]` | 15-joint Euler angles |
| `[51:57]` | 6 inter-segment flexion angles (degrees) |
| `[57:120]` | 21 MediaPipe **world** landmark positions — 63 floats, (x,y,z) per landmark in metres |
| `[120:183]` | 21 MediaPipe **image** landmark positions — 63 floats, (x,y,z) per landmark; normalised image coords (x,y ∈ [0,1], z = MediaPipe pseudo-depth). Drives the arm palm-frame normal (more stable near edge-on wrist poses than world-landmark depth) |

The multi-camera fusion node ([`teleop/hand_fusion_node.py`](teleop/hand_fusion_node.py)) publishes the **same 183-float message** — the triangulated 3D skeleton fills both the world- and image-landmark blocks — so all downstream teleop is identical whether the source is the single-camera publisher or the fused rig.

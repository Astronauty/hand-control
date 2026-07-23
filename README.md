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

**Multi-camera (fused) input.** Both teleop modes take their hand pose from `/hand/joint_angles`, which by default the built-in single-camera publisher supplies. Pass `--multicam` to instead auto-launch the [multi-camera pipeline](teleop/MULTICAM.md) (triangulated, occlusion-robust) as a child process — no separate terminal:

```bash
python kinova_leap_pick_place.py --mode dexpilot \
    --multicam c0:0 --multicam c1:2 --multicam rs:8 --multicam-realsense rs \
    --skeleton-view --camera-views
```

Per-camera resolution is read from `camera_intrinsics_<name>.json`, so a bare `NAME:INDEX` is enough. Add `--recalibrate-extrinsics` to re-solve each camera's board pose interactively before teleop starts. See [CLI flags](#cli-flags) and [`teleop/MULTICAM.md`](teleop/MULTICAM.md).

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
| `--camera N` | auto | *(teleop modes only)* Camera index forwarded to the built-in single-camera MediaPipe publisher. Defaults to auto-select (prefers external/USB camera at index ≥ 1). Run `python ui/mediapipe_joint_angles.py --list-cameras` to see available indices. |
| `--multicam NAME:INDEX[:WxH]` | none | *(teleop modes)* Auto-launch the [multi-camera pipeline](teleop/MULTICAM.md) (`teleop/run_multicam.py`) as a child process instead of the single-camera publisher, so `/hand/joint_angles` comes from the **fused** cameras. Repeat per camera (≥ 2), e.g. `--multicam c0:0 --multicam c1:2`. Implies `--no-mediapipe`. Resolution is read from each camera's `camera_intrinsics_<name>.json` when `:WxH` is omitted. |
| `--multicam-realsense NAME` | none | *(with `--multicam`)* Mark a camera as an Intel RealSense — its node captures the COLOR stream via `pyrealsense2` (the `:INDEX` is then ignored; the SDK picks the device). Default 640×480 @ 30 fps (the D435I's 1080p color is only 8 fps). Repeatable. |
| `--multicam-max-res` | off | *(with `--multicam`)* Open each camera at its highest supported resolution (forwards `--max-res`). Prefer omitting it — resolutions come from the intrinsics files automatically. |
| `--recalibrate-extrinsics` | off | *(with `--multicam`)* Run the interactive extrinsics solve for each camera **before** teleop starts (fix the ChArUco board at the world origin, press **SPACE** per camera). Reuses each camera's calibrated resolution. Default reuses the saved extrinsics. |
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

# multi-camera (fused) teleop — resolutions auto-read from intrinsics:
python kinova_leap_pick_place.py --mode dexpilot \
    --multicam c0:0 --multicam c1:2 --multicam rs:8 --multicam-realsense rs \
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

## Camera calibration (ChArUco board)

Calibration produces, per camera, the intrinsics (`camera_matrix`, `dist_coeffs`) and the extrinsic pose (`R_cam_world`, `t_cam_world`) that place the camera in a fixed **world frame** anchored to the board. Single-camera teleop uses these for metric wrist positioning; multi-camera triangulation uses them to fuse views (see [`teleop/MULTICAM.md`](teleop/MULTICAM.md)).

### Board setup

Generate a printable ChArUco board PNG (5×7 squares, 35 mm nominal, sized for letter paper):

```bash
python calibration/charuco_calibration.py generate
```

Writes `calibration/board.png` (~7.2 × 9.9 in at 300 DPI). Options: `--square-mm` (nominal square size, default 35), `--dpi` (default 300), `--out`.

1. Print at **100% / Actual size** — never "fit to page", which silently rescales and invalidates the metric calibration. The raw PNG has no DPI metadata, so viewers that assume 96 DPI size it wrong; if your viewer misbehaves, place the image on a letter-size 300 DPI canvas first or print from an application that lets you set the scale explicitly.
2. Glue or tape the print completely flat to something rigid (foam board, clipboard).
3. Measure one black square edge with calipers — printers drift a few percent — and pass the **measured** value as `--square-mm` to every step below.

### Single camera

```bash
python calibration/charuco_calibration.py intrinsics --camera 1 --square-mm 34.8
python calibration/charuco_calibration.py extrinsics --camera 1 --square-mm 34.8
```

`intrinsics`: wave the board across the frame; **SPACE** captures a view (collect 12+ at varied angle/tilt/distance), **C** calibrates. Aim for RMS < 1.0 px. `extrinsics`: fix the board at the desired world origin facing the camera; **SPACE** averages `--n-avg` frames and solves. Writes `calibration/camera_intrinsics.json` and `camera_extrinsics.json`.

### Multiple cameras (triangulation rig)

Setting up a rig on a new machine from scratch. Each camera gets its own named calibration files (`camera_intrinsics_<name>.json`, `camera_extrinsics_<name>.json`). Two rules make triangulation valid:

- **Same board pose for all extrinsics.** Every camera's extrinsic must be solved against the *same* fixed board placement — that shared pose is what puts all cameras in one world frame (no stereo calibration needed). Do not move the board between cameras.
- **Calibrate at the capture resolution.** Intrinsics are only valid at the resolution they were estimated at. Each camera's calibrated size is stored in its `camera_intrinsics_<name>.json` (`image_size`), which `run_multicam.py` and the `--multicam` launch **read back automatically** — so a camera streams at its calibrated resolution with no `:WxH` on the CLI. The fusion node hard-errors on a resolution mismatch.

**0. Find your camera indices.** OpenCV indices are assigned by USB enumeration and differ per machine/session:

```bash
python ui/mediapipe_joint_angles.py --list-cameras       # lists openable indices
python calibration/camera_identity.py                    # index -> hardware id + model label
```

Pick a stable `<name>` for each physical camera (e.g. `c0`, `c1`, `rs`) and note its current index. A RealSense exposes several `/dev/video*` nodes; only the **color** one is usable for tracking (`camera_identity.py` labels it; on Linux it is typically the highest-numbered readable node).

**1. Intrinsics.** Easiest is **auto-discovery** — one command finds every connected color camera, auto-names them (`c0`, `c1`, … and `rs` for a RealSense), and walks through calibrating each:

```bash
python calibration/charuco_calibration.py intrinsics-all --square-mm 34.8 --max-res
```

Per camera: **SPACE** captures a view (12+ at varied angle/tilt/distance, fill the edges), **C** solves, **S** skips, **Q** quits. It prints the `extrinsics-all` command to run next. Because each intrinsic is stamped with the camera's hardware id, the auto-assigned names are just labels — the *identity* is what binds them at launch (so enumeration order doesn't matter later).

Or calibrate cameras individually (independent of board placement; `--max-res` opens each at its highest mode):

```bash
python calibration/charuco_calibration.py intrinsics --camera 0 --name c0 --square-mm 34.8 --max-res
python calibration/charuco_calibration.py intrinsics --camera 2 --name c1 --square-mm 34.8 --max-res
```

Each run also **stamps the camera's hardware id** (USB `vendor:product:serial`, or RealSense SDK serial) into the intrinsics file — see *Camera identity* below.

**2. Extrinsics — all cameras in one command** (board fixed for the whole run). Fix the board at the world origin, then `--auto` discovers every calibrated camera and walks through each:

```bash
python calibration/charuco_calibration.py extrinsics-all --auto --square-mm 34.8
# or name them explicitly:
# ... extrinsics-all --cam c0:0 --cam c1:2 --cam rs:8 --square-mm 34.8
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
python calibration/charuco_calibration.py stamp-id --camera 0 --name c0
python calibration/charuco_calibration.py stamp-id --camera 8 --name rs --realsense
```

#### Intel RealSense

A RealSense participates as a plain RGB camera, but its color stream is captured through `pyrealsense2` (bundled in the env) rather than bare OpenCV. Pass `--multicam-realsense <name>` (in the app) or `--realsense <name>` (in `run_multicam.py`), and use `--realsense` on the calibration commands so they capture the same stream:

```bash
python calibration/charuco_calibration.py intrinsics --camera 8 --name rs --realsense --square-mm 34.8
python calibration/charuco_calibration.py extrinsics-all --cam c0:0 --cam c1:2 --cam rs:8 --realsense rs --square-mm 34.8
```

The D435I's 1080p color runs at only 8 fps, so RealSense capture defaults to 640×480 @ 30 fps — calibrate its intrinsics at the size you'll stream. (Factory SDK intrinsics report zero distortion; if fused landmarks warp near the frame edges, board-calibrate to recover the real coefficients.)

See [`teleop/MULTICAM.md`](teleop/MULTICAM.md) for the full multi-camera pipeline (per-camera resolution, the combined viewer, RealSense specifics, calibration validation, and the reprojection auto-reject safety net).

---

## MediaPipe publisher (standalone)

```bash
python ui/mediapipe_joint_angles.py               # auto-select camera
python ui/mediapipe_joint_angles.py --camera 1    # specify camera index
python ui/mediapipe_joint_angles.py --list-cameras
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

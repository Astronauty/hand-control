## Installation

Install and source the UV environment:
```bash
conda create -f environment.yml
conda activate hand-control
source .venv/bin/activate
```

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

#### Autonomous grasp (default)
RRT path planning + constrained IK + internal-force grasp controller. Keyboard-driven target selection.

```bash
python kinova_leap_pick_place.py
```

#### DexPilot teleoperation
Live MediaPipe hand retargeting via ROS 2. Launches the MediaPipe publisher and MuJoCo viewer together.

```bash
python kinova_leap_pick_place.py --mode dexpilot
```

Requires a sourced ROS 2 environment and `CYCLONEDDS_URI` set to an UP interface (see [ROS 2 setup](#ros-2-setup) below).

**Start tracking:** the robot holds its home pose until you press **`G`**. Hold your hand at a comfortable neutral orientation and press `G` — that instant is captured as home, so the robot's wrist orientation is treated as matching yours, and it then follows your movement and rotation relative to that pose. Press `G` again at any time to re-zero. `Q`/`Esc` quits.

---

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--mode {rrt,dexpilot}` | `rrt` | **rrt**: autonomous RRT+IK grasp. **dexpilot**: live MediaPipe retargeting teleop via ROS 2. |
| `--ik-solver {sqp,ipopt}` | `sqp` | IK solver backend (see below). |
| `--dashboard` | off | Launch a live pyqtgraph metrics dashboard (separate process): planning mode (Approach/Grasp/Transport), proximity-based active object, scrolling fingertip→object distances, net hand→object wrench, per-finger contact normal forces, and a combined RRT+IK planner solution log. |
| `--viz-only` | off | Debug mode: disables arm/hand collision physics and never calls `mj_step`. REACH and GRASP phases hold their IK solution kinematically so you can inspect the IK/RRT result without dynamics interference. |
| `--seed N` | none | RNG seed for object randomization — the same seed reproduces the same layout (positions and sizes). Default: fresh entropy every run. Ignored with `--no-randomize`. |
| `--no-randomize` | off | Skip object randomization entirely: objects keep the positions, sizes, and colors authored in `models/scene_pick_place.xml`. |
| `--camera N` | auto | *(dexpilot only)* Camera index forwarded to the MediaPipe publisher. Defaults to auto-select (prefers external/USB camera at index ≥ 1). Run `python ui/mediapipe_joint_angles.py --list-cameras` to see available indices. |

Flags can be combined, e.g.:
```bash
python kinova_leap_pick_place.py --mode dexpilot --camera 1
python kinova_leap_pick_place.py --viz-only --dashboard
python kinova_leap_pick_place.py --ik-solver ipopt
python kinova_leap_pick_place.py --seed 42          # reproducible object layout
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

### Autonomous mode controls

| Key | Action |
|---|---|
| `Ctrl+1` … `Ctrl+6` | Select grasp target (objects 1–6) |
| `Ctrl+0` | Return to home pose |
| `Enter` | Commit to GRASP / release back to pregrasp |
| `← →` | Jog object x-position while grasping |
| `↑ ↓` | Jog object z-position (height) while grasping |
| `K` | Cycle IK config visualisation (pregrasp → grasp → off) |
| `B` | Toggle IK collision bounding-sphere overlay |
| `Q` / `Esc` | Quit |

---

### ROS 2 setup

Both the mediapipe publisher and the MuJoCo simulation run on the same machine, so any UP network interface works for CycloneDDS discovery. The default `lo` (loopback) transport is unreliable with CycloneDDS multicast; use a physical interface instead:

```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General></Domain></CycloneDDS>'
```

Replace `eno1` with your UP interface (`ip link show` to list; pick one with `state UP` and a carrier). Add the export to `~/.bashrc` to make it permanent.

Source ROS 2 before running:
```bash
source /opt/ros/humble/setup.bash
```

---

## 2D Environment

```bash
python internal_force_control.py
```

---

## MediaPipe publisher (standalone)

```bash
python ui/mediapipe_joint_angles.py               # auto-select camera
python ui/mediapipe_joint_angles.py --camera 1    # specify camera index
python ui/mediapipe_joint_angles.py --list-cameras
```

Publishes to `/hand/joint_angles` (`Float32MultiArray`, 120 floats):

| Slice | Content |
|---|---|
| `[0:3]` | Wrist position (camera space) |
| `[3:6]` | Wrist Euler angles |
| `[6:51]` | 15-joint Euler angles |
| `[51:57]` | 6 inter-segment flexion angles (degrees) |
| `[57:120]` | 21 MediaPipe world landmark positions — 63 floats, (x,y,z) per landmark in metres |

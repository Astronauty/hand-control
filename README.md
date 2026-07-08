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
| `--dashboard` | off | Launch a live pyqtgraph metrics dashboard (separate process): scrolling fingertip→object distances, RRT solve times, and IPOPT iteration counts. |
| `--viz-only` | off | Debug mode: disables arm/hand collision physics and never calls `mj_step`. REACH and GRASP phases hold their IK solution kinematically so you can inspect the IK/RRT result without dynamics interference. |
| `--camera N` | auto | *(dexpilot only)* Camera index forwarded to the MediaPipe publisher. Defaults to auto-select (prefers external/USB camera at index ≥ 1). Run `python ui/mediapipe_joint_angles.py --list-cameras` to see available indices. |

Flags can be combined, e.g.:
```bash
python kinova_leap_pick_place.py --mode dexpilot --camera 1
python kinova_leap_pick_place.py --viz-only --dashboard
```

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

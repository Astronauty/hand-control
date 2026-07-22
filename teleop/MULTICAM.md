# Multi-camera hand tracking

Triangulate MediaPipe hand landmarks from **N cameras** (configurable, ≥2) into
one metric 3D skeleton in a shared world frame, and publish the **same
183-float `/hand/joint_angles` message** the single-camera publisher emits — so
DexPilot / contact-aware teleop run against it **unchanged**.

This fixes occlusion-driven failures: a pinch hidden in one camera's view no
longer breaks grasp detection or the palm-normal orientation, as long as ≥2
cameras see the relevant landmarks.

## Pipeline

```
cam c0 → hand_landmark_node → /hand/cam_c0/landmarks ─┐
cam c1 → hand_landmark_node → /hand/cam_c1/landmarks ─┼→ hand_fusion_node → /hand/joint_angles [183]
cam rs → hand_landmark_node → /hand/cam_rs/landmarks ─┘   triangulate 21 LM      (unchanged contract)
```

Files:
- `triangulation.py` — pure math (DLT / weighted SVD), no ROS. `test_triangulation.py` covers it.
- `hand_message.py` — shared 183-float message builders (Euler / flexion / wrist).
- `hand_landmark_node.py` — one per camera: capture + MediaPipe → raw 2D landmarks.
- `hand_fusion_node.py` — subscribe all cameras, triangulate, publish the 183-float msg.
- `run_multicam.py` — process supervisor: launches all nodes with one command.

## 1. Calibrate each camera (the gate)

Every camera needs its own intrinsics **and** an extrinsic solved against the
**same fixed board at the same world origin** — that shared board is what puts
all cameras in one common world frame (no stereo calibration needed).

For each camera `<name>` at OpenCV index `<idx>`:

Cameras may run at **different resolutions** — triangulation uses each camera's
own intrinsics, so there is no shared-resolution requirement. Use `--max-res` to
calibrate each camera at its highest supported mode (best landmark precision):

**Intrinsics** are per-camera (lens/sensor), so run them one at a time:

```bash
# wave the board around, at THIS camera's max resolution
python calibration/charuco_calibration.py intrinsics --camera <idx> --name <name> \
    --square-mm <measured> --max-res
```

**Extrinsics for ALL cameras in one command** — fix the board at the world
origin and walk through every camera without restarting:

```bash
python calibration/charuco_calibration.py extrinsics-all \
    --cam c0:0 --cam c1:2 --cam rs:4 --square-mm <measured> --max-res
```

It opens each camera in turn (`[k/N] <name>` in the title). Per camera:
**SPACE** = average & solve + save, **S** = skip, **Q** = stop the walkthrough.
Keep the board FIXED the whole run — that shared pose is what puts every camera
in one world frame. (The single-camera `extrinsics --camera <idx> --name <name>`
still exists if you want to redo just one.)

The tool prints the resolution it opened at (`[camera] index N opened at WxH`).
Use a fixed `--width/--height` instead of `--max-res` if you want a specific size.

> **The one hard rule: each camera must CAPTURE at the resolution its intrinsics
> were CALIBRATED at.** Intrinsics (`fx`, `cx`, …) are only valid at their
> calibration resolution — calibrate at 1280×720 but stream at 640×480 and every
> ray is mis-scaled 2×, giving a steady ~15–20 px triangulation error that still
> looks fine per-camera. The fusion node **hard-errors** on a size mismatch, and
> `--max-res` on both the calibration tool and `run_multicam` keeps them aligned.

This writes `calibration/camera_intrinsics_<name>.json` and
`camera_extrinsics_<name>.json`. Do **not** move the board between cameras'
extrinsics steps — they must all reference the same physical pose.

> Accuracy here dominates everything. Validate two ways:
> - **Per camera (by eye):**
>   `python calibration/assess_extrinsics.py --camera <idx> --name <name> --square-mm <measured>`
>   — checks the world axes are stable, scale is right, and jitter is low. Press
>   `S` to (re)save that camera's extrinsic. Do NOT move the board between cameras.
> - **Cross-camera (the real check):** launch the pipeline (step 2) and watch the
>   fusion node's median reprojection-error log — it warns above ~8 px. This is
>   the true shared-frame validation.

## 2. Launch the pipeline

```bash
# every camera at its highest resolution (match how you calibrated)
python teleop/run_multicam.py --cam c0:0 --cam c1:2 --max-res

# explicit per-camera resolution: name:index:WxH (must match that cam's intrinsics)
python teleop/run_multicam.py --cam c0:0:1920x1080 --cam c1:2:1280x960

# fixed same size for all, with per-camera debug windows
python teleop/run_multicam.py --cam c0:0 --cam c1:2 --width 640 --height 480 --show
```

Resolution precedence per camera: explicit `:WxH` → `--max-res` → `--width/--height`.
`--list-cameras` on the old publisher (`python ui/mediapipe_joint_angles.py
--list-cameras`) still lists OpenCV indices.

### Visualization

```bash
# per-camera windows: hand skeleton + the shared world axes for THAT camera
python teleop/run_multicam.py --cam c0:0 --cam c1:2 --max-res --show

# a 3D viewer of the FUSED skeleton on a black window (orbit with the mouse)
python teleop/run_multicam.py --cam c0:0 --cam c1:2 --max-res --show-fused

# both
python teleop/run_multicam.py --cam c0:0 --cam c1:2 --max-res --show --show-fused
```

- `--show-fused` opens ONE combined window:
  - **Left panel** — the **triangulated 3D** skeleton with an orbit camera (drag
    to rotate, `z` flip up-axis, `r` reset, `q` close). World axes anchor the
    origin; dashed X/Y/Z guides show the wrist position with distance labels;
    fingertips + wrist accented. Each **physical camera** is drawn as a marker +
    optical-axis stub at its calibrated world position, so you can see the rig
    geometry and confirm the extrinsics place cameras sensibly.
  - **Right column** — a live **video preview** from each camera (small JPEG
    thumbnails on a separate throttled topic, ~10 Hz) with the detected skeleton
    and world axes overlaid. Lightweight: it does NOT ship full-res frames, so it
    doesn't affect tracking throughput.
- `--show` (optional) additionally opens the old separate per-camera windows for
  close-up debugging. Not needed for normal use — everything is in the combined
  `--show-fused` window.

The RealSense participates as a **plain RGB camera** in v1 (its depth is ignored
for now — a phase-2 add-on). Give it any name, e.g. `rs`, calibrate it like the
others, and pass `--cam rs:<idx>`.

## 3. Run teleop as usual

The fusion node publishes `/hand/joint_angles`, so your existing teleop is
untouched:

```bash
python kinova_leap_pick_place.py        # or whatever launches DexPilotController
```

Do **not** also run `ui/mediapipe_joint_angles.py` at the same time — both
publish to the same topic. Use one publisher OR the multi-camera pipeline.

## Tuning

`run_multicam.py` / `hand_fusion_node.py` flags:
- `--sync-window` (default 0.033 s) — max staleness for a camera frame to join a
  fused solve. Raise if cameras run slower than 30 fps.
- `--vis-thresh` (default 0.3) — per-view visibility below which a landmark's
  view is dropped from the fit (occlusion rejection).
- `--min-views` (default 2) — minimum simultaneous views to accept a landmark.

## Notes / limitations (v1)

- **World & image landmark blocks are both the triangulated metric skeleton**
  (wrist-relative). This is deliberate: the whole point of triangulating is a
  more stable frame than any single view's image landmarks. The downstream
  retargeter/arm-controller consume these exactly as before.
- **Un-triangulated landmarks** (seen by <2 cameras this frame) hold their last
  good value so the 21×3 block never goes NaN. Persistent gaps mean a camera is
  down or mis-calibrated — check the reprojection log.
- **RealSense depth** is not yet used. Adding it as a per-landmark prior is the
  natural next step for landmarks only one camera can see.
```

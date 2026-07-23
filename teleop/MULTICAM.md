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

### RealSense cameras (D435I etc.)

A RealSense is a **special case for both rules above**: the pipeline captures its
COLOR stream through **pyrealsense2** (not cv2/V4L2), and the D435I color at
640×480 is what `--multicam-realsense`/`run_multicam --realsense` opens by
default. So its calibration must use the *same* capture path and size, or you get
exactly the "hundreds of px, unfixable by re-doing extrinsics" failure.

1. **Intrinsics.** Two options:

   **(a) SDK factory values (fast, no session):**
   ```bash
   python calibration/realsense_intrinsics.py --name rs           # 640x480 (pipeline default)
   ```
   Queries pyrealsense2 for the color stream's intrinsics at the exact pipeline
   size and writes `camera_intrinsics_rs.json`. **Caveat:** the D435I's factory
   color intrinsics report **zero distortion**. If fused landmarks drift/warp
   near the FRAME EDGES, that's real uncorrected radial lens distortion the zeros
   don't model — board-calibrate instead:

   **(b) Board calibration (recovers real k1,k2,p1,p2,k3):**
   ```bash
   python calibration/charuco_calibration.py intrinsics \
       --camera 8 --name rs --realsense --square-mm <measured>
   # --realsense defaults to 640x480 and captures via pyrealsense2 (same stream
   # as the pipeline). Fill the frame EDGES/CORNERS with the board across views —
   # that's what pins down the edge distortion. C = calibrate, aim for RMS < 0.5 px.
   ```

2. **Extrinsics — capture the RealSense via pyrealsense2 too**, by naming it in
   `--realsense`. RealSense cameras then use `--rs-width/--rs-height` (default
   640×480), while webcams keep `--width/--height`:
   ```bash
   python calibration/charuco_calibration.py extrinsics-all \
       --cam c0:0 --cam c1:2 --cam rs:8 --realsense rs \
       --square-mm <measured>
   ```
   (The single-camera form takes a bare `--realsense` flag:
   `extrinsics --camera 8 --name rs --realsense --width 640 --height 480`.)

> Changing intrinsics **invalidates the old extrinsics** (they were solved on top
> of the old K) — always re-run extrinsics for that camera after re-doing its
> intrinsics.

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
>
> **Auto-reject of a bad view:** the fusion node scores each participating camera
> by its own median reprojection error against the triangulation and, if the
> worst breaches `--reproj-gate` (default 40 px) while enough views remain to
> keep `--min-views`, drops that camera and re-triangulates without it. So a 3rd
> camera whose extrinsics landed in the wrong board frame no longer drags the
> fused skeleton off — the health log shows `reject[<name>:N]` and the reported
> reproj drops back to the clean two-view value. Set `--reproj-gate 0` to disable
> (e.g. while diagnosing the calibration itself). This is a safety net, not a
> fix: re-solve that camera's extrinsics so it can rejoin the solve.

## 2. Launch the pipeline

```bash
# resolutions auto-read from each camera's intrinsics — no :WxH needed
python teleop/run_multicam.py --cam c0:0 --cam c1:2

# explicit per-camera resolution: name:index:WxH (overrides the intrinsics size)
python teleop/run_multicam.py --cam c0:0:1920x1080 --cam c1:2:1280x960

# force each camera to its highest mode, with per-camera debug windows
python teleop/run_multicam.py --cam c0:0 --cam c1:2 --max-res --show
```

Resolution precedence per camera:
**explicit `:WxH` → `camera_intrinsics_<name>.json` `image_size` → `--max-res` →
`--width/--height`.** Because the calibrated size is stored in the intrinsics file,
a bare `NAME:INDEX` streams at exactly the resolution it was calibrated at — no
`:WxH` on the CLI, and no separate rig-config file to keep in sync.
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

### Intel RealSense

The RealSense participates as an **RGB camera** (its depth is not used yet — a
phase-2 add-on). Its color stream can't be opened reliably at higher resolutions
through bare OpenCV/V4L2, so it's captured via **`pyrealsense2`** instead. Mark it
with `--realsense <name>`:

```bash
python teleop/run_multicam.py --cam c0:0 --cam c1:2 --cam rs:8 --realsense rs
```

- The `:INDEX` for a `--realsense` camera is **ignored** — the SDK selects the
  device (single-RealSense rigs need nothing more).
- The D435I's 1080p color runs at only **8 fps** (too slow for teleop), so a
  RealSense defaults to **640×480 @ 30 fps** (matching the webcams' framerate for
  clean fusion sync). Calibrate its intrinsics at the size you'll stream.
- Note the `:INDEX` still matters for the **calibration tool** (which uses OpenCV):
  point `intrinsics`/`extrinsics --camera <idx>` at the RealSense **color** V4L2
  node (find it with `v4l2-ctl --list-devices`; it's the one that reads a normal
  3-channel frame, not depth/IR).

## 3. Run teleop as usual

The fusion node publishes `/hand/joint_angles`, so your existing teleop is
untouched. Two ways to wire it up:

**A. Let the teleop app launch the pipeline** (`--multicam`, recommended — one
command, no separate terminal):

```bash
python kinova_leap_pick_place.py --mode dexpilot \
    --multicam c0:0 --multicam c1:2 --multicam rs:8 --multicam-realsense rs \
    --skeleton-view --camera-views
```

Resolutions come from the intrinsics files. `--multicam` implies `--no-mediapipe`
(so the single-cam publisher isn't also started). Add `--recalibrate-extrinsics`
to re-solve each camera's board pose interactively before teleop starts.

**B. Run the pipeline separately**, then the teleop app pointed at it:

```bash
python teleop/run_multicam.py --cam c0:0 --cam c1:2      # terminal 1
python kinova_leap_pick_place.py --mode dexpilot --no-mediapipe   # terminal 2
```

Do **not** also run `ui/mediapipe_joint_angles.py` (or omit `--no-mediapipe` in
option B) at the same time — two publishers on `/hand/joint_angles` interleave
poses. Use exactly one hand-pose source.

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

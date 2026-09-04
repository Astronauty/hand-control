# AnyTeleop retargeting backend

A **separable** finger-retargeting backend that wraps the published
[`dex-retargeting`](https://github.com/dexsuite/dex-retargeting) library — the
retargeting core of [AnyTeleop](https://yzqin.github.io/anyteleop/) (Qin et al.) — so it
can be compared head-to-head against this repo's hand-rolled DexPilot retargeter on the
same LEAP hand.

Nothing here is imported unless an AnyTeleop run mode is selected, so the existing
DexPilot / contact-aware code paths are untouched (and still run) when this package or its
optional dependency is absent.

## The comparison (2×2 of pipeline × finger retargeter)

| `--mode` | Pipeline | Finger retargeter |
|---|---|---|
| `dexpilot` | plain teleop | our hand-rolled DexPilot |
| `anyteleop` | plain teleop | dex-retargeting |
| `contact_aware_w_dexpilot` | Ours (NLP recommender + lock-in) | our hand-rolled DexPilot |
| `contact_aware_w_anyteleop` | Ours | dex-retargeting |

`contact_aware_w_dexpilot` is identical to the legacy `contact_aware_teleop`; the two
`*_anyteleop` modes are the new conditions. The four names normalize inside
`kinova_leap_pick_place.py` (beside the `rrt` alias) to a canonical pipeline mode + a
retargeter selector, so no downstream code changed.

## Install

```bash
uv sync --extra anyteleop
```

This pulls `dex-retargeting==0.4.6` plus a **CPU** build of torch (the PyTorch CPU index
is scoped to torch in `pyproject.toml`, so it never drags in the ~5GB CUDA stack and never
touches the base environment). 0.4.6 is pinned because 0.5.0+ needs `numpy>=2` (this repo
pins `numpy<2` for mediapipe); per the dex-retargeting changelog, 0.5.0 is packaging-only,
so 0.4.6's retargeting is algorithmically identical.

To remove: `uv sync` (without `--extra anyteleop`) prunes it.

## Configuration

`calibration/anyteleop_config.json` (hot-reloaded each frame):

| field | values | meaning |
|---|---|---|
| `retargeting_type` | `"vector"` \| `"dexpilot"` | which dex-retargeting optimizer. `vector` is what AnyTeleop uses in its live video demos; `dexpilot` is its DexPilot-style pinch-aware optimizer. |
| `hand_type` | `"right"` \| `"left"` | operator hand (the sim LEAP is a right hand). |
| `output_alpha` | `0<a<=1` | extra EMA our wrapper applies on top of dex-retargeting's own `low_pass_alpha`. `1.0` = none. |
| `scaling_factor` | number \| `null` | dex-retargeting's robot-to-human hand-size gain (higher = more finger curl/reach). `null` keeps the vendored YAML value (`1.6`). |
| `low_pass_alpha` | `0..1` \| `null` | dex-retargeting's internal low-pass (smaller = smoother/laggier; `<0` disables). `null` keeps the YAML value (`0.2`). |

`scaling_factor` / `low_pass_alpha` override the vendored `leap_hand_right*.yml` so you tune
them here instead of editing the asset files; changing either rebuilds the solver on
hot-reload. **Pinch detection** (`EPS`, enter/exit fracs, median N) is *not* here — the
backend reuses the shared DexPilot debounce from `calibration/retarget_config.json`, so
[`ui/hand_tune.py`](../ui/hand_tune.py) tunes pinch identically for AnyTeleop and DexPilot.

## How it works

`anyteleop/`
- **`retargeter.py`** — `AnyTeleopRetargeter`, the drop-in backend. Same public surface as
  `DexPilotRetargeter` (`retarget(world_lm,…) -> (16,)`, `human_palm_frame_robot_aligned`,
  `poll_config`, `reset`, `last_d_s1[_filt]`). The finger *solve* goes to dex-retargeting;
  everything geometric/optimizer-independent (the robot-aligned palm frame for the arm, the
  pinch-state debounce for the trial logger) is **delegated to an internal
  `DexPilotRetargeter`**, so those behaviors stay identical across backends and no code is
  copied out of `teleop/`.
- **`landmarks.py`** — converts our MediaPipe world landmarks (message `raw[57:120]`) into
  dex-retargeting's MANO-convention keypoints, ported verbatim from its
  `SingleHandDetector` (and reusing its `OPERATOR2MANO` constant) so the input basis
  matches the library exactly.
- **`joint_remap.py`** — dex-retargeting returns 16 qpos in its URDF's numeric joint order;
  this remaps them onto the sim's `_HAND_JOINT_NAMES` order. The permutation is *derived
  empirically* at construction (drive each URDF DOF, match by which fingertip moves, order
  within a finger by root→tip tree order) and cross-checked against a verified table.
- **`config.py`** — permissive JSON loader (missing file/keys → defaults; bad JSON ignored),
  matching the DexPilot config pattern.
- **`factory.py`** — `make_retargeter(name, model, …)`; lazy-imports this backend only when
  `name == "anyteleop"`.
- **`assets/leap_hand/`** — vendored dexsuite LEAP URDF + meshes (the wheel ships the
  retargeting YAML but not the URDF). This is the *retargeter's* kinematic model; the sim
  itself is still driven by `models/kinova_leap.xml` (the same physical hand).

## Headless retargeting comparison

`anyteleop/compare_retargeting.py` compares DexPilot vs AnyTeleop(vector) vs
AnyTeleop(dexpilot) with **no camera, ROS, or objects** — retargeting is a pure
`landmarks → 16 joints` function, so it runs fully headless. It reports per-backend solve
latency, joint-limit saturation, robot-vs-human fingertip-direction error (reusing
`DexPilotRetargeter._human_vectors/_robot_vectors` as the tracking definition), pairwise
joint-trajectory RMS divergence, and pinch-decision agreement.

```bash
# designed synthetic pose sweep (open / graded pinches / fist / per-finger curls)
python3 anyteleop/compare_retargeting.py

# replay REAL recorded hand motion (from ui/hand_quality.py's hq_*.npz) — the absolute
# tracking metric is only meaningful on real poses; synthetic poses are for relative
# (latency / divergence / saturation) comparison.
python3 anyteleop/compare_retargeting.py --replay logs/hand_quality/hq_YYYYMMDD_HHMMSS.npz
```

**Task (pick/place) performance is NOT headless-testable** — it needs live hand input
driving a grasp. Use the `./start_teleop.sh sim <mode> --trial-log` sweep on the
workstation for that, then `parse_trials_tables.py`.

## The one seam into existing code

`teleop/dexpilot_controller.py` gained a `retargeter="dexpilot"` kwarg; when it's
`"anyteleop"` it calls `make_retargeter` instead of constructing `DexPilotRetargeter`
directly. The off-thread finger worker, arm controller, EMA smoothing, and PD-torque
application are all unchanged — the backend just has to honor `retarget() -> (16,)`.

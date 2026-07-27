#!/usr/bin/env python3
"""Kinematic replay of a logged contact-aware / dexpilot run.

Ingests a run directory written by kinova_leap_pick_place.py --trial-log:
    logs/<run>/pose_trace.npz   — one row per ~50 Hz sample, ALL phases, full state
    logs/<run>/events.jsonl     — state transitions (optional; used for an overlay)

"Kinematic" replay = for each recorded row we overwrite data.qpos with the logged
full state and call mj_forward (no mj_step, no controllers, no operator input). qpos
in the trace is the COMPLETE model state (nq = N_ROBOT robot DOF + object freejoints)
recorded against the unmodified models/scene_pick_place.xml — the run is always
--no-randomize (see kinova_leap_pick_place.py:511), so the scene needs no seed to
reconstruct. This reproduces exactly what the arm/hand/object DID, frame for frame;
it does NOT re-simulate physics or re-run the operator's teleop decisions.

Usage:
    python3 replay_pose_trace.py logs/<run>            # real-time in the MuJoCo viewer
    python3 replay_pose_trace.py logs/<run> --speed 2  # 2x
    python3 replay_pose_trace.py logs/<run> --no-viewer --dump frames/  # headless PNGs
    python3 replay_pose_trace.py logs/<run>/pose_trace.npz   # point straight at the npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mujoco as mj
except ImportError:
    sys.exit("mujoco not importable — run in the same env as kinova_leap_pick_place.py")

SCENE_XML = 'models/scene_pick_place.xml'   # fixed layout; runs are always --no-randomize


def _resolve_paths(target: Path):
    """Accept either a run directory or a direct .npz path; return (npz, events_or_None,
    run_label). SCENE_XML is resolved relative to this script's directory so the replay
    works from any CWD."""
    if target.is_dir():
        npz = target / 'pose_trace.npz'
        events = target / 'events.jsonl'
        label = target.name
    else:
        npz = target
        events = target.parent / 'events.jsonl'
        label = target.parent.name
    if not npz.exists():
        sys.exit(f"no pose_trace.npz found at {npz}")
    return npz, (events if events.exists() else None), label


def _logged_object_names(events_path: Path | None) -> list[str] | None:
    """The distinct target-object body names recorded across this run's trial_start events,
    in first-seen order. Returns None if there is no events.jsonl or no trial_start carries
    an 'object' field (older logs) — the caller then falls back to the full scene.

    This is what lets replay rebuild the SAME pruned model the run used: kinova_leap_pick_place
    spawns a single --object body by DELETING the other sweep objects from the MjSpec before
    compile, so the recorded qpos width matches only that pruned model, not the full scene XML."""
    if events_path is None:
        return None
    names: list[str] = []
    try:
        with open(events_path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('event') == 'trial_start':
                    obj = row.get('object')
                    if obj and obj not in names:
                        names.append(obj)
    except OSError:
        return None
    return names or None


# Object bodies subject to the --object spawn prune, mirroring the loader in
# kinova_leap_pick_place.py (baseline obj_red_box + the obj_box_* sweep siblings). Any
# body NOT in this family is always kept; a swept body is kept only if the log used it.
def _is_sweep_body(name: str) -> bool:
    return name == 'obj_red_box' or name.startswith('obj_box_')


def _build_replay_model(scene_path: Path, keep_objects: list[str] | None):
    """Rebuild the model the run was logged against. If keep_objects is a non-empty list of
    sweep-object body names, load the scene as an MjSpec and delete every OTHER sweep body
    (plus the single-object test keyframes, which otherwise corrupt the registry on a body
    delete) before compiling — reproducing the run's pruned nq. Otherwise (no object info),
    compile the scene unchanged. Returns the compiled MjModel."""
    if not keep_objects:
        return mj.MjModel.from_xml_path(str(scene_path))
    spec = mj.MjSpec.from_file(str(scene_path))
    keep = set(keep_objects)
    to_delete = [b for b in spec.bodies if _is_sweep_body(b.name) and b.name not in keep]
    if to_delete:
        for k in list(spec.keys):     # keyframes are single-object; drop before body deletes
            spec.delete(k)
        for b in to_delete:
            spec.delete(b)
    return spec.compile()


def _load_trace(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    trace = {k: d[k] for k in d.files}
    if 'qpos' not in trace or 't' not in trace:
        sys.exit(f"{npz_path} missing required 'qpos'/'t' arrays (keys: {sorted(trace)})")
    return trace


def _load_events(events_path: Path | None, t_lo: float, t_hi: float) -> list[dict]:
    """Return events whose `t` is a SIM-time within [t_lo, t_hi]. Some rows (e.g. 'solve')
    log a wall-clock t (~1.7e9) instead of sim-time; those fall outside the sim-time span
    and are dropped from the replay overlay."""
    if events_path is None:
        return []
    out = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get('t')
        if isinstance(t, (int, float)) and t_lo - 1e-6 <= t <= t_hi + 1e-6:
            out.append(ev)
    out.sort(key=lambda e: e['t'])
    return out


def _event_label(ev: dict) -> str:
    name = ev.get('event', '?')
    if name == 'phase_enter':
        sub = ev.get('approach_sub')
        return f"→ {ev.get('phase')}" + (f"/{sub}" if sub else "")
    if name == 'attempt_start':
        return f"attempt #{ev.get('attempt')} start"
    if name == 'attempt_result':
        return f"attempt #{ev.get('attempt')} {ev.get('outcome')}"
    if name == 'pick_confirmed':
        return f"pick confirmed (dwell {ev.get('dwell_s')}s)"
    if name == 'drop':
        return f"DROP (#{ev.get('count')})"
    if name == 'arrival':
        return f"arrival (xy {ev.get('xy_offset_m')}m)"
    if name == 'contact_violation':
        return f"inadvertent contact (#{ev.get('count')})"
    if name == 'trial_end':
        return f"trial_end: {ev.get('outcome')} ({ev.get('duration_s')}s)"
    if name == 'trial_start':
        return f"trial_start: {ev.get('object')}"
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path,
                    help="run directory (logs/<run>) or a pose_trace.npz path")
    ap.add_argument('--speed', type=float, default=1.0,
                    help="playback rate multiplier (2 = twice real-time; 0 = as fast as "
                         "possible). Scales whichever --pace clock is active.")
    ap.add_argument('--pace', choices=['sim', 'index'], default='index',
                    help="pacing clock. 'index' (default): one row per fixed wall-clock "
                         "interval (see --fps), so kinematic phases like the RRT reach — "
                         "whose sim-time stamps are COMPRESSED because kinematic replay "
                         "never advances data.time — play out visibly instead of jumping. "
                         "'sim': pace off the trace's own sim-time `t` (true physics rate, "
                         "but kinematic phases whoosh past).")
    ap.add_argument('--fps', type=float, default=50.0,
                    help="rows per second in --pace index (default 50, matching the ~50 Hz "
                         "trace rate). Scaled by --speed. Ignored in --pace sim.")
    ap.add_argument('--loop', action='store_true', help="restart from the top at the end")
    ap.add_argument('--start', type=float, default=0.0,
                    help="skip to this sim-time (s) before playing")
    ap.add_argument('--phase', default=None,
                    help="only replay rows whose standardized phase == this "
                         "(APPROACH/PICK/TRANSPORT/PLACE)")
    ap.add_argument('--no-viewer', action='store_true',
                    help="headless: don't open the interactive viewer")
    ap.add_argument('--dump', type=Path, default=None, metavar='DIR',
                    help="headless render: write one PNG per row to DIR (implies "
                         "--no-viewer). Requires the offscreen GL backend.")
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    scene_path = script_dir / SCENE_XML
    if not scene_path.exists():
        sys.exit(f"scene XML not found at {scene_path} — run from the repo, or the layout "
                 f"moved")

    npz_path, events_path, label = _resolve_paths(args.run)
    trace = _load_trace(npz_path)
    n_rows = len(trace['t'])

    # Rebuild the model the run was logged against. The run may have spawned a single
    # --object body (the others deleted from the MjSpec before compile), so the raw scene
    # XML (all sweep objects present) has a larger nq than the recorded qpos. Read the
    # object name(s) from events.jsonl and reproduce the same prune.
    keep_objects = _logged_object_names(events_path)
    model = _build_replay_model(scene_path, keep_objects)
    data = mj.MjData(model)

    qpos = np.asarray(trace['qpos'])
    if qpos.shape[1] != model.nq:
        kept = ', '.join(keep_objects) if keep_objects else '(none logged — full scene)'
        sys.exit(f"qpos width {qpos.shape[1]} != model.nq {model.nq} after rebuilding the "
                 f"model for logged object(s) [{kept}]. The scene XML's robot/object layout "
                 f"has changed since this run was logged (e.g. a body added/removed, or "
                 f"geometry edited), so replay would be misaligned. Check out the "
                 f"scene_pick_place.xml revision from the run's date to replay it.")

    t_sim = np.asarray(trace['t'], dtype=float)
    phase_arr = trace.get('phase')
    phase_arr = np.asarray(phase_arr).astype(str) if phase_arr is not None else None

    # Row selection: --start skips early sim-time, --phase filters to one phase.
    keep = t_sim >= args.start
    if args.phase is not None:
        if phase_arr is None:
            sys.exit("--phase given but the trace has no 'phase' array")
        keep &= (phase_arr == args.phase)
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        sys.exit("no rows match the --start/--phase filters")

    events = _load_events(events_path, float(t_sim[0]), float(t_sim[-1]))
    if args.pace == 'index':
        _pace_desc = f"index-paced {args.fps * args.speed:.0f} rows/s"
    else:
        _pace_desc = f"sim-time @ {args.speed}x"
    print(f"[replay] {label}: {n_rows} rows, sim-time {t_sim[0]:.2f}→{t_sim[-1]:.2f}s, "
          f"{len(events)} sim-time events, playing {idx.size} rows ({_pace_desc})")

    def apply_row(i: int):
        """Write the logged full state and forward-kinematics it (no integration)."""
        data.qpos[:] = qpos[i]
        # qvel/act, when present, make the forward pass reproduce the recorded contact
        # forces/sensors too — harmless for pure kinematics, useful if you read them back.
        if 'qvel' in trace:
            data.qvel[:] = trace['qvel'][i]
        if 'act' in trace and model.na:
            data.act[:] = trace['act'][i]
        data.time = float(t_sim[i])
        mj.mj_forward(model, data)

    # Event overlay: print each event once, when playback first passes its sim-time.
    # Seed the pointer past any events before the first played row so a --start/--phase
    # jump doesn't dump the earlier timeline. reset_events() re-seeds it for --loop.
    first_t = float(t_sim[idx[0]])
    def reset_events():
        p = 0
        while p < len(events) and events[p]['t'] < first_t - 1e-9:
            p += 1
        ev_ptr[0] = p
    ev_ptr = [0]
    reset_events()
    def flush_events_upto(t_now: float):
        while ev_ptr[0] < len(events) and events[ev_ptr[0]]['t'] <= t_now + 1e-9:
            ev = events[ev_ptr[0]]
            print(f"    [{ev['t']:7.3f}s] {_event_label(ev)}")
            ev_ptr[0] += 1

    headless = args.no_viewer or args.dump is not None

    if args.dump is not None:
        _dump_frames(model, data, apply_row, idx, t_sim, args, flush_events_upto)
        return

    if headless:
        # No viewer, no render: just walk the rows (useful to sanity-check/emit events).
        for i in idx:
            apply_row(int(i))
            flush_events_upto(float(t_sim[i]))
        print("[replay] headless walk complete (no rendering; use --dump for PNGs)")
        return

    _play_in_viewer(model, data, apply_row, idx, t_sim, args, flush_events_upto,
                    reset_events)


def _play_in_viewer(model, data, apply_row, idx, t_sim, args, flush_events_upto,
                    reset_events):
    import mujoco.viewer as viewer
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            wall0 = time.time()
            sim0 = float(t_sim[idx[0]])
            for n, i in enumerate(idx):
                if not v.is_running():
                    break
                i = int(i)
                apply_row(i)
                flush_events_upto(float(t_sim[i]))
                v.sync()
                # Pace to a target wall-clock time for this row, scaled by --speed.
                #   index: n / (fps*speed) — constant rate, so compressed-sim-time
                #          kinematic phases (RRT reach) play out at a visible pace.
                #   sim:   trace sim-time delta / speed — true physics rate.
                if args.speed > 0:
                    if args.pace == 'index':
                        target_wall = wall0 + n / (args.fps * args.speed)
                    else:
                        target_wall = wall0 + (float(t_sim[i]) - sim0) / args.speed
                    dt = target_wall - time.time()
                    if dt > 0:
                        time.sleep(dt)
            if not args.loop:
                break
            reset_events()   # replay the event overlay on the next loop
        print("[replay] done")


def _dump_frames(model, data, apply_row, idx, t_sim, args, flush_events_upto):
    out = args.dump
    out.mkdir(parents=True, exist_ok=True)
    renderer = mj.Renderer(model, height=args.height, width=args.width)
    print(f"[replay] rendering {idx.size} frames → {out}/ ({args.width}x{args.height})")
    for n, i in enumerate(idx):
        i = int(i)
        apply_row(i)
        flush_events_upto(float(t_sim[i]))
        renderer.update_scene(data)
        pixels = renderer.render()
        _write_png(out / f"frame_{n:05d}.png", pixels)
    renderer.close()
    # One PNG per row → the video is inherently index-paced. Match the encode framerate
    # to the index rate so the RRT reach plays at the same visible cadence as the viewer;
    # in --pace sim this is just a sensible default (the trace is ~50 Hz).
    _enc_fps = max(1, round(args.fps * args.speed)) if args.pace == 'index' else 50
    print(f"[replay] wrote {idx.size} PNGs. Encode e.g.:\n"
          f"    ffmpeg -framerate {_enc_fps} -i {out}/frame_%05d.png -c:v libx264 "
          f"-pix_fmt yuv420p {out}/replay.mp4")


def _write_png(path: Path, pixels: np.ndarray):
    try:
        from PIL import Image
        Image.fromarray(pixels).save(path)
    except ImportError:
        # Fallback: matplotlib is a heavier dep but usually present in sci envs.
        import matplotlib.image as mpimg
        mpimg.imsave(path, pixels)


if __name__ == '__main__':
    main()

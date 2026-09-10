"""Offscreen video recording of the MuJoCo scene from named cameras.

Records one MP4 per camera (e.g. a 3rd-person `overview` and the 1st-person `wrist`
camera) into the trial-log run directory, alongside events.jsonl / the trace npz. Uses a
single reused mj.Renderer + one cv2.VideoWriter per camera (mp4v). Frame capture is driven
by the caller (throttled to the recorder's fps) so it adds a bounded, off-the-hot-path cost.

Construction is best-effort: if the GL context / renderer / video writer can't be created,
recording is disabled with a warning and the run continues (recording is a diagnostic aid,
never load-bearing).
"""
from __future__ import annotations

import os

import numpy as np
import mujoco as mj


class SceneRecorder:
    def __init__(self, model, run_dir, run_name, cameras=("overview", "wrist"),
                 fps: float = 30.0, size=(1280, 960)):
        self.enabled = False
        self._writers = {}
        self._renderer = None
        self._model = model
        self._fps = float(fps)
        self._w, self._h = int(size[0]), int(size[1])
        self._last_t = None       # sim-time of the last captured frame (fps throttle)

        # Resolve requested cameras to ids; skip any the scene doesn't define.
        self._cams = []
        for name in cameras:
            cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                print(f"[record] camera {name!r} not in the scene — skipping.")
            else:
                self._cams.append((name, cid))
        if not self._cams:
            print("[record] no valid cameras — recording disabled.")
            return

        try:
            import cv2
            self._cv2 = cv2
            self._renderer = mj.Renderer(model, height=self._h, width=self._w)
            os.makedirs(run_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            for name, cid in self._cams:
                path = os.path.join(str(run_dir), f"{run_name}_{name}.mp4")
                w = cv2.VideoWriter(path, fourcc, self._fps, (self._w, self._h))
                if not w.isOpened():
                    print(f"[record] could not open writer for {path} — skipping {name}.")
                    continue
                self._writers[cid] = (w, path)
            if not self._writers:
                self._teardown()
                return
            self.enabled = True
            print(f"[record] recording {', '.join(n for n, _ in self._cams)} "
                  f"-> {run_dir} at {self._fps:g} fps ({self._w}x{self._h}).")
        except Exception as e:   # GL / cv2 / codec failure — non-fatal
            print(f"[record] disabled ({type(e).__name__}: {e}).")
            self._teardown()

    def capture(self, data, sim_time: float | None = None):
        """Render each camera and append a frame, throttled to fps by sim_time (or wall
        time if sim_time is None). Cheap no-op when disabled."""
        if not self.enabled:
            return
        t = sim_time if sim_time is not None else self._wall()
        # A backspace/BADQACC reset snaps data.time back to 0 (or any earlier value). Without
        # this guard the throttle below would then see (t - _last_t) < 0 for the whole span
        # back to the pre-reset time and skip EVERY frame — recording appeared to "stop" after
        # a reset. On any backward jump, re-anchor and capture this frame so recording
        # continues seamlessly into the reset run.
        if self._last_t is not None and t < self._last_t:
            self._last_t = None
        if self._last_t is not None and (t - self._last_t) < (1.0 / self._fps):
            return
        self._last_t = t
        try:
            for cid, (w, _path) in self._writers.items():
                self._renderer.update_scene(data, camera=cid)
                rgb = self._renderer.render()
                w.write(self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"[record] frame capture failed ({type(e).__name__}: {e}); "
                  f"stopping recording.")
            self.close()

    @staticmethod
    def _wall():
        import time
        return time.monotonic()

    def _teardown(self):
        for w, _ in self._writers.values():
            try:
                w.release()
            except Exception:
                pass
        self._writers = {}
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def close(self):
        if not self._writers and self._renderer is None:
            return
        paths = [p for _, (_, p) in self._writers.items()]
        self._teardown()
        self.enabled = False
        if paths:
            print(f"[record] saved: {', '.join(paths)}")

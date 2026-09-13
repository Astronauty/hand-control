"""H.264 MP4 writing for the offscreen recorders.

Why this exists: `cv2.VideoWriter` with the "mp4v" fourcc produces MPEG-4 Part 2
(DivX-era) video. It is a valid .mp4, but browsers, QuickTime, most desktop
preview/thumbnailer stacks and the GitHub/Slack inline players decode only
H.264 (MPEG-4 Part 10 / AVC) -- so an mp4v clip reads as a broken or
un-previewable file everywhere except VLC/ffplay.

We cannot simply ask cv2 for "avc1": the pip `opencv-python` wheels bundle an
FFmpeg built WITHOUT an H.264 encoder (libx264 is GPL, so the wheels ship
without it), and the writer silently fails to open. The system `ffmpeg` binary
does have libx264, so we pipe raw BGR frames to it instead.

Falls back to the old cv2/mp4v writer when no usable ffmpeg is present, so
recording still happens on a bare machine -- just in the less portable codec.
"""
from __future__ import annotations

import shutil
import subprocess


def _ffmpeg_h264():
    """Path to an ffmpeg that can actually encode H.264, or None."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    return exe if "libx264" in out else None


class H264Writer:
    """Drop-in stand-in for cv2.VideoWriter: write(bgr_frame) / release().

    Frames must all be `size` (w, h) uint8 BGR, matching cv2's convention.
    """

    def __init__(self, path, fps, size, crf=20, preset="medium"):
        self.path = str(path)
        self._w, self._h = int(size[0]), int(size[1])
        self._proc = None
        self._backend = None

        exe = _ffmpeg_h264()
        if exe is not None:
            cmd = [
                exe, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self._w}x{self._h}", "-r", f"{float(fps):g}",
                "-i", "pipe:0",
                "-an",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                # yuv420p + even dimensions: required by QuickTime and most
                # hardware decoders. +faststart moves the moov atom to the
                # front so a player can start without fetching the whole file.
                "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-movflags", "+faststart",
                self.path,
            ]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.PIPE)
                self._backend = "h264"
            except Exception as e:
                print(f"[video] ffmpeg spawn failed ({type(e).__name__}: {e}); "
                      f"falling back to mp4v.")
                self._proc = None

        if self._proc is None:
            import cv2
            self._cv2 = cv2
            self._writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
                (self._w, self._h))
            self._backend = "mp4v"
            if not self._writer.isOpened():
                self._backend = None
            else:
                print(f"[video] no ffmpeg+libx264 found; writing MPEG-4 Part 2 "
                      f"(mp4v) to {self.path} -- may not preview in browsers.")

    def isOpened(self):
        return self._backend is not None

    def write(self, bgr):
        if self._backend == "h264":
            try:
                self._proc.stdin.write(bgr.tobytes())
            except (BrokenPipeError, ValueError):
                # Encoder died; surface its complaint once and stop writing.
                err = b""
                try:
                    err = self._proc.stderr.read() or b""
                except Exception:
                    pass
                print(f"[video] ffmpeg stopped accepting frames for {self.path}"
                      + (f": {err.decode(errors='replace').strip()}" if err else ""))
                self._backend = None
        elif self._backend == "mp4v":
            self._writer.write(bgr)

    def release(self):
        if self._backend == "h264" or self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=60)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            if self._proc.returncode not in (0, None):
                err = b""
                try:
                    err = self._proc.stderr.read() or b""
                except Exception:
                    pass
                print(f"[video] ffmpeg exited {self._proc.returncode} for "
                      f"{self.path}: {err.decode(errors='replace').strip()}")
            self._proc = None
        elif self._backend == "mp4v":
            self._writer.release()
        self._backend = None

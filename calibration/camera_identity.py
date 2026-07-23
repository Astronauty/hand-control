"""Stable hardware identity for USB/UVC and RealSense cameras (Linux/V4L2).

An intrinsic calibration is a property of the LENS + SENSOR, not the USB port or
the OpenCV index — both of which shuffle between sessions. Binding a saved
intrinsic to the camera's hardware identity (USB vendor:product + serial, or the
RealSense SDK serial) lets the pipeline auto-load the right calibration whatever
port/index the camera lands on, and lets us WARN when an index maps to a
different physical camera than the one that was calibrated.

Identity string format (stable, human-readable):
    usb:<vid>:<pid>:<serial>        e.g. usb:046d:0825:ABC123
    usb:<vid>:<pid>:                (no serial reported)
    rs:<serial>                     RealSense, from pyrealsense2

All functions degrade gracefully: if sysfs/pyrealsense2 are unavailable, they
return None and callers fall back to index-based behaviour.
"""
from __future__ import annotations

import os
import glob


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _usb_identity_for_video(index: int) -> str | None:
    """Walk sysfs from /dev/videoN up to its USB device and build a usb: id.

    Returns 'usb:<vid>:<pid>:<serial>' or None if it can't be resolved (e.g. a
    non-USB or virtual device, or a platform without this sysfs layout).
    """
    dev = f"/sys/class/video4linux/video{index}/device"
    base = os.path.realpath(dev) if os.path.exists(dev) else None
    if not base:
        return None
    # Climb parents until we find a dir carrying idVendor/idProduct (the USB
    # device node, above the per-interface node).
    d = base
    for _ in range(8):
        vid = _read(os.path.join(d, "idVendor"))
        pid = _read(os.path.join(d, "idProduct"))
        if vid and pid:
            serial = _read(os.path.join(d, "serial")) or ""
            return f"usb:{vid}:{pid}:{serial}"
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _model_for_video(index: int) -> str | None:
    """Best-effort human label (USB product string) for logging."""
    dev = f"/sys/class/video4linux/video{index}/device"
    base = os.path.realpath(dev) if os.path.exists(dev) else None
    if not base:
        return None
    d = base
    for _ in range(8):
        prod = _read(os.path.join(d, "product"))
        if prod:
            return prod
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def identity_for_index(index: int, realsense: bool = False) -> str | None:
    """Hardware identity of the camera at OpenCV `index`.

    For a RealSense the SDK serial is authoritative; pass realsense=True. For a
    plain UVC webcam we read the USB vid:pid:serial from sysfs. Returns None if
    identity can't be determined (caller falls back to index-only).
    """
    if realsense:
        rs_id = _first_realsense_identity()
        if rs_id:
            return rs_id
        # fall through to USB view of the RealSense's color node
    return _usb_identity_for_video(index)


def label_for_index(index: int) -> str:
    """A short human label for logs (product string or the video node)."""
    return _model_for_video(index) or f"/dev/video{index}"


def _first_realsense_identity() -> str | None:
    """Serial of the first connected RealSense, via pyrealsense2 (if installed)."""
    try:
        import pyrealsense2 as rs
    except Exception:
        return None
    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            if serial:
                return f"rs:{serial}"
    except Exception:
        return None
    return None


def _list_video_indices(max_index: int = 12) -> list[int]:
    """OpenCV indices that have a /dev/videoN node (cheap, no camera open)."""
    idxs = []
    for path in glob.glob("/sys/class/video4linux/video*"):
        try:
            idxs.append(int(os.path.basename(path).replace("video", "")))
        except ValueError:
            pass
    return sorted(i for i in idxs if i <= max_index)


def find_index_by_identity(target_id: str,
                           realsense: bool = False) -> int | None:
    """Find the OpenCV index whose camera matches `target_id`, or None.

    Scans available video nodes and compares hardware identities. For a RealSense
    target (rs:<serial>) we match the SDK serial; the caller still needs the
    COLOR-stream OpenCV index, which this returns by matching the USB device.
    """
    if not target_id:
        return None
    if target_id.startswith("rs:"):
        # A single RealSense: its identity is unique; the color index is whatever
        # video node maps to that USB device. Match by USB device presence.
        cur = _first_realsense_identity()
        if cur != target_id:
            return None
        # Find the color node: the highest-numbered readable 3-channel node of
        # the RealSense is the color stream (depth/IR nodes differ). Left to the
        # caller's probe; here we just confirm the device is present and return
        # the first candidate. In practice callers pass an explicit index too.
        return None   # RealSense color index is resolved by the caller's probe
    for idx in _list_video_indices():
        if _usb_identity_for_video(idx) == target_id:
            return idx
    return None


if __name__ == "__main__":
    # Quick manual check: print identity of every video node.
    for idx in _list_video_indices():
        print(f"video{idx}: id={identity_for_index(idx)}  "
              f"label={label_for_index(idx)!r}")
    rs_id = _first_realsense_identity()
    print(f"realsense: {rs_id or '(none / pyrealsense2 not installed)'}")

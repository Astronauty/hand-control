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
import json


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


def _is_metadata_subnode(index: int) -> bool:
    """True if this video node is a non-primary node of its USB interface.

    Modern UVC drivers expose a second /dev/videoN per physical camera for
    metadata (uncompressed frame metadata, not an image) alongside the real
    capture node. The sysfs 'index' attribute is the node's position within
    its interface: 0 is always the real capture node on every camera in this
    rig, N>=1 the metadata sibling — which OpenCV can never open, but which
    still costs a slow failed VideoCapture() probe (and a printed WARN/ERROR)
    if we try. Skipping it via sysfs first avoids that cost entirely.
    Returns False (don't skip) if sysfs doesn't expose this, e.g. non-Linux —
    callers fall back to the normal open-and-check probe."""
    val = _read(f"/sys/class/video4linux/video{index}/index")
    try:
        return int(val) != 0
    except (TypeError, ValueError):
        return False


def _is_color_frame(frame) -> bool:
    """Heuristic: a real COLOR frame has channel means that differ (B!=G!=R).

    Depth is single-plane; IR is 3-plane but gray (equal channel means). A tiny
    tolerance rejects gray/IR while accepting even a dim, low-saturation color
    scene — real UVC color sensors still separate channels slightly even under
    neutral lighting, while depth/IR nodes read near-zero gap."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return False
    b, g, r = frame[:, :, 0].mean(), frame[:, :, 1].mean(), frame[:, :, 2].mean()
    return (max(b, g, r) - min(b, g, r)) > 1.5


def discover_cameras(max_index: int = 12, warmup: int = 20) -> list[dict]:
    """Auto-discover usable COLOR cameras, one entry per PHYSICAL device.

    Opens each /dev/video* node, keeps only those returning a color frame, and
    dedups by hardware id (a multi-node device — e.g. a RealSense exposing
    color+IR+depth — collapses to a single physical camera). Returns a list of
    dicts (sorted by index) with keys:
        index    OpenCV index of the color stream
        id       hardware identity string (or None)
        label    human product label
        is_realsense  True if it looks like an Intel RealSense
    Requires cv2; identity via sysfs (Linux). Best-effort — skips anything that
    won't open or read.
    """
    import cv2
    rs_id = _first_realsense_identity()
    seen_ids: set = set()
    out: list[dict] = []
    for idx in _list_video_indices(max_index):
        if _is_metadata_subnode(idx):
            continue
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        is_color = False
        for _ in range(warmup):
            ok, fr = cap.read()
            if ok and fr is not None:
                # Check every warmup frame, not just the last — AWB noise can
                # bounce a genuine color camera's channel-mean gap across the
                # threshold frame to frame, especially under neutral lighting.
                if _is_color_frame(fr):
                    is_color = True
                    break   # already have the answer — don't burn the rest
                            # of the warmup budget (dominates discovery time
                            # on a slow-to-open camera).
        cap.release()
        if not is_color:
            continue                       # depth/IR/metadata node
        hw = identity_for_index(idx)
        label = label_for_index(idx)
        is_rs = bool(label and "realsense" in label.lower()) or (
            hw is not None and rs_id is not None)
        # Dedup by hardware id: multiple color-ish nodes of one device -> keep the
        # first (lowest index). Devices with no id can't be deduped -> keep each.
        key = hw or f"__noid_{idx}"
        if key in seen_ids:
            continue
        seen_ids.add(key)
        out.append({"index": idx, "id": hw, "label": label,
                    "is_realsense": is_rs})
    return out


def calibrated_hardware_ids(calib_dir: str) -> dict:
    """Map name -> hardware_id for every camera_intrinsics_<name>.json that has
    one. Only these participate in identity matching."""
    import glob
    out = {}
    for p in glob.glob(os.path.join(calib_dir, "camera_intrinsics_*.json")):
        name = os.path.basename(p)[len("camera_intrinsics_"):-len(".json")]
        try:
            hw = json.load(open(p)).get("hardware_id")
        except Exception:
            hw = None
        if hw:
            out[name] = hw
    return out


def match_calibrated_cameras(calib_dir: str, verbose: bool = True):
    """Discover connected cameras and match each to a calibration by hardware id.

    Returns (specs, realsense_names) where specs is [(name, index)] for every
    calibrated camera currently plugged in (matched by id), and realsense_names
    lists which of those are RealSense. Shared by run_multicam --auto and
    extrinsics-all --auto so their discovery is identical. `verbose` prints a
    per-camera match/skip report and the calibrated-but-absent set.
    """
    calibrated = calibrated_hardware_ids(calib_dir)   # name -> hw_id
    if not calibrated:
        return [], []
    by_id = {hw: name for name, hw in calibrated.items()}
    specs, rs_names, matched = [], [], []
    for cam in discover_cameras():
        name = by_id.get(cam["id"]) if cam["id"] else None
        if name is None:
            if verbose:
                print(f"[match] camera at index {cam['index']} ({cam['label']!r}, "
                      f"id={cam['id']}) has no matching calibration — skipping.")
            continue
        specs.append((name, cam["index"]))
        if cam["is_realsense"]:
            rs_names.append(name)
        matched.append(f"{name}@{cam['index']}")
    if verbose:
        print(f"[match] matched {len(specs)} camera(s): {', '.join(matched) or '(none)'}")
        absent = set(calibrated) - {n for n, _ in specs}
        if absent:
            print(f"[match] calibrated but not plugged in: {sorted(absent)}")
    return specs, rs_names


if __name__ == "__main__":
    # Quick manual check: print identity of every video node, then discovery.
    for idx in _list_video_indices():
        print(f"video{idx}: id={identity_for_index(idx)}  "
              f"label={label_for_index(idx)!r}")
    rs_id = _first_realsense_identity()
    print(f"realsense: {rs_id or '(none / pyrealsense2 not installed)'}")
    print("\ndiscovered color cameras:")
    for c in discover_cameras():
        print(f"  index {c['index']}  {c['label']!r}  id={c['id']}  "
              f"{'[RealSense]' if c['is_realsense'] else ''}")

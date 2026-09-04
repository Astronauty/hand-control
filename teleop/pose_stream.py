#!/usr/bin/env python3
"""
pose_stream.py — send MuJoCo geom poses to the headset, 72 Hz, ~1.4 KB/frame.

The downlink half of the teleop loop. hand_sender.gd pushes hand joints up on
port 9870; this pushes scene poses back down on 9871, so the operator sees the
simulated robot and objects in stereo while their hands drive it.

This streams POSES, not pixels. The geometry was baked into the APK once by
scene_export.py, so a whole scene costs about the same bandwidth as the hand
data — versus a video encoder's 30-50 ms of latency and a monoscopic image.

Two ways to run it:

  In-process (preferred). Import and call from the sim loop, so poses are
  exactly the frame that was just stepped:

      from teleop.pose_stream import PoseStreamer
      streamer = PoseStreamer('godot_scene/scene.json')
      ...
      streamer.send(model, data)          # once per viewer frame

  Standalone, for testing the link with no sim attached:

      python3 teleop/pose_stream.py --demo

Setup
-----
      adb reverse tcp:9871 tcp:9871

Wire format (little-endian), one datagram per frame:
      uint32 magic 0x4E454353 ('SCEN')
      uint16 version | uint16 n_geoms
      uint32 seq | float64 sim_time
      float32[n_geoms*7]  px,py,pz,qx,qy,qz,qw   MuJoCo frame, Z-up, metres

Quaternions go out xyzw to match the uplink's convention; MuJoCo stores wxyz
internally, so they are reordered here. Poses stay in MuJoCo coordinates the
whole way — the Godot side applies one Z-up -> Y-up basis change at the scene
root, so there is exactly one place to get the convention wrong.
"""

import argparse
import json
import socket
import struct
import sys
import time

import numpy as np

MAGIC = 0x4E454353
VERSION = 1
HDR_FMT = "<IHHId"          # 20 bytes
HDR_SIZE = struct.calcsize(HDR_FMT)


def mat_to_quat_xyzw(m):
    """3x3 rotation (row-major, flat 9) -> quaternion xyzw. Shepperd's method,
    branching on the largest diagonal term for numerical stability."""
    m = m.reshape(3, 3)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


class PoseStreamer:
    """
    Non-blocking TCP server. Accepts one headset client and drops frames rather
    than ever stalling the sim loop — a blocked send() inside the control loop
    would be far worse than a dropped frame.
    """

    def __init__(self, scene_json, host="127.0.0.1", port=9871, hz=72.0,
                 quiet=False):
        with open(scene_json) as f:
            scene = json.load(f)
        self.geom_names = [g.get("name", "") for g in scene["geoms"]]
        self.geom_gids = [int(g.get("gid", -1)) for g in scene["geoms"]]
        self.export_ngeom = int(scene.get("model_ngeom", -1))
        self.n = len(self.geom_names)
        self.quiet = quiet
        self.min_dt = 1.0 / hz
        self._gids = None            # resolved lazily against the live model
        self._last = 0.0
        self.seq = 0
        self.sent = 0
        self.dropped = 0

        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(1)
        self.srv.setblocking(False)
        self.conn = None
        self._log(f"scene stream on {host}:{port} — {self.n} geoms, "
                  f"{self.n*7*4} bytes/frame")

    def _log(self, m):
        if not self.quiet:
            print(f"[pose-stream] {m}")

    def _resolve(self, model):
        """
        Map each exported geom to a live id ONCE.

        Named geoms resolve by name, which survives scene pruning. Unnamed ones
        — the Kinova and LEAP visual meshes, which Menagerie ships without geom
        names — can only be resolved by raw id, and that is valid only while the
        live model has the same geom count as the export. Anything unresolvable
        streams identity, so a stale export shows a collapsed robot rather than
        crashing the sim.
        """
        import mujoco as mj
        self._gids = []
        missing = []
        id_shift = (self.export_ngeom >= 0 and self.export_ngeom != model.ngeom)

        for name, gid_export in zip(self.geom_names, self.geom_gids):
            if name:
                gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
                if gid < 0:
                    missing.append(name)
            elif not id_shift and 0 <= gid_export < model.ngeom:
                gid = gid_export
            else:
                gid = -1
                missing.append(f"<unnamed gid {gid_export}>")
            self._gids.append(gid)

        if id_shift:
            self._log(f"WARNING: export had {self.export_ngeom} geoms, live "
                      f"model has {model.ngeom} — geom ids have shifted, so "
                      f"unnamed geoms (the robot meshes) cannot be resolved. "
                      f"Re-run scene_export.py against THIS scene, matching "
                      f"--object.")
        if missing:
            self._log(f"WARNING: {len(missing)} of {self.n} geoms unresolved "
                      f"(first: {missing[0]}) — they will sit at the origin")
        else:
            self._log(f"resolved all {self.n} geoms")

    def _accept(self):
        try:
            self.conn, _ = self.srv.accept()
            self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.conn.setblocking(False)
            self._log("headset connected")
        except BlockingIOError:
            pass

    def send(self, model, data, force=False):
        """Call once per viewer frame. Rate-limited; returns True if sent."""
        now = time.monotonic()
        if not force and now - self._last < self.min_dt:
            return False
        self._last = now

        if self.conn is None:
            self._accept()
            if self.conn is None:
                return False
        if self._gids is None:
            self._resolve(model)

        buf = np.zeros((self.n, 7), dtype=np.float32)
        for i, gid in enumerate(self._gids):
            if gid < 0:
                buf[i, 6] = 1.0                     # identity for a missing geom
                continue
            buf[i, 0:3] = data.geom_xpos[gid]
            buf[i, 3:7] = mat_to_quat_xyzw(data.geom_xmat[gid])

        self.seq += 1
        pkt = struct.pack(HDR_FMT, MAGIC, VERSION, self.n, self.seq,
                          float(data.time)) + buf.tobytes()
        try:
            self.conn.sendall(pkt)
            self.sent += 1
            return True
        except (BlockingIOError, BrokenPipeError, ConnectionResetError, OSError):
            # Drop the frame (or the client). Never block the sim loop.
            self.dropped += 1
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None
            self._log("headset disconnected — waiting for reconnect")
            return False

    def close(self):
        if self.conn:
            self.conn.close()
        self.srv.close()


def demo(scene_json, xml, port, hz):
    """Run the sim headless and stream it, to test the link with no viewer."""
    import mujoco as mj
    model = mj.MjModel.from_xml_path(xml)
    data = mj.MjData(model)
    s = PoseStreamer(scene_json, port=port, hz=hz)
    print("stepping — ctrl-c to stop")
    try:
        while True:
            mj.mj_step(model, data)
            s.send(model, data)
            if s.seq % 200 == 0 and s.seq:
                print(f"  t={data.time:6.2f}s  sent={s.sent}  dropped={s.dropped}")
            time.sleep(model.opt.timestep)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="godot_scene/scene.json")
    ap.add_argument("--xml", default="models/scene_pick_place.xml")
    ap.add_argument("--port", type=int, default=9871)
    ap.add_argument("--hz", type=float, default=72.0)
    ap.add_argument("--demo", action="store_true",
                    help="step the sim headless and stream it")
    args = ap.parse_args()
    if args.demo:
        demo(args.scene, args.xml, args.port, args.hz)
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
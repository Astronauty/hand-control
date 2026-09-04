#!/usr/bin/env python3
"""
vive_hand_publisher.py — drop-in replacement for ui/mediapipe_joint_angles.py.

Receives OpenXR hand-joint frames from a VIVE Focus Vision over TCP (tunnelled
by `adb reverse` across USB) and republishes them on /hand/joint_angles in the
layout DexPilotController expects.

Setup
-----
    adb reverse tcp:9870 tcp:9870
    python3 vive_hand_publisher.py --hand right

Self-test without ROS or a headset:
    python3 vive_hand_publisher.py --selftest

Wire format (little-endian, 780 bytes per frame, per hand)
----------------------------------------------------------
    uint32  magic     0x45564956
    uint16  version   1
    uint16  hand      0 = left, 1 = right
    uint32  seq
    uint64  t_ns
    uint32  tracked
    float32[26*7]     joints: px,py,pz,qx,qy,qz,qw
    float32[7]        headset pose

Poses arrive in Godot/OpenXR convention: right-handed, Y-up, -Z forward,
metres, in the OpenXR STAGE reference space. All conversion happens here.
"""

import argparse
import socket
import struct
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

MAGIC = 0x45564956
N_XR_JOINTS = 26
HDR_FMT = "<IHHIQI"                        # 24 bytes, no padding under '<'
HDR_SIZE = struct.calcsize(HDR_FMT)
BODY_FLOATS = N_XR_JOINTS * 7 + 7
PACKET_SIZE = HDR_SIZE + BODY_FLOATS * 4   # 780

# ---------------------------------------------------------------------------
# Joint remap: OpenXR (26) -> MediaPipe (21)
#
# MediaPipe's "MCP" is OpenXR's *proximal*, not metacarpal, so the OpenXR
# metacarpals (6, 11, 16, 21) are dropped. Getting this wrong shortens every
# proximal phalanx and quietly corrupts the DexPilot keyvectors.
# ---------------------------------------------------------------------------

XR_TO_MP = [
    1,                    # 0      wrist
    2, 3, 4, 5,           # 1-4    thumb: metacarpal, proximal, distal, tip
    7, 8, 9, 10,          # 5-8    index: proximal, intermediate, distal, tip
    12, 13, 14, 15,       # 9-12   middle
    17, 18, 19, 20,       # 13-16  ring
    22, 23, 24, 25,       # 17-20  pinky
]
MP_WRIST = 0

# ---------------------------------------------------------------------------
# Frame conversion: OpenXR (RH, Y-up) -> MuJoCo (RH, Z-up)
#
#   p_mj = (x, -z, y)         a +90 deg rotation about X
#
# For orientations this is a basis change, q_mj = q_c * q_xr * q_c^-1, not a
# rotation. MuJoCo stores quaternions wxyz; OpenXR uses xyzw.
# ---------------------------------------------------------------------------

_S = float(np.sqrt(0.5))
Q_C = np.array([_S, _S, 0.0, 0.0])         # wxyz
Q_C_INV = np.array([_S, -_S, 0.0, 0.0])

# ---------------------------------------------------------------------------
# Yaw alignment — the calibration the ChArUco board used to provide.
#
# The webcam rig solved each camera's extrinsics against a board placed at the
# world origin, so the resulting world frame was aligned to the robot BY
# CONSTRUCTION. OpenXR's STAGE space has no such anchor: gravity fixes Z (which
# is why height is always correct), but the origin and YAW come from wherever
# the operator happened to be facing when the play space was established.
#
# So a yaw offset between hand and robot is expected, not a bug — and it will
# come back if the guardian is re-run or the headset is set up facing a
# different way. It lives here rather than in the app because changing it is a
# restart instead of an APK rebuild, and rather than in dexpilot because every
# other frame convention is already in this file.
#
# Applied to POSITIONS and the wrist QUATERNION together, as a proper rotation
# about the vertical axis. Note a bare X<->Y swap is a REFLECTION (det = -1)
# and would mirror the hand, putting the thumb on the wrong side; --yaw 90 is
# the rotation that swaps the axes correctly.
# ---------------------------------------------------------------------------

_YAW_R = np.eye(3)          # applied in MuJoCo frame (Z up)
_YAW_Q = np.array([1.0, 0.0, 0.0, 0.0])     # wxyz, identity


def set_yaw(deg):
    """Build the yaw rotation once, at startup."""
    global _YAW_R, _YAW_Q
    a = np.radians(float(deg))
    c, s_ = np.cos(a), np.sin(a)
    _YAW_R = np.array([[c, -s_, 0.0],
                       [s_,  c, 0.0],
                       [0.0, 0.0, 1.0]])
    _YAW_Q = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])   # wxyz, about Z


def quat_mul(a, b):
    """Hamilton product; operands and result in wxyz."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def xr_to_mujoco(poses):
    """(N,7) [pos xyz | quat xyzw] -> (N,3) positions, (N,4) quats wxyz.

    Two stages: the fixed OpenXR(Y-up) -> MuJoCo(Z-up) basis change, then the
    operator-set yaw. Both are proper rotations, so the hand is never mirrored.
    """
    p = poses[:, 0:3]
    pos = np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])
    pos = pos @ _YAW_R.T

    quat = np.empty((poses.shape[0], 4))
    for i, (qx, qy, qz, qw) in enumerate(poses[:, 3:7]):
        q = quat_mul(quat_mul(Q_C, np.array([qw, qx, qy, qz])), Q_C_INV)
        quat[i] = quat_mul(_YAW_Q, q)
    return pos, quat


def decode(buf, want_hand):
    """Parse one packet. Returns a dict, or None if it should be ignored."""
    magic, ver, hand, seq, t_ns, tracked = struct.unpack_from(HDR_FMT, buf, 0)
    if magic != MAGIC or ver != 1 or hand != want_hand:
        return None
    if not tracked:
        return None

    floats = np.frombuffer(buf, dtype="<f4", count=BODY_FLOATS,
                           offset=HDR_SIZE).astype(np.float64)
    joints = floats[:N_XR_JOINTS * 7].reshape(N_XR_JOINTS, 7)

    pos_mj, quat_mj = xr_to_mujoco(joints)

    lm = pos_mj[XR_TO_MP]                      # (21,3)
    wrist_pos = lm[MP_WRIST]
    wrist_quat = quat_mj[XR_TO_MP[MP_WRIST]]

    # Raw OpenXR-frame landmarks, kept alongside the MuJoCo-converted ones so
    # --frame can select between them without re-deriving anything.
    # The --frame openxr branch needs the same yaw, applied about OpenXR's
    # vertical (its +Y) rather than MuJoCo's +Z.
    lm_xr = joints[XR_TO_MP, 0:3]
    if not np.allclose(_YAW_R, np.eye(3)):
        _c, _s = _YAW_R[0, 0], _YAW_R[1, 0]
        _Ry = np.array([[_c, 0.0, _s], [0.0, 1.0, 0.0], [-_s, 0.0, _c]])
        lm_xr = lm_xr @ _Ry.T
    wrist_xr = lm_xr[MP_WRIST]

    # Headset pose, the last 7 floats of the payload. Carried through so the
    # egocentric debug view can reconstruct what the tracker sees.
    head = floats[N_XR_JOINTS * 7:N_XR_JOINTS * 7 + 7]

    return {
        "seq": seq,
        "t_ns": t_ns,
        "lm": lm,
        "lm_rel": lm - wrist_pos,              # palm-frame, what keyvectors use
        "wrist_pos": wrist_pos,
        "wrist_quat": wrist_quat,
        "lm_rel_xr": lm_xr - wrist_xr,
        "wrist_xr": wrist_xr,
        "head_xr": head,                       # px,py,pz,qx,qy,qz,qw (OpenXR)
    }


# ---------------------------------------------------------------------------
# Message layout — derived from teleop/dexpilot_controller.py step():
#
#     raw[0:3]     wrist position (camera space)     -> cam_wrist
#     raw[3:57]    54 floats, never read by the controller  -> zeros
#     raw[57:120]  21 world landmarks, WRIST-RELATIVE -> fingertips + retargeting
#     raw[120:183] 21 image landmarks (optional)      -> ARM palm orientation only
#
# The controller requires len(raw) >= 120 and uses the image block only when
# len(raw) >= 183, falling back to world landmarks otherwise.
#
# FRAME CONVENTION — the one thing to verify empirically.
# dexpilot_controller describes the world basis as {x-right, y-UP, z-toward-cam},
# which is exactly OpenXR's convention, implying landmarks go out unconverted.
# But kinova_leap_pick_place's --no-mediapipe path sets R_cam_robot=diag([1,-1,-1])
# so that R_des = palm_R maps the "fused world palm" straight to the robot, which
# reads as MuJoCo Z-up. Those two comments describe different pipelines and I
# can't tell from the source which governs here — so --frame switches between
# them. Try openxr first; if the hand appears rotated 90 deg about X, use mujoco.
# ---------------------------------------------------------------------------

N_MSG_FLOATS = 183
_IMG_BASIS = np.diag([1.0, -1.0, -1.0])   # y-up,z-toward-cam -> y-down,z-pseudodepth
                                          # det = +1, a proper rotation, so the palm
                                          # frame built from it is NOT reflected


def pack_message(d, frame="openxr"):
    """Build the 183-float /hand/joint_angles message."""
    if frame == "openxr":
        wrist = d["wrist_xr"]
        lm_rel = d["lm_rel_xr"]
    else:
        wrist = d["wrist_pos"]
        lm_rel = d["lm_rel"]

    msg = np.zeros(N_MSG_FLOATS, dtype=np.float32)
    msg[0:3] = wrist                       # absolute wrist
    # raw[3:57] is 54 floats the controller never reads, so debug payload rides
    # here without affecting retargeting at all. Only 3:10 is used.
    msg[3:10] = d["head_xr"]                # headset pose, raw OpenXR frame
    msg[57:120] = lm_rel.ravel()           # wrist-relative world landmarks

    # Image landmarks: same points re-expressed in the image basis. Built by a
    # proper rotation rather than by negating coordinates — the controller's own
    # comment warns that negating then cross-producting reflects the frame (det -1),
    # which SVD "repairs" by flipping an arbitrary axis.
    msg[120:183] = (lm_rel @ _IMG_BASIS.T).ravel()
    return msg


# ---------------------------------------------------------------------------
# Framed TCP reader
# ---------------------------------------------------------------------------

class FrameReader:
    """
    Accepts one client and yields only the newest complete frame per poll.

    Drop-to-latest matters: under a stall TCP delivers a backlog, and feeding
    a burst of stale poses into IK would drive the arm through old targets.
    """

    def __init__(self, host, port, want_hand, log):
        self.log = log
        self.want_hand = want_hand
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(1)
        self.srv.setblocking(False)
        self.conn = None
        self.rx = b""
        self.superseded = 0        # newer frame for OUR hand arrived first
        self.other_hand = 0        # frames for the hand we aren't tracking

    def _accept(self):
        try:
            self.conn, _ = self.srv.accept()
            self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.conn.setblocking(False)
            self.rx = b""
            self.log("headset connected")
        except BlockingIOError:
            pass

    def _disconnect(self):
        try:
            self.conn.close()
        except OSError:
            pass
        self.conn = None
        self.rx = b""
        self.log("headset disconnected — waiting for reconnect")

    def latest(self):
        """Newest complete packet, or None."""
        if self.conn is None:
            self._accept()
            return None

        try:
            while True:
                chunk = self.conn.recv(65536)
                if not chunk:
                    self._disconnect()
                    return None
                self.rx += chunk
        except BlockingIOError:
            pass
        except (ConnectionResetError, OSError):
            self._disconnect()
            return None

        n = len(self.rx) // PACKET_SIZE
        if n == 0:
            return None

        # Godot streams BOTH hands, so the newest packet in the buffer is often
        # the hand we are not tracking. Taking it blindly and letting decode()
        # reject it throws away every frame for our hand that arrived just
        # before it. Scan instead, and keep the newest packet that is ours.
        best = None
        for i in range(n):
            pkt = self.rx[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
            hand = struct.unpack_from("<H", pkt, 6)[0]
            if hand == self.want_hand:
                if best is not None:
                    self.superseded += 1
                best = pkt
            else:
                self.other_hand += 1

        self.rx = self.rx[n * PACKET_SIZE:]
        return best

    def close(self):
        if self.conn:
            self.conn.close()
        self.srv.close()


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

def run_ros(args):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    class ViveHandPublisher(Node):

        def __init__(self):
            super().__init__("vive_hand_publisher")
            self.want_hand = 0 if args.hand == "left" else 1
            self.stale_s = args.stale_ms / 1000.0

            self.pub = self.create_publisher(
                Float32MultiArray, args.topic, 10)
            self.reader = FrameReader(
                args.host, args.port, self.want_hand,
                lambda m: self.get_logger().info(m))

            self.last_seq = -1
            self.untracked = 0
            self.published = 0
            self.last_rx = 0.0
            self.warned_stale = False

            self.get_logger().info(
                f"listening on {args.host}:{args.port} for the {args.hand} "
                f"hand, publishing {args.topic}")

            self.create_timer(1.0 / 200.0, self.tick)
            self.create_timer(5.0, self.report)

        def tick(self):
            pkt = self.reader.latest()
            if pkt is None:
                if (self.last_rx and not self.warned_stale
                        and time.monotonic() - self.last_rx > self.stale_s):
                    self.get_logger().warn(
                        "no frames — app closed, or hand out of view?")
                    self.warned_stale = True
                return

            self.warned_stale = False
            self.last_rx = time.monotonic()

            d = decode(pkt, self.want_hand)
            if d is None:
                self.untracked += 1      # hand present in stream, not in view
                return

            self.last_seq = d["seq"]

            msg = Float32MultiArray()
            msg.data = pack_message(d, args.frame).tolist()
            self.pub.publish(msg)
            self.published += 1

        def report(self):
            hz = self.published / 5.0
            self.get_logger().info(
                f"{hz:5.1f} Hz published | {self.untracked} frames with no "
                f"hand in view | {self.reader.other_hand} other-hand | "
                f"{self.reader.superseded} superseded")
            self.published = 0
            self.untracked = 0
            self.reader.other_hand = 0
            self.reader.superseded = 0

    rclpy.init()
    node = ViveHandPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.reader.close()
        node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# Self-test — verifies framing and conversion with no ROS and no headset
# ---------------------------------------------------------------------------

def selftest():
    poses = np.zeros((N_XR_JOINTS, 7), dtype=np.float32)
    poses[:, 6] = 1.0                      # identity quaternions
    poses[1] = [0.10, 1.20, -0.30, 0, 0, 0, 1]   # wrist, 1.2 m up, 0.3 m fwd
    poses[10] = [0.10, 1.35, -0.30, 0, 0, 0, 1]  # index tip, 15 cm above wrist

    body = np.concatenate([poses.ravel(), np.zeros(7, dtype=np.float32)])
    pkt = struct.pack(HDR_FMT, MAGIC, 1, 1, 7, 123456789, 1) + \
        body.astype("<f4").tobytes()

    assert len(pkt) == PACKET_SIZE, f"packet is {len(pkt)}, expected {PACKET_SIZE}"

    d = decode(pkt, want_hand=1)
    assert d is not None, "decode rejected a valid packet"

    print(f"packet size      {len(pkt)} bytes")
    print(f"wrist (MuJoCo)   {np.round(d['wrist_pos'], 3)}")
    print(f"index tip rel    {np.round(d['lm_rel'][8], 3)}")
    for _f in ("openxr", "mujoco"):
        _m = pack_message(d, _f)
        print(f"message ({_f:6s}) {len(_m)} floats, wrist={np.round(_m[0:3],3)}")

    # OpenXR y=+0.15 above the wrist must land on MuJoCo +z.
    assert abs(d["lm_rel"][8][2] - 0.15) < 1e-5, "Y-up -> Z-up conversion wrong"
    # OpenXR -z (forward) must land on MuJoCo +y.
    assert abs(d["wrist_pos"][1] - 0.30) < 1e-5, "forward axis wrong"
    print("\nOK — framing and frame conversion behave as expected.")


def check_yaw():
    """Show what each candidate yaw does, so the right one can be picked by
    matching observed behaviour instead of by trial and error in the sim."""
    moves = {"right (hand +X)": [0.20, 0.0, 0.0],
             "forward (hand -Z)": [0.0, 0.0, -0.20],
             "up (hand +Y)": [0.0, 0.20, 0.0]}
    print("hand motion -> robot motion, per --yaw setting")
    print("(robot axes: +X right, +Y forward/away, +Z up)\n")
    for deg in (0, 90, 180, 270):
        set_yaw(deg)
        print(f"  --yaw {deg}")
        for name, v in moves.items():
            pose = np.zeros((1, 7)); pose[0, 0:3] = v; pose[0, 6] = 1.0
            out = xr_to_mujoco(pose)[0][0]
            ax = "XYZ"[int(np.argmax(np.abs(out)))]
            sg = "+" if out[np.argmax(np.abs(out))] > 0 else "-"
            print(f"      hand {name:<18} -> robot {sg}{ax}  "
                  f"{np.round(out, 3)}")
        print()
    set_yaw(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9870)
    ap.add_argument("--hand", choices=["left", "right"], default="right")
    ap.add_argument("--topic", default="/hand/joint_angles")
    ap.add_argument("--stale-ms", type=int, default=250)
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="degrees of yaw between the headset's STAGE frame and "
                         "the robot base. OpenXR fixes Z by gravity but its yaw "
                         "comes from wherever the operator faced when the play "
                         "space was set, so this replaces the ChArUco board's "
                         "alignment. Try 90 / -90 / 180; see --check-yaw.")
    ap.add_argument("--check-yaw", action="store_true",
                    help="print how each yaw maps hand motion to robot axes, "
                         "then exit")
    ap.add_argument("--frame", choices=["openxr", "mujoco"], default="openxr",
                    help="landmark frame convention (see the layout notes above); "
                         "try openxr first, switch if the hand looks rotated")
    ap.add_argument("--selftest", action="store_true",
                    help="check framing and conversion, then exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0
    if args.check_yaw:
        check_yaw()
        return 0
    set_yaw(args.yaw)
    if args.yaw:
        print(f"[yaw] hand frame rotated {args.yaw:+.0f} deg about vertical")
    run_ros(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

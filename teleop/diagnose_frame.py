"""Diagnose the board->robot direction mapping by MEASURING it, not deriving it.

The teleop position mapping assumes the published board-frame axes align with
MuJoCo world axes (pos_target = home + _Rwb @ (hand_board - board_ref), _Rwb=I).
When "the directions don't seem right", that assumption is wrong — some
published-board axis maps to the wrong robot axis / sign.

This tool prints the live published wrist board position (raw[0:3]) and, after
you mark a reference, the DELTA as you move your hand. Move your hand along ONE
known real-world direction at a time and read which board coordinate changes:

  1. Start the publisher:  uv run python kinova_leap_pick_place.py --mode dexpilot --camera 0
     (or run teleop/ui.py directly). Make sure a RIGHT hand is
     tracked and calibration is loaded (absolute board coords published).
  2. In another sourced terminal:  uv run python teleop/diagnose_frame.py
  3. Hold your hand still, press ENTER to mark the reference.
  4. Move your hand a known amount in ONE direction (e.g. 20 cm to YOUR RIGHT),
     hold still, read the printed delta. Repeat for UP and for AWAY-from-you.

Interpretation: the axis with the largest |delta| is the board axis that your
real motion maps to; its sign tells you the orientation. Fill the resulting
signed-permutation into the controller's world_from_board (_Rwb) so hand-right
-> robot-right, hand-up -> robot-up (MuJoCo: +X forward, +Y left, +Z up).

Keys:  ENTER = (re)mark reference    q + ENTER = quit
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class _Diag(Node):
    def __init__(self):
        super().__init__("diagnose_frame")
        self._wrist = None
        self.create_subscription(
            Float32MultiArray, "/hand/joint_angles",
            lambda m: self._cb(list(m.data)), 10)

    def _cb(self, data):
        if len(data) >= 3:
            self._wrist = np.array(data[0:3], float)
        # Rebuild the ROBOT-ALIGNED palm frame and the robot orientation target
        # (R_des = R_mp_to_robot @ palm_R) using the SAME math as the controller,
        # so we can compare R_des directly to what the robot wrist should be.
        if len(data) >= 120:
            lm = np.array(data[57:120], float).reshape(21, 3)
            self._palm_R = _palm_robot_aligned(lm)
            # Raw fingers direction (wrist->knuckle midpoint) in MediaPipe world
            # axes — to check whether MediaPipe world Y is up or down.
            fingers = 0.5 * (lm[5] + lm[17]) - lm[0]
            self._fingers_mp = fingers / (np.linalg.norm(fingers) + 1e-9)
            if _R_MP_TO_ROBOT is not None:
                self._R_des = _R_MP_TO_ROBOT @ self._palm_R

    @property
    def wrist(self):
        return self._wrist

    @property
    def palm_R(self):
        return getattr(self, "_palm_R", None)

    @property
    def fingers_mp(self):
        return getattr(self, "_fingers_mp", None)

    @property
    def R_des(self):
        return getattr(self, "_R_des", None)


def _palm_robot_aligned(lm):
    """Mirror of DexPilotRetargeter.human_palm_frame_robot_aligned."""
    wrist, idx_mcp, pky_mcp = lm[0], lm[5], lm[17]
    z = 0.5 * (idx_mcp + pky_mcp) - wrist; z /= np.linalg.norm(z) + 1e-9
    thumb_dir = idx_mcp - pky_mcp
    x = np.cross(thumb_dir, z); x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x); y /= np.linalg.norm(y) + 1e-9
    return np.column_stack([x, y, z])


# MediaPipe-world -> robot-world rotation, computed straight from the JSON so
# this tool has NO dependency on the heavy dexpilot_arm_controller import chain
# (mujoco/casadi), which was failing silently and blanking the robot readout.
#   R_mp_to_robot = world_from_board @ R_world_cam @ R_mp_to_cv
def _load_R_mp_to_robot():
    import json
    calib_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration")
    try:
        with open(os.path.join(calib_dir, "camera_extrinsics.json")) as f:
            R_world_cam = np.array(json.load(f)["R_world_cam"], float)
    except Exception as e:
        print(f"[diag] could not load extrinsics for robot readout: {e}")
        return None
    world_from_board = np.diag([1.0, -1.0, -1.0])
    R_mp_to_cv = np.diag([1.0, -1.0, -1.0])
    return world_from_board @ R_world_cam @ R_mp_to_cv


_R_MP_TO_ROBOT = _load_R_mp_to_robot()


def _key_thread(state):
    """stdin keys:
      1 / 2 / 3 : capture the full R_des for pose 1/2/3 (hold the described pose)
      s         : solve the correction M from all captured poses
      p         : RIGIDITY TEST — snapshot palm_R; press again after rotating your
                  hand a KNOWN amount to see the measured rotation angle
      q         : quit
    """
    for line in sys.stdin:
        s = line.strip().lower()
        if s == "q":
            state["quit"] = True
            return
        if s in ("1", "2", "3"):
            state["capture"] = int(s)
        elif s == "s":
            state["solve"] = True
        elif s == "p":
            state["rigidity"] = True


def main():
    rclpy.init()
    node = _Diag()
    state = {"quit": False, "capture": None, "solve": False, "rigidity": False}
    threading.Thread(target=_key_thread, args=(state,), daemon=True).start()
    rigidity_ref = [None]  # palm_R snapshot for the rigidity test

    # THREE reference poses. Columns are [normal, thumb, fingers] (robot roles),
    # each built RIGHT-HANDED (thumb = fingers × normal). MuJoCo world: +X fwd,
    # +Y left, +Z up. Hold each pose, press its number to capture R_des.
    def _rh(normal, fingers):
        n = np.array(normal, float); f = np.array(fingers, float)
        t = np.cross(f, n)
        return np.column_stack([n, t, f])
    DESIRED = {
        1: _rh(normal=[0, 0, -1], fingers=[1, 0, 0]),   # palm DOWN, fingers FWD
        2: _rh(normal=[1, 0, 0], fingers=[0, 0, 1]),    # palm FWD(+X), fingers UP
        3: _rh(normal=[1, 0, 0], fingers=[0, 1, 0]),    # palm FWD(+X), fingers LEFT
    }
    POSE_DESC = {
        1: "PALM DOWN, fingers FORWARD (away from you)",
        2: "PALM FACING the robot (+X fwd), fingers UP",
        3: "PALM FACING the robot (+X fwd), fingers to your LEFT",
    }
    captured = {}  # pose_id -> measured R_des

    print(__doc__)
    print(f"[diag] R_mp_to_robot loaded: {_R_MP_TO_ROBOT is not None}")
    print("[diag] 3-POSE CALIBRATION. Hold each pose, press its number to capture:")
    for i in (1, 2, 3):
        print(f"        {i}: {POSE_DESC[i]}")
    print("[diag] Then press 's' to solve the correction. 'q' quits.")
    ref = None
    last_print = None
    _warned_len = False
    while rclpy.ok() and not state["quit"]:
        rclpy.spin_once(node, timeout_sec=0.05)
        w = node.wrist

        # Handle KEY ACTIONS FIRST, before the 'no wrist' guard, so number/solve
        # keys always respond even if the hand momentarily isn't tracked.
        if state["capture"] is not None:
            pid = state["capture"]; state["capture"] = None
            rd = node.R_des
            if rd is None:
                print("[diag] no R_des yet — is a RIGHT hand tracked & calib loaded? "
                      "(hold the pose steadily in view, then press the number)")
            else:
                captured[pid] = rd.copy()
                print(f"\n[diag] captured pose {pid} ({POSE_DESC[pid]})")
                print(f"       measured R_des =\n{np.round(rd,3)}\n")
            continue
        if state["solve"]:
            state["solve"] = False
            if len(captured) < 2:
                print(f"[diag] need >=2 captured poses (have {len(captured)}).")
                continue
            # Solve M with M @ R_meas = R_desired for all captured poses.
            # Stack: [R_meas1 R_meas2 ...] and [R_des1 R_des2 ...] as 3x(3k),
            # M = A_des @ A_meas^+  (least squares). Then orthonormalise via SVD.
            A_meas = np.hstack([captured[i] for i in sorted(captured)])
            A_des = np.hstack([DESIRED[i] for i in sorted(captured)])
            M = A_des @ np.linalg.pinv(A_meas)
            U, _, Vt = np.linalg.svd(M)
            M_ortho = U @ Vt
            det = float(np.linalg.det(M_ortho))
            print("\n[diag] ===== SOLVED correction M "
                  "(paste as R_align=) =====")
            print("np.array([")
            for row in M_ortho:
                print(f"    [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}],")
            print("])")
            print(f"[diag] det(M)={det:+.3f}  "
                  f"({'ROTATION' if det > 0 else 'REFLECTION — handedness mismatch confirmed'})")
            # residual per pose
            for i in sorted(captured):
                err = np.linalg.norm(M_ortho @ captured[i] - DESIRED[i])
                print(f"[diag]   pose {i} residual: {err:.3f} (0=perfect)")
            print()
            continue
        if state["rigidity"]:
            state["rigidity"] = False
            pr = node.palm_R
            if pr is None:
                print("[diag] no palm_R yet — is a right hand tracked?")
            elif rigidity_ref[0] is None:
                rigidity_ref[0] = pr.copy()
                print("\n[diag] RIGIDITY: reference palm_R snapshotted. Now rotate "
                      "your hand a KNOWN amount (e.g. exactly 90°) and press 'p' again.\n")
            else:
                R_rel = pr @ rigidity_ref[0].T
                # angle of the relative rotation
                ang = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
                ortho = np.allclose(pr @ pr.T, np.eye(3), atol=1e-3)
                det = float(np.linalg.det(pr))
                print(f"\n[diag] RIGIDITY RESULT: palm_R rotated {ang:.1f}° since ref")
                print(f"       (compare to how far you ACTUALLY rotated your hand)")
                print(f"       palm_R orthonormal={ortho}  det={det:+.2f} (must be +1)")
                print(f"       If the angle MATCHES your real rotation -> frame is "
                      f"rigid (bug is elsewhere).")
                print(f"       If it does NOT match -> palm_R is non-rigid (the bug).\n")
                rigidity_ref[0] = None
            continue

        # Below here needs live wrist data; skip if none yet.
        if w is None:
            continue
        if node.R_des is None and not _warned_len:
            _warned_len = True
            print("[diag] NOTE: no robot-target readout yet — message has <120 "
                  "floats (no world landmarks) or calibration didn't load.")
        # Throttle prints to meaningful changes.
        if last_print is not None and np.linalg.norm(w - last_print) < 0.005:
            continue
        last_print = w.copy()
        # Orientation readout: palm frame axes (MediaPipe world axes:
        # x-right, y-up, z-toward-cam). Watch the palm-normal (Z col) as you
        # rotate — if it flips sign between poses, the construction is the bug.
        rd = node.R_des
        if rd is not None:
            # Robot wrist target axes in ROBOT world (X=palm-normal, Y=thumb,
            # Z=fingers). Compare these to what the robot wrist SHOULD be for your
            # held pose. MuJoCo world: +X forward, +Y left, +Z up.
            def _dir(v):
                d = ["+X(fwd)", "-X(back)", "+Y(left)", "-Y(right)",
                     "+Z(up)", "-Z(down)"]
                c = [v[0], -v[0], v[1], -v[1], v[2], -v[2]]
                return d[int(np.argmax(c))]
            ori = (f" | robot target: normal->{_dir(rd[:,0])} "
                   f"thumb->{_dir(rd[:,1])} fingers->{_dir(rd[:,2])}")
        else:
            ori = ""
        # Raw MediaPipe-world fingers direction — the definitive Y-axis check.
        fm = node.fingers_mp
        if fm is not None:
            mp_dirs = ["mp+X(right)", "mp-X(left)", "mp+Y", "mp-Y",
                       "mp+Z(toward-cam)", "mp-Z(away)"]
            mc = [fm[0], -fm[0], fm[1], -fm[1], fm[2], -fm[2]]
            ori += f" | fingers(raw MP)={np.round(fm,2)} ->{mp_dirs[int(np.argmax(mc))]}"

        if ref is None:
            print(f"[diag] board pos = [{w[0]:+.3f} {w[1]:+.3f} {w[2]:+.3f}] m"
                  f"{ori}   (ENTER=mark ref)")
        else:
            d = w - ref
            dom = int(np.argmax(np.abs(d)))
            axis = "XYZ"[dom]
            sign = "+" if d[dom] >= 0 else "-"
            print(f"[diag] Δboard = [{d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}] m   "
                  f"-> dominant board {sign}{axis}{ori}")

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()

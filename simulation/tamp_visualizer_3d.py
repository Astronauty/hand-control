#!/usr/bin/env python3
"""
simulation/tamp_visualizer_3d.py

3D TAMP Visualiser — Kinova Gen3 + LEAP hand
=============================================
Phases
  APPROACH        : move robot wrist in 3-D; object position is FIXED
  GRASP_RECOMMEND : background MultiStartGraspPlanner3D; browse candidates
  EXECUTING       : stub
  TRANSPORTING    : stub
  PLACING         : stub

Input modalities (switch with M key)
  keyboard   (default)  WASD + Q/E move the robot wrist target in XYZ;
                        the arm tracks via damped Jacobian IK every frame
  mediapipe             webcam wrist XY -> robot wrist target XY;
                        pinch triggers grasp plan

Usage
-----
  conda activate hand-control
  cd c:/Users/sures/Documents/GitHub/hand-control_backup_jun_8
  python simulation/tamp_visualizer_3d.py
  python simulation/tamp_visualizer_3d.py --mediapipe
  python simulation/tamp_visualizer_3d.py --obj-pos 0.4 0.1 0.05 --nc 2

Key bindings (also printed at startup)
---------------------------------------
  IDLE             G  : enter APPROACH
  APPROACH         W/S: wrist +/-Y    A/D: wrist +/-X    Q/E: wrist +/-Z
                   G  : run grasp planner -> GRASP_RECOMMEND
  GRASP_RECOMMEND  J/K: next/prev candidate
                   Y  : accept -> EXECUTING (stub)
                   N  : reject all -> APPROACH
                   G  : replan
  STUB phases      SPACE : back to APPROACH
  Any state        M  : toggle keyboard / mediapipe
                   R  : reset to APPROACH
                   P  : print status
"""

import argparse
import enum
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import mujoco
import mujoco.viewer
import numpy as np

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from simulation.grasp_planner_3d import GraspConfig3D, MultiStartGraspPlanner3D   # noqa: E402


def _get_act_idx(model) -> list:
    """qpos address for each actuated joint, in actuator order."""
    return [int(model.jnt_qposadr[model.actuator_trnid[i, 0]])
            for i in range(model.nu)]


# --------------------------------------------------------------------------- #
# Optional MediaPipe
# --------------------------------------------------------------------------- #
_MP_AVAILABLE = False
try:
    import cv2                  # noqa: F401
    import mediapipe as mp      # noqa: F401
    _MP_AVAILABLE = True
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class VizConfig:
    xml_path:       str           = ""
    use_mediapipe:  bool          = False
    mp_cam_id:      int           = 0
    nc:             int           = 3       # seeds per plan
    log_dir:        Optional[str] = None

    # keyboard wrist movement
    wrist_step:     float         = 0.01    # metres per keypress
    wrist_x_min:    float         = 0.10
    wrist_x_max:    float         = 0.70
    wrist_y_min:    float         = -0.50
    wrist_y_max:    float         =  0.50
    wrist_z_min:    float         =  0.05
    wrist_z_max:    float         =  0.60

    # Jacobian IK per-frame
    ik_step_scale:  float         = 0.3     # fraction of error corrected per step
    ik_damp:        float         = 1e-2    # damping for pseudo-inverse

    # mediapipe workspace mapping: normalised image [0,1] -> world metres
    mp_wrist_x_range: tuple       = (0.10, 0.60)
    mp_wrist_y_range: tuple       = (-0.30, 0.30)
    mp_wrist_z:       float       = 0.25    # Z held constant in MP mode

    # misc
    replan_cooldown:  float       = 1.0     # min seconds between replans
    sim_dt_sleep:     float       = 0.002


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #

class State(enum.Enum):
    IDLE            = "IDLE"
    APPROACH        = "APPROACH"
    GRASP_RECOMMEND = "GRASP_RECOMMEND"
    EXECUTING       = "EXECUTING"       # stub
    TRANSPORTING    = "TRANSPORTING"    # stub
    PLACING         = "PLACING"         # stub


# --------------------------------------------------------------------------- #
# MediaPipe worker (optional)
# --------------------------------------------------------------------------- #

class _MediaPipeWorker:
    """
    Daemon thread: reads webcam, runs MediaPipe Hands.
    Exposes wrist normalised XY in [0,1]^2 and pinch flag (thread-safe).
    """

    def __init__(self, cam_id: int = 0):
        self._cam_id = cam_id
        self._lock   = threading.Lock()
        self._wrist_xy: Optional[np.ndarray] = None
        self._pinch  = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not _MP_AVAILABLE:
            raise RuntimeError("mediapipe / cv2 not installed")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="mp-worker")
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def wrist_xy(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._wrist_xy is None else self._wrist_xy.copy()

    @property
    def pinch(self) -> bool:
        with self._lock:
            return self._pinch

    def _loop(self):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(self._cam_id)
        while self._running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.033); continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            wxy, pinch = None, False
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                wxy = np.array([1.0 - lm[0].x, lm[0].y])   # mirror X
                pinch = np.hypot(lm[4].x - lm[8].x,
                                 lm[4].y - lm[8].y) < 0.05
            with self._lock:
                self._wrist_xy = wxy
                self._pinch    = pinch
            cv2.waitKey(1)
        cap.release()
        hands.close()


# --------------------------------------------------------------------------- #
# Home configuration
# --------------------------------------------------------------------------- #

def _home_q(nu: int) -> np.ndarray:
    """Neutral ready pose: arm pointing forward-down, hand open."""
    q = np.zeros(nu)
    q[1] = np.deg2rad(30)
    q[3] = np.deg2rad(-60)
    q[5] = np.deg2rad(60)
    curl = np.deg2rad([10, 0, 15, 10])
    for s in range(7, 19, 4):
        q[s:s+4] = curl
    q[19:23] = np.deg2rad([20, 0, 10, 10])
    return q


# --------------------------------------------------------------------------- #
# Main visualiser
# --------------------------------------------------------------------------- #

class TAMPVisualizer3D:
    """
    MuJoCo TAMP visualiser with Jacobian-IK wrist teleop and grasp planner.

    update()       — call once per viewer frame
    key_callback   — pass to mujoco.viewer.launch_passive
    cleanup()      — stop MediaPipe worker
    """

    _HELP = (
        "\n"
        "  3D TAMP Visualiser\n"
        "  -------------------\n"
        "  IDLE             G  : start (-> APPROACH)\n"
        "  APPROACH         W/S: wrist +/-Y   A/D: wrist +/-X   Q/E: wrist +/-Z\n"
        "                   G  : run grasp planner\n"
        "  GRASP_RECOMMEND  J/K: next/prev candidate\n"
        "                   Y  : accept (EXECUTING stub)\n"
        "                   N  : reject all -> APPROACH\n"
        "                   G  : replan\n"
        "  STUB phases      SPACE : back to APPROACH\n"
        "  Any state        M  : toggle keyboard/mediapipe\n"
        "                   R  : reset   P  : print status\n"
    )

    def __init__(self, cfg: VizConfig):
        self.cfg = cfg
        self._setup_logging()
        self._load_scene()
        self._setup_ik()
        self._setup_planner()
        self._setup_input()
        self._init_state()
        self.log.info(self._HELP)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _setup_logging(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.cfg.log_dir or str(_ROOT / "logs" / f"tamp3d_{ts}")
        os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir
        self.log = logging.getLogger("tamp3d")
        if not self.log.handlers:
            self.log.setLevel(logging.INFO)
            fmt = logging.Formatter(
                "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            self.log.addHandler(sh)

    def _load_scene(self):
        xml = self.cfg.xml_path or str(
            _ROOT / "models" / "scene_pick_place_pos.xml")
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.log.info(f"Scene: {xml}  nu={self.model.nu}  nq={self.model.nq}")

        self._act_idx = _get_act_idx(self.model)

        # object freejoint qpos address (to place object at startup)
        obj_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj_box")
        if obj_bid < 0:
            raise ValueError("Body 'obj_box' not found")
        self._obj_qpos_addr = int(
            self.model.jnt_qposadr[self.model.body_jntadr[obj_bid]])

        # contact marker mocap IDs (4 fingers: thumb, index, middle, ring)
        self._cp1_mocap = self._mocap_id("cp1_body")
        self._cp2_mocap = self._mocap_id("cp2_body")
        self._cp3_mocap = self._mocap_id("cp3_body")
        self._cp4_mocap = self._mocap_id("cp4_body")

    def _mocap_id(self, body_name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found")
        mid = int(self.model.body_mocapid[bid])
        if mid < 0:
            raise ValueError(f"'{body_name}' is not a mocap body")
        return mid

    def _setup_ik(self):
        """Locate the pinch site and arm velocity DOF indices for Jacobian IK."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        if sid < 0:
            # fall back to bracelet_link body centre
            self.log.warning("Site 'pinch_site' not found; using bracelet_link body")
            self._use_site = False
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "bracelet_link")
            if bid < 0:
                raise ValueError("Neither 'pinch_site' nor 'bracelet_link' found")
            self._ik_body_id = bid
        else:
            self._use_site  = True
            self._ik_site_id = sid

        # velocity DOF index for each of the 7 arm actuators
        self._arm_dof_idx = [
            int(self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]])
            for i in range(7)
        ]

    def _setup_planner(self):
        grasp_cfg = GraspConfig3D(
            w_reg=0.01, on_object=True, ik_constraint=True,
            penetration_constraint=False)
        self._ms_planner = MultiStartGraspPlanner3D(
            self.model, self.data, grasp_cfg, log_dir=self._log_dir)
        self.log.info("Planner ready")

    def _setup_input(self):
        self._modality = "mediapipe" if self.cfg.use_mediapipe else "keyboard"
        self._mp_worker: Optional[_MediaPipeWorker] = None
        if self.cfg.use_mediapipe:
            if not _MP_AVAILABLE:
                self.log.warning("mediapipe/cv2 not installed; using keyboard")
                self._modality = "keyboard"
            else:
                self._mp_worker = _MediaPipeWorker(self.cfg.mp_cam_id)
                self._mp_worker.start()
                self.log.info("MediaPipe worker started")

    def _init_state(self):
        self.state = State.IDLE

        self._q_home = _home_q(self.model.nu)
        self._apply_arm(self._q_home)

        # initialise wrist target from FK at home pose
        self._wrist_target = self._current_wrist_pos().copy()

        # grasp plan state
        self._candidates:    List[dict] = []
        self._candidate_idx: int        = 0
        self._plan_status:   str        = ""
        self._plan_thread: Optional[threading.Thread] = None
        self._last_replan_t: float      = 0.0

        # object position — set externally before starting, then fixed
        self._obj_pos = np.array([0.35, 0.0, 0.05])

        self._hide_markers()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_obj_pos(self, pos: np.ndarray):
        """Place the object at a fixed world position (call before starting)."""
        pos = np.asarray(pos, float)
        self._obj_pos = pos.copy()
        a = self._obj_qpos_addr
        self.data.qpos[a:a+3]   = pos
        self.data.qpos[a+3:a+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

    def update(self):
        """Call once per viewer frame."""
        mujoco.mj_step(self.model, self.data)
        if self.state in (State.APPROACH, State.GRASP_RECOMMEND):
            if self._modality == "mediapipe":
                self._update_mediapipe()
            self._ik_step()

    def key_callback(self, keycode: int):
        self._on_key(keycode)

    def cleanup(self):
        if self._mp_worker:
            self._mp_worker.stop()

    # ------------------------------------------------------------------ #
    # MuJoCo helpers
    # ------------------------------------------------------------------ #

    def _apply_arm(self, q: np.ndarray):
        for i, idx in enumerate(self._act_idx):
            self.data.qpos[idx] = q[i]
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _current_wrist_pos(self) -> np.ndarray:
        if self._use_site:
            return self.data.site_xpos[self._ik_site_id].copy()
        else:
            return self.data.xpos[self._ik_body_id].copy()

    def _arm_jacobian(self) -> np.ndarray:
        """3 × 7 position Jacobian of the wrist site w.r.t. the 7 arm DOFs."""
        jac_pos = np.zeros((3, self.model.nv))
        if self._use_site:
            mujoco.mj_jacSite(self.model, self.data, jac_pos, None,
                              self._ik_site_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jac_pos, None,
                              self._ik_body_id)
        return jac_pos[:, self._arm_dof_idx]   # (3, 7)

    def _ik_step(self):
        """Damped Jacobian IK: move arm joints toward _wrist_target."""
        cur = self._current_wrist_pos()
        dx  = self._wrist_target - cur
        if np.linalg.norm(dx) < 1e-4:
            return
        J     = self._arm_jacobian()
        lam   = self.cfg.ik_damp
        J_inv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3))
        dq    = J_inv @ (dx * self.cfg.ik_step_scale)

        q_cur = np.array([self.data.qpos[i] for i in self._act_idx])
        q_new = q_cur.copy()
        q_new[:7] += dq
        q_new[:7]  = np.clip(q_new[:7], -np.pi, np.pi)
        self._apply_arm(q_new)

    def _clamp_wrist_target(self):
        c = self.cfg
        self._wrist_target[0] = np.clip(
            self._wrist_target[0], c.wrist_x_min, c.wrist_x_max)
        self._wrist_target[1] = np.clip(
            self._wrist_target[1], c.wrist_y_min, c.wrist_y_max)
        self._wrist_target[2] = np.clip(
            self._wrist_target[2], c.wrist_z_min, c.wrist_z_max)

    def _show_markers(self, p1: np.ndarray, p2: np.ndarray,
                      p3: np.ndarray | None = None, p4: np.ndarray | None = None):
        far = np.array([0.0, 0.0, -10.0])
        self.data.mocap_pos[self._cp1_mocap] = p1
        self.data.mocap_pos[self._cp2_mocap] = p2
        self.data.mocap_pos[self._cp3_mocap] = p3 if p3 is not None else far
        self.data.mocap_pos[self._cp4_mocap] = p4 if p4 is not None else far

    def _hide_markers(self):
        far = np.array([0.0, 0.0, -10.0])
        self.data.mocap_pos[self._cp1_mocap] = far
        self.data.mocap_pos[self._cp2_mocap] = far
        self.data.mocap_pos[self._cp3_mocap] = far
        self.data.mocap_pos[self._cp4_mocap] = far

    # ------------------------------------------------------------------ #
    # State transitions
    # ------------------------------------------------------------------ #

    def _to_approach(self):
        self.state        = State.APPROACH
        self._candidates  = []
        self._plan_status = ""
        self._hide_markers()
        self._apply_arm(self._q_home)
        self._wrist_target = self._current_wrist_pos().copy()
        self.log.info(
            f"[APPROACH]  obj fixed at {np.round(self._obj_pos,3)}  "
            f"wrist at {np.round(self._wrist_target,3)}  "
            f"modality={self._modality}  —  W/S/A/D/Q/E to move wrist, G to plan")

    def _to_grasp_recommend(self):
        self.state        = State.GRASP_RECOMMEND
        self._candidates  = []
        self._plan_status = "planning..."
        self.log.info("[GRASP_RECOMMEND]  planning in background ...")
        self._start_replan()

    def _to_executing_stub(self):
        self.state = State.EXECUTING
        if self._candidates:
            c = self._candidates[self._candidate_idx]
            self.log.info(
                f"[EXECUTING stub]  status={c.get('status')}  "
                f"cost={c.get('cost')}  "
                f"arm q={np.round(np.rad2deg(c['q'][:7]),1)} deg")
            self.log.info("  press SPACE to return to APPROACH")
        else:
            self.log.warning("[EXECUTING stub]  no candidate to execute")

    def _reset(self):
        self.log.info("[RESET]")
        self._to_approach()

    # ------------------------------------------------------------------ #
    # Grasp planning
    # ------------------------------------------------------------------ #

    def _start_replan(self):
        now = time.monotonic()
        if now - self._last_replan_t < self.cfg.replan_cooldown:
            self.log.info("Replan throttled"); return
        self._last_replan_t = now

        q_snap   = np.array([self.data.qpos[i] for i in self._act_idx])
        obj_snap = self._obj_pos.copy()
        self.log.info(f"  seeds={self.cfg.nc}  obj={np.round(obj_snap,3)}")

        def _run():
            t0     = time.monotonic()
            result = self._ms_planner.solve(q_snap, obj_snap,
                                            max_seeds=self.cfg.nc)
            dt = time.monotonic() - t0

            all_r      = result.get("all_results") or [result]
            candidates = [r for r in all_r if r.get("q") is not None]
            candidates.sort(key=lambda r: (
                {"converged": 0, "best-effort": 1, "failed": 2}
                .get(r.get("status", "failed"), 2),
                r.get("cost") or 1e9))

            self._candidates    = candidates
            self._candidate_idx = 0
            n_ok = sum(1 for r in candidates if r.get("status") == "converged")
            self._plan_status = f"{n_ok}/{len(candidates)} converged ({dt:.1f}s)"
            self.log.info(f"[planner] {self._plan_status}")
            if candidates:
                self._refresh_display()

        self._plan_thread = threading.Thread(
            target=_run, daemon=True, name="planner")
        self._plan_thread.start()

    def _refresh_display(self):
        if not self._candidates:
            self._hide_markers(); return
        c   = self._candidates[self._candidate_idx]
        p1, p2 = c.get("p1"), c.get("p2")
        p3, p4 = c.get("p3"), c.get("p4")
        if p1 is not None and p2 is not None:
            self._show_markers(p1, p2, p3, p4)
        cost_s = f"{c.get('cost'):.5f}" if c.get("cost") is not None else "n/a"
        self.log.info(
            f"  candidate {self._candidate_idx+1}/{len(self._candidates)}"
            f"  status={c.get('status')}  cost={cost_s}"
            f"  p1={np.round(p1,3) if p1 is not None else '?'}"
            f"  p2={np.round(p2,3) if p2 is not None else '?'}"
            f"  p3={np.round(p3,3) if p3 is not None else '?'}"
            f"  p4={np.round(p4,3) if p4 is not None else '?'}")

    def _next_candidate(self):
        if not self._candidates:
            self.log.info("No candidates yet"); return
        self._candidate_idx = (self._candidate_idx + 1) % len(self._candidates)
        self._refresh_display()

    def _prev_candidate(self):
        if not self._candidates:
            return
        self._candidate_idx = (self._candidate_idx - 1) % len(self._candidates)
        self._refresh_display()

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #

    def _on_key(self, keycode: int):
        s    = self.state
        k    = keycode
        step = self.cfg.wrist_step

        # global
        if k == ord("R"): self._reset(); return
        if k == ord("M"): self._toggle_modality(); return
        if k == ord("P"): self._print_status(); return

        if s == State.IDLE:
            if k == ord("G"):
                self._to_approach()

        elif s == State.APPROACH:
            moved = False
            if   k == ord("W"): self._wrist_target[1] += step;  moved = True
            elif k == ord("S"): self._wrist_target[1] -= step;  moved = True
            elif k == ord("A"): self._wrist_target[0] -= step;  moved = True
            elif k == ord("D"): self._wrist_target[0] += step;  moved = True
            elif k == ord("Q"): self._wrist_target[2] += step;  moved = True
            elif k == ord("E"): self._wrist_target[2] -= step;  moved = True
            elif k == ord("G"): self._to_grasp_recommend()
            if moved:
                self._clamp_wrist_target()

        elif s == State.GRASP_RECOMMEND:
            if   k == ord("J"): self._next_candidate()
            elif k == ord("K"): self._prev_candidate()
            elif k == ord("Y"): self._to_executing_stub()
            elif k == ord("N"):
                self.log.info("Rejected -> APPROACH")
                self._to_approach()
            elif k == ord("G"):
                self.log.info("Replanning...")
                self._start_replan()

        elif s in (State.EXECUTING, State.TRANSPORTING, State.PLACING):
            if k == ord(" "):
                self._to_approach()

    def _update_mediapipe(self):
        """Map MediaPipe wrist XY to robot wrist target XY."""
        if not self._mp_worker:
            return
        wxy = self._mp_worker.wrist_xy
        if wxy is None:
            return
        xmin, xmax = self.cfg.mp_wrist_x_range
        ymin, ymax = self.cfg.mp_wrist_y_range
        self._wrist_target[0] = xmin + wxy[0] * (xmax - xmin)
        self._wrist_target[1] = ymin + (1.0 - wxy[1]) * (ymax - ymin)
        self._wrist_target[2] = self.cfg.mp_wrist_z   # Z fixed; use Q/E to adjust
        self._clamp_wrist_target()

        # pinch gesture triggers grasp plan
        if (self._mp_worker.pinch
                and self.state == State.APPROACH
                and time.monotonic() - self._last_replan_t > self.cfg.replan_cooldown):
            self._to_grasp_recommend()

    def _toggle_modality(self):
        if self._modality == "keyboard":
            if not _MP_AVAILABLE:
                self.log.warning("mediapipe not installed"); return
            if not self._mp_worker:
                self._mp_worker = _MediaPipeWorker(self.cfg.mp_cam_id)
                self._mp_worker.start()
            self._modality = "mediapipe"
            self.log.info("Switched to MEDIAPIPE")
        else:
            self._modality = "keyboard"
            self.log.info("Switched to KEYBOARD")

    def _print_status(self):
        self.log.info(
            f"state={self.state.value}  modality={self._modality}  "
            f"wrist_target={np.round(self._wrist_target,3)}  "
            f"wrist_actual={np.round(self._current_wrist_pos(),3)}  "
            f"obj={np.round(self._obj_pos,3)}  "
            f"candidates={len(self._candidates)}  plan='{self._plan_status}'")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _parse_args():
    p = argparse.ArgumentParser(description="3D TAMP Visualiser (Kinova+LEAP)")
    p.add_argument("--xml",        default="",
                   help="Scene XML path (default: models/scene_pick_place_pos.xml)")
    p.add_argument("--mediapipe",  action="store_true",
                   help="Start in mediapipe input mode")
    p.add_argument("--cam",        type=int, default=0,
                   help="Camera index for mediapipe")
    p.add_argument("--nc",         type=int, default=3,
                   help="Grasp seeds per plan (default 3)")
    p.add_argument("--log-dir",    default="",
                   help="Directory for IPOPT logs")
    p.add_argument("--obj-pos",    type=float, nargs=3, default=[0.35, 0.0, 0.05],
                   metavar=("X", "Y", "Z"), help="Fixed object position (default 0.35 0 0.05)")
    p.add_argument("--wrist-step", type=float, default=0.01,
                   help="Wrist target step per keypress in metres (default 0.01)")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg  = VizConfig(
        xml_path      = args.xml,
        use_mediapipe = args.mediapipe,
        mp_cam_id     = args.cam,
        nc            = args.nc,
        log_dir       = args.log_dir or None,
        wrist_step    = args.wrist_step,
    )
    mgr = TAMPVisualizer3D(cfg)
    mgr.set_obj_pos(np.array(args.obj_pos))
    mgr._to_approach()

    with mujoco.viewer.launch_passive(
            mgr.model, mgr.data,
            key_callback=mgr.key_callback) as viewer:
        viewer.cam.azimuth   = 140
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.8
        viewer.cam.lookat[:] = [0.35, 0.0, 0.30]

        try:
            while viewer.is_running():
                mgr.update()
                viewer.sync()
                time.sleep(cfg.sim_dt_sleep)
        finally:
            mgr.cleanup()


if __name__ == "__main__":
    main()

"""DexPilot kinematic retargeting for the LEAP hand.

Implements the cost function from:
  Handa et al. (2019) "DexPilot: Vision Based Teleoperation of
  Dexterous Robotic Hand-Arm System", Section VII-A.

  C(q_a) = (1/2) Σ_i s(d_i) · ‖r_i(q_a) − f(d_i)·r̂_i(q_h)‖²  +  γ‖q_a‖²

The 10 inter-fingertip vectors (all expressed in the palm frame):
  [0-3] Palm → each fingertip       (s=1,   f=β·d;  always in far regime)
  [4-6] Primary-tip → thumb-tip     (S1: s=200, f=η₁ when d<ε)
  [7-9] Primary-tip ↔ primary-tip   (S2: s=400, f=η₂ when both primaries in S1)

Solved with scipy SLSQP, warm-started from previous solution.
DIP joints on the three primary fingers are constrained to equal their MCP joints.
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.optimize
import mujoco as mj

# Tuned retargeting constants persist here (written by the RetargetTuner's save
# key, loaded on construction). Lives next to the other teleop calibration files.
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "calibration", "retarget_config.json",
)


class DexPilotRetargeter:
    """Retargets 21 MediaPipe world landmarks → 16 LEAP hand joint angles."""

    # Paper constants. These are also the fields the live RetargetTuner reads and
    # writes per frame — every value the retargeting cost consumes is an instance
    # attribute (never a bare literal in _switching), so a slider can override it.
    BETA  = 1.6       # robot-to-human hand size scale
    GAMMA = 2.5e-3    # regularisation weight (open hand = zero)
    EPS   = 0.03      # proximity threshold [m] — 3 cm
    ETA1  = 1e-4      # S1 target distance: near-contact [m]
    ETA2  = 3e-2      # S2 minimum inter-primary separation [m]
    S1_GAIN = 200.0   # S1 (pinch) cost weight when a primary tip is near the thumb
    S2_GAIN = 400.0   # S2 cost weight when both involved primaries are pinching

    # Attribute names the tuner sweeps, in slider order. Kept here so the tuner
    # and any config file agree on exactly which fields are tunable.
    TUNABLE = ('BETA', 'GAMMA', 'EPS', 'ETA1', 'ETA2', 'S1_GAIN', 'S2_GAIN')

    # MediaPipe landmark indices
    _LM_WRIST   = 0
    _LM_IDX_MCP = 5   # index finger MCP (defines palm x-axis)
    _LM_PKY_MCP = 17  # pinky MCP (defines palm plane)
    # [index, middle, ring, thumb] fingertip landmark IDs
    _HUMAN_TIPS = [8, 12, 16, 4]

    # LEAP fingertip site names in DexPilot order: [index, middle, ring, thumb]
    _TIP_SITES = [
        'leap_if_ds_tip',
        'leap_mf_ds_tip',
        'leap_rf_ds_tip',
        'leap_th_ds_tip',
    ]
    _PALM_BODY = 'leap_palm'

    # LEAP hand joint names in qpos order (16 joints, arm joints precede these)
    _HAND_JOINT_NAMES = [
        'leap_if_mcp', 'leap_if_rot', 'leap_if_pip', 'leap_if_dip',   # 0-3
        'leap_mf_mcp', 'leap_mf_rot', 'leap_mf_pip', 'leap_mf_dip',   # 4-7
        'leap_rf_mcp', 'leap_rf_rot', 'leap_rf_pip', 'leap_rf_dip',   # 8-11
        'leap_th_cmc', 'leap_th_axl', 'leap_th_mcp', 'leap_th_ipl',   # 12-15
    ]

    # DIP = MCP equality constraints within the 16-element hand vector:
    #   if_dip[3] = if_mcp[0],  mf_dip[7] = mf_mcp[4],  rf_dip[11] = rf_mcp[8]
    _DIP_MCP = [(3, 0), (7, 4), (11, 8)]

    def __init__(self, model: mj.MjModel, n_arm: int = 7,
                 debug: bool = False, eps: float | None = None,
                 load_config: bool = True) -> None:
        self._model = model
        self._n_arm = n_arm
        self._n_hand = 16
        self.debug = debug   # print S1 pinch distances vs EPS each frame

        # Promote the class-attribute defaults to per-instance fields so the live
        # tuner (and the eps override / saved config below) mutate only this
        # instance, never the class.
        for name in self.TUNABLE:
            setattr(self, name, getattr(type(self), name))

        # Proximity threshold seed. The paper's 3 cm assumes an open-hand
        # thumb→fingertip separation of ~10 cm; MediaPipe world landmarks
        # compress the thumb toward the palm plane (noisy z), so the open-hand
        # baseline reads ~4-8 cm here. Set eps between your open and pinched
        # clusters (empirically ~1.8-2.0 cm) to avoid false pinches.
        if eps is not None:
            self.EPS = float(eps)

        # Precedence: class defaults -> eps arg (a code-level seed) -> saved
        # config. A tuned retarget_config.json is the operator's latest word, so
        # it wins over the hardcoded eps once the tuner has saved one.
        if load_config:
            self.load_config()

        self._tip_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, s)
            for s in self._TIP_SITES
        ]
        self._palm_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, self._PALM_BODY)

        # qpos start for the first LEAP hand joint (robustly via joint ID)
        _first_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT,
                                    self._HAND_JOINT_NAMES[0])
        self._hand_qpos0 = int(model.jnt_qposadr[_first_jid])

        # Joint bounds from model (looked up by name, not by raw index)
        self._bounds: list[tuple[float, float]] = []
        for name in self._HAND_JOINT_NAMES:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if model.jnt_limited[jid]:
                lo = float(model.jnt_range[jid, 0])
                hi = float(model.jnt_range[jid, 1])
            else:
                lo, hi = -np.pi, np.pi
            self._bounds.append((lo, hi))

        # Equality constraints: DIP = MCP for the three primary fingers.
        # Each lambda captures dip/mcp at definition time via default args.
        self._constraints = [
            {'type': 'eq', 'fun': lambda q, d=dip, m=mcp: q[d] - q[m]}
            for dip, mcp in self._DIP_MCP
        ]

        # Scratch MjData for FK evaluations (arm joints held fixed at zero —
        # vectors are expressed in the palm frame so arm pose cancels out)
        self._scratch = mj.MjData(model)

        self._q_prev: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Human palm frame
    # ------------------------------------------------------------------

    def human_palm_frame(
        self, lm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a right-handed palm frame from MediaPipe world landmarks.

        Returns (R_WP, origin) where R_WP is the 3×3 world-from-palm rotation
        and origin is the palm origin in world space.

        Frame chosen to MATCH the LEAP robot palm-frame convention (verified via
        FK at q=0): fingers extend along +x, the index→ring spread runs along −y,
        and +z is the palm normal. The retargeting cost matches vector directions,
        so the human and robot palm frames MUST use the same axis convention — an
        earlier version put finger-extension on +y, ~90° off the robot's +x, which
        made the optimizer curl the fingers sideways to a single spot even for an
        open hand.

        Column layout of R_WP:
          [:, 0]  x — wrist → knuckles (up the hand, finger-extension direction)
          [:, 1]  y — across the palm (ring→index side; +y toward index/thumb)
          [:, 2]  z — palm normal (≈ toward camera when palm faces camera)
        """
        origin  = lm[self._LM_WRIST]
        idx_mcp = lm[self._LM_IDX_MCP]
        pky_mcp = lm[self._LM_PKY_MCP]

        # x: wrist → knuckle line midpoint (up the hand toward the fingers).
        # Using the index/pinky MCP midpoint (not just index-MCP) makes this the
        # true finger-extension axis, matching the robot's +x.
        knuckle_mid = 0.5 * (idx_mcp + pky_mcp)
        x = knuckle_mid - origin
        x /= np.linalg.norm(x) + 1e-9

        # z: palm normal — cross(up-the-hand, pinky→index across direction).
        # Ordered so +z points out of the palm (toward camera for a facing hand).
        across_ptoi = idx_mcp - pky_mcp          # pinky → index
        z = np.cross(x, across_ptoi)
        z /= np.linalg.norm(z) + 1e-9

        # y: completes right-handed frame (points toward the index/thumb side)
        y = np.cross(z, x)
        y /= np.linalg.norm(y) + 1e-9

        return np.column_stack([x, y, z]), origin

    def human_palm_frame_robot_aligned(
        self, lm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Palm frame whose axis ROLES match the robot pinch_site frame, so that
        an identical physical hand/wrist orientation yields an identical rotation
        matrix — making the arm-orientation map an IDENTITY (no R_align / no
        press-8 offset needed). Verified against the model: pinch_site +X = palm
        normal (= thumb×fingers), +Y = toward thumb, +Z = along fingers.

        This is SEPARATE from human_palm_frame() (used by the finger retargeting
        cost, which needs the fingers-on-X convention) — do not merge them.

        Columns of the returned world-from-palm rotation:
          [:, 0]  X — palm normal  (thumb_dir × fingers, matching robot +X)
          [:, 1]  Y — toward the thumb/index side (across the palm)
          [:, 2]  Z — along the fingers (wrist → knuckle midpoint)

        """
        origin  = lm[self._LM_WRIST]
        idx_mcp = lm[self._LM_IDX_MCP]
        pky_mcp = lm[self._LM_PKY_MCP]

        z = 0.5 * (idx_mcp + pky_mcp) - origin      # along fingers -> robot +Z
        z /= np.linalg.norm(z) + 1e-9
        thumb_dir = idx_mcp - pky_mcp               # pinky -> index (thumb side)
        x = np.cross(thumb_dir, z)                  # palm normal = thumb × fingers
        x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)                          # toward thumb, completes RH
        y /= np.linalg.norm(y) + 1e-9

        return np.column_stack([x, y, z]), origin

    # ------------------------------------------------------------------
    # Vector lists
    # ------------------------------------------------------------------

    # Fingertip-source options ----------------------------------------------
    # Image-landmark fingertips can be more stable near edge-on wrist poses than
    # world landmarks (the world-3D model head has a depth-flip ambiguity there).
    # But image coords are ANISOTROPIC: x,y are frame-FRACTIONS on a 4:3 (640x480)
    # sensor, so image-y is inflated 640/480 vs image-x, and z is pseudodepth in a
    # model-defined unit. A single scalar EPS can't threshold an anisotropic space
    # consistently (a pinch triggers at a different gap depending on orientation),
    # so we apply a per-axis scale S to make image space ISOTROPIC before use, then
    # map image basis -> world basis with C (so tips compose with the WORLD palm
    # frame, which is left exactly as-is). EPS/ETA are then just re-tuned numbers.
    _IMG_ASPECT   = 480.0 / 640.0      # y-scale: undo the 4:3 frame aspect
    Z_SCALE       = 1.0                # pseudodepth scale vs image-x (empirical; 1:1 for now)
    _C_IMG_TO_WORLD = np.diag([1.0, -1.0, -1.0])   # image basis -> world basis

    def _tip_scale(self) -> np.ndarray:
        """Per-axis image->world tip map: C @ diag([1, aspect, Z_SCALE])."""
        S = np.diag([1.0, self._IMG_ASPECT, float(self.Z_SCALE)])
        return self._C_IMG_TO_WORLD @ S

    def _human_vectors(self, lm: np.ndarray,
                       tip_lm: np.ndarray | None = None) -> list[dict]:
        """Compute 10 inter-fingertip vectors in the human palm frame.

        Args:
          lm:     (21,3) WORLD landmarks. ALWAYS defines the palm FRAME (unchanged).
          tip_lm: optional (21,3) IMAGE landmarks. When given, fingertip POSITIONS
                  (and the palm origin) come from these instead of world landmarks,
                  isotropy-corrected and mapped into world axes by _tip_scale(), so
                  they compose with the world palm frame. The palm frame itself is
                  still built from world `lm` — only the tip source changes.

        Returns a list of dicts with keys:
          'r'    : ndarray(3) — the vector
          'type' : 'palm' | 's1' | 's2'
          For 's1': 'primary' (int 0-2, 0=index)
          For 's2': 's1_dep' ((int, int) — S1 indices that must both be near)
        """
        # Palm frame: ALWAYS from world landmarks (this handling is unchanged).
        R_WP, origin_W = self.human_palm_frame(lm)
        R_PW = R_WP.T

        if tip_lm is not None:
            # Isotropy-correct + image->world map, then use image-landmark tips.
            M = self._tip_scale()
            origin = M @ tip_lm[self._LM_WRIST]
            tips_src = [M @ p for p in tip_lm[self._HUMAN_TIPS]]
        else:
            origin = origin_W
            tips_src = [p for p in lm[self._HUMAN_TIPS]]         # (4,3) world

        tips = [R_PW @ (t - origin) for t in tips_src]           # in palm frame

        vecs: list[dict] = []

        # [0-3] Palm → tip
        for i, t in enumerate(tips):
            vecs.append({'r': t.copy(), 'type': 'palm', 'idx': i})

        # [4-6] S1: primary tip → thumb tip  (th - primary, expressed in palm frame)
        for i in range(3):   # 0=index, 1=middle, 2=ring
            vecs.append({'r': (tips[3] - tips[i]).copy(), 'type': 's1', 'primary': i})

        # [7-9] S2: primary ↔ primary
        for (i, j), (si, sj) in zip([(0,1),(0,2),(1,2)], [(0,1),(0,2),(1,2)]):
            vecs.append({'r': (tips[j] - tips[i]).copy(), 'type': 's2', 's1_dep': (si, sj)})

        return vecs

    def _robot_vectors(self, q_hand: np.ndarray) -> list[np.ndarray]:
        """Compute 10 robot inter-fingertip vectors in the LEAP palm frame via FK."""
        qp0 = self._hand_qpos0
        self._scratch.qpos[qp0: qp0 + self._n_hand] = q_hand
        mj.mj_kinematics(self._model, self._scratch)

        p_palm = self._scratch.xpos[self._palm_id]
        R_WP   = self._scratch.xmat[self._palm_id].reshape(3, 3)
        R_PW   = R_WP.T

        tips = [R_PW @ (self._scratch.site_xpos[sid] - p_palm)
                for sid in self._tip_ids]   # [if, mf, rf, th] in palm frame

        vecs: list[np.ndarray] = []
        for t in tips:                              # [0-3] palm → tip
            vecs.append(t.copy())
        for i in range(3):                         # [4-6] S1: primary → thumb
            vecs.append(tips[3] - tips[i])
        for i, j in [(0,1),(0,2),(1,2)]:           # [7-9] S2: primary ↔ primary
            vecs.append(tips[j] - tips[i])

        return vecs

    # ------------------------------------------------------------------
    # Switching functions
    # ------------------------------------------------------------------

    def _switching(
        self, hv: dict, d_s1: list[float]
    ) -> tuple[float, float]:
        """Return (s, f) for one human vector entry.

        Args:
            hv:   One element from _human_vectors().
            d_s1: Distances [d_if→th, d_mf→th, d_rf→th] from the S1 vectors.
        """
        vtype = hv['type']
        d = float(np.linalg.norm(hv['r']))

        if vtype == 'palm':
            return 1.0, self.BETA * d

        if vtype == 's1':
            if d <= self.EPS:
                return self.S1_GAIN, self.ETA1
            return 1.0, self.BETA * d

        # s2: active only when BOTH involved primary fingers are in S1 pinch
        si, sj = hv['s1_dep']
        if d_s1[si] <= self.EPS and d_s1[sj] <= self.EPS:
            return self.S2_GAIN, self.ETA2
        return 1.0, self.BETA * d

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _cost(
        self,
        q_hand: np.ndarray,
        human_vecs: list[dict],
        d_s1: list[float],
    ) -> float:
        robot_vecs = self._robot_vectors(q_hand)
        c = self.GAMMA * float(np.dot(q_hand, q_hand))
        for rv, hv in zip(robot_vecs, human_vecs):
            s, f = self._switching(hv, d_s1)
            d = float(np.linalg.norm(hv['r']))
            r_h_hat = hv['r'] / (d + 1e-9)
            diff = rv - f * r_h_hat
            c += 0.5 * s * float(np.dot(diff, diff))
        return c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retarget(
        self,
        world_lm: np.ndarray,
        q_prev: np.ndarray | None = None,
        image_lm: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map 21 MediaPipe world landmarks to 16 LEAP hand joint angles.

        Args:
            world_lm: (21, 3) array of world landmark positions in metres.
                      MediaPipe convention: wrist ≈ origin, hand facing camera.
                      ALWAYS used for the palm frame.
            q_prev:   Optional external warm-start (overrides internal state).
            image_lm: optional (21, 3) IMAGE landmarks. When given, fingertip
                      POSITIONS come from these (isotropy-corrected — see
                      _human_vectors) instead of world landmarks; often more stable
                      near edge-on wrist poses. Re-tune EPS/ETA for the new units.
        Returns:
            q_hand: (16,) LEAP joint angles in radians.
        """
        human_vecs = self._human_vectors(world_lm, tip_lm=image_lm)
        d_s1 = [float(np.linalg.norm(human_vecs[4 + i]['r'])) for i in range(3)]

        if self.debug:
            d_palm = [float(np.linalg.norm(human_vecs[i]['r'])) for i in range(4)]
            pinch = ['PINCH' if d <= self.EPS else 'open' for d in d_s1]
            print(f"[retarget] S1 d(if,mf,rf→th)={[round(d,3) for d in d_s1]} "
                  f"EPS={self.EPS} -> {pinch} | palm→tip={[round(d,3) for d in d_palm]}")

        x0 = (q_prev if q_prev is not None
               else (self._q_prev if self._q_prev is not None
                     else np.zeros(self._n_hand)))

        result = scipy.optimize.minimize(
            self._cost, x0,
            args=(human_vecs, d_s1),
            method='SLSQP',
            bounds=self._bounds,
            constraints=self._constraints,
            options={'maxiter': 100, 'ftol': 1e-6},
        )

        q = result.x.copy()
        self._q_prev = q
        return q

    def reset(self) -> None:
        """Clear warm-start state (call when hand tracking is lost)."""
        self._q_prev = None

    # ------------------------------------------------------------------
    # Live-tuning config (see teleop/retarget_tuner.py)
    # ------------------------------------------------------------------

    def tunables(self) -> dict[str, float]:
        """Current values of every tunable retargeting constant."""
        return {name: float(getattr(self, name)) for name in self.TUNABLE}

    def load_config(self, path: str | None = None) -> bool:
        """Overlay saved tunables from JSON onto this instance.

        Only keys in TUNABLE are applied (unknown keys ignored), so a stale or
        hand-edited file can't inject arbitrary attributes. Returns True if a file
        was found and read. Missing file is not an error — defaults stand.
        """
        path = path or _CONFIG_PATH
        if not os.path.exists(path):
            return False
        with open(path) as f:
            cfg = json.load(f) or {}
        for name in self.TUNABLE:
            if name in cfg:
                setattr(self, name, float(cfg[name]))
        print(f"[retarget] loaded tuned constants from {path}: {self.tunables()}")
        return True

    def save_config(self, path: str | None = None) -> str:
        """Write the current tunables to JSON so they auto-load next launch."""
        path = path or _CONFIG_PATH
        with open(path, "w") as f:
            json.dump(self.tunables(), f, indent=2)
        print(f"[retarget] saved tuned constants -> {path}")
        return path

"""MuJoCo forward-kinematics adaptor for the VWJ optimizer.

Provides, over the first `n_opt` actuated joints (7 arm + 16 LEAP hand = 23):
  * site world positions for a candidate qpos,
  * the wrist site's world rotation (as a wxyz quaternion),
  * site translational + rotational Jacobians (world frame),
all on a SCRATCH mj.MjData so the live sim state is never touched.

This replaces the upstream method's pinocchio RobotAdaptor. The objective code
(vwj/objective.py) only ever talks to this interface, so the kinematics backend is
swappable and — being the same MuJoCo model the simulator runs — is guaranteed
consistent with the robot the other baselines use.
"""
from __future__ import annotations

import numpy as np
import mujoco as mj


class MujocoKinematics:
    def __init__(self, model: mj.MjModel, opt_joint_names: list[str],
                 site_names: list[str], wrist_site: str):
        """
        model:            the compiled Gen3+LEAP MjModel (from the live scene is fine;
                          we only read its structure and use a private scratch MjData).
        opt_joint_names:  the joints the optimizer controls, in optimization order
                          (7 arm + 16 hand). Their qpos addresses define the x-vector.
        site_names:       every site the objective references (tips + DIPs + wrist),
                          deduplicated; positions are returned in this order.
        wrist_site:       the site whose rotation feeds the wrist-orientation term.
        """
        self.model = model
        self.data = mj.MjData(model)          # private scratch — never the sim's data
        self.site_names = list(site_names)
        self.site_ids = np.array(
            [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, s) for s in self.site_names], int)
        if np.any(self.site_ids < 0):
            missing = [s for s, i in zip(self.site_names, self.site_ids) if i < 0]
            raise ValueError(f"VWJ: sites not found in model: {missing}")
        self.wrist_idx = self.site_names.index(wrist_site)

        # qpos addresses AND dof (velocity/Jacobian) columns of the optimized joints,
        # in optimization order. For the 1-DOF hinge joints here qpos and dof indices
        # coincide, but resolve both explicitly so the objective/Jacobian stay correct.
        self._qadr = np.empty(len(opt_joint_names), int)
        self._dof_cols = np.empty(len(opt_joint_names), int)
        for i, jn in enumerate(opt_joint_names):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise ValueError(f"VWJ: joint not found: {jn}")
            self._qadr[i] = model.jnt_qposadr[jid]
            self._dof_cols[i] = model.jnt_dofadr[jid]
        self.n_opt = len(opt_joint_names)

        # per-joint limits (optimization order); continuous joints report (0,0) -> unbounded.
        self.lo = np.empty(self.n_opt); self.hi = np.empty(self.n_opt)
        for i, jn in enumerate(opt_joint_names):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jn)
            r = model.jnt_range[jid]
            if r[0] == 0.0 and r[1] == 0.0:
                self.lo[i], self.hi[i] = -np.inf, np.inf
            else:
                self.lo[i], self.hi[i] = r[0], r[1]

    def set_qpos(self, q_opt: np.ndarray) -> None:
        """Write the optimized joints into the scratch data (other DOF left at 0)."""
        self.data.qpos[self._qadr] = q_opt

    def forward(self, q_opt: np.ndarray) -> None:
        """Position-only kinematics update for the candidate q (cheap; no Jacobians)."""
        self.set_qpos(q_opt)
        mj.mj_kinematics(self.model, self.data)

    def forward_with_jac(self, q_opt: np.ndarray) -> None:
        """Kinematics + the comp needed for site Jacobians (mj_comPos)."""
        self.set_qpos(q_opt)
        mj.mj_kinematics(self.model, self.data)
        mj.mj_comPos(self.model, self.data)   # required before mj_jacSite

    def site_positions(self) -> np.ndarray:
        """(n_site, 3) world positions, in site_names order (call after forward*)."""
        return self.data.site_xpos[self.site_ids].copy()

    def wrist_quat(self) -> np.ndarray:
        """Wrist site world rotation as a wxyz quaternion (call after forward*)."""
        R = self.data.site_xmat[self.site_ids[self.wrist_idx]].reshape(3, 3)
        q = np.empty(4)
        mj.mju_mat2Quat(q, R.reshape(9))
        return q                               # (w, x, y, z)

    def site_jacobians(self):
        """Per-site translational + rotational Jacobians restricted to the optimized
        joints. Returns (Jp, Jr) each (n_site, 3, n_opt). Call after forward_with_jac."""
        nv = self.model.nv
        cols = self._dof_cols
        Jp = np.empty((len(self.site_ids), 3, self.n_opt))
        Jr = np.empty((len(self.site_ids), 3, self.n_opt))
        _jp = np.zeros((3, nv)); _jr = np.zeros((3, nv))
        for k, sid in enumerate(self.site_ids):
            mj.mj_jacSite(self.model, self.data, _jp, _jr, sid)
            Jp[k] = _jp[:, cols]
            Jr[k] = _jr[:, cols]
        return Jp, Jr

"""Grasping controller: joint-space PD hold of a grasp configuration + gravity
compensation, with an optional pure internal (pinching) force superimposed via
fingertip Jacobian transposes:

    tau = Kp (q_target - q) - Kd qdot + qfrc_bias + sum_k J_k^T (R_WSk f_ck)
    f_c = null(G) @ gamma          (w_des = 0 — no net object wrench)

The internal force squeezes the object along the grasp map's null space, so with
an exact grasp map it produces zero net wrench on the object — the object should
not accelerate, only be pinched. Scene-agnostic: all ids/gains are injected.
"""

import numpy as np
import mujoco as mj

from grasp_control.grasp_map import SpatialGraspMapComputer
from grasp_control.force_control import GraspForceAllocator


class GraspController:
    def __init__(self, model, n_robot, tip_site_ids, obj_site_ids, obj_body_id,
                 kp, kd, gamma=5.0, squeeze_pd_scale=1.0,
                 active_joint_slices=((7, 11), (19, 23)),
                 grasp_map_computer=None, allocator=None):
        """
        Args:
            model: MjModel.
            n_robot: number of robot DOFs (object joints follow in qpos/qvel).
            tip_site_ids: fingertip site ids, one per grasping finger.
            obj_site_ids: object contact-site ids, same order as tip_site_ids.
            obj_body_id: body id of the grasped object.
            kp, kd: (n_robot,) PD gains for the q_target hold.
            gamma: internal squeeze force scale (null-space weight); negate if
                fingers pull apart (the inward_dirs anchor should prevent that).
            squeeze_pd_scale: multiplier on kp/kd over active_joint_slices while
                squeezing — lower it (e.g. 0.25) if the finger PD fights the
                squeeze and measured contact force falls short of gamma/sqrt(2).
            active_joint_slices: (start, stop) qpos slices of the grasping
                fingers' joints (default: LEAP index 7:11 and thumb 19:23).
            grasp_map_computer / allocator: injected for testability.
        """
        self.model = model
        self.n_robot = n_robot
        self.tip_site_ids = list(tip_site_ids)
        self.obj_site_ids = list(obj_site_ids)
        self.obj_body_id = obj_body_id
        self.kp = np.asarray(kp, dtype=float).copy()
        self.kd = np.asarray(kd, dtype=float).copy()
        self.squeeze_pd_scale = squeeze_pd_scale
        self.active_joint_slices = tuple(active_joint_slices)
        self.grasp_map_computer = grasp_map_computer or SpatialGraspMapComputer()
        self.allocator = allocator or GraspForceAllocator(gamma)

        self.q_target = None
        self.squeeze = False
        self.last_f_c = None   # (3*n_fingers,) contact-frame forces from last compute()

    def set_target(self, q_target):
        """Set the (n_robot,) joint-space PD setpoint."""
        self.q_target = np.asarray(q_target, dtype=float).copy()

    def set_squeeze(self, on):
        self.squeeze = bool(on)
        if not on:
            self.last_f_c = None

    def _effective_gains(self):
        if not self.squeeze or self.squeeze_pd_scale == 1.0:
            return self.kp, self.kd
        kp, kd = self.kp.copy(), self.kd.copy()
        for lo, hi in self.active_joint_slices:
            kp[lo:hi] *= self.squeeze_pd_scale
            kd[lo:hi] *= self.squeeze_pd_scale
        return kp, kd

    def compute(self, data):
        """Return a full nv-length torque vector for data.qfrc_applied (object
        DOFs zero). Pure torque map — never mutates data; the caller owns
        quasi-static stabilization (qvel zeroing) and mj_step."""
        assert self.q_target is not None, "call set_target() before compute()"
        n = self.n_robot
        tau = np.zeros(self.model.nv)

        kp, kd = self._effective_gains()
        tau[:n] = kp * (self.q_target - data.qpos[:n]) + kd * (0 - data.qvel[:n])

        if self.squeeze:
            tau[:n] += self._internal_force_torques(data)

        # Gravity/bias compensation for the robot chain only.
        tau[:n] += data.qfrc_bias[:n]
        return tau

    def _internal_force_torques(self, data):
        """Pure internal force from the live grasp map's null space, mapped to
        joint torques through the fingertip Jacobians."""
        n = self.n_robot
        p_WoO = data.xpos[self.obj_body_id]
        R_WO = data.xmat[self.obj_body_id].reshape(3, 3)

        contacts, inward_dirs, R_WS_list, J_list = [], [], [], []
        for k, (sid_obj, sid_tip) in enumerate(zip(self.obj_site_ids, self.tip_site_ids)):
            R_WSk = data.site_xmat[sid_obj].reshape(3, 3)
            p_WoSk = data.site_xpos[sid_obj]
            p_OSk_O = R_WO.T @ (p_WoSk - p_WoO)
            R_OSk = R_WO.T @ R_WSk
            contacts.append({'p': p_OSk_O, 'R': R_OSk})
            R_WS_list.append(R_WSk)

            # Squeeze-sign anchor on the first contact only: its null-space
            # component must point toward the object center (compressive).
            if k == 0:
                inward_dirs.append(R_OSk.T @ (-p_OSk_O / np.linalg.norm(p_OSk_O)))
            else:
                inward_dirs.append(None)

            Jk_full = np.zeros((3, self.model.nv))
            mj.mj_jacSite(self.model, data, Jk_full, None, sid_tip)
            J_list.append(Jk_full[:3, :n])

        G = self.grasp_map_computer.compute(contacts)
        f_c = self.allocator.allocate(G, np.zeros(6), contact_dof=3,
                                      inward_dirs=inward_dirs)
        self.last_f_c = f_c

        tau_int = np.zeros(n)
        for k in range(len(J_list)):
            f_ck_W = R_WS_list[k] @ f_c[3*k:3*k+3]
            tau_int += J_list[k].T @ f_ck_W
        return tau_int

"""Grasping controller: joint-space PD hold of a grasp configuration + gravity
compensation, with an optional pure internal (pinching) force superimposed via
fingertip Jacobian transposes:

    tau = Kp (q_target - q) - Kd qdot + qfrc_bias + sum_k J_k^T (R_WSk f_ck)
    f_c = pinv(G) @ w_des + null(G) @ gamma

By default w_des = 0 (pure internal force: zero net object wrench — the object
is only pinched, and its weight must be carried by friction reacting through the
finger PD). With support_weight=True, w_des is set each step to the wrench that
statically supports the object's weight, so the load is explicitly allocated to
the contacts instead of left to friction. Scene-agnostic: all ids/gains injected.
"""

import numpy as np
import mujoco as mj

from grasp_control.grasp_map import SpatialGraspMapComputer
from grasp_control.force_control import GraspForceAllocator


class GraspController:
    def __init__(self, model, n_robot, tip_site_ids, obj_site_ids, obj_body_id,
                 kp, kd, gamma=5.0, squeeze_pd_scale=1.0,
                 active_joint_slices=((7, 11), (19, 23)),
                 support_weight=False, pad_offsets=None,
                 grasp_map_computer=None, allocator=None,
                 obj_contact_provider=None):
        """
        Args:
            model: MjModel.
            n_robot: number of robot DOFs (object joints follow in qpos/qvel).
            tip_site_ids: fingertip site ids, one per grasping finger.
            obj_site_ids: object contact-site ids, same order as tip_site_ids.
                May be None when obj_contact_provider is supplied (teleop mode:
                the recommended contacts have no MuJoCo sites).
            obj_body_id: body id of the grasped object.
            obj_contact_provider: optional callable provider(data) -> list of
                (p_W (3,), R_W_inward (3,3)) per contact, same order as
                tip_site_ids, giving the live world-frame contact position and a
                rotation whose col0 is the INWARD surface normal (same convention
                as the contact SITES' x-axis). When None, contact geometry is read
                from obj_site_ids exactly as before (autonomous/site-based path).
                Used for NLP-recommended contacts that track the object body via
                stored object-local offsets rather than authored sites.
            kp, kd: (n_robot,) PD gains for the q_target hold.
            gamma: internal squeeze force scale (null-space weight); negate if
                fingers pull apart (the inward_dirs anchor should prevent that).
            squeeze_pd_scale: multiplier on kp/kd over active_joint_slices while
                squeezing — lower it (e.g. 0.25) if the finger PD fights the
                squeeze and measured contact force falls short of gamma/sqrt(2).
            active_joint_slices: (start, stop) qpos slices of the grasping
                fingers' joints (default: LEAP index 7:11 and thumb 19:23).
            support_weight: when True, allocate w_des = the wrench that statically
                supports the object's weight (recomputed each step from the live
                object pose) on top of the null-space squeeze, so gravity is
                carried by commanded contact forces instead of friction alone.
            pad_offsets: optional per-contact distance (m) from each tip SITE to its
                fingerpad SURFACE along the pad normal. Tip sites sit at the tip-mesh
                centroid, ~10mm behind the pad — without this offset the slip
                correction anchors the centroid onto the object surface, a constant
                phantom error that biases the fingers inward instead of only
                countering true tangential slip.
            grasp_map_computer / allocator: injected for testability.
        """
        self.model = model
        self.n_robot = n_robot
        self.tip_site_ids = list(tip_site_ids)
        self.obj_site_ids = list(obj_site_ids) if obj_site_ids is not None else None
        self.obj_contact_provider = obj_contact_provider
        if self.obj_site_ids is None and obj_contact_provider is None:
            raise ValueError("GraspController needs either obj_site_ids or "
                             "obj_contact_provider")
        self.obj_body_id = obj_body_id
        self.kp = np.asarray(kp, dtype=float).copy()
        self.kd = np.asarray(kd, dtype=float).copy()
        self.squeeze_pd_scale = squeeze_pd_scale
        self.active_joint_slices = tuple(active_joint_slices)
        self.support_weight = bool(support_weight)
        self.pad_offsets = (list(pad_offsets) if pad_offsets is not None
                            else [0.0] * len(self.tip_site_ids))
        self.grasp_map_computer = grasp_map_computer or SpatialGraspMapComputer()
        self.allocator = allocator or GraspForceAllocator(gamma)

        self.q_target = None
        self.squeeze = False
        self.last_f_c = None   # (3*n_fingers,) contact-frame forces from last compute()
        self.last_f_c_W = None # (n_fingers, 3) same forces mapped to WORLD frame (R_WS @ f_ck)
        self.last_contacts_W = None  # (n_fingers, 3) grasp-map contact points in WORLD frame

    def set_target(self, q_target):
        """Set the (n_robot,) joint-space PD setpoint."""
        self.q_target = np.asarray(q_target, dtype=float).copy()

    def set_squeeze(self, on):
        self.squeeze = bool(on)
        if not on:
            self.last_f_c = None

    def effective_gains(self):
        """kp/kd with squeeze_pd_scale applied over active_joint_slices while
        squeezing. Public so callers that hand-roll their own PD (e.g.
        kinova_leap_pick_place's GRASP phase) can still get the softened gains
        without going through compute()."""
        if not self.squeeze or self.squeeze_pd_scale == 1.0:
            return self.kp, self.kd
        kp, kd = self.kp.copy(), self.kd.copy()
        for lo, hi in self.active_joint_slices:
            kp[lo:hi] *= self.squeeze_pd_scale
            kd[lo:hi] *= self.squeeze_pd_scale
        return kp, kd

    def _live_contacts(self, data):
        """Per-contact (p_W (3,), R_W_inward (3,3)) at the current data, in
        tip_site order. Uses obj_contact_provider when set, else reads the
        contact SITES (col0 of site_xmat is the inward normal)."""
        if self.obj_contact_provider is not None:
            out = self.obj_contact_provider(data)
            return [(np.asarray(p, float).reshape(3),
                     np.asarray(R, float).reshape(3, 3)) for p, R in out]
        return [(data.site_xpos[sid].copy(),
                 data.site_xmat[sid].reshape(3, 3).copy())
                for sid in self.obj_site_ids]

    def compute(self, data):
        """Return a full nv-length torque vector for data.qfrc_applied (object
        DOFs zero). Pure torque map — never mutates data; the caller owns
        quasi-static stabilization (qvel zeroing) and mj_step."""
        assert self.q_target is not None, "call set_target() before compute()"
        n = self.n_robot
        tau = np.zeros(self.model.nv)

        kp, kd = self.effective_gains()
        tau[:n] = kp * (self.q_target - data.qpos[:n]) + kd * (0 - data.qvel[:n])

        if self.squeeze:
            tau[:n] += self.internal_force_torques(data)

        # Gravity/bias compensation for the robot chain only.
        tau[:n] += data.qfrc_bias[:n]
        return tau

    def slip_correction_torques(self, data, kp=200.0, f_max=10.0):
        """Anchor each fingertip to its object contact site:
        tau = sum_k J_k^T kp (p_Sk - p_tipk), applied through the FINGER joints only
        (active_joint_slices). The object sites move with the object, so this is
        contact-frame position feedback that counters tangential slip of the grasp
        under load — the soft finger joint PD alone cannot hold the tangential
        friction force, so the tips shear off the object during transport. Kept off
        the arm columns deliberately: there the same virtual spring acts as a
        constant drag opposing arm motion (any accumulated tip-site offset never
        resets), fighting the jog instead of maintaining the grip.

        f_max caps each finger's virtual spring force: if the grasp is ever lost and
        the object gets away, the tip<->site error is no longer a slip (it can be
        decimetres), and an uncapped kp*err would command enormous torques at a
        dislocated grasp geometry."""
        n = self.n_robot
        tau = np.zeros(n)
        live = self._live_contacts(data)
        for k, (sid_tip, (p_WoSk, R_WSk)) in enumerate(zip(self.tip_site_ids, live)):
            J = np.zeros((3, self.model.nv))
            mj.mj_jacSite(self.model, data, J, None, sid_tip)
            # Anchor = contact point backed off by the pad-surface offset along the
            # inward normal (col0 of the contact frame) — where the tip SITE sits
            # when the pad SURFACE is flush on the object.
            inward_W = R_WSk[:, 0]
            anchor_W = p_WoSk - self.pad_offsets[k] * inward_W
            f_k = kp * (anchor_W - data.site_xpos[sid_tip])
            f_norm = float(np.linalg.norm(f_k))
            if f_norm > f_max:
                f_k *= f_max / f_norm
            for lo, hi in self.active_joint_slices:
                tau[lo:hi] += J[:3, lo:hi].T @ f_k
        return tau

    def internal_force_torques(self, data, scale=1.0):
        """Pure internal force from the live grasp map's null space, mapped to
        joint torques through the fingertip Jacobians. Public so callers that own
        their PD hold (e.g. kinova_leap_pick_place's GRASP phase) can superimpose
        just the squeeze; updates last_f_c.

        scale multiplies the whole allocated force (squeeze + weight support) —
        callers should ramp it 0->1 over ~0.5s at squeeze-on: the pair of contact
        forces is only 'internal' once BOTH contacts exist, and full force applied
        while a finger is still closing a gap arrives as an unbalanced shove that
        knocks the object out of the grasp."""
        n = self.n_robot
        p_WoO = data.xpos[self.obj_body_id]
        R_WO = data.xmat[self.obj_body_id].reshape(3, 3)

        contacts, inward_dirs, R_WS_list, J_list = [], [], [], []
        live = self._live_contacts(data)
        for k, (sid_tip, (p_WoSk, R_WSk)) in enumerate(zip(self.tip_site_ids, live)):
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
        # w_des: zero (pure pinch) by default; with support_weight, the object-frame
        # wrench that statically supports the object's weight. G's torque reference
        # is the object BODY ORIGIN (contacts use p_OSk_O), while gravity acts at
        # the COM (xipos) — hence the r x f moment arm between the two.
        w_des = np.zeros(6)
        if self.support_weight:
            m_obj = float(self.model.body_mass[self.obj_body_id])
            f_W   = -m_obj * self.model.opt.gravity          # hand-on-object support force
            r_W   = data.xipos[self.obj_body_id] - p_WoO     # origin -> COM, world frame
            w_des[:3] = R_WO.T @ f_W
            w_des[3:] = R_WO.T @ np.cross(r_W, f_W)
        f_c = scale * self.allocator.allocate(G, w_des, contact_dof=3,
                                              inward_dirs=inward_dirs)
        self.last_f_c = f_c

        n_f = len(J_list)
        f_c_W = np.zeros((n_f, 3))
        pts_W = np.zeros((n_f, 3))
        tau_int = np.zeros(n)
        for k in range(n_f):
            f_ck_W = R_WS_list[k] @ f_c[3*k:3*k+3]
            f_c_W[k] = f_ck_W
            # Grasp-map contact point in world (contacts store object-frame p_OSk_O).
            pts_W[k] = p_WoO + R_WO @ contacts[k]['p']
            tau_int += J_list[k].T @ f_ck_W
        self.last_f_c_W = f_c_W
        self.last_contacts_W = pts_W
        return tau_int

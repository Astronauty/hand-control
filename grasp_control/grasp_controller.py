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


def _peak_internal_normal(f_int, n_contacts, contact_dof=3):
    """Largest per-contact NORMAL component of a stacked contact-force vector.

    Contact frames in this repo put the inward normal in col0, and G's columns are
    built in those same frames, so each contact's normal component is element 0 of
    its own block. Used to normalise the internal force to unit peak normal, which
    is what makes gamma mean newtons (see internal_force_torques)."""
    f_int = np.asarray(f_int, float)
    return max(abs(float(f_int[contact_dof * k])) for k in range(n_contacts))


class GraspController:
    def __init__(self, model, n_robot, tip_site_ids, obj_site_ids, obj_body_id,
                 kp, kd, gamma=5.0, squeeze_pd_scale=1.0, transport_pd_scale=1.0,
                 squeeze_pd_per_finger=False, squeeze_pd_ratio_target=2.0,
                 squeeze_pd_min_scale=0.05,
                 active_joint_slices=((7, 11), (19, 23)),
                 support_weight=False, pad_offsets=None,
                 grasp_map_computer=None, allocator=None,
                 obj_contact_provider=None, cone_gamma=True,
                 cone_mu=0.7, cone_f_min=0.5, cone_margin=0.2,
                 gamma_ref=None, nullspace_tracking=False):
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
            gamma: internal squeeze force -- the per-contact internal NORMAL force
                in NEWTONS on the cone_gamma path, which normalises the null-space
                force to unit peak normal before scaling. That is the contract
                scripts/3D_minimum_NCF.py defines and solve_gamma_live solves
                against, so a solved gamma can be commanded directly. (On the
                sign-anchor fallback path it remains a raw null-space weight, since
                that path has no cone solve to normalise against.) Negate if the
                fingers pull apart (the inward_dirs anchor should prevent that).
            squeeze_pd_scale: multiplier on kp/kd over active_joint_slices while
                squeezing — lower it (e.g. 0.25) if the finger PD fights the
                squeeze and the measured contact force falls short of the
                commanded gamma (which is now in newtons; the old 'gamma/sqrt(2)'
                rule of thumb predates the normalisation fix).
            transport_pd_scale: multiplier used INSTEAD once set_transporting(True)
                is called — i.e. while the grasp is bearing load rather than
                closing. Defaults to 1.0 (full gains), because a softened finger
                PD is back-driven by the object's weight and bleeds normal force
                until the pinch fails. See effective_gains for the measurements.
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
            gamma_ref: the gamma that kp/kd were tuned at. When set, the finger
                gains are additionally scaled by gamma/gamma_ref, so the PD keeps
                a CONSTANT ratio of authority against the internal force instead
                of being a constant absolute stiffness. This is what makes a gain
                tuned once survive a change in gamma (e.g. a new disturbance
                budget); see effective_gains for the measured failure boundary.
                None (default) keeps the legacy absolute-gain behaviour.
            nullspace_tracking: apply FRoGGeR's eq. (18) projector
                (I - Jh^T (Jh^T)^dagger) to the TRACKING torque while squeezing, so
                tracking cannot push the fingertips across the object's surface.
                Their stated purpose is that applying the tracking torques "does not
                change the contact positions between the hand and object". Scoped to
                the finger columns so the arm jog is unaffected, matching their note
                that the projection leaves the arm torques alone. Default False:
                every existing result was measured without it.
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
        # Per-finger squeeze authority. OFF by default: every measured result in
        # this repo was taken with the single global scale, and this changes the
        # gains a running grasp sees. See _per_finger_pd_scales().
        self.squeeze_pd_per_finger  = squeeze_pd_per_finger
        # Ratio |tau_int| / |tau_pd| a finger is considered to have enough
        # authority at. 2.0, not 1.0: at exactly 1.0 the two torques balance and
        # the finger stalls, which is the measured failure (0.93x stalled at
        # 4.24 mm). The three fingers that DID seat in that run were at 1.5x,
        # 8.5x and 15.1x, so 2.0 sits just above the lowest observed success and
        # leaves the comfortable fingers untouched.
        self.squeeze_pd_ratio_target = squeeze_pd_ratio_target
        # Floor on the per-finger cut. A finger with no tracking authority at all
        # drifts across the surface under the squeeze -- the failure
        # effective_gains' own docstring records for the lift phase -- so the cut
        # is bounded rather than allowed to go to zero.
        self.squeeze_pd_min_scale   = squeeze_pd_min_scale
        # Previous step's finger torques, the input to the per-finger ratio.
        # None until the first compute().
        self._last_tau_pd  = None
        self._last_tau_int = None
        self.transport_pd_scale = transport_pd_scale
        self.transporting = False
        self.active_joint_slices = tuple(active_joint_slices)
        self.support_weight = bool(support_weight)
        self.pad_offsets = (list(pad_offsets) if pad_offsets is not None
                            else [0.0] * len(self.tip_site_ids))
        self.grasp_map_computer = grasp_map_computer or SpatialGraspMapComputer()
        self.allocator = allocator or GraspForceAllocator(gamma)
        # CONE-CONSTRAINED gamma. The sign-anchor path only orients each
        # null-space basis vector using the FIRST non-None inward_dirs entry, so
        # with 2 antipodal contacts (null(G) 1-D) one sign is the whole answer,
        # but with 3 contacts the null space is 3-D and nothing holds the
        # non-anchor contacts compressive. Measured on an asymmetric tripod:
        # sign-anchor normal forces [1.213, 0.047, 0.325] N vs cone-solve
        # [1.0, 1.0, 1.0] N. See tests/test_force_allocator_cone.py.
        self.cone_gamma  = bool(cone_gamma)
        self.cone_mu     = float(cone_mu)
        self.cone_f_min  = float(cone_f_min)
        self.cone_margin = float(cone_margin)
        self.last_cone_info = None
        # Gamma the kp/kd above were tuned at. None = absolute gains (legacy).
        self.gamma_ref = None if gamma_ref is None else float(gamma_ref)
        # FRoGGeR eq. (18): project the tracking torque out of the hand
        # Jacobian's range while squeezing. Off by default -- see compute().
        self.nullspace_tracking = bool(nullspace_tracking)

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

    def set_transporting(self, on):
        """Switch the finger joints from CLOSING gains to HOLDING gains.

        Call this once the squeeze has converged and the grasp starts bearing
        load (a lift/transport jog). See effective_gains for why the two phases
        need opposite gains.
        """
        self.transporting = bool(on)

    def effective_gains(self):
        """kp/kd with the finger-joint gains scaled over active_joint_slices.

        The CLOSING and HOLDING phases of a grasp want OPPOSITE gains, and this
        picks between them:

        * Closing (squeeze): stiff finger gains fight the squeeze -- the PD is
          holding the fingers at the planned pre-contact posture while
          internal_force_torques tries to drive them inward, so a high kp
          suppresses the very motion that closes the gap. Measured: sweeping kp
          0.8 -> 20 monotonically REDUCED grip force 6.03 -> 1.28 N. This is
          what squeeze_pd_scale (0.25) exists for.

        * Holding (transport): the object's weight now acts THROUGH the
          fingertips, back-driving those same joints. Softened gains lose ground
          to it, the fingers open, penetration drops and normal force bleeds
          away. Measured on 036_wood_block seed 2 (7.15 N object) during the
          lift: fn decayed 8.20 -> 6.71 N over ~200ms with the squeeze command
          held constant, finger position error creeping 2.83 -> 3.55 deg, and
          the object began falling while both contacts were still present.
          Sweeping the lift-phase scale:

              scale  peak rise   contact lost   final carry
              0.25    27.98mm      2.50s          -0.6%
              0.5     48.20mm      3.05s          44.5%
              1.0     69.58mm      never          71.9%
              2.0     75.78mm      never          78.4%

          017_orange improves too (87.5% -> 95.3% carry at 1.0), so this is not
          a heavy-object special case.

        transport_pd_scale therefore defaults to 1.0 (full gains): a grasp that
        is holding should hold. Note this does NOT re-litigate the closing-phase
        tuning above -- that measurement was of grip force DURING the squeeze,
        which is a different quantity from force RETENTION under load.

        GAMMA-REFERENCED SCALING (gamma_ref). Both tunings above were measured at
        one squeeze magnitude, so they are really ratios masquerading as absolute
        gains. The finger PD and the internal force act on the SAME joints in
        opposition:

            tau_finger = kp * (q_target - q)  +  J^T (R f_c)
                         \______ PD ______/     \___ squeeze ___/

        and |f_c| scales with gamma. Only their RATIO decides whether the tips stay
        on the planned contacts, so a gain that is right at one gamma is wrong by
        that same factor at another.

        Measured on 036_wood_block seed 2 (after the equilibrium-scaling fix but
        BEFORE the gamma normalisation, so these gammas are in the old inflated
        units), lift_ok at squeeze_pd_scale=1.0:

            gamma   kp=0.8   kp=3.0   kp=20.0
              1.0     ok       --        --
              4.0     ok       --        --
              8.0     ok       ok        --
             12.0   FAIL       ok        ok

        Setting gamma_ref pins the ratio: gains are multiplied by gamma/gamma_ref,
        so the PD keeps constant authority against the squeeze.

        STILL NEEDED AFTER THE NORMALISATION FIX, for a reason worth being precise
        about. Normalising made gamma mean newtons, which removed the mass-driven
        unit drift -- but it did NOT make one absolute finger gain sufficient,
        because how much PD authority a given geometry has against a given squeeze
        still varies per grasp. 036_wood_block seed 1 is the case: at the solved
        gamma=25.9 with absolute gains the object never moves and the force tracks
        the command (16-20 N), yet |tau_int| runs ~60x |tau_pd| and the tips creep
        2 -> 15 mm off the planned contacts until the grasp slides off (fn 0.0).
        With gamma_ref=1.0 the same seed lifts at that gamma (20.7/22.0 N). Seed 2
        of the same object does not need it. So this is about PD authority per
        grasp, not about gamma's units -- which is why fixing the units did not
        retire it.
        Left None the behaviour is exactly as before, so existing callers and
        their tuned constants are unaffected.

        Public so callers that hand-roll their own PD (e.g.
        kinova_leap_pick_place's GRASP phase) get the same gains compute() uses.
        """
        scale = (self.transport_pd_scale if self.transporting
                 else self.squeeze_pd_scale if self.squeeze else 1.0)
        # Track the commanded squeeze magnitude, so a gain tuned at gamma_ref
        # keeps the same authority at any other gamma.
        if self.gamma_ref is not None:
            g_cmd = float(np.max(np.atleast_1d(self.allocator.gamma)))
            scale *= max(g_cmd, 1e-9) / self.gamma_ref
        # PER-FINGER authority (squeeze_pd_per_finger). One global scale is the
        # wrong instrument for the failure it was introduced to fix, and the
        # measurement says so directly. At the squeeze plateau on
        # 036_wood_block seed 0 (n=4), |tau_int| / |tau_pd| per finger read
        #
        #     thumb 15.1x   index 8.5x   ring 1.5x   middle 0.93x
        #
        # and ONLY the finger below 1.0 failed to close -- it stalled 4.24 mm off
        # the surface at 0.00 N while the other three were seated. The object had
        # moved 0.6 mm, the joints had 0.84 rad of margin and the actuators were
        # at 0.7 of +/-8 Nm, so this is a quasi-static standoff between two
        # COMMANDED torques, not a limit and not a reach failure.
        #
        # Lowering the global scale does fix that finger (0.00 -> 8.46 N at 0.10,
        # and the first four-finger lift) but it pays for it everywhere: measured
        # max fingertip drift off the planned contact went 5.2 -> 10.7 mm, and
        # because the allocator re-solves from the LIVE contact geometry its own
        # command then moved too (middle 7.75 -> 14.13 N at the same gamma). A
        # finger that is already seated does not need its tracking weakened.
        #
        # So: scale each finger's slice by what THAT finger needs. A finger whose
        # ratio is already comfortable keeps full tracking authority.
        #
        # MEASURED, AND IT DOES NOT FIX THE 0 N FINGER. Kept because the negative
        # result is what locates the real cause, and re-deriving it costs a day.
        # On the same cell it drives middle's ratio to exactly the target
        # (|pd| 0.743 -> 0.340, ratio 0.93x -> 2.0x) and the finger STILL reads
        # 0.00 N at 5.7 mm. Raising the target to 4.0 and 8.0 changes nothing.
        # The global scale=0.10 that DOES work differs in a way this cannot
        # reproduce: softening EVERY finger lets the whole hand settle, the live
        # contact geometry moves, and the allocator -- which re-solves G from
        # `_live_contacts` every step -- then hands the middle finger a much
        # larger share. Measured |f_c| at the same gamma:
        #
        #     global 0.10   ring  7.69  middle 14.13  index  9.16  thumb 27.41
        #     per-finger    ring 12.75  middle  7.72  index 10.99  thumb 27.37
        #
        # i.e. the middle finger's ALLOCATED force nearly doubles under the
        # global softening. The 0 N finger is not short of PD authority, it is
        # short of ALLOCATED FORCE, and softening its own tracking cannot give it
        # more. The fix belongs in the allocator (a finger not yet in contact
        # should be commanded harder, not the others weaker), not here.
        #
        # OFF by default, accordingly.
        per = self._per_finger_pd_scales() if (
            self.squeeze and not self.transporting
            and self.squeeze_pd_per_finger) else None
        if scale == 1.0 and per is None:
            return self.kp, self.kd
        kp, kd = self.kp.copy(), self.kd.copy()
        for _k, (lo, hi) in enumerate(self.active_joint_slices):
            s_k = scale * (1.0 if per is None else float(per[_k]))
            kp[lo:hi] *= s_k
            kd[lo:hi] *= s_k
        return kp, kd

    def _per_finger_pd_scales(self):
        """Multiplier per finger slice, in active_joint_slices order.

        Derived from the ratio the standoff is actually decided by:

            r_k = |tau_int,k| / |tau_pd,k|

        both measured on the SAME joints on the previous control step. Where
        r_k >= squeeze_pd_ratio_target the finger is winning comfortably and
        keeps its gain; where it is below, the gain is cut by exactly the
        shortfall, r_k / target, so the finger is given just enough authority to
        close rather than being softened wholesale. Clamped below by
        squeeze_pd_min_scale so a finger can never be left with no tracking at
        all -- an unconstrained cut is how a finger ends up drifting across the
        surface, which is the failure mode squeeze_pd_scale's own docstring
        records for the LIFT phase.

        Returns None before the first compute() has populated the torque
        history, so the first step is unchanged.

        NOTE this reads the PREVIOUS step's torques, which is what makes it
        cheap and non-circular: scaling kp changes tau_pd on the NEXT step, and
        the ratio re-measures it. It is a slow outer loop around the PD, not an
        algebraic solve.
        """
        if self._last_tau_pd is None or self._last_tau_int is None:
            return None
        tgt = float(self.squeeze_pd_ratio_target)
        lo_s = float(self.squeeze_pd_min_scale)
        out = []
        for (lo, hi) in self.active_joint_slices:
            pd_k = float(np.abs(self._last_tau_pd[lo:hi]).max())
            it_k = float(np.abs(self._last_tau_int[lo:hi]).max())
            if pd_k <= 1e-9:
                out.append(1.0)          # already at target, nothing to fight
                continue
            r_k = it_k / pd_k
            out.append(1.0 if r_k >= tgt else max(lo_s, r_k / tgt))
        return out

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

        # FRoGGeR's eq. (18) structure: the TRACKING term is projected into the null
        # space of the hand Jacobian before the contact-force term is added, so
        # "applying them does not change the contact positions between the hand and
        # object" (their App. B). Our tracking PD is otherwise free to push along
        # directions that move the fingertips across the object's surface, which is
        # what the squeeze then has to fight -- hence squeeze_pd_scale, which solves
        # the same problem by weakening tracking everywhere rather than only where
        # it conflicts.
        #
        # Off by default: every existing tabletop result was measured without it,
        # and the projector changes the hold's behaviour, not just its numbers.
        if self.squeeze and self.nullspace_tracking:
            tau[:n] = self._project_out_contact_dirs(data, tau[:n])

        if self.squeeze:
            # Recorded SEPARATELY, before they are summed, because
            # _per_finger_pd_scales needs the two in isolation -- once added
            # there is no way to recover which part is tracking and which is
            # squeeze. tau[:n] at this point is the tracking term (plus the
            # null-space projection when enabled), which is exactly the torque
            # the squeeze has to overcome.
            _tau_int = self.internal_force_torques(data)
            self._last_tau_pd  = tau[:n].copy()
            self._last_tau_int = _tau_int[:n].copy()
            tau[:n] += _tau_int

        # Gravity/bias compensation for the robot chain only.
        tau[:n] += data.qfrc_bias[:n]
        return tau

    def _project_out_contact_dirs(self, data, tau_n):
        """(I - Jh^T (Jh^T)^dagger) tau -- FRoGGeR's eq. (18) projector.

        Jh stacks the fingertip translational Jacobians of the GRASPING fingers
        (3*n_contacts x n_robot). Left-multiplying by the projector removes exactly
        the component of the tracking torque that would produce fingertip motion,
        leaving everything orthogonal to it untouched.

        Scoped to the FINGER columns (active_joint_slices). The paper notes "this
        projection does not affect the arm torques at all" -- true for them because
        Jh there is the HAND Jacobian and the arm lives outside it. Our J_list is
        built over all n_robot columns, so the arm columns would otherwise be
        projected too, which would fight the resolved-rate jog that carries the
        object. Restricting the projector to the finger columns reproduces their
        intent on our kinematics.
        """
        import numpy as _np
        n = self.n_robot
        cols = _np.concatenate([_np.arange(a, b)
                                for (a, b) in self.active_joint_slices])
        Jh = _np.zeros((3 * len(self.tip_site_ids), len(cols)))
        for k, sid in enumerate(self.tip_site_ids):
            Jk = _np.zeros((3, self.model.nv))
            mj.mj_jacSite(self.model, data, Jk, None, sid)
            Jh[3 * k:3 * k + 3, :] = Jk[:3, cols]
        JhT = Jh.T                                   # (len(cols) x 3*nc)
        P = _np.eye(len(cols)) - JhT @ _np.linalg.pinv(JhT)
        out = _np.array(tau_n, float)
        out[cols] = P @ out[cols]
        return out

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
        # Cone-constrained gamma when enabled: solves for null-space weights that
        # keep EVERY contact compressive and inside its friction cone, rather than
        # sign-flipping an arbitrary SVD basis off one anchor contact. Contact
        # frames put the inward normal in col0, and G's columns are built in those
        # same frames, so the per-contact normal is [1,0,0] by construction.
        #
        # gamma carries the caller's commanded squeeze magnitude. The LP returns a
        # MINIMAL in-cone allocation whose peak normal force is cone_f_min, and
        # gamma says how hard to actually squeeze.
        #
        # Only the INTERNAL (null-space) part may be scaled. f_c splits as
        #
        #     f_c = pinv(G) w_des  +  N gamma_LP
        #           \__________/     \________/
        #            equilibrium       internal
        #
        # and only the second term is wrench-neutral (G N = 0). The first term is
        # what holds the object in equilibrium -- with support_weight it IS the
        # object's weight -- so multiplying the WHOLE vector by gamma/f_min (as
        # this did) commands a net object wrench gamma/f_min times gravity. At the
        # post-budget-change gamma=12 on 036_wood_block that is a 24x amplifier:
        # the commanded support force became 171.6 N on a 7.15 N block and the
        # squeeze ejected it (object z 0.625 -> 0.978 m, tips drifting ~190 mm off
        # the planned grasp, measured contact force 0.0 N because nothing was left
        # to touch). No PD gain can oppose that -- it is a wrench applied to the
        # OBJECT, not a torque the fingers fight: sweeping finger_kp 0.8 -> 80 at
        # gamma=12 left the measured force at exactly 0.0 N throughout.
        #
        # Scaling only the null-space term keeps G f_c == w_des for every gamma, so
        # the equilibrium the allocation was solved for is invariant to the squeeze
        # magnitude, and gamma means "peak INTERNAL normal force (N)" -- a physical
        # unit that does not shift when the disturbance budget changes.
        f_c = None
        if self.cone_gamma:
            _g_cmd = float(np.max(np.atleast_1d(self.allocator.gamma)))
            _gam, _info = self.allocator.solve_gamma_cone(
                G, w_des, contact_dof=3,
                normals=[np.array([1.0, 0.0, 0.0])] * len(contacts),
                mu=self.cone_mu, f_min=self.cone_f_min, margin=self.cone_margin)
            self.last_cone_info = _info
            if _gam is not None:
                _f_eq = np.linalg.pinv(G) @ w_des        # equilibrium part: never scaled
                _f_int = np.asarray(_info['f_c'], float) - _f_eq   # == N @ gamma_LP
                # NORMALISE the internal part to UNIT peak normal force, so gamma
                # multiplies a unit vector and therefore IS the commanded peak
                # internal normal force in newtons -- the contract
                # scripts/3D_minimum_NCF.py defines and solve_gamma_live solves
                # against (it passes ncf=[1.0]*n; see _peak_internal_normal).
                #
                # This replaces dividing by cone_f_min, which assumed
                # |f_int| == cone_f_min. It is not: solve_gamma_cone minimises the
                # PEAK normal force subject to f_min at EVERY contact, so once
                # w_des != 0 the equilibrium part already loads one contact and the
                # internal correction needed to bring the others up to f_min grows
                # with w_des. Measured peak |f_int| against an f_min of 0.5:
                # 0.745 / 0.976 / 2.364 / 4.076 N at object weights 0.49 / 0.95 /
                # 3.73 / 7.15 N. Dividing by f_min therefore under-normalised by a
                # factor that TRACKED OBJECT MASS, inflating gamma 1.5x (014_lemon)
                # to 8.2x (036_wood_block) -- which is why the block commanded ~98 N
                # of internal force at gamma=12 and measured ~55 N where friction
                # needs ~19 N. See tests/test_gamma_is_newtons.py.
                _pk = _peak_internal_normal(_f_int, len(contacts))
                if _pk > 1e-9:
                    f_c = scale * (_f_eq + _g_cmd * (_f_int / _pk))
        if f_c is None:
            # Fall back to the sign-anchor path: either cone_gamma is off, or the
            # LP found no compressive in-cone force for this geometry (which is a
            # real property of the contacts, not a solver failure).
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

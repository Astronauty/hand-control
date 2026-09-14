"""Damped-least-squares IK over the first n_robot joints, for an arbitrary number of
site position targets (planar: xy, spatial: xyz). Generalizes the fixed-2-site,
2D-only `solve_ik` from internal_force_control.py.
"""

import mujoco as mj
import numpy as np


class IKSolver:
    def __init__(self, pos_dim, n_robot, damping=0.01, max_iter=500, step=0.5, tol=1e-3,
                 adaptive_damping=False, lambda_max=0.1, w0=1e-3,
                 selective_damping=False, sigma0=0.05):
        self.pos_dim = pos_dim
        self.n_robot = n_robot
        self.damping = damping
        self.max_iter = max_iter
        self.step = step
        self.tol = tol
        # Singularity-robust (Levenberg-Marquardt) damping. When enabled, the
        # damping factor GROWS as the Yoshikawa manipulability w=sqrt(det(J J^T))
        # drops below w0, keeping dq bounded near singularities instead of letting
        # the near-singular (J J^T)^-1 blow up (the "whipping" in teleop). Away
        # from singularities damping stays at `damping` so tracking stays crisp.
        # Off by default so RRT/grasp callers are unchanged.
        self.adaptive_damping = adaptive_damping
        self.lambda_max = lambda_max
        self.w0 = w0
        # PER-DIRECTION (selective) damping. Damps ONLY the near-singular singular
        # directions instead of every direction equally, via the SVD form
        #     J^+ = sum_i sigma_i / (sigma_i^2 + lambda_i^2) v_i u_i^T
        #     lambda_i = 0                                  if sigma_i > sigma0
        #                lambda_max * (1 - sigma_i/sigma0)^2 otherwise
        #
        # Why this exists. A SCALAR lambda has to be large enough for the WORST
        # direction, so it over-damps every healthy one. Measured on the 3-site
        # seeding Jacobian (9x23) at the wood-block home pose:
        #     sigma = [2.25, .315, .147, .104, .0896, .0847, .0241, .0132, 7.0e-4]
        # eight directions are fine and one is weak, yet lambda=0.01 suppresses all
        # nine (a direction is suppressed when sigma^2 << lambda; here the weakest
        # has sigma^2 = 4.9e-7, four orders below lambda).
        #
        # Two measured consequences of the scalar form, both fixed by damping only
        # the offending axis:
        #   1. Task error on REACHABLE targets: a 20mm move converges to 8.29mm
        #      with damping=0.01/null_gain=0.3, and to 0.06mm at damping=1e-4.
        #      It is a FIXED POINT, not slow convergence -- 10x the iterations
        #      leaves it at 8.29mm.
        #   2. The null-space projector leaks. dq += null_gain*(I - J^+ J)(q_bias-q)
        #      is only orthogonal to the task when J^+ is the TRUE pseudo-inverse;
        #      the damped one makes I - J^+ J a non-projector, so the posture bias
        #      pulls the fingertips off target. Damping only the weak direction
        #      keeps I - J^+ J close to a true projector.
        #
        # sigma0 gates on the SMALLEST SINGULAR VALUE, deliberately not the
        # Yoshikawa w=sqrt(det(J J^T)) that adaptive_damping uses: w is the PRODUCT
        # of the singular values, so (a) it shrinks geometrically with the task
        # dimension -- measured 4.84e-3 / 1.22e-6 / 1.83e-11 for 1/2/3 sites at the
        # same pose, so a single w0 cannot serve different site counts (w0=1e-3
        # fires ALWAYS at 3 sites, making adaptive_damping a flat 11x damping
        # increase rather than an adaptive one) -- and (b) a volume masks a
        # collapsing axis: scaling sigma_min up 100x and sigma_max down 100x leaves
        # w bit-identical while the conditioning that actually matters improves
        # 100-fold. sigma_min has task-space units and does not move with the row
        # count, so one threshold serves 2 and 3 contacts alike.
        #
        # Off by default: selective_damping=False reproduces the scalar path exactly.
        self.selective_damping = selective_damping
        self.sigma0 = sigma0

    def solve(self, model, data, site_ids: list[int], targets: list[np.ndarray],
              orientations: list[np.ndarray] = None, q_bias=None, null_gain=0.1) -> np.ndarray:
        """Object joints beyond n_robot are left unchanged. Returns n_robot-length q.

        Limited joints are clipped to model.jnt_range after each step — without this,
        a heavily redundant chain (e.g. a 23-DOF arm+hand solving a 6D position
        constraint) can wander to joint values far outside their physical range, which
        then makes that "solution" unreachable by anything that respects joint limits
        (e.g. RRTPlanner, which samples only within range). Unlimited joints (continuous
        revolute, e.g. some arm joints) are left unclipped.

        orientations: optional list aligned with site_ids. Each entry is one of:
          - None: position-only (the original 2-fixed-site planar behavior).
          - a 3x3 desired rotation matrix: full SO(3) match, all 3 rotational DOF
            constrained (e_omega = 0.5 * sum(cross(R_cur[:,i], R_des[:,i])), the same
            convention used for object orientation error in the GRASP-phase controller).
          - a (local_axis, world_target) tuple: only constrains *that* site axis to point
            along world_target, leaving roll about it free (e_omega = cross(R_cur @
            local_axis, world_target)). Use this instead of a full matrix when only an
            approach direction matters — full 3-DOF matching forced an arbitrary,
            independently-chosen roll on each of 2 fingertips sharing one wrist, which
            conflicted between fingers and produced floor-penetrating, non-converging
            solutions (found empirically on the Kinova+LEAP pregrasp).

        Without any orientation target, a redundant chain's IK is free to pick *any*
        approach direction that places the fingertip at the right point, including
        reaching in from a direction that makes no sense for a lateral pinch grasp
        (verified visually: position-only IK reached down from above instead of
        flanking the object).

        q_bias: optional secondary objective, projected through the null space of the
        task Jacobian, pulling the redundant solution toward a preferred posture (e.g. a
        "ready pose" with the elbow up) instead of leaving the extra DOF unconstrained.
        On a heavily redundant chain (a 7-DOF arm solving a 6D position task) the
        unconstrained solution can wander into configurations that dip the arm/hand
        through the floor — irrelevant for the original 2D 2-contact case (no redundant
        DOF beyond the task), so this is a no-op by default (q_bias=None) and doesn't
        change planar behavior.
        """
        limited = model.jnt_limited[: self.n_robot].astype(bool)
        lo = model.jnt_range[: self.n_robot, 0]
        hi = model.jnt_range[: self.n_robot, 1]

        q = data.qpos[: self.n_robot].copy()
        for _ in range(self.max_iter):
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            err_blocks, Js = [], []
            for idx, (s, t) in enumerate(zip(site_ids, targets)):
                Jp = np.zeros((3, model.nv))
                Jr = np.zeros((3, model.nv))
                mj.mj_jacSite(model, data, Jp, Jr, s)
                err_blocks.append(t - data.site_xpos[s][: self.pos_dim])
                Js.append(Jp[: self.pos_dim, : self.n_robot])
                if orientations is not None and orientations[idx] is not None:
                    spec = orientations[idx]
                    R_cur = data.site_xmat[s].reshape(3, 3)
                    if isinstance(spec, tuple):
                        local_axis, world_target = spec
                        v_cur = R_cur @ local_axis
                        v_des = world_target / np.linalg.norm(world_target)
                        e_omega = np.cross(v_cur, v_des)
                    else:
                        R_des = spec
                        e_omega = 0.5 * sum(np.cross(R_cur[:, i], R_des[:, i]) for i in range(3))
                    err_blocks.append(e_omega)
                    Js.append(Jr[:, : self.n_robot])
            err = np.concatenate(err_blocks)
            if np.linalg.norm(err) < self.tol:
                break
            J = np.vstack(Js)
            JJt = J @ J.T
            lam = self.damping
            if self.adaptive_damping:
                # Yoshikawa manipulability; det(JJt) >= 0. Ramp damping up
                # quadratically once w drops below the threshold w0.
                w = float(np.sqrt(max(np.linalg.det(JJt), 0.0)))
                if w < self.w0:
                    ratio = 1.0 - w / self.w0
                    lam = self.damping + self.lambda_max * ratio * ratio
            if self.selective_damping:
                # Per-direction damping: build the inverse from the SVD so each
                # singular direction gets its OWN lambda_i, zero for the healthy
                # ones. Equivalent to the scalar form when every sigma_i <= sigma0.
                U, sv, Vt = np.linalg.svd(J, full_matrices=False)
                lam_i = np.where(
                    sv > self.sigma0, 0.0,
                    self.lambda_max * (1.0 - sv / self.sigma0) ** 2)
                # sigma/(sigma^2 + lambda^2); guard the exactly-zero-sigma case,
                # where the gain is 0 (that direction is unreachable, not infinite).
                denom = sv ** 2 + lam_i ** 2
                gain = np.divide(sv, denom, out=np.zeros_like(sv), where=denom > 0)
                J_pinv_damped = (Vt.T * gain) @ U.T
            else:
                J_pinv_damped = J.T @ np.linalg.inv(JJt + lam * np.eye(J.shape[0]))
            dq = J_pinv_damped @ err
            if q_bias is not None:
                null_proj = np.eye(self.n_robot) - J_pinv_damped @ J
                dq += null_gain * (null_proj @ (q_bias - q))
            q += self.step * dq
            q[limited] = np.clip(q[limited], lo[limited], hi[limited])
            data.qpos[: self.n_robot] = q
        return q


class PlanarIKSolver(IKSolver):
    def __init__(self, n_robot, **kwargs):
        super().__init__(pos_dim=2, n_robot=n_robot, **kwargs)


class SpatialIKSolver(IKSolver):
    def __init__(self, n_robot, **kwargs):
        super().__init__(pos_dim=3, n_robot=n_robot, **kwargs)

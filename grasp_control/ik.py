"""Damped-least-squares IK over the first n_robot joints, for an arbitrary number of
site position targets (planar: xy, spatial: xyz). Generalizes the fixed-2-site,
2D-only `solve_ik` from internal_force_control.py.
"""

import mujoco as mj
import numpy as np


class IKSolver:
    def __init__(self, pos_dim, n_robot, damping=0.01, max_iter=500, step=0.5, tol=1e-3):
        self.pos_dim = pos_dim
        self.n_robot = n_robot
        self.damping = damping
        self.max_iter = max_iter
        self.step = step
        self.tol = tol

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
            J_pinv_damped = J.T @ np.linalg.inv(J @ J.T + self.damping * np.eye(J.shape[0]))
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

"""VWJ retargeting objective + analytic gradient (clean-room, arXiv:2506.09384).

One scalar cost over the optimized joints q (7 arm + 16 hand), with an nlopt-style
`(x, grad) -> float` signature. Four Huber terms (beta = huber_delta):

    L = links_vec_cost + wrist_rot_cost + joint_pos_cost + joint_vel_cost

  links_vec : for each robot link-pair (origin->task), match the pair's world
              vector to the human ref vector, weighted per pair (the ref + weights
              come from vwj/refvalues.build_ref_values).
  wrist_rot : squared quaternion angular error of the wrist site vs the human wrist.
  joint_pos : per-joint regularization toward a rest posture.
  joint_vel : per-joint smoothing vs the previous solved q.

Gradient: the link-pair and wrist terms go through the MuJoCo site Jacobians
(vwj.kinematics); the joint terms are direct in q. Verified against finite
differences in the module test.
"""
from __future__ import annotations

import numpy as np

from vwj.kinematics import MujocoKinematics


def _huber(r: np.ndarray, beta: float):
    """Elementwise Huber h(r) and its derivative h'(r) (SmoothL1 with transition beta).
    Matches torch.nn.SmoothL1Loss(beta): quadratic |r|<beta, linear beyond."""
    r = np.asarray(r, float)
    a = np.abs(r)
    quad = a < beta
    h = np.where(quad, 0.5 * r * r / beta, a - 0.5 * beta)
    dh = np.where(quad, r / beta, np.sign(r))
    return h, dh


def _quat_angular_error(q_ref, q_cur):
    """Angle (rad) between two wxyz quaternions: 2*acos(|<q_ref,q_cur>|)."""
    d = abs(float(np.dot(q_ref, q_cur)))
    d = min(1.0, max(-1.0, d))
    return 2.0 * np.arccos(d)


class VWJObjective:
    """Holds the static config; get_objective(ref) returns a per-frame nlopt callback."""

    def __init__(self, kin: MujocoKinematics, link_pairs: list[tuple[str, str]],
                 huber_delta: float = 0.02, wrist_rot_weight: float = 0.1,
                 joint_pos_weight: np.ndarray = None, joint_vel_weight: np.ndarray = None):
        self.kin = kin
        self.beta = float(huber_delta)
        self.w_wrist_rot = float(wrist_rot_weight)
        n = kin.n_opt
        self.w_joint_pos = (np.zeros(n) if joint_pos_weight is None
                            else np.asarray(joint_pos_weight, float))
        self.w_joint_vel = (np.zeros(n) if joint_vel_weight is None
                            else np.asarray(joint_vel_weight, float))

        # Resolve each link-pair's origin/task to a site index into kin.site_names, or
        # -1 for the special "world" origin (fixed point at the world origin, pos 0).
        names = kin.site_names
        self.origin_idx = np.array(
            [-1 if o == "world" else names.index(o) for o, _ in link_pairs], int)
        self.task_idx = np.array([names.index(t) for _, t in link_pairs], int)

    def get_objective(self, ref_link_vec: np.ndarray, ref_weights: np.ndarray,
                      ref_wrist_quat: np.ndarray, q_rest: np.ndarray, q_last: np.ndarray):
        ref_vec = np.asarray(ref_link_vec, float)
        w_vec = np.asarray(ref_weights, float)
        q_rest = np.asarray(q_rest, float)
        q_last = np.asarray(q_last, float)
        kin = self.kin
        beta = self.beta

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            need_grad = grad.size > 0
            if need_grad:
                kin.forward_with_jac(x)
            else:
                kin.forward(x)
            pos = kin.site_positions()                       # (n_site,3)
            wquat = kin.wrist_quat()

            # ---- link-vector term ----
            origin_pos = np.where(self.origin_idx[:, None] < 0, 0.0,
                                  pos[np.clip(self.origin_idx, 0, None)])
            task_pos = pos[self.task_idx]
            cur_vec = task_pos - origin_pos                   # (M,3)
            diff = cur_vec - ref_vec                          # (M,3)
            dist = np.linalg.norm(diff, axis=1)               # (M,)
            wdist = w_vec * dist
            h, dh = _huber(wdist, beta)
            links_cost = float(np.sum(h))

            # ---- wrist orientation term ----
            ang = _quat_angular_error(ref_wrist_quat, wquat)
            wrist_cost = self.w_wrist_rot * ang * ang

            # ---- joint pos / vel terms (Huber, per-joint weighted) ----
            jp = self.w_joint_pos * (x - q_rest)
            jv = self.w_joint_vel * (x - q_last)
            hjp, dhjp = _huber(jp, beta)
            hjv, dhjv = _huber(jv, beta)
            joint_pos_cost = float(np.sum(hjp))
            joint_vel_cost = float(np.sum(hjv))

            total = links_cost + wrist_cost + joint_pos_cost + joint_vel_cost

            if need_grad:
                Jp, Jr = kin.site_jacobians()                 # (n_site,3,n_opt) each
                g = np.zeros(kin.n_opt)

                # link-vec gradient: d/dq [ sum_m h(w_m ||diff_m||) ]
                #   = sum_m h'(w_m d_m) * w_m * (diff_m/d_m) . d(cur_vec_m)/dq
                # d(cur_vec_m)/dq = Jp[task_m] - Jp[origin_m]  (origin=world -> 0).
                safe = dist > 1e-9
                for m in range(len(self.task_idx)):
                    if not safe[m] or w_vec[m] == 0.0:
                        continue
                    unit = diff[m] / dist[m]                  # (3,)
                    Jtask = Jp[self.task_idx[m]]              # (3,n_opt)
                    Jvec = Jtask if self.origin_idx[m] < 0 else Jtask - Jp[self.origin_idx[m]]
                    g += dh[m] * w_vec[m] * (unit @ Jvec)

                # wrist-rot gradient: d/dq [ w_wr * ang^2 ] = 2 w_wr ang * d(ang)/dq.
                # d(ang)/dq maps the wrist angular Jacobian Jr[wrist] through the
                # geodesic-angle derivative. For small residual angles ang≈|e_omega|,
                # and d(ang)/dq ≈ (axis) . Jr[wrist], with axis the unit rotation-error
                # vector from ref to current. Use a finite, stable form.
                if self.w_wrist_rot > 0.0 and ang > 1e-6:
                    axis = _wrist_error_axis(ref_wrist_quat, wquat)   # (3,), world frame
                    Jw = Jr[kin.wrist_idx]                             # (3,n_opt)
                    g += 2.0 * self.w_wrist_rot * ang * (axis @ Jw)

                # joint terms (direct in q).
                g += dhjp * self.w_joint_pos
                g += dhjv * self.w_joint_vel

                grad[:] = g

            return total

        return objective


def _wrist_error_axis(q_ref, q_cur):
    """Unit axis (world frame) of the rotation taking q_cur -> q_ref, i.e. the
    direction increasing the current wrist angle reduces the error. Returns the axis
    of q_err = q_ref * conj(q_cur), expressed so that (axis . omega) is d(angle)/dt."""
    # q_err = q_ref ⊗ q_cur^{-1}  (wxyz)
    w1, x1, y1, z1 = q_ref
    w2, x2, y2, z2 = q_cur
    # conj(q_cur) = (w2, -x2, -y2, -z2)
    cw, cx, cy, cz = w2, -x2, -y2, -z2
    ew = w1 * cw - x1 * cx - y1 * cy - z1 * cz
    ex = w1 * cx + x1 * cw + y1 * cz - z1 * cy
    ey = w1 * cy - x1 * cz + y1 * cw + z1 * cx
    ez = w1 * cz + x1 * cy - y1 * cx + z1 * cw
    v = np.array([ex, ey, ez])
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3)
    axis = v / n
    # sign: moving current toward ref DEcreases angle, so the gradient axis is -axis
    # (the world-frame angular velocity that reduces the error). Return -axis so that
    # (axis . Jw . dq) with a descent step reduces ang.
    return -axis

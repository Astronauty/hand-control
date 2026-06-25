"""Allocate a desired object wrench to stacked contact forces:

    f_c = pinv(G) @ w_des + null(G) @ gamma

Dimension- and contact-count-agnostic: works for any grasp map G (3xM planar or 6xM
spatial) and any null-space dimension k (k > 1 once there are more contacts than the
minimum needed for force closure, e.g. 3+ fingers). Generalizes the 2-contact,
1D-null-space, single-scalar-gamma logic in internal_force_control.py.
"""

import numpy as np
import scipy.linalg


class GraspForceAllocator:
    def __init__(self, gamma):
        """gamma: scalar (applied to every null-space direction) or a sequence with one
        weight per null-space basis vector."""
        self.gamma = np.atleast_1d(np.asarray(gamma, dtype=float))

    def allocate(self, G, w_des, contact_dof, inward_dirs=None):
        """
        Args:
            G: grasp map, w_o = G @ f_c.
            w_des: desired object wrench (length G.shape[0]).
            contact_dof: per-contact force dimension (2 planar PCWF, 3 spatial PCWF).
            inward_dirs: optional list, one entry per contact (or None for contacts that
                shouldn't anchor a sign check), giving the compressive direction in that
                contact's own frame. Used to orient each null-space basis vector so the
                internal force squeezes rather than pulls apart — generalizes the single
                dot-product sign flip in internal_force_control.py to N contacts and to
                a null space of any dimension (each basis vector is independently
                sign-corrected using the first available anchor contact).
        """
        null = scipy.linalg.null_space(G)  # (n_contacts * contact_dof, k)

        if inward_dirs is not None and null.shape[1] > 0:
            null = null.copy()
            for j in range(null.shape[1]):
                for i, d in enumerate(inward_dirs):
                    if d is None:
                        continue
                    seg = null[i * contact_dof : (i + 1) * contact_dof, j][: len(d)]
                    if np.dot(seg, d) < 0:
                        null[:, j] *= -1
                    break

        if null.shape[1] == 0:
            f_internal = np.zeros(G.shape[1])
        else:
            gamma = self.gamma if len(self.gamma) == null.shape[1] else np.full(null.shape[1], self.gamma[0])
            f_internal = null @ gamma

        return np.linalg.pinv(G) @ w_des + f_internal

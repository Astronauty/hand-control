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

    def solve_gamma_cone(self, G, w_des, contact_dof, normals, mu,
                         f_min=0.5, f_max=None, margin=0.0):
        """Pick null-space weights gamma so the ALLOCATED force is compressive and
        inside every friction cone -- at EVERY contact, not just an anchor one.

        f_c(gamma) = pinv(G) @ w_des + N @ gamma,  N = null(G), gamma in R^k.

        The sign-anchor approach this replaces only ever fixed the ORIENTATION of
        each basis vector, using the first non-None inward_dirs entry (see
        allocate(): the loop breaks after that contact). With 2 antipodal contacts
        null(G) is 1-D, so one sign is the whole answer and it works. With 3
        contacts the null space is 3-D and a uniform gamma over an arbitrary SVD
        basis lands anywhere in it -- nothing holds the non-anchor contacts
        compressive, let alone inside their cones.

        Solved as an LP in (gamma, t) -- linear because f_c is affine in gamma and
        the cone is imposed as a polyhedral (n_side-facet) approximation:

            min  t                                    (t bounds the normal forces)
            s.t. f_k . n_k          >= f_min          compressive, per contact
                 f_k . n_k          <= t              t is the peak normal force
                 f_k . e_ki         <= mu_eff * f_k . n_k   per facet i, per contact
                 t                  <= f_max          optional budget cap

        where e_ki are n_side tangent directions spanning contact k's tangent
        plane. The pyramid is INSCRIBED in the true cone (mu_eff = mu*cos(pi/n_side))
        so a solution is always feasible for the real cone, never merely for the
        approximation.

        normals : list of per-contact INWARD unit normals, expressed in that
            contact's own frame (the same frame G's columns use). For the contact
            frames this repo builds, col0 of R is the inward normal, so the normal
            in-frame is [1,0,0].
        mu      : scalar or per-contact friction coefficient.
        f_min   : minimum normal force per contact (N). Must be > 0 or the
            all-zeros gamma trivially satisfies everything.
        margin  : shrink factor on mu, in [0,1), for safety against the friction
            estimate. mu_used = mu * (1 - margin).

        Returns (gamma, info). gamma is None when no compressive in-cone force
        exists -- that is a real property of the contact geometry (e.g. three
        nearly-parallel normals cannot squeeze), and the caller should treat it as
        "this grasp cannot be executed", not fall back to an arbitrary gamma.
        """
        import scipy.optimize

        N = scipy.linalg.null_space(G)
        k = N.shape[1]
        M = G.shape[1]
        n_c = M // contact_dof
        if k == 0:
            return None, {'reason': 'null space is empty; no internal force exists'}

        f0 = np.linalg.pinv(G) @ w_des            # particular solution
        mus = np.full(n_c, float(mu)) if np.isscalar(mu) else np.asarray(mu, float)
        n_side = 8
        # Inscribed pyramid: a facet normal at angle pi/n_side from a sampled
        # tangent under-approximates the circular cone, so feasibility here implies
        # feasibility for the true cone.
        mu_eff = mus * np.cos(np.pi / n_side) * (1.0 - float(margin))

        rows_ub, rhs_ub = [], []
        nvar = k + 1                               # [gamma (k), t]
        for c in range(n_c):
            sl = slice(c * contact_dof, (c + 1) * contact_dof)
            n_hat = np.asarray(normals[c], float)[:contact_dof]
            n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
            # normal force as an affine function of gamma: a . gamma + b
            a_n = n_hat @ N[sl, :]
            b_n = float(n_hat @ f0[sl])
            # compressive: -(a.g + b) <= -f_min
            r = np.zeros(nvar); r[:k] = -a_n; rows_ub.append(r); rhs_ub.append(b_n - f_min)
            # t bounds it: (a.g + b) - t <= 0
            r = np.zeros(nvar); r[:k] = a_n; r[k] = -1.0
            rows_ub.append(r); rhs_ub.append(-b_n)
            # friction facets, spanning contact c's tangent plane
            ref = np.array([0.0, 0.0, 1.0]) if abs(n_hat[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
            t1 = np.cross(n_hat, ref); t1 /= (np.linalg.norm(t1) + 1e-12)
            t2 = np.cross(n_hat, t1)
            for i in range(n_side):
                ang = 2.0 * np.pi * i / n_side
                e = np.cos(ang) * t1 + np.sin(ang) * t2
                a_t = e @ N[sl, :]
                b_t = float(e @ f0[sl])
                # f.e - mu_eff * f.n <= 0
                r = np.zeros(nvar); r[:k] = a_t - mu_eff[c] * a_n
                rows_ub.append(r); rhs_ub.append(-(b_t - mu_eff[c] * b_n))

        cost = np.zeros(nvar); cost[k] = 1.0       # minimize peak normal force
        bounds = [(None, None)] * k + [(0.0, float(f_max) if f_max else None)]
        res = scipy.optimize.linprog(cost, A_ub=np.array(rows_ub), b_ub=np.array(rhs_ub),
                                     bounds=bounds, method='highs')
        if not res.success:
            return None, {'reason': f'LP infeasible: {res.message}', 'status': res.status}
        gamma = res.x[:k]
        f_c = f0 + N @ gamma
        fn = []
        for c in range(n_c):
            sl = slice(c * contact_dof, (c + 1) * contact_dof)
            n_hat = np.asarray(normals[c], float)[:contact_dof]
            n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
            fn.append(float(n_hat @ f_c[sl]))
        return gamma, {'f_c': f_c, 'normal_forces': fn, 'peak_normal': float(res.x[k]),
                       'null_dim': k, 'n_contacts': n_c}

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

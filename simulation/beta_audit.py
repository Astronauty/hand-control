"""True-normal recomputation of the min-weight metric beta, for auditing the NLP's
reported beta against the geometry it actually solved.

Why this exists
---------------
`_embed_gws_ca` embeds the FRoGGeR min-weight LP as NLP variables, so the beta it
reports is the LP's optimum *on the wrench matrix the NLP built* -- which is assembled
from the QUADRATIC PATCH's symbolic normals, not from the object's true surface. The
planner docstring records the consequence: on a 35-solve tabletop sweep, 6 solves (17%)
reported beta > 0.01 on contacts the wrench certificate rejects as infeasible, and on
061_foam_brick a reported beta = +0.0996 sat on a grasp whose true normals span
76.3 deg against a 61.4 deg friction limit -- force closure geometrically impossible.

This module answers "what would beta be on the TRUE normals at the SAME solved points?"
It rebuilds W from `_geom_normal_np` (the mesh/primitive surface normal) and solves the
same LP in numpy, so reported-vs-true is a like-for-like comparison of one quantity.

It is deliberately standalone and read-only: it imports the planner's own primitives
(_friction_cone_verts, _geom_normal_np, _build_contact_frame_3d, _span_margin) rather
than reimplementing the cone or the frame convention, so it cannot drift from what the
NLP means by a "primitive wrench". Nothing here is on the solve path.

Generalizes to n contacts, so it reads a tripod's beta correctly -- unlike verify(),
which is still hardcoded n=2.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from simulation.grasp_planner_3d import (
    _build_contact_frame_3d,
    _friction_cone_verts,
    _geom_normal_np,
    _span_margin,
)


def build_W_np(points, normals_out, obj_center, obj_R, mu, mu_t: float = 0.0):
    """Primitive wrench matrix W (6 x n*s) in the OBJECT BODY frame, from true normals.

    Mirrors build_W_ca's conventions exactly -- same _friction_cone_verts, same
    contact frame (col0 = INWARD normal), same [tau; f] row order, same object-frame
    change of basis -- but numerically, at fixed points, with no CasADi.

    points      : [(3,)] contact positions, WORLD frame.
    normals_out : [(3,)] OUTWARD surface normals at those points, WORLD frame.
    obj_center  : (3,) object origin, world frame.
    obj_R       : (3,3) object->world rotation.
    mu          : sliding friction coefficient.
    mu_t        : soft-finger torsional coefficient; > 0 adds 2 columns per contact.
                  Left at 0.0 (PCwF, s=5) to match the standardized GWS preset.
    """
    # 4-sided (no origin row), matching build_W_ca -- the audit must build the SAME
    # W the NLP does, and a zero column would let the LP report beta = 0 spuriously.
    verts_c = _friction_cone_verts(mu, include_origin=False)
    R_ow = np.asarray(obj_R, float).T          # world -> object body frame
    c = np.asarray(obj_center, float)
    cols = []
    for p, n_out in zip(points, normals_out):
        p = np.asarray(p, float)
        n_out = np.asarray(n_out, float)
        n_in = -n_out / (np.linalg.norm(n_out) + 1e-12)
        _, t1, t2 = _build_contact_frame_3d(n_in)
        R = np.column_stack([n_in, t1, t2])    # col0 = inward normal (build_W_ca's R_param)
        p_O = R_ow @ (p - c)
        for v in verts_c:
            f_O = R_ow @ (R @ np.asarray(v, float))
            cols.append(np.concatenate([np.cross(p_O, f_O), f_O]))
        if mu_t > 0.0:
            # Soft-finger torsion columns: pure normal force with +/- spin moment.
            n_O = R_ow @ n_in
            f_O = R_ow @ (R @ np.array([1.0, 0.0, 0.0]))
            for s in (+1.0, -1.0):
                cols.append(np.concatenate([np.cross(p_O, f_O) + s * mu_t * n_O, f_O]))
    return np.column_stack(cols)


def min_weight_beta(W) -> tuple[float, np.ndarray | None]:
    """Solve the FRoGGeR min-weight LP on a fixed W and return (beta, alpha).

        max_{alpha, beta}  beta   s.t.  W alpha = 0,  sum(alpha) = 1,  alpha >= beta

    alpha is FREE (not >= 0), matching _embed_gws_ca: out of closure the correct
    answer is beta < 0 with some alpha_j < 0, not infeasibility.

    Variables are packed as x = [alpha (n_cols); beta (1)]. The `alpha >= beta*1`
    constraint becomes `beta - alpha_j <= 0` in linprog's `A_ub x <= b_ub` form.
    Returns (nan, None) if the LP fails to solve.
    """
    W = np.asarray(W, float)
    n = W.shape[1]
    c = np.zeros(n + 1)
    c[-1] = -1.0                                  # maximize beta == minimize -beta
    A_eq = np.zeros((7, n + 1))
    A_eq[:6, :n] = W                              # W alpha = 0
    A_eq[6, :n] = 1.0                             # sum(alpha) = 1
    b_eq = np.zeros(7)
    b_eq[6] = 1.0
    A_ub = np.hstack([-np.eye(n), np.ones((n, 1))])   # beta - alpha_j <= 0
    b_ub = np.zeros(n)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(None, None)] * (n + 1), method="highs")
    if not res.success:
        return float("nan"), None
    return float(res.x[-1]), res.x[:n]


def audit(result: dict, *, geom_type: int, obj_center, obj_R, obj_size,
          mesh_entry=None, mu: float, mu_t: float = 0.0) -> dict:
    """Recompute beta on the TRUE surface normals at a solve's own contact points.

    result : a GraspPlanner3D.solve() dict (needs 'p1','p2', optionally 'p3' and
             the reported 'beta'/'gws_beta' if present).

    Returns a dict with:
      n_contacts     -- how many contacts entered W (2 or 3)
      beta_true      -- min-weight metric on true normals, at the solved points
      beta_reported  -- what the NLP reported, if the result carries it
      beta_delta     -- reported - true (positive = the NLP was optimistic)
      n_cols         -- columns in W (beta's arithmetic ceiling is 1/n_cols)
      beta_true_scaled / beta_reported_scaled -- beta * n_cols, the budget-invariant
                        form (1.0 == saturating the ceiling), comparable across
                        n_contacts and soft-finger settings
      span_margin    -- pure geometric closure test on the true normals (n=2 only;
                        positive = closure achievable at this mu). None at n>=3,
                        where a pairwise opposition test is not the right question.
      normals        -- the true outward normals used
      contradiction  -- True when beta_reported > 0.01 but the geometry says closure
                        is impossible (beta_true <= 0, or span_margin < 0). This is
                        the 17%-of-solves failure mode the planner docstring records.
    """
    pts = [np.asarray(result[k], float)
           for k in ("p1", "p2", "p3")
           if result.get(k) is not None]
    if len(pts) < 2:
        return {}
    normals = [_geom_normal_np(p, geom_type, obj_center, obj_R, obj_size,
                               mesh_entry=mesh_entry) for p in pts]
    W = build_W_np(pts, normals, obj_center, obj_R, mu, mu_t=mu_t)
    beta_true, alpha = min_weight_beta(W)
    n_cols = W.shape[1]

    reported = result.get("beta", result.get("gws_beta"))
    reported = float(reported) if reported is not None else None

    sm = _span_margin(normals[0], normals[1], mu) if len(pts) == 2 else None

    contradiction = bool(
        reported is not None and reported > 0.01
        and (not np.isfinite(beta_true) or beta_true <= 0.0
             or (sm is not None and sm < 0.0)))

    return {
        "n_contacts": len(pts),
        "n_cols": n_cols,
        "beta_true": beta_true,
        "beta_reported": reported,
        "beta_delta": (reported - beta_true)
                      if (reported is not None and np.isfinite(beta_true)) else None,
        "beta_true_scaled": beta_true * n_cols if np.isfinite(beta_true) else float("nan"),
        "beta_reported_scaled": reported * n_cols if reported is not None else None,
        "span_margin": sm,
        "points": [np.asarray(p, float).tolist() for p in pts],
        "sdf_mm": [None] * len(pts),   # filled by the caller when it has the geom args
        "normals": [n.tolist() for n in normals],
        "alpha_min": float(np.min(alpha)) if alpha is not None else None,
        "contradiction": contradiction,
    }


def audit_embedded_lp(result: dict) -> dict:
    """Re-solve the NLP's OWN min-weight LP on its OWN wrench matrix.

    Separates two failure modes that both present as a reported beta disagreeing with
    the geometry:

      * SURROGATE error -- W is assembled from quadratic-patch normals, which differ
        from the surface normal. `audit()` above measures this by rebuilding W from
        true normals.
      * STOPPING artifact -- alpha and beta are IPOPT decision variables inside the
        main NLP, not a nested LP solved to optimality. On a best-effort exit the
        reported beta is whatever the final iterate held, which need not be the
        optimum of even its own W.

    This function isolates the second: it takes `result['gws_W']` unchanged (patch
    normals and all) and solves the min-weight LP on it to optimality. Any difference
    from `result['gws_beta']` is attributable to the NLP not converging its own
    embedded LP, with the surrogate held fixed by construction.

    Returns {} when the result carries no W (w_gws == 0, or the stage built no GWS
    block).

    Keys:
      beta_embedded  -- what the NLP reported (result['gws_beta'])
      beta_relp      -- the same LP re-solved to optimality on the same W
      lp_gap         -- beta_relp - beta_embedded; > 0 means the NLP stopped short
                        of its own LP's optimum
      resid_Walpha   -- ||W @ alpha|| at the NLP's alpha. The LP constrains this to
                        0, so a nonzero value is direct evidence the equality is
                        unconverged rather than merely the objective being loose.
      resid_sum      -- |sum(alpha) - 1| at the NLP's alpha, same reasoning.
      alpha_min_gap  -- min(alpha) - beta at the NLP's values. The LP constrains
                        alpha >= beta*1 and beta is maximized, so at optimality the
                        smallest alpha equals beta and this is 0.
    """
    W = result.get("gws_W")
    if W is None:
        return {}
    W = np.asarray(W, float)
    if W.ndim != 2 or W.shape[0] != 6:
        return {}
    beta_relp, _ = min_weight_beta(W)
    beta_emb = result.get("gws_beta")
    beta_emb = float(beta_emb) if beta_emb is not None else None

    out = {
        "beta_embedded": beta_emb,
        "beta_relp": beta_relp,
        "lp_gap": (beta_relp - beta_emb)
                  if (beta_emb is not None and np.isfinite(beta_relp)) else None,
        "n_cols": int(W.shape[1]),
    }
    a = result.get("gws_alpha")
    if a is not None:
        a = np.asarray(a, float).flatten()
        if a.size == W.shape[1]:
            out["resid_Walpha"] = float(np.linalg.norm(W @ a))
            out["resid_sum"] = float(abs(a.sum() - 1.0))
            out["alpha_min_gap"] = (float(a.min() - beta_emb)
                                    if beta_emb is not None else None)
    return out

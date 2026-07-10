"""
Collision-aware IK via CasADi + IPOPT.

Formulation:
    minimize_q  Σ_k ||site_pos_k(q) - target_k||²  +  posture_weight * ||q - q_bias||²
    subject to:
        q_lo[i] <= q[i] <= q_hi[i]                          (joint limits, hard)
        sphere_box_distance(arm_geom_i, obj_geom_j, q) >= d_min  (collision avoidance, hard)

FK callbacks are CasADi external Callbacks with enable_fd=True so CasADi computes
finite-difference Jacobians for them. Collision avoidance uses a smooth analytic
sphere-vs-box/plane/cylinder SDF (not mj_geomDistance) so CasADi can differentiate
the distance expressions via its own AD after chaining through the FK callback Jacobians.

IPOPT is run with jacobian_approximation="finite-difference-values", meaning IPOPT
computes its own FD of the full NLP (not just the callbacks). This smooths kinks at
face/edge/vertex transitions in the box SDF that would otherwise break L-BFGS.

The companion Jacobian callbacks (_SitePositionJacCallback etc.) are retained here as
candidate analytic-Jacobian implementations. They are NOT currently used (verified
numerically via check_analytic_jacobians()). Enabling them requires switching to
jacobian_approximation="exact" in the solver options AND verifying no kink issues.
"""
import itertools
import time

import casadi as ca
import mujoco as mj
import numpy as np

_call_counter = itertools.count()


def _skew(v):
    """3-vector → 3×3 skew-symmetric matrix satisfying skew(v) @ w == v × w."""
    return np.array([[ 0.,    -v[2],  v[1]],
                     [ v[2],  0.,    -v[0]],
                     [-v[1],  v[0],  0.  ]])


# ---------------------------------------------------------------------------
# Analytic-Jacobian companion callbacks (NOT currently wired up)
# ---------------------------------------------------------------------------
# These implement the Jacobian of each FK callback analytically via mj_jacSite /
# mj_jac. They are NOT enabled in the main callbacks because:
#   1. _sphere_box_distance has kinks at face/edge/vertex transitions; CasADi's
#      symbolic AD exposes them, while IPOPT's own FD smooths them — so
#      jacobian_approximation="finite-difference-values" converges better here.
#   2. The has_jacobian()/get_jacobian() API contract has not been validated end-
#      to-end against the CasADi version in use (check_analytic_jacobians() below).
#
# To enable: (a) fix the kinks (smooth fabs/fmax in _sphere_box_distance),
# (b) verify check_analytic_jacobians() passes, (c) restore has_jacobian()/
# get_jacobian() on the main callbacks, (d) switch solver to "exact".
# ---------------------------------------------------------------------------

class _SitePositionJacCallback(ca.Callback):
    """d(site_xpos)/dq via mj_jacSite — companion for _SitePositionCallback."""

    def __init__(self, name, model, site_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._sid   = site_id
        self._n     = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {})

    def get_n_in(self):  return 2   # q, nominal_output (ignored)
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self._n, 1) if i == 0 else ca.Sparsity.dense(3, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, self._n)
    def has_jacobian(self): return False

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        mj.mj_comPos(self._model, self._data)   # needed: populates cdof for mj_jacSite
        jacp = np.zeros((3, self._model.nv))
        mj.mj_jacSite(self._model, self._data, jacp, None, self._sid)
        return [jacp[:, :self._n]]


class _SiteAxisJacCallback(ca.Callback):
    """d(R @ local_axis)/dq via angular Jacobian — companion for _SiteAxisCallback.

    Chain rule: d(R @ v)/dq_j = ω_j × (R @ v)  where ω_j = jacr[:, j].
    In matrix form: J = -skew(R @ v) @ jacr[:, :n_robot].
    """

    def __init__(self, name, model, site_id, local_axis, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model     = model
        self._data      = mj.MjData(model)
        self._sid       = site_id
        self._localaxis = np.asarray(local_axis, dtype=float).flatten()
        self._n         = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {})

    def get_n_in(self):  return 2
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self._n, 1) if i == 0 else ca.Sparsity.dense(3, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, self._n)
    def has_jacobian(self): return False

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        mj.mj_comPos(self._model, self._data)   # needed: populates cdof for mj_jacSite
        R          = self._data.site_xmat[self._sid].reshape(3, 3)
        axis_world = R @ self._localaxis
        jacr       = np.zeros((3, self._model.nv))
        mj.mj_jacSite(self._model, self._data, None, jacr, self._sid)
        return [-_skew(axis_world) @ jacr[:, :self._n]]


class _GeomPositionJacCallback(ca.Callback):
    """d(geom_xpos)/dq via mj_jacGeom — companion for _GeomPositionCallback."""

    def __init__(self, name, model, geom_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._gid   = geom_id
        self._n     = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {})

    def get_n_in(self):  return 2
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self._n, 1) if i == 0 else ca.Sparsity.dense(3, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, self._n)
    def has_jacobian(self): return False

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        mj.mj_comPos(self._model, self._data)   # needed: populates cdof for mj_jac
        jacp    = np.zeros((3, self._model.nv))
        pt      = self._data.geom_xpos[self._gid].copy()
        body_id = int(self._model.geom_bodyid[self._gid])
        mj.mj_jac(self._model, self._data, jacp, None, pt, body_id)
        return [jacp[:, :self._n]]


# ---------------------------------------------------------------------------
# Main FK callbacks (function evaluation only — Jacobians delegated above)
# ---------------------------------------------------------------------------

class _SitePositionCallback(ca.Callback):
    """q[:n_robot] → site_xpos[site_id] (3-vector) via MuJoCo FK."""

    def __init__(self, name, model, site_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model    = model
        self._data     = mj.MjData(model)
        self._sid      = site_id
        self._n        = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, 1)

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        return [self._data.site_xpos[self._sid].copy()]


class _SiteAxisCallback(ca.Callback):
    """q[:n_robot] → site_R(q) @ local_axis (world-frame 3-vector) via MuJoCo FK."""

    def __init__(self, name, model, site_id, local_axis, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model     = model
        self._data      = mj.MjData(model)
        self._sid       = site_id
        self._localaxis = np.asarray(local_axis, dtype=float).flatten()
        self._n         = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, 1)

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        R = self._data.site_xmat[self._sid].reshape(3, 3)
        return [R @ self._localaxis]



class _GeomPositionCallback(ca.Callback):
    """q[:n_robot] → geom_xpos[geom_id] (3-vector) via MuJoCo FK. Used to feed an
    arm geom's sphere center into the analytic sphere-vs-box distance formula below."""

    def __init__(self, name, model, geom_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model    = model
        self._data     = mj.MjData(model)
        self._gid      = geom_id
        self._n        = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, 1)

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        return [self._data.geom_xpos[self._gid].copy()]


class _BatchedGeomPositionCallback(ca.Callback):
    """q[:n_robot] → stacked world positions (3G vector) of G geoms via ONE MuJoCo FK.

    Replaces G per-geom _GeomPositionCallback instances in solve(): one Python↔CasADi
    crossing and one mj_kinematics per NLP evaluation instead of G of each. Profiled on
    the pick-place grasp solve (71 geoms), the per-geom form spent ~30% of solve time on
    callback dispatch + DM↔numpy conversion (~75k crossings); batching cuts that to ~1k
    crossings with identical iterates."""

    def __init__(self, name, model, geom_ids, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._gids  = list(geom_ids)
        self._n     = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3 * len(self._gids), 1)

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        return [self._data.geom_xpos[self._gids].reshape(-1)]


def _softplus(x, alpha=500.0):
    """Smooth approximation to max(x, 0), element-wise on a CasADi expression.

    softplus(x, α) = max(x,0) + log(1 + exp(−α|x|)) / α

    This is C∞ everywhere (gradient = sigmoid 1/(1+exp(−αx))).  Max error
    vs true max(x,0) is log(2)/α ≈ 1.4 mm at α=500 — acceptable against
    a 5 mm clearance budget.  Error decays exponentially: at ±5 mm < 0.2 mm.

    Using the max(x,0)+correction form keeps both tails numerically stable:
    exp(−α|x|) → 0 in both tails so log(1+~0)/α → 0.

    NOT currently used in the production SDFs: with jacobian_approximation=
    "finite-difference-values" IPOPT's own FD already smooths the fmax kinks,
    and adding softplus double-smooths the boundary, weakening repulsion and
    changing local minima (benchmarked worse in 4/6 cases with DLS warm-start).
    Kept here for the analytic-Jacobian path: switching to jacobian_approximation=
    "exact" exposes the raw fmax kink to CasADi's AD, breaking L-BFGS — softplus
    fixes that.  To enable: replace ca.fmax(q,0) with _softplus(q) in
    _sphere_box_distance and _sphere_cylinder_distance, then switch the solver to
    "exact" and restore has_jacobian()/get_jacobian() on the FK callbacks.
    """
    return ca.fmax(x, 0) + ca.log(1 + ca.exp(-alpha * ca.fabs(x))) / alpha


def _sphere_box_distance(p_arm, arm_radius, box_center, box_R, half_extents):
    """Smooth, exact (not bounding-sphere) signed distance from a sphere (center
    p_arm, radius arm_radius) to an oriented box (center box_center, rotation box_R,
    half_extents), minus the sphere radius.

    Standard point-to-box clamp formula (e.g. Quilez's box SDF), evaluated in the
    box's local frame:
        q = |R^T (p_arm - box_center)| - half_extents
        outside = ||softplus(q)||   (smooth approx to ||max(q,0)||)
        inside  = min(max(q), 0)   (negative penetration depth if inside)

    max(q,0) in the outside term is replaced by _softplus(q) to remove the
    gradient kink at the face/edge/vertex transition (q_i = 0).  That kink
    caused L-BFGS in IPOPT to accumulate conflicting curvature estimates
    across the boundary and stall.  The inside term keeps fmin/fmax because
    its kinks only arise in infeasible states (sphere inside the box) and do
    not affect convergence from a good warm-start.
    """
    p_local = box_R.T @ (p_arm - box_center)
    q       = ca.fabs(p_local) - half_extents
    outside = ca.sqrt(ca.sumsqr(ca.fmax(q, 0)) + 1e-12)
    inside  = ca.fmin(ca.fmax(ca.fmax(q[0], q[1]), q[2]), 0)
    return outside + inside - arm_radius


def _sphere_plane_distance(p_arm, arm_radius, plane_point, plane_normal):
    """Signed distance from a sphere (center p_arm, radius arm_radius) to a plane given
    by a point on it and its unit outward normal, minus the radius. Exact and smooth
    (a single dot product), so CasADi differentiates it analytically — used for the
    ground/floor plane, which the box and bounding-sphere machinery can't represent (a
    plane has no finite bounding sphere / half-extents)."""
    return ca.dot(plane_normal, p_arm - plane_point) - arm_radius


def _sphere_cylinder_distance(p_arm, arm_radius, cyl_center, cyl_R, cyl_radius, cyl_halfheight):
    """Smooth, exact (not bounding-sphere) signed distance from a sphere to a cylinder
    (local z = axis, radius cyl_radius, half-height cyl_halfheight), minus the sphere
    radius. The cylinder reduces to a 2-D box in (radial, axial) coordinates, so this is
    the same face/edge clamp as _sphere_box_distance applied to (sqrt(x^2+y^2), |z|).
    Unlike the bounding-sphere proxy it isn't inflated by the cylinder's diagonal, so a
    finger can actually reach the curved surface to grasp. geom_size for a cylinder is
    (radius, half_height, _) in MuJoCo. Small epsilons keep the sqrt gradients finite."""
    p        = cyl_R.T @ (p_arm - cyl_center)
    radial   = ca.sqrt(p[0] * p[0] + p[1] * p[1] + 1e-12)
    dr       = radial - cyl_radius
    dz       = ca.fabs(p[2]) - cyl_halfheight
    outside  = ca.sqrt(ca.fmax(dr, 0) ** 2 + ca.fmax(dz, 0) ** 2 + 1e-12)
    inside   = ca.fmin(ca.fmax(dr, dz), 0)
    return outside + inside - arm_radius


def _sphere_sphere_distance(p_arm, arm_radius, obj_center, obj_radius):
    """Conservative center-to-center bounding-sphere distance minus both radii — the smooth
    stand-in for a non-box, non-plane object geom (e.g. a cylinder), matching the old
    _SphereDistCallback but built symbolically from a precomputed object center so CasADi
    differentiates it analytically (no finite-difference callback per pair)."""
    return ca.norm_2(p_arm - obj_center) - arm_radius - obj_radius


# ---------------------------------------------------------------------------
# Softplus SDF variants — used by the SQP solver path.
#
# Replace ca.fmax → _softplus in the outside term only so the SDF is C∞.
# The inside term keeps fmin/fmax: its kinks only appear deep inside the
# geometry (infeasible), which the warm-start never reaches.
# ---------------------------------------------------------------------------

def _softplus_sphere_box_distance(p_arm, arm_radius, box_center, box_R, half_extents):
    p_local = box_R.T @ (p_arm - box_center)
    q       = ca.fabs(p_local) - half_extents
    outside = ca.sqrt(ca.sumsqr(_softplus(q)) + 1e-12)
    inside  = ca.fmin(ca.fmax(ca.fmax(q[0], q[1]), q[2]), 0)
    return outside + inside - arm_radius


def _softplus_sphere_cylinder_distance(p_arm, arm_radius, cyl_center, cyl_R,
                                       cyl_radius, cyl_halfheight):
    p       = cyl_R.T @ (p_arm - cyl_center)
    radial  = ca.sqrt(p[0] * p[0] + p[1] * p[1] + 1e-12)
    dr      = radial - cyl_radius
    dz      = ca.fabs(p[2]) - cyl_halfheight
    outside = ca.sqrt(_softplus(dr) ** 2 + _softplus(dz) ** 2 + 1e-12)
    inside  = ca.fmin(ca.fmax(dr, dz), 0)
    return outside + inside - arm_radius


# ---------------------------------------------------------------------------
# Analytic-Jacobian FK callback wrappers — used by the SQP solver path.
#
# Each wraps a companion JacCallback and exposes it via has_jacobian() /
# get_jacobian() so CasADi uses the analytic Jacobian in its chain-rule
# differentiation of the NLP instead of FD-perturbing the callback.
# self._jac_cb is held as an instance variable to prevent GC.
# ---------------------------------------------------------------------------

class _SitePositionCallbackAnalytic(ca.Callback):
    def __init__(self, name, model, site_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._sid   = site_id
        self._n     = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self._jac_cb = _SitePositionJacCallback(name + "_J", model, site_id, n_robot, obj_qpos)
        self.construct(name, {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, _):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, _): return ca.Sparsity.dense(3, 1)
    def has_jacobian(self):        return True
    def get_jacobian(self, *_):    return self._jac_cb

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        return [self._data.site_xpos[self._sid].copy()]


class _SiteAxisCallbackAnalytic(ca.Callback):
    def __init__(self, name, model, site_id, local_axis, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model     = model
        self._data      = mj.MjData(model)
        self._sid       = site_id
        self._localaxis = np.asarray(local_axis, dtype=float).flatten()
        self._n         = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self._jac_cb = _SiteAxisJacCallback(name + "_J", model, site_id, local_axis,
                                            n_robot, obj_qpos)
        self.construct(name, {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, _):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, _): return ca.Sparsity.dense(3, 1)
    def has_jacobian(self):        return True
    def get_jacobian(self, *_):    return self._jac_cb

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        R = self._data.site_xmat[self._sid].reshape(3, 3)
        return [R @ self._localaxis]


class _GeomPositionCallbackAnalytic(ca.Callback):
    def __init__(self, name, model, geom_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._gid   = geom_id
        self._n     = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self._jac_cb = _GeomPositionJacCallback(name + "_J", model, geom_id, n_robot, obj_qpos)
        self.construct(name, {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, _):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, _): return ca.Sparsity.dense(3, 1)
    def has_jacobian(self):        return True
    def get_jacobian(self, *_):    return self._jac_cb

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        return [self._data.geom_xpos[self._gid].copy()]


class _BatchedGeomPositionJacCallback(ca.Callback):
    """d(stacked geom_xpos)/dq for G geoms via one FK pass + mj_jac per geom —
    companion for _BatchedGeomPositionCallbackAnalytic."""

    def __init__(self, name, model, geom_ids, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._gids  = list(geom_ids)
        self._n     = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {})

    def get_n_in(self):  return 2   # q, nominal_output (ignored)
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        return (ca.Sparsity.dense(self._n, 1) if i == 0
                else ca.Sparsity.dense(3 * len(self._gids), 1))
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3 * len(self._gids), self._n)
    def has_jacobian(self): return False

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        mj.mj_comPos(self._model, self._data)   # needed: populates cdof for mj_jac
        J    = np.zeros((3 * len(self._gids), self._n))
        jacp = np.zeros((3, self._model.nv))
        for k, gid in enumerate(self._gids):
            mj.mj_jac(self._model, self._data, jacp, None,
                      self._data.geom_xpos[gid].copy(), int(self._model.geom_bodyid[gid]))
            J[3 * k:3 * k + 3] = jacp[:, :self._n]
        return [J]


class _BatchedGeomPositionCallbackAnalytic(ca.Callback):
    def __init__(self, name, model, geom_ids, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._gids  = list(geom_ids)
        self._n     = n_robot
        self.eval_count = 0
        if obj_qpos is not None:
            self._data.qpos[n_robot:n_robot + len(obj_qpos)] = obj_qpos
        self._jac_cb = _BatchedGeomPositionJacCallback(name + "_J", model, geom_ids,
                                                       n_robot, obj_qpos)
        self.construct(name, {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, _):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, _): return ca.Sparsity.dense(3 * len(self._gids), 1)
    def has_jacobian(self):        return True
    def get_jacobian(self, *_):    return self._jac_cb

    def eval(self, arg):
        self.eval_count += 1
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        return [self._data.geom_xpos[self._gids].reshape(-1)]


# Solver options for sqpmethod + OSQP.  OSQP is used instead of qpOASES
# because with O(500) inequality constraints on 23 DOFs the linearised QP
# subproblem is often primal-infeasible at the initial (DLS warm-start) point;
# OSQP finds the minimum-constraint-violation direction and lets SQP continue.
_SQP_SOLVER_OPTS = {
    'print_time':            False,
    'qpsol':                 'osqp',
    'qpsol_options':         {'error_on_fail': False,
                              'osqp': {'verbose': False, 'polish': True}},
    'max_iter':              500,
    # Slightly relaxed from the sqpmethod defaults (tol_pr/tol_du = 1e-6): the dual
    # tolerance drives most of the tail iterations (sub-mm cost polishing), while the
    # primal tolerance is kept tight so constraint (collision) feasibility is unaffected.
    # Measured on the pick-place scene: defaults ran 285-500 iters/object (one hitting
    # max_iter); 1e-4 cuts that roughly in half at <2mm tip-error change. tol_du=1e-3 /
    # tol_pr=1e-5 is the next notch (~90 iters, +9mm) if latency ever matters more.
    'tol_du':                1e-4,
    'hessian_approximation': 'limited-memory',
    'lbfgs_memory':          20,
    'convexify_strategy':    'regularize',
    'print_iteration':       False,
    'print_header':          False,
    'print_status':          False,
}


def configure_sqp(solver):
    """Switch a ConstrainedIKSolver instance to the SQP + softplus-SDF mode.

    Replaces the module-level SDF functions and FK callbacks with their
    analytic-Jacobian / softplus-smoothed counterparts, and switches the
    CasADi solver from IPOPT to sqpmethod/OSQP.  Call once after construction.
    """
    import grasp_control.constrained_ik as _m
    _m._sphere_box_distance          = _softplus_sphere_box_distance
    _m._sphere_cylinder_distance     = _softplus_sphere_cylinder_distance
    _m._SitePositionCallback         = _SitePositionCallbackAnalytic
    _m._SiteAxisCallback             = _SiteAxisCallbackAnalytic
    _m._GeomPositionCallback         = _GeomPositionCallbackAnalytic
    _m._BatchedGeomPositionCallback  = _BatchedGeomPositionCallbackAnalytic
    solver._solver_name = 'sqpmethod'
    solver._solver_opts = _SQP_SOLVER_OPTS


class ConstrainedIKSolver:
    """
    IPOPT-based IK with hard joint-limit and collision-avoidance constraints.

    Unlike the DLS solver (grasp_control/ik.py), which clips joint limits post-hoc
    and has no collision awareness, this enforces:
      - Box constraints on each limited joint (no clipping, true hard constraint)
      - mj_geomDistance(arm_geom_i, obj_geom_j) >= clearance for each pair

    arm_geom_names : geom names on the manipulator to protect from objects.
                     Build this list with build_arm_geom_names() or supply manually.
    obj_geom_names : geom names on scene objects (e.g. ['obj_box_geom', 'obj_cylinder_geom']).
    clearance      : minimum signed distance (m) required between each arm/obj geom pair.
    posture_weight : weight on ||q - q_bias||² in the cost. Small (0.01–0.1) keeps the
                     position task dominant while still regularizing the null space.
    pad_axis       : selects the direction (as a unit vector) in the frame of the finger end 
                     effector to attempt to orient to the normal of object surface during grasping
    orient_weight  : weight in the IK cost function for orienting the finger pad with the object surface normal 
    """

    def __init__(self, model, n_robot,
                 arm_geom_names, obj_geom_names,
                 clearance=0.005, posture_weight=0.05,
                 pad_axis=(-1.0, 0.0, 0.0), orient_weight=0.0,
                 max_iter=500, verbose=False):
        self._model          = model
        self._n              = n_robot
        self._arm_geom_names = list(arm_geom_names)
        self._obj_geom_names = list(obj_geom_names)
        self._clearance      = clearance
        self._posture_weight = posture_weight
        self._max_iter       = max_iter
        self._verbose        = verbose
        self._pad_axis = np.asarray(pad_axis, dtype=float).flatten()
        self._orient_weight = orient_weight
        self.last_metrics = {}   # populated by solve(); consumed by the live dashboard

        self._arm_gids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n)
            for n in self._arm_geom_names
        ]
        self._obj_gids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n)
            for n in self._obj_geom_names
        ]
        # Box-shaped object geoms get the exact sphere-vs-box distance formula
        # (_sphere_box_distance) instead of the bounding-sphere proxy, since a sphere
        # bound has to use the box's diagonal to stay conservative in every direction
        # (e.g. 0.104m for a 0.06m half-extent cube) — a big, unnecessary penalty for
        # the common case of approaching a face. geom_size for a box geom *is* its
        # (hx, hy, hz) half-extents (MuJoCo convention).
        self._obj_is_box       = [model.geom_type[gid] == mj.mjtGeom.mjGEOM_BOX
                                   for gid in self._obj_gids]
        # Plane object geoms (e.g. the ground) use the exact sphere-vs-plane distance;
        # they have no finite bounding sphere, so the box/sphere paths don't apply.
        self._obj_is_plane     = [model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE
                                   for gid in self._obj_gids]
        self._obj_is_cyl       = [model.geom_type[gid] == mj.mjtGeom.mjGEOM_CYLINDER
                                   for gid in self._obj_gids]
        self._obj_half_extents = [model.geom_size[gid].copy() for gid in self._obj_gids]
        # Bounding-sphere radius for the sphere-vs-sphere fallback (non-box, non-plane geoms).
        self._obj_rbound       = [float(model.geom_rbound[gid]) for gid in self._obj_gids]

        limited        = model.jnt_limited[:n_robot].astype(bool)
        self._lo       = model.jnt_range[:n_robot, 0]
        self._hi       = model.jnt_range[:n_robot, 1]
        self._limited  = limited

        # Bounds for unlimited joints are left at +-1e19 (effectively inf) so this can be
        # passed to opti.bounded() as one vectorized call below. Building it per-scalar-index
        # (opti.bounded(lo[i], q[i], hi[i]) in a loop) makes CasADi treat each one as a
        # general inequality constraint instead of a true variable bound (it only recognizes
        # the bound-only fast path when the constrained expression *is* the whole decision
        # variable, not an indexed slice of it) — confirmed via IPOPT's own problem summary
        # ("variables with lower/upper bounds: 0" vs "19 inequality constraints"). The extra
        # multipliers from 19 spurious general constraints caused IPOPT to repeatedly enter
        # its restoration phase and sometimes report false "Infeasible_Problem_Detected" exits
        # even on a trivially feasible box-constrained problem.
        self._lo_vec = np.where(limited, self._lo, -1e19)
        self._hi_vec = np.where(limited, self._hi,  1e19)

        self._solver_name = "ipopt"
        self._solver_opts = {
            "print_time": False,
            "ipopt": {
                # IPOPT computes its own FD Jacobian of the NLP (not CasADi's chain-rule
                # Jacobian). This is intentional: CasADi's chain-rule differentiates
                # _sphere_box_distance symbolically, exposing kinks at face/edge/vertex
                # transitions where the SDF gradient is discontinuous. IPOPT's FD straddles
                # those kinks in the full distance expression, giving effectively smoothed
                # gradients that L-BFGS handles far better than the exact one-sided derivatives.
                # Switching to "exact" here requires smoothing the box SDF kinks first.
                "jacobian_approximation": "finite-difference-values",
                "hessian_approximation":  "limited-memory",
                "print_level":  5 if verbose else 0,
                "sb":           "yes",
                "max_iter":     max_iter,
                "tol":          1e-4,
                "constr_viol_tol": 1e-6,
                "mu_strategy":              "adaptive",
                "acceptable_tol":           1e-3,
                "acceptable_iter":          10,
                "acceptable_constr_viol_tol": 1e-5,
            },
        }

    def solve(self, data, site_ids, targets,
              q_bias=None, q_init=None, skip_arm_geoms=frozenset(),
              reduced_clearance_geoms=frozenset(), reduced_clearance=0.0005,
              inward_dirs=None, prune_margin=0.15):
        """
        Solve IK with collision-avoidance constraints.

        Parameters
        ----------
        data           : live MjData — object qpos is snapshotted so FK inside
                         callbacks reflects the current scene positions.
        site_ids       : list[int] — fingertip site IDs to drive to targets.
        targets        : list[ndarray (3,)] — world-frame fingertip target positions.
        inward_dirs    : optional list[ndarray (3,)] — per-site world-frame unit vectors the
                         fingerpad normal should align with (each contact site's inward surface
                         normal). When given and orient_weight > 0, adds an alignment term to
                         the cost. None (default) leaves orientation unconstrained.
        q_bias         : (n_robot,) posture reference used in the secondary cost term
                         and as the default warm-start.
        q_init         : (n_robot,) explicit IPOPT warm-start; overrides q_bias.
        skip_arm_geoms : set of arm geom names to exclude entirely from collision
                         constraints.
        reduced_clearance_geoms : arm geoms that use a reduced clearance (instead of the
                         solver's default) against the OBJECT geoms — for the active
                         grasping fingers, which must get close to the object surface.
                         Either a set of geom names (all using `reduced_clearance`) or a
                         dict {geom_name: clearance_m} for per-geom values (e.g. distal
                         links disabled, proximal links small-but-finite). The full
                         clearance is always kept against any plane geom (the floor), so
                         these fingers never drop underground.
        reduced_clearance : clearance (m) applied to reduced_clearance_geoms vs objects
                         when it is a set; ignored when it is a dict.
        prune_margin   : per-pair constraint pruning threshold (m), or None to disable.
                         Arm-geom/object pairs whose exact distance at the warm start
                         exceeds clearance + prune_margin are left out of the NLP (only
                         ~7 of ~500 constraints ever bind; measured ~3x faster per
                         iteration). Plane (floor) pairs are always kept. Every pruned
                         pair is re-checked at the solution; a violation triggers one
                         automatic re-solve with pruning disabled.

        Returns
        -------
        q : (n_robot,) — IPOPT solution, or best-iterate on failure.
        """
        n        = self._n
        ctr      = next(_call_counter)
        obj_qpos = data.qpos[n:].copy()
        q0 = (q_init if q_init is not None
              else (q_bias if q_bias is not None else np.zeros(n)))
        _t0_solve = time.time()

        site_cbs = [
            _SitePositionCallback(f"cik_site_{ctr}_{i}", self._model, sid, n, obj_qpos)
            for i, sid in enumerate(site_ids)
        ]

        axis_cbs = [
            _SiteAxisCallback(f"cik_axis_{ctr}_{i}", self._model, sid, self._pad_axis, n, obj_qpos) 
            for i, sid in enumerate(site_ids)
        ]

        # Object qpos is snapshotted (not optimized), so each object geom's world pose
        # is a constant for this solve — precompute box poses once via a scratch FK
        # rather than re-deriving them symbolically. The robot is posed at the warm
        # start q0 in the same pass so the pair-pruning distances below reflect it.
        _scratch = mj.MjData(self._model)
        _scratch.qpos[:n] = q0
        _scratch.qpos[n : n + len(obj_qpos)] = obj_qpos
        mj.mj_kinematics(self._model, _scratch)
        obj_pose = [
            (_scratch.geom_xpos[gid].copy(), _scratch.geom_xmat[gid].reshape(3, 3).copy())
            for gid in self._obj_gids
        ]

        opti = ca.Opti()
        q    = opti.variable(n)

        cost = ca.MX(0)
        for cb, tgt in zip(site_cbs, targets):
            diff = cb(q) - ca.DM(tgt)
            cost = cost + ca.dot(diff, diff)

        if q_bias is not None:
            dq   = q - ca.DM(q_bias)
            cost = cost + self._posture_weight * ca.dot(dq, dq)

        if inward_dirs is not None and self._orient_weight > 0:
            for cb, d_in in zip(axis_cbs, inward_dirs):
                e = cb(q) - ca.DM(d_in)
                cost = cost + self._orient_weight * ca.dot(e,e)

        opti.minimize(cost)

        # Single vectorized bound on the whole decision variable -> true lbx/ubx in IPOPT,
        # not n separate general inequality constraints (see _lo_vec/_hi_vec comment above).
        opti.subject_to(opti.bounded(ca.DM(self._lo_vec), q, ca.DM(self._hi_vec)))

        # ONE batched position callback feeds every arm geom's sphere center (its world
        # position depends only on q, not on the object): a single Python↔CasADi crossing
        # and a single mj_kinematics per NLP evaluation instead of one per geom. The
        # per-object distance is then a purely symbolic expression that CasADi
        # differentiates analytically:
        #   box   -> exact sphere-vs-box       (_sphere_box_distance)
        #   plane -> exact sphere-vs-plane      (_sphere_plane_distance)   e.g. the floor
        #   cyl   -> exact sphere-vs-cylinder   (_sphere_cylinder_distance)
        #   other -> conservative sphere-sphere (_sphere_sphere_distance)
        # Pair pruning (prune_margin): only a handful of the ~500 pair constraints ever
        # bind, so pairs whose exact warm-start distance clears the threshold are left
        # out of the NLP — plane pairs always kept (the whole hand can descend toward
        # the floor). Pruned pairs are re-verified at the solution before returning.
        # mj_geomDistance is guarded by the bounding-sphere lower bound: MuJoCo's GJK
        # can return a phantom 0.0 for well-separated box-box pairs.
        # _cb_keepalive: CasADi Callback objects are only weakly referenced once their
        # symbolic output is consumed — without holding a Python reference for the rest of
        # this solve they get garbage-collected and the solver's later gradient pass fails
        # with "Callback object has been deleted".
        # Each dist_exprs entry is (distance_expr, required_clearance, arm_geom, obj_geom).
        # Active grasping fingers get a reduced clearance vs objects so they can reach the
        # surface, but the FULL clearance is always kept against a plane so no finger drops
        # underground.
        arm_entries = []   # (arm_geom_name, geom_id, bounding_radius, clearance_vs_objects)
        for ag, gid1 in zip(self._arm_geom_names, self._arm_gids):
            if ag in skip_arm_geoms:
                continue
            if isinstance(reduced_clearance_geoms, dict):
                obj_clr = reduced_clearance_geoms.get(ag, self._clearance)
            else:
                obj_clr = reduced_clearance if ag in reduced_clearance_geoms else self._clearance
            arm_entries.append((ag, gid1, float(self._model.geom_rbound[gid1]), obj_clr))

        batched_cb = _BatchedGeomPositionCallback(
            f"cik_bgp_{ctr}", self._model, [e[1] for e in arm_entries], n, obj_qpos)
        _cb_keepalive = [batched_cb]
        p_all = batched_cb(q)   # stacked (3G,) symbolic geom positions

        dist_exprs   = []
        pruned_pairs = []   # (arm_gid, obj_gid, arm_geom_name, obj_geom_name, clearance)
        _ft6 = np.zeros(6)
        for i, (ag, gid1, arm_radius, obj_clr) in enumerate(arm_entries):
            p_arm = p_all[3 * i : 3 * i + 3]
            for j, ogn in enumerate(self._obj_geom_names):
                center, R = obj_pose[j]
                gid2 = self._obj_gids[j]
                clr  = self._clearance if self._obj_is_plane[j] else obj_clr
                if prune_margin is not None and not self._obj_is_plane[j]:
                    lb = (float(np.linalg.norm(_scratch.geom_xpos[gid2]
                                               - _scratch.geom_xpos[gid1]))
                          - arm_radius - self._obj_rbound[j])
                    dist = lb
                    if lb <= clr + prune_margin:   # bound can't decide — ask GJK
                        dist = max(mj.mj_geomDistance(
                            self._model, _scratch, gid1, gid2,
                            max(clr, 0.0) + prune_margin + 0.05, _ft6), lb)
                    if dist > clr + prune_margin:
                        pruned_pairs.append((gid1, gid2, ag, ogn, clr))
                        continue
                if self._obj_is_box[j]:
                    expr = _sphere_box_distance(
                        p_arm, arm_radius,
                        ca.DM(center), ca.DM(R), ca.DM(self._obj_half_extents[j]))
                elif self._obj_is_plane[j]:
                    expr = _sphere_plane_distance(
                        p_arm, arm_radius, ca.DM(center), ca.DM(R[:, 2]))
                elif self._obj_is_cyl[j]:
                    expr = _sphere_cylinder_distance(
                        p_arm, arm_radius, ca.DM(center), ca.DM(R),
                        float(self._obj_half_extents[j][0]), float(self._obj_half_extents[j][1]))
                else:
                    expr = _sphere_sphere_distance(
                        p_arm, arm_radius, ca.DM(center), self._obj_rbound[j])
                dist_exprs.append((expr, clr, ag, ogn))

        for d, clr, _ag, _og in dist_exprs:
            opti.subject_to(d >= clr)

        opti.set_initial(q, q0)
        t_setup = time.time() - _t0_solve
        opti.solver(self._solver_name, self._solver_opts)
        t_setup_ms = (time.time() - _t0_solve) * 1e3
        _slabel = self._solver_name.upper()

        print(f"[{_slabel}] setup {t_setup_ms:.0f}ms  |  "
              f"{n} vars  {len(dist_exprs)} constraints ({len(pruned_pairs)} pruned)  "
              f"(1 batched geom cb [{len(arm_entries)} geoms] + "
              f"{len(site_cbs)} site + {len(axis_cbs)} axis cbs)")

        def _diagnostics(value_fn, t_solve_ms, iters):
            """value_fn: sol.value or opti.debug.value.
            Reports site error, posture, collision slack, orientation, callback counts,
            and timing so it is clear where time is going.
            Stashes results into self.last_metrics for the live dashboard."""
            site_err_mm = [
                float(np.linalg.norm(value_fn(cb(q)) - ca.DM(tgt))) * 1000.0
                for cb, tgt in zip(site_cbs, targets)
            ]
            self.last_metrics['site_err_mm'] = site_err_mm
            print(f"[{_slabel}]   site errors (mm): "
                  f"{['%.2f' % e for e in site_err_mm]}  max={max(site_err_mm):.2f}")

            if q_bias is not None:
                dq = np.array(value_fn(q)).flatten() - np.asarray(q_bias)
                print(f"[{_slabel}]   posture term: {self._posture_weight * float(dq @ dq):.4g}"
                      f"  (||q-q_bias||={float(np.linalg.norm(dq)):.3g} rad)")

            if dist_exprs:
                # One pass: evaluate slack and keep geom names together.
                slack_rows = [((float(value_fn(expr)) - clr) * 1000.0, ag, og)
                              for expr, clr, ag, og in dist_exprs]
                slacks_mm = [r[0] for r in slack_rows]
                i_min  = int(np.argmin(slacks_mm))
                n_bind = sum(s < 0.1 for s in slacks_mm)
                self.last_metrics['min_slack_mm'] = slacks_mm[i_min]
                print(f"[{_slabel}]   collision margin (mm): min={slacks_mm[i_min]:.2f}"
                      f"  n_binding(<0.1mm)={n_bind}/{len(slacks_mm)}")
                # Top-5 tightest pairs — the main signal for what's bottlenecking convergence
                for sl, ag, og in sorted(slack_rows, key=lambda r: r[0])[:5]:
                    print(f"[{_slabel}]     {sl:+7.2f}mm  {ag}  vs  {og}")

            if inward_dirs is not None and self._orient_weight > 0:
                ang = [float(np.degrees(np.arccos(np.clip(
                        float(np.dot(np.array(value_fn(cb(q))).flatten(), d_in)), -1.0, 1.0))))
                    for cb, d_in in zip(axis_cbs, inward_dirs)]
                self.last_metrics['pad_deg'] = ang
                print(f"[{_slabel}]   pad-normal error (deg): "
                    f"{['%.1f' % a for a in ang]}  max={max(ang):.1f}")

            # FK call counts. SQP/analytic mode: geom eval_count ≈ n_iters ×
            # n_line_search_steps (one batched crossing per NLP evaluation, plus one
            # Jacobian crossing per iteration, not counted here). IPOPT/FD mode: IPOPT
            # perturbs each of n DOFs and re-evaluates the full NLP, so eval_count ≈
            # n_iters × (n_robot + 1) × n_line_search_steps — but still per batched
            # callback, not per geom.
            site_fk  = sum(cb.eval_count for cb in site_cbs)
            axis_fk  = sum(cb.eval_count for cb in axis_cbs)
            geom_fk  = batched_cb.eval_count
            total_fk = site_fk + axis_fk + geom_fk
            ms_per_iter = t_solve_ms / iters if iters else float('nan')
            self.last_metrics.update(
                total_fk_calls=total_fk, t_setup_ms=t_setup_ms,
                t_solve_ms=t_solve_ms, ms_per_iter=ms_per_iter)
            print(f"[{_slabel}]   timing: setup={t_setup_ms:.0f}ms  solve={t_solve_ms:.0f}ms"
                  f"  {ms_per_iter:.1f}ms/iter")
            print(f"[{_slabel}]   FK calls: {total_fk:,}  "
                  f"(site={site_fk}  axis={axis_fk}  geom={geom_fk})"
                  f"  ≈ {total_fk / max(iters, 1):.0f} FK/iter")

        def _violated_pruned_pair(q_sol):
            """Exact-distance recheck of every pruned pair at the solution. Returns the
            worst (dist, clr, arm_geom, obj_geom) violation, or None. Same GJK guard as
            the pruning pass (bounding-sphere lower bound vs phantom 0.0)."""
            if not pruned_pairs:
                return None
            _scratch.qpos[:n] = q_sol
            mj.mj_kinematics(self._model, _scratch)
            worst = None
            for gid1, gid2, ag, ogn, clr in pruned_pairs:
                lb = (float(np.linalg.norm(_scratch.geom_xpos[gid2]
                                           - _scratch.geom_xpos[gid1]))
                      - float(self._model.geom_rbound[gid1]) - float(self._model.geom_rbound[gid2]))
                if lb >= clr:
                    continue
                dist = max(mj.mj_geomDistance(self._model, _scratch, gid1, gid2,
                                              max(clr, 0.0) + 0.05, _ft6), lb)
                if dist < clr and (worst is None or dist - clr < worst[0] - worst[1]):
                    worst = (dist, clr, ag, ogn)
            return worst

        # Reset per-solve metrics; _diagnostics fills in the physical quantities and the
        # try/except branches add solver status. Read via solver.last_metrics after solve().
        self.last_metrics = {'n_pruned': len(pruned_pairs)}
        _t0_ipopt = time.time()
        try:
            sol  = opti.solve()
            t_solve_ms = (time.time() - _t0_ipopt) * 1e3
            st   = opti.stats()
            iters = st['iter_count']
            self.last_metrics.update(status=st['return_status'], iters=iters,
                                     obj=float(sol.value(opti.f)), success=True)
            print(f"[{_slabel}] {st['return_status']}  iters={iters}"
                  f"  obj={float(sol.value(opti.f)):.4g}")
            _diagnostics(sol.value, t_solve_ms, iters)
            q_sol = np.array(sol.value(q))
            viol  = _violated_pruned_pair(q_sol)
            if viol is not None:
                dist, clr, ag, ogn = viol
                print(f"[{_slabel}] pruned pair violated at solution: {ag} vs {ogn} "
                      f"{dist * 1e3:.1f}mm < clearance {clr * 1e3:.1f}mm — "
                      f"re-solving with pruning disabled")
                return self.solve(data, site_ids, targets,
                                  q_bias=q_bias, q_init=q_init,
                                  skip_arm_geoms=skip_arm_geoms,
                                  reduced_clearance_geoms=reduced_clearance_geoms,
                                  reduced_clearance=reduced_clearance,
                                  inward_dirs=inward_dirs, prune_margin=None)
            return q_sol
        except RuntimeError:
            t_solve_ms = (time.time() - _t0_ipopt) * 1e3
            try:
                st = opti.stats()
            except Exception:
                # opti.stats() throws if solve() bailed before the solver
                # ever ran (e.g. sqpmethod QP failure on iteration 0)
                st = {}
            iters = st.get('iter_count', 0)
            self.last_metrics.update(status=st.get('return_status', 'failed'),
                                     iters=iters, success=False)
            print(f"[{_slabel}] FAILED: {st.get('return_status','?')}  iters={iters}")
            _diagnostics(opti.debug.value, t_solve_ms, iters)
            return np.array(opti.debug.value(q))


def check_analytic_jacobians(model, n_robot, site_ids, geom_ids,
                              pad_axis=(-1., 0., 0.), eps=1e-6, atol=1e-3):
    """Numerically verify the analytic-Jacobian companion callbacks against FD.

    Call this once at startup (with the model and a representative set of site/geom
    IDs) to confirm the analytic Jacobians are correct before enabling them in the
    main solver. Prints pass/fail per callback type and returns True if all pass.

    Parameters
    ----------
    model      : MjModel
    n_robot    : int — number of robot DOFs
    site_ids   : list[int] — fingertip site IDs to check
    geom_ids   : list[int] — arm geom IDs to check
    pad_axis   : local pad axis for SiteAxis check
    eps        : FD step size (m or rad)
    atol       : absolute tolerance for max Jacobian element error

    Returns True if all checks pass, False if any fail.
    """
    import mujoco as _mj
    q_test = np.zeros(n_robot)
    obj_qpos = np.zeros(model.nq - n_robot)

    d = _mj.MjData(model)
    d.qpos[:n_robot] = q_test
    _mj.mj_kinematics(model, d)

    all_ok = True

    def _fd_jacobian(eval_fn, q, n):
        f0 = np.array(eval_fn(q)).flatten()
        J  = np.zeros((len(f0), n))
        for j in range(n):
            dq = q.copy(); dq[j] += eps
            J[:, j] = (np.array(eval_fn(dq)).flatten() - f0) / eps
        return J

    for i, sid in enumerate(site_ids[:2]):  # check first 2 sites
        cb  = _SitePositionJacCallback(f"chk_spos_{i}", model, sid, n_robot, obj_qpos)
        def _f(q): return _SitePositionCallback(f"tmp{i}", model, sid, n_robot, obj_qpos).eval([q])[0]
        J_analytic = np.array(cb.eval([q_test, np.zeros(3)])[0])
        J_fd       = _fd_jacobian(_f, q_test, n_robot)
        err = np.abs(J_analytic - J_fd).max()
        ok  = err < atol
        print(f"[Jac check] SitePosition  site={sid}  max_err={err:.2e}  {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    for i, sid in enumerate(site_ids[:2]):
        cb  = _SiteAxisJacCallback(f"chk_sax_{i}", model, sid, pad_axis, n_robot, obj_qpos)
        def _f(q):
            d2 = _mj.MjData(model)
            d2.qpos[:n_robot] = q
            _mj.mj_kinematics(model, d2)
            R = d2.site_xmat[sid].reshape(3, 3)
            return R @ np.array(pad_axis)
        J_analytic = np.array(cb.eval([q_test, np.zeros(3)])[0])
        J_fd       = _fd_jacobian(_f, q_test, n_robot)
        err = np.abs(J_analytic - J_fd).max()
        ok  = err < atol
        print(f"[Jac check] SiteAxis      site={sid}  max_err={err:.2e}  {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    for i, gid in enumerate(geom_ids[:3]):  # check first 3 geoms
        cb  = _GeomPositionJacCallback(f"chk_gpos_{i}", model, gid, n_robot, obj_qpos)
        def _f(q):
            d2 = _mj.MjData(model)
            d2.qpos[:n_robot] = q
            _mj.mj_kinematics(model, d2)
            return d2.geom_xpos[gid].copy()
        J_analytic = np.array(cb.eval([q_test, np.zeros(3)])[0])
        J_fd       = _fd_jacobian(_f, q_test, n_robot)
        err = np.abs(J_analytic - J_fd).max()
        ok  = err < atol
        print(f"[Jac check] GeomPosition  geom={gid}  max_err={err:.2e}  {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    return all_ok


def build_arm_geom_names(model, body_names):
    """
    Return all geom names attached to the given body names.

    Typical usage for the Kinova+LEAP arm:
        arm_geoms = build_arm_geom_names(model, [
            'forearm_link', 'spherical_wrist_1_link',
            'spherical_wrist_2_link', 'bracelet_link',
            'leap_palm',
            'leap_if_bs', 'leap_if_px', 'leap_if_md',
            'leap_mf_bs', 'leap_mf_px', 'leap_mf_md',
            'leap_rf_bs', 'leap_rf_px', 'leap_rf_md',
            'leap_th_mp', 'leap_th_pp', 'leap_th_dp',
        ])
        # Then add distal links with skip_arm_geoms for grasp IK:
        all_geoms = build_arm_geom_names(model, body_names + distal_bodies)
        solver.solve(..., skip_arm_geoms={'leap_if_ds_collision', 'leap_th_ds_collision'})
    """
    body_ids = {
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b)
        for b in body_names
    }
    return [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
        if model.geom_bodyid[i] in body_ids
        and mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
    ]

"""
Collision-aware IK via CasADi + IPOPT.

Formulation:
    minimize_q  Σ_k ||site_pos_k(q) - target_k||²  +  posture_weight * ||q - q_bias||²
    subject to:
        q_lo[i] <= q[i] <= q_hi[i]                          (joint limits, hard)
        mj_geomDistance(arm_geom_i, obj_geom_j, q) >= d_min  (collision avoidance, hard)

Both the fingertip-position task and the collision constraints are evaluated through
MuJoCo forward kinematics wrapped in CasADi external callbacks; IPOPT uses
finite-difference Jacobians (no hand-coded gradients needed). Each callback owns a
private MjData copy so that IPOPT's FD perturbation of one constraint doesn't corrupt
the state seen by another.

See also: simulation/tamp_manager.py (_SDCActuated) for the same callback pattern
used in the TAMP approach controller.
"""
import itertools

import casadi as ca
import mujoco as mj
import numpy as np

_call_counter = itertools.count()


class _SitePositionCallback(ca.Callback):
    """q[:n_robot] → site_xpos[site_id] (3-vector) via MuJoCo FK."""

    def __init__(self, name, model, site_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._sid   = site_id
        self._n     = n_robot
        if obj_qpos is not None:
            self._data.qpos[n_robot : n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        mj.mj_comPos(self._model, self._data)
        return [self._data.site_xpos[self._sid].copy()]

class _SiteAxisCallback(ca.Callback):
    """q[:n_robot] → site_R(q) @ local_axis (world-frame unit 3-vector) via MuJoCo FK. 
    Returns the fingerpad normal used to orient the fingerpad with the contact surface normal."""
    def __init__(self, name, model, site_id, local_axis, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data = mj.MjData(model)
        self._sid = site_id
        self._localaxis = np.asarray(local_axis, dtype=float).flatten()
        self._n = n_robot

        if obj_qpos is not None:
            self._data.qpos[n_robot : n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {"enable_fd": True})
    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i): return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, 1)
    def has_jacobian(self): return False

    def eval(self, arg):
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)

        R = self._data.site_xmat[self._sid].reshape(3,3)
        return [R @ self._localaxis]





class _GeomPositionCallback(ca.Callback):
    """q[:n_robot] → geom_xpos[geom_id] (3-vector) via MuJoCo FK. Same pattern as
    _SitePositionCallback but for a geom origin rather than a site — used to feed an
    arm geom's sphere center into the analytic sphere-vs-box distance formula below."""

    def __init__(self, name, model, geom_id, n_robot, obj_qpos=None):
        ca.Callback.__init__(self)
        self._model = model
        self._data  = mj.MjData(model)
        self._gid   = geom_id
        self._n     = n_robot
        if obj_qpos is not None:
            self._data.qpos[n_robot : n_robot + len(obj_qpos)] = obj_qpos
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):  return ca.Sparsity.dense(self._n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(3, 1)
    def has_jacobian(self):        return False

    def eval(self, arg):
        self._data.qpos[:self._n] = np.array(arg[0]).flatten()
        mj.mj_kinematics(self._model, self._data)
        return [self._data.geom_xpos[self._gid].copy()]


def _sphere_box_distance(p_arm, arm_radius, box_center, box_R, half_extents):
    """Smooth, exact (not bounding-sphere) signed distance from a sphere (center
    p_arm, radius arm_radius) to an oriented box (center box_center, rotation box_R,
    half_extents), minus the sphere radius.

    Standard point-to-box clamp formula (e.g. Quilez's box SDF), evaluated in the
    box's local frame:
        q = |R^T (p_arm - box_center)| - half_extents
        outside = ||max(q, 0)||      (0 if inside, true face/edge/vertex distance otherwise)
        inside  = min(max(q), 0)     (negative penetration depth if inside)
    Unlike the bounding-sphere proxy (which has to use the box's *diagonal* to never
    underestimate clearance, e.g. 0.104m for a 0.06m half-extent cube), this uses the
    box's true half-extents, so it isn't artificially conservative for a face-on
    approach. It is built from plain CasADi symbolic ops (fabs/fmax/sqrt), so CasADi
    differentiates it exactly via autodiff instead of finite-differencing through a
    Callback — the box still has a genuine gradient kink where the closest feature
    switches between face/edge/vertex, but that kink is now a true zero-measure
    crossing with a correct one-sided gradient, not a region smeared by an FD step
    straddling it (which is what caused IPOPT to cycle with the exact mj_geomDistance
    callback). The small epsilon under the sqrt avoids an infinite gradient exactly
    at outside == 0 (i.e. exactly touching a face).
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

        self._solver_opts = {
            "print_time": False,
            "ipopt": {
                "jacobian_approximation": "finite-difference-values",
                "hessian_approximation":  "limited-memory",
                "print_level":  5 if verbose else 0,
                "sb":           "yes",
                "max_iter":     max_iter,
                "tol":          1e-4,
                "constr_viol_tol": 1e-6,
                # mj_geomDistance has genuine kinks (closest point jumps between a box's
                # face/edge/vertex as q varies) — limited-memory BFGS fed FD gradients across
                # those kinks can take wild steps and either grind to max_iter or report a
                # spurious Infeasible_Problem_Detected, even when a perfectly good iterate was
                # visited along the way. "adaptive" mu_strategy is IPOPT's standard robustness
                # recommendation for exactly this class of poorly-conditioned/noisy problem.
                # acceptable_* lets IPOPT return the best iterate once progress stalls near a
                # decent point, instead of continuing to chase full first-order optimality
                # through more non-smooth terrain and potentially wandering somewhere worse.
                "mu_strategy":              "adaptive",
                "acceptable_tol":           1e-2,
                "acceptable_iter":          15,
                "acceptable_constr_viol_tol": 1e-4,
            },
        }

    def solve(self, data, site_ids, targets,
              q_bias=None, q_init=None, skip_arm_geoms=frozenset(),
              reduced_clearance_geoms=frozenset(), reduced_clearance=0.0005,
              inward_dirs=None):
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

        Returns
        -------
        q : (n_robot,) — IPOPT solution, or best-iterate on failure.
        """
        n        = self._n
        ctr      = next(_call_counter)
        obj_qpos = data.qpos[n:].copy()

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
        # rather than re-deriving them symbolically.
        _scratch = mj.MjData(self._model)
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

        # One finite-difference position callback PER ARM GEOM (its world position depends
        # only on q, not on the object), reused across all objects; the per-object distance
        # is then a purely symbolic expression that CasADi differentiates analytically:
        #   box   -> exact sphere-vs-box       (_sphere_box_distance)
        #   plane -> exact sphere-vs-plane      (_sphere_plane_distance)   e.g. the floor
        #   other -> conservative sphere-sphere (_sphere_sphere_distance)  e.g. a cylinder
        # This is what makes checking every hand collision geom affordable — the FD cost is
        # n_arm_geoms (not n_arm_geoms x n_objects, as when a callback was built per pair).
        # _cb_keepalive: CasADi Callback objects are only weakly referenced once their
        # symbolic output is consumed — without holding a Python reference for the rest of
        # this solve they get garbage-collected and IPOPT's later gradient pass fails with
        # "Callback object has been deleted".
        # Each entry is (distance_expr, required_clearance). Active grasping fingers get a
        # reduced clearance vs objects so they can reach the surface, but the FULL clearance
        # is always kept against a plane (the floor) so no finger drops underground.
        dist_exprs   = []
        _cb_keepalive = []
        for ag, gid1 in zip(self._arm_geom_names, self._arm_gids):
            if ag in skip_arm_geoms:
                continue
            arm_radius = float(self._model.geom_rbound[gid1])
            if isinstance(reduced_clearance_geoms, dict):
                obj_clr = reduced_clearance_geoms.get(ag, self._clearance)
            else:
                obj_clr = reduced_clearance if ag in reduced_clearance_geoms else self._clearance
            pos_cb = _GeomPositionCallback(f"cik_gp_{ctr}_{gid1}", self._model, gid1, n, obj_qpos)
            _cb_keepalive.append(pos_cb)
            p_arm = pos_cb(q)
            for j in range(len(self._obj_gids)):
                center, R = obj_pose[j]
                if self._obj_is_box[j]:
                    dist_exprs.append((_sphere_box_distance(
                        p_arm, arm_radius,
                        ca.DM(center), ca.DM(R), ca.DM(self._obj_half_extents[j])), obj_clr))
                elif self._obj_is_plane[j]:
                    # Plane outward normal is its local +z axis (3rd column of geom_xmat).
                    # Floor always uses the full clearance, never the reduced one.
                    dist_exprs.append((_sphere_plane_distance(
                        p_arm, arm_radius, ca.DM(center), ca.DM(R[:, 2])), self._clearance))
                elif self._obj_is_cyl[j]:
                    # geom_size = (radius, half_height, _) for a cylinder.
                    dist_exprs.append((_sphere_cylinder_distance(
                        p_arm, arm_radius, ca.DM(center), ca.DM(R),
                        float(self._obj_half_extents[j][0]), float(self._obj_half_extents[j][1])), obj_clr))
                else:
                    dist_exprs.append((_sphere_sphere_distance(
                        p_arm, arm_radius, ca.DM(center), self._obj_rbound[j]), obj_clr))

        for d, clr in dist_exprs:
            opti.subject_to(d >= clr)

        q0 = (q_init if q_init is not None
              else (q_bias if q_bias is not None else np.zeros(n)))
        opti.set_initial(q, q0)
        opti.solver("ipopt", self._solver_opts)

        def _diagnostics(value_fn):
            """value_fn: sol.value or opti.debug.value — evaluates a casadi expr at
            the current iterate. Reports the physically meaningful quantities (mm of
            site error, mm of collision slack) that IPOPT's scaled `tol`/`obj` don't
            directly convey, so a converged-but-still-far-off solution is visible.
            Also stashes the numeric values into self.last_metrics for external
            consumers (e.g. the live dashboard in kinova_leap_pick_place.py)."""
            site_err_mm = [
                float(np.linalg.norm(value_fn(cb(q)) - ca.DM(tgt))) * 1000.0
                for cb, tgt in zip(site_cbs, targets)
            ]
            self.last_metrics['site_err_mm'] = site_err_mm
            print(f"[IPOPT]   site errors (mm): "
                  f"{['%.2f' % e for e in site_err_mm]}  max={max(site_err_mm):.2f}")
            if q_bias is not None:
                dq = np.array(value_fn(q)).flatten() - np.asarray(q_bias)
                print(f"[IPOPT]   posture term: {self._posture_weight * float(dq @ dq):.4g}"
                      f"  (||q-q_bias||={float(np.linalg.norm(dq)):.3g} rad)")
            if dist_exprs:
                # Margin above each constraint's own required clearance (mixed: reduced for
                # active fingers vs objects, full vs the floor).
                slacks_mm = [(float(value_fn(expr)) - clr) * 1000.0 for expr, clr in dist_exprs]
                i_min = int(np.argmin(slacks_mm))
                self.last_metrics['min_slack_mm'] = slacks_mm[i_min]
                print(f"[IPOPT]   collision margin (mm): min={slacks_mm[i_min]:.2f}"
                      f"  n_binding(<0.1mm)={sum(s < 0.1 for s in slacks_mm)}/{len(slacks_mm)}")
            if inward_dirs is not None and self._orient_weight > 0:
                ang = [float(np.degrees(np.arccos(np.clip(
                        float(np.dot(np.array(value_fn(cb(q))).flatten(), d_in)), -1.0, 1.0))))
                    for cb, d_in in zip(axis_cbs, inward_dirs)]
                self.last_metrics['pad_deg'] = ang
                print(f"[IPOPT]   pad-normal error (deg): "
                    f"{['%.1f' % a for a in ang]}  max={max(ang):.1f}")

        # Reset per-solve metrics; _diagnostics fills in the physical quantities and the
        # try/except branches add solver status. Read via solver.last_metrics after solve().
        self.last_metrics = {}
        try:
            sol  = opti.solve()
            st   = opti.stats()
            self.last_metrics.update(status=st['return_status'], iters=st['iter_count'],
                                     obj=float(sol.value(opti.f)), success=True)
            print(f"[IPOPT] {st['return_status']}  iters={st['iter_count']}"
                  f"  obj={float(sol.value(opti.f)):.4g}")
            _diagnostics(sol.value)
            return np.array(sol.value(q))
        except RuntimeError:
            st = opti.stats()
            self.last_metrics.update(status=st.get('return_status', '?'),
                                     iters=st.get('iter_count', '?'), success=False)
            print(f"[IPOPT] FAILED: {st.get('return_status','?')}  iters={st.get('iter_count','?')}")
            _diagnostics(opti.debug.value)
            return np.array(opti.debug.value(q))


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

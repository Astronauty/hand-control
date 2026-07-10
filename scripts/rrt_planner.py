import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d


class RRTPlanner:
    """
    RRT-Connect planner in joint space with MuJoCo geometry-based collision checking.
    Grows two trees simultaneously (from start and goal) and connects them.
    """

    def __init__(
        self,
        model,
        finger_geom_names,
        obj_body_names,
        extra_obj_geom_names=(),
        step_size=0.05,
        goal_bias=0.1,
        goal_tol=0.05,
        max_iter=30000,
        clearance=0.020,
        n_smooth=100,
        densify_spacing=0.02,
        smooth_sigma=3.0,
        n_robot=None,
        n_plan=None,
    ):
        self.model = model
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_tol = goal_tol
        self.max_iter = max_iter
        self.clearance = clearance
        self.n_smooth = n_smooth
        self.densify_spacing = densify_spacing
        self.smooth_sigma = smooth_sigma
        # Restrict planning to the first n_robot joints; objects occupy the rest.
        self._n_robot = n_robot if n_robot is not None else model.nv
        # Only randomise the first n_plan joints during sampling; the remaining joints
        # (typically hand DOF) are fixed at the goal values in every random sample and
        # are excluded from nearest-neighbour distance so the high-dimensional finger
        # space doesn't swamp the lower-dimensional arm space.
        self._n_plan = n_plan if n_plan is not None else self._n_robot

        self._data = mujoco.MjData(model)
        self._q_lo = model.jnt_range[:self._n_robot, 0].copy()
        self._q_hi = model.jnt_range[:self._n_robot, 1].copy()
        self._pair_clearance = {}   # (finger_gid, obj_gid) -> clearance override; set per plan()

        # Continuous (unlimited) revolute joints within the planned range live on a circle
        # (S^1): theta and theta+-2pi are the same configuration. Mark them so distance,
        # steering, and edge interpolation take the SHORT arc across the +-pi seam instead
        # of unwinding a near-full turn. Keyed off the model (hinge + not limited) rather
        # than hardcoded indices.
        self._circular = np.zeros(self._n_robot, dtype=bool)
        for j in range(model.njnt):
            adr = model.jnt_qposadr[j]
            if (adr < self._n_plan
                    and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                    and not model.jnt_limited[j]):
                self._circular[adr] = True
        # Unlimited joints compile with jnt_range == [0, 0], so sampling uniform(lo, hi)
        # would pin every circular joint to exactly 0 in every random sample — the planner
        # would only ever explore the limited joints (3 of the Gen3's 7) plus goal-bias
        # pulls. Sample circular joints over a full turn instead; with the wrap-aware
        # metric/steer any 2pi branch is equivalent, so [-pi, pi) covers the whole circle.
        self._q_lo[self._circular] = -np.pi
        self._q_hi[self._circular] = np.pi

        self._finger_geoms = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in finger_geom_names
        ]
        # Bounding-sphere radii for the broadphase prefilter in _is_free: the exact
        # distance can never be below ||c1-c2|| - rb1 - rb2, so pairs whose sphere bound
        # already clears the threshold skip mj_geomDistance entirely. Besides the speedup,
        # this guards against a MuJoCo 3.3.x GJK instability where mj_geomDistance
        # spuriously returns 0.0 for well-SEPARATED box-box pairs at near-face-parallel
        # poses (flips with a 1-ulp qpos change) — those phantom "contacts" rejected huge
        # swaths of genuinely free space and starved the planner. Planes have rbound == 0
        # (no bounding sphere); they get an analytic point-plane bound instead (see
        # _pair_lower_bounds).
        self._rbound = model.geom_rbound.copy()

        self._obj_geoms = []
        for body_name in obj_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            start = model.body_geomadr[body_id]
            for i in range(model.body_geomnum[body_id]):
                self._obj_geoms.append(start + i)
        # Extra individual obstacle geoms by name (e.g. the ground plane, which lives on
        # the world body alongside unrelated visual markers we don't want to sweep in).
        for gname in extra_obj_geom_names:
            self._obj_geoms.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname))

        # Vectorized-broadphase precomputation. _is_free runs tens of thousands of times
        # per plan and profiling showed it at ~97% of plan() wall time, dominated by (a)
        # the pure-Python pair loop over all finger×obstacle pairs and (b) exact
        # mj_geomDistance queries against the floor PLANE, which rbound==0 exempted from
        # the sphere prefilter (measured: 100% of all GJK calls). Both are fixed by
        # computing every pair's distance lower bound in one numpy pass (sphere-sphere
        # for finite geoms, point-plane for planes) and only running the exact query on
        # pairs whose bound fails to clear their required clearance (~0.3 per check).
        self._fg_arr     = np.array(self._finger_geoms, dtype=int)
        self._og_arr     = np.array(self._obj_geoms, dtype=int)
        self._rb_f       = self._rbound[self._fg_arr]
        self._rb_o       = self._rbound[self._og_arr]
        self._plane_cols = np.nonzero(np.array(
            [model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in self._obj_geoms]))[0]
        self._fg_index   = {g: i for i, g in enumerate(self._finger_geoms)}
        self._og_index   = {g: j for j, g in enumerate(self._obj_geoms)}
        # Per-pair clearance matrix — the vectorized counterpart of _pair_clearance,
        # rebuilt by plan() and updated in place by _endpoint_grace.
        self._rebuild_clearance_matrix()

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def _rebuild_clearance_matrix(self):
        """Bake self._pair_clearance into the (n_finger, n_obj) matrix _is_free compares
        the pair lower bounds against. Must be called whenever _pair_clearance is
        replaced wholesale (plan() does); _endpoint_grace maintains both in step."""
        C = np.full((len(self._fg_arr), len(self._og_arr)), self.clearance)
        for (fg, og), clr in self._pair_clearance.items():
            i = self._fg_index.get(fg)
            j = self._og_index.get(og)
            if i is not None and j is not None:
                C[i, j] = clr
        self._clr_mat = C

    def _pair_lower_bounds(self):
        """(n_finger, n_obj) matrix of distance lower bounds at the pose currently in
        self._data (mj_kinematics already run). Finite-geom pairs use the bounding-sphere
        bound ||c1-c2|| - rb1 - rb2; plane columns use the exact point-plane bound
        n̂·(c - p_plane) - rb (the plane's world normal is local +z, i.e. the third
        column of its geom_xmat). Every entry is a true lower bound on the exact
        geom-geom distance, so comparing it against the clearance matrix can only skip
        pairs mj_geomDistance provably could not flag."""
        xpos = self._data.geom_xpos
        P_f  = xpos[self._fg_arr]
        P_o  = xpos[self._og_arr]
        lb   = (np.linalg.norm(P_f[:, None, :] - P_o[None, :, :], axis=2)
                - self._rb_f[:, None] - self._rb_o[None, :])
        for k in self._plane_cols:
            n_hat     = self._data.geom_xmat[self._og_arr[k]].reshape(3, 3)[:, 2]
            lb[:, k]  = (P_f - P_o[k]) @ n_hat - self._rb_f
        return lb

    def _is_free(self, q):
        self._data.qpos[:self._n_robot] = q   # only set robot DOFs; objects stay at snapshot
        mujoco.mj_kinematics(self.model, self._data)
        # Broadphase: one vectorized pass over all pairs; the exact query below runs only
        # for pairs whose lower bound fails their clearance. Per-pair clearance overrides
        # (e.g. active fingertips vs the target object at 0.0 so a close grasp goal isn't
        # self-disqualified) live in _clr_mat — the distance is still CHECKED, so an
        # exempted finger may touch but can never sweep through the object.
        lb = self._pair_lower_bounds()
        fromto = np.zeros(6)
        for i, j in zip(*np.nonzero(lb < self._clr_mat)):
            if mujoco.mj_geomDistance(self.model, self._data, int(self._fg_arr[i]),
                                      int(self._og_arr[j]), 10.0, fromto) < self._clr_mat[i, j]:
                return False
        return True

    def _wrap_diff(self, d):
        """Wrap the circular-joint components of a difference vector into [-pi, pi] (the
        short arc). Operates on the last axis, so it handles both a single delta (n_robot,)
        and a stack of diffs (N, n_plan)."""
        d = np.array(d, dtype=float)
        m = self._circular[:d.shape[-1]]
        if m.any():
            d[..., m] = (d[..., m] + np.pi) % (2 * np.pi) - np.pi
        return d

    def rebranch(self, q_ref, q):
        """Return q with its circular joints shifted onto the 2pi branch nearest q_ref
        (within +-pi). Same physical configuration, but the numeric values no longer force
        a near-full turn relative to q_ref. Use on the goal before planning."""
        q = np.asarray(q, dtype=float).copy()
        m = self._circular[:q.shape[0]]
        q[m] = np.asarray(q_ref)[m] + self._wrap_diff(q - np.asarray(q_ref))[m]
        return q

    def _unwrap_path(self, path):
        """Remove 2pi jumps on circular joints along the path (np.unwrap per joint) so the
        stored waypoints are continuous — the connection between the two trees can meet at
        configs equal mod 2pi but 2pi apart numerically, and every downstream consumer
        (densify, gaussian smooth, the waypoint follower, ghost markers) interpolates
        LINEARLY, which would otherwise re-introduce the long way around."""
        if not self._circular.any() or len(path) < 2:
            return path
        arr = np.array(path)
        for i in np.nonzero(self._circular[:arr.shape[1]])[0]:
            arr[:, i] = np.unwrap(arr[:, i])
        return [row.copy() for row in arr]

    def _edge_free(self, q_a, q_b):
        """Check strictly interior points of edge (endpoints trusted by caller)."""
        delta = self._wrap_diff(q_b - q_a)   # short-arc on circular joints
        # Sample at 0.25× step_size intervals for tighter coverage.
        n_steps = max(2, int(np.ceil(np.linalg.norm(delta) / (0.25 * self.step_size))))
        for i in range(1, n_steps):
            if not self._is_free(q_a + delta * (i / n_steps)):
                return False
        return True

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    def _nearest_idx(self, nodes_arr, q):
        diffs = self._wrap_diff(nodes_arr[:, :self._n_plan] - q[:self._n_plan])
        return int(np.argmin((diffs * diffs).sum(axis=1)))

    def _steer(self, q_from, q_to):
        delta = self._wrap_diff(q_to - q_from)   # shortest arc on circular joints
        d = np.linalg.norm(delta)
        # Move along the (wrapped) delta; on circular joints the result may leave [-pi, pi],
        # which is fine (those joints are unlimited) and is resolved by _unwrap_path at the end.
        return q_from + delta if d <= self.step_size else q_from + delta / d * self.step_size

    def _extend(self, nodes, arr_ref, parents, q_target):
        """One RRT step toward q_target. Returns ('reached'|'advanced'|'trapped', q_new)."""
        idx = self._nearest_idx(arr_ref[0], q_target)
        q_new = self._steer(nodes[idx], q_target)
        if self._is_free(q_new) and self._edge_free(nodes[idx], q_new):
            nodes.append(q_new)
            arr_ref[0] = np.vstack([arr_ref[0], q_new])
            parents.append(idx)
            # Wrap-aware reached test: q_new can equal q_target mod 2pi but differ by 2pi.
            reached = np.linalg.norm(self._wrap_diff(q_new - q_target)[:self._n_plan]) < 1e-9
            return ("reached" if reached else "advanced"), q_new
        return "trapped", None

    def _connect(self, nodes, arr_ref, parents, q_target):
        """Greedily extend tree toward q_target until reached or trapped."""
        status = "advanced"
        q_new = None
        while status == "advanced":
            status, q_new = self._extend(nodes, arr_ref, parents, q_target)
        return status, q_new

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------

    def _extract_path(self, nodes, parents):
        path, i = [], len(nodes) - 1
        while i != -1:
            path.append(nodes[i])
            i = parents[i]
        path.reverse()
        return path

    def _smooth(self, path):
        for _ in range(self.n_smooth):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 1)
            j = np.random.randint(i + 1, len(path))
            if self._edge_free(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path

    def _gauss_smooth(self, path):
        """Smooth the densified path with a Gaussian kernel applied per joint.
        Any waypoint the kernel pushes into clearance violation is reverted to its
        original (pre-smooth) value so the clearance guarantee is preserved."""
        if self.smooth_sigma is None or self.smooth_sigma <= 0 or len(path) < 3:
            return path
        original = np.array(path)                                    # (N, nq)
        arr = gaussian_filter1d(original, sigma=self.smooth_sigma, axis=0, mode='nearest')
        for i in range(len(arr)):
            if not self._is_free(arr[i]):
                arr[i] = original[i]
        return list(arr)

    def _densify(self, path):
        """Linearly interpolate between waypoints at densify_spacing intervals.
        Points on a verified edge are collision-free by construction."""
        if self.densify_spacing is None or len(path) < 2:
            return path
        dense = []
        for i in range(len(path) - 1):
            q_a, q_b = path[i], path[i + 1]
            delta = q_b - q_a
            n = max(1, int(np.ceil(np.linalg.norm(delta) / self.densify_spacing)))
            for k in range(n):
                dense.append(q_a + delta * (k / n))
        dense.append(path[-1])
        return dense

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def geom_id(self, name):
        """Look up a geom id by name (helper for building pair_clearance at the call site)."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _endpoint_grace(self, q):
        """Relax the clearance of any pair that already violates it at endpoint q (which the
        caller commits to regardless — the arm may still hug the object it just released at
        q_start, and the IK can leave an active fingertip marginally inside its allowance at
        q_goal). The pair's clearance drops to just under its distance at q, so the endpoint
        is admissible and the pair can move AWAY freely, but can never get any deeper than
        it already is. Pairs satisfying their clearance at q are untouched."""
        self._data.qpos[:self._n_robot] = q
        mujoco.mj_kinematics(self.model, self._data)
        # Same vectorized broadphase as _is_free: a pair whose lower bound clears its
        # requirement can't need grace (and the exact query is the one vulnerable to
        # phantom 0.0 results — see __init__). Grace updates the clearance matrix and
        # the dict together so the two views never diverge.
        lb = self._pair_lower_bounds()
        fromto = np.zeros(6)
        for i, j in zip(*np.nonzero(lb < self._clr_mat)):
            fg, og = int(self._fg_arr[i]), int(self._og_arr[j])
            d = mujoco.mj_geomDistance(self.model, self._data, fg, og, 10.0, fromto)
            if d < self._clr_mat[i, j]:
                self._pair_clearance[(fg, og)] = d - 1e-4
                self._clr_mat[i, j]            = d - 1e-4

    def plan(self, q_start, q_goal, pair_clearance=None):
        """
        Plan a collision-free joint-space path from q_start to q_goal.
        Uses RRT-Connect (bidirectional) for reliability.
        Returns a list of configs (start…goal), or None on failure.

        q_start is trusted to be collision-free (not checked).
        q_goal should be clearly in free space (e.g. a pre-grasp config).

        pair_clearance : dict {(finger_geom_id, obj_geom_id): clearance_m} overriding the
                         default clearance per pair — use 0.0 to let the active fingertips
                         approach (touch) the target object so a close pregrasp goal is
                         admissible while still forbidding penetration. Pairs already closer
                         than their clearance at q_start/q_goal are further relaxed to their
                         endpoint distance (see _endpoint_grace), never below it.
        """
        self._pair_clearance = dict(pair_clearance or {})
        self._rebuild_clearance_matrix()
        self._endpoint_grace(q_start)
        self._endpoint_grace(q_goal)
        # Stable references to start/goal trees — names never change even after swap.
        s_nodes, s_arr, s_par = [q_start.copy()], [np.array([q_start])], [-1]
        g_nodes, g_arr, g_par = [q_goal.copy()],  [np.array([q_goal])],  [-1]

        # Working aliases; Python rebinds these on swap but the underlying list objects
        # (s_nodes, g_nodes, …) are still reachable via their stable names for extraction.
        a_nodes, a_arr, a_par = s_nodes, s_arr, s_par
        b_nodes, b_arr, b_par = g_nodes, g_arr, g_par

        for _ in range(self.max_iter):
            # Goal-biased sampling: with probability goal_bias, pull toward the opposite
            # tree's root rather than a random config — the main driver of convergence on
            # high-DOF chains where pure-random sampling rarely lands near the other tree.
            if np.random.random() < self.goal_bias:
                q_rand = b_nodes[0].copy()
            else:
                # Non-planned joints (e.g. hand DOF) are fixed at goal values in every
                # random sample so only the arm joints vary during tree expansion.
                q_rand = q_goal.copy()
                q_rand[:self._n_plan] = np.random.uniform(
                    self._q_lo[:self._n_plan], self._q_hi[:self._n_plan])

            status, q_new = self._extend(a_nodes, a_arr, a_par, q_rand)
            if status != "trapped":
                conn_status, _ = self._connect(b_nodes, b_arr, b_par, q_new)
                if conn_status == "reached":
                    # Always extract start→goal regardless of which alias holds which tree.
                    path_s = self._extract_path(s_nodes, s_par)
                    path_g = self._extract_path(g_nodes, g_par)
                    path_g.reverse()
                    # Unwrap circular joints FIRST (removes the 2pi jump where the two trees
                    # meet) so the subsequent linear densify/smooth take the short arc.
                    raw = self._unwrap_path(path_s + path_g)
                    path = self._gauss_smooth(self._densify(self._smooth(raw)))
                    print(f"[RRT] Found path: {len(path)} waypoints")
                    return path

            # Swap so both trees grow at roughly equal rates.
            a_nodes, b_nodes = b_nodes, a_nodes
            a_arr,   b_arr   = b_arr,   a_arr
            a_par,   b_par   = b_par,   a_par

        print(f"[RRT] Failed after {self.max_iter} iterations")
        return None

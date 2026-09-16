import os
import time

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

        # Diagnostic: RRT_NO_COLLISION=1 makes _is_free() unconditionally True (see there).
        # Timing experiments only -- the path it yields is not collision-free. Announced
        # loudly at construction so a run that accidentally has it set is obvious.
        self._no_collision = os.environ.get('RRT_NO_COLLISION', '') == '1'
        if self._no_collision:
            print("[RRT] *** RRT_NO_COLLISION=1 -- collision checking DISABLED. "
                  "Paths are UNSAFE; timing diagnostics only. ***")

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
        # Per-geom type + half-extents for the PHANTOM-0.0 verification (see _is_free):
        # mj_geomDistance's GJK returns spurious 0.0 for well-separated BOX-BOX pairs at
        # near-face-parallel poses. When a flagged pair is box-box, we cross-check with an
        # analytic OBB-vs-OBB lower bound (separating-axis on the 6 face normals); if the
        # geoms are clearly apart, the mj 0.0 is a phantom and the pair is not a real hit.
        self._geom_type = model.geom_type.copy()
        self._geom_size = model.geom_size.copy()
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

    def admissibility(self, q_start, q_goal, pair_clearance=None):
        """Diagnostic: replicate plan()'s setup (rebuild clearance + endpoint grace) and
        report whether each endpoint is collision-free, plus the first blocking (finger,
        object) geom pair and its distance for a failing endpoint. Does NOT plan — a cheap
        pre-flight so 'RRT failed' can be attributed to an in-collision goal/start (the
        multi-hull-clearance bug's signature) vs a genuine narrow-passage planning failure.
        Returns a dict; leaves _pair_clearance/_clr_mat set up as plan() would (harmless)."""
        self._pair_clearance = dict(pair_clearance or {})
        self._rebuild_clearance_matrix()
        self._endpoint_grace(q_start)
        self._endpoint_grace(q_goal)

        def _probe(q):
            self._data.qpos[:self._n_robot] = q
            mujoco.mj_kinematics(self.model, self._data)
            lb = self._pair_lower_bounds()
            fromto = np.zeros(6)
            worst = None
            for i, j in zip(*np.nonzero(lb < self._clr_mat)):
                fg, og = int(self._fg_arr[i]), int(self._og_arr[j])
                d = mujoco.mj_geomDistance(self.model, self._data, fg, og, 10.0, fromto)
                if d < self._clr_mat[i, j]:
                    if worst is None or d < worst[2]:
                        worst = (fg, og, float(d), float(self._clr_mat[i, j]))
            return worst

        w_s, w_g = _probe(q_start), _probe(q_goal)

        def _name(gid):
            return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or gid

        def _pack(w):
            if w is None:
                return None
            return {'finger': _name(w[0]), 'obj': _name(w[1]),
                    'dist_m': round(w[2], 5), 'need_m': round(w[3], 5)}

        return {'start_free': w_s is None, 'goal_free': w_g is None,
                'start_block': _pack(w_s), 'goal_block': _pack(w_g)}

    def _obb_separation(self, ga, gb):
        """Conservative lower bound on the distance between two BOX geoms (ga, gb) at the
        current self._data pose, via the separating-axis theorem on the 6 face normals (3
        per box). For each candidate axis, project both boxes' half-extents and compare
        against the centre-offset along that axis; the largest positive gap over all 6 axes
        is a valid lower bound on the true OBB distance (0 if none separates). Used ONLY to
        reject mj_geomDistance's box-box phantom 0.0 — a lower bound can only ever mark a
        pair as MORE separated, never falsely clear a real collision."""
        d = self._data
        ca = d.geom_xpos[ga]; Ra = d.geom_xmat[ga].reshape(3, 3); ha = self._geom_size[ga]
        cb = d.geom_xpos[gb]; Rb = d.geom_xmat[gb].reshape(3, 3); hb = self._geom_size[gb]
        t = cb - ca
        best = 0.0
        for R, h in ((Ra, ha), (Rb, hb)):
            for k in range(3):
                axis = R[:, k]
                # projected radius of each box onto this axis = sum |axis·(box axis)|*half
                ra = np.sum(np.abs(Ra.T @ axis) * ha)
                rb = np.sum(np.abs(Rb.T @ axis) * hb)
                gap = abs(float(t @ axis)) - (ra + rb)
                if gap > best:
                    best = gap
        return best

    def _is_free(self, q):
        _t0 = time.perf_counter()
        # DIAGNOSTIC BYPASS (RRT_NO_COLLISION=1). Declares EVERY configuration free, so
        # plan() measures pure search cost with the collision check removed. The resulting
        # path is UNSAFE BY CONSTRUCTION -- it may sweep the hand through the table, the
        # bin, or the objects -- so this is for timing experiments ONLY, never a data run.
        # Deliberately placed AFTER the timer start and BEFORE mj_kinematics so the call
        # is still counted in _isfree_n: that keeps the "checks" figure comparable between
        # the on and off runs (same number of queries, ~0 cost each).
        if self._no_collision:
            if hasattr(self, "_isfree_n"):
                self._isfree_ms += (time.perf_counter() - _t0) * 1e3
                self._isfree_n += 1
            return True
        self._data.qpos[:self._n_robot] = q   # only set robot DOFs; objects stay at snapshot
        mujoco.mj_kinematics(self.model, self._data)
        # Broadphase: one vectorized pass over all pairs; the exact query below runs only
        # for pairs whose lower bound fails their clearance. Per-pair clearance overrides
        # (e.g. active fingertips vs the target object at 0.0 so a close grasp goal isn't
        # self-disqualified) live in _clr_mat — the distance is still CHECKED, so an
        # exempted finger may touch but can never sweep through the object.
        lb = self._pair_lower_bounds()
        fromto = np.zeros(6)
        result = True
        _BOX = mujoco.mjtGeom.mjGEOM_BOX
        for i, j in zip(*np.nonzero(lb < self._clr_mat)):
            fg, og = int(self._fg_arr[i]), int(self._og_arr[j])
            d = mujoco.mj_geomDistance(self.model, self._data, fg, og, 10.0, fromto)
            if d < self._clr_mat[i, j]:
                # PHANTOM-0.0 REJECTION: mj_geomDistance's GJK spuriously returns ~0 for
                # well-separated BOX-BOX pairs near face-parallel (verified live: a palm box
                # 35cm above the table slab reported 0.0 while analytic distance was 347mm —
                # freezing the RRT). For a box-box pair reporting ~0, verify with the analytic
                # OBB separating-axis lower bound; if the boxes are clearly apart it's a
                # phantom, not a collision. The lower bound only ever declares MORE
                # separation, so this can never mask a genuine overlap.
                if (d <= 1e-6 and self._geom_type[fg] == _BOX
                        and self._geom_type[og] == _BOX
                        and self._obb_separation(fg, og) > max(self._clr_mat[i, j], 1e-3)):
                    continue   # phantom — treat as free for this pair
                result = False
                break
        # Diagnostic accounting (see plan()): total collision-check time + call count, so a
        # SPEED failure (per-check cost dominating on mesh/SDF geoms) is separable from a
        # CONNECTIVITY failure. Guarded on the attr so pre-plan() calls don't error.
        if hasattr(self, "_isfree_n"):
            self._isfree_ms += (time.perf_counter() - _t0) * 1e3
            self._isfree_n += 1
        return result

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
        """Shortcut-smooth the raw path. RRT-Connect raw paths can be long and convoluted
        (an unbalanced search grows one tree to ~1000 nodes, so the extracted path threads
        through many of them). A fixed 100 random shortcuts barely dents such a path — the
        result stayed convoluted. Two upgrades:
          1. A GREEDY forward pass: from each anchor, connect to the FARTHEST reachable
             later waypoint in one shot, then continue from there. One O(n·checks) sweep
             collapses most of the detour deterministically (not luck-of-the-draw i/j).
          2. Then the random shortcut polish, its budget scaled to the (now-short) path,
             to clean up whatever the greedy pass left."""
        if len(path) <= 2:
            return path

        # --- greedy forward shortcut: farthest CONTIGUOUS reachable jump per anchor ---
        # Forward scan (not far-end backtrack): from anchor i, advance j while edge i->j+1
        # stays free; jump to the last free j, repeat. ~O(n) checks. One pass can stall at a
        # tricky mid-path spot (a single blocked edge caps the jump), leaving many short
        # segments — so run forward+reverse passes ALTERNATELY to convergence: a reverse
        # pass shortcuts what the forward pass's stall left behind. This is what collapses a
        # genuinely convoluted RRT path (start_tree in the hundreds) to a near-straight one.
        def _forward(p):
            out = [p[0]]; i = 0; n = len(p)
            while i < n - 1:
                j = i + 1
                while j + 1 < n and self._edge_free(p[i], p[j + 1]):
                    j += 1
                out.append(p[j]); i = j
            return out

        prev_len = None
        for _ in range(8):                      # converges in 1-3 passes in practice
            path = _forward(path)
            path = list(reversed(_forward(list(reversed(path)))))
            if prev_len is not None and len(path) >= prev_len:
                break                            # no further collapse
            prev_len = len(path)
            if len(path) <= 2:
                break

        # --- random shortcut polish: catches non-contiguous shortcuts the greedy passes
        # (which only join along the existing order) can't. Budget scaled to path length.
        budget = max(self.n_smooth, 20 * len(path))
        for _ in range(budget):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 1)
            j = np.random.randint(i + 1, len(path))
            if j - i >= 2 and self._edge_free(path[i], path[j]):
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

        # Diagnostics (read by callers after plan()): distinguishes a SPEED failure (few
        # iterations reached before the time/iter budget, per-check cost dominating) from a
        # CONNECTIVITY failure (all iters ran, trees grew large but never met — a narrow
        # passage). _isfree_ms/_isfree_n time the collision check that dominates cost.
        self.last_iters = 0
        self._isfree_ms = 0.0
        self._isfree_n = 0
        self.first_step_block = None   # diagnostic: why the first extension is rejected

        # DIAGNOSTIC: probe one steered step from each root toward the other root and, if it
        # is rejected, name the first blocking (finger/arm geom, object/obstacle geom) pair.
        # A tree frozen at 1 node means every _extend is 'trapped' — this says exactly which
        # collision pair does it (e.g. a curling finger hitting the object, or arm vs table).
        try:
            _probe_from, _probe_to = q_start, q_goal
            _q_step = self._steer(_probe_from, _probe_to)
            if not self._is_free(_q_step):
                self._data.qpos[:self._n_robot] = _q_step
                mujoco.mj_kinematics(self.model, self._data)
                lb = self._pair_lower_bounds(); _ft = np.zeros(6); _worst = None
                for i, j in zip(*np.nonzero(lb < self._clr_mat)):
                    fg, og = int(self._fg_arr[i]), int(self._og_arr[j])
                    d = mujoco.mj_geomDistance(self.model, self._data, fg, og, 10.0, _ft)
                    if d < self._clr_mat[i, j] and (_worst is None or d < _worst[2]):
                        _worst = (fg, og, float(d))
                if _worst is not None:
                    _nm = lambda g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or g
                    fg, og = _worst[0], _worst[1]
                    # PHANTOM CHECK: mj_geomDistance box-box can spuriously return ~0 for
                    # well-separated pairs (GJK instability). Cross-check against the TRUE
                    # geometry: the finger geom's world center, the object/obstacle box's
                    # world AABB, and the analytic point-to-box distance from the finger
                    # center to that box. If mj says ~0 but the finger center is clearly
                    # outside the box (analytic dist >> 0), it's a phantom, not a real hit.
                    fc = self._data.geom_xpos[fg].copy()
                    oc = self._data.geom_xpos[og].copy()
                    os_ = self.model.geom_size[og].copy()
                    otype = int(self.model.geom_type[og])
                    # box surface distance from finger center (box frame), if og is a box
                    analytic = None
                    if otype == mujoco.mjtGeom.mjGEOM_BOX:
                        R = self._data.geom_xmat[og].reshape(3, 3)
                        loc = R.T @ (fc - oc)                 # finger center in box frame
                        outside = np.maximum(np.abs(loc) - os_, 0.0)
                        analytic = float(np.linalg.norm(outside))  # >0 iff center outside box
                    self.first_step_block = {
                        'finger': _nm(fg), 'obj': _nm(og),
                        'mj_dist_mm': round(_worst[2] * 1e3, 2),
                        'analytic_center_to_box_mm': (round(analytic * 1e3, 2)
                                                      if analytic is not None else None),
                        'finger_xyz': [round(v, 3) for v in fc],
                        'obj_box_center': [round(v, 3) for v in oc],
                        'obj_box_half': [round(v, 3) for v in os_],
                    }
                    _ph = (analytic is not None and _worst[2] < 1e-4 and analytic > 0.02)
                    print(f"[RRT] first step blocked: {_nm(fg)} vs {_nm(og)}  "
                          f"mj={_worst[2]*1e3:.1f}mm  "
                          f"analytic(center->box)={'?' if analytic is None else f'{analytic*1e3:.1f}mm'}"
                          f"{'  <-- PHANTOM (mj lies)' if _ph else ''}")
                    print(f"       finger_xyz={self.first_step_block['finger_xyz']}  "
                          f"box_center={self.first_step_block['obj_box_center']}  "
                          f"half={self.first_step_block['obj_box_half']}")
        except Exception:
            import traceback; traceback.print_exc()

        for _n_it in range(self.max_iter):
            self.last_iters = _n_it + 1
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
                    _sc = self._smooth(raw)          # shortcut path (pre-densify)
                    path = self._gauss_smooth(self._densify(_sc))
                    self.last_start_tree = len(s_nodes)
                    self.last_goal_tree = len(g_nodes)
                    # Convolution diagnostics on the SHORTCUT path (arm joints only): a
                    # straight path has ratio ~1.0; >1 means the shortcut left detours.
                    _a = np.array(_sc)[:, :self._n_plan]
                    _seg = float(np.linalg.norm(np.diff(_a, axis=0), axis=1).sum()) if len(_a) > 1 else 0.0
                    _straight = float(np.linalg.norm(_a[-1] - _a[0])) if len(_a) > 1 else 0.0
                    self.last_raw_wp = len(raw)
                    self.last_shortcut_wp = len(_sc)
                    self.last_conv_ratio = round(_seg / _straight, 3) if _straight > 1e-6 else None
                    print(f"[RRT] Found path: {len(path)} wp "
                          f"(raw={len(raw)} -> shortcut={len(_sc)} -> densified={len(path)}; "
                          f"conv={self.last_conv_ratio}x; {self.last_iters} iters, "
                          f"{self._isfree_n} checks, {self._isfree_ms/1e3:.1f}s)")
                    return path

            # Swap so both trees grow at roughly equal rates.
            a_nodes, b_nodes = b_nodes, a_nodes
            a_arr,   b_arr   = b_arr,   a_arr
            a_par,   b_par   = b_par,   a_par

        # Record final tree sizes so the caller can tell CONNECTIVITY failure (both trees
        # grew large but never met) from a STUCK-goal-tree failure (goal tree ~1 node — it
        # can't take a single valid step out of the grasp pose).
        self.last_start_tree = len(s_nodes)
        self.last_goal_tree = len(g_nodes)
        _avg = (self._isfree_ms / self._isfree_n) if self._isfree_n else 0.0
        print(f"[RRT] Failed after {self.max_iter} iters — trees start={len(s_nodes)} "
              f"goal={len(g_nodes)}; {self._isfree_n} collision-checks, "
              f"{self._isfree_ms/1e3:.1f}s in checks ({_avg:.3f}ms avg)")
        return None

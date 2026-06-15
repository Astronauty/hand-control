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
        step_size=0.05,
        goal_bias=0.1,
        goal_tol=0.05,
        max_iter=30000,
        clearance=0.020,
        n_smooth=100,
        densify_spacing=0.02,
        smooth_sigma=3.0,
        n_robot=None,
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

        self._data = mujoco.MjData(model)
        self._q_lo = model.jnt_range[:self._n_robot, 0].copy()
        self._q_hi = model.jnt_range[:self._n_robot, 1].copy()

        self._finger_geoms = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in finger_geom_names
        ]

        self._obj_geoms = []
        for body_name in obj_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            start = model.body_geomadr[body_id]
            for i in range(model.body_geomnum[body_id]):
                self._obj_geoms.append(start + i)

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def _is_free(self, q):
        self._data.qpos[:self._n_robot] = q   # only set robot DOFs; objects stay at snapshot
        mujoco.mj_kinematics(self.model, self._data)
        fromto = np.zeros(6)
        for fg in self._finger_geoms:
            for og in self._obj_geoms:
                if mujoco.mj_geomDistance(self.model, self._data, fg, og, 10.0, fromto) < self.clearance:
                    return False
        return True

    def _edge_free(self, q_a, q_b):
        """Check strictly interior points of edge (endpoints trusted by caller)."""
        delta = q_b - q_a
        # Sample at 0.25× step_size intervals for tighter coverage.
        n_steps = max(2, int(np.ceil(np.linalg.norm(delta) / (0.25 * self.step_size))))
        for i in range(1, n_steps):
            if not self._is_free(q_a + delta * (i / n_steps)):
                return False
        return True

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_idx(nodes_arr, q):
        diffs = nodes_arr - q
        return int(np.argmin((diffs * diffs).sum(axis=1)))

    def _steer(self, q_from, q_to):
        delta = q_to - q_from
        d = np.linalg.norm(delta)
        return q_to.copy() if d <= self.step_size else q_from + delta / d * self.step_size

    def _extend(self, nodes, arr_ref, parents, q_target):
        """One RRT step toward q_target. Returns ('reached'|'advanced'|'trapped', q_new)."""
        idx = self._nearest_idx(arr_ref[0], q_target)
        q_new = self._steer(nodes[idx], q_target)
        if self._is_free(q_new) and self._edge_free(nodes[idx], q_new):
            nodes.append(q_new)
            arr_ref[0] = np.vstack([arr_ref[0], q_new])
            parents.append(idx)
            status = "reached" if np.linalg.norm(q_new - q_target) < 1e-9 else "advanced"
            return status, q_new
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

    def plan(self, q_start, q_goal):
        """
        Plan a collision-free joint-space path from q_start to q_goal.
        Uses RRT-Connect (bidirectional) for reliability.
        Returns a list of configs (start…goal), or None on failure.

        q_start is trusted to be collision-free (not checked).
        q_goal should be clearly in free space (e.g. a pre-grasp config).
        """
        # Stable references to start/goal trees — names never change even after swap.
        s_nodes, s_arr, s_par = [q_start.copy()], [np.array([q_start])], [-1]
        g_nodes, g_arr, g_par = [q_goal.copy()],  [np.array([q_goal])],  [-1]

        # Working aliases; Python rebinds these on swap but the underlying list objects
        # (s_nodes, g_nodes, …) are still reachable via their stable names for extraction.
        a_nodes, a_arr, a_par = s_nodes, s_arr, s_par
        b_nodes, b_arr, b_par = g_nodes, g_arr, g_par

        for _ in range(self.max_iter):
            q_rand = np.random.uniform(self._q_lo, self._q_hi)

            status, q_new = self._extend(a_nodes, a_arr, a_par, q_rand)
            if status != "trapped":
                conn_status, _ = self._connect(b_nodes, b_arr, b_par, q_new)
                if conn_status == "reached":
                    # Always extract start→goal regardless of which alias holds which tree.
                    path_s = self._extract_path(s_nodes, s_par)
                    path_g = self._extract_path(g_nodes, g_par)
                    path_g.reverse()
                    path = self._gauss_smooth(self._densify(self._smooth(path_s + path_g)))
                    print(f"[RRT] Found path: {len(path)} waypoints")
                    return path

            # Swap so both trees grow at roughly equal rates.
            a_nodes, b_nodes = b_nodes, a_nodes
            a_arr,   b_arr   = b_arr,   a_arr
            a_par,   b_par   = b_par,   a_par

        print(f"[RRT] Failed after {self.max_iter} iterations")
        return None

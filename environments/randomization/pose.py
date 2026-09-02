"""Object-pose randomization primitives.

sample_nonoverlapping_xy is the generic rejection-sampling routine extracted from
kinova_leap_pick_place.py's _randomize_objects (originally inlined there for its
hardcoded 7-primitive scene). benchmarks/ycb_grasp/workspace.py's sample_positions is
the complementary reachable-workspace sampler for YCB objects — this module doesn't
duplicate that; grasp_bench/env.py (Phase 4) composes both as needed.
"""
import numpy as np


def sample_nonoverlapping_xy(n: int, center: np.ndarray, half_extent: np.ndarray,
                             min_sep: float, rng: np.random.Generator,
                             max_tries: int = 2000) -> list[np.ndarray]:
    """Rejection-sample n 2-D positions uniformly in [center-half_extent,
    center+half_extent] with pairwise center-to-center separation >= min_sep.

    Falls back to an unchecked draw for any position that can't find a valid spot
    within max_tries (matches the original _randomize_objects behavior — better to
    place every object than to leave one out because the box got crowded).
    """
    center = np.asarray(center, float)
    half_extent = np.asarray(half_extent, float)
    xy_list: list[np.ndarray] = []
    for _ in range(n):
        for _ in range(max_tries):
            xy = center + rng.uniform(-half_extent, half_extent)
            if all(np.linalg.norm(xy - p) >= min_sep for p in xy_list):
                xy_list.append(xy)
                break
        else:
            xy_list.append(center + rng.uniform(-half_extent, half_extent))
    return xy_list

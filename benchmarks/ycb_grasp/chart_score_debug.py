"""Print the antipodal chart-pair (and best triple) scoring output for one or
more YCB objects, using the new object_uv_atlas.chart_normals_centroids /
filter_bottom_facing_charts. Scratch/debug tool for validating the chart-aware
seeding heuristic BEFORE it's wired into GraspPlanner3D's actual seed
generation (_fixed_antipodal_seed / _seed_pair) -- see the session's seeding
rethink notes.

Score convention (matches the "2 vs 1" / N-finger brainstorm): for a
combination of chart NORMALS {n_1..n_k}, score = -||sum(n_i)|| -- a small
resultant vector norm means the normals surround the object well (good
force-closure candidate geometry), independent of k. For k=2 this reduces to
maximizing -n_1.n_2 (pure antipodal) up to a monotonic transform, so it's a
strict generalization, not a different metric for the 2-finger case.

    python benchmarks/ycb_grasp/chart_score_debug.py
    python benchmarks/ycb_grasp/chart_score_debug.py --object 036_wood_block --top 10
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import mujoco as mj
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from ycb_grasp import scene as S                                                # noqa: E402
from grasp_control import object_uv_atlas as oua                                # noqa: E402


def combo_score(chart_normal: np.ndarray, combo: tuple[int, ...]) -> float:
    """-||sum of unit normals|| -- higher (less negative) is better; 0 is the
    theoretical best for a combo whose normals sum to exactly zero (perfectly
    balanced closure directions)."""
    s = chart_normal[list(combo)].sum(axis=0)
    return -float(np.linalg.norm(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", nargs="+",
                    default=["013_apple", "077_rubiks_cube", "036_wood_block",
                            "009_gelatin_box", "017_orange"])
    ap.add_argument("--top", type=int, default=5, help="how many top pairs/triples to print")
    ap.add_argument("--max-down-deg", type=float, default=60.0)
    args = ap.parse_args()

    for oid in args.object:
        print(f"\n{'='*70}\n{oid}\n{'='*70}")
        # Placed upright-ish (identity quat) just to build the atlas/body — the
        # bottom-facing filter below re-derives up_local from THIS placement's
        # object_R, same as GraspPlanner3D would at solve time.
        m, d, info = S.build([(oid, (0, 0, 0.1), (1, 0, 0, 0))])
        body_name = next(iter(info))
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, body_name)
        atlas, big, area_frac = oua.load_or_build(m, bid)
        chart_normal, chart_centroid = oua.chart_normals_centroids(atlas)

        print(f"  {len(big)} big charts (of {atlas['n_charts']} total): "
             f"{big.tolist()}  area%={np.round(area_frac[big]*100,1).tolist()}")

        # object-local "up" at this placement: identity quat here, so world +z
        # IS object-local +z -- still route it through obj_mat for correctness
        # if this script is ever pointed at a rotated placement.
        d_bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, body_name)
        mj.mj_forward(m, d)
        obj_R = d.xmat[d_bid].reshape(3, 3)
        up_local = obj_R.T @ np.array([0.0, 0.0, 1.0])

        big_filt = oua.filter_bottom_facing_charts(big, chart_normal, up_local,
                                                   max_down_deg=args.max_down_deg)
        dropped = sorted(set(big.tolist()) - set(big_filt.tolist()))
        print(f"  after bottom-facing filter (max_down_deg={args.max_down_deg}): "
             f"{big_filt.tolist()}  (dropped: {dropped})")
        for ci in big:
            n = chart_normal[ci]
            down_deg = np.degrees(np.arccos(np.clip(-np.dot(n, up_local), -1, 1)))
            flag = " <- DROPPED (bottom-facing)" if ci in dropped else ""
            print(f"    chart {ci:3d}  normal={np.round(n,3).tolist()}  "
                 f"area%={area_frac[ci]*100:5.1f}  angle-from-down={down_deg:6.1f}deg{flag}")

        if len(big_filt) < 2:
            print("  fewer than 2 usable charts after filtering -- skip scoring")
            continue

        print(f"\n  -- top {args.top} PAIRS (2-finger) by score = -||n_i+n_j|| --")
        pair_scores = [(combo_score(chart_normal, c), c) for c in combinations(big_filt.tolist(), 2)]
        pair_scores.sort(key=lambda x: -x[0])
        for score, combo in pair_scores[:args.top]:
            ni, nj = chart_normal[combo[0]], chart_normal[combo[1]]
            ang = np.degrees(np.arccos(np.clip(np.dot(ni, nj), -1, 1)))
            dist = np.linalg.norm(chart_centroid[combo[0]] - chart_centroid[combo[1]])
            print(f"    charts {combo}  score={score:+.4f}  normal-angle={ang:6.1f}deg  "
                 f"centroid-dist={dist*1000:6.1f}mm")

        if len(big_filt) >= 3:
            print(f"\n  -- top {args.top} TRIPLES (3-finger) by score = -||n_i+n_j+n_k|| --")
            tri_scores = [(combo_score(chart_normal, c), c) for c in combinations(big_filt.tolist(), 3)]
            tri_scores.sort(key=lambda x: -x[0])
            for score, combo in tri_scores[:args.top]:
                print(f"    charts {combo}  score={score:+.4f}")


if __name__ == "__main__":
    main()

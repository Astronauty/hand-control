"""Does the live recommender NLP push free contacts UP for IK feasibility?

The contact-height study (test_contact_height_sweep.py) showed that raising the antipodal
contacts toward the top of the face dramatically improves reach. Question: when the
recommender treats the contacts as DECISION VARIABLES (free on the face plane), does the
IK cost alone drive them upward on its own — starting from the face-center seed?

To isolate IK behavior we DISABLE the wrench/gamma cost (wrench_constraint=False) so only
w_ik*ik + w_reg*reg drives the solve. Contacts p1/p2 remain free (box face-pin fixes only
the normal-axis coord; vertical + horizontal stay free). We seed at the face CENTER
(fixed antipodal along local x) and measure how far the solver raises the contacts
(dz = solved_contact_z - object_center_z) and the final IK site error, per cube size,
over the 12 operator seeds.

Usage:
    python simulation/test_recommender_contact_lift.py
    python simulation/test_recommender_contact_lift.py --sizes 0.04,0.05,0.06
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco as mj

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from grasp_planner_3d import (GraspConfig3D, GraspPlanner3D,  # noqa: E402
                              _fixed_antipodal_seed)

_SCENE = os.path.join(os.path.dirname(_HERE), 'models', 'scene_pick_place.xml')
_BOX = 'obj_red_box'


def _resize_box(model, half):
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, _BOX + '_geom')
    model.geom_size[gid] = np.array([half, half, half], float)
    s1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _BOX + '_c1')
    s2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _BOX + '_c2')
    model.site_pos[s1] = np.array([-half, 0.0, 0.0])
    model.site_pos[s2] = np.array([half, 0.0, 0.0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', default='0.040,0.050,0.060')
    ap.add_argument('--max-seeds', type=int, default=1,
                    help="seeds per solve; 1 = just the center-face fixed seed")
    args = ap.parse_args()
    sizes = [float(x) for x in args.sizes.split(',')]

    with open(os.path.join(os.path.dirname(_HERE), 'samples.jsonl')) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    t0 = time.time()
    print(f"\nContact-LIFT analysis — recommender NLP, wrench/gamma DISABLED, contacts free")
    print(f"  seed = face center (fixed antipodal, local x);  {len(samples)} operator seeds")
    print("=" * 84)
    print(f"{'full(cm)':>8} {'converged':>10} {'mean_dz(mm)':>12} {'max_dz(mm)':>11} "
          f"{'med_ik(mm)':>11} {'note':>18}")
    print("-" * 84)

    for half in sizes:
        model = mj.MjModel.from_xml_path(_SCENE)
        _resize_box(model, half)
        act_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _BOX)

        dzs, iks, n_conv = [], [], 0
        for s in samples:
            data = mj.MjData(model)
            oq = np.asarray(s['obj_qpos'], float)
            jadr = model.jnt_qposadr[model.body_jntadr[bid]]
            data.qpos[jadr:jadr + 7] = oq[:7]
            data.qpos[jadr + 2] = half   # rest bottom on table
            for i, idx in enumerate(act_idx):
                data.qpos[idx] = s['q_seed'][i]
            mj.mj_forward(model, data)

            # wrench_constraint=False -> only IK+posture drive the solve; contacts free.
            cfg = GraspConfig3D(obj_geom=_BOX + '_geom', obj_body=_BOX, max_iter=200,
                                wrench_constraint=False)
            planner = GraspPlanner3D(model, mj.MjData(model), cfg)
            planner.data.qpos[:] = data.qpos[:]
            mj.mj_forward(model, planner.data)
            q_ref = np.array([data.qpos[i] for i in act_idx])
            obj_pos = data.xpos[bid].copy()
            R_WO = data.xmat[bid].reshape(3, 3)

            # Seed exactly at the face-center antipodal pair (local x).
            seed = _fixed_antipodal_seed(
                planner._obj_geom_type, planner._obj_size, obj_pos, R_WO, [1.0, 0.0, 0.0])
            res = planner.solve(q_ref, obj_pos,
                                p1_init=seed['p1'], p2_init=seed['p2'],
                                d1=-seed['n1_in'], d2=-seed['n2_in'])
            if res.get('p1') is None:
                continue
            if res.get('status') == 'converged':
                n_conv += 1
            # Contact z relative to object center, in object frame (dz>0 = raised up).
            p1_O = R_WO.T @ (np.asarray(res['p1'], float) - obj_pos)
            p2_O = R_WO.T @ (np.asarray(res['p2'], float) - obj_pos)
            dz = 0.5 * (p1_O[2] + p2_O[2])       # avg vertical shift from center
            dzs.append(dz * 1e3)
            # IK error: verify()'s site-to-target residual
            info = planner.verify(res)
            iks.append(max(info.get('ik_thumb_mm', 1e9), info.get('ik_index_mm', 1e9)))

        dzs = np.array(dzs); iks = np.array(iks)
        note = 'lifts up' if (len(dzs) and np.mean(dzs) > 2) else \
               ('sinks' if (len(dzs) and np.mean(dzs) < -2) else 'stays ~center')
        print(f"{half*200:>8.1f} {n_conv:>7}/{len(samples):<2} "
              f"{np.mean(dzs) if len(dzs) else 0:>12.1f} "
              f"{np.max(dzs) if len(dzs) else 0:>11.1f} "
              f"{np.median(iks) if len(iks) else 0:>11.1f} {note:>18}  "
              f"[{time.time()-t0:.0f}s]")
    print("=" * 84)
    print("dz = mean contact height above object center (mm). POSITIVE = the free-contact")
    print("solve raised the contacts toward the top on its own (what we want). If dz stays")
    print("~0, the IK cost does NOT lift contacts and a seeding/bias change is needed.\n")


if __name__ == '__main__':
    main()

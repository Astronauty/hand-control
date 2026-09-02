"""Run a few constrained-IK solves against floating YCB objects and render them.

    python benchmarks/ycb_grasp/ik_demo.py                  # 3 solves, render to PNG
    python benchmarks/ycb_grasp/ik_demo.py --view           # interactive viewer
    python benchmarks/ycb_grasp/ik_demo.py --objects 025_mug 011_banana

Targets are an antipodal pair on the object's minor principal axis -- the
direction a two-finger pinch would naturally close along -- placed on the hull
surface and handed to the index and thumb fingertip sites.

NOTE ON OBJECT COLLISION. `ConstrainedIKSolver` has no mesh branch: a mesh geom
falls through to `_sphere_sphere_distance` and is modelled by its bounding
sphere (constrained_ik.py:890). With the current fine decomposition that is a
union of one sphere per hull -- much better than a single sphere over the whole
object, but still an over-approximation that inflates thin features. `--no-obj-
collision` drops the object from the constraint set to separate "the IK cannot
reach" from "the sphere proxy is blocking it".
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from grasp_control import ConstrainedIKSolver, SpatialIKSolver      # noqa: E402
from grasp_control.constrained_ik import configure_sqp              # noqa: E402
from kinova_common.constants import (FINGER_CODE, FINGER_SET,       # noqa: E402
                                     FINGER_TIP_SITES, GEN3_XML)
from ycb_grasp import scene as S, workspace as W                    # noqa: E402

N_ROBOT = 23
PAD_BACKOFF = 0.004    # fingertip site sits behind the pad surface


def home_bias():
    """Gen3 home arm pose + a lightly curled hand, as the IK null-space bias."""
    hm = mj.MjModel.from_xml_path(str(REPO / GEN3_XML))
    q = np.zeros(N_ROBOT)
    q[:7] = hm.key("home").qpos[:7]
    q[11:15] = [1.2, 0.0, 0.5, 0.5]
    q[15:19] = [1.2, 0.0, 0.5, 0.5]
    return q


def robot_geom_names(model):
    """Named collision geoms on the LEAP fingers, palm and wrist."""
    prefixes = tuple(f"leap_{c}_" for c in FINGER_CODE.values())
    out = []
    for g in range(model.ngeom):
        n = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g)
        if not n or model.geom_contype[g] == 0:
            continue
        bn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if bn.startswith(prefixes) or bn in ("leap_palm", "bracelet_link"):
            out.append(n)
    return out


def clearance_by_geom(names):
    """Live tiers: contact tips unconstrained, inner links relaxed, rest +2mm."""
    out = {}
    for g in names:
        if "_ds_" in g or g.endswith("_tip"):
            out[g] = -1.0
        elif g.startswith(("leap_if_md", "leap_th_px")):
            out[g] = -0.010
        else:
            out[g] = 0.002
    return out


_VERT_CACHE = {}


def object_hull_verts(obj_id):
    """Collision-hull vertices of a YCB object in its body frame, from a
    standalone compile -- so poses and targets can be worked out before the
    benchmark scene exists."""
    if obj_id in _VERT_CACHE:
        return _VERT_CACHE[obj_id]
    m = mj.MjSpec.from_file(str(S.YCB_DIR / obj_id / f"{obj_id}.xml")).compile()
    out = []
    for g in range(m.ngeom):
        if m.geom_group[g] != 3:
            continue
        vid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[vid], m.mesh_vertnum[vid]
        R = np.zeros(9)
        mj.mju_quat2Mat(R, m.geom_quat[g])
        out.append(m.mesh_vert[a:a + n] @ R.reshape(3, 3).T + m.geom_pos[g])
    V = np.vstack(out)
    _VERT_CACHE[obj_id] = V
    return V


def quat_mat(q):
    R = np.zeros(9)
    mj.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def pinch_targets_from(V, pos, quat):
    """Antipodal pair on the object's minor principal axis, in world coords.

    Returns (p_index, p_thumb, axis_world). Support points are used rather than a
    ray cast: they are guaranteed to lie on the hull, needing no intersection test.
    """
    c = V.mean(0)
    # minor principal axis = thinnest direction = natural pinch axis
    _, _, Vt = np.linalg.svd(V - c, full_matrices=False)
    u = Vt[2]
    p_plus, p_minus = V[np.argmax(V @ u)], V[np.argmin(V @ u)]

    R = quat_mat(quat)
    uw = R @ u
    # Offset OUTWARD along the pinch axis. The fingertip site sits behind the pad
    # surface, so for the pad to meet the object the site must stop short of it.
    # Offsetting inward instead (the original sign) buried every target 2-5 mm
    # inside the object, which only became visible once the SDF could report it.
    a = np.asarray(pos) + R @ p_plus + uw * PAD_BACKOFF
    b = np.asarray(pos) + R @ p_minus - uw * PAD_BACKOFF
    return a, b, uw


def place_objects(obj_ids, ws, rng, floor_clear=0.06, x_min=0.30, r_max=0.72,
                  min_sep=0.18, tries=4000):
    """Sample a floating pose per object: reachable, clear of the floor, mutually apart.

    The workspace cloud alone is not enough. It is the *palm* envelope, so it
    admits poses at the floor and beside the base where a two-finger pinch has
    nowhere to go -- the first run put a bottle at z=0.041 with every binding
    constraint against the floor, and the solve failed at 44 mm. Rejecting on the
    object's own lowest vertex is what makes "floating" true rather than nominal.
    """
    placed = []
    for oid in obj_ids:
        V = object_hull_verts(oid)
        for _ in range(tries):
            p = W.sample_positions(ws, 1, rng, margin=0.01)[0]
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)

            if p[0] < x_min or np.linalg.norm(p[:2]) > r_max:
                continue
            if p[2] + (V @ quat_mat(q).T)[:, 2].min() < floor_clear:
                continue
            if any(np.linalg.norm(p - pp) < min_sep for _, pp, _ in placed):
                continue
            placed.append((oid, p, q))
            break
        else:
            raise RuntimeError(f"no valid placement for {oid} in {tries} tries")
    return placed


def solve_one(model, data, obj_body, tgt, q_bias,
              use_obj_collision=True, backend="ipopt", collision="sdf"):
    site_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
                for f in FINGER_SET]
    rgeoms = robot_geom_names(model)
    obj_geoms = ["floor"]
    sdf_bodies = ()
    if use_obj_collision:
        if collision == "sdf":
            sdf_bodies = (obj_body,)
        else:
            obj_geoms += S.hull_geoms(model, obj_body)

    # Unconstrained DLS first, as the live path does. Without it IPOPT starts from
    # the home pose, has to travel the whole way to the object, and stalls: measured
    # 12 mm residual at 300 iterations and *worse* at 800 (42 mm) and 2000 (45 mm) --
    # wandering, not a budget problem.
    t0 = time.time()
    dls = SpatialIKSolver(n_robot=N_ROBOT)
    ik_data = mj.MjData(model)
    ik_data.qpos[:] = data.qpos
    q_warm = dls.solve(model, ik_data, site_ids, tgt, q_bias=q_bias, null_gain=0.3)
    dls_ms = (time.time() - t0) * 1e3

    solver = ConstrainedIKSolver(
        model, N_ROBOT, rgeoms, obj_geoms, sdf_bodies=sdf_bodies,
        clearance=0.005, posture_weight=np.r_[np.full(7, 1e-5), np.full(16, 1e-4)],
        tip_weight=100.0, orient_weight=0.0, max_iter=300,
    )
    if backend == "sqp":
        configure_sqp(solver)

    t0 = time.time()
    q = solver.solve(data, site_ids, tgt, q_bias=q_bias, q_init=q_warm,
                     reduced_clearance_geoms=clearance_by_geom(rgeoms))
    wall = (time.time() - t0) * 1e3
    m = dict(solver.last_metrics)
    m["wall_ms"] = wall
    m["dls_ms"] = dls_ms
    m["n_obj_geoms"] = len(obj_geoms)
    m["collision"] = "none" if not use_obj_collision else collision
    return q, m


def render(model, data, path, w=1200, h=900, groups=(0, 1, 2, 5),
           lookat=(0.45, 0.0, 0.3), dist=1.5, azim=135, elev=-20):
    """Render to PNG. Group 4 (workspace shell) is off unless asked for -- 10k
    voxel boxes drawn over the scene hide everything behind them."""
    opt = mj.MjvOption()
    mj.mjv_defaultOption(opt)
    for g in range(6):
        opt.geomgroup[g] = 1 if g in groups else 0

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, azim, elev

    r = mj.Renderer(model, h, w)
    try:
        r.update_scene(data, camera=cam, scene_option=opt)
        px = r.render()
        from PIL import Image
        Image.fromarray(px).save(path)
    finally:
        del r
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*",
                    default=["006_mustard_bottle", "025_mug", "011_banana"])
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--backend", choices=["ipopt", "sqp"], default="ipopt")
    ap.add_argument("--no-obj-collision", action="store_true")
    ap.add_argument("--collision", choices=["sdf", "hulls"], default="sdf",
                    help="sdf: one precomputed distance table per object. "
                         "hulls: per-hull bounding spheres (the old fallback).")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "ycb_grasp" / "out"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    base = mj.MjModel.from_xml_path(str(REPO / "models" / "scene_kinova_leap.xml"))
    ws = W.load_or_build(base, n=200_000, seed=0)
    print(f"workspace: {len(ws['centers'])} voxels @ {ws['voxel']} m "
          f"({ws['n_kept']}/{ws['n_samples']} samples kept)")

    placements = place_objects(args.objects, ws, rng)
    q_bias = home_bias()

    # Coarser shell for drawing: the solver's 3 cm cloud renders as a near-solid
    # mass of boxes that hides everything behind it.
    shell = W.shell_mesh(W.load_or_build(base, n=200_000, seed=0, voxel=0.07))

    rows, last = [], None
    for oid, pos, quat in placements:
        # One object per scene: each test is its own trial, so a render shows only
        # the object under test and nothing else can enter the constraint set.
        V = object_hull_verts(oid)
        a, b, _ = pinch_targets_from(V, pos, quat)
        targets = {"index": a, "thumb": b}
        tgt = [targets[f] for f in FINGER_SET]
        markers = [("tgt_index", a, (0.15, 0.9, 0.35, 0.95)),
                   ("tgt_thumb", b, (0.15, 0.75, 1.0, 0.95))]

        model, data, info = S.build([(oid, pos, quat)], workspace=shell,
                                    target_sites=markers)
        bn = next(iter(info))
        data.qpos[:N_ROBOT] = q_bias
        mj.mj_forward(model, data)

        print(f"\n{'=' * 72}\n{oid}   {model.body_mass[info[bn]['bid']]:.3f} kg   "
              f"{len(S.hull_geoms(model, bn))} hulls   pos={np.round(pos, 3)}   "
              f"span={np.linalg.norm(a - b) * 1e3:.0f} mm")

        q, met = solve_one(model, data, bn, tgt, q_bias,
                           use_obj_collision=not args.no_obj_collision,
                           backend=args.backend, collision=args.collision)
        data.qpos[:N_ROBOT] = q
        mj.mj_forward(model, data)
        met["site_err_check_mm"] = [
            round(float(np.linalg.norm(data.site_xpos[mj.mj_name2id(
                model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])] - t) * 1e3), 2)
            for f, t in zip(FINGER_SET, tgt)]
        rows.append((oid, met))

        png = render(model, data, out / f"solve_{oid}.png",
                     lookat=pos, dist=0.75)
        over = render(model, data, out / f"scene_{oid}.png",
                      groups=(0, 1, 2, 4, 5), dist=2.4, azim=140, elev=-22)
        print(f"  {met.get('status')}  solve={met.get('t_solve_ms', float('nan')):.0f} ms  "
              f"iters={met.get('iters')}  tip err (mm)={met['site_err_check_mm']}  "
              f"min slack={met.get('min_slack_mm')}  pruned={met.get('n_pruned')}")
        print(f"  -> {png.name}, {over.name}")
        last = (model, data)

    print("\n" + "=" * 72)
    print(f"{'object':24s} {'hulls':>6s} {'ms':>7s} {'iters':>6s} {'max tip err mm':>15s}")
    for oid, m in rows:
        print(f"{oid:24s} {m['n_obj_geoms'] - 1:6d} "
              f"{m.get('t_solve_ms', float('nan')):7.0f} {str(m.get('iters')):>6s} "
              f"{max(m['site_err_check_mm']):15.2f}")

    if args.view and last is not None:
        import mujoco.viewer
        model, data = last
        with mujoco.viewer.launch_passive(model, data) as v:
            v.opt.geomgroup[3] = 0
            v.opt.geomgroup[4] = 1
            v.sync()
            while v.is_running():
                mj.mj_forward(model, data)
                v.sync()


if __name__ == "__main__":
    main()

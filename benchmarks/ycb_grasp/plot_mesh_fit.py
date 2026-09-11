"""Visualize the mesh-vertex quadratic/plane fit at a contact seed.

Shows, per fit radius: which mesh vertices were selected, the fitted plane, the
fitted paraboloid, and the residuals of each -- so the choice between "this face
is flat" and "this face is curved" can be read off the picture rather than
inferred from two kappa numbers.

Also answers directly whether a larger sample radius smooths the surface
variance out: each panel reports the plane's own RMS residual and the fitted
curvature, so a radius sweep shows whether kappa is converging to a stable value
(real curvature) or shrinking as ~1/r (noise being fitted).
"""
import sys
from pathlib import Path

try:
    import out_paths as OP
except ImportError:
    from ycb_grasp import out_paths as OP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = Path("/home/aipexws5/daniel/hand-control")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "benchmarks"))
import mujoco as mj                                              # noqa: E402
from ycb_grasp import pick_from_floor as P, scene as S, workspace as W   # noqa: E402
from grasp_control import object_uv_atlas as oua                 # noqa: E402
from simulation.grasp_planner_3d import (_geom_normal_np, _project_to_surface_np,
    _sdf_hessian_np, _tangent_basis_np, _principal_curvature_axes_np,
    MultiStartGraspPlanner3D)                                    # noqa: E402
from simulation.grasp_config_builder import for_ablation_default # noqa: E402
from ycb_grasp.ik_demo import clearance_by_geom, robot_geom_names # noqa: E402


def fit_at(Vv, seed_l, t1_l, t2_l, n_l, radius, w_band=0.004):
    """Select verts within `radius` of the seed in the tangent plane (and within
    w_band along the normal, so points wrapping onto an adjacent face are not
    mixed into a local fit), then fit BOTH a plane and a full quadratic."""
    d = Vv - seed_l
    u, v, w = d @ t1_l, d @ t2_l, d @ n_l
    m = ((u**2 + v**2) <= radius**2) & (np.abs(w) <= w_band)
    if m.sum() < 8:
        return None
    U, Vq, W = u[m], v[m], w[m]
    # plane: w = du + ev + f
    Ap = np.stack([U, Vq, np.ones(m.sum())], axis=1)
    cp, *_ = np.linalg.lstsq(Ap, W, rcond=None)
    rp = W - Ap @ cp
    # quadratic: w = a u^2 + b uv + c v^2 + d u + e v + f
    Aq = np.stack([U**2, U*Vq, Vq**2, U, Vq, np.ones(m.sum())], axis=1)
    cq, *_ = np.linalg.lstsq(Aq, W, rcond=None)
    rq = W - Aq @ cq
    Hq = np.array([[2*cq[0], cq[1]], [cq[1], 2*cq[2]]])
    ev = np.linalg.eigvalsh(Hq)
    return dict(mask=m, U=U, V=Vq, W=W, cp=cp, cq=cq, kappa=ev,
                rms_plane=float(np.sqrt((rp**2).mean())),
                rms_quad=float(np.sqrt((rq**2).mean())),
                max_plane=float(np.abs(rp).max()), n=int(m.sum()))


def main():
    obj = sys.argv[1] if len(sys.argv) > 1 else "036_wood_block"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    radii = [0.015, 0.025, 0.04, 0.06]

    rng = np.random.default_rng(4)
    base = mj.MjModel.from_xml_path(str(REPO/"models"/"scene_kinova_leap.xml"))
    ws = W.load_or_build(base, n=200_000, seed=0)
    pos, quat = P.place_object_on_floor(obj, ws, rng)
    model, data, info = S.build([(obj, pos, quat)])
    bn = next(iter(info)); bid = info[bn]["bid"]
    mj.mj_forward(model, data); pos, quat = P.settle_object_on_floor(model, data, bid)
    rg = robot_geom_names(model)
    cfg = for_ablation_default(obj_geom=S.hull_geoms(model, bn)[0], obj_body=bn, n_seeds=1,
                               arm_geom_names=rg, obj_clearance_by_geom=clearance_by_geom(rg))
    pl = MultiStartGraspPlanner3D(model, data, cfg); p = pl._planner
    gid = p._obj_gid; gt = int(model.geom_type[gid]); gs = model.geom_size[gid]
    c = data.geom_xpos[gid].copy(); R = data.geom_xmat[gid].reshape(3,3).copy()
    me = p._mesh_entry
    Vv, _ = oua.body_visual_mesh(model, bid)
    Vc = S.hull_vertices(model, bn); ext = Vc.max(0) - Vc.min(0)

    sl = Vc.mean(0).copy(); sl[2] = Vc[:,2].min() + frac*ext[2]
    ps = _project_to_surface_np(c + R@(sl + np.array([1.,0,0])*ext[0]*2.0),
                                gt, c, R, gs, mesh_entry=me)
    n_out = _geom_normal_np(ps, gt, c, R, gs, mesh_entry=me)
    seed_l = R.T @ (ps - c)
    n_l = np.asarray(me["grad_fn"](seed_l), float).reshape(3); n_l /= np.linalg.norm(n_l)
    t1, t2 = _tangent_basis_np(n_out); t1_l = R.T@t1; t2_l = R.T@t2
    H = _sdf_hessian_np(me, seed_l); T = np.stack([t1_l, t2_l], axis=1)
    _,_,k0s,k1s = _principal_curvature_axes_np(T.T@H@T, t1_l, t2_l)

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(f"{obj}: mesh-vertex fit at seed z={ps[2]:.3f} "
                 f"(SDF-Hessian says kappa=({k0s:+.3f}, {k1s:+.3f}))\n"
                 "selected verts (dots, colored by height above the fitted plane), "
                 "fitted PLANE (grey) and QUADRATIC (blue mesh)", fontsize=11)

    for i, radius in enumerate(radii):
        f = fit_at(Vv, seed_l, t1_l, t2_l, n_l, radius)
        ax = fig.add_subplot(2, len(radii), i+1, projection="3d")
        if f is None:
            ax.set_title(f"r={radius*1000:.0f}mm: too few pts"); continue
        U, Vq, Wv = f["U"]*1000, f["V"]*1000, f["W"]*1000
        resid = (f["W"] - np.stack([f["U"], f["V"], np.ones(f["n"])],1) @ f["cp"])*1000
        sc = ax.scatter(U, Vq, Wv, c=resid, cmap="coolwarm", s=6,
                        vmin=-2.5, vmax=2.5, depthshade=False)
        g = np.linspace(-radius, radius, 12)*1000
        GU, GV = np.meshgrid(g, g)
        cp = f["cp"]; cq = f["cq"]
        PL = (cp[0]*GU/1000 + cp[1]*GV/1000 + cp[2])*1000
        ax.plot_surface(GU, GV, PL, color="0.6", alpha=0.30, linewidth=0)
        QD = (cq[0]*(GU/1000)**2 + cq[1]*(GU/1000)*(GV/1000) + cq[2]*(GV/1000)**2
              + cq[3]*GU/1000 + cq[4]*GV/1000 + cq[5])*1000
        ax.plot_wireframe(GU, GV, QD, color="tab:blue", alpha=0.55,
                          rstride=2, cstride=2, linewidth=0.7)
        ax.set_title(f"r={radius*1000:.0f}mm  n={f['n']}\n"
                     f"kappa=({f['kappa'][0]:+.2f}, {f['kappa'][1]:+.2f})",
                     fontsize=9)
        ax.set_xlabel("u (mm)", fontsize=7); ax.set_ylabel("v (mm)", fontsize=7)
        ax.set_zlabel("h (mm)", fontsize=7); ax.tick_params(labelsize=6)
        ax.set_zlim(-4, 4)

        # residual panel below
        ax2 = fig.add_subplot(2, len(radii), len(radii)+i+1)
        ax2.axhline(0, color="k", lw=0.6)
        rad = np.sqrt(f["U"]**2 + f["V"]**2)*1000
        ax2.scatter(rad, resid, s=6, c=resid, cmap="coolwarm", vmin=-2.5, vmax=2.5)
        ax2.set_title(f"plane RMS={f['rms_plane']*1000:.2f}mm   "
                      f"quad RMS={f['rms_quad']*1000:.2f}mm   "
                      f"improvement={100*(1-f['rms_quad']/max(f['rms_plane'],1e-12)):.0f}%",
                      fontsize=8)
        ax2.set_xlabel("distance from seed (mm)", fontsize=7)
        ax2.set_ylabel("height above plane (mm)", fontsize=7)
        ax2.tick_params(labelsize=6); ax2.set_ylim(-4, 4); ax2.grid(alpha=0.3)

    out = OP.fig_path(Path(f"meshfit_{obj}_{frac:.2f}.png"))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OP.savefig(fig, out, dpi=110)
    print(f"-> {out.resolve()}")
    for radius in radii:
        f = fit_at(Vv, seed_l, t1_l, t2_l, n_l, radius)
        if f:
            print(f"  r={radius*1000:4.0f}mm n={f['n']:5d} kappa=({f['kappa'][0]:+8.3f},{f['kappa'][1]:+8.3f}) "
                  f"planeRMS={f['rms_plane']*1000:.3f}mm quadRMS={f['rms_quad']*1000:.3f}mm")


if __name__ == "__main__":
    main()

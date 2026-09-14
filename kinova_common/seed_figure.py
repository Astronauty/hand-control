"""One figure showing ALL contact seeds a solve considered — accepted and rejected.

The teleop counterpart of benchmarks/ycb_grasp/plot_seed_quadratic.py, and
deliberately drawn in the SAME grammar (its draw_mesh / draw_seed_rays /
draw_quadratic / quad_frame helpers are imported, not reimplemented), so a teleop
figure and a seed_quadratic figure can be read side by side.

The difference is WHERE the seeds come from. plot_seed_quadratic generates its own
seeds in its own process with its own RNG, so its picture is representative of the
seeding DISTRIBUTION but is not the seed set any particular solve used. This module
draws planner.last_seed_accept_table / last_seed_reject_table -- recorded by
MultiStartGraspPlanner3D.solve() as it gates -- so the figure is PAIRED with the
solve that produced the committed grasp: same seeds, same gates, same RNG draw.

Rejected seeds are drawn too (red, with the gate's own reason). solve() stops at
the first n_seeds ACCEPTED, so the rejects recorded are exactly those the gates
threw away on the way to this grasp -- and on an object where a gate refuses
everything, they are the whole story.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from ycb_grasp import out_paths as OP                       # noqa: E402
from ycb_grasp import plot_seed_quadratic as SQ             # noqa: E402


def _scene_from_planner(planner, model, data):
    """The `sc` dict plot_seed_quadratic's helpers expect, built from a LIVE planner
    instead of from its own build_object(). Mesh pose comes from the BODY (object_sdf
    is body-frame) exactly as verify()/solve() do -- see the frame note in
    grasp_planner_3d.verify()."""
    pl = planner._planner if hasattr(planner, "_planner") else planner
    gid = pl._obj_gid
    gtype = int(model.geom_type[gid])
    me = pl._mesh_entry
    if gtype == SQ._GEOM_TYPE_MESH:
        center = data.xpos[pl._obj_bid].copy()
        R = data.xmat[pl._obj_bid].reshape(3, 3).copy()
    else:
        center = data.geom_xpos[gid].copy()
        R = data.geom_xmat[gid].reshape(3, 3).copy()
    Vvis, Fvis = SQ.oua.body_visual_mesh(model, pl._obj_bid)
    cfg = pl.cfg
    return dict(obj=cfg.obj_body, model=model, data=data, cfg=cfg, planner=pl,
                geom_type=gtype, geom_size=model.geom_size[gid].copy(),
                mesh_entry=me, center=center, R=R, Vvis=Vvis, Fvis=Fvis,
                ground_z=float(cfg.ground_z),
                r_tip=(float(cfg.seed_ground_clearance_m)
                       if cfg.seed_ground_clearance_m is not None
                       else float(getattr(pl, "_r_tip_min", 0.0194))))


def _as_rec(entry, ok):
    """One accept/reject table row -> the `rec` shape draw_seed_rays wants.

    p3s/n3_in ride along when the row has them (a tripod seed that cleared the
    third-contact fan). Absent at n=2, and the caller then draws two contacts --
    never a fabricated third."""
    seed = dict(p1s=np.asarray(entry["p1s"], float),
                p2s=np.asarray(entry["p2s"], float),
                n1_in=np.asarray(entry["n1_in"], float),
                n2_in=np.asarray(entry["n2_in"], float))
    seed["p1"], seed["p2"] = seed["p1s"], seed["p2s"]
    if entry.get("p3s") is not None:
        seed["p3s"] = np.asarray(entry["p3s"], float)
        seed["n3_in"] = np.asarray(entry["n3_in"], float)
        seed["p3"] = seed["p3s"]
    return dict(seed=seed, kind=entry.get("kind", "random"), ok=ok,
                why=entry.get("why", "accepted"),
                ray=dict(origin=None, u=None))


def _contacts_of(rec):
    """[(role_key, point, inward_normal), ...] for whatever contacts this seed has.

    'index' is the SLOT-2 key plot_seed_quadratic colours. The NLP makes the middle
    finger share slot 2's patch (grasp_planner_3d.py _run_stage reconstructs p3 from
    _t2_frame), so no second patch is drawn -- but the SEED does not respect that
    sharing, and the panel says so rather than hiding it. See the in/out-of-bounds
    annotation in _draw_block: measured 47-59mm index-to-middle seed separation
    against a ~19mm patch half-extent on 017_orange, i.e. every third seed starts
    2-3x OUTSIDE the region it is later confined to."""
    s_ = rec["seed"]
    out = [("thumb", s_["p1s"], s_["n1_in"]), ("index", s_["p2s"], s_["n2_in"])]
    if s_.get("p3s") is not None:
        out.append(("middle", s_["p3s"], s_["n3_in"]))
    return out


_MIDDLE_COLOR = "#238b45"   # green -- distinct from thumb orange / index blue

# One colour per finger, used for BOTH ends of that contact's story: the o at
# the seed and the ^ at the NLP's converged position. Rejected seeds keep the
# finger colours too -- the panel header already says the seed was rejected, so
# recolouring everything red only destroys the finger encoding.
_FINGER_COLORS = dict(SQ.FINGER_COLORS, middle=_MIDDLE_COLOR)


def _FCOL(key):
    return _FINGER_COLORS.get(key, "0.3")


def _overlay_solved(ax, solved, normals=None):
    """Draw the SOLVED contacts over a seed panel as a triangle per finger.

    Same colour as that finger's seed marker: the pair (o -> ^) is the NLP's
    travel for that contact, and colour is what ties the two ends together.
    No connecting axis is drawn -- the only line on this panel is the dashed
    chord the antipodal seed scan ran along.
    """
    p1 = solved.get("p1")
    p2 = solved.get("p2")
    if p1 is None or p2 is None:
        return
    for key in ("thumb", "index", "middle"):
        p = solved.get({"thumb": "p1", "index": "p2", "middle": "p3"}[key])
        if p is None:
            continue
        # Lifted off the surface for the same reason the seed o is -- the NLP
        # converges ONTO the patch, so the solved contact is co-planar with it.
        # The normal is taken from this finger's SEED: the solved point has no
        # recorded normal here, and over a patch-sized move the surface normal
        # turns little enough that it serves only to pick a side.
        _n = (normals or {}).get(key)
        _q = _lift(p, _n) if _n is not None else np.asarray(p, float).reshape(3)
        _mark(ax, _q, "^", _FCOL(key), 8.0, 21)


def _mark(ax, p, marker, color, size, z):
    """One contact marker, drawn with ax.plot rather than ax.scatter.

    This is not cosmetic. A scatter on a 3D axes is a Path3DCollection, which
    matplotlib folds into its depth sort REGARDLESS of zorder, so a marker
    lying on a patch is composited behind that patch and disappears -- measured
    against an alpha=0.92 surface, the marker is invisible at any lift and at
    any zorder. A marker drawn through ax.plot is a Line3D, which does honour
    zorder, so it stays on top of the patch it belongs to. The patches keep
    their depth sort against EACH OTHER; only the markers opt out.
    """
    p = np.asarray(p, float).reshape(3)
    ax.plot([p[0]], [p[1]], [p[2]], marker=marker, ms=size, mfc=color,
            mec="k", mew=0.4, ls="none", zorder=z)


def _lift(p, n_in, mm=1.2):
    """A contact point pushed `mm` OUT of the surface (against the inward
    normal). Markers are drawn here rather than at the contact so they never
    land co-planar with an opaque patch -- see the note at the seed scatter."""
    p = np.asarray(p, float).reshape(3)
    n = np.asarray(n_in, float).reshape(3)
    d = float(np.linalg.norm(n))
    return p - (n / d) * (mm * 1e-3) if d > 1e-12 else p


def _draw_block(fig, gs, col0, recs, sc, header, ok, solved=None):
    """One column block (all accepted, or all rejected), one panel per seed.

    solved: optional dict(p1=..., p2=..., p3=...) of the contacts the NLP
        actually converged to. Overlaid on ACCEPTED panels only (a rejected
        seed never reached the NLP, so it has no solved counterpart). This is
        what makes the figure legible next to the grasp-contacts figure: the
        seed is only a STARTING POINT, and the NLP moves it -- measured 23mm of
        travel and 45.6 deg of grasp-axis rotation on 014_lemon seed 0 -- so a
        seed figure with no solved marker looks like it is showing a different
        grasp than the one that executed. It was; both were correct.
    """
    for j, rec in enumerate(recs):
        ax = fig.add_subplot(gs[0, col0 + j], projection="3d")
        # WHICH ACCEPTED SEED WON. solve() runs a full NLP per accepted seed and
        # keeps the cost-ranked best, so without this every accepted panel looks
        # equally chosen and the reader cannot tell which one produced the grasp
        # that actually executed. rec['winner'] is set by the caller from
        # res['seed_index'].
        is_winner = bool(ok and rec.get("winner"))
        _stage_frames = rec.get("stage_frames") if is_winner else None
        # ONE PANEL PER SEED. The patch row used to be a separate, patch-zoomed
        # axes below this one, on the assumption that a patch is small next to
        # its object and needs its own framing. Measured, it is not: patch
        # half-extent vs object half-extent is 0.96-1.02 on 056_tennis_ball and
        # 1.17-2.15 on 036_wood_block (the largest object in the set), so the
        # two rows were drawing nearly the same framing twice. Merging them puts
        # the seed, the solved contact and the patch each contact is confined to
        # in ONE picture, which is the comparison the figure is actually for.
        SQ.draw_mesh(ax, sc, alpha=0.16)

        pts = []
        for key, p, nin in _contacts_of(rec):
            if key == "middle":
                continue          # shares slot 2's patch; marked, not re-drawn
            # PREFER THE FRAME THE SOLVE ACTUALLY USED. Rebuilding it here with
            # quad_frame() re-runs _mesh_quadratic_contact_ca at THIS record's
            # seed, which is not the point the winning stage was built on --
            # measured 24mm apart in y on 036_wood_block seed 1 -- so the drawn
            # rectangle did not correspond to the solved t_var, and a solution
            # strictly inside its bounds (t1 = +86.87 against +86.88) appeared
            # to sit outside its own trust region. res['quad_frames'] carries
            # the real thing; the rebuild stays as the fallback for records
            # that have none (rejected seeds never reached a stage).
            frame = (_stage_frames or {}).get(key) if is_winner else None
            if frame is None:
                try:
                    frame = SQ.quad_frame(sc, np.asarray(p, float),
                                          np.asarray(nin, float), sc["cfg"])
                except Exception:
                    frame = None
            if frame is None:
                continue
            # OPAQUE + DEPTH SORTED, so a nearer patch actually hides the part
            # of the farther one behind it. Both were previously drawn at
            # alpha=0.55 with a hard zorder=8; an explicit zorder overrides
            # matplotlib's 3D depth sort entirely, so the two patches were
            # composited in CALL order regardless of which faced the camera,
            # and a pair on opposite sides of the object read as
            # interpenetrating rather than front-and-back.
            # alpha just under 1: opaque enough that the nearer patch clearly
            # occludes the farther one, sheer enough that a contact marker
            # lying ON the surface still reads through it. At alpha=1.0 the
            # seed markers vanished entirely -- they sit on the patch, so an
            # opaque patch swallows them, which costs the panel its subject.
            SQ.draw_quadratic(ax, sc, frame, key, alpha=0.92, depth_sort=True)
            pts.append(SQ.patch_points(frame, sc["center"], sc["R"],
                                       (frame["t_lo_0"], frame["t_hi_0"]),
                                       (frame["t_lo_1"], frame["t_hi_1"]),
                                       n=7).reshape(-1, 3))

        # The chord the antipodal scan ran along -- the only line on the panel.
        ax.plot(*np.array([rec["seed"]["p1s"], rec["seed"]["p2s"]]).T,
                color="0.45", lw=1.1, ls="--", zorder=9)
        # SEED positions: one o per finger, in that finger's colour, nudged
        # OUTWARD off the surface. A contact point is by construction ON the
        # patch, so drawn at its true position it is exactly co-planar with an
        # opaque surface and the depth sort may place it either side; a ~1mm
        # lift along the outward normal makes it unambiguously in front
        # without moving it anywhere the reader can measure.
        for key, p, nin in _contacts_of(rec):
            _mark(ax, _lift(p, nin), "o", _FCOL(key), 7.0, 20)
            pts.append(np.asarray(p, float)[None, :])
        # SOLVED positions: same colour, as a triangle. ONLY on the winning
        # panel -- res['p1']/['p2'] are the contacts of the seed that WON, so
        # drawing them on a losing accepted panel would show that panel's seed
        # next to a different seed's solution and invite exactly the
        # "solution is off its patch" misreading this figure exists to settle.
        if is_winner and solved is not None:
            _overlay_solved(ax, solved,
                            normals={k: n for k, _p, n in _contacts_of(rec)})

        _title = ("accepted -- SELECTED" if is_winner
                  else "accepted" if ok else rec["why"])
        ax.set_title(_title, fontsize=8.5,
                     color=("#08519c" if is_winner else "0.15") if ok else "#cb181d",
                     fontweight=("bold" if is_winner else "normal"), pad=1.0)
        # WORLD frame, framed on the whole object: draw_mesh draws the full
        # shell in world, and with the patches this large there is nothing to
        # gain by cropping to them.
        SQ._equal_axes(ax, sc["center"] + sc["Vvis"] @ sc["R"].T)
        ax.set_axis_off()


def write_seed_figure(planner, model, data, out_path, title_extra="", res=None):
    """Draw every seed this solve considered. Returns the written path, or None
    when the planner recorded no seeds (e.g. a solve that never reached seeding).

    res: the solve result dict. When given, the contacts the NLP CONVERGED TO
        (res['p1']/['p2']/['p3']) are overlaid on the accepted panels as a
        triangle in each finger's colour, against that finger's o at its seed.

        This exists because the seed figure and the grasp-contacts figure
        legitimately disagree: a seed is a STARTING POINT and the NLP moves it
        (measured 23.5/22.7mm of travel and 45.6 deg of grasp-axis rotation on
        014_lemon seed 0, seed width 53.7mm -> solved 62.9mm). Without the
        overlay the two figures look like they describe different grasps, and
        the only way to tell they don't is to diff the coordinates by hand.
    """
    pl = planner._planner if hasattr(planner, "_planner") else planner
    acc = list(getattr(planner, "last_seed_accept_table", None)
               or getattr(pl, "last_seed_accept_table", None) or [])
    rej = list(getattr(planner, "last_seed_reject_table", None)
               or getattr(pl, "last_seed_reject_table", None) or [])
    if not acc and not rej:
        return None
    sc = _scene_from_planner(planner, model, data)
    recs = [_as_rec(e, True) for e in acc] + [_as_rec(e, False) for e in rej]
    # Solved contacts, when the caller handed us the result. Only p1/p2 are
    # required; p3 rides along at n_contacts>=3.
    _solved = None
    if res is not None and res.get("p1") is not None and res.get("p2") is not None:
        _solved = {"p1": res["p1"], "p2": res["p2"]}
        if res.get("p3") is not None:
            _solved["p3"] = res["p3"]
    # WINNER + ITS REAL PATCH FRAMES. solve() runs one NLP per accepted seed and
    # keeps the cost-ranked best; res['seed_index'] says which accepted seed
    # that was, and res['quad_frames'] carries the paraboloid frames that stage
    # actually built. Tagging the record lets the panel both mark itself as the
    # selected one and draw the patch the contact was genuinely confined to
    # rather than a plot-time reconstruction at a different seed point.
    _win = res.get("seed_index") if res is not None else None
    if _win is not None and 0 <= int(_win) < len(acc):
        recs[int(_win)]["winner"] = True
        recs[int(_win)]["stage_frames"] = (res.get("quad_frames") or {})

    n = len(recs)
    n_acc = len(acc)
    # ACCEPTED LEFT, REJECTED RIGHT, with a narrow spacer column between them, so
    # the two populations read as two groups instead of one undifferentiated run.
    _SPACER = 0.22
    _acc_n = len([r for r in recs if r["ok"]])
    _rej_all = [r for r in recs if not r["ok"]]
    _REJ_CAP = 4
    _rej_n = min(len(_rej_all), _REJ_CAP)
    n_acc, n_rej = _acc_n, _rej_n
    widths = ([1.0] * n_acc) + ([_SPACER] if (n_acc and n_rej) else []) + ([1.0] * n_rej)
    n_cols = len(widths)
    fig = plt.figure(figsize=(max(3.3 * (n_acc + n_rej) + 1.0, 10.0), 4.2))
    gs = fig.add_gridspec(1, n_cols, width_ratios=widths)
    # COMPACT TITLE: object, and how the seeds split. The gate thresholds
    # (kappa, DLS pool) are config, not a property of this picture -- they
    # belong in the run's log, and on the figure they only compete with it.
    fig.suptitle(f"{sc['obj']}{title_extra}  —  "
                 f"{n_acc} accepted, {len(rej)} rejected", fontsize=11)

    acc_recs = [r for r in recs if r["ok"]]
    rej_recs = [r for r in recs if not r["ok"]]
    # CAP the rejected columns. 009_gelatin_box rejects 20 seeds (its faces are
    # mostly within seed_ground_clearance_m of the table), and drawing all of them
    # made the figure 8843px wide with ~2mm panels -- the ACCEPTED seeds, which are
    # the subject, became unreadable. Rejects are a diagnostic tail: a
    # representative handful plus a count carries the same information.
    _REJ_CAP = 4
    _n_rej_total = len(rej_recs)
    if _n_rej_total > _REJ_CAP:
        # Spread the sample across the run rather than taking the first few, so a
        # gate that only fires late is still represented.
        _idx = np.linspace(0, _n_rej_total - 1, _REJ_CAP).astype(int)
        rej_recs = [rej_recs[i] for i in _idx]
    _draw_block(fig, gs, 0, acc_recs, sc, "accepted", True, solved=_solved)
    _draw_block(fig, gs, n_acc + (1 if (n_acc and n_rej) else 0),
                rej_recs, sc, "rejected", False)

    # ONE SHARED LEGEND. Finger role is carried by colour in every panel, and
    # marker shape separates the two ends of a contact's story (seed vs the
    # NLP's converged position), so a single key replaces every per-panel tag.
    _has_mid = (bool(_solved) and _solved.get("p3") is not None) or any(
        r["seed"].get("p3s") is not None for r in recs)
    _keys = ["thumb", "index"] + (["middle"] if _has_mid else [])
    _h = [plt.Line2D([], [], color=_FCOL(k), marker="o", ls="none", ms=7,
                     mec="k", mew=0.4, label=k) for k in _keys]
    _h += [plt.Line2D([], [], color="0.35", marker="o", ls="none", ms=7,
                      mec="k", mew=0.4, label="seed"),
           plt.Line2D([], [], color="0.35", marker="^", ls="none", ms=7,
                      mec="k", mew=0.4, label="NLP solution")]
    fig.legend(handles=_h, loc="lower center", ncol=len(_h), frameon=False,
               fontsize=9, handletextpad=0.35, columnspacing=1.4,
               bbox_to_anchor=(0.5, 0.005))

    # Tight packing. The default 3D-axes bbox leaves most of a panel empty
    # around the rendered object, so the margins are pulled in hard.
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.10,
                        wspace=0.02)
    out = OP.savefig(fig, Path(out_path), dpi=115)
    plt.close(fig)
    return out

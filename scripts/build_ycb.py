"""Convert the YCB object set into MJCF with quality-driven collision proxies.

CoACD's hull count is an *output*, not a setting. Each object is sealed into a
watertight solid (mesh_seal.py), then decomposed at progressively finer concavity
thresholds until the union of hulls stops claiming more than --max-false-solid of
the object's convex hull as solid when it is actually empty. Fidelity is what is
held fixed across the set; the hull count lands wherever the shape requires it.

The knob this replaces was `max_convex_hull=8`, a fixed count applied to all 84
objects. That fixes cost instead of fidelity, and the two are not related across
shapes -- 8 hulls is more than a banana needs and nowhere near enough for a cup,
which is why the near-convex objects looked right and the cavities filled in.

Bodies also carry an explicit <inertial> built from the published YCB mass and the
sealed solid's tensor (ycb_masses.py). Without it MuJoCo infers mass from the
geoms, counting the cosmetic visual mesh and every overlapping hull -- 0.379 kg on
a 0.066 kg banana, and drifting whenever the hull count changed.

    python scripts/build_ycb.py                     # build everything not yet built
    python scripts/build_ycb.py --max-false-solid 0.02 --force
    python scripts/build_ycb.py --audit             # grade what is already on disk
"""
import argparse
import json
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import coacd
import trimesh

import ycb_masses
from mesh_fit import FitMetric
from mesh_seal import rasterise, seal, volume_error

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "assets/ycb_raw"
OUT = REPO / "assets/ycb_mjcf"
REPORT = OUT / "fit_report.json"

# Coarse -> fine. Hull count rises monotonically down this ladder (011_banana:
# 3, 4, 5, 8, ...; 025_mug: 26, 73, 144, 241, ...), so the first rung that meets
# the tolerance is also the cheapest one that does, and once a rung overruns the
# hull budget no finer rung can come back under it.
LADDER = (0.10, 0.075, 0.05, 0.035, 0.025, 0.015)

# Falsely-solid volume the proxy may carry, as a fraction of the object's convex
# hull. Normalising by the hull rather than by the object is what makes one number
# mean the same thing on a mug as on a banana -- see mesh_seal.VolumeError.
#
# 0.05 is set just above the floor CoACD imposes, not plucked. CoACD decomposes a
# voxel remesh of the scan at preprocess_resolution=50, i.e. voxels of
# extent/50 -- 4.3 mm on 003_cracker_box, 5.0 mm on 021_bleach_cleanser -- so its
# hulls bulge about one such voxel outside the true surface however many of them
# it makes. That floor is roughly scale-invariant once normalised by hull volume
# (the bulge goes as area x voxel, the hull as extent^3), and it lands at 0.03-0.05
# on the larger objects. A 0.02 target sat underneath it: objects with 1.3-3.1 mm
# of actual surface error churned the whole ladder and still reported as failures.
# Going below ~0.05 means raising CoACD's preprocess_resolution first.
DEFAULT_TOL_FRAC = 0.05

# Second limb of the gate: share of the convex hull's false solid that must be
# recovered, for objects whose concavity is too large to reach tol_frac at all.
MIN_RECOVERED = 0.85

# Hull ceiling per object, a contact-cost backstop rather than a quality setting --
# set high enough that the tolerance is what actually governs. Thin-walled objects
# are the expensive ones (025_mug reaches 3 mm at ~73 hulls, 065-a_cups at 36): a
# ceiling under that would cap exactly the objects a fixed count already failed,
# which is the failure this replaces. Objects that hit it are reported as
# budget_limited rather than silently degraded.
DEFAULT_MAX_HULLS = 128

# CoACD's max_ch_vertex is ignored unless decimate=True, so hulls arrive fully
# detailed (700-3700 faces each). Decimating is tempting -- MuJoCo pays per hull
# vertex in convex collision -- but it is not a free win, and which way it goes
# depends on the shape of the hulls, not the object:
#
#   025_mug   @0.10  25 hulls  697 faces  p99 4.87 mm  ->  64 verts: p99 4.74 mm
#   011_banana@0.05   4 hulls 3028 faces  p99 3.90 mm  ->  64 verts: p99 13.56 mm
#
# The mug decomposes into small boxy wedges that 64 vertices describe exactly; the
# banana into a few large smooth curved hulls that they do not. Off by default,
# opt in with --max-hull-verts. The gate makes that safe rather than silent: a
# decimated object that misses the tolerance descends the ladder and spends hulls
# to win the fidelity back.
DEFAULT_MAX_HULL_VERTS = 0      # 0 = no decimation


def extract_archives():
    """YCB downloads land as <name>_google_16k.tgz; unpack to <name>/google_16k/."""
    for tgz in sorted(RAW.rglob("*_google_16k.tgz")):
        name = tgz.name.removesuffix("_google_16k.tgz")
        if (RAW / name / "google_16k").is_dir():
            continue
        with tarfile.open(tgz) as tf:
            tf.extractall(RAW)
        print(f"extracted {name}")


def find_visual_obj(obj_dir):
    p = obj_dir / "google_16k" / "textured.obj"
    return p if p.exists() else None


def good_enough(error, tol_frac, min_recovered=MIN_RECOVERED):
    """Whether a decomposition's volume agreement is acceptable.

    Two limbs, because one threshold on false-solid volume does not mean the same
    thing on a convex object as on a hollow one:

    * `false_solid_frac <= tol_frac` -- the proxy claims almost nothing solid that
      is not. Near-convex objects clear this immediately, which is right: 013_apple
      scores 0.018 with a *single* hull, and no hull count improves on it, because
      what is left is CoACD's own remeshing bulge rather than concavity. Gating
      only on `recovered` would chase that noise forever.

    * `recovered >= min_recovered` -- of the concavity a plain convex hull gets
      wrong, this much has been won back. Hollow objects live here: 011_banana's
      hull is 33% false solid, and no achievable decomposition takes that to 2%,
      but recovering ~87% of it lands at 1.9 mm of surface error.

    Objects pass on whichever limb suits their shape, which is what makes the
    number comparable across the set.
    """
    return error.false_solid_frac <= tol_frac or error.recovered >= min_recovered


def decompose_to_tolerance(mesh, solid, hull_occupancy, tol_frac, max_hulls,
                           max_hull_verts=0, ladder=LADDER):
    """Walk the ladder until the fit is good enough; return (parts, error, threshold, status).

    Status is one of:
      tolerance_met    -- the cheapest decomposition that is good enough
      budget_limited   -- a finer rung would have overrun max_hulls; best kept
      ladder_exhausted -- the whole ladder ran without passing, hull count is fine;
                          the shape is simply not one CoACD resolves to this gate
      over_budget      -- even the coarsest rung overruns max_hulls; fewest kept

    The last two are worth telling apart: budget_limited says spend more hulls,
    ladder_exhausted says more hulls were on offer and did not help.
    """
    # Rung zero is the plain convex hull -- the cheapest proxy that exists, and the
    # right answer for balls and dice. Without it the ladder starts at a
    # decomposition and can never come back down to one hull.
    hull = mesh.convex_hull
    hull_parts = [(hull.vertices, hull.faces)]
    error = volume_error(solid, hull_parts, hull_occupancy)
    if good_enough(error, tol_frac):
        return hull_parts, error, None, "tolerance_met"
    attempts = [(None, hull_parts, error)]
    hit_budget = False

    for threshold in ladder:
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces),
                                threshold=threshold, max_convex_hull=-1,
                                decimate=max_hull_verts > 0,
                                max_ch_vertex=max_hull_verts or 256, seed=0)
        error = volume_error(solid, parts, hull_occupancy)
        attempts.append((threshold, parts, error))
        if error.n_hulls > max_hulls:
            hit_budget = True
            break
        if good_enough(error, tol_frac):
            return parts, error, threshold, "tolerance_met"

    within = [a for a in attempts if a[2].n_hulls <= max_hulls]
    if within:
        threshold, parts, error = min(within, key=lambda a: a[2].false_solid_frac)
        return parts, error, threshold, "budget_limited" if hit_budget else "ladder_exhausted"

    threshold, parts, error = min(attempts, key=lambda a: a[2].n_hulls)
    return parts, error, threshold, "over_budget"


def process(obj_dir, tol_frac, max_hulls, max_hull_verts=0, assembly_volumes=None):
    name = obj_dir.name                      # e.g. 006_mustard_bottle
    src = find_visual_obj(obj_dir)
    if src is None:
        print(f"skip {name}: no google_16k mesh")
        return None

    dst = OUT / name
    collision_dir = dst / "collision"
    collision_dir.mkdir(parents=True, exist_ok=True)

    # copy the visual assets MuJoCo will render
    for f in ["textured.obj", "textured.mtl", "texture_map.png"]:
        s = src.parent / f
        if s.exists():
            shutil.copy(s, dst / f)

    mesh = trimesh.load(src, force="mesh")
    started = time.time()
    solid = seal(mesh)
    hull = mesh.convex_hull
    hull_occupancy = rasterise([(hull.vertices, hull.faces)], solid)

    parts, error, threshold, status = decompose_to_tolerance(
        mesh, solid, hull_occupancy, tol_frac, max_hulls, max_hull_verts)
    elapsed = time.time() - started
    if not parts:
        print(f"skip {name}: CoACD returned no hulls")
        return None

    mass, mass_source = ycb_masses.resolve(name, solid.volume, assembly_volumes)
    com, tensor = solid.inertia(mass)

    # A rebuild can produce fewer hulls than the last run left behind; stale
    # part_*.obj would otherwise stay on disk and be silently wrong.
    for stale in collision_dir.glob("part_*.obj"):
        stale.unlink()

    collision_files = []
    for i, (v, f) in enumerate(parts):
        fn = f"collision/part_{i}.obj"
        trimesh.Trimesh(v, f).export(dst / fn)
        collision_files.append(fn)

    write_mjcf(dst, name, collision_files, mass, com, tensor)

    # The surface metric is a diagnostic, not the gate, so it is measured once on
    # the decomposition that was chosen rather than on every rung of the ladder.
    surface = FitMetric(mesh)(parts)
    record = dict(status=status, threshold=threshold, seconds=round(elapsed, 1),
                  tol_frac=tol_frac, max_hulls=max_hulls, max_hull_verts=max_hull_verts,
                  mass_kg=round(mass, 6), mass_source=mass_source,
                  overshoot_p99_mm=round(surface.p99_mm, 2), **error.as_dict())
    rung = "hull" if threshold is None else f"{threshold:.3f}"
    print(f"{name:26s} {status:16s} thresh={rung:<6s} hulls={error.n_hulls:<4d} "
          f"false={error.false_solid_frac:6.4f} recov={error.recovered:6.3f} "
          f"p99={surface.p99_mm:5.2f}mm mass={mass*1000:7.1f}g/{mass_source} {elapsed:5.1f}s",
          flush=True)
    return name, record


def audit(name):
    """Score the decomposition already on disk, without re-running CoACD."""
    src = find_visual_obj(RAW / name)
    parts_dir = OUT / name / "collision"
    if src is None or not parts_dir.is_dir():
        return None
    mesh = trimesh.load(src, force="mesh")
    parts = []
    for p in sorted(parts_dir.glob("part_*.obj")):
        m = trimesh.load(p, force="mesh")
        parts.append((m.vertices, m.faces))
    if not parts:
        return None

    solid = seal(mesh)
    hull = mesh.convex_hull
    error = volume_error(solid, parts, rasterise([(hull.vertices, hull.faces)], solid))
    surface = FitMetric(mesh)(parts)
    print(f"{name:26s} hulls={error.n_hulls:<4d} false={error.false_solid_frac:6.4f} "
          f"({error.false_solid_cm3:7.1f}cm3) recov={error.recovered:6.3f} "
          f"p99={surface.p99_mm:6.2f}mm", flush=True)
    return name, dict(status="audit", overshoot_p99_mm=round(surface.p99_mm, 2),
                      **error.as_dict())


def inertial_xml(mass, com, tensor):
    """<inertial> from the published mass and the sealed solid's tensor.

    fullinertia takes the matrix in body-frame axes about the centre of mass, in
    the order Ixx Iyy Izz Ixy Ixz Iyz; the compiler diagonalises it itself. With
    this present MuJoCo stops inferring inertia from the geoms, which is the point
    -- the visual mesh and the overlapping hulls were all contributing mass.
    """
    # 12 significant figures, not 6: the report stores mass to 6 decimals, and at
    # 6 sig figs a 2.4 kg body round-trips to 2.41695 against a recorded 2.416951,
    # which is a spurious mismatch for anything checking the two agree.
    p = " ".join(f"{v:.12g}" for v in com)
    i = " ".join(f"{v:.12g}" for v in (tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                       tensor[0, 1], tensor[0, 2], tensor[1, 2]))
    return f'      <inertial pos="{p}" mass="{mass:.12g}" fullinertia="{i}"/>'


def write_mjcf(dst, name, collision_files, mass, com, tensor):
    # relative paths so it works on any clone
    assets = [f'    <mesh name="{name}_visual" file="textured.obj"/>']
    for i, cf in enumerate(collision_files):
        assets.append(f'    <mesh name="{name}_col_{i}" file="{cf}"/>')

    # type="2d" is required: the default "cube" mapping garbles UV-mapped scans.
    # MuJoCo ignores textured.mtl, so the texture must be declared here and
    # reached via a material -- geoms have no texture attribute of their own.
    if (dst / "texture_map.png").exists():
        assets.append(f'    <texture name="{name}_tex" type="2d" '
                      f'file="texture_map.png"/>')
        assets.append(f'    <material name="{name}_mat" '
                      f'texture="{name}_tex"/>')
        visual_mat = f'material="{name}_mat" '
    else:
        visual_mat = ""

    geoms = [f'      <geom type="mesh" mesh="{name}_visual" '
             f'{visual_mat}class="visual"/>']
    for i in range(len(collision_files)):
        geoms.append(f'      <geom type="mesh" mesh="{name}_col_{i}" '
                     f'class="collision"/>')

    xml = f"""<mujoco model="{name}">
  <compiler meshdir="." texturedir="."/>
  <default>
    <default class="visual">
      <geom group="2" contype="0" conaffinity="0"/>
    </default>
    <default class="collision">
      <geom group="3"/>
    </default>
  </default>
  <asset>
{chr(10).join(assets)}
  </asset>
  <worldbody>
    <body name="{name}" pos="0 0 0">
{inertial_xml(mass, com, tensor)}
      <freejoint/>
{chr(10).join(geoms)}
    </body>
  </worldbody>
</mujoco>
"""
    (dst / f"{name}.xml").write_text(xml)


def rewrite(name, assembly_volumes=None):
    """Re-emit one MJCF from the hulls already on disk, no CoACD.

    Sealing is deterministic, so mass and inertia come back identical -- this is
    for when the *writer* changes (a new attribute, a formatting fix) and the
    decompositions are still good. Re-running the ladder for that would be an hour
    of CoACD to produce the same hulls.
    """
    src = find_visual_obj(RAW / name)
    dst = OUT / name
    files = sorted(p.name for p in (dst / "collision").glob("part_*.obj"))
    if src is None or not files:
        return None
    solid = seal(trimesh.load(src, force="mesh"))
    mass, source = ycb_masses.resolve(name, solid.volume, assembly_volumes)
    com, tensor = solid.inertia(mass)
    write_mjcf(dst, name, [f"collision/{f}" for f in files], mass, com, tensor)
    print(f"rewrote {name:26s} hulls={len(files):<4d} mass={mass*1000:7.1f}g/{source}",
          flush=True)
    return name, mass


def sealed_volume(name):
    """Sealed volume of one object, for splitting an assembly's published mass."""
    src = find_visual_obj(RAW / name)
    if src is None:
        return name, 0.0
    return name, seal(trimesh.load(src, force="mesh")).volume


def assembly_volumes(dirs, jobs):
    """{group prefix: {sibling name: sealed volume}} for the multi-part sets.

    A part's share of its assembly's published mass depends on its siblings, so
    these are measured up front in the parent -- the workers stay independent.
    Only the ~15 objects in ycb_masses.ASSEMBLY_TOTAL_G are touched.
    """
    groups = {}
    for prefix in {ycb_masses.group_prefix(d.name) for d in dirs} - {None}:
        groups[prefix] = sorted(
            p.name for p in RAW.iterdir()
            if p.is_dir() and ycb_masses.group_prefix(p.name) == prefix)
    if not groups:
        return {}

    names = [n for members in groups.values() for n in members]
    print(f"measuring {len(names)} assembly parts to split their published masses")
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        volumes = dict(pool.map(sealed_volume, names))
    return {prefix: {n: volumes[n] for n in members} for prefix, members in groups.items()}


def load_report():
    if REPORT.exists():
        return json.loads(REPORT.read_text())
    return {}


def save_report(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True))


def summarise(objects):
    if not objects:
        print("nothing to summarise")
        return
    key = "false_solid_frac"
    ranked = sorted(objects.items(), key=lambda kv: -kv[1][key])
    hulls = [r["n_hulls"] for _, r in ranked]
    print(f"\n{len(ranked)} objects | hulls total {sum(hulls)}, "
          f"median {sorted(hulls)[len(hulls) // 2]}, max {max(hulls)}")

    sources = {}
    for _, r in ranked:
        sources[r.get("mass_source", "n/a")] = sources.get(r.get("mass_source", "n/a"), 0) + 1
    print("mass: " + ", ".join(f"{n} {s}" for s, n in sorted(sources.items())))

    flagged = [(n, r) for n, r in ranked if r["status"] not in ("tolerance_met", "audit")]
    if flagged:
        print(f"\n{len(flagged)} did not reach the gate:")
        for n, r in flagged:
            print(f"  {n:26s} {r['status']:16s} hulls={r['n_hulls']:<4d} "
                  f"false={r[key]:.4f} recov={r['recovered']:.3f}")

    print("\nworst fits:")
    for n, r in ranked[:8]:
        print(f"  {n:26s} hulls={r['n_hulls']:<4d} false={r[key]:7.4f} "
              f"({r['false_solid_cm3']:7.1f}cm3)  p99={r.get('overshoot_p99_mm', float('nan')):6.2f}mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-false-solid", type=float, default=DEFAULT_TOL_FRAC,
                    help="allowed falsely-solid volume, as a fraction of the object's "
                         f"convex hull (default {DEFAULT_TOL_FRAC})")
    ap.add_argument("--max-hulls", type=int, default=DEFAULT_MAX_HULLS,
                    help="contact-cost ceiling per object (default 128)")
    ap.add_argument("--max-hull-verts", type=int, default=DEFAULT_MAX_HULL_VERTS,
                    help="cap vertices per hull for cheaper contact geometry; costs "
                         "fidelity on large smooth hulls, which the ladder then buys "
                         "back with more hulls (default 0 = off)")
    ap.add_argument("--jobs", type=int, default=6,
                    help="objects decomposed in parallel; CoACD is itself threaded "
                         "and will saturate ~28 cores per object, so more is not "
                         "always faster (default 6)")
    ap.add_argument("--only", nargs="+", metavar="OBJECT",
                    help="build just these objects, e.g. 024_bowl 025_mug")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the object already meets the current settings")
    ap.add_argument("--audit", action="store_true",
                    help="measure the decompositions already on disk; runs no CoACD")
    ap.add_argument("--rewrite", action="store_true",
                    help="re-emit the MJCFs from the hulls already on disk, for when "
                         "the writer changed but the decompositions are still good")
    args = ap.parse_args()

    coacd.set_log_level("error")

    if args.rewrite:
        names = args.only or sorted(p.name for p in OUT.iterdir() if p.is_dir())
        assembly = assembly_volumes([RAW / n for n in names], args.jobs)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            done = [r for r in pool.map(
                rewrite, names,
                [assembly.get(ycb_masses.group_prefix(n)) for n in names]) if r]
        print(f"\nrewrote {len(done)} MJCFs")
        return

    if args.audit:
        names = args.only or sorted(p.name for p in OUT.iterdir() if p.is_dir())
        # Scoring is pure numpy, so unlike the build this scales with the pool.
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = pool.map(audit, names)
        summarise(dict(r for r in results if r))
        return

    extract_archives()
    # "data" holds the downloaded .tgz files, not an object
    dirs = [d for d in sorted(RAW.iterdir()) if d.is_dir() and d.name != "data"]
    if args.only:
        wanted = set(args.only)
        dirs = [d for d in dirs if d.name in wanted]

    report = load_report()
    objects = report.setdefault("objects", {})

    if not args.force:
        # Settings live on each record, not on the file, so rebuilding one object
        # at a tighter tolerance does not invalidate the other 83.
        def current(name):
            record = objects.get(name)
            return (record is not None
                    and record.get("tol_frac") == args.max_false_solid
                    and record.get("max_hulls") == args.max_hulls
                    and record.get("max_hull_verts") == args.max_hull_verts
                    and (OUT / name / f"{name}.xml").exists())

        todo = [d for d in dirs if not current(d.name)]
        if len(dirs) - len(todo):
            print(f"{len(dirs) - len(todo)} objects already built at these settings "
                  f"(--force to redo)")
        dirs = todo

    if dirs:
        print(f"decomposing {len(dirs)} objects to false-solid <= {args.max_false_solid} "
              f"of the convex hull (<= {args.max_hulls} hulls), {args.jobs} at a time\n")
        assembly = assembly_volumes(dirs, args.jobs)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(process, d, args.max_false_solid, args.max_hulls,
                                   args.max_hull_verts, assembly.get(ycb_masses.group_prefix(d.name)))
                       for d in dirs]
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    name, record = result
                    objects[name] = record
                    save_report(report)      # checkpoint: a long run can be interrupted

    save_report(report)
    summarise(objects)
    print(f"\nreport: {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

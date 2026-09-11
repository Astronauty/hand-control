"""Single source of truth for where benchmark artifacts (renders, videos, CSVs) go.

Before this module every script invented its own layout directly under out/, and
sweeps made it worse: a --tag run wrote a SIBLING top-level directory
(out/pick_and_place_base, out/pick_and_place_imp20, ...) rather than nesting, so
a handful of parameter sweeps left 350MB spread across seven unrelated
top-level folders with no way to tell which environment produced what.

The layout is grouped by ENVIRONMENT, because that is the thing that actually
differs between runs -- a floor result and a tabletop result are not comparable
and should not sit in the same folder:

    out/
      floor/                    scene.py: objects floating over a bare floor
        <run tag>/<object>/...
      tabletop/                 table_scene.py: table + bin (pick-and-place)
        <run tag>/<object>/...
      analysis/                 environment-independent diagnostics
        quadratic_path/ mesh_fit/ uv_atlas/ ablate_ik.csv

`run tag` defaults to "default", so an untagged run is stable and overwrites
itself instead of accumulating. Sweeps pass --out-tag and nest underneath their
environment rather than beside it.

Scripts should call env_dir()/analysis_dir() rather than building paths from
REPO themselves, so the convention stays in one place.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Figure format for MATPLOTLIB figures (not MuJoCo camera renders, which are
# inherently raster and stay .png). Vector by default: these are diagnostic
# figures that get zoomed hard and land in a LaTeX paper, where a rasterized
# 3D panel goes soft and its text does not match the document font.
#
# Measured on the seed-quadratic figures: 017_orange 268KB pdf vs 396KB png,
# 036_wood_block 742KB pdf vs 741KB png (1.7s vs 1.2s to write). So the win is
# scalability and text quality, NOT file size -- a mesh-heavy panel emits one
# vector path per triangle and roughly breaks even.
#
# PFF_FIG_FORMAT=png restores raster output for quick previewing.
FIG_FORMAT = os.environ.get("PFF_FIG_FORMAT", "pdf").lstrip(".").lower()


def fig_path(path) -> Path:
    """Re-extension a figure path to the configured FIG_FORMAT.

    Call sites keep writing `.../name.png` literals -- readable, greppable, and
    matching the docstrings that name the artifact -- and this decides the
    actual container in one place.
    """
    return Path(path).with_suffix("." + FIG_FORMAT)


def savefig(fig, path, dpi=150, **kw):
    """fig.savefig() honouring FIG_FORMAT. dpi is ignored for vector formats
    (it only affects any rasterized sub-artist), kept so raster previews via
    PFF_FIG_FORMAT=png retain their intended resolution."""
    out = fig_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **({"dpi": dpi} if FIG_FORMAT in ("png", "jpg", "jpeg") else {}), **kw)
    return out
OUT_ROOT = REPO / "benchmarks" / "ycb_grasp" / "out"

FLOOR = "floor"
TABLETOP = "tabletop"
ANALYSIS = "analysis"

DEFAULT_TAG = "default"


def env_dir(env, tag=DEFAULT_TAG, object_id=None, create=True):
    """Directory for one environment's run artifacts.

    env       : FLOOR or TABLETOP.
    tag       : run tag (a sweep's --out-tag). Nests UNDER the environment.
    object_id : optional per-object subdirectory.
    """
    if env not in (FLOOR, TABLETOP):
        raise ValueError(f"unknown env {env!r} (expected {FLOOR!r} or {TABLETOP!r})")
    p = OUT_ROOT / env / (tag or DEFAULT_TAG)
    if object_id:
        p = p / str(object_id)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def analysis_dir(name=None, create=True):
    """Directory for environment-independent analysis output (quadratic-path
    plots, mesh fits, UV atlases, ablation CSVs)."""
    p = OUT_ROOT / ANALYSIS
    if name:
        p = p / str(name)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def add_out_args(ap, env, default_tag=DEFAULT_TAG):
    """Add the standard --out / --out-tag pair to an ArgumentParser.

    --out overrides the environment root outright (for one-off runs); --out-tag
    nests a named run under it. Resolve with resolve_out().
    """
    ap.add_argument("--out", default=None,
                    help="override the output directory outright (default: "
                         f"out/{env}/<--out-tag>)")
    ap.add_argument("--out-tag", default=default_tag,
                    help="name this run; nests under the environment directory "
                         "so sweeps do not create sibling top-level folders")
    return ap


def resolve_out(args, env):
    """The output directory for a parsed argparse namespace."""
    if getattr(args, "out", None):
        p = Path(args.out)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return env_dir(env, getattr(args, "out_tag", DEFAULT_TAG))

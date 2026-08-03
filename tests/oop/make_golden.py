"""
Capture the golden-master outputs from the OLD procedural pipeline in ``src/``.

Run this **before** the OOP refactor, with only the old ``stentfit`` package
importable (``pip install -e .`` from the repo root, which points at ``src/``)::

    python tests/oop/make_golden.py

It runs ``stentfit.stent_pipeline`` on ``stent01.stl`` into a temp folder using
the settings in :mod:`tests.oop.params`, then copies the key outputs into
``tests/oop/golden/stent01/``. If BeamMe/GMSH are available it also runs
``build_smoketest_pipeline`` and copies the generated 4C ``.yaml`` files.

The oracle is the committed files on disk — once captured, the old package is
never needed again, which is what lets ``src_oop/stentfit`` reuse the same
import name.

:raises RuntimeError: If the imported ``stentfit`` is the new ``src_oop`` one
    rather than the old procedural package.
"""

import gzip
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import (  # noqa: E402
    FINALIZE_PARAMS,
    SIMULATION_GOLDEN_FILES,
    LEGACY_SIMULATION_PARAMS,
    SKELETON_GOLDEN_FILES,
    SKELETON_PARAMS,
    STENT_NAME,
    STENT_STL,
    WRAP_MAX_SURF,
)


def _capture(src: Path, out_dir: Path) -> None:
    """
    Copy one produced file into the golden folder, gzip-compressed.

    The raw outputs total ~34 MB, which is too much to keep in git history
    forever; gzipped they are ~8 MB. :func:`tests.oop.test_equivalence.read_golden`
    decompresses them on read, so the stored form is an implementation detail.

    :param src: File the pipeline produced.
    :param out_dir: Golden folder to write ``<name>.gz`` into.
    """
    dst = out_dir / (src.name + ".gz")
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)


def _check_old_package() -> None:
    """
    Fail loudly if the importable ``stentfit`` is the new OOP package.

    The golden master must come from the procedural code in ``src/``; capturing
    it from ``src_oop`` would make the equivalence test tautological.

    :raises RuntimeError: If ``stentfit`` resolves to ``src_oop`` or does not
        expose the procedural ``stent_pipeline`` entry point.
    """
    import stentfit

    where = Path(stentfit.__file__).resolve()
    if "src_oop" in where.parts:
        raise RuntimeError(
            f"imported stentfit is the NEW oop package ({where}). "
            f"Run this with only the old package installed.")
    if not hasattr(stentfit, "stent_pipeline"):
        raise RuntimeError(
            f"imported stentfit ({where}) has no stent_pipeline - "
            f"this is not the old procedural package.")
    print(f"[golden] using old procedural package: {where}")


def make_skeleton_golden(out_dir: Path) -> Path:
    """
    Run the old skeletonisation pipeline and copy its outputs into ``out_dir``.

    :param out_dir: Golden folder for this stent, e.g. ``tests/oop/golden/stent01``.
    :returns: The temp folder the pipeline actually wrote into, so a caller can
        feed it to the simulation-side capture without recomputing.
    """
    from stentfit import stent_pipeline

    tmp_root = Path(tempfile.mkdtemp(prefix="stentfit_golden_"))
    run_dir = tmp_root / STENT_NAME

    # The old pipeline takes both phases' parameters in one flat call, and
    # max_display/random_seed appear in both param sets with the same values.
    old_kwargs = {**SKELETON_PARAMS, **FINALIZE_PARAMS}

    # "" means: accept the detected ring count, and make no manual 2D edits.
    with mock.patch("builtins.input", return_value=""):
        stent_pipeline(
            stl_file=str(REPO_ROOT / STENT_STL),
            output_dir=str(run_dir),
            stent_name=STENT_NAME,
            wrap_max_surf=WRAP_MAX_SURF,
            **old_kwargs,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SKELETON_GOLDEN_FILES:
        _capture(run_dir / name, out_dir)
        print(f"[golden] captured {name}")
    return run_dir


def make_simulation_golden(stent_run_dir: Path, out_dir: Path) -> bool:
    """
    Run the old simulation smoke-test pipeline and copy its 4C inputs into ``out_dir``.

    :param stent_run_dir: Skeletonisation output folder from
        :func:`make_skeleton_golden`.
    :param out_dir: Golden folder for this stent.
    :returns: ``True`` if the outputs were captured, ``False`` if the
        environment could not run the simulation setup (missing GMSH/BeamMe/4C
        schema), in which case the oracle stays limited to the skeleton files.
    """
    try:
        from stentfit import build_smoketest_pipeline
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[golden] simulation side skipped: {type(exc).__name__}: {exc}")
        return False

    sim_dir = stent_run_dir.parent / "simulation_input"
    sim_dir.mkdir(parents=True, exist_ok=True)
    try:
        build_smoketest_pipeline(
            stent_name=STENT_NAME,
            stent_dir=stent_run_dir,
            sim_input_dir=sim_dir,
            **LEGACY_SIMULATION_PARAMS,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[golden] simulation side FAILED: {type(exc).__name__}: {exc}")
        return False

    captured = 0
    for name in SIMULATION_GOLDEN_FILES:
        src = sim_dir / name
        if src.exists():
            _capture(src, out_dir)
            print(f"[golden] captured {name}")
            captured += 1
        else:
            print(f"[golden] MISSING {name} - not captured")
    return captured == len(SIMULATION_GOLDEN_FILES)


def main() -> None:
    """Capture every golden output this environment can produce."""
    _check_old_package()
    out_dir = Path(__file__).resolve().parent / "golden" / STENT_NAME
    run_dir = make_skeleton_golden(out_dir)
    ok = make_simulation_golden(run_dir, out_dir)
    print(f"\n[golden] skeleton outputs captured -> {out_dir}")
    print(f"[golden] simulation outputs captured: {ok}")
    print(f"[golden] pipeline scratch folder left at: {run_dir.parent}")


if __name__ == "__main__":
    main()

"""
Golden-master equivalence tests for the ``src_oop`` class API.

Each test re-runs the new classes with the same settings the golden master was
captured under (:mod:`tests.oop.params`) and compares the outputs against the
committed files in ``tests/oop/golden/``. Because the numerical kernels were
copied verbatim, the outputs should be *identical*; the tolerances here are a
safety margin against float-reduction reordering, not expected drift.

The pipeline runs are expensive, so both the skeletonisation and the simulation
setup are session-scoped fixtures shared by every assertion.
"""

import gzip
import io
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from params import (
    FINALIZE_PARAMS,
    SIMULATION_PARAMS,
    SKELETON_PARAMS,
    STENT_NAME,
    STENT_STL,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / STENT_NAME

RTOL = 1e-6
ATOL = 1e-9


# ---------------------------------------------------------------------------
# Golden-file access
# ---------------------------------------------------------------------------

def read_golden(name: str) -> bytes:
    """
    Read one golden file, transparently decompressing it.

    Goldens are stored gzipped to keep them out of git at their full ~34 MB.

    :param name: Uncompressed file name, e.g. ``"skeleton_points.csv"``.
    :raises FileNotFoundError: If the golden was never captured.
    :returns: The file's decompressed bytes.
    """
    path = GOLDEN_DIR / (name + ".gz")
    if not path.exists():
        raise FileNotFoundError(
            f"golden {name} missing - regenerate with tests/oop/make_golden.py")
    with gzip.open(path, "rb") as f:
        return f.read()


def golden_exists(name: str) -> bool:
    """
    :param name: Uncompressed golden file name.
    :returns: ``True`` if that golden was captured.
    """
    return (GOLDEN_DIR / (name + ".gz")).exists()


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def assert_json_equal(actual, expected, path: str = "$") -> None:
    """
    Compare two parsed JSON structures, with a tolerance on numbers.

    Walks both structures in parallel so a mismatch reports the exact key path
    rather than dumping two large documents.

    :param actual: Value produced by the new pipeline.
    :param expected: Value from the golden master.
    :param path: JSON path of the current node, used in failure messages.
    :raises AssertionError: On any structural or numeric mismatch.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert set(actual) == set(expected), (
            f"{path}: key mismatch; "
            f"missing={sorted(set(expected) - set(actual))} "
            f"extra={sorted(set(actual) - set(expected))}")
        for k in expected:
            assert_json_equal(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual)}"
        assert len(actual) == len(expected), (
            f"{path}: length {len(actual)} != {len(expected)}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_json_equal(a, e, f"{path}[{i}]")
    elif isinstance(expected, bool) or expected is None:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)), (
            f"{path}: expected a number, got {type(actual)}")
        assert np.isclose(actual, expected, rtol=RTOL, atol=ATOL), (
            f"{path}: {actual!r} != {expected!r}")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def assert_frames_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """
    Compare two DataFrames: same columns, same length, numerics within tolerance.

    :param actual: Table produced by the new pipeline.
    :param expected: Table from the golden master.
    :raises AssertionError: On any column, length, or value mismatch.
    """
    assert list(actual.columns) == list(expected.columns), (
        f"columns differ: {list(actual.columns)} != {list(expected.columns)}")
    assert len(actual) == len(expected), (
        f"row count differs: {len(actual)} != {len(expected)}")

    for col in expected.columns:
        exp, act = expected[col], actual[col]
        if pd.api.types.is_numeric_dtype(exp):
            assert np.allclose(act.to_numpy(), exp.to_numpy(),
                               rtol=RTOL, atol=ATOL, equal_nan=True), (
                f"column {col!r} differs numerically")
        else:
            assert act.astype(str).equals(exp.astype(str)), (
                f"column {col!r} differs")


def normalise_4c_yaml(text: str) -> list[str]:
    """
    Strip the volatile provenance block from a BeamMe-generated 4C input file.

    BeamMe stamps every file it writes with a ``TITLE.BeamMe`` block holding the
    creation timestamp and the calling script's path and git sha. Those differ
    between the golden capture and any later run without the mesh differing at
    all, so they are dropped before comparing.

    :param text: Full contents of a ``.4C.yaml`` file.
    :returns: The remaining lines, with the provenance fields removed.
    """
    volatile = ("creation_date:", "path:", "git_sha:", "git_date:")
    return [line for line in text.splitlines()
            if not line.strip().startswith(volatile)]


# ---------------------------------------------------------------------------
# Pipeline runs (session-scoped: expensive)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def stent_run(tmp_path_factory, repo_root):
    """
    Run the new :class:`~stentfit.stent.Stent` pipeline once for the session.

    ``builtins.input`` is mocked to return ``""``, which accepts the detected
    ring count and declines any manual 2D edits — the same fully-automatic path
    the golden master was captured under.

    :param tmp_path_factory: pytest's session temp-dir factory.
    :param repo_root: Repository root, for resolving the input STL.
    :returns: ``(stent, output_dir)`` — the finished object and its output folder.
    """
    from stentfit import Stent

    out_dir = tmp_path_factory.mktemp("oop_stent") / STENT_NAME
    with mock.patch("builtins.input", return_value=""):
        stent = Stent(str(repo_root / STENT_STL), STENT_NAME, str(out_dir),
                      **SKELETON_PARAMS)
        stent.skeletonize(prune_tip_frac=FINALIZE_PARAMS["prune_tip_frac"])
    return stent, Path(stent.output_dir)


@pytest.fixture(scope="session")
def simulation_run(tmp_path_factory, stent_run):
    """
    Run the new :class:`~stentfit.simulation.Simulation` setup once for the session.

    :param tmp_path_factory: pytest's session temp-dir factory.
    :param stent_run: The finished stent to build the simulation around.
    :returns: ``(simulation, sim_input_dir)``.
    :raises pytest.skip.Exception: If the environment cannot run the setup
        (no GMSH, BeamMe, or 4C schema).
    """
    from stentfit import Simulation

    stent, _ = stent_run
    sim_dir = tmp_path_factory.mktemp("oop_simulation")
    try:
        sim = Simulation(stent, sim_input_dir=sim_dir, **SIMULATION_PARAMS).setup()
    except ImportError as exc:                      # pragma: no cover
        pytest.skip(f"simulation dependencies unavailable: {exc}")
    return sim, Path(sim_dir)


# ---------------------------------------------------------------------------
# Stent equivalence
# ---------------------------------------------------------------------------

def test_stent_features_matches_golden(stent_run):
    """``stent_features.json`` matches the golden master."""
    _, out_dir = stent_run
    actual = json.loads((out_dir / "stent_features.json").read_text())
    expected = json.loads(read_golden("stent_features.json"))
    assert_json_equal(actual, expected)


def test_skeleton_points_matches_golden(stent_run):
    """``skeleton_points.csv`` matches the golden master."""
    _, out_dir = stent_run
    actual = pd.read_csv(out_dir / "skeleton_points.csv")
    expected = pd.read_csv(io.BytesIO(read_golden("skeleton_points.csv")))
    assert_frames_equal(actual, expected)


def test_skeleton_splines_matches_golden(stent_run):
    """``skeleton_splines.json`` matches the golden master, curve by curve."""
    _, out_dir = stent_run
    actual = json.loads((out_dir / "skeleton_splines.json").read_text())
    expected = json.loads(read_golden("skeleton_splines.json"))

    assert actual["n_curves"] == expected["n_curves"]
    assert len(actual["curves"]) == len(expected["curves"])
    for i, (a, e) in enumerate(zip(actual["curves"], expected["curves"])):
        assert_json_equal(a, e, f"curves[{i}]")


def test_stent_object_state(stent_run):
    """
    The finished object exposes the pipeline results, with no duplicated fields.

    Guards the Section 4.1 decisions: ``r_mid``/``strut_thickness`` are read out
    of ``stent_features`` rather than copied onto the instance, and
    ``circumference`` is a derived property.
    """
    stent, _ = stent_run

    assert stent.stent_df is not None
    assert stent.skeleton_df is not None
    assert stent.skeleton_curves is not None
    assert stent.skeleton_splines is not None
    assert len(stent.skeleton_curves) == len(stent.skeleton_splines)
    assert stent.ring_2d and stent.ring_order

    # circumference is derived, never stored
    assert "circumference" not in vars(stent)
    assert np.isclose(stent.circumference,
                      2 * np.pi * stent.stent_features["r_mid"])

    # r_mid / strut_thickness live only in stent_features
    assert "r_mid" not in vars(stent)
    assert "strut_thickness" not in vars(stent)


def test_stent_load_round_trip(stent_run):
    """
    :meth:`~stentfit.stent.Stent.load` restores a run from its output folder.

    This is the replacement for the old ``state=None`` resume branch, so it has
    to bring back everything the manual-edit phase needs.
    """
    from stentfit import Stent

    stent, out_dir = stent_run
    reloaded = Stent.load(str(out_dir))

    assert reloaded.stent_name == STENT_NAME
    assert set(reloaded.ring_2d) == set(stent.ring_2d)
    assert np.isclose(reloaded.circumference, stent.circumference)
    assert np.allclose(reloaded.ring_edges, stent.ring_edges)
    assert len(reloaded.stent_df) == len(stent.stent_df)
    for label, rec in stent.ring_2d.items():
        assert np.allclose(reloaded.ring_2d[label]["arc"], rec["arc"])
        assert np.allclose(reloaded.ring_2d[label]["z"], rec["z"])


def test_stent_missing_stl_raises(tmp_path):
    """A non-existent STL fails fast, before any output folder work."""
    from stentfit import Stent

    stent = Stent(str(tmp_path / "nope.stl"), "nope", str(tmp_path / "out"))
    with pytest.raises(FileNotFoundError, match="No STL file"):
        stent.skeletonize_2d()


# ---------------------------------------------------------------------------
# Simulation equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "artery_solid.4C.yaml",
    "stent_warped.4C.yaml",
    "artery_stent.4C.yaml",
    "simulation.4C.yaml",
])
def test_simulation_yaml_matches_golden(simulation_run, name):
    """
    Each generated 4C input matches the golden master.

    Compared line by line with BeamMe's volatile provenance block normalised
    out (see :func:`normalise_4c_yaml`).
    """
    if not golden_exists(name):
        pytest.skip(f"golden {name} was not captured in this environment")

    _, sim_dir = simulation_run
    produced = sim_dir / name
    assert produced.exists(), f"{name} was not written"

    actual = normalise_4c_yaml(produced.read_text())
    expected = normalise_4c_yaml(read_golden(name).decode())

    assert len(actual) == len(expected), (
        f"{name}: {len(actual)} lines != {len(expected)} golden lines")
    for i, (a, e) in enumerate(zip(actual, expected), start=1):
        assert a == e, f"{name}: line {i} differs\n  actual:   {a}\n  expected: {e}"


def test_simulation_coupling_checks_pass(simulation_run):
    """The mixed-dimensional coupling checks all pass, as they did for the golden."""
    sim, _ = simulation_run
    assert sim.coupling_report is not None
    assert sim.coupling_report["all_passed"] is True


def test_simulation_object_state(simulation_run):
    """
    The finished simulation exposes its meshes and reads through to its parts.

    Guards the Section 4.3 decisions: the stent features and the artery solid
    path are read off the composed objects rather than duplicated onto the
    simulation.
    """
    sim, sim_dir = simulation_run

    assert sim.beam_mesh is not None
    assert sim.full_mesh is not None
    assert sim.artery is not None
    assert sim.artery.solid_yaml == sim_dir / "artery_solid.4C.yaml"

    # features / strut thickness / solid yaml are never copied onto the sim
    for duplicated in ("features", "stent_strut_thickness", "artery_solid_yaml"):
        assert duplicated not in vars(sim)

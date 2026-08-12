import json
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from stentfit import Stent

REPO_ROOT = Path(__file__).parent.parent
STL_FILE = REPO_ROOT / "examples/data/input/stent_designs/stent01.stl"
SETTINGS = {
    "n_points": None,
    "max_display": 500_000,
    "remove_supports": False,
    "random_seed": 0,       
    "n_rings": None,
    "auto_tune": True,
    "pixels_per_strut": 10,
    "dilate_px": 3,
    "pad_fraction": 0.20,
    "tune_time_limit": 100_000,  # so high that the clock never stops the search
    "quality_gamma": 2.0,
    "ring_halo_frac": 0.4,
}

@pytest.fixture(scope="session")
def stent():
    """Run the pipeline on stent01 once, and hand the result to every test.

    ``scope="session"`` makes it run one time instead of once per test.

    ``input()`` is replaced so the pipeline never stops to ask us anything.
    Answering ``""`` accepts the ring count it found and skips the manual edits.
    """
    output_dir = Path(tempfile.mkdtemp()) / "stent01"

    with mock.patch("builtins.input", return_value=""):
        finished = Stent(str(STL_FILE), "stent01", str(output_dir), **SETTINGS)
        finished.skeletonize()

    return finished


@pytest.fixture(scope="session")
def output(stent):
    """The folder the run wrote its files into."""
    return Path(stent.output_dir)


@pytest.fixture(scope="session")
def points(output):
    """The skeleton points the run produced, as a table."""
    return pd.read_csv(output / "skeleton_points.csv")


@pytest.fixture(scope="session")
def splines(output):
    """The curves the run fitted, as a dictionary."""
    return json.loads((output / "skeleton_splines.json").read_text())

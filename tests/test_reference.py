"""Check the pipeline still gives the same answer it gave before.

Every test here compares the pipeline's output against the reference files in
``reference/stent01/``. Those files were made by this same code earlier and were
checked by hand. If a test here fails, either the code is broken or the pipeline
was changed on purpose and the new answer is better. So these tests answer 
"did anything change?", not "is it correct?".
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = Path(__file__).parent / "reference" / "stent01"


def test_features_match_reference(output):
    """The measured stent geometry is the same as before.

    This is length, diameter, the radii, the strut thickness, the centreline
    direction, and the ring boundaries.
    """
    actual = json.loads((output / "stent_features.json").read_text())
    expected = json.loads((REFERENCE / "stent_features.json").read_text())

    assert sorted(actual) == sorted(expected)

    for name in expected:
        assert np.allclose(actual[name], expected[name]), name


def test_skeleton_points_match_reference(output):
    """Every point of the 3D skeleton is the same as before.

    This is the strongest test in the suite. The file has one row per skeleton
    point: where it is, what kind of point it is, and which points it connects
    to. There are about 70,000 of them.
    """
    actual = pd.read_csv(output / "skeleton_points.csv")
    expected = pd.read_csv(REFERENCE / "skeleton_points.csv")

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-6, atol=1e-9)


def test_splines_match_reference(output):
    """Every fitted curve is the same as before.

    Compared one curve at a time, so the error message tells us which curve is
    different instead of just saying "the file is different".
    """
    actual = json.loads((output / "skeleton_splines.json").read_text())
    expected = json.loads((REFERENCE / "skeleton_splines.json").read_text())

    assert actual["n_curves"] == expected["n_curves"]

    for i in range(expected["n_curves"]):
        a = actual["curves"][i]
        e = expected["curves"][i]

        assert a["degree"] == e["degree"], f"curve {i}"
        assert a["is_loop"] == e["is_loop"], f"curve {i}"
        assert np.isclose(a["length"], e["length"]), f"curve {i}"
        assert np.allclose(a["control_points"], e["control_points"]), f"curve {i}"
        assert np.allclose(a["knot_vector"], e["knot_vector"]), f"curve {i}"

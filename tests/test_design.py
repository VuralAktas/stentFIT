"""Check the pipeline found the stent we actually gave it.

Every test here compares the pipeline's output against the stent01 design itself, not
against any earlier run of the pipeline. These values inside ``reference/stent01/design.json`` 
come from user visual inspection of the stent01 design, not from any run of the pipeline, and these 
tests check that the pipeline found the same numbers.
"""

import json
from pathlib import Path

DESIGN = json.loads(
    (Path(__file__).parent / "reference" / "stent01" / "design.json").read_text())


def test_the_right_number_of_rings_was_found(stent):
    """Ring detection finds as many rings as the design really has.

    The pipeline finds the rings by itself, by looking for dips in how many
    points there are along the stent axis. If it gets this wrong, every ring
    boundary moves, and the whole stent is skeletonised differently. No other
    test would notice, because the wrong answer would still be tidy and
    repeatable.
    """
    assert len(stent.ring_order) == DESIGN["n_rings"]


def test_the_right_number_of_curves_was_fitted(stent):
    """One curve comes out for each strut in the design.

    Too few curves means two struts were joined into one across a junction. Too
    many means one strut was cut in half. Either way the beam mesh built from
    these curves would not be the real stent.
    """
    assert len(stent.skeleton_splines) == DESIGN["n_curves"]


def test_the_junctions_agree_with_the_design(points, splines):
    """The junctions found also point to the design's number of struts.

    Every strut ends at two junctions, so the junction degrees add up to twice
    the number of struts.

    This reaches the same design number from the other side. The test above
    counts what the curve fitter made; this one counts what the graph found. If
    they ever disagree, the two halves of the pipeline are describing different
    stents.
    """
    junctions = points[points["node_type"] != "line"]
    loops = sum(1 for curve in splines["curves"] if curve["is_loop"])

    assert junctions["degree"].sum() / 2 + loops == DESIGN["n_curves"]

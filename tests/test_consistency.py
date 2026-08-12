"""Check the output makes sense on its own.

Every test here checks something that has to be true for any correct skeleton of
any stent. They are true because of geometry, because of how a graph works, or
because of what a B-spline is.
"""

import ast
import json
import numpy as np
import pytest
from stentfit import Stent


def test_point_ids_are_0_1_2_and_so_on(points):
    """The point ids count up from 0 with no gaps and no repeats.

    Points refer to each other by id, so a missing or repeated id would make
    those references point at the wrong place.
    """
    ids = points["skeleton_point_id"].to_numpy()

    assert points["skeleton_point_id"].is_unique
    assert list(ids) == list(range(len(points)))


def test_degree_is_the_number_of_neighbours(points):
    """A point's ``degree`` is how many neighbours it actually has.

    The degree is stored in its own column, separate from the neighbour list, so
    the two can disagree. Other code decides what to do based on the degree, so
    a disagreement would be a real bug.
    """
    for degree, text in zip(points["degree"], points["neighbor_ids"]):
        assert degree == len(ast.literal_eval(text))


def test_neighbours_point_both_ways(points):
    """If point A says B is its neighbour, then B must say A as well.

    The skeleton is an undirected graph, so every connection goes both ways. A
    one-way connection would be a strut attached at one end only.
    """
    neighbours = {point_id: ast.literal_eval(text)
                  for point_id, text
                  in zip(points["skeleton_point_id"], points["neighbor_ids"])}

    for point_id, ids in neighbours.items():
        for other in ids:
            assert point_id in neighbours[other], f"{point_id} -> {other} is one way"


def test_no_point_is_its_own_neighbour(points):
    """A point is never connected to itself."""
    for point_id, text in zip(points["skeleton_point_id"], points["neighbor_ids"]):
        assert point_id not in ast.literal_eval(text)


def test_node_type_matches_the_degree(points):
    """The label on a point agrees with how many neighbours it has.

    A ``line`` point is in the middle of a strut, so it has 2 neighbours.
    A ``junction`` is where struts meet, so it has 3 or more.
    An ``end`` is a loose tip, so it has 1.
    """
    for node_type, degree in zip(points["node_type"], points["degree"]):
        if node_type == "line":
            assert degree == 2
        elif node_type == "junction":
            assert degree >= 3
        elif node_type == "end":
            assert degree == 1
        else:
            pytest.fail(f"unknown node type: {node_type}")


def test_r_and_theta_match_x_and_y(points):
    """``r`` and ``theta`` are just ``x`` and ``y`` written a different way.

    Both are saved in the same file. If they disagree, then two pieces of code
    reading the same row would be looking at two different points.
    """
    x = points["x"].to_numpy()
    y = points["y"].to_numpy()

    assert np.allclose(points["r"].to_numpy(), np.sqrt(x**2 + y**2))
    assert np.allclose(points["theta"].to_numpy(), np.arctan2(y, x))


def test_points_are_inside_the_stent(points, stent):
    """No skeleton point sticks out of the stent it came from.

    The skeleton runs down the middle of the strut material, so it cannot go
    past the ends of the stent, and it cannot leave the wall.
    """
    features = stent.stent_features

    assert points["z"].min() >= features["z_min"] - 1e-9
    assert points["z"].max() <= features["z_max"] + 1e-9
    assert points["r"].min() >= features["r_inner"]
    assert points["r"].max() <= features["r_outer"]


def test_features_match_their_own_definitions(stent):
    """The measured numbers add up.

    Each one is worked out somewhere in the pipeline. Checking them against each
    other finds a wrong formula without needing to know the right answer.
    """
    features = stent.stent_features

    assert np.isclose(features["length"], features["z_max"] - features["z_min"])
    assert np.isclose(features["diameter"], 2 * features["radius"])
    assert np.isclose(features["r_mid"],
                      (features["r_inner"] + features["r_outer"]) / 2)
    assert np.isclose(features["strut_thickness"],
                      features["r_outer"] - features["r_inner"])

    assert features["r_inner"] < features["r_outer"]
    assert features["strut_thickness"] > 0
    assert features["num_points"] > 0


def test_centreline_direction_has_length_1(stent):
    """The stent's long axis is a unit vector.

    It comes out of a PCA and is used as a direction. If it were not length 1,
    everything measured along it would be quietly scaled by the wrong amount.
    """
    direction = np.asarray(stent.stent_centerline_direction, float).ravel()

    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_ring_boundaries_go_up_and_cover_the_stent(stent):
    """The ring boundaries increase, and they reach both ends of the stent.

    They cut the stent into rings. If they were out of order, or stopped short,
    the rings would overlap or leave a gap.
    """
    edges = np.asarray(stent.ring_edges, float).ravel()

    assert (np.diff(edges) > 0).all()
    assert np.isclose(edges[0], stent.stent_features["z_min"])
    assert np.isclose(edges[-1], stent.stent_features["z_max"])


def test_every_ring_was_skeletonised(stent):
    """``n`` boundaries make ``n - 1`` rings, and every one of them was used.

    A ring that was found but then quietly dropped would leave a hole in the
    middle of the stent, which is easy to miss when looking at the wireframe.
    """
    edges = np.asarray(stent.ring_edges, float).ravel()

    assert len(edges) - 1 == len(stent.ring_order)


def test_number_of_curves_is_consistent(splines, stent):
    """The curve count is the same everywhere it is written down."""
    assert splines["n_curves"] == len(splines["curves"])
    assert splines["n_curves"] == len(stent.skeleton_splines)


def test_curve_count_matches_the_junctions(points, splines):
    """The junctions tell us how many curves there should be, and they agree.

    Every curve ends at two junctions, and each of those ends uses up one of
    that junction's connections. So adding up the degrees of all the junctions
    counts every curve exactly twice.

    The graph and the curve fitter are two different steps, so it means
    something when they agree.
    """
    junctions = points[points["node_type"] != "line"]
    loops = sum(1 for curve in splines["curves"] if curve["is_loop"])

    assert junctions["degree"].sum() / 2 + loops == splines["n_curves"]


def test_curves_are_valid_bsplines(splines):
    """Every curve follows the rule ``knots = control points + degree + 1``.

    This is what makes a B-spline a B-spline. If it is broken, the curve cannot
    be drawn at all, and splinepy and BeamMe will both refuse it later.
    """
    for i, curve in enumerate(splines["curves"]):
        if curve["knot_vector"] is None:
            continue  # a straight polyline, which has no knots

        assert (len(curve["knot_vector"])
                == len(curve["control_points"]) + curve["degree"] + 1), f"curve {i}"


def test_knot_vectors_never_go_backwards(splines):
    """The numbers in a knot vector never decrease, as the definition requires."""
    for i, curve in enumerate(splines["curves"]):
        if curve["knot_vector"] is None:
            continue

        knots = np.asarray(curve["knot_vector"], float)
        assert (np.diff(knots) >= -1e-12).all(), f"curve {i}"


def test_curves_are_3d_and_not_empty(splines):
    """Control points have x, y and z, and no curve has zero length.

    A curve with no length would be a strut that turns into nothing when the
    beam mesh is built.
    """
    for i, curve in enumerate(splines["curves"]):
        control_points = np.asarray(curve["control_points"], float)

        assert control_points.shape[1] == 3, f"curve {i}"
        assert curve["length"] > 0, f"curve {i}"


def test_the_finished_object_has_everything(stent):
    """The Stent object holds all the results at the end of the run."""
    assert stent.stent_df is not None
    assert stent.skeleton_df is not None
    assert stent.skeleton_curves is not None
    assert stent.skeleton_splines is not None
    assert stent.ring_edges is not None
    assert stent.stent_centerline_direction is not None

    assert len(stent.skeleton_curves) == len(stent.skeleton_splines)
    assert len(stent.ring_2d) == len(stent.ring_order)


def test_the_written_features_match_the_object(output, stent):
    """``stent_features.json`` says the same thing the object does.

    The file is not just a copy of ``stent.stent_features``. When it is written,
    two more values are added to it: the centreline direction and the ring
    boundaries, which the object keeps as separate attributes. So the file and
    the object can end up disagreeing, and nothing else would notice.
    """
    written = json.loads((output / "stent_features.json").read_text())

    for name, value in stent.stent_features.items():
        assert name in written, name
        assert np.isclose(written[name], float(value)), name

    assert np.allclose(written["ring_boundaries"],
                       np.asarray(stent.ring_edges, float).ravel())
    assert np.allclose(written["stent_centerline_direction"],
                       np.asarray(stent.stent_centerline_direction, float).ravel())


def test_a_finished_run_can_be_loaded_again(output, stent):
    """``Stent.load()`` brings a finished run back from its folder.

    This is what you use after restarting the kernel, so it has to bring back
    everything the next step needs.
    """
    reloaded = Stent.load(str(output))

    assert reloaded.stent_name == "stent01"
    assert set(reloaded.ring_2d) == set(stent.ring_2d)
    assert len(reloaded.stent_df) == len(stent.stent_df)
    assert np.allclose(reloaded.ring_edges, stent.ring_edges)
    assert np.isclose(reloaded.stent_features["r_mid"],
                      stent.stent_features["r_mid"])


def test_a_missing_stl_gives_a_clear_error(tmp_path):
    """Asking for an STL that is not there fails straight away.

    It should fail before doing any work, with a message that says what is
    wrong, not with a confusing error much later on.
    """
    missing = Stent(str(tmp_path / "not_here.stl"), "not_here", str(tmp_path / "out"))

    with pytest.raises(FileNotFoundError, match="No STL file"):
        missing.skeletonize_2d()

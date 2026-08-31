"""
The stent as a 1D beam mesh: meshing it, welding its crowns, and measuring it.

Every simulation type starts here. What differs between them is how the stent is loaded, not how
it is built, so all of this is shared and owned by no case.

The mesh itself comes from :func:`stentfit.core.splines.mesh_skeleton_beams`, unchanged.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from beamme.core.conf import bme
from beamme.core.coupling import Coupling

from ..core.splines import mesh_skeleton_beams
from .materials import beam_material, section_properties

#: How a crown is welded. ``fix`` welds the 3 translations and 3 rotations and leaves the Hermite
#: tangent DOFs free, because struts arriving from different directions have different tangents.
JUNCTION_COUPLING_DOF = bme.coupling_dof.fix


# --------------------------------------------------------------------------------------
# Reading what the skeletonisation measured
# --------------------------------------------------------------------------------------


def load_features(stent_dir) -> dict:
    """
    Read a stent's measured features.

    :param stent_dir: The stent's output folder.
    :returns: The parsed ``stent_features.json``.
    :raises FileNotFoundError: If the stent has not been skeletonised.
    """
    stent_dir = Path(stent_dir)
    if not (stent_dir / "skeleton_splines.json").exists():
        raise FileNotFoundError(
            f"no skeletonisation in {stent_dir} - run Stent.skeletonize() first")
    return json.loads((stent_dir / "stent_features.json").read_text())


def read_junctions(stent_dir) -> tuple:
    """
    Read the stent's junction points from its own skeletonisation output.

    ``skeleton_points.csv`` labels every skeleton point with a ``node_type`` and a graph
    ``degree``, and a point where three or more struts meet is a junction. Using that label rather
    than a geometric guess keeps this correct for any stent, whatever its crown count.

    :param stent_dir: The stent's output folder.
    :returns: ``(coords, degrees)``, the ``(n, 3)`` junction positions and the degree of each.
    :raises FileNotFoundError: If ``skeleton_points.csv`` is absent. Without it the struts stay
        disconnected and the stent is not a structure, so this is fatal.
    """
    csv = Path(stent_dir) / "skeleton_points.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"no skeleton_points.csv in {stent_dir} - junction coupling needs it, and without "
            f"the struts staying welded the stent is not a structure")

    points = pd.read_csv(csv)
    junctions = points[points["node_type"] == "junction"]
    return junctions[["x", "y", "z"]].to_numpy(float), junctions["degree"].to_numpy(int)


def degree_counts(degrees) -> dict:
    """
    :param degrees: Graph degrees from :func:`read_junctions`.
    :returns: How many junctions have each degree, e.g. ``{3: 90}``.
    """
    return dict(sorted(Counter(np.asarray(degrees).tolist()).items()))


# --------------------------------------------------------------------------------------
# Building the mesh
# --------------------------------------------------------------------------------------


def build_stent_beams(stent_dir, settings, l_el: float,
                      strut_thickness: float) -> tuple:
    """
    Mesh the stent's fitted splines as beams and give them their material.

    ``mesh_skeleton_beams`` builds an elastic ``MaterialReissner`` internally, taking the
    cross-section radius from ``stent_features.json`` rather than from anything passed in. That
    material is always replaced afterwards with the one from
    :func:`~stentfit.sim.materials.section_properties`, so the mesh cannot be meshed for one
    thickness and shaped like another. For an elastic run at the settings' own thickness the
    replacement is numerically identical to what was there.

    :param stent_dir: The stent's output folder, holding the fitted splines.
    :param settings: The simulation's settings, for the strut material and element type.
    :param l_el: Target element length, in mm.
    :param strut_thickness: Strut thickness, in mm, for the section properties.
    :returns: ``(mesh, section)`` -- the BeamMe mesh and its section properties.
    """
    mesh = mesh_skeleton_beams(str(stent_dir), l_el=l_el,
                               youngs_modulus=settings.youngs,
                               poisson_ratio=settings.poisson,
                               density=settings.density,
                               beam_class_label=settings.beam_class)

    section = section_properties(strut_thickness, settings)

    material = beam_material(section, settings)
    mesh.materials = [material]
    for element in mesh.elements:
        element.material = material

    return mesh, section


def centreline_nodes(mesh) -> list:
    """
    Return the beam nodes that carry translational DOF.

    ``Beam3rHerm2Line3`` nodes hold 9 DOF, ``[disp(3), rot(3), tangent(3)]``, but only the Hermite
    centreline nodes have translations. The interior "middle" nodes carry rotations only, so a
    displacement condition on them is meaningless and 4C rejects it.

    :param mesh: The beam mesh.
    :returns: The centreline nodes.
    """
    return [node for node in mesh.nodes if not node.is_middle_node]


# --------------------------------------------------------------------------------------
# Welding the crowns
# --------------------------------------------------------------------------------------


def couple_junctions(mesh, nodes: list, junction_coords, junction_degrees) -> dict:
    """
    Weld the beam nodes at each labelled crown together.

    ``mesh_skeleton_beams`` meshes each strut on its own, so a crown where three struts meet
    becomes three nodes at identical coordinates with nothing relating them. Undeformed they
    overlap and the stent looks intact, but under load each strut moves on its own and the
    structure comes apart.

    For a junction of degree ``d`` the ``d`` nearest beam nodes are the ones meshed from the struts
    ending there, so the degree says how many nodes to take and no distance threshold is needed.

    :param mesh: The beam mesh, modified in place.
    :param nodes: The centreline nodes.
    :param junction_coords: ``(n, 3)`` junction positions from :func:`read_junctions`.
    :param junction_degrees: How many struts meet at each of those junctions.
    :returns: Dict with ``n_junctions``, ``degrees`` and ``max_gap``. The gap is the largest
        distance from a labelled junction to the beam nodes welded there; it should be tiny, and a
        large value means the wrong nodes were selected.
    """
    tree = cKDTree(np.array([node.coordinates for node in nodes], float))

    gaps = []
    for coord, degree in zip(junction_coords, junction_degrees):
        distance, index = tree.query(coord, k=int(degree))
        distance, index = np.atleast_1d(distance), np.atleast_1d(index)
        gaps.append(float(distance.max()))

        # Coupling checks that the nodes it is given really are at one position, so a bad
        # selection raises here instead of silently welding the wrong struts together.
        mesh.add(Coupling([nodes[i] for i in index],
                          bme.bc.point_coupling, JUNCTION_COUPLING_DOF))

    return {"n_junctions": len(junction_coords),
            "degrees": degree_counts(junction_degrees),
            "max_gap": max(gaps) if gaps else 0.0}


# --------------------------------------------------------------------------------------
# Measuring the mesh
# --------------------------------------------------------------------------------------


def axis_report(nodes: list) -> dict:
    """
    Measure the stent's axis alignment and extent from its own nodes.

    Everything downstream assumes the stent runs along z and is centred on the origin in x-y, which
    is what ``Stent.skeletonize()`` produces. It is cheap to measure rather than trust, because a
    non-zero ``(cx, cy)`` would quietly turn a radial scaling into a translation.

    ``stent_features.json`` also carries ``stent_centerline_direction``, but that is the axis in
    the original STL frame, before alignment, so it must not be used here.

    :param nodes: The centreline nodes.
    :returns: Dict with ``cx``, ``cy``, ``z_min``, ``z_max``, ``r_mean``, and the derived
        ``length`` and ``diameter``, all in mm.
    """
    points = np.array([node.coordinates for node in nodes], float)
    cx, cy = float(points[:, 0].mean()), float(points[:, 1].mean())
    radii = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
    r_mean = float(radii.mean())

    return {"cx": cx, "cy": cy, "z_min": z_min, "z_max": z_max, "r_mean": r_mean,
            "length": z_max - z_min, "diameter": 2.0 * r_mean}


def beam_element_lengths(mesh) -> np.ndarray:
    """
    Measure every beam element's length, along the element.

    The ``l_el`` handed to the mesher is only a target. BeamMe divides each strut into a whole
    number of elements, so the lengths produced scatter either side of it. The coupling rules apply
    to the elements that exist rather than to the target, and with the target near the limit that
    difference decides whether a mesh is valid.

    The length is measured end to middle to end rather than as the straight chord. The chord is
    wrong for tightly curved struts, because at a folded crown the strut doubles back and the
    distance between an element's two ends can be half the length of the element itself. Measured
    by chord, a crimped stent's elements look far shorter than they are and the coupling check
    rejects a mesh that is fine.

    Two straight segments through the middle node still slightly under-measure a curved element,
    which keeps the error in the safe direction, so the check can only be stricter than reality.

    :param mesh: The beam mesh.
    :returns: Element lengths, in mm.
    """
    lengths = []
    for element in mesh.elements:
        ends = [np.asarray(n.coordinates, float)
                for n in element.nodes if not n.is_middle_node]
        middle = [np.asarray(n.coordinates, float)
                  for n in element.nodes if n.is_middle_node]
        if middle:
            lengths.append(float(np.linalg.norm(middle[0] - ends[0])
                                 + np.linalg.norm(ends[-1] - middle[0])))
        else:
            lengths.append(float(np.linalg.norm(ends[-1] - ends[0])))
    return np.array(lengths)


def innermost_surface_radius(nodes: list, section: dict) -> float:
    """
    Radius of the stent's innermost strut *surface*, measured from the mesh.

    The features file's ``r_inner`` is an average, and how far the innermost node sits inside that
    average changes from stent to stent. So anything placed relative to ``r_inner`` means a
    different clearance on each one. Measuring instead makes a requested clearance the clearance
    that results.

    :param nodes: The centreline nodes.
    :param section: The strut section, from :func:`~stentfit.sim.materials.section_properties`.
    :returns: Radius of the innermost strut surface, in mm.
    """
    points = np.array([node.coordinates for node in nodes], float)
    return float(np.hypot(points[:, 0], points[:, 1]).min()) - section["radius"]

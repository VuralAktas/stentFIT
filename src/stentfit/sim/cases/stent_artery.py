"""
The stent tied into a test artery: a mixed-dimensional smoke test.

This exercises the whole beam-to-solid chain end to end. The stent is warped onto a curved
centreline, the artery wall is meshed as a 3D solid, the two are tied together, and a runnable
input is written. It is not the physics of the reference papers, since the artery uses a
placeholder ``StVenantKirchhoff`` material, the coupling is tied meshtying rather than true
contact, and the balloon is a simplified radial point force.

For deployment physics use ``stent_balloon``, which is loaded only through contact and has
converged runs behind it. This case is kept because it is the only one involving an artery, and it
is where beam-to-solid meshtying lives.
"""

import numpy as np
from scipy.spatial import cKDTree

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySet
from beamme.core.rotation import Rotation
from beamme.cosserat_curve.cosserat_curve import CosseratCurve
from beamme.cosserat_curve.warping_along_cosserat_curve import warp_mesh_along_curve
from beamme.four_c.header_functions import (set_beam_to_solid_meshtying,
                                            set_header_static, set_runtime_output)
from beamme.four_c.input_file import InputFile

from ...core import artery_geom as _geom
from .. import beam_model as bm
from ..coupling import (check_coupling, coupling_limits, print_coupling,
                        raise_if_coupling_failed)
from ..record import base_record
from ..settings import settings_to_dict
from . import BuildContext, built_case, register

def radial_directions(points: np.ndarray, centreline: np.ndarray) -> np.ndarray:
    """
    Find each point's outward radial direction relative to a centreline.

    For each point the nearest centreline point is found, and the vector to it has its tangent
    component projected out, leaving only the perpendicular part. This points the balloon force
    straight out from the artery's local axis at each stent node, however the artery bends.

    :param points: ``(n, 3)`` points to find the direction at.
    :param centreline: The artery centreline.
    :returns: ``(n, 3)`` outward unit vectors.
    """
    centreline = np.asarray(centreline, dtype=float)
    tangent = np.gradient(centreline, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    _, index = cKDTree(centreline).query(points)
    radial = points - centreline[index]
    radial -= np.einsum("ij,ij->i", radial, tangent[index])[:, None] * tangent[index]
    norm = np.linalg.norm(radial, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return radial / norm


def warp_onto_artery(beam_mesh, artery, features: dict) -> None:
    """
    Warp the straight stent onto the artery centreline.

    Represents the centreline as a BeamMe ``CosseratCurve``, rotates the stent's own straight axis
    onto the curve's tangent, and centres the stent's mid-length on the curve's arc mid-point.

    :param beam_mesh: The stent beam mesh, modified in place.
    :param artery: The artery to warp onto.
    :param features: The stent's measured features.
    """
    centreline = artery.centreline
    curve = CosseratCurve(centreline)

    reference = Rotation([0.0, 1.0, 0.0], -np.pi / 2.0)      # first basis vector -> +Z
    arc = np.linalg.norm(np.diff(centreline, axis=0), axis=1).sum()
    z_center = 0.5 * (features["z_min"] + features["z_max"])
    origin = np.array([0.0, 0.0, z_center - arc / 2.0])

    warp_mesh_along_curve(beam_mesh, curve, origin=origin, reference_rotation=reference)


def build(ctx: BuildContext) -> list:
    """
    Build the tied beam-to-solid input.

    :param ctx: The build context. Everything comes from
        :class:`~stentfit.sim.settings.StentArterySettings`; ``options["artery"]`` carries the
        :class:`~stentfit.artery.Artery`.
    :returns: A single entry.
    :raises ValueError: If no artery was given, if it was built for a different stent, or if the
        coupling rules are violated.
    """
    artery = ctx.options.get("artery")
    if artery is None:
        raise ValueError("stent_artery needs an Artery: "
                         "Simulation(stent, sim_type='stent_artery', artery=Artery(stent, ...))")

    settings = ctx.settings
    force = settings.expansion_force

    strut = ctx.strut_thickness()
    limits = coupling_limits(strut, settings)
    l_el = limits["l_beam"]

    run_dir = ctx.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"stent {ctx.stent_name}: diameter {ctx.features['diameter']:.3f} mm, "
          f"strut {strut:.4f} mm")
    print(f"  sizing: solid {limits['h_solid']:.4f} mm, beam {l_el:.4f} mm")

    # --- stent, meshed straight then warped onto the artery -------------------------
    beam_mesh, section = bm.build_stent_beams(ctx.stent_dir, settings,
                                              l_el=l_el, strut_thickness=strut)

    # Measured before the warp, on purpose. Bending the stent onto the centreline shortens the
    # elements on the inside of the bend and lengthens those on the outside -- but the artery
    # solid is warped by the *same* centreline, so its elements shrink there too. The solid size
    # this is compared against is the straight-tube design size handed to gmsh below, so the beam
    # has to be measured in the same configuration. Comparing a warped beam against an unwarped
    # solid target penalises the beam for a bend they both undergo: on stent01's curved artery it
    # turns a ratio of 1.12 into 0.99 and rejects a mesh that is fine.
    lengths = bm.beam_element_lengths(beam_mesh)

    warp_onto_artery(beam_mesh, artery, ctx.features)

    stent_yaml = run_dir / "stent_warped.4C.yaml"
    warped = InputFile()
    warped.add(beam_mesh)
    warped.dump(str(stent_yaml), validate=False, add_footer_application_script=False)
    print(f"[saved] {stent_yaml.name}")

    # --- artery wall, meshed as a 3D solid ------------------------------------------
    artery.mesh_solid(out_path=run_dir / "artery_solid.4C.yaml",
                      element_size=limits["h_solid"])

    # --- coupling checks -------------------------------------------------------------
    report = check_coupling(lengths, {"solid": limits["h_solid"]},
                            d_beam=limits["d_beam"], e_beam=settings.youngs,
                            e_solid=artery.artery_youngs)
    print("mixed-dimensional coupling (Steinbrecher et al.):")
    print_coupling(report)
    raise_if_coupling_failed(report, limits)

    # --- assemble the beam and the solid, tied together -----------------------------
    inp, solid = _geom.import_artery_solid(artery.solid_yaml)
    combined_path = run_dir / "artery_stent.4C.yaml"
    _, full_mesh = _geom.assemble_beam_solid(
        inp, solid, beam_mesh,
        lumen_surface_index=settings.lumen_surface_index,
        # Defaults to beam_to_solid_surface_meshtying. A real deployment run would need
        # beam_to_solid_surface_contact, which is what the stent_balloon case uses.
        bc_type=ctx.options.get("bc_type"),
        output_path=combined_path)
    print(f"[saved] {combined_path.name}")

    full_mesh.write_vtk(output_name="artery_stent_mesh", output_directory=str(run_dir))

    # --- the runnable input: solver, BCs, and the radial "balloon" force -------------
    out_path = run_dir / "simulation.4C.yaml"
    run = InputFile()
    set_header_static(run, n_steps=settings.n_steps, total_time=settings.total_time)
    set_runtime_output(run)
    set_beam_to_solid_meshtying(run, bme.bc.beam_to_solid_surface_meshtying,
                                contact_discretization="mortar", mortar_shape="line2")

    inlet, outlet = settings.inlet_surface_index, settings.outlet_surface_index
    surfaces = full_mesh.geometry_sets.get(bme.geo.surface, [])
    if len(surfaces) <= max(inlet, outlet):
        raise ValueError("imported solid is missing the inlet/outlet surface sets")
    for index in (inlet, outlet):
        full_mesh.add(BoundaryCondition(
            surfaces[index],
            {"NUMDOF": 3, "ONOFF": [1, 1, 1], "VAL": [0, 0, 0], "FUNCT": [0, 0, 0]},
            bc_type=bme.bc.dirichlet))

    ramp = Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}])
    full_mesh.add(ramp)

    nodes = bm.centreline_nodes(beam_mesh)
    coords = np.array([node.coordinates for node in nodes])
    radial = radial_directions(coords, artery.centreline)

    if settings.fix_stent_node:      # remove stent rigid-body translation
        full_mesh.add(BoundaryCondition(
            GeometrySet([nodes[0]]),
            {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
             "VAL": [0] * 9, "FUNCT": [0] * 9},
            bc_type=bme.bc.dirichlet))

    for node, direction in zip(nodes, radial):
        f = force * direction
        full_mesh.add(BoundaryCondition(
            GeometrySet([node]),
            {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
             "VAL": [f[0], f[1], f[2], 0, 0, 0, 0, 0, 0],
             "FUNCT": [ramp, ramp, ramp, 0, 0, 0, 0, 0, 0]},
            bc_type=bme.bc.neumann))

    run.add(full_mesh)
    run.dump(str(out_path), validate=True, add_footer_application_script=False)

    print(f"[sim] static smoke test: {settings.n_steps} steps, radial expansion force "
          f"{force:g} N "
          f"ramped over {len(nodes):,} stent nodes")
    print(f"[sim] BCs: artery inlet+outlet fixed; coupling = beam-to-solid meshtying (tied)")
    print(f"[saved] {out_path.name}   (schema-validated)")

    record = base_record("stent_artery", run_dir)
    record |= {
        # what solve() needs to find this run again, without rebuilding it
        "input": {"file": out_path.name, "output_base": "out_simulation"},
        "stent": {"name": ctx.stent_name, "source": str(ctx.stent_dir),
                  "strut_thickness_mm": round(strut, 5)},
        "beam_model": {"element": settings.beam_class, "material_law": settings.material,
                       "target_element_length_mm": round(l_el, 5),
                       "n_elements": len(beam_mesh.elements),
                       "section": section},
        "settings": settings_to_dict(settings),
        "artery": {"type": artery.artery_type, "radius_mm": round(artery.radius, 4),
                   "length_mm": round(artery.length, 4),
                   "wall_thickness_mm": artery.wall_thickness,
                   "youngs_modulus_MPa": artery.artery_youngs,
                   "material": "StVenantKirchhoff (placeholder)"},
        "loading": {"mode": "radial point force (placeholder balloon)",
                    "expansion_force_N": force, "profile": "ramp"},
        "coupling_constraints": {**limits, **report},
        "coupling_method": "beam-to-solid surface meshtying (tied), mortar line2",
        "results": {"note": "filled in after the solve"},
    }

    return [built_case(name="stent_artery", input_path=out_path, run_dir=run_dir,
                       output_base="out_simulation", record=record, coupling=report)]


register("stent_artery", build)

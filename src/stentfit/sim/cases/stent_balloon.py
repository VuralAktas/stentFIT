"""
The stent opened by contact with an expanding balloon.

The stent is opened by contact rather than by prescribing where its nodes must go, and that is the
reason this case exists. ``stent_only`` prescribes every beam node, which is fine for an elastic
material but over-constrained for a plastic one, because once a crown reaches its yield moment
there is nowhere to redistribute the load.

Datz et al. open the stent through contact with a pressure-inflated balloon and prescribe only a
few fixation points. This reproduces that arrangement in the simplest form that keeps the essential
property, so the stent is loaded only by contact forces and is restrained by the constraints needed
to remove its rigid-body modes and nothing more.

The balloon is a :class:`~stentfit.balloon.Balloon`, built and passed in. This module owns the
contact and the stent's restraint, and nothing else.
"""

import numpy as np
from scipy.spatial import cKDTree

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.geometry_set import GeometrySet
from beamme.four_c.beam_interaction_conditions import add_beam_interaction_condition
from beamme.four_c.header_functions import (set_binning_strategy_section,
                                            set_header_static, set_runtime_output)

from ...core.artery_geom import import_artery_solid
from .. import beam_model as bm
from ..coupling import (check_coupling, coupling_limits, print_coupling,
                        raise_if_coupling_failed)
from ..materials import describe
from ..record import base_record
from ..settings import settings_to_dict
from . import BuildContext, built_case, register

def restrain_stent(mesh, nodes: list, junction_coords) -> list:
    """
    Remove the stent's rigid-body modes, and nothing else.

    Loaded only by contact against a smooth cylinder, the stent is free to slide along the axis,
    drift off-centre and spin about the axis. Those four rigid-body modes make the stiffness matrix
    singular.

    Every constraint has to act tangentially rather than radially, because a radial component would
    fight the expansion the contact is driving. 4C's Dirichlet conditions are along the global axes,
    so each node is chosen where the axis being fixed points tangentially. ``x`` is tangential at 90
    and 270 degrees, and ``y`` is tangential at 0 and 180.

    ===============  =========  =================================================
    crown            fixed      removes
    ===============  =========  =================================================
    nearest 90 deg   ``x``      x-translation, and half of the rotation
    nearest 270 deg  ``x``      the other half of the rotation
    nearest 0/180    ``y``      y-translation
    the first one    ``z``      axial sliding
    ===============  =========  =================================================

    Using the 90/270 pair for ``x`` matters. Those two crowns usually sit exactly on the axis, so
    both constraints are tangential and leak nothing radial. Only the ``y`` constraint has to settle
    for whatever crown is nearest 0 or 180 degrees, which for an odd number of crowns per ring can
    be some way off. A constraint that leaks resists the expansion, and enough of it will stop the
    solve shortly after contact engages.

    The reaction forces are the check, and they should stay small next to the contact forces.

    :param mesh: The combined mesh, modified in place.
    :param nodes: The stent's centreline nodes.
    :param junction_coords: The labelled crown positions.
    :returns: One dict per restrained crown, with ``crown_deg``, ``fixed`` and ``radial_leak``.
    """
    points = np.array([node.coordinates for node in nodes], float)
    z_mid = 0.5 * (points[:, 2].min() + points[:, 2].max())
    span = points[:, 2].max() - points[:, 2].min()

    tree = cKDTree(points)
    coords = junction_coords
    angles = np.arctan2(coords[:, 1], coords[:, 0])

    def best_crown(*wanted):
        """
        Crown closest to any of the wanted angles, with mid-length only breaking ties.

        Angular accuracy dominates because every degree of error leaks a radial component into
        what is meant to be a purely tangential constraint.
        """
        distance = np.min([np.abs(np.arctan2(np.sin(angles - a), np.cos(angles - a)))
                           for a in wanted], axis=0)
        index = int(np.argmin(distance + 0.01 * np.abs(coords[:, 2] - z_mid) / span))
        return coords[index], float(distance[index])

    half = np.pi / 2
    plan = [(*best_crown(half), "x"),                # x is tangential at 90 deg
            (*best_crown(-half), "x"),               # ... and at 270; the pair kills rotation
            (*best_crown(0.0, np.pi), "y")]          # y is tangential at 0 and 180 deg

    detail = []
    for i, (coord, error, axis) in enumerate(plan):
        node = nodes[tree.query(coord)[1]]
        onoff = [0, 0, 0]
        onoff["xyz".index(axis)] = 1
        if i == 0:
            onoff[2] = 1                             # z on the first crown only: axial sliding
        mesh.add(BoundaryCondition(
            GeometrySet([node]),
            {"NUMDOF": 9, "ONOFF": onoff + [0] * 6, "VAL": [0] * 9, "FUNCT": [0] * 9},
            bc_type=bme.bc.dirichlet))
        degrees = np.degrees(np.arctan2(node.coordinates[1], node.coordinates[0])) % 360
        detail.append({"crown_deg": f"{degrees:.1f} deg",
                       "fixed": "".join("xyz"[j] for j, v in enumerate(onoff) if v),
                       "radial_leak": f"{abs(np.sin(error)) * 100:.1f}% radial"})

    return detail


def add_contact(inp, mesh, beam_mesh, balloon_surface, settings, penalty_g0: float) -> None:
    """
    Couple the stent beams to the balloon's outer surface with penalty contact.

    BeamMe's ``set_beam_to_solid_meshtying`` handles only the meshtying variants and raises for
    contact, so the contact section is written directly. ``inp.dump(validate=True)`` checks it
    against 4C's own schema, so a wrong key or value fails at build time rather than being
    ignored.

    :param inp: The 4C input file.
    :param mesh: The combined mesh, modified in place.
    :param beam_mesh: The stent beam mesh, for the beam side of the pair.
    :param balloon_surface: The balloon's outer surface geometry set.
    :param settings: The simulation's settings, for the contact fields.
    :param penalty_g0: Penetration over which the contact stiffness ramps in, in mm.
    """
    add_beam_interaction_condition(mesh, GeometrySet(beam_mesh.elements), balloon_surface,
                                   bme.bc.beam_to_solid_surface_contact)

    inp.add({"BEAM INTERACTION": {"REPARTITIONSTRATEGY": "everydt"}})
    set_binning_strategy_section(inp)
    inp.add({"BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT":
             settings.contact_section(penalty_g0)})


def build(ctx: BuildContext) -> list:
    """
    Build the stent-balloon contact input.

    :param ctx: The build context. Everything comes from
        :class:`~stentfit.sim.settings.StentBalloonSettings`; ``options["balloon"]`` carries the
        :class:`~stentfit.balloon.Balloon` that ``Simulation`` built from it.
    :returns: A single entry, in its own numbered run folder.
    :raises ValueError: If no balloon was given, if the coupling rules are violated, or if the
        balloon overlaps the stent.
    """
    from ..record import next_run_dir

    balloon = ctx.options.get("balloon")
    if balloon is None:
        raise ValueError("stent_balloon needs a Balloon - Simulation builds one from the "
                         "settings, so this means the case was called directly")

    settings = ctx.settings
    target_strain = settings.radial_strain

    strut = ctx.strut_thickness()
    limits = coupling_limits(strut, settings)
    l_el = limits["l_beam"]

    print(f"stent {ctx.stent_name}: diameter {ctx.features['diameter']:.3f} mm, "
          f"length {ctx.features['length']:.3f} mm, strut {strut:.4f} mm")
    print(f"  sizing: solid {limits['h_solid']:.4f} mm "
          f"(x{limits['factor_solid']:g} beam diameter), "
          f"beam {l_el:.4f} mm (x{limits['factor_beam']:g} solid)")

    # --- stent ---------------------------------------------------------------------
    beam_mesh, section = bm.build_stent_beams(ctx.stent_dir, settings,
                                              l_el=l_el, strut_thickness=strut)
    for line in describe(section, settings):
        print(f"  {line}")

    nodes = bm.centreline_nodes(beam_mesh)
    junction_coords, junction_degrees = bm.read_junctions(ctx.stent_dir)
    welded = bm.couple_junctions(beam_mesh, nodes, junction_coords, junction_degrees)
    inner_face = bm.innermost_surface_radius(nodes, section)

    # --- balloon -------------------------------------------------------------------
    run_dir = next_run_dir(ctx.output_dir)
    print(f"  run folder {run_dir.name}")

    balloon.mesh_solid(run_dir / "balloon.4C.yaml",
                       stent_inner_face=inner_face, limits=limits)
    print()
    for line in balloon.summary():
        print(line)

    strain = balloon.expansion_strain(target_strain)
    target_r = float(ctx.features["r_inner"]) * (1 + target_strain)
    print(f"  target outer radius {target_r:.4f} mm -- an outcome here, not imposed.")

    # --- coupling constraints ------------------------------------------------------
    # Checked before the input is written, against the elements that were actually created rather
    # than the target length. A violation is not a solver failure: the run would complete and hand
    # back a mesh-dependent answer, so nothing downstream would catch it.
    lengths = bm.beam_element_lengths(beam_mesh)
    print(f"\nbeam elements: {len(lengths):,}, length {lengths.min():.4f} .. "
          f"{lengths.max():.4f} mm (mean {lengths.mean():.4f}, target {l_el:.4f})")

    report = check_coupling(lengths,
                            {"circumferential": balloon.info["h_circ"],
                             "axial": balloon.info["h_axial"]},
                            d_beam=limits["d_beam"], e_beam=settings.youngs,
                            e_solid=balloon.effective_youngs)
    print("mixed-dimensional coupling (Steinbrecher et al.):")
    print_coupling(report)
    raise_if_coupling_failed(report, limits)

    # The gap must be positive before solving: a balloon that already overlaps the stent starts
    # with contact penetrating, which fails for reasons unrelated to the solver.
    clearance = inner_face - balloon.r_outer
    print(f"\nsurface-to-surface clearance: {clearance:+.4f} mm")
    if clearance <= 0:
        raise ValueError("balloon overlaps the stent - raise clearance_frac")

    # --- assemble ------------------------------------------------------------------
    inp, solid = import_artery_solid(balloon.solid_yaml)
    surfaces = solid.geometry_sets.get(bme.geo.surface, [])
    if len(surfaces) != 4:
        raise ValueError(f"balloon solid has {len(surfaces)} surface sets, expected 4")
    inner_surface, outer_surface = surfaces[0], surfaces[1]
    end_surfaces = (surfaces[2], surfaces[3])

    solid.add(beam_mesh)

    # The contact regularisation width is a physical length, set by the stent's own strut
    # diameter, so changing n_steps or the load profile does not change how contact behaves.
    # The advance is still measured, but only for the run record.
    advance = balloon.advance_per_step(target_strain, settings.n_steps)
    penalty_g0 = settings.penalty_g0_per_strut * strut

    set_header_static(inp, n_steps=settings.n_steps, total_time=settings.total_time,
                      max_iter=settings.max_iter,
                      tol_residuum=settings.tol_residuum,
                      tol_increment=settings.tol_increment,
                      predictor=settings.predictor)
    set_runtime_output(inp, output_solid=True, btsvmt_output=False, btss_output=False,
                       absolute_beam_positions=True, output_strains=True)
    beams = inp.fourc_input["IO/RUNTIME VTK OUTPUT/BEAMS"]
    beams["MATERIAL_FORCES_GAUSSPOINT"] = True
    inp.fourc_input["IO/RUNTIME VTK OUTPUT/BEAMS"] = beams

    balloon.add_pressure(solid, inner_surface)
    balloon.add_end_springs(solid, end_surfaces)
    # A follower load depends on the displacement, so it belongs in the stiffness matrix.
    # 4C aborts on "orthopressure" without this rather than dropping the term silently.
    dynamic = inp.fourc_input["STRUCTURAL DYNAMIC"]
    dynamic["LOADLIN"] = True
    inp.fourc_input["STRUCTURAL DYNAMIC"] = dynamic

    restraint = restrain_stent(solid, nodes, junction_coords)
    print(f"stent restrained at {len(restraint)} crowns (tangential directions only):")
    for crown in restraint:
        print(f"    {crown['crown_deg']:>10s}  fix {crown['fixed']:2s}   "
              f"{crown['radial_leak']}")

    add_contact(inp, solid, beam_mesh, outer_surface, settings, penalty_g0)
    print(f"\ncontact: {settings.discretization}, {settings.penalty_law}")
    print(f"  penalty  {settings.penalty:g} N/mm^2   (Datz et al.)")
    print(f"  regularise over g0 = {penalty_g0:.5f} mm "
          f"({settings.penalty_g0_per_strut:g} x strut diameter)")
    if settings.discretization == "mortar":
        print(f"  multipliers {settings.mortar_shape_function} on "
              f"{settings.mortar_contact_defined_in}, regularised by the penalty "
              f"strategy")
    else:
        # The one number that says how over-constrained GPTS is here: three constraints per Gauss
        # point, against a solid element of nearly the same size as the beam element.
        h = max(balloon.info["h_circ"], balloon.info["h_axial"])
        print(f"  {3 * settings.gauss_points} constraints per beam element "
              f"(h_beam/h_solid {lengths.min() / h:.2f}-{lengths.max() / h:.2f}; "
              f"GPTS locking grows as this approaches 1)")

    print(f"\nsolver: {settings.n_steps} steps, max {settings.max_iter} iterations, "
          f"predictor {settings.predictor}")

    inp.add(solid)
    out_path = run_dir / "stent_balloon.4C.yaml"
    inp.dump(str(out_path), validate=True, add_footer_application_script=False)
    print(f"\n[saved] {out_path.name}   (schema-validated)")

    record = base_record("stent_balloon", run_dir)
    record |= {
        # what solve() needs to find this run again, without rebuilding it
        "input": {"file": out_path.name, "output_base": "out_stent_balloon"},
        "stent": {"name": ctx.stent_name, "source": str(ctx.stent_dir),
                  "diameter_mm": round(float(ctx.features["diameter"]), 4),
                  "length_mm": round(float(ctx.features["length"]), 4),
                  "strut_thickness_mm": round(strut, 5),
                  "junctions": welded},
        "beam_model": {
            "element": settings.beam_class, "material_law": settings.material,
            "target_element_length_mm": round(l_el, 5),
            "actual_element_length_mm": {"min": round(float(lengths.min()), 5),
                                         "mean": round(float(lengths.mean()), 5),
                                         "max": round(float(lengths.max()), 5)},
            "n_elements": len(beam_mesh.elements), "section": section},
        "settings": settings_to_dict(settings),
        "balloon": balloon.to_dict(),
        "loading": {"profile": balloon.load_profile,
                    "scale_of_t": balloon.load_expression,
                    "target_stent_radial_strain": target_strain,
                    "target_r_outer_mm": round(target_r, 4),
                    "balloon_target_strain": round(strain, 5),
                    "travel_mm": round(balloon.travel(target_strain), 5),
                    "peak_rate_vs_ramp": balloon.peak_rate,
                    "advance_per_step_mm": round(advance, 5)},
        "coupling_constraints": {**limits, **report},
        "contact": {"penalty_parameter_g0_mm": round(penalty_g0, 5),
                    "penalty_g0_per_strut": settings.penalty_g0_per_strut},
        "stent_restraint": restraint,
        "initial_clearance_mm": round(clearance, 5),
        "results": {"note": "filled in after the solve"},
    }

    return [built_case(name=run_dir.name, input_path=out_path, run_dir=run_dir,
                       output_base="out_stent_balloon", record=record, coupling=report)]


register("stent_balloon", build)

"""
Stent-only load cases: radial expansion and axial stretch.

There is no artery and no balloon. The beam mesh is driven by prescribed displacements, and
beam-to-beam contact stops the struts passing through each other.

The radial case drives every centreline node outwards by a uniform radial scaling, written as a 4C
symbolic space-time function, so one Dirichlet condition covers the whole mesh. The axial direction
is left free, so the foreshortening comes out as a result rather than being imposed. That leaves
translation along z as the only rigid-body mode, and it is pinned at mid-length.

The axial case clamps a band at each end instead, standing in for a test machine's grips, and moves
the top band along z. Both bands are held laterally, which is what rigid grips do, and the middle
is left free, so the stent necks in through the gauge length as a test specimen does.

Elements near a welded crown are kept out of the contact set, because the crowns are coincident but
separate nodes that 4C would otherwise read as touching from the first step.

Prescribing every node is fine for an elastic material, which supplies whatever bending moment the
imposed shape demands. It is over-constrained for a plastic one, because once a crown reaches its
yield moment there is nowhere to redistribute the load. The plastic path is still built, so the
runs can be kept and compared, but it is not expected to converge.
"""

import numpy as np
from scipy.spatial import cKDTree

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySet
from beamme.four_c.beam_interaction_conditions import add_beam_interaction_condition
from beamme.four_c.header_functions import (set_beam_contact_runtime_output,
                                            set_beam_contact_section, set_header_static,
                                            set_runtime_output)
from beamme.four_c.input_file import InputFile

from .. import beam_model as bm
from ..materials import describe
from ..record import base_record
from ..settings import element_length, settings_to_dict
from . import BuildContext, built_case, register

#: The two load cases. Both strains are engineering strains at ``t = 1``, signed and relative to
#: the undeformed stent, so they mean the same thing on any stent. Positive is outward or longer.
CASES = {
    "radial_expand": {"kind": "radial", "strain": +0.50},    # diameter x1.5
    "axial_stretch": {"kind": "axial", "strain": +0.10},
}

#: How much of the tip ring the axial cases clamp at each end. It is a fraction of the tip ring's
#: own length, taken from the stent's ring boundaries, so it scales with the crown spacing rather
#: than with the total length. A whole ring would be far too stiff, and a single node would be a
#: point load rather than a clamp.
TIP_RING_GRIP_FRAC = 0.2


def case_folder_name(name: str, material: str) -> str:
    """
    Folder name for a case, tagged when the material is not the elastic one.

    Elastic and elastoplastic runs of the same case have to coexist so they can be compared.

    :param name: A key of :data:`CASES`.
    :param material: The strut material law.
    :returns: e.g. ``radial_expand`` or ``radial_expand__plastic``.
    """
    return name if material != "elastoplastic" else f"{name}__plastic"


def add_radial_case(mesh, nodes: list, strain: float, axis: dict) -> dict:
    """
    Prescribe a uniform radial scaling, leaving axial displacement free.

    The radial displacement of a node at radius r is ``r * strain``, so the strain is exactly the
    coefficient in the displacement functions and there is nothing to convert.

    :param mesh: The beam mesh, modified in place.
    :param nodes: The centreline nodes to drive.
    :param strain: Imposed diameter change at ``t = 1``, as a fraction of the original diameter.
    :param axis: Where the stent's axis sits, from :func:`~stentfit.sim.beam_model.axis_report`.
    :returns: Dict describing what was applied, for the run record.
    """
    # 4C prescribes VAL * FUNCT(x, y, z, t) with the space part evaluated at the reference
    # position, so VAL = 1 and the function carries the whole displacement.
    fx = Function([{"SYMBOLIC_FUNCTION_OF_SPACE_TIME":
                    f"(x-({axis['cx']:.10g}))*({strain:.10g})*t"}])
    fy = Function([{"SYMBOLIC_FUNCTION_OF_SPACE_TIME":
                    f"(y-({axis['cy']:.10g}))*({strain:.10g})*t"}])
    mesh.add(fx)
    mesh.add(fy)

    mesh.add(BoundaryCondition(
        GeometrySet(nodes),
        {"NUMDOF": 9,
         "ONOFF": [1, 1, 0, 0, 0, 0, 0, 0, 0],      # z free -> foreshortening is a result
         "VAL": [1.0, 1.0, 0, 0, 0, 0, 0, 0, 0],
         "FUNCT": [fx, fy, 0, 0, 0, 0, 0, 0, 0]},
        bc_type=bme.bc.dirichlet))

    # Prescribing x and y everywhere leaves exactly one rigid-body mode: translation along z.
    # Pin it at mid-length, where it perturbs the deformation least.
    zs = np.array([node.coordinates[2] for node in nodes])
    mid = nodes[int(np.argmin(np.abs(zs - zs.mean())))]
    mesh.add(BoundaryCondition(
        GeometrySet([mid]),
        {"NUMDOF": 9, "ONOFF": [0, 0, 1, 0, 0, 0, 0, 0, 0],
         "VAL": [0] * 9, "FUNCT": [0] * 9},
        bc_type=bme.bc.dirichlet))

    return {"radial_strain": strain,
            "n_driven_nodes": len(nodes),
            "pinned_node_z_mm": float(mid.coordinates[2])}


def add_axial_case(mesh, nodes: list, strain: float, axis: dict, ring_boundaries,
                   grip_frac: float = TIP_RING_GRIP_FRAC) -> dict:
    """
    Hold both end rings on the axis and move the top one along z.

    The stent still necks in under tension or bulges out under compression through the free middle,
    exactly as a test specimen does between its grips.

    :param mesh: The beam mesh, modified in place.
    :param nodes: The centreline nodes to select the end bands from.
    :param strain: Imposed axial displacement as a fraction of stent length.
    :param axis: Where the stent's axis sits, from :func:`~stentfit.sim.beam_model.axis_report`.
    :param ring_boundaries: Ring z edges from ``stent_features.json``, used to size the grip
        against the tip ring rather than against the whole stent.
    :param grip_frac: Grip band as a fraction of the tip ring's own length. See
        :data:`TIP_RING_GRIP_FRAC`.
    :returns: Dict describing what was applied, for the run record.
    :raises ValueError: If either end band caught no nodes.
    """
    rb = np.asarray(ring_boundaries, float)
    grip_bottom = grip_frac * (rb[1] - rb[0])
    grip_top = grip_frac * (rb[-1] - rb[-2])

    bottom = [n for n in nodes if n.coordinates[2] < axis["z_min"] + grip_bottom]
    top = [n for n in nodes if n.coordinates[2] > axis["z_max"] - grip_top]
    if not bottom or not top:
        raise ValueError(
            f"grips of {grip_bottom:.4f} / {grip_top:.4f} mm caught {len(bottom)} bottom / "
            f"{len(top)} top nodes - raise grip_frac")

    # Declared as space-time even though it only depends on t: 4C looks up Dirichlet functions as
    # FunctionOfSpaceTime, and a SYMBOLIC_FUNCTION_OF_TIME registers as the incompatible
    # FunctionOfTime, which aborts at setup. (Neumann conditions accept the time-only form, which
    # is why the artery case can use it.)
    ramp = Function([{"SYMBOLIC_FUNCTION_OF_SPACE_TIME": "t"}])
    mesh.add(ramp)

    mesh.add(BoundaryCondition(                       # bottom: fully fixed in translation
        GeometrySet(bottom),
        {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
         "VAL": [0] * 9, "FUNCT": [0] * 9},
        bc_type=bme.bc.dirichlet))

    mesh.add(BoundaryCondition(                       # top: moved along z, held on the axis
        GeometrySet(top),
        {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
         "VAL": [0, 0, strain * axis["length"], 0, 0, 0, 0, 0, 0],
         "FUNCT": [0, 0, ramp, 0, 0, 0, 0, 0, 0]},
        bc_type=bme.bc.dirichlet))

    return {"axial_strain": strain, "grip_frac": grip_frac,
            "n_bottom_nodes": len(bottom), "n_top_nodes": len(top),
            "grip_bottom_mm": float(grip_bottom), "grip_top_mm": float(grip_top),
            "imposed_displacement_mm": float(strain * axis["length"])}


def contact_elements(mesh, junction_coords, exclude_radius: float) -> list:
    """
    The elements that take part in self-contact.

    :param mesh: The beam mesh.
    :param junction_coords: The welded crowns, from :func:`~stentfit.sim.beam_model.read_junctions`.
    :param exclude_radius: Elements with a node closer than this to a crown are left out.
    :returns: The elements to put in the contact set.
    """
    tree = cKDTree(np.asarray(junction_coords, float))

    keep = []
    for element in mesh.elements:
        coords = [node.coordinates for node in element.nodes]
        if tree.query(coords)[0].min() > exclude_radius:
            keep.append(element)
    return keep


#: Elements this many steps apart along the mesh sit within an element length of each other simply
#: by being on the same strut. That is not a clearance, so :func:`reference_clearance` skips them.
NEIGHBOUR_DEPTH = 2


def neighbours(mesh, depth: int = NEIGHBOUR_DEPTH) -> dict:
    """
    Which elements are within ``depth`` steps of each other along the mesh.

    Taken from the whole mesh, not from the contact set: two elements on one strut stay neighbours
    whether or not the element between them is in the set.

    :param mesh: The beam mesh.
    :param depth: How many steps out to walk.
    :returns: Element index -> the set of element indices within ``depth`` steps, itself included.
    """
    elements = list(mesh.elements)

    by_node = {}
    for index, element in enumerate(elements):
        for node in element.nodes:
            by_node.setdefault(id(node), []).append(index)

    touching = {index: {index} for index in range(len(elements))}
    for shared in by_node.values():
        for index in shared:
            touching[index].update(shared)

    near = {}
    for index in range(len(elements)):
        seen = {index}
        front = {index}
        for _ in range(depth):
            front = {j for i in front for j in touching[i]} - seen
            seen |= front
        near[index] = seen
    return near


def reference_clearance(mesh, elements: list, strut: float) -> float:
    """
    How close the contact set already is to touching, before any load.

    Measured node to node rather than element to element, which is close enough with elements about
    one strut thickness long. Below 1.0 means 4C sees contact at ``t = 0`` and the first step fails.

    :param mesh: The beam mesh, for the element topology.
    :param elements: The contact set, from :func:`contact_elements`.
    :param strut: Strut thickness, in mm.
    :returns: Smallest gap between two elements that are not neighbours, in strut thicknesses.
        ``inf`` if nothing is anywhere near.
    """
    near = neighbours(mesh)
    index_of = {id(element): index for index, element in enumerate(mesh.elements)}

    coords, owner = [], []
    for element in elements:
        for node in element.nodes:
            coords.append(node.coordinates)
            owner.append(index_of[id(element)])
    coords, owner = np.array(coords, float), np.array(owner)

    best = np.inf
    for a, b in cKDTree(coords).query_pairs(3.0 * strut):
        if owner[b] not in near[owner[a]]:
            best = min(best, float(np.linalg.norm(coords[a] - coords[b])))
    return best / strut


def add_self_contact(mesh, elements: list) -> None:
    """
    Pair the contact set with itself, so the struts collide with each other.

    :param mesh: The beam mesh, modified in place.
    :param elements: The contact set, from :func:`contact_elements`.
    """
    add_beam_interaction_condition(mesh, GeometrySet(elements), GeometrySet(elements),
                                   bme.bc.beam_to_beam_contact)


def binning_box(nodes: list, strain: float, margin: float) -> list:
    """
    Search box for the contact pairs, grown to cover the deformed stent.

    :param nodes: The centreline nodes.
    :param strain: The case's strain, so an expanding stent stays inside the box.
    :param margin: Extra room on every side, in mm.
    :returns: ``[x0, y0, z0, x1, y1, z1]``.
    """
    points = np.array([node.coordinates for node in nodes], float)
    low, high = points.min(axis=0), points.max(axis=0)
    grow = (high - low) * max(strain, 0.0) + margin
    return [float(v) for v in np.concatenate([low - grow, high + grow])]


def contact_section(settings, section: dict, l_el: float, strut: float,
                    box: list, cutoff: float) -> dict:
    """
    Keyword arguments for 4C's beam-to-beam contact section.

    :param settings: The simulation's settings, for the contact fields.
    :param section: Strut section properties, for ``EA``.
    :param l_el: Target beam element length, in mm.
    :param strut: Strut thickness, in mm.
    :param box: The binning box, from :func:`binning_box`.
    :param cutoff: Smallest allowed bin size, in mm.
    :returns: Arguments for BeamMe's ``set_beam_contact_section``.
    """
    return {
        "btb_penalty": settings.contact_penalty_frac * section["EA"] / l_el,
        "btb_line_penalty": 0.0,
        "penalty_law": "LinPosQuadPen",
        "penalty_regularization_g0": settings.contact_g0_per_strut * strut,
        # A tiny, identical window for both keeps point contact active above ~1 degree - as
        # close to "every angle" as 4C allows - while trivially satisfying its requirement that
        # the large-angle and small-angle windows overlap. BeamMe defaults these to 70/80, which
        # would switch angle scaling back on; 4C also rejects the schema's own -1 "unset" value
        # and a zero-width [0, 0] window at runtime.
        "per_shift_angle": [0.0, 1.0],
        "par_shift_angle": [0.0, 1.0],
        # Coarser than BeamMe's default of 12 degrees. The near-parallel tip folds are the one
        # geometry this stent has that a fine segmentation cannot project onto reliably, and 4C's
        # own error message on that pair pointed at this value.
        "b_seg_angle": 30.0,
        "binning_parameters": {"binning_bounding_box": box,
                               "binning_cutoff_radius": cutoff},
    }


def write_input(mesh, settings, out_path, contact: dict = None) -> None:
    """
    Write the solver header, the output settings and the mesh, schema-validated.

    :param mesh: The beam mesh.
    :param settings: The simulation's settings, for the solver fields.
    :param out_path: Where to write.
    :param contact: Contact arguments from :func:`contact_section`, or ``None`` for no contact.
    """
    inp = InputFile()
    set_header_static(inp, n_steps=settings.n_steps, total_time=settings.total_time,
                      max_iter=settings.max_iter, tol_residuum=settings.tol_residuum,
                      predictor=settings.predictor)
    set_runtime_output(inp, output_solid=False, btsvmt_output=False, btss_output=False,
                       absolute_beam_positions=True, output_strains=True)

    # Also write the internal forces and moments at every Gauss point. For an elastic material
    # these are just E*A*strain and E*I*curvature, but they stop being simple multiples once
    # plasticity is on, and they are what the yield limits are compared against.
    #
    # This cannot go through InputFile.add(): that calls combine_sections(), which raises if a
    # section already exists, and set_runtime_output has just written this one.
    beams = inp.fourc_input["IO/RUNTIME VTK OUTPUT/BEAMS"]
    beams["MATERIAL_FORCES_GAUSSPOINT"] = True
    inp.fourc_input["IO/RUNTIME VTK OUTPUT/BEAMS"] = beams

    # set_header_static writes no line search section, so this one is new and can go through add().
    if settings.line_search != "Full Step":
        inp.add({"STRUCT NOX/Line Search": {"Method": settings.line_search}})

    if contact is not None:
        # This also writes the binning and beam interaction sections.
        set_beam_contact_section(inp, **contact)
        set_beam_contact_runtime_output(inp)     # gaps and contact forces, to check in ParaView

    inp.add(mesh)
    inp.dump(str(out_path), validate=True, add_footer_application_script=False)


def build(ctx: BuildContext) -> list:
    """
    Build one 4C input per requested load case.

    A fresh mesh is built per case: boundary conditions mutate the mesh in place, so reusing one
    mesh would stack every case's conditions on top of each other.

    :param ctx: The build context. Everything comes from
        :class:`~stentfit.sim.settings.StentOnlySettings`.
    :returns: One entry per case written, as described in :mod:`stentfit.sim.cases`.
    :raises ValueError: If an unknown case or strain name was requested.
    """
    settings = ctx.settings

    wanted = list(settings.cases) or list(CASES)
    unknown = [name for name in wanted if name not in CASES]
    if unknown:
        raise ValueError(f"unknown case(s) {unknown}, expected from {sorted(CASES)}")

    strains = settings.strains or {}
    unknown = [name for name in strains if name not in CASES]
    if unknown:
        raise ValueError(f"unknown case(s) in strains {unknown}, "
                         f"expected from {sorted(CASES)}")
    grip_frac = settings.grip_frac

    strut = ctx.strut_thickness()
    l_el = element_length(settings, strut)
    junction_coords, junction_degrees = bm.read_junctions(ctx.stent_dir)

    print(f"stent {ctx.stent_name}: length {ctx.features['length']:.3f} mm, "
          f"diameter {ctx.features['diameter']:.3f} mm, strut {strut:.4f} mm")
    print(f"  element length {l_el:.4f} mm ({settings.l_el_per_strut:g} x strut thickness)")
    print(f"  junctions      {len(junction_coords)} "
          f"(degrees {bm.degree_counts(junction_degrees)})")
    print(f"  solver         {settings.n_steps} steps, predictor {settings.predictor}, "
          f"line search {settings.line_search}")

    built = []
    for name in wanted:
        spec = CASES[name]
        strain = strains.get(name, spec["strain"])
        folder = case_folder_name(name, settings.material)
        print(f"\n{'=' * 78}\n=== {folder}  ({spec['kind']}, strain {strain:+.2f}"
              + f", l_el {l_el:.4f} mm)\n{'=' * 78}")

        mesh, section = bm.build_stent_beams(ctx.stent_dir, settings,
                                             l_el=l_el, strut_thickness=strut)
        for line in describe(section, settings):
            print(f"  {line}")

        nodes = bm.centreline_nodes(mesh)
        welded = bm.couple_junctions(mesh, nodes, junction_coords, junction_degrees)
        axis = bm.axis_report(nodes)
        print(f"  junctions welded {welded['n_junctions']}, worst node gap "
              f"{welded['max_gap']:.3e} mm")
        print(f"  axis offset      x={axis['cx']:+.5f} y={axis['cy']:+.5f} mm (want ~0)")
        print(f"  z extent         {axis['z_min']:+.4f} .. {axis['z_max']:+.4f} mm "
              f"(length {axis['length']:.4f})")

        if spec["kind"] == "radial":
            info = add_radial_case(mesh, nodes, strain, axis)
            print(f"  radial strain    {strain * 100:+.0f}% -> diameter "
                  f"{axis['diameter']:.4f} -> "
                  f"{axis['diameter'] * (1 + strain):.4f} mm")
        else:
            info = add_axial_case(mesh, nodes, strain, axis,
                                  ctx.features["ring_boundaries"], grip_frac)
            print(f"  axial strain     {strain * 100:+.0f}% -> "
                  f"{info['imposed_displacement_mm']:+.4f} mm "
                  f"(grips {info['grip_bottom_mm']:.4f} / {info['grip_top_mm']:.4f} mm, "
                  f"{info['n_bottom_nodes']} / {info['n_top_nodes']} nodes)")

        lengths = bm.beam_element_lengths(mesh)

        contact = None
        contact_info = {"enabled": False}
        if settings.self_contact:
            exclude = settings.contact_exclusion_per_strut * strut
            elements = contact_elements(mesh, junction_coords, exclude)
            add_self_contact(mesh, elements)

            clearance = reference_clearance(mesh, elements, strut)
            box = binning_box(nodes, strain, 2.0 * strut)
            contact = contact_section(settings, section, l_el, strut,
                                      box, float(lengths.max()) + 2.0 * strut)

            print(f"  self-contact     {len(elements)} of {len(mesh.elements)} elements "
                  f"(crowns excluded within {exclude:.4f} mm)")
            print(f"                   penalty {contact['btb_penalty']:.4g} N/mm, "
                  f"g0 {contact['penalty_regularization_g0']:.5f} mm")
            # 4C measures the gap surface to surface, so contact is already live at t = 0 once
            # the centrelines are within one strut thickness plus the regularisation width.
            quiet = 1.0 + settings.contact_g0_per_strut
            note = ""
            if clearance < 1.0:
                note = "   <-- struts already overlap, step 1 will fail"
            elif clearance < quiet:
                note = f"   <-- under {quiet:.2f}, contact is already live at t = 0"
            print(f"                   worst clearance {clearance:.2f} strut thicknesses{note}")

            contact_info = {"enabled": True,
                            "n_elements": len(elements),
                            "n_elements_total": len(mesh.elements),
                            "exclusion_mm": round(exclude, 5),
                            "penalty_N_per_mm": round(contact["btb_penalty"], 4),
                            "penalty_g0_mm": round(contact["penalty_regularization_g0"], 5),
                            "reference_clearance_per_strut": round(float(clearance), 3)}

        run_dir = ctx.output_dir / folder
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / f"stent_{folder}.4C.yaml"
        write_input(mesh, settings, out_path, contact)
        print(f"  [saved] {out_path.name}   (schema-validated)")

        # Mesh preview for ParaView, so the geometry can be checked before spending a solve.
        mesh.write_vtk(output_name=f"stent_{folder}_mesh", output_directory=str(run_dir))

        # ``base_record`` carries the build date, the status and the 4C build, so this type appears
        # in ``runs_summary.csv`` with the same columns filled in as the others.
        record = base_record("stent_only", run_dir) | {
            # what solve() needs to find this run again, without rebuilding it
            "input": {"file": out_path.name, "output_base": f"out_{folder}"},
            "case": {"name": name, "kind": spec["kind"], "strain": strain, **info},
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
                "n_elements": len(mesh.elements), "n_nodes": len(mesh.nodes),
                "section": section},
            "contact": contact_info,
            "geometry": axis,
            "settings": settings_to_dict(settings),
        }

        built.append(built_case(name=folder, input_path=out_path, run_dir=run_dir,
                                output_base=f"out_{folder}", record=record))

    return built


register("stent_only", build)

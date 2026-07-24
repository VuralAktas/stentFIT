import json
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from plotly import graph_objects as go
from plotly import io as pio
from scipy.spatial import cKDTree


from beamme.core.mesh import Mesh
from .stent_splines import (mesh_skeleton_beams)
from .artery_mesh import ( mesh_artery_gmsh,
                          import_artery_solid,
                          assemble_beam_solid)
from .artery_generate import (generate_artery_for_stent)

from beamme.core.rotation import Rotation
from beamme.cosserat_curve.cosserat_curve import CosseratCurve
from beamme.cosserat_curve.warping_along_cosserat_curve import warp_mesh_along_curve
from beamme.four_c.input_file import InputFile
from beamme.core.conf import bme
from beamme.core.geometry_set import GeometrySet
from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.function import Function
from beamme.four_c.header_functions import (set_header_static, set_runtime_output, set_beam_to_solid_meshtying)



def stent_feature_extraction(stent_dir: str | Path) -> dict:
    """
    Load a stent's skeletonisation output and print a summary of its features.

    Reads ``stent_features.json`` and ``skeleton_points.csv`` from
    ``stent_dir`` (as produced by :func:`~stentfit.stent_pipeline.stent_pipeline`),
    prints the stent's key dimensions and skeleton size for a quick sanity
    check, and returns everything the rest of this simulation setup needs.

    :param stent_dir: Stent skeletonisation output folder.
    :raises FileNotFoundError: If ``stent_dir`` is not a directory.
    :returns: Dict with the stent folder (``stent_dir``), the raw features
        dict (``features``), the skeleton points DataFrame (``stent_skel``,
        with ``neighbor_ids`` parsed back into lists), the centreline
        direction (``stent_centerline_dir``), and the individual geometry
        values (``stent_length``, ``stent_diameter``, ``stent_r_outer``,
        ``stent_strut_thickness``, ``stent_z_min``, ``stent_z_max``).
    """
    stent_dir = Path(stent_dir)
    if not stent_dir.is_dir():
        raise FileNotFoundError(
            f"Stent folder not found: {stent_dir}"
        )

    with open(stent_dir / "stent_features.json") as f:
        features = json.load(f)

    stent_centerline_dir = np.array(features["stent_centerline_direction"])

    stent_skel = pd.read_csv(stent_dir / "skeleton_points.csv")
    stent_skel["neighbor_ids"] = stent_skel["neighbor_ids"].apply(ast.literal_eval)

    stent_length          = features["length"]
    stent_diameter        = features["diameter"]
    stent_r_outer         = features["r_outer"]
    stent_strut_thickness = features["strut_thickness"]
    stent_z_min, stent_z_max = features["z_min"], features["z_max"]

    print(f"Loaded stent result from : {stent_dir.resolve()}")
    print(f"Skeleton nodes           : {len(stent_skel):,}")
    print(f"Centreline direction     : {stent_centerline_dir.round(4)}")
    print(f"length          : {stent_length:8.3f} mm")
    print(f"diameter        : {stent_diameter:8.3f} mm")
    print(f"r_outer         : {stent_r_outer:8.3f} mm")
    print(f"strut_thickness : {stent_strut_thickness:8.3f} mm")
    print(f"z range         : [{stent_z_min:.3f}, {stent_z_max:.3f}] mm")
    print(f"sampled points  : {features['num_points']:,}")
    print(f"skeleton nodes  : {len(stent_skel):,}")

    return {
        "stent_dir": stent_dir,
        "features": features,
        "stent_skel": stent_skel,
        "stent_centerline_dir": stent_centerline_dir,
        "stent_length": stent_length,
        "stent_diameter": stent_diameter,
        "stent_r_outer": stent_r_outer,
        "stent_strut_thickness": stent_strut_thickness,
        "stent_z_min": stent_z_min,
        "stent_z_max": stent_z_max,
    }



def stent_meshing_alignment(stent_dir: str | Path, sim_input_dir: str | Path, features: dict, artery_cl: np.ndarray,
                            youngs_modulus: float, poisson_ratio: float, density: float,
                            beam_class_label: str, beam_element_size: float) -> dict:
    """
    Mesh the straight stent as beams and warp it onto the artery centreline.

    Builds the stent's beam mesh from its fitted splines
    (:func:`~stentfit.stent_splines.mesh_skeleton_beams`), represents the
    artery centreline as a BeamMe ``CosseratCurve``, then warps the straight
    stent onto it (rotating the stent's own straight axis onto the curve's
    tangent, and centring the stent's ``z_min``/``z_max`` mid-point on the
    curve's arc mid-point). Writes the warped beam mesh to
    ``stent_warped.4C.yaml``.

    :param stent_dir: Stent skeletonisation output folder, read by
        :func:`~stentfit.stent_splines.mesh_skeleton_beams`.
    :param sim_input_dir: Folder ``stent_warped.4C.yaml`` is written into.
    :param features: Stent features dict; only ``z_min``/``z_max`` are used here.
    :param artery_cl: Artery centreline points to warp the stent onto.
    :param youngs_modulus: Beam material Young's modulus, in MPa.
    :param poisson_ratio: Beam material Poisson's ratio.
    :param density: Beam material density.
    :param beam_class_label: BeamMe beam element type, either
        ``'Beam3rHerm2Line3'`` or ``'Beam3rLine2Line2'``.
    :param beam_element_size: Target beam element length, in mm.
    :returns: Dict with the warped BeamMe ``Mesh`` (``beam_mesh``), its
        warped node coordinates (``stent_warped``), the ``beam_element_size``
        passed through, and the path to the written ``stent_yaml``.
    """
    stent_dir = Path(stent_dir)
    sim_input_dir = Path(sim_input_dir)

    # 1. Build the straight stent as a BeamMe beam mesh from the fitted splines in the folder.
    beam_mesh = mesh_skeleton_beams(input_dir=str(stent_dir),
                                      l_el=beam_element_size,
                                    youngs_modulus=youngs_modulus,
                                    poisson_ratio=poisson_ratio,
                                    density=density,
                                    beam_class_label=beam_class_label)

    # 2. Represent the artery centreline as a Cosserat curve.
    curve = CosseratCurve(artery_cl)

    # 3. Warp the straight stent onto the curve.
    ref_rot   = Rotation([0.0, 1.0, 0.0], -np.pi / 2.0)          # first basis vector -> +Z (stent axis)
    total_arc = np.linalg.norm(np.diff(artery_cl, axis=0), axis=1).sum()
    z_center  = 0.5 * (features["z_min"] + features["z_max"])
    origin    = np.array([0.0, 0.0, z_center - total_arc / 2.0])

    warp_mesh_along_curve(beam_mesh, curve, origin=origin, reference_rotation=ref_rot)

    # Warped node coordinates, for inspection and downstream use.
    stent_warped = np.array([node.coordinates for node in beam_mesh.nodes])

    # Save the warped stent beam mesh as a 4C .yaml

    stent_yaml = sim_input_dir / "stent_warped.4C.yaml"
    stent_input = InputFile()
    stent_input.add(beam_mesh)
    stent_input.dump(str(stent_yaml), validate=False, add_footer_application_script=False)
    print(f"[saved] {stent_yaml}")

    return {
        "beam_mesh": beam_mesh,
        "stent_warped": stent_warped,
        "beam_element_size": beam_element_size,
        "stent_yaml": stent_yaml,
    }


def create_assembly_mesh(artery_solid_yaml: str | Path | None,
                         beam_mesh: Mesh,
                         sim_input_dir: str | Path,
                         lumen_surface_index: int = 0,
                         bc_type=None,
                         output_filename: str = "artery_stent.4C.yaml"):
    """
    Import the artery solid and tie the stent beam mesh to it, as one 4C input.

    Imports ``artery_solid_yaml``
    (:func:`~stentfit.artery_mesh.import_artery_solid`), then couples it
    to ``beam_mesh`` with BeamMe's mortar beam-to-solid method
    (:func:`~stentfit.artery_mesh.assemble_beam_solid`), writing the
    combined 4C input file. Coupling defaults to tied meshtying
    (``bme.bc.beam_to_solid_surface_meshtying``); pass
    ``bme.bc.beam_to_solid_surface_contact`` for a real deployment
    simulation instead of this smoke test.

    :param artery_solid_yaml: Path to the artery solid ``.yaml``, from
        :func:`~stentfit.artery_mesh.mesh_artery_gmsh`. ``None`` skips
        assembly entirely (e.g. if meshing failed upstream).
    :param beam_mesh: Warped stent beam mesh, from :func:`stent_meshing_alignment`.
    :param sim_input_dir: Folder ``output_filename`` is written into.
    :param lumen_surface_index: Index into the artery solid's surface sets
        for the lumen surface the beams couple to. ``0`` is the lumen
        (``DSURFACE 1``, written first by :func:`~stentfit.artery_mesh.mesh_artery_gmsh`).
    :param bc_type: BeamMe beam-to-solid coupling type. ``None`` defaults to
        tied meshtying.
    :param output_filename: Filename for the assembled 4C input, written
        into ``sim_input_dir``.
    :returns: ``(None, None)`` if ``artery_solid_yaml`` is ``None``.
        Otherwise ``(input_file, full_mesh)`` — the BeamMe ``InputFile`` and
        the combined beam+solid mesh.
    """
    if artery_solid_yaml is None:
        print("No artery solid mesh from the previous cell — skipping assembly.")
        return None, None

    sim_input_dir = Path(sim_input_dir)
    input_file, solid = import_artery_solid(artery_solid_yaml)

    out_path = sim_input_dir / output_filename
    input_file, full_mesh = assemble_beam_solid(
        input_file, solid, beam_mesh,
        lumen_surface_index=lumen_surface_index,  # DSURFACE 1 = lumen (written first by the mesher)
        # bc_type defaults to beam_to_solid_surface_meshtying; switch to
        # bme.bc.beam_to_solid_surface_contact for a real deployment simulation.
        bc_type=bc_type,
        output_path=out_path,
    )

    print(f"Wrote assembled beam-to-solid 4C input file -> {out_path}")
    print("Next: materials (HGO-C artery), boundary conditions, expansion driver, solver, run 4C.")
    return input_file, full_mesh


def paraview_mesh_files(full_mesh: Mesh | None,
                        sim_input_dir: str | Path,
                        output_name: str = "artery_stent_mesh") -> tuple[Path, Path] | None:
    """
    Export the assembled beam+solid mesh as separate ``.vtu`` files for ParaView.

    BeamMe's ``write_vtk`` splits beams and solid elements into two files by
    itself; this just names them and reports their paths.

    :param full_mesh: Combined beam+solid mesh, from :func:`create_assembly_mesh`.
        ``None`` skips export (e.g. if assembly hasn't run yet).
    :param sim_input_dir: Folder the ``.vtu`` files are written into.
    :param output_name: Base filename; ``_beam.vtu`` / ``_solid.vtu`` are appended.
    :returns: ``None`` if ``full_mesh`` is ``None``. Otherwise
        ``(beam_vtu, solid_vtu)`` — the paths to the two written files.
    """
    if full_mesh is None:
        print("No assembled mesh — run the meshing / assembly cells first.")
        return None

    sim_input_dir = Path(sim_input_dir)
    full_mesh.write_vtk(output_name=output_name, output_directory=str(sim_input_dir))
    beam_vtu  = sim_input_dir / f"{output_name}_beam.vtu"
    solid_vtu = sim_input_dir / f"{output_name}_solid.vtu"
    print(f"[vtk] {beam_vtu}")
    print(f"[vtk] {solid_vtu}")
    print("Open the .vtu files in ParaView to inspect the meshes.")
    return beam_vtu, solid_vtu




def check_coupling_assumptions(
    beam_youngs: float,
    solid_youngs: float,
    beam_diameter: float,
    beam_element_length: float,
    solid_element_length: float,
    stiffness_ratio_min: float = 10.0,
    length_ratio_min: float = 1,
    length_ratio_max: float = 6,
    length_ratio_accuracy_max: float = 8.0,
) -> dict:
    """
    Check whether the beam and solid meshes satisfy the mixed-dimensional
    beam-to-solid coupling assumptions (Steinbrecher et al.), and print a
    pass/fail summary.

    Three checks, each independent of the others:

    1. **Stiffness** — the beam must be much stiffer than the solid
       (``E_beam / E_solid >= stiffness_ratio_min``), since the coupling
       assumes the solid deforms around an effectively rigid-ish beam.
    2. **Solid size vs. beam diameter** — the solid element size must be at
       least as large as the beam's cross-section diameter
       (``L_solid >= D_beam``), the spatial-resolution limit the mortar
       coupling is only valid above.
    3. **Element length ratio** — beam elements should be longer than solid
       elements, but not by too much: ``L_beam / L_solid`` has a valid band
       (``length_ratio_min`` to ``length_ratio_accuracy_max``, above which
       the coupling's L2 error grows) and, within that, a narrower optimal
       band (``length_ratio_min`` to ``length_ratio_max``, ~2.5-5 per the paper).

    :param beam_youngs: Beam material Young's modulus, in MPa.
    :param solid_youngs: Solid material Young's modulus, in MPa.
    :param beam_diameter: Beam cross-section diameter, in mm.
    :param beam_element_length: Actual mean beam element length, in mm.
    :param solid_element_length: Target solid element size, in mm.
    :param stiffness_ratio_min: Minimum acceptable ``E_beam / E_solid``.
    :param length_ratio_min: Lower bound of both the valid and optimal
        ``L_beam / L_solid`` bands.
    :param length_ratio_max: Upper bound of the optimal ``L_beam / L_solid`` band.
    :param length_ratio_accuracy_max: Upper bound of the valid ``L_beam / L_solid``
        band, above which coupling accuracy degrades.
    :returns: Dict with one entry per check (``stiffness``,
        ``solid_size_vs_beam_diameter``, ``element_length_ratio``), each
        holding its computed ratio, threshold(s), a human-readable ``note``,
        and a ``passed`` bool — plus ``all_passed``, ``True`` only if every
        check passed.
    """
    checks = {}

    # 1. Stiffness ratio -------------------------------------------------------
    stiff = beam_youngs / solid_youngs if solid_youngs > 0 else float("inf")
    ok_stiff = stiff >= stiffness_ratio_min
    checks["stiffness"] = dict(
        E_beam_MPa    = beam_youngs,
        E_solid_MPa   = solid_youngs,
        ratio         = round(stiff, 2),
        threshold_min = stiffness_ratio_min,
        passed        = bool(ok_stiff),
        note = (f"E_beam/E_solid = {stiff:.1f} "
                f"({'>=' if ok_stiff else '<'} {stiffness_ratio_min}) "
                f"- beam {'is' if ok_stiff else 'is NOT'} much stiffer than the solid"),
    )

    # 2. Solid element size >= beam cross-section diameter ---------------------
    rot = solid_element_length / beam_diameter if beam_diameter > 0 else float("inf")
    ok_rot = solid_element_length >= beam_diameter
    checks["solid_size_vs_beam_diameter"] = dict(
        solid_element_mm = round(solid_element_length, 4),
        beam_diameter_mm = round(beam_diameter, 4),
        ratio            = round(rot, 2),
        threshold_min    = 1.0,
        passed           = bool(ok_rot),
        note = (f"L_solid/D_beam = {rot:.2f} "
                f"({'>=' if ok_rot else '<'} 1) - solid element "
                f"{'>=' if ok_rot else '<'} beam cross-section diameter"),
    )

    # 3. Element length ratio: beam elements long vs solid, but not too long ---
    #    Valid band [length_ratio_min, length_ratio_accuracy_max]: below -> mortar
    #    coupling poorly conditioned; above ~8 -> L2 error grows (Steinbrecher).
    lr = beam_element_length / solid_element_length if solid_element_length > 0 else float("inf")
    ok_lr = length_ratio_min <= lr <= length_ratio_accuracy_max
    within_optimal = length_ratio_min <= lr <= length_ratio_max
    if lr > length_ratio_accuracy_max:
        lr_note = (f"L_beam/L_solid = {lr:.2f} (> {length_ratio_accuracy_max}) "
                   f"- too long: L2 coupling error grows; refine the solid or coarsen the beam")
    elif lr < length_ratio_min:
        lr_note = (f"L_beam/L_solid = {lr:.2f} (< {length_ratio_min}) "
                   f"- beam elements are NOT fairly long vs solid elements")
    elif within_optimal:
        lr_note = (f"L_beam/L_solid = {lr:.2f} - in the optimal "
                   f"{length_ratio_min}-{length_ratio_max} band")
    else:
        lr_note = (f"L_beam/L_solid = {lr:.2f} - acceptable "
                   f"(above the {length_ratio_max} optimum, below the "
                   f"{length_ratio_accuracy_max} accuracy limit)")
    checks["element_length_ratio"] = dict(
        beam_element_mm  = round(beam_element_length, 4),
        solid_element_mm = round(solid_element_length, 4),
        ratio            = round(lr, 2),
        valid_band       = (length_ratio_min, length_ratio_accuracy_max),
        optimal_band     = (length_ratio_min, length_ratio_max),
        within_optimal   = bool(within_optimal),
        passed           = bool(ok_lr),
        note             = lr_note,
    )

    all_passed = all(c["passed"] for c in checks.values())
    checks["all_passed"] = all_passed

    print("\nMixed-dimensional coupling assumption check")
    print("-------------------------------------------")
    for name in ("stiffness", "solid_size_vs_beam_diameter", "element_length_ratio"):
        c = checks[name]
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name:28s} {c['note']}")
    print(f"  => {'ALL CHECKS PASSED' if all_passed else 'ONE OR MORE CHECKS FAILED'}")

    return checks


def _radial_directions(points: np.ndarray, artery_cl: np.ndarray) -> np.ndarray:
    """
    Find each point's outward radial direction relative to a centreline.

    For each point, the nearest centreline point is found (KD-tree), and the
    vector from that centreline point to the point has its tangent
    component projected out — leaving only the component perpendicular to
    the centreline, normalised to a unit vector. Used to point the balloon
    expansion force straight out from the artery's local axis at each stent
    node, however the artery bends.

    :param points: ``(n, 3)`` points to find the radial direction at.
    :param artery_cl: Artery centreline points.
    :returns: ``(n, 3)`` unit vectors, each point's outward radial direction.
    """
    artery_cl = np.asarray(artery_cl, dtype=float)
    tang = np.gradient(artery_cl, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)

    _, idx = cKDTree(artery_cl).query(points)
    radial = points - artery_cl[idx]
    radial -= np.einsum("ij,ij->i", radial, tang[idx])[:, None] * tang[idx]
    norm = np.linalg.norm(radial, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return radial / norm


def build_smoketest_input(
    mesh: Mesh,
    beam_mesh: Mesh,
    artery_cl: np.ndarray,
    out_path: str | Path,
    n_steps: int = 10,
    total_time: float = 1.0,
    expansion_force: float = 1e-4,
    inlet_surface_index: int = 1,
    outlet_surface_index: int = 2,
    fix_stent_node: bool = True,
) -> Path:
    """
    Build a runnable, schema-validated 4C static simulation input.

    Adds a static solver header and runtime VTK output, fixes the artery's
    inlet and outlet surfaces (3 translational DOF, Dirichlet), and applies a
    quasi-static radial "balloon" expansion: a point force at each beam
    centreline node, directed radially outward from the artery centreline
    (:func:`_radial_directions`) and ramped from 0 to ``expansion_force``
    over ``n_steps`` by a time function. If ``fix_stent_node``, one stent
    node is also pinned in translation to remove the stent's rigid-body
    motion, since the radial forces alone don't constrain it. Coupling is
    tied beam-to-solid meshtying, matching what
    :func:`~stentfit.artery_mesh.assemble_beam_solid` already put on
    the mesh. Validates the assembled input against the 4C schema and writes it.

    :param mesh: Combined beam+solid mesh, from :func:`create_assembly_mesh`.
    :param beam_mesh: Warped stent beam mesh (only its nodes are used, to
        find the centreline nodes the expansion force is applied to).
    :param artery_cl: Artery centreline points, used to compute each beam
        node's outward radial direction.
    :param out_path: File path the simulation input is written to.
    :param n_steps: Number of quasi-static load steps.
    :param total_time: Total simulation time for the static solver.
    :param expansion_force: Radial point-force magnitude at full ramp.
    :param inlet_surface_index: Index into the solid's surface geometry sets
        for the inlet (fixed) surface.
    :param outlet_surface_index: Index into the solid's surface geometry sets
        for the outlet (fixed) surface.
    :param fix_stent_node: Pin one stent centreline node's translation, to
        remove rigid-body motion.
    :raises ValueError: If the imported solid is missing the inlet/outlet surface sets.
    :returns: ``out_path``, for chaining into a caller's own return value.
    """

    inp = InputFile()

    # --- Solver control (static, quasi-static steps) + runtime VTK output ------
    set_header_static(inp, n_steps=n_steps, total_time=total_time)
    set_runtime_output(inp)
    # Beam-to-solid interaction header (the coupling *condition* is already on the mesh).
    set_beam_to_solid_meshtying(inp, bme.bc.beam_to_solid_surface_meshtying,
                                contact_discretization="mortar", mortar_shape="line2")

    # --- Boundary conditions: fix the artery inlet + outlet ends (solid, 3 DOF) -
    surf = mesh.geometry_sets.get(bme.geo.surface, [])
    if len(surf) <= max(inlet_surface_index, outlet_surface_index):
        raise ValueError("Imported solid is missing the inlet/outlet surface sets.")
    for i in (inlet_surface_index, outlet_surface_index):
        mesh.add(BoundaryCondition(
            surf[i], {"NUMDOF": 3, "ONOFF": [1, 1, 1], "VAL": [0, 0, 0], "FUNCT": [0, 0, 0]},
            bc_type=bme.bc.dirichlet))

    # --- Radial "balloon" expansion force on the stent centreline nodes --------
    # Beam3rHerm2Line3 nodes carry 9 DOF [disp(3), rot(3), tangent(3)]; only the
    # Hermite centreline nodes (is_middle_node == False) have translational DOF.
    # A time-ramp FUNCT (f(t) = t) scales the force from 0 to full over the steps.
    ramp = Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}])
    mesh.add(ramp)

    cnodes = [n for n in beam_mesh.nodes if not n.is_middle_node]
    coords = np.array([n.coordinates for n in cnodes])
    radial = _radial_directions(coords, artery_cl)

    if fix_stent_node:                      # remove stent rigid-body translation
        mesh.add(BoundaryCondition(
            GeometrySet([cnodes[0]]),
            {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0], "VAL": [0] * 9, "FUNCT": [0] * 9},
            bc_type=bme.bc.dirichlet))

    for node, r in zip(cnodes, radial):
        f = expansion_force * r
        mesh.add(BoundaryCondition(
            GeometrySet([node]),
            {"NUMDOF": 9,
             "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
             "VAL": [f[0], f[1], f[2], 0, 0, 0, 0, 0, 0],
             "FUNCT": [ramp, ramp, ramp, 0, 0, 0, 0, 0, 0]},
            bc_type=bme.bc.neumann))

    # --- Assemble + schema-validate + write ------------------------------------
    inp.add(mesh)
    inp.dump(str(out_path), validate=True, add_footer_application_script=False)

    print(f"[sim] static smoke test: {n_steps} steps, radial expansion force "
          f"{expansion_force:g} N ramped over {len(cnodes):,} stent nodes")
    print(f"[sim] BCs: artery inlet+outlet fixed"
          + (", one stent node pinned" if fix_stent_node else "")
          + "; coupling = beam-to-solid meshtying (tied)")
    print(f"[saved] {out_path}")
    print("[sim] schema-validated. Run in 4C on Linux: set BEAMME_FOUR_C_EXE and launch 4C on this file.")
    return out_path

def build_smoketest_pipeline(   stent_name: str,
                                    stent_dir: str | Path,
                                    sim_input_dir: str | Path,
                                    artery_type: str = 'straight',
                                    inner_margin: float = 0.5,
                                    wall_thickness: float = 0.5,
                                    noise_amplitude: float = 0.15,
                                    noise_seed: float = 0,
                                    bend_angle_deg: float = 180.0,
                                    mesh_type: str = 'HEX8',
                                    artery_youngs: float = 2.0,
                                    factor_solid: float = 1.5,
                                    stent_youngs: float = 2.0e5,
                                    stent_poisson: float = 0.3,
                                    stent_density: float = 0.0,
                                    beam_class_label: str = 'Beam3rHerm2Line3',
                                    factor_beam: float = 1.2,
                                    n_steps: int = 10,
                                    expansion_force: float = 1e-4) -> None:
    """
    Build a parametric test artery, warp the stent onto it, check that the
    two meet the mixed-dimensional coupling assumptions, and — if they do —
    write a runnable 4C simulation input.

    Chains the whole synthetic pipeline: reads the stent's output
    (:func:`stent_feature_extraction`), generates a parametric test artery
    sized to it (:func:`~stentfit.artery_generate.generate_artery_for_stent`),
    meshes the stent as a beam mesh and warps it onto the artery centreline
    (:func:`stent_meshing_alignment`), meshes the artery wall as a 3D solid
    with GMSH (:func:`~stentfit.artery_mesh.mesh_artery_gmsh`), assembles
    the beam-to-solid mesh (:func:`create_assembly_mesh`) and exports it for
    ParaView (:func:`paraview_mesh_files`). It then checks the mixed-dimensional
    coupling assumptions (:func:`check_coupling_assumptions`), shows an inline
    Plotly view of the artery/centreline/stent, and — only if those checks
    pass — builds the runnable simulation input
    (:func:`build_smoketest_input`).

    :param stent_name: Name used to label the inline plot.
    :param stent_dir: Stent skeletonisation output folder (from :func:`~stentfit.stent_pipeline.stent_pipeline`).
    :param sim_input_dir: Folder every generated ``.4C.yaml`` and ``.vtu`` is written into.
    :param artery_type: Parametric artery shape: ``'straight'``, ``'curved'``, or ``'s_bend'``.
    :param inner_margin: Extra clearance, in mm, between the stent and the artery inner wall.
    :param wall_thickness: Artery wall thickness, in mm, for the 3D solid.
    :param noise_amplitude: Fractional wall-roughness noise added to the artery.
    :param noise_seed: Seed for the artery wall noise, for repeatable runs.
    :param bend_angle_deg: Total bend angle, in degrees, for a ``'curved'`` or ``'s_bend'`` artery.
    :param mesh_type: GMSH element type for the artery solid: ``'TET4'``, ``'TET10'``, or ``'HEX8'``.
    :param artery_youngs: Artery wall Young's modulus, in MPa (placeholder StVenantKirchhoff material).
    :param factor_solid: Safety factor sizing the artery solid element size relative to the beam diameter.
    :param stent_youngs: Stent beam Young's modulus, in MPa.
    :param stent_poisson: Stent beam Poisson's ratio.
    :param stent_density: Stent beam material density.
    :param beam_class_label: BeamMe beam element type, either
        ``'Beam3rHerm2Line3'`` or ``'Beam3rLine2Line2'``.
    :param factor_beam: Additional safety factor sizing the beam element length beyond ``factor_solid``.
    :param n_steps: Number of load steps for the balloon expansion ramp.
    :param expansion_force: Radial point-force magnitude for the balloon expansion.
    :returns: Nothing. Writes ``artery_solid.4C.yaml``, ``stent_warped.4C.yaml``,
        ``artery_stent.4C.yaml``, the ``*_beam.vtu`` / ``*_solid.vtu`` ParaView
        exports, and — only if the coupling checks pass — ``simulation.4C.yaml``,
        all into ``sim_input_dir``. Also displays an inline Plotly figure.
    """
    stent_dir = Path(stent_dir)
    sim_input_dir = Path(sim_input_dir)

    # Extract stent features for test_artery generation
    print("\nStent features")
    print("--------------")
    stent_info = stent_feature_extraction(stent_dir)
    stent_dir      = stent_info["stent_dir"]
    features       = stent_info["features"]
    stent_strut_thickness = stent_info["stent_strut_thickness"]

    # Element size configuration for stent and artery meshing
    d_beam = features["strut_thickness"]  # [mm] beam cross-section diameter (circular)
    solid_element_size = d_beam * factor_solid  # [mm] solid element size for the stent struts, with a safety factor
    beam_element_size = d_beam * factor_solid * factor_beam  # [mm] beam element length, with a safety factor

    # Test artery generation
    print("\nTest_Artery features")
    print("--------------")
    stent_feat_w = {k: {"value": v} for k, v in features.items() if isinstance(v, (int, float))}
    artery_geometry, artery_cl, artery_radius = generate_artery_for_stent(
        stent_feat_w,
        artery_type=artery_type,
        noise_amplitude=noise_amplitude,
        noise_seed=noise_seed,
        bend_angle_deg=bend_angle_deg,
        inner_margin=inner_margin,
        wall_thickness=wall_thickness,
    )

    # Stent meshing and alignment with the artery
    print("\n Stent Meshing and Alignment")
    print("--------------")
    stent_meshing = stent_meshing_alignment(
        stent_dir, sim_input_dir, features, artery_cl,
        beam_element_size=beam_element_size,
        youngs_modulus=stent_youngs,
        poisson_ratio=stent_poisson,
        density=stent_density,
        beam_class_label=beam_class_label,
    )

    stent_mesh          = stent_meshing["beam_mesh"]

    # Mesh the artery WALL into a 3D solid with GMSH and write it as a 4C .yaml.
    print("\n Test_Artery Meshing")
    print("--------------")
    artery_solid_yaml = mesh_artery_gmsh(
        r_inner=artery_radius,                    
        r_outer=artery_radius + wall_thickness,  
        centreline=artery_cl,                     
        mesh_type=mesh_type,
        element_size=solid_element_size,
        noise_amplitude=noise_amplitude,  
        noise_seed=noise_seed,
        youngs_modulus=artery_youngs,
        out_path=sim_input_dir / "artery_solid.4C.yaml",
    )

    # Create the assembly mesh for the stent and artery, and write it as a 4C .yaml.
    print('\n Assembly of Stent and Test_Artery')
    print("--------------")
    input_file, assembly_mesh = create_assembly_mesh(artery_solid_yaml, stent_mesh, sim_input_dir)

    # Export the assembled mesh as separate .vtu files for ParaView visualization. 
    print("\n Paraview")
    paraview_mesh_files(assembly_mesh, sim_input_dir)

    # Check the stent-artery fit and coupling assumptions 
    # Actual mean beam element length (chord, end-to-end) from the meshed beams.
    stent_elem_len = float(np.mean([np.linalg.norm(np.asarray(el.nodes[-1].coordinates) - np.asarray(el.nodes[0].coordinates)) for el in stent_mesh.elements]))
    coupling = check_coupling_assumptions(
        beam_youngs=stent_youngs,             
        solid_youngs=artery_youngs,            
        beam_diameter=stent_strut_thickness,    
        beam_element_length=stent_elem_len,      
        solid_element_length=solid_element_size,
    )
    coupling_ok = coupling["all_passed"]
    if not coupling_ok:
        print("\n[!] Coupling assumptions not satisfied — retune L_EL / SOLID_ELEMENT_SIZE (and the moduli) "
            "before building the simulation input.")
        
    # Visualization of geometries
    elem_coords = np.array([[n.coordinates for n in el.nodes] for el in stent_mesh.elements])  # (n_el, 3, 3)
    seg = np.full((len(elem_coords), 4, 3), np.nan)   # 3 nodes + a NaN gap to break the line between elements
    seg[:, :3, :] = elem_coords
    beam_lines = seg.reshape(-1, 3)

    verts = np.asarray(artery_geometry.vertices)
    faces = np.asarray(artery_geometry.faces)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="lightpink", opacity=0.25, name="artery", showscale=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=artery_cl[:, 0], y=artery_cl[:, 1], z=artery_cl[:, 2],
        mode="lines", line=dict(color="gray", width=3, dash="dash"), name="centreline",
    ))
    fig.add_trace(go.Scatter3d(
        x=beam_lines[:, 0], y=beam_lines[:, 1], z=beam_lines[:, 2],
        mode="lines", line=dict(color="crimson", width=2), name="stent beams",
    ))
    fig.update_layout(
        title=f"Stent warped onto {artery_type} artery — {stent_name} "
            f"({len(stent_mesh.elements):,} beam elements)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    stent_artery_html = sim_input_dir / "stent_artery_view.html"
    pio.write_html(fig, str(stent_artery_html), auto_open=False)
    print(f"[saved] {stent_artery_html}")
    try:
        fig.show()
    except Exception as e:
        print(f"[plotly] interactive view skipped ({e}); use {stent_artery_html.name} instead")

    # Build the runnable 4C simulation input: static solver + boundary conditions + a radial
    # "balloon" expansion force on the stent, on top of the assembled beam-to-solid mesh.
    # Gated on the coupling + physical-fit checks. Smoke test: placeholder material + meshtying
    # (tied). The file is schema-validated here; running it needs a 4C binary on Linux.
    if not (coupling_ok):
        print("[skip] Coupling checks failed — fix those before building the simulation.")
    else:
        simulation_yaml = build_smoketest_input(
            assembly_mesh, stent_mesh, artery_cl,
            out_path=sim_input_dir / "simulation.4C.yaml",
            n_steps=n_steps,
            expansion_force=expansion_force,
        )
        print(f"\nSimulation input ready -> {simulation_yaml}")
        
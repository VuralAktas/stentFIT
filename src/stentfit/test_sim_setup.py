import json
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from stentfit import mesh_skeleton_beams
from stentfit.test_artery_mesh import assemble_beam_solid, import_artery_solid

from beamme.core.rotation import Rotation
from beamme.cosserat_curve.cosserat_curve import CosseratCurve
from beamme.cosserat_curve.warping_along_cosserat_curve import warp_mesh_along_curve
from beamme.four_c.input_file import InputFile
from beamme.core.conf import bme
from beamme.core.geometry_set import GeometrySet
from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.function import Function
from beamme.four_c.header_functions import (set_header_static, set_runtime_output, set_beam_to_solid_meshtying)



def stent_feature_extraction(stent_dir: Path) -> dict:
    """In order to generate an artery that fits the stent, we need to extract the stent features from the outputs folder."""


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



def stent_meshing_alignment(stent_dir: Path, sim_input_dir: Path, features: dict, artery_cl: np.ndarray,
                            youngs_modulus: float, poisson_ratio: float, density: float,
                            beam_class_label: str, beam_element_size: float):
    """Mesh the stent skeleton into beams and warp it onto the artery centreline.

    1. Builds the straight stent as a BeamMe beam mesh from the fitted splines.
    2. Represents the artery centreline as a Cosserat curve.
    3. Warps the straight stent onto that curve and saves the warped mesh as a
       4C .yaml.
    """


    # 1. Build the straight stent as a BeamMe beam mesh from the fitted splines in the folder.
    beam_mesh = mesh_skeleton_beams(input_dir=str(stent_dir), output_dir=str(sim_input_dir),
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


def create_assembly_mesh(artery_solid_yaml, beam_mesh, sim_input_dir, *,
                         lumen_surface_index=0,
                         bc_type=None,
                         output_filename="artery_stent.4C.yaml"):
    """Assemble the artery solid + the warped stent beams into one BeamMe mesh
    and write the 4C input."""


    if artery_solid_yaml is None:
        print("No artery solid mesh from the previous cell — skipping assembly.")
        return None, None

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


def paraview_mesh_files(full_mesh, sim_input_dir: Path, output_name: str="artery_stent_mesh"):
    """Export the assembled beam + solid mesh as SEPARATE .vtu files for ParaView."""
    if full_mesh is None:
        print("No assembled mesh — run the meshing / assembly cells first.")
        return None

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
    """Validity checks for mixed-dimensional 1D-beam-to-3D-solid (mortar) coupling.

    These are the Steinbrecher et al. conditions the coupling relies on:

    1. **stiffness** — the beam (stent) must be much stiffer than the solid
       (artery): ``E_beam / E_solid >= stiffness_ratio_min``.
    2. **rule_of_thumb** — the solid element must be at least as large as the beam
       cross-section: ``L_solid >= D_beam`` (i.e. the beam cross-section is small
       compared to the solid elements).
    3. **element_length_ratio** — the beam-to-solid element ratio ``r = L_beam / L_solid``
       must sit in the valid band ``length_ratio_min <= r <= length_ratio_accuracy_max``:
       a **lower** bound (beam elements fairly long vs the solid, so the mortar
       coupling is well-conditioned) and an **upper accuracy** bound — the L2 error
       of the coupling is flat up to ``r ~ 8`` and then climbs steeply (Steinbrecher
       et al. convergence study), so ``r`` must not exceed ``length_ratio_accuracy_max``.
       The recommended optimum is ``length_ratio_min .. length_ratio_max`` (~2.5-5).

    Parameters
    ----------
    beam_youngs, solid_youngs : float   Young's moduli of the beam / solid [MPa]
    beam_diameter             : float   beam cross-section diameter [mm] (= 2 x beam radius = strut thickness)
    beam_element_length       : float   beam (1D) element length [mm]
    solid_element_length      : float   representative solid element edge length [mm]
    stiffness_ratio_min       : float   min E_beam / E_solid
    length_ratio_min, length_ratio_max     : float   recommended (optimal) band for L_beam / L_solid
    length_ratio_accuracy_max : float   hard upper limit on L_beam / L_solid (accuracy cliff, ~8)

    Returns
    -------
    dict  {check_name: {passed, note, ...values...}, "all_passed": bool}
        Also prints a short pass/fail table.
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

    # 2. Rule of thumb: solid element size >= beam cross-section diameter -------
    rot = solid_element_length / beam_diameter if beam_diameter > 0 else float("inf")
    ok_rot = solid_element_length >= beam_diameter
    checks["rule_of_thumb"] = dict(
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
    for name in ("stiffness", "rule_of_thumb", "element_length_ratio"):
        c = checks[name]
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name:22s} {c['note']}")
    print(f"  => {'ALL CHECKS PASSED' if all_passed else 'ONE OR MORE CHECKS FAILED'}")

    return checks


def _radial_directions(points: np.ndarray, artery_cl: np.ndarray) -> np.ndarray:
    """Outward radial unit vector per point, relative to the artery centreline.

    For each point, take the vector from its nearest centreline point, remove the
    component along the local centreline tangent, and normalise — so the force is
    purely radial (perpendicular to the vessel axis) for straight and curved arteries.
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
    mesh,
    beam_mesh,
    artery_cl,
    out_path,
    *,
    n_steps: int = 10,
    total_time: float = 1.0,
    expansion_force: float = 1e-4,
    inlet_surface_index: int = 1,
    outlet_surface_index: int = 2,
    fix_stent_node: bool = True,
):
    """Add solver / BCs / expansion load to the assembled mesh and dump a 4C input.

    Parameters
    ----------
    mesh        : assembled BeamMe Mesh (solid artery + stent beams + coupling),
                  as returned by ``assemble_beam_solid`` — carries the surface sets
                  (0 = lumen, 1 = inlet, 2 = outlet) and the beam-to-solid condition.
    beam_mesh   : the warped stent beam Mesh (used to pick the loaded stent nodes).
    artery_cl   : (M, 3) artery centreline (for the radial load direction).
    out_path    : where to write ``simulation.4C.yaml``.
    n_steps, total_time : quasi-static load stepping (force ramps 0 -> full over the time).
    expansion_force     : radial outward point force per stent node at full load [N].
    inlet_surface_index, outlet_surface_index : which imported surface sets are the ends.
    fix_stent_node      : pin one stent node's translation to remove rigid-body modes.

    Returns
    -------
    out_path : the written, schema-validated ``simulation.4C.yaml``.
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

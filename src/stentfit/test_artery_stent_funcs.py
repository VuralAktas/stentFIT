import json
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from stentfit import mesh_skeleton_beams

from beamme.core.rotation import Rotation
from beamme.cosserat_curve.cosserat_curve import CosseratCurve
from beamme.cosserat_curve.warping_along_cosserat_curve import warp_mesh_along_curve

from stentfit.test_artery_mesh import assemble_beam_solid, import_artery_solid
from beamme.four_c.input_file import InputFile


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
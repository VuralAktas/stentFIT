"""stentFIT — semi-automated virtual stent implantation"""

from importlib.metadata import version

from .stent_sampling import (sample_stent_points,
                             load_update_stent_data)
from .stent_crowns import detect_crowns
from .stent_skeleton_2d import (skeletonize_crowns_2d,
                                save_crown_2d_checkpoint,
                                load_crown_2d_checkpoint,
                                edit_crowns_2d_interactive,
                                assemble_2d_skeleton)
from .stent_skeleton_3d import (wrap_skeleton_to_3d,
                                save_stent_features_and_views)
from .stent_splines import (fit_skeleton_splines,
                            mesh_skeleton_beams)
from .stent_plotting import (plot_skeleton_splines_2d,
                             plot_skeleton_splines_trimesh)
from .stent_pipeline import (stent_pipeline,
                             run_skeletonization_2d,
                             resume_and_edit_crowns,
                             finalize_skeleton)

from .test_artery_mesh import (mesh_artery_gmsh,
                               import_artery_solid,
                               assemble_beam_solid)
from .test_artery_generate import generate_artery_for_stent
from .test_sim_setup import (stent_feature_extraction,
                                      stent_meshing_alignment,
                                      create_assembly_mesh,
                                      paraview_mesh_files,
                                      check_coupling_assumptions,
                                      build_smoketest_input)



__version__ = version("stentfit")

__all__ = [
    "sample_stent_points",
    "load_update_stent_data",
    "detect_crowns",
    "skeletonize_crowns_2d",
    "save_crown_2d_checkpoint",
    "load_crown_2d_checkpoint",
    "edit_crowns_2d_interactive",
    "assemble_2d_skeleton",
    "wrap_skeleton_to_3d",
    "save_stent_features_and_views",
    "fit_skeleton_splines",
    "mesh_skeleton_beams",
    "plot_skeleton_splines_2d",
    "plot_skeleton_splines_trimesh",
    "stent_pipeline",
    "run_skeletonization_2d",
    "resume_and_edit_crowns",
    "finalize_skeleton",
    "mesh_artery_gmsh",
    "import_artery_solid",
    "assemble_beam_solid",
    "generate_artery_for_stent",
    "stent_feature_extraction",
    "stent_meshing_alignment",
    "create_assembly_mesh",
    "paraview_mesh_files",
    "check_coupling_assumptions",
    "build_smoketest_input",
]

import os

import numpy as np
import trimesh

# Relative imports so the pipeline works from an installed wheel, independent of the
# repository layout or the caller's working directory.
from .stent_sampling import sample_stent_points
from .stent_crowns import detect_crowns
from .stent_skeleton_2d import (
    skeletonize_crowns_2d,
    save_crown_2d_checkpoint,
    load_crown_2d_checkpoint,
    edit_crowns_2d_interactive,
    assemble_2d_skeleton,
)
from .stent_skeleton_3d import wrap_skeleton_to_3d, save_stent_features_and_views
from .stent_splines import fit_skeleton_splines, mesh_skeleton_beams
from .stent_plotting import plot_skeleton_splines_2d, plot_skeleton_splines_trimesh


def _pick_existing_output_dir(output_dir:str) -> str:
    """Return the output folder to reuse for this stent.

    Versioned runs live as siblings of `output_dir` (<name>, <name>_v02, _v03, ...).
    If more than one exists, list them and let the user choose; otherwise return the
    only candidate unchanged.
    """
    import re

    parent, base = os.path.split(output_dir.rstrip('/\\'))
    parent = parent or '.'
    pattern = re.compile(rf"^{re.escape(base)}(_v\d+)?$")

    candidates = sorted(
        os.path.join(parent, name)
        for name in os.listdir(parent)
        if pattern.match(name) and os.path.isdir(os.path.join(parent, name)))

    if len(candidates) <= 1:
        return candidates[0] if candidates else output_dir

    print("Multiple output folders exist for this stent:")
    for i, path in enumerate(candidates, start=1):
        print(f"  [{i}] {os.path.basename(path)}")
    while True:
        choice = input(f"Which folder to use? [1-{len(candidates)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print("  invalid choice, try again.")


def run_skeletonization_2d(stl_file:str,
                           stent_name:str,
                           output_dir:str,
                           n_points:int = None,
                           max_display:int = 500_000,
                           remove_supports:bool = False,
                           random_seed:int = 0,
                           auto_tune:bool = True,
                           pixels_per_strut:int = 10,
                           dilate_px:int = 3,
                           pad_fraction:float = 0.20,
                           tune_time_limit:int = 120,
                           quality_gamma:float = 2.0,
                           crown_halo_frac:float = 0.4,
                           ):
    """sample -> detect crowns -> 2D skeletonise -> checkpoint"""


    # Check the STL file exists; if not, there is nothing to skeletonise.
    if not os.path.isfile(stl_file):
        raise FileNotFoundError(
            f"No STL file for stent '{stent_name}' at: {stl_file}")

    # If the requested output folder already exists and is non-empty, ask whether to
    # reuse it as-is, overwrite (wipe) it, or branch off into a new versioned folder
    # (<name>_v02, _v03, ...).
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        choice = input(
            f"Output folder already exists and is not empty:\n  {output_dir}\n"
            f"  [e] use it as-is   [o] overwrite (wipe it first)   "
            f"[n] make a new versioned folder\nChoose [e/o/n]: ").strip().lower()
        if choice.startswith('o'):
            import shutil
            shutil.rmtree(output_dir)
            print(f"[output] wiped and reusing {output_dir}")
        elif choice.startswith('n'):
            parent, base = os.path.split(output_dir.rstrip('/\\'))
            v = 2
            while os.path.isdir(os.path.join(parent, f"{base}_v{v:02d}")):
                v += 1
            output_dir = os.path.join(parent, f"{base}_v{v:02d}")
            print(f"[output] using a new folder: {output_dir}")
        else:
            # Reuse an existing folder. There may be several versioned folders for this
            # stent (<name>, <name>_v02, _v03, ...); if so, let the user pick which one.
            output_dir = _pick_existing_output_dir(output_dir)
            print(f"[output] reusing existing folder as-is: {output_dir}")
            state = load_crown_2d_checkpoint(output_dir)
            state['output_dir'] = output_dir
            return state
    os.makedirs(output_dir, exist_ok=True)

    mesh = trimesh.load(stl_file)
    print(f"Loaded mesh: {stl_file}")

    sampled = sample_stent_points(
        mesh, stent_name, output_dir, n_points=n_points, max_display=max_display,
        remove_supports=remove_supports, random_seed=random_seed)
    stent_df       = sampled['stent_df']
    stent_features = sampled['stent_features']
    centerline_dir = sampled['stent_centerline_direction']

    crowns = detect_crowns(stent_df, stent_features, stent_name, output_dir,
                           max_display=max_display)
    stent_df       = crowns['stent_df']
    crown_edges    = crowns['crown_edges']
    conn_radius_3d = crowns['conn_radius_3d']

    skel = skeletonize_crowns_2d(
        stent_df, stent_features, crown_edges, conn_radius_3d, output_dir,
        auto_tune=auto_tune, pixels_per_strut=pixels_per_strut, dilate_px=dilate_px,
        pad_fraction=pad_fraction, tune_time_limit=tune_time_limit,
        quality_gamma=quality_gamma,
        crown_halo_frac=crown_halo_frac)
    crown_2d    = skel['crown_2d']
    crown_order = skel['crown_order']

    r_mid           = stent_features['r_mid']
    strut_thickness = stent_features['strut_thickness']
    circumference   = 2 * np.pi * r_mid

    save_crown_2d_checkpoint(crown_2d, stent_features, centerline_dir, r_mid,
                             strut_thickness, circumference, crown_edges, output_dir)

    return {
        'crown_2d'                  : crown_2d,
        'crown_order'               : crown_order,
        'stent_df'                  : stent_df,
        'stent_features'            : stent_features,
        'stent_centerline_direction': centerline_dir,
        'r_mid'                     : r_mid,
        'strut_thickness'           : strut_thickness,
        'circumference'             : circumference,
        'crown_edges'               : crown_edges,
        'output_dir'                : output_dir
    }



def resume_and_edit_crowns(output_dir:str, 
                           state:dict=None,
                           interactive:bool=True):
    """Resume + Step 5.5: (optionally reload the checkpoint) -> interactive 2D edits
    -> assemble the full 2D skeleton.

    Pass the `state` from run_skeletonization_2d() in a normal run; pass state=None
    (e.g. after a kernel restart, or when resuming straight from an output folder) to
    reload crown_2d.pkl + crown_points.csv from disk. Set interactive=False to skip the
    manual-edit prompts (useful for scripted / non-interactive runs). Returns the state
    updated with the assembled 2D skeleton (skel_arc/skel_z/...)."""
    if state is None:
        state = load_crown_2d_checkpoint(output_dir)
    else:
        print("[resume] using in-memory state (normal run).")

    if interactive:
        state['crown_2d'] = edit_crowns_2d_interactive(
            state['crown_2d'], state['stent_features'], state['stent_centerline_direction'],
            state['r_mid'], state['strut_thickness'], state['circumference'],
            state['crown_edges'], output_dir)
    else:
        print("[resume] interactive=False -> skipping manual 2D edits.")

    state.update(assemble_2d_skeleton(state['crown_2d']))
    state['surf_df'] = state['stent_df']
    return state


def finalize_skeleton(state:dict,
                      output_dir:str,
                      stent_name:str,
                      wrap_max_surf:int = None,
                      prune_tip_frac:float = 0,
                      max_display:int = 500_000,
                      random_seed:int = 0):
    """wrap the 2D skeleton to 3D, fit splines,
    write the final exports, and render the unrolled-2D + trimesh-3D views."""


    skeleton_df = wrap_skeleton_to_3d(
        state['skel_arc'], state['skel_z'], state['skel_px'], state['pixel_size'],
        state['surf_df'], state['r_mid'], state['circumference'],
        state['strut_thickness'], output_dir, stent_name, wrap_max_surf=wrap_max_surf,
        prune_tip_frac=prune_tip_frac,
        max_display=max_display, random_seed=random_seed)

    fitted = fit_skeleton_splines(skeleton_df, output_dir)
    skeleton_curves  = fitted['curves']
    skeleton_splines = fitted['splines']

    save_stent_features_and_views(
        skeleton_df, state['stent_df'], state['stent_features'],
        state['stent_centerline_direction'], state['crown_edges'], output_dir,
        stent_name, max_display=max_display)

    plot_skeleton_splines_2d(
        skeleton_curves, skeleton_splines, state['stent_df'], state['r_mid'],
        state['circumference'], state['crown_edges'], state.get('crown_order'),
        output_dir, stent_name)
    plot_skeleton_splines_trimesh(skeleton_splines, output_dir)

    state['skeleton_df']      = skeleton_df
    state['skeleton_curves']  = skeleton_curves
    state['skeleton_splines'] = skeleton_splines
    return state

def stent_pipeline( stl_file:str,
                    output_dir:str,
                    stent_name:str,
                    mesh_beams:bool = False,
                    n_points:int = None,
                    max_display:int = 500_000,
                    remove_supports:bool = False,
                    random_seed:int = 0,
                    auto_tune:bool = True,
                    pixels_per_strut:int = 10,
                    dilate_px:int = 3,
                    pad_fraction:float = 0.20,
                    tune_time_limit:int = 120,
                    quality_gamma:float = 2.0,
                    crown_halo_frac:float = 0.4,

                    wrap_max_surf:int = 2_000_000,
                    prune_tip_frac:float = 0,

                    l_el:float=0.1,
                    youngs_modulus:float=2.0e5,
                    poisson_ratio:float=0.3,
                    density:float=0.0,
                    beam_class_label:str='Beam3rHerm2Line3', 
                    ):
    

    state = run_skeletonization_2d( stl_file=stl_file,
                                    stent_name=stent_name,
                                    output_dir=output_dir,
                                    n_points=n_points,
                                    max_display=max_display,
                                    remove_supports=remove_supports,
                                    random_seed=random_seed,
                                    auto_tune=auto_tune,
                                    pixels_per_strut=pixels_per_strut,
                                    dilate_px=dilate_px,
                                    pad_fraction=pad_fraction,
                                    tune_time_limit=tune_time_limit,
                                    quality_gamma=quality_gamma,
                                    crown_halo_frac=crown_halo_frac)
    # run_skeletonization_2d may have branched to a versioned folder; use the actual one.
    output_dir = state['output_dir']

    # Pass the in-memory `state` straight through (normal run). If instead you resume
    # from disk, call resume_and_edit_crowns(output_dir) with state=None.
    state = resume_and_edit_crowns(output_dir, state)
    

    state = finalize_skeleton(  state=state,
                                output_dir=output_dir,
                                stent_name=stent_name,
                                wrap_max_surf=wrap_max_surf,
                                prune_tip_frac=prune_tip_frac,
                                max_display=max_display,
                                random_seed=random_seed)
    
    if mesh_beams:
        beam_mesh = mesh_skeleton_beams(output_dir=output_dir,
                                        l_el=l_el,
                                        youngs_modulus=youngs_modulus,
                                        poisson_ratio=poisson_ratio,
                                        density=density,
                                        beam_class_label=beam_class_label)


    print(f"[pipeline] done -> {output_dir}") 
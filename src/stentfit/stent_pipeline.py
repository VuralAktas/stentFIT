import os
import numpy as np
import trimesh
import shutil   # For overwriting the selected stent folder in case of user selection
import re       # For using the existing folder in case of user selection

from .stent_sampling import sample_stent_points
from .stent_rings import detect_rings
from .stent_skeleton_2d import (
    skeletonize_rings_2d,
    save_ring_2d_checkpoint,
    load_ring_2d_checkpoint,
    edit_rings_2d_interactive,
    assemble_2d_skeleton,
)
from .stent_skeleton_3d import wrap_skeleton_to_3d, save_stent_features_and_views
from .stent_splines import fit_skeleton_splines, mesh_skeleton_beams
from .stent_plotting import plot_skeleton_splines_2d, plot_skeleton_splines_trimesh


def _pick_existing_output_dir(output_dir: str) -> str:
    """
    Resolve which versioned output folder to reuse for this stent.

    Looks for ``output_dir`` itself plus any versioned siblings
    (``<name>_v02``, ``_v03``, ...) in the same parent folder. With none
    found, returns ``output_dir`` unchanged; with exactly one, returns it;
    with several, prints them and prompts the user to pick one.

    :param output_dir: The originally requested output folder.
    :returns: The resolved folder path to actually use.
    """
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


def run_skeletonization_2d(
        stl_file:str,
        stent_name:str,
        output_dir:str,
        n_points:int = None,
        max_display:int = 500_000,
        remove_supports:bool = False,
        random_seed:int = 0,
        n_rings:int = None,
        auto_tune:bool = True,
        pixels_per_strut:int = 10,
        dilate_px:int = 3,
        pad_fraction:float = 0.20,
        tune_time_limit:int = 120,
        quality_gamma:float = 2.0,
        ring_halo_frac:float = 0.4) -> dict:
    """
    Sample the stent mesh, detect its rings, and 2D-skeletonise each ring.

    Loads the STL, samples a point cloud, splits the stent into rings, then
    runs the 2D skeletonisation per ring. The assembled ring data is
    checkpointed to disk so this can resume after a kernel restart, via
    :func:`resume_and_edit_rings`.

    If ``output_dir`` already exists and is non-empty, the user is asked
    whether to reuse it as-is, overwrite it, or branch into a new versioned
    folder (``<name>_v02``, ``_v03``, ...). Reusing an existing folder loads
    its checkpoint from disk instead of recomputing anything.

    :param stl_file: Path to the stent surface mesh (STL).
    :param stent_name: Name used to label outputs and plots.
    :param output_dir: Folder for all outputs. May be replaced with a
        versioned or reused folder, depending on the user's choice above.
    :param n_points: Number of points to sample from the mesh. ``None`` picks
        the count automatically from the stent size.
    :param max_display: Maximum number of points drawn in the HTML views.
    :param remove_supports: Drop print-support points during sampling.
    :param random_seed: Seed for the point sampling, for repeatable runs.
    :param n_rings: Expected ring count. ``None`` lets ring detection decide.
    :param auto_tune: Search for the best 2D skeletonisation parameters per ring.
    :param pixels_per_strut: Raster resolution, in pixels across one strut width.
    :param dilate_px: Dilation radius, in pixels, before thinning.
    :param pad_fraction: Seam padding added when unrolling a ring to 2D.
    :param tune_time_limit: Time budget, in seconds, for the auto-tune search.
    :param quality_gamma: Weight of the skeleton quality score during tuning.
    :param ring_halo_frac: Z-halo, as a fraction of ring height, so struts
        reconnect across neighbouring rings.
    :returns: State dict with the per-ring 2D skeleton (``ring_2d``), the
        stent point cloud and features, and the (possibly updated) ``output_dir``.
    """


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
            state = load_ring_2d_checkpoint(output_dir)
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

    rings = detect_rings(stent_df, stent_features, stent_name, output_dir,
                          max_display=max_display, n_rings=n_rings)
    stent_df       = rings['stent_df']
    ring_edges    = rings['ring_edges']
    conn_radius_3d = rings['conn_radius_3d']

    skel = skeletonize_rings_2d(
        stent_df, stent_features, ring_edges, conn_radius_3d, output_dir,
        auto_tune=auto_tune, pixels_per_strut=pixels_per_strut, dilate_px=dilate_px,
        pad_fraction=pad_fraction, tune_time_limit=tune_time_limit,
        quality_gamma=quality_gamma,
        ring_halo_frac=ring_halo_frac)
    ring_2d    = skel['ring_2d']
    ring_order = skel['ring_order']

    r_mid           = stent_features['r_mid']
    strut_thickness = stent_features['strut_thickness']
    circumference   = 2 * np.pi * r_mid

    save_ring_2d_checkpoint(ring_2d, stent_features, centerline_dir, r_mid,
                            strut_thickness, circumference, ring_edges, output_dir)

    return {
        'ring_2d'                  : ring_2d,
        'ring_order'               : ring_order,
        'stent_df'                  : stent_df,
        'stent_features'            : stent_features,
        'stent_centerline_direction': centerline_dir,
        'r_mid'                     : r_mid,
        'strut_thickness'           : strut_thickness,
        'circumference'             : circumference,
        'ring_edges'                : ring_edges,
        'output_dir'                : output_dir
    }



def resume_and_edit_rings(output_dir: str,
                          state: dict | None = None) -> dict:
    """
    Apply the interactive manual edits, then assemble the full 2D skeleton.

    Runs after :func:`run_skeletonization_2d`. Pass the in-memory ``state``
    from that call for a normal run; pass ``state=None`` (e.g.
    after a kernel restart, or when resuming straight from an output folder)
    to reload ``ring_2d.pkl`` + ``ring_points.csv`` from disk via
    :func:`~stentfit.stent_skeleton_2d.load_ring_2d_checkpoint`. Always runs
    :func:`~stentfit.stent_skeleton_2d.edit_rings_2d_interactive`, which
    prompts to manually edit any ring before assembling
    (:func:`~stentfit.stent_skeleton_2d.assemble_2d_skeleton`).

    :param output_dir: Folder to reload the checkpoint from, if ``state`` is
        ``None``. Also passed through to the interactive edit prompts.
    :param state: State dict from :func:`run_skeletonization_2d`. ``None``
        reloads it from ``output_dir`` instead.
    :returns: The state dict, updated with the assembled 2D skeleton
        (``skel_arc``/``skel_z``/...) and the surface point cloud (``surf_df``).
    """
    if state is None:
        state = load_ring_2d_checkpoint(output_dir)
    else:
        print("[resume] using in-memory state (normal run).")

    state['ring_2d'] = edit_rings_2d_interactive(
        state['ring_2d'], state['stent_features'], state['stent_centerline_direction'],
        state['r_mid'], state['strut_thickness'], state['circumference'],
        state['ring_edges'], output_dir)

    state.update(assemble_2d_skeleton(state['ring_2d']))
    state['surf_df'] = state['stent_df']
    return state


def finalize_skeleton(state: dict,
                      output_dir: str,
                      stent_name: str,
                      wrap_max_surf: int | None = None,
                      prune_tip_frac: float = 0,
                      max_display: int = 500_000,
                      random_seed: int = 0) -> dict:
    """
    Wrap the 2D skeleton to 3D, fit splines, and write the final exports.

    Runs after :func:`resume_and_edit_rings`. Lifts the assembled 2D
    skeleton onto the 3D stent surface
    (:func:`~stentfit.stent_skeleton_3d.wrap_skeleton_to_3d`, which also
    cleans up the graph), fits a B-spline per curve
    (:func:`~stentfit.stent_splines.fit_skeleton_splines`), then writes the
    final feature/view exports
    (:func:`~stentfit.stent_skeleton_3d.save_stent_features_and_views`) and
    the unrolled-2D and trimesh-3D spline views.

    :param state: State dict from :func:`resume_and_edit_rings`, with the
        assembled 2D skeleton and stent features/geometry.
    :param output_dir: Folder all exports and views are written into.
    :param stent_name: Name used to label outputs and plots.
    :param wrap_max_surf: Maximum surface points used when wrapping to 3D.
        ``None`` uses every surface point.
    :param prune_tip_frac: Fraction of each curve tip to prune after wrapping.
    :param max_display: Maximum number of points drawn in the HTML views.
    :param random_seed: Seed for any subsampling during the 3D wrap.
    :returns: ``state``, updated with the 3D skeleton graph
        (``skeleton_df``), the grouped curves (``skeleton_curves``), and the
        fitted splines (``skeleton_splines``).
    """
    skeleton_df = wrap_skeleton_to_3d(
        state['skel_arc'], state['skel_z'], state['skel_px'],
        state['surf_df'], state['r_mid'], state['circumference'],
        state['strut_thickness'], output_dir, stent_name, wrap_max_surf=wrap_max_surf,
        prune_tip_frac=prune_tip_frac,
        max_display=max_display, random_seed=random_seed)

    fitted = fit_skeleton_splines(skeleton_df, output_dir)
    skeleton_curves  = fitted['curves']
    skeleton_splines = fitted['splines']

    save_stent_features_and_views(
        skeleton_df, state['stent_df'], state['stent_features'],
        state['stent_centerline_direction'], state['ring_edges'], output_dir,
        max_display=max_display)

    plot_skeleton_splines_2d(
        skeleton_curves, skeleton_splines, state['stent_df'], state['r_mid'],
        state['circumference'], state['ring_edges'], state.get('ring_order'),
        output_dir, stent_name)
    plot_skeleton_splines_trimesh(skeleton_splines, output_dir)

    state['skeleton_df']      = skeleton_df
    state['skeleton_curves']  = skeleton_curves
    state['skeleton_splines'] = skeleton_splines
    return state

def stent_pipeline(
        stl_file:str,
        output_dir:str,
        stent_name:str,
        n_points:int = None,
        max_display:int = 500_000,
        remove_supports:bool = False,
        random_seed:int = 0,
        n_rings:int = None,
        auto_tune:bool = True,
        pixels_per_strut:int = 10,
        dilate_px:int = 3,
        pad_fraction:float = 0.20,
        tune_time_limit:int = 120,
        quality_gamma:float = 2.0,
        ring_halo_frac:float = 0.4,
        wrap_max_surf:int = 2_000_000,
        prune_tip_frac:float = 0,) -> dict:
    """
    Run the full stent skeletonisation, from an STL mesh to fitted splines.

    Runs, in order: sampling the mesh and building the 2D skeleton per ring
    (:func:`run_skeletonization_2d`), applying the interactive 2D edits and
    assembling the flat skeleton (:func:`resume_and_edit_rings`), then
    wrapping it to 3D and fitting the splines (:func:`finalize_skeleton`).
    Each of these three calls also writes its own intermediate files into
    ``output_dir``.

    :param stl_file: Path to the stent surface mesh (STL).
    :param output_dir: Folder for all outputs. If it already exists, the user
        is asked whether to reuse, overwrite, or branch into a versioned folder.
    :param stent_name: Name used to label outputs and plots.
    :param n_points: Number of points to sample from the mesh. ``None`` picks
        the count automatically from the stent size.
    :param max_display: Maximum number of points drawn in the HTML views.
    :param remove_supports: Drop print-support points during sampling.
    :param random_seed: Seed for the point sampling, for repeatable runs.
    :param n_rings: Expected ring count. ``None`` lets ring detection decide.
    :param auto_tune: Search for the best 2D skeletonisation parameters per ring.
    :param pixels_per_strut: Raster resolution, in pixels across one strut width.
    :param dilate_px: Dilation radius, in pixels, before thinning.
    :param pad_fraction: Seam padding added when unrolling a ring to 2D.
    :param tune_time_limit: Time budget, in seconds, for the auto-tune search.
    :param quality_gamma: Weight of the skeleton quality score during tuning.
    :param ring_halo_frac: Z-halo, as a fraction of ring height, so struts
        reconnect across neighbouring rings.
    :param wrap_max_surf: Maximum surface points used when wrapping to 3D.
    :param prune_tip_frac: Fraction of each curve tip to prune after wrapping.
    :returns: The pipeline state dict, holding the 2D skeleton, the 3D skeleton
        graph, the fitted curves and splines, and the final ``output_dir``.
    """

    state = run_skeletonization_2d( stl_file=stl_file,
                                    stent_name=stent_name,
                                    output_dir=output_dir,
                                    n_points=n_points,
                                    max_display=max_display,
                                    remove_supports=remove_supports,
                                    random_seed=random_seed,
                                    n_rings=n_rings,
                                    auto_tune=auto_tune,
                                    pixels_per_strut=pixels_per_strut,
                                    dilate_px=dilate_px,
                                    pad_fraction=pad_fraction,
                                    tune_time_limit=tune_time_limit,
                                    quality_gamma=quality_gamma,
                                    ring_halo_frac=ring_halo_frac)
    
    # run_skeletonization_2d may have branched to a versioned folder; use the actual one.
    output_dir = state['output_dir']

    # Pass the in-memory `state` straight through (normal run). If instead you resume
    # from disk, call resume_and_edit_rings(output_dir) with state=None.
    state = resume_and_edit_rings(output_dir, state)
    

    state = finalize_skeleton(  state=state,
                                output_dir=output_dir,
                                stent_name=stent_name,
                                wrap_max_surf=wrap_max_surf,
                                prune_tip_frac=prune_tip_frac,
                                max_display=max_display,
                                random_seed=random_seed)



    print(f"[pipeline] done -> {output_dir}") 

    return state
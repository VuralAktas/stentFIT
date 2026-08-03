"""
The :class:`Stent` class: an STL stent design carried through skeletonisation.
"""

import os
import re
import shutil

import numpy as np
import trimesh

from .core import plotting
from .core import rings as _rings
from .core import sampling as _sampling
from .core import skeleton_2d as _skeleton_2d
from .core import skeleton_3d as _skeleton_3d
from .core import splines as _splines


class Stent:
    """
    A stent design, from its STL surface mesh to a fitted spline wireframe.

    Holds one stent's data as it moves through the skeletonisation pipeline.
    Every stage sets more attributes and writes its own inspectable
    intermediate (CSV + Plotly HTML) into :attr:`output_dir`, so the run can be
    checked before committing to an expensive contact simulation.

    Run the whole thing with :meth:`skeletonize`, or drive the three phases
    separately for the interactive per-ring workflow::

        stent = Stent(stl_file, "stent01", "outputs/stent01").skeletonize()

        # or, phase by phase, inspecting the plots in between:
        stent = Stent(stl_file, "stent01", "outputs/stent01")
        stent.skeletonize_2d()        # sample -> rings -> 2D skeleton + checkpoint
        stent.edit_and_assemble()     # manual 2D fixes -> assembled flat skeleton
        stent.finalize()              # wrap to 3D -> clean graph -> fit splines

    After a kernel restart, :meth:`load` rebuilds the object from the
    checkpoint in an existing output folder, so the run can pick up at
    :meth:`edit_and_assemble` without recomputing anything.

    The tuning parameters below are set once on the instance; per-operation
    parameters stay as arguments on the method that uses them.

    :param stl_file: Path to the stent surface mesh (STL).
    :param stent_name: Name used to label outputs and plots.
    :param output_dir: Folder for all outputs. If it already exists and is
        non-empty, :meth:`skeletonize_2d` asks whether to reuse, overwrite, or
        branch into a versioned folder, and may replace this with the folder
        actually used.
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
    """

    def __init__(self: "Stent",
                 stl_file: str,
                 stent_name: str,
                 output_dir: str,
                 n_points: int | None = None,
                 max_display: int = 500_000,
                 remove_supports: bool = False,
                 random_seed: int = 0,
                 n_rings: int | None = None,
                 auto_tune: bool = True,
                 pixels_per_strut: int = 10,
                 dilate_px: int = 3,
                 pad_fraction: float = 0.20,
                 tune_time_limit: int = 120,
                 quality_gamma: float = 2.0,
                 ring_halo_frac: float = 0.4):
        # --- inputs ---
        self.stl_file = stl_file
        self.stent_name = stent_name
        self.output_dir = output_dir

        # --- tuning parameters (former pipeline config) ---
        self.n_points = n_points
        self.max_display = max_display
        self.remove_supports = remove_supports
        self.random_seed = random_seed
        self.n_rings = n_rings
        self.auto_tune = auto_tune
        self.pixels_per_strut = pixels_per_strut
        self.dilate_px = dilate_px
        self.pad_fraction = pad_fraction
        self.tune_time_limit = tune_time_limit
        self.quality_gamma = quality_gamma
        self.ring_halo_frac = ring_halo_frac

        # --- data, filled in progressively as the pipeline runs ---
        self.mesh = None                        # loaded trimesh surface mesh
        self.stent_df = None                    # sampled surface point cloud
        self.stent_features = None              # geometry features dict
        self.stent_centerline_direction = None  # PCA long-axis unit vector
        self.ring_edges = None                  # ring z-boundaries
        self.ring_order = None                  # ring ids, bottom to top
        self.ring_2d = None                     # per-ring 2D skeletons
        self.skel_arc = None                    # assembled 2D skeleton, arc
        self.skel_z = None                      # assembled 2D skeleton, z
        self.skel_px = None                     # per-point pixel size
        self.surf_df = None                     # surface cloud used for the 3D wrap
        self.skeleton_df = None                 # final 3D skeleton graph
        self.skeleton_curves = None             # skeleton grouped into curves
        self.skeleton_splines = None            # one fitted B-spline per curve

    # ------------------------------------------------------------------
    # Output folder handling
    # ------------------------------------------------------------------

    def _versioned_candidates(self: "Stent") -> list[str]:
        """
        List the existing output folders belonging to this stent.

        Matches :attr:`output_dir` itself plus any versioned siblings
        (``<name>_v02``, ``_v03``, ...) in the same parent folder.

        :returns: Sorted folder paths, empty if none exist yet.
        """
        parent, base = os.path.split(self.output_dir.rstrip('/\\'))
        parent = parent or '.'
        pattern = re.compile(rf"^{re.escape(base)}(_v\d+)?$")
        return sorted(
            os.path.join(parent, name)
            for name in os.listdir(parent)
            if pattern.match(name) and os.path.isdir(os.path.join(parent, name)))

    def _pick_existing_output_dir(self: "Stent", action: str = "use") -> str:
        """
        Resolve which of this stent's versioned output folders to act on.

        With no candidate found, returns :attr:`output_dir` unchanged; with
        exactly one, returns it; with several, prints them and prompts the user
        to pick one. Used for both reusing and overwriting, so a stent with
        several versions never has one silently picked for it.

        :param action: Verb used in the prompt, so it reads as the operation
            actually about to happen — ``"use"`` or ``"overwrite"``.
        :returns: The resolved folder path to actually act on.
        """
        candidates = self._versioned_candidates()
        if len(candidates) <= 1:
            return candidates[0] if candidates else self.output_dir

        print("Multiple output folders exist for this stent:")
        for i, path in enumerate(candidates, start=1):
            print(f"  [{i}] {os.path.basename(path)}")
        while True:
            choice = input(f"Which folder to {action}? "
                           f"[1-{len(candidates)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                return candidates[int(choice) - 1]
            print("  invalid choice, try again.")

    def _resolve_output_dir(self: "Stent") -> bool:
        """
        Settle on the output folder, asking the user if one already exists.

        If :attr:`output_dir` exists and is non-empty, offers to reuse it
        as-is, overwrite (wipe) it, or branch into a new versioned folder
        (``<name>_v02``, ``_v03``, ...). Reusing loads that folder's checkpoint
        straight onto this object instead of recomputing anything. Updates
        :attr:`output_dir` in place, and creates the folder.

        Reusing *and* overwriting both go through
        :meth:`_pick_existing_output_dir`, so when several versions of this
        stent exist the user is asked which one — an overwrite never silently
        wipes the unversioned base folder while other versions sit alongside it.

        :returns: ``True`` if an existing checkpoint was loaded and the caller
            should skip recomputation, ``False`` to carry on with a fresh run.
        """
        if os.path.isdir(self.output_dir) and os.listdir(self.output_dir):
            choice = input(
                f"Output folder already exists and is not empty:\n  {self.output_dir}\n"
                f"  [e] use it as-is   [o] overwrite (wipe it first)   "
                f"[n] make a new versioned folder\nChoose [e/o/n]: ").strip().lower()
            if choice.startswith('o'):
                # Several versioned folders may exist for this stent; wipe the
                # one the user names, not blindly the unversioned base folder.
                self.output_dir = self._pick_existing_output_dir(action="overwrite")
                shutil.rmtree(self.output_dir)
                print(f"[output] wiped and reusing {self.output_dir}")
            elif choice.startswith('n'):
                parent, base = os.path.split(self.output_dir.rstrip('/\\'))
                v = 2
                while os.path.isdir(os.path.join(parent, f"{base}_v{v:02d}")):
                    v += 1
                self.output_dir = os.path.join(parent, f"{base}_v{v:02d}")
                print(f"[output] using a new folder: {self.output_dir}")
            else:
                # Reuse an existing folder. There may be several versioned folders
                # for this stent (<name>, <name>_v02, ...); if so, let the user pick.
                self.output_dir = self._pick_existing_output_dir()
                print(f"[output] reusing existing folder as-is: {self.output_dir}")
                self._load_checkpoint_into(self.output_dir)
                return True
        os.makedirs(self.output_dir, exist_ok=True)
        return False

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self: "Stent", verbose: bool = True) -> str:
        """
        Write the per-ring 2D skeletons and stent geometry to ``ring_2d.pkl``.

        Together with the already-saved ``ring_points.csv``, this lets
        :meth:`load` rebuild the object after a kernel restart, without
        rerunning sampling, ring detection, or 2D skeletonisation.

        :param verbose: Print the saved path and ring count.
        :returns: Path to the written ``ring_2d.pkl``.
        """
        r_mid = self.stent_features['r_mid']
        return _skeleton_2d.save_ring_2d_checkpoint(
            self.ring_2d, self.stent_features, self.stent_centerline_direction,
            r_mid, self.stent_features['strut_thickness'],
            2 * np.pi * r_mid, self.ring_edges, self.output_dir, verbose=verbose)

    def _load_checkpoint_into(self: "Stent", output_dir: str) -> None:
        """
        Restore this object's state from an output folder's checkpoint.

        :param output_dir: Folder holding ``ring_2d.pkl`` and ``ring_points.csv``.
        """
        state = _skeleton_2d.load_ring_2d_checkpoint(output_dir)
        self.output_dir = output_dir
        self.ring_2d = state['ring_2d']
        self.stent_features = state['stent_features']
        self.stent_centerline_direction = state['stent_centerline_direction']
        self.ring_edges = state['ring_edges']
        self.stent_df = state['stent_df']

    @classmethod
    def load(cls: type["Stent"],
             output_dir: str,
             stl_file: str = "",
             stent_name: str = "",
             **kwargs) -> "Stent":
        """
        Rebuild a :class:`Stent` from an existing output folder.

        Reads back the ``ring_2d.pkl`` checkpoint and ``ring_points.csv``, so
        the pipeline can resume at :meth:`edit_and_assemble` after a kernel
        restart. This replaces the procedural pipeline's ``state=None`` resume
        branch — a reloaded object *is* the resumed state.

        The stent's own geometry (features, centreline direction, ring edges,
        surface cloud) all comes from the checkpoint. ``stl_file`` only matters
        if you intend to re-run :meth:`skeletonize_2d` on the reloaded object.

        :param output_dir: Folder holding ``ring_2d.pkl`` and ``ring_points.csv``.
        :param stl_file: Path to the original STL, if it is needed again.
        :param stent_name: Name used to label outputs and plots. Empty uses
            the output folder's basename.
        :param kwargs: Any :class:`Stent` tuning parameter, to override the
            constructor defaults on the reloaded object.
        :raises FileNotFoundError: If the folder has no checkpoint to load.
        :returns: The reconstructed stent.
        """
        stent = cls(stl_file,
                    stent_name or os.path.basename(output_dir.rstrip('/\\')),
                    output_dir, **kwargs)
        stent._load_checkpoint_into(output_dir)
        return stent

    # ------------------------------------------------------------------
    # Pipeline phases
    # ------------------------------------------------------------------

    def skeletonize_2d(self: "Stent") -> "Stent":
        """
        Sample the stent mesh, detect its rings, and 2D-skeletonise each ring.

        Loads the STL, samples a point cloud, splits the stent into rings, then
        runs the 2D skeletonisation per ring. The assembled ring data is
        checkpointed to disk (:meth:`save_checkpoint`) so the run can resume
        after a kernel restart.

        If :attr:`output_dir` already exists and is non-empty, the user is
        asked whether to reuse it as-is, overwrite it, or branch into a new
        versioned folder. Reusing loads that folder's checkpoint instead of
        recomputing anything.

        Sets :attr:`stent_df`, :attr:`stent_features`,
        :attr:`stent_centerline_direction`, :attr:`ring_edges`,
        :attr:`ring_2d`, and :attr:`ring_order`.

        :raises FileNotFoundError: If :attr:`stl_file` does not exist.
        :returns: ``self``, so phases can be chained.
        """
        # Check the STL file exists; if not, there is nothing to skeletonise.
        if not os.path.isfile(self.stl_file):
            raise FileNotFoundError(
                f"No STL file for stent '{self.stent_name}' at: {self.stl_file}")

        if self._resolve_output_dir():
            return self          # reused an existing folder: state came off disk

        self.mesh = trimesh.load(self.stl_file)
        print(f"Loaded mesh: {self.stl_file}")

        sampled = _sampling.sample_stent_points(
            self.mesh, self.stent_name, self.output_dir, n_points=self.n_points,
            max_display=self.max_display, remove_supports=self.remove_supports,
            random_seed=self.random_seed)
        self.stent_df = sampled['stent_df']
        self.stent_features = sampled['stent_features']
        self.stent_centerline_direction = sampled['stent_centerline_direction']

        detected = _rings.detect_rings(
            self.stent_df, self.stent_features, self.stent_name, self.output_dir,
            max_display=self.max_display, n_rings=self.n_rings)
        self.stent_df = detected['stent_df']
        self.ring_edges = detected['ring_edges']
        conn_radius_3d = detected['conn_radius_3d']

        skel = _skeleton_2d.skeletonize_rings_2d(
            self.stent_df, self.stent_features, self.ring_edges, conn_radius_3d,
            self.output_dir, auto_tune=self.auto_tune,
            pixels_per_strut=self.pixels_per_strut, dilate_px=self.dilate_px,
            pad_fraction=self.pad_fraction, tune_time_limit=self.tune_time_limit,
            quality_gamma=self.quality_gamma, ring_halo_frac=self.ring_halo_frac)
        self.ring_2d = skel['ring_2d']
        self.ring_order = skel['ring_order']

        self.save_checkpoint()
        return self

    def edit_and_assemble(self: "Stent") -> "Stent":
        """
        Apply the interactive manual 2D edits, then assemble the flat skeleton.

        Runs after :meth:`skeletonize_2d`. Always prompts once to manually edit
        any ring, so defects the automatic detector missed can be fixed by hand
        on the flat ring skeleton, then concatenates every ring into one 2D
        skeleton.

        Sets :attr:`skel_arc`, :attr:`skel_z`, :attr:`skel_px`, and
        :attr:`surf_df`.

        :returns: ``self``, so phases can be chained.
        """
        r_mid = self.stent_features['r_mid']
        self.ring_2d = _skeleton_2d.edit_rings_2d_interactive(
            self.ring_2d, self.stent_features, self.stent_centerline_direction,
            r_mid, self.stent_features['strut_thickness'],
            2 * np.pi * r_mid, self.ring_edges, self.output_dir)

        assembled = _skeleton_2d.assemble_2d_skeleton(self.ring_2d)
        self.skel_arc = assembled['skel_arc']
        self.skel_z = assembled['skel_z']
        self.skel_px = assembled['skel_px']
        self.surf_df = self.stent_df
        return self

    def finalize(self: "Stent",
                 prune_tip_frac: float = 0,
                 max_display: int = 500_000,
                 random_seed: int = 0) -> "Stent":
        """
        Wrap the 2D skeleton to 3D, fit splines, and write the final exports.

        Runs after :meth:`edit_and_assemble`. Lifts the assembled 2D skeleton
        onto the 3D stent surface (which also cleans up the graph: junction
        blobs contracted to a centroid, short dead-ends pruned), fits a
        B-spline per curve, then writes the final feature/view exports plus the
        unrolled-2D and trimesh-3D spline views.

        Sets :attr:`skeleton_df`, :attr:`skeleton_curves`, and
        :attr:`skeleton_splines`.

        :param prune_tip_frac: Fraction of each curve tip to prune after wrapping.
        :param max_display: Maximum number of points drawn in the HTML views.
        :param random_seed: Seed for any subsampling during the 3D wrap.
        :returns: ``self``, so phases can be chained.
        """
        r_mid = self.stent_features['r_mid']
        self.skeleton_df = _skeleton_3d.wrap_skeleton_to_3d(
            self.skel_arc, self.skel_z, self.skel_px, self.surf_df,
            r_mid, 2 * np.pi * r_mid,
            self.stent_features['strut_thickness'], self.output_dir,
            # Ceiling on the surface points fed to the wrap's KD-tree: a memory
            # guard, not a modelling choice, so it is not a public argument.
            # Past this many points the cloud is randomly downsampled.
            self.stent_name, wrap_max_surf=2_000_000,
            prune_tip_frac=prune_tip_frac, max_display=max_display,
            random_seed=random_seed)

        fitted = _splines.fit_skeleton_splines(self.skeleton_df, self.output_dir)
        self.skeleton_curves = fitted['curves']
        self.skeleton_splines = fitted['splines']

        _skeleton_3d.save_stent_features_and_views(
            self.skeleton_df, self.stent_df, self.stent_features,
            self.stent_centerline_direction, self.ring_edges, self.output_dir,
            max_display=max_display)

        self.plot_splines_2d()
        self.plot_splines_trimesh()
        return self

    def skeletonize(self: "Stent", prune_tip_frac: float = 0) -> "Stent":
        """
        Run the full skeletonisation, from the STL mesh to fitted splines.

        Chains all three phases: :meth:`skeletonize_2d`,
        :meth:`edit_and_assemble`, then :meth:`finalize`. Each writes its own
        intermediates into :attr:`output_dir`.

        (:meth:`skeletonize_2d` is only the first phase, despite the similar
        name — the verb alone means the whole run, the ``_2d`` suffix marks the
        sub-step.)

        :param prune_tip_frac: Fraction of each curve tip to prune after wrapping.
        :returns: ``self``, holding the 2D skeleton, the 3D skeleton graph, the
            grouped curves, and the fitted splines.
        """
        self.skeletonize_2d()
        self.edit_and_assemble()
        self.finalize(prune_tip_frac=prune_tip_frac,
                      max_display=self.max_display,
                      random_seed=self.random_seed)

        print(f"[pipeline] done -> {self.output_dir}")
        return self

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def plot_splines_2d(self: "Stent") -> None:
        """
        Draw the fitted splines on the unrolled (arc, z) plane.

        Writes ``skeleton_splines_2d.html`` / ``.png`` into :attr:`output_dir`.
        """
        r_mid = self.stent_features['r_mid']
        plotting.plot_skeleton_splines_2d(
            self.skeleton_curves, self.skeleton_splines, self.stent_df,
            r_mid, 2 * np.pi * r_mid, self.ring_edges,
            self.ring_order, self.output_dir, self.stent_name)

    def plot_splines_trimesh(self: "Stent") -> None:
        """
        Draw the fitted splines as 3D tubes with trimesh.

        Writes ``skeleton_splines_trimesh.html`` and ``skeleton_splines.glb``
        into :attr:`output_dir`.
        """
        plotting.plot_skeleton_splines_trimesh(self.skeleton_splines, self.output_dir)

    def __repr__(self: "Stent") -> str:
        """:returns: A short summary of how far this stent has been processed."""
        stage = "empty"
        if self.skeleton_splines is not None:
            stage = f"{len(self.skeleton_splines)} splines"
        elif self.skel_arc is not None:
            stage = f"2D skeleton ({len(self.skel_arc):,} pts)"
        elif self.ring_2d is not None:
            stage = f"{len(self.ring_2d)} rings"
        return f"<Stent {self.stent_name!r} [{stage}] -> {self.output_dir}>"

"""
The :class:`Artery` class: a parametric test artery and its 3D wall solid.
"""

from pathlib import Path

import numpy as np

from .core import artery_geom as _geom
from .stent import Stent


class Artery:
    """
    A test artery: a wall surface, its centreline, and the 3D solid meshed from them.

    The geometry is parametric and generated to fit a given stent, rather than
    imported from imaging — enough to exercise the whole mixed-dimensional
    chain end to end. The shape is a *parameter*, not a separate constructor::

        artery = Artery(stent, artery_type="curved", inner_margin=0.5)
        sim = Simulation(stent, artery, sim_input_dir)
        sim.setup()

    Everything is resolved and built at construction, so :attr:`radius`,
    :attr:`geometry` and :attr:`centreline` are real the moment the object
    exists. Note this builds only the **wall surface** (a ``trimesh`` tube);
    the 3D finite-element solid 4C actually solves on is a separate step,
    :meth:`mesh_solid`, which :meth:`~stentfit.simulation.Simulation.setup`
    runs for you.

    Every dimension is derived from the stent: the lumen radius is the stent's
    outer radius plus ``inner_margin`` clearance, the length a multiple of the
    stent length (so the stent always sits well inside), and any bend radius is
    picked so the arc roughly spans that length at the given bend angle. There
    is deliberately no way to set those dimensions by hand — a hand-sized tube
    is still a synthetic artery, and the real use for a specific geometry is
    importing patient anatomy, which is a different construction path.

    :param stent: The stent this artery is sized to hold. Its skeletonisation
        must have run, so its features are populated.
    :param artery_type: Shape: ``'straight'``, ``'curved'``, or ``'s_bend'``.
    :param inner_margin: Clearance, in mm, between the stent's outer radius and
        the lumen wall.
    :param wall_thickness: Wall thickness, in mm. ``0`` builds the lumen
        surface only, with no separate wall.
    :param noise_amplitude: Fractional wall-roughness noise, as a fraction of
        the radius. ``0`` gives a smooth pipe.
    :param noise_seed: Seed for the wall noise, for repeatable runs.
    :param bend_angle_deg: Total bend angle, in degrees. Used only by
        ``'curved'`` and ``'s_bend'``.
    :param mesh_type: GMSH element type for the solid: ``'TET4'``, ``'TET10'``,
        or ``'HEX8'``. Used by :meth:`mesh_solid`.
    :param artery_youngs: Wall Young's modulus, in MPa (placeholder
        StVenantKirchhoff material). Used by :meth:`mesh_solid`.
    :param n_circumference: Number of vertices around each cross-section.
    :param n_axial: Number of cross-sections along the length.
    :raises ValueError: If ``artery_type`` is unknown, or the stent has not
        been skeletonised yet.
    """

    def __init__(self: "Artery",
                 stent: Stent,
                 artery_type: str = "straight",
                 inner_margin: float = 0.5,
                 wall_thickness: float = 0.5,
                 noise_amplitude: float = 0.15,
                 noise_seed: float = 0,
                 bend_angle_deg: float = 180.0,
                 mesh_type: str = "HEX8",
                 artery_youngs: float = 2.0,
                 n_circumference: int = 64,
                 n_axial: int = 150):
        # What each shape needs, as (length factor, widest arc radius in mm):
        #   length factor - how much longer than the stent the artery is built,
        #     so its clamped inlet/outlet ends sit clear of the stent. An S-bend
        #     needs more again, because its two arcs eat into the straight runs.
        #   widest arc radius - caps how gentle a bend may get. Without it a
        #     shallow bend angle produces an arc so wide the artery is visually
        #     straight. None for a straight artery, which has no arc.
        shapes = {"straight": (1.5, None),
                  "curved":   (1.5, 20.0),
                  "s_bend":   (2.0, 15.0)}
        if artery_type not in shapes:
            raise ValueError(
                f"unknown artery_type {artery_type!r}; "
                f"expected one of {sorted(shapes)}")
        length_factor, bend_radius_cap = shapes[artery_type]

        self.stent = stent
        self.artery_type = artery_type
        self.inner_margin = inner_margin
        self.wall_thickness = float(wall_thickness)
        self.noise_amplitude = noise_amplitude
        self.noise_seed = noise_seed
        self.bend_angle_deg = bend_angle_deg

        # --- solid meshing defaults, applied by mesh_solid() ---
        self.mesh_type = mesh_type
        self.artery_youngs = artery_youngs

        # --- dimensions, all derived from the stent ---
        features = self._stent_features()
        self.radius = float(features["r_outer"] + self.inner_margin)
        self.length = float(features["length"] * length_factor)
        self.bend_radius = self._resolve_bend_radius(bend_radius_cap)

        # --- build the wall surface and the centreline it follows ---
        self.geometry, self.centreline = self._build(n_circumference, n_axial)

        #: Path to the 4C solid ``.yaml``, once :meth:`mesh_solid` has run.
        self.solid_yaml: Path | None = None

        self._print_summary()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _stent_features(self: "Artery") -> dict:
        """
        Read the stent's geometry features, checking it has been skeletonised.

        :raises ValueError: If the stent's pipeline has not run yet.
        :returns: The stent's features dict.
        """
        if self.stent.stent_features is None:
            raise ValueError(
                "stent has no features yet - run Stent.skeletonize() (or "
                "Stent.load()) before generating an artery for it.")
        return self.stent.stent_features

    def _resolve_bend_radius(self: "Artery", cap: float | None) -> float | None:
        """
        Work out the arc radius for a bent artery.

        Picked so the arc spans about 90% of the artery's own length at the
        given bend angle, then limited by ``cap`` so a shallow angle cannot
        produce an arc so wide the artery looks straight. An S-bend splits its
        length across two arcs.

        :param cap: Widest arc radius allowed, in mm. ``None`` for a straight
            artery, which has no arc.
        :returns: The arc radius in mm, or ``None`` for a straight artery.
        """
        if cap is None:
            return None

        angle = np.radians(self.bend_angle_deg)
        n_arcs = 2.0 if self.artery_type == "s_bend" else 1.0
        return min(cap, 0.9 * self.length / (n_arcs * angle))

    def _build(self: "Artery",
               n_circumference: int,
               n_axial: int) -> tuple:
        """
        Build the wall surface mesh and the centreline it is swept along.

        :param n_circumference: Number of vertices around each cross-section.
        :param n_axial: Number of cross-sections along the length.
        :returns: ``(geometry, centreline)`` — the ``trimesh`` wall surface and
            the ``(n, 3)`` centreline points.
        """
        common = dict(radius=self.radius, length=self.length,
                      wall_thickness=self.wall_thickness,
                      n_circumference=n_circumference, n_axial=n_axial,
                      noise_amplitude=self.noise_amplitude,
                      noise_seed=self.noise_seed)

        if self.artery_type == "straight":
            geometry = _geom.generate_straight_artery(**common)
            centreline = _geom.straight_centreline(self.length, n_axial)
        elif self.artery_type == "curved":
            geometry = _geom.generate_curved_artery(
                bend_radius=self.bend_radius,
                bend_angle_deg=self.bend_angle_deg, **common)
            centreline = _geom.curved_centreline(
                self.length, self.bend_radius, self.bend_angle_deg, n_axial)
        else:
            geometry = _geom.generate_s_bend_artery(
                bend_radius=self.bend_radius,
                bend_angle_deg=self.bend_angle_deg, **common)
            centreline = _geom.s_bend_centreline(
                self.length, self.bend_radius, self.bend_angle_deg, n_axial)

        return geometry, centreline

    def _print_summary(self: "Artery") -> None:
        """Print the built artery's dimensions, matching the old pipeline's output."""
        total_arc = float(np.linalg.norm(np.diff(self.centreline, axis=0),
                                         axis=1).sum())
        print(f"Artery type      : {self.artery_type}")
        print(f"Artery radius    : {self.radius:.3f} mm (lumen)")
        print(f"Wall thickness   : {self.wall_thickness:.3f} mm"
              + (f"  (outer radius {self.radius + self.wall_thickness:.3f} mm)"
                 if self.wall_thickness > 0 else "  (lumen surface only)"))
        print(f"Noise amplitude  : {self.noise_amplitude} "
              f"({self.noise_amplitude * 100:.0f}% of radius)  seed={self.noise_seed}")
        if self.bend_radius is not None:
            angle = np.radians(self.bend_angle_deg)
            n_arcs = 2 if self.artery_type == "s_bend" else 1
            arc = n_arcs * self.bend_radius * angle
            print(f"Bend angle       : {self.bend_angle_deg:.1f} deg")
            print(f"Bend radius      : {self.bend_radius:.2f} mm  "
                  f"({'2x arc' if n_arcs == 2 else 'arc'} = {arc:.2f} mm)")
        stent_length = self.stent.stent_features["length"]
        print(f"Arc length       : {total_arc:.2f} mm  "
              f"(stent {stent_length:.2f} mm = "
              f"{stent_length / total_arc * 100:.0f}% of artery)")
        print(f"Centreline       : {len(self.centreline)} points  "
              f"bounds {self.centreline.min(0).round(2)} → "
              f"{self.centreline.max(0).round(2)}")
        print(f"Mesh             : {len(self.geometry.vertices):,} vertices  "
              f"{len(self.geometry.faces):,} faces  "
              f"watertight={self.geometry.is_watertight}")

    # ------------------------------------------------------------------
    # Solid meshing
    # ------------------------------------------------------------------

    def mesh_solid(self: "Artery",
                   out_path: str | Path,
                   element_size: float,
                   mesh_type: str | None = None,
                   youngs_modulus: float | None = None,
                   poisson_ratio: float = 0.3,
                   density: float = 1.0,
                   material_id: int = 1) -> Path:
        """
        Mesh the artery wall as a hollow 3D solid with GMSH and write a 4C ``.yaml``.

        This is the finite-element mesh 4C solves on, as opposed to the wall
        *surface* built at construction. Meshes the annulus between
        :attr:`radius` and ``radius + wall_thickness`` as a straight tube,
        classifies its boundary nodes into ``DSURFACE`` sets (``1`` = lumen,
        ``2`` = inlet, ``3`` = outlet), then warps the whole tube onto
        :attr:`centreline` using the same frame convention as the stent warp,
        so beam and solid stay aligned. Writes the mesh with a placeholder
        ``MAT_Struct_StVenantKirchhoff`` material.

        Stores the written path on :attr:`solid_yaml`, which
        :meth:`~stentfit.simulation.Simulation.assemble` reads back.

        :param out_path: File path the 4C ``.yaml`` solid is written to.
        :param element_size: Target element size, in mm.
        :param mesh_type: Element type: ``'TET4'``, ``'TET10'``, or ``'HEX8'``.
            ``None`` uses :attr:`mesh_type` from the constructor.
        :param youngs_modulus: Material Young's modulus, in MPa. ``None`` uses
            :attr:`artery_youngs` from the constructor.
        :param poisson_ratio: Placeholder material Poisson's ratio.
        :param density: Placeholder material density.
        :param material_id: Material ID written into the 4C input.
        :raises ValueError: If the wall has no thickness, or the element type is
            not supported.
        :returns: The path written, also stored on :attr:`solid_yaml`.
        """
        self.solid_yaml = _geom.mesh_artery_gmsh(
            r_inner=self.radius,
            r_outer=self.radius + self.wall_thickness,
            centreline=self.centreline,
            out_path=out_path,
            mesh_type=self.mesh_type if mesh_type is None else mesh_type,
            element_size=element_size,
            noise_amplitude=self.noise_amplitude,
            noise_seed=self.noise_seed,
            material_id=material_id,
            youngs_modulus=(self.artery_youngs if youngs_modulus is None
                            else youngs_modulus),
            poisson_ratio=poisson_ratio,
            density=density,
        )
        return self.solid_yaml

    def __repr__(self: "Artery") -> str:
        """:returns: A short summary of the artery's shape and size."""
        meshed = "" if self.solid_yaml is None else f", solid -> {self.solid_yaml.name}"
        arc = float(np.linalg.norm(np.diff(self.centreline, axis=0), axis=1).sum())
        return (f"<Artery {self.artery_type} r={self.radius:.3f} "
                f"wall={self.wall_thickness:.3f} arc={arc:.2f}mm{meshed}>")

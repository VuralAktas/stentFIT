"""
The :class:`Artery` class: a parametric test artery and its 3D wall solid.
"""

from pathlib import Path

import numpy as np
import trimesh

from .kernels import artery_geom as _geom


class Artery:
    """
    A test artery: a wall surface, its centreline, and the 3D solid meshed from them.

    The geometry is parametric and generated to fit a given stent, rather than
    imported from imaging — enough to exercise the whole mixed-dimensional
    chain end to end. Build one with a factory rather than the constructor::

        artery = Artery.for_stent(stent, artery_type="curved")
        artery.mesh_solid(element_size=0.2, out_path="artery_solid.4C.yaml")

    :meth:`for_stent` sizes everything off the stent's own features;
    :meth:`straight`, :meth:`curved`, and :meth:`s_bend` take explicit
    dimensions instead.

    :param geometry: The artery wall surface, as a ``trimesh.Trimesh``.
    :param centreline: ``(n, 3)`` points along the artery centreline.
    :param radius: Lumen (inner) radius, in mm.
    :param wall_thickness: Wall thickness, in mm.
    :param artery_type: Shape this was built as: ``'straight'``, ``'curved'``,
        or ``'s_bend'``. Empty if built directly through the constructor.
    :param length: Centreline length the artery was built to, in mm.
    :param bend_radius: Arc radius used, in mm. ``None`` for a straight artery.
    :param bend_angle_deg: Bend angle used, in degrees. ``None`` for a straight artery.
    :param noise_amplitude: Fractional wall-roughness noise the surface was built with.
    :param noise_seed: Seed used for that wall noise.
    """

    def __init__(self,
                 geometry: trimesh.Trimesh,
                 centreline: np.ndarray,
                 radius: float,
                 *,
                 wall_thickness: float = 0.5,
                 artery_type: str = "",
                 length: float | None = None,
                 bend_radius: float | None = None,
                 bend_angle_deg: float | None = None,
                 noise_amplitude: float = 0.0,
                 noise_seed: int | None = None):
        self.geometry = geometry
        self.centreline = np.asarray(centreline, dtype=float)
        self.radius = float(radius)
        self.wall_thickness = float(wall_thickness)

        # --- shape parameters this artery was built with ---
        self.artery_type = artery_type
        self.length = length
        self.bend_radius = bend_radius
        self.bend_angle_deg = bend_angle_deg
        self.noise_amplitude = noise_amplitude
        self.noise_seed = noise_seed

        #: Path to the 4C solid ``.yaml``, once :meth:`mesh_solid` has run.
        self.solid_yaml: Path | None = None

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    @property
    def r_outer(self) -> float:
        """
        :returns: The wall's outer radius, ``radius + wall_thickness``, in mm.
        """
        return self.radius + self.wall_thickness

    @property
    def arc_length(self) -> float:
        """
        :returns: Total centreline arc length, summed segment by segment, in mm.
        """
        return float(np.linalg.norm(np.diff(self.centreline, axis=0), axis=1).sum())

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def for_stent(cls,
                  stent,
                  *,
                  artery_type: str = "straight",
                  inner_margin: float = 0.5,
                  wall_thickness: float = 0.5,
                  noise_amplitude: float = 0.15,
                  noise_seed: float = 0,
                  bend_angle_deg: float = 180.0) -> "Artery":
        """
        Build a parametric test artery sized to fit a given stent.

        The lumen radius is the stent's outer radius plus ``inner_margin``
        clearance; the length is a multiple of the stent length (1.5x for
        ``'straight'``/``'curved'``, 2x for ``'s_bend'``, so the stent always
        sits well inside it), and any bend radius is picked so the artery's arc
        roughly matches its length at the given bend angle.

        :param stent: A :class:`~stentfit.stent.Stent` whose skeletonisation has
            run, so :attr:`~stentfit.stent.Stent.stent_features` is populated.
        :param artery_type: Artery shape: ``'straight'``, ``'curved'``, or ``'s_bend'``.
        :param inner_margin: Extra clearance, in mm, between the stent and the lumen.
        :param wall_thickness: Artery wall thickness, in mm. ``0`` builds the
            lumen surface only.
        :param noise_amplitude: Fractional wall-roughness noise, as a fraction
            of the radius.
        :param noise_seed: Seed for the wall noise. ``None`` draws a fresh
            pattern each call.
        :param bend_angle_deg: Total bend angle, in degrees, for
            ``'curved'``/``'s_bend'``.
        :raises ValueError: If the stent has not been skeletonised yet.
        :returns: The generated artery.
        """
        if stent.stent_features is None:
            raise ValueError(
                "stent has no features yet - run Stent.skeletonize() (or "
                "Stent.load()) before generating an artery for it.")

        # The kernel reads features in the {"value": ...} shape used elsewhere
        # in the simulation setup, so wrap the plain scalars for it.
        stent_feat_w = {k: {"value": v} for k, v in stent.stent_features.items()
                        if isinstance(v, (int, float))}

        geometry, centreline, radius = _geom.generate_artery_for_stent(
            stent_feat_w,
            artery_type=artery_type,
            noise_amplitude=noise_amplitude,
            noise_seed=noise_seed,
            bend_angle_deg=bend_angle_deg,
            inner_margin=inner_margin,
            wall_thickness=wall_thickness,
        )

        return cls(geometry, centreline, radius,
                   wall_thickness=wall_thickness,
                   artery_type=artery_type,
                   length=float(np.linalg.norm(np.diff(centreline, axis=0),
                                               axis=1).sum()),
                   bend_angle_deg=(None if artery_type == "straight"
                                   else bend_angle_deg),
                   noise_amplitude=noise_amplitude,
                   noise_seed=noise_seed)

    @classmethod
    def straight(cls,
                 radius: float = 1.5,
                 length: float = 25.0,
                 *,
                 wall_thickness: float = 0.0,
                 n_circumference: int = 32,
                 n_axial: int = 100,
                 noise_amplitude: float = 0.0,
                 noise_seed: int | None = None) -> "Artery":
        """
        Build a straight artery of constant radius along the z-axis.

        :param radius: Lumen radius, in mm.
        :param length: Artery length, in mm.
        :param wall_thickness: Wall thickness, in mm. ``0`` builds the lumen
            surface only, with no separate wall.
        :param n_circumference: Number of vertices around each cross-section.
        :param n_axial: Number of cross-sections along the length.
        :param noise_amplitude: Fractional wall-roughness noise, as a fraction
            of the radius.
        :param noise_seed: Seed for the wall noise. ``None`` draws a fresh
            pattern each call.
        :returns: The generated artery.
        """
        geometry = _geom.generate_straight_artery(
            radius=radius, length=length, wall_thickness=wall_thickness,
            n_circumference=n_circumference, n_axial=n_axial,
            noise_amplitude=noise_amplitude, noise_seed=noise_seed)
        centreline = _geom.straight_centreline(length, n_axial)
        return cls(geometry, centreline, radius, wall_thickness=wall_thickness,
                   artery_type="straight", length=length,
                   noise_amplitude=noise_amplitude, noise_seed=noise_seed)

    @classmethod
    def curved(cls,
               radius: float = 1.5,
               length: float = 30.0,
               *,
               bend_radius: float = 20.0,
               bend_angle_deg: float = 45.0,
               wall_thickness: float = 0.0,
               n_circumference: int = 32,
               n_axial: int = 100,
               noise_amplitude: float = 0.0,
               noise_seed: int | None = None) -> "Artery":
        """
        Build an artery along a straight → arc → straight centreline.

        :param radius: Lumen radius, in mm.
        :param length: Total artery length along the centreline, in mm.
        :param bend_radius: Radius of the circular arc, in mm.
        :param bend_angle_deg: Total bend angle, in degrees.
        :param wall_thickness: Wall thickness, in mm. ``0`` builds the lumen
            surface only, with no separate wall.
        :param n_circumference: Number of vertices around each cross-section.
        :param n_axial: Number of cross-sections along the length.
        :param noise_amplitude: Fractional wall-roughness noise, as a fraction
            of the radius.
        :param noise_seed: Seed for the wall noise. ``None`` draws a fresh
            pattern each call.
        :raises ValueError: If the arc alone is longer than ``length``.
        :returns: The generated artery.
        """
        geometry = _geom.generate_curved_artery(
            radius=radius, length=length, bend_radius=bend_radius,
            bend_angle_deg=bend_angle_deg, wall_thickness=wall_thickness,
            n_circumference=n_circumference, n_axial=n_axial,
            noise_amplitude=noise_amplitude, noise_seed=noise_seed)
        centreline = _geom.curved_centreline(length, bend_radius,
                                             bend_angle_deg, n_axial)
        return cls(geometry, centreline, radius, wall_thickness=wall_thickness,
                   artery_type="curved", length=length,
                   bend_radius=bend_radius, bend_angle_deg=bend_angle_deg,
                   noise_amplitude=noise_amplitude, noise_seed=noise_seed)

    @classmethod
    def s_bend(cls,
               radius: float = 1.5,
               length: float = 40.0,
               *,
               bend_radius: float = 25.0,
               bend_angle_deg: float = 25.0,
               wall_thickness: float = 0.0,
               n_circumference: int = 32,
               n_axial: int = 150,
               noise_amplitude: float = 0.0,
               noise_seed: int | None = None) -> "Artery":
        """
        Build an artery along an S-shaped centreline (two opposite bends).

        ``n_axial`` is only a target here: the S-bend centreline splits it
        across five segments, so the actual point count can come out slightly
        different.

        :param radius: Lumen radius, in mm.
        :param length: Total artery length along the centreline, in mm.
        :param bend_radius: Radius of each circular arc, in mm.
        :param bend_angle_deg: Bend angle of each arc, in degrees.
        :param wall_thickness: Wall thickness, in mm. ``0`` builds the lumen
            surface only, with no separate wall.
        :param n_circumference: Number of vertices around each cross-section.
        :param n_axial: Target number of cross-sections along the length.
        :param noise_amplitude: Fractional wall-roughness noise, as a fraction
            of the radius.
        :param noise_seed: Seed for the wall noise. ``None`` draws a fresh
            pattern each call.
        :raises ValueError: If the two arcs alone are longer than ``length``.
        :returns: The generated artery.
        """
        geometry = _geom.generate_s_bend_artery(
            radius=radius, length=length, bend_radius=bend_radius,
            bend_angle_deg=bend_angle_deg, wall_thickness=wall_thickness,
            n_circumference=n_circumference, n_axial=n_axial,
            noise_amplitude=noise_amplitude, noise_seed=noise_seed)
        centreline = _geom.s_bend_centreline(length, bend_radius,
                                             bend_angle_deg, n_axial)
        return cls(geometry, centreline, radius, wall_thickness=wall_thickness,
                   artery_type="s_bend", length=length,
                   bend_radius=bend_radius, bend_angle_deg=bend_angle_deg,
                   noise_amplitude=noise_amplitude, noise_seed=noise_seed)

    # ------------------------------------------------------------------
    # Solid meshing
    # ------------------------------------------------------------------

    def mesh_solid(self,
                   *,
                   out_path: str | Path,
                   element_size: float,
                   mesh_type: str = "HEX8",
                   noise_amplitude: float = 0.15,
                   noise_seed: float = 0,
                   youngs_modulus: float = 2.0,
                   poisson_ratio: float = 0.3,
                   density: float = 1.0,
                   material_id: int = 1) -> Path:
        """
        Mesh the artery wall as a hollow 3D solid with GMSH and write a 4C ``.yaml``.

        Meshes the annulus between :attr:`radius` and :attr:`r_outer` as a
        straight tube, classifies its boundary nodes into ``DSURFACE`` sets
        (``1`` = lumen, ``2`` = inlet, ``3`` = outlet), then warps the whole
        tube onto :attr:`centreline` using the same frame convention as the
        stent warp, so beam and solid stay aligned. Writes the mesh with a
        placeholder ``MAT_Struct_StVenantKirchhoff`` material.

        Stores the written path on :attr:`solid_yaml`, which
        :meth:`~stentfit.simulation.Simulation.assemble` reads back.

        :param out_path: File path the 4C ``.yaml`` solid is written to.
        :param element_size: Target element size, in mm.
        :param mesh_type: Element type: ``'TET4'``, ``'TET10'``, or ``'HEX8'``.
        :param noise_amplitude: Fractional radial wall-roughness noise.
        :param noise_seed: Seed for the wall noise.
        :param youngs_modulus: Placeholder material Young's modulus, in MPa.
        :param poisson_ratio: Placeholder material Poisson's ratio.
        :param density: Placeholder material density.
        :param material_id: Material ID written into the 4C input.
        :raises ValueError: If the wall has no thickness, or ``mesh_type`` is
            not supported.
        :returns: The path written, also stored on :attr:`solid_yaml`.
        """
        self.solid_yaml = _geom.mesh_artery_gmsh(
            r_inner=self.radius,
            r_outer=self.r_outer,
            centreline=self.centreline,
            out_path=out_path,
            mesh_type=mesh_type,
            element_size=element_size,
            noise_amplitude=noise_amplitude,
            noise_seed=noise_seed,
            material_id=material_id,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            density=density,
        )
        return self.solid_yaml

    def __repr__(self) -> str:
        """:returns: A short summary of the artery's shape and size."""
        kind = self.artery_type or "custom"
        meshed = "" if self.solid_yaml is None else f", solid -> {self.solid_yaml.name}"
        return (f"<Artery {kind} r={self.radius:.3f} wall={self.wall_thickness:.3f} "
                f"arc={self.arc_length:.2f}mm{meshed}>")

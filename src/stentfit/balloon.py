"""The balloon that opens the stent."""

from pathlib import Path

import numpy as np

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function

from .core import balloon_geom as _geom
from .sim.coupling import coupling_limits
from .sim.settings import resolve_strut_thickness
from .stent import Stent

#: Balloon wall thickness, in mm, from Datz et al.
#:
#: A catheter balloon's wall is a manufacturing property and it does not scale with the stent it
#: carries, so the value is used as it is. It is about a third of the strut diameter, which keeps
#: the balloon a membrane a strut can dent, and that local give is what lets the stent redistribute
#: load.
DEFAULT_WALL = 0.04

#: Fewest elements around the circumference before the tube stops behaving as a tube.
#:
#: The mesh is a prism with flat facets, and a facet carries load by bending where a membrane
#: carries it by stretching. Below about a dozen facets the inscribed polygon falls well under the
#: true radius, the circumferential fibre direction stops turning with the surface, and a beam
#: landing on an edge between two facets flips between them from one Newton iteration to the next.
#:
#: The symptom is not a crash. The residual converges while the displacement increment does not, so
#: the solve stalls until it runs out of iterations, with nothing saying the mesh was the problem.
#: That is why it is checked at build time.
MIN_CIRCUMFERENTIAL_ELEMENTS = 12

#: Largest wall thickness, as a fraction of the outer radius, that still behaves as a membrane.
#:
#: Bending stiffness goes as the thickness cubed, so past roughly twice the paper's ratio the wall
#: resists a strut pressing into it instead of denting.
MAX_WALL_PER_RADIUS = 0.15

BALLOON_MATERIALS = ("orthotropic", "isotropic")

#: How the load varies over the analysis, as an expression in ``t`` peaking at 1.
#:
#: ``peak_rate`` is how much faster the profile moves at its quickest than a one-way ramp covering
#: the same distance. It sizes the per-step advance reported in the run record.
#:
#: ==========================  =====================  =========  ==============================
#: name                        expression             peak_rate  shape
#: ==========================  =====================  =========  ==============================
#: ``ramp_inflate``            ``t``                        1.0  up, and stays there
#: ``ramp_inflate_deflate``    ``1-fabs(2*t-1)``            2.0  straight up, straight down
#: ``parabola_inflate_deflate`` ``4*t*(1-t)``               4.0  smooth up and down
#: ==========================  =====================  =========  ==============================
#:
#: ``ramp_inflate_deflate`` is the paper's. A triangle changes at a constant rate throughout, so
#: the approach to the unloaded end state is as gentle as the rest of the run. The parabola is
#: steepest at ``t = 0`` and ``t = 1``, so it arrives at the unloaded state faster than anywhere
#: else, which is where the contact segmentation has the least room to work.
#:
#: The spelling is ``fabs`` rather than ``abs``, because that is what 4C's expression parser uses.
LOAD_PROFILES = {
    "ramp_inflate": {"expr": "t", "peak_rate": 1.0},
    "ramp_inflate_deflate": {"expr": "1-fabs(2*t-1)", "peak_rate": 2.0},
    "parabola_inflate_deflate": {"expr": "4*t*(1-t)", "peak_rate": 4.0},
}


class Balloon:
    """
    A catheter balloon sized to sit just inside a stent, and the pressure that inflates it.

    It is driven by a follower pressure on its inner surface with its ends on springs, so the
    radius it reaches is an outcome of the solve rather than something imposed.

    It is built against a stent the same way :class:`~stentfit.artery.Artery` is::

        balloon = Balloon(stent, material="orthotropic")
        balloon.mesh_solid(out_path, stent_inner_face=..., coupling=...)

    The two length scales come from different places. The in-plane element size follows from the
    coupling rule, so the number of elements around and along the tube is set by the stent's strut
    thickness. The wall thickness is a property of the balloon itself, and it is much thinner than
    an element is wide.

    :param stent: The stent this balloon goes inside. Its skeletonisation must have run.
    :param material: ``"orthotropic"`` is the paper's, a Neo-Hooke base with two artificial fibre
        families, one very stiff along the tube and one very soft around it. That contrast is what
        makes a plain tube behave like a folded catheter balloon. ``"isotropic"`` is the same base
        with no fibres.
    :param clearance_frac: Clearance between the balloon's outer surface and the stent's innermost
        strut surface, as a multiple of strut thickness. This is the clearance that results rather
        than a target, because the balloon is placed from the measured innermost surface.
    :param overhang_frac: How far the balloon reaches past each stent end, as a fraction of stent
        length, so the end crowns do not hang over the edge of the contact surface.
    :param wall: Wall thickness in mm. ``None`` uses :data:`DEFAULT_WALL`.
    :param pressure_max: Peak inflation pressure, in MPa. This is the load, given directly, and it
        is also what the paper reports.

        To stretch balloons of different size equally rather than pressurise them equally, work
        from the wall tension instead, where the parameters are set::

            r_outer = features["r_inner"] - clearance_frac * features["strut_thickness"]
            pressure_max = 0.6 * neohooke_youngs * wall / r_outer
    :param end_spring_stiffness: Stiffness of the end supports, in MPa/mm. The paper uses springs
        rather than fixed ends, because it reduces strong element deformation at the boundary and
        because a real balloon is bonded to a catheter shaft rather than clamped. Springs in all
        three directions also stop a pressure-loaded free tube from drifting off.
    :param load_profile: One of :data:`LOAD_PROFILES`. ``"ramp_inflate"`` inflates and stops. The
        two ``*_inflate_deflate`` profiles rise to the peak and return to zero, which is the only
        way to measure recoil, and they differ in how they get there.
    :param neohooke_youngs: The balloon wall's own Young's modulus, in MPa. The pressure is derived
        from this, so raising it raises the pressure with it and the wall is stretched by the same
        fraction of its stiffness. It is also the stiffness the beam-to-solid coupling check
        compares the stent against.
    :param neohooke_poisson: The wall's Poisson's ratio. Exactly zero is deliberate in the paper:
        it stops radial expansion from pulling the tube shorter.
    :param fibre_longitudinal: ``{"k1": ..., "k2": ...}`` for the along-the-tube fibre family.
        ``None`` uses the paper's very stiff k1 = 1000, which is what stops the tube stretching
        lengthwise. Read only when ``material="orthotropic"``.
    :param fibre_circumferential: The same for the around-the-tube family. ``None`` uses the
        paper's very soft k1 = 1.5e-7, which lets the diameter grow almost freely. Ten orders of
        magnitude separate the two families, and that contrast is the mechanism. Swapped, the tube
        stretches instead of inflating.
    :param load_expression: A custom expression in ``t``, overriding ``load_profile``. Must peak
        at 1, and must return to 0 for recoil to be measurable.
    :param peak_rate: How much faster the custom expression moves at its quickest than a steady
        ramp. Required with ``load_expression``, because it sizes the contact regularisation: a
        parabola covering the same distance moves 4x faster than a ramp.
    :raises ValueError: If the material is unknown, if only one of ``load_expression`` and
        ``peak_rate`` is given, or if the stent has not been skeletonised.
    """

    def __init__(self: "Balloon",
                 stent: Stent,
                 material: str = "orthotropic",
                 clearance_frac: float = 0.1,
                 overhang_frac: float = 0.02,
                 wall=None,
                 pressure_max: float = 0.3,
                 end_spring_stiffness: float = 100.0,
                 load_profile: str = "parabola_inflate_deflate",
                 neohooke_youngs: float = _geom.NEOHOOKE_YOUNGS,
                 neohooke_poisson: float = _geom.NEOHOOKE_POISSON,
                 fibre_longitudinal: dict = None,
                 fibre_circumferential: dict = None,
                 load_expression: str = None,
                 peak_rate: float = None):
        if material not in BALLOON_MATERIALS:
            raise ValueError(f"material must be one of {BALLOON_MATERIALS}, got {material!r}")
        custom = load_expression is not None or peak_rate is not None
        if custom and (load_expression is None or peak_rate is None):
            raise ValueError("a custom profile needs both load_expression and peak_rate - "
                             "peak_rate sizes the contact regularisation and cannot be "
                             "guessed from the expression")
        if not custom and load_profile not in LOAD_PROFILES:
            raise ValueError(f"load_profile must be one of {sorted(LOAD_PROFILES)}, "
                             f"got {load_profile!r}")
        if stent.stent_features is None:
            raise ValueError("stent has no features yet - run Stent.skeletonize() (or "
                             "Stent.load()) before building a balloon for it")

        self.stent = stent
        self.material = material
        self.clearance_frac = float(clearance_frac)
        self.overhang_frac = float(overhang_frac)
        self.wall = DEFAULT_WALL if wall is None else float(wall)
        self.pressure = float(pressure_max)
        self.end_spring_stiffness = float(end_spring_stiffness)
        self.load_profile = "custom" if custom else load_profile
        self.neohooke_youngs = float(neohooke_youngs)
        self.neohooke_poisson = float(neohooke_poisson)
        self.fibre_longitudinal = fibre_longitudinal or dict(_geom.FIBRE_LONGITUDINAL)
        self.fibre_circumferential = (fibre_circumferential
                                      or dict(_geom.FIBRE_CIRCUMFERENTIAL))

        # Fixed the moment the balloon exists, so they are plain attributes rather than
        # properties. Only the three that depend on mesh_solid() having run stay properties.
        if custom:
            self.load_expression = load_expression
            self.peak_rate = float(peak_rate)
        else:
            self.load_expression = LOAD_PROFILES[load_profile]["expr"]
            self.peak_rate = LOAD_PROFILES[load_profile]["peak_rate"]
        # The stiffness that actually resists the beam: the rubbery Neo-Hooke base.
        self.effective_youngs = self.neohooke_youngs

        #: Geometry produced by :meth:`mesh_solid`, or ``None`` before it runs.
        self.info = None
        #: Path to the 4C solid ``.yaml``, once :meth:`mesh_solid` has run.
        self.solid_yaml = None

    # ------------------------------------------------------------------
    # Sizing and meshing
    # ------------------------------------------------------------------

    def features(self: "Balloon") -> dict:
        """:returns: The stent's measured features."""
        return self.stent.stent_features

    def mesh_solid(self: "Balloon", out_path, stent_inner_face: float,
                   limits: dict = None, settings=None) -> Path:
        """
        Size the balloon against the stent and write it as a 4C solid.

        The balloon is placed against the stent's measured innermost strut surface rather than
        against the averaged ``r_inner`` from the features file. How far the innermost node sits
        inside that average changes from stent to stent, so a clearance expressed against the
        average would mean a different real gap on each one, and a value that is safe on one stent
        would overlap on another.

        No element count is passed. Both counts follow from the coupling rule, so the strut
        thickness alone fixes the whole mesh. It is read from ``limits["d_beam"]``, which is the
        thickness the stent itself was built at, so the balloon cannot be sized for one thickness
        and placed against a stent of another.

        :param out_path: Where to write the ``.4C.yaml``.
        :param stent_inner_face: Radius of the stent's innermost strut *surface*, in mm, measured
            from the beam mesh by
            :func:`~stentfit.sim.beam_model.innermost_surface_radius`.
        :param limits: Pre-computed coupling limits from
            :func:`~stentfit.sim.coupling.coupling_limits`, which the caller normally already has.
            Also the source of the strut thickness every length here is scaled by.
        :param settings: Used to compute the limits when ``limits`` is not given, and to resolve
            the strut thickness in that case.
        :returns: The path written.
        :raises ValueError: If the wall would reach the axis, or the balloon overlaps the stent.
        """
        features = self.features()
        z_min, z_max = float(features["z_min"]), float(features["z_max"])
        length = z_max - z_min

        if limits is None:
            limits = coupling_limits(resolve_strut_thickness(settings, features), settings)

        # The strut thickness the *stent* was meshed and sectioned at, not the one in the features
        # file. The two are the same until settings.strut_thickness overrides the measurement, and
        # then they differ: the stent's section, its innermost surface and both element sizes all
        # follow the override, so a clearance scaled by the scan's value would place the balloon
        # against a stent of a thickness it was never sized for. Taken from the limits rather than
        # resolved again here, so there is one source and the two cannot drift apart.
        strut = float(limits["d_beam"])
        h = limits["h_solid"]                       # in-plane element size, from the coupling rule

        clearance_target = self.clearance_frac * strut
        r_outer = stent_inner_face - clearance_target
        r_inner = r_outer - self.wall            # a membrane, not a solid element thick
        if r_inner <= 0:
            raise ValueError(f"balloon wall reaches the axis: r_inner={r_inner:.4g} mm - "
                             f"reduce clearance_frac or wall")
        if r_outer >= stent_inner_face:
            raise ValueError(f"balloon overlaps the stent: outer radius {r_outer:.4f} mm against "
                             f"an innermost strut surface at {stent_inner_face:.4f} mm")
        if self.wall / r_outer > MAX_WALL_PER_RADIUS:
            raise ValueError(
                f"balloon wall is {100 * self.wall / r_outer:.0f}% of its outer radius "
                f"({self.wall:.4f} mm against {r_outer:.4f} mm), so it bends where a membrane "
                f"would stretch - at most {100 * MAX_WALL_PER_RADIUS:.0f}%. The radius is "
                f"{stent_inner_face:.4f} mm of stent bore less {clearance_target:.4f} mm of "
                f"clearance, so lower clearance_frac (now {self.clearance_frac:g}) or set "
                f"strut_thickness by hand if the scan measured it too thick")

        overhang = self.overhang_frac * length
        z0, z1 = z_min - overhang, z_max + overhang

        # Element counts follow from the target size, rounded up so elements never exceed it.
        n_circ = int(np.ceil(2.0 * np.pi * r_outer / h))
        n_axial = int(np.ceil((z1 - z0) / h))

        if n_circ < MIN_CIRCUMFERENTIAL_ELEMENTS:
            # h is factor_solid x strut, so a thick strut coarsens the mesh and shrinks the radius
            # at the same time. Both push this the wrong way, which is why the message names the
            # strut as well as the clearance.
            raise ValueError(
                f"balloon has only {n_circ} elements around its circumference, so it is a "
                f"{n_circ}-sided prism rather than a tube - at least "
                f"{MIN_CIRCUMFERENTIAL_ELEMENTS} are needed. Circumference "
                f"{2.0 * np.pi * r_outer:.4f} mm at element size {h:.4f} mm "
                f"({h / strut:.3g} x a {strut:.4f} mm strut). Lower clearance_frac "
                f"(now {self.clearance_frac:g}) to widen the balloon, or set strut_thickness by "
                f"hand to refine the mesh - a crimped scan measures folded struts as one thick "
                f"band and reads several times the true thickness")

        coords, conn, surfaces = _geom.build_balloon(r_inner, r_outer, z0, z1,
                                                     n_circ=n_circ, n_axial=n_axial)

        if self.material == "isotropic":
            material_lines, mat_id = _geom.isotropic_material(
                1, self.neohooke_youngs, self.neohooke_poisson)
        else:
            material_lines, mat_id = _geom.orthotropic_material(
                1, self.neohooke_youngs, self.neohooke_poisson,
                self.fibre_longitudinal, self.fibre_circumferential)
        fibers = _geom.fibre_directions(coords, conn) if self.material == "orthotropic" else None

        path = _geom.write_4c_solid(out_path, coords, conn, surfaces,
                                    material_lines=material_lines, material_id=mat_id,
                                    fibers=fibers)
        vtk = _geom.write_vtu(out_path, coords, conn, fibers)

        h_circ = 2.0 * np.pi * r_outer / n_circ
        h_axial = (z1 - z0) / n_axial
        self.info = {
            "material": self.material,
            "r_inner": r_inner, "r_outer": r_outer, "wall": self.wall,
            "wall_per_strut": self.wall / strut,
            "aspect_ratio": max(h_circ, h_axial) / self.wall,
            "clearance_frac": self.clearance_frac,
            "clearance_target_mm": clearance_target,
            "stent_inner_face_mm": stent_inner_face,
            "gap": float(features["r_inner"]) - r_outer,
            "overhang_frac": self.overhang_frac,
            "z_min": z0, "z_max": z1,
            "n_circ": n_circ, "n_axial": n_axial, "n_radial": 1,
            "h_circ": h_circ, "h_axial": h_axial,
            "n_nodes": len(coords), "n_elements": len(conn),
            "n_inner_surface_nodes": len(surfaces[1]),
            "n_outer_surface_nodes": len(surfaces[2]),
            "has_fibers": fibers is not None,
            "youngs_modulus_MPa": self.effective_youngs,
            "path": path, "vtk": vtk,
        }
        self.solid_yaml = path
        return path

    # ------------------------------------------------------------------
    # Drive
    # ------------------------------------------------------------------

    @property
    def r_outer(self: "Balloon") -> float:
        """
        :returns: The balloon's outer radius, in mm.
        :raises ValueError: If it has not been meshed yet.
        """
        if self.info is None:
            raise ValueError("balloon not meshed yet - call mesh_solid() first")
        return self.info["r_outer"]

    def wall_tension(self: "Balloon") -> float:
        """
        The hoop stress the applied pressure produces, in MPa.

        Thin-wall Laplace, ``sigma = p * r / t``. This is the consequence of the pressure and not
        an input. It is reported so a run can be compared against a balloon of a different size,
        where the same pressure means a different stretch.

        Datz et al. inflate to 13 atm on a balloon of 0.49 mm outer radius and 0.04 mm wall, which
        is 12.4 MPa of tension, or 0.73 times their 17 MPa wall modulus.

        :returns: The hoop stress, in MPa.
        :raises ValueError: If the balloon has not been meshed yet.
        """
        return self.pressure * self.r_outer / self.wall

    def expansion_strain(self: "Balloon", stent_radial_strain: float) -> float:
        """
        How far the balloon must expand to push the stent to a target radius.

        The stent and the balloon do not start at the same radius, so they do not share a strain.
        What has to match is the final position, where the balloon's outer surface arrives at the
        place the stent's inner surface should end up.

        It drives nothing, since the pressure is the load. It is used only to report the target and
        to size the contact regularisation.

        :param stent_radial_strain: Target stent expansion, as a fraction of its starting radius.
        :returns: The balloon's radial strain at peak inflation.
        """
        target = float(self.features()["r_inner"]) * (1.0 + stent_radial_strain)
        return target / self.r_outer - 1.0

    def travel(self: "Balloon", stent_radial_strain: float) -> float:
        """
        :param stent_radial_strain: Target stent expansion.
        :returns: Full radial travel of the balloon surface, in mm.
        """
        return self.r_outer * self.expansion_strain(stent_radial_strain)

    def advance_per_step(self: "Balloon", stent_radial_strain: float, n_steps: int) -> float:
        """
        Largest per-step advance of the balloon surface, in mm.

        This is not simply travel over steps. A profile that goes out and back covers the distance
        twice in the same time, and a parabola is four times faster than a one-way ramp at its
        steepest. See :data:`LOAD_PROFILES`.

        :param stent_radial_strain: Target stent expansion.
        :param n_steps: Number of load steps.
        :returns: The advance, in mm.
        """
        return self.travel(stent_radial_strain) * self.peak_rate / n_steps

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def add_pressure(self: "Balloon", mesh, inner_surface) -> None:
        """
        Inflate the balloon with a follower pressure on its inner surface.

        ``TYPE: orthopressure`` keeps the load normal to the surface as it deforms. On a tube that
        nearly doubles its diameter this matters, because a fixed-direction load would point the
        wrong way over most of the circumference.

        The sign is negative because 4C's surface normal points out of the element, which on an
        inner surface means inwards towards the axis, so a negative pressure along it pushes the
        wall outwards.

        The caller must also set ``LOADLIN``. A follower load depends on the displacement, so it
        contributes to the stiffness matrix, and 4C refuses to run ``orthopressure`` without it.

        :param mesh: The combined mesh, modified in place.
        :param inner_surface: Geometry set for the balloon's inner surface (``DSURFACE 1``).
        """
        ramp = Function([{"SYMBOLIC_FUNCTION_OF_SPACE_TIME": self.load_expression}])
        mesh.add(ramp)
        mesh.add(BoundaryCondition(
            inner_surface,
            {"NUMDOF": 6, "ONOFF": [1, 0, 0, 0, 0, 0],
             "VAL": [-self.pressure, 0, 0, 0, 0, 0],
             "FUNCT": [ramp, 0, 0, 0, 0, 0],
             "TYPE": "orthopressure"},
            bc_type=bme.bc.neumann))

    def add_end_springs(self: "Balloon", mesh, end_surfaces) -> None:
        """
        Support the balloon's two end caps on springs.

        BeamMe has no enum for this condition, but a raw section name can be passed as ``bc_type``
        and is written verbatim, which is the route ``beam_potential`` uses.

        ``DIRECTION: xyz`` makes the springs act in all three directions rather than only along the
        surface normal. That is the paper's omnidirectional support, and it also pins down the
        rigid-body modes of a tube loaded by pressure alone.

        :param mesh: The combined mesh, modified in place.
        :param end_surfaces: Geometry sets for the two end caps (``DSURFACE 3`` and ``4``).
        """
        for surface in end_surfaces:
            mesh.add(BoundaryCondition(
                surface,
                {"NUMDOF": 3, "ONOFF": [1, 1, 1],
                 "STIFF": [self.end_spring_stiffness] * 3, "TIMEFUNCTSTIFF": [0, 0, 0],
                 "VISCO": [0.0] * 3, "TIMEFUNCTVISCO": [0, 0, 0],
                 "DISPLOFFSET": [0.0] * 3, "TIMEFUNCTDISPLOFFSET": [0, 0, 0],
                 "FUNCTNONLINSTIFF": [0, 0, 0],
                 "DIRECTION": "xyz"},
                bc_type="DESIGN SURF ROBIN SPRING DASHPOT CONDITIONS"))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self: "Balloon") -> list:
        """:returns: The balloon as printable lines, once meshed."""
        if self.info is None:
            return [f"balloon: {self.material} (not meshed yet)"]
        i = self.info
        lines = [
            f"balloon: {self.material}, radii {i['r_inner']:.4f} .. {i['r_outer']:.4f} mm, "
            f"wall {i['wall']:.4f} mm",
            f"  mesh          {i['n_elements']:,} HEX8 "
            f"({i['n_circ']} x {i['n_axial']} x 1), aspect {i['aspect_ratio']:.1f}:1",
            f"  element size  {i['h_circ']:.4f} circ, {i['h_axial']:.4f} axial, "
            f"{i['wall']:.4f} radial mm",
            f"  clearance     {i['clearance_target_mm']:.5f} mm "
            f"({self.clearance_frac:g} x strut) from the measured inner face "
            f"{i['stent_inner_face_mm']:.5f} mm",
        ]
        lines += [f"  pressure      {self.pressure:.4f} MPa peak "
                  f"(wall tension {self.wall_tension():.2f} MPa = "
                  f"{self.wall_tension() / self.neohooke_youngs:.2f} x the "
                  f"{self.neohooke_youngs:g} MPa wall)",
                  f"  ends          {self.end_spring_stiffness:g} MPa/mm springs",
                  f"  profile       {self.load_profile}, scale(t) = {self.load_expression}"]
        return lines

    def to_dict(self: "Balloon") -> dict:
        """:returns: Everything about this balloon, for the run record."""
        record = {
            "material": self.material,
            "clearance_frac": self.clearance_frac, "overhang_frac": self.overhang_frac,
            "wall_mm": self.wall, "load_profile": self.load_profile,
            "load_expression": self.load_expression,
        }
        if self.info:
            record |= {
                "r_inner_mm": round(self.info["r_inner"], 5),
                "r_outer_mm": round(self.info["r_outer"], 5),
                "wall_per_strut_thickness": round(self.info["wall_per_strut"], 3),
                "element_aspect_ratio": round(self.info["aspect_ratio"], 2),
                "clearance_target_mm": round(self.info["clearance_target_mm"], 5),
                "stent_inner_face_mm": round(self.info["stent_inner_face_mm"], 5),
                "gap_mm": round(self.info["gap"], 5),
                "n_circumferential": self.info["n_circ"], "n_axial": self.info["n_axial"],
                "n_radial": 1, "n_elements": self.info["n_elements"],
                "element_size_mm": {"circumferential": round(self.info["h_circ"], 5),
                                    "axial": round(self.info["h_axial"], 5),
                                    "radial": round(self.wall, 5)},
                "youngs_modulus_MPa": self.effective_youngs,
            }
        record |= {
            "pressure_max_MPa": self.pressure,
            "wall_tension_MPa": (round(self.wall_tension(), 4) if self.info else None),
            "wall_tension_per_youngs": (round(self.wall_tension() / self.neohooke_youngs, 4)
                                        if self.info else None),
            "end_spring_stiffness_MPa_per_mm": self.end_spring_stiffness,
            "neohooke_youngs_MPa": self.neohooke_youngs,
            "neohooke_poisson": self.neohooke_poisson,
            "fibre_longitudinal": self.fibre_longitudinal,
            "fibre_circumferential": self.fibre_circumferential,
        }
        return record

    def __repr__(self: "Balloon") -> str:
        """:returns: A short summary."""
        where = (f"r {self.info['r_outer']:.3f} mm, {self.info['n_elements']:,} HEX8"
                 if self.info else "not meshed")
        return f"<Balloon {self.material} [{where}]>"

"""
Every parameter of a simulation, one flat class per type.

Each simulation type has one settings class, holding exactly the parameters it needs. Nothing is
shared between the types, so setting one field never moves another::

    settings = StentBalloonSettings(penalty_law="linear_quadratic", n_steps=200)
    sim = Simulation(stent, sim_type="stent_balloon", settings=settings)

The classes are frozen, so a settings object cannot change behind a run that is already using it.
Use :func:`dataclasses.replace` to make a modified copy.
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any

# --------------------------------------------------------------------------------------
# Round-trip, for the run record
# --------------------------------------------------------------------------------------


def settings_to_dict(settings) -> dict:
    """
    Turn a settings object into a plain dict, ready for YAML.

    :param settings: Any settings or config object in this package.
    :returns: Its fields as a dict.
    """
    return asdict(settings)


def settings_from_dict(settings_class, data: dict | None):
    """
    Rebuild a settings object from a recorded dict.

    Unknown keys are ignored and missing ones fall back to the default, so a record written by an
    earlier version can still be read.

    :param settings_class: The class to build, e.g. :class:`StentBalloonSettings`.
    :param data: A dict from a ``run_parameters.yaml``. ``None`` gives the defaults.
    :returns: The settings object.
    """
    if not data:
        return settings_class()

    known = {f.name for f in fields(settings_class)}
    return settings_class(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------------------
# Values shared by every type, checked the same way
# --------------------------------------------------------------------------------------

BEAM_MATERIALS = ("elastic", "elastoplastic")
BEAM_CLASSES = ("Beam3rHerm2Line3", "Beam3rLine2Line2")


def _check_beam(settings) -> None:
    """
    Validate the strut fields every type carries.

    :param settings: A settings object with the beam fields.
    :raises ValueError: If any value is impossible.
    """
    if settings.material not in BEAM_MATERIALS:
        raise ValueError(f"material must be one of {BEAM_MATERIALS}, got {settings.material!r}")
    if settings.beam_class not in BEAM_CLASSES:
        raise ValueError(f"beam_class must be one of {BEAM_CLASSES}, got {settings.beam_class!r}")
    if settings.youngs <= 0:
        raise ValueError(f"youngs must be positive, got {settings.youngs}")
    if not 0.0 <= settings.poisson < 0.5:
        raise ValueError(f"poisson must be in [0, 0.5), got {settings.poisson}")
    if settings.l_el_per_strut <= 0:
        raise ValueError(f"l_el_per_strut must be positive, got {settings.l_el_per_strut}")
    if settings.strut_thickness is not None and settings.strut_thickness <= 0:
        raise ValueError(f"strut_thickness must be positive when set, "
                         f"got {settings.strut_thickness}")
    if settings.material == "elastoplastic":
        if settings.yield_strength <= 0:
            raise ValueError(f"yield_strength must be positive for an elastoplastic material, "
                             f"got {settings.yield_strength}")
        if not 0.0 < settings.tangent_modulus_ratio < 1.0:
            raise ValueError(f"tangent_modulus_ratio must be in (0, 1), got "
                             f"{settings.tangent_modulus_ratio}")


def _check_solver(settings) -> None:
    """
    Validate the solver fields every type carries.

    :param settings: A settings object with the solver fields.
    :raises ValueError: If any value is impossible.
    """
    if settings.n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {settings.n_steps}")
    if settings.max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {settings.max_iter}")
    if settings.total_time <= 0:
        raise ValueError(f"total_time must be positive, got {settings.total_time}")
    if settings.tol_residuum <= 0:
        raise ValueError(f"tol_residuum must be positive, got {settings.tol_residuum}")


def resolve_strut_thickness(settings, features) -> float:
    """
    The one strut thickness a run is built from.

    ``settings.strut_thickness`` is the value to use, and ``None`` means the one the skeletonisation
    measured, which is the usual case.

    Every length in a simulation is scaled by the strut thickness, so this is the one place that
    decides which value that is.

    :param settings: Any settings object.
    :param features: The parsed ``stent_features.json``.
    :returns: The strut thickness, in mm.
    """
    if getattr(settings, "strut_thickness", None) is not None:
        return float(settings.strut_thickness)
    return float(features["strut_thickness"])


def element_length(settings, strut_thickness: float) -> float:
    """
    Target beam element length for a stent, in mm.

    The length is a multiple of the strut thickness rather than an absolute number, so a stent with
    thicker struts is resolved just as well without changing the setting.

    :param settings: Any settings object.
    :param strut_thickness: The stent's strut thickness, in mm.
    :returns: The element length, in mm.
    """
    return settings.l_el_per_strut * strut_thickness


# --------------------------------------------------------------------------------------
# Stent only
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StentOnlySettings:
    """
    Everything the stent-only simulation needs.

    The beam mesh is driven by prescribed displacements, with no artery and no balloon, and
    beam-to-beam contact stops the struts passing through each other.

    The defaults are the elastic ones. An elastoplastic run needs a different solver as well as a
    different material, so use :data:`STENT_ONLY_PLASTIC` rather than changing ``material`` alone.

    :param material: ``"elastic"`` or ``"elastoplastic"``. An elastic strut stays proportional
        however hard it is loaded, so it cannot show yielding or permanent deformation. An
        elastoplastic one bends elastically up to the yield moment and then hinges.
    :param youngs: Young's modulus, in MPa. 200 GPa is 316L and CoCr to within a few percent.
    :param poisson: Poisson's ratio.
    :param density: Mass density. Zero, since the analysis is static.
    :param yield_strength: Yield strength in MPa, read only when the material is elastoplastic.
        Follows Datz et al. (Comput. Biol. Med. 189 (2025) 109914).
    :param tangent_modulus_ratio: Post-yield tangent modulus, as a fraction of ``youngs``. Datz et
        al. use E = 380 GPa and E_ep = 64 GPa, which is this ratio.
    :param beam_class: How each strut is interpolated along its length. This is the element
        formulation and not the material, and the two are chosen independently.
    :param l_el_per_strut: Beam element length, as a multiple of the strut thickness. Only
        ``stent_only`` reads it. The types with a solid body derive the beam length from the solid
        instead, as ``factor_beam x factor_solid x strut``, because the coupling rules constrain
        the ratio between the two meshes. Refine those with ``factor_beam``.
    :param strut_thickness: The strut thickness the simulation uses, in mm. ``None`` means the
        measured value, which is the usual case. Everything scaled by the strut follows it, through
        :func:`resolve_strut_thickness`. Worth setting by hand on a stent scanned crimped, where
        the folded struts make the measured band several struts deep.
    :param cases: Which load cases to build. Empty builds both.
    :param strains: Overrides a case's strain, e.g. ``{"radial_expand": 0.60}``. The defaults are
        ``radial_expand`` +0.50 and ``axial_stretch`` +0.10, both engineering strains at ``t = 1``.
    :param self_contact: Beam-to-beam penalty contact between the struts. ``False`` leaves nothing
        stopping them passing through each other.
    :param contact_penalty_frac: Contact penalty, as a fraction of ``E * A / l_el``. Too high and
        Newton stalls when a pair comes into contact, too low and the struts sink into each other.
        Lower it first if a run diverges.
    :param contact_g0_per_strut: The gap over which the contact force eases in, as a fraction of
        the strut thickness. A hard switch at zero gap is what makes penalty contact stall. 4C
        measures the gap surface to surface, so struts closer than ``1 + this`` strut thicknesses
        already feel a force at ``t = 0``, and the build prints the worst case.
    :param contact_exclusion_per_strut: Elements with a node this close to a welded crown are kept
        out of the contact set, in strut thicknesses. The crowns are coincident but separate nodes,
        so without this 4C reads every crown as touching at ``t = 0``. Past about one strut
        thickness it only removes struts from the contact set.
    :param grip_frac: How much of the tip ring the axial cases clamp at each end, standing in for a
        test machine's grips. It is a fraction of the tip ring's own length, so it scales with the
        crown spacing rather than the total length.
    :param n_steps: Load steps. 20 is plenty for an elastic run, which is path-independent.
        Plasticity is not, so it needs far more.
    :param total_time: Total pseudo-time for the static analysis.
    :param max_iter: Newton iteration cap per step.
    :param tol_residuum: Force residual tolerance, scaled 2-norm.
    :param tol_increment: Displacement update tolerance, scaled 2-norm. ``None`` leaves BeamMe's
        default.
    :param predictor: Starting guess for each Newton solve. ``"TangDis"`` extrapolates along the
        tangent stiffness and is fast while the response is smooth, but it overshoots once the
        struts yield. Use ``"ConstDis"`` for a plastic run, which starts from the last converged
        state.
    :param line_search: ``"Full Step"`` always takes the whole Newton step. ``"Backtrack"`` halves
        a step that makes the residual worse.
    """

    simulation_type = "stent_only"

    # --- 1. what to simulate ---------------------------------------------------------
    material: str = "elastic"
    cases: tuple = ()

    # --- 2. stent material -----------------------------------------------------------
    youngs: float = 2.0e5
    poisson: float = 0.3
    density: float = 0.0
    yield_strength: float = 300.0
    tangent_modulus_ratio: float = 64.0 / 380.0

    # --- 3. mesh ---------------------------------------------------------------------
    beam_class: str = "Beam3rHerm2Line3"
    l_el_per_strut: float = 1.0
    strut_thickness: float = None

    # --- 4. load cases ---------------------------------------------------------------
    strains: dict = field(default_factory=dict)
    grip_frac: float = 0.2

    # --- 4b. self-contact ------------------------------------------------------------
    self_contact: bool = True
    contact_penalty_frac: float = 0.01
    contact_g0_per_strut: float = 0.05
    contact_exclusion_per_strut: float = 0.5

    # --- 5. solver -------------------------------------------------------------------
    n_steps: int = 20
    total_time: float = 1.0
    max_iter: int = 20
    tol_residuum: float = 1e-8
    tol_increment: float = None
    predictor: str = "TangDis"
    line_search: str = "Full Step"

    def __post_init__(self: "StentOnlySettings") -> None:
        # YAML has no tuple, so a recorded run reads ``cases`` back as a list. Normalising here
        # means a run rebuilt from its record compares equal to the settings that produced it.
        object.__setattr__(self, "cases", tuple(self.cases))

        _check_beam(self)
        _check_solver(self)
        if not 0.0 < self.grip_frac <= 1.0:
            raise ValueError(f"grip_frac must be in (0, 1], got {self.grip_frac}")

        for name in ("contact_penalty_frac", "contact_g0_per_strut",
                     "contact_exclusion_per_strut"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative, got {getattr(self, name)}")


#: The stent-only settings that produced the converged elastic runs. The same as the class
#: defaults, named so the two materials can be read side by side.
STENT_ONLY_ELASTIC = StentOnlySettings(
    material="elastic",
    n_steps=20, max_iter=20, tol_residuum=1e-8,
    predictor="TangDis", line_search="Full Step")

#: The stent-only settings for an elastoplastic run. The solver changes together with the
#: material, so the two are set here as one.
#:
#: These runs are not expected to converge, because prescribing every node is over-constrained
#: once the struts yield.
STENT_ONLY_PLASTIC = StentOnlySettings(
    material="elastoplastic",
    n_steps=200, max_iter=20, tol_residuum=1e-8,
    predictor="ConstDis", line_search="Backtrack")


# --------------------------------------------------------------------------------------
# Stent and balloon
# --------------------------------------------------------------------------------------

BALLOON_MATERIALS = ("orthotropic", "isotropic")
LOAD_PROFILES = ("ramp_inflate", "ramp_inflate_deflate", "parabola_inflate_deflate")
CONTACT_DISCRETIZATIONS = ("gauss_point_to_segment", "mortar")


@dataclass(frozen=True)
class StentBalloonSettings:
    """
    Everything the stent-balloon contact simulation needs.

    The stent is opened by contact with a pressure-inflated balloon rather than by prescribing
    where its nodes must go, and the modelling follows Datz et al. (Comput. Biol. Med. 189 (2025)
    109914). The balloon is driven by a follower pressure with its ends on springs, so the radius
    it reaches is an outcome of the solve and not something imposed.

    The defaults are the ones that completed all 200 steps.

    :param material: The struts' material law, as for :class:`StentOnlySettings`.
    :param balloon_material: ``"orthotropic"`` is the paper's, a Neo-Hooke base with two artificial
        fibre families, one very stiff along the tube and one very soft around it. That contrast is
        what makes a plain tube behave like a folded catheter balloon. ``"isotropic"`` is the same
        base with no fibres.
    :param load_profile: How the pressure varies over the analysis. ``"ramp_inflate"`` goes up and
        stays there. The two ``*_inflate_deflate`` profiles return to zero, which is the only way
        to measure recoil. ``"ramp_inflate_deflate"`` is the paper's triangle, changing at a
        constant rate. ``"parabola_inflate_deflate"`` is smooth but steepest at the two ends, so it
        reaches the unloaded state faster than anywhere else in the run.
    :param youngs: Strut Young's modulus, in MPa.
    :param poisson: Strut Poisson's ratio.
    :param density: Strut density.
    :param yield_strength: Strut yield strength in MPa, read only when elastoplastic. The paper's
        values for a plastic run are ``youngs=3.8e5`` and ``yield_strength=471.6``, calibrated to
        stand in for the kinematic hardening of crimping.
    :param tangent_modulus_ratio: Post-yield tangent modulus, as a fraction of ``youngs``.
    :param beam_class: Beam element formulation.
    :param l_el_per_strut: Unused here. The beam element length comes from the coupling rule
        instead, so the strut thickness alone fixes both meshes. Kept so every settings class
        carries the same beam fields.
    :param strut_thickness: The strut thickness the simulation uses, in mm. ``None`` means the
        measured value.
    :param clearance_frac: Gap between the balloon's outer surface and the stent's innermost strut
        surface, as a multiple of strut thickness. It is measured from the beam mesh rather than
        from the averaged ``r_inner``, because how far the innermost node sits inside that average
        changes from stent to stent, so the same number would mean a different real gap on each one.
    :param overhang_frac: How far the balloon reaches past each stent end, as a fraction of stent
        length, so the end crowns do not hang over the edge of the contact surface.
    :param wall: Balloon wall thickness in mm, from the paper. It is roughly a third of the strut
        diameter, and thinner than one solid element on purpose, because the balloon has to behave
        as a membrane a strut can dent rather than as a rigid shell.
    :param radial_strain: How far the stent is meant to open, as a fraction of its starting radius.
        It drives nothing, since the pressure is the load. It is only the target the run is
        reported against.
    :param pressure_max: Peak inflation pressure, in MPa. This is the load.

        The paper's 1.01 MPa does not transfer directly, because their balloon starts at a much
        smaller radius. What transfers between balloons of different size is the wall tension, and
        :meth:`~stentfit.balloon.Balloon.wall_tension` reports it.
    :param end_spring_stiffness: Balloon end supports, in MPa/mm. The paper uses springs rather
        than fixed ends, because it reduces strong element deformation at the boundary and because
        a real balloon is bonded to a catheter shaft rather than clamped.
    :param neohooke_youngs: The balloon wall's own Young's modulus, in MPa. This is the stiffness
        the coupling check compares the stent against.
    :param neohooke_poisson: The wall's Poisson's ratio. Exactly zero stops the radial expansion
        pulling the tube shorter.
    :param fibre_longitudinal: ``{"k1": ..., "k2": ...}`` for the along-the-tube fibre family.
    :param fibre_circumferential: The same for the around-the-tube family. Ten orders of magnitude
        separate the two, and that contrast is the mechanism. Swapped, the tube stretches instead
        of inflating.
    :param penalty: How hard contact pushes back, in N/mm^2, from the paper.
    :param penalty_law: ``"linear"`` or ``"linear_quadratic"``.
    :param penalty_g0_per_strut: Width of the zone over which the contact stiffness eases in, as a
        multiple of the strut diameter. At 0 contact is a hard on/off switch and the whole
        transition happens inside one load step, which is what makes the final zero-load step of an
        inflate-deflate run fail. It is a physical length rather than a count of load steps, so
        changing ``n_steps`` or the load profile does not change the contact behaviour with it.
    :param discretization: Where along the beam the no-overlap condition is checked.
        ``"gauss_point_to_segment"`` measures the gap at each Gauss point, so each point becomes
        three constraints. ``"mortar"`` carries a traction field along the beam and satisfies the
        condition in a weighted sense, which guards against the locking Steinbrecher et al. report
        when the beam element length approaches the solid element size.
    :param gauss_points: Quadrature points per beam element. Under GPTS this sets the number of
        constraints, so raising it stiffens the coupling. Under mortar the count comes from
        ``mortar_shape_function`` and this only controls integration accuracy, which is why mortar
        does not lock.
    :param contact_type: Where the contact force comes from. 4C's own default passes schema
        validation and then aborts at runtime, so this always has to be set.
    :param constraint_strategy: How the constraint is enforced.
    :param geometry_pair_strategy: ``"segmentation"`` cuts the beam where it crosses between solid
        elements before integrating, because the integrand has kinks there.
    :param mortar_shape_function: Read only under mortar. ``"line2"`` is the linear multiplier
        interpolation Steinbrecher et al. measured as locking-free.
    :param mortar_contact_defined_in: Read only under mortar. The reference configuration does not
        move, so the weights are computed once instead of every Newton iteration.
    :param factor_solid: Solid element size, as a multiple of the beam cross-section diameter.
    :param factor_beam: Beam element length, as a multiple of the solid element size. Both are
        margins above a limit rather than the limit itself, because the meshers divide a curve or a
        circumference into a whole number of elements, so the result scatters around the target.
    :param n_steps: Load steps. The base is 100, doubled because an inflate-deflate profile goes
        out and back. A plastic run wants double again, so the yielding is followed as it spreads.
    :param total_time: Total pseudo-time.
    :param max_iter: Newton iteration cap. It is generous at 60 because most steps use 3 to 6
        iterations and the occasional one needs about 40.
    :param tol_residuum: Force residual tolerance.
    :param tol_increment: Displacement update tolerance. Set explicitly here, unlike the stent-only
        case, because contact makes the displacement update a meaningful second criterion.
    :param predictor: Starting guess for each Newton solve.
    :param line_search: How far along the Newton direction to move.
    """

    simulation_type = "stent_balloon"

    # --- 1. what to simulate ---------------------------------------------------------
    material: str = "elastic"
    balloon_material: str = "orthotropic"
    load_profile: str = "parabola_inflate_deflate"

    # --- 2. stent material -----------------------------------------------------------
    youngs: float = 2.0e5
    poisson: float = 0.3
    density: float = 0.0
    yield_strength: float = 300.0
    tangent_modulus_ratio: float = 64.0 / 380.0
    beam_class: str = "Beam3rHerm2Line3"
    l_el_per_strut: float = 1.0
    strut_thickness: float = None

    # --- 3. balloon shape ------------------------------------------------------------
    clearance_frac: float = 1.0
    overhang_frac: float = 0.1
    wall: float = 0.04
    radial_strain: float = 0.50

    # --- 4. balloon loading ----------------------------------------------------------
    pressure_max: float = 0.3
    end_spring_stiffness: float = 100.0
    neohooke_youngs: float = 17.0
    neohooke_poisson: float = 0.0
    fibre_longitudinal: dict = field(default_factory=lambda: {"k1": 1000.0, "k2": 0.01})
    fibre_circumferential: dict = field(default_factory=lambda: {"k1": 1.5e-7, "k2": 0.35})

    # --- 5. contact ------------------------------------------------------------------
    penalty: float = 10.0
    penalty_law: str = "linear_quadratic"
    penalty_g0_per_strut: float = 0.5
    discretization: str = "gauss_point_to_segment"
    gauss_points: int = 6
    contact_type: str = "gap_variation"
    constraint_strategy: str = "penalty"
    geometry_pair_strategy: str = "segmentation"
    mortar_shape_function: str = "line2"
    mortar_contact_defined_in: str = "reference_configuration"

    # --- 6. mesh sizes ---------------------------------------------------------------
    factor_solid: float = 1.5
    factor_beam: float = 1.2

    # --- 7. solver -------------------------------------------------------------------
    n_steps: int = 200
    total_time: float = 1.0
    max_iter: int = 60
    tol_residuum: float = 1e-8
    tol_increment: float = 1e-10
    predictor: str = "ConstDis"
    line_search: str = "Full Step"

    def __post_init__(self: "StentBalloonSettings") -> None:
        _check_beam(self)
        _check_solver(self)
        if self.balloon_material not in BALLOON_MATERIALS:
            raise ValueError(f"balloon_material must be one of {BALLOON_MATERIALS}, "
                             f"got {self.balloon_material!r}")
        if self.load_profile not in LOAD_PROFILES:
            raise ValueError(f"load_profile must be one of {LOAD_PROFILES}, "
                             f"got {self.load_profile!r}")
        if self.discretization not in CONTACT_DISCRETIZATIONS:
            raise ValueError(f"discretization must be one of {CONTACT_DISCRETIZATIONS}, "
                             f"got {self.discretization!r}")
        if self.penalty <= 0:
            raise ValueError(f"penalty must be positive, got {self.penalty}")
        if self.pressure_max <= 0:
            raise ValueError(f"pressure_max must be positive - it is the load, "
                             f"got {self.pressure_max}")
        if self.penalty_g0_per_strut < 0:
            raise ValueError(f"penalty_g0_per_strut must not be negative, "
                             f"got {self.penalty_g0_per_strut}")
        if self.gauss_points < 1:
            raise ValueError(f"gauss_points must be at least 1, got {self.gauss_points}")
        if self.wall <= 0:
            raise ValueError(f"wall must be positive, got {self.wall}")
        if self.factor_solid < 1.0:
            raise ValueError(f"factor_solid must be at least 1 (a solid element may not be "
                             f"smaller than the beam diameter), got {self.factor_solid}")
        if not 1.0 <= self.factor_beam <= 6.0:
            raise ValueError(f"factor_beam must lie in the 1 to 6 band, got {self.factor_beam}")

    def contact_section(self: "StentBalloonSettings", penalty_g0: float) -> dict:
        """
        Build the ``BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT`` section.

        The two mortar keys are written only under mortar, because 4C's default for both has no
        valid meaning at runtime even though it passes schema validation.

        :param penalty_g0: Penetration over which the contact stiffness ramps in, in mm.
        :returns: The section, ready to hand to ``InputFile.add``.
        """
        section = {
            "CONTACT_TYPE": self.contact_type,
            "CONTACT_DISCRETIZATION": self.discretization,
            "CONSTRAINT_STRATEGY": self.constraint_strategy,
            "PENALTY_PARAMETER": self.penalty,
            "PENALTY_LAW": self.penalty_law,
            "PENALTY_PARAMETER_G0": penalty_g0,
            "GEOMETRY_PAIR_STRATEGY": self.geometry_pair_strategy,
            "GAUSS_POINTS": self.gauss_points,
        }
        if self.discretization == "mortar":
            section["MORTAR_SHAPE_FUNCTION"] = self.mortar_shape_function
            section["MORTAR_CONTACT_DEFINED_IN"] = self.mortar_contact_defined_in
        return section


#: The stent-balloon settings that produced the run which finished all 200 steps. The same as the
#: class defaults, named so it can be cited.
STENT_BALLOON_ELASTIC = StentBalloonSettings()

#: The stent-balloon settings for an elastoplastic run. It takes twice the steps, so the yielding
#: is followed as it spreads. This is the case the plastic work is aimed at.
STENT_BALLOON_PLASTIC = StentBalloonSettings(material="elastoplastic", n_steps=400)


# --------------------------------------------------------------------------------------
# Stent and artery
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StentArterySettings:
    """
    Everything the stent-artery smoke test needs.

    This exercises the whole beam-to-solid chain end to end, but it is not the physics of the
    reference papers. The artery wall uses a placeholder ``StVenantKirchhoff`` material, the
    coupling is tied meshtying rather than contact, and the balloon is a radial point force. It has
    never been run in 4C, so these are BeamMe's defaults rather than calibrated values. For
    deployment physics use :class:`StentBalloonSettings`.

    :param material: The struts' material law.
    :param youngs: Strut Young's modulus, in MPa.
    :param poisson: Strut Poisson's ratio.
    :param density: Strut density.
    :param yield_strength: Strut yield strength, in MPa.
    :param tangent_modulus_ratio: Post-yield tangent modulus as a fraction of ``youngs``.
    :param beam_class: Beam element formulation.
    :param l_el_per_strut: Unused here; the beam element length comes from the coupling rule.
    :param strut_thickness: The strut thickness the simulation uses, in mm. ``None`` means the
        value the skeletonisation measured.
    :param expansion_force: Radial point-force magnitude standing in for the balloon, in N.
    :param fix_stent_node: Pin one stent node's translation, to remove the rigid-body motion the
        radial forces alone do not constrain.
    :param lumen_surface_index: Which solid surface set the beams couple to.
    :param inlet_surface_index: Which solid surface set is the fixed inlet.
    :param outlet_surface_index: Which solid surface set is the fixed outlet.
    :param factor_solid: Solid element size as a multiple of the beam diameter.
    :param factor_beam: Beam element length as a multiple of the solid element size.
    :param n_steps: Load steps.
    :param total_time: Total pseudo-time.
    :param max_iter: Newton iteration cap per step.
    :param tol_residuum: Force residual tolerance.
    :param tol_increment: Displacement update tolerance.
    :param predictor: Starting guess for each Newton solve.
    :param line_search: How far along the Newton direction to move.
    """

    simulation_type = "stent_artery"

    # --- 1. what to simulate ---------------------------------------------------------
    material: str = "elastic"

    # --- 2. stent material -----------------------------------------------------------
    youngs: float = 2.0e5
    poisson: float = 0.3
    density: float = 0.0
    yield_strength: float = 300.0
    tangent_modulus_ratio: float = 64.0 / 380.0
    beam_class: str = "Beam3rHerm2Line3"
    l_el_per_strut: float = 1.0
    strut_thickness: float = None

    # --- 3. loading ------------------------------------------------------------------
    expansion_force: float = 1e-4
    fix_stent_node: bool = True

    # --- 4. surface sets -------------------------------------------------------------
    lumen_surface_index: int = 0
    inlet_surface_index: int = 1
    outlet_surface_index: int = 2

    # --- 5. mesh sizes ---------------------------------------------------------------
    factor_solid: float = 1.5
    factor_beam: float = 1.2

    # --- 6. solver -------------------------------------------------------------------
    n_steps: int = 10
    total_time: float = 1.0
    max_iter: int = 20
    tol_residuum: float = 1e-8
    tol_increment: float = None
    predictor: str = "TangDis"
    line_search: str = "Full Step"

    def __post_init__(self: "StentArterySettings") -> None:
        _check_beam(self)
        _check_solver(self)
        if self.factor_solid < 1.0:
            raise ValueError(f"factor_solid must be at least 1, got {self.factor_solid}")
        if not 1.0 <= self.factor_beam <= 6.0:
            raise ValueError(f"factor_beam must lie in the 1 to 6 band, got {self.factor_beam}")


#: Every simulation type's settings class, by name.
SETTINGS_CLASSES = {
    "stent_only": StentOnlySettings,
    "stent_balloon": StentBalloonSettings,
    "stent_artery": StentArterySettings,
}


def default_settings(sim_type: str):
    """
    Build a type's default settings.

    :param sim_type: ``"stent_only"``, ``"stent_balloon"`` or ``"stent_artery"``.
    :returns: That type's settings, at their defaults.
    :raises ValueError: If the type is unknown.
    """
    if sim_type not in SETTINGS_CLASSES:
        raise ValueError(f"unknown sim_type {sim_type!r}, "
                         f"expected one of {sorted(SETTINGS_CLASSES)}")
    return SETTINGS_CLASSES[sim_type]()

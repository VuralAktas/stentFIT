"""
Strut cross-section properties and beam materials.

One strut is treated as a solid circle of radius ``strut_thickness / 2``, which is the assumption
``stentfit.core.splines.mesh_skeleton_beams`` makes as well. Everything else follows from that
radius, so nothing here is tied to a particular stent.
"""

import numpy as np

from beamme.four_c.material import MaterialReissner, MaterialReissnerElastoplastic



def section_properties(strut_thickness: float, settings) -> dict:
    """
    Cross-section and yield properties of one strut, from its thickness alone.

    ``Np`` and ``Mp`` are the loads at which the section first yields in pure tension and in pure
    bending. Postprocessing divides the simulated force and moment by them to report how close a
    case gets.

    ``Mp`` is the moment at which the outer fibre first reaches the yield stress,
    ``sigma_y * pi * r^3 / 4``, following Datz et al.'s ``M_y0``. This is not the fully plastic
    moment, which is 1.70 times larger and is only reached once the whole cross-section has yielded.
    4C's ``YIELDM`` is the onset of yielding, so the fully plastic value would keep the struts
    elastic far too long and then yield them abruptly.

    ``ISOHARDM`` is the moment-curvature hardening modulus, from Datz et al. eq. (C.12),
    ``H = E_ep / (1 - E_ep/E)`` in stress units, multiplied by the second moment.

    The yield quantities are always computed, whether or not the material can yield. An elastic run
    still reports how close it came, which is what says whether the elastic answer is believable.

    :param strut_thickness: Strut thickness, in mm, as measured by the skeletonisation.
    :param settings: The simulation's settings, for the strut material and stiffness.
    :returns: Dict with the radius, area, second moment, axial and bending stiffness, the yield
        force and moment, and the isotropic hardening modulus.
    :raises ValueError: If the thickness is not positive.
    """
    if strut_thickness <= 0:
        raise ValueError(f"strut_thickness must be positive, got {strut_thickness}")

    r = strut_thickness / 2.0
    area = np.pi * r ** 2
    mom = np.pi * r ** 4 / 4.0                       # second moment of area, both directions

    ratio = settings.tangent_modulus_ratio
    hardening = settings.youngs * ratio / (1.0 - ratio) * mom

    return {"radius": r,
            "area": float(area),
            "mom": float(mom),
            "EA": float(settings.youngs * area),
            "EI": float(settings.youngs * mom),
            "Np": float(settings.yield_strength * area),                    # yield axial force, N
            "Mp": float(settings.yield_strength * np.pi * r ** 3 / 4.0),    # first yield moment, N*mm
            "ISOHARDM": float(hardening)}                               # N*mm^2


def beam_material(section: dict, settings):
    """
    Build the beam material for this strut section.

    The elastoplastic material is plastic in **bending only**, following Datz et al. Their
    Appendix C is explicit that purely elastic behaviour is assumed for the torsion moment and
    that only the bending moment is elasto-plastic, so the struts can still exhibit elastic
    torsion, shear and axial extension. Their justification is physical: the radial expansion of a
    stent is governed by local bending of the struts, not by stretching them.

    4C also offers axial yielding (``YIELDN`` / ``ISOHARDN``). It is deliberately left at 4C's
    ``-1`` default, so it is simply not set. An earlier attempt switched it on because the measured
    axial utilisation exceeded 1. That was wrong: the axial force reaching yield is a consequence
    of driving every node radially, not evidence that the physical mechanism is axial. Every extra
    plastic mechanism is another nonlinearity for Newton to fight.

    :param section: The strut section, from :func:`section_properties`.
    :param settings: The simulation's settings, for the strut material and stiffness.
    :returns: A BeamMe material, ready to attach to a mesh.
    """
    common = dict(radius=section["radius"], youngs_modulus=settings.youngs,
                  nu=settings.poisson, density=settings.density)

    if settings.material != "elastoplastic":
        return MaterialReissner(**common)

    return MaterialReissnerElastoplastic(**common,
                                         yield_moment=section["Mp"],
                                         isohardening_modulus_moment=section["ISOHARDM"])


def describe(section: dict, settings) -> list:
    """
    Summarise the section and material as printable lines.

    :param section: The strut section, from :func:`section_properties`.
    :param settings: The simulation's settings, for the strut material and stiffness.
    :returns: One line per fact, for the build log.
    """
    lines = [
        f"material   {settings.material}, E = {settings.youngs:g} MPa, nu = {settings.poisson:g}",
        f"section    r = {section['radius']:.4f} mm, A = {section['area']:.4g} mm^2, "
        f"I = {section['mom']:.4g} mm^4",
        f"yield      Np = {section['Np']:.4g} N, Mp = {section['Mp']:.4g} N.mm "
        f"(at {settings.yield_strength:g} MPa)",
    ]
    if settings.material == "elastoplastic":
        lines.append(f"plasticity bending only, ISOHARDM = {section['ISOHARDM']:.4g} "
                     f"N.mm^2 (E_ep/E = {settings.tangent_modulus_ratio:.3f}); axial elastic")
    else:
        lines.append("plasticity none (elastic: no yield limit, Np and Mp reported only)")
    return lines

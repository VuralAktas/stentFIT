"""
The mixed-dimensional coupling rules, in one place.

Coupling a 1D beam to a 3D solid is only valid inside a window. The rules are Steinbrecher et
al.'s, and there are three of them.

1. A solid element must not be smaller than the beam's cross-section diameter, in the plane of the
   coupled surface. Below that the beam acts as a line load on a solid that resolves finer than the
   load itself, and the solution runs towards the singular point-load solution instead of
   converging. It does so quietly, because the run still completes and only the answer is
   mesh-dependent.

   In-plane is the operative part. The load is a line on a surface, so the resolution that can
   chase the singularity is the one along that surface. A wall thickness is a property of the
   structure rather than of the load discretisation, so it is not checked here.

2. Beam elements must be longer than solid elements, by a factor of 1 to 6, and up to 8 before the
   accuracy degrades. Shorter beams put several load points inside one solid element and the
   coupling becomes ill-conditioned, and much longer ones grow the L2 coupling error.

3. The beam must be much stiffer than the solid, ``E_beam / E_solid >= 10``, which is what makes
   the 1D reduction of the strut reasonable in the first place.

Breaking a rule is not a solver failure. The run would complete and hand back a mesh-dependent
answer, so nothing downstream would catch it. That is why this is checked before an input is
written, and against the elements that were created rather than against the target length.
"""

import numpy as np


#: Below this ratio the beam is not stiff enough next to the solid for the 1D reduction to hold.
STIFFNESS_RATIO_MIN = 10.0

#: Beam elements shorter than a solid element make the coupling ill-conditioned.
LENGTH_RATIO_MIN = 1.0

#: Above this the coupling is still valid, but no longer in the preferred band.
LENGTH_RATIO_OPTIMAL_MAX = 6.0

#: Above this the L2 coupling error grows and the result is no longer trustworthy.
LENGTH_RATIO_MAX = 8.0


def coupling_limits(strut_thickness: float, settings) -> dict:
    """
    Size both meshes from the beam diameter.

    The beam length is derived here as well, so a simulation needs no element-count setting of its
    own and the strut thickness alone fixes both meshes.

    :param strut_thickness: Strut thickness, in mm. This is the beam cross-section diameter.
    :param settings: The simulation's settings, for ``factor_solid`` and ``factor_beam``.
    :returns: Dict with the beam diameter, both target element sizes, and the bands they sit in.
    :raises ValueError: If the thickness is not positive.
    """
    if strut_thickness <= 0:
        raise ValueError(f"strut_thickness must be positive, got {strut_thickness}")

    d_beam = float(strut_thickness)
    h_solid = d_beam * settings.factor_solid

    return {"d_beam": d_beam,
            "factor_solid": settings.factor_solid,
            "factor_beam": settings.factor_beam,
            "h_solid": h_solid,
            "l_beam": h_solid * settings.factor_beam,
            "l_beam_min": LENGTH_RATIO_MIN * h_solid,          # below this: ill-conditioned
            "l_beam_optimal_max": LENGTH_RATIO_OPTIMAL_MAX * h_solid,
            "l_beam_max": LENGTH_RATIO_MAX * h_solid}          # above this: L2 error grows


def check_coupling(beam_lengths, solid_sizes: dict, d_beam: float,
                   e_beam: float, e_solid: float) -> dict:
    """
    Check every coupling rule against the elements that actually exist.

    Beam lengths are checked at both ends of the measured distribution rather than against the
    target, because BeamMe divides each strut into a whole number of elements and what comes out
    scatters around what was asked for.

    ``solid_sizes`` is a dict, so a body with different resolutions in different directions can
    report each one. A balloon passes its circumferential and axial sizes separately, and an artery
    passes a single size. Wall or through-thickness resolution must not be passed here, since the
    rule does not apply there.

    Each rule reports ``passed`` and ``optimal`` separately, because a ratio can be valid while
    still sitting outside the preferred band, which is worth reporting without failing the build.

    :param beam_lengths: Every beam element's length, in mm. Measure along the element, end to
        middle to end, not straight through it.
    :param solid_sizes: In-plane solid element sizes, keyed by direction.
    :param d_beam: Beam cross-section diameter, in mm.
    :param e_beam: Beam Young's modulus, in MPa.
    :param e_solid: Solid Young's modulus, in MPa. Use whatever actually resists the beam: for a
        body whose every node is prescribed, that is the placeholder stiffness, not the material
        the run record names.
    :returns: Dict with a ``rules`` list, plus ``all_passed`` and ``all_optimal``. Each rule is a
        dict with ``name``, ``value``, ``limit``, ``passed`` and ``optimal``.
    :raises ValueError: If no beam lengths or no solid sizes were given.
    """
    lengths = np.asarray(beam_lengths, dtype=float)
    if lengths.size == 0:
        raise ValueError("no beam element lengths to check")
    if not solid_sizes:
        raise ValueError("no solid element sizes to check")

    rules = []

    # Rule 1, once per in-plane direction the solid resolves differently.
    for direction, size in solid_sizes.items():
        rules.append({"name": f"solid element >= beam diameter, {direction}",
                      "value": float(size), "limit": f">= {d_beam:.4f}",
                      "passed": bool(size >= d_beam), "optimal": bool(size >= d_beam)})

    # Rule 2, against the coarsest in-plane direction: that is the one a beam element has to be
    # longer than.
    h = max(solid_sizes.values())
    shortest = float(lengths.min()) / h
    longest = float(lengths.max()) / h
    rules.append({"name": "shortest beam / solid element",
                  "value": shortest, "limit": f">= {LENGTH_RATIO_MIN:g}",
                  "passed": shortest >= LENGTH_RATIO_MIN,
                  "optimal": shortest >= LENGTH_RATIO_MIN})
    rules.append({"name": "longest beam / solid element",
                  "value": longest,
                  "limit": f"<= {LENGTH_RATIO_OPTIMAL_MAX:g} optimal, "
                           f"<= {LENGTH_RATIO_MAX:g} valid",
                  "passed": longest <= LENGTH_RATIO_MAX,
                  "optimal": longest <= LENGTH_RATIO_OPTIMAL_MAX})

    # Rule 3.
    stiffness = e_beam / e_solid if e_solid > 0 else float("inf")
    rules.append({"name": "beam / solid stiffness",
                  "value": float(stiffness), "limit": f">= {STIFFNESS_RATIO_MIN:g}",
                  "passed": stiffness >= STIFFNESS_RATIO_MIN,
                  "optimal": stiffness >= STIFFNESS_RATIO_MIN})

    return {"rules": rules,
            "all_passed": all(rule["passed"] for rule in rules),
            "all_optimal": all(rule["optimal"] for rule in rules)}


def coupling_lines(report: dict) -> list:
    """
    Format a coupling report as printable lines, one rule each.

    :param report: A report from :func:`check_coupling`.
    :returns: One line per rule.
    """
    width = max(len(rule["name"]) for rule in report["rules"])
    lines = []
    for rule in report["rules"]:
        if not rule["passed"]:
            mark = "FAIL"
        elif rule["optimal"]:
            mark = "ok  "
        else:
            mark = "warn"
        lines.append(f"  [{mark}] {rule['name']:{width}s} {rule['value']:9.4f}  "
                     f"({rule['limit']})")
    return lines


def print_coupling(report: dict) -> None:
    """
    Print a coupling report.

    :param report: A report from :func:`check_coupling`.
    """
    for line in coupling_lines(report):
        print(line)


def raise_if_coupling_failed(report: dict, limits: dict) -> None:
    """
    Stop the build if any rule failed, saying what to change.

    :param report: A report from :func:`check_coupling`.
    :param limits: The limits the mesh was sized from, from :func:`coupling_limits`, used to spell
        out the valid band.
    :raises ValueError: If any rule failed.
    """
    if report["all_passed"]:
        return

    failed = ", ".join(rule["name"] for rule in report["rules"] if not rule["passed"])
    raise ValueError(
        f"mixed-dimensional coupling rules violated ({failed}) - the result would depend on "
        f"the mesh.\n"
        f"  the solid element size is {limits['h_solid']:.4f} mm, so every beam element must "
        f"fall between {limits['l_beam_min']:.4f} and {limits['l_beam_max']:.4f} mm.\n"
        f"  adjust factor_solid / factor_beam in the settings. Both meshers round to a "
        f"whole number of elements, so the targets must sit inside the bands with margin, "
        f"not on their edges.")

"""
The simulation types, one module each.

Every module here exposes ``build(ctx) -> list`` and is registered in :data:`CASE_BUILDERS`, so
adding a simulation type means adding one module and one registry entry.

Each entry the builder returns is a plain dict describing one written input:

===============  ===========================================================
key              meaning
===============  ===========================================================
``name``         the case's name, which is also its folder name
``input_path``   the written ``.4C.yaml``
``run_dir``      the folder holding it, where results and logs will go
``output_base``  base name 4C writes its results under, inside ``run_dir``
``record``       everything that determined this input, for the run record
``coupling``     the coupling report, or ``None`` for a type with no solid
===============  ===========================================================

A module in this package may import from :mod:`stentfit.sim`, but never from another module in
this package. That is what keeps the simulation types independent of each other.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ..settings import resolve_strut_thickness


@dataclass
class BuildContext:
    """
    Everything a case builder is handed.

    :param stent_dir: The stent's skeletonisation output folder.
    :param stent_name: The stent's name, used for folder names and printing.
    :param features: The parsed ``stent_features.json``.
    :param output_dir: Where this simulation type writes for this stent.
    :param settings: Every parameter of this simulation, from
        :mod:`stentfit.sim.settings`. One flat object; the case reads what it needs.
    :param runner: How 4C is launched, carried so it reaches the run record.
    :param options: Anything not part of the settings, such as the composed ``balloon`` or
        ``artery`` object.
    """

    stent_dir: Path
    stent_name: str
    features: dict
    output_dir: Path
    settings: object = None
    runner: object = None
    options: dict = field(default_factory=dict)

    def strut_thickness(self: "BuildContext") -> float:
        """
        The strut thickness every mesh size and section property is derived from.

        ``settings.strut_thickness`` is the value to use. ``None`` means "the one the
        skeletonisation measured", which is the usual case.

        :returns: The strut thickness, in mm.
        """
        return resolve_strut_thickness(self.settings, self.features)


def built_case(name, input_path, run_dir, output_base, record, coupling=None) -> dict:
    """
    Describe one written input.

    :param name: The case's name, which is also its folder name.
    :param input_path: The written ``.4C.yaml``.
    :param run_dir: The folder holding it.
    :param output_base: Base name 4C writes its results under.
    :param record: Everything that determined this input, for ``run_parameters.yaml``.
    :param coupling: The coupling report, where the type has one.
    :returns: The dict described in the module docstring.
    """
    return {"name": name, "input_path": input_path, "run_dir": run_dir,
            "output_base": output_base, "record": record, "coupling": coupling}


#: Every simulation type, by name. Populated at import time below.
CASE_BUILDERS = {}


def register(name: str, builder) -> None:
    """
    Add a simulation type to the registry.

    :param name: The type's name, as passed to ``Simulation``.
    :param builder: Its ``build`` function.
    """
    CASE_BUILDERS[name] = builder


def get_builder(name: str):
    """
    Look up a simulation type.

    :param name: The type's name.
    :returns: Its ``build`` function.
    :raises ValueError: If no such type is registered.
    """
    if name not in CASE_BUILDERS:
        raise ValueError(f"unknown simulation_type {name!r}, "
                         f"expected one of {sorted(CASE_BUILDERS)}")
    return CASE_BUILDERS[name]


def _register_builtin() -> None:
    """Import the built-in types so they register themselves."""
    from . import stent_artery, stent_balloon, stent_only  # noqa: F401


_register_builtin()

__all__ = ["BuildContext", "built_case", "CASE_BUILDERS", "register", "get_builder"]

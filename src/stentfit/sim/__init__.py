"""
Shared library for the 4C simulation pipeline.

Nothing here belongs to one simulation type. The types themselves live in
:mod:`stentfit.sim.cases`, one module each, and they import from this package on equal terms. A
case must never import another case.

Every parameter of a simulation lives in one flat settings class per type, in
:mod:`stentfit.sim.settings`. Reports are plain dicts with a matching ``print_*`` helper.
"""

from .beam_model import (axis_report, beam_element_lengths, build_stent_beams,
                         centreline_nodes, couple_junctions, degree_counts,
                         innermost_surface_radius, load_features, read_junctions)
from .coupling import (check_coupling, coupling_limits, coupling_lines, print_coupling,
                       raise_if_coupling_failed)
from .materials import beam_material, describe, section_properties
from .record import (next_run_dir, read_run_parameters, update_run_results,
                     write_run_index, write_run_parameters)
from .runner import (FourCRunner, RunLock, RunnerConfig, RunnerError, preflight_ok,
                     preflight_lines, print_preflight, raise_if_not_ready)
from .settings import (SETTINGS_CLASSES, STENT_BALLOON_ELASTIC, STENT_BALLOON_PLASTIC,
                       STENT_ONLY_ELASTIC, STENT_ONLY_PLASTIC, StentArterySettings,
                       StentBalloonSettings, StentOnlySettings, default_settings,
                       element_length, settings_from_dict, settings_to_dict)

__all__ = [
    # settings: one flat class per simulation type
    "StentOnlySettings", "StentBalloonSettings", "StentArterySettings",
    "STENT_ONLY_ELASTIC", "STENT_ONLY_PLASTIC",
    "STENT_BALLOON_ELASTIC", "STENT_BALLOON_PLASTIC",
    "SETTINGS_CLASSES", "default_settings", "element_length",
    "settings_to_dict", "settings_from_dict",
    # materials
    "section_properties", "beam_material", "describe",
    # coupling
    "coupling_limits", "check_coupling", "coupling_lines", "print_coupling",
    "raise_if_coupling_failed",
    # beam model
    "load_features", "read_junctions", "degree_counts", "build_stent_beams",
    "centreline_nodes", "couple_junctions", "axis_report", "beam_element_lengths",
    "innermost_surface_radius",
    # running 4C
    "FourCRunner", "RunnerConfig", "RunLock", "RunnerError", "preflight_ok",
    "preflight_lines", "print_preflight", "raise_if_not_ready",
    # run records
    "next_run_dir", "write_run_parameters", "read_run_parameters", "update_run_results",
    "write_run_index",
]

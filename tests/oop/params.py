"""
Shared run parameters for the golden master and the OOP equivalence tests.

The golden outputs in ``tests/oop/golden/`` were produced by running the old
procedural pipeline in ``src/stentfit`` with exactly these settings. The
equivalence tests re-run the new ``src_oop`` class API with the same settings,
so any difference in the outputs is a refactor bug rather than a parameter
mismatch.

Determinism notes:

* ``random_seed=0`` fixes the mesh surface sampling.
* ``auto_tune=True`` with a deliberately generous ``tune_time_limit`` so the
  2D-skeleton parameter search always stops on its own convergence / cycle
  detector rather than on the wall-clock budget. The wall-clock stop is the one
  genuinely non-deterministic exit in the pipeline, so it must never fire.
* Every automated run mocks ``builtins.input`` to return ``""``, which means
  "accept the detected ring count" in ring detection and "no manual edits" in
  the 2D-edit prompt.
"""

STENT_NAME = "stent01"
STENT_STL = "examples/data/input/stent_designs/stent01.stl"

#: Keyword arguments shared by ``stent_pipeline`` (old) and ``Stent`` (new).
SKELETON_PARAMS = dict(
    n_points=None,
    max_display=500_000,
    remove_supports=False,
    random_seed=0,
    n_rings=None,
    auto_tune=True,
    pixels_per_strut=10,
    dilate_px=3,
    pad_fraction=0.20,
    tune_time_limit=100_000,   # effectively no wall-clock stop -> deterministic
    quality_gamma=2.0,
    ring_halo_frac=0.4,
)

#: Keyword arguments for the finalisation phase (3D wrap + spline fitting).
FINALIZE_PARAMS = dict(
    prune_tip_frac=0,
    max_display=500_000,
    random_seed=0,
)

#: ``wrap_max_surf`` is hardcoded inside the OOP ``Stent.finalize()`` — it is a
#: memory guard, not a modelling knob, so it is not a public argument there. The
#: old pipeline took it as an argument, so ``make_golden.py`` passes this value
#: explicitly. Keep the two in sync so the golden comparison stays valid.
WRAP_MAX_SURF = 2_000_000

#: The old ``build_smoketest_pipeline`` took one flat parameter list. The class
#: API splits it in two: artery shape and wall material on ``Artery``, stent
#: material / element sizing / load stepping on ``Simulation``. Their union is
#: still exactly the old set, so the golden comparison stays apples-to-apples.
ARTERY_PARAMS = dict(
    artery_type="straight",
    inner_margin=0.5,
    wall_thickness=0.5,
    noise_amplitude=0.15,
    noise_seed=0,
    bend_angle_deg=180.0,
    mesh_type="HEX8",
    artery_youngs=2.0,
)

#: Keyword arguments for ``Simulation``.
SIMULATION_PARAMS = dict(
    stent_youngs=2.0e5,
    stent_poisson=0.3,
    stent_density=0.0,
    beam_class_label="Beam3rHerm2Line3",
    factor_solid=1.5,
    factor_beam=1.2,
    n_steps=10,
    expansion_force=1e-4,
)

#: The flat form the old procedural pipeline took, used by ``make_golden.py``.
LEGACY_SIMULATION_PARAMS = {**ARTERY_PARAMS, **SIMULATION_PARAMS}

#: Files copied into ``tests/oop/golden/<stent>/`` from a skeletonisation run.
SKELETON_GOLDEN_FILES = (
    "skeleton_points.csv",
    "skeleton_splines.json",
    "stent_features.json",
)

#: Files copied into ``tests/oop/golden/<stent>/`` from a simulation run.
SIMULATION_GOLDEN_FILES = (
    "artery_solid.4C.yaml",
    "stent_warped.4C.yaml",
    "artery_stent.4C.yaml",
    "simulation.4C.yaml",
)

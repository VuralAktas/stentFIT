# Changelog

<!--next-version-placeholder-->

## v0.1.0 (24/07/2026)

- First release of `stentfit`!

## v0.1.1 (24/07/2026)

- Fix Windows install crash when `git` isn't on `PATH` (BeamMe requires it to write commit metadata).
- `sim_setup.py` functions now accept plain string paths, not just `pathlib.Path`.
- Replace blocking `fig.show()` in `build_smoketest_pipeline` with a saved HTML file plus a guarded optional interactive view.

## v0.1.2 (05/08/2026)

- The pipeline is now driven by three classes which are `Stent`, `Artery` and `Simulation`, instead of the previous module-level functions but outputs are unchanged.

## v0.1.3 (07/08/2026)

- Relicensed from MIT to the GNU General Public License v3.0 or later. Releases up to and including v0.1.2 remain available under MIT.
- Added `CITATION.cff` so the repository can be cited directly from GitHub.
- No functional changes.

## v0.2.0 (31/08/2026)

- Added `stentfit.sim`, which builds, runs and measures 4C simulations. Three simulation types are available: `stent_only` drives every beam node by prescribed displacement, `stent_balloon` opens the stent through contact with a pressure-inflated balloon, and `stent_artery` ties it into a test artery with meshtying. Each one is a module in `stentfit.sim.cases` and none of them knows about the others.
- Every type follows the same four steps, `build_input()`, `check()`, `run()` and `postprocess()`, and each build lands in its own numbered run folder with a `run_parameters.yaml` recording every parameter behind it.
- **Breaking change.** `Simulation` is now built as `Simulation(stent, sim_type, settings)` instead of `Simulation(stent, artery, sim_input_dir, ...)`. `setup()` still exists, but `align()`, `mesh_artery()`, `assemble()`, `export_paraview()`, `check_coupling()`, `plot_overview()` and `write_input()` are gone. Code written against v0.1.3 will not run.
- Every parameter of a simulation now lives in one frozen settings class per type, in `stentfit.sim.settings`. Nothing is shared between the types, so setting one field never moves another.
- Added `Balloon`, a catheter balloon sized to sit just inside a stent, driven by a follower pressure with its ends on springs.
- Added a command line, `python -m stentfit.run`, with `doctor`, `build`, `solve`, `report` and `all`. Long solves belong in a terminal rather than in a notebook cell, and several can run at once because each run locks its own folder.
- 4C now runs inside a pinned Docker container, so there is nothing to compile. `python -m stentfit.run doctor` checks the toolchain before a solve is started.
- `splinepy` is now declared as a core dependency. It was always imported by `stentfit.core.splines`, but arrived only because BeamMe requires it.
- Replaced `examples/test_sim_generation.ipynb` with one notebook per simulation type: `simulation_stent_only`, `simulation_stent_and_balloon` and `simulation_stent_and_artery`.
- Added `tests/test_simulation.py`, which checks that every type builds an input 4C's own schema accepts. Nothing is solved there, because 4C runs for hours.

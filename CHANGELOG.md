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

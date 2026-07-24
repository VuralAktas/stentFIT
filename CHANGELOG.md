# Changelog

<!--next-version-placeholder-->

## v0.1.0 (24/07/2026)

- First release of `stentfit`!

## v0.1.1 (24/07/2026)

- Fix Windows install crash when `git` isn't on `PATH` (BeamMe requires it to write commit metadata).
- `sim_setup.py` functions now accept plain string paths, not just `pathlib.Path`.
- Replace blocking `fig.show()` in `build_smoketest_pipeline` with a saved HTML file plus a guarded optional interactive view.

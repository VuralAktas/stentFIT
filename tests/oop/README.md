# OOP refactor equivalence tests

Golden-master tests proving `src_oop/stentfit` (class API) reproduces the
behaviour of `src/stentfit` (procedural API).

## Running

```bash
pytest tests/oop -q
```

`conftest.py` puts `src_oop/` at the front of `sys.path` and then asserts the
`stentfit` that actually got imported lives there — both packages use the import
name `stentfit`, and the editable install of the old one would otherwise shadow
it silently. No uninstall is needed.

The full run takes ~2 minutes: the skeletonisation and simulation pipelines each
run once, as session-scoped fixtures shared by every assertion.

## Layout

| Path | What it is |
|---|---|
| `params.py` | The exact settings both the golden capture and the tests run under |
| `make_golden.py` | Regenerates the goldens from the **old** `src/` pipeline |
| `golden/stent01/*.gz` | The committed oracle: outputs of the old pipeline |
| `test_equivalence.py` | Re-runs the new classes and compares against the oracle |

Goldens are stored gzipped (34 MB → 8.3 MB) and decompressed on read.

## Determinism

The comparison is only meaningful because the pipeline is deterministic under
`params.py`:

- `random_seed=0` fixes the mesh surface sampling.
- `tune_time_limit` is set far higher than the search ever needs, so the 2D
  skeleton tuner always stops on its own convergence detector. Its wall-clock
  stop is the one genuinely non-deterministic exit in the pipeline and must
  never fire.
- Every automated run mocks `builtins.input` to return `""` — "accept the
  detected ring count", then "no manual edits".

Verified byte-identical across two consecutive runs of the old pipeline before
any class was written. The kernels were copied verbatim, so outputs should be
*identical*; the tolerances in the tests are a safety margin against
float-reduction reordering, not expected drift.

The generated 4C `.yaml` files are the one exception: BeamMe stamps each with a
`TITLE.BeamMe` provenance block (creation timestamp, calling script path, git
sha). `normalise_4c_yaml` strips it from both sides before comparing.

## Regenerating the goldens

Only needed if the reference behaviour in `src/` intentionally changes. Requires
the old package to be the importable `stentfit`:

```bash
pip install -e .        # points at src/
python tests/oop/make_golden.py
```

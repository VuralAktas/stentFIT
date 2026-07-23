# stentFIT

Semi-automated virtual stent implantation via contact simulation (BeamMe / 4C).

`stentFIT` turns a stent surface mesh (`.stl`) into a 1D beam-element model ready
for beam-to-solid contact simulation. It samples the stent surface, detects its
rings, extracts a 2D skeleton per ring, wraps that skeleton back onto the 3D
mid-surface, fits a B-spline to each strut curve, and meshes the result into
Simo–Reissner beams with [BeamMe](https://beamme-py.github.io/beamme/).

## Installation

```bash
pip install stentfit
```

Requires **Python 3.13**. The beam-meshing step additionally uses `beamme` and
`splinepy`, installed automatically.

## What it does

1. **2D skeletonisation** — sample a point cloud from the STL, align it to its axis,
   detect the rings, and skeletonise each ring in 2D (optional auto-tuning + manual edits).
2. **Wrap to 3D** — wrap the 2D skeleton onto the local mid-surface, clean the graph,
   and fit a B-spline to each strut curve.
3. **Beam meshing** — mesh the fitted splines into a 1D Simo–Reissner beam mesh with BeamMe.

Each stage writes `skeleton_points.csv`, `skeleton_splines.json`, `stent_features.json`,
and interactive HTML views into the output directory.

## Usage

The full, interactive workflow — including the manual-edit and checkpoint steps — is in
[`examples/stent_skeleton.ipynb`](examples/stent_skeleton.ipynb). The pipeline API is
exposed through a single import point:

```python
from stentfit import (
    # Stage 1 — sample the STL, detect rings, skeletonise each ring in 2D
    run_skeletonization_2d, resume_and_edit_rings,
    sample_stent_points, detect_rings, skeletonize_rings_2d,
    save_ring_2d_checkpoint, load_ring_2d_checkpoint,
    edit_rings_2d_interactive, assemble_2d_skeleton,
    # Stage 2 — wrap the 2D skeleton onto the 3D mid-surface and fit a B-spline per strut
    finalize_skeleton, wrap_skeleton_to_3d, fit_skeleton_splines,
    save_stent_features_and_views,
    plot_skeleton_splines_2d, plot_skeleton_splines_trimesh,
    # Stage 3 — mesh the skeleton into 1D beams with BeamMe
    mesh_skeleton_beams,
)
```

The example notebook wraps these into three stages you run in sequence:

```python
# STL_FILE, STENT_NAME, and OUTPUT_DIR are yours to set — see the notebook for a full example.
state = run_skeletonization_2d(STL_FILE, STENT_NAME, OUTPUT_DIR)  # Stage 1: sample → rings → 2D skeleton (+ checkpoint)
state = resume_and_edit_rings(OUTPUT_DIR, state)                  # optional manual 2D edits → assemble skeleton
state = finalize_skeleton(state, OUTPUT_DIR, STENT_NAME)          # Stage 2: wrap to 3D, clean graph, fit splines, export
beam_mesh = mesh_skeleton_beams(OUTPUT_DIR)                        # Stage 3: mesh splines into 1D beams
```

Each stage writes CSV/JSON exports and interactive HTML views into the output
directory, and the `ring_2d.pkl` checkpoint (paired with `ring_points.csv`) lets the
pipeline resume after a kernel restart. See the example notebook for the complete,
parameterised pipeline.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that
this project is released with a Code of Conduct. By contributing to this project, you
agree to abide by its terms.

## License

`stentfit` was created by Vural Aktas. It is licensed under the terms of the MIT license.

## Credits

`stentfit` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/)
and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

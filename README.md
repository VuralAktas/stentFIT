# stentFIT

Semi-automated virtual stent implantation via contact simulation (BeamMe / 4C).

`stentFIT` turns a stent surface mesh (`.stl`) into a 1D beam-element model ready
for beam-to-solid contact simulation. It samples the stent surface, detects its
crowns, extracts a 2D skeleton per crown, wraps that skeleton back onto the 3D
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
   detect the crowns, and skeletonise each crown in 2D (optional auto-tuning + manual edits).
2. **Wrap to 3D** — wrap the 2D skeleton onto the local mid-surface, clean the graph,
   and fit a B-spline to each strut curve.
3. **Beam meshing** — mesh the fitted splines into a 1D Simo–Reissner beam mesh with BeamMe.

Each stage writes `skeleton_points.csv`, `skeleton_splines.json`, `stent_features.json`,
and interactive HTML views into the output directory.

## Usage

The full, interactive workflow — including the manual-edit and checkpoint steps — is in
[`examples/stent_skeleton.ipynb`](examples/stent_skeleton.ipynb). The pipeline runs in
three stages:

```python
import trimesh
from stentfit import stent_funcs

mesh = trimesh.load("data/stent_designs/stent04.stl")

# Stage 1 — sample the STL, detect crowns, skeletonise each crown in 2D
# Stage 2 — wrap the 2D skeleton onto the 3D mid-surface and fit a B-spline per strut
# Stage 3 — mesh the skeleton into 1D beams with BeamMe
```

Each stage writes CSV/JSON exports and interactive HTML views into the output
directory. See the example notebook for the complete, parameterised pipeline.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that
this project is released with a Code of Conduct. By contributing to this project, you
agree to abide by its terms.

## License

`stentfit` was created by Vural Aktas. It is licensed under the terms of the MIT license.

## Credits

`stentfit` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/)
and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

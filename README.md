# stentFIT

[![Documentation Status](https://readthedocs.org/projects/stentfit/badge/?version=latest)](https://stentfit.readthedocs.io/en/latest/?badge=latest)

Semi-automated virtual stent implantation with mixed-dimensional modelling

Full documentation, including the API reference and workflow diagrams, is hosted at [stentfit.readthedocs.io](https://stentfit.readthedocs.io/en/latest/).

`stentFIT` turns a stent surface mesh (`.stl`) into a 1D beam-element model ready for beam-to-solid contact simulation. It samples the stent surface, detects its rings, extracts a 2D skeleton per ring, wraps that skeleton back onto the 3D mid-surface, fits a B-spline to each strut curve, and meshes the result into Simo–Reissner beams with [BeamMe](https://beamme-py.github.io/beamme/). Extracting the stent's 1D wireframe is a semi-automated process: it allows manual edits of intermediate steps such as ring detection and 2D skeletonisation. It also provides a smoke test that verifies the quality of the 1D stent model inside a generated pipe-like 3D vessel.

## Installation

Requires **Python 3.13** (pinned to match [BeamMe](https://beamme-py.github.io/beamme/)'s supported range) and **Git** installed and available on your `PATH` — BeamMe uses the `git` executable to write commit metadata into generated 4C input files, and raises an error at runtime if it can't find one. Git is not a Python package, so `pip`/`poetry` cannot install it for you — the steps below install it via conda alongside everything else.

**Option A — Just use the package**

Same steps on macOS, Linux, and Windows (PowerShell) — on Windows, just make sure to run `python`/`pip`, not the `py` launcher, which ignores the active conda environment:

```bash
conda create -n stentfit-env python=3.13
conda activate stentfit-env
conda install -c conda-forge git
pip install stentfit
```

Core dependencies such as `beamme` (stent beam meshing and artery solid meshing), `gmsh` (artery solid meshing), and `fourcipp` are installed automatically by `pip`. Running the generated simulation input files additionally requires a compiled **4C** executable, which is not included in this package.

**Option B — Clone the repo and start developing**

```bash
git clone https://github.com/VuralAktas/stentFIT.git
cd stentFIT
conda env create -f environment.yml   # Python 3.13 + pip + git
conda activate env_stentfit
poetry install                        # installs stentfit + all dependency groups
```

This path pulls `git` in automatically via `environment.yml`, and also gives you the example notebooks and locked dependency versions (`poetry.lock`) matching the rest of the project.

## What it does & How to use it

```python
from stentfit import Stent, Artery, Simulation

# 1. STL -> 1D spline wireframe
stent = Stent(stl_file="stent01.stl", stent_name="stent01", output_dir="outputs/stent01")
stent.skeletonize()

# 2. A test artery sized to that stent
artery = Artery(stent, artery_type="curved")

# 3. Warp, tie, check, and write the 4C input
sim = Simulation(stent, artery, sim_input_dir="outputs/simulation")
sim.setup()
```

Just note that, this example is just a simplified version of the full workflow. Each call runs a fixed sequence of steps, and every step is also callable on its own — see [Workflows](https://stentfit.readthedocs.io/en/latest/workflow.html) for the breakdown.

**1. Stent skeletonisation** ([`examples/stent_skeleton.ipynb`](examples/stent_skeleton.ipynb))

- Sample a point cloud from the stent STL and align it to its centreline axis.
- Detect rings and skeletonise each ring in 2D (optional auto-tuning + manual edits).
- Wrap the 2D skeleton onto the local mid-surface, clean up the graph, and fit a B-spline per strut.
- Mesh the fitted splines into a 1D Simo–Reissner beam mesh with BeamMe.

Each stage writes `skeleton_points.csv`, `skeleton_splines.json`, `stent_features.json`, and interactive HTML views into the output directory.

![Unrolled 2D skeleton with per-ring tuning diagnostics](https://raw.githubusercontent.com/VuralAktas/stentFIT/main/docs/images/skeleton_splines_2d.png)

**2. Test artery generation & simulation setup** ([`examples/test_sim_generation.ipynb`](examples/test_sim_generation.ipynb))

A synthetic/parametric smoke test exercises the full mixed-dimensional chain end-to-end:

- Generate a parametric test artery (straight / curved / S-bend) sized to the stent, and mesh its wall as a 3D solid with GMSH.
- Warp the stent beam mesh onto the artery centreline.
- Check beam-to-solid coupling compatibility (stiffness ratio, element-size ratios) and visualize the stent inside the artery via Paraview.
- Tie the beam mesh to the artery lumen and write a schema-validated 4C simulation input file with a quasi-static radial expansion load.

This confirms the stent-to-artery mapping and 4C input generation work end-to-end, using placeholder materials and tied meshtying rather than real contact such as full deployment physics (contact, HGO-C artery material, elasto-plastic beam bending) is planned but not yet implemented.

![Stent beam mesh warped into a curved artery, viewed in ParaView](https://raw.githubusercontent.com/VuralAktas/stentFIT/main/docs/images/paraview_stent_artery.png)

## License

`stentfit` was created by Vural Aktas. It is licensed under the terms of the MIT license.

## Contributing

Interested in contributing? Reach out at [vural.aktas@rwth-aachen.de](mailto:vural.aktas@rwth-aachen.de).

## Credits

`stentfit` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/)
and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

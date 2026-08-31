# Workflows

`stentfit` is built around three classes. Each one holds the data it produces,
so the way to read a workflow is: which method you call, which attributes it
fills in, and which files it leaves on disk.

The diagrams below follow the source (`stent.py`, `artery.py`,
`simulation.py`). Click any name to jump to its full API reference.

## `Stent`

One stent, from its STL surface mesh to a fitted spline wireframe.
{py:meth}`~stentfit.stent.Stent.skeletonize` runs all three phases, and each
phase is also callable on its own for the interactive, per-ring workflow.

These are the attributes the object carries. They all start as `None` and the
phases below fill them in, which is what the "Fills in" columns refer to.

| Attribute | Holds |
|---|---|
| `mesh` | the loaded trimesh surface mesh |
| `stent_df` | the sampled point cloud, in cylindrical and Cartesian coordinates |
| `stent_features` | the geometry values: `length`, `diameter`, `radius`, `strut_thickness`, `r_inner`, `r_outer`, `r_mid`, `z_min`, `z_max`, `center_cylinder_radius`, `num_points` |
| `stent_centerline_direction` | the PCA long-axis unit vector |
| `ring_edges` | the z-boundaries between rings |
| `ring_order` | the ring ids, bottom to top |
| `ring_2d` | the per-ring 2D skeletons, keyed `ring_00`, `ring_01`, ... |
| `skel_arc`, `skel_z` | the assembled flat skeleton |
| `skel_px` | the pixel size behind each skeleton point |
| `surf_df` | the surface cloud used for the 3D wrap |
| `skeleton_df` | the final 3D skeleton graph |
| `skeleton_curves` | the skeleton grouped into curves, as lists of point ids |
| `skeleton_splines` | one fitted B-spline per curve |

The values you pass to the constructor, such as `output_dir`, `auto_tune` or
`pixels_per_strut`, are also attributes, but they are settings rather than
results so they are not listed here.

```{mermaid}
graph LR
    subgraph P1[skeletonize_2d]
        direction TB
        A1[sample_stent_points] --> A2[detect_rings] --> A3[skeletonize_rings_2d] --> A4[save_checkpoint]
    end
    subgraph P2[edit_and_assemble]
        direction TB
        B1[edit_rings_2d_interactive] --> B2[assemble_2d_skeleton]
    end
    subgraph P3[finalize]
        direction TB
        C1[wrap_skeleton_to_3d] --> C2[fit_skeleton_splines] --> C3[save_stent_features_and_views]
    end
    S[["Stent(stl_file,<br/>stent_name,<br/>output_dir)"]] --> P1 --> P2 --> P3 --> D[["splines ready<br/>for meshing"]]
```

**Phase 1 - {py:meth}`~stentfit.stent.Stent.skeletonize_2d`**

| Step | Fills in | Writes |
|---|---|---|
| {py:func}`~stentfit.core.sampling.sample_stent_points` | `stent_df`, `stent_features`, `stent_centerline_direction` | `sampling_points.csv`, `sampling_points.html`, `thickness_diagnostics.html` |
| {py:func}`~stentfit.core.rings.detect_rings` | `ring_edges` | `ring_points.csv`, `ring_dips.html`, `ring_assignment.html` |
| {py:func}`~stentfit.core.skeleton_2d.skeletonize_rings_2d` | `ring_2d`, `ring_order` | `skeleton_plots/ring_XX.html`, `ring_XX_convergence.html`, `ring_XX_2d.csv` |
| {py:meth}`~stentfit.stent.Stent.save_checkpoint` | | `ring_2d.pkl` |

**Phase 2 - {py:meth}`~stentfit.stent.Stent.edit_and_assemble`**

| Step | Fills in | Writes |
|---|---|---|
| {py:func}`~stentfit.core.skeleton_2d.edit_rings_2d_interactive` | `ring_2d` (edited in place) | updated `ring_2d.pkl` on each confirmed edit |
| {py:func}`~stentfit.core.skeleton_2d.assemble_2d_skeleton` | `skel_arc`, `skel_z`, `skel_px`, `surf_df` | |

This phase prompts once, so you can fix any ring the detector got wrong before
the skeleton is lifted to 3D. After a kernel restart,
{py:meth}`~stentfit.stent.Stent.load` rebuilds the object from `ring_2d.pkl`
and you can carry on from here without recomputing phase 1.

**Phase 3 - {py:meth}`~stentfit.stent.Stent.finalize`**

| Step | Fills in | Writes |
|---|---|---|
| {py:func}`~stentfit.core.skeleton_3d.wrap_skeleton_to_3d` | `skeleton_df` | `skeleton_points.csv`, `skeleton_only.html` |
| {py:func}`~stentfit.core.splines.fit_skeleton_splines` | `skeleton_curves`, `skeleton_splines` | `skeleton_splines.json`, `splines.html` |
| {py:func}`~stentfit.core.skeleton_3d.save_stent_features_and_views` | | `stent_features.json`, `skeleton_with_cloud.html` |
| {py:meth}`~stentfit.stent.Stent.plot_splines_2d` | | `skeleton_splines_2d.html`, `skeleton_splines_2d.png` |
| {py:meth}`~stentfit.stent.Stent.plot_splines_trimesh` | | `skeleton_splines_trimesh.html`, `skeleton_splines.glb` |

`skeleton_curves` and `skeleton_splines` are both kept because they are
different things: the curves are the topology, meaning which skeleton points
form each strut, and the splines are the smooth geometry fitted through them.

## `Artery`

A parametric test artery sized to hold a given stent. Everything is resolved in
the constructor, so the object is complete as soon as it exists.

| Attribute | Holds |
|---|---|
| `stent` | the stent this artery was sized against |
| `radius`, `length`, `bend_radius` | the resolved dimensions, in mm |
| `geometry` | the wall **surface** mesh, as a trimesh tube |
| `centreline` | the `(n, 3)` centreline points |
| `solid_yaml` | the path to the 4C solid, `None` until `mesh_solid` has run |

The shape settings you pass in, such as `artery_type`, `inner_margin` and
`wall_thickness`, are kept as attributes too.

```{mermaid}
graph LR
    S[["Stent<br/>(skeletonised)"]] --> A[["Artery(stent,<br/>artery_type,<br/>inner_margin)"]]
    A --> G[radius, length, bend_radius]
    G --> M[geometry + centreline<br/>wall surface]
    M --> MS[mesh_solid]
    MS --> Y[["artery_solid.4C.yaml<br/>3D solid mesh"]]
```

| Step | Fills in | Writes |
|---|---|---|
| {py:class}`~stentfit.artery.Artery` constructor | `radius`, `length`, `bend_radius`, `geometry`, `centreline` | |
| {py:meth}`~stentfit.artery.Artery.mesh_solid` | `solid_yaml` | `artery_solid.4C.yaml` |

The dimensions all come from the stent. The lumen radius is the stent's outer
radius plus `inner_margin`, the length is a multiple of the stent length so the
clamped ends sit clear of the stent, and any bend radius is picked so the arc
spans most of that length.

Note that the constructor builds only the wall **surface**, which is a trimesh
tube. The finite-element solid that 4C solves on comes from
{py:meth}`~stentfit.artery.Artery.mesh_solid`, and the build calls that for you.


## `Balloon`

A catheter balloon sized to sit just inside a stent, and the pressure that
inflates it. Like `Artery`, it is built against a stent, and
{py:class}`~stentfit.simulation.Simulation` builds it from the settings, so
there is one block to read rather than two objects to keep in step.

| Attribute | Holds |
|---|---|
| `stent` | the stent this balloon goes inside |
| `r_inner`, `r_outer`, `wall` | the resolved dimensions, in mm |
| `n_circ`, `n_axial` | the element counts, derived from the coupling rule |
| `solid_yaml` | the path to the 4C solid, `None` until `mesh_solid` has run |

The balloon is driven by a follower pressure on its inner surface with its ends
on springs, so the radius it reaches is an outcome of the solve rather than
something imposed. Its outer surface is what the stent touches.

| Step | Fills in | Writes |
|---|---|---|
| {py:class}`~stentfit.balloon.Balloon` constructor | the material and loading settings | |
| {py:meth}`~stentfit.balloon.Balloon.mesh_solid` | `r_inner`, `r_outer`, `n_circ`, `n_axial`, `solid_yaml` | `balloon.4C.yaml` |
| {py:meth}`~stentfit.balloon.Balloon.add_pressure` | | the pressure condition, into the mesh |
| {py:meth}`~stentfit.balloon.Balloon.add_end_springs` | | the end supports, into the mesh |

The two length scales come from different places. The in-plane element size
follows from the coupling rule, so the number of elements around and along the
tube is set by the stent's strut thickness. The wall thickness is a property of
the balloon itself, and it is much thinner than an element is wide.

The balloon is placed against the stent's **measured** innermost strut surface,
not against the averaged `r_inner` from the features file, because how far the
innermost node sits inside that average changes from stent to stent.

## `Simulation`

One stent, one simulation type, from input file to measured result. Three types
are available, and they all behave the same way.

| Attribute | Holds |
|---|---|
| `stent`, `settings`, `runner` | what you passed in |
| `sim_type` | `"stent_only"`, `"stent_balloon"` or `"stent_artery"` |
| `output_dir` | the root, with the type and the stent name appended |
| `balloon` | built from the settings, for `stent_balloon` only |
| `artery` | the artery, required by `stent_artery` |
| `built` | what {py:meth}`~stentfit.simulation.Simulation.build_input` produced, one dict per written input |
| `runs` | what {py:meth}`~stentfit.simulation.Simulation.run` produced, by case name |

Every parameter of a simulation lives in one flat settings class per type, in
{py:mod}`stentfit.sim.settings`. Nothing is shared between the types, so setting
one field never moves another.

```{mermaid}
graph LR
    S[["Stent<br/>(skeletonised)"]] --> I[["Simulation(stent,<br/>sim_type,<br/>settings)"]]
    I --> B[build_input]
    B --> C[check]
    C --> R[run]
    R --> P[postprocess]
    B --> F[["runNNN/<br/>input .4C.yaml<br/>run_parameters.yaml"]]
    P --> M[["results/<br/>metrics.csv<br/>summary.yaml"]]
    I -.->|from_run| B
```

| Step | Fills in | Writes |
|---|---|---|
| {py:meth}`~stentfit.simulation.Simulation.build_input` | `built` | `<case>.4C.yaml`, `*_mesh.vtu`, `run_parameters.yaml`, `runs_summary.csv` |
| {py:meth}`~stentfit.simulation.Simulation.check` | | |
| {py:meth}`~stentfit.simulation.Simulation.find_runs` | `built`, from disk | |
| {py:meth}`~stentfit.simulation.Simulation.run` | `runs` | `run.log`, `out_*/` |
| {py:meth}`~stentfit.simulation.Simulation.postprocess` | | `results/metrics.csv`, `results/summary.yaml`, `results/balloon_profile.csv` |

`build_input` is the only step that creates a folder. `find_runs` is what lets
`run` and `postprocess` work on a folder that already exists, so the run that
was built, with the parameters that were set, is the one that gets solved.
{py:meth}`~stentfit.simulation.Simulation.from_run` goes further and rebuilds
the whole `Simulation` from a finished run's own record, which is how a study
changes one parameter and compares.

### The three simulation types

Each one is a module in {py:mod}`stentfit.sim.cases`, and none of them knows
about the others. Everything they share lives in {py:mod}`stentfit.sim`: the
meshing, the materials, the coupling rules, the runner and the run record.

| | `stent_only` | `stent_balloon` | `stent_artery` |
|---|---|---|---|
| Second body | none | balloon, solid elements | test artery |
| How the stent is loaded | prescribed displacement | pressure through contact | radial point force |
| Interaction | self-contact | beam-to-solid contact | meshtying |
| Settings class | {py:class}`~stentfit.sim.settings.StentOnlySettings` | {py:class}`~stentfit.sim.settings.StentBalloonSettings` | {py:class}`~stentfit.sim.settings.StentArterySettings` |

`stent_only` has two load cases. Radial expansion drives every centreline node
outwards and leaves the axial direction free, so the foreshortening comes out as
a result rather than being imposed. Axial stretch clamps a band at each end and
pulls, so the stent necks in through the gauge length as a test specimen does.

### Element sizing

Both factors are relative to the stent's strut thickness, and their ratio is
what {py:func}`~stentfit.sim.coupling.check_coupling` tests:

```
solid_element_size = strut_thickness × factor_solid
beam_element_size  = strut_thickness × factor_solid × factor_beam
```

`stent_only` is the exception. It has no solid to couple to, so its element
length comes from `l_el_per_strut` directly.

### Coupling checks

Following Steinbrecher et al., where the beam is one stent strut and the solid
is the artery wall or the balloon:

| # | Check | Criterion |
|---|---|---|
| 1 | Stiffness ratio | `E_beam / E_solid ≥ 10` |
| 2 | Solid element vs beam diameter | `L_solid ≥ D_beam`, in the plane of the coupled surface |
| 3 | Element length ratio | `L_beam / L_solid` within `[1, 8]`, optimal `[1, 6]` |

The lengths are checked at both ends of the measured distribution rather than
against the target, because the mesher divides each strut into a whole number of
elements and what comes out scatters around what was asked for.

Breaking a rule is not a solver failure. The run would complete and hand back a
mesh-dependent answer, so nothing downstream would catch it. That is why the
build refuses to write an input that violates them.

### Running from a terminal

Long solves belong in a terminal rather than in a notebook cell, and several can
run at once, because each run locks its own folder:

```bash
python -m stentfit.run doctor
python -m stentfit.run build  stent_balloon stent01
python -m stentfit.run solve  stent_balloon stent01 --cpus 4
python -m stentfit.run report stent_balloon stent01 run001
```

See the [README](index.md) for what each output file contains, and the
{doc}`API reference <autoapi/index>` for full signatures.

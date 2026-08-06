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

**Phase 1 — {py:meth}`~stentfit.stent.Stent.skeletonize_2d`**

| Step | Fills in | Writes |
|---|---|---|
| {py:func}`~stentfit.core.sampling.sample_stent_points` | `stent_df`, `stent_features`, `stent_centerline_direction` | `sampling_points.csv`, `sampling_points.html`, `thickness_diagnostics.html` |
| {py:func}`~stentfit.core.rings.detect_rings` | `ring_edges` | `ring_points.csv`, `ring_dips.html`, `ring_assignment.html` |
| {py:func}`~stentfit.core.skeleton_2d.skeletonize_rings_2d` | `ring_2d`, `ring_order` | `skeleton_plots/ring_XX.html`, `ring_XX_convergence.html`, `ring_XX_2d.csv` |
| {py:meth}`~stentfit.stent.Stent.save_checkpoint` | | `ring_2d.pkl` |

**Phase 2 — {py:meth}`~stentfit.stent.Stent.edit_and_assemble`**

| Step | Fills in | Writes |
|---|---|---|
| {py:func}`~stentfit.core.skeleton_2d.edit_rings_2d_interactive` | `ring_2d` (edited in place) | updated `ring_2d.pkl` on each confirmed edit |
| {py:func}`~stentfit.core.skeleton_2d.assemble_2d_skeleton` | `skel_arc`, `skel_z`, `skel_px`, `surf_df` | |

This phase prompts once, so you can fix any ring the detector got wrong before
the skeleton is lifted to 3D. After a kernel restart,
{py:meth}`~stentfit.stent.Stent.load` rebuilds the object from `ring_2d.pkl`
and you can carry on from here without recomputing phase 1.

**Phase 3 — {py:meth}`~stentfit.stent.Stent.finalize`**

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
tube. The finite-element solid that 4C actually solves on comes from
{py:meth}`~stentfit.artery.Artery.mesh_solid`, and
{py:meth}`~stentfit.simulation.Simulation.setup` calls that for you.

## `Simulation`

Composes a stent and an artery into a runnable 4C input.
{py:meth}`~stentfit.simulation.Simulation.setup` runs the whole chain, and
every step is also callable on its own.

| Attribute | Holds |
|---|---|
| `stent`, `artery` | the two composed objects |
| `sim_input_dir` | the folder every generated file goes into |
| `beam_mesh` | the warped stent beam mesh |
| `full_mesh` | the combined beam and solid mesh |
| `coupling_report` | the pass/fail checks, including `all_passed` |

Only the last three are results; the rest are what you passed in. Nothing is
copied from the composed objects, so the stent features are read through as
`sim.stent.stent_features` and the solid path as `sim.artery.solid_yaml`.

```{mermaid}
graph TD
    I[["Simulation(stent, artery,<br/>sim_input_dir)"]] --> A[align]
    A --> B[mesh_artery]
    B --> C[assemble]
    C --> D[export_paraview]
    D --> E[check_coupling]
    E -->|all_passed| F[write_input]
    E -.->|fails| X([skip: fix element sizes or moduli])
```

| Step | Fills in | Writes |
|---|---|---|
| {py:meth}`~stentfit.simulation.Simulation.print_stent_summary` | | |
| {py:meth}`~stentfit.simulation.Simulation.align` | `beam_mesh` | `stent_warped.4C.yaml` |
| {py:meth}`~stentfit.simulation.Simulation.mesh_artery` | `artery.solid_yaml` | `artery_solid.4C.yaml` |
| {py:meth}`~stentfit.simulation.Simulation.assemble` | `full_mesh` | `artery_stent.4C.yaml` |
| {py:meth}`~stentfit.simulation.Simulation.export_paraview` | | `artery_stent_mesh_beam.vtu`, `artery_stent_mesh_solid.vtu` |
| {py:meth}`~stentfit.simulation.Simulation.check_coupling` | `coupling_report` | |
| {py:meth}`~stentfit.simulation.Simulation.plot_overview` | | `stent_artery_view.html` |
| {py:meth}`~stentfit.simulation.Simulation.write_input` | | `simulation.4C.yaml` |

{py:meth}`~stentfit.simulation.Simulation.align` meshes the straight stent as
beams with {py:func}`~stentfit.core.splines.mesh_skeleton_beams`, then warps it
onto the artery centreline.
{py:meth}`~stentfit.simulation.Simulation.assemble` ties the beams to the lumen
surface with {py:func}`~stentfit.core.artery_geom.import_artery_solid` and
{py:func}`~stentfit.core.artery_geom.assemble_beam_solid`.

`write_input` only runs when all three coupling checks pass, because an input
file that breaks the mixed-dimensional assumptions should not look runnable.

### Element sizing

Both factors are relative to the stent's strut thickness, and their ratio is
what {py:meth}`~stentfit.simulation.Simulation.check_coupling` tests:

```
solid_element_size = strut_thickness × factor_solid
beam_element_size  = strut_thickness × factor_solid × factor_beam
```

They are read-only properties, so they always follow the stent rather than
drifting from it.

### Coupling checks

Following Steinbrecher et al., where the beam is one stent strut and the solid
is the artery wall:

| # | Check | Criterion |
|---|---|---|
| 1 | Stiffness ratio | `E_beam / E_solid ≥ 10` |
| 2 | Rule of thumb | `L_solid ≥ D_beam` |
| 3 | Element length ratio | `L_beam / L_solid` within `[1, 8]`, optimal `[1, 6]` |

See the [README](index.md) for what each output file contains, and the
{doc}`API reference <autoapi/index>` for full signatures.

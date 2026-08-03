# Workflows

`stentfit` has two pipelines. Each is driven by one call that runs a fixed
sequence of steps, and every step is also callable on its own when you want to
inspect or redo part of a run. The diagrams below trace those calls directly
from the source (`stent.py`, `simulation.py`); click any name to jump to its
full API reference.

## Step 1 — Stent skeletonisation

{py:meth}`~stentfit.stent.Stent.skeletonize` runs three phases in order, each
also callable on its own for the interactive, per-ring workflow.

```{mermaid}
graph TD
    P[Stent.skeletonize] --> A
    P --> B
    P --> C

    subgraph A[skeletonize_2d]
        A1[sample_stent_points]
        A2[detect_rings]
        A3[skeletonize_rings_2d]
        A4[save_checkpoint]
        A1 --> A2 --> A3 --> A4
    end

    subgraph B[edit_and_assemble]
        B1[edit_rings_2d_interactive]
        B2[assemble_2d_skeleton]
        B1 --> B2
    end

    subgraph C[finalize]
        C1[wrap_skeleton_to_3d]
        C2[fit_skeleton_splines]
        C3[save_stent_features_and_views]
        C4[plot_skeleton_splines_2d]
        C5[plot_skeleton_splines_trimesh]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    A --> B --> C
```

**Phase 1 —** {py:meth}`~stentfit.stent.Stent.skeletonize_2d`
: {py:func}`~stentfit.core.sampling.sample_stent_points` samples the STL
  into a point cloud, {py:func}`~stentfit.core.rings.detect_rings` splits it
  into rings, {py:func}`~stentfit.core.skeleton_2d.skeletonize_rings_2d`
  2D-skeletonises each ring (optionally auto-tuned), and
  {py:meth}`~stentfit.stent.Stent.save_checkpoint` writes the `ring_2d.pkl`
  resume checkpoint.

**Phase 2 —** {py:meth}`~stentfit.stent.Stent.edit_and_assemble`
: runs {py:func}`~stentfit.core.skeleton_2d.edit_rings_2d_interactive`, which
  prompts once to manually fix any ring, then
  {py:func}`~stentfit.core.skeleton_2d.assemble_2d_skeleton` concatenates the
  rings into the full 2D skeleton. After a kernel restart,
  {py:meth}`~stentfit.stent.Stent.load` rebuilds everything from the
  checkpoint so a run can pick up here without recomputing.

**Phase 3 —** {py:meth}`~stentfit.stent.Stent.finalize`
: {py:func}`~stentfit.core.skeleton_3d.wrap_skeleton_to_3d` lifts the 2D
  skeleton onto the 3D mid-surface and cleans up the graph,
  {py:func}`~stentfit.core.splines.fit_skeleton_splines` fits a B-spline per
  curve, {py:func}`~stentfit.core.skeleton_3d.save_stent_features_and_views`
  writes the final feature/view exports, and
  {py:func}`~stentfit.core.plotting.plot_skeleton_splines_2d` /
  {py:func}`~stentfit.core.plotting.plot_skeleton_splines_trimesh` render
  the spline views.

## Steps 2–5 — Test artery generation & simulation setup

{py:class}`~stentfit.artery.Artery` builds a test artery sized to the stent,
and {py:meth}`~stentfit.simulation.Simulation.setup` chains the whole
synthetic/parametric smoke test from there to a runnable 4C simulation input.

```{mermaid}
graph TD
    A[Artery] --> S[Simulation.setup]
    S --> S1[print_stent_summary]
    S1 --> S2[align]
    S2 --> S2a[mesh_skeleton_beams]
    S2 --> S3[mesh_artery]
    S3 --> S3a[mesh_artery_gmsh]
    S3 --> S4[assemble]
    S4 --> S4a[import_artery_solid]
    S4 --> S4b[assemble_beam_solid]
    S4 --> S5[export_paraview]
    S5 --> S6[check_coupling]
    S6 -->|all_passed| S7[write_input]
    S6 -.->|fails| S8([skip: fix element sizes / moduli])
```

1. {py:class}`~stentfit.artery.Artery` builds a parametric test artery
   (straight / curved / s-bend) sized to the stent: its lumen radius is the
   stent's outer radius plus a clearance margin, and its length a multiple of
   the stent length.
2. {py:meth}`~stentfit.simulation.Simulation.print_stent_summary` reports the
   stent's key dimensions as a sanity check before meshing.
3. {py:meth}`~stentfit.simulation.Simulation.align` calls
   {py:func}`~stentfit.core.splines.mesh_skeleton_beams` to mesh the straight
   stent as beams, then warps it onto the artery centreline.
4. {py:meth}`~stentfit.simulation.Simulation.mesh_artery` calls
   {py:meth}`~stentfit.artery.Artery.mesh_solid`, which meshes the artery wall
   as a 3D solid with {py:func}`~stentfit.core.artery_geom.mesh_artery_gmsh`.
5. {py:meth}`~stentfit.simulation.Simulation.assemble` calls
   {py:func}`~stentfit.core.artery_geom.import_artery_solid` then
   {py:func}`~stentfit.core.artery_geom.assemble_beam_solid` to tie the beam
   mesh to the artery lumen (mortar beam-to-solid coupling).
6. {py:meth}`~stentfit.simulation.Simulation.export_paraview` exports separate
   beam/solid `.vtu` files for inspection.
7. {py:meth}`~stentfit.simulation.Simulation.check_coupling` checks the
   stiffness ratio, solid-size-vs-beam-diameter, and element-length-ratio
   assumptions from Steinbrecher et al. — **only if all three pass**:
8. {py:meth}`~stentfit.simulation.Simulation.write_input` adds the static
   solver header, fixes the artery inlet/outlet, and applies the radial
   "balloon" expansion load, writing the schema-validated
   `simulation.4C.yaml`.

See the [README](index.md) for what each step's output file is, and the
{doc}`API reference <autoapi/index>` for full signatures.

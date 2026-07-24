# Workflows

`stentfit` has two pipelines, each a thin orchestrator function that calls a
fixed sequence of module-level functions. The diagrams below trace those
calls directly from the source (`stent_pipeline.py`, `sim_setup.py`); click
any function name to jump to its full API reference.

## Step 1 — Stent skeletonisation

{py:func}`~stentfit.stent_pipeline.stent_pipeline` runs three phases in
order, each also callable on its own for the interactive, per-ring workflow.

```{mermaid}
graph TD
    P[stent_pipeline] --> A
    P --> B
    P --> C

    subgraph A[run_skeletonization_2d]
        A1[sample_stent_points]
        A2[detect_rings]
        A3[skeletonize_rings_2d]
        A4[save_ring_2d_checkpoint]
        A1 --> A2 --> A3 --> A4
    end

    subgraph B[resume_and_edit_rings]
        B1[load_ring_2d_checkpoint]
        B2[edit_rings_2d_interactive]
        B3[assemble_2d_skeleton]
        B1 -.->|"state=None only"| B2
        B2 --> B3
    end

    subgraph C[finalize_skeleton]
        C1[wrap_skeleton_to_3d]
        C2[fit_skeleton_splines]
        C3[save_stent_features_and_views]
        C4[plot_skeleton_splines_2d]
        C5[plot_skeleton_splines_trimesh]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    A --> B --> C
```

**Phase 1 —** {py:func}`~stentfit.stent_pipeline.run_skeletonization_2d`
: {py:func}`~stentfit.stent_sampling.sample_stent_points` samples the STL
  into a point cloud, {py:func}`~stentfit.stent_rings.detect_rings` splits it
  into rings, {py:func}`~stentfit.stent_skeleton_2d.skeletonize_rings_2d`
  2D-skeletonises each ring (optionally auto-tuned), and
  {py:func}`~stentfit.stent_skeleton_2d.save_ring_2d_checkpoint` writes the
  `ring_2d.pkl` resume checkpoint.

**Phase 2 —** {py:func}`~stentfit.stent_pipeline.resume_and_edit_rings`
: reloads the checkpoint via
  {py:func}`~stentfit.stent_skeleton_2d.load_ring_2d_checkpoint` only when
  called with `state=None` (e.g. after a kernel restart); otherwise it reuses
  the in-memory state from phase 1. Always runs
  {py:func}`~stentfit.stent_skeleton_2d.edit_rings_2d_interactive` (prompts
  once to manually fix any ring), then
  {py:func}`~stentfit.stent_skeleton_2d.assemble_2d_skeleton` concatenates
  the rings into the full 2D skeleton.

**Phase 3 —** {py:func}`~stentfit.stent_pipeline.finalize_skeleton`
: {py:func}`~stentfit.stent_skeleton_3d.wrap_skeleton_to_3d` lifts the 2D
  skeleton onto the 3D mid-surface and cleans up the graph,
  {py:func}`~stentfit.stent_splines.fit_skeleton_splines` fits a B-spline per
  curve, {py:func}`~stentfit.stent_skeleton_3d.save_stent_features_and_views`
  writes the final feature/view exports, and
  {py:func}`~stentfit.stent_plotting.plot_skeleton_splines_2d` /
  {py:func}`~stentfit.stent_plotting.plot_skeleton_splines_trimesh` render
  the spline views.

## Steps 2–5 — Test artery generation & simulation setup

{py:func}`~stentfit.sim_setup.build_smoketest_pipeline` chains the whole
synthetic/parametric smoke test, from the stent's skeletonisation output to
a runnable 4C simulation input.

```{mermaid}
graph TD
    S[build_smoketest_pipeline] --> S1[stent_feature_extraction]
    S1 --> S2[generate_artery_for_stent]
    S2 --> S3[stent_meshing_alignment]
    S3 --> S3a[mesh_skeleton_beams]
    S3 --> S4[mesh_artery_gmsh]
    S4 --> S5[create_assembly_mesh]
    S5 --> S5a[import_artery_solid]
    S5 --> S5b[assemble_beam_solid]
    S5 --> S6[paraview_mesh_files]
    S6 --> S7[check_coupling_assumptions]
    S7 -->|all_passed| S8[build_smoketest_input]
    S7 -.->|fails| S9([skip: fix element sizes / moduli])
```

1. {py:func}`~stentfit.sim_setup.stent_feature_extraction` loads
   `stent_features.json` / `skeleton_points.csv` from the Step 1 output.
2. {py:func}`~stentfit.artery_generate.generate_artery_for_stent` builds a
   parametric test artery (straight / curved / s-bend) sized to the stent.
3. {py:func}`~stentfit.sim_setup.stent_meshing_alignment` calls
   {py:func}`~stentfit.stent_splines.mesh_skeleton_beams` to mesh the
   straight stent as beams, then warps it onto the artery centreline.
4. {py:func}`~stentfit.artery_mesh.mesh_artery_gmsh` meshes the artery wall
   as a 3D solid with GMSH.
5. {py:func}`~stentfit.sim_setup.create_assembly_mesh` calls
   {py:func}`~stentfit.artery_mesh.import_artery_solid` then
   {py:func}`~stentfit.artery_mesh.assemble_beam_solid` to tie the beam mesh
   to the artery lumen (mortar beam-to-solid coupling).
6. {py:func}`~stentfit.sim_setup.paraview_mesh_files` exports separate
   beam/solid `.vtu` files for inspection.
7. {py:func}`~stentfit.sim_setup.check_coupling_assumptions` checks the
   stiffness ratio, solid-size-vs-beam-diameter, and element-length-ratio
   assumptions from Steinbrecher et al. — **only if all three pass**:
8. {py:func}`~stentfit.sim_setup.build_smoketest_input` adds the static
   solver header, fixes the artery inlet/outlet, and applies the radial
   "balloon" expansion load, writing the schema-validated
   `simulation.4C.yaml`.

See the [README](index.md) for what each step's output file is, and the
{doc}`API reference <autoapi/index>` for full function signatures.

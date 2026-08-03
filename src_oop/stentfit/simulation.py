from pathlib import Path

import numpy as np
from plotly import graph_objects as go
from plotly import io as pio
from scipy.spatial import cKDTree

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySet
from beamme.core.rotation import Rotation
from beamme.cosserat_curve.cosserat_curve import CosseratCurve
from beamme.cosserat_curve.warping_along_cosserat_curve import warp_mesh_along_curve
from beamme.four_c.header_functions import (set_beam_to_solid_meshtying,
                                            set_header_static,
                                            set_runtime_output)
from beamme.four_c.input_file import InputFile

from .artery import Artery
from .stent import Stent
from .core import artery_geom as _geom
from .core import splines as _splines


def _radial_directions(points: np.ndarray, artery_cl: np.ndarray) -> np.ndarray:
    """
    Find each point's outward radial direction relative to a centreline.

    For each point, the nearest centreline point is found (KD-tree), and the
    vector from that centreline point to the point has its tangent component
    projected out — leaving only the component perpendicular to the centreline,
    normalised to a unit vector. Used to point the balloon expansion force
    straight out from the artery's local axis at each stent node, however the
    artery bends.

    :param points: ``(n, 3)`` points to find the radial direction at.
    :param artery_cl: Artery centreline points.
    :returns: ``(n, 3)`` unit vectors, each point's outward radial direction.
    """
    artery_cl = np.asarray(artery_cl, dtype=float)
    tang = np.gradient(artery_cl, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)

    _, idx = cKDTree(artery_cl).query(points)
    radial = points - artery_cl[idx]
    radial -= np.einsum("ij,ij->i", radial, tang[idx])[:, None] * tang[idx]
    norm = np.linalg.norm(radial, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return radial / norm


class Simulation:
    """
    A mixed-dimensional beam-to-solid simulation setup for one stent and artery.

    Composes a :class:`~stentfit.stent.Stent` and an
    :class:`~stentfit.artery.Artery` into a runnable 4C input: the stent is
    meshed as 1D beams and warped onto the artery centreline, the artery wall
    is meshed as a 3D solid, the two are tied together with BeamMe's mortar
    beam-to-solid coupling, and — provided the coupling assumptions hold — a
    static solver header, boundary conditions and a quasi-static radial
    expansion load are written out::

        artery = Artery(stent, artery_type="curved", inner_margin=0.5)
        sim = Simulation(stent, artery, "outputs/simulation/input")
        sim.setup()

    The artery is built first and passed in, so every artery-shape and
    wall-material knob lives on :class:`~stentfit.artery.Artery` and everything
    here concerns the stent, the coupling, and the load. Build both against the
    *same* stent — the constructor rejects a mismatch.

    This is a **smoke test**, not the physics of the reference papers: the
    artery uses a placeholder ``StVenantKirchhoff`` material, coupling is tied
    meshtying rather than true contact, and the balloon is a simplified radial
    point force.

    :param stent: The stent to deploy. Its skeletonisation must have run, since
        the beam mesh is built from the splines in its output folder.
    :param artery: The artery to deploy into, already built by
        :class:`~stentfit.artery.Artery`. Every artery-shape and wall-material
        parameter lives on that object, not here.
    :param sim_input_dir: Folder every generated ``.4C.yaml`` and ``.vtu`` is
        written into.
    :param stent_youngs: Stent beam Young's modulus, in MPa.
    :param stent_poisson: Stent beam Poisson's ratio.
    :param stent_density: Stent beam material density.
    :param beam_class_label: BeamMe beam element type, either
        ``'Beam3rHerm2Line3'`` or ``'Beam3rLine2Line2'``.
    :param factor_solid: Safety factor sizing the artery solid element size
        relative to the beam diameter.
    :param factor_beam: Additional safety factor sizing the beam element length
        beyond ``factor_solid``.
    :param n_steps: Number of load steps for the balloon expansion ramp.
    :param expansion_force: Radial point-force magnitude for the balloon expansion.
    :raises ValueError: If ``artery`` was built for a different stent.
    """

    def __init__(self: "Simulation",
                 stent: Stent,
                 artery: Artery,
                 sim_input_dir: str | Path,
                 stent_youngs: float = 2.0e5,
                 stent_poisson: float = 0.3,
                 stent_density: float = 0.0,
                 beam_class_label: str = "Beam3rHerm2Line3",
                 factor_solid: float = 1.5,
                 factor_beam: float = 1.2,
                 n_steps: int = 10,
                 expansion_force: float = 1e-4):
        # An artery sized against a different stent would pass the coupling
        # checks against one stent and be meshed around another, so catch it here.
        if artery.stent is not stent:
            raise ValueError(
                "this artery was built for a different stent "
                f"({artery.stent.stent_name!r} vs {stent.stent_name!r}) - build "
                f"the artery with Artery(stent, ...) using the same stent.")

        # --- composed parts ---
        self.stent = stent
        self.artery = artery
        self.sim_input_dir = Path(sim_input_dir)

        # --- element sizing, both factors relative to the stent's strut ---
        self.factor_solid = factor_solid

        # --- stent beam material / discretisation ---
        self.stent_youngs = stent_youngs
        self.stent_poisson = stent_poisson
        self.stent_density = stent_density
        self.beam_class_label = beam_class_label
        self.factor_beam = factor_beam

        # --- load stepping ---
        self.n_steps = n_steps
        self.expansion_force = expansion_force

        # --- data this simulation produces ---
        self.beam_mesh = None        # warped stent beam mesh (align)
        self.full_mesh = None        # combined beam + solid mesh (assemble)
        self.coupling_report = None  # pass/fail checks (check_coupling)

    # ------------------------------------------------------------------
    # Derived element sizing
    # ------------------------------------------------------------------

    @property
    def beam_diameter(self: "Simulation") -> float:
        """
        :returns: The beam cross-section diameter — the stent's strut
            thickness, read straight off the composed stent, in mm.
        """
        return self.stent.stent_features["strut_thickness"]

    @property
    def solid_element_size(self: "Simulation") -> float:
        """
        :returns: Target artery solid element size: the beam diameter with the
            ``factor_solid`` safety factor applied, in mm.
        """
        return self.beam_diameter * self.factor_solid

    @property
    def beam_element_size(self: "Simulation") -> float:
        """
        :returns: Target beam element length: the solid element size with the
            further ``factor_beam`` safety factor applied, in mm.
        """
        return self.beam_diameter * self.factor_solid * self.factor_beam

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def print_stent_summary(self: "Simulation") -> None:
        """
        Print the stent's key dimensions, as a sanity check before meshing.

        The values come from the live :class:`~stentfit.stent.Stent` object, so
        unlike the procedural pipeline nothing is re-read from
        ``stent_features.json`` / ``skeleton_points.csv``. A stent restored with
        :meth:`~stentfit.stent.Stent.load` has no 3D skeleton in memory, so the
        node count is only printed when it is available.
        """
        f = self.stent.stent_features
        print(f"Loaded stent result from : {Path(self.stent.output_dir).resolve()}")
        print(f"Centreline direction     : "
              f"{np.asarray(self.stent.stent_centerline_direction).round(4)}")
        print(f"length          : {f['length']:8.3f} mm")
        print(f"diameter        : {f['diameter']:8.3f} mm")
        print(f"r_outer         : {f['r_outer']:8.3f} mm")
        print(f"strut_thickness : {f['strut_thickness']:8.3f} mm")
        print(f"z range         : [{f['z_min']:.3f}, {f['z_max']:.3f}] mm")
        print(f"sampled points  : {f['num_points']:,}")
        if self.stent.skeleton_df is not None:
            print(f"skeleton nodes  : {len(self.stent.skeleton_df):,}")

    def mesh_artery(self: "Simulation") -> Path:
        """
        Mesh the artery wall as a 3D solid, sized to this simulation's stent.

        Thin wrapper over :meth:`~stentfit.artery.Artery.mesh_solid` that fills
        in the element size (which depends on the stent's strut thickness, so
        the artery cannot work it out alone) and the output path.

        :returns: Path to the written ``artery_solid.4C.yaml``.
        """
        return self.artery.mesh_solid(
            out_path=self.sim_input_dir / "artery_solid.4C.yaml",
            element_size=self.solid_element_size)

    def align(self: "Simulation") -> "Simulation":
        """
        Mesh the straight stent as beams and warp it onto the artery centreline.

        Builds the beam mesh from the stent's fitted splines, represents the
        artery centreline as a BeamMe ``CosseratCurve``, then warps the straight
        stent onto it — rotating the stent's own straight axis onto the curve's
        tangent, and centring the stent's ``z_min``/``z_max`` mid-point on the
        curve's arc mid-point. Writes ``stent_warped.4C.yaml``.

        Sets :attr:`beam_mesh`.

        :returns: ``self``, so steps can be chained.
        """
        # 1. Build the straight stent as a BeamMe beam mesh from the fitted
        #    splines in the stent's output folder.
        self.beam_mesh = _splines.mesh_skeleton_beams(
            input_dir=str(self.stent.output_dir),
            l_el=self.beam_element_size,
            youngs_modulus=self.stent_youngs,
            poisson_ratio=self.stent_poisson,
            density=self.stent_density,
            beam_class_label=self.beam_class_label)

        # 2. Represent the artery centreline as a Cosserat curve.
        artery_cl = self.artery.centreline
        curve = CosseratCurve(artery_cl)

        # 3. Warp the straight stent onto the curve.
        features = self.stent.stent_features
        ref_rot = Rotation([0.0, 1.0, 0.0], -np.pi / 2.0)   # first basis vector -> +Z
        total_arc = np.linalg.norm(np.diff(artery_cl, axis=0), axis=1).sum()
        z_center = 0.5 * (features["z_min"] + features["z_max"])
        origin = np.array([0.0, 0.0, z_center - total_arc / 2.0])

        warp_mesh_along_curve(self.beam_mesh, curve, origin=origin,
                              reference_rotation=ref_rot)

        # Save the warped stent beam mesh as a 4C .yaml
        stent_yaml = self.sim_input_dir / "stent_warped.4C.yaml"
        stent_input = InputFile()
        stent_input.add(self.beam_mesh)
        stent_input.dump(str(stent_yaml), validate=False,
                         add_footer_application_script=False)
        print(f"[saved] {stent_yaml}")
        return self

    def assemble(self: "Simulation",
                 lumen_surface_index: int = 0,
                 bc_type=None,
                 output_filename: str = "artery_stent.4C.yaml") -> "Simulation":
        """
        Import the artery solid and tie the stent beam mesh to it, as one 4C input.

        Imports the artery's solid ``.yaml`` (written by
        :meth:`~stentfit.artery.Artery.mesh_solid`), then couples it to
        :attr:`beam_mesh` with BeamMe's mortar beam-to-solid method, writing the
        combined 4C input file. Coupling defaults to tied meshtying; pass
        ``bme.bc.beam_to_solid_surface_contact`` for a real deployment
        simulation instead of this smoke test.

        Sets :attr:`full_mesh`.

        :param lumen_surface_index: Index into the artery solid's surface sets
            for the lumen surface the beams couple to. ``0`` is the lumen
            (``DSURFACE 1``, written first by the mesher).
        :param bc_type: BeamMe beam-to-solid coupling type. ``None`` defaults to
            tied meshtying.
        :param output_filename: Filename for the assembled 4C input, written
            into :attr:`sim_input_dir`.
        :returns: ``self``, so steps can be chained.
        """
        if self.artery is None or self.artery.solid_yaml is None:
            print("No artery solid mesh from the previous step — skipping assembly.")
            return self

        input_file, solid = _geom.import_artery_solid(self.artery.solid_yaml)

        out_path = self.sim_input_dir / output_filename
        _, self.full_mesh = _geom.assemble_beam_solid(
            input_file, solid, self.beam_mesh,
            lumen_surface_index=lumen_surface_index,
            # bc_type defaults to beam_to_solid_surface_meshtying; switch to
            # bme.bc.beam_to_solid_surface_contact for a real deployment run.
            bc_type=bc_type,
            output_path=out_path,
        )

        print(f"Wrote assembled beam-to-solid 4C input file -> {out_path}")
        print("Next: materials (HGO-C artery), boundary conditions, expansion "
              "driver, solver, run 4C.")
        return self

    def export_paraview(self: "Simulation", output_name: str = "artery_stent_mesh") -> tuple | None:
        """
        Export the assembled beam+solid mesh as separate ``.vtu`` files for ParaView.

        BeamMe's ``write_vtk`` splits beams and solid elements into two files by
        itself; this just names them and reports their paths.

        :param output_name: Base filename; ``_beam.vtu`` / ``_solid.vtu`` are appended.
        :returns: ``None`` if nothing has been assembled yet. Otherwise
            ``(beam_vtu, solid_vtu)`` — the paths to the two written files.
        """
        if self.full_mesh is None:
            print("No assembled mesh — run assemble() first.")
            return None

        self.full_mesh.write_vtk(output_name=output_name,
                                 output_directory=str(self.sim_input_dir))
        beam_vtu = self.sim_input_dir / f"{output_name}_beam.vtu"
        solid_vtu = self.sim_input_dir / f"{output_name}_solid.vtu"
        print(f"[vtk] {beam_vtu}")
        print(f"[vtk] {solid_vtu}")
        print("Open the .vtu files in ParaView to inspect the meshes.")
        return beam_vtu, solid_vtu

    def check_coupling(self: "Simulation",
                       stiffness_ratio_min: float = 10.0,
                       length_ratio_min: float = 1,
                       length_ratio_max: float = 6,
                       length_ratio_accuracy_max: float = 8.0) -> dict:
        """
        Check the mixed-dimensional beam-to-solid coupling assumptions.

        Three checks, following Steinbrecher et al., each independent:

        1. **Stiffness** — the beam must be much stiffer than the solid
           (``E_beam / E_solid >= stiffness_ratio_min``), since the coupling
           assumes the solid deforms around an effectively rigid-ish beam.
        2. **Solid size vs. beam diameter** — the solid element size must be at
           least the beam's cross-section diameter, the spatial-resolution limit
           the mortar coupling is only valid above.
        3. **Element length ratio** — beam elements should be longer than solid
           elements, but not by too much: a valid band up to
           ``length_ratio_accuracy_max``, and a narrower optimal band up to
           ``length_ratio_max``.

        The beam element length is measured from the meshed beams themselves
        (mean end-to-end chord), not from the requested target.

        Sets :attr:`coupling_report`.

        :param stiffness_ratio_min: Minimum acceptable ``E_beam / E_solid``.
        :param length_ratio_min: Lower bound of both the valid and optimal
            ``L_beam / L_solid`` bands.
        :param length_ratio_max: Upper bound of the optimal band.
        :param length_ratio_accuracy_max: Upper bound of the valid band, above
            which coupling accuracy degrades.
        :raises ValueError: If the beam mesh has not been built yet.
        :returns: The report dict, one entry per check plus ``all_passed``.
        """
        if self.beam_mesh is None:
            raise ValueError("no beam mesh yet - call align() first.")

        # Actual mean beam element length (chord, end-to-end) from the meshed beams.
        beam_element_length = float(np.mean([
            np.linalg.norm(np.asarray(el.nodes[-1].coordinates)
                           - np.asarray(el.nodes[0].coordinates))
            for el in self.beam_mesh.elements]))

        beam_youngs = self.stent_youngs
        solid_youngs = self.artery.artery_youngs
        beam_diameter = self.beam_diameter
        solid_element_length = self.solid_element_size

        checks = {}

        # 1. Stiffness ratio ---------------------------------------------------
        stiff = beam_youngs / solid_youngs if solid_youngs > 0 else float("inf")
        ok_stiff = stiff >= stiffness_ratio_min
        checks["stiffness"] = dict(
            E_beam_MPa=beam_youngs,
            E_solid_MPa=solid_youngs,
            ratio=round(stiff, 2),
            threshold_min=stiffness_ratio_min,
            passed=bool(ok_stiff),
            note=(f"E_beam/E_solid = {stiff:.1f} "
                  f"({'>=' if ok_stiff else '<'} {stiffness_ratio_min}) "
                  f"- beam {'is' if ok_stiff else 'is NOT'} much stiffer than the solid"),
        )

        # 2. Solid element size >= beam cross-section diameter -----------------
        rot = solid_element_length / beam_diameter if beam_diameter > 0 else float("inf")
        ok_rot = solid_element_length >= beam_diameter
        checks["solid_size_vs_beam_diameter"] = dict(
            solid_element_mm=round(solid_element_length, 4),
            beam_diameter_mm=round(beam_diameter, 4),
            ratio=round(rot, 2),
            threshold_min=1.0,
            passed=bool(ok_rot),
            note=(f"L_solid/D_beam = {rot:.2f} "
                  f"({'>=' if ok_rot else '<'} 1) - solid element "
                  f"{'>=' if ok_rot else '<'} beam cross-section diameter"),
        )

        # 3. Element length ratio: beam elements long vs solid, but not too long
        #    Valid band [length_ratio_min, length_ratio_accuracy_max]: below ->
        #    mortar coupling poorly conditioned; above ~8 -> L2 error grows.
        lr = (beam_element_length / solid_element_length
              if solid_element_length > 0 else float("inf"))
        ok_lr = length_ratio_min <= lr <= length_ratio_accuracy_max
        within_optimal = length_ratio_min <= lr <= length_ratio_max
        if lr > length_ratio_accuracy_max:
            lr_note = (f"L_beam/L_solid = {lr:.2f} (> {length_ratio_accuracy_max}) "
                       f"- too long: L2 coupling error grows; refine the solid or "
                       f"coarsen the beam")
        elif lr < length_ratio_min:
            lr_note = (f"L_beam/L_solid = {lr:.2f} (< {length_ratio_min}) "
                       f"- beam elements are NOT fairly long vs solid elements")
        elif within_optimal:
            lr_note = (f"L_beam/L_solid = {lr:.2f} - in the optimal "
                       f"{length_ratio_min}-{length_ratio_max} band")
        else:
            lr_note = (f"L_beam/L_solid = {lr:.2f} - acceptable "
                       f"(above the {length_ratio_max} optimum, below the "
                       f"{length_ratio_accuracy_max} accuracy limit)")
        checks["element_length_ratio"] = dict(
            beam_element_mm=round(beam_element_length, 4),
            solid_element_mm=round(solid_element_length, 4),
            ratio=round(lr, 2),
            valid_band=(length_ratio_min, length_ratio_accuracy_max),
            optimal_band=(length_ratio_min, length_ratio_max),
            within_optimal=bool(within_optimal),
            passed=bool(ok_lr),
            note=lr_note,
        )

        all_passed = all(c["passed"] for c in checks.values())
        checks["all_passed"] = all_passed

        print("\nMixed-dimensional coupling assumption check")
        print("-------------------------------------------")
        for name in ("stiffness", "solid_size_vs_beam_diameter", "element_length_ratio"):
            c = checks[name]
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name:28s} {c['note']}")
        print(f"  => {'ALL CHECKS PASSED' if all_passed else 'ONE OR MORE CHECKS FAILED'}")

        self.coupling_report = checks
        return checks

    def plot_overview(self: "Simulation", show: bool = True) -> Path:
        """
        Draw the artery surface, its centreline, and the warped stent together.

        Writes ``stent_artery_view.html`` into :attr:`sim_input_dir`, and shows
        the figure inline when running in a notebook.

        :param show: Try to display the figure inline as well as saving it.
        :returns: Path to the written HTML view.
        """
        elem_coords = np.array([[n.coordinates for n in el.nodes]
                                for el in self.beam_mesh.elements])   # (n_el, 3, 3)
        # 3 nodes + a NaN gap, so the line breaks between elements
        seg = np.full((len(elem_coords), 4, 3), np.nan)
        seg[:, :3, :] = elem_coords
        beam_lines = seg.reshape(-1, 3)

        artery_cl = self.artery.centreline
        verts = np.asarray(self.artery.geometry.vertices)
        faces = np.asarray(self.artery.geometry.faces)

        fig = go.Figure()
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="lightpink", opacity=0.25, name="artery", showscale=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=artery_cl[:, 0], y=artery_cl[:, 1], z=artery_cl[:, 2],
            mode="lines", line=dict(color="gray", width=3, dash="dash"),
            name="centreline",
        ))
        fig.add_trace(go.Scatter3d(
            x=beam_lines[:, 0], y=beam_lines[:, 1], z=beam_lines[:, 2],
            mode="lines", line=dict(color="crimson", width=2), name="stent beams",
        ))
        fig.update_layout(
            title=f"Stent warped onto {self.artery.artery_type} artery — "
                  f"{self.stent.stent_name} "
                  f"({len(self.beam_mesh.elements):,} beam elements)",
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        stent_artery_html = self.sim_input_dir / "stent_artery_view.html"
        pio.write_html(fig, str(stent_artery_html), auto_open=False)
        print(f"[saved] {stent_artery_html}")
        if show:
            try:
                fig.show()
            except Exception as e:
                print(f"[plotly] interactive view skipped ({e}); "
                      f"use {stent_artery_html.name} instead")
        return stent_artery_html

    def write_input(self: "Simulation",
                    out_path: str | Path | None = None,
                    total_time: float = 1.0,
                    inlet_surface_index: int = 1,
                    outlet_surface_index: int = 2,
                    fix_stent_node: bool = True) -> Path:
        """
        Build a runnable, schema-validated 4C static simulation input.

        Adds a static solver header and runtime VTK output, fixes the artery's
        inlet and outlet surfaces (3 translational DOF, Dirichlet), and applies
        a quasi-static radial "balloon" expansion: a point force at each beam
        centreline node, directed radially outward from the artery centreline
        and ramped from 0 to :attr:`expansion_force` over :attr:`n_steps` by a
        time function. If ``fix_stent_node``, one stent node is also pinned in
        translation to remove the stent's rigid-body motion, since the radial
        forces alone do not constrain it.

        :param out_path: File path the simulation input is written to. ``None``
            writes ``simulation.4C.yaml`` into :attr:`sim_input_dir`.
        :param total_time: Total simulation time for the static solver.
        :param inlet_surface_index: Index into the solid's surface geometry sets
            for the inlet (fixed) surface.
        :param outlet_surface_index: Index into the solid's surface geometry
            sets for the outlet (fixed) surface.
        :param fix_stent_node: Pin one stent centreline node's translation, to
            remove rigid-body motion.
        :raises ValueError: If nothing has been assembled yet, or the imported
            solid is missing the inlet/outlet surface sets.
        :returns: The path written.
        """
        if self.full_mesh is None:
            raise ValueError("no assembled mesh yet - call assemble() first.")
        if out_path is None:
            out_path = self.sim_input_dir / "simulation.4C.yaml"

        mesh = self.full_mesh
        artery_cl = self.artery.centreline
        inp = InputFile()

        # --- Solver control (static, quasi-static steps) + runtime VTK output --
        set_header_static(inp, n_steps=self.n_steps, total_time=total_time)
        set_runtime_output(inp)
        # Beam-to-solid interaction header (the coupling *condition* is already
        # on the mesh).
        set_beam_to_solid_meshtying(inp, bme.bc.beam_to_solid_surface_meshtying,
                                    contact_discretization="mortar",
                                    mortar_shape="line2")

        # --- BCs: fix the artery inlet + outlet ends (solid, 3 DOF) -----------
        surf = mesh.geometry_sets.get(bme.geo.surface, [])
        if len(surf) <= max(inlet_surface_index, outlet_surface_index):
            raise ValueError("Imported solid is missing the inlet/outlet surface sets.")
        for i in (inlet_surface_index, outlet_surface_index):
            mesh.add(BoundaryCondition(
                surf[i],
                {"NUMDOF": 3, "ONOFF": [1, 1, 1], "VAL": [0, 0, 0], "FUNCT": [0, 0, 0]},
                bc_type=bme.bc.dirichlet))

        # --- Radial "balloon" expansion force on the stent centreline nodes ---
        # Beam3rHerm2Line3 nodes carry 9 DOF [disp(3), rot(3), tangent(3)]; only
        # the Hermite centreline nodes (is_middle_node == False) have
        # translational DOF. A time-ramp FUNCT (f(t) = t) scales the force from 0
        # to full over the steps.
        ramp = Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}])
        mesh.add(ramp)

        cnodes = [n for n in self.beam_mesh.nodes if not n.is_middle_node]
        coords = np.array([n.coordinates for n in cnodes])
        radial = _radial_directions(coords, artery_cl)

        if fix_stent_node:              # remove stent rigid-body translation
            mesh.add(BoundaryCondition(
                GeometrySet([cnodes[0]]),
                {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
                 "VAL": [0] * 9, "FUNCT": [0] * 9},
                bc_type=bme.bc.dirichlet))

        for node, r in zip(cnodes, radial):
            f = self.expansion_force * r
            mesh.add(BoundaryCondition(
                GeometrySet([node]),
                {"NUMDOF": 9,
                 "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
                 "VAL": [f[0], f[1], f[2], 0, 0, 0, 0, 0, 0],
                 "FUNCT": [ramp, ramp, ramp, 0, 0, 0, 0, 0, 0]},
                bc_type=bme.bc.neumann))

        # --- Assemble + schema-validate + write -------------------------------
        inp.add(mesh)
        inp.dump(str(out_path), validate=True, add_footer_application_script=False)

        print(f"[sim] static smoke test: {self.n_steps} steps, radial expansion "
              f"force {self.expansion_force:g} N ramped over {len(cnodes):,} stent nodes")
        print(f"[sim] BCs: artery inlet+outlet fixed"
              + (", one stent node pinned" if fix_stent_node else "")
              + "; coupling = beam-to-solid meshtying (tied)")
        print(f"[saved] {out_path}")
        print("[sim] schema-validated. Run in 4C on Linux: set BEAMME_FOUR_C_EXE "
              "and launch 4C on this file.")
        return Path(out_path)

    # ------------------------------------------------------------------
    # Full chain
    # ------------------------------------------------------------------

    def setup(self: "Simulation", show_plot: bool = True) -> "Simulation":
        """
        Prepare a runnable 4C input, from the stent and artery through to the load.

        Chains the whole synthetic pipeline: prints the stent summary, meshes
        the stent as beams and warps it onto the artery centreline
        (:meth:`align`), meshes the artery wall as a 3D solid
        (:meth:`mesh_artery`), assembles the
        beam-to-solid mesh (:meth:`assemble`) and exports it for ParaView
        (:meth:`export_paraview`). It then checks the coupling assumptions
        (:meth:`check_coupling`), shows the overview plot, and — **only if those
        checks pass** — writes the runnable input (:meth:`write_input`).

        Named ``setup`` rather than ``run`` on purpose: it *prepares* a runnable
        4C input, it does not execute the analysis. Running the solve is 4C's
        job, external to this package.

        :param show_plot: Display the artery/stent overview figure inline.
        :returns: ``self``, holding the meshes and the coupling report.
        """
        self.sim_input_dir.mkdir(parents=True, exist_ok=True)

        # Stent features, for the test-artery sizing
        print("\nStent features")
        print("--------------")
        self.print_stent_summary()

        # Stent meshing and alignment with the artery
        print("\n Stent Meshing and Alignment")
        print("--------------")
        self.align()

        # Mesh the artery WALL into a 3D solid with GMSH and write it as a 4C .yaml.
        print("\n Test_Artery Meshing")
        print("--------------")
        self.mesh_artery()

        # Create the assembly mesh for the stent and artery, and write it as a 4C .yaml.
        print('\n Assembly of Stent and Test_Artery')
        print("--------------")
        self.assemble()

        # Export the assembled mesh as separate .vtu files for ParaView.
        print("\n Paraview")
        self.export_paraview()

        # Check the stent-artery fit and coupling assumptions
        self.check_coupling()
        coupling_ok = self.coupling_report["all_passed"]
        if not coupling_ok:
            print("\n[!] Coupling assumptions not satisfied — retune L_EL / "
                  "SOLID_ELEMENT_SIZE (and the moduli) before building the "
                  "simulation input.")

        # Visualisation of geometries
        self.plot_overview(show=show_plot)

        # Build the runnable 4C simulation input: static solver + boundary
        # conditions + a radial "balloon" expansion force on the stent, on top of
        # the assembled beam-to-solid mesh. Gated on the coupling checks. Smoke
        # test: placeholder material + meshtying (tied). The file is
        # schema-validated here; running it needs a 4C binary on Linux.
        if not coupling_ok:
            print("[skip] Coupling checks failed — fix those before building "
                  "the simulation.")
        else:
            simulation_yaml = self.write_input()
            print(f"\nSimulation input ready -> {simulation_yaml}")

        return self

    def __repr__(self: "Simulation") -> str:
        """:returns: A short summary of how far this simulation has been set up."""
        bits = []
        if self.beam_mesh is not None:
            bits.append(f"{len(self.beam_mesh.elements):,} beams")
        if self.full_mesh is not None:
            bits.append(f"{len(self.full_mesh.elements):,} total elements")
        if self.coupling_report is not None:
            bits.append("coupling "
                        + ("OK" if self.coupling_report["all_passed"] else "FAILED"))
        stage = ", ".join(bits) or "not set up"
        return f"<Simulation {self.stent.stent_name!r} [{stage}] -> {self.sim_input_dir}>"

import numpy as np
from scipy.spatial import cKDTree
from beamme.core.conf import bme
from beamme.core.geometry_set import GeometrySet
from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.function import Function
from beamme.four_c.input_file import InputFile
from beamme.four_c.header_functions import (set_header_static, set_runtime_output, set_beam_to_solid_meshtying)


def _radial_directions(points: np.ndarray, artery_cl: np.ndarray) -> np.ndarray:
    """Outward radial unit vector per point, relative to the artery centreline.

    For each point, take the vector from its nearest centreline point, remove the
    component along the local centreline tangent, and normalise — so the force is
    purely radial (perpendicular to the vessel axis) for straight and curved arteries.
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


def build_smoketest_input(
    mesh,
    beam_mesh,
    artery_cl,
    out_path,
    *,
    n_steps: int = 10,
    total_time: float = 1.0,
    expansion_force: float = 1e-4,
    inlet_surface_index: int = 1,
    outlet_surface_index: int = 2,
    fix_stent_node: bool = True,
):
    """Add solver / BCs / expansion load to the assembled mesh and dump a 4C input.

    Parameters
    ----------
    mesh        : assembled BeamMe Mesh (solid artery + stent beams + coupling),
                  as returned by ``assemble_beam_solid`` — carries the surface sets
                  (0 = lumen, 1 = inlet, 2 = outlet) and the beam-to-solid condition.
    beam_mesh   : the warped stent beam Mesh (used to pick the loaded stent nodes).
    artery_cl   : (M, 3) artery centreline (for the radial load direction).
    out_path    : where to write ``simulation.4C.yaml``.
    n_steps, total_time : quasi-static load stepping (force ramps 0 -> full over the time).
    expansion_force     : radial outward point force per stent node at full load [N].
    inlet_surface_index, outlet_surface_index : which imported surface sets are the ends.
    fix_stent_node      : pin one stent node's translation to remove rigid-body modes.

    Returns
    -------
    out_path : the written, schema-validated ``simulation.4C.yaml``.
    """


    inp = InputFile()

    # --- Solver control (static, quasi-static steps) + runtime VTK output ------
    set_header_static(inp, n_steps=n_steps, total_time=total_time)
    set_runtime_output(inp)
    # Beam-to-solid interaction header (the coupling *condition* is already on the mesh).
    set_beam_to_solid_meshtying(inp, bme.bc.beam_to_solid_surface_meshtying,
                                contact_discretization="mortar", mortar_shape="line2")

    # --- Boundary conditions: fix the artery inlet + outlet ends (solid, 3 DOF) -
    surf = mesh.geometry_sets.get(bme.geo.surface, [])
    if len(surf) <= max(inlet_surface_index, outlet_surface_index):
        raise ValueError("Imported solid is missing the inlet/outlet surface sets.")
    for i in (inlet_surface_index, outlet_surface_index):
        mesh.add(BoundaryCondition(
            surf[i], {"NUMDOF": 3, "ONOFF": [1, 1, 1], "VAL": [0, 0, 0], "FUNCT": [0, 0, 0]},
            bc_type=bme.bc.dirichlet))

    # --- Radial "balloon" expansion force on the stent centreline nodes --------
    # Beam3rHerm2Line3 nodes carry 9 DOF [disp(3), rot(3), tangent(3)]; only the
    # Hermite centreline nodes (is_middle_node == False) have translational DOF.
    # A time-ramp FUNCT (f(t) = t) scales the force from 0 to full over the steps.
    ramp = Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}])
    mesh.add(ramp)

    cnodes = [n for n in beam_mesh.nodes if not n.is_middle_node]
    coords = np.array([n.coordinates for n in cnodes])
    radial = _radial_directions(coords, artery_cl)

    if fix_stent_node:                      # remove stent rigid-body translation
        mesh.add(BoundaryCondition(
            GeometrySet([cnodes[0]]),
            {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0], "VAL": [0] * 9, "FUNCT": [0] * 9},
            bc_type=bme.bc.dirichlet))

    for node, r in zip(cnodes, radial):
        f = expansion_force * r
        mesh.add(BoundaryCondition(
            GeometrySet([node]),
            {"NUMDOF": 9,
             "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
             "VAL": [f[0], f[1], f[2], 0, 0, 0, 0, 0, 0],
             "FUNCT": [ramp, ramp, ramp, 0, 0, 0, 0, 0, 0]},
            bc_type=bme.bc.neumann))

    # --- Assemble + schema-validate + write ------------------------------------
    inp.add(mesh)
    inp.dump(str(out_path), validate=True, add_footer_application_script=False)

    print(f"[sim] static smoke test: {n_steps} steps, radial expansion force "
          f"{expansion_force:g} N ramped over {len(cnodes):,} stent nodes")
    print(f"[sim] BCs: artery inlet+outlet fixed"
          + (", one stent node pinned" if fix_stent_node else "")
          + "; coupling = beam-to-solid meshtying (tied)")
    print(f"[saved] {out_path}")
    print("[sim] schema-validated. Run in 4C on Linux: set BEAMME_FOUR_C_EXE and launch 4C on this file.")
    return out_path

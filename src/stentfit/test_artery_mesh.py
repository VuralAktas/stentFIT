import numpy as np
import gmsh
from pathlib import Path
from stentfit.process_funcs import _compute_rmf
from pathlib import Path

from beamme.four_c.model_importer import (import_cubitpy_model,
                                          import_four_c_model)
from beamme.core.conf import bme
from beamme.core.geometry_set import GeometrySet
from beamme.four_c.beam_interaction_conditions import add_beam_interaction_condition
from beamme.four_c.header_functions import set_beam_to_solid_meshtying


# ---------------------------------------------------------------------------
# 1. Artery solid meshing (GMSH) — Mac-native, writes a 4C .yaml solid
# ---------------------------------------------------------------------------

# gmsh element type -> (4C keyword, nodes/elem, gmsh->4C node permutation).
# TET4/HEX8 share gmsh's ordering with VTK/4C (identity). For TET10, gmsh's last
# two mid-edge nodes (edges 2-3, 1-3) are swapped relative to VTK/4C (edges 1-3, 2-3),
# so nodes 8 and 9 are swapped. (Derived from gmsh local node coords + BeamMe's
# VolumeTET*/HEX8 vtk_topology == range, i.e. 4C order == VTK order.)
_MESH_SPEC = {
    "TET4":  dict(gtype=4,  nnode=4,  keyword="TET4",  perm=[0, 1, 2, 3]),
    "TET10": dict(gtype=11, nnode=10, keyword="TET10", perm=[0, 1, 2, 3, 4, 5, 6, 7, 9, 8]),
    "HEX8":  dict(gtype=5,  nnode=8,  keyword="HEX8",  perm=[0, 1, 2, 3, 4, 5, 6, 7]),
}


def _apply_radial_noise(coords, length, amplitude, seed):
    """Fractional radial perturbation of node radii in the straight (+Z) config.

    A smooth low-frequency field of the axial position and angle, tapered to zero
    over the outer 10% at each end so the inlet/outlet faces stay clean rings.
    ``amplitude`` is the fractional radius variation (e.g. 0.1 = +/-10%).
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    rng = np.random.default_rng(seed)
    field = np.zeros_like(z)
    for kz, kphi in [(1, 2), (2, 3), (3, 5), (2, 6)]:
        field += (rng.uniform(-1, 1)
                  * np.sin(2 * np.pi * kz * z / length + rng.uniform(0, 2 * np.pi))
                  * np.cos(kphi * phi + rng.uniform(0, 2 * np.pi)))
    field /= np.abs(field).max() + 1e-12
    taper = np.clip(np.minimum(z, length - z) / (0.1 * length), 0.0, 1.0)
    r_new = r * (1.0 + amplitude * field * taper)
    out = coords.copy()
    out[:, 0] = r_new * np.cos(phi)
    out[:, 1] = r_new * np.sin(phi)
    return out


def _warp_coords_to_centreline(coords, centreline):
    """Warp straight (+Z) node coordinates onto a 3D centreline with a
    rotation-minimising frame (the same ``_compute_rmf`` used by
    ``map_stent_to_artery``): each node's z is its arc-length coordinate, and its
    (x, y) cross-section is placed via the local normal / binormal.

    The artery wall's cross-section is a circle, so the azimuthal orientation of
    the frame does not change the solid — the tube simply follows the centreline
    path and stays co-located with the stent (also warped onto ``artery_cl``). A
    straight centreline along +Z leaves the tube unchanged. Returns (N, 3).
    """

    cl = np.asarray(centreline, dtype=float)
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])       # arc length at each cl point
    _, N, B = _compute_rmf(cl)

    s = np.clip(coords[:, 2], 0.0, arc[-1])             # tube built along +Z from 0
    cl_s = np.column_stack([np.interp(s, arc, cl[:, k]) for k in range(3)])
    N_s = np.column_stack([np.interp(s, arc, N[:, k]) for k in range(3)])
    B_s = np.column_stack([np.interp(s, arc, B[:, k]) for k in range(3)])
    N_s /= np.linalg.norm(N_s, axis=1, keepdims=True)
    B_s /= np.linalg.norm(B_s, axis=1, keepdims=True)
    return cl_s + coords[:, 0:1] * N_s + coords[:, 1:2] * B_s


def _count_inverted(coords, conn, mesh_type):
    """Number of negative-Jacobian elements (corner orientation check)."""
    P = coords[conn - 1]
    if mesh_type.startswith("TET"):
        v = np.einsum("ij,ij->i", P[:, 1] - P[:, 0],
                      np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))
    else:  # HEX8 corner triple product
        v = np.einsum("ij,ij->i", P[:, 1] - P[:, 0],
                      np.cross(P[:, 3] - P[:, 0], P[:, 4] - P[:, 0]))
    return int((v < 0).sum())


def mesh_artery_gmsh(
    r_inner: float,
    r_outer: float,
    centreline,
    out_path,
    *,
    mesh_type: str = "TET4",
    element_size: float | None = None,
    noise_amplitude: float = 0.0,
    noise_seed: int | None = None,
    material_id: int = 1,
    youngs_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
    density: float = 1.0,
):
    """Mesh a hollow artery wall along a centreline with GMSH; write a 4C ``.yaml`` solid.

    Builds the wall between the lumen radius ``r_inner`` and outer radius ``r_outer``
    as a straight hollow tube (length = arc length of ``centreline``), optionally adds
    biological wall roughness, then **warps it onto ``centreline``** so the solid
    follows any straight / curved / s-bend artery — using the same ``CosseratCurve``
    convention as the stent warp, so the solid stays aligned with the warped stent.
    Writes a 4C input file with:

    * ``NODE COORDS`` and ``STRUCTURE ELEMENTS`` (``SOLID <TYPE> ... MAT n KINEM nonlinear``),
    * ``DSURF-NODE TOPOLOGY`` with **DSURFACE 1 = lumen** (inner surface — the
      beam-to-solid coupling target), **2 = inlet**, **3 = outlet** (end faces, for BCs),
    * a placeholder ``MAT_Struct_StVenantKirchhoff`` material (upgrade to HGO-C later).

    The resulting file feeds ``import_artery_solid`` unchanged.

    ``mesh_type`` selects the solid element:

    * ``"TET4"``  — linear tets (unstructured; fast, fine for the placeholder material).
    * ``"TET10"`` — quadratic tets (unstructured; use for the near-incompressible HGO
      artery material, where linear tets lock).
    * ``"HEX8"``  — structured hexahedra (transfinite sectors swept along the axis;
      matches the reference papers, fewest elements, best for the wall material).

    Parameters
    ----------
    r_inner, r_outer : float   lumen and outer wall radii [mm] (r_outer > r_inner)
    centreline       : (M, 3) array   artery centreline the wall is warped onto
                               (``artery_cl`` from ``generate_artery_for_stent``)
    out_path         : path    where to write the 4C ``.yaml`` solid
    mesh_type        : str     "TET4" | "TET10" | "HEX8"
    element_size     : float   target element edge length [mm]; default = wall
                               thickness. Sets tet size, and the structured hex
                               divisions (circumference / radial / axial). Tune vs.
                               beam length so the solid/beam element ratio stays
                               ~2.5-5 and the solid element size >= the beam diameter
    noise_amplitude  : float   fractional wall roughness (0 = smooth; 0.1 = +/-10% of
                               radius), tapered to zero at the inlet / outlet ends
    noise_seed       : int     RNG seed for reproducible roughness
    material_id      : int     4C material id referenced by the solid elements
    youngs_modulus, poisson_ratio, density : placeholder solid material values

    Returns
    -------
    out_path : the written 4C ``.yaml`` file path
    """


    if not (r_outer > r_inner > 0):
        raise ValueError(f"Need r_outer > r_inner > 0, got {r_inner=} {r_outer=}")
    mesh_type = mesh_type.upper()
    if mesh_type not in _MESH_SPEC:
        raise ValueError(f"mesh_type must be one of {list(_MESH_SPEC)}, got {mesh_type!r}")
    spec = _MESH_SPEC[mesh_type]

    out_path = Path(out_path)
    centreline = np.asarray(centreline, dtype=float)
    length = float(np.linalg.norm(np.diff(centreline, axis=0), axis=1).sum())
    z_min = 0.0                                 # build the straight tube from z=0
    if element_size is None:
        element_size = r_outer - r_inner        # ~one element across the wall thickness

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("artery_wall")

        if mesh_type in ("TET4", "TET10"):
            # Unstructured hollow wall = outer cylinder minus inner cylinder.
            occ = gmsh.model.occ
            outer = occ.addCylinder(0, 0, z_min, 0, 0, length, r_outer)
            inner = occ.addCylinder(0, 0, z_min, 0, 0, length, r_inner)
            occ.cut([(3, outer)], [(3, inner)])
            occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMin", element_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
            gmsh.model.mesh.generate(3)
            if mesh_type == "TET10":
                gmsh.model.mesh.setOrder(2)
        else:
            # Structured hexahedra: transfinite annular sectors swept along +Z.
            n_sec    = 8                                                   # 45deg arcs (<180deg)
            n_circ_s = max(1, round(2 * np.pi * 0.5 * (r_inner + r_outer)
                                    / element_size / n_sec))
            n_radial = max(1, round((r_outer - r_inner) / element_size))
            n_axial  = max(1, round(length / element_size))
            geo = gmsh.model.geo
            centre = geo.addPoint(0, 0, z_min)
            ang  = [2 * np.pi * k / n_sec for k in range(n_sec)]
            p_in  = [geo.addPoint(r_inner * np.cos(a), r_inner * np.sin(a), z_min) for a in ang]
            p_out = [geo.addPoint(r_outer * np.cos(a), r_outer * np.sin(a), z_min) for a in ang]
            rad = [geo.addLine(p_in[k], p_out[k]) for k in range(n_sec)]   # shared radial lines
            surfs = []
            for k in range(n_sec):
                k2 = (k + 1) % n_sec
                iarc = geo.addCircleArc(p_in[k], centre, p_in[k2])
                oarc = geo.addCircleArc(p_out[k], centre, p_out[k2])
                s = geo.addPlaneSurface([geo.addCurveLoop([rad[k], oarc, -rad[k2], -iarc])])
                geo.mesh.setTransfiniteCurve(rad[k], n_radial + 1)
                geo.mesh.setTransfiniteCurve(rad[k2], n_radial + 1)
                geo.mesh.setTransfiniteCurve(iarc, n_circ_s + 1)
                geo.mesh.setTransfiniteCurve(oarc, n_circ_s + 1)
                geo.mesh.setTransfiniteSurface(s)
                geo.mesh.setRecombine(2, s)
                surfs.append(s)
            geo.synchronize()
            geo.extrude([(2, s) for s in surfs], 0, 0, length,
                        numElements=[n_axial], recombine=True)
            geo.synchronize()
            gmsh.model.mesh.generate(3)

        # Nodes, renumbered to a contiguous 1..N (gmsh tags may be sparse).
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        coords = ncoords.reshape(-1, 3)
        tag2idx = {int(t): i + 1 for i, t in enumerate(ntags)}

        # Volume elements of the requested type, reordered gmsh -> 4C.
        etypes, _, enodes = gmsh.model.mesh.getElements(dim=3)
        idx = list(etypes).index(spec["gtype"])
        conn = np.vectorize(tag2idx.get)(
            enodes[idx].reshape(-1, spec["nnode"]).astype(int))[:, spec["perm"]]

        # Classify boundary surfaces by their mesh nodes' radius / z (robust for both
        # the occ cylinders and the sectored hex): a lumen face has all nodes at
        # r ~ r_inner, an end face is flat in z; internal radial faces (nodes at both
        # radii) fall through and are ignored.
        lumen_ids, inlet_ids, outlet_ids = set(), set(), set()
        for _, tag in gmsh.model.getEntities(2):
            nt, nc, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
            if len(nt) == 0:
                continue
            c = nc.reshape(-1, 3)
            r = np.hypot(c[:, 0], c[:, 1]); z = c[:, 2]
            ids = [tag2idx[int(x)] for x in nt]
            if z.max() - z.min() < 1e-6:                         # flat end cap
                (inlet_ids if abs(z.mean() - z_min) < 1e-6 else outlet_ids).update(ids)
            elif np.all(np.abs(r - r_inner) < 1e-3):             # inner (lumen) surface
                lumen_ids.update(ids)
        lumen_ids, inlet_ids, outlet_ids = map(sorted, (lumen_ids, inlet_ids, outlet_ids))
    finally:
        gmsh.finalize()

    # Optional biological wall roughness (radial, on the straight tube; the surface
    # sets were classified on the clean tube above so they are unaffected).
    if noise_amplitude > 0:
        coords = _apply_radial_noise(coords, length, noise_amplitude, noise_seed)

    # Warp the straight tube onto the artery centreline (identity for a straight
    # centreline). Same CosseratCurve convention as the stent -> solid stays aligned.
    coords = _warp_coords_to_centreline(coords, centreline)
    n_inverted = _count_inverted(coords, conn, mesh_type)

    # --- Write the 4C .yaml solid --------------------------------------------
    kw = spec["keyword"]
    lines = [
        "MATERIALS:",
        f"  - MAT: {material_id}",
        "    MAT_Struct_StVenantKirchhoff:",
        f"      YOUNG: {youngs_modulus}",
        f"      NUE: {poisson_ratio}",
        f"      DENS: {density}",
        "STRUCTURE ELEMENTS:",
    ]
    for e, nodes in enumerate(conn, 1):
        node_str = " ".join(str(n) for n in nodes)
        lines.append(f'  - "{e} SOLID {kw} {node_str} MAT {material_id} KINEM nonlinear"')
    lines.append("NODE COORDS:")
    for i, (x, y, z) in enumerate(coords, 1):
        lines.append(f'  - "NODE {i} COORD {x:.10g} {y:.10g} {z:.10g}"')
    lines.append("DSURF-NODE TOPOLOGY:")
    for gid, ids in ((1, lumen_ids), (2, inlet_ids), (3, outlet_ids)):
        for n in ids:
            lines.append(f'  - "NODE {n} DSURFACE {gid}"')
    out_path.write_text("\n".join(lines) + "\n")

    print(f"[gmsh] artery wall meshed ({kw}): r_inner={r_inner:.3f} r_outer={r_outer:.3f} "
          f"arc length={length:.3f} mm  (element size {element_size:.3f} mm)")
    print(f"[gmsh] {len(coords):,} nodes, {len(conn):,} {kw} elements  "
          f"noise={noise_amplitude:g}, warped onto centreline")
    print(f"[gmsh] surface node sets: lumen={len(lumen_ids)} inlet={len(inlet_ids)} "
          f"outlet={len(outlet_ids)}")
    if n_inverted:
        print(f"[gmsh] WARNING: {n_inverted} inverted element(s) after warp/noise "
              f"— reduce noise_amplitude or bend, or refine the mesh")
    print(f"[saved] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2. Import the artery solid into BeamMe (4C .yaml file *or* a CubitPy model)
# ---------------------------------------------------------------------------

def import_artery_solid(source):
    """Import an artery solid mesh into BeamMe.

    ``source`` is either a path to a 4C ``.yaml`` file (from ``mesh_artery_gmsh``)
    or a ``cubitpy.CubitPy`` instance. Returns ``(input_file, solid_mesh)`` where
    ``solid_mesh`` carries the volume elements plus the surface geometry sets
    (index 0 = lumen, 1 = inlet, 2 = outlet).
    """

    if hasattr(source, "cmd") or type(source).__name__ == "CubitPy":
        input_file, solid = import_cubitpy_model(source, convert_input_to_mesh=True)
    else:
        input_file, solid = import_four_c_model(Path(source),
                                                convert_input_to_mesh=True)

    n_surf = len(solid.geometry_sets.get(bme.geo.surface, []))
    print(f"[import] solid: {len(solid.nodes):,} nodes, {len(solid.elements):,} "
          f"elements, {n_surf} surface set(s)")
    return input_file, solid


# ---------------------------------------------------------------------------
# 3. Assemble solid + beams and write the 4C input file  (fully testable)
# ---------------------------------------------------------------------------

def assemble_beam_solid(
    input_file,
    solid_mesh,
    beam_mesh,
    *,
    lumen_surface_index: int = 0,
    bc_type=None,
    contact_discretization: str = "mortar",
    mortar_shape: str = "line2",
    n_gauss_points: int = 6,
    output_path=None,
):
    """Combine the artery solid with the warped stent beams and tie them.

    Adds ``beam_mesh`` to ``solid_mesh`` ("add the beam elements to the solid"),
    creates a beam-to-solid coupling between the beam curves and the lumen surface
    set, adds the interaction header options, and — if ``output_path`` is given —
    dumps the assembled ``.4C.yaml``.

    Parameters
    ----------
    input_file            : BeamMe InputFile from ``import_artery_solid``
    solid_mesh, beam_mesh : BeamMe Meshes (solid = artery, beam = warped stent)
    lumen_surface_index   : which imported surface set is the lumen (default 0)
    bc_type               : coupling type; default
                            ``bme.bc.beam_to_solid_surface_meshtying``. Switch to
                            ``bme.bc.beam_to_solid_surface_contact`` for a real
                            deployment (allows separation).
    contact_discretization, mortar_shape, n_gauss_points : mortar options
    output_path           : if set, dump the assembled 4C input file here

    Returns
    -------
    (input_file, mesh) : ready to dump / extend with materials, BCs, solver
    """

    if bc_type is None:
        bc_type = bme.bc.beam_to_solid_surface_meshtying

    surf_sets = solid_mesh.geometry_sets.get(bme.geo.surface, [])
    if not surf_sets:
        raise ValueError("Imported solid has no surface geometry set to couple to. "
                         "Check the lumen DSURF set in the artery mesh.")
    lumen_set = surf_sets[lumen_surface_index]

    # Capture the beam curves before merging, then add the beams to the solid.
    beam_elems = list(beam_mesh.elements)
    solid_mesh.add(beam_mesh)
    beam_set = GeometrySet(beam_elems)

    coupling_id = add_beam_interaction_condition(
        solid_mesh, beam_set, lumen_set, bc_type=bc_type)

    # Header options for the interaction (meshtying variants). Contact uses a
    # different setter (set_beam_contact_section) — wire that when we switch.
    if bc_type in (bme.bc.beam_to_solid_surface_meshtying,
                   bme.bc.beam_to_solid_volume_meshtying):
        set_beam_to_solid_meshtying(
            input_file, bc_type,
            contact_discretization=contact_discretization,
            mortar_shape=mortar_shape,
            n_gauss_points=n_gauss_points,
        )

    input_file.add(solid_mesh)
    print(f"[assemble] coupling id {coupling_id}: {len(beam_elems):,} beam elements "
          f"<-> lumen surface ({bc_type})")
    print(f"[assemble] combined mesh: {len(solid_mesh.nodes):,} nodes, "
          f"{len(solid_mesh.elements):,} elements")

    if output_path is not None:
        input_file.dump(str(output_path), validate=False,
                        add_footer_application_script=False)
        print(f"[saved] {output_path}")

    return input_file, solid_mesh

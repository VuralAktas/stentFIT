"""
The balloon's geometry and material, as files 4C can read.

A balloon is a far simpler object than an artery, since it is a straight tube with no wall
roughness and no centreline warping. A structured grid in cylindrical coordinates covers it in
little code, and it lets every surface be labelled exactly instead of inferred from coordinates
afterwards.

``stentfit.core.artery_geom.mesh_artery_gmsh`` is not reused because it classifies only the inner
surface and the two end caps, and faces at the outer radius fall through its check. For the balloon
it is the outer surface that matters, since that is what the stent touches.

The output is the same 4C ``.yaml`` solid format ``mesh_artery_gmsh`` writes, so
``stentfit.core.artery_geom.import_artery_solid`` reads it back unchanged.

Surface sets written:

============  ==========================================================
``DSURFACE``  meaning
============  ==========================================================
1             inner surface (``r = r_inner``) -- the pressurised face
2             **outer surface** (``r = r_outer``) -- the contact surface
3             end cap at ``z_min``
4             end cap at ``z_max``
============  ==========================================================

Everything here is geometry and file format. What the balloon *is* -- its size, its drive, its
loading -- lives on :class:`stentfit.balloon.Balloon`.
"""

from pathlib import Path

import numpy as np


def build_balloon(r_inner, r_outer, z_min, z_max, n_circ, n_axial=None, n_radial=1):
    """
    Build a structured hexahedral tube.

    Nodes are laid out on a regular grid in (radius, angle, z). The grid is periodic in angle:
    the last angular station wraps to the first, so no duplicate seam nodes are created and the
    tube closes exactly.

    :param r_inner: Inner radius, in mm.
    :param r_outer: Outer radius, in mm.
    :param z_min: Lower end, in mm.
    :param z_max: Upper end, in mm.
    :param n_circ: Number of elements around the circumference.
    :param n_axial: Number of elements along the axis. ``None`` picks a count giving roughly
        cubic elements, based on the circumferential element size.
    :param n_radial: Number of elements across the wall.
    :returns: ``(coords, conn, surfaces)`` -- ``(n, 3)`` node coordinates, ``(m, 8)`` 1-based
        HEX8 connectivity, and a dict of 1-based node id lists per surface set.
    """
    if not r_outer > r_inner > 0:
        raise ValueError(f"need r_outer > r_inner > 0, got {r_inner=} {r_outer=}")

    length = z_max - z_min
    if n_axial is None:
        # Aim for roughly cubic elements: match the axial size to the circumferential one.
        circ_size = 2.0 * np.pi * 0.5 * (r_inner + r_outer) / n_circ
        n_axial = max(1, int(round(length / circ_size)))

    radii = np.linspace(r_inner, r_outer, n_radial + 1)
    angles = np.linspace(0.0, 2.0 * np.pi, n_circ, endpoint=False)   # periodic, no seam
    zs = np.linspace(z_min, z_max, n_axial + 1)

    # Node id at grid position (i_radial, j_angular, k_axial), 1-based for 4C. The angular
    # index wraps, which is what closes the tube.
    def nid(i, j, k):
        return 1 + (i * n_circ + (j % n_circ)) * len(zs) + k

    coords = np.empty((len(radii) * n_circ * len(zs), 3), float)
    for i, r in enumerate(radii):
        for j, a in enumerate(angles):
            for k, z in enumerate(zs):
                coords[nid(i, j, k) - 1] = (r * np.cos(a), r * np.sin(a), z)

    # HEX8 node ordering: the first four nodes are the face at z_k, wound counter-clockwise
    # seen from +z, then the matching four at z_(k+1). The local axes here are (radial,
    # angular, axial), which is right-handed since e_r x e_theta = e_z, so the winding has to
    # run radial-first: (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1). Going angular-first instead
    # reverses it and every element comes out with negative volume.
    conn = []
    for i in range(n_radial):
        for j in range(n_circ):
            for k in range(n_axial):
                conn.append([nid(i, j, k), nid(i + 1, j, k),
                             nid(i + 1, j + 1, k), nid(i, j + 1, k),
                             nid(i, j, k + 1), nid(i + 1, j, k + 1),
                             nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)])
    conn = np.array(conn, dtype=int)

    # Surface sets read straight off the grid indices, so each is exact by construction rather
    # than recovered from coordinates.
    surfaces = {
        1: sorted(nid(0, j, k) for j in range(n_circ) for k in range(len(zs))),
        2: sorted(nid(n_radial, j, k) for j in range(n_circ) for k in range(len(zs))),
        3: sorted(nid(i, j, 0) for i in range(len(radii)) for j in range(n_circ)),
        4: sorted(nid(i, j, len(zs) - 1) for i in range(len(radii)) for j in range(n_circ)),
    }
    return coords, conn, surfaces


# Balloon material parameters, from Datz et al. Section 2.3.
#
# The balloon is not made of a real material. It is a plain tube standing in for a folded
# catheter balloon, and the paper makes it behave like one by giving it two artificial fibre
# families with wildly different stiffnesses:
#
#   longitudinal    k1 = 1000      very stiff  -> the tube refuses to stretch lengthwise
#   circumferential k1 = 1.5e-7    very soft   -> it grows in diameter almost freely
#
# Ten orders of magnitude between them. That contrast is the entire mechanism, which is why the
# fibre directions have to be right; swapped, the tube would stretch instead of inflate.
#
# The base is Neo-Hooke with E = 17 MPa and nu = 0. Poisson's ratio of exactly zero is
# deliberate: it stops radial expansion from pulling the tube shorter.
NEOHOOKE_YOUNGS = 17.0
NEOHOOKE_POISSON = 0.0
FIBRE_LONGITUDINAL = {"k1": 1000.0, "k2": 0.01}
FIBRE_CIRCUMFERENTIAL = {"k1": 1.5e-7, "k2": 0.35}


def orthotropic_material(mat_id=1, neohooke_youngs=NEOHOOKE_YOUNGS,
                         neohooke_poisson=NEOHOOKE_POISSON,
                         fibre_longitudinal=None, fibre_circumferential=None):
    """
    The paper's balloon material: Neo-Hooke plus two exponential fibre families.

    ``MAT_ElastHyper`` sums the summands listed in ``MATIDS``. The structural tensor is not one of
    them, and is referenced by ``STR_TENS_ID`` instead, which is how 4C's own test inputs write it.

    ``INIT: 1`` is ``INIT_MODE_ELEMENT_FIBERS``. In that mode 4C looks for a cylinder coordinate
    system on the element first and, not finding one, falls back to the explicit ``FIBER1`` and
    ``FIBER2`` vectors written on every element line. ``FIBER_ID`` selects which family reads which
    vector, and ``GAMMA`` is only used by the coordinate-system branch, so it is set to zero.

    Each family contributes ``k1/(2*k2) * (exp(k2*(I4-1)^2) - 1)``, the paper's Eq. (B.4).
    ``K1COMP = 0`` switches the fibres off in compression, which is what fibres do.

    :param mat_id: Id for the combined material, the one the elements point at.
    :param neohooke_youngs: Base Young's modulus, in MPa.
    :param neohooke_poisson: Base Poisson's ratio.
    :param fibre_longitudinal: ``{"k1": ..., "k2": ...}`` for the along-the-tube family.
        ``None`` uses :data:`FIBRE_LONGITUDINAL`.
    :param fibre_circumferential: The same for the around-the-tube family. ``None`` uses
        :data:`FIBRE_CIRCUMFERENTIAL`.
    :returns: ``(lines, mat_id)`` -- YAML lines for the MATERIALS section, and the id to use.
    """
    fibre_longitudinal = fibre_longitudinal or FIBRE_LONGITUDINAL
    fibre_circumferential = fibre_circumferential or FIBRE_CIRCUMFERENTIAL

    neo, lon, cir, tens = mat_id + 1, mat_id + 2, mat_id + 3, mat_id + 4
    lines = [f"  - MAT: {mat_id}",
             "    MAT_ElastHyper:",
             "      NUMMAT: 3",
             f"      MATIDS: [{neo}, {lon}, {cir}]",
             "      DENS: 0.0",
             f"  - MAT: {neo}",
             "    ELAST_CoupNeoHooke:",
             f"      YOUNG: {neohooke_youngs}",
             f"      NUE: {neohooke_poisson}"]
    for fid, (mid, fib) in enumerate([(lon, fibre_longitudinal),
                                      (cir, fibre_circumferential)], start=1):
        lines += [f"  - MAT: {mid}",
                  "    ELAST_CoupAnisoExpo:",
                  f"      K1: {fib['k1']}",
                  f"      K2: {fib['k2']}",
                  "      GAMMA: 0.0",
                  "      K1COMP: 0.0",
                  "      K2COMP: 1.0",
                  f"      STR_TENS_ID: {tens}",
                  "      INIT: 1",
                  f"      FIBER_ID: {fid}"]
    lines += [f"  - MAT: {tens}",
              "    ELAST_StructuralTensor:",
              '      STRATEGY: "Standard"']
    return lines, mat_id


def isotropic_material(mat_id=1, neohooke_youngs=NEOHOOKE_YOUNGS,
                       neohooke_poisson=NEOHOOKE_POISSON):
    """
    The same Neo-Hooke base with the fibres left out, for comparison.

    Running this against :func:`orthotropic_material` isolates what the artificial anisotropy
    actually does. It should still inflate radially, because ``nu = 0`` already decouples
    diameter from length, but it cannot reproduce dogboning -- the ends inflating before the
    middle -- because nothing distinguishes one part of the tube from another.

    :param mat_id: Id for the combined material.
    :param neohooke_youngs: Base Young's modulus, in MPa.
    :param neohooke_poisson: Base Poisson's ratio.
    :returns: ``(lines, mat_id)``.
    """
    neo = mat_id + 1
    return ([f"  - MAT: {mat_id}",
             "    MAT_ElastHyper:",
             "      NUMMAT: 1",
             f"      MATIDS: [{neo}]",
             "      DENS: 0.0",
             f"  - MAT: {neo}",
             "    ELAST_CoupNeoHooke:",
             f"      YOUNG: {neohooke_youngs}",
             f"      NUE: {neohooke_poisson}"], mat_id)


def fibre_directions(coords, conn):
    """
    Longitudinal and circumferential unit vectors at each element's centre.

    Written per element rather than left to 4C's cylinder coordinate system. The mesher builds
    the tube analytically, so it knows both directions exactly at every element -- there is
    nothing to infer, and nothing to get wrong about which axis 4C thinks the tube runs along.

    The tube's axis is z by construction, so longitudinal is ``(0, 0, 1)`` everywhere and
    circumferential is ``(-sin, cos, 0)`` at the element centre's own angle.

    :param coords: ``(n, 3)`` node coordinates.
    :param conn: ``(m, 8)`` 1-based HEX8 connectivity.
    :returns: ``(m, 2, 3)`` array -- per element, the longitudinal then circumferential vector.
    """
    centres = coords[conn - 1].mean(axis=1)            # (m, 3)
    angle = np.arctan2(centres[:, 1], centres[:, 0])
    out = np.zeros((len(conn), 2, 3))
    out[:, 0, 2] = 1.0                                 # longitudinal: along the axis
    out[:, 1, 0] = -np.sin(angle)                      # circumferential: around it
    out[:, 1, 1] = np.cos(angle)
    return out


def write_4c_solid(out_path, coords, conn, surfaces, *, material_lines, material_id=1,
                   fibers=None):
    """
    Write the mesh as a 4C solid ``.yaml``, in the format ``import_artery_solid`` expects.

    :param out_path: File to write.
    :param coords: ``(n, 3)`` node coordinates.
    :param conn: ``(m, 8)`` 1-based HEX8 connectivity.
    :param surfaces: Dict of ``DSURFACE`` id to 1-based node ids.
    :param material_lines: YAML lines for the MATERIALS section, from one of the material
        builders. Required: the wall's stiffness is what the pressure inflates against, so there
        is no meaningful placeholder to fall back on.
    :param material_id: Material id referenced by the elements.
    :param fibers: ``(m, 2, 3)`` fibre vectors per element, from :func:`fibre_directions`.
        Written as ``FIBER1``/``FIBER2`` on each element line. ``None`` omits them.
    :returns: The path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["MATERIALS:"] + list(material_lines) + ["STRUCTURE ELEMENTS:"]
    for e, nodes in enumerate(conn, 1):
        node_str = " ".join(str(n) for n in nodes)
        fib = ""
        if fibers is not None:
            for i, v in enumerate(fibers[e - 1], start=1):
                fib += f" FIBER{i} {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}"
        lines.append(f'  - "{e} SOLID HEX8 {node_str} MAT {material_id} '
                     f'KINEM nonlinear{fib}"')

    lines.append("NODE COORDS:")
    for i, (x, y, z) in enumerate(coords, 1):
        lines.append(f'  - "NODE {i} COORD {x:.10g} {y:.10g} {z:.10g}"')

    lines.append("DSURF-NODE TOPOLOGY:")
    for gid in sorted(surfaces):
        for n in surfaces[gid]:
            lines.append(f'  - "NODE {n} DSURFACE {gid}"')

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_vtu(out_path, coords, conn, fibers=None):
    """
    Write the mesh as a legacy ASCII VTK file, for a look in ParaView before solving anything.

    Checking the balloon visually is cheap and catches a wrong radius or a tube that already
    overlaps the stent, neither of which the solver would report in a recognisable way.

    When fibres are given they are written as cell vectors, so ParaView's Glyph filter draws
    them. This is the only practical check on the fibre convention: swapping the two families
    produces a balloon that stretches instead of inflating, and nothing in the solve would say
    so. The longitudinal arrows must run along the tube and the circumferential ones around it.

    :param out_path: File to write. A ``.vtk`` suffix is used.
    :param coords: ``(n, 3)`` node coordinates.
    :param conn: ``(m, 8)`` 1-based HEX8 connectivity.
    :param fibers: ``(m, 2, 3)`` fibre vectors per element, or ``None``.
    :returns: The path written.
    """
    out_path = Path(out_path).with_suffix(".vtk")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# vtk DataFile Version 3.0", "balloon", "ASCII", "DATASET UNSTRUCTURED_GRID",
             f"POINTS {len(coords)} double"]
    lines += [f"{x:.10g} {y:.10g} {z:.10g}" for x, y, z in coords]
    lines.append(f"CELLS {len(conn)} {len(conn) * 9}")
    lines += ["8 " + " ".join(str(n - 1) for n in cell) for cell in conn]   # VTK is 0-based
    lines.append(f"CELL_TYPES {len(conn)}")
    lines += ["12"] * len(conn)                                            # 12 = VTK_HEXAHEDRON

    if fibers is not None:
        lines.append(f"CELL_DATA {len(conn)}")
        for i, name in enumerate(("longitudinal", "circumferential")):
            lines.append(f"VECTORS {name} double")
            lines += [f"{v[0]:.10g} {v[1]:.10g} {v[2]:.10g}" for v in fibers[:, i, :]]

    out_path.write_text("\n".join(lines) + "\n")
    return out_path

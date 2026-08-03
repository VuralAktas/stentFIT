import numpy as np
import trimesh
import gmsh
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from beamme.four_c.model_importer import (import_cubitpy_model,
                                          import_four_c_model)
from beamme.core.conf import bme
from beamme.core.mesh import Mesh
from beamme.core.geometry_set import GeometrySet
from beamme.four_c.input_file import InputFile
from beamme.four_c.beam_interaction_conditions import add_beam_interaction_condition
from beamme.four_c.header_functions import set_beam_to_solid_meshtying


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _make_circle_points(radius: float, n_circumference: int) -> np.ndarray:
    """
    Sample evenly-spaced points around a circle in the XY plane, at z=0.

    :param radius: Circle radius.
    :param n_circumference: Number of points around the circle.
    :returns: ``(n_circumference, 3)`` array of points.
    """
    theta = np.linspace(0, 2 * np.pi, n_circumference, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    return points * radius


def _build_tube_mesh(
        centerline: np.ndarray,
        radii: np.ndarray,
        n_circumference: int = 32,
        cap_ends: bool = True,
        noise_amplitude: float = 0.0,
        noise_seed: int | None = None,
    ) -> trimesh.Trimesh:
    """
    Sweep a circular cross-section along a centreline into a tube mesh.

    A local frame (tangent, normal, binormal) is propagated along the
    centreline by parallel-transporting the normal at each step (a
    rotation-minimising-frame approximation), so the cross-section doesn't
    twist. If ``noise_amplitude`` is positive, a "biological" wall-roughness
    field is added as a fractional radius deviation: a smooth low-frequency
    component along the length (75% weight, mimicking natural
    stenosis/dilatation) plus a small-scale per-vertex component (25%
    weight, mimicking endothelial texture), clamped so the radius never
    drops below 10% of its nominal value. Both ends are capped with a
    triangle fan if ``cap_ends``.

    :param centerline: ``(n, 3)`` points along the tube's centreline.
    :param radii: Per-section radius, one value per centreline point.
    :param n_circumference: Number of vertices around each cross-section.
    :param cap_ends: Close both ends of the tube with a triangle fan.
    :param noise_amplitude: Fractional wall-roughness noise, as a fraction of the radius.
    :param noise_seed: Seed for the wall noise. ``None`` draws a fresh pattern each call.
    :returns: The tube surface mesh, with normals fixed to point outward.
    """
    n_sections = len(centerline)
    nc = n_circumference

    # --- Compute local coordinate frames along the centreline (RMF-like) ---
    tangents = np.zeros_like(centerline)
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    tangents[1:-1] = centerline[2:] - centerline[:-2]

    # Normalise
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    tangents = tangents / norms

    # Build an initial normal vector perpendicular to the first tangent
    t0 = tangents[0]
    # Pick the axis least aligned with t0 to form an initial normal
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(t0, seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    normal = np.cross(t0, seed)
    normal /= np.linalg.norm(normal)

    # Propagate the normal along the curve (rotation-minimising approximation)
    normals = np.zeros_like(centerline)
    normals[0] = normal
    for i in range(1, n_sections):
        # Project previous normal onto plane perpendicular to current tangent
        n_prev = normals[i - 1]
        t_cur = tangents[i]
        n_cur = n_prev - np.dot(n_prev, t_cur) * t_cur
        length = np.linalg.norm(n_cur)
        if length < 1e-12:
            n_cur = normals[i - 1]
        else:
            n_cur /= length
        normals[i] = n_cur

    binormals = np.cross(tangents, normals)

    # --- Build biological noise field (n_sections, nc) if requested ----------
    # Two superimposed components:
    #   1. Axial (75%): smooth low-frequency radius variation along the length,
    #      mimicking natural stenosis/dilatation.  Correlation length ≈ 12% of arc.
    #   2. Roughness (25%): small-scale per-vertex surface irregularity,
    #      mimicking endothelial texture.  Correlation length ≈ 3% of arc axially.
    # The combined field is expressed as a *fractional* deviation of the local
    # radius, clamped so the radius never drops below 10% of its nominal value.
    if noise_amplitude > 0.0:
        rng = np.random.default_rng(noise_seed)

        raw_axial = rng.standard_normal(n_sections)
        sigma_a = max(2, int(0.12 * n_sections))
        axial = gaussian_filter1d(raw_axial, sigma_a)
        axial /= axial.std() + 1e-12

        raw_rough = rng.standard_normal((n_sections, nc))
        sigma_r = (max(1, int(0.03 * n_sections)), 1)
        rough = gaussian_filter(raw_rough, sigma_r)
        rough /= rough.std() + 1e-12

        noise_field = noise_amplitude * (0.75 * axial[:, np.newaxis] + 0.25 * rough)
    else:
        noise_field = None

    # --- Vectorised vertex placement -----------------------------------------
    circle_2d = _make_circle_points(1.0, nc)  # (nc, 3) unit circle
    cos_t = circle_2d[:, 0]                   # (nc,)
    sin_t = circle_2d[:, 1]                   # (nc,)

    # ring_dirs[i, j] = cos(θ_j)·N_i + sin(θ_j)·B_i,  shape (n_sections, nc, 3)
    ring_dirs = (
        cos_t[np.newaxis, :, np.newaxis] * normals[:, np.newaxis, :]
        + sin_t[np.newaxis, :, np.newaxis] * binormals[:, np.newaxis, :]
    )

    # Effective per-vertex radius with optional noise, shape (n_sections, nc)
    if noise_field is not None:
        R_eff = np.clip(
            radii[:, np.newaxis] * (1.0 + noise_field),
            radii[:, np.newaxis] * 0.1,  # never shrink below 10% of nominal
            None,
        )
    else:
        R_eff = radii[:, np.newaxis]

    # (n_sections, nc, 3) → (n_sections*nc, 3)
    vertices = (
        centerline[:, np.newaxis, :] + R_eff[:, :, np.newaxis] * ring_dirs
    ).reshape(-1, 3)

    # --- Build quad faces (split into triangles) between consecutive rings ---
    faces = []
    for i in range(n_sections - 1):
        for j in range(nc):
            j_next = (j + 1) % nc
            # Current ring indices
            v00 = i * nc + j
            v01 = i * nc + j_next
            # Next ring indices
            v10 = (i + 1) * nc + j
            v11 = (i + 1) * nc + j_next
            # Two triangles per quad
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])

    # --- Cap the ends ---
    if cap_ends:
        # Start cap (ring 0) — fan from a centre vertex
        centre_start = centerline[0].copy()
        idx_cs = len(vertices)
        vertices = np.vstack([vertices, centre_start.reshape(1, 3)])
        for j in range(nc):
            j_next = (j + 1) % nc
            faces.append([idx_cs, 0 * nc + j_next, 0 * nc + j])  # wound inward

        # End cap (last ring) — fan from a centre vertex
        centre_end = centerline[-1].copy()
        idx_ce = len(vertices)
        vertices = np.vstack([vertices, centre_end.reshape(1, 3)])
        last = (n_sections - 1) * nc
        for j in range(nc):
            j_next = (j + 1) % nc
            faces.append([idx_ce, last + j, last + j_next])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def _add_outer_wall(
        inner_mesh: trimesh.Trimesh,
        centerline: np.ndarray,
        radii: np.ndarray,
        wall_thickness: float,
        n_circumference: int) -> trimesh.Trimesh:
    """
    Turn a lumen-only tube mesh into a hollow-walled tube.

    Builds a second, larger, independently-capped tube at
    ``radii + wall_thickness`` as the outer wall, flips ``inner_mesh``'s
    normals so they face into the wall material instead of outward, and
    concatenates both into one mesh representing the wall's inner and outer
    boundary together.

    :param inner_mesh: The lumen surface, from :func:`_build_tube_mesh`.
    :param centerline: ``(n, 3)`` points along the tube's centreline, matching
        ``inner_mesh``.
    :param radii: Per-section lumen radius, matching ``inner_mesh``.
    :param wall_thickness: Wall thickness added to ``radii`` for the outer surface.
    :param n_circumference: Number of vertices around each cross-section.
    :returns: The combined outer + inner (inverted) surface mesh.
    """
    outer_radii = radii + wall_thickness
    outer = _build_tube_mesh(centerline, outer_radii, n_circumference)
    inner_mesh.invert()
    return trimesh.util.concatenate([outer, inner_mesh])


# ---------------------------------------------------------------------------
# Centreline generators
# ---------------------------------------------------------------------------

def straight_centreline(length: float, n_points: int = 100) -> np.ndarray:
    """
    Build a straight centreline of the given length along the z-axis.

    :param length: Centreline length, in mm.
    :param n_points: Number of points along the centreline.
    :returns: ``(n_points, 3)`` array of points, at ``x = y = 0``.
    """
    z = np.linspace(0, length, n_points)
    return np.column_stack([np.zeros(n_points), np.zeros(n_points), z])


def curved_centreline(
        length: float,
        bend_radius: float,
        bend_angle_deg: float = 45.0,
        n_points: int = 100) -> np.ndarray:
    """
    Build a straight → arc → straight centreline, in the XZ plane.

    The middle section is a circular arc of ``bend_radius`` sweeping
    ``bend_angle_deg``; the two straight segments before and after it are
    equal length, sized so the whole centreline's arc length adds up to
    ``length``. Points are sampled at even arc-length steps along the whole
    path.

    :param length: Total centreline arc length, in mm.
    :param bend_radius: Radius of the circular arc, in mm.
    :param bend_angle_deg: Total bend angle, in degrees.
    :param n_points: Number of points along the centreline.
    :raises ValueError: If the arc alone (``bend_radius * bend_angle``) is
        longer than ``length``, leaving no room for the two straight segments.
    :returns: ``(n_points, 3)`` array of points.
    """
    bend_angle = np.radians(bend_angle_deg)
    arc_length = bend_radius * bend_angle
    straight_each = (length - arc_length) / 2.0

    if straight_each < 0:
        raise ValueError(
            f"Total length ({length}) is shorter than the arc length ({arc_length:.2f}). "
            f"Increase length or reduce bend_radius/bend_angle."
        )

    points = []
    t_vals = np.linspace(0, 1, n_points)
    total = length

    for t in t_vals:
        s = t * total  # arc-length parameter

        if s <= straight_each:
            # First straight segment along +Z
            points.append([0.0, 0.0, s])

        elif s <= straight_each + arc_length:
            # Circular arc in XZ plane, centre at (bend_radius, 0, straight_each)
            s_arc = s - straight_each
            angle = s_arc / bend_radius
            x = bend_radius * (1 - np.cos(angle))
            z = straight_each + bend_radius * np.sin(angle)
            points.append([x, 0.0, z])

        else:
            # Second straight segment, continuing in the exit direction
            s_extra = s - straight_each - arc_length
            exit_angle = bend_angle
            dx = np.sin(exit_angle)
            dz = np.cos(exit_angle)
            x_arc_end = bend_radius * (1 - np.cos(exit_angle))
            z_arc_end = straight_each + bend_radius * np.sin(exit_angle)
            points.append([
                x_arc_end + s_extra * dx,
                0.0,
                z_arc_end + s_extra * dz,
            ])

    return np.array(points)


def s_bend_centreline(
        length: float,
        bend_radius: float,
        bend_angle_deg: float = 30.0,
        n_points: int = 100) -> np.ndarray:
    """
    Build an S-shaped centreline: straight → arc right → straight → arc left → straight.

    Walked piecewise in 3D, tracking position and heading segment by
    segment, so each arc continues smoothly from where the previous segment
    left off. The two arcs both use ``bend_radius``/``bend_angle_deg`` but
    curve in opposite directions; the three straight segments share what's
    left of ``length`` equally. Each of the 5 segments gets roughly
    ``n_points / 5`` points (floored at 10), so the actual point count may
    come out slightly different from ``n_points``.

    :param length: Total centreline arc length, in mm.
    :param bend_radius: Radius of each circular arc, in mm.
    :param bend_angle_deg: Bend angle of each arc, in degrees.
    :param n_points: Target number of points along the centreline.
    :raises ValueError: If the two arcs alone are longer than ``length``,
        leaving no room for the three straight segments.
    :returns: Array of points along the centreline.
    """
    # Build as: straight → bend_right → straight → bend_left → straight
    bend_angle = np.radians(bend_angle_deg)
    arc_length = bend_radius * bend_angle
    n_seg = 5  # straight, arc1, straight_mid, arc2, straight
    seg_lengths = [
        (length - 2 * arc_length) / 3.0,  # entry straight
        arc_length,                         # first bend
        (length - 2 * arc_length) / 3.0,  # middle straight
        arc_length,                         # second bend (opposite)
        (length - 2 * arc_length) / 3.0,  # exit straight
    ]

    if any(s < 0 for s in seg_lengths):
        raise ValueError("Length too short for the requested S-bend parameters.")

    # Build piecewise: track position and direction
    points = []
    pos = np.array([0.0, 0.0, 0.0])
    direction_angle = 0.0  # angle from +Z in the XZ plane

    def direction_vec(a):
        return np.array([np.sin(a), 0.0, np.cos(a)])

    n_per_seg = max(n_points // n_seg, 10)

    # Segment 1: entry straight
    for i in range(n_per_seg):
        frac = i / n_per_seg
        p = pos + frac * seg_lengths[0] * direction_vec(direction_angle)
        points.append(p.copy())
    pos = pos + seg_lengths[0] * direction_vec(direction_angle)

    # Segment 2: first bend (positive angle → bends to the right in XZ)
    centre1 = pos + bend_radius * np.array([np.cos(direction_angle), 0.0, -np.sin(direction_angle)])
    start_a1 = direction_angle
    for i in range(n_per_seg):
        frac = i / n_per_seg
        a = frac * bend_angle
        offset = np.array([-np.cos(start_a1 + a), 0.0, np.sin(start_a1 + a)])
        points.append((centre1 + bend_radius * offset).copy())
    direction_angle += bend_angle
    pos = centre1 + bend_radius * np.array([-np.cos(direction_angle), 0.0, np.sin(direction_angle)])

    # Segment 3: middle straight
    for i in range(n_per_seg):
        frac = i / n_per_seg
        p = pos + frac * seg_lengths[2] * direction_vec(direction_angle)
        points.append(p.copy())
    pos = pos + seg_lengths[2] * direction_vec(direction_angle)

    # Segment 4: second bend (negative angle → bends to the left)
    centre2 = pos - bend_radius * np.array([np.cos(direction_angle), 0.0, -np.sin(direction_angle)])
    start_a2 = direction_angle
    for i in range(n_per_seg):
        frac = i / n_per_seg
        a = frac * bend_angle
        offset = np.array([np.cos(start_a2 - a), 0.0, -np.sin(start_a2 - a)])
        points.append((centre2 + bend_radius * offset).copy())
    direction_angle -= bend_angle
    pos = centre2 + bend_radius * np.array([np.cos(direction_angle), 0.0, -np.sin(direction_angle)])

    # Segment 5: exit straight
    for i in range(n_per_seg + 1):  # +1 to include endpoint
        frac = i / n_per_seg
        p = pos + frac * seg_lengths[4] * direction_vec(direction_angle)
        points.append(p.copy())

    return np.array(points)


# ---------------------------------------------------------------------------
# High-level generators
# ---------------------------------------------------------------------------

def generate_straight_artery(
        radius: float = 1.5,
        length: float = 25.0,
        wall_thickness: float = 0.0,
        n_circumference: int = 32,
        n_axial: int = 100,
        noise_amplitude: float = 0.0,
        noise_seed: int | None = None,
    ) -> trimesh.Trimesh:
    """
    Build a straight tube mesh of constant radius.

    The simplest of the three artery shapes: a :func:`straight_centreline`
    with a uniform radius, tubed by :func:`_build_tube_mesh`. If
    ``wall_thickness`` is positive, a second, larger tube is added as the
    outer wall (:func:`_add_outer_wall`); otherwise the mesh is the lumen
    surface only.

    :param radius: Lumen radius, in mm.
    :param length: Artery length, in mm.
    :param wall_thickness: Wall thickness, in mm. ``0`` builds the lumen
        surface only, with no separate wall.
    :param n_circumference: Number of vertices around each cross-section.
    :param n_axial: Number of cross-sections along the length.
    :param noise_amplitude: Fractional wall-roughness noise, as a fraction of the radius.
    :param noise_seed: Seed for the wall noise. ``None`` draws a fresh pattern each call.
    :returns: The artery wall mesh.
    """
    cl = straight_centreline(length, n_axial)
    radii = np.full(n_axial, radius)
    mesh = _build_tube_mesh(cl, radii, n_circumference,
                            noise_amplitude=noise_amplitude, noise_seed=noise_seed)

    if wall_thickness > 0:
        mesh = _add_outer_wall(mesh, cl, radii, wall_thickness, n_circumference)

    return mesh


def generate_curved_artery(
    radius: float = 1.5,
    length: float = 30.0,
    bend_radius: float = 20.0,
    bend_angle_deg: float = 45.0,
    wall_thickness: float = 0.0,
    n_circumference: int = 32,
    n_axial: int = 100,
    noise_amplitude: float = 0.0,
    noise_seed: int | None = None,
) -> trimesh.Trimesh:
    """
    Build a tube mesh of constant radius along a single-bend centreline.

    Same construction as :func:`generate_straight_artery`, but tubed along a
    :func:`curved_centreline` (straight → arc → straight) instead of a
    straight line.

    :param radius: Lumen radius, in mm.
    :param length: Total artery length along the centreline, in mm.
    :param bend_radius: Radius of the circular arc, in mm.
    :param bend_angle_deg: Total bend angle, in degrees.
    :param wall_thickness: Wall thickness, in mm. ``0`` builds the lumen
        surface only, with no separate wall.
    :param n_circumference: Number of vertices around each cross-section.
    :param n_axial: Number of cross-sections along the length.
    :param noise_amplitude: Fractional wall-roughness noise, as a fraction of the radius.
    :param noise_seed: Seed for the wall noise. ``None`` draws a fresh pattern each call.
    :returns: The artery wall mesh.
    """
    cl = curved_centreline(length, bend_radius, bend_angle_deg, n_axial)
    radii = np.full(n_axial, radius)
    mesh = _build_tube_mesh(cl, radii, n_circumference,
                            noise_amplitude=noise_amplitude, noise_seed=noise_seed)
    if wall_thickness > 0:
        mesh = _add_outer_wall(mesh, cl, radii, wall_thickness, n_circumference)
    return mesh


def generate_s_bend_artery(
    radius: float = 1.5,
    length: float = 40.0,
    bend_radius: float = 25.0,
    bend_angle_deg: float = 25.0,
    wall_thickness: float = 0.0,
    n_circumference: int = 32,
    n_axial: int = 150,
    noise_amplitude: float = 0.0,
    noise_seed: int | None = None,
) -> trimesh.Trimesh:
    """
    Build a tube mesh of constant radius along an S-shaped centreline.

    Same construction as :func:`generate_straight_artery`, but tubed along a
    :func:`s_bend_centreline` (two opposite bends). ``n_axial`` is only a
    target: the S-bend centreline's actual point count can come out
    slightly different, so the radius array is sized to match the
    centreline it actually returns rather than ``n_axial`` directly.

    :param radius: Lumen radius, in mm.
    :param length: Total artery length along the centreline, in mm.
    :param bend_radius: Radius of each circular arc, in mm.
    :param bend_angle_deg: Bend angle of each arc, in degrees.
    :param wall_thickness: Wall thickness, in mm. ``0`` builds the lumen
        surface only, with no separate wall.
    :param n_circumference: Number of vertices around each cross-section.
    :param n_axial: Target number of cross-sections along the length.
    :param noise_amplitude: Fractional wall-roughness noise, as a fraction of the radius.
    :param noise_seed: Seed for the wall noise. ``None`` draws a fresh pattern each call.
    :returns: The artery wall mesh.
    """
    cl = s_bend_centreline(length, bend_radius, bend_angle_deg, n_axial)
    radii = np.full(len(cl), radius)
    mesh = _build_tube_mesh(cl, radii, n_circumference,
                            noise_amplitude=noise_amplitude, noise_seed=noise_seed)
    if wall_thickness > 0:
        mesh = _add_outer_wall(mesh, cl, radii, wall_thickness, n_circumference)
    return mesh


# ---------------------------------------------------------------------------
# 1. Artery solid meshing (GMSH)
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


def _apply_radial_noise(coords: np.ndarray,
                        length: float,
                        amplitude: float,
                        seed: int | None) -> np.ndarray:
    """
    Perturb a straight tube mesh's radius with a smooth, organic-looking pattern.

    Sums 4 sinusoidal modes, each oscillating along both the axial (z) and
    circumferential (phi) directions with a random amplitude sign and phase,
    to build a wall-roughness field that reads as smooth and undulating
    rather than uniform per-vertex noise. The field is normalised to peak at
    1 before scaling by ``amplitude``, and tapered to zero within the last
    10% of the length at each end, so the flat inlet/outlet caps stay flat.
    Only the radius changes; ``z`` is left untouched.

    :param coords: ``(n, 3)`` node coordinates of the straight tube mesh.
    :param length: Tube length along z, used to scale the noise wavelengths and end taper.
    :param amplitude: Fractional radial noise, as a fraction of the local radius.
    :param seed: Seed for the noise pattern. ``None`` draws a fresh pattern each call.
    :returns: The perturbed coordinates.
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


def _compute_rmf(cl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate a rotation-minimising frame (tangent, normal, binormal) along a centreline.

    The tangent is the local curve direction; the normal starts
    perpendicular to the first tangent and is then parallel-transported
    along the curve (projected onto each new tangent's perpendicular plane
    and renormalised), so the frame doesn't twist. The binormal completes
    the frame. Same construction as
    :func:`_build_tube_mesh`'s frame.

    :param cl: ``(n, 3)`` centreline points.
    :returns: ``(T, N, B)`` — the per-point tangent, normal, and binormal,
        each ``(n, 3)``.
    """
    n   = len(cl)
    T   = np.zeros_like(cl, dtype=float)
    T[0]    = cl[1] - cl[0]
    T[-1]   = cl[-1] - cl[-2]
    T[1:-1] = cl[2:] - cl[:-2]
    nrm = np.linalg.norm(T, axis=1, keepdims=True)
    T  /= np.where(nrm > 1e-12, nrm, 1.0)

    seed = np.array([1.0, 0.0, 0.0])
    if abs(T[0] @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    N0 = np.cross(T[0], seed);  N0 /= np.linalg.norm(N0)

    N = np.zeros_like(cl, dtype=float);  N[0] = N0
    for i in range(1, n):
        Ni = N[i-1] - (N[i-1] @ T[i]) * T[i]
        l  = np.linalg.norm(Ni)
        N[i] = Ni / l if l > 1e-12 else N[i-1]

    return T, N, np.cross(T, N)

def _warp_coords_to_centreline(coords: np.ndarray, centreline: np.ndarray) -> np.ndarray:
    """
    Bend a straight tube mesh's node coordinates onto an artery centreline.

    Each node's ``z`` is treated as its arc-length position along the
    straight tube (clamped to the centreline's own arc range), used to
    interpolate that point's position and rotation-minimising frame
    (:func:`_compute_rmf`) on ``centreline``. The node's local ``x``/``y``
    offset is then re-expressed along the interpolated normal/binormal
    instead of the original straight-tube axes — the same warping
    convention used for the stent beam mesh, so the two stay aligned.

    :param coords: ``(n, 3)`` node coordinates of the straight tube mesh.
    :param centreline: Points the straight tube is warped onto.
    :returns: The warped coordinates.
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


def _count_inverted(coords: np.ndarray, conn: np.ndarray, mesh_type: str) -> int:
    """
    Count elements that ended up inverted (negative signed volume).

    For each element, takes the scalar triple product of the first three
    edge vectors from one corner node (nodes 0,1,2,3 for a tet; nodes
    0,1,3,4 for a hex, treating one corner as a representative tetrahedron).
    A negative value means that element's node ordering now describes a
    flipped, degenerate shape — a sign that a large warp or noise amplitude
    has folded the mesh over on itself.

    :param coords: ``(n, 3)`` node coordinates.
    :param conn: ``(n_elem, nnode)`` element connectivity, with 1-based node IDs.
    :param mesh_type: Element type: any ``'TET*'`` label, or ``'HEX8'``.
    :returns: Number of elements with negative signed volume.
    """
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
    centreline: np.ndarray,
    out_path: str | Path,
    mesh_type: str = "TET4",
    element_size: float | None = None,
    noise_amplitude: float = 0.0,
    noise_seed: int | None = None,
    material_id: int = 1,
    youngs_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
    density: float = 1.0,
) -> Path:
    """
    Mesh the artery wall as a hollow 3D solid with GMSH and write it as a 4C ``.yaml``.

    Builds a straight annular tube between ``r_inner`` and ``r_outer`` —
    unstructured tets (``TET4``/``TET10``, via an OCC cylinder-minus-cylinder
    boolean cut) or structured hexahedra (``HEX8``, via transfinite annular
    sectors extruded along the tube) — then classifies its boundary nodes
    into ``DSURFACE`` sets purely from their coordinates: the inner surface
    is the lumen (``DSURFACE 1``), and the two flat ends are the inlet/outlet
    (``DSURFACE 2``/``3``). Optional radial wall-roughness noise is applied,
    then the whole straight tube is warped onto ``centreline`` using the same
    frame convention as the stent warp, so the solid stays aligned with the
    beam mesh. Elements that end up inverted by the warp or noise are
    counted and reported as a warning. Writes the mesh plus a placeholder
    ``MAT_Struct_StVenantKirchhoff`` material as a 4C solid input file.

    :param r_inner: Lumen (inner) radius, in mm.
    :param r_outer: Outer wall radius, in mm.
    :param centreline: Points the straight tube is warped onto.
    :param out_path: File path the 4C ``.yaml`` solid is written to.
    :param mesh_type: Element type: ``'TET4'``, ``'TET10'``, or ``'HEX8'``.
    :param element_size: Target element size, in mm. ``None`` uses roughly
        one element across the wall thickness (``r_outer - r_inner``).
    :param noise_amplitude: Fractional radial wall-roughness noise.
    :param noise_seed: Seed for the wall noise. ``None`` draws a fresh pattern each call.
    :param material_id: Material ID written into the 4C input.
    :param youngs_modulus: Placeholder material Young's modulus, in MPa.
    :param poisson_ratio: Placeholder material Poisson's ratio.
    :param density: Placeholder material density.
    :raises ValueError: If ``r_outer <= r_inner`` or ``r_inner <= 0``, or if
        ``mesh_type`` isn't one of the supported keys.
    :returns: ``out_path``, for chaining into a caller's own return value.
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

def import_artery_solid(source) -> tuple[InputFile, Mesh]:
    """
    Import the artery solid mesh into BeamMe, from a 4C ``.yaml`` or a CubitPy model.

    Duck-types ``source``: a CubitPy model (detected by a ``cmd`` attribute
    or class name) is imported via ``import_cubitpy_model``; anything else
    is treated as a path to a 4C ``.yaml`` solid (e.g. from
    :func:`mesh_artery_gmsh`) and imported via ``import_four_c_model``. The
    CubitPy path is legacy — Cubit Coreform is no longer used in this
    project (no macOS build), so in practice ``source`` is always a
    ``.yaml`` path.

    :param source: A CubitPy model, or a path to a 4C ``.yaml`` solid.
    :returns: ``(input_file, solid)`` — the BeamMe ``InputFile`` and the
        imported solid ``Mesh``.
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
    input_file: InputFile,
    solid_mesh: Mesh,
    beam_mesh: Mesh,
    lumen_surface_index: int = 0,
    bc_type=None,
    contact_discretization: str = "mortar",
    mortar_shape: str = "line2",
    n_gauss_points: int = 6,
    output_path: str | Path | None = None,
) -> tuple[InputFile, Mesh]:
    """
    Tie the stent beam mesh to the artery solid's lumen surface and write the combined 4C input.

    Merges ``beam_mesh`` into ``solid_mesh``, couples the beam elements to
    the solid's lumen surface set (``lumen_surface_index``) via BeamMe's
    mortar beam-to-solid method (``add_beam_interaction_condition``), and —
    for the meshtying variants — writes the matching header options
    (``set_beam_to_solid_meshtying``). Contact uses a different header
    setter, not yet wired up here. If ``output_path`` is given, dumps the
    assembled input file there.

    :param input_file: BeamMe ``InputFile`` the solid (and, once merged, the
        beams) is added to.
    :param solid_mesh: Artery solid mesh, from
        :func:`import_artery_solid`. Modified in place: the beam elements
        are merged into it.
    :param beam_mesh: Warped stent beam mesh, from
        :meth:`~stentfit.simulation.Simulation.align`.
    :param lumen_surface_index: Index into the solid's surface geometry sets
        for the lumen surface the beams couple to.
    :param bc_type: BeamMe beam-to-solid coupling type. ``None`` defaults to
        tied meshtying (``bme.bc.beam_to_solid_surface_meshtying``).
    :param contact_discretization: Mortar discretization passed to
        ``set_beam_to_solid_meshtying``.
    :param mortar_shape: Mortar shape function passed to
        ``set_beam_to_solid_meshtying``.
    :param n_gauss_points: Number of Gauss points passed to
        ``set_beam_to_solid_meshtying``.
    :param output_path: File path to dump the assembled 4C input to.
        ``None`` skips writing.
    :raises ValueError: If ``solid_mesh`` has no surface geometry set to couple to.
    :returns: ``(input_file, solid_mesh)`` — the input file with the solid
        (and merged beams) added, and the combined mesh.
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

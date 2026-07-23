import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d, gaussian_filter

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


def generate_artery_for_stent(
    features: dict,
    artery_type: str = "s_bend",
    noise_amplitude: float = 0.1,
    noise_seed: int | None = 0,
    bend_angle_deg: float = 45.0,
    inner_margin: float = 0.5,
    wall_thickness: float = 0.5,
) -> tuple:
    """
    Build a parametric test artery sized to fit a given stent.

    The artery's lumen radius is the stent's outer radius plus
    ``inner_margin`` clearance; its length is a multiple of the stent length
    (1.5x for ``'straight'``/``'curved'``, 2x for ``'s_bend'``, so the stent
    always sits well inside it), and any bend radius is picked so the
    artery's arc roughly matches its length at the given bend angle. Builds
    both the wall mesh (:func:`generate_straight_artery`,
    :func:`generate_curved_artery`, or :func:`generate_s_bend_artery`) and
    the matching centreline (:func:`straight_centreline`,
    :func:`curved_centreline`, or :func:`s_bend_centreline`) for whichever
    ``artery_type`` is requested, then prints a summary comparing the two.

    :param features: Stent features, each entry wrapped as ``{"value": ...}``
        (only ``r_outer`` and ``length`` are read).
    :param artery_type: Artery shape: ``'straight'``, ``'curved'``, or ``'s_bend'``.
    :param noise_amplitude: Fractional wall-roughness noise, as a fraction of the radius.
    :param noise_seed: Seed for the wall noise. ``None`` draws a fresh pattern each call.
    :param bend_angle_deg: Total bend angle, in degrees, for ``'curved'``/``'s_bend'``.
    :param inner_margin: Extra clearance, in mm, between the stent and the artery lumen.
    :param wall_thickness: Artery wall thickness, in mm. ``0`` builds the lumen surface only.
    :returns: ``(artery_mesh, artery_cl, artery_radius)`` — the wall
        ``trimesh.Trimesh``, the centreline points, and the lumen radius.
    """
    artery_radius = features["r_outer"]["value"] + inner_margin
    stent_length  = features["length"]["value"]

    _len_c = stent_length * 1.5
    _len_s = stent_length * 2.0
    _ba    = np.radians(bend_angle_deg)
    _br_c  = min(20.0, 0.9 * _len_c / _ba)
    _br_s  = min(15.0, 0.9 * _len_s / (2.0 * _ba))

    configs = {
        "straight": dict(
            radius=artery_radius, length=stent_length * 1.5,
            wall_thickness=wall_thickness,
            n_circumference=64, n_axial=150,
        ),
        "curved": dict(
            radius=artery_radius, length=stent_length * 1.5,
            bend_radius=_br_c, bend_angle_deg=bend_angle_deg,
            wall_thickness=wall_thickness,
            n_circumference=64, n_axial=150,
        ),
        "s_bend": dict(
            radius=artery_radius, length=stent_length * 2.0,
            bend_radius=_br_s, bend_angle_deg=bend_angle_deg,
            wall_thickness=wall_thickness,
            n_circumference=64, n_axial=150,
        ),
    }
    mesh_gen = {
        "straight": generate_straight_artery,
        "curved":   generate_curved_artery,
        "s_bend":   generate_s_bend_artery,
    }

    p           = configs[artery_type]
    artery_mesh = mesh_gen[artery_type](**p, noise_amplitude=noise_amplitude, noise_seed=noise_seed)

    if artery_type == "straight":
        artery_cl = straight_centreline(p["length"], p["n_axial"])
    elif artery_type == "curved":
        artery_cl = curved_centreline(p["length"], p["bend_radius"], p["bend_angle_deg"], p["n_axial"])
    elif artery_type == "s_bend":
        artery_cl = s_bend_centreline(p["length"], p["bend_radius"], p["bend_angle_deg"], p["n_axial"])

    total_arc = np.linalg.norm(np.diff(artery_cl, axis=0), axis=1).sum()


    print(f"Artery type      : {artery_type}")
    print(f"Artery radius    : {artery_radius:.3f} mm (lumen)")
    print(f"Wall thickness   : {wall_thickness:.3f} mm"
          + (f"  (outer radius {artery_radius + wall_thickness:.3f} mm)" if wall_thickness > 0
             else "  (lumen surface only)"))
    print(f"Noise amplitude  : {noise_amplitude} ({noise_amplitude*100:.0f}% of radius)  seed={noise_seed}")
    if artery_type == "curved":
        print(f"Bend angle       : {bend_angle_deg:.1f} deg")
        print(f"Bend radius      : {_br_c:.2f} mm  (arc = {_br_c * _ba:.2f} mm)")
    if artery_type == "s_bend":
        print(f"Bend angle       : {bend_angle_deg:.1f} deg")
        print(f"Bend radius      : {_br_s:.2f} mm  (2× arc = {2 * _br_s * _ba:.2f} mm)")
    print(f"Arc length       : {total_arc:.2f} mm  (stent {stent_length:.2f} mm = {stent_length/total_arc*100:.0f}% of artery)")
    print(f"Centreline       : {len(artery_cl)} points  bounds {artery_cl.min(0).round(2)} → {artery_cl.max(0).round(2)}")
    print(f"Mesh             : {len(artery_mesh.vertices):,} vertices  {len(artery_mesh.faces):,} faces  watertight={artery_mesh.is_watertight}")

    return artery_mesh, artery_cl, artery_radius




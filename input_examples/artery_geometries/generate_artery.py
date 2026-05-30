"""
generate_artery.py — Parametric artery STL generator for stentFIT

Generates cylindrical vessel geometries (straight, curved, S-bend, tapered)
as watertight STL meshes for use in virtual stent implantation simulations.

Usage:
    python generate_artery.py                        # generates all default geometries
    python generate_artery.py --type curved           # generates only the curved artery
    python generate_artery.py --type straight --radius 2.0 --length 30.0 --output my_artery.stl

All dimensions are in millimeters.
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import trimesh
except ImportError:
    raise ImportError(
        "trimesh is required: pip install trimesh\n"
        "Add it to your env_requirements.txt"
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _make_circle_points(radius: float, n_circumference: int) -> np.ndarray:
    """Generate points on a unit circle in the XY plane, centred at origin."""
    theta = np.linspace(0, 2 * np.pi, n_circumference, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    return points * radius


def _build_tube_mesh(
    centerline: np.ndarray,
    radii: np.ndarray,
    n_circumference: int = 32,
    cap_ends: bool = True,
) -> trimesh.Trimesh:
    """
    Build a watertight tube mesh around a 3D centerline with varying radii.

    Parameters
    ----------
    centerline : (M, 3) array
        Ordered 3D points defining the vessel centreline.
    radii : (M,) array
        Cross-sectional radius at each centreline point.
    n_circumference : int
        Number of vertices around each cross-section ring.
    cap_ends : bool
        If True, close both ends with triangulated caps.

    Returns
    -------
    trimesh.Trimesh
        Watertight triangulated surface mesh of the tube.
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

    # --- Place circle rings at each centreline point ---
    circle_2d = _make_circle_points(1.0, nc)  # unit circle

    vertices = np.zeros((n_sections * nc, 3))
    for i in range(n_sections):
        R = radii[i]
        centre = centerline[i]
        n_vec = normals[i]
        b_vec = binormals[i]
        for j in range(nc):
            vertices[i * nc + j] = (
                centre + R * (circle_2d[j, 0] * n_vec + circle_2d[j, 1] * b_vec)
            )

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


# ---------------------------------------------------------------------------
# Centreline generators
# ---------------------------------------------------------------------------

def straight_centreline(length: float, n_points: int = 100) -> np.ndarray:
    """Straight centreline along the Z-axis."""
    z = np.linspace(0, length, n_points)
    return np.column_stack([np.zeros(n_points), np.zeros(n_points), z])


def curved_centreline(
    length: float,
    bend_radius: float,
    bend_angle_deg: float = 45.0,
    n_points: int = 100,
) -> np.ndarray:
    """
    Centreline with a single circular bend in the XZ plane.

    The vessel starts straight along Z, bends through `bend_angle_deg`,
    then continues straight in the new direction.

    Parameters
    ----------
    length : float
        Total arc-length of the centreline (mm).
    bend_radius : float
        Radius of curvature at the bend (mm).
    bend_angle_deg : float
        Total bend angle in degrees.
    n_points : int
        Number of discretisation points.
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
    n_points: int = 100,
) -> np.ndarray:
    """
    S-shaped centreline: two opposite bends separated by a straight section.

    The bends are symmetric and lie in the XZ plane.
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


def tapered_radii(
    r_start: float, r_end: float, n_points: int
) -> np.ndarray:
    """Linearly varying radius from r_start to r_end."""
    return np.linspace(r_start, r_end, n_points)


# ---------------------------------------------------------------------------
# High-level generators
# ---------------------------------------------------------------------------

def generate_straight_artery(
    radius: float = 1.5,
    length: float = 25.0,
    wall_thickness: float = 0.0,
    n_circumference: int = 32,
    n_axial: int = 100,
) -> trimesh.Trimesh:
    """
    Generate a straight cylindrical artery.

    Parameters
    ----------
    radius : float
        Inner lumen radius in mm (typical coronary: 1.25–2.0 mm).
    length : float
        Vessel length in mm.
    wall_thickness : float
        If > 0, generates a hollow tube (outer - inner). If 0, single-wall surface.
    n_circumference : int
        Vertices around each ring.
    n_axial : int
        Number of cross-section rings along the length.
    """
    cl = straight_centreline(length, n_axial)
    radii = np.full(n_axial, radius)
    mesh = _build_tube_mesh(cl, radii, n_circumference)

    if wall_thickness > 0:
        outer_radii = np.full(n_axial, radius + wall_thickness)
        outer = _build_tube_mesh(cl, outer_radii, n_circumference)
        # Flip normals of inner surface and combine
        mesh.invert()
        mesh = trimesh.util.concatenate([outer, mesh])

    return mesh


def generate_curved_artery(
    radius: float = 1.5,
    length: float = 30.0,
    bend_radius: float = 20.0,
    bend_angle_deg: float = 45.0,
    n_circumference: int = 32,
    n_axial: int = 100,
) -> trimesh.Trimesh:
    """
    Generate a curved artery with a single bend.

    Parameters
    ----------
    radius : float
        Lumen radius in mm.
    length : float
        Total arc-length in mm.
    bend_radius : float
        Radius of curvature at the bend in mm.
    bend_angle_deg : float
        Bend angle in degrees.
    """
    cl = curved_centreline(length, bend_radius, bend_angle_deg, n_axial)
    radii = np.full(n_axial, radius)
    return _build_tube_mesh(cl, radii, n_circumference)


def generate_s_bend_artery(
    radius: float = 1.5,
    length: float = 40.0,
    bend_radius: float = 25.0,
    bend_angle_deg: float = 25.0,
    n_circumference: int = 32,
    n_axial: int = 150,
) -> trimesh.Trimesh:
    """
    Generate an S-shaped artery with two opposite bends.

    Parameters
    ----------
    radius : float
        Lumen radius in mm.
    length : float
        Total arc-length in mm.
    bend_radius : float
        Radius of curvature for each bend in mm.
    bend_angle_deg : float
        Bend angle for each curve segment in degrees.
    """
    cl = s_bend_centreline(length, bend_radius, bend_angle_deg, n_axial)
    radii = np.full(len(cl), radius)
    return _build_tube_mesh(cl, radii, n_circumference)


def generate_tapered_artery(
    radius_proximal: float = 2.0,
    radius_distal: float = 1.2,
    length: float = 30.0,
    n_circumference: int = 32,
    n_axial: int = 100,
) -> trimesh.Trimesh:
    """
    Generate a straight artery that tapers from proximal to distal end.

    Parameters
    ----------
    radius_proximal : float
        Radius at the start (proximal) end in mm.
    radius_distal : float
        Radius at the far (distal) end in mm.
    length : float
        Vessel length in mm.
    """
    cl = straight_centreline(length, n_axial)
    radii = tapered_radii(radius_proximal, radius_distal, n_axial)
    return _build_tube_mesh(cl, radii, n_circumference)


# ---------------------------------------------------------------------------
# Convenience: generate all defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "straight": {
        "func": generate_straight_artery,
        "kwargs": {"radius": 1.5, "length": 25.0},
    },
    "curved": {
        "func": generate_curved_artery,
        "kwargs": {"radius": 1.5, "length": 30.0, "bend_radius": 20.0, "bend_angle_deg": 45.0},
    },
    "s_bend": {
        "func": generate_s_bend_artery,
        "kwargs": {"radius": 1.5, "length": 40.0, "bend_radius": 25.0, "bend_angle_deg": 25.0},
    },
    "tapered": {
        "func": generate_tapered_artery,
        "kwargs": {"radius_proximal": 2.0, "radius_distal": 1.2, "length": 30.0},
    },
}


def generate_all(output_dir: str | Path = ".") -> dict[str, Path]:
    """Generate all default artery geometries and save as STL files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    for name, spec in DEFAULTS.items():
        mesh = spec["func"](**spec["kwargs"])
        filepath = output_dir / f"artery_{name}.stl"
        mesh.export(filepath)

        # Print mesh summary
        print(f"  {name:12s} → {filepath.name}")
        print(f"               vertices: {len(mesh.vertices):,}")
        print(f"               faces:    {len(mesh.faces):,}")
        print(f"               watertight: {mesh.is_watertight}")
        print(f"               bounds:   {mesh.bounds[0].round(2)} → {mesh.bounds[1].round(2)}")
        print()

        saved[name] = filepath

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate parametric artery STL files for stentFIT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_artery.py                          # all defaults\n"
            "  python generate_artery.py --type curved            # just curved\n"
            "  python generate_artery.py --type straight -r 2.0 -l 35 -o custom.stl\n"
        ),
    )
    parser.add_argument(
        "--type", "-t",
        choices=["all", "straight", "curved", "s_bend", "tapered"],
        default="all",
        help="Artery type to generate (default: all)",
    )
    parser.add_argument("--radius", "-r", type=float, default=None, help="Lumen radius in mm")
    parser.add_argument("--length", "-l", type=float, default=None, help="Vessel length in mm")
    parser.add_argument("--bend-radius", type=float, default=None, help="Bend radius in mm")
    parser.add_argument("--bend-angle", type=float, default=None, help="Bend angle in degrees")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output STL filename")
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        default=".",
        help="Output directory (used when --type=all)",
    )

    args = parser.parse_args()

    if args.type == "all":
        print("Generating all default artery geometries:\n")
        generate_all(args.output_dir)
        print("Done.")
        return

    # Single geometry mode
    spec = DEFAULTS[args.type]
    kwargs = spec["kwargs"].copy()

    # Override with CLI args if provided
    if args.radius is not None:
        if "radius" in kwargs:
            kwargs["radius"] = args.radius
        if "radius_proximal" in kwargs:
            kwargs["radius_proximal"] = args.radius
    if args.length is not None:
        kwargs["length"] = args.length
    if args.bend_radius is not None and "bend_radius" in kwargs:
        kwargs["bend_radius"] = args.bend_radius
    if args.bend_angle is not None and "bend_angle_deg" in kwargs:
        kwargs["bend_angle_deg"] = args.bend_angle

    mesh = spec["func"](**kwargs)
    output_path = args.output or f"artery_{args.type}.stl"
    mesh.export(output_path)
    print(f"Saved: {output_path}")
    print(f"  vertices:   {len(mesh.vertices):,}")
    print(f"  faces:      {len(mesh.faces):,}")
    print(f"  watertight: {mesh.is_watertight}")


if __name__ == "__main__":
    main()

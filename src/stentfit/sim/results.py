"""
Measure what a run produced.

Reads the ``.pvd`` time series a run wrote and reduces the deformed beam geometry at every step to
the numbers that mean something for a stent:

``diameter``
    Twice the 95th percentile radius about the stent's own axis, so one stray node cannot define it.
``length``
    The z extent of the deformed stent.
``radial_strain`` / ``axial_strain``
    ``(D - D0) / D0`` and ``(L - L0) / L0``. Whichever one the case imposes checks the boundary
    condition, and the other is the structural response.
``foreshortening_pct``
    ``-axial_strain`` as a percentage, so positive means the stent shortened. This is the headline
    number, because a stent that shortens while deploying stops covering the lesion.
``peak strain``
    The largest magnitude over all Gauss points, which says whether the run stayed near the elastic
    range of a real metal.
``N/Np``, ``M/Mp``
    How close the struts came to yielding.

It works for any simulation type, since all it needs is a beam ``.pvd``.

Beams and solids store their coordinates differently. The beam output sets
``USE_ABSOLUTE_POSITIONS``, so its points are already deformed and there is nothing to add back on.
The solid output does not, so the balloon's points are its reference coordinates and all of its
motion sits in the ``displacement`` array. In ParaView this means the beams must not be given a
Warp By Vector and the balloon must.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


#: Radius percentile used for the diameter. A single crown flicking outward would set the maximum,
#: so a high percentile is taken instead.
RADIUS_PERCENTILE = 95

#: The central fraction of the length the diameter is measured over. The axial cases clamp both end
#: bands, so those nodes sit at the original radius whatever the stent does, and including them
#: would hide the necking.
MID_BAND_FRAC = 0.5


def _pyvista():
    """
    Import pyvista lazily, with a useful message when it is missing.

    :returns: The pyvista module.
    :raises ImportError: If pyvista is not installed.
    """
    try:
        import pyvista
    except ImportError as error:
        raise ImportError("reading 4C results needs pyvista: pip install pyvista") from error
    return pyvista


def read_pvd(pvd_path):
    """
    Read a 4C ``.pvd`` collection into ``(timestep, vtu_path)`` pairs.

    :param pvd_path: Path to the ``.pvd`` file.
    :returns: List of ``(float, Path)``, ordered by timestep.
    """
    root = ET.parse(pvd_path).getroot()
    base = Path(pvd_path).parent
    out = [(float(ds.attrib["timestep"]), base / ds.attrib["file"])
           for ds in root.iter("DataSet")]
    return sorted(out)


#: Names a 4C load expression may use, mapped to their Python equivalents. 4C spells absolute value
#: ``fabs``, which Python does not know. The namespace is listed explicitly rather than handing the
#: expression the builtins, since it comes out of a file.
_EXPR_NAMES = {"fabs": abs, "abs": abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
               "sin": np.sin, "cos": np.cos, "tan": np.tan, "pi": np.pi}


def pressure_at(t: float, pressure_max: float, expression: str) -> float:
    """
    The balloon pressure at one pseudo-time.

    The run record stores the load profile as the 4C expression that was written into the input,
    such as ``1-fabs(2*t-1)``. Evaluating that string keeps this correct for a custom profile as
    well as the named ones, and there is no second copy of the shapes to drift out of step with
    :data:`~stentfit.balloon.LOAD_PROFILES`.

    :param t: Pseudo-time, 0 to 1.
    :param pressure_max: Peak pressure in MPa, i.e. the value the profile scales.
    :param expression: The profile as a 4C expression in ``t``, peaking at 1.
    :returns: Pressure in MPa.
    """
    scale = eval(expression, {"__builtins__": {}},          # noqa: S307 - own generated file
                 {**_EXPR_NAMES, "t": float(t)})
    return float(pressure_max) * float(scale)


def align_to_reference(points, reference):
    """
    Remove rigid-body motion, mapping points back into the reference configuration's frame.

    The stent is held by only the constraints needed to stop it drifting, so it picks up a small
    translation and rotation over a full inflate-deflate cycle. Radius is measured about a fixed
    axis, so that rotation would read as diameter and a fully recovered elastic stent would report
    permanent expansion it does not have. The rigid transform is removed by the Kabsch
    construction.

    :param points: ``(n, 3)`` deformed coordinates.
    :param reference: ``(n, 3)`` coordinates of the same nodes, undeformed.
    :returns: ``(n, 3)`` deformed coordinates with translation and rotation taken out.
    """
    a = np.asarray(reference, float)
    b = np.asarray(points, float)
    a_mid, b_mid = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - a_mid, b - b_mid

    u, _, vt = np.linalg.svd(ac.T @ bc)
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, flip]) @ u.T       # ac @ rotation.T ~= bc
    return bc @ rotation + a_mid


def junction_cell_map(mesh, junction_coords, junction_degrees):
    """
    Find the beam elements meeting at each labelled junction.

    A junction sits at the same coordinates in every mesh, however finely the struts are divided,
    so it is a valid place to compare results across refinements, where a global peak is not. A
    crown of degree ``d`` has ``d`` coincident points, one per strut ending there, so the degree
    says how many to collect and no distance threshold is needed.

    :param mesh: The mesh at the first timestep.
    :param junction_coords: ``(n, 3)`` junction positions.
    :param junction_degrees: How many struts meet at each junction.
    :returns: List of cell-id arrays, one per junction.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(mesh.points)
    out = []
    for coord, degree in zip(junction_coords, junction_degrees):
        _, idx = tree.query(coord, k=int(degree))
        cells = set()
        for i in np.atleast_1d(idx):
            cells.update(int(c) for c in mesh.point_cell_ids(int(i)))
        out.append(np.array(sorted(cells), dtype=int))
    return out


def reference_frame(mesh, junction_coords=None, junction_degrees=None, section=None):
    """
    Fix the measurement frame from the undeformed configuration.

    Both parts of the frame are chosen once and reused for every timestep:

    ``core``
        The nodes in the free middle of the stent. Choosing them again from each step's deformed
        coordinates would let the band drift as the stent foreshortens, so a slightly different set
        of nodes would be sampled at every step.
    ``cx, cy``
        The stent's own axis. The radial cases scale the stent about this axis, so measuring the
        radius from the origin instead would make an exactly imposed strain read back as something
        slightly different.

    :param mesh: The mesh at the first timestep, i.e. undeformed.
    :param junction_coords: ``(n, 3)`` junction positions, to also map the crowns.
    :param junction_degrees: How many struts meet at each junction.
    :param section: Section properties from ``build_input.section_properties``, supplying the
        yield limits the internal forces are compared against.
    :returns: Dict with ``core``, ``cx``, ``cy`` and optionally ``junction_cells``, ``Np``,
        ``Mp``.
    """
    p = np.asarray(mesh.points, float)
    z = p[:, 2]
    z_mid, half = 0.5 * (z.min() + z.max()), 0.5 * MID_BAND_FRAC * (z.max() - z.min())
    ref = {"core": np.abs(z - z_mid) <= half,
           "cx": float(p[:, 0].mean()), "cy": float(p[:, 1].mean()),
           "points": p.copy(), "z_range": (float(z.min()), float(z.max()))}
    if junction_coords is not None:
        ref["junction_cells"] = junction_cell_map(mesh, junction_coords, junction_degrees)
    if section is not None:
        ref["Np"], ref["Mp"] = section["Np"], section["Mp"]
    return ref


def measure(mesh, ref):
    """
    Reduce one deformed beam mesh to scalar metrics.

    :param mesh: The mesh for one timestep, with deformed point coordinates.
    :param ref: Measurement frame from :func:`reference_frame`, fixed across timesteps.
    :returns: Dict of metrics for this step.
    """
    p = np.asarray(mesh.points, float)
    core = ref["core"]
    r = np.hypot(p[:, 0] - ref["cx"], p[:, 1] - ref["cy"])
    z = p[:, 2]

    out = {"diameter": 2.0 * float(np.percentile(r[core], RADIUS_PERCENTILE)),
           "diameter_ends": 2.0 * float(np.percentile(r[~core], RADIUS_PERCENTILE)),
           "diameter_max": 2.0 * float(r.max()),
           "length": float(z.max() - z.min())}
    out["stent_dogboning_pct"] = 100.0 * (out["diameter_ends"] / out["diameter"] - 1.0)

    # The same diameter with rigid-body motion removed. Reported alongside rather than instead:
    # comparing the two columns is how you see how far the stent drifted.
    aligned = align_to_reference(p, ref["points"])
    r_aligned = np.hypot(aligned[:, 0] - ref["cx"], aligned[:, 1] - ref["cy"])
    out["diameter_aligned"] = 2.0 * float(np.percentile(r_aligned[core], RADIUS_PERCENTILE))

    # Strain arrays are per Gauss point and named by 4C, not by us -- pick up whatever the
    # run wrote rather than assuming a fixed set.
    for name in mesh.cell_data:
        if "strain" not in name and "curvature" not in name:
            continue
        v = np.abs(np.asarray(mesh.cell_data[name], float))

        # Global peak: whichever Gauss point happens to be worst. Useful as an order of
        # magnitude, but it is not a fixed physical location, so it does not converge under
        # mesh refinement -- the maximum simply moves.
        out[f"peak_{name}"] = float(v.max())

        # At the crowns: the same physical points in every mesh, so these can be compared
        # across refinements. Each junction's value is averaged over the elements meeting
        # there; the averaging window shrinks as the mesh refines, so the sequence converges
        # towards the pointwise value at the crown.
        if "junction_cells" in ref:
            per_junction = np.array([v[cells].mean() for cells in ref["junction_cells"]])
            out[f"crown_mean_{name}"] = float(per_junction.mean())
            out[f"crown_max_{name}"] = float(per_junction.max())

    out.update(utilisation(mesh, ref))
    return out


def utilisation(mesh, ref):
    """
    How close the struts come to yielding, from 4C's internal force output.

    Needs ``MATERIAL_FORCES_GAUSSPOINT`` in the input. Two ratios are reported:

    ``N/Np``
        axial force over the force that would yield the section in pure tension
    ``M/Mp``
        bending moment over the moment that would yield it in pure bending

    Below 1 the strut is still elastic, and above 1 a real strut would have yielded. Under an
    elastic material law nothing stops these going above 1, which is the finding rather than a
    problem.

    :param mesh: The mesh for one timestep.
    :param ref: Measurement frame, carrying ``Np`` and ``Mp``.
    :returns: Dict of utilisation ratios, empty if the forces were not written.
    """
    if "Np" not in ref or "material_axial_force_GPs" not in mesh.cell_data:
        return {}

    n = np.abs(np.asarray(mesh.cell_data["material_axial_force_GPs"], float))
    m2 = np.asarray(mesh.cell_data["material_bending_moment_2_GPs"], float)
    m3 = np.asarray(mesh.cell_data["material_bending_moment_3_GPs"], float)
    m = np.hypot(m2, m3)                     # resultant moment, both bending directions

    out = {"peak_N_over_Np": float(n.max() / ref["Np"]),
           "peak_M_over_Mp": float(m.max() / ref["Mp"])}
    if "junction_cells" in ref:
        cells = ref["junction_cells"]
        out["crown_N_over_Np"] = float(
            np.mean([n[c].mean() for c in cells]) / ref["Np"])
        out["crown_M_over_Mp"] = float(
            np.mean([m[c].mean() for c in cells]) / ref["Mp"])
    return out


def balloon_outer_surface(mesh, r_mid: float):
    """
    The balloon's outer surface, undeformed and deformed.

    The solid output does not set ``USE_ABSOLUTE_POSITIONS``, so unlike the beams its point
    coordinates are the reference ones and all the motion sits in the ``displacement`` array.
    Reading ``mesh.points`` alone would report a balloon that never inflates. Only the outer surface
    is wanted, since that is the face the struts press on, and it is picked by reference radius
    above mid-wall.

    :param mesh: The balloon mesh for one timestep.
    :param r_mid: Mid-wall radius, i.e. halfway between the balloon's inner and outer radii.
    :returns: ``(reference, deformed)``, each ``(n, 3)``, for the outer-surface nodes only.
    """
    p = np.asarray(mesh.points, float)
    u = np.asarray(mesh.point_data["displacement"], float)
    outer = np.hypot(p[:, 0], p[:, 1]) > r_mid
    return p[outer], (p + u)[outer]


def contact_gap(beam_points, balloon_points, beam_radius: float):
    """
    Signed distance from every beam node to the balloon surface.

    Penalty contact is a spring rather than a wall. The force it applies is the overlap times the
    penalty parameter, so some overlap always exists and it grows with the load. Measuring it is
    the only way to tell whether the contact is tight enough for the answer to be trusted.

    :param beam_points: ``(n, 3)`` deformed beam centreline nodes.
    :param balloon_points: ``(m, 3)`` deformed balloon outer-surface nodes.
    :param beam_radius: Strut cross-section radius, since the beam is a centreline and the
        contact geometry is a tube of this radius around it.
    :returns: ``(n,)`` gaps. Negative means the strut has sunk into the balloon.
    """
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(np.asarray(balloon_points, float)).query(np.asarray(beam_points, float))
    return distance - beam_radius


def ring_profile(reference_outer, deformed_outer):
    """
    Outer radius against axial position, averaged around each ring of nodes.

    This is the shape the paper plots in its Fig. 5. The rings are grouped by reference z, because
    the balloon stretches axially as it inflates, and binning the deformed coordinates would mix
    neighbouring rings together near the ends, which is where the interesting part of the profile
    is.

    :param reference_outer: ``(n, 3)`` undeformed outer-surface nodes.
    :param deformed_outer: ``(n, 3)`` the same nodes, deformed.
    :returns: ``(z, radius)``, one entry per ring, ordered along the axis.
    """
    z = np.round(np.asarray(reference_outer, float)[:, 2], 6)
    r = np.hypot(deformed_outer[:, 0], deformed_outer[:, 1])
    rings = np.unique(z)
    return rings, np.array([r[z == value].mean() for value in rings])


def balloon_metrics(reference_outer, deformed_outer, stent_z, r_outer_0: float) -> dict:
    """
    Reduce the balloon's shape to the numbers that can be compared with the paper.

    ``balloon_dogboning_pct``
        How much further the balloon opens where the stent does not hold it back. It is a
        transient, which Datz et al.'s Fig. 5 shows growing through mid-inflation and vanishing
        once the balloon's fibres reach their stiffening knee. So a large value means the balloon
        is under-inflated rather than that anything is wrong.
    ``balloon_r_tip_ratio``
        The end ring's radius over its starting value. The ends are sprung to stand in for the
        catheter shaft, so this should stay near 1. It is read from the end ring itself, because an
        averaged slice would mix in the free overhang.

    :param reference_outer: ``(n, 3)`` undeformed outer-surface nodes.
    :param deformed_outer: ``(n, 3)`` the same nodes, deformed.
    :param stent_z: ``(z_min, z_max)`` of the stent, splitting the balloon into the part the
        stent restrains and the free overhang.
    :param r_outer_0: The balloon's undeformed outer radius.
    :returns: Dict of balloon shape metrics.
    """
    z = np.asarray(reference_outer, float)[:, 2]
    r = np.hypot(deformed_outer[:, 0], deformed_outer[:, 1])
    under = (z > stent_z[0]) & (z < stent_z[1])
    past = ~under
    tip = np.isclose(z, z.min()) | np.isclose(z, z.max())

    out = {"balloon_r_tip": float(r[tip].mean()),
           "balloon_r_tip_ratio": float(r[tip].mean() / r_outer_0)}
    if not under.any():
        return out

    out["balloon_r_under_stent"] = float(r[under].mean())
    if past.any():
        out["balloon_r_bulge"] = float(r[past].max())
        out["balloon_dogboning_pct"] = 100.0 * (r[past].max() / r[under].mean() - 1.0)
    return out


def process_case(case_dir, junctions=None, section: dict = None,
                 balloon: dict = None, pressure: dict = None) -> list:
    """
    Measure every timestep of one run.

    ``balloon`` and ``pressure`` are optional, because not every simulation type has them.
    ``stent_only`` has no balloon, so it gets none of those columns.

    :param case_dir: The run folder, holding the ``.pvd`` and its ``.vtu`` files.
    :param junctions: ``(coords, degrees)`` from
        :func:`~stentfit.sim.beam_model.read_junctions`, to also report quantities at the
        crowns. ``None`` reports global quantities only.
    :param section: The strut section, supplying the yield limits the internal forces are
        compared against, and the strut radius the contact gap is measured from. ``None`` skips
        the utilisation ratios.
    :param balloon: ``{"r_inner": mm, "r_outer": mm}`` for the undeformed balloon, enabling the
        contact and balloon-shape columns. ``None`` skips them.
    :param pressure: ``{"max": MPa, "expression": str}``, enabling the ``pressure_MPa`` column.
        ``None`` skips it.
    :returns: One dict per timestep, or ``None`` if the run has no results yet.
    """
    pv = _pyvista()
    case_dir = Path(case_dir)
    pair = tuple(junctions) if junctions is not None else (None, None)
    pvds = sorted(case_dir.glob("*-structure-beams.pvd"))
    if not pvds:
        return None

    steps = read_pvd(pvds[0])

    # The balloon is a separate time series. Its steps line up with the beams', so they are
    # looked up by timestep rather than by position -- a run killed mid-step can leave the two
    # collections one entry apart.
    balloon_steps, r_mid = {}, None
    if balloon:
        solid = sorted(case_dir.glob("*-structure.pvd"))
        if solid:
            balloon_steps = dict(read_pvd(solid[0]))
            r_mid = 0.5 * (balloon["r_inner"] + balloon["r_outer"])

    rows, ref = [], None
    for t, vtu in steps:
        if not vtu.exists():
            continue
        mesh = pv.read(vtu)
        if ref is None:                        # fix the measurement frame on the first step
            ref = reference_frame(mesh, *pair, section=section)
        m = measure(mesh, ref)
        m["time"] = t
        if pressure:
            m["pressure_MPa"] = pressure_at(t, pressure["max"], pressure["expression"])

        solid_vtu = balloon_steps.get(t)
        if solid_vtu is not None and solid_vtu.exists():
            reference_outer, deformed_outer = balloon_outer_surface(pv.read(solid_vtu), r_mid)
            m.update(balloon_metrics(reference_outer, deformed_outer,
                                     ref["z_range"], balloon["r_outer"]))
            if section is not None:
                gap = contact_gap(mesh.points, deformed_outer, section["radius"])
                overlap = -gap[gap < 0]
                m["penetration_max_mm"] = float(overlap.max()) if overlap.size else 0.0
                m["penetration_per_strut"] = m["penetration_max_mm"] / (2.0 * section["radius"])
                m["n_nodes_in_contact"] = int(overlap.size)
        rows.append(m)

    if not rows:
        return None

    # Everything is relative to the undeformed state, which is the first step.
    d0, l0 = rows[0]["diameter"], rows[0]["length"]
    da0 = rows[0]["diameter_aligned"]
    for m in rows:
        m["radial_strain"] = (m["diameter"] - d0) / d0
        m["radial_strain_aligned"] = (m["diameter_aligned"] - da0) / da0
        m["axial_strain"] = (m["length"] - l0) / l0
        m["foreshortening_pct"] = -100.0 * m["axial_strain"]
    return rows


def write_csv(path, rows: list) -> Path:
    """
    Write per-step metrics as CSV.

    Columns are the union of every row's keys, so a run that wrote extra fields keeps them.

    :param path: Where to write.
    :param rows: Per-step metrics from :func:`process_case`.
    :returns: The path written.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    fields = ["time"] + [f for f in fields if f != "time"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def peak_index(rows: list) -> int:
    """
    :param rows: Per-step metrics from :func:`process_case`.
    :returns: Index of the step with the largest stent diameter.
    """
    return max(range(len(rows)), key=lambda i: rows[i]["diameter"])


def radial_stiffness(rows: list):
    """
    How far the stent opens per unit of balloon pressure, in mm/MPa.

    This is a secant from the unloaded state to the peak diameter rather than a tangent. The
    response is strongly non-linear, since the balloon has to cross the clearance gap before
    anything happens at all, so a tangent would say more about where it was taken than about the
    stent.

    :param rows: Per-step metrics from :func:`process_case`.
    :returns: The secant slope, or ``None`` without a pressure column.
    """
    if not rows or "pressure_MPa" not in rows[0]:
        return None
    key = "diameter_aligned" if "diameter_aligned" in rows[0] else "diameter"
    top = peak_index(rows)
    span = rows[top]["pressure_MPa"] - rows[0]["pressure_MPa"]
    if span <= 0:
        return None
    return (rows[top][key] - rows[0][key]) / span


def summarise(rows: list) -> dict:
    """
    Reduce a run to its headline numbers.

    Two states matter and they are rarely the same step. The final state is what the stent is left
    as, and the peak state is what it was asked to do. For an inflate-deflate profile the peak is
    where every comparison with the paper is made.

    :param rows: Per-step metrics from :func:`process_case`.
    :returns: The final state, the peak state, and how far the run actually got.
    """
    last = rows[-1]
    top = rows[peak_index(rows)]

    out = {"steps": len(rows), "t_end": last["time"],
           "diameter_mm": last["diameter"], "length_mm": last["length"],
           "radial_strain": last["radial_strain"],
           "axial_strain": last["axial_strain"],
           "foreshortening_pct": last["foreshortening_pct"],
           "diameter_peak_mm": top["diameter"],
           "diameter_peak_aligned_mm": top["diameter_aligned"],
           "t_at_peak": top["time"],
           "foreshortening_pct_at_peak": top["foreshortening_pct"],
           "stent_dogboning_pct_at_peak": top["stent_dogboning_pct"],
           **{k: v for k, v in last.items() if k.startswith(("peak_", "crown_"))}}

    for key, name in [("pressure_MPa", "pressure_at_peak_MPa"),
                      ("balloon_dogboning_pct", "balloon_dogboning_pct_at_peak"),
                      ("balloon_r_tip_ratio", "balloon_r_tip_ratio_at_peak"),
                      ("penetration_max_mm", "penetration_max_mm_at_peak"),
                      ("penetration_per_strut", "penetration_per_strut_at_peak")]:
        if key in top:
            out[name] = top[key]

    stiffness = radial_stiffness(rows)
    if stiffness is not None:
        out["radial_stiffness_mm_per_MPa"] = stiffness
    return out


def recoil(rows: list) -> dict:
    """
    How much of the expansion was given back once the load came off.

    This only means something for a load profile that returns to zero. An elastic stent recovers
    essentially all of it, so a recoil near 100% is the signature of a material that cannot keep
    its new shape. It is measured on ``diameter_aligned`` when that is available, because on the raw
    diameter the rigid-body drift shows up as permanent gain.

    :param rows: Per-step metrics from :func:`process_case`.
    :returns: The peak and final diameters and the recovered fraction, or ``None`` if the run
        never came back off its peak.
    """
    key = "diameter_aligned" if rows and "diameter_aligned" in rows[0] else "diameter"
    diameters = [row[key] for row in rows]
    peak = max(diameters)
    peak_index = diameters.index(peak)
    if peak_index >= len(diameters) - 1:
        return None
    start, final = diameters[0], diameters[-1]
    gained = peak - start
    if gained <= 0:
        return None
    return {"diameter_start_mm": start, "diameter_peak_mm": peak, "diameter_final_mm": final,
            "peak_at_t": rows[peak_index]["time"],
            "recoil_pct": 100.0 * (peak - final) / gained,
            "permanent_gain_mm": final - start}


def profile_table(case_dir, rows: list, balloon: dict, n_curves: int = 6) -> dict:
    """
    Balloon radius against axial position, at several pressures.

    This is the paper's Fig. 5, and it is the most direct comparison available. The curves are
    taken from the inflation branch only, up to and including the peak, because an inflate-deflate
    profile passes every pressure twice and the stent has opened in between.

    :param case_dir: The run folder.
    :param rows: Per-step metrics from :func:`process_case`, carrying ``pressure_MPa``.
    :param balloon: ``{"r_inner": mm, "r_outer": mm}`` for the undeformed balloon.
    :param n_curves: How many pressures to sample, spread evenly up to the peak.
    :returns: ``{"z": array, "curves": [(pressure, radii), ...]}``, or ``None`` if the run has no
        balloon output or no pressure column.
    """
    pv = _pyvista()
    solid = sorted(Path(case_dir).glob("*-structure.pvd"))
    if not solid or not rows or "pressure_MPa" not in rows[0]:
        return None

    steps = dict(read_pvd(solid[0]))
    r_mid = 0.5 * (balloon["r_inner"] + balloon["r_outer"])
    inflating = rows[:peak_index(rows) + 1]
    if not inflating:
        return None

    top = inflating[-1]["pressure_MPa"]
    wanted = np.linspace(0.0, top, n_curves)

    z, curves = None, []
    for target in wanted:
        row = min(inflating, key=lambda r: abs(r["pressure_MPa"] - target))
        vtu = steps.get(row["time"])
        if vtu is None or not vtu.exists():
            continue
        reference_outer, deformed_outer = balloon_outer_surface(pv.read(vtu), r_mid)
        z, radii = ring_profile(reference_outer, deformed_outer)
        curves.append((row["pressure_MPa"], radii))
    return {"z": z, "curves": curves} if curves else None


def write_profile_csv(path, profile: dict) -> Path:
    """
    Write a balloon profile as CSV: one row per node ring, one column per pressure.

    :param path: Where to write.
    :param profile: The table from :func:`profile_table`.
    :returns: The path written.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z_mm"] + [f"r_mm_at_{p:.4f}MPa" for p, _ in profile["curves"]])
        for i, z in enumerate(profile["z"]):
            writer.writerow([f"{z:.6f}"] + [f"{radii[i]:.6f}" for _, radii in profile["curves"]])
    return path

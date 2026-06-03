import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


def _compute_rmf(cl: np.ndarray):
    """Rotation-minimising frame (Bishop frame) along a polyline centreline."""
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


def map_stent_to_artery(
    skel: pd.DataFrame,
    features: dict,
    artery_cl: np.ndarray,
) -> tuple:
    """
    Map straight stent skeleton nodes onto the artery centreline using a
    rotation-minimising frame (RMF). The stent is centred within the artery arc.

    Parameters
    ----------
    skel      : skeleton DataFrame (skeleton_point_id, x, y, z, node_type, neighbor_ids)
    features  : dict  each key maps to {"value": ..., "unit": "..."}
    artery_cl : (M, 3) ndarray  artery centreline points

    Returns
    -------
    skel_mapped : (N, 3) ndarray  skeleton nodes mapped onto the artery
    cl_m        : (N, 3) ndarray  corresponding centreline position per node
    """
    seg_lens = np.linalg.norm(np.diff(artery_cl, axis=0), axis=1)
    arc_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    t_artery = arc_lens / arc_lens[-1]

    _, N_arr, B_arr = _compute_rmf(artery_cl)

    stent_arc_frac = features["length"]["value"] / arc_lens[-1]
    t_start        = (1.0 - stent_arc_frac) / 2.0

    z_min, z_max = features["z_min"]["value"], features["z_max"]["value"]
    xyz     = skel[["x", "y", "z"]].values
    t_stent = (xyz[:, 2] - z_min) / (z_max - z_min)
    t_a     = np.clip(t_start + t_stent * stent_arc_frac, 0.0, 1.0)

    cl_m = np.column_stack([np.interp(t_a, t_artery, artery_cl[:, k]) for k in range(3)])
    N_m  = np.column_stack([np.interp(t_a, t_artery, N_arr[:, k])     for k in range(3)])
    B_m  = np.column_stack([np.interp(t_a, t_artery, B_arr[:, k])     for k in range(3)])
    N_m /= np.linalg.norm(N_m, axis=1, keepdims=True)
    B_m /= np.linalg.norm(B_m, axis=1, keepdims=True)

    skel_mapped = cl_m + xyz[:, 0:1] * N_m + xyz[:, 1:2] * B_m
    print(f"Mapped {len(skel_mapped):,} skeleton nodes")

    return skel_mapped, cl_m


def _inside_mesh_kdtree(mesh: trimesh.Trimesh, points: np.ndarray):
    """
    Point-in-closed-mesh test — no rtree required.

    Builds a KD-tree on mesh *vertices*, finds the nearest surface vertex for
    each query point, then checks the dot product of the vector
    (surface_vertex → query_point) with the outward vertex normal.

      dot < 0  →  inside  (vector opposes outward normal)
      dot ≥ 0  →  outside

    Returns
    -------
    inside : (N,) bool ndarray
    dists  : (N,) float ndarray — distance to nearest surface vertex [mm]
    """
    tree = cKDTree(mesh.vertices)
    dists, vidx = tree.query(points, workers=-1)
    to_pt   = points - mesh.vertices[vidx]
    normals = mesh.vertex_normals[vidx]
    dot     = np.einsum("ij,ij->i", to_pt, normals)
    return dot < 0, dists


def check_stent_artery_fit(
    skel_mapped: np.ndarray,
    skel: "pd.DataFrame",
    features: dict,
    artery_mesh: trimesh.Trimesh,
    artery_cl: np.ndarray,
    artery_radius: float,
    n_sample: int = 3000,
    rng_seed: int = 42,
) -> dict:
    """
    Five compatibility checks between a placed stent skeleton and an artery mesh.
    Check 5 (bending strain) is material-based when 'max_elastic_strain' is present
    in features, otherwise falls back to a geometric heuristic.

    Parameters
    ----------
    skel_mapped   : (N, 3) ndarray
    skel          : DataFrame  (skeleton table with node_type, neighbor_ids)
    features      : dict       (each key maps to {"value": ..., "unit": "..."})
    artery_mesh   : trimesh.Trimesh
    artery_cl     : (M, 3) ndarray  artery centreline
    artery_radius : float           nominal artery lumen radius [mm]
    n_sample      : int             nodes sampled for checks 3 & 4

    Returns
    -------
    dict  {check_name: {passed, note, …values…}}
    """
    rng    = np.random.default_rng(rng_seed)
    checks = {}

    # ── 1. Length ──────────────────────────────────────────────────────────────
    seg_l = np.linalg.norm(np.diff(artery_cl, axis=0), axis=1)
    arc   = seg_l.sum()
    slen  = features["length"]["value"]
    fill  = slen / arc
    checks["length"] = dict(
        stent_length_mm = round(slen, 3),
        artery_arc_mm   = round(arc,  3),
        fill_fraction   = round(fill, 4),
        passed          = fill < 0.95,
        note = f"Stent {slen:.2f} mm fills {fill*100:.1f}% of {arc:.2f} mm artery arc",
    )

    # ── 2. Delivery feasibility ────────────────────────────────────────────────
    stent_od = 2.0 * features["r_outer"]["value"]
    artery_d = 2.0 * artery_radius
    fits     = stent_od < artery_d
    margin   = (artery_d - stent_od) / artery_d * 100
    checks["delivery"] = dict(
        crimped_OD_mm     = round(stent_od, 3),
        artery_ID_mm      = round(artery_d,  3),
        radial_margin_pct = round(margin, 1),
        passed            = fits,
        note = (
            f"Crimped OD {stent_od:.3f} mm < artery ID {artery_d:.3f} mm "
            f"({margin:.1f}% radial margin)"
            if fits else
            f"Crimped OD {stent_od:.3f} mm exceeds artery ID {artery_d:.3f} mm — delivery blocked"
        ),
    )

    # ── 3 & 4. Containment + wall clearance (shared KD-tree query) ────────────
    print("  [3/5] Containment + [4/5] Clearance … ", end="", flush=True)
    idx = rng.choice(len(skel_mapped), min(n_sample, len(skel_mapped)), replace=False)
    inside, dists = _inside_mesh_kdtree(artery_mesh, skel_mapped[idx])

    pct = inside.mean() * 100
    checks["containment"] = dict(
        sampled    = len(idx),
        pct_inside = round(pct, 1),
        passed     = pct >= 95.0,
        note = f"{pct:.1f}% of {len(idx):,} sampled nodes inside artery lumen",
    )

    n_penet  = int((~inside).sum())
    clr      = np.where(inside, dists, -dists)
    min_cl   = float(clr.min())
    strut_r  = features["strut_thickness"]["value"] / 2.0 if "strut_thickness" in features else 0.0
    passed_cl = (n_penet == 0) and (min_cl >= strut_r)
    print(f"{pct:.1f}% inside, min clearance {min_cl:.3f} mm, {n_penet} penetrating")
    checks["clearance"] = dict(
        min_clearance_mm = round(min_cl,  4),
        strut_radius_mm  = round(strut_r, 4),
        n_penetrating    = n_penet,
        sampled          = len(idx),
        passed           = passed_cl,
        note = (
            f"Min wall clearance {min_cl:.3f} mm "
            f"(strut radius {strut_r:.3f} mm), {n_penet} penetrating nodes"
        ),
    )

    # ── 5. Bending strain at artery curvature ─────────────────────────────────
    T    = np.diff(artery_cl, axis=0)
    sl   = np.linalg.norm(T, axis=1)
    T_u  = T / sl[:, None]
    dT   = np.diff(T_u, axis=0)
    avg  = (sl[:-1] + sl[1:]) / 2.0
    kap  = np.linalg.norm(dT, axis=1) / avg
    kmax = float(kap.max()) if len(kap) else 0.0
    Rmin = (1.0 / kmax) if kmax > 1e-9 else float("inf")

    E    = features["youngs_modulus"]["value"] if "youngs_modulus" in features else None
    EI   = (E * np.pi * strut_r**4 / 4.0) if (E and strut_r > 0) else None

    eps_max = features["max_elastic_strain"]["value"] if "max_elastic_strain" in features else None

    if eps_max is not None and strut_r > 0:
        eps_actual = (strut_r / Rmin) if np.isfinite(Rmin) else 0.0
        ok_curv    = eps_actual < eps_max
        if np.isfinite(Rmin):
            curv_note = (
                f"Max strut bending strain {eps_actual*100:.3f}% at R_min={Rmin:.1f} mm "
                f"({'<' if ok_curv else '≥'} elastic limit {eps_max*100:.1f}%)"
            )
        else:
            curv_note = "Artery is straight — zero bending strain"
            ok_curv   = True
        check_basis = f"material  [ε = r_strut/R,  limit = {eps_max*100:.1f}%]"
    else:
        thresh      = slen / 2.0
        eps_actual  = None
        ok_curv     = Rmin > thresh
        if np.isfinite(Rmin):
            curv_note = (
                f"Min artery bend radius {Rmin:.1f} mm "
                f"({'>' if ok_curv else '<'} heuristic limit {thresh:.1f} mm = stent_length/2)"
            )
        else:
            curv_note = "Artery is straight — no curvature constraint"
            ok_curv   = True
        check_basis = "geometric heuristic  (set max_elastic_strain for material check)"

    checks["bending_strain"] = dict(
        max_curvature_per_mm      = round(kmax, 6),
        min_bend_radius_mm        = round(Rmin, 2) if np.isfinite(Rmin) else "inf",
        max_bending_strain_pct    = round(eps_actual * 100, 4) if eps_actual is not None else "N/A",
        elastic_limit_pct         = round(eps_max * 100, 2) if eps_max is not None else "N/A",
        bending_stiffness_EI_Nmm2 = round(EI, 4) if EI is not None else "N/A",
        check_basis               = check_basis,
        passed                    = ok_curv,
        note                      = curv_note,
    )

    return checks

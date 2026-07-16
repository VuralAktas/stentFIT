import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


def downsample_skeleton(skel: pd.DataFrame, factor: int = 1) -> pd.DataFrame:
    """
    Topology-preserving downsampling of a stent skeleton graph.

    Every junction and endpoint (degree != 2) is always kept so the wireframe
    topology is preserved. The straight degree-2 chains *between* them are
    thinned by ``factor`` (every ``factor``-th node along the chain is kept),
    so strut curvature is retained rather than collapsed to a single edge.

    The returned DataFrame keeps the same schema and re-numbers
    ``skeleton_point_id`` to a contiguous ``0..M-1`` matching row order, with
    ``neighbor_ids`` remapped accordingly — this is required by the downstream
    mapping / beam-mesh code, which indexes the mapped node array positionally.

    Parameters
    ----------
    skel   : skeleton DataFrame with columns
             skeleton_point_id, x, y, z, r, theta, node_type, degree, neighbor_ids
    factor : int  keep every ``factor``-th node along each chain (1 = no change)

    Returns
    -------
    pd.DataFrame  reduced skeleton with the same columns
    """
    if factor is None or factor <= 1:
        return skel.reset_index(drop=True)

    # ── Build adjacency (filtered to existing ids) ────────────────────────
    adj = {int(r.skeleton_point_id): list(r.neighbor_ids)
           for r in skel.itertuples()}
    ids = set(adj)
    adj = {k: [n for n in v if n in ids] for k, v in adj.items()}
    deg = {k: len(v) for k, v in adj.items()}
    essential = {k for k, d in deg.items() if d != 2}

    def _walk_chain(start, second):
        """Ordered node ids from an essential node along a degree-2 chain
        until the next essential node (or back to start for a pure loop)."""
        path = [start, second]
        prev, cur = start, second
        while cur not in essential:
            nxt = next((n for n in adj[cur] if n != prev), None)
            if nxt is None:
                break
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == start:           # closed loop
                break
        return path

    kept_edges = set()                  # frozenset({a, b}) in ORIGINAL ids
    kept_nodes = set(essential)         # isolated nodes (deg 0) included here
    processed = set()                   # every node touched by a walked chain
    visited_dir = set()                 # directed (a, b) first-steps already walked

    # ── Chains anchored at essential nodes ────────────────────────────────
    for e in essential:
        for nb in adj[e]:
            if (e, nb) in visited_dir:
                continue
            path = _walk_chain(e, nb)
            visited_dir.add((e, nb))
            visited_dir.add((path[-1], path[-2]))   # reverse end of this chain
            processed.update(path)
            # Thin the interior by `factor` but never to zero: keeping >= 1
            # interior node retains strut curvature AND prevents parallel
            # struts (A-i-B and A-j-B) from collapsing onto the same A-B edge.
            interior = path[1:-1]
            kept_interior = interior[::factor]          # non-empty if interior is
            if path[0] == path[-1] and len(kept_interior) < 2:
                # self-loop A..A needs >= 2 distinct interior nodes, otherwise
                # its two edges dedup into a single dangling spur
                kept_interior = interior[:2]
            keep = [path[0]] + kept_interior + [path[-1]]
            kept_nodes.update(keep)
            for a, b in zip(keep[:-1], keep[1:]):
                if a != b:
                    kept_edges.add(frozenset((a, b)))

    # ── Pure degree-2 loops with no essential anchor (e.g. closed rings) ──
    # Only nodes never touched above; decimated-away chain interiors are
    # already in `processed` and must NOT be re-created as spurious rings.
    unvisited = {k for k in adj if k not in processed and deg[k] == 2}
    while unvisited:
        start = unvisited.pop()
        path = _walk_chain(start, adj[start][0])    # returns to start
        if path[-1] == start:
            path = path[:-1]
        processed.update(path)
        unvisited -= set(path)
        keep = path[::factor] or [start]
        kept_nodes.update(keep)
        ring = keep + [keep[0]]                     # close the loop
        for a, b in zip(ring[:-1], ring[1:]):
            if a != b:
                kept_edges.add(frozenset((a, b)))

    # ── Re-number to contiguous 0..M-1 in original-id order ───────────────
    keep_sorted = sorted(kept_nodes)
    old_to_new = {old: new for new, old in enumerate(keep_sorted)}

    new_adj = {new: [] for new in range(len(keep_sorted))}
    for e in kept_edges:
        a, b = tuple(e)
        na, nb = old_to_new[a], old_to_new[b]
        new_adj[na].append(nb)
        new_adj[nb].append(na)

    src = skel.set_index("skeleton_point_id")
    rows = []
    for old in keep_sorted:
        new = old_to_new[old]
        nbrs = sorted(new_adj[new])
        d = len(nbrs)
        ntype = ("isolated" if d == 0 else "endpoint" if d == 1
                 else "line" if d == 2 else "junction")
        s = src.loc[old]
        rows.append({
            "skeleton_point_id": new,
            "x": s["x"], "y": s["y"], "z": s["z"],
            "r": s["r"], "theta": s["theta"],
            "node_type": ntype,
            "degree": d,
            "neighbor_ids": nbrs,
        })

    out = pd.DataFrame(rows).sort_values("skeleton_point_id").reset_index(drop=True)
    print(f"Downsampled skeleton (factor={factor}): "
          f"{len(skel):,} → {len(out):,} nodes, "
          f"{len(kept_edges):,} edges  (kept {len(essential):,} junctions/endpoints)")
    return out


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
    artery_cl: np.ndarray,
    artery_radius: float,
    *,
    artery_mesh: "trimesh.Trimesh | None" = None,
    n_sample: int = 3000,
    rng_seed: int = 42,
) -> dict:
    """
    Five compatibility checks between a placed stent skeleton and an artery mesh.
    Check 5 (bending strain) is material-based when 'max_elastic_strain' is present
    in features, otherwise falls back to a geometric heuristic.

    Containment / clearance (checks 3 & 4) are measured as each stent node's
    **radial distance from the artery centreline** vs. the lumen radius. This is
    mesh-independent, so it works for any wall thickness — a two-shell *walled*
    artery mesh has an ambiguous surface inside/outside test, which used to make
    containment misreport. It uses the nominal ``artery_radius`` (wall roughness is
    approximated, not resolved).

    Parameters
    ----------
    skel_mapped   : (N, 3) ndarray  placed (warped) stent node positions
    skel          : DataFrame  (skeleton table with node_type, neighbor_ids)
    features      : dict       (each key maps to {"value": ..., "unit": "..."})
    artery_cl     : (M, 3) ndarray  artery centreline
    artery_radius : float           nominal artery lumen radius [mm]
    artery_mesh   : trimesh.Trimesh, optional  no longer used (kept for API compat)
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

    # ── 3 & 4. Containment + wall clearance (radial distance to the centreline) ─
    # Mesh-independent: compare each sampled stent node's radial distance from the
    # artery centreline against the lumen radius. Robust for any wall thickness /
    # shape (a two-shell walled mesh has an ambiguous inside/outside test).
    print("  [3/5] Containment + [4/5] Clearance … ", end="", flush=True)
    idx = rng.choice(len(skel_mapped), min(n_sample, len(skel_mapped)), replace=False)
    pts = np.asarray(skel_mapped)[idx]

    tang = np.gradient(artery_cl, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    _, nidx = cKDTree(artery_cl).query(pts, workers=-1)
    vec = pts - artery_cl[nidx]
    axial = np.einsum("ij,ij->i", vec, tang[nidx])                 # component along the axis
    radial_dist = np.sqrt(np.maximum(np.einsum("ij,ij->i", vec, vec) - axial ** 2, 0.0))

    inside = radial_dist < artery_radius
    pct = inside.mean() * 100
    checks["containment"] = dict(
        sampled    = len(idx),
        pct_inside = round(pct, 1),
        passed     = pct >= 95.0,
        note = (f"{pct:.1f}% of {len(idx):,} sampled nodes inside artery lumen "
                f"(radial distance < {artery_radius:.3f} mm)"),
    )

    clearance = artery_radius - radial_dist          # + inside the lumen, - penetrating
    n_penet   = int((radial_dist >= artery_radius).sum())
    min_cl    = float(clearance.min())
    strut_r   = features["strut_thickness"]["value"] / 2.0 if "strut_thickness" in features else 0.0
    passed_cl = (n_penet == 0) and (min_cl >= strut_r)
    print(f"{pct:.1f}% inside, min clearance {min_cl:.3f} mm, {n_penet} penetrating")
    checks["clearance"] = dict(
        min_clearance_mm = round(min_cl,  4),
        strut_radius_mm  = round(strut_r, 4),
        n_penetrating    = n_penet,
        sampled          = len(idx),
        passed           = passed_cl,
        note = (
            f"Min lumen clearance {min_cl:.3f} mm "
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


def check_coupling_assumptions(
    beam_youngs: float,
    solid_youngs: float,
    beam_diameter: float,
    beam_element_length: float,
    solid_element_length: float,
    stiffness_ratio_min: float = 10.0,
    length_ratio_min: float = 1,
    length_ratio_max: float = 6,
    length_ratio_accuracy_max: float = 8.0,
) -> dict:
    """Validity checks for mixed-dimensional 1D-beam-to-3D-solid (mortar) coupling.

    These are the Steinbrecher et al. conditions the coupling relies on, and are
    separate from the geometric ``check_stent_artery_fit`` (which checks fit, not
    the numerical coupling assumptions):

    1. **stiffness** — the beam (stent) must be much stiffer than the solid
       (artery): ``E_beam / E_solid >= stiffness_ratio_min``.
    2. **rule_of_thumb** — the solid element must be at least as large as the beam
       cross-section: ``L_solid >= D_beam`` (i.e. the beam cross-section is small
       compared to the solid elements).
    3. **element_length_ratio** — the beam-to-solid element ratio ``r = L_beam / L_solid``
       must sit in the valid band ``length_ratio_min <= r <= length_ratio_accuracy_max``:
       a **lower** bound (beam elements fairly long vs the solid, so the mortar
       coupling is well-conditioned) and an **upper accuracy** bound — the L2 error
       of the coupling is flat up to ``r ~ 8`` and then climbs steeply (Steinbrecher
       et al. convergence study), so ``r`` must not exceed ``length_ratio_accuracy_max``.
       The recommended optimum is ``length_ratio_min .. length_ratio_max`` (~2.5-5).

    Parameters
    ----------
    beam_youngs, solid_youngs : float   Young's moduli of the beam / solid [MPa]
    beam_diameter             : float   beam cross-section diameter [mm] (= 2 x beam radius = strut thickness)
    beam_element_length       : float   beam (1D) element length [mm]
    solid_element_length      : float   representative solid element edge length [mm]
    stiffness_ratio_min       : float   min E_beam / E_solid
    length_ratio_min, length_ratio_max     : float   recommended (optimal) band for L_beam / L_solid
    length_ratio_accuracy_max : float   hard upper limit on L_beam / L_solid (accuracy cliff, ~8)

    Returns
    -------
    dict  {check_name: {passed, note, ...values...}, "all_passed": bool}
        Also prints a short pass/fail table.
    """
    checks = {}

    # 1. Stiffness ratio -------------------------------------------------------
    stiff = beam_youngs / solid_youngs if solid_youngs > 0 else float("inf")
    ok_stiff = stiff >= stiffness_ratio_min
    checks["stiffness"] = dict(
        E_beam_MPa    = beam_youngs,
        E_solid_MPa   = solid_youngs,
        ratio         = round(stiff, 2),
        threshold_min = stiffness_ratio_min,
        passed        = bool(ok_stiff),
        note = (f"E_beam/E_solid = {stiff:.1f} "
                f"({'>=' if ok_stiff else '<'} {stiffness_ratio_min}) "
                f"- beam {'is' if ok_stiff else 'is NOT'} much stiffer than the solid"),
    )

    # 2. Rule of thumb: solid element size >= beam cross-section diameter -------
    rot = solid_element_length / beam_diameter if beam_diameter > 0 else float("inf")
    ok_rot = solid_element_length >= beam_diameter
    checks["rule_of_thumb"] = dict(
        solid_element_mm = round(solid_element_length, 4),
        beam_diameter_mm = round(beam_diameter, 4),
        ratio            = round(rot, 2),
        threshold_min    = 1.0,
        passed           = bool(ok_rot),
        note = (f"L_solid/D_beam = {rot:.2f} "
                f"({'>=' if ok_rot else '<'} 1) - solid element "
                f"{'>=' if ok_rot else '<'} beam cross-section diameter"),
    )

    # 3. Element length ratio: beam elements long vs solid, but not too long ---
    #    Valid band [length_ratio_min, length_ratio_accuracy_max]: below -> mortar
    #    coupling poorly conditioned; above ~8 -> L2 error grows (Steinbrecher).
    lr = beam_element_length / solid_element_length if solid_element_length > 0 else float("inf")
    ok_lr = length_ratio_min <= lr <= length_ratio_accuracy_max
    within_optimal = length_ratio_min <= lr <= length_ratio_max
    if lr > length_ratio_accuracy_max:
        lr_note = (f"L_beam/L_solid = {lr:.2f} (> {length_ratio_accuracy_max}) "
                   f"- too long: L2 coupling error grows; refine the solid or coarsen the beam")
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
        beam_element_mm  = round(beam_element_length, 4),
        solid_element_mm = round(solid_element_length, 4),
        ratio            = round(lr, 2),
        valid_band       = (length_ratio_min, length_ratio_accuracy_max),
        optimal_band     = (length_ratio_min, length_ratio_max),
        within_optimal   = bool(within_optimal),
        passed           = bool(ok_lr),
        note             = lr_note,
    )

    all_passed = all(c["passed"] for c in checks.values())
    checks["all_passed"] = all_passed

    print("\nMixed-dimensional coupling assumption check")
    print("-------------------------------------------")
    for name in ("stiffness", "rule_of_thumb", "element_length_ratio"):
        c = checks[name]
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name:22s} {c['note']}")
    print(f"  => {'ALL CHECKS PASSED' if all_passed else 'ONE OR MORE CHECKS FAILED'}")

    return checks

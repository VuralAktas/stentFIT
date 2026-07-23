import numpy as np
import pandas as pd
import time
import os
import pickle
from collections import deque, Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, dilation, closing, disk

from .stent_rings import segment_stent
from .stent_plotting import plot_ring_skeleton_2d_html, plot_ring_convergence_html, _render_ring_2d


def open_stent_to_plane(stent_df: pd.DataFrame, r_mid: float) -> dict:
    """
    Unroll a cylindrical stent point cloud onto a flat (arc, z) plane.

    Converts each point's ``theta`` to an arc-length coordinate (``r_mid *
    theta``), leaving ``z`` unchanged. This flat representation is what
    :func:`compute_skeleton_2d` rasterises and thins.

    :param stent_df: Stent point cloud with ``theta`` and ``z_cylindrical`` columns.
    :param r_mid: Mid-wall radius, used to convert angle to arc length.
    :returns: Dict with the unrolled coordinates (``arc_flat``, ``z_flat``),
        the minimum arc value (``arc_min_flat``), and the full ``circumference``
        at ``r_mid``.
    """
    circumference = 2 * np.pi * r_mid
    arc_flat      = r_mid * stent_df['theta'].values
    z_flat        = stent_df['z_cylindrical'].values
    arc_min_flat  = arc_flat.min()

    return {
        'arc_flat'     : arc_flat,
        'z_flat'       : z_flat,
        'arc_min_flat' : arc_min_flat,
        'circumference': circumference,
    }



def compute_skeleton_2d(
    arc_flat: np.ndarray,
    z_flat: np.ndarray,
    arc_min_flat: float,
    circumference: float,
    stent_df: pd.DataFrame,
    stent_geometry: dict,
    pixels_per_strut: int,
    dilate_px: int,
    pad_fraction: float,
) -> dict:
    """
    Rasterise, dilate, and thin the flat surface points into a 2D skeleton.

    The (arc, z) points are rasterised onto a pixel grid sized so one strut
    is ``pixels_per_strut`` pixels wide, with the arc axis wrapped (the last
    column borders the first) so a strut sitting on the seam stays one
    continuous line. The raster is dilated by ``dilate_px``, closed to fill
    tiny gaps, then thinned to a 1-pixel-wide skeleton
    (``skimage.morphology.skeletonize``). A padded copy of the wrap is used
    only during thinning so the seam sees its true neighbours; the padding is
    cropped back off before returning.

    :param arc_flat: Flat arc-coordinates of the ring's surface points, from
        :func:`open_stent_to_plane`.
    :param z_flat: Flat z-coordinates of the ring's surface points.
    :param arc_min_flat: Minimum arc value, used as the raster's column origin.
    :param circumference: Full circumference at the ring's mid-wall radius,
        sets the raster width.
    :param stent_df: Ring's surface point cloud (unused directly here, kept
        for a consistent call signature with the quality check).
    :param stent_geometry: Stent features dict; only ``strut_thickness`` is used.
    :param pixels_per_strut: Raster resolution, in pixels across one strut width.
    :param dilate_px: Dilation radius, in pixels, before thinning.
    :param pad_fraction: Seam padding, as a fraction of the raster width, so
        thinning sees the wrap correctly.
    :returns: Dict with the skeleton's flat coordinates (``skel_arc``,
        ``skel_z``), the ``pixel_size`` used, and the same points as a
        DataFrame (``df_skeleton_2d``).
    """
    pixel_size = stent_geometry['strut_thickness'] / pixels_per_strut

    # periodic (cylindrical) raster: one circumference wide, the arc wraps 
    # The last column is a neighbour of the first, so there is NO seam cut. Nothing
    # gets sliced, so a strut sitting on the seam stays one continuous line.
    n_cols = max(1, int(round(circumference / pixel_size)))
    z_lo   = z_flat.min()
    n_rows = int(np.ceil((z_flat.max() - z_lo) / pixel_size)) + 1

    # Every point gets a column 0..n_cols-1; going past the end wraps back to 0.
    col = np.mod(np.round((arc_flat - arc_min_flat) / pixel_size).astype(int), n_cols)
    row = np.clip(np.round((z_flat - z_lo) / pixel_size).astype(int), 0, n_rows - 1)

    img = np.zeros((n_rows, n_cols), dtype=bool)
    img[row, col] = True

    # Copy a strip from each side onto the other side ("wrap"), so the thinning sees
    # the strut continuously across the seam. pad_cols is just working room.
    pad_cols = min(n_cols, max(dilate_px + 1, int(round(pad_fraction * n_cols))))
    img_wrap = np.pad(img, ((0, 0), (pad_cols, pad_cols)), mode='wrap')

    img_solid = dilation(img_wrap, footprint=disk(dilate_px))
    img_solid = closing (img_solid, footprint=disk(1))
    img_skel  = skeletonize(img_solid)

    # Keep just the middle one turn (drop the wrap strips we added).
    sk_rows, sk_cols_w = np.where(img_skel)
    keep    = (sk_cols_w >= pad_cols) & (sk_cols_w < pad_cols + n_cols)
    sk_rows = sk_rows[keep]
    sk_cols = sk_cols_w[keep] - pad_cols

    skel_arc = arc_min_flat + (sk_cols + 0.5) * pixel_size
    skel_z   = z_lo         + (sk_rows + 0.5) * pixel_size

    return {
        'skel_arc'      : skel_arc,
        'skel_z'        : skel_z,
        'pixel_size'    : pixel_size,
        'df_skeleton_2d': pd.DataFrame({'arc': skel_arc, 'z': skel_z}),
    }



def _two_core_mask(V: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Find a graph's 2-core: every node with degree >= 2 once dead-ends are
    peeled away.

    Repeatedly removes degree-<=1 nodes (leaves), updating their neighbours'
    degree as they go, until only nodes that are part of a cycle remain.
    Used to distinguish a real loop from a simple dead-end branch: a node
    survives here only if it's actually on a cycle, not just reachable from one.

    :param V: Total number of nodes in the graph.
    :param a: Edge endpoints, first side (parallel to ``b``).
    :param b: Edge endpoints, second side (parallel to ``a``).
    :returns: Boolean mask, ``True`` for every node in the 2-core.
    """
    deg = np.zeros(V, dtype=int)
    np.add.at(deg, a, 1)
    np.add.at(deg, b, 1)
    adj = [[] for _ in range(V)]
    for u, v in zip(a.tolist(), b.tolist()):
        adj[u].append(v)
        adj[v].append(u)
    alive = np.ones(V, dtype=bool)
    q = deque(np.where(deg <= 1)[0].tolist())
    while q:
        u = q.popleft()
        if not alive[u]:
            continue
        alive[u] = False
        for w in adj[u]:
            if alive[w]:
                deg[w] -= 1
                if deg[w] == 1:
                    q.append(w)
    return alive



def check_skeleton_quality(
    df_skeleton_2d: pd.DataFrame,
    pixel_size: float,
    stent_df: pd.DataFrame,
    r_mid: float,
    region_allowed: np.ndarray,
    strut_thickness: float | None = None,
    loop_size_factor: float = 2.0,
    surf_tree: cKDTree | None = None,
    surf_reg: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """
    Score a 2D skeleton for three defect types and report where they are.

    Every skeleton point is assigned to its nearest surface region, and the
    skeleton's pixel-grid graph is rebuilt from ``df_skeleton_2d`` to check:
    (1) **bad connections** — an edge joins two regions that are not actually
    adjacent in 3D (per ``region_allowed``), meaning the skeleton bridged two
    unrelated struts; (2) **loops** — a small closed cycle inside a region or
    straddling a border, found via the graph's 2-core, and only flagged if
    its bounding diagonal is under ``loop_size_factor * strut_thickness`` (a
    larger loop is a real design cell, not a defect); (3) **empty regions** —
    a region with no skeleton point at all. This is the scoring function
    :func:`tune_skeleton_params` minimises over.

    :param df_skeleton_2d: Skeleton points with ``arc`` and ``z`` columns,
        from :func:`compute_skeleton_2d`.
    :param pixel_size: Pixel size the skeleton was rasterised at, used to
        recover its integer grid coordinates.
    :param stent_df: Ring's surface point cloud with a ``region`` column.
    :param r_mid: Mid-wall radius, used to unroll ``theta`` to arc length.
    :param region_allowed: Region-adjacency matrix from
        :func:`~stentfit.stent_rings.segment_stent`.
    :param strut_thickness: Strut thickness, used to size the small-loop
        cutoff. ``None`` treats every loop as small (no size filtering).
    :param loop_size_factor: A loop wider than this many strut thicknesses is
        a design cell, not a defect.
    :param surf_tree: Prebuilt KD-tree of the surface points, to skip
        rebuilding it across repeated calls (e.g. inside the tuning loop).
        ``None`` builds it from ``stent_df``.
    :param surf_reg: Region labels matching ``surf_tree``, required together with it.
    :param verbose: Print a pass/fail summary for each defect type.
    :returns: Dict with the region and skeleton-point counts (``n_regions``,
        ``n_skel_points``), the defects found (``bad_connections``,
        ``region_loops``, ``border_loops``, ``empty_regions``), the
        per-point region labels (``skel_region``), and (arc, z) markers for
        plotting each defect (``bad_edge_xy``, ``loop_points_xy``, ``empty_xy``).
    """
    skel = df_skeleton_2d[['arc', 'z']].to_numpy()
    n_sk = len(skel)
    n_regions = int(stent_df['region'].max())
    all_regions = np.arange(1, n_regions + 1)

    # Step 1: assign each skeleton point to the nearest surface region
    if surf_tree is None or surf_reg is None:
        surf_arc  = r_mid * stent_df['theta'].to_numpy()
        surf_z    = stent_df['z'].to_numpy()
        surf_reg  = stent_df['region'].to_numpy()
        surf_tree = cKDTree(np.column_stack([surf_arc, surf_z]))
    _, nn       = surf_tree.query(skel)
    skel_region = surf_reg[nn]

    # Step 2: recover integer pixel-grid coords
    gi = np.round((skel[:, 0] - skel[:, 0].min()) / pixel_size).astype(int)  # col (arc)
    gj = np.round((skel[:, 1] - skel[:, 1].min()) / pixel_size).astype(int)  # row (z)
    coord_set = set(zip(gi.tolist(), gj.tolist()))
    coord_idx = {(int(gi[k]), int(gj[k])): k for k in range(n_sk)}

    # Step 3: build the 8-neighbour skeleton graph (drop diagonals at 4-connected corners)
    edges = []
    for k in range(n_sk):
        ci, cj = int(gi[k]), int(gj[k])
        for di, dj in ((1, 0), (0, 1)):
            nb = coord_idx.get((ci + di, cj + dj))
            if nb is not None:
                edges.append((k, nb))
        for di, dj in ((1, 1), (1, -1)):
            nb = coord_idx.get((ci + di, cj + dj))
            if nb is not None:
                corner = ((ci + di, cj) in coord_set) or ((ci, cj + dj) in coord_set)
                if not corner:
                    edges.append((k, nb))
    edges = np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)

    # Step 4: connections between non-touching regions
    bad_connections = []
    bad_edge_xy     = np.empty((0, 2))
    if len(edges):
        ra, rb = skel_region[edges[:, 0]], skel_region[edges[:, 1]]
        diff   = ra != rb
        bad    = diff & ~region_allowed[ra, rb]
        if bad.any():
            pairs = np.unique(np.sort(np.column_stack([ra[bad], rb[bad]]), axis=1), axis=0)
            bad_connections = [tuple(int(x) for x in p) for p in pairs]
            bad_edge_xy = 0.5 * (skel[edges[bad, 0]] + skel[edges[bad, 1]])   # (arc, z) midpoints

    # Step 5: small loops, inside a region or on a border (design cells are large)
    loop_max_diag = (loop_size_factor * strut_thickness
                     if strut_thickness is not None else np.inf)
    region_loops = {}      # {region_id: small-loop count}
    border_loops = {}      # {(A, B): count}
    loop_points  = []      # (arc, z) markers for every flagged small loop

    def _loop_components(node_mask):
        node_idx = np.where(node_mask)[0]
        V = len(node_idx)
        if V == 0 or not len(edges):
            return []
        e_mask = node_mask[edges[:, 0]] & node_mask[edges[:, 1]]
        sub    = edges[e_mask]
        E = len(sub)
        if not E:
            return []
        remap = {int(g): i for i, g in enumerate(node_idx)}
        a = np.fromiter((remap[int(x)] for x in sub[:, 0]), dtype=int, count=E)
        b = np.fromiter((remap[int(x)] for x in sub[:, 1]), dtype=int, count=E)
        core = _two_core_mask(V, a, b)
        if not core.any():
            return []
        # connected components of the 2-core = the separate loops
        ce = core[a] & core[b]
        g  = csr_matrix((np.ones(int(ce.sum())), (a[ce], b[ce])), shape=(V, V))
        g  = g + g.T
        ncomp, lab = connected_components(g, directed=False)
        comps = []
        for c in range(ncomp):
            sel = (lab == c) & core
            if sel.sum() < 3:                 # need >=3 nodes to enclose anything
                continue
            gidx = node_idx[sel]
            xy   = skel[gidx]
            comps.append((set(int(r) for r in np.unique(skel_region[gidx])), xy))
        return comps

    def _is_small(xy):
        return float(np.hypot(*(xy.max(0) - xy.min(0)))) <= loop_max_diag

    # Cycles in the per-region skeleton graph.
    for reg in np.unique(skel_region):
        for _, xy in _loop_components(skel_region == reg):
            if _is_small(xy):
                region_loops[int(reg)] = region_loops.get(int(reg), 0) + 1
                loop_points.append(xy)

    # border loops: a seam loop is split across both per-region passes, so recheck
    # each touching pair on the union of the two regions
    if len(edges):
        ra, rb = skel_region[edges[:, 0]], skel_region[edges[:, 1]]
        seam   = (ra != rb) & region_allowed[ra, rb]      # touching neighbours only
        if seam.any():
            pairs = np.unique(np.sort(np.column_stack([ra[seam], rb[seam]]), axis=1), axis=0)
            for A, B in pairs:
                A, B = int(A), int(B)
                for regs, xy in _loop_components((skel_region == A) | (skel_region == B)):
                    if regs == {A, B} and _is_small(xy):   # straddles the seam, and small
                        border_loops[(A, B)] = border_loops.get((A, B), 0) + 1
                        loop_points.append(xy)

    loop_points_xy = np.vstack(loop_points) if loop_points else np.empty((0, 2))

    # Step 6: empty regions (no skeleton point assigned)
    present       = np.unique(skel_region)
    empty_regions = [int(r) for r in np.setdiff1d(all_regions, present)]
    empty_xy      = np.array([[ (r_mid * stent_df.loc[stent_df['region'] == er, 'theta']).mean(),
                                 stent_df.loc[stent_df['region'] == er, 'z'].mean() ]
                              for er in empty_regions]) if empty_regions else np.empty((0, 2))

    result = {
        'n_regions'      : n_regions,
        'n_skel_points'  : n_sk,
        'bad_connections': bad_connections,
        'region_loops'   : region_loops,
        'border_loops'   : border_loops,
        'empty_regions'  : empty_regions,
        'skel_region'    : skel_region,
        'bad_edge_xy'    : bad_edge_xy,      # (K, 2) (arc, z) midpoints of bad edges
        'loop_points_xy' : loop_points_xy,   # (K, 2) (arc, z) loop-structure points
        'empty_xy'       : empty_xy,         # (K, 2) (arc, z) centroids of empty regions
    }

    if verbose:
        print(f"Skeleton quality report - {n_regions} regions, {n_sk:,} skeleton points")
        print("-" * 60)
        if bad_connections:
            print(f"[FAIL] {len(bad_connections)} connection(s) between non-touching regions:")
            for a, b in bad_connections:
                print(f"        region {a} <-> region {b}")
        else:
            print("[ OK ] no connections between non-touching regions")

        if region_loops:
            print(f"[FAIL] {len(region_loops)} region(s) contain a loop:")
            for reg, n in sorted(region_loops.items()):
                print(f"        region {reg}: {n} loop(s)")
        else:
            print("[ OK ] no loops inside any region")

        if border_loops:
            print(f"[FAIL] {len(border_loops)} loop(s) on the border between touching regions:")
            for (a, b), n in sorted(border_loops.items()):
                print(f"        region {a} <-> region {b}: {n} loop(s)")
        else:
            print("[ OK ] no loops on region borders")

        if empty_regions:
            print(f"[FAIL] {len(empty_regions)} empty region(s) (no skeleton point): {empty_regions}")
        else:
            print("[ OK ] every region has at least one skeleton point")

    return result



def tune_skeleton_params(
    arc: np.ndarray,
    z: np.ndarray,
    stent_df: pd.DataFrame,
    stent_features: dict,
    region_allowed: np.ndarray,
    pps0: float = 10.0,           # starting pixels_per_strut (continuous)
    dil0: int = 3,                # starting dilate_px (integer disk radius)
    pps_min: float = 5.0,
    pps_max: float = 120.0,        # hard cap requested (keeps runtime bounded)
    dil_min: int = 1,
    dil_max: int = 30,            # cap on disk radius -> bounds per-step cost
    s_pps_conn: float = 25.0,     # pps push from bad connections
    s_pps_empty: float = 20.0,    # pps push from empty regions (size-weighted)
    s_pps_loop: float = 15.0,     # pps pull-down for loops when dilate_px is capped
    s_pps_explore: float = 5.0,   # when defect-free, step pps up by this to shrink the quality residual
    s_dil_loop: float = 20.0,     # dilate_px push from loops (thicken to fill holes)
    s_dil_conn: float = 8.0,      # dilate_px pull-down from connections (thin)
    w_conn: float = 2.0,
    w_loop: float = 10.0,
    w_empty: float = 1.0,
    loop_size_factor: float = 2.0,  # a loop wider than this x strut_thickness is a design cell, not a defect
    q_eps: float = 1e-3,          # floor on the quality residual so total_error stays > 0
    quality_gamma: float = 2.0,   # quality_error convexity: 1 = linear, >1 = diminishing returns
    res_no_improve_max: int = 5,  # steps (after the first clean step) with no total_error improvement
    target_penalty: float = 1.0,  # defect_error <= this counts as "feasible" (clean) AND gates best selection
    max_repeats: int = 2,        # stop once the SAME (pps, dil) state has been visited this many times
    pad_fraction: float = 0.20,
    time_limit: float = 100.0,
    predictive_stop: bool = True,
    max_iters: int = int(1e3),
    verbose: bool = True,
) -> dict:
    """
    Search for the ``pixels_per_strut`` / ``dilate_px`` pair that gives the
    cleanest 2D skeleton in the least time.

    Each step runs :func:`compute_skeleton_2d` at the current (pps, dil) and
    scores it with :func:`check_skeleton_quality`. The score is a
    ``defect_error`` (bad connections, loops, empty regions — zero for a
    clean skeleton) plus a ``quality_error`` residual that only shrinks as
    pps/dil gets finer. While defects remain, pps and dil are nudged in
    proportion to which defect dominates (sharpen for bad connections or
    empty regions, thicken for loops, thin back down once dilation hits its
    cap). Once clean, pps is raised step by step to shrink the residual. The
    search stops on the time limit, on no improvement for
    ``res_no_improve_max`` steps after going clean, or if the same (pps, dil)
    state repeats ``max_repeats`` times.

    :param arc: Flat arc-coordinates of the ring's surface points.
    :param z: Flat z-coordinates of the ring's surface points.
    :param stent_df: Ring's surface point cloud with a ``region`` column.
    :param stent_features: Stent features dict (``r_mid``, ``strut_thickness``).
    :param region_allowed: Region-adjacency matrix from
        :func:`~stentfit.stent_rings.segment_stent`.
    :param pps0: Starting ``pixels_per_strut``.
    :param dil0: Starting ``dilate_px``.
    :param pps_min: Lower bound on ``pixels_per_strut``.
    :param pps_max: Upper bound on ``pixels_per_strut``, keeps runtime bounded.
    :param dil_min: Lower bound on ``dilate_px``.
    :param dil_max: Upper bound on ``dilate_px``.
    :param s_pps_conn: How much ``pps`` rises per unit of bad-connection error.
    :param s_pps_empty: How much ``pps`` rises per unit of empty-region error.
    :param s_pps_loop: How much ``pps`` falls per unit of loop error, once
        ``dilate_px`` is already capped.
    :param s_pps_explore: Step size ``pps`` is raised by once the skeleton is clean.
    :param s_dil_loop: How much ``dilate_px`` rises per unit of loop error.
    :param s_dil_conn: How much ``dilate_px`` falls per unit of bad-connection error.
    :param w_conn: Weight of bad connections in ``defect_error``.
    :param w_loop: Weight of loops in ``defect_error``.
    :param w_empty: Weight of empty regions in ``defect_error``.
    :param loop_size_factor: A loop wider than this many strut thicknesses is
        a real design cell, not a defect.
    :param q_eps: Floor on ``quality_error`` so ``total_error`` never hits zero.
    :param quality_gamma: Convexity of ``quality_error``; 1 is linear, higher
        gives diminishing returns as it approaches zero.
    :param res_no_improve_max: Steps without improvement, after first going
        clean, before the search stops.
    :param target_penalty: ``defect_error`` at or below this counts as clean.
    :param max_repeats: Times the same (pps, dil) state may repeat before the
        search stops as a cycle.
    :param pad_fraction: Seam padding passed through to :func:`compute_skeleton_2d`.
    :param time_limit: Time budget, in seconds, for the whole search.
    :param predictive_stop: Stop early if the next step is projected to blow
        the time budget, instead of only checking after it runs.
    :param max_iters: Hard cap on the number of steps, regardless of time.
    :param verbose: Print the per-step error table.
    :returns: Dict with the best ``pps``/``dilate_px`` found (``best_pps``,
        ``best_dilate_px``), their errors (``best_defect_error``,
        ``best_quality_error``, ``best_total_error``), the full step history
        (``history``), and the winning ``skeleton_2d`` / ``quality_report``.
    """
    r_mid = stent_features['r_mid']
    circ  = 2 * np.pi * r_mid

    # Prebuild the (arc, z) surface tree ONCE (reused every iteration).
    surf_tree = cKDTree(np.column_stack([r_mid * stent_df['theta'].to_numpy(),
                                         stent_df['z'].to_numpy()]))
    surf_reg     = stent_df['region'].to_numpy()
    n_regions    = int(stent_df['region'].max())
    region_sizes = stent_df['region'].value_counts()
    med_size     = float(np.median(region_sizes.to_numpy()))

    def _errors(qr):
        n_conn = len(qr['bad_connections'])
        n_loop = int(sum(qr['region_loops'].values())) \
               + int(sum(qr.get('border_loops', {}).values()))
        if qr['empty_regions']:
            e_empty = sum(float(region_sizes.get(er, 0)) for er in qr['empty_regions']) / med_size
        else:
            e_empty = 0.0
        return n_conn, n_loop, e_empty

    pps = float(pps0)
    # dilate_px can never exceed the strut spacing (dil <= pps).
    dil = int(np.clip(round(dil0), dil_min, min(dil_max, int(pps))))
    best, history = None, []
    seen = Counter()              # (round(pps), dil) -> times visited (cycle / pinned detector)
    feasible_reached = False      # latches True at the first defect_error<=target step
    no_improve = 0                # steps (post-feasibility) since the incumbent improved
    t0 = time.time()
    durs = []                     # per-iteration wall times -> predict the next step's cost

    # Constant span used to normalise pps/dil into the quality residual [0, 100].
    ratio_span = (pps_max / dil_min) - (pps_min / dil_max)

    if verbose:
        print(f"{'step':>4} {'pps':>7} {'dil_px':>6} | {'conn':>5} {'loop':>5} {'empty':>6} "
              f"| {'defect':>8} {'qual':>6} {'total':>8} | {'t(s)':>5}")
        print("-" * 82)

    for it in range(max_iters):
        elapsed = time.time() - t0
        # Hard stop: always active.
        if elapsed > time_limit:
            print(f"[tune] time limit ({time_limit:.0f}s) reached - stopping")
            break
        # predictive stop: project the next step's cost, stop if it won't fit
        if predictive_stop and durs:
            growth = durs[-1] / durs[-2] if len(durs) >= 2 and durs[-2] > 0 else 1.0
            proj   = durs[-1] * max(1.0, growth)
            if elapsed + proj > time_limit:
                print(f"[tune] next step projected ~{proj:.0f}s - would exceed the "
                      f"{time_limit:.0f}s budget at t={elapsed:.0f}s - stopping")
                break

        _it_t0 = time.time()
        res = compute_skeleton_2d(
            arc_flat=arc, z_flat=z, arc_min_flat=arc.min(),
            circumference=circ, stent_df=stent_df, stent_geometry=stent_features,
            pixels_per_strut=pps, dilate_px=dil, pad_fraction=pad_fraction,
        )
        qr = check_skeleton_quality(
            res['df_skeleton_2d'], res['pixel_size'], stent_df, r_mid, region_allowed,
            strut_thickness=stent_features['strut_thickness'],
            loop_size_factor=loop_size_factor,
            surf_tree=surf_tree, surf_reg=surf_reg, verbose=False,
        )
        durs.append(time.time() - _it_t0)
        n_conn, n_loop, e_empty = _errors(qr)
        # total error: defect_error -> 0 for a clean skeleton, quality_error is a
        # strictly-positive resolution residual (finer pps/dil shrinks it)
        defect_error  = w_conn * n_conn + w_loop * n_loop + w_empty * e_empty
        fineness      = float(np.clip((pps/dil - pps_min/dil_max) / ratio_span, 0.0, 1.0))
        quality_error = 100.0 * float(np.clip((1.0 - fineness) ** quality_gamma, q_eps, 1.0))
        total_error   = defect_error + quality_error
        elapsed = time.time() - t0
        history.append(dict(step=it, pps=round(pps, 2), dil_px=dil, conn=n_conn,
                            loop=n_loop, empty=round(e_empty, 2),
                            defect_error=round(defect_error, 3),
                            quality_error=round(quality_error, 4),
                            total_error=round(total_error, 3), t=round(elapsed, 1)))
        if verbose:
            print(f"{it:>4} {pps:>7.2f} {dil:>6d} | {n_conn:>5d} {n_loop:>5d} "
                  f"{e_empty:>6.2f} | {defect_error:>8.3f} {quality_error:>6.3f} "
                  f"{total_error:>8.3f} | {elapsed:>5.1f}")

        # selection: lexicographic argmin over (dirty_flag, total_error), so a
        # clean skeleton always beats a defective one
        clean    = defect_error <= target_penalty
        cand_key = (0 if clean else 1, total_error)
        improved = best is None or cand_key < best['key']
        if improved:
            best = dict(key=cand_key, total_error=total_error, defect_error=defect_error,
                        quality_error=quality_error, pps=pps, dil=dil,
                        conn=n_conn, loop=n_loop, empty=e_empty,
                        result=res, quality_report=qr)

        # convergence stop: once clean, count steps with no incumbent improvement
        if defect_error <= target_penalty:
            feasible_reached = True
        if feasible_reached:
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= res_no_improve_max:
                    print(f"[tune] total_error not improved for {no_improve} steps since "
                          f"reaching a clean skeleton - stopping (converged)")
                    break

        # cycle / pinned detector: a repeated (pps, dil) state means we are looping
        key = (round(pps, 2), dil)
        seen[key] += 1
        if seen[key] >= max_repeats:
            print(f"[tune] same parameters (pps={pps:.2f}, dil_px={dil}) seen "
                  f"{max_repeats}x - stopping (cycle / pinned)")
            break

        # parameter update
        if defect_error <= target_penalty:
            # feasible: raise pps while holding dil, so pps/dil rises and the residual shrinks
            pps_new = pps + s_pps_explore
            dil_new = dil
        else:
            # infeasible: error-proportional correction (errors normalised to [0,1])
            e_conn_n  = min(1.0, n_conn  / max(1, n_regions))
            e_loop_n  = min(1.0, n_loop  / max(1, n_regions))
            e_empty_n = min(1.0, e_empty / max(1, n_regions))
            dil_cap   = min(dil_max, int(pps))            # hard cap AND strut spacing
            pps_new = pps + s_pps_conn * e_conn_n + s_pps_empty * e_empty_n      # sharpen
            # thicken to close loops, thin to break connections (min move 1 so it never stalls)
            d_loop = max(1, int(np.ceil(s_dil_loop * e_loop_n))) if n_loop > 0 else 0
            d_conn = max(1, int(np.ceil(s_dil_conn * e_conn_n))) if n_conn > 0 else 0
            dil_new = dil + d_loop - d_conn
            if n_loop > 0 and dil >= dil_cap:     # dilation capped but holes remain -> coarsen pps
                pps_new -= s_pps_loop * e_loop_n

        pps = float(np.clip(pps_new, pps_min, pps_max))
        # Re-clamp dil against the (possibly changed) strut spacing: dil <= pps.
        dil = int(np.clip(dil_new, dil_min, min(dil_max, int(pps))))

    hist_df = pd.DataFrame(history)
    print("\n[tune] BEST  pps={:.2f}  dilate_px={}  total_error={:.3f}  "
          "(defect={:.3f}, quality={:.4f}; conn={}, loop={}, empty={:.2f})".format(
              best['pps'], best['dil'], best['total_error'],
              best['defect_error'], best['quality_error'],
              int(best['conn']), int(best['loop']), best['empty']))

    return {
        'best_pps'          : best['pps'],
        'best_dilate_px'    : best['dil'],
        'best_defect_error' : best['defect_error'],
        'best_quality_error': best['quality_error'],
        'best_total_error'  : best['total_error'],
        'history'           : hist_df,
        'skeleton_2d'       : best['result'],
        'quality_report'    : best['quality_report'],
    }



def _grid_adjacency(arc: np.ndarray,
                    z: np.ndarray,
                    pixel_size: float) -> tuple[np.ndarray, list[list[int]]]:
    """
    Rebuild the 8-neighbour pixel-grid graph for a set of 2D skeleton points.

    Recovers each point's integer grid coordinates from its (arc, z)
    position and connects it to the neighbours sharing an edge or a
    non-corner-cutting diagonal — the same connectivity rule used in
    :func:`check_skeleton_quality`. Used by the manual/automatic bad-edge
    fixers (:func:`fix_ring_loop_2d`, :func:`fix_ring_connection_2d`,
    :func:`auto_clean_bad_connections_2d`) to walk the skeleton as a graph.

    :param arc: Flat arc-coordinates of the ring's 2D skeleton.
    :param z: Flat z-coordinates of the ring's 2D skeleton.
    :param pixel_size: Pixel size the skeleton was rasterised at.
    :returns: ``(edges, adj)`` — an ``(E, 2)`` array of point-index pairs,
        and a per-point adjacency list of neighbour indices.
    """
    arc = np.asarray(arc, float)
    z   = np.asarray(z, float)
    n   = len(arc)
    if n == 0:
        return np.empty((0, 2), int), []
    gi = np.round((arc - arc.min()) / pixel_size).astype(int)
    gj = np.round((z   - z.min())   / pixel_size).astype(int)
    coord_set = set(zip(gi.tolist(), gj.tolist()))
    coord_idx = {(int(gi[k]), int(gj[k])): k for k in range(n)}
    edges = []
    for k in range(n):
        ci, cj = int(gi[k]), int(gj[k])
        for di, dj in ((1, 0), (0, 1)):
            nb = coord_idx.get((ci + di, cj + dj))
            if nb is not None:
                edges.append((k, nb))
        for di, dj in ((1, 1), (1, -1)):
            nb = coord_idx.get((ci + di, cj + dj))
            if nb is not None:
                corner = ((ci + di, cj) in coord_set) or ((ci, cj + dj) in coord_set)
                if not corner:
                    edges.append((k, nb))
    edges = np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return edges, adj



def _interp_2d(a: np.ndarray, b: np.ndarray, spacing: float) -> np.ndarray:
    """
    Generate evenly-spaced points strictly between two 2D points.

    Used by :func:`fix_ring_loop_2d` to fill a collapsed loop back in as a
    single straight line between its two kept anchor points, at roughly the
    skeleton's own pixel spacing.

    :param a: Start point, as ``(arc, z)``.
    :param b: End point, as ``(arc, z)``.
    :param spacing: Target distance between consecutive inserted points.
    :returns: ``(n, 2)`` array of points strictly between ``a`` and ``b``
        (excluding both endpoints); empty if they're already closer than
        ``spacing``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    dist = float(np.hypot(*(b - a)))
    n_mid = max(int(round(dist / spacing)) - 1, 0) if spacing > 0 else 0
    if n_mid == 0:
        return np.empty((0, 2), float)
    ts = (np.arange(1, n_mid + 1) / (n_mid + 1))[:, None]
    return a + ts * (b - a)



def fix_ring_loop_2d(arc: np.ndarray,
                     z: np.ndarray,
                     pixel_size: float,
                     anchor_a: int,
                     anchor_b: int,
                     verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse a loop (bubble) in the 2D skeleton into a single straight path.

    ``anchor_a`` and ``anchor_b`` must both sit on the same loop (found via
    the grid graph's 2-core, same as :func:`check_skeleton_quality`'s loop
    check). Every other point on that loop is deleted, and an evenly-spaced
    straight line is inserted between the two anchors instead. Everything
    outside the loop is left untouched. If the anchors aren't both on the
    same loop, nothing changes.

    :param arc: Flat arc-coordinates of the ring's 2D skeleton.
    :param z: Flat z-coordinates of the ring's 2D skeleton.
    :param pixel_size: Pixel size the skeleton was rasterised at, used to
        rebuild its grid adjacency and space the inserted points.
    :param anchor_a: Point index at one end of the loop to keep.
    :param anchor_b: Point index at the other end of the loop to keep.
    :param verbose: Print what was changed, or why nothing was.
    :returns: ``(arc, z, changed_idx)`` — the edited coordinates and the
        indices of the newly inserted points. Unchanged, with an empty
        ``changed_idx``, if the anchors aren't both on the same loop.
    """
    arc = np.asarray(arc, float)
    z   = np.asarray(z, float)
    n   = len(arc)
    a, b = int(anchor_a), int(anchor_b)

    edges, _ = _grid_adjacency(arc, z, pixel_size)
    core = (_two_core_mask(n, edges[:, 0], edges[:, 1]) if len(edges)
            else np.zeros(n, bool))
    if not (core[a] and core[b]):
        if verbose:
            print(f"[fix loop 2d] anchors {a}, {b} are not both on a loop "
                  f"(2-core) — no change")
        return arc, z, np.array([], int)

    e_core = edges[core[edges[:, 0]] & core[edges[:, 1]]]
    g = csr_matrix((np.ones(len(e_core)), (e_core[:, 0], e_core[:, 1])), shape=(n, n))
    g = g + g.T
    _, lab = connected_components(g, directed=False)
    if lab[a] != lab[b]:
        if verbose:
            print(f"[fix loop 2d] anchors {a}, {b} are on different loops — no change")
        return arc, z, np.array([], int)

    comp = lab == lab[a]                 # the loop's nodes (all in the 2-core)
    delete = comp.copy()
    delete[a] = False
    delete[b] = False

    keep = ~delete
    new_arc = arc[keep]
    new_z   = z[keep]
    mids = _interp_2d((arc[a], z[a]), (arc[b], z[b]), pixel_size)
    if len(mids):
        new_arc = np.concatenate([new_arc, mids[:, 0]])
        new_z   = np.concatenate([new_z,   mids[:, 1]])
    changed_idx = (np.arange(len(new_arc) - len(mids), len(new_arc))
                   if len(mids) else np.array([], int))
    if verbose:
        print(f"[fix loop 2d] removed {int(delete.sum())} loop point(s), inserted "
              f"{len(mids)} between anchors {a} and {b}")
    return new_arc, new_z, changed_idx



def fix_ring_connection_2d(arc: np.ndarray,
                           z: np.ndarray,
                           pixel_size: float,
                           point_a: int,
                           point_b: int,
                           verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Delete a whole wrong bridge between two points on the 2D skeleton.

    Walks the shortest grid-graph path between ``point_a`` and ``point_b``,
    then extends outward from each end along the thin (degree-2) chain until
    it hits a junction or endpoint (degree != 2) — that junction is kept as
    the boundary, everything strictly inside is removed. ``point_a`` and
    ``point_b`` should sit on the bridge itself (e.g. its two ends), not on
    the junctions bounding it. If there's no path between them, or nothing
    thin to remove, nothing changes.

    :param arc: Flat arc-coordinates of the ring's 2D skeleton.
    :param z: Flat z-coordinates of the ring's 2D skeleton.
    :param pixel_size: Pixel size the skeleton was rasterised at, used to
        rebuild its grid adjacency.
    :param point_a: Point index on one end of the wrong bridge.
    :param point_b: Point index on the other end of the wrong bridge.
    :param verbose: Print what was removed, or why nothing was.
    :returns: ``(arc, z, changed_idx)`` — the edited coordinates and the
        indices of any newly inserted points (always empty here — this fix
        only deletes). Unchanged if there was no path or nothing removable.
    """
    arc = np.asarray(arc, float)
    z   = np.asarray(z, float)
    n   = len(arc)
    a, b = int(point_a), int(point_b)

    _, adj = _grid_adjacency(arc, z, pixel_size)
    deg = [len(adj[i]) for i in range(n)]

    # shortest path a -> b
    prev = {a: -1}
    dq = deque([a])
    while dq:
        u = dq.popleft()
        if u == b:
            break
        for w in adj[u]:
            if w not in prev:
                prev[w] = u
                dq.append(w)
    if b not in prev:
        if verbose:
            print(f"[fix connection 2d] no path between {a} and {b} — no change")
        return arc, z, np.array([], int)
    path = []
    cur = b
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path = path[::-1]

    # the bridge = the maximal thin (degree-2) chain covering the path; keep the
    # bounding junctions (degree != 2) that attach it to the struts
    remove = set(p for p in path if deg[p] == 2)

    def _walk_out(start, inside, max_steps=400):
        prev_n, curn = inside, start
        for _ in range(max_steps):
            nbrs = [w for w in adj[curn] if w != prev_n]
            if not nbrs:
                return
            prev_n, curn = curn, nbrs[0]
            if deg[curn] != 2:          # junction/endpoint -> boundary, keep it
                return
            remove.add(curn)

    if deg[a] == 2:
        _walk_out(a, path[1] if len(path) > 1 else -1)
    if deg[b] == 2:
        _walk_out(b, path[-2] if len(path) > 1 else -1)

    if not remove:
        if verbose:
            print(f"[fix connection 2d] nothing to remove between {a} and {b} "
                  f"(both look like junctions) — pick points on the bridge itself; "
                  f"no change")
        return arc, z, np.array([], int)

    keep = np.ones(n, bool)
    keep[list(remove)] = False
    if verbose:
        print(f"[fix connection 2d] removed the {len(remove)}-point bridge between "
              f"{a} and {b} (up to the bounding junctions) to open a clean gap")
    return arc[keep], z[keep], np.array([], int)



def auto_clean_bad_connections_2d(arc: np.ndarray,
                                  z: np.ndarray,
                                  pixel_size: float,
                                  bad_edge_xy: np.ndarray,
                                  verbose: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Remove the skeleton bridges flagged as bad connections by
    :func:`check_skeleton_quality`.

    Each entry in ``bad_edge_xy`` is the (arc, z) midpoint of one flagged
    edge. For each, the nearest skeleton point is used as a seed, and the
    whole degree-2 chain it belongs to (the thin bridge between two
    junctions) is removed. A seed that sits on a junction (degree != 2) has
    no removable chain and is left alone, for the manual-edit step to fix by
    hand.

    :param arc: Flat arc-coordinates of the ring's 2D skeleton.
    :param z: Flat z-coordinates of the ring's 2D skeleton.
    :param pixel_size: Pixel size the skeleton was rasterised at, used to
        rebuild its grid adjacency.
    :param bad_edge_xy: (K, 2) array of (arc, z) midpoints, from
        ``quality_report['bad_edge_xy']``.
    :param verbose: Print how many bridges (and points) were removed.
    :returns: ``(arc, z, n_bridges)`` — the cleaned coordinates and the
        number of bridges removed. Unchanged, with ``n_bridges=0``, if
        nothing was removable.
    """
    arc = np.asarray(arc, float)
    z   = np.asarray(z, float)
    n   = len(arc)
    bad = np.asarray(bad_edge_xy, float).reshape(-1, 2)
    if not len(bad) or n == 0:
        return arc, z, 0

    _, adj = _grid_adjacency(arc, z, pixel_size)
    deg    = [len(adj[i]) for i in range(n)]
    seeds  = cKDTree(np.column_stack([arc, z])).query(bad)[1]

    def _deg2_component(start):
        if deg[start] != 2:
            comp = set()
            for nb in adj[start]:
                if deg[nb] == 2:
                    comp |= _deg2_component(nb)
            return comp
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if deg[w] == 2 and w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    remove, n_bridges = set(), 0
    for s in np.unique(seeds):
        s = int(s)
        if s in remove:
            continue
        comp = _deg2_component(s)
        if comp:
            remove |= comp
            n_bridges += 1

    if not remove:
        if verbose:
            print(f"[auto-clean] {len(bad)} flagged bad connection(s), none removable "
                  f"(seeds sit on junctions) — left for the manual fix")
        return arc, z, 0

    keep = np.ones(n, bool)
    keep[list(remove)] = False
    if verbose:
        print(f"[auto-clean] removed {n_bridges} detected bad-connection bridge(s) "
              f"({len(remove)} points) automatically")
    return arc[keep], z[keep], n_bridges



def _downsample_surface_pair(a: np.ndarray,
                             b: np.ndarray,
                             n: int = 40000,
                             seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly subsample two matching coordinate arrays down to ``n`` points.

    Used to shrink a ring's surface points before storing them for the 2D
    skeleton plot, so :func:`~stentfit.stent_plotting.plot_ring_skeleton_2d_html`
    stays responsive.

    :param a: First coordinate array (e.g. arc).
    :param b: Second coordinate array (e.g. z), same length as ``a``.
    :param n: Maximum number of points to keep.
    :param seed: Seed for the row sampling, for repeatable plots.
    :returns: ``(a, b)`` unchanged if ``len(a) <= n``, otherwise the same
        random ``n`` indices taken from both.
    """
    a = np.asarray(a); b = np.asarray(b)
    if len(a) <= n:
        return a, b
    idx = np.random.default_rng(seed).choice(len(a), n, replace=False)
    return a[idx], b[idx]



def skeletonize_rings_2d(stent_df: pd.DataFrame,
                         stent_features: dict,
                         ring_edges: np.ndarray,
                         conn_radius_3d: float,
                         output_dir: str,
                         auto_tune: bool = True,
                         pixels_per_strut: int = 10,
                         dilate_px: int = 3,
                         pad_fraction: float = 0.20,
                         tune_time_limit: int = 120,
                         quality_gamma: float = 2.0,
                         ring_halo_frac: float = 0.4) -> dict:
    """
    2D-skeletonise every ring of the stent, one ring at a time.

    Runs :func:`~stentfit.stent_rings.segment_stent` once on the whole stent
    to get a shared region map, then for each ring: unrolls its points (plus
    a z-halo, so struts still reconnect across ring boundaries) to a flat
    (arc, z) plane with :func:`open_stent_to_plane`, skeletonises it (either a
    single pass with :func:`compute_skeleton_2d`, or an auto-tuned search over
    ``pixels_per_strut`` / ``dilate_px`` via :func:`tune_skeleton_params`), then
    trims the result back down to the ring's own z-band. Detected bad
    connections are auto-cleaned with :func:`auto_clean_bad_connections_2d`.
    Writes per-ring plots and a CSV under ``output_dir/skeleton_plots/``.

    :param stent_df: Stent point cloud with a ``ring_id`` column (from
        :func:`~stentfit.stent_rings.detect_rings`).
    :param stent_features: Stent features dict (``r_mid``, ``strut_thickness``, ...).
    :param ring_edges: Z-boundaries between rings, used to size each ring's halo.
    :param conn_radius_3d: 3D connectivity radius used for the region segmentation.
    :param output_dir: Folder the per-ring plots and CSVs are written into
        (under a ``skeleton_plots`` subfolder).
    :param auto_tune: Search for the best skeletonisation parameters per ring,
        instead of using ``pixels_per_strut`` / ``dilate_px`` directly.
    :param pixels_per_strut: Raster resolution, in pixels across one strut width.
    :param dilate_px: Dilation radius, in pixels, before thinning.
    :param pad_fraction: Seam padding added when unrolling a ring to 2D.
    :param tune_time_limit: Time budget, in seconds, for the auto-tune search.
    :param quality_gamma: Weight of the skeleton quality score during tuning.
    :param ring_halo_frac: Z-halo, as a fraction of ring height, so struts
        reconnect across neighbouring rings.
    :returns: Dict with the per-ring 2D skeletons (``ring_2d``, keyed by
        ``ring_XX`` label) and the ring order along the stent axis (``ring_order``).
    """
    r_mid           = stent_features['r_mid']
    strut_thickness = stent_features['strut_thickness']

    plots_dir = os.path.join(output_dir, 'skeleton_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Single segmentation for the whole stent (each ring -> 3 pieces) 
    segmented      = segment_stent(stent_df, strut_thickness, conn_radius_3d,
                                   n_sub_per_ring=3)
    seg_df_full    = segmented['stent_df']          # whole stent, now has 'region'
    region_allowed = segmented['region_allowed']    # global region adjacency

    ring_order = (seg_df_full.groupby('ring_id')['z_cylindrical'].mean()
                              .sort_values().index.tolist())
    use_halo = ring_edges is not None and len(ring_edges) >= 2
    z_all    = seg_df_full['z_cylindrical'].values

    ring_2d = {}          # ring_label -> per-ring 2D skeleton record
    print(f"Skeletonising {len(ring_order)} rings "
          f"({'auto-tune' if auto_tune else 'fixed params'})...")

    for ci, c in enumerate(ring_order):
        label = f"ring_{int(c):02d}"
        own_z = seg_df_full.loc[seg_df_full['ring_id'] == c, 'z_cylindrical']
        if use_halo:
            k = int(np.clip(np.digitize(own_z.mean(), ring_edges) - 1,
                            0, len(ring_edges) - 2))
            z_lo, z_hi = float(ring_edges[k]), float(ring_edges[k + 1])
            halo    = ring_halo_frac * (z_hi - z_lo)
            seg_sub = seg_df_full[(z_all >= z_lo - halo) & (z_all <= z_hi + halo)].copy()
        else:
            z_lo, z_hi = float(own_z.min()), float(own_z.max())
            seg_sub    = seg_df_full[seg_df_full['ring_id'] == c].copy()

        opened = open_stent_to_plane(seg_sub, r_mid)
        arc_flat, z_flat = opened['arc_flat'], opened['z_flat']

        print(f"\n===== {label} ({ci + 1}/{len(ring_order)}) — "
              f"{len(seg_sub):,} pts, band z=[{z_lo:.4f}, {z_hi:.4f}] =====")

        if auto_tune:
            tuned   = tune_skeleton_params(
                arc=arc_flat, z=z_flat, stent_df=seg_sub, stent_features=stent_features,
                region_allowed=region_allowed, pps0=pixels_per_strut, dil0=dilate_px,
                pad_fraction=pad_fraction,
                time_limit=tune_time_limit, quality_gamma=quality_gamma,
                verbose=True)
            sk_2d          = tuned['skeleton_2d']
            history        = tuned['history']
            quality_report = tuned['quality_report']
            diag_pps, diag_dil = tuned['best_pps'], tuned['best_dilate_px']
        else:
            sk_2d = compute_skeleton_2d(
                arc_flat=arc_flat, z_flat=z_flat, arc_min_flat=opened['arc_min_flat'],
                circumference=opened['circumference'],
                stent_df=seg_sub, stent_geometry=stent_features,
                pixels_per_strut=pixels_per_strut, dilate_px=dilate_px,
                pad_fraction=pad_fraction)
            history = None
            quality_report = check_skeleton_quality(
                df_skeleton_2d=sk_2d['df_skeleton_2d'], pixel_size=sk_2d['pixel_size'],
                stent_df=seg_sub, r_mid=r_mid, region_allowed=region_allowed,
                strut_thickness=strut_thickness, verbose=True)
            diag_pps, diag_dil = pixels_per_strut, dilate_px

        # trim the final skeleton to the ring's own z-band (drop most of the halo).
        # Unlike the seam, neighbouring rings are skeletonised on SEPARATE, unaligned
        # pixel grids, so the two halves of a boundary strut do not coincide. We keep a
        # small overlap band past each edge so both rings share points there and the
        # halves always reconnect in 3D. The overlap is sized in PIXELS (intrinsic
        # resolution), not a design fraction; the old 0.01*strut slack was sub-pixel and
        # therefore unreliable, which is why boundary struts were sometimes pruned.
        skel_df2d = sk_2d['df_skeleton_2d']
        arc_all   = skel_df2d['arc'].to_numpy()
        z_2d_all  = skel_df2d['z'].to_numpy()
        z_ov  = 2.0 * sk_2d['pixel_size']
        tmask = (z_2d_all >= z_lo - z_ov) & (z_2d_all <= z_hi + z_ov)
        arc_c = arc_all[tmask]
        z_c   = z_2d_all[tmask]

        # auto-clean the DETECTED bad connections (whole-bridge removal, same as the
        # manual fix), then re-check so the plot flags only errors the detector missed.
        bad_xy = np.asarray(quality_report.get('bad_edge_xy', np.empty((0, 2)))).reshape(-1, 2)
        bad_in = bad_xy[(bad_xy[:, 1] >= z_lo) & (bad_xy[:, 1] <= z_hi)] if len(bad_xy) else bad_xy
        if len(bad_in):
            arc_c, z_c, n_auto = auto_clean_bad_connections_2d(
                arc_c, z_c, sk_2d['pixel_size'], bad_in, verbose=True)
            if n_auto:
                quality_report = check_skeleton_quality(
                    df_skeleton_2d=pd.DataFrame({'arc': arc_c, 'z': z_c}),
                    pixel_size=sk_2d['pixel_size'], stent_df=seg_sub, r_mid=r_mid,
                    region_allowed=region_allowed, strut_thickness=strut_thickness,
                    verbose=False)

        # store this ring's 2D skeleton (editable in Step 5.5)
        surf_arc_ds, surf_z_ds = _downsample_surface_pair(arc_flat, z_flat)
        ring_2d[label] = {
            'ring_id'  : c,
            'arc'       : arc_c.copy(),
            'z'         : z_c.copy(),
            'pixel_size': sk_2d['pixel_size'],
            'z_lo'      : z_lo,
            'z_hi'      : z_hi,
            'surf_arc'  : surf_arc_ds,
            'surf_z'    : surf_z_ds,
            'n_edits'   : 0,
        }

        # The 2D skeleton alone (single panel, zoomable, remaining issues flagged)
        skel_html = os.path.join(plots_dir, f'{label}.html')
        plot_ring_skeleton_2d_html(
            arc_c, z_c, surf_arc_ds, surf_z_ds, skel_html, label,
            ring_band=(z_lo, z_hi), quality_report=quality_report)
        # The tuning convergence / issue-summary, in a separate file
        conv_html = os.path.join(plots_dir, f'{label}_convergence.html')
        plot_ring_convergence_html(history, conv_html, c, quality_report=quality_report,
                                    pps=diag_pps, dil_px=diag_dil)
        # The 2D CSV (i, arc, z) for the record
        pd.DataFrame({'i': np.arange(len(arc_c)), 'arc': arc_c, 'z': z_c}).to_csv(
            os.path.join(plots_dir, f'{label}_2d.csv'), index=False)
        print(f"  {label}: {len(arc_c):,} skeleton pts  ->  {skel_html}")

    n_total = sum(len(v['arc']) for v in ring_2d.values())
    print(f"\nStored {len(ring_2d)} rings ({n_total:,} 2D skeleton points) in RING_2D.")
    print(f"Per-ring plots (ring_XX.html + _convergence.html) + CSVs saved in {plots_dir}")
    print("Detected bad connections were auto-cleaned; review each ring_XX.html for any "
          "remaining errors and fix them by hand in the manual-edit step.")

    return {'ring_2d': ring_2d, 'ring_order': ring_order}



def save_ring_2d_checkpoint(ring_2d: dict,
                            stent_features: dict,
                            stent_centerline_direction: np.ndarray,
                            r_mid: float,
                            strut_thickness: float,
                            circumference: float,
                            ring_edges: np.ndarray,
                            output_dir: str,
                            verbose: bool = True) -> str:
    """
    Save the assembled per-ring 2D skeletons to disk as a resume checkpoint.

    Pickles ``ring_2d`` plus the scalars needed to rebuild it, into
    ``ring_2d.pkl``. Together with the already-saved ``ring_points.csv``,
    this lets :func:`load_ring_2d_checkpoint` restore the pipeline state
    after a kernel restart, without rerunning sampling, ring detection, or
    2D skeletonisation.

    :param ring_2d: Per-ring 2D skeletons, from :func:`skeletonize_rings_2d`.
    :param stent_features: Stent features dict.
    :param stent_centerline_direction: Stent centreline unit vector.
    :param r_mid: Mid-wall radius.
    :param strut_thickness: Strut thickness.
    :param circumference: Full circumference at ``r_mid``.
    :param ring_edges: Z-boundaries between rings.
    :param output_dir: Folder ``ring_2d.pkl`` is written into.
    :param verbose: Print the saved path and ring count.
    :returns: Path to the written ``ring_2d.pkl``.
    """
    ring2d_pkl = os.path.join(output_dir, 'ring_2d.pkl')
    with open(ring2d_pkl, 'wb') as f:
        pickle.dump({
            'RING_2D'                  : ring_2d,
            'stent_features'            : stent_features,
            'stent_centerline_direction': np.asarray(stent_centerline_direction),
            'r_mid'                     : float(r_mid),
            'strut_thickness'           : float(strut_thickness),
            'circumference'             : float(circumference),
            'ring_edges'               : ring_edges,
        }, f)
    if verbose:
        print(f"[checkpoint] saved {len(ring_2d)} per-ring 2D skeletons -> {ring2d_pkl}")
    return ring2d_pkl



def load_ring_2d_checkpoint(output_dir: str) -> dict:
    """
    Reload the per-ring 2D skeletons and surface point cloud from disk.

    Reads back ``ring_2d.pkl`` (from :func:`save_ring_2d_checkpoint`) and
    ``ring_points.csv``, so the pipeline can resume the manual-edit step
    after a kernel restart without rerunning sampling, ring detection, or 2D
    skeletonisation.

    :param output_dir: Folder containing ``ring_2d.pkl`` and ``ring_points.csv``.
    :raises FileNotFoundError: If either file is missing.
    :returns: State dict with the per-ring 2D skeletons (``ring_2d``), the
        stent features and geometry (``stent_features``,
        ``stent_centerline_direction``, ``r_mid``, ``strut_thickness``,
        ``circumference``, ``ring_edges``), and the surface point cloud
        (``stent_df``).
    """
    ring2d_pkl = os.path.join(output_dir, 'ring_2d.pkl')
    ring_csv   = os.path.join(output_dir, 'ring_points.csv')
    for p in (ring2d_pkl, ring_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing {p} - run Steps 1-5.4 first (per-ring checkpoint not found).")

    with open(ring2d_pkl, 'rb') as f:
        ck = pickle.load(f)

    os.makedirs(os.path.join(output_dir, 'skeleton_plots'), exist_ok=True)
    print(f"[resume] loading point cloud from {ring_csv} ...")
    stent_df = pd.read_csv(ring_csv)

    state = {
        'ring_2d'                  : ck['RING_2D'],
        'stent_features'            : ck['stent_features'],
        'stent_centerline_direction': np.asarray(ck['stent_centerline_direction']),
        'r_mid'                     : float(ck['r_mid']),
        'strut_thickness'           : float(ck['strut_thickness']),
        'circumference'             : float(ck['circumference']),
        'ring_edges'               : ck.get('ring_edges'),
        'stent_df'                  : stent_df,
    }
    print(f"[resume] restored {len(state['ring_2d'])} per-ring 2D skeletons + "
          f"{len(stent_df):,} surface points. Ready for the manual-edit step.")
    return state



def _parse_two_ids(s: str) -> tuple[int, int]:
    """
    Parse a "a, b" / "a b" string into exactly two int point indices.

    :param s: User input like ``"12, 40"`` or ``"12 40"``.
    :raises ValueError: If ``s`` does not contain exactly two integers.
    :returns: The two parsed indices, as ``(a, b)``.
    """
    vals = [int(x) for x in (s or '').replace(',', ' ').split()]
    if len(vals) != 2:
        raise ValueError("please give exactly two point indices, e.g. '12, 40'")
    return vals[0], vals[1]



def edit_rings_2d_interactive(ring_2d: dict,
                              stent_features: dict,
                              stent_centerline_direction: np.ndarray,
                              r_mid: float,
                              strut_thickness: float,
                              circumference: float,
                              ring_edges: np.ndarray,
                              output_dir: str) -> dict:
    """
    Prompt the user to manually fix defects in any ring's 2D skeleton.

    Asks once whether to edit any ring at all. If yes, loops: pick a ring by
    label, describe the problem (``loop`` or ``connection``), give the two
    point indices at the defect, and the matching fixer runs
    (:func:`fix_ring_loop_2d` or :func:`fix_ring_connection_2d`). Each edit is
    applied tentatively, rendered to a preview PNG/HTML via
    :func:`~stentfit.stent_plotting._render_ring_2d`, and only kept (and
    checkpointed to disk) if the user confirms it looks right — otherwise the
    ring is reverted to its state before that edit.

    :param ring_2d: Per-ring 2D skeletons, from :func:`skeletonize_rings_2d`.
        Edited in place and also returned.
    :param stent_features: Stent features dict, passed through to the
        checkpoint save on each confirmed edit.
    :param stent_centerline_direction: Stent centreline unit vector, passed
        through to the checkpoint save.
    :param r_mid: Mid-wall radius, passed through to the checkpoint save.
    :param strut_thickness: Strut thickness, passed through to the checkpoint save.
    :param circumference: Full circumference at ``r_mid``, passed through to
        the checkpoint save.
    :param ring_edges: Z-boundaries between rings, passed through to the checkpoint save.
    :param output_dir: Folder edit previews are rendered into and the
        checkpoint is saved to.
    :returns: ``ring_2d``, with any confirmed edits applied.
    """
    plots_dir = os.path.join(output_dir, 'skeleton_plots')
    os.makedirs(plots_dir, exist_ok=True)

    if input("Do you want to manually edit any ring? [y/n] ").strip().lower() == 'y':
        while True:
            ans = input("\nWhich ring would you like to change? "
                        "(e.g. ring_01, ring_02, ...; empty = done) ").strip()
            if not ans:
                break
            try:
                label = ans if ans.startswith('ring_') else f"ring_{int(ans):02d}"
            except ValueError:
                print(f"  '{ans}' is not a valid ring name. "
                      f"Available: {', '.join(ring_2d)}")
                continue
            if label not in ring_2d:
                print(f"  '{label}' not found. Available: {', '.join(ring_2d)}")
                continue

            rec  = ring_2d[label]
            prob = input("What is the problem in the skeleton? (loop / connection) ").strip().lower()

            if prob.startswith('loop'):
                print("  -> I'll remove the loop (bubble) and join the two points you give "
                      "with a single evenly-spaced straight line. Everything else stays.")
                a, b = _parse_two_ids(input("  Two point indices at the loop ends (e.g. 12, 40): "))
                new_arc, new_z, changed = fix_ring_loop_2d(rec['arc'], rec['z'],
                                                            rec['pixel_size'], a, b)
            elif prob.startswith('conn'):
                print("  -> I'll remove the WHOLE wrong bridge (extending up to where it "
                      "meets the struts) to open a clean gap. Everything else stays.")
                a, b = _parse_two_ids(input("  Two point indices ON the wrong bridge itself, "
                                            "e.g. its two ends (e.g. 12, 40): "))
                new_arc, new_z, changed = fix_ring_connection_2d(rec['arc'], rec['z'],
                                                                  rec['pixel_size'], a, b)
            else:
                print("  Unknown problem — choose 'loop' or 'connection'.")
                continue

            if len(new_arc) == len(rec['arc']):
                print("  No change made (see the note above). Re-check the point indices.")
                continue

            # keep a backup so the edit can be undone, then tentatively apply + render
            prev_arc, prev_z, prev_n = rec['arc'].copy(), rec['z'].copy(), rec['n_edits']
            rec['arc'], rec['z'] = new_arc, new_z
            rec['n_edits'] += 1
            out = _render_ring_2d(ring_2d, label, plots_dir, changed_idx=changed,
                                   suffix='edited')
            print(f"  preview saved as {label}_edited_{rec['n_edits']} -> {out}")

            if input("  Are you happy with this edit? (y/n) ").strip().lower().startswith('y'):
                save_ring_2d_checkpoint(ring_2d, stent_features, stent_centerline_direction,
                                         r_mid, strut_thickness, circumference, ring_edges,
                                         output_dir, verbose=False)
                print(f"  kept edit {rec['n_edits']} on {label}. (ring_2d.pkl updated)")
            else:
                rec['arc'], rec['z'], rec['n_edits'] = prev_arc, prev_z, prev_n
                try:
                    os.remove(out)
                except OSError:
                    pass
                print(f"  reverted — {label} restored to its previous state. You can try again.")

    return ring_2d



def assemble_2d_skeleton(ring_2d: dict) -> dict:
    """
    Concatenate every ring's 2D skeleton into one flat skeleton.

    Rings are ordered by ``ring_id`` (bottom to top along the stent axis) and
    their (arc, z) points are stacked into single arrays. Each point also
    carries its ring's ``pixel_size``, since different rings may have been
    skeletonised at different resolutions.

    :param ring_2d: Per-ring 2D skeletons, from :func:`skeletonize_rings_2d`
        (optionally edited by :func:`edit_rings_2d_interactive`).
    :returns: Dict with the concatenated coordinates (``skel_arc``,
        ``skel_z``), the per-point pixel size (``skel_px``), and the median
        ``pixel_size`` across all rings.
    """
    order    = sorted(ring_2d, key=lambda L: ring_2d[L]['ring_id'])
    skel_arc = np.concatenate([ring_2d[L]['arc'] for L in order])
    skel_z   = np.concatenate([ring_2d[L]['z']   for L in order])
    skel_px  = np.concatenate([np.full(len(ring_2d[L]['arc']), ring_2d[L]['pixel_size'])
                               for L in order])
    pixel_size = float(np.median([ring_2d[L]['pixel_size'] for L in order]))
    print(f"\nAssembled 2D skeleton: {len(skel_arc):,} points "
          f"| pixel_size median = {pixel_size:.5f}")
    return {'skel_arc': skel_arc, 'skel_z': skel_z, 'skel_px': skel_px,
            'pixel_size': pixel_size}


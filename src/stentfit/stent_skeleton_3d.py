import numpy as np
import pandas as pd
import ast
import datetime
import json
import os
from scipy.spatial import cKDTree

from .stent_plotting import plot_skeleton_html, plot_skeleton_with_cloud_html


def adjust_skeleton_to_local_midsurface(
    skel_arc: np.ndarray,
    skel_z: np.ndarray,
    stent_df: pd.DataFrame,
    r_mid: float,
    circumference: float,
    search_radius: float,
) -> dict:
    """Wrap the 2D skeleton to 3D using a per-point local radius.

    Each skeleton point's r is the mean r of surface points within search_radius
    (in the arc/z plane), instead of the constant r_mid.
    """
    # Triple-tile surface points so the periodic seam is seamless
    arc_all = r_mid * stent_df['theta'].values
    z_all   = stent_df['z_cylindrical'].values
    r_all   = stent_df['r'].values

    arc_three = np.concatenate([arc_all - circumference, arc_all, arc_all + circumference])
    z_three   = np.concatenate([z_all,                   z_all,   z_all])
    r_three   = np.concatenate([r_all,                   r_all,   r_all])

    tree    = cKDTree(np.column_stack([arc_three, z_three]))
    nb_idx  = tree.query_ball_point(np.column_stack([skel_arc, skel_z]), r=search_radius)
    local_r = np.array([r_three[idx].mean() if len(idx) else r_mid for idx in nb_idx])

    theta_skel = skel_arc / r_mid
    x = local_r * np.cos(theta_skel)
    y = local_r * np.sin(theta_skel)
    z = skel_z

    return {
        'df_skeleton_3d' : pd.DataFrame({
            'theta' : theta_skel,
            'x'     : x,
            'y'     : y,
            'z'     : z,
            'r'     : local_r,
        }),
        'skeleton_points': np.column_stack([x, y, z]),
    }



def analyze_skeleton_connectivity(
    df_skeleton_3d: pd.DataFrame,
    pixel_size,                              # scalar, OR a per-point array (len N)
    neighbor_radius_factor: float = 1.8,
) -> pd.DataFrame:
    """Classify each skeleton point by degree: isolated/endpoint/line/junction.

    pixel_size may be a scalar (uniform skeleton) or a per-point array of length N.
    The per-point form is for a skeleton assembled from pieces tuned to different
    resolutions: points i, j link only when their distance is within
    neighbor_radius_factor * max(px_i, px_j), so a fine piece is never
    over-connected by a coarse piece's radius.
    """
    pts = df_skeleton_3d[['x', 'y', 'z']].values
    N   = len(pts)
    px  = np.asarray(pixel_size, dtype=float)

    tree = cKDTree(pts)
    # Query at the largest neighbour radius any point could use, then (per-point
    # case) keep only pairs within the local radius set by the COARSER of the two.
    radius = neighbor_radius_factor * float(px.max())
    pairs  = tree.query_pairs(r=radius, output_type='ndarray')
    if px.ndim and len(pairs):
        d   = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
        thr = neighbor_radius_factor * np.maximum(px[pairs[:, 0]], px[pairs[:, 1]])
        pairs = pairs[d <= thr]

    neighbors = [[] for _ in range(N)]
    for i, j in pairs:
        neighbors[i].append(int(j))
        neighbors[j].append(int(i))

    degrees   = np.array([len(nb) for nb in neighbors])
    node_type = np.select(
        [degrees == 0, degrees == 1, degrees == 2],
        ['isolated',   'endpoint',   'line'],
        default='junction',
    )

    df_result = df_skeleton_3d.copy().reset_index(drop=True)
    df_result['skeleton_point_id'] = np.arange(len(df_result))
    df_result['degree']            = degrees
    df_result['node_type']         = node_type
    df_result['neighbor_ids']      = neighbors

    return df_result



def reconnect_skeleton_endpoints(
    df_connectivity: pd.DataFrame,
    pixel_size: float,
    max_gap_factor: float = 12.0,
    min_cos: float = 0.5,
    exclude_hops: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Bridge spurious breaks where thinning split a strut into two endpoints.

    Each degree-1 endpoint looks forward along its own tangent and reconnects to
    the nearest skeleton point inside a forward cone within max_gap. Genuine tips
    have no forward continuation, so they find no candidate and stay endpoints.

    max_gap_factor scales pixel_size to the largest bridgeable gap; min_cos is the
    forward-cone half-angle as a cosine (0.5 => 60 deg); exclude_hops skips
    candidates within that many graph hops so an endpoint never joins its own stub.

    Returns a copy with neighbor_ids, degree and node_type recomputed.
    """
    df  = df_connectivity.reset_index(drop=True).copy()
    pts = df[['x', 'y', 'z']].values
    N   = len(pts)
    max_gap = max_gap_factor * pixel_size

    # Symmetric adjacency from neighbor_ids
    adj = [set() for _ in range(N)]
    for i, nbrs in enumerate(df['neighbor_ids']):
        if isinstance(nbrs, str):
            nbrs = ast.literal_eval(nbrs)
        for j in nbrs:
            j = int(j)
            adj[i].add(j); adj[j].add(i)

    deg = np.array([len(a) for a in adj])

    def _within_hops(src, hops):
        seen, frontier = {src}, {src}
        for _ in range(hops):
            nxt = set()
            for u in frontier:
                nxt |= adj[u]
            nxt -= seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return seen

    tree      = cKDTree(pts)
    endpoints = np.where(deg == 1)[0].tolist()

    new_edges = []
    resolved  = set()

    for e in endpoints:
        if e in resolved or deg[e] != 1:
            continue
        nb      = next(iter(adj[e]))
        tangent = pts[e] - pts[nb]                 # points outward, away from stub
        tlen    = np.linalg.norm(tangent)
        if tlen < 1e-12:
            continue
        tangent /= tlen

        near = _within_hops(e, exclude_hops)
        cand = tree.query_ball_point(pts[e], r=max_gap)

        best, best_score = None, -np.inf
        for c in cand:
            if c == e or c in near or c in resolved:
                continue
            d    = pts[c] - pts[e]
            dist = np.linalg.norm(d)
            if dist < 1e-9:
                continue
            cos = float(np.dot(tangent, d / dist))
            if cos < min_cos:                      # outside the forward cone
                continue
            score = cos / dist                     # aligned & near wins
            if score > best_score:
                best, best_score = c, score

        if best is not None:
            new_edges.append((e, best))
            adj[e].add(best); adj[best].add(e)
            deg[e] += 1; deg[best] += 1
            resolved.add(e)
            if deg[best] == 2 and best in set(endpoints):
                resolved.add(best)                 # bridged endpoint<->endpoint pair

    # Recompute connectivity fields from the final adjacency
    neighbors = [sorted(adj[i]) for i in range(N)]
    degrees   = np.array([len(a) for a in neighbors])
    node_type = np.select(
        [degrees == 0, degrees == 1, degrees == 2],
        ['isolated',   'endpoint',   'line'],
        default='junction',
    )

    df['neighbor_ids'] = neighbors
    df['degree']       = degrees
    df['node_type']    = node_type

    if verbose:
        print(f"[reconnect] endpoints {len(endpoints)} -> {int((degrees == 1).sum())}  "
              f"({len(new_edges)} bridges added, max_gap = {max_gap:.4f} mm, "
              f"cone = {np.degrees(np.arccos(min_cos)):.0f} deg)")

    return df



def stitch_seam_endpoints(
    df_connectivity: pd.DataFrame,
    r_mid: float,
    strut_thickness: float,
    z_tol_frac: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reconnect strut pieces cut by the unroll seam (the theta = +/-pi wrap).

    Cropping the unrolled skeleton cuts every seam-crossing strut into two stubs,
    one on each arc edge at the same z. The 3D wrap puts them at nearly the same
    point but as two degree-1 endpoints. They are joined when all three hold:
    opposite seam edges, same height z (within z_tol_frac strut thicknesses), and
    closer than one strut thickness in 3D. The only scale is strut_thickness.

    Returns a copy with neighbor_ids, degree and node_type recomputed.
    """
    df  = df_connectivity.reset_index(drop=True).copy()
    pts = df[['x', 'y', 'z']].values
    N   = len(pts)

    max_gap   = strut_thickness                 # 3D join distance (condition 3)
    z_tol     = z_tol_frac * strut_thickness    # same-height tolerance (condition 2)
    seam_band = strut_thickness / r_mid         # seam-edge band in theta (= one strut thickness of arc)

    adj = [set() for _ in range(N)]
    for i, nbrs in enumerate(df['neighbor_ids']):
        if isinstance(nbrs, str):
            nbrs = ast.literal_eval(nbrs)
        for j in nbrs:
            j = int(j)
            adj[i].add(j); adj[j].add(i)
    deg = np.array([len(a) for a in adj])

    theta     = df['theta'].values
    th_min, th_max = theta.min(), theta.max()
    endpoints = np.where(deg == 1)[0]

    low  = [e for e in endpoints if theta[e] <= th_min + seam_band]   # on theta_min edge
    high = [e for e in endpoints if theta[e] >= th_max - seam_band]   # on theta_max edge

    # All low/high pairs that satisfy conditions 2 and 3, shortest 3D gap first so
    # each stub greedily takes its true partner; every endpoint is used at most once.
    cands = []
    for e in low:
        for h in high:
            if h == e:
                continue
            if abs(pts[e, 2] - pts[h, 2]) > z_tol:          # condition 2: same height
                continue
            d = float(np.linalg.norm(pts[e] - pts[h]))
            if d <= max_gap:                                # condition 3: within a strut thickness
                cands.append((d, e, h))
    cands.sort()

    new_edges = []
    used = set()
    for d, e, h in cands:
        if e in used or h in used:
            continue
        adj[e].add(h); adj[h].add(e)
        used.add(e); used.add(h)
        new_edges.append((e, h))

    neighbors = [sorted(adj[i]) for i in range(N)]
    degrees   = np.array([len(a) for a in neighbors])
    node_type = np.select(
        [degrees == 0, degrees == 1, degrees == 2],
        ['isolated',   'endpoint',   'line'],
        default='junction',
    )
    df['neighbor_ids'] = neighbors
    df['degree']       = degrees
    df['node_type']    = node_type

    if verbose:
        print(f"[seam-stitch] seam endpoints: {len(low)} low / {len(high)} high "
              f"-> {len(new_edges)} bridge(s) added "
              f"(join < {max_gap:.4f} mm = 1 strut thickness, z_tol = {z_tol:.4f} mm)")

    return df



def prune_skeleton_spurs(
    df_connectivity: pd.DataFrame,
    pixel_size: float,
    tip_frac: float = 0,
    max_spur_len: float = None,
    max_iter: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove spurs: short dead-end branches that are not genuine stent tips.

    A degree-1 node is kept as a tip when it lies within tip_frac of the axial (z)
    span from either end. Every other dead-end is walked back along its degree-2
    chain to the first junction and deleted; the walk repeats max_iter times so
    chained spurs collapse. max_spur_len optionally keeps any non-tip dead-end
    longer than that (likely a real strut whose far end failed to bridge).

    Returns a re-indexed copy (contiguous skeleton_point_id, remapped
    neighbor_ids, recomputed degree / node_type).
    """
    df  = df_connectivity.reset_index(drop=True).copy()
    pts = df[['x', 'y', 'z']].values
    N   = len(pts)
    z   = pts[:, 2]
    z_min, z_max = z.min(), z.max()
    tip_band = tip_frac * (z_max - z_min)

    def _is_tip(i):
        # tip_band <= 0 protects no tips: every dead-end is prunable (strict band)
        if tip_band <= 0:
            return False
        return (z[i] - z_min) <= tip_band or (z_max - z[i]) <= tip_band

    adj = [set() for _ in range(N)]
    for i, nbrs in enumerate(df['neighbor_ids']):
        if isinstance(nbrs, str):
            nbrs = ast.literal_eval(nbrs)
        for j in nbrs:
            j = int(j)
            adj[i].add(j); adj[j].add(i)

    alive = np.ones(N, dtype=bool)

    def _deg(i):
        return sum(1 for n in adj[i] if alive[n])

    n_branches   = 0
    n_removed    = 0
    skipped_long = 0

    for _ in range(max_iter):
        endpoints = [i for i in range(N) if alive[i] and _deg(i) == 1]
        to_remove = set()

        for e in endpoints:
            if not alive[e] or _deg(e) != 1 or _is_tip(e):
                continue

            # Walk the dead-end chain to the first junction / other endpoint
            chain   = [e]
            visited = {e}
            prev    = e
            cur      = next(n for n in adj[e] if alive[n])
            length   = float(np.linalg.norm(pts[cur] - pts[prev]))
            terminal = None
            while True:
                d_cur = _deg(cur)
                if d_cur >= 3:
                    terminal = ('junction', cur); break           # stop before junction
                if d_cur == 1:
                    chain.append(cur); terminal = ('endpoint', cur); break
                nxts = [n for n in adj[cur] if alive[n] and n != prev]
                if not nxts or nxts[0] in visited:
                    chain.append(cur); terminal = ('loop', cur); break
                chain.append(cur); visited.add(cur)
                nxt = nxts[0]
                length += float(np.linalg.norm(pts[nxt] - pts[cur]))
                prev, cur = cur, nxt

            # Keep floating segments that terminate on a real tip
            if terminal[0] == 'endpoint' and _is_tip(terminal[1]):
                continue
            if max_spur_len is not None and length > max_spur_len:
                skipped_long += 1
                continue

            to_remove.update(chain)
            n_branches += 1

        if not to_remove:
            break
        for n in to_remove:
            alive[n] = False
        n_removed += len(to_remove)

    # Re-index surviving nodes and remap neighbours
    keep       = np.where(alive)[0]
    old_to_new = {int(o): i for i, o in enumerate(keep)}
    new        = df.iloc[keep].reset_index(drop=True).copy()

    new_neighbors = [sorted(old_to_new[n] for n in adj[o] if alive[n]) for o in keep]
    degrees       = np.array([len(n) for n in new_neighbors])
    node_type     = np.select(
        [degrees == 0, degrees == 1, degrees == 2],
        ['isolated',   'endpoint',   'line'],
        default='junction',
    )

    new['skeleton_point_id'] = np.arange(len(new))
    new['neighbor_ids']      = new_neighbors
    new['degree']            = degrees
    new['node_type']         = node_type

    if verbose:
        nz       = new['z'].values
        rem_nontip = int(np.sum((degrees == 1) &
                                ~(((nz - z_min) <= tip_band) | ((z_max - nz) <= tip_band))))
        msg = (f"[prune_spurs] removed {n_removed} nodes in {n_branches} spur branch(es); "
               f"remaining non-tip endpoints: {rem_nontip}")
        if skipped_long:
            msg += f"; {skipped_long} dead-end(s) kept (> max_spur_len)"
        print(msg)

    return new



def collapse_tiny_loops(df_connectivity: pd.DataFrame, strut_thickness: float,
                        max_diag_factor: float = 2.0, verbose: bool = True) -> pd.DataFrame:
    """Remove tiny spurious loops (small closed cycles / lasso hooks).

    A stray little loop makes an otherwise-normal strut point look like a junction
    (it gains a 3rd neighbour), and the spur-pruner can't touch it because it is not
    a loose end. This targets two clearly-artifact shapes and leaves genuine large
    design cells alone:
      * lasso self-loop: a degree-2 chain that leaves a junction and returns to the
        SAME junction — its interior is deleted, demoting the false junction back to
        a normal point / tip.
      * isolated tiny cycle: a free-floating small degree-2 loop — deleted whole.
    A loop counts as tiny when its bounding-box diagonal is <=
    max_diag_factor * strut_thickness (design cells are much larger). Curves running
    between two DIFFERENT junctions are never touched (they may be real connectors).

    Returns a re-indexed copy (contiguous skeleton_point_id, remapped neighbor_ids,
    recomputed degree / node_type).
    """
    df  = df_connectivity.reset_index(drop=True).copy()
    N   = len(df)
    pts = df[['x', 'y', 'z']].values

    adj = [set() for _ in range(N)]
    for i, nbrs in enumerate(df['neighbor_ids']):
        if isinstance(nbrs, str):
            nbrs = ast.literal_eval(nbrs)
        for j in nbrs:
            j = int(j)
            adj[i].add(j); adj[j].add(i)
    deg      = np.array([len(a) for a in adj])
    specials = set(np.where(deg != 2)[0].tolist())

    # group the graph into curves (degree-2 chains bounded by specials, or closed
    # degree-2 loops), tracing each undirected edge once
    visited = set()

    def walk(s, nxt):
        path = [s, nxt]
        visited.add(frozenset((s, nxt)))
        prev, cur = s, nxt
        while cur not in specials:
            others = [n for n in adj[cur] if n != prev]
            if not others:
                break
            nb = others[0]
            visited.add(frozenset((cur, nb)))
            path.append(nb)
            prev, cur = cur, nb
            if cur == s:
                break
        return path

    curves = []
    for s in specials:
        for nb in adj[s]:
            if frozenset((s, nb)) not in visited:
                curves.append(walk(s, nb))
    for i in range(N):
        for nb in adj[i]:
            if frozenset((i, nb)) not in visited:
                curves.append(walk(i, nb))

    max_diag = max_diag_factor * strut_thickness
    remove, n_loops = set(), 0
    for c in curves:
        cp   = pts[c]
        diag = float(np.linalg.norm(cp.max(0) - cp.min(0)))
        if diag > max_diag:
            continue
        if c[0] == c[-1] and c[0] in specials:          # lasso self-loop at a junction
            remove.update(n for n in c if n != c[0])
            n_loops += 1
        elif c[0] == c[-1]:                             # isolated tiny cycle
            remove.update(c)
            n_loops += 1

    alive = np.ones(N, dtype=bool)
    alive[list(remove)] = False
    keep       = np.where(alive)[0]
    old_to_new = {int(o): i for i, o in enumerate(keep)}
    new        = df.iloc[keep].reset_index(drop=True).copy()

    new_neighbors = [sorted(old_to_new[n] for n in adj[o] if alive[n]) for o in keep]
    degrees       = np.array([len(n) for n in new_neighbors])
    node_type     = np.select(
        [degrees == 0, degrees == 1, degrees == 2],
        ['isolated',   'endpoint',   'line'],
        default='junction',
    )
    new['skeleton_point_id'] = np.arange(len(new))
    new['neighbor_ids']      = new_neighbors
    new['degree']            = degrees
    new['node_type']         = node_type

    if verbose:
        print(f"[collapse_tiny_loops] removed {n_loops} tiny loop(s) "
              f"(<= {max_diag:.4f} mm diag), {len(remove)} points; "
              f"total nodes {N} -> {len(new)}")
    return new



def merge_seam_duplicates(
    df_connectivity: pd.DataFrame,
    r_mid: float,
    strut_thickness: float,
    seam_band_frac: float = 3.0,
    z_tol_frac: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fold the doubled unroll-seam truss into a single centerline.

    The seam padding skeletonises a seam-crossing strut twice, once near each arc
    edge; wrapped to 3D the two copies form near-parallel rails that get
    cross-linked into a dense ribbon. A low-rail node and its high-rail partner are
    the same strut at the same height z, so each low/high pair within z_tol is
    merged to its midpoint (neighbours unioned). There is no 3D-distance cap, since
    the gap can be large at the wide end of the truss.

    seam_band_frac is the seam-edge band width in strut thicknesses (the key knob);
    z_tol_frac is the same-height tolerance. Returns a re-indexed copy.
    """
    df  = df_connectivity.reset_index(drop=True).copy()
    pts = df[['x', 'y', 'z']].values.copy()   # writable (the merge overwrites survivor coords)
    N   = len(pts)

    # Symmetric adjacency from neighbor_ids
    adj = [set() for _ in range(N)]
    for i, nbrs in enumerate(df['neighbor_ids']):
        if isinstance(nbrs, str):
            nbrs = ast.literal_eval(nbrs)
        for j in nbrs:
            j = int(j)
            adj[i].add(j); adj[j].add(i)

    theta          = df['theta'].values
    z              = pts[:, 2]
    th_min, th_max = theta.min(), theta.max()
    band           = seam_band_frac * strut_thickness / r_mid   # seam-edge band in theta
    z_tol          = z_tol_frac * strut_thickness

    low  = np.where(theta <= th_min + band)[0]
    high = np.where(theta >= th_max - band)[0]

    # candidate low/high pairs at the same height (no 3D cap), shortest |dz| first
    cands = []
    for e in low:
        for h in high:
            if h == e:
                continue
            dz = abs(z[e] - z[h])
            if dz <= z_tol:
                d3 = float(np.linalg.norm(pts[e] - pts[h]))
                cands.append((dz, d3, int(e), int(h)))
    cands.sort()

    alive   = np.ones(N, dtype=bool)
    used    = set()
    n_pairs = 0
    for dz, d3, e, h in cands:
        if e in used or h in used:
            continue

        # Merge h into survivor e at the midpoint; union neighbours, drop h.
        mid    = 0.5 * (pts[e] + pts[h])
        pts[e] = mid
        df.at[e, 'x'] = mid[0]
        df.at[e, 'y'] = mid[1]
        df.at[e, 'z'] = mid[2]
        df.at[e, 'r'] = float(np.hypot(mid[0], mid[1]))
        df.at[e, 'theta'] = float(np.arctan2(mid[1], mid[0]))

        for nb in list(adj[h]):
            adj[nb].discard(h)
            if nb != e:
                adj[e].add(nb); adj[nb].add(e)
        adj[h] = set()
        adj[e].discard(e)
        alive[h] = False
        used.add(e); used.add(h)
        n_pairs += 1

    # Re-index surviving nodes and remap neighbours
    keep       = np.where(alive)[0]
    old_to_new = {int(o): i for i, o in enumerate(keep)}
    new        = df.iloc[keep].reset_index(drop=True).copy()

    new_neighbors = [sorted(old_to_new[n] for n in adj[o] if alive[n]) for o in keep]
    degrees       = np.array([len(n) for n in new_neighbors])
    node_type     = np.select(
        [degrees == 0, degrees == 1, degrees == 2],
        ['isolated',   'endpoint',   'line'],
        default='junction',
    )

    new['skeleton_point_id'] = np.arange(len(new))
    new['neighbor_ids']      = new_neighbors
    new['degree']            = degrees
    new['node_type']         = node_type

    if verbose:
        print(f"[seam-merge] rails: {len(low)} low / {len(high)} high -> "
              f"{n_pairs} pair(s) merged to midpoint; "
              f"total nodes {N} -> {len(new)} "
              f"(band={seam_band_frac}xt, z_tol={z_tol:.4f} mm)")

    return new



def collapse_junction_clusters(
    df_connectivity: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Collapse each connected blob of junction nodes into one centroid node.

    Thinning leaves a small blob of degree>=3 pixels at every strut crossing. This
    takes the junction-only subgraph (edges whose both endpoints are junctions),
    finds its connected components, and replaces each blob of two or more
    junctions with a single node at their centroid, rewiring every attached strut.
    Genuinely separate crossings are not edge-connected, so they stay distinct.

    Returns a re-indexed copy (contiguous skeleton_point_id, remapped
    neighbor_ids, recomputed degree / node_type).
    """
    df  = df_connectivity.reset_index(drop=True).copy()
    pts = df[['x', 'y', 'z']].values
    N   = len(pts)

    # Symmetric adjacency from neighbor_ids
    adj = [set() for _ in range(N)]
    for i, nbrs in enumerate(df['neighbor_ids']):
        if isinstance(nbrs, str):
            nbrs = ast.literal_eval(nbrs)
        for j in nbrs:
            j = int(j)
            adj[i].add(j); adj[j].add(i)

    deg          = np.array([len(a) for a in adj])
    junctions    = np.where(deg >= 3)[0]
    junction_set = {int(j) for j in junctions}

    # group edge-connected junction nodes into clusters (one blob per crossing)
    clusters = []
    visited  = set()
    for j in junction_set:
        if j in visited:
            continue
        comp  = []
        stack = [j]
        visited.add(j)
        while stack:
            u = stack.pop()
            comp.append(u)
            for nb in adj[u]:
                if nb in junction_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if len(comp) >= 2:
            clusters.append(np.array(sorted(comp)))

    alive = np.ones(N, dtype=bool)
    n_clusters = 0

    for members in clusters:
        members = sorted(int(m) for m in members)
        survivor = members[0]
        member_set = set(members)

        # Centroid of the cluster -> overwrite survivor's coordinates
        centroid = pts[members].mean(axis=0)
        df.at[survivor, 'x'] = centroid[0]
        df.at[survivor, 'y'] = centroid[1]
        df.at[survivor, 'z'] = centroid[2]
        df.at[survivor, 'r'] = float(np.hypot(centroid[0], centroid[1]))
        df.at[survivor, 'theta'] = float(np.arctan2(centroid[1], centroid[0]))

        # Rewire: external neighbours of any member attach to the survivor
        external = set()
        for m in members:
            for nb in adj[m]:
                if nb not in member_set:
                    external.add(nb)
        # Drop old member<->external edges, then connect survivor<->external
        for m in members:
            for nb in list(adj[m]):
                adj[nb].discard(m)
            adj[m] = set()
        for nb in external:
            adj[survivor].add(nb); adj[nb].add(survivor)

        # Keep only the survivor; delete the rest of the cluster
        for m in members[1:]:
            alive[m] = False
        n_clusters += 1

    # Re-index surviving nodes and remap neighbours
    keep       = np.where(alive)[0]
    old_to_new = {int(o): i for i, o in enumerate(keep)}
    new        = df.iloc[keep].reset_index(drop=True).copy()

    new_neighbors = [sorted(old_to_new[n] for n in adj[o] if alive[n]) for o in keep]
    degrees       = np.array([len(n) for n in new_neighbors])
    node_type     = np.select(
        [degrees == 0, degrees == 1, degrees == 2],
        ['isolated',   'endpoint',   'line'],
        default='junction',
    )

    new['skeleton_point_id'] = np.arange(len(new))
    new['neighbor_ids']      = new_neighbors
    new['degree']            = degrees
    new['node_type']         = node_type

    if verbose:
        n_junc_after = int((degrees >= 3).sum())
        print(f"[collapse_junctions] collapsed {n_clusters} cluster(s); "
              f"junction nodes {len(junctions)} -> {n_junc_after}; "
              f"total nodes {N} -> {len(new)} (edge-connected contraction)")

    return new



def _write_skeleton_stp(skeleton_df: pd.DataFrame, output_path: str) -> None:
    """Write skeleton wireframe as an ISO 10303-21 (STEP) file."""
    pts_data = skeleton_df.reset_index(drop=True)
    pts = pts_data[['x', 'y', 'z']].values
    id_to_idx = {int(row.skeleton_point_id): i for i, row in enumerate(pts_data.itertuples())}

    # Build unique edge set from neighbor_ids
    edges = set()
    for row in pts_data.itertuples():
        pid = int(row.skeleton_point_id)
        nbrs = row.neighbor_ids
        if isinstance(nbrs, str):
            nbrs = ast.literal_eval(nbrs)
        for nid in nbrs:
            edges.add((min(pid, nid), max(pid, nid)))
    edges = sorted(edges)

    buf = []
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    buf.append('ISO-10303-21;')
    buf.append('HEADER;')
    buf.append("FILE_DESCRIPTION(('Stent Skeleton Wireframe'),'2;1');")
    buf.append(f"FILE_NAME('skeleton_points.stp','{now}',(''),(''),(''),(''),(''));")
    buf.append("FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));")
    buf.append('ENDSEC;')
    buf.append('DATA;')

    eid = 1

    unit_id = eid
    buf.append(f'#{eid}=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));')
    eid += 1

    angle_id = eid
    buf.append(f'#{eid}=(PLANE_ANGLE_UNIT()NAMED_UNIT(*)SI_UNIT($,.RADIAN.));')
    eid += 1

    solid_id = eid
    buf.append(f'#{eid}=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());')
    eid += 1

    unc_id = eid
    buf.append(f"#{eid}=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{unit_id},'distance_accuracy_value','confusion accuracy');")
    eid += 1

    ctx_id = eid
    buf.append(
        f"#{eid}=(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc_id}))"
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{unit_id},#{angle_id},#{solid_id}))"
        f"REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'));"
    )
    eid += 1

    # One CARTESIAN_POINT per skeleton vertex
    cp_ids = []
    for pt in pts:
        buf.append(f"#{eid}=CARTESIAN_POINT('',({pt[0]:.6f},{pt[1]:.6f},{pt[2]:.6f}));")
        cp_ids.append(eid)
        eid += 1

    # DIRECTION + VECTOR + LINE for each unique edge
    line_ids = []
    for (pi, pj) in edges:
        idx_i = id_to_idx.get(pi)
        idx_j = id_to_idx.get(pj)
        if idx_i is None or idx_j is None:
            continue
        d = pts[idx_j] - pts[idx_i]
        length = np.linalg.norm(d)
        if length < 1e-10:
            continue
        dn = d / length

        dir_id = eid
        buf.append(f"#{eid}=DIRECTION('',({dn[0]:.6f},{dn[1]:.6f},{dn[2]:.6f}));")
        eid += 1

        vec_id = eid
        buf.append(f"#{eid}=VECTOR('',#{dir_id},{length:.6f});")
        eid += 1

        line_id = eid
        buf.append(f"#{eid}=LINE('',#{cp_ids[idx_i]},#{vec_id});")
        eid += 1
        line_ids.append(line_id)

    # GEOMETRIC_CURVE_SET collecting all lines
    gcs_id = eid
    refs = ','.join(f'#{lid}' for lid in line_ids)
    buf.append(f"#{eid}=GEOMETRIC_CURVE_SET('Skeleton Wireframe',({refs}));")
    eid += 1

    # SHAPE_REPRESENTATION referencing the curve set and context
    buf.append(f"#{eid}=SHAPE_REPRESENTATION('Stent Skeleton',(#{gcs_id}),#{ctx_id});")
    eid += 1

    buf.append('ENDSEC;')
    buf.append('END-ISO-10303-21;')

    with open(output_path, 'w') as f:
        f.write('\n'.join(buf))



def wrap_skeleton_to_3d(skel_arc, skel_z, skel_px, pixel_size, surf_df, r_mid,
                        circumference, strut_thickness, output_dir, stent_name,
                        wrap_max_surf=None, seam_tol_frac=0.01, seam_z_tol_frac=3.0,
                        prune_tip_frac=0, max_display=500_000, random_seed=0):
    """Wrap the 2D skeleton onto the local mid-surface and run graph cleanup (Step 6).

    Lifts each 2D point to 3D via a per-point local mid-surface radius, classifies
    node degree, then folds the seam, contracts junction clusters, prunes short
    spurs and collapses tiny loops. On a huge cloud a downsampled throwaway copy is
    used for the radius estimate only (``wrap_max_surf``). Saves
    ``skeleton_points.csv`` + ``skeleton_only.html`` and returns the final SKELETON_DF.
    """
    if wrap_max_surf is not None and len(surf_df) > wrap_max_surf:
        wrap_surf = surf_df.sample(wrap_max_surf, random_state=random_seed)
        print(f"[wrap] downsampled surface for the wrap: {len(surf_df):,} -> "
              f"{len(wrap_surf):,} pts (wrap_max_surf)")
    else:
        wrap_surf = surf_df

    skel3d = adjust_skeleton_to_local_midsurface(
        skel_arc, skel_z, wrap_surf, r_mid, circumference, search_radius=strut_thickness)

    df = analyze_skeleton_connectivity(skel3d['df_skeleton_3d'], skel_px)
    df = merge_seam_duplicates(df, r_mid, strut_thickness,
                               seam_band_frac=seam_tol_frac, z_tol_frac=seam_z_tol_frac)
    df = collapse_junction_clusters(df)
    df = prune_skeleton_spurs(df, pixel_size, tip_frac=prune_tip_frac)
    df = collapse_tiny_loops(df, strut_thickness)

    skeleton_df = df[['skeleton_point_id', 'x', 'y', 'z', 'r', 'theta',
                      'node_type', 'degree', 'neighbor_ids']].copy()

    skeleton_only_html = os.path.join(output_dir, 'skeleton_only.html')
    skeleton_csv       = os.path.join(output_dir, 'skeleton_points.csv')
    skeleton_df.to_csv(skeleton_csv, index=False)
    plot_skeleton_html(skeleton_df, skeleton_only_html,
                       title=f'{stent_name} skeleton', max_display=max_display)

    print(f"\nSkeleton finalised: {len(skeleton_df):,} points "
          f"({(skeleton_df['node_type'] == 'junction').sum()} junctions, "
          f"{(skeleton_df['degree'] == 1).sum()} endpoints)")
    print(f"[saved] {skeleton_csv}")
    print(f"[saved] {skeleton_only_html}  (inspect the final 3D skeleton)")
    return skeleton_df



def save_stent_features_and_views(skeleton_df, stent_df, stent_features,
                                  stent_centerline_direction, crown_edges, output_dir,
                                  stent_name, max_display=500_000):
    """Write the skeleton+cloud overlay and the final ``stent_features.json`` (Step 8).

    Renders ``skeleton_with_cloud.html`` (sparse cloud underlay) and folds the
    centreline direction + crown boundaries into ``stent_features.json``. Crown
    boundaries prefer the detected ``crown_edges``, else are derived from the
    per-point ``crown_id`` groups. Returns ``{features_path, crown_boundaries}``.
    """
    # Keep the cloud underlay sparse so the skeleton stays easy to see.
    cloud_display = int(max_display / 5)

    skeleton_cloud_html = os.path.join(output_dir, 'skeleton_with_cloud.html')
    plot_skeleton_with_cloud_html(skeleton_df, stent_df, skeleton_cloud_html,
                                  max_cloud=cloud_display)

    # crown boundaries (z): prefer detected crown_edges; else derive from crown_id groups.
    if crown_edges is not None and len(np.asarray(crown_edges).ravel()) >= 2:
        crown_boundaries = np.asarray(crown_edges, float).ravel()
    elif 'crown_id' in stent_df.columns:
        _g = (stent_df.groupby('crown_id')['z'].agg(['min', 'max', 'mean'])
                      .sort_values('mean'))
        _lo, _hi = _g['min'].to_numpy(), _g['max'].to_numpy()
        crown_boundaries = np.concatenate([[_lo[0]], 0.5 * (_hi[:-1] + _lo[1:]), [_hi[-1]]])
    else:
        crown_boundaries = None

    features_path = os.path.join(output_dir, 'stent_features.json')
    features_out = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in stent_features.items()}
    features_out['stent_centerline_direction'] = [
        float(v) for v in np.asarray(stent_centerline_direction).ravel()]
    features_out['crown_boundaries'] = (None if crown_boundaries is None
                                        else [float(e) for e in crown_boundaries])
    with open(features_path, 'w') as f:
        json.dump(features_out, f, indent=2)

    skeleton_csv = os.path.join(output_dir, 'skeleton_points.csv')
    splines_html = os.path.join(output_dir, 'splines.html')
    splines_json = os.path.join(output_dir, 'skeleton_splines.json')
    print("[saved] final outputs:")
    for p in (skeleton_csv, splines_html, splines_json, skeleton_cloud_html,
              features_path):
        print(f"   {p}")
    print(f"   crown_boundaries: "
          f"{0 if crown_boundaries is None else len(crown_boundaries)} values "
          f"folded into stent_features.json")
    print(f"\nThe results of the skeletonization and spline fitting have been saved "
          f"in {output_dir}. Please check the results. The stent 1D wireframe is now "
          f"ready for the simulation.")
    return {'features_path': features_path, 'crown_boundaries': crown_boundaries}


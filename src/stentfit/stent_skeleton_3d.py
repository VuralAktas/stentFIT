import numpy as np
import pandas as pd
import ast
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



def wrap_skeleton_to_3d(skel_arc, skel_z, skel_px, pixel_size, surf_df, r_mid,
                        circumference, strut_thickness, output_dir, stent_name,
                        wrap_max_surf=None,
                        prune_tip_frac=0, max_display=500_000, random_seed=0):
    """Wrap the 2D skeleton onto the local mid-surface and run graph cleanup (Step 6).

    Lifts each 2D point to 3D via a per-point local mid-surface radius, classifies
    node degree, then contracts junction clusters and prunes short spurs. On a huge
    cloud a downsampled throwaway copy is used for the radius estimate only
    (``wrap_max_surf``). Saves ``skeleton_points.csv`` + ``skeleton_only.html`` and
    returns the final SKELETON_DF.
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
    # The periodic raster keeps every strut continuous across the seam, so no seam
    # step is needed — analyze_skeleton_connectivity's 3D KD-tree already rejoins the
    # two coincident seam ends.
    df = collapse_junction_clusters(df)
    df = prune_skeleton_spurs(df, pixel_size, tip_frac=prune_tip_frac)
    
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


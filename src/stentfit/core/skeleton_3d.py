import numpy as np
import pandas as pd
import ast
import json
import os
from scipy.spatial import cKDTree

from .plotting import plot_skeleton_html, plot_skeleton_with_cloud_html


def adjust_skeleton_to_local_midsurface(
    skel_arc: np.ndarray,
    skel_z: np.ndarray,
    stent_df: pd.DataFrame,
    r_mid: float,
    circumference: float,
    search_radius: float,
) -> dict:
    """
    Lift each flat 2D skeleton point to 3D at its own local mid-wall radius.

    A skeleton point is placed at the angle and z given by its (arc, z)
    coordinates, but its radius is the mean radius of the nearby surface
    points within ``search_radius`` (falling back to ``r_mid`` if none are
    found) — not a fixed radius. This lets the 3D skeleton follow the
    stent's real surface undulation instead of sitting on a perfect
    cylinder. Surface points are triple-tiled along arc (one copy shifted
    left, one right) first, so the neighbour search sees across the seam
    without a periodic-distance correction.

    :param skel_arc: Flat arc-coordinates of the assembled 2D skeleton.
    :param skel_z: Flat z-coordinates of the assembled 2D skeleton.
    :param stent_df: Stent surface point cloud with ``theta``,
        ``z_cylindrical``, and ``r`` columns.
    :param r_mid: Mid-wall radius, used to convert arc back to angle, and as
        the fallback radius where no surface points are nearby.
    :param circumference: Full circumference at ``r_mid``, used to tile the
        surface points across the seam.
    :param search_radius: Radius, in the (arc, z) plane, used to gather
        nearby surface points for the local radius average.
    :returns: Dict with the lifted skeleton as a DataFrame
        (``df_skeleton_3d``, with ``theta``, ``x``, ``y``, ``z``, ``r``
        columns) and the same points as a plain ``(N, 3)`` array
        (``skeleton_points``).
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
    pixel_size: float | np.ndarray,
    neighbor_radius_factor: float = 1.8,
) -> pd.DataFrame:
    """
    Build the 3D skeleton graph and classify every point by its degree.

    Two points are neighbours if they're within ``neighbor_radius_factor``
    times the local pixel size — this is also what rejoins the two
    coincident seam ends left over from the periodic 2D raster, since they
    sit at the same 3D position once wrapped. ``pixel_size`` may be a single
    scalar (one skeletonisation resolution for the whole stent) or a
    per-point array (one entry per ring, from
    :func:`~stentfit.core.skeleton_2d.assemble_2d_skeleton`), in which case
    a pair only counts if it's within the coarser of the two points' radii.
    Each point's degree then sets its ``node_type``: 0 is ``isolated``, 1 is
    ``endpoint``, 2 is ``line``, and 3+ is ``junction``.

    :param df_skeleton_3d: 3D skeleton points with ``x``, ``y``, ``z`` columns,
        from :func:`adjust_skeleton_to_local_midsurface`.
    :param pixel_size: Pixel size the skeleton was rasterised at — a scalar,
        or a per-point array matching ``df_skeleton_3d``.
    :param neighbor_radius_factor: How many pixel-sizes apart two points may
        be and still count as neighbours.
    :returns: ``df_skeleton_3d``, with added ``skeleton_point_id``,
        ``degree``, ``node_type``, and ``neighbor_ids`` (list of connected
        point indices) columns.
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
    tip_frac: float = 0,
    max_spur_len: float | None = None,
    max_iter: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove short dead-end branches (spurs) from the 3D skeleton graph.

    Each round, every degree-1 endpoint not protected as a tip is walked
    along its dead-end chain until it hits a junction, another endpoint, or
    a loop. The whole chain is removed, unless it terminates on a real tip
    (kept so the stent's actual ends survive) or is longer than
    ``max_spur_len`` (kept as a real branch, not a stray spur). This repeats
    for up to ``max_iter`` rounds, since removing one spur can expose a new
    endpoint one step further in. Degrees and node types are recomputed
    afterward.

    :param df_connectivity: 3D skeleton graph with ``x``, ``y``, ``z``, and
        ``neighbor_ids`` columns, from :func:`analyze_skeleton_connectivity`
        or :func:`collapse_junction_clusters`.
    :param tip_frac: Fraction of the stent's z-range, from each axial end,
        protected as a real tip — endpoints in that band are never pruned.
        ``0`` protects nothing, so every dead-end is prunable.
    :param max_spur_len: Chains longer than this are kept as real branches
        instead of being pruned. ``None`` prunes regardless of length.
    :param max_iter: Maximum number of pruning rounds.
    :param verbose: Print how many nodes/branches were removed, and how many
        non-tip endpoints remain.
    :returns: The skeleton graph with spurs removed, and
        ``skeleton_point_id``, ``degree``, ``node_type``, ``neighbor_ids``
        recomputed to match.
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
    """
    Contract each blob of edge-connected junction nodes to a single centroid.

    A real strut crossing should be one junction point, but the raster ->
    thin -> lift pipeline can leave a small cluster of several
    degree-3+ nodes sitting right next to each other instead. Every such
    cluster (found by walking edges between junction nodes only) is
    collapsed: one member survives at the cluster's centroid, every external
    neighbour of any member is rewired to point at the survivor, and the
    rest of the cluster is dropped. Degrees and node types are recomputed
    afterward, since collapsing can turn a junction into a plain line point.

    :param df_connectivity: 3D skeleton graph with ``x``, ``y``, ``z``, and
        ``neighbor_ids`` columns, from :func:`analyze_skeleton_connectivity`.
    :param verbose: Print how many clusters were collapsed and the resulting
        node counts.
    :returns: The skeleton graph with junction clusters collapsed, and
        ``skeleton_point_id``, ``degree``, ``node_type``, ``neighbor_ids``
        recomputed to match.
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



def wrap_skeleton_to_3d(skel_arc: np.ndarray,
                        skel_z: np.ndarray,
                        skel_px: np.ndarray,
                        surf_df: pd.DataFrame,
                        r_mid: float,
                        circumference: float,
                        strut_thickness: float,
                        output_dir: str,
                        stent_name: str,
                        wrap_max_surf: int | None = None,
                        prune_tip_frac: float = 0,
                        max_display: int = 500_000,
                        random_seed: int = 0) -> pd.DataFrame:
    """
    Lift the flat 2D skeleton onto the 3D stent surface and clean up its graph.

    Each point is lifted to 3D using a per-point local mid-surface radius
    (:func:`adjust_skeleton_to_local_midsurface`), then the graph is
    classified by node degree and its 3D KD-tree rejoins the coincident seam
    ends (:func:`analyze_skeleton_connectivity`). Junction blobs are
    contracted to a centroid (:func:`collapse_junction_clusters`) and short
    dead-ends are pruned (:func:`prune_skeleton_spurs`). Saves
    ``skeleton_points.csv`` and ``skeleton_only.html``.

    :param skel_arc: Flat arc-coordinates of the assembled 2D skeleton.
    :param skel_z: Flat z-coordinates of the assembled 2D skeleton.
    :param skel_px: Per-point pixel size, from :func:`~stentfit.core.skeleton_2d.assemble_2d_skeleton`.
    :param surf_df: Stent surface point cloud, used to find each skeleton
        point's local mid-surface radius.
    :param r_mid: Mid-wall radius.
    :param circumference: Full circumference at ``r_mid``.
    :param strut_thickness: Strut thickness, used as the local-midsurface search radius.
    :param output_dir: Folder the CSV and HTML view are written into.
    :param stent_name: Name used to label outputs and plots.
    :param wrap_max_surf: Maximum surface points used for the wrap. ``None``
        uses every surface point.
    :param prune_tip_frac: Fraction of each curve tip to prune as a spur.
    :param max_display: Maximum number of points drawn in the HTML view.
    :param random_seed: Seed for the surface downsampling, when ``wrap_max_surf`` applies.
    :returns: The final 3D skeleton graph, with ``skeleton_point_id``, ``x``,
        ``y``, ``z``, ``r``, ``theta``, ``node_type``, ``degree``, and
        ``neighbor_ids`` columns.
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
    df = collapse_junction_clusters(df)
    df = prune_skeleton_spurs(df, tip_frac=prune_tip_frac)
    
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



def save_stent_features_and_views(skeleton_df: pd.DataFrame,
                                  stent_df: pd.DataFrame,
                                  stent_features: dict,
                                  stent_centerline_direction: np.ndarray,
                                  ring_edges: np.ndarray | None,
                                  output_dir: str,
                                  max_display: int = 500_000) -> dict:
    """
    Write the final ``stent_features.json`` and the skeleton-with-cloud view.

    Draws the 3D skeleton overlaid on a sparse cloud of the original surface
    points (``skeleton_with_cloud.html``), then folds
    ``stent_centerline_direction`` and the ring z-boundaries into
    ``stent_features.json``. Ring boundaries come from ``ring_edges`` if
    given, else are derived from ``stent_df``'s ``ring_id`` groups (the
    midpoint between each pair of neighbouring rings). Also prints a summary
    of every file this pipeline run has produced — the skeleton CSV and
    spline exports were already written by earlier steps
    (:func:`wrap_skeleton_to_3d`, :func:`~stentfit.core.splines.fit_skeleton_splines`);
    only the JSON and cloud view are written here.

    :param skeleton_df: Final 3D skeleton graph.
    :param stent_df: Stent surface point cloud, used as the cloud underlay
        and, if needed, to derive ring boundaries from its ``ring_id`` column.
    :param stent_features: Stent features dict to write out (length,
        diameter, strut_thickness, ...).
    :param stent_centerline_direction: Stent centreline unit vector, folded
        into the saved features.
    :param ring_edges: Z-boundaries between rings. ``None`` (or fewer than 2
        values) falls back to deriving them from ``stent_df``.
    :param output_dir: Folder the JSON and HTML view are written into.
    :param max_display: Maximum skeleton points drawn in the HTML view; the
        surface cloud underlay is drawn at a fifth of this.
    :returns: Dict with the path to the written JSON (``features_path``) and
        the ring z-boundaries actually used (``ring_boundaries``, ``None``
        if neither source was available).
    """
    # Keep the cloud underlay sparse so the skeleton stays easy to see.
    cloud_display = int(max_display / 5)

    skeleton_cloud_html = os.path.join(output_dir, 'skeleton_with_cloud.html')
    plot_skeleton_with_cloud_html(skeleton_df, stent_df, skeleton_cloud_html,
                                  max_cloud=cloud_display)

    # ring boundaries (z): prefer detected ring_edges; else derive from ring_id groups.
    if ring_edges is not None and len(np.asarray(ring_edges).ravel()) >= 2:
        ring_boundaries = np.asarray(ring_edges, float).ravel()
    elif 'ring_id' in stent_df.columns:
        _g = (stent_df.groupby('ring_id')['z'].agg(['min', 'max', 'mean'])
                      .sort_values('mean'))
        _lo, _hi = _g['min'].to_numpy(), _g['max'].to_numpy()
        ring_boundaries = np.concatenate([[_lo[0]], 0.5 * (_hi[:-1] + _lo[1:]), [_hi[-1]]])
    else:
        ring_boundaries = None

    features_path = os.path.join(output_dir, 'stent_features.json')
    features_out = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in stent_features.items()}
    features_out['stent_centerline_direction'] = [
        float(v) for v in np.asarray(stent_centerline_direction).ravel()]
    features_out['ring_boundaries'] = (None if ring_boundaries is None
                                        else [float(e) for e in ring_boundaries])
    with open(features_path, 'w') as f:
        json.dump(features_out, f, indent=2)

    skeleton_csv = os.path.join(output_dir, 'skeleton_points.csv')
    splines_html = os.path.join(output_dir, 'splines.html')
    splines_json = os.path.join(output_dir, 'skeleton_splines.json')
    print("[saved] final outputs:")
    for p in (skeleton_csv, splines_html, splines_json, skeleton_cloud_html,
              features_path):
        print(f"   {p}")
    print(f"   ring_boundaries: "
          f"{0 if ring_boundaries is None else len(ring_boundaries)} values "
          f"folded into stent_features.json")
    print(f"\nThe results of the skeletonization and spline fitting have been saved "
          f"in {output_dir}. Please check the results. The stent 1D wireframe is now "
          f"ready for the simulation.")
    return {'features_path': features_path, 'ring_boundaries': ring_boundaries}


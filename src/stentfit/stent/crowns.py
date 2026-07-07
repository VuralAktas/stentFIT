import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .plotting import plot_points_3d_html, plot_crown_dips_html


def find_crowns(
    stent_df: pd.DataFrame,
    strut_thickness: float,
    min_crown_frac: float = 0.2,      # crown < frac * (90th-pct point-count) = "tiny"
    show_plots: bool = False,
    out_path: str = None,             # if set, save the dip-detection plot here (PNG)
) -> dict:
    """Split the stent into crown-to-crown bands and number them.

    The point count per z-slice dips at every crown where the struts converge.
    find_crowns locates those dips, cuts the stent at the dip z-values, labels
    the connected piece in each band, and absorbs any undersized spurious crown
    into the nearest real crown. The crown number goes in stent_df['crown_id'].

    Returns stent_df (with 'crown_id'), crown_edges (band boundaries), n_crowns
    and conn_radius_3d.
    """

    pts3d     = stent_df[['x', 'y', 'z']].values
    z_vals    = stent_df['z_cylindrical'].values
    n_samples = len(pts3d)

    # conn_radius_3d: min of density-based and strut-based estimates
    r_mean         = stent_df['r'].values.mean()
    z_range        = z_vals.max() - z_vals.min()
    surface_area   = 2 * np.pi * r_mean * z_range
    avg_spacing    = np.sqrt(surface_area / n_samples)
    conn_radius_3d = min(3.0 * avg_spacing, strut_thickness)

    # Step 1: detect crown dips (each dip is a cut)
    n_diag  = 200
    edges_d = np.linspace(z_vals.min(), z_vals.max(), n_diag + 1)
    sid_d   = np.clip(np.digitize(z_vals, edges_d) - 1, 0, n_diag - 1)
    npts_d  = np.bincount(sid_d, minlength=n_diag).astype(float)
    npts_s  = uniform_filter1d(npts_d, size=5)

    # keep only the deepest dips (true crowns), not shallow local notches
    depth_frac   = 0.4
    prom         = 0.25 * (npts_s.max() - npts_s.min())
    depth_thresh = npts_s.min() + depth_frac * (np.median(npts_s) - npts_s.min())
    dips, _      = find_peaks(-npts_s, prominence=prom, distance=3,
                              height=-depth_thresh)        # dip value <= depth_thresh
    n_dips       = len(dips)

    # Step 2: cut at the dip z-values (n_dips cuts -> n_dips+1 bands)
    dip_z   = np.sort((0.5 * (edges_d[:-1] + edges_d[1:]))[dips])
    z_edges = np.concatenate([[z_vals.min()], dip_z, [z_vals.max()]])
    n_bands = len(z_edges) - 1
    '''print(f"[crown] crown dips={n_dips} -> cutting at {len(dip_z)} z-values "
          f"-> {n_bands} bands")'''

    if show_plots or out_path is not None:
        zc_d = 0.5 * (edges_d[:-1] + edges_d[1:])
        fig, axd = plt.subplots(figsize=(11, 3))
        axd.plot(zc_d, npts_s, color='gray', label='points/slice (smoothed)')
        axd.plot(zc_d[dips], npts_s[dips], 'rv', ms=9, label=f'{n_dips} crown dips')
        axd.axhline(depth_thresh, color='red', ls=':', lw=1,
                    label=f'depth cutoff ({depth_frac:.2f})')
        axd.set_xlabel('z_cylindrical'); axd.set_ylabel('points / slice')
        axd.set_title(f'Crown dips -> {n_bands} crown-to-crown bands')
        axd.legend(); axd.grid(True, alpha=0.3)
        plt.tight_layout()
        if out_path is not None:
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # Step 3: label connected pieces within each band
    slc   = np.clip(np.digitize(z_vals, z_edges) - 1, 0, n_bands - 1)

    tree  = cKDTree(pts3d)
    all_p = tree.query_pairs(r=conn_radius_3d, output_type='ndarray')

    same  = slc[all_p[:, 0]] == slc[all_p[:, 1]]
    pairs = all_p[same]

    adj = csr_matrix(
        (np.ones(2 * len(pairs), np.uint8),
         (np.concatenate([pairs[:, 0], pairs[:, 1]]),
          np.concatenate([pairs[:, 1], pairs[:, 0]]))),
        shape=(n_samples, n_samples))
    _, crown_id = connected_components(adj, directed=False)

    # Step 4: absorb spurious tiny crowns into the nearest real crown
    # (size against the 90th-pct count, not the median, since fragments can outnumber crowns)
    ids, counts = np.unique(crown_id, return_counts=True)
    size_thresh = max(min_crown_frac * float(np.percentile(counts, 90)), 1.0)
    small_ids   = ids[counts <  size_thresh]
    normal_ids  = ids[counts >= size_thresh]

    if len(small_ids) and len(normal_ids):
        s_mask = np.isin(crown_id, small_ids)
        s_idx  = np.where(s_mask)[0]
        n_idx  = np.where(~s_mask)[0]

        # nearest normal point (-> its crown) and its distance, per tiny point
        s_dist, nn = cKDTree(pts3d[n_idx]).query(pts3d[s_idx])
        s_ncrown   = crown_id[n_idx][nn]
        s_crown    = crown_id[s_idx]

        # one target per tiny crown = crown of its single closest normal point
        pick   = (pd.DataFrame({'scrown': s_crown, 'dist': s_dist})
                    .groupby('scrown')['dist'].idxmin())          # position in s_* arrays
        target = dict(zip(pick.index.to_numpy(), s_ncrown[pick.to_numpy()]))

        crown_id        = crown_id.copy()
        crown_id[s_idx] = np.array([target[c] for c in s_crown], dtype=crown_id.dtype)
        print(f"[crown] absorbed {len(small_ids)} tiny crowns "
              f"(<{size_thresh:.0f} pts) into nearest normal crowns")
    else:
        '''print(f"[crown] no tiny crowns to absorb (threshold {size_thresh:.0f} pts)")'''

    # Step 5: renumber crowns 1..n in axial order (lowest mean z = crown 1)
    labels  = np.unique(crown_id)
    mean_z  = np.array([z_vals[crown_id == lbl].mean() for lbl in labels])
    order   = labels[np.argsort(mean_z)]                     # labels low-z -> high-z
    relabel = {int(lbl): i + 1 for i, lbl in enumerate(order)}
    crown_f = np.array([relabel[int(c)] for c in crown_id], dtype=np.int32)
    C       = int(crown_f.max())

    stent_df['crown_id'] = crown_f

    return {
        'stent_df'          : stent_df,
        'crown_edges'       : z_edges,
        'n_crowns'          : C,
        'conn_radius_3d'    : conn_radius_3d,
        # dip-detection diagnostic data (for interactive HTML plot)
        'dip_z_centers'     : 0.5 * (edges_d[:-1] + edges_d[1:]),
        'dip_counts_smoothed': npts_s,
        'dip_indices'       : dips,
        'dip_depth_thresh'  : depth_thresh,
        'n_bands'           : n_bands,
    }



def segment_stent(
    stent_df: pd.DataFrame,
    strut_thickness: float,
    conn_radius_3d: float,
    n_sub_per_crown: int = 3,         # cut each crown into this many equal z-pieces
    min_region_frac: float = 0.2,     # region < frac * median point-count = "tiny"
) -> dict:
    """Slice each crown into z-pieces, label components, clean tiny regions.

    Each crown band (the 'crown_id' column from find_crowns) is cut into
    n_sub_per_crown equal z-pieces; connected components are found within each
    (crown, piece) slice. Tiny non-pipe-like regions are absorbed into the
    nearest normal region.

    Returns stent_df (with 'region'), region_allowed adjacency, whole_stent_region,
    n_regions and conn_radius_3d.
    """
    
    if 'crown_id' not in stent_df.columns:
        raise KeyError("segment_stent now slices per crown - run crown_stent first "
                       "so stent_df has a 'crown_id' column.")

    whole_stent_region = np.array([[True]])  # all points belong to the same whole-stent region

    pts3d     = stent_df[['x', 'y', 'z']].values
    z_vals    = stent_df['z_cylindrical'].values
    crowns    = stent_df['crown_id'].values
    n_samples = len(pts3d)

    # Step 1: cut each crown into n_sub_per_crown equal z-pieces
    slc      = np.empty(n_samples, dtype=np.int64)
    next_id  = 0
    for c in np.unique(crowns):
        m  = crowns == c
        zc = z_vals[m]
        if zc.max() > zc.min():
            edges = np.linspace(zc.min(), zc.max(), n_sub_per_crown + 1)
            sub   = np.clip(np.digitize(zc, edges) - 1, 0, n_sub_per_crown - 1)
        else:
            sub   = np.zeros(m.sum(), dtype=np.int64)   # degenerate crown -> single piece
        slc[m]  = next_id + sub
        next_id += n_sub_per_crown
    n_slices = len(np.unique(slc))
    print(f"[segmentation] {len(np.unique(crowns))} crowns x {n_sub_per_crown} pieces "
          f"-> {n_slices} z-slices")

    # Step 2: connected components within each slice
    tree  = cKDTree(pts3d)
    all_p = tree.query_pairs(r=conn_radius_3d, output_type='ndarray')

    same  = slc[all_p[:, 0]] == slc[all_p[:, 1]]
    pairs = all_p[same]
    cross = all_p[~same]          # kept for region_allowed (no second query)

    adj = csr_matrix(
        (np.ones(2 * len(pairs), np.uint8),
         (np.concatenate([pairs[:, 0], pairs[:, 1]]),
          np.concatenate([pairs[:, 1], pairs[:, 0]]))),
        shape=(n_samples, n_samples))
    _, region = connected_components(adj, directed=False)

    # Step 3: absorb tiny regions into the nearest normal region
    ids, counts = np.unique(region, return_counts=True)
    size_thresh = max(min_region_frac * float(np.median(counts)), 1.0)
    small_ids   = ids[counts <  size_thresh]
    normal_ids  = ids[counts >= size_thresh]

    if len(small_ids) and len(normal_ids):
        s_mask = np.isin(region, small_ids)
        s_idx  = np.where(s_mask)[0]
        n_idx  = np.where(~s_mask)[0]

        # nearest normal point (-> its region) and its distance, per tiny point
        s_dist, nn = cKDTree(pts3d[n_idx]).query(pts3d[s_idx])
        s_nreg     = region[n_idx][nn]
        s_reg      = region[s_idx]

        # one target per tiny region = region of its single closest normal point
        pick   = (pd.DataFrame({'sreg': s_reg, 'dist': s_dist})
                    .groupby('sreg')['dist'].idxmin())          # position in s_* arrays
        target = dict(zip(pick.index.to_numpy(), s_nreg[pick.to_numpy()]))

        region        = region.copy()
        region[s_idx] = np.array([target[r] for r in s_reg], dtype=region.dtype)
        print(f"[segmentation] NOTE: absorbed {len(small_ids)} tiny regions "
              f"(<{size_thresh:.0f} pts) into nearest normal regions")
    else:
        '''print(f"[segmentation] no tiny regions to absorb (threshold {size_thresh:.0f} pts)")'''

    # Step 4: compact renumbering
    _, inv   = np.unique(region, return_inverse=True)
    region_f = (inv + 1).astype(np.int32)   # 1-based, dense
    R        = int(region_f.max())
    print(f"[segmentation] {R} regions are found")

    stent_df           = stent_df.copy()
    stent_df['region'] = region_f

    # Step 5: region adjacency from the cross-slice pairs
    region_allowed = np.zeros((R + 1, R + 1), dtype=bool)
    np.fill_diagonal(region_allowed, True)
    if len(cross):
        ra, rb = region_f[cross[:, 0]], region_f[cross[:, 1]]
        diff   = ra != rb
        if diff.any():
            adj_pairs = np.unique(np.sort(np.column_stack([ra[diff], rb[diff]]), axis=1), axis=0)
            region_allowed[adj_pairs[:, 0], adj_pairs[:, 1]] = True
            region_allowed[adj_pairs[:, 1], adj_pairs[:, 0]] = True
    region_allowed[0, :] = True
    region_allowed[:, 0] = True

    return {
        'stent_df'          : stent_df,
        'region_allowed'    : region_allowed,
        'whole_stent_region': whole_stent_region,
        'n_regions'         : R,
        'conn_radius_3d'    : conn_radius_3d,
    }



def detect_crowns(stent_df, stent_features, stent_name, output_dir, max_display=500_000):
    """Auto-detect crowns and label every point with a ``crown_id`` (Step 3).

    Runs ``find_crowns`` (no count enforcement), renders the 3D crown assignment
    (categorical colours) and the interactive dip-detection plot, and saves
    ``crown_points.csv``. Returns ``{stent_df, crown_edges, conn_radius_3d, n_crowns}``.
    """
    strut_thickness = stent_features['strut_thickness']

    crown_dips_html = os.path.join(output_dir, 'crown_dips.html')
    crown_html      = os.path.join(output_dir, 'crown_assignment.html')
    crown_csv       = os.path.join(output_dir, 'crown_points.csv')

    crown_res      = find_crowns(stent_df, strut_thickness=strut_thickness)
    stent_df       = crown_res['stent_df']          # now has 'crown_id'
    crown_edges    = crown_res['crown_edges']
    conn_radius_3d = crown_res['conn_radius_3d']
    n_crowns       = crown_res['n_crowns']
    print(f"Detected {n_crowns} crowns.")

    plot_crown_dips_html(crown_res, crown_dips_html)
    plot_points_3d_html(stent_df, 'point_id', crown_html, color_col='crown_id',
                        title=f'{stent_name} crowns ({n_crowns})', max_display=max_display,
                        categorical=True)
    print(f"[plot] {crown_html}")
    print(f"[plot] {crown_dips_html}")

    stent_df[['point_id', 'r', 'theta', 'z_cylindrical', 'x', 'y', 'z', 'crown_id']].to_csv(
        crown_csv, index=False)
    print(f"[saved] {crown_csv}")

    return {'stent_df': stent_df, 'crown_edges': crown_edges,
            'conn_radius_3d': conn_radius_3d, 'n_crowns': n_crowns}


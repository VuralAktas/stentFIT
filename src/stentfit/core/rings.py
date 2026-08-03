import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .plotting import plot_points_3d_html, plot_ring_dips_html


def find_rings(
    stent_df: pd.DataFrame,
    strut_thickness: float,
    n_rings: int | None = None,
    show_plots: bool = False,
    out_path: str | None = None,
) -> dict:
    """
    Split the stent point cloud into axial rings and label every point.

    Builds a smoothed point-count profile along z; the profile dips where
    struts converge, marking candidate ring boundaries. When ``n_rings`` is
    not forced, the ring pitch is read from the profile's autocorrelation
    (falling back to the raw dip count if no clear period exists). An even
    grid of boundaries is then laid down and only snapped onto a nearby dip
    when it sits close, so the rings stay even overall.

    :param stent_df: Stent point cloud with ``x``, ``y``, ``z``,
        ``z_cylindrical``, and ``r`` columns.
    :param strut_thickness: Strut thickness, used as an upper bound for the
        3D connectivity radius.
    :param n_rings: Ring count to force. ``None`` auto-detects it from the
        z-profile.
    :param show_plots: Show the dip-detection diagnostic plot inline (matplotlib).
    :param out_path: File path to save the dip-detection plot as a PNG.
        ``None`` skips saving.
    :returns: Dict with the point cloud labelled with ``ring_id``
        (``stent_df``), the ring z-boundaries (``ring_edges``), the ring
        count (``n_rings``), the 3D connectivity radius (``conn_radius_3d``),
        and the dip-detection diagnostics (``dip_z_centers``,
        ``dip_counts_smoothed``, ``dip_indices``, ``dip_depth_thresh``,
        ``boundary_z``, ``n_bands``) used by
        :func:`~stentfit.core.plotting.plot_ring_dips_html`.
    """

    pts3d     = stent_df[['x', 'y', 'z']].values
    z_vals    = stent_df['z_cylindrical'].values
    n_samples = len(pts3d)

    # conn_radius_3d: min of density-based and strut-based estimates
    r_mean         = stent_df['r'].values.mean()
    z_min, z_max   = z_vals.min(), z_vals.max()
    z_range        = z_max - z_min
    surface_area   = 2 * np.pi * r_mean * z_range
    avg_spacing    = np.sqrt(surface_area / n_samples)
    conn_radius_3d = min(3.0 * avg_spacing, strut_thickness)

    # Step 1: z point-count profile (dips where struts converge / links only)
    n_diag  = 200
    edges_d = np.linspace(z_min, z_max, n_diag + 1)
    zc_d    = 0.5 * (edges_d[:-1] + edges_d[1:])
    slice_w = z_range / n_diag
    sid_d   = np.clip(np.digitize(z_vals, edges_d) - 1, 0, n_diag - 1)
    npts_d  = np.bincount(sid_d, minlength=n_diag).astype(float)
    npts_s  = uniform_filter1d(npts_d, size=5)

    # Step 2: candidate dips (real strut-free gaps); we only ever cut AT these
    depth_frac   = 0.4
    prom         = 0.25 * (npts_s.max() - npts_s.min())
    depth_thresh = npts_s.min() + depth_frac * (np.median(npts_s) - npts_s.min())
    dips, _      = find_peaks(-npts_s, prominence=prom, distance=3,
                              height=-depth_thresh)        # dip value <= depth_thresh
    n_dips       = len(dips)
    dip_z_all    = np.sort(zc_d[dips]) if n_dips else np.array([])

    # Step 3: decide how many rings N. If the caller forces n_rings, use it;
    # otherwise auto-detect the ring pitch from the profile's autocorrelation.
    # Sliding npts_s against itself, the lag of the *strongest* peak is the
    # dominant repeat distance (one full ring). The strongest peak, not the
    # first, because the profile also has finer within-ring structure whose
    # short-lag peaks would over-segment the stent.
    if n_rings is not None:
        N       = max(1, int(n_rings))
        pitch_z = z_range / N
    else:
        prof     = npts_s - npts_s.mean()
        ac       = np.correlate(prof, prof, mode='full')[n_diag - 1:]   # lags 0 .. n_diag-1
        ac_peaks, _ = find_peaks(ac, distance=3)
        # keep only peaks whose pitch yields a plausible ring count (2 .. n_diag//4)
        if len(ac_peaks):
            lag_N   = np.round(z_range / (ac_peaks * slice_w)).astype(int)
            valid   = ac_peaks[(lag_N >= 2) & (lag_N <= n_diag // 4)]
        else:
            valid   = np.array([], dtype=int)
        if len(valid):
            pitch_slices = int(valid[np.argmax(ac[valid])])     # strongest valid repeat
            pitch_z      = pitch_slices * slice_w
            N            = max(1, int(round(z_range / pitch_z)))
        else:
            # fallback: no clear period -> trust the raw dip count
            N            = max(1, n_dips + 1)
            pitch_z      = z_range / N

    # Step 4: lay an even grid of N-1 boundaries. Equal width is the priority, so
    # only nudge a boundary onto a nearby dip when the gap sits very close (a
    # minor adjustment); a far-off dip is ignored so the rings stay even.
    snap_tol = 0.2 * pitch_z
    ideal    = z_min + np.arange(1, N) * (z_range / N)
    if len(dip_z_all):
        snapped = []
        for zi in ideal:
            j = int(np.argmin(np.abs(dip_z_all - zi)))
            snapped.append(dip_z_all[j] if abs(dip_z_all[j] - zi) < snap_tol else zi)
        dip_z = np.unique(np.round(np.sort(snapped), 12))
    else:
        dip_z = ideal
    z_edges = np.concatenate([[z_min], dip_z, [z_max]])
    n_bands = len(z_edges) - 1

    if show_plots or out_path is not None:
        fig, axd = plt.subplots(figsize=(11, 3))
        axd.plot(zc_d, npts_s, color='gray', label='points/slice (smoothed)')
        if n_dips:
            axd.plot(dip_z_all, npts_s[dips], 'v', color='lightgray', ms=7,
                     label=f'{n_dips} candidate dips')
        for zb in dip_z:
            axd.axvline(zb, color='red', lw=1.2)
        axd.axhline(depth_thresh, color='red', ls=':', lw=1,
                    label=f'depth cutoff ({depth_frac:.2f})')
        axd.set_xlabel('z_cylindrical'); axd.set_ylabel('points / slice')
        axd.set_title(f'{n_bands} rings (pitch~{pitch_z:.4g})')
        axd.legend(); axd.grid(True, alpha=0.3)
        plt.tight_layout()
        if out_path is not None:
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # Step 5: label every point by axial band -> ring is a full-circumference z-slab
    ring_f = (np.clip(np.digitize(z_vals, z_edges) - 1, 0, n_bands - 1) + 1).astype(np.int32)
    C       = int(ring_f.max())
    stent_df['ring_id'] = ring_f

    return {
        'stent_df'          : stent_df,
        'ring_edges'       : z_edges,
        'n_rings'          : C,
        'conn_radius_3d'    : conn_radius_3d,
        # dip-detection diagnostic data (for interactive HTML plot)
        'dip_z_centers'     : zc_d,
        'dip_counts_smoothed': npts_s,
        'dip_indices'       : dips,
        'dip_depth_thresh'  : depth_thresh,
        'boundary_z'        : dip_z,          # the N-1 boundaries actually used
        'n_bands'           : n_bands,
    }



def segment_stent(
    stent_df: pd.DataFrame,
    strut_thickness: float,
    conn_radius_3d: float,
    n_sub_per_ring: int = 3,
    min_region_frac: float = 0.2,
) -> dict:
    """
    Split the whole stent into small 3D-connected regions, once for all rings.

    Each ring is first cut into ``n_sub_per_ring`` equal z-pieces. Within each
    piece, points closer than ``conn_radius_3d`` are grouped into connected
    regions, and any region smaller than ``min_region_frac`` of the median
    region size is absorbed into its nearest normal-sized neighbour. The
    result is a per-point ``region`` label plus which region pairs are
    actually adjacent in 3D (``region_allowed``) — used later by
    :func:`~stentfit.core.skeleton_2d.check_skeleton_quality` to tell a real
    strut connection from a false one bridging two unrelated regions.

    :param stent_df: Stent point cloud with a ``ring_id`` column (from
        :func:`detect_rings`).
    :param strut_thickness: Strut thickness, used indirectly via ``conn_radius_3d``.
    :param conn_radius_3d: 3D connectivity radius; points closer than this are
        grouped into the same region.
    :param n_sub_per_ring: Number of equal z-pieces each ring is cut into
        before segmenting, so a single ring's points still split into several
        regions.
    :param min_region_frac: Regions smaller than this fraction of the median
        region point-count are absorbed into their nearest normal region.
    :returns: Dict with the point cloud labelled with ``region`` (``stent_df``),
        the region-adjacency matrix (``region_allowed``), the region count
        (``n_regions``), and the ``conn_radius_3d`` passed through unchanged.
    """
    if 'ring_id' not in stent_df.columns:
        raise KeyError("segment_stent now slices per ring - run ring_stent first "
                       "so stent_df has a 'ring_id' column.")

    whole_stent_region = np.array([[True]])  # all points belong to the same whole-stent region

    pts3d     = stent_df[['x', 'y', 'z']].values
    z_vals    = stent_df['z_cylindrical'].values
    rings    = stent_df['ring_id'].values
    n_samples = len(pts3d)

    # Step 1: cut each ring into n_sub_per_ring equal z-pieces
    slc      = np.empty(n_samples, dtype=np.int64)
    next_id  = 0
    for c in np.unique(rings):
        m  = rings == c
        zc = z_vals[m]
        if zc.max() > zc.min():
            edges = np.linspace(zc.min(), zc.max(), n_sub_per_ring + 1)
            sub   = np.clip(np.digitize(zc, edges) - 1, 0, n_sub_per_ring - 1)
        else:
            sub   = np.zeros(m.sum(), dtype=np.int64)   # degenerate ring -> single piece
        slc[m]  = next_id + sub
        next_id += n_sub_per_ring
    n_slices = len(np.unique(slc))
    print(f"[segmentation] {len(np.unique(rings))} rings x {n_sub_per_ring} pieces "
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



def detect_rings(
        stent_df: pd.DataFrame,
        stent_features: dict,
        stent_name: str,
        output_dir: str,
        max_display: int = 500_000,
        n_rings: int | None = None) -> dict:
    """
    Split the stent point cloud into rings and let the user confirm the count.

    Runs :func:`find_rings` to locate the point-count dips where struts
    converge, then draws the dip profile and the ring-colored point cloud so
    the user can check the split. If ``n_rings`` is not given, the user is
    asked to accept the detected count or type a different one to redo the
    split with a forced count. Saves ``ring_points.csv``, ``ring_dips.html``,
    and ``ring_assignment.html`` into ``output_dir``.

    :param stent_df: Stent point cloud, as returned by :func:`sample_stent_points`.
    :param stent_features: Stent features dict; only ``strut_thickness`` is used.
    :param stent_name: Name used to label outputs and plots.
    :param output_dir: Folder the CSV and HTML views are written into.
    :param max_display: Maximum number of points drawn in the ring-assignment view.
    :param n_rings: Ring count to force. ``None`` auto-detects it, then still
        prompts the user to accept or override it.
    :returns: Dict with the point cloud labelled with ``ring_id``
        (``stent_df``), the ``ring_edges`` (z-boundaries), ``conn_radius_3d``,
        and the final ``n_rings``.
    """
    strut_thickness = stent_features['strut_thickness']

    ring_dips_html = os.path.join(output_dir, 'ring_dips.html')
    ring_html      = os.path.join(output_dir, 'ring_assignment.html')
    ring_csv       = os.path.join(output_dir, 'ring_points.csv')

    forced = n_rings   # None on the first pass -> auto-detect
    while True:
        ring_res      = find_rings(stent_df, strut_thickness=strut_thickness,
                                     n_rings=forced)
        stent_df       = ring_res['stent_df']          # now has 'ring_id'
        ring_edges    = ring_res['ring_edges']
        conn_radius_3d = ring_res['conn_radius_3d']
        n_found        = ring_res['n_rings']
        how            = 'forced' if forced is not None else 'auto-detected'
        print(f"Detected {n_found} rings ({how}).")

        plot_ring_dips_html(ring_res, ring_dips_html)
        plot_points_3d_html(stent_df, 'point_id', ring_html, color_col='ring_id',
                            title=f'{stent_name} rings ({n_found})', max_display=max_display,
                            categorical=True)
        print(f"[plot] {ring_html}")
        print(f"[plot] {ring_dips_html}")

        # let the user inspect the plots and redo with a different count if wanted
        try:
            ans = input(f"Accept {n_found} rings? "
                        f"[Enter = accept, or type a new count e.g. 3 to redo]: ").strip()
        except EOFError:
            ans = ''            # no console -> accept the current (auto) result
        if not ans:
            break
        try:
            forced = max(1, int(ans))
        except ValueError:
            print(f"  '{ans}' is not a whole number — keeping {n_found} rings.")
            break

    n_rings = n_found
    stent_df[['point_id', 'r', 'theta', 'z_cylindrical', 'x', 'y', 'z', 'ring_id']].to_csv(
        ring_csv, index=False)
    print(f"[saved] {ring_csv}")

    return {'stent_df': stent_df, 'ring_edges': ring_edges,
            'conn_radius_3d': conn_radius_3d, 'n_rings': n_rings}


import numpy as np
import trimesh
from trimesh.path.entities import Line
import matplotlib.pyplot as plt
import pandas as pd
import ast
import datetime
import pathlib
import json
import time
from collections import deque, Counter
from sklearn.cluster import DBSCAN

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from skimage.morphology import skeletonize, dilation, closing, disk

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()


def compute_pre_stent_size_ratio(mesh: trimesh.Trimesh) -> dict:
    """Estimate stent length, diameter and their ratio from the raw mesh."""
    pts      = mesh.vertices
    centered = pts - pts.mean(axis=0)

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis   = eigvecs[:, np.argmax(eigvals)]

    projections = centered @ principal_axis
    length      = projections.max() - projections.min()

    axial_component  = np.outer(projections, principal_axis)
    radial_distances = np.linalg.norm(centered - axial_component, axis=1)
    diameter         = 2.0 * radial_distances.max()

    return {
        'length'    : length,
        'diameter'  : diameter,
        'size_ratio': length / diameter,
    }

def preprocess_stent(
    mesh: trimesh.Trimesh,
    n_samples: int,
    samples_per_face: int,
    n_thickness_slices: int,
    slice_cutoff: int,
    thickness_calc_plot: bool,
    remove_supports: bool,
    random_seed: int = None,
) -> dict:
    """Sample the mesh, align it to z, and extract stent features.

    Samples a point cloud from the surface, rotates the PCA axis onto [0,0,1],
    builds a cylindrical-coordinate DataFrame, optionally strips the end
    supports, then measures length, diameter, radii and strut thickness.
    """

    # Step 1: sample the surface (random_seed=None draws fresh; int = reproducible)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples, seed=random_seed)

    # Step 2: PCA centroid and main axis
    pca_mean  = pts.mean(axis=0)
    centered  = pts - pca_mean
    cov       = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    pca_axis  = eigvecs[:, np.argmax(eigvals)]

    center_cylinder_radius = np.linalg.norm(centered, axis=1).min() * 0.5

    # Step 3: rotation matrix mapping pca_axis onto [0, 0, 1]
    z_hat = np.array([0.0, 0.0, 1.0])
    v     = np.cross(pca_axis, z_hat)
    s     = np.linalg.norm(v)
    c     = float(np.dot(pca_axis, z_hat))

    if s < 1e-10:
        R = np.eye(3)
    else:
        vx = np.array([[  0,   -v[2],  v[1]],
                       [ v[2],   0,   -v[0]],
                       [-v[1],  v[0],   0  ]])
        R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / s**2)

    # Step 4: build the cylindrical-coordinate point cloud
    shifted = (R @ centered.T).T

    r     = np.sqrt(shifted[:, 0]**2 + shifted[:, 1]**2)
    theta = np.arctan2(shifted[:, 1], shifted[:, 0])
    z_cyl = shifted[:, 2]

    if remove_supports:
        # Step 4a: inner wall radius from the central band, drop points inside it
        z_mid_center = 0.5 * (z_cyl.min() + z_cyl.max())
        z_span       = z_cyl.max() - z_cyl.min()
        middle_band  = np.abs(z_cyl - z_mid_center) < 0.20 * z_span
        r_inner_mid  = np.percentile(r[middle_band], 1)
        radial_keep  = r >= r_inner_mid * 0.8

        # Step 4b: drop the +z support by keeping only the largest DBSCAN cluster
        strut_est = np.percentile(r, 98) - r_inner_mid

        kept_idx = np.flatnonzero(radial_keep)
        coords   = shifted[kept_idx]

        db_labels  = DBSCAN(eps=strut_est * 0.1, min_samples=10).fit(coords).labels_
        body_label = np.bincount(db_labels[db_labels >= 0]).argmax()
        body_keep  = db_labels == body_label

        keep_mask = np.zeros(len(pts), dtype=bool)
        keep_mask[kept_idx[body_keep]] = True

        pts     = pts[keep_mask]
        shifted = shifted[keep_mask]
        r       = r[keep_mask]
        theta   = theta[keep_mask]
        z_cyl   = z_cyl[keep_mask]

        # Step 4c: trim the solid closing ring at the -z end (walk in until lattice)
        n_az_slices = 200
        cov_solid   = 0.9           # theta coverage above this counts as a solid ring
        gap_deg     = 15.0          # theta gap wider than this counts as an open cell
        min_in_slice = 5

        edges       = np.linspace(z_cyl.min(), z_cyl.max(), n_az_slices + 1)
        slice_id    = np.clip(np.digitize(z_cyl, edges) - 1, 0, n_az_slices - 1)
        theta_edges = np.linspace(-np.pi, np.pi, 73)   # 72 bins of 5 deg

        def _slice_is_solid(idx):
            if idx.size < min_in_slice:
                return False                            # too sparse to call solid
            t       = np.sort(theta[idx])
            cov     = (np.histogram(t, bins=theta_edges)[0] > 0).mean()
            gaps    = np.r_[np.diff(t), (t[0] + 2 * np.pi) - t[-1]]
            n_cells = (gaps > np.deg2rad(gap_deg)).sum()
            return cov >= cov_solid and n_cells == 0

        band_keep = np.ones(len(z_cyl), dtype=bool)
        for i in range(n_az_slices):                    # i = 0 is the -z end
            idx = np.flatnonzero(slice_id == i)
            if idx.size == 0:
                continue
            if _slice_is_solid(idx):
                band_keep[idx] = False
            else:
                break                                   # hit the lattice -> stop

        pts     = pts[band_keep]
        shifted = shifted[band_keep]
        r       = r[band_keep]
        theta   = theta[band_keep]
        z_cyl   = z_cyl[band_keep]

        # Step 4d: hard axial clamp for this stent
        pts     = pts[z_cyl > -9.8]
        shifted = shifted[z_cyl > -9.8]
        r       = r[z_cyl > -9.8]
        theta   = theta[z_cyl > -9.8]
        z_cyl   = z_cyl[z_cyl > -9.8]

        pts     = pts[z_cyl < 10.1]
        shifted = shifted[z_cyl < 10.1]
        r       = r[z_cyl < 10.1]
        theta   = theta[z_cyl < 10.1]
        z_cyl   = z_cyl[z_cyl < 10.1]

        print(f"[preprocess] removed supports: {len(pts)} points remain")


    df = pd.DataFrame({
        'point_id'     : np.arange(len(pts)),
        'r'            : r,
        'theta'        : theta,
        'z_cylindrical': z_cyl,
        'x'            : shifted[:, 0],
        'y'            : shifted[:, 1],
        'z'            : shifted[:, 2],
    })

    # Step 5: bounding box (plots only)
    z_cyl_min, z_cyl_max = z_cyl.min(), z_cyl.max()
    stent_length   = z_cyl_max - z_cyl_min
    stent_diameter = 2.0 * r.max()

    # Step 6: strut thickness (robust percentile + per-slice mean)
    r_inner_pct = np.percentile(r, 2)
    r_outer_pct = np.percentile(r, 98)
    strut_thick_robust = r_outer_pct - r_inner_pct

    z_edges   = np.linspace(z_cyl_min, z_cyl_max, n_thickness_slices + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    bin_idx   = np.clip(np.digitize(z_cyl, z_edges) - 1, 0, n_thickness_slices - 1)

    rows = []
    for i in range(n_thickness_slices):
        mask = bin_idx == i
        if mask.sum() < 50:
            rows.append({'slice_idx': i, 'z': z_centers[i],
                         'r_inner': np.nan, 'r_outer': np.nan,
                         'thickness': np.nan, 'n_points': int(mask.sum())})
        else:
            ri, ro = r[mask].min(), r[mask].max()
            rows.append({'slice_idx': i, 'z': z_centers[i],
                         'r_inner': ri, 'r_outer': ro,
                         'thickness': ro - ri, 'n_points': int(mask.sum())})

    df_thick = (pd.DataFrame(rows)
                .iloc[slice_cutoff: n_thickness_slices - slice_cutoff]
                .dropna()
                .reset_index(drop=True))
    strut_thick_slice_mean = df_thick['thickness'].mean()
    strut_thick_final = strut_thick_slice_mean if not df_thick.empty else strut_thick_robust

    stent_features = {
        'length'                : stent_length,
        'diameter'              : stent_diameter,
        'radius'                : stent_diameter / 2.0,
        'strut_thickness'       : strut_thick_final,
        'z_min'                 : z_cyl_min,
        'z_max'                 : z_cyl_max,
        'r_inner'               : float(df_thick['r_inner'].mean()),
        'r_outer'               : float(df_thick['r_outer'].mean()),
        'r_mid'                 : float(df_thick['r_inner'].mean() + df_thick['r_outer'].mean()) / 2.0,
        'center_cylinder_radius': center_cylinder_radius,
        'num_points'            : len(df),
    }

    # Step 7: optional thickness plots
    if thickness_calc_plot and not df_thick.empty:
        fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(df_thick['z'], df_thick['r_outer'], color='green', label='r_outer')
        axes[0].plot(df_thick['z'], df_thick['r_inner'], color='red',   label='r_inner')
        axes[0].set_ylabel('r (radial distance)')
        axes[0].set_title('Inner and outer radius along stent axis')
        axes[0].legend()
        axes[0].grid(True)

        mean_t = df_thick['thickness'].mean()
        axes[1].plot(df_thick['z'], df_thick['thickness'], color='steelblue')
        axes[1].axhline(mean_t, color='orange', linestyle='--',
                        label=f'mean = {mean_t:.4f}')
        axes[1].set_ylabel('Thickness (r_outer - r_inner)')
        axes[1].set_xlabel('z_cylindrical (axial position)')
        axes[1].set_title('Strut radial thickness along stent axis')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(r, bins=300, color='steelblue', edgecolor='none')
        ax.axvline(r_inner_pct, color='red',   linestyle='--',
                   label=f'r_inner = {r_inner_pct:.4f}  (2nd pct)')
        ax.axvline(r_outer_pct, color='green', linestyle='--',
                   label=f'r_outer = {r_outer_pct:.4f}  (98th pct)')
        ax.set_xlabel('r  (radial distance from stent axis)')
        ax.set_ylabel('Point count')
        ax.set_title('Global r distribution - two peaks = inner and outer strut wall')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {
        'stent_df'                   : df,
        'stent_features'             : stent_features,
        'stent_centerline_direction' : pca_axis,
    }

def find_crowns(
    stent_df: pd.DataFrame,
    strut_thickness: float,
    min_crown_frac: float = 0.2,      # crown < frac * (90th-pct point-count) = "tiny"
    show_plots: bool = False,
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

    if show_plots:
        zc_d = 0.5 * (edges_d[:-1] + edges_d[1:])
        fig, axd = plt.subplots(figsize=(11, 3))
        axd.plot(zc_d, npts_s, color='gray', label='points/slice (smoothed)')
        axd.plot(zc_d[dips], npts_s[dips], 'rv', ms=9, label=f'{n_dips} crown dips')
        axd.axhline(depth_thresh, color='red', ls=':', lw=1,
                    label=f'depth cutoff ({depth_frac:.2f})')
        axd.set_xlabel('z_cylindrical'); axd.set_ylabel('points / slice')
        axd.set_title(f'Crown dips -> {n_bands} crown-to-crown bands')
        axd.legend(); axd.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

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


def open_stent_to_plane(stent_df: pd.DataFrame, r_mid: float, pad_fraction: float) -> dict:
    """
    Unwrap all surface points onto a 2D (arc, z) plane with periodic padding.
    """
    circumference = 2 * np.pi * r_mid
    arc_flat      = r_mid * stent_df['theta'].values
    z_flat        = stent_df['z_cylindrical'].values
    arc_min_flat  = arc_flat.min()
    arc_max_flat  = arc_flat.max()
    pad_width     = pad_fraction * circumference

    arc_three = np.concatenate([arc_flat - circumference, arc_flat, arc_flat + circumference])
    z_three   = np.concatenate([z_flat,                   z_flat,   z_flat])
    pad_mask  = ((arc_three >= arc_min_flat - pad_width) &
                 (arc_three <= arc_max_flat + pad_width))

    return {
        'arc_padded'   : arc_three[pad_mask],
        'z_padded'     : z_three[pad_mask],
        'arc_flat'     : arc_flat,
        'z_flat'       : z_flat,
        'arc_min_flat' : arc_min_flat,
        'arc_max_flat' : arc_max_flat,
        'circumference': circumference,
    }


def compute_skeleton_2d(
    arc_flat: np.ndarray,
    z_flat: np.ndarray,
    arc_min_flat: float,
    arc_max_flat: float,
    circumference: float,
    stent_df: pd.DataFrame,
    stent_geometry: dict,
    pixels_per_strut: int,
    dilate_px: int,
    pad_fraction: float,
    plot: bool,
    seam_tol_frac: float = 0.0,
) -> dict:
    """Rasterise, dilate, skeletonise, then map back to (arc, z).

    The unroll introduces a seam at the arc boundary, so the points are
    replicated +/-circumference and a pad_fraction margin is kept while
    rasterising; the skeleton is then cropped back to the real arc window.
    seam_tol_frac widens that crop by seam_tol_frac * strut_thickness on both
    edges so a strut crossing the seam reconnects in 3D (0.0 = hard crop).
    """
    pixel_size = stent_geometry['strut_thickness'] / pixels_per_strut

    # Step 1: seam padding (replicate along arc, keep a margin band)
    arc_three = np.concatenate([arc_flat - circumference, arc_flat, arc_flat + circumference])
    z_three   = np.concatenate([z_flat,                   z_flat,   z_flat])
    band      = ((arc_three >= arc_flat.min() - pad_fraction * circumference) &
                 (arc_three <= arc_flat.max() + pad_fraction * circumference))
    arc_pad, z_pad = arc_three[band], z_three[band]   # padded surface points (incl. seam copies)

    # Step 2: canvas spans the padded band so seam copies are not clipped
    arc_lo, arc_hi = arc_pad.min(), arc_pad.max()
    z_lo,   z_hi   = z_pad.min(),   z_pad.max()
    n_cols = int(np.ceil((arc_hi - arc_lo) / pixel_size)) + 1
    n_rows = int(np.ceil((z_hi   - z_lo)   / pixel_size)) + 1

    col_idx = np.clip(((arc_pad - arc_lo) / pixel_size).astype(int), 0, n_cols - 1)
    row_idx = np.clip(((z_pad   - z_lo)   / pixel_size).astype(int), 0, n_rows - 1)

    # Step 3: rasterise, dilate, skeletonise
    img = np.zeros((n_rows, n_cols), dtype=bool)
    img[row_idx, col_idx] = True

    img_solid = dilation(img,       footprint=disk(dilate_px))
    img_solid = closing (img_solid, footprint=disk(1))

    img_skel = skeletonize(img_solid)

    sk_rows, sk_cols = np.where(img_skel)
    skel_arc_pad = arc_lo + (sk_cols + 0.5) * pixel_size
    skel_z_pad   = z_lo   + (sk_rows + 0.5) * pixel_size

    # Step 4: crop back to the real arc window (plus seam tolerance)
    seam_tol  = seam_tol_frac * stent_geometry['strut_thickness']
    seam_mask = ((skel_arc_pad >= arc_min_flat - seam_tol) &
                 (skel_arc_pad <= arc_max_flat + seam_tol))
    skel_arc  = skel_arc_pad[seam_mask]
    skel_z    = skel_z_pad[seam_mask]

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 11))

        axes[0].imshow(img_skel.T, cmap='gray_r', origin='lower',
                       extent=(z_lo, z_hi, arc_lo, arc_hi), aspect='equal')
        axes[0].axhline(arc_min_flat, color='red', linestyle='--', linewidth=1)
        axes[0].axhline(arc_max_flat, color='red', linestyle='--', linewidth=1)
        axes[0].set_title('Skeleton on padded band  (red dashes = kept arc window; outside = padding)')
        axes[0].set_xlabel('z (mm)'); axes[0].set_ylabel('arc (mm)')

        axes[1].scatter(z_pad,      arc_pad,      s=0.3, c='lightsteelblue', linewidths=0)
        axes[1].scatter(skel_z_pad, skel_arc_pad, s=1.0, c='0.65',   linewidths=0, label='padding (dropped)')
        axes[1].scatter(skel_z,     skel_arc,     s=1.5, c='crimson', linewidths=0, label='kept skeleton')
        axes[1].axhline(arc_min_flat, color='red', linestyle='--', linewidth=1)
        axes[1].axhline(arc_max_flat, color='red', linestyle='--', linewidth=1)
        axes[1].set_title('Skeleton over surface points (incl. padding region)')
        axes[1].set_xlabel('z (mm)'); axes[1].set_ylabel('arc (mm)')
        axes[1].set_aspect('equal'); axes[1].legend(loc='upper right', markerscale=6)

        plt.tight_layout()
        plt.show()

    return {
        'skel_arc'             : skel_arc,
        'skel_z'               : skel_z,
        'skel_arc_padded'      : skel_arc_pad,
        'skel_z_padded'        : skel_z_pad,
        'arc_window'           : (arc_min_flat, arc_max_flat),
        'arc_band'             : (arc_lo, arc_hi),
        'pixel_size'           : pixel_size,
        'df_skeleton_2d'       : pd.DataFrame({'arc': skel_arc,     'z': skel_z}),
        'df_skeleton_2d_padded': pd.DataFrame({'arc': skel_arc_pad, 'z': skel_z_pad}),
    }


def _two_core_mask(V: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mask of nodes in the 2-core (iteratively remove degree<2 nodes)."""
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
    strut_thickness: float = None,
    loop_size_factor: float = 2.0,
    surf_tree: cKDTree = None,
    surf_reg: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """Report skeleton quality against the segmented regions (does not modify it).

    Flags four issues: bad_connections (edges joining regions that don't touch),
    region_loops and border_loops (small thinning bubbles, not design cells), and
    empty_regions (regions with no skeleton point). A loop counts only when its
    bounding-box diagonal is below loop_size_factor * strut_thickness; with
    strut_thickness None the size test is off and every loop is flagged.

    Also returns the (arc, z) location of each issue for plotting. surf_tree /
    surf_reg let a caller pass a prebuilt KD-tree so a tuner avoids rebuilding it.
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
        """Loops on `node_mask`, each as (region labels spanned, (arc, z) points).

        A loop is one connected component of the sub-graph's 2-core, so call this
        on a small node set (one region, or a touching pair), not the whole stent.
        """
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
        """True when the loop's bounding box is no wider than loop_max_diag."""
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


def remove_bad_connections(
    df_skeleton_2d: pd.DataFrame,
    pixel_size: float,
    stent_df: pd.DataFrame,
    r_mid: float,
    region_allowed: np.ndarray,
    surf_tree: cKDTree = None,
    surf_reg: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """Delete skeleton points on edges that join two non-touching regions.

    Region labels come from the nearest surface point, so removing every point on
    a bad edge clears all bad edges in one pass and leaves a >=2 px gap (the 3D
    rebuild won't re-bridge). Loops and empty regions are left untouched.
    """
    skel = df_skeleton_2d[['arc', 'z']].to_numpy()
    n_sk = len(skel)

    # Step 1: nearest-surface region per skeleton point
    if surf_tree is None or surf_reg is None:
        surf_arc  = r_mid * stent_df['theta'].to_numpy()
        surf_z    = stent_df['z'].to_numpy()
        surf_reg  = stent_df['region'].to_numpy()
        surf_tree = cKDTree(np.column_stack([surf_arc, surf_z]))
    _, nn       = surf_tree.query(skel)
    skel_region = surf_reg[nn]

    # Step 2: build the 8-neighbour grid graph (same as check_skeleton_quality)
    gi = np.round((skel[:, 0] - skel[:, 0].min()) / pixel_size).astype(int)
    gj = np.round((skel[:, 1] - skel[:, 1].min()) / pixel_size).astype(int)
    coord_set = set(zip(gi.tolist(), gj.tolist()))
    coord_idx = {(int(gi[k]), int(gj[k])): k for k in range(n_sk)}
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

    # Step 3: drop points on edges between non-touching regions
    remove = np.zeros(n_sk, dtype=bool)
    n_bad_edges, bad_pairs = 0, []
    if len(edges):
        ra, rb = skel_region[edges[:, 0]], skel_region[edges[:, 1]]
        bad    = (ra != rb) & ~region_allowed[ra, rb]
        n_bad_edges = int(bad.sum())
        if bad.any():
            remove[np.unique(edges[bad].ravel())] = True
            bad_pairs = [tuple(int(x) for x in p) for p in
                         np.unique(np.sort(np.column_stack([ra[bad], rb[bad]]), axis=1), axis=0)]

    df_clean = df_skeleton_2d.loc[~remove].reset_index(drop=True)
    if verbose:
        if n_bad_edges:
            print(f"[clean] removed {int(remove.sum())} point(s) on {n_bad_edges} bad edge(s) "
                  f"across {len(bad_pairs)} region pair(s) {bad_pairs} "
                  f"-> {len(df_clean):,} points remain")
        else:
            print(f"[clean] no bad connections found -> skeleton unchanged "
                  f"({len(df_clean):,} points)")

    return {
        'df_skeleton_2d': df_clean,
        'skel_arc'      : df_clean['arc'].to_numpy(),
        'skel_z'        : df_clean['z'].to_numpy(),
        'pixel_size'    : pixel_size,
        'n_removed'     : int(remove.sum()),
        'n_bad_edges'   : n_bad_edges,
        'removed_xy'    : skel[remove],   # (K, 2) (arc, z) of the cut points
        'bad_pairs'     : bad_pairs,
    }


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
    w_conn: float = 5.0,
    w_loop: float = 3.0,
    w_empty: float = 1.0,
    loop_size_factor: float = 2.0,  # a loop wider than this x strut_thickness is a design cell, not a defect
    q_eps: float = 1e-3,          # floor on the quality residual so total_error stays > 0
    quality_gamma: float = 2.0,   # quality_error convexity: 1 = linear, >1 = diminishing returns
    res_no_improve_max: int = 5,  # steps (after the first clean step) with no total_error improvement
    target_penalty: float = 0.0,  # defect_error <= this counts as "feasible" (clean) AND gates best selection
    max_repeats: int = 2,        # stop once the SAME (pps, dil) state has been visited this many times
    pad_fraction: float = 0.20,
    seam_tol_frac: float = 0.0,   # widen the seam crop so wrap-crossing struts reconnect in 3D
    time_limit: float = 100.0,
    predictive_stop: bool = True,
    max_iters: int = int(1e3),
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    """Error-driven tuner for pixels_per_strut and dilate_px.

    Each step skeletonises, scores with check_skeleton_quality, and nudges the two
    parameters by the region-normalised errors: bad connections raise pps and
    lower dilate_px, empty regions raise pps, loops raise dilate_px (or lower pps
    once dilate_px is capped). dilate_px is an integer disk radius bounded by the
    strut spacing (dil <= pps).

    Selection minimises total_error = defect_error + quality_error, gated so a
    clean skeleton (defect_error <= target_penalty) always beats a defective one;
    quality_error is a strictly-positive resolution residual on a 0-100 scale.
    Stops on a repeated (pps, dil) state, on no improvement for res_no_improve_max
    steps after the first clean skeleton, or on time_limit / max_iters.
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
            arc_flat=arc, z_flat=z, arc_min_flat=arc.min(), arc_max_flat=arc.max(),
            circumference=circ, stent_df=stent_df, stent_geometry=stent_features,
            pixels_per_strut=pps, dilate_px=dil, pad_fraction=pad_fraction, plot=False,
            seam_tol_frac=seam_tol_frac,
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

    if plot:
        # total-error trajectory: defect_error -> 0, leaving the quality residual
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(hist_df['step'], hist_df['total_error'],  '-', color='crimson',
                label='total_error', linewidth=2)
        ax.plot(hist_df['step'], hist_df['defect_error'], '-', color='steelblue',
                label='defect_error')
        ax.plot(hist_df['step'], hist_df['quality_error'], '-', color='darkorange',
                label='quality_error')
        ax.axhline(0, color='grey', lw=0.8, ls=':')   # ideal (unreachable) total_error = 0
        ax.set_ylim(bottom=0)
        ax.set_xlabel('step'); ax.set_ylabel('error'); ax.set_title('tuning trajectory (minimise total_error)')
        ax.legend(loc='upper right')
        plt.tight_layout(); plt.show()

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


def plot_skeleton_quality(
    skeleton_2d_result: dict,
    stent_df: pd.DataFrame,
    r_mid: float,
    quality_report: dict,
    figsize=(16, 7),
) -> None:
    """Overlay the quality issues on the unrolled (z, arc) skeleton.

    Skeleton coloured by region inside the kept window, the dropped padding
    skeleton in grey, and bad connections / loops / empty regions highlighted.
    """
    skel        = skeleton_2d_result['df_skeleton_2d'][['arc', 'z']].to_numpy()
    skel_pad    = skeleton_2d_result['df_skeleton_2d_padded'][['arc', 'z']].to_numpy()
    skel_region = quality_report['skel_region']
    arc_lo, arc_hi   = skeleton_2d_result['arc_band']
    arc_min, arc_max = skeleton_2d_result['arc_window']
    circ = 2 * np.pi * r_mid

    # Replicate surface points across the seam to fill the padded band
    surf_arc = r_mid * stent_df['theta'].to_numpy()
    surf_z   = stent_df['z'].to_numpy()
    a3 = np.concatenate([surf_arc - circ, surf_arc, surf_arc + circ])
    z3 = np.concatenate([surf_z,          surf_z,   surf_z])
    in_band = (a3 >= arc_lo) & (a3 <= arc_hi)
    a3, z3  = a3[in_band], z3[in_band]
    sel = np.random.default_rng(0).choice(len(a3), min(len(a3), int(3e5)), replace=False)

    rng    = np.random.default_rng(1)
    cmap   = {r: rng.random(3) for r in np.unique(skel_region)}
    colors = np.array([cmap[r] for r in skel_region])

    # padding-only skeleton points (outside the kept window)
    pad_only = (skel_pad[:, 0] < arc_min) | (skel_pad[:, 0] > arc_max)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(z3[sel], a3[sel], s=0.3, c='0.88', linewidths=0, zorder=0)
    ax.scatter(skel_pad[pad_only, 1], skel_pad[pad_only, 0], s=2.0, c='0.6',
               linewidths=0, zorder=1, label='padding skeleton (dropped)')
    ax.scatter(skel[:, 1], skel[:, 0], s=2.0, c=colors, linewidths=0, zorder=2)

    ax.axhline(arc_min, color='red', linestyle='--', linewidth=1, zorder=3)
    ax.axhline(arc_max, color='red', linestyle='--', linewidth=1, zorder=3)

    bad_xy   = quality_report['bad_edge_xy']
    loop_xy  = quality_report['loop_points_xy']
    empty_xy = quality_report['empty_xy']

    if len(bad_xy):
        ax.scatter(bad_xy[:, 1], bad_xy[:, 0], s=140, marker='x', c='red',
                   linewidths=2.5, zorder=5,
                   label=f'bad connection ({len(quality_report["bad_connections"])} pair(s))')
    if len(loop_xy):
        n_reg_loops    = len(quality_report['region_loops'])
        n_border_loops = len(quality_report.get('border_loops', {}))
        ax.scatter(loop_xy[:, 1], loop_xy[:, 0], s=40, marker='o',
                   facecolors='none', edgecolors='magenta', linewidths=1.4, zorder=4,
                   label=f'loop ({n_reg_loops} region, {n_border_loops} border)')
    if len(empty_xy):
        ax.scatter(empty_xy[:, 1], empty_xy[:, 0], s=220, marker='s',
                   facecolors='none', edgecolors='black', linewidths=2.0, zorder=6,
                   label=f'empty region ({len(empty_xy)})')
        for (a, z_c), er in zip(empty_xy, quality_report['empty_regions']):
            ax.annotate(f'R{er}', (z_c, a), color='black', fontsize=8,
                        ha='center', va='bottom')

    ax.set_aspect('equal')
    ax.set_ylim(arc_lo, arc_hi)          # show the full padded band
    ax.set_xlabel('z (mm)'); ax.set_ylabel('arc (mm)')
    ax.set_title('Skeleton quality issues (unrolled surface, incl. padding band)')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()


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


def load_update_stent_data(stent_name, material_name, youngs_modulus, poissons_ratio,
                    density, max_elastic_strain):
    """Load stent geometry/skeleton data and extend features with material parameters.

    Each key in the returned features dict maps to {"value": ..., "unit": "..."}.
    Density is stored in kg/m³; callers using the mm-N-tonne system must convert to t/mm³.
    stent_features.json is updated only when material parameters are new or have changed.
    The JSON stores {"value", "unit"} wrapped entries for all keys.
    """
    _GEO_UNITS = {
        "length":                 "mm",
        "diameter":               "mm",
        "radius":                 "mm",
        "strut_thickness":        "mm",
        "z_min":                  "mm",
        "z_max":                  "mm",
        "r_inner":                "mm",
        "r_outer":                "mm",
        "r_mid":                  "mm",
        "center_cylinder_radius": "mm",
        "connection_radius":      "mm",
        "conn_radius_3d":         "mm",
        "num_points":             "-",
        "n_regions":              "-",
    }
    _MAT_KEYS = {"material_name", "youngs_modulus", "poissons_ratio",
                 "shear_modulus", "density", "max_elastic_strain"}

    stent_dir = REPO_ROOT / "examples" / "notebook_outputs" / stent_name
    print(f"Loading stent data from: {stent_dir.resolve()}")

    with open(stent_dir / "stent_features.json") as f:
        raw = json.load(f)

    # unwrap any {"value": ..., "unit": ...} entries (handles previously wrapped JSON)
    def _unwrap(v):
        while isinstance(v, dict) and "value" in v:
            v = v["value"]
        return v
    raw = {k: _unwrap(v) for k, v in raw.items()}

    with open(stent_dir / "stent_centerline_direction.json") as f:
        cl_dir = np.array(json.load(f))

    sk = pd.read_csv(stent_dir / "skeleton_points.csv")
    sk["neighbor_ids"] = sk["neighbor_ids"].apply(ast.literal_eval)

    # Material params as flat raw values (for comparison + JSON storage)
    new_mat = {
        "material_name":      material_name,
        "youngs_modulus":     youngs_modulus,
        "poissons_ratio":     poissons_ratio,
        "shear_modulus":      youngs_modulus / (2.0 * (1.0 + poissons_ratio)),
        "density":            density,
        "max_elastic_strain": max_elastic_strain,
    }

    _MAT_UNITS = {
        "material_name": "-", "youngs_modulus": "MPa", "poissons_ratio": "-",
        "shear_modulus": "MPa", "density": "kg/m³", "max_elastic_strain": "-",
    }

    # Write JSON only when material params are absent or have changed
    needs_update = any(raw.get(k) != v for k, v in new_mat.items())
    if needs_update:
        geo_raw_flat = {k: v for k, v in raw.items() if k not in _MAT_KEYS}
        geo_wrapped  = {k: {"value": v, "unit": _GEO_UNITS.get(k, "-")} for k, v in geo_raw_flat.items()}
        mat_wrapped  = {k: {"value": v, "unit": _MAT_UNITS[k]} for k, v in new_mat.items()}
        with open(stent_dir / "stent_features.json", "w") as f:
            json.dump({**geo_wrapped, **mat_wrapped}, f, indent=4)
        print(f"  stent_features.json updated with material parameters.")
    else:
        print(f"  Material parameters unchanged - stent_features.json not rewritten.")

    # Wrap for in-memory use
    geo_raw = {k: v for k, v in raw.items() if k not in _MAT_KEYS}
    feats = {k: {"value": v, "unit": _GEO_UNITS.get(k, "-")} for k, v in geo_raw.items()}
    feats.update({
        "material_name":      {"value": material_name,           "unit": "-"},
        "youngs_modulus":     {"value": youngs_modulus,          "unit": "MPa"},
        "poissons_ratio":     {"value": poissons_ratio,          "unit": "-"},
        "shear_modulus":      {"value": new_mat["shear_modulus"],"unit": "MPa"},
        "density":            {"value": density,                 "unit": "kg/m³"},
        "max_elastic_strain": {"value": max_elastic_strain,      "unit": "-"},
    })

    return feats, cl_dir, sk


import numpy as np
import trimesh
from trimesh.path.entities import Line
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, to_rgb
from matplotlib.lines import Line2D
import pandas as pd
import ast
import datetime
import pathlib
import json
import time
import os
import glob
import pickle
import base64
from collections import deque, Counter
from sklearn.cluster import DBSCAN

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize, dilation, closing, disk

import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as pcolors

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

    if n_samples is None:
        n_samples = len(mesh.faces) * samples_per_face
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples, seed=random_seed)

    # Step 2: PCA centroid and main axis
    pca_mean  = pts.mean(axis=0)
    centered  = pts - pca_mean
    cov       = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
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

        '''# Step 4b: drop the +z support by keeping only the largest DBSCAN cluster
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
        z_cyl   = z_cyl[band_keep]'''

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
    w_conn: float = 2.0,
    w_loop: float = 10.0,
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


# ---------------------------------------------------------------------------
# Interactive-pipeline helpers: outlier removal, Plotly HTML views,
# per-crown diagnostics, and manual skeleton edits.
# ---------------------------------------------------------------------------

def drop_points(stent_df: pd.DataFrame, point_ids) -> pd.DataFrame:
    """Remove rows from the point cloud by ``point_id``.

    ``point_ids`` is any iterable of ids (empty / None -> no-op). The original
    ``point_id`` values are preserved (no renumbering) so ids the user reads off
    an earlier HTML view stay valid across rounds. Returns a new DataFrame.
    """
    if point_ids is None:
        return stent_df
    ids = [int(p) for p in point_ids]
    if not ids:
        return stent_df
    keep = ~stent_df['point_id'].isin(ids)
    return stent_df[keep].reset_index(drop=True)


def _downsample_df(df: pd.DataFrame, max_display: int, random_state: int = 0) -> pd.DataFrame:
    """Return df unchanged if small, else a random subset of ``max_display`` rows
    (ids preserved). Display-only — never used for processing."""
    if max_display is None or len(df) <= max_display:
        return df
    return df.sample(max_display, random_state=random_state)


def plot_points_3d_html(
    df: pd.DataFrame,
    id_col: str,
    out_path: str,
    color_col: str = None,
    max_display: int = 40000,
    title: str = "",
    point_size: float = 1,
    categorical: bool = False,
) -> str:
    """Write an interactive Plotly 3D scatter of a point cloud to ``out_path`` (HTML).

    Hovering a point shows its ``id_col`` (e.g. point_id / skeleton_point_id) and,
    when ``color_col`` is given, that value too (e.g. crown_id, used for colour).
    The view is downsampled to ``max_display`` points for browser performance, but
    every displayed point keeps its true id so outlier removal stays valid.

    ``categorical=True`` treats ``color_col`` as discrete labels (e.g. crown_id):
    each label gets its own high-contrast qualitative colour and legend entry, so
    neighbouring groups are easy to tell apart (a continuous scale like Turbo makes
    adjacent crowns look nearly identical). ``categorical=False`` keeps the
    continuous colour scale.
    """

    disp = _downsample_df(df, max_display)
    ids  = disp[id_col].to_numpy()
    xyz  = disp[['x', 'y', 'z']].to_numpy()
    note = f"  ({len(disp):,}/{len(df):,} shown)" if len(disp) < len(df) else ""

    if color_col is not None and color_col in disp.columns and categorical:
        # one trace per label with a distinct qualitative colour + legend
        palette = (pcolors.qualitative.Dark24 + pcolors.qualitative.Light24)
        labels  = sorted(disp[color_col].unique())
        fig = go.Figure()
        for i, lab in enumerate(labels):
            sub  = disp[disp[color_col] == lab]
            cdat = np.column_stack([sub[id_col].to_numpy(),
                                    sub[color_col].to_numpy()])
            fig.add_trace(go.Scatter3d(
                x=sub['x'], y=sub['y'], z=sub['z'], mode='markers',
                marker=dict(size=point_size, color=palette[i % len(palette)]),
                name=f'{color_col}={lab}', customdata=cdat,
                hovertemplate=(f"{id_col}=%{{customdata[0]}}<br>"
                               f"{color_col}=%{{customdata[1]}}<extra></extra>")))
        fig.update_layout(
            template='plotly_dark', height=800, margin=dict(l=0, r=0, t=40, b=0),
            title=(title + note), legend=dict(itemsizing='constant'),
            scene=dict(aspectmode='data', xaxis_title='x', yaxis_title='y',
                       zaxis_title='z'))
        pio.write_html(fig, out_path, auto_open=False)
        return out_path

    if color_col is not None and color_col in disp.columns:
        cval       = disp[color_col].to_numpy()
        customdata = np.column_stack([ids, cval])
        hovertemplate = (f"{id_col}=%{{customdata[0]}}<br>"
                         f"{color_col}=%{{customdata[1]}}<extra></extra>")
        marker = dict(size=point_size, color=cval, colorscale='Turbo',
                      showscale=True, colorbar=dict(title=color_col))
    else:
        customdata    = ids[:, None]
        hovertemplate = f"{id_col}=%{{customdata[0]}}<extra></extra>"
        marker        = dict(size=point_size, color='royalblue')

    fig = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
        marker=marker, customdata=customdata, hovertemplate=hovertemplate))
    fig.update_layout(
        template='plotly_dark', height=800, margin=dict(l=0, r=0, t=40, b=0),
        title=(title + note),
        scene=dict(aspectmode='data', xaxis_title='x', yaxis_title='y', zaxis_title='z'))
    pio.write_html(fig, out_path, auto_open=False)
    return out_path


def _skeleton_edge_segments(skeleton_df: pd.DataFrame):
    """Return (xe, ye, ze) line arrays with NaN separators for all unique edges,
    built from the explicit ``neighbor_ids`` graph (ids = skeleton_point_id)."""
    coords = skeleton_df.set_index('skeleton_point_id')[['x', 'y', 'z']]
    xe, ye, ze = [], [], []
    seen = set()
    for _, row in skeleton_df.iterrows():
        pid  = int(row['skeleton_point_id'])
        nbrs = row['neighbor_ids']
        if isinstance(nbrs, str):
            nbrs = ast.literal_eval(nbrs)
        for nid in nbrs:
            nid = int(nid)
            key = (min(pid, nid), max(pid, nid))
            if key in seen or nid not in coords.index:
                continue
            seen.add(key)
            p0 = coords.loc[pid].to_numpy()
            p1 = coords.loc[nid].to_numpy()
            xe += [p0[0], p1[0], np.nan]
            ye += [p0[1], p1[1], np.nan]
            ze += [p0[2], p1[2], np.nan]
    return np.array(xe), np.array(ye), np.array(ze)


def plot_skeleton_html(
    skeleton_df: pd.DataFrame,
    out_path: str,
    title: str = "Skeleton",
    max_display: int = 40000,
) -> str:
    """Write an interactive Plotly 3D view of the skeleton alone to ``out_path``.

    Draws every edge as grey lines (from ``neighbor_ids``) and overlays the nodes
    as markers grouped/coloured by ``node_type``; hovering a node shows its
    ``skeleton_point_id``, ``node_type`` and ``degree`` so the user can name the
    points involved in any loop / wrong-connection error.
    """
    fig = go.Figure()

    xe, ye, ze = _skeleton_edge_segments(skeleton_df)
    if len(xe):
        fig.add_trace(go.Scatter3d(
            x=xe, y=ye, z=ze, mode='lines',
            line=dict(width=3, color='rgba(150,150,150,0.6)'),
            name='edges', hoverinfo='skip', showlegend=False))

    colors = {'line': 'royalblue', 'junction': 'limegreen',
              'endpoint': 'red', 'isolated': 'orange'}
    disp = _downsample_df(skeleton_df, max_display)
    for ntype, col in colors.items():
        sub = disp[disp['node_type'] == ntype]
        if not len(sub):
            continue
        cdata = np.column_stack([sub['skeleton_point_id'].to_numpy(),
                                 sub['degree'].to_numpy()])
        fig.add_trace(go.Scatter3d(
            x=sub['x'], y=sub['y'], z=sub['z'], mode='markers',
            marker=dict(size=3, color=col), name=ntype, customdata=cdata,
            hovertemplate=("skeleton_point_id=%{customdata[0]}<br>"
                           f"node_type={ntype}<br>"
                           "degree=%{customdata[1]}<extra></extra>")))

    fig.update_layout(
        template='plotly_dark', height=800, margin=dict(l=0, r=0, t=40, b=0),
        title=title,
        scene=dict(aspectmode='data', xaxis_title='x', yaxis_title='y', zaxis_title='z'))
    pio.write_html(fig, out_path, auto_open=False)
    return out_path


def plot_skeleton_with_cloud_html(
    skeleton_df: pd.DataFrame,
    stent_df: pd.DataFrame,
    out_path: str,
    max_cloud: int = 40000,
) -> str:
    """Write the final combined 3D view (skeleton edges + nodes over a faint,
    downsampled point cloud) to ``out_path``. Hovering a skeleton node shows its
    ``skeleton_point_id``; hovering a cloud point shows its ``point_id``."""
    cloud = _downsample_df(stent_df, max_cloud)
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=cloud['x'], y=cloud['y'], z=cloud['z'], mode='markers',
        marker=dict(size=1.5, color='rgba(180,180,180,0.25)'),
        name='point cloud', customdata=cloud['point_id'].to_numpy()[:, None],
        hovertemplate="point_id=%{customdata}<extra></extra>"))

    xe, ye, ze = _skeleton_edge_segments(skeleton_df)
    if len(xe):
        fig.add_trace(go.Scatter3d(
            x=xe, y=ye, z=ze, mode='lines',
            line=dict(width=4, color='rgba(255,80,80,0.9)'),
            name='skeleton', hoverinfo='skip', showlegend=True))

    sk = _downsample_df(skeleton_df, max_cloud)
    fig.add_trace(go.Scatter3d(
        x=sk['x'], y=sk['y'], z=sk['z'], mode='markers',
        marker=dict(size=2.5, color='red'), name='skeleton nodes',
        customdata=sk['skeleton_point_id'].to_numpy()[:, None],
        hovertemplate="skeleton_point_id=%{customdata}<extra></extra>"))

    fig.update_layout(
        template='plotly_dark', height=800, margin=dict(l=0, r=0, t=40, b=0),
        title="Final skeleton over stent point cloud",
        scene=dict(aspectmode='data', xaxis_title='x', yaxis_title='y', zaxis_title='z'))
    pio.write_html(fig, out_path, auto_open=False)
    return out_path


def plot_splines_html(splines: list, out_path: str, n_eval: int = 100) -> str:
    """Write an interactive Plotly 3D view of the fitted skeleton splines.

    ``splines`` is the list returned by the notebook's spline fitter (each item a
    dict with a ``tck`` and ``ctrl`` polyline fallback, or None). Evaluates each
    spline at ``n_eval`` samples and renders one coloured curve per spline.
    """
    cmap    = plt.get_cmap('tab20')
    palette = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
               for r, g, b, _ in (cmap(i % 20) for i in range(len(splines)))]
    fig = go.Figure()
    n_drawn = 0
    for i, (spl, color) in enumerate(zip(splines, palette), start=1):
        if spl is None:
            continue
        if spl.get('tck') is None:
            sp = np.asarray(spl['ctrl'])
        else:
            uu = np.linspace(0.0, 1.0, n_eval)
            x, y, z = splev(uu, spl['tck'])
            sp = np.column_stack([x, y, z])
        fig.add_trace(go.Scatter3d(
            x=sp[:, 0], y=sp[:, 1], z=sp[:, 2], mode='lines',
            line=dict(width=5, color=color), name=f"curve {i}", hoverinfo='name',
            showlegend=False))
        n_drawn += 1
    fig.update_layout(
        template='plotly_dark', height=800, margin=dict(l=0, r=0, t=40, b=0),
        title=f"{n_drawn} fitted spline curves",
        scene=dict(aspectmode='data', xaxis_title='x', yaxis_title='y', zaxis_title='z'))
    pio.write_html(fig, out_path, auto_open=False)
    return out_path


def plot_crown_dips_html(crown_res: dict, out_path: str) -> str:
    """Write the interactive crown dip-detection plot (HTML) from a find_crowns result.

    Plots smoothed points/slice vs z; hovering any point shows its z and count.
    Detected crown dips are marked, and the depth-cutoff threshold is drawn as a
    horizontal line. Uses the diagnostic arrays returned by find_crowns
    (dip_z_centers, dip_counts_smoothed, dip_indices, dip_depth_thresh, n_bands).
    """
    zc     = np.asarray(crown_res['dip_z_centers'])
    cnt    = np.asarray(crown_res['dip_counts_smoothed'])
    dips   = np.asarray(crown_res['dip_indices'], dtype=int)
    thresh = float(crown_res['dip_depth_thresh'])
    n_bands = crown_res.get('n_bands', '?')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=zc, y=cnt, mode='lines+markers', name='points / slice',
        line=dict(color='gray'), marker=dict(size=4, color='gray'),
        hovertemplate="z=%{x:.4f}<br>points/slice=%{y:.1f}<extra></extra>"))
    if len(dips):
        fig.add_trace(go.Scatter(
            x=zc[dips], y=cnt[dips], mode='markers', name='crown dips',
            marker=dict(size=12, color='red', symbol='triangle-down'),
            hovertemplate="DIP<br>z=%{x:.4f}<br>points/slice=%{y:.1f}<extra></extra>"))
    fig.add_hline(y=thresh, line=dict(color='red', dash='dot', width=1),
                  annotation_text='depth cutoff', annotation_position='top left')
    fig.update_layout(
        template='plotly_white', height=420, margin=dict(l=40, r=20, t=50, b=40),
        title=f"Crown dips -> {n_bands} crown-to-crown bands  ({len(dips)} dips)",
        xaxis_title='z_cylindrical', yaxis_title='points / slice')
    pio.write_html(fig, out_path, auto_open=False, config={'scrollZoom': True})
    return out_path


def plot_crown_convergence_html(history, out_path, crown_id,
                                quality_report=None, pps=None, dil_px=None):
    """Save the per-crown tuning diagnostic (separate from the skeleton plot).

    * auto-tune ON  (``history`` given): the tuning error convergence line plot;
      hovering a step shows defect/quality/total error and the pps / dil_px tried.
    * auto-tune OFF (``history`` None but ``quality_report`` given): a quality
      summary bar chart — the count of each issue type (green when 0, red when >0)
      for the fixed ``pps`` / ``dil_px`` used.
    * neither available: a plain annotation.
    """
    fig = go.Figure()

    if history is not None and len(history):
        step    = np.asarray(history['step'])
        total   = np.asarray(history['total_error'])
        defect  = np.asarray(history['defect_error'])
        quality = np.asarray(history['quality_error'])
        pps_a   = np.asarray(history['pps']) if 'pps' in history else np.full_like(step, np.nan, float)
        dil_a   = np.asarray(history['dil_px']) if 'dil_px' in history else np.full_like(step, np.nan, float)
        cdata   = np.column_stack([pps_a, dil_a, defect, quality, total])
        htmpl   = ("step=%{x}<br>pps=%{customdata[0]:.2f}<br>"
                   "dil_px=%{customdata[1]:.0f}<br>defect=%{customdata[2]:.4f}<br>"
                   "quality=%{customdata[3]:.4f}<br>total=%{customdata[4]:.4f}<extra></extra>")
        for name, yv, col in (('total', total, 'black'),
                              ('defect', defect, 'royalblue'),
                              ('quality', quality, 'orange')):
            fig.add_trace(go.Scatter(
                x=step, y=yv, mode='lines+markers', name=name,
                line=dict(color=col), marker=dict(size=5, color=col),
                customdata=cdata, hovertemplate=htmpl))
        fig.update_layout(title=f'Crown {crown_id} — error convergence',
                          xaxis_title='tuning step', yaxis_title='error')
    elif quality_report is not None:
        cats = ['bad connections', 'region loops', 'border loops', 'empty regions']
        vals = [len(quality_report.get('bad_connections', [])),
                len(quality_report.get('region_loops', {})),
                len(quality_report.get('border_loops', {})),
                len(quality_report.get('empty_regions', []))]
        colors = ['crimson' if v > 0 else 'seagreen' for v in vals]
        fig.add_trace(go.Bar(
            x=cats, y=vals, marker_color=colors, text=vals, textposition='outside',
            hovertemplate="%{x}: %{y}<extra></extra>", showlegend=False))
        ptag = f' (pps={pps}, dil_px={dil_px})' if pps is not None else ''
        fig.update_layout(title=f'Crown {crown_id} — quality summary{ptag}',
                          yaxis=dict(title='count', rangemode='tozero'))
    else:
        fig.add_annotation(text='no tuning history (auto-tune off)',
                           xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=f'Crown {crown_id} — tuning')

    fig.update_layout(template='plotly_white', height=450,
                      margin=dict(l=50, r=20, t=50, b=40))
    pio.write_html(fig, out_path, auto_open=False, config={'scrollZoom': True})
    return out_path


# ---------------------------------------------------------------------------
# 2D per-crown manual edits (Step 5.5). The user fixes errors on the flat
# (arc, z) crown skeleton BEFORE the 3D wrap, pointing at a problem with two
# anchor indices; the tool auto-removes what's between them. Edits touch only
# the targeted bubble/bridge (everything else in the crown is untouched) and
# keep ~pixel_size spacing. Connectivity is rebuilt later in 3D (Step 6).
# ---------------------------------------------------------------------------

def _grid_adjacency(arc, z, pixel_size):
    """8-neighbour pixel-grid graph for a 2D skeleton (arc, z).

    Returns (edges, adj): edges is an (K, 2) int array of index pairs, adj a list
    of neighbour-index lists. Same rule as remove_bad_connections /
    check_skeleton_quality (a diagonal is dropped when it short-cuts a
    4-connected corner).
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


def _interp_2d(a, b, spacing):
    """Evenly-spaced interior points (~spacing apart) on the segment a->b,
    exclusive of the endpoints. a, b are (arc, z). Returns (M, 2)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    dist = float(np.hypot(*(b - a)))
    n_mid = max(int(round(dist / spacing)) - 1, 0) if spacing > 0 else 0
    if n_mid == 0:
        return np.empty((0, 2), float)
    ts = (np.arange(1, n_mid + 1) / (n_mid + 1))[:, None]
    return a + ts * (b - a)


def fix_crown_loop_2d(arc, z, pixel_size, anchor_a, anchor_b, verbose=True):
    """Collapse a loop (bubble) on the 2D crown skeleton into a single path.

    ``anchor_a`` / ``anchor_b`` are the two point indices (the plot's hover ``i``)
    where the loop attaches. The loop is the 2-core connected component containing
    both anchors; its points are deleted except the two anchors, and an
    evenly-spaced straight segment is inserted between them. All other points are
    left unchanged. Returns (new_arc, new_z, changed_idx) where changed_idx marks
    the inserted points; a no-op returns the inputs and an empty changed_idx.
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


def fix_crown_connection_2d(arc, z, pixel_size, point_a, point_b, verbose=True):
    """Cut a wrong connection on the 2D crown skeleton by removing the whole bridge.

    Pick two points ON the wrong link (the little bridge between two struts). The
    shortest graph path between them identifies the bridge; the removal is then
    extended along the thin (degree-2) chain in both directions up to the junctions
    where the bridge meets the real struts, so the ENTIRE bridge is deleted (not
    just the piece between your two clicks — that would leave two stubs). The
    bounding junctions and everything else are kept, opening a clean gap. Returns
    (new_arc, new_z, changed_idx=empty); a no-op returns the inputs.
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
        """From a path end, walk AWAY from the path through degree-2 nodes, adding
        them to `remove`, stopping at the first junction/endpoint (kept)."""
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


def auto_clean_bad_connections_2d(arc, z, pixel_size, bad_edge_xy, verbose=True):
    """Auto-remove the detected bad connections the same way the manual fix does.

    For each flagged bad-edge location in ``bad_edge_xy`` (the (arc, z) midpoints
    from check_skeleton_quality), find the nearest skeleton point and delete the
    entire thin (degree-2) bridge chain it lies on, up to the bounding junctions —
    identical to ``fix_crown_connection_2d`` but seeded automatically from the
    detector instead of two user clicks. This clears the automatically-detectable
    connection errors so only the ones the detector missed remain (fixed by hand in
    Step 5.5). Returns (new_arc, new_z, n_bridges_removed).
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
        """Maximal set of degree-2 nodes connected to `start` through degree-2
        nodes (the bridge). If `start` is a junction, union its degree-2 arms."""
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


def plot_crown_skeleton_2d_html(arc, z, surface_arc, surface_z, out_path, crown_label,
                                crown_band=None, changed_idx=None, quality_report=None,
                                title=""):
    """Single-panel interactive 2D view of a crown skeleton (crown_XX.html + editor).

    x=z, y=arc. Surface points grey (halo cut when crown_band is given), skeleton
    points red; hovering a skeleton point shows its local index i and (arc, z).
    Scroll/drag zoom enabled (no equal-aspect lock). ``changed_idx`` points are
    ringed to show what an edit changed. When ``quality_report`` is given, the
    detected issues INSIDE the crown band are overlaid — bad connections (blue x),
    loops (magenta open circles), empty regions (black open squares) — and the
    title shows the in-band issue count.
    """
    arc   = np.asarray(arc)
    z     = np.asarray(z)
    s_arc = np.asarray(surface_arc)
    s_z   = np.asarray(surface_z)

    def _in_band(xy_arc, xy_z):
        if crown_band is None:
            return xy_arc, xy_z
        lo, hi = float(crown_band[0]), float(crown_band[1])
        m = (xy_z >= lo) & (xy_z <= hi)
        return xy_arc[m], xy_z[m]

    s_arc, s_z = _in_band(s_arc, s_z)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s_z, y=s_arc, mode='markers', name='surface',
        marker=dict(size=2, color='lightgray'), hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=z, y=arc, mode='markers', name='skeleton',
        marker=dict(size=4, color='red'), customdata=np.arange(len(arc)),
        hovertemplate="i=%{customdata}<br>arc=%{y:.4f}<br>z=%{x:.4f}<extra></extra>"))

    if changed_idx is not None and len(changed_idx):
        ci = np.asarray(changed_idx, int)
        fig.add_trace(go.Scatter(
            x=z[ci], y=arc[ci], mode='markers', name='changed',
            marker=dict(size=8, color='deepskyblue', symbol='circle-open',
                        line=dict(width=2)), customdata=ci,
            hovertemplate="changed i=%{customdata}<br>arc=%{y:.4f}<br>z=%{x:.4f}<extra></extra>"))

    # overlay flagged quality issues (in-band only) so the user can verify detection
    n_issues = None
    if quality_report is not None:
        def _pts_in_band(xy):
            xy = np.asarray(xy)
            if not len(xy):
                return xy
            a, zz = _in_band(xy[:, 0], xy[:, 1])
            return np.column_stack([a, zz])
        bad_xy   = _pts_in_band(quality_report.get('bad_edge_xy',    np.empty((0, 2))))
        loop_xy  = _pts_in_band(quality_report.get('loop_points_xy', np.empty((0, 2))))
        empty_xy = _pts_in_band(quality_report.get('empty_xy',       np.empty((0, 2))))
        n_issues = len(bad_xy) + len(loop_xy) + len(empty_xy)

        # so the point index i is still readable where a marker covers a skeleton
        # point, tag each issue marker with its nearest skeleton point's index
        sk_tree = cKDTree(np.column_stack([arc, z])) if len(arc) else None
        def _nearest_i(xy):
            if sk_tree is None or not len(xy):
                return np.full(len(xy), -1, int)
            return sk_tree.query(xy)[1].astype(int)   # xy is (K,2) [arc, z]

        if len(bad_xy):
            fig.add_trace(go.Scatter(
                x=bad_xy[:, 1], y=bad_xy[:, 0], mode='markers',
                name=f'bad connection ({len(bad_xy)})',
                marker=dict(symbol='x', size=7, color='blue', line=dict(width=1)),
                customdata=_nearest_i(bad_xy),
                hovertemplate="BAD CONNECTION<br>nearest i=%{customdata}<br>"
                              "arc=%{y:.4f}<br>z=%{x:.4f}<extra></extra>"))
        if len(loop_xy):
            fig.add_trace(go.Scatter(
                x=loop_xy[:, 1], y=loop_xy[:, 0], mode='markers',
                name=f'loop ({len(loop_xy)} pts)',
                marker=dict(symbol='circle-open', size=6, color='magenta', line=dict(width=1)),
                customdata=_nearest_i(loop_xy),
                hovertemplate="LOOP<br>i=%{customdata}<br>"
                              "arc=%{y:.4f}<br>z=%{x:.4f}<extra></extra>"))
        if len(empty_xy):
            fig.add_trace(go.Scatter(
                x=empty_xy[:, 1], y=empty_xy[:, 0], mode='markers',
                name=f'empty region ({len(empty_xy)})',
                marker=dict(symbol='square-open', size=8, color='black', line=dict(width=1)),
                hovertemplate="EMPTY REGION<br>arc=%{y:.4f}<br>z=%{x:.4f}<extra></extra>"))

    ttl = title or f'{crown_label} — 2D skeleton'
    if n_issues is not None:
        ttl += '  (clean)' if n_issues == 0 else f'  ({n_issues} issue marker(s))'

    # Size the figure to the data aspect (equal px per unit in z and arc) so the
    # default view isn't stretched. We avoid scaleanchor (which would disable
    # scroll zoom); free axes still allow scroll/drag zoom.
    allz = np.concatenate([z, s_z]) if len(s_z) else z
    alla = np.concatenate([arc, s_arc]) if len(s_arc) else arc
    z_rng = float(np.ptp(allz)) or 1.0
    a_rng = float(np.ptp(alla)) or 1.0
    scale = 780.0 / max(z_rng, a_rng)
    fig_w = int(np.clip(z_rng * scale + 130, 380, 1500))
    fig_h = int(np.clip(a_rng * scale + 120, 380, 1100))

    fig.update_layout(
        template='plotly_white', width=fig_w, height=fig_h,
        margin=dict(l=55, r=20, t=60, b=45), title=ttl,
        xaxis_title='z', yaxis=dict(title='arc'),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0))
    pio.write_html(fig, out_path, auto_open=False, config={'scrollZoom': True})
    return out_path


# =====================================================================================
# Pipeline step orchestrators (Steps 1-9)
# -------------------------------------------------------------------------------------
# One function per notebook step: each takes explicit parameters (no notebook globals),
# writes the same intermediate files into ``output_dir``, and returns the data the next
# step needs. The Step-1 notebook wires these together in three thin wrappers.
# =====================================================================================


def sample_stent_points(mesh, stent_name, output_dir, n_points=None,
                        samples_per_face=1, max_display=500_000,
                        remove_supports=False, random_seed=0):
    """Sample the STL surface into a cylindrical-coordinate point cloud (Steps 1-2).

    Auto-decides the sample count from the stent size (small/short stents -> ~1e6,
    large ones -> ~1e7, scaled from the mesh face count) unless ``n_points`` forces
    a count. Aligns the PCA axis to [0,0,1] via ``preprocess_stent``, renders an
    HTML scatter (hover shows point_id) and saves ``sampling_points.csv``. Returns
    ``{stent_df, stent_features, stent_centerline_direction}``.
    """
    pre_stent_info = compute_pre_stent_size_ratio(mesh)
    pre_size_ratio = pre_stent_info['size_ratio']
    pre_length     = pre_stent_info['length']
    target_samples = int(1e6) if (pre_size_ratio < 10 or pre_length < 15) else int(1e7)

    auto_n_samples = len(mesh.faces)            # samples_per_face = 1
    if auto_n_samples > 10 * target_samples:
        auto_n_samples = auto_n_samples // 10
    elif auto_n_samples < target_samples * 0.5 and auto_n_samples > 0.1 * target_samples:
        auto_n_samples = auto_n_samples * 10
    elif auto_n_samples < target_samples * 0.1:
        auto_n_samples = auto_n_samples * 100

    print(f"[preanalysis] size_ratio={pre_size_ratio:.2f}, length={pre_length:.2f} mm "
          f"-> target_samples={target_samples:,}")
    print(f"[preanalysis] mesh faces={len(mesh.faces):,} -> auto n_samples={auto_n_samples:,}")

    n_use = auto_n_samples if n_points is None else int(n_points)
    print(f"[preanalysis] sampling {n_use:,} points "
          f"({'auto' if n_points is None else 'overridden by n_points'})")

    pre = preprocess_stent(
        mesh=mesh, n_samples=n_use, samples_per_face=samples_per_face,
        n_thickness_slices=100, slice_cutoff=5, thickness_calc_plot=False,
        remove_supports=remove_supports, random_seed=random_seed)
    stent_df       = pre['stent_df']
    stent_features = pre['stent_features']
    centerline_dir = pre['stent_centerline_direction']

    sampling_html = os.path.join(output_dir, 'sampling_points.html')
    sampling_csv  = os.path.join(output_dir, 'sampling_points.csv')
    plot_points_3d_html(stent_df, 'point_id', sampling_html,
                        title=f'{stent_name} sampling', max_display=max_display)
    print(f"[plot] {sampling_html}  (inspect the sampled point cloud)")

    stent_df[['point_id', 'r', 'theta', 'z_cylindrical', 'x', 'y', 'z']].to_csv(
        sampling_csv, index=False)
    print(f"\n[saved] {sampling_csv}  ({len(stent_df):,} points)")

    return {'stent_df': stent_df, 'stent_features': stent_features,
            'stent_centerline_direction': centerline_dir}


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


def _downsample_surface_pair(a, b, n=40000, seed=0):
    """Downsample paired arrays to ``n`` points (display only)."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) <= n:
        return a, b
    idx = np.random.default_rng(seed).choice(len(a), n, replace=False)
    return a[idx], b[idx]


def skeletonize_crowns_2d(stent_df, stent_features, crown_edges, conn_radius_3d,
                          output_dir, auto_tune=True, pixels_per_strut=10, dilate_px=3,
                          pad_fraction=0.20, tune_time_limit=120, quality_gamma=2.0,
                          seam_tol_frac=0.01, crown_halo_frac=0.4):
    """Skeletonise each crown in 2D and store the results in a CROWN_2D dict (Step 5).

    Segments the whole stent once (each crown -> 3 z-pieces) for the region map +
    ``region_allowed`` adjacency, then per crown (with a ``crown_halo_frac`` z-halo):
    unroll -> (auto-tune or fixed) skeletonise -> trim to the crown band. Detected
    bad connections are auto-cleaned (whole-bridge removal) and the skeleton
    re-checked so per-crown plots flag only what the detector MISSED. Writes
    ``skeleton_plots/crown_XX.html`` / ``crown_XX_convergence.html`` / ``crown_XX_2d.csv``.
    Returns ``{crown_2d, crown_order}``.
    """
    r_mid           = stent_features['r_mid']
    strut_thickness = stent_features['strut_thickness']

    plots_dir = os.path.join(output_dir, 'skeleton_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # --- single segmentation for the whole stent (each crown -> 3 pieces) ---
    segmented      = segment_stent(stent_df, strut_thickness, conn_radius_3d,
                                   n_sub_per_crown=3)
    seg_df_full    = segmented['stent_df']          # whole stent, now has 'region'
    region_allowed = segmented['region_allowed']    # global region adjacency

    crown_order = (seg_df_full.groupby('crown_id')['z_cylindrical'].mean()
                              .sort_values().index.tolist())
    use_halo = crown_edges is not None and len(crown_edges) >= 2
    z_all    = seg_df_full['z_cylindrical'].values

    crown_2d = {}          # crown_label -> per-crown 2D skeleton record
    print(f"Skeletonising {len(crown_order)} crowns "
          f"({'auto-tune' if auto_tune else 'fixed params'})...")

    for ci, c in enumerate(crown_order):
        label = f"crown_{int(c):02d}"
        own_z = seg_df_full.loc[seg_df_full['crown_id'] == c, 'z_cylindrical']
        if use_halo:
            k = int(np.clip(np.digitize(own_z.mean(), crown_edges) - 1,
                            0, len(crown_edges) - 2))
            z_lo, z_hi = float(crown_edges[k]), float(crown_edges[k + 1])
            halo    = crown_halo_frac * (z_hi - z_lo)
            seg_sub = seg_df_full[(z_all >= z_lo - halo) & (z_all <= z_hi + halo)].copy()
        else:
            z_lo, z_hi = float(own_z.min()), float(own_z.max())
            seg_sub    = seg_df_full[seg_df_full['crown_id'] == c].copy()

        opened = open_stent_to_plane(seg_sub, r_mid, pad_fraction)
        arc_flat, z_flat = opened['arc_flat'], opened['z_flat']

        print(f"\n===== {label} ({ci + 1}/{len(crown_order)}) — "
              f"{len(seg_sub):,} pts, band z=[{z_lo:.4f}, {z_hi:.4f}] =====")

        if auto_tune:
            tuned   = tune_skeleton_params(
                arc=arc_flat, z=z_flat, stent_df=seg_sub, stent_features=stent_features,
                region_allowed=region_allowed, pps0=pixels_per_strut, dil0=dilate_px,
                pad_fraction=pad_fraction, seam_tol_frac=seam_tol_frac,
                time_limit=tune_time_limit, quality_gamma=quality_gamma,
                plot=False, verbose=True)
            sk_2d          = tuned['skeleton_2d']
            history        = tuned['history']
            quality_report = tuned['quality_report']
            diag_pps, diag_dil = tuned['best_pps'], tuned['best_dilate_px']
        else:
            sk_2d = compute_skeleton_2d(
                arc_flat=arc_flat, z_flat=z_flat, arc_min_flat=opened['arc_min_flat'],
                arc_max_flat=opened['arc_max_flat'], circumference=opened['circumference'],
                stent_df=seg_sub, stent_geometry=stent_features,
                pixels_per_strut=pixels_per_strut, dilate_px=dilate_px,
                pad_fraction=pad_fraction, plot=False, seam_tol_frac=seam_tol_frac)
            history = None
            quality_report = check_skeleton_quality(
                df_skeleton_2d=sk_2d['df_skeleton_2d'], pixel_size=sk_2d['pixel_size'],
                stent_df=seg_sub, r_mid=r_mid, region_allowed=region_allowed,
                strut_thickness=strut_thickness, verbose=True)
            diag_pps, diag_dil = pixels_per_strut, dilate_px

        # trim the final skeleton to the crown's own z-band (drop the halo)
        skel_df2d = sk_2d['df_skeleton_2d']
        arc_all   = skel_df2d['arc'].to_numpy()
        z_2d_all  = skel_df2d['z'].to_numpy()
        z_tol = 0.01 * strut_thickness
        tmask = (z_2d_all >= z_lo - z_tol) & (z_2d_all <= z_hi + z_tol)
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

        # store this crown's 2D skeleton (editable in Step 5.5)
        surf_arc_ds, surf_z_ds = _downsample_surface_pair(arc_flat, z_flat)
        crown_2d[label] = {
            'crown_id'  : c,
            'arc'       : arc_c.copy(),
            'z'         : z_c.copy(),
            'pixel_size': sk_2d['pixel_size'],
            'z_lo'      : z_lo,
            'z_hi'      : z_hi,
            'surf_arc'  : surf_arc_ds,
            'surf_z'    : surf_z_ds,
            'n_edits'   : 0,
        }

        # (1) the 2D skeleton alone (single panel, zoomable, remaining issues flagged)
        skel_html = os.path.join(plots_dir, f'{label}.html')
        plot_crown_skeleton_2d_html(
            arc_c, z_c, surf_arc_ds, surf_z_ds, skel_html, label,
            crown_band=(z_lo, z_hi), quality_report=quality_report)
        # (2) the tuning convergence / issue-summary, in a separate file
        conv_html = os.path.join(plots_dir, f'{label}_convergence.html')
        plot_crown_convergence_html(history, conv_html, c, quality_report=quality_report,
                                    pps=diag_pps, dil_px=diag_dil)
        # (3) a plain 2D CSV (i, arc, z) for the record
        pd.DataFrame({'i': np.arange(len(arc_c)), 'arc': arc_c, 'z': z_c}).to_csv(
            os.path.join(plots_dir, f'{label}_2d.csv'), index=False)
        print(f"  {label}: {len(arc_c):,} skeleton pts  ->  {skel_html}")

    n_total = sum(len(v['arc']) for v in crown_2d.values())
    print(f"\nStored {len(crown_2d)} crowns ({n_total:,} 2D skeleton points) in CROWN_2D.")
    print(f"Per-crown plots (crown_XX.html + _convergence.html) + CSVs saved in {plots_dir}")
    print("Detected bad connections were auto-cleaned; review each crown_XX.html for any "
          "remaining errors and fix them by hand in the manual-edit step.")

    return {'crown_2d': crown_2d, 'crown_order': crown_order}


def save_crown_2d_checkpoint(crown_2d, stent_features, stent_centerline_direction,
                             r_mid, strut_thickness, circumference, crown_edges,
                             output_dir, verbose=True):
    """Pickle the per-crown 2D skeletons + scalars to ``crown_2d.pkl`` (Step 5.4).

    Lets the manual-edit step resume after a kernel restart without re-running the
    Step-5 auto-tune. The surface cloud is reused from ``crown_points.csv``.
    Returns the checkpoint path.
    """
    crown2d_pkl = os.path.join(output_dir, 'crown_2d.pkl')
    with open(crown2d_pkl, 'wb') as f:
        pickle.dump({
            'CROWN_2D'                  : crown_2d,
            'stent_features'            : stent_features,
            'stent_centerline_direction': np.asarray(stent_centerline_direction),
            'r_mid'                     : float(r_mid),
            'strut_thickness'           : float(strut_thickness),
            'circumference'             : float(circumference),
            'crown_edges'               : crown_edges,
        }, f)
    if verbose:
        print(f"[checkpoint] saved {len(crown_2d)} per-crown 2D skeletons -> {crown2d_pkl}")
    return crown2d_pkl


def load_crown_2d_checkpoint(output_dir):
    """Reload the ``crown_2d.pkl`` checkpoint + point cloud after a kernel restart.

    Restores the per-crown 2D skeletons and the scalars the manual-edit / wrap
    steps rely on, and reads the surface cloud from ``crown_points.csv``. Returns a
    state dict (``crown_2d``, ``stent_features``, ``stent_centerline_direction``,
    ``r_mid``, ``strut_thickness``, ``circumference``, ``crown_edges``, ``stent_df``).
    """
    crown2d_pkl = os.path.join(output_dir, 'crown_2d.pkl')
    crown_csv   = os.path.join(output_dir, 'crown_points.csv')
    for p in (crown2d_pkl, crown_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing {p} - run Steps 1-5.4 first (per-crown checkpoint not found).")

    with open(crown2d_pkl, 'rb') as f:
        ck = pickle.load(f)

    os.makedirs(os.path.join(output_dir, 'skeleton_plots'), exist_ok=True)
    print(f"[resume] loading point cloud from {crown_csv} ...")
    stent_df = pd.read_csv(crown_csv)

    state = {
        'crown_2d'                  : ck['CROWN_2D'],
        'stent_features'            : ck['stent_features'],
        'stent_centerline_direction': np.asarray(ck['stent_centerline_direction']),
        'r_mid'                     : float(ck['r_mid']),
        'strut_thickness'           : float(ck['strut_thickness']),
        'circumference'             : float(ck['circumference']),
        'crown_edges'               : ck.get('crown_edges'),
        'stent_df'                  : stent_df,
    }
    print(f"[resume] restored {len(state['crown_2d'])} per-crown 2D skeletons + "
          f"{len(stent_df):,} surface points. Ready for the manual-edit step.")
    return state


def _parse_two_ids(s):
    """Parse a 'a, b' / 'a b' string into exactly two int point indices."""
    vals = [int(x) for x in (s or '').replace(',', ' ').split()]
    if len(vals) != 2:
        raise ValueError("please give exactly two point indices, e.g. '12, 40'")
    return vals[0], vals[1]


def _render_crown_2d(crown_2d, label, plots_dir, changed_idx=None, suffix=None):
    """Render one crown's current 2D skeleton to ``crown_XX[_edited_N].html``."""
    rec  = crown_2d[label]
    name = f"{label}_edited_{rec['n_edits']}" if suffix == 'edited' else label
    out  = os.path.join(plots_dir, f"{name}.html")
    plot_crown_skeleton_2d_html(
        rec['arc'], rec['z'], rec['surf_arc'], rec['surf_z'], out, label,
        crown_band=(rec['z_lo'], rec['z_hi']), changed_idx=changed_idx,
        title=f"{name} — 2D skeleton")
    return out


def edit_crowns_2d_interactive(crown_2d, stent_features, stent_centerline_direction,
                               r_mid, strut_thickness, circumference, crown_edges,
                               output_dir):
    """Interactively fix per-crown 2D skeleton defects by naming two anchors (Step 5.5).

    Prompts for a crown and a problem type: 'loop' collapses a bubble into a single
    straight path (``fix_crown_loop_2d``); 'connection' deletes a whole wrong bridge
    up to its bounding junctions (``fix_crown_connection_2d``). Each edit is previewed
    as ``crown_XX_edited_N.html`` and kept only if confirmed (kept edits re-save
    ``crown_2d.pkl`` so a restart reloads the edited crowns). Mutates and returns
    ``crown_2d``.
    """
    plots_dir = os.path.join(output_dir, 'skeleton_plots')
    os.makedirs(plots_dir, exist_ok=True)

    if input("Do you want to manually edit any crown? [y/n] ").strip().lower() == 'y':
        while True:
            ans = input("\nWhich crown would you like to change? "
                        "(e.g. crown_01, crown_02, ...; empty = done) ").strip()
            if not ans:
                break
            try:
                label = ans if ans.startswith('crown_') else f"crown_{int(ans):02d}"
            except ValueError:
                print(f"  '{ans}' is not a valid crown name. "
                      f"Available: {', '.join(crown_2d)}")
                continue
            if label not in crown_2d:
                print(f"  '{label}' not found. Available: {', '.join(crown_2d)}")
                continue

            rec  = crown_2d[label]
            prob = input("What is the problem in the skeleton? (loop / connection) ").strip().lower()

            if prob.startswith('loop'):
                print("  -> I'll remove the loop (bubble) and join the two points you give "
                      "with a single evenly-spaced straight line. Everything else stays.")
                a, b = _parse_two_ids(input("  Two point indices at the loop ends (e.g. 12, 40): "))
                new_arc, new_z, changed = fix_crown_loop_2d(rec['arc'], rec['z'],
                                                            rec['pixel_size'], a, b)
            elif prob.startswith('conn'):
                print("  -> I'll remove the WHOLE wrong bridge (extending up to where it "
                      "meets the struts) to open a clean gap. Everything else stays.")
                a, b = _parse_two_ids(input("  Two point indices ON the wrong bridge itself, "
                                            "e.g. its two ends (e.g. 12, 40): "))
                new_arc, new_z, changed = fix_crown_connection_2d(rec['arc'], rec['z'],
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
            out = _render_crown_2d(crown_2d, label, plots_dir, changed_idx=changed,
                                   suffix='edited')
            print(f"  preview saved as {label}_edited_{rec['n_edits']} -> {out}")

            if input("  Are you happy with this edit? (y/n) ").strip().lower().startswith('y'):
                save_crown_2d_checkpoint(crown_2d, stent_features, stent_centerline_direction,
                                         r_mid, strut_thickness, circumference, crown_edges,
                                         output_dir, verbose=False)
                print(f"  kept edit {rec['n_edits']} on {label}. (crown_2d.pkl updated)")
            else:
                rec['arc'], rec['z'], rec['n_edits'] = prev_arc, prev_z, prev_n
                try:
                    os.remove(out)
                except OSError:
                    pass
                print(f"  reverted — {label} restored to its previous state. You can try again.")

    return crown_2d


def assemble_2d_skeleton(crown_2d):
    """Concatenate the (edited) per-crown 2D skeletons into one flat skeleton.

    Orders crowns by ``crown_id`` and stacks their ``arc`` / ``z`` / per-point
    ``pixel_size``. Returns ``{skel_arc, skel_z, skel_px, pixel_size}`` (the last a
    median pixel size across crowns), ready for the 3D wrap.
    """
    order    = sorted(crown_2d, key=lambda L: crown_2d[L]['crown_id'])
    skel_arc = np.concatenate([crown_2d[L]['arc'] for L in order])
    skel_z   = np.concatenate([crown_2d[L]['z']   for L in order])
    skel_px  = np.concatenate([np.full(len(crown_2d[L]['arc']), crown_2d[L]['pixel_size'])
                               for L in order])
    pixel_size = float(np.median([crown_2d[L]['pixel_size'] for L in order]))
    print(f"\nAssembled 2D skeleton: {len(skel_arc):,} points "
          f"| pixel_size median = {pixel_size:.5f}")
    return {'skel_arc': skel_arc, 'skel_z': skel_z, 'skel_px': skel_px,
            'pixel_size': pixel_size}


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


def group_skeleton_curves(skeleton_points_df):
    """Split the skeleton into curves (degree-2 chains bounded by junctions /
    endpoints, or closed degree-2 loops). Returns lists of skeleton_point_ids."""
    dfc = skeleton_points_df.reset_index(drop=True)
    adj = {}
    for _, row in dfc.iterrows():
        nbrs = row['neighbor_ids']
        if isinstance(nbrs, str):
            nbrs = ast.literal_eval(nbrs)
        adj[int(row['skeleton_point_id'])] = list(int(n) for n in nbrs)

    degree   = {pid: len(nbrs) for pid, nbrs in adj.items()}
    specials = {pid for pid, d in degree.items() if d != 2}
    curves, visited_edges = [], set()

    def walk(start, nxt):
        path = [start, nxt]
        visited_edges.add(frozenset((start, nxt)))
        prev, cur = start, nxt
        while cur not in specials:
            others = [n for n in adj[cur] if n != prev]
            if not others:
                break
            nb = others[0]
            visited_edges.add(frozenset((cur, nb)))
            path.append(nb)
            prev, cur = cur, nb
            if cur == start:
                break
        return path

    for s in specials:
        for nb in adj[s]:
            if frozenset((s, nb)) not in visited_edges:
                curves.append(walk(s, nb))
    for pid in adj:
        for nb in adj[pid]:
            if frozenset((pid, nb)) not in visited_edges:
                curves.append(walk(pid, nb))
    return curves


def fit_curve_spline(point_ids, coords, every, k, s):
    """Fit a B-spline through every Nth point of a curve; fall back to the raw
    control polyline (tck=None) if the fit fails."""
    pts  = coords.loc[point_ids].to_numpy()
    keep = np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]
    pts  = pts[keep]
    if len(pts) < 2:
        return None
    is_loop = (point_ids[0] == point_ids[-1]) and len(pts) > 3
    every   = max(1, min(every, (len(pts) - 1) // max(k, 1)))
    idx     = np.unique(np.r_[np.arange(0, len(pts), every), len(pts) - 1])
    ctrl    = pts[idx]
    if is_loop:
        if np.allclose(ctrl[0], ctrl[-1]):
            ctrl = ctrl[:-1]
        if len(ctrl) < 4:
            is_loop = False
    seg_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    kk = min(k, len(ctrl) - 1)
    if kk >= 1:
        try:
            (tck, u) = splprep([ctrl[:, 0], ctrl[:, 1], ctrl[:, 2]],
                               s=s, k=kk, per=int(is_loop))
            return {'tck': tck, 'u': u, 'ctrl': ctrl, 'n_ctrl': len(ctrl),
                    'k': kk, 'is_loop': is_loop, 'length': seg_len}
        except Exception:
            pass
    return {'tck': None, 'u': None, 'ctrl': ctrl, 'n_ctrl': len(ctrl),
            'k': 0, 'is_loop': False, 'length': seg_len}


def _spline_record(spl):
    """Neutral (degree, knot_vector, control_points) record for one curve, ready
    to rebuild as splinepy.BSpline. tck=None -> degree-1 control polyline."""
    if spl is None:
        return None
    if spl['tck'] is not None:
        t, c, k = spl['tck']
        ctrl = np.asarray(c).T                       # (n_ctrl, 3) xyz
        return {'degree': int(k),
                'knot_vector': [float(v) for v in t],
                'control_points': ctrl.tolist(),
                'is_loop': bool(spl['is_loop']),
                'length': float(spl['length'])}
    ctrl = np.asarray(spl['ctrl'])
    return {'degree': 1, 'knot_vector': None,        # polyline fallback
            'control_points': ctrl.tolist(),
            'is_loop': False, 'length': float(spl['length'])}


def fit_skeleton_splines(skeleton_df, output_dir, spline_every=10, spline_degree=3,
                         smooth=0.0):
    """Group the skeleton graph into curves and fit a B-spline per curve (Step 7).

    Renders the smooth curves to ``splines.html`` and exports ``skeleton_splines.json``
    (per-curve degree / knot_vector / control_points), which rebuilds directly as a
    ``splinepy.BSpline`` for BeamMe. Returns ``{curves, splines}``.
    """
    coords  = skeleton_df.set_index('skeleton_point_id')[['x', 'y', 'z']]
    curves  = group_skeleton_curves(skeleton_df)
    splines = [fit_curve_spline(c, coords, spline_every, spline_degree, smooth)
               for c in curves]
    print(f"Grouped {len(curves)} curves; fitted "
          f"{sum(s is not None for s in splines)} splines.")

    splines_html = os.path.join(output_dir, 'splines.html')
    plot_splines_html(splines, splines_html)

    splines_json = os.path.join(output_dir, 'skeleton_splines.json')
    with open(splines_json, 'w') as f:
        json.dump({
            'n_curves': len(splines),
            'note': ('Per-curve B-splines. Rebuild each as '
                     'splinepy.BSpline(degrees=[degree], knot_vectors=[knot_vector], '
                     'control_points=control_points) then pass to BeamMe '
                     'create_beam_mesh_from_splinepy (or create_beam_mesh_parametric_curve '
                     'via the spline evaluator). control_points are (n_ctrl, 3) xyz in mm. '
                     'knot_vector=null marks a degree-1 polyline fallback.'),
            'curves': [_spline_record(s) for s in splines],
        }, f, indent=2)

    print(f"[saved] {splines_html}")
    print(f"[saved] {splines_json}  ({len(splines)} curves)")
    return {'curves': curves, 'splines': splines}


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


def _to_arc_z(x, y, z, r_mid):
    """3D (x, y, z) -> unrolled (z_axial, arc) with arc = r_mid * atan2(y, x)."""
    return np.asarray(z, float), r_mid * np.arctan2(np.asarray(y, float),
                                                    np.asarray(x, float))


def _break_seam(z_ax, arc, thresh):
    """Insert NaNs where arc jumps across the theta seam so the polyline does not
    draw a spurious wrap-around segment across the plot."""
    z_ax = np.asarray(z_ax, float).copy()
    arc  = np.asarray(arc, float).copy()
    for j in np.where(np.abs(np.diff(arc)) > thresh)[0][::-1]:
        z_ax = np.insert(z_ax, j + 1, np.nan)
        arc  = np.insert(arc,  j + 1, np.nan)
    return z_ax, arc


def _plotly_decode(o):
    """Decode a Plotly typed-array ({'bdata','dtype'[, 'shape']}) or a plain list."""
    if isinstance(o, dict) and 'bdata' in o:
        dt = {'f8': '<f8', 'f4': '<f4', 'i1': 'i1', 'i2': '<i2', 'i4': '<i4',
              'u1': 'u1', 'u4': '<u4'}[o['dtype']]
        a = np.frombuffer(base64.b64decode(o['bdata']), dtype=dt)
        if 'shape' in o:
            a = a.reshape(tuple(int(s) for s in str(o['shape']).split(',')))
        return a
    return np.asarray(o)


def _load_convergence(path):
    """Pull trace data from a saved crown_XX_convergence.html.
    Returns one of:
      {'kind':'convergence', 'total':(x,y), 'defect':(x,y), 'quality':(x,y)}
      {'kind':'quality_bar', 'x':labels, 'y':counts, 'colors':colors}
    or None on failure."""
    try:
        html = open(path, encoding='utf-8').read()
        i = html.rfind('Plotly.newPlot(')
        j = html.find('[', i)
        depth, k, instr, esc = 0, j, False, False        # bracket scan, string-aware
        while k < len(html):
            ch = html[k]
            if instr:
                if esc:            esc = False
                elif ch == '\\':   esc = True
                elif ch == '"':    instr = False
            elif ch == '"':        instr = True
            elif ch == '[':        depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        data = json.loads(html[j:k + 1])
        # --- auto-tune ON: scatter traces named total / defect / quality ---
        out = {}
        for tr in data:
            nm = tr.get('name')
            if nm in ('total', 'defect', 'quality') and 'x' in tr and 'y' in tr:
                out[nm] = (_plotly_decode(tr['x']).astype(float),
                           _plotly_decode(tr['y']).astype(float))
        if out:
            out['kind'] = 'convergence'
            return out
        # --- auto-tune OFF: single bar trace (quality summary) ---
        for tr in data:
            if tr.get('type') == 'bar' and 'x' in tr and 'y' in tr:
                return {'kind': 'quality_bar',
                        'x':      list(_plotly_decode(tr['x'])),
                        'y':      _plotly_decode(tr['y']).astype(float),
                        'colors': tr.get('marker', {}).get('color', ['steelblue'])}
        return None
    except Exception as e:
        print(f"  [tuning] could not read {os.path.basename(path)}: {e}")
        return None


def _hue_gap(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def _band_conv(k, crown_order, n_bands, conv_files, conv_dir):
    """Map unrolled band k to its crown_XX_convergence.html path + crown id."""
    if crown_order is not None and len(crown_order) == n_bands:
        cid = int(crown_order[k])
        return os.path.join(conv_dir, f'crown_{cid:02d}_convergence.html'), cid
    if k < len(conv_files):
        base = os.path.basename(conv_files[k])
        cid  = int(base.split('_')[1])
        return conv_files[k], cid
    return None, None


def plot_skeleton_splines_2d(skeleton_curves, skeleton_splines, stent_df, r_mid,
                             circumference, crown_edges, crown_order, output_dir,
                             stent_name):
    """Render the unrolled 2D skeleton with a per-crown tuning strip (Steps 9-10).

    Draws every fitted spline on an unrolled (z, arc) plane (touching curves given
    different hues), overlays the point cloud + crown boundaries, and puts each
    crown's tuning-convergence curves (read back from its convergence HTML) above
    its band. Saves ``skeleton_splines_2d.png`` and a self-contained
    ``skeleton_splines_2d.html``.
    """
    # --- varied colours, touching curves differ in HUE (rotating greedy colouring) ---
    palette = []
    for _nm in ('tab20', 'tab20b', 'tab20c'):
        _cm = plt.get_cmap(_nm)
        palette.extend(_cm(i) for i in range(_cm.N))
    n_pal   = len(palette)
    pal_hue = np.array([rgb_to_hsv(to_rgb(c))[0] for c in palette])
    HUE_TOL = 0.06                                 # min circular hue gap to neighbours

    pt2curves = {}
    for ci, cpts in enumerate(skeleton_curves):
        for p in cpts:
            pt2curves.setdefault(int(p), []).append(ci)
    curve_adj = {ci: set() for ci in range(len(skeleton_curves))}
    for shared in pt2curves.values():
        for a in shared:
            for b in shared:
                if a != b:
                    curve_adj[a].add(b)

    curve_color_idx = {}
    ptr = 0
    for ci in range(len(skeleton_curves)):
        nbr_idx  = [curve_color_idx[n] for n in curve_adj[ci] if n in curve_color_idx]
        nbr_hues = [pal_hue[j] for j in nbr_idx]
        chosen   = None
        for step in range(n_pal):
            cand = (ptr + step) % n_pal
            if cand in nbr_idx:
                continue
            if all(_hue_gap(pal_hue[cand], h) >= HUE_TOL for h in nbr_hues):
                chosen = cand
                break
        if chosen is None:
            for step in range(n_pal):
                cand = (ptr + step) % n_pal
                if cand not in nbr_idx:
                    chosen = cand
                    break
        curve_color_idx[ci] = chosen
        ptr = (chosen + 1) % n_pal

    # --- crown boundaries (z) for the vertical dashed lines ---------------------------
    features_path = os.path.join(output_dir, 'stent_features.json')
    crown_lines   = None
    if os.path.exists(features_path):
        with open(features_path) as f:
            cb = json.load(f).get('crown_boundaries')
        if cb is not None:
            crown_lines = np.asarray(cb, float).ravel()
    if crown_lines is None and crown_edges is not None \
            and len(np.asarray(crown_edges).ravel()) >= 2:
        crown_lines = np.asarray(crown_edges, float).ravel()
    if crown_lines is None and 'crown_id' in stent_df.columns:
        g   = (stent_df.groupby('crown_id')['z'].agg(['min', 'max', 'mean'])
                       .sort_values('mean'))
        lo  = g['min'].to_numpy()
        hi  = g['max'].to_numpy()
        crown_lines = np.concatenate([[lo[0]], 0.5 * (hi[:-1] + lo[1:]), [hi[-1]]])
    crown_lines = None if crown_lines is None else np.sort(np.asarray(crown_lines, float))

    # --- map each crown band to its convergence file ---
    conv_dir   = os.path.join(output_dir, 'skeleton_plots')
    conv_files = sorted(glob.glob(os.path.join(conv_dir, 'crown_*_convergence.html')))
    n_bands    = 0 if crown_lines is None else len(crown_lines) - 1

    # --- figure: main unrolled plot (bottom) + per-crown tuning strip (top) ------------
    L, B, W, H = 0.05, 0.06, 0.93, 0.52            # main axes box (figure coords)
    TY0, TH    = 0.66, 0.28                         # tuning strip band (figure coords)

    fig = plt.figure(figsize=(16, 9))
    ax  = fig.add_axes([L, B, W, H])

    # point-cloud underlay (grey)
    ax.scatter(stent_df['z'].to_numpy(), r_mid * stent_df['theta'].to_numpy(),
               s=1, c='0.8', alpha=0.5, linewidths=0, rasterized=True, zorder=1)

    # each spline curve, coloured so touching curves differ
    seam_thresh = 0.5 * circumference
    n_drawn     = 0
    for ci, spl in enumerate(skeleton_splines):
        if spl is None:
            continue
        if spl['tck'] is not None:
            xx, yy, zz = splev(np.linspace(0.0, 1.0, 200), spl['tck'])
        else:
            ctrl = np.asarray(spl['ctrl'])
            xx, yy, zz = ctrl[:, 0], ctrl[:, 1], ctrl[:, 2]
        z_ax, arc = _break_seam(*_to_arc_z(xx, yy, zz, r_mid), seam_thresh)
        ax.plot(z_ax, arc, '-', lw=1.8, color=palette[curve_color_idx[ci]], zorder=2)
        n_drawn += 1

    if crown_lines is not None:
        xlo, xhi = float(crown_lines[0]), float(crown_lines[-1])
        span = (xhi - xlo) or 1.0
        xlo -= 0.01 * span
        xhi += 0.01 * span
        ax.set_xlim(xlo, xhi)
        for e in crown_lines:
            ax.axvline(e, ls='--', lw=1.6, color='k', alpha=0.9, zorder=10)
    else:
        xlo, xhi = ax.get_xlim()
        span = (xhi - xlo) or 1.0

    fig.suptitle(f'{stent_name} — unrolled 2D skeleton + per-crown tuning',
                 fontsize=11, y=0.995)
    ax.set_xlabel('z  (axial position, mm)')
    ax.set_ylabel('arc = r_mid · θ  (circumferential, mm)')

    # per-crown tuning plots above their bands
    tune_colors = (('total', 'black'), ('defect', 'royalblue'), ('quality', 'orange'))
    tax = None
    n_tuned = 0
    for k in range(n_bands):
        a, b = float(crown_lines[k]), float(crown_lines[k + 1])
        fx0  = L + (a - xlo) / (xhi - xlo) * W
        fw   = (b - a) / (xhi - xlo) * W
        pad  = 0.14 * fw
        tax  = fig.add_axes([fx0 + pad, TY0, max(fw - 2 * pad, 1e-3), TH])
        path, cid = _band_conv(k, crown_order, n_bands, conv_files, conv_dir)
        conv = _load_convergence(path) if path and os.path.exists(path) else None
        if conv is None:
            tax.axis('off')
            continue
        if conv['kind'] == 'convergence':
            for nm, col in tune_colors:
                if nm in conv:
                    xs, ys = conv[nm]
                    tax.plot(xs, ys, '-', color=col, lw=1.0, marker='o', ms=2, label=nm)
            tax.margins(x=0.03)
        else:  # quality_bar
            short_x = [l.replace(' ', '\n') for l in conv['x']]
            tax.bar(short_x, conv['y'], color=conv['colors'], width=0.6)
            tax.set_ylim(bottom=0)
        tax.set_title(f'crown {cid}', fontsize=7, pad=2)
        tax.tick_params(labelsize=5, length=2, pad=1)
        n_tuned += 1

    # shared legend on the last tuning axes (loc='best' finds the emptiest spot)
    if n_tuned and tax is not None:
        handles = [Line2D([0], [0], color=col, marker='o', ms=3, lw=1.2, label=nm)
                   for nm, col in tune_colors]
        tax.legend(handles=handles, fontsize=7, loc='best', frameon=True,
                   title='tuning error', title_fontsize=7)

    unrolled_png  = os.path.join(output_dir, 'skeleton_splines_2d.png')
    unrolled_html = os.path.join(output_dir, 'skeleton_splines_2d.html')
    fig.savefig(unrolled_png, dpi=150, bbox_inches='tight')

    # embed the PNG as base64 in a self-contained HTML file
    with open(unrolled_png, 'rb') as _f:
        _png_b64 = base64.b64encode(_f.read()).decode()
    with open(unrolled_html, 'w') as _f:
        _f.write(
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{stent_name} — unrolled 2D skeleton</title></head>'
            '<body style="margin:0;background:#fff;">'
            f'<img src="data:image/png;base64,{_png_b64}" '
            'style="width:100%;height:auto;">'
            '</body></html>'
        )
    plt.show()
    print(f"[saved] {unrolled_png}  ({n_drawn} curves, "
          f"{0 if crown_lines is None else len(crown_lines)} crown boundaries, "
          f"{n_tuned}/{n_bands} crown tuning plots)")
    print(f"[saved] {unrolled_html}")
    return {'png': unrolled_png, 'html': unrolled_html}


def plot_skeleton_splines_trimesh(skeleton_splines, output_dir, show=True):
    """Export the fitted splines as a coloured 3D path (.glb + trimesh .html).

    Evaluates each spline into a polyline, builds a ``trimesh.path.Path3D`` with one
    colour per curve, writes a portable ``skeleton_splines.glb`` and a self-contained
    ``skeleton_splines_trimesh.html``, and (when ``show``) opens the in-notebook view.
    """
    _verts, _ents, _cols = [], [], []
    _off  = 0
    _cmap = plt.get_cmap('tab20')
    _n    = 0
    for spl in skeleton_splines:
        if spl is None:
            continue
        if spl['tck'] is not None:
            xx, yy, zz = splev(np.linspace(0.0, 1.0, 120), spl['tck'])
            pts = np.column_stack([xx, yy, zz])
        else:                                    # degree-1 polyline fallback
            pts = np.asarray(spl['ctrl'], float)
        if len(pts) < 2:
            continue
        _ents.append(Line(np.arange(_off, _off + len(pts))))
        _verts.append(pts)
        _cols.append((np.array(_cmap(_n % 20)) * 255).astype(np.uint8))
        _off += len(pts)
        _n   += 1

    spline_path = trimesh.path.Path3D(entities=_ents, vertices=np.vstack(_verts))
    try:
        spline_path.colors = np.array(_cols, dtype=np.uint8)
    except Exception as e:
        print(f"[trimesh] per-curve colouring skipped: {e}")

    splines_glb   = os.path.join(output_dir, 'skeleton_splines.glb')
    splines_thtml = os.path.join(output_dir, 'skeleton_splines_trimesh.html')
    with open(splines_glb, 'wb') as f:
        f.write(spline_path.scene().export(file_type='glb'))
    try:
        from trimesh.viewer import notebook as _tvn
        with open(splines_thtml, 'w') as f:
            f.write(_tvn.scene_to_html(spline_path.scene()))
        print(f"[saved] {splines_thtml}")
    except Exception as e:
        print(f"[trimesh] html export skipped: {e}")
    print(f"[saved] {splines_glb}  ({_n} spline curves)")

    if show:
        spline_path.show()
    return spline_path


def _feat(stent_features, key):
    """Read a feature value, unwrapping the {'value', 'unit'} form if present."""
    v = stent_features[key]
    return v['value'] if isinstance(v, dict) and 'value' in v else v


def _bspline_from_record(rec):
    """Rebuild a splinepy.BSpline from a skeleton_splines.json curve record.
    knot_vector=None marks a degree-1 polyline fallback -> clamped uniform knots."""
    import splinepy
    ctrl = np.asarray(rec['control_points'], float)
    if rec['knot_vector'] is not None:
        return splinepy.BSpline(degrees=[int(rec['degree'])],
                                knot_vectors=[rec['knot_vector']],
                                control_points=ctrl)
    n  = len(ctrl)                                   # degree-1 polyline fallback
    kv = [0.0, 0.0] + list(np.linspace(0.0, 1.0, n)[1:-1]) + [1.0, 1.0]
    return splinepy.BSpline(degrees=[1], knot_vectors=[kv], control_points=ctrl)


def mesh_skeleton_beams(output_dir, l_el=0.1, youngs_modulus=2.0e5, poisson_ratio=0.3,
                        density=0.0, beam_class_label='Beam3rHerm2Line3'):
    """Mesh the fitted splines into a 1D Simo-Reissner beam mesh with BeamMe (Step 9).

    Rebuilds each ``skeleton_splines.json`` curve as a ``splinepy.BSpline`` and meshes
    it into one BeamMe ``Mesh`` (beam radius = strut_thickness / 2). Material +
    element choices are provisional placeholders; only the geometry is final. Writes
    ``skeleton_beam_mesh_beam.vtu`` and returns the mesh. Imports ``splinepy`` /
    ``beamme`` lazily so the rest of ``stent_funcs`` runs without those heavy deps.
    """
    import splinepy  # noqa: F401  (used indirectly via _bspline_from_record)
    from beamme.core.mesh import Mesh
    from beamme.four_c.material import MaterialReissner
    from beamme.four_c.element_beam import Beam3rHerm2Line3, Beam3rLine2Line2
    from beamme.mesh_creation_functions.beam_splinepy import create_beam_mesh_from_splinepy

    beam_classes = {'Beam3rHerm2Line3': Beam3rHerm2Line3,
                    'Beam3rLine2Line2': Beam3rLine2Line2}
    beam_class = beam_classes[beam_class_label]

    # --- read the stent's saved data from the output folder (self-contained) ---
    with open(os.path.join(output_dir, 'stent_features.json')) as f:
        stent_features = json.load(f)
    with open(os.path.join(output_dir, 'skeleton_splines.json')) as f:
        splines_data = json.load(f)

    strut_thickness = float(_feat(stent_features, 'strut_thickness'))
    print(f"[folder] read stent_features.json from {output_dir}")
    print(f"[folder] strut_thickness = {strut_thickness:.4f} mm")

    beam_radius = strut_thickness / 2.0            # circular cross-section radius (mm)

    beam_mesh = Mesh()
    beam_mat  = MaterialReissner(radius=beam_radius, youngs_modulus=youngs_modulus,
                                 nu=poisson_ratio, density=density)

    n_meshed, n_skipped = 0, 0
    for rec in splines_data['curves']:
        if rec is None or len(rec['control_points']) < 2:
            n_skipped += 1
            continue
        try:
            create_beam_mesh_from_splinepy(beam_mesh, beam_class, beam_mat,
                                           _bspline_from_record(rec), l_el=l_el)
            n_meshed += 1
        except Exception as e:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  [skip] curve failed: {type(e).__name__}: {str(e)[:100]}")

    print(f"\nMeshed {n_meshed}/{splines_data['n_curves']} curves "
          f"({n_skipped} skipped) into a 1D beam mesh:")
    print(f"  {len(beam_mesh.nodes):,} nodes, {len(beam_mesh.elements):,} "
          f"{beam_class_label} elements")
    print(f"  cross-section radius {beam_radius:.4f} mm, target element length "
          f"{l_el:.4f} mm")

    beam_mesh.write_vtk(output_name='skeleton_beam_mesh', output_directory=output_dir)
    print(f"[saved] {os.path.join(output_dir, 'skeleton_beam_mesh_beam.vtu')}")
    return beam_mesh

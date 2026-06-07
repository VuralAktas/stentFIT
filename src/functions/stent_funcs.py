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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d
from skimage.morphology import skeletonize, dilation, closing, disk


def compute_pre_stent_size_ratio(mesh: trimesh.Trimesh) -> dict:
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
) -> dict:

    # ------------------------------------------------------------------
    # 1. Sample surface
    # ------------------------------------------------------------------
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples)

    # ------------------------------------------------------------------
    # 2. PCA — centroid + main axis
    # ------------------------------------------------------------------
    pca_mean  = pts.mean(axis=0)
    centered  = pts - pca_mean
    cov       = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    pca_axis  = eigvecs[:, np.argmax(eigvals)]

    center_cylinder_radius = np.linalg.norm(centered, axis=1).min() * 0.5

    # ------------------------------------------------------------------
    # 3. Rotation matrix: pca_axis → [0, 0, 1]
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Build unified DataFrame
    # ------------------------------------------------------------------
    shifted = (R @ centered.T).T

    r     = np.sqrt(shifted[:, 0]**2 + shifted[:, 1]**2)
    theta = np.arctan2(shifted[:, 1], shifted[:, 0])
    z_cyl = shifted[:, 2]

    if remove_supports:
        z_min, z_max = z_cyl.min(), z_cyl.max()
        z_cutoff     = 3.0
        cutoff_mask  = (z_cyl >= z_min + z_cutoff) & (z_cyl <= z_max - z_cutoff)
        n_removed    = int((~cutoff_mask).sum())
        print(f"[3.4] End cutoff: {n_removed:,} points dropped "
              f"({100 * n_removed / len(r):.1f}%) for z_cyl ∉ "
              f"[{z_min + z_cutoff:.4f}, {z_max - z_cutoff:.4f}] mm")
        if n_removed > 0:
            pts     = pts[cutoff_mask]
            shifted = shifted[cutoff_mask]
            r       = r[cutoff_mask]
            theta   = theta[cutoff_mask]
            z_cyl   = z_cyl[cutoff_mask]

    df = pd.DataFrame({
        'point_id'     : np.arange(len(pts)),
        'r'            : r,
        'theta'        : theta,
        'z_cylindrical': z_cyl,
        'x'            : shifted[:, 0],
        'y'            : shifted[:, 1],
        'z'            : shifted[:, 2],
    })

    # ------------------------------------------------------------------
    # 5a. Bounding box in shifted frame (used for plots only)
    # ------------------------------------------------------------------
    z_cyl_min, z_cyl_max = z_cyl.min(), z_cyl.max()
    stent_length   = z_cyl_max - z_cyl_min
    stent_diameter = 2.0 * r.max()

    # ------------------------------------------------------------------
    # 5b. Strut thickness — two methods, both stored
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Optional thickness plots
    # ------------------------------------------------------------------
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
        axes[1].set_ylabel('Thickness (r_outer − r_inner)')
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
        ax.set_title('Global r distribution — two peaks = inner and outer strut wall')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {
        'stent_df'                   : df,
        'stent_features'             : stent_features,
        'stent_centerline_direction' : pca_axis,
    }

def segment_stent(
    stent_df: pd.DataFrame,
    n_long_slices: int,
    segmentation_wanted: bool,
    strut_thickness: float,
    small_logic: str = 'or',          # 'or' = small by count OR by span; 'and' = both
    show_plots: bool = False,
) -> dict:
    """
    Longitudinal-slice connected components + valley-threshold merge of small fragments.

    A fragment is judged "small" using TWO measures, each with its own
    valley threshold:
        • point count           — number of surface points in the region
        • bbox diagonal (mm)     — 3-D bounding-box diagonal of the region
                                   (orientation-agnostic physical span; chosen
                                   over a z-only span, which is biased toward
                                   circumferential struts and capped by slicing)
    `small_logic` decides how to combine them ('or' catches more, 'and' is
    conservative).

    Returns:
        stent_df       — with merged, compactly renumbered 'region' column
        region_allowed — (R+1 x R+1) bool adjacency matrix for merged regions
        whole_stent_region — trivial (1x1) adjacency for the un-segmented case
        n_regions      — number of regions after merge
        conn_radius_3d — the derived connection radius
    """
    if not segmentation_wanted:
        stent_df           = stent_df.copy()
        stent_df['region'] = 1
        region_allowed     = np.array([[True]])
        whole_stent_region = np.array([[True]])
        print("[6.1.5] Segmentation skipped — all points in one region")
        return {
            'stent_df'          : stent_df,
            'region_allowed'    : region_allowed,
            'whole_stent_region': whole_stent_region,
            'n_regions'         : 1,
            'conn_radius_3d'    : 0.0,
        }

    whole_stent_region = np.array([[True]])  # all points belong to the same whole-stent region

    pts3d     = stent_df[['x', 'y', 'z']].values
    z_vals    = stent_df['z_cylindrical'].values
    n_samples = len(pts3d)

    # conn_radius_3d: min of density-based and strut-based estimates
    r_mean         = stent_df['r'].values.mean()
    z_range        = z_vals.max() - z_vals.min()
    surface_area   = 2 * np.pi * r_mean * z_range
    avg_spacing    = np.sqrt(surface_area / n_samples)
    conn_radius_3d = min(3.0 * avg_spacing, strut_thickness)
    print(f"[6.1.5] conn_radius_3d={conn_radius_3d:.4f}  (density-based={3*avg_spacing:.4f}, strut-based={strut_thickness:.4f})")

    # ── Initial segmentation: slice + connected components ────────────────────
    z_edges = np.linspace(z_vals.min(), z_vals.max(), n_long_slices + 1)
    slc     = np.clip(np.digitize(z_vals, z_edges) - 1, 0, n_long_slices - 1)

    tree  = cKDTree(pts3d)
    all_p = tree.query_pairs(r=conn_radius_3d, output_type='ndarray')

    same  = slc[all_p[:, 0]] == slc[all_p[:, 1]]
    pairs = all_p[same]
    cross = all_p[~same]          # kept for region_allowed — no second query_pairs needed

    adj = csr_matrix(
        (np.ones(2 * len(pairs), np.uint8),
         (np.concatenate([pairs[:, 0], pairs[:, 1]]),
          np.concatenate([pairs[:, 1], pairs[:, 0]]))),
        shape=(n_samples, n_samples))
    _, region = connected_components(adj, directed=False)

    # ── Valley-threshold helper ───────────────────────────────────────────────
    def _valley_thresholds(sizes, near_zero_frac=0.4, n_bins=50):
        s           = sizes[sizes > 0]
        hist, edges = np.histogram(s, bins=n_bins)
        smooth      = uniform_filter1d(hist.astype(float), size=3)
        centers     = 0.5 * (edges[:-1] + edges[1:])
        peak_idx    = int(np.argmax(smooth))
        near_zero   = near_zero_frac * smooth[peak_idx]
        mode        = float(centers[peak_idx])
        small_thresh = None
        for i in range(peak_idx - 1, -1, -1):
            if smooth[i] <= near_zero:
                small_thresh = float(centers[i]); break
        large_thresh = None
        for i in range(peak_idx + 1, len(smooth)):
            if smooth[i] <= near_zero:
                large_thresh = float(centers[i]); break
        return mode, small_thresh, large_thresh, smooth, centers, edges

    # ── Per-region size measures: point count + 3-D bounding-box diagonal ─────
    ids, counts = np.unique(region, return_counts=True)

    geom = pd.DataFrame({'region': region,
                         'x': pts3d[:, 0], 'y': pts3d[:, 1], 'z': pts3d[:, 2]})
    agg  = geom.groupby('region').agg(['min', 'max'])
    span = np.sqrt(sum((agg[(c, 'max')] - agg[(c, 'min')]) ** 2 for c in ('x', 'y', 'z')))
    diag = span.loc[ids].to_numpy()              # bbox diagonal aligned to `ids`

    mode_c, small_c, large_c, sm_c, ct_c, ed_c = _valley_thresholds(counts)
    mode_d, small_d, large_d, sm_d, ct_d, ed_d = _valley_thresholds(diag)

    if show_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.4))
        for ax, centers, smooth, edges, mode, small_t, xlabel, fmt in (
            (ax1, ct_c, sm_c, ed_c, mode_c, small_c, 'region point-count',        '.0f'),
            (ax2, ct_d, sm_d, ed_d, mode_d, small_d, 'region bbox diagonal (mm)', '.3g'),
        ):
            ax.bar(centers, smooth, width=np.diff(edges) * 0.85, color='steelblue', align='center')
            ax.axvline(mode, color='orange', ls='--', lw=2, label=f'mode={mode:{fmt}}')
            if small_t is not None:
                ax.axvline(small_t, color='red', ls='--', lw=2, label=f'small<{small_t:{fmt}}')
            ax.set_xlabel(xlabel); ax.set_ylabel('# regions'); ax.legend()
        fig.suptitle('region-size distributions — valley thresholds')
        plt.tight_layout(); plt.show()

    # ── Combine the two "small" criteria ──────────────────────────────────────
    small_by_count = counts < small_c if small_c is not None else np.zeros(len(ids), bool)
    small_by_diag  = diag   < small_d if small_d is not None else np.zeros(len(ids), bool)
    small_mask     = (small_by_count & small_by_diag) if small_logic == 'and' \
                     else (small_by_count | small_by_diag)

    n_raw   = len(ids)
    ct_txt  = f"{small_c:.0f}" if small_c is not None else "—"
    dg_txt  = f"{small_d:.3g}" if small_d is not None else "—"
    small_ids  = ids[small_mask]
    normal_ids = ids[~small_mask]

    if len(small_ids) and len(normal_ids):
        print(f"[6.1.5] {n_raw} raw regions | small if count<{ct_txt} {small_logic.upper()} "
              f"diag<{dg_txt} mm | merging {len(small_ids)} "
              f"(count-only={small_by_count.sum()}, diag-only={small_by_diag.sum()})")

        s_idx = np.where(np.isin(region, small_ids))[0]
        n_idx = np.where(~np.isin(region, small_ids))[0]

        # Group touching small fragments into connected groups
        k        = len(small_ids)
        sid_to_i = {sid: i for i, sid in enumerate(small_ids)}
        s_node   = np.array([sid_to_i[region[i]] for i in s_idx])

        sp = cKDTree(pts3d[s_idx]).query_pairs(r=conn_radius_3d, output_type='ndarray')
        if len(sp):
            a, b = s_node[sp[:, 0]], s_node[sp[:, 1]]
            m    = a != b
            g    = csr_matrix((np.ones(m.sum()), (a[m], b[m])), shape=(k, k))
            g    = g + g.T
        else:
            g = csr_matrix((k, k))
        _, grp_label = connected_components(g, directed=False)
        s_grp = grp_label[s_node]

        # Absorb each group into the nearest normal/large region
        n_tree       = cKDTree(pts3d[n_idx])
        n_reg        = region[n_idx]
        s_dist, s_nn = n_tree.query(pts3d[s_idx])
        s_nreg       = n_reg[s_nn]

        pick   = (pd.DataFrame({'grp': s_grp, 'dist': s_dist, 'nreg': s_nreg})
                    .groupby('grp')['dist'].idxmin())
        n_grps        = int(grp_label.max()) + 1
        target        = np.empty(n_grps, dtype=region.dtype)
        target[s_grp[pick.to_numpy()]] = s_nreg[pick.to_numpy()]

        region        = region.copy()
        region[s_idx] = target[s_grp]
        print(f"[6.1.5] → {len(np.unique(region))} regions after merge")
    else:
        print(f"[6.1.5] Nothing to merge (count<{ct_txt} {small_logic.upper()} diag<{dg_txt})")

    # ── Compact renumbering ───────────────────────────────────────────────────
    _, inv   = np.unique(region, return_inverse=True)
    region_f = (inv + 1).astype(np.int32)   # 1-based, dense
    R        = int(region_f.max())

    stent_df           = stent_df.copy()
    stent_df['region'] = region_f

    # ── Efficient region_allowed from cross pairs ─────────────────────────────
    # Two different regions are adjacent only across slice boundaries (within a
    # slice, connected components already merged all nearby points into one region).
    # We reuse the cross pairs from the initial query_pairs — no second KDTree
    # query needed.  Remapping through region_f gives adjacency for the merged,
    # renumbered regions automatically.
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
) -> dict:
    """
    Rasterise → dilate → skeletonise → back to physical (arc, z) coords.

    Unrolling the stent into the (arc, z) plane introduces an artificial seam at
    the arc boundary. To skeletonise continuously across it, the surface points
    are replicated ±circumference and a `pad_fraction` margin of that band is
    kept. The canvas is sized to the *padded* band (so the seam copies are NOT
    clipped onto the boundary columns), then the skeleton is cropped back to the
    real arc window [arc_min_flat, arc_max_flat].
    """
    pixel_size = stent_geometry['strut_thickness'] / pixels_per_strut

    # ── Seam padding: replicate along arc and keep a margin band ──────────────
    arc_three = np.concatenate([arc_flat - circumference, arc_flat, arc_flat + circumference])
    z_three   = np.concatenate([z_flat,                   z_flat,   z_flat])
    band      = ((arc_three >= arc_flat.min() - pad_fraction * circumference) &
                 (arc_three <= arc_flat.max() + pad_fraction * circumference))
    arc_pad, z_pad = arc_three[band], z_three[band]   # padded surface points (incl. seam copies)

    # ── Canvas spans the *padded* band so seam copies are not clipped ─────────
    arc_lo, arc_hi = arc_pad.min(), arc_pad.max()
    z_lo,   z_hi   = z_pad.min(),   z_pad.max()
    n_cols = int(np.ceil((arc_hi - arc_lo) / pixel_size)) + 1
    n_rows = int(np.ceil((z_hi   - z_lo)   / pixel_size)) + 1

    col_idx = np.clip(((arc_pad - arc_lo) / pixel_size).astype(int), 0, n_cols - 1)
    row_idx = np.clip(((z_pad   - z_lo)   / pixel_size).astype(int), 0, n_rows - 1)

    img = np.zeros((n_rows, n_cols), dtype=bool)
    img[row_idx, col_idx] = True

    img_solid = dilation(img,       footprint=disk(dilate_px))
    img_solid = closing (img_solid, footprint=disk(1))

    img_skel = skeletonize(img_solid)

    sk_rows, sk_cols = np.where(img_skel)
    skel_arc_pad = arc_lo + (sk_cols + 0.5) * pixel_size
    skel_z_pad   = z_lo   + (sk_rows + 0.5) * pixel_size

    # ── Crop the padded skeleton back to the real arc window ──────────────────
    seam_mask = (skel_arc_pad >= arc_min_flat) & (skel_arc_pad <= arc_max_flat)
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
    """Boolean mask of nodes that survive iterative removal of degree<2 nodes
    (the 2-core) — i.e. the loop structure of the sub-graph."""
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
    surf_tree: cKDTree = None,
    surf_reg: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """
    Quality report for a whole-stent skeleton, using the segmented regions only
    as a reference (the skeleton itself is computed without segmentation).

    Reports three issues — it does NOT modify the skeleton:
      1. bad_connections : skeleton edges that join two regions which do not
                           actually touch (region_allowed is False).
      2. region_loops    : regions whose skeleton sub-graph contains a cycle.
      3. empty_regions   : regions that received no skeleton point at all.

    Also returns the (arc, z) locations of the offending features so they can
    be drawn on top of the skeleton.

    `surf_tree` / `surf_reg` let a caller pass a prebuilt (arc, z) KD-tree and
    its region labels so a tuner doesn't rebuild it every iteration; when not
    given they are built from `stent_df`.
    """
    skel = df_skeleton_2d[['arc', 'z']].to_numpy()
    n_sk = len(skel)
    n_regions = int(stent_df['region'].max())
    all_regions = np.arange(1, n_regions + 1)

    # ── Assign each skeleton point to the nearest surface point's region ──────
    if surf_tree is None or surf_reg is None:
        surf_arc  = r_mid * stent_df['theta'].to_numpy()
        surf_z    = stent_df['z'].to_numpy()
        surf_reg  = stent_df['region'].to_numpy()
        surf_tree = cKDTree(np.column_stack([surf_arc, surf_z]))
    _, nn       = surf_tree.query(skel)
    skel_region = surf_reg[nn]

    # ── Recover integer pixel-grid coords (skeleton sits on a regular grid) ───
    gi = np.round((skel[:, 0] - skel[:, 0].min()) / pixel_size).astype(int)  # col (arc)
    gj = np.round((skel[:, 1] - skel[:, 1].min()) / pixel_size).astype(int)  # row (z)
    coord_set = set(zip(gi.tolist(), gj.tolist()))
    coord_idx = {(int(gi[k]), int(gj[k])): k for k in range(n_sk)}

    # ── Build the skeleton graph (8-neighbour, diagonal de-duplicated) ────────
    # Each undirected edge is emitted once. A diagonal is dropped when a
    # 4-connected corner exists, so staircase pixels don't create spurious
    # triangle-loops.
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

    # ── 1) Connections between non-touching regions ──────────────────────────
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

    # ── 2) Loops within a region (cycles in the per-region sub-graph) ─────────
    region_loops   = {}
    loop_points    = []
    for reg in np.unique(skel_region):
        node_mask = skel_region == reg
        node_idx  = np.where(node_mask)[0]
        V = len(node_idx)
        if len(edges):
            e_mask = node_mask[edges[:, 0]] & node_mask[edges[:, 1]]
            sub    = edges[e_mask]
        else:
            sub = edges
        E = len(sub)
        if E:
            remap = {int(g): i for i, g in enumerate(node_idx)}
            a = np.fromiter((remap[int(x)] for x in sub[:, 0]), dtype=int, count=E)
            b = np.fromiter((remap[int(x)] for x in sub[:, 1]), dtype=int, count=E)
            g = csr_matrix((np.ones(E), (a, b)), shape=(V, V))
            g = g + g.T
            ncomp, _ = connected_components(g, directed=False)
        else:
            ncomp = V
        n_cycles = E - V + ncomp          # independent loops (Betti-1) of the sub-graph
        if n_cycles > 0:
            region_loops[int(reg)] = int(n_cycles)
            core = _two_core_mask(V, a, b)        # the actual loop structure
            loop_points.append(skel[node_idx[core]])
    loop_points_xy = np.vstack(loop_points) if loop_points else np.empty((0, 2))

    # ── 3) Empty regions (no skeleton point assigned) ────────────────────────
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
        'empty_regions'  : empty_regions,
        'skel_region'    : skel_region,
        'bad_edge_xy'    : bad_edge_xy,      # (K, 2) (arc, z) midpoints of bad edges
        'loop_points_xy' : loop_points_xy,   # (K, 2) (arc, z) loop-structure points
        'empty_xy'       : empty_xy,         # (K, 2) (arc, z) centroids of empty regions
    }

    if verbose:
        print(f"Skeleton quality report — {n_regions} regions, {n_sk:,} skeleton points")
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
    """
    Delete skeleton points that bridge two regions which do not actually touch
    (`region_allowed` is False), so the bad connection disappears from the
    (arc, z) skeleton. Region labels come from the nearest surface point and are
    fixed, so removing every point incident to a bad edge clears ALL bad edges in
    a single pass (and leaves a >=2 px gap, so the 3D rebuild won't re-bridge).
    Loops and empty regions are NOT touched.
    """
    skel = df_skeleton_2d[['arc', 'z']].to_numpy()
    n_sk = len(skel)

    # nearest-surface region for each skeleton point
    if surf_tree is None or surf_reg is None:
        surf_arc  = r_mid * stent_df['theta'].to_numpy()
        surf_z    = stent_df['z'].to_numpy()
        surf_reg  = stent_df['region'].to_numpy()
        surf_tree = cKDTree(np.column_stack([surf_arc, surf_z]))
    _, nn       = surf_tree.query(skel)
    skel_region = surf_reg[nn]

    # 8-neighbour grid graph (same construction as check_skeleton_quality)
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

    # flag points on edges between non-touching regions and drop them
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
    s_pps_explore: float = 5.0,   # when feasible, step pps up by this to chase a higher score (resolution)
    s_dil_loop: float = 20.0,     # dilate_px push from loops (thicken to fill holes)
    s_dil_conn: float = 8.0,      # dilate_px pull-down from connections (thin)
    w_conn: float = 1.0,
    w_loop: float = 1.0,
    w_empty: float = 1.0,
    w_pps_dil_ratio_reward: float = 4,   # score reward: prefer higher pps (<1 so it never beats a real error)
    reward_no_improve_max: int = 3,   # steps (after the first clean step) with no reward gain
    reward_improve_tol: float = 1e-2,     # min score gain to reset the no-improve counter (avoid noise)
    target_penalty: float = 0.0,  # penalty <= this counts as "feasible"
    max_repeats: int = 4,        # stop once the SAME (pps, dil) state has been visited this many times
    pad_fraction: float = 0.20,
    time_limit: float = 100.0,
    predictive_stop: bool = False,
    max_iters: int = int(1e3),
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Error-driven continuous tuner for `pixels_per_strut` and `dilate_px`.

    Each step skeletonises, scores with `check_skeleton_quality`, and nudges the
    two parameters in proportion to the (region-normalised) errors — a large
    error moves a parameter more than a small one:

        • bad connections (non-touching regions joined)
              -> pixels_per_strut up   (sharpen so regions separate)
              -> dilate_px        down (thin so struts stop bridging)
        • empty regions (no skeleton point)
              -> pixels_per_strut up    (size-weighted by the empty region)
        • loops inside a region (holes in an under-filled strut)
              -> dilate_px        up    (thicken to close the holes)
              -> if dilate_px is already at its cap, pixels_per_strut down
                 instead (coarsen so the existing fill closes the holes)

    `pixels_per_strut` is continuous; `dilate_px` is an integer disk radius so it
    moves in (proportional) integer steps. `dilate_px` is additionally bounded by
    the current `pixels_per_strut` (dil <= pps): a disk radius larger than the
    strut spacing always bridges neighbouring struts, so the effective ceiling is
    min(dil_max, floor(pps)).

    Penalty = w_conn*conn + w_loop*loop + w_empty*empty. The step kept as `best`
    maximises a *score* = reward - penalty, where the reward is a small bonus for
    higher `pixels_per_strut` (so bigger score = better):

        reward = w_pps_dil_ratio_reward * (pps/dil - pps_min/dil_max) / (pps_max/dil_min - pps_min/dil_max)   in [0, w_pps_dil_ratio_reward]

    Reaching penalty <= `target_penalty` does NOT stop the search — once feasible
    it keeps pushing resolution up (pps += s_pps_explore while dil is held fixed,
    so the ratio pps/dil and hence the reward genuinely rises). Holding dil thins
    the physical dilation; if loops/empties reappear the infeasible branch thickens
    dil back, and the post-feasibility no-improve counter bounds the search.

    STOPPING is purely cycle-based: the update is deterministic, so if a (pps, dil)
    pair ever recurs the whole trajectory from it repeats. The tuner counts how
    often each (pps, dil) state is visited and stops once any state has been seen
    `max_repeats` times (a true limit cycle, or pinned at a bound). Once the
    skeleton is clean for the first time, a no-improve counter starts and runs on
    EVERY subsequent step (it is not reset by later infeasible excursions); the
    tuner stops after `reward_no_improve_max` steps with no gain in the reward
    bucket (converged). It also stops after `time_limit` s / `max_iters`. Returns the
    highest-score step; ties within `reward_improve_tol` (the same-reward band
    used for the no-improve count) are broken by the highest pps/dil ratio.
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
        n_loop = int(sum(qr['region_loops'].values()))
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
    best_bucket = None            # best reward bucket since feasibility was first reached
    feasible_reached = False      # latches True at the first penalty<=target step
    no_improve = 0                # steps (post-feasibility) since the reward bucket improved
    t0 = time.time()
    durs = []                     # per-iteration wall times -> predict the next step's cost

    if verbose:
        print(f"{'step':>4} {'pps':>7} {'dil_px':>6} | {'conn':>5} {'loop':>5} {'empty':>6} "
              f"| {'penalty':>8} {'score':>8} | {'t(s)':>5}")
        print("-" * 75)

    for it in range(max_iters):
        elapsed = time.time() - t0
        # Hard stop: always active.
        if elapsed > time_limit:
            print(f"[tune] time limit ({time_limit:.0f}s) reached — stopping")
            break
        # Predictive stop: project the next step's cost and stop before launching
        # one that won't finish in time. Disable with predictive_stop=False.
        if predictive_stop and durs:
            growth = durs[-1] / durs[-2] if len(durs) >= 2 and durs[-2] > 0 else 1.0
            proj   = durs[-1] * max(1.0, growth)
            if elapsed + proj > time_limit:
                print(f"[tune] next step projected ~{proj:.0f}s — would exceed the "
                      f"{time_limit:.0f}s budget at t={elapsed:.0f}s — stopping")
                break

        _it_t0 = time.time()
        res = compute_skeleton_2d(
            arc_flat=arc, z_flat=z, arc_min_flat=arc.min(), arc_max_flat=arc.max(),
            circumference=circ, stent_df=stent_df, stent_geometry=stent_features,
            pixels_per_strut=pps, dilate_px=dil, pad_fraction=pad_fraction, plot=False,
        )
        qr = check_skeleton_quality(
            res['df_skeleton_2d'], res['pixel_size'], stent_df, r_mid, region_allowed,
            surf_tree=surf_tree, surf_reg=surf_reg, verbose=False,
        )
        durs.append(time.time() - _it_t0)
        n_conn, n_loop, e_empty = _errors(qr)
        P = w_conn * n_conn + w_loop * n_loop + w_empty * e_empty
        # Reward higher resolution, if pps/dil ratio is more, it shows higher resolution skeleton
        # A clean skeleton (P=0) scores +reward (small, in [0, w_pps_dil_ratio_reward]);
        # any error makes P large and drags the score negative.
        reward = w_pps_dil_ratio_reward * (pps/dil - pps_min/dil_max) / (pps_max/dil_min - pps_min/dil_max)
        score  = reward - P
        elapsed = time.time() - t0
        history.append(dict(step=it, pps=round(pps, 2), dil_px=dil, conn=n_conn,
                            loop=n_loop, empty=round(e_empty, 2),
                            penalty=round(P, 3), score=round(score, 3), t=round(elapsed, 1)))
        if verbose:
            print(f"{it:>4} {pps:>7.2f} {dil:>6d} | {n_conn:>5d} {n_loop:>5d} "
                  f"{e_empty:>6.2f} | {P:>8.3f} {score:>8.3f} | {elapsed:>5.1f}")

        # Selection key: prefer a higher score, but treat scores within
        # `reward_improve_tol` of each other as the *same reward* (one tolerance
        # bucket) and break that tie by the highest pps/dil ratio (finest
        # resolution). So among the equal-reward steps the highest-ratio one is
        # kept — not the first seen.
        ratio        = pps / dil
        score_bucket = round(score / reward_improve_tol)
        cand_key     = (score_bucket, ratio)
        if best is None or cand_key > best['key']:
            best = dict(key=cand_key, score=score, ratio=ratio, penalty=P, pps=pps,
                        dil=dil, conn=n_conn, loop=n_loop, empty=e_empty,
                        result=res, quality_report=qr)

        # Convergence stop: counting begins the first time the skeleton is clean
        # (penalty <= target) and then runs EVERY step — it is NOT reset by later
        # infeasible excursions (e.g. an explore step that overshoots into loops).
        # A step whose reward bucket beats the best-so-far resets the counter;
        # `reward_no_improve_max` steps with no reward gain -> stop.
        if P <= target_penalty:
            feasible_reached = True
        if feasible_reached:
            if best_bucket is None or score_bucket > best_bucket:
                best_bucket = score_bucket
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= reward_no_improve_max:
                    print(f"[tune] reward not improved for {no_improve} steps since "
                          f"reaching a clean skeleton — stopping (converged)")
                    break

        # Cycle / pinned detector: the update is deterministic, so a repeated
        # (pps, dil) state means we are looping. Stop once one is seen max_repeats×.
        key = (round(pps, 2), dil)
        seen[key] += 1
        if seen[key] >= max_repeats:
            print(f"[tune] same parameters (pps={pps:.2f}, dil_px={dil}) seen "
                  f"{max_repeats}× — stopping (cycle / pinned)")
            break

        # ── Parameter update ──────────────────────────────────────────────────
        if P <= target_penalty:
            # Feasible: push resolution up by raising pps while HOLDING dil fixed,
            # so the ratio pps/dil (the reward) actually increases. This thins the
            # physical dilation (dil*pixel_size shrinks as pps grows); if that
            # reintroduces loops/empties the next step's infeasible branch will
            # thicken dil back, and the post-feasibility no-improve counter bounds
            # the search either way.
            pps_new = pps + s_pps_explore
            dil_new = dil
        else:
            # Infeasible: error-proportional correction (errors normalised to [0,1]).
            e_conn_n  = min(1.0, n_conn  / max(1, n_regions))
            e_loop_n  = min(1.0, n_loop  / max(1, n_regions))
            e_empty_n = min(1.0, e_empty / max(1, n_regions))
            dil_cap   = min(dil_max, int(pps))            # hard cap AND strut spacing
            pps_new = pps + s_pps_conn * e_conn_n + s_pps_empty * e_empty_n      # sharpen
            # Thicken to close loops, thin to break bad connections.
            dil_new = dil + int(round(s_dil_loop * e_loop_n)) \
                          - int(round(s_dil_conn * e_conn_n))
            if n_loop > 0 and dil >= dil_cap:     # dilation capped but holes remain -> coarsen pps
                pps_new -= s_pps_loop * e_loop_n

        pps = float(np.clip(pps_new, pps_min, pps_max))
        # Re-clamp dil against the (possibly changed) strut spacing: dil <= pps.
        dil = int(np.clip(dil_new, dil_min, min(dil_max, int(pps))))

    hist_df = pd.DataFrame(history)
    print("\n[tune] BEST  pps={:.2f}  dilate_px={}  penalty={:.3f}  score={:.3f}  "
          "(conn={}, loop={}, empty={:.2f})".format(
              best['pps'], best['dil'], best['penalty'], best['score'],
              int(best['conn']), int(best['loop']), best['empty']))

    if plot:
        # penalty trajectory
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(hist_df['step'], hist_df['penalty'], '-', color='steelblue', label='penalty')
        ax.plot(hist_df['step'], hist_df['score'],   '-', color='crimson',   label='score')
        ax.axhline(0, color='grey', lw=0.8, ls=':')   # score > 0 = clean skeleton
        ax.set_xlabel('step'); ax.set_ylabel('penalty / score'); ax.set_title('tuning trajectory')
        ax2 = ax.twinx()
        ax2.plot(hist_df['step'], hist_df['pps'],    '--', color='darkorange', label='pps')
        ax2.plot(hist_df['step'], hist_df['dil_px'], '--', color='seagreen',   label='dilate_px')
        ax2.set_ylabel('pps / dilate_px')
        ax.legend(loc='upper right'); ax2.legend(loc='center right')
        plt.tight_layout(); plt.show()

    return {
        'best_pps'       : best['pps'],
        'best_dilate_px' : best['dil'],
        'best_penalty'   : best['penalty'],
        'history'        : hist_df,
        'skeleton_2d'    : best['result'],
        'quality_report' : best['quality_report'],
    }


def plot_skeleton_quality(
    skeleton_2d_result: dict,
    stent_df: pd.DataFrame,
    r_mid: float,
    quality_report: dict,
    figsize=(16, 7),
) -> None:
    """Overlay the quality issues on the unrolled (z, arc) skeleton, including
    the seam padding band: skeleton coloured by region inside the kept window,
    the dropped padding skeleton in grey, and bad connections / loops / empty
    regions highlighted. Red dashes mark the kept arc window."""
    skel        = skeleton_2d_result['df_skeleton_2d'][['arc', 'z']].to_numpy()
    skel_pad    = skeleton_2d_result['df_skeleton_2d_padded'][['arc', 'z']].to_numpy()
    skel_region = quality_report['skel_region']
    arc_lo, arc_hi   = skeleton_2d_result['arc_band']
    arc_min, arc_max = skeleton_2d_result['arc_window']
    circ = 2 * np.pi * r_mid

    # ── Replicate surface points across the seam to fill the padded band ──────
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
        ax.scatter(loop_xy[:, 1], loop_xy[:, 0], s=40, marker='o',
                   facecolors='none', edgecolors='magenta', linewidths=1.4, zorder=4,
                   label=f'loop ({len(quality_report["region_loops"])} region(s))')
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
    """
    Replace the constant r_mid wrap with a per-point local r, computed as
    the mean r of surface points within `search_radius` (in arc/z plane)
    around each skeleton point.
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
    pixel_size: float,
    neighbor_radius_factor: float = 1.8,
) -> pd.DataFrame:
    """
    Classify each skeleton point by its connectivity degree.
    node_type: 'isolated' (0), 'endpoint' (1), 'line' (2), 'junction' (3+).
    """
    pts    = df_skeleton_3d[['x', 'y', 'z']].values
    radius = neighbor_radius_factor * pixel_size

    tree  = cKDTree(pts)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')

    neighbors = [[] for _ in range(len(pts))]
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
    """
    Bridge spurious breaks in the skeleton graph.

    Rasterise → thin can split a continuous strut into two pieces, each ending in
    a degree-1 node. Left untouched these artificial endpoints act as free strut
    ends in the contact simulation and concentrate large spurious bending /
    deformation. This step walks every endpoint, looks *forward* along the
    strut's own tangent, and reconnects it to the nearest skeleton point lying
    inside a forward cone within `max_gap`.

    Genuine tips (e.g. open crowns at a free stent end) have no continuation in
    their forward direction, so no candidate is found and they stay endpoints —
    the geometry decides, no explicit/per-design tip detection is needed.

    Parameters
    ----------
    max_gap_factor : multiplied by `pixel_size` to set the largest bridgeable gap.
    min_cos        : forward-cone half-angle as a cosine (0.5 => 60°). Stops the
                     search from jumping sideways onto a parallel strut.
    exclude_hops   : candidates within this many graph hops of the endpoint are
                     skipped, so an endpoint never reconnects to its own stub.

    Returns a copy of `df_connectivity` with `neighbor_ids`, `degree` and
    `node_type` recomputed after the bridges are inserted.
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
                resolved.add(best)                 # bridged endpoint↔endpoint pair

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
        print(f"[reconnect] endpoints {len(endpoints)} → {int((degrees == 1).sum())}  "
              f"({len(new_edges)} bridges added, max_gap = {max_gap:.4f} mm, "
              f"cone = {np.degrees(np.arccos(min_cos)):.0f}°)")

    return df


def prune_skeleton_spurs(
    df_connectivity: pd.DataFrame,
    pixel_size: float,
    tip_frac: float = 0.05,
    max_spur_len: float = None,
    max_iter: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove spurs: short dead-end branches that are not genuine stent tips.

    After reconnect_skeleton_endpoints, any remaining degree-1 node is either a
    real tip (open crown at an axial end of the stent) or a spur — a small twig
    left by the rasterise→thin step. Spurs act as free strut ends in the contact
    simulation and concentrate spurious bending, so they are pruned here.

    A degree-1 node counts as a *tip* (and is kept) when it lies within
    `tip_frac` of the axial (z) span from either end. Every other dead-end is
    walked back along its degree-2 chain to the first junction and the whole
    twig is deleted. The walk repeats (`max_iter`) so chained spurs collapse.

    Parameters
    ----------
    tip_frac     : axial tip band as a fraction of (z_max - z_min). Endpoints
                   inside the band at either end are treated as real tips.
    max_spur_len : optional safety cap (mm). A non-tip dead-end longer than this
                   is left in place (it is likely a real strut whose far end
                   failed to bridge — raise the reconnect gap instead of pruning).
                   None = prune every non-tip dead-end regardless of length.

    Returns a re-indexed copy of `df_connectivity` (contiguous
    skeleton_point_id, remapped neighbor_ids, recomputed degree / node_type).
    """
    df  = df_connectivity.reset_index(drop=True).copy()
    pts = df[['x', 'y', 'z']].values
    N   = len(pts)
    z   = pts[:, 2]
    z_min, z_max = z.min(), z.max()
    tip_band = tip_frac * (z_max - z_min)

    def _is_tip(i):
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

    stent_dir = pathlib.Path("../../notebook_outputs") / stent_name
    print(f"Loading stent data from: {stent_dir.resolve()}")

    with open(stent_dir / "stent_features.json") as f:
        raw = json.load(f)

    # Unwrap any {"value": ..., "unit": ...} entries — handles JSON files that were
    # previously written with the wrapped format (strips all nesting levels).
    def _unwrap(v):
        while isinstance(v, dict) and "value" in v:
            v = v["value"]
        return v
    raw = {k: _unwrap(v) for k, v in raw.items()}

    with open(stent_dir / "stent_centerline_direction.json") as f:
        cl_dir = np.array(json.load(f))

    sk = pd.read_csv(stent_dir / "skeleton_points.csv")
    sk["neighbor_ids"] = sk["neighbor_ids"].apply(ast.literal_eval)

    # ── Material params as flat raw values (for comparison + JSON storage) ─────
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

    # ── Write JSON only when material params are absent or have changed ────────
    needs_update = any(raw.get(k) != v for k, v in new_mat.items())
    if needs_update:
        geo_raw_flat = {k: v for k, v in raw.items() if k not in _MAT_KEYS}
        geo_wrapped  = {k: {"value": v, "unit": _GEO_UNITS.get(k, "-")} for k, v in geo_raw_flat.items()}
        mat_wrapped  = {k: {"value": v, "unit": _MAT_UNITS[k]} for k, v in new_mat.items()}
        with open(stent_dir / "stent_features.json", "w") as f:
            json.dump({**geo_wrapped, **mat_wrapped}, f, indent=4)
        print(f"  stent_features.json updated with material parameters.")
    else:
        print(f"  Material parameters unchanged — stent_features.json not rewritten.")

    # ── Wrap for in-memory use ────────────────────────────────────────────────
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


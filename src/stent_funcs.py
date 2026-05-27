# Step 0: Import libraries and load mesh
import numpy as np
import trimesh
from trimesh.path.entities import Line
import matplotlib.pyplot as plt
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, convolve, label as cc_label
from skimage.morphology import skeletonize, binary_dilation, binary_closing, disk

from scipy.spatial import cKDTree


def preprocess_stent(mesh: trimesh.Trimesh, 
                     n_samples: int,
                     samples_per_face: int,
                     n_thickness_slices: int, 
                     slice_cutoff: int, 
                     thickness_calc_plot: bool,
                     remove_supports: bool) -> dict:


    # ------------------------------------------------------------------
    # 1. Sample surface  (clamp: at least 1 pt/face, at most 10 M total)
    # ------------------------------------------------------------------
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples)

    # ------------------------------------------------------------------
    # 2. PCA — centroid + main axis
    # ------------------------------------------------------------------
    pca_mean    = pts.mean(axis=0) 
    centered    = pts - pca_mean
    cov         = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    pca_axis    = eigvecs[:, np.argmax(eigvals)]          # (3,) unit-ish vector
    
    # approximate inner-hollow radius (used later for bin sizing in step 10)
    center_cylinder_radius = np.linalg.norm(centered, axis=1).min() * 0.5

    # ------------------------------------------------------------------
    # 3. Rotation matrix: pca_axis → [0, 0, 1]  (align main axis to z)
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
    # 4. Build unified DataFrame  (one rotation, all coordinate systems)
    # ------------------------------------------------------------------
    shifted = (R @ centered.T).T                          # (N, 3)

    r     = np.sqrt(shifted[:, 0]**2 + shifted[:, 1]**2)
    theta = np.arctan2(shifted[:, 1], shifted[:, 0])
    z_cyl = shifted[:, 2]

    # Cut 3mm off each end to discard support structures and delete the corresponding points from all arrays
    if remove_supports:
        z_min, z_max = z_cyl.min(), z_cyl.max()
        z_cutoff = 3.0  # mm
        cutoff_mask = (z_cyl >= z_min + z_cutoff) & (z_cyl <= z_max - z_cutoff)
        n_removed = int((~cutoff_mask).sum())
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
        'x_cartesian'  : shifted[:, 0],
        'y_cartesian'  : shifted[:, 1],
        'z_cartesian'  : shifted[:, 2],
    })

    # verify alignment (informational only)
    axis_check = np.linalg.eig(np.cov((shifted - shifted.mean(axis=0)).T))[1]
    main_after = axis_check[:, np.argmax(np.linalg.eigvals(
                         np.cov((shifted - shifted.mean(axis=0)).T)))]
    '''print(f"Main axis after rotation (expect ≈ [0,0,1]): {main_after.real.round(3)}")'''

    # ------------------------------------------------------------------
    # 5a. Bounding box in shifted frame
    # ------------------------------------------------------------------
    z_cyl_min, z_cyl_max = z_cyl.min(), z_cyl.max()
    stent_length   = z_cyl_max - z_cyl_min
    stent_diameter = 2.0 * r.max()

    sh_min = shifted.min(axis=0)
    sh_max = shifted.max(axis=0)
    bbox = {
        'x_min': sh_min[0], 'x_max': sh_max[0],
        'y_min': sh_min[1], 'y_max': sh_max[1],
        'z_min': sh_min[2], 'z_max': sh_max[2],
        'center': (sh_min + sh_max) / 2,
    }

    # ------------------------------------------------------------------
    # 5b. Strut thickness — two methods, both stored
    # ------------------------------------------------------------------
    # method A: robust global percentiles (recommended for tapered stents)
    r_inner_pct = np.percentile(r, 2)
    r_outer_pct = np.percentile(r, 98)
    strut_thick_robust = r_outer_pct - r_inner_pct

    # method B: per-slice mean (from original notebook)
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

    # Store all features in a dictionary for downstream use
    stent_features = {
        # primary geometric properties
        'length'                : stent_length,
        'diameter'              : stent_diameter,
        'radius'                : stent_diameter / 2.0,
        # strut wall — robust method is default (matches STENT_GEOMETRY in notebook)
        'strut_thickness'       : strut_thick_final,
        'z_min'                 : z_cyl_min,
        'z_max'                 : z_cyl_max,
        'r_inner'               : float(df_thick['r_inner'].mean()),
        'r_outer'               : float(df_thick['r_outer'].mean()),
        'r_mid'                 : float(df_thick['r_inner'].mean() + df_thick['r_outer'].mean()) / 2.0,
        # used downstream for bin sizing in per-component PCA (step 10)
        'center_cylinder_radius': center_cylinder_radius,
        'num_points'            : len(df),
    }

    # ------------------------------------------------------------------
    # Optional plot  (pass plot=True to see it)
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

        # --- plot 2: global r distribution ---
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

    
    # ------------------------------------------------------------------
    # 5c. Trimesh scene objects (call .show() yourself after composing)
    # ------------------------------------------------------------------
    scene_points = trimesh.PointCloud(shifted, colors=[200, 200, 200, 30])

    scene_centerline = trimesh.path.Path3D(
        entities=[Line(points=[0, 1], color=[255, 0, 0, 255])],
        vertices=np.array([[0.0, 0.0, z_cyl_min],
                           [0.0, 0.0, z_cyl_max]])
    )

    xn, xx = bbox['x_min'], bbox['x_max']
    yn, yx = bbox['y_min'], bbox['y_max']
    zn, zx = bbox['z_min'], bbox['z_max']
    corners = np.array([[xn,yn,zn],[xx,yn,zn],[xx,yx,zn],[xn,yx,zn],
                        [xn,yn,zx],[xx,yn,zx],[xx,yx,zx],[xn,yx,zx]])
    bbox_edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
                  [0,4],[1,5],[2,6],[3,7]]
    scene_bbox = trimesh.path.Path3D(
        entities=[Line(points=e, color=[255, 165, 0, 255]) for e in bbox_edges],
        vertices=corners
    )

    return {
        'stent_df'        : df,
        'stent_features'  : stent_features,
        'stent_centerline': pca_axis,
    }


def segment_stent(
    stent_df: pd.DataFrame,
    n_long_slices: int,
    segmentation_wanted: bool
    ) -> dict:
    """
    Longitudinal-slice connected components on all surface points.
    If n_samples is provided, conn_radius_3d is automatically scaled
    to the actual point spacing so connectivity works regardless of
    how large the stent is.
    """
    if not segmentation_wanted:
        region_allowed = np.array([[True]])
        stent_df = stent_df.copy()
        stent_df['region'] = 1
        print(f"[6.1.5] Segmentation skipped — all points in one region")
        print(f"        3D-adj pairs       : 0")
        print(f"        Allowed density    : 100.000%")
        return {
            'stent_df'      : stent_df,
            'region_allowed': region_allowed,
            'n_regions'     : 1
        }

    pts3d  = stent_df[['x_cartesian', 'y_cartesian', 'z_cartesian']].values
    z_vals = stent_df['z_cylindrical'].values
    n_samples    = len(pts3d)

    # ── Auto-scale conn_radius_3d to actual point spacing ──────────────
    # Estimate surface area from the bounding cylinder: 2π r z_range.
    r_vals   = stent_df['r'].values
    r_approx = r_vals.mean()
    z_range  = z_vals.max() - z_vals.min()
    surface_area_est = 2 * np.pi * r_approx * z_range

    # Mean spacing between nearest neighbours on the surface
    avg_spacing    = np.sqrt(surface_area_est / n_samples)
    # Use 3× avg spacing: comfortably bridges any gap between same-surface
    # neighbours while still much smaller than the inter-strut gap
    # conn_radius_3d = max(conn_radius_3d, 3.0 * avg_spacing)

    # ── conn_radius_3d: pure geometric derivation ──────────────────────
    # Surface area of the stent shell ≈ outer cylinder: 2π × r × length
    # Average point spacing on that surface = √(surface_area / n_samples)
    # 3× spacing guarantees connectivity along the surface without ever
    # bridging across struts (gap between struts >> spacing).
    r_mean           = r_vals.mean()
    z_range          = z_vals.max() - z_vals.min()
    surface_area     = 2 * np.pi * r_mean * z_range
    avg_spacing      = np.sqrt(surface_area / n_samples)
    conn_radius_3d   = 3.0 * avg_spacing

    # ── Slicing and graph build (unchanged) ────────────────────────────
    z_edges = np.linspace(z_vals.min(), z_vals.max(), n_long_slices + 1)
    slc     = np.clip(np.digitize(z_vals, z_edges) - 1, 0, n_long_slices - 1)

    tree  = cKDTree(pts3d)
    all_p = tree.query_pairs(r=conn_radius_3d, output_type='ndarray')

    same  = slc[all_p[:, 0]] == slc[all_p[:, 1]]
    pairs = all_p[same]
    cross = all_p[~same]

    adj = csr_matrix(
        (np.ones(2 * len(pairs), np.uint8),
         (np.concatenate([pairs[:, 0], pairs[:, 1]]),
          np.concatenate([pairs[:, 1], pairs[:, 0]]))),
        shape=(n_samples, n_samples))
    _, region = connected_components(adj, directed=False)

    R        = int(region.max() + 1)
    stent_df = stent_df.copy()
    stent_df['region'] = region + 1

    if len(cross):
        adj_pairs = np.unique(np.sort(region[cross], axis=1) + 1, axis=0)
    else:
        adj_pairs = np.empty((0, 2), dtype=int)

    region_allowed = np.zeros((R + 1, R + 1), dtype=bool)
    np.fill_diagonal(region_allowed, True)
    if len(adj_pairs):
        region_allowed[adj_pairs[:, 0], adj_pairs[:, 1]] = True
        region_allowed[adj_pairs[:, 1], adj_pairs[:, 0]] = True
    region_allowed[0, :] = True
    region_allowed[:, 0] = True

    return {
        'stent_df'      : stent_df,
        'region_allowed': region_allowed,
        'n_regions'     : R,
        'conn_radius_3d': conn_radius_3d
    }

def open_stent_to_plane(stent_df: pd.DataFrame, r_mid: float, pad_fraction: float) -> dict:
    """
    Unwrap all surface points onto a 2D (arc, z) plane with periodic padding.

    Returns
    -------
    'arc_padded'    'z_padded'    with padding
    'arc_flat'      'z_flat'      original (no padding)
    'arc_min_flat'  'arc_max_flat'  seam boundaries
    'circumference' float
    """
    circumference = 2 * np.pi * r_mid
    arc_flat      = r_mid * stent_df['theta'].values          # ← stent_df → df_outer
    z_flat        = stent_df['z_cylindrical'].values
    arc_min_flat  = arc_flat.min()
    arc_max_flat  = arc_flat.max()
    pad_width     = pad_fraction * circumference

    arc_three = np.concatenate([arc_flat - circumference, arc_flat, arc_flat + circumference])
    z_three   = np.concatenate([z_flat,                   z_flat,   z_flat])
    pad_mask  = ((arc_three >= arc_min_flat - pad_width) &
                 (arc_three <= arc_max_flat + pad_width))
    arc_padded = arc_three[pad_mask]
    z_padded   = z_three[pad_mask]

    return {
        'arc_padded'   : arc_padded,
        'z_padded'     : z_padded,
        'arc_flat'     : arc_flat,
        'z_flat'       : z_flat,
        'arc_min_flat' : arc_min_flat,
        'arc_max_flat' : arc_max_flat,
        'circumference': circumference,
    }


def _prune_spur_branches(skel: np.ndarray, min_branch_length: int) -> np.ndarray:
    """Remove leaf branches shorter than min_branch_length pixels."""

    skel = skel.copy().astype(bool)
    K8   = np.ones((3, 3), int); K8[1, 1] = 0
    while True:
        nb        = convolve(skel.astype(int), K8, mode='constant', cval=0)
        junctions = skel & (nb >= 3)
        endpoints = skel & (nb == 1)
        cut, nl   = cc_label(skel & ~junctions, structure=np.ones((3, 3), int))
        removed   = False
        for lbl in range(1, nl + 1):
            br = (cut == lbl)
            if br.sum() < min_branch_length and (br & endpoints).any():
                skel   &= ~br
                removed = True
        if not removed:
            break
    return skel

def _cut_bad_bridges(skel: np.ndarray, label_img: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Remove skeleton pixels whose region label conflicts with an 8-neighbour."""
    from scipy.ndimage import convolve
    skel = skel.copy()
    while True:
        sk_lbl   = np.where(skel, label_img, 0).astype(np.int32)
        conflict = np.zeros_like(skel)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted        = np.zeros_like(sk_lbl)
                r0, r1         = max(0, dr),  skel.shape[0] + min(0, dr)
                c0, c1         = max(0, dc),  skel.shape[1] + min(0, dc)
                shifted[r0:r1, c0:c1] = sk_lbl[r0-dr:r1-dr, c0-dc:c1-dc]
                bad            = (sk_lbl > 0) & (shifted > 0) & ~allowed[sk_lbl, shifted]
                conflict      |= bad
        if not conflict.any():
            break
        skel &= ~conflict
    return skel


def compute_skeleton_2d(
    arc_padded: np.ndarray,
    z_padded: np.ndarray,
    arc_flat: np.ndarray,
    z_flat: np.ndarray,
    arc_min_flat: float,
    arc_max_flat: float,
    circumference: float,
    stent_df: pd.DataFrame,
    region_allowed: np.ndarray,
    stent_geometry: dict,
    pixels_per_strut: int,
    dilate_px: int,
    plot: bool) -> dict:
    """
    Rasterise → dilate → skeletonise → prune spurs → cut cross-region bridges
    → back to physical (arc, z) coords.

    Returns
    -------
    'skel_arc'       'skel_z'    physical skeleton coordinates (mm)
    'df_skeleton_2d' pd.DataFrame columns: arc, z
    """

    pixel_size     = stent_geometry['strut_thickness'] / pixels_per_strut
    arc_lo, arc_hi = arc_padded.min(), arc_padded.max()
    z_lo,   z_hi   = z_padded.min(),   z_padded.max()
    n_cols = int(np.ceil((arc_hi - arc_lo) / pixel_size)) + 1
    n_rows = int(np.ceil((z_hi   - z_lo)   / pixel_size)) + 1

    reg_flat  = stent_df['region'].values               
    arc_three = np.concatenate([arc_flat - circumference, arc_flat, arc_flat + circumference])
    z_three   = np.concatenate([z_flat,                   z_flat,   z_flat])
    reg_three = np.concatenate([reg_flat,                 reg_flat, reg_flat])

    pad_mask  = ((arc_three >= arc_flat.min() - 0.20 * circumference) &
                 (arc_three <= arc_flat.max() + 0.20 * circumference))
    arc_use, z_use, reg_use = arc_three[pad_mask], z_three[pad_mask], reg_three[pad_mask]

    col_idx   = np.clip(((arc_use - arc_lo) / pixel_size).astype(int), 0, n_cols - 1)
    row_idx   = np.clip(((z_use   - z_lo)   / pixel_size).astype(int), 0, n_rows - 1)

    img       = np.zeros((n_rows, n_cols), dtype=bool)
    img_label = np.zeros((n_rows, n_cols), dtype=np.int32)
    img[row_idx, col_idx]       = True
    img_label[row_idx, col_idx] = reg_use

    img_solid  = binary_dilation(img,       footprint=disk(dilate_px))
    img_solid  = binary_closing (img_solid, footprint=disk(1))
    _, (sr, sc) = distance_transform_edt(~img, return_indices=True)
    img_label   = np.where(img_solid, img_label[sr, sc], 0)

    img_skel = skeletonize(img_solid)

    # If there is only one region or all regions are mutually allowed, skip the cutting step to avoid over-pruning
    if region_allowed.sum() == region_allowed.size:
        pass
    else:
        img_skel = _prune_spur_branches(img_skel, min_branch_length=2 * pixels_per_strut)
        img_skel = _cut_bad_bridges(img_skel, img_label, region_allowed)
        img_skel = _prune_spur_branches(img_skel, min_branch_length=2 * pixels_per_strut)

    sk_rows, sk_cols = np.where(img_skel)
    skel_arc_all     = arc_lo + (sk_cols + 0.5) * pixel_size
    skel_z_all       = z_lo   + (sk_rows + 0.5) * pixel_size
    seam_mask        = (skel_arc_all >= arc_min_flat) & (skel_arc_all <= arc_max_flat)
    skel_arc         = skel_arc_all[seam_mask]
    skel_z           = skel_z_all[seam_mask]
    df_skeleton_2d   = pd.DataFrame({'arc': skel_arc, 'z': skel_z})


    if plot:
        np.random.seed(0)
        cmap    = np.random.rand(int(img_label.max()) + 1, 3)
        cmap[0] = [1, 1, 1]
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))

        # Panel 1 — region labels
        axes[0].imshow(cmap[img_label].transpose(1, 0, 2), origin='lower',
                    extent=(z_lo, z_hi, arc_lo, arc_hi), aspect='equal')
        axes[0].set_title('Pixel region labels (3D)')
        axes[0].set_xlabel('z (mm)'); axes[0].set_ylabel('arc (mm)')

        # Panel 2 — skeleton
        axes[1].imshow(img_skel.T, cmap='gray_r', origin='lower',
                    extent=(z_lo, z_hi, arc_lo, arc_hi), aspect='equal')
        axes[1].axhline(arc_min_flat, color='red', linestyle='--', linewidth=1)
        axes[1].axhline(arc_max_flat, color='red', linestyle='--', linewidth=1)
        axes[1].set_title('Skeleton (post-cut)')
        axes[1].set_xlabel('z (mm)'); axes[1].set_ylabel('arc (mm)')

        # Panel 3 — skeleton overlay on point cloud
        axes[2].scatter(z_padded, arc_padded, s=0.3, c='lightsteelblue', linewidths=0)
        axes[2].scatter(skel_z,   skel_arc,   s=1.0, c='crimson',        linewidths=0)
        axes[2].axhline(arc_min_flat, color='red', linestyle='--', linewidth=1)
        axes[2].axhline(arc_max_flat, color='red', linestyle='--', linewidth=1)
        axes[2].set_title('Skeleton over outer points')
        axes[2].set_xlabel('z (mm)'); axes[2].set_ylabel('arc (mm)')
        axes[2].set_aspect('equal')

        plt.tight_layout()
        plt.show()

    return {
        'skel_arc'      : skel_arc,
        'skel_z'        : skel_z,
        'df_skeleton_2d': df_skeleton_2d,
    }


def wrap_skeleton_to_3d(
    skel_arc: np.ndarray,
    skel_z: np.ndarray,
    r_mid: float,
) -> dict:
    """
    Inverse of open_stent_to_plane: (arc, z) → (x, y, z) at radius r_mid.

    Returns
    -------
    'df_skeleton_3d'   pd.DataFrame   arc, theta, x, y, z  (shifted frame)
    'skeleton_points'  (N, 3) ndarray
    'scene_skeleton'   trimesh.Scene
    """
    theta_skel = skel_arc / r_mid
    x_skel     = r_mid * np.cos(theta_skel)
    y_skel     = r_mid * np.sin(theta_skel)

    df_skeleton_3d  = pd.DataFrame({
        'arc': skel_arc, 'theta': theta_skel,
        'x': x_skel, 'y': y_skel, 'z': skel_z,
    })
    skeleton_points = df_skeleton_3d[['x', 'y', 'z']].values

    return {
        'df_skeleton_3d' : df_skeleton_3d,
        'skeleton_points': skeleton_points,
    }



def adjust_skeleton_to_local_midsurface(
    skel_arc,
    skel_z,
    stent_df,
    r_mid,
    circumference,
    search_radius,
) -> dict:
    """
    Replace the constant r_mid wrap with a per-point local r, computed as
    the mean r of surface points within `search_radius` (in arc/z plane)
    around each skeleton point.

    Parameters
    ----------
    skel_arc, skel_z : ndarray       from SKELETON_2D
    stent_df         : pd.DataFrame  must have theta, z_cylindrical, r
    r_mid            : float         original global wrap radius
    circumference    : float         2π r_mid  (for periodic tiling)
    search_radius    : float         neighbourhood size (mm)  — see note below
    verbose          : bool

    Returns
    -------
    'df_skeleton_3d'   pd.DataFrame   columns: arc, theta, x, y, z, local_r
    'skeleton_points'  (N, 3) ndarray
    'local_r'          (N,)  ndarray  per-point adjusted radius
    """
    # Triple-tile surface points so the periodic seam is seamless
    arc_all = r_mid * stent_df['theta'].values
    z_all   = stent_df['z_cylindrical'].values
    r_all   = stent_df['r'].values

    arc_three = np.concatenate([arc_all - circumference, arc_all, arc_all + circumference])
    z_three   = np.concatenate([z_all,                   z_all,   z_all])
    r_three   = np.concatenate([r_all,                   r_all,   r_all])

    tree = cKDTree(np.column_stack([arc_three, z_three]))

    # Vectorised radius query: returns one list of neighbour indices per skeleton point
    nb_idx  = tree.query_ball_point(np.column_stack([skel_arc, skel_z]),
                                    r=search_radius)
    local_r = np.array([
        r_three[idx].mean() if len(idx) else r_mid
        for idx in nb_idx
    ])

    # Wrap with the local radius — same theta, same z, only the wrap radius changes
    theta_skel = skel_arc / r_mid
    x = local_r * np.cos(theta_skel)
    y = local_r * np.sin(theta_skel)
    z = skel_z

    return {
        'df_skeleton_3d' : pd.DataFrame({
            'arc'    : skel_arc,
            'theta'  : theta_skel,
            'x'      : x,
            'y'      : y,
            'z'      : z,
            'local_r': local_r,
        }),
        'skeleton_points': np.column_stack([x, y, z]),
        'local_r'        : local_r,
    }


def compute_pre_stent_size_ratio(mesh: trimesh.Trimesh) -> dict:
    pts = mesh.vertices
    centered = pts - pts.mean(axis=0)

    # PCA: largest eigenvector = principal (longest) axis
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)          # eigh: stable for symmetric matrices
    principal_axis = eigvecs[:, np.argmax(eigvals)]  # unit vector along stent axis

    # Project all points onto the principal axis → 1D extent = length
    projections = centered @ principal_axis
    length = projections.max() - projections.min()

    # Radial distance from the principal axis → diameter
    axial_component = np.outer(projections, principal_axis)
    radial_vectors  = centered - axial_component
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    diameter = 2.0 * radial_distances.max()

    return {
        'length'    : length,
        'diameter'  : diameter,
        'size_ratio': length / diameter,
    }
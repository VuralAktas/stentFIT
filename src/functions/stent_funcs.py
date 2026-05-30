import numpy as np
import trimesh
from trimesh.path.entities import Line
import matplotlib.pyplot as plt
import pandas as pd
import ast
import datetime

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, convolve, label as cc_label
from skimage.morphology import skeletonize, binary_dilation, binary_closing, disk


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
) -> dict:
    """
    Longitudinal-slice connected components on all surface points.
    conn_radius_3d is automatically scaled to the actual point spacing.
    """
    if not segmentation_wanted:
        stent_df           = stent_df.copy()
        stent_df['region'] = 1
        region_allowed     = np.array([[True]])
        print(f"[6.1.5] Segmentation skipped — all points in one region")
        print(f"        3D-adj pairs       : 0")
        print(f"        Allowed density    : 100.000%")
        return {
            'stent_df'      : stent_df,
            'region_allowed': region_allowed,
            'n_regions'     : 1,
            'conn_radius_3d': 0.0,
        }

    pts3d    = stent_df[['x', 'y', 'z']].values
    z_vals   = stent_df['z_cylindrical'].values
    n_samples = len(pts3d)

    # Derive conn_radius_3d from point density on the surface
    r_mean       = stent_df['r'].values.mean()
    z_range      = z_vals.max() - z_vals.min()
    surface_area = 2 * np.pi * r_mean * z_range
    avg_spacing  = np.sqrt(surface_area / n_samples)
    conn_radius_3d = 3.0 * avg_spacing

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
        'conn_radius_3d': conn_radius_3d,
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
    plot: bool,
) -> dict:
    """
    Rasterise → dilate → skeletonise → prune spurs → cut cross-region bridges
    → back to physical (arc, z) coords.
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

    if region_allowed.sum() < region_allowed.size:
        img_skel = _prune_spur_branches(img_skel, min_branch_length=2 * pixels_per_strut)
        img_skel = _cut_bad_bridges(img_skel, img_label, region_allowed)
        img_skel = _prune_spur_branches(img_skel, min_branch_length=2 * pixels_per_strut)

    sk_rows, sk_cols = np.where(img_skel)
    skel_arc_all     = arc_lo + (sk_cols + 0.5) * pixel_size
    skel_z_all       = z_lo   + (sk_rows + 0.5) * pixel_size
    seam_mask        = (skel_arc_all >= arc_min_flat) & (skel_arc_all <= arc_max_flat)
    skel_arc         = skel_arc_all[seam_mask]
    skel_z           = skel_z_all[seam_mask]

    if plot:
        np.random.seed(0)
        cmap    = np.random.rand(int(img_label.max()) + 1, 3)
        cmap[0] = [1, 1, 1]
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))

        axes[0].imshow(cmap[img_label].transpose(1, 0, 2), origin='lower',
                    extent=(z_lo, z_hi, arc_lo, arc_hi), aspect='equal')
        axes[0].set_title('Pixel region labels (3D)')
        axes[0].set_xlabel('z (mm)'); axes[0].set_ylabel('arc (mm)')

        axes[1].imshow(img_skel.T, cmap='gray_r', origin='lower',
                    extent=(z_lo, z_hi, arc_lo, arc_hi), aspect='equal')
        axes[1].axhline(arc_min_flat, color='red', linestyle='--', linewidth=1)
        axes[1].axhline(arc_max_flat, color='red', linestyle='--', linewidth=1)
        axes[1].set_title('Skeleton (post-cut)')
        axes[1].set_xlabel('z (mm)'); axes[1].set_ylabel('arc (mm)')

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
        'df_skeleton_2d': pd.DataFrame({'arc': skel_arc, 'z': skel_z}),
    }

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



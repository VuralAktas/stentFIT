import numpy as np
import trimesh
import pandas as pd
from pathlib import Path
import os
from sklearn.cluster import DBSCAN
from skimage.morphology import closing

from .stent_plotting import plot_points_3d_html, plot_thickness_diagnostics_html


def compute_pre_stent_size_ratio(
        mesh: trimesh.Trimesh) -> dict:
    """
    Estimate the stent length and diameter from the raw mesh vertices.

    This is a quick, cheap pre-check run before any sampling. It fits a PCA
    axis to the mesh vertices, measures the length along that axis and the
    diameter across it, and returns their ratio. :func:`sample_stent_points`
    uses this ratio to pick how many points to sample.

    :param mesh: The stent surface mesh, already loaded as a ``trimesh`` object.
    :returns: Dict with the stent ``length``, ``diameter``, and their
        ``size_ratio`` (length / diameter).
    """
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
        remove_supports: bool,
        random_seed: int | None = None,
        out_dir: Path | None = None) -> dict:
    """
    Sample the mesh, align it, and extract the stent point cloud and features.

    This does the main sampling work behind :func:`sample_stent_points`. It
    samples points on the mesh surface, fits a PCA axis and rotates it onto
    ``[0, 0, 1]``, then builds a cylindrical-coordinate point cloud. Along the
    axis it measures the length, diameter, and strut thickness. Thickness is
    read per axial slice and averaged over the middle of the stent. Support
    points can be trimmed first.

    :param mesh: The stent surface mesh, already loaded as a ``trimesh`` object.
    :param n_samples: Number of points to sample. ``None`` uses one sample per
        face, scaled by ``samples_per_face``.
    :param samples_per_face: Samples per mesh face when ``n_samples`` is ``None``.
    :param n_thickness_slices: Number of axial slices used to measure thickness.
    :param slice_cutoff: Slices dropped from each axial end before averaging,
        so the closed stent ends do not skew the thickness.
    :param remove_supports: Trim print-support points before extracting features.
    :param random_seed: Seed for the surface sampling. ``None`` draws a fresh
        cloud each call; an int makes it reproducible.
    :param out_dir: Folder to write ``thickness_diagnostics.html`` into. ``None``
        skips the diagnostic plot.
    :returns: Dict with the point cloud (``stent_df``), the ``stent_features``
        (length, diameter, radius, strut_thickness, z-bounds, inner/outer/mid
        radii, center_cylinder_radius, num_points), and the
        ``stent_centerline_direction`` (the PCA axis before rotation).
    """

    if n_samples is None:
        n_samples = len(mesh.faces) * samples_per_face
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples, seed=random_seed)

    # PCA centroid and main axis
    pca_mean  = pts.mean(axis=0)
    centered  = pts - pca_mean
    cov       = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pca_axis  = eigvecs[:, np.argmax(eigvals)]

    center_cylinder_radius = np.linalg.norm(centered, axis=1).min() * 0.5

    # Rotation matrix mapping pca_axis onto [0, 0, 1]
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

    # Build the cylindrical-coordinate point cloud
    shifted = (R @ centered.T).T

    r     = np.sqrt(shifted[:, 0]**2 + shifted[:, 1]**2)
    theta = np.arctan2(shifted[:, 1], shifted[:, 0])
    z_cyl = shifted[:, 2]

    # Removing support points is optional and it is hard coded for a specific stent design. 
    if remove_supports:
        
        # Drop the -z support by keeping only the middle band of points and the outer radial points
        z_mid_center = 0.5 * (z_cyl.min() + z_cyl.max())
        z_span       = z_cyl.max() - z_cyl.min()
        middle_band  = np.abs(z_cyl - z_mid_center) < 0.20 * z_span
        r_inner_mid  = np.percentile(r[middle_band], 1)
        radial_keep  = r >= r_inner_mid * 0.8

        '''# Drop the +z support by keeping only the largest DBSCAN cluster
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

        # Trim the solid closing ring at the -z end (walk in until lattice)
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

        # Hard axial clamp for this stent
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

    # Bounding box (plots only)
    z_cyl_min, z_cyl_max = z_cyl.min(), z_cyl.max()
    stent_length   = z_cyl_max - z_cyl_min
    stent_diameter = 2.0 * r.max()

    # Strut thickness: read per axial slice, averaged over the middle of the stent
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
    if df_thick.empty:
        raise ValueError(
            "No valid thickness slices: every slice was empty or below the point "
            "cutoff. Try more sample points, fewer thickness slices, or a smaller "
            "slice_cutoff.")
    strut_thick_final = df_thick['thickness'].mean()

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

    if out_dir is not None:
        thickness_html = os.path.join(out_dir, 'thickness_diagnostics.html')
        plot_thickness_diagnostics_html(df_thick, r, thickness_html, strut_thick_final)
        print(f"[plot] {thickness_html}  (inspect the strut thickness)")

    return {
        'stent_df'                   : df,
        'stent_features'             : stent_features,
        'stent_centerline_direction' : pca_axis,
    }



def sample_stent_points(
        mesh: trimesh.Trimesh,
        stent_name: str,
        output_dir: Path,
        n_points: int = None,
        samples_per_face: int = 1,
        max_display: int = 500_000,
        remove_supports: bool = False,
        random_seed: int = 0) -> dict:
    """
    Turn the stent mesh into a cleaned, aligned point cloud.

    This is the first step of the pipeline. When ``n_points`` is not given,
    it first picks a sample count from the stent size (small or short stents
    get fewer points). It then calls :func:`preprocess_stent` to align the
    PCA axis to ``[0, 0, 1]``, extract the stent features, and build a
    cylindrical-coordinate point cloud. The cloud is saved as
    ``sampling_points.csv`` and drawn to ``sampling_points.html``, and the
    strut thickness is drawn to ``thickness_diagnostics.html``.

    :param mesh: The stent surface mesh, already loaded as a ``trimesh`` object.
    :param stent_name: Name used to label outputs and plots.
    :param output_dir: Folder where the CSV and HTML view are written.
    :param n_points: Number of points to sample. ``None`` picks the count
        automatically from the stent size.
    :param samples_per_face: Number of samples taken per mesh face.
    :param max_display: Maximum number of points drawn in the HTML view.
    :param remove_supports: Drop print-support points during sampling.
    :param random_seed: Seed for the sampling, for repeatable runs.
    :returns: Dict with the point cloud (``stent_df``), the extracted
        ``stent_features``, and the ``stent_centerline_direction`` unit vector.
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
        n_thickness_slices=100, slice_cutoff=5,
        remove_supports=remove_supports, random_seed=random_seed, out_dir=output_dir)
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


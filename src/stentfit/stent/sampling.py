import numpy as np
import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import ast
import pathlib
import json
import os
from sklearn.cluster import DBSCAN
from skimage.morphology import closing

from .plotting import plot_points_3d_html

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent.absolute()


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


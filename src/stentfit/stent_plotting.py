import numpy as np
import trimesh
from trimesh.path.entities import Line
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, to_rgb
from matplotlib.lines import Line2D
import pandas as pd
import ast
import json
import os
import glob
import base64
from scipy.spatial import cKDTree
from scipy.interpolate import splev
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as pcolors


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
    bounds = np.asarray(crown_res.get('boundary_z', []), dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=zc, y=cnt, mode='lines+markers', name='points / slice',
        line=dict(color='gray'), marker=dict(size=4, color='gray'),
        hovertemplate="z=%{x:.4f}<br>points/slice=%{y:.1f}<extra></extra>"))
    if len(dips):
        fig.add_trace(go.Scatter(
            x=zc[dips], y=cnt[dips], mode='markers', name='candidate dips',
            marker=dict(size=10, color='lightgray', symbol='triangle-down',
                        line=dict(color='gray', width=1)),
            hovertemplate="candidate dip<br>z=%{x:.4f}<br>points/slice=%{y:.1f}<extra></extra>"))
    for i, zb in enumerate(bounds):
        fig.add_vline(x=float(zb), line=dict(color='red', width=1.5),
                      annotation_text='boundary' if i == 0 else None,
                      annotation_position='top')
    fig.add_hline(y=thresh, line=dict(color='red', dash='dot', width=1),
                  annotation_text='depth cutoff', annotation_position='top left')
    fig.update_layout(
        template='plotly_white', height=420, margin=dict(l=40, r=20, t=50, b=40),
        title=f"Crown boundaries -> {n_bands} crowns  "
              f"({len(bounds)} cuts from {len(dips)} candidate dips)",
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



def plot_skeleton_splines_trimesh(skeleton_splines, output_dir, show=False):
    """Export the fitted splines as a coloured 3D path (.glb + trimesh .html).

    Evaluates each spline into a polyline, builds a ``trimesh.path.Path3D`` with one
    colour per curve, writes a portable ``skeleton_splines.glb`` and a self-contained
    ``skeleton_splines_trimesh.html``, and (when ``show``) opens the interactive
    trimesh window. ``show`` defaults to ``False`` so the function stays headless-safe
    (opening the window needs a GUI backend such as ``pyglet``); the ``.html`` file is
    the portable view.
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
        try:
            spline_path.show()
        except Exception as e:
            print(f"[trimesh] interactive window skipped ({e}); "
                  f"use skeleton_splines_trimesh.html instead")
    return spline_path


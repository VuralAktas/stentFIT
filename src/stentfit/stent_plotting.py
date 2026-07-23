import numpy as np
import trimesh
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
from trimesh.viewer import notebook as _tvn
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as pcolors
from plotly.subplots import make_subplots


def _downsample_df(
        df: pd.DataFrame,
        max_display: int | None,
        random_state: int = 0) -> pd.DataFrame:
    """
    Randomly subsample rows so a plot draws at most ``max_display`` points.

    :param df: Rows to subsample.
    :param max_display: Maximum rows to keep. ``None`` or a value at least as
        large as ``len(df)`` returns ``df`` unchanged.
    :param random_state: Seed for the row sampling, for repeatable plots.
    :returns: ``df`` itself, or a random ``max_display``-row subset of it.
    """
    if max_display is None or len(df) <= max_display:
        return df
    return df.sample(max_display, random_state=random_state)



def plot_points_3d_html(
    df: pd.DataFrame,
    id_col: str,
    out_path: str,
    color_col: str | None = None,
    max_display: int = 40000,
    title: str = "",
    point_size: float = 1,
    categorical: bool = False) -> str:
    """
    Draw a point cloud as an interactive 3D scatter and save it as HTML.

    ``df`` is downsampled to ``max_display`` points first, so large clouds
    stay responsive in the browser. Coloring has three modes: no ``color_col``
    draws every point in one flat color; ``color_col`` with ``categorical=True``
    draws one trace per label with its own legend entry; ``color_col`` without
    ``categorical`` draws a single trace with a continuous colorbar.

    :param df: Point cloud with at least ``x``, ``y``, ``z``, and ``id_col`` columns.
    :param id_col: Column shown as the point ID on hover.
    :param out_path: File path the HTML view is written to.
    :param color_col: Column used to color the points. ``None`` disables coloring.
    :param max_display: Maximum number of points drawn, downsampled if ``df`` is larger.
    :param title: Plot title. The shown/total point count is appended automatically.
    :param point_size: Marker size for the scatter points.
    :param categorical: Treat ``color_col`` as discrete labels instead of a
        continuous value.
    :returns: ``out_path``, for chaining into a caller's own return value.
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



def _skeleton_edge_segments(skeleton_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the x/y/z arrays to draw every skeleton edge as one Plotly line trace.

    Each edge contributes its two endpoints followed by a ``NaN``, which
    breaks the line so Plotly draws many disconnected segments from a single
    ``Scatter3d`` trace instead of one per edge. Each undirected edge
    (``neighbor_ids`` is stored both ways) is only emitted once.

    :param skeleton_df: Skeleton graph with ``skeleton_point_id``, ``x``,
        ``y``, ``z``, and ``neighbor_ids`` columns.
    :returns: ``(xe, ye, ze)`` coordinate arrays, ``NaN``-separated, ready to
        pass straight to ``go.Scatter3d(mode='lines')``.
    """
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
    """
    Draw the 3D skeleton graph as edges and node-type markers, save as HTML.

    Every edge is drawn once as a single line trace
    (:func:`_skeleton_edge_segments`); nodes are downsampled to
    ``max_display`` and colored by ``node_type`` (``line``, ``junction``,
    ``endpoint``, ``isolated``), each as its own legend-toggleable trace.

    :param skeleton_df: Skeleton graph with ``x``, ``y``, ``z``,
        ``skeleton_point_id``, ``degree``, ``node_type``, and
        ``neighbor_ids`` columns.
    :param out_path: File path the HTML view is written to.
    :param title: Plot title.
    :param max_display: Maximum number of nodes drawn, downsampled if
        ``skeleton_df`` is larger. Edges are always drawn in full.
    :returns: ``out_path``, for chaining into a caller's own return value.
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
    """
    Draw the final 3D skeleton overlaid on a sparse stent surface cloud, as HTML.

    Both the surface cloud and the skeleton nodes are downsampled to
    ``max_cloud`` points; skeleton edges are always drawn in full
    (:func:`_skeleton_edge_segments`). The cloud is drawn faint and small so
    the skeleton stays the clear focal point.

    :param skeleton_df: Final 3D skeleton graph with ``x``, ``y``, ``z``,
        ``skeleton_point_id``, and ``neighbor_ids`` columns.
    :param stent_df: Stent surface point cloud with ``x``, ``y``, ``z``, and
        ``point_id`` columns.
    :param out_path: File path the HTML view is written to.
    :param max_cloud: Maximum number of points drawn for the surface cloud
        and for the skeleton nodes, each downsampled independently.
    :returns: ``out_path``, for chaining into a caller's own return value.
    """
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
    """
    Draw every fitted spline curve in 3D, each in its own color, as HTML.

    Each spline is evaluated at ``n_eval`` points along its parameter range
    (``scipy.interpolate.splev``); a curve with no fitted spline (the
    polyline fallback from :func:`~stentfit.stent_splines.fit_curve_spline`)
    is drawn from its raw control points instead. ``None`` entries (curves
    where fitting produced nothing) are skipped.

    :param splines: Per-curve fit results from
        :func:`~stentfit.stent_splines.fit_skeleton_splines`.
    :param out_path: File path the HTML view is written to.
    :param n_eval: Number of points each spline is evaluated at for drawing.
    :returns: ``out_path``, for chaining into a caller's own return value.
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



def plot_ring_dips_html(ring_res: dict, out_path: str) -> str:
    """
    Draw the ring-boundary dip detection profile and save it as HTML.

    Plots the smoothed points-per-slice curve along z, marks the candidate
    dips and the depth cutoff used to filter them, and draws a vertical line
    at each boundary that was actually used to cut the stent into rings.

    :param ring_res: Dict returned by :func:`find_rings`; must have
        ``dip_z_centers``, ``dip_counts_smoothed``, ``dip_indices``,
        ``dip_depth_thresh``, and optionally ``n_bands`` / ``boundary_z``.
    :param out_path: File path the HTML view is written to.
    :returns: ``out_path``, for chaining into a caller's own return value.
    """
    zc     = np.asarray(ring_res['dip_z_centers'])
    cnt    = np.asarray(ring_res['dip_counts_smoothed'])
    dips   = np.asarray(ring_res['dip_indices'], dtype=int)
    thresh = float(ring_res['dip_depth_thresh'])
    n_bands = ring_res.get('n_bands', '?')
    bounds = np.asarray(ring_res.get('boundary_z', []), dtype=float)

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
        title=f"Ring boundaries -> {n_bands} rings  "
              f"({len(bounds)} cuts from {len(dips)} candidate dips)",
        xaxis_title='z_cylindrical', yaxis_title='points / slice')
    pio.write_html(fig, out_path, auto_open=False, config={'scrollZoom': True})
    return out_path



def plot_thickness_diagnostics_html(df_thick: pd.DataFrame,
                                    r: np.ndarray,
                                    out_path: str,
                                    strut_thickness: float) -> str:
    z      = df_thick['z'].to_numpy()
    r_in   = df_thick['r_inner'].to_numpy()
    r_out  = df_thick['r_outer'].to_numpy()
    thick  = df_thick['thickness'].to_numpy()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        subplot_titles=('Inner and outer radius along stent axis',
                        'Strut radial thickness along stent axis',
                        'Global r distribution (two peaks = inner and outer wall)'))

    fig.add_trace(go.Scatter(x=z, y=r_out, mode='lines', name='r_outer',
                             line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=z, y=r_in, mode='lines', name='r_inner',
                             line=dict(color='red')), row=1, col=1)

    fig.add_trace(go.Scatter(x=z, y=thick, mode='lines', name='thickness',
                             line=dict(color='steelblue')), row=2, col=1)
    fig.add_hline(y=strut_thickness, line=dict(color='orange', dash='dash'),
                  annotation_text=f'mean = {strut_thickness:.4f}',
                  annotation_position='top left', row=2, col=1)

    fig.add_trace(go.Histogram(x=r, nbinsx=300, marker_color='steelblue',
                               name='r', showlegend=False), row=3, col=1)

    fig.update_xaxes(title_text='z_cylindrical', row=1, col=1)
    fig.update_xaxes(title_text='z_cylindrical', row=2, col=1)
    fig.update_xaxes(title_text='r (radial distance from stent axis)', row=3, col=1)
    fig.update_yaxes(title_text='r', row=1, col=1)
    fig.update_yaxes(title_text='thickness', row=2, col=1)
    fig.update_yaxes(title_text='point count', row=3, col=1)
    fig.update_layout(template='plotly_white', height=900,
                      margin=dict(l=50, r=20, t=50, b=40),
                      title='Strut thickness diagnostics')

    pio.write_html(fig, out_path, auto_open=False, config={'scrollZoom': True})
    return out_path



def plot_ring_convergence_html(history: pd.DataFrame | None,
                               out_path: str,
                               ring_id: int,
                               quality_report: dict | None = None,
                               pps: float | None = None,
                               dil_px: int | None = None) -> str:
    """
    Draw one ring's auto-tune convergence (or a quality summary) and save it as HTML.

    Three cases: with a non-empty ``history`` (from
    :func:`~stentfit.stent_skeleton_2d.tune_skeleton_params`), draws the
    ``total``/``defect``/``quality`` error trajectory across tuning steps.
    Without a history but with a ``quality_report``, draws a bar chart of the
    defect counts instead (used for the fixed-params, no-auto-tune case).
    With neither, draws a placeholder noting auto-tune was off.

    :param history: Per-step tuning history from :func:`~stentfit.stent_skeleton_2d.tune_skeleton_params`.
        ``None`` or empty falls back to the quality-summary or placeholder case.
    :param out_path: File path the HTML view is written to.
    :param ring_id: Ring identifier, used in the plot title.
    :param quality_report: Dict from :func:`~stentfit.stent_skeleton_2d.check_skeleton_quality`,
        used for the quality-summary bar chart when ``history`` is unavailable.
    :param pps: ``pixels_per_strut`` used, shown in the quality-summary title if given.
    :param dil_px: ``dilate_px`` used, shown in the quality-summary title if given.
    :returns: ``out_path``, for chaining into a caller's own return value.
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
        fig.update_layout(title=f'Ring {ring_id} — error convergence',
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
        fig.update_layout(title=f'Ring {ring_id} — quality summary{ptag}',
                          yaxis=dict(title='count', rangemode='tozero'))
    else:
        fig.add_annotation(text='no tuning history (auto-tune off)',
                           xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=f'Ring {ring_id} — tuning')

    fig.update_layout(template='plotly_white', height=450,
                      margin=dict(l=50, r=20, t=50, b=40))
    pio.write_html(fig, out_path, auto_open=False, config={'scrollZoom': True})
    return out_path



def plot_ring_skeleton_2d_html(arc: np.ndarray,
                               z: np.ndarray,
                               surface_arc: np.ndarray,
                               surface_z: np.ndarray,
                               out_path: str,
                               ring_label: str,
                               ring_band: tuple[float, float] | None = None,
                               changed_idx: np.ndarray | None = None,
                               quality_report: dict | None = None,
                               title: str = "") -> str:
    """
    Draw one ring's flat 2D skeleton over its surface points, with any
    flagged defects overlaid, and save it as HTML.

    The surface points are cropped to ``ring_band`` first, if given, so a
    ring skeletonised with a z-halo is shown next to only its own surface
    band. When ``quality_report`` is passed, its bad connections, loops, and
    empty regions are drawn as markers, tagged with the nearest skeleton
    point's index so they line up with the manual-edit prompts. When
    ``changed_idx`` is passed, those skeleton points are highlighted, useful
    for showing what a manual edit changed.

    :param arc: Flat arc-coordinates of the ring's 2D skeleton.
    :param z: Flat z-coordinates of the ring's 2D skeleton.
    :param surface_arc: Flat arc-coordinates of the ring's surface points.
    :param surface_z: Flat z-coordinates of the ring's surface points.
    :param out_path: File path the HTML view is written to.
    :param ring_label: Ring label used in the default title.
    :param ring_band: ``(z_lo, z_hi)`` the surface points are cropped to.
        ``None`` shows every surface point passed in.
    :param changed_idx: Skeleton point indices to highlight as changed.
    :param quality_report: Dict from :func:`~stentfit.stent_skeleton_2d.check_skeleton_quality`;
        its ``bad_edge_xy``, ``loop_points_xy``, and ``empty_xy`` are drawn as
        defect markers, and the issue count is appended to the title.
    :param title: Plot title. ``ring_label`` and the issue count are used if empty.
    :returns: ``out_path``, for chaining into a caller's own return value.
    """
    arc   = np.asarray(arc)
    z     = np.asarray(z)
    s_arc = np.asarray(surface_arc)
    s_z   = np.asarray(surface_z)

    def _in_band(xy_arc, xy_z):
        if ring_band is None:
            return xy_arc, xy_z
        lo, hi = float(ring_band[0]), float(ring_band[1])
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

    ttl = title or f'{ring_label} — 2D skeleton'
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



def _render_ring_2d(ring_2d: dict,
                    label: str,
                    plots_dir: str,
                    changed_idx: np.ndarray | None = None,
                    suffix: str | None = None) -> str:
    """
    Render one ring's current 2D skeleton via :func:`plot_ring_skeleton_2d_html`.

    Used by the interactive edit loop to preview a tentative edit: with
    ``suffix='edited'``, the file is named
    ``<label>_edited_<rec['n_edits']>.html`` instead of ``<label>.html``, so
    each edit gets its own preview without overwriting the original.

    :param ring_2d: Per-ring 2D skeletons, keyed by ``label``.
    :param label: Ring label to render (e.g. ``"ring_01"``).
    :param plots_dir: Folder the HTML view is written into.
    :param changed_idx: Skeleton point indices to highlight as changed.
    :param suffix: ``'edited'`` names the file after the ring's current edit
        count instead of its plain label.
    :returns: Path to the written HTML file.
    """
    rec  = ring_2d[label]
    name = f"{label}_edited_{rec['n_edits']}" if suffix == 'edited' else label
    out  = os.path.join(plots_dir, f"{name}.html")
    plot_ring_skeleton_2d_html(
        rec['arc'], rec['z'], rec['surf_arc'], rec['surf_z'], out, label,
        ring_band=(rec['z_lo'], rec['z_hi']), changed_idx=changed_idx,
        title=f"{name} — 2D skeleton")
    return out



def _to_arc_z(x: np.ndarray, y: np.ndarray, z: np.ndarray, r_mid: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Unroll 3D points onto the (z, arc) plane, recomputing angle from x/y.

    Unlike :func:`~stentfit.stent_skeleton_2d.open_stent_to_plane`, which
    reads a ``theta`` column directly, this recomputes it from ``x``/``y``
    via ``arctan2`` — used for spline points, which only have xyz coordinates.

    :param x: X-coordinates.
    :param y: Y-coordinates.
    :param z: Z-coordinates.
    :param r_mid: Mid-wall radius, used to convert angle to arc length.
    :returns: ``(z, arc)`` coordinate arrays.
    """
    return np.asarray(z, float), r_mid * np.arctan2(np.asarray(y, float),
                                                    np.asarray(x, float))



def _break_seam(z_ax: np.ndarray, arc: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert a NaN wherever an unrolled curve jumps across the arc seam.

    A curve that crosses the seam (e.g. from ``+circumference/2`` back to
    ``-circumference/2``) would otherwise be drawn as one long spurious line
    all the way across the plot. Any gap in ``arc`` wider than ``thresh`` is
    cut by inserting a ``NaN`` at that point in both arrays.

    :param z_ax: Z-coordinates (or another axial coordinate) of the unrolled curve.
    :param arc: Arc-coordinates of the unrolled curve.
    :param thresh: Minimum arc jump between consecutive points that counts as a seam crossing.
    :returns: ``(z_ax, arc)``, each with a ``NaN`` inserted at every seam crossing.
    """
    z_ax = np.asarray(z_ax, float).copy()
    arc  = np.asarray(arc, float).copy()
    for j in np.where(np.abs(np.diff(arc)) > thresh)[0][::-1]:
        z_ax = np.insert(z_ax, j + 1, np.nan)
        arc  = np.insert(arc,  j + 1, np.nan)
    return z_ax, arc



def _plotly_decode(o: dict | list) -> np.ndarray:
    """
    Decode one Plotly-exported array back into a plain numpy array.

    Plotly's HTML export sometimes stores array data compactly as
    base64-encoded typed arrays (a dict with ``bdata``/``dtype``, optionally
    ``shape``) instead of a plain JSON list. This reverses that encoding;
    anything else is passed straight to ``np.asarray``.

    :param o: A trace's raw ``x``/``y`` value from the parsed Plotly JSON —
        either a typed-array dict or a plain list.
    :returns: The decoded array.
    """
    if isinstance(o, dict) and 'bdata' in o:
        dt = {'f8': '<f8', 'f4': '<f4', 'i1': 'i1', 'i2': '<i2', 'i4': '<i4',
              'u1': 'u1', 'u4': '<u4'}[o['dtype']]
        a = np.frombuffer(base64.b64decode(o['bdata']), dtype=dt)
        if 'shape' in o:
            a = a.reshape(tuple(int(s) for s in str(o['shape']).split(',')))
        return a
    return np.asarray(o)



def _load_convergence(path: str) -> dict | None:
    """
    Re-extract the tuning data plotted in a saved ring convergence HTML file.

    Reads the Plotly figure written by :func:`plot_ring_convergence_html`
    back off disk: finds its embedded ``Plotly.newPlot(...)`` call with a
    string-aware bracket scan (so brackets inside trace names don't confuse
    it), parses that JSON, and pulls out either the ``total``/``defect``/
    ``quality`` trajectory traces (auto-tune on) or the single quality-summary
    bar trace (auto-tune off), decoding any typed-array values via
    :func:`_plotly_decode`. Used to redraw those tuning plots as small
    matplotlib strips in :func:`plot_skeleton_splines_2d`, without needing
    the original tuning history in memory.

    :param path: Path to a ``ring_XX_convergence.html`` file.
    :returns: ``None`` if the file can't be read or parsed. Otherwise a dict
        with ``kind`` set to ``'convergence'`` (plus ``total``/``defect``/
        ``quality`` as ``(x, y)`` arrays) or ``'quality_bar'`` (plus ``x``,
        ``y``, ``colors``).
    """
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



def _hue_gap(a: float, b: float) -> float:
    """
    Circular distance between two hues in ``[0, 1)``.

    Hue wraps around (0 and 1 are the same color), so a plain difference
    would overstate the gap between hues on opposite sides of the wrap.

    :param a: First hue, in ``[0, 1)``.
    :param b: Second hue, in ``[0, 1)``.
    :returns: The shorter of the two distances around the circle.
    """
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)



def _band_conv(k: int,
              ring_order: list | None,
              n_bands: int,
              conv_files: list[str],
              conv_dir: str) -> tuple[str | None, int | None]:
    """
    Resolve the convergence-plot file and ring ID for the k-th ring band.

    If ``ring_order`` is available and matches ``n_bands``, the file path is
    built directly from the k-th ring's ID. Otherwise, falls back to
    indexing into ``conv_files`` (sorted by filename) and parsing the ring
    ID back out of that file's name.

    :param k: Index of the ring band, in axial order.
    :param ring_order: Ring IDs in axial order, from
        :func:`~stentfit.stent_rings.detect_rings` / :func:`~stentfit.stent_skeleton_2d.skeletonize_rings_2d`.
        ``None`` or a length mismatch falls back to ``conv_files``.
    :param n_bands: Total number of ring bands.
    :param conv_files: Sorted list of ``ring_XX_convergence.html`` paths, used as the fallback.
    :param conv_dir: Folder the convergence files live in, used to build the
        path when ``ring_order`` is available.
    :returns: ``(path, ring_id)``, or ``(None, None)`` if neither source
        could resolve this band.
    """
    if ring_order is not None and len(ring_order) == n_bands:
        cid = int(ring_order[k])
        return os.path.join(conv_dir, f'ring_{cid:02d}_convergence.html'), cid
    if k < len(conv_files):
        base = os.path.basename(conv_files[k])
        cid  = int(base.split('_')[1])
        return conv_files[k], cid
    return None, None



def plot_skeleton_splines_2d(skeleton_curves: list[list[int]],
                             skeleton_splines: list[dict | None],
                             stent_df: pd.DataFrame,
                             r_mid: float,
                             circumference: float,
                             ring_edges: np.ndarray | None,
                             ring_order: list | None,
                             output_dir: str,
                             stent_name: str) -> dict:
    """
    Draw the unrolled 2D splines over the stent cloud, with per-ring tuning
    plots stacked above their band, and save it as a static PNG + HTML.

    Curves are colored with a greedy rotating palette so any two curves that
    share a point differ in hue. Ring boundaries are read from
    ``stent_features.json`` if present (else ``ring_edges``, else derived
    from ``stent_df``'s ``ring_id`` groups) and drawn as vertical dashed
    lines; each ring's saved convergence/quality-summary HTML
    (:func:`plot_ring_convergence_html`) is parsed back out
    (:func:`_load_convergence`) and redrawn as a small matplotlib strip
    above that ring's band. The figure is saved as a PNG, embedded as a
    self-contained HTML page, and also shown inline.

    :param skeleton_curves: Grouped point-id curves, from
        :func:`~stentfit.stent_splines.group_skeleton_curves`.
    :param skeleton_splines: Per-curve fit results, from
        :func:`~stentfit.stent_splines.fit_skeleton_splines`.
    :param stent_df: Stent surface point cloud, drawn as a grey underlay.
    :param r_mid: Mid-wall radius, used to unroll splines and the cloud to
        (z, arc) coordinates.
    :param circumference: Full circumference at ``r_mid``, used to detect
        and break the seam when unrolling each spline.
    :param ring_edges: Z-boundaries between rings, used if
        ``stent_features.json`` has no ``ring_boundaries``.
    :param ring_order: Ring IDs in axial order, used to match each band to
        its convergence file. ``None`` falls back to parsing the ring ID
        from each convergence file's name.
    :param output_dir: Folder the PNG/HTML are written into, and where
        ``stent_features.json`` and the per-ring convergence plots are read from.
    :param stent_name: Name used to label the plot title.
    :returns: Dict with the paths to the written PNG and HTML (``png``, ``html``).
    """
    #  varied colours, touching curves differ in HUE (rotating greedy colouring)
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

    # ring boundaries (z) for the vertical dashed lines 
    features_path = os.path.join(output_dir, 'stent_features.json')
    ring_lines   = None
    if os.path.exists(features_path):
        with open(features_path) as f:
            cb = json.load(f).get('ring_boundaries')
        if cb is not None:
            ring_lines = np.asarray(cb, float).ravel()
    if ring_lines is None and ring_edges is not None \
            and len(np.asarray(ring_edges).ravel()) >= 2:
        ring_lines = np.asarray(ring_edges, float).ravel()
    if ring_lines is None and 'ring_id' in stent_df.columns:
        g   = (stent_df.groupby('ring_id')['z'].agg(['min', 'max', 'mean'])
                       .sort_values('mean'))
        lo  = g['min'].to_numpy()
        hi  = g['max'].to_numpy()
        ring_lines = np.concatenate([[lo[0]], 0.5 * (hi[:-1] + lo[1:]), [hi[-1]]])
    ring_lines = None if ring_lines is None else np.sort(np.asarray(ring_lines, float))

    #  map each ring band to its convergence file 
    conv_dir   = os.path.join(output_dir, 'skeleton_plots')
    conv_files = sorted(glob.glob(os.path.join(conv_dir, 'ring_*_convergence.html')))
    n_bands    = 0 if ring_lines is None else len(ring_lines) - 1

    #  figure: main unrolled plot (bottom) + per-ring tuning strip (top) 
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

    if ring_lines is not None:
        xlo, xhi = float(ring_lines[0]), float(ring_lines[-1])
        span = (xhi - xlo) or 1.0
        xlo -= 0.01 * span
        xhi += 0.01 * span
        ax.set_xlim(xlo, xhi)
        for e in ring_lines:
            ax.axvline(e, ls='--', lw=1.6, color='k', alpha=0.9, zorder=10)
    else:
        xlo, xhi = ax.get_xlim()
        span = (xhi - xlo) or 1.0

    fig.suptitle(f'{stent_name} — unrolled 2D skeleton + per-ring tuning',
                 fontsize=11, y=0.995)
    ax.set_xlabel('z  (axial position, mm)')
    ax.set_ylabel('arc = r_mid · θ  (circumferential, mm)')

    # per-ring tuning plots above their bands
    tune_colors = (('total', 'black'), ('defect', 'royalblue'), ('quality', 'orange'))
    tax = None
    n_tuned = 0
    for k in range(n_bands):
        a, b = float(ring_lines[k]), float(ring_lines[k + 1])
        fx0  = L + (a - xlo) / (xhi - xlo) * W
        fw   = (b - a) / (xhi - xlo) * W
        pad  = 0.14 * fw
        tax  = fig.add_axes([fx0 + pad, TY0, max(fw - 2 * pad, 1e-3), TH])
        path, cid = _band_conv(k, ring_order, n_bands, conv_files, conv_dir)
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
        tax.set_title(f'ring {cid}', fontsize=7, pad=2)
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
          f"{0 if ring_lines is None else len(ring_lines)} ring boundaries, "
          f"{n_tuned}/{n_bands} ring tuning plots)")
    print(f"[saved] {unrolled_html}")
    return {'png': unrolled_png, 'html': unrolled_html}



def plot_skeleton_splines_trimesh(skeleton_splines: list[dict | None],
                                  output_dir: str,
                                  show: bool = False,
                                  tube_radius: float | None = None,
                                  sections: int = 6) -> trimesh.Trimesh | None:
    """
    Build a 3D tube mesh of every fitted spline and save it as GLB + HTML.

    Each spline is evaluated (or, for the polyline fallback, taken as-is)
    and turned into a chain of cylinder segments, colored per-curve. The
    combined mesh is exported as a ``.glb`` and, where the trimesh notebook
    viewer supports it, as a self-contained HTML page.

    :param skeleton_splines: Per-curve fit results, from
        :func:`~stentfit.stent_splines.fit_skeleton_splines`.
    :param output_dir: Folder the GLB and HTML are written into.
    :param show: Open an interactive trimesh viewer window.
    :param tube_radius: Cylinder radius for each curve. ``None`` picks it
        automatically as a fraction of the mesh's bounding-box diagonal.
    :param sections: Number of sides on each cylinder's cross-section.
    :returns: The combined mesh, or ``None`` if there were no curves to draw.
    """
    _curves = []
    _cmap = plt.get_cmap('tab20')
    for spl in skeleton_splines:
        if spl is None:
            continue
        if spl['tck'] is not None:
            xx, yy, zz = splev(np.linspace(0.0, 1.0, 60), spl['tck'])
            pts = np.column_stack([xx, yy, zz])
        else:                                    # degree-1 polyline fallback
            pts = np.asarray(spl['ctrl'], float)
        if len(pts) < 2:
            continue
        _curves.append(pts)

    if not _curves:
        print("[trimesh] no curves to plot")
        return None

    if tube_radius is None:
        _all_pts = np.vstack(_curves)
        _diag = np.linalg.norm(_all_pts.max(axis=0) - _all_pts.min(axis=0))
        tube_radius = 0.003 * _diag

    _tubes = []
    for _n, pts in enumerate(_curves):
        _color = (np.array(_cmap(_n % 20)) * 255).astype(np.uint8)
        _segments = [
            trimesh.creation.cylinder(radius=tube_radius, segment=[pts[i], pts[i + 1]], sections=sections)
            for i in range(len(pts) - 1)
        ]
        _curve_mesh = trimesh.util.concatenate(_segments)
        _curve_mesh.visual.face_colors = _color
        _tubes.append(_curve_mesh)

    spline_mesh = trimesh.util.concatenate(_tubes)

    splines_glb   = os.path.join(output_dir, 'skeleton_splines.glb')
    splines_thtml = os.path.join(output_dir, 'skeleton_splines_trimesh.html')
    with open(splines_glb, 'wb') as f:
        f.write(spline_mesh.scene().export(file_type='glb'))
    try:
        with open(splines_thtml, 'w') as f:
            f.write(_tvn.scene_to_html(spline_mesh.scene()))
        print(f"[saved] {splines_thtml}")
    except Exception as e:
        print(f"[trimesh] html export skipped: {e}")
    print(f"[saved] {splines_glb}  ({len(_curves)} spline curves, tube_radius={tube_radius:.4g})")

    if show:
        try:
            spline_mesh.show()
        except Exception as e:
            print(f"[trimesh] interactive window skipped ({e}); "
                  f"use skeleton_splines_trimesh.html instead")
    return spline_mesh


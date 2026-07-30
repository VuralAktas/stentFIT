import numpy as np
import pandas as pd
import ast
import json
import os
from collections import Counter
from scipy.interpolate import splprep

from ..plotting import plot_splines_html
import splinepy 
from beamme.core.mesh import Mesh
from beamme.four_c.material import MaterialReissner
from beamme.four_c.element_beam import Beam3rHerm2Line3, Beam3rLine2Line2
from beamme.mesh_creation_functions.beam_splinepy import create_beam_mesh_from_splinepy


def group_skeleton_curves(skeleton_points_df: pd.DataFrame) -> list[list[int]]:
    """
    Split the skeleton graph into curves: chains between junctions/endpoints,
    plus closed loops.

    A "special" node is anything with degree != 2 (an endpoint, junction, or
    isolated point). From every special node, each unvisited edge is walked
    until it reaches another special node, tracing out one curve. Any edges
    left unvisited afterward belong to a closed loop with no special node
    at all (every node on it has degree 2), so those are walked separately,
    starting anywhere on the loop and ending back where they started.

    :param skeleton_points_df: 3D skeleton graph with ``skeleton_point_id``
        and ``neighbor_ids`` columns.
    :returns: List of curves, each a list of point IDs in path order. A
        closed loop's first and last ID are the same point.
    """
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



def prune_spur_curves(curves: list[list[int]]) -> list[list[int]]:
    """
    Drop curves that end in a free (unshared) endpoint, iteratively.

    A closed loop (same start and end point) is always kept. Otherwise, a
    curve is dropped if either end is "free" — touched by only that one
    curve, meaning it's a dead-end rather than a real junction shared with
    another curve. This repeats until a full pass drops nothing, since
    removing one spur can free up the endpoint of its neighbour, exposing a
    new spur to drop next round.

    :param curves: Curves from :func:`group_skeleton_curves`, each a list of
        point IDs in path order.
    :returns: The curves that survived pruning.
    """
    curves = [list(c) for c in curves]
    while True:
        endpoint_count = Counter()
        for c in curves:
            endpoint_count[c[0]]  += 1
            endpoint_count[c[-1]] += 1

        kept, dropped = [], 0
        for c in curves:
            if c[0] == c[-1]:                                  # closed loop -> keep
                kept.append(c)
                continue
            free_start = endpoint_count[c[0]]  <= 1
            free_end   = endpoint_count[c[-1]] <= 1
            if free_start or free_end:
                dropped += 1
            else:
                kept.append(c)

        curves = kept
        if dropped == 0:                                       # fixed point reached
            break
    return curves



def fit_curve_spline(point_ids: list[int],
                     coords: pd.DataFrame,
                     every: int,
                     k: int,
                     s: float) -> dict | None:
    """
    Fit one B-spline to a single curve's skeleton points.

    Consecutive duplicate points are dropped first. Every ``every``-th point
    is kept as a spline control point (always including the last), with
    ``every`` reduced automatically if the curve is too short to leave
    enough control points for degree ``k``. A closed loop is fit as a
    periodic spline, dropping its duplicated start/end control point first;
    if that leaves too few points to close the loop, it falls back to an
    open curve. If ``scipy.interpolate.splprep`` fails (or there aren't
    enough control points for any spline), the control points themselves are
    returned as a polyline fallback instead.

    :param point_ids: One curve's point IDs in path order, from
        :func:`group_skeleton_curves`.
    :param coords: Skeleton points' ``x``, ``y``, ``z`` coordinates, indexed
        by point ID.
    :param every: Take every Nth point as a control point.
    :param k: Target B-spline degree.
    :param s: Smoothing factor passed to ``splprep``. ``0`` interpolates the
        control points exactly.
    :returns: ``None`` if fewer than 2 distinct points remain. Otherwise a
        dict with the fitted ``tck``/``u`` (``None`` if fitting failed or
        wasn't attempted), the ``ctrl`` points used, ``n_ctrl``, the actual
        degree ``k`` used, whether it closed as a loop (``is_loop``), and
        the curve's physical ``length``.
    """
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



def _spline_record(spl: dict | None) -> dict | None:
    """
    Convert one :func:`fit_curve_spline` result into a JSON-serializable record.

    Unpacks scipy's ``tck`` tuple into plain ``degree``/``knot_vector``/
    ``control_points`` fields. Where ``tck`` is ``None`` (the polyline
    fallback), records ``degree=1`` and ``knot_vector=None`` with the raw
    control points instead.

    :param spl: One curve's fit result from :func:`fit_curve_spline`, or ``None``.
    :returns: ``None`` if ``spl`` is ``None``. Otherwise a dict with
        ``degree``, ``knot_vector`` (``None`` for the polyline fallback),
        ``control_points`` (as a plain nested list), ``is_loop``, and ``length``.
    """
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



def fit_skeleton_splines(skeleton_df: pd.DataFrame,
                         output_dir: str,
                         spline_every: int = 10,
                         spline_degree: int = 3,
                         smooth: float = 0.0,
                         prune_spurs: bool = True) -> dict:
    """
    Group the skeleton graph into curves and fit a B-spline to each.

    Groups the 3D skeleton into curves with :func:`group_skeleton_curves`
    (degree-2 chains between junctions/endpoints, plus closed loops), drops
    any spur curve with a free end if ``prune_spurs`` (via
    :func:`prune_spur_curves`), then fits each remaining curve with
    :func:`fit_curve_spline`. Saves ``splines.html`` and
    ``skeleton_splines.json`` (per-curve degree, knot vector, control points
    — with a plain polyline as the fallback where a curve is too short to
    fit a spline to).

    :param skeleton_df: 3D skeleton graph with ``skeleton_point_id``, ``x``,
        ``y``, ``z``, and ``neighbor_ids`` columns, from
        :func:`~stentfit.kernels.skeleton_3d.wrap_skeleton_to_3d`.
    :param output_dir: Folder the HTML view and JSON export are written into.
    :param spline_every: Take every Nth skeleton point as a spline control
        point, to keep the control polygon from being one point per sample.
    :param spline_degree: Target B-spline degree, reduced automatically for
        short curves.
    :param smooth: Smoothing factor passed to ``scipy.interpolate.splprep``.
        ``0`` interpolates the control points exactly.
    :param prune_spurs: Drop curves with a free (non-junction, non-loop) end,
        instead of fitting a spline to them.
    :returns: Dict with the grouped point-id curves (``curves``) and their
        fitted splines (``splines``, one entry per curve, ``None`` where
        fitting failed).
    """
    coords  = skeleton_df.set_index('skeleton_point_id')[['x', 'y', 'z']]
    curves  = group_skeleton_curves(skeleton_df)
    if prune_spurs:
        n_before = len(curves)
        curves   = prune_spur_curves(curves)
        if len(curves) < n_before:
            print(f"Pruned {n_before - len(curves)} spur curve(s) "
                  f"({n_before} -> {len(curves)}).")
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



def _feat(stent_features: dict, key: str):
    """
    Read one stent feature, whichever of its two saved shapes it's stored in.

    A feature is either a plain scalar or a ``{'value': ..., 'unit': ...}``
    dict (the shape used for material parameters elsewhere in the
    pipeline); this reads either the same way.

    :param stent_features: Stent features dict (from ``stent_features.json``
        or in-memory).
    :param key: Feature name to read.
    :returns: The feature's plain value.
    """
    v = stent_features[key]
    return v['value'] if isinstance(v, dict) and 'value' in v else v



def _bspline_from_record(rec: dict) -> "splinepy.BSpline":
    """
    Rebuild a ``splinepy.BSpline`` from one :func:`_spline_record` entry.

    Where ``knot_vector`` is ``None`` (the polyline fallback), builds a
    degree-1 B-spline with a uniform clamped knot vector instead, so the
    polyline can still be meshed the same way as a real spline.

    :param rec: One curve's record, from ``skeleton_splines.json`` (or
        :func:`_spline_record` directly) — with ``degree``, ``knot_vector``,
        and ``control_points``.
    :returns: The reconstructed B-spline.
    """
    ctrl = np.asarray(rec['control_points'], float)
    if rec['knot_vector'] is not None:
        return splinepy.BSpline(degrees=[int(rec['degree'])],
                                knot_vectors=[rec['knot_vector']],
                                control_points=ctrl)
    n  = len(ctrl)                                   # degree-1 polyline fallback
    kv = [0.0, 0.0] + list(np.linspace(0.0, 1.0, n)[1:-1]) + [1.0, 1.0]
    return splinepy.BSpline(degrees=[1], knot_vectors=[kv], control_points=ctrl)



def mesh_skeleton_beams(input_dir: str,
                        l_el: float = 0.1,
                        youngs_modulus: float = 2.0e5,
                        poisson_ratio: float = 0.3,
                        density: float = 0.0,
                        beam_class_label: str = 'Beam3rHerm2Line3') -> Mesh:
    """
    Build a BeamMe 1D beam mesh from a stent's saved splines.

    Reads back ``stent_features.json`` (for the strut thickness, used as the
    circular cross-section radius) and ``skeleton_splines.json`` (for the
    per-curve splines) from ``input_dir``, then meshes each curve with
    BeamMe's ``create_beam_mesh_from_splinepy``, using
    :func:`_bspline_from_record` to rebuild each spline. A curve is skipped
    if it has fewer than 2 control points or meshing raises. Each curve's
    new elements are tagged with a ``curve_color`` VTK cell scalar (folded
    into 16 bins) for ParaView inspection — this is visualisation-only and
    has no effect on the 4C simulation input.

    :param input_dir: Stent output folder to read ``stent_features.json``
        and ``skeleton_splines.json`` from — normally the same folder
        :meth:`~stentfit.stent.Stent.skeletonize` wrote them to.
    :param l_el: Target element length, in mm.
    :param youngs_modulus: Beam material Young's modulus, in MPa.
    :param poisson_ratio: Beam material Poisson's ratio.
    :param density: Beam material density.
    :param beam_class_label: BeamMe beam element type, either
        ``'Beam3rHerm2Line3'`` or ``'Beam3rLine2Line2'``.
    :returns: The assembled BeamMe ``Mesh``.
    """
    beam_classes = {'Beam3rHerm2Line3': Beam3rHerm2Line3,
                    'Beam3rLine2Line2': Beam3rLine2Line2}
    beam_class = beam_classes[beam_class_label]

    # --- read the stent's saved data from the output folder (self-contained) ---
    with open(os.path.join(input_dir, 'stent_features.json')) as f:
        stent_features = json.load(f)
    with open(os.path.join(input_dir, 'skeleton_splines.json')) as f:
        splines_data = json.load(f)

    strut_thickness = float(_feat(stent_features, 'strut_thickness'))

    beam_radius = strut_thickness / 2.0            # circular cross-section radius (mm)

    beam_mesh = Mesh()
    beam_mat  = MaterialReissner(radius=beam_radius, youngs_modulus=youngs_modulus,
                                 nu=poisson_ratio, density=density)

    n_meshed, n_skipped = 0, 0
    for curve_idx, rec in enumerate(splines_data['curves']):
        if rec is None or len(rec['control_points']) < 2:
            n_skipped += 1
            continue
        try:
            n_before = len(beam_mesh.elements)
            create_beam_mesh_from_splinepy(beam_mesh, beam_class, beam_mat,
                                           _bspline_from_record(rec), l_el=l_el)
            # Tag this curve's new elements with a per-element cell scalar for ParaView.
            # This is VTK-only visualization data (element.vtk_cell_data -> get_vtk -> .vtu);
            # InputFile.dump ignores it, so it has no effect on the 4C simulation.
            #   curve_color : curve id folded into <=32 bins so ParaView's categorical
            #                 colouring works (it caps at 32 distinct values)
            for el in beam_mesh.elements[n_before:]:
                el.vtk_cell_data["curve_color"] = curve_idx % 16
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

    '''beam_mesh.write_vtk(output_name='skeleton_beam_mesh', output_directory=output_dir)
    print(f"[saved] {os.path.join(output_dir, 'skeleton_beam_mesh_beam.vtu')}")'''
    return beam_mesh


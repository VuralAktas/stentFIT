import numpy as np
import ast
import json
import os
from collections import Counter
from scipy.interpolate import splprep

from .stent_plotting import plot_splines_html
import splinepy 
from beamme.core.mesh import Mesh
from beamme.four_c.material import MaterialReissner
from beamme.four_c.element_beam import Beam3rHerm2Line3, Beam3rLine2Line2
from beamme.mesh_creation_functions.beam_splinepy import create_beam_mesh_from_splinepy


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



def prune_spur_curves(curves):
    """Drop spur curves left over after grouping (Step 7 double-check).

    Each curve is a chain of skeleton_point_ids whose endpoints (first / last id) are
    the junctions or dead-ends it meets. A curve is kept only if BOTH of its endpoints
    are shared with at least one other curve; a curve with a free endpoint (an id that
    no other curve touches) is a dangling spur and is removed. Closed loops
    (first id == last id) are always kept. Pruning iterates because removing one spur
    can turn a neighbouring segment into a new spur (a chain of dead-end pieces).

    Complements the node-level `prune_skeleton_spurs` in the 3D graph cleanup.
    Returns the filtered list of curves.
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
                         smooth=0.0, prune_spurs=True):
    """Group the skeleton graph into curves and fit a B-spline per curve (Step 7).

    With ``prune_spurs=True`` (default), a curve-level double-check removes dangling
    spur curves (a curve with a free endpoint no other curve shares) before fitting.

    Renders the smooth curves to ``splines.html`` and exports ``skeleton_splines.json``
    (per-curve degree / knot_vector / control_points), which rebuilds directly as a
    ``splinepy.BSpline`` for BeamMe. Returns ``{curves, splines}``.
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



def _feat(stent_features, key):
    """Read a feature value, unwrapping the {'value', 'unit'} form if present."""
    v = stent_features[key]
    return v['value'] if isinstance(v, dict) and 'value' in v else v



def _bspline_from_record(rec):
    """Rebuild a splinepy.BSpline from a skeleton_splines.json curve record.
    knot_vector=None marks a degree-1 polyline fallback -> clamped uniform knots."""

    ctrl = np.asarray(rec['control_points'], float)
    if rec['knot_vector'] is not None:
        return splinepy.BSpline(degrees=[int(rec['degree'])],
                                knot_vectors=[rec['knot_vector']],
                                control_points=ctrl)
    n  = len(ctrl)                                   # degree-1 polyline fallback
    kv = [0.0, 0.0] + list(np.linspace(0.0, 1.0, n)[1:-1]) + [1.0, 1.0]
    return splinepy.BSpline(degrees=[1], knot_vectors=[kv], control_points=ctrl)



def mesh_skeleton_beams(input_dir, output_dir, l_el=0.1, youngs_modulus=2.0e5, poisson_ratio=0.3,
                        density=0.0, beam_class_label='Beam3rHerm2Line3'):
    """Mesh the fitted splines into a 1D Simo-Reissner beam mesh with BeamMe (Step 9).

    Rebuilds each ``skeleton_splines.json`` curve as a ``splinepy.BSpline`` and meshes
    it into one BeamMe ``Mesh`` (beam radius = strut_thickness / 2). Material +
    element choices are provisional placeholders; only the geometry is final. Writes
    ``skeleton_beam_mesh_beam.vtu`` and returns the mesh. Imports ``splinepy`` /
    ``beamme`` lazily so the rest of ``stent_funcs`` runs without those heavy deps.
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


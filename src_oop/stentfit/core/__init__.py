"""
The function modules behind the three domain classes.

Every module here holds module-level functions only — no classes. The package
top level is exactly the three domain classes
(:class:`~stentfit.stent.Stent`, :class:`~stentfit.artery.Artery`,
:class:`~stentfit.simulation.Simulation`); everything function-based lives
here, so the split between "things that carry state" and "things that don't" is
visible from the folder layout alone.

Two kinds of module:

* **Numerical kernels** (:mod:`~stentfit.core.sampling`,
  :mod:`~stentfit.core.rings`, :mod:`~stentfit.core.skeleton_2d`,
  :mod:`~stentfit.core.skeleton_3d`, :mod:`~stentfit.core.splines`,
  :mod:`~stentfit.core.artery_geom`) — arrays in, arrays out, no shared
  state. The procedural package's numerical core, re-homed unchanged.
* **Views** (:mod:`~stentfit.core.plotting`) — the interactive Plotly HTML
  and trimesh figures each pipeline stage writes out for inspection.

The submodules are imported on demand rather than eagerly here, so importing
one kernel does not pull in the whole dependency stack (GMSH, BeamMe, splinepy)
that only the artery and spline kernels need.
"""

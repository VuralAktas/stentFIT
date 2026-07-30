"""
Pure numerical kernels behind the three domain classes.

Every module here holds module-level functions only — arrays in, arrays out,
no shared state and no classes. They are the procedural package's numerical
core, re-homed unchanged; :class:`~stentfit.stent.Stent`,
:class:`~stentfit.artery.Artery`, and :class:`~stentfit.simulation.Simulation`
are the only things that carry state, and they call into these.

The submodules are imported on demand rather than eagerly here, so importing
one kernel does not pull in the whole dependency stack (GMSH, BeamMe, splinepy)
that only the artery and spline kernels need.
"""

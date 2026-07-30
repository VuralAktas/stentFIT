"""
stentFIT — semi-automated virtual stent implantation via contact simulation.

Object-oriented API built around three domain classes:

* :class:`~stentfit.stent.Stent` — an STL stent design through sampling, ring
  detection, 2D/3D skeletonisation, and spline fitting.
* :class:`~stentfit.artery.Artery` — a parametric test artery surface and its
  GMSH-meshed 3D wall solid.
* :class:`~stentfit.simulation.Simulation` — composes the two into a runnable
  4C beam-to-solid input.

Everything else in the package is module-level functions: the numerical
kernels in :mod:`stentfit.kernels` and the Plotly views in
:mod:`stentfit.plotting`.
"""

from .stent import Stent
from .artery import Artery

# Hardcoded while both src/ and src_oop/ share the import name `stentfit`, so
# this package is used from PYTHONPATH rather than installed. Switch to
# importlib.metadata.version("stentfit") once src_oop/ is the installed package.
__version__ = "0.2.0"

__all__ = ["Stent", "Artery"]

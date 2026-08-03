from .stent import Stent
from .artery import Artery
from .simulation import Simulation

# Hardcoded while both src/ and src_oop/ share the import name `stentfit`, so
# this package is used from PYTHONPATH rather than installed. Switch to
# importlib.metadata.version("stentfit") once src_oop/ is the installed package.
__version__ = "0.2.0"

__all__ = ["Stent", "Artery", "Simulation"]

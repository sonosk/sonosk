"""SonoSK: simulation of medical ultrasound imaging and therapy.

Modules
-------
grid        regular grids in 1, 2 and 3 dimensions
transducer  single elements, annular, linear, phased, convex, matrix and
            cylindrical sources
field       CW pressure fields (Rayleigh integral, angular spectrum)
medium      materials and property maps
phantom     shapes, label maps and point scatterers
imaging     pulse-echo simulation, beamforming and B-mode images
thermal     heat source, Pennes bioheat equation, thermal dose, lesions
stack       piezoelectric stack model (Mason) for probe design
safety      acoustic output quantities and regulatory limits
quality     resolution, contrast, CNR, gCNR and speckle measures
analytic    closed-form reference fields
water       properties of water
plot        figures (needs matplotlib)
"""

__version__ = "0.1.0"

from . import (analytic, field, imaging, medium, phantom, quality, safety, stack,
               thermal, transducer, water)
from .grid import Grid
from .medium import Material, Medium

__all__ = [
    "Grid", "Material", "Medium",
    "analytic", "field", "imaging", "medium", "phantom", "quality", "safety",
    "stack", "thermal", "transducer", "water", "__version__",
]

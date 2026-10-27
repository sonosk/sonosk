"""SonoSK: simulation of medical ultrasound imaging and therapy.

For research and teaching only. The models rest on stated assumptions
(see the Limits page of the documentation); results are not validated for
clinical decisions, treatment planning of patients or regulatory claims.

Modules
-------
grid        regular grids in 1, 2 and 3 dimensions
transducer  single elements, annular, linear, phased, convex, matrix and
            cylindrical sources
field       CW pressure fields (Rayleigh integral, angular spectrum, hybrid
            angular spectrum for heterogeneous media)
fullwave    k-space pseudospectral solver: heterogeneous, absorbing,
            nonlinear media in 1, 2 and 3 dimensions
medium      materials and property maps
tissues     tissue property database with a source for every value
phantom     shapes, label maps, random structure and point scatterers
organs      anatomical phantoms of the breast, prostate, uterus and face
imaging     pulse-echo simulation, beamforming and B-mode images
paths       straight-ray travel times and attenuation in heterogeneous media
thermal     heat sources, Pennes bioheat equation, thermal dose, Arrhenius
            damage, perfusion shutdown and multi-point treatments
stack       piezoelectric stack model (Mason) for probe design
safety      acoustic output quantities and regulatory limits
quality     resolution, contrast, CNR, gCNR and speckle measures
benchmarks  convergence and reference cases with machine-readable results
analytic    closed-form reference fields
water       properties of water
plot        figures (needs matplotlib)
"""

__version__ = "0.2.0"

from . import (analytic, benchmarks, field, fullwave, imaging, medium, organs, paths, phantom,
               quality, safety, stack, thermal, tissues, transducer, water)
from .grid import Grid
from .medium import Material, Medium


def run_info():
    """Versions and platform, for the metadata of saved results."""
    import datetime
    import platform

    import numpy
    info = {"sonosk": __version__, "python": platform.python_version(),
            "numpy": numpy.__version__, "platform": platform.platform(),
            "machine": platform.machine(),
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = None
    return info


__all__ = [
    "Grid", "Material", "Medium", "run_info",
    "analytic", "benchmarks", "field", "fullwave", "imaging", "medium", "organs", "paths",
    "phantom", "quality", "safety", "stack", "thermal", "tissues", "transducer", "water",
    "__version__",
]

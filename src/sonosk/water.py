"""Properties of pure water.

References
----------
Marczak W. Water as a standard in the measurements of speed of sound in
liquids. J. Acoust. Soc. Am. 102(5):2776-2779, 1997.

Kell G.S. Density, thermal expansivity, and compressibility of liquid
water from 0 to 150 C. J. Chem. Eng. Data 20(1):97-105, 1975.
"""

import numpy as np

# Marczak (1997), valid 0-95 C
_C = (1.402385e3, 5.038813, -5.799136e-2, 3.287156e-4, -1.398845e-6, 2.787860e-9)

# Kell (1975), valid 0-150 C
_RHO_NUM = (999.83952, 16.945176, -7.9870401e-3, -46.170461e-6, 105.56302e-9, -280.54253e-12)
_RHO_DEN = 16.879850e-3


def _check(t, lo, hi):
    t = np.asarray(t, dtype=float)
    if np.any(t < lo) or np.any(t > hi):
        raise ValueError(f"temperature outside {lo}-{hi} C")
    return t


def sound_speed(temp):
    """Speed of sound [m/s] at temperature temp [C]."""
    t = _check(temp, 0.0, 95.0)
    return np.polynomial.polynomial.polyval(t, _C)


def density(temp):
    """Density [kg/m^3] at temperature temp [C]."""
    t = _check(temp, 0.0, 150.0)
    return np.polynomial.polynomial.polyval(t, _RHO_NUM) / (1.0 + _RHO_DEN * t)


def impedance(temp):
    """Characteristic acoustic impedance [kg/(m^2 s)]."""
    return density(temp) * sound_speed(temp)

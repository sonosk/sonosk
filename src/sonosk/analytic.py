"""Closed-form on-axis pressure of circular sources in a lossless fluid.

These solutions are used as references for the numerical field code.
Results are pressure amplitudes normalised by p0 = rho c u0, where u0 is
the normal velocity amplitude of the source surface.

References
----------
Kinsler L.E., Frey A.R., Coppens A.B., Sanders J.V. Fundamentals of
Acoustics, 4th ed. Wiley, 2000.

O'Neil H.T. Theory of focusing radiators. J. Acoust. Soc. Am.
21(5):516-526, 1949.
"""

import numpy as np


def wavenumber(f, c=1500.0):
    return 2.0 * np.pi * f / c


def piston_axial(z, a, f, c=1500.0):
    """Pressure amplitude p/p0 on the axis of a baffled flat piston.

    z distance from the piston face [m], a piston radius [m],
    f frequency [Hz], c sound speed [m/s].
    """
    z = np.asarray(z, dtype=float)
    k = wavenumber(f, c)
    return 2.0 * np.abs(np.sin(0.5 * k * (np.sqrt(z * z + a * a) - z)))


def piston_last_maximum(a, f, c=1500.0):
    """Position of the last on-axis maximum of a flat piston [m]."""
    lam = c / f
    return a * a / lam - lam / 4.0


def bowl_depth(a, roc):
    """Depth h of a spherical bowl of aperture radius a and curvature roc."""
    if a > roc:
        raise ValueError("aperture radius larger than radius of curvature")
    return roc - np.sqrt(roc * roc - a * a)


def focal_gain(a, roc, f, c=1500.0):
    """Linear pressure gain k h at the geometric focus of a bowl."""
    return wavenumber(f, c) * bowl_depth(a, roc)


def bowl_axial(z, a, roc, f, c=1500.0):
    """Pressure amplitude p/p0 on the axis of a focused bowl (O'Neil 1949).

    z is measured from the apex of the bowl [m], a is the aperture radius
    and roc the radius of curvature [m]. The geometric focus is at z = roc.
    """
    z = np.asarray(z, dtype=float)
    k = wavenumber(f, c)
    h = bowl_depth(a, roc)
    r = np.sqrt((z - h) ** 2 + a * a)
    q = 1.0 - z / roc
    near = np.abs(q) < 1e-9
    qs = np.where(near, 1.0, q)
    p = np.abs(2.0 / qs * np.sin(0.5 * k * (r - z)))
    return np.where(near, k * h, p)

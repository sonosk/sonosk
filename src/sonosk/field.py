"""Continuous-wave pressure fields.

rayleigh() sums the Rayleigh integral over the patches of a transducer,

    p(r) / p0 = (j k / 2 pi) sum_n w_n A_n exp(-(j k + a) R_n) / R_n,

where p0 = rho c u0, w_n is the complex patch weight (apodization and
delay), A_n the patch area, R_n the distance and a the amplitude
attenuation of a homogeneous medium. For curved sources this is the usual
approximation that treats each patch as part of a baffled plane.

angular_spectrum() propagates a field given on a plane by the angular
spectrum method.

Time convention: exp(j omega t).

References
----------
O'Neil H.T. Theory of focusing radiators. J. Acoust. Soc. Am.
21(5):516-526, 1949.

Zeng X., McGough R.J. Evaluation of the angular spectrum approach for
simulations of near-field pressures. J. Acoust. Soc. Am. 123(1):68-76, 2008.
"""

import numpy as np

from .medium import attenuation_np


def rayleigh(transducer, points, f, c=1540.0, alpha0=0.0, y=1.0, max_pairs=4_000_000):
    """Complex pressure p/p0 at points.

    Parameters
    ----------
    transducer : Transducer
    points : (M, 3) array
        Field points [m].
    f : float
        Frequency [Hz].
    c : float
        Sound speed [m/s].
    alpha0, y : float
        Power-law attenuation of the medium, alpha0 in dB/(cm MHz^y).
    max_pairs : int
        Number of point-patch pairs handled at once; lower it if memory is short.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    k = 2 * np.pi * f / c
    a = float(attenuation_np(alpha0, f, y))
    src = transducer.pos
    w = transducer.weights(f) * transducer.area
    g = 1j * k / (2 * np.pi)

    out = np.empty(pts.shape[0], dtype=complex)
    step = max(1, max_pairs // src.shape[0])
    for i in range(0, pts.shape[0], step):
        d = pts[i:i + step, None, :] - src[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        r = np.maximum(r, 1e-12)
        out[i:i + step] = (np.exp(-(1j * k + a) * r) / r) @ w
    return g * out


def grid_points(grid, y=0.0):
    """Field points for a grid: 1-D on the z axis, 2-D in the x-z plane at y,
    3-D everywhere. Returns an (N, 3) array."""
    if grid.ndim == 1:
        z = grid.axis(0)
        return np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
    if grid.ndim == 2:
        x, z = grid.mesh()
        return np.stack([x.ravel(), np.full(x.size, y), z.ravel()], 1)
    x, yy, z = grid.mesh()
    return np.stack([x.ravel(), yy.ravel(), z.ravel()], 1)


def rayleigh_grid(transducer, grid, f, c=1540.0, alpha0=0.0, y=1.0, plane_y=0.0, **kw):
    """rayleigh() evaluated on a Grid; result has the grid shape."""
    p = rayleigh(transducer, grid_points(grid, plane_y), f, c, alpha0, y, **kw)
    return p.reshape(grid.shape)


def angular_spectrum(p0, dx, f, z, c=1540.0, alpha0=0.0, y=1.0, pad=2):
    """Propagate a complex field from the plane z = 0 to the depths z.

    Parameters
    ----------
    p0 : 1-D or 2-D complex array
        Field on a line (x) or a plane (x, y). A 1-D input is a field that
        does not change along y, so the result is a 2-D (x, z) problem.
    dx : float
        Sample spacing of p0 [m], the same along both axes.
    f : float
        Frequency [Hz].
    z : array
        Depths [m], measured from the input plane.
    pad : int
        Zero padding factor against wrap-around.

    Returns
    -------
    array of shape (len(z),) + p0.shape
    """
    p0 = np.asarray(p0, dtype=complex)
    z = np.atleast_1d(np.asarray(z, dtype=float))
    k = 2 * np.pi * f / c
    a = float(attenuation_np(alpha0, f, y))

    shape = tuple(pad * n for n in p0.shape)
    axes = tuple(range(p0.ndim))
    spec = np.fft.fftn(p0, shape, axes=axes)
    ks = [2 * np.pi * np.fft.fftfreq(n, dx) for n in shape]
    kk = np.meshgrid(*ks, indexing="ij")
    kt2 = sum(q * q for q in kk)
    kz = np.sqrt((k * k - kt2).astype(complex))
    kz = np.where(kz.imag > 0, np.conj(kz), kz)    # evanescent waves decay

    out = np.empty((z.size,) + p0.shape, dtype=complex)
    sl = tuple(slice(0, n) for n in p0.shape)
    for i, zi in enumerate(z):
        q = np.fft.ifftn(spec * np.exp(-1j * kz * zi), axes=axes)
        out[i] = q[sl] * np.exp(-a * zi)
    return out


def scale_to_pressure(p_rel, rho=1000.0, c=1540.0, u0=None, power=None, area=None):
    """Convert p/p0 to pascals from the surface velocity u0 [m/s] or from the
    radiated acoustic power [W] of a uniform plane source of given area [m^2]."""
    if u0 is None:
        if power is None or area is None:
            raise ValueError("give u0, or power and area")
        u0 = np.sqrt(2 * power / (rho * c * area))
    return np.asarray(p_rel) * rho * c * u0


def intensity(p, rho=1000.0, c=1540.0):
    """Time-average intensity [W/m^2] of a CW pressure amplitude p [Pa]
    in the plane-wave approximation."""
    return np.abs(p) ** 2 / (2 * rho * c)

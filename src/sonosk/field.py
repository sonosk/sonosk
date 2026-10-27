"""Continuous-wave pressure fields.

rayleigh() sums the Rayleigh integral over the patches of a transducer,

    p(r) / p0 = (j k / 2 pi) sum_n w_n A_n exp(-(j k + a) R_n) / R_n,

where p0 = rho c u0, w_n is the complex patch weight (apodization and
delay), A_n the patch area, R_n the distance and a the amplitude
attenuation of a homogeneous medium. For curved sources this is the usual
approximation that treats each patch as part of a baffled plane.

angular_spectrum() propagates a field given on a plane by the angular
spectrum method. hybrid_angular_spectrum() extends it to heterogeneous
media: the field is propagated slice by slice, with the angular spectrum
step taken at the mean wavenumber of each slice and the local differences
in speed, attenuation and impedance applied in the space domain (Vyas and
Christensen 2012). It is fast enough for 3-D treatment planning and is the
method used in several MR-guided focused ultrasound studies; it neglects
reflected waves, scattering and nonlinearity (see sonosk.fullwave for
those).

Time convention: exp(j omega t).

References
----------
O'Neil H.T. Theory of focusing radiators. J. Acoust. Soc. Am.
21(5):516-526, 1949.

Zeng X., McGough R.J. Evaluation of the angular spectrum approach for
simulations of near-field pressures. J. Acoust. Soc. Am. 123(1):68-76, 2008.

Vyas U., Christensen D. Ultrasound beam simulations in inhomogeneous tissue
geometries using the hybrid angular spectrum method. IEEE Trans. Ultrason.
Ferroelectr. Freq. Control 59(6):1093-1100, 2012.

Johnson S.L., Christensen D.A., Dillon C.R., Payne A. Validation of hybrid
angular spectrum acoustic and thermal modelling in phantoms. Int. J.
Hyperthermia 35(1):578-590, 2018.
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


def pulsed(transducer, points, pulse, c=1540.0, alpha0=0.0, y=1.0, fs=None, level_db=-40.0,
           max_pairs=4_000_000):
    """Transient pressure p(t)/p0 at points for a pulsed excitation.

    The face velocity of every element follows pulse (any object with
    analytic(t) and f0, for example imaging.Pulse used as a one-way
    transmit pulse), delayed and weighted by the transducer's delays and
    apodization. The field is synthesised from Rayleigh integrals at the
    frequencies of the pulse spectrum down to level_db, so attenuation acts
    on each frequency with its own power law.

    Returns
    -------
    t : (n_t,) array [s], time from the start of the pulse at the face
    p : (n_points, n_t) real array, in units of p0 = rho c u0
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    f0 = pulse.f0
    fs = 16 * f0 if fs is None else float(fs)
    # spectrum of the pulse, centred so it starts near t = 0
    dur = pulse.duration
    t_c = 0.5 * dur
    sub = transducer.pos[::max(1, transducer.n_patches // 500)]
    r_max = np.max(np.linalg.norm(pts[:, None, :] - sub[None], axis=2))
    T = r_max / c + dur + float(np.max(transducer.delays)) + 2 / f0
    n = int(2 ** np.ceil(np.log2(T * fs)))
    t = np.arange(n) / fs
    u = pulse.analytic(t - t_c).real
    U = np.fft.rfft(u)
    f = np.fft.rfftfreq(n, 1 / fs)
    keep = np.abs(U) >= np.abs(U).max() * 10 ** (level_db / 20)
    keep[0] = False
    P = np.zeros((pts.shape[0], f.size), dtype=complex)
    for i in np.flatnonzero(keep):
        # both rayleigh() and the inverse FFT use exp(+j w t)
        P[:, i] = U[i] * rayleigh(transducer, pts, f[i], c, alpha0, y, max_pairs)
    p = np.fft.irfft(P, n, axis=1)
    return t, p


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


def hybrid_angular_spectrum(p0, medium, f, pad=2, taper=0.15):
    """Propagate a field through a heterogeneous medium, slice by slice.

    Parameters
    ----------
    p0 : complex array
        Field on the first depth plane of the medium grid: shape
        grid.shape[:-1] (x for a 2-D grid, (x, y) for a 3-D grid).
    medium : Medium
        2-D or 3-D medium; uses c, rho, alpha0 and y.
    f : float
        Frequency [Hz].
    pad : float
        Lateral zero padding factor against wrap-around.
    taper : float
        Fraction of the padded width, on each side, over which the field is
        smoothly absorbed.

    Returns
    -------
    complex array with the grid shape: the pressure, in the units of p0.
    """
    g = medium.grid
    if g.ndim < 2:
        raise ValueError("use angular_spectrum for 1-D problems")
    p0 = np.asarray(p0, dtype=complex)
    lat = g.shape[:-1]
    if p0.shape != lat:
        raise ValueError(f"p0 must have shape {lat}")
    nz = g.shape[-1]
    dz = g.spacing[-1]
    w = 2 * np.pi * f
    shape = tuple(int(np.ceil(pad * n)) for n in lat)
    off = tuple((m - n) // 2 for m, n in zip(shape, lat))
    inner = tuple(slice(o, o + n) for o, n in zip(off, lat))
    padw = [(o, m - n - o) for o, m, n in zip(off, shape, lat)]
    axes = tuple(range(len(lat)))

    def lateral_pad(a):
        return np.pad(a, padw, mode="edge")

    ks = [2 * np.pi * np.fft.fftfreq(m, g.spacing[i]) for i, m in enumerate(shape)]
    kt2 = sum(q * q for q in np.meshgrid(*ks, indexing="ij"))

    # smooth absorbing border (none with taper = 0, for periodic fields)
    win = np.ones(shape)
    for i, m in enumerate(shape if taper > 0 else ()):
        nt = max(int(taper * m), 1)
        prof = np.ones(m)
        ramp = np.sin(0.5 * np.pi * np.arange(nt) / nt) ** 2
        prof[:nt] = ramp
        prof[m - nt:] = ramp[::-1]
        sh = [1] * len(shape)
        sh[i] = m
        win = win * prof.reshape(sh)

    c = medium.c
    att = attenuation_np(medium.alpha0, f, medium.y)
    z_imp = medium.rho * medium.c

    out = np.empty(g.shape, dtype=complex)
    p = np.zeros(shape, dtype=complex)
    p[inner] = p0
    out[..., 0] = p0
    for i in range(nz - 1):
        ci = lateral_pad(c[..., i])
        ai = lateral_pad(att[..., i])
        kbar = w / np.mean(c[..., i])
        kz = np.sqrt((kbar * kbar - kt2).astype(complex))
        kz = np.where(kz.imag > 0, np.conj(kz), kz)
        p = np.fft.ifftn(np.fft.fftn(p, axes=axes) * np.exp(-1j * kz * dz), axes=axes)
        p = p * np.exp(-1j * (w / ci - kbar) * dz - ai * dz)
        z1 = lateral_pad(z_imp[..., i])
        z2 = lateral_pad(z_imp[..., i + 1])
        p = p * (2 * z2 / (z1 + z2)) * win
        out[..., i + 1] = p[inner]
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

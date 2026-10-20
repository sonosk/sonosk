"""Pulse-echo simulation and image formation.

The forward model is linear (first-order Born scattering) in a homogeneous
medium with sound speed c:

- the transmit wave is described by a virtual source: a plane wave with a
  steering angle, a diverging wave from a point behind the array, or a
  wave focused on a point in front of it;
- each receive element is a point receiver with the directivity of a
  rectangular element in a soft baffle, sinc(w sin(theta) / lambda) cos(theta);
- spreading is 1/R on the receive path and 1/R from a diverging source;
  attenuation is applied at the centre frequency along both paths.

The echo of every scatterer is a copy of the pulse delayed by its travel
time. Signals are kept in analytic (complex) form, so the envelope is the
modulus of the beamformed image.

Beamforming is delay-and-sum with a receive f-number. Images from several
transmits can be summed (coherent compounding).

References
----------
Montaldo G., Tanter M., Bercoff J., Benech N., Fink M. Coherent plane-wave
compounding for very high frame rate ultrasonography and transient
elastography. IEEE Trans. Ultrason. Ferroelectr. Freq. Control
56(3):489-506, 2009.

Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
Part I: theory & examples. Comput. Methods Programs Biomed. 218:106726, 2022.
"""

from dataclasses import dataclass

import numpy as np

from .medium import attenuation_np


@dataclass
class Pulse:
    """Gaussian-modulated pulse.

    f0 centre frequency [Hz], bandwidth fractional -6 dB bandwidth of the
    spectrum (0.6 means 60 %).
    """

    f0: float
    bandwidth: float = 0.6

    @property
    def sigma(self):
        # spectrum exp(-(2 pi df sigma)^2 / 2) is 0.5 at df = bandwidth f0 / 2
        return np.sqrt(2 * np.log(2)) / (np.pi * self.bandwidth * self.f0)

    @property
    def duration(self):
        """Support used for the simulation, 8 sigma [s]."""
        return 8 * self.sigma

    def analytic(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-0.5 * (t / self.sigma) ** 2 + 2j * np.pi * self.f0 * t)

    def __call__(self, t):
        return self.analytic(t).real


class Transmit:
    """Transmit wave described by a virtual source.

    Use the constructors plane(), diverging() and focused().
    """

    def __init__(self, kind, angle=0.0, source=None):
        self.kind = kind
        self.angle = float(angle)
        self.source = None if source is None else np.asarray(source, dtype=float)

    @classmethod
    def plane(cls, angle=0.0):
        """Plane wave with steering angle [rad] in the x-z plane."""
        return cls("plane", angle=angle)

    @classmethod
    def diverging(cls, source):
        """Diverging wave from a virtual point source behind the array (z < 0)."""
        return cls("diverging", source=source)

    @classmethod
    def focused(cls, point):
        """Wave focused on point (z > 0)."""
        return cls("focused", source=point)

    def element_delays(self, elements, c):
        """Firing delays [s] of the elements, the first element at 0."""
        e = np.asarray(elements, dtype=float)
        if self.kind == "plane":
            s = e @ self._dir()
            return (s - s.min()) / c
        d = np.linalg.norm(e - self.source, axis=1)
        if self.kind == "diverging":
            return (d - d.min()) / c
        return (d.max() - d) / c

    def _dir(self):
        return np.array([np.sin(self.angle), 0.0, np.cos(self.angle)])

    def travel(self, points, elements, c):
        """Arrival time [s] of the transmit wave and its path length [m]
        and amplitude at points, with time zero at the first firing."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        e = np.asarray(elements, dtype=float)
        if self.kind == "plane":
            n = self._dir()
            s0 = (e @ n).min()
            path = p @ n - s0
            # insonified strip: trace the point back to the array along n
            x0 = p[:, 0] - p[:, 2] * np.tan(self.angle)
            amp = ((x0 >= e[:, 0].min()) & (x0 <= e[:, 0].max())).astype(float)
            return path / c, path, amp
        d_e = np.linalg.norm(e - self.source, axis=1)
        r = np.linalg.norm(p - self.source, axis=1)
        if self.kind == "diverging":
            path = r - d_e.min()
            amp = d_e.min() / np.maximum(r, 1e-9)
            return path / c, path, amp
        beyond = p[:, 2] > self.source[2]
        path = d_e.max() + np.where(beyond, r, -r)
        amp = np.ones(p.shape[0])
        return path / c, path, amp


def _directivity(sin_t, cos_t, width, f0, c):
    return np.sinc(width * sin_t * f0 / c) * cos_t


def _element_frame(array):
    e = array.centers
    n = array.element_normals
    return e, n


def simulate(array, transmit, pos, amp, pulse, c=1540.0, fs=None, alpha0=0.0,
             depth=None, oversample=4, element_chunk=16):
    """Simulate the received signals of one transmit.

    Every echo is placed as an impulse on a time axis oversample times
    finer than fs, split between the two nearest samples, then convolved
    with the pulse and decimated to fs.

    Parameters
    ----------
    array : Transducer
        Linear, phased, convex or matrix array.
    transmit : Transmit
    pos, amp : arrays
        Scatterer positions (N, 3) [m] and amplitudes (N,).
    pulse : Pulse
        Two-way pulse (the electrical excitation convolved with both
        transducer responses).
    c : float
        Sound speed [m/s].
    fs : float, optional
        Sampling frequency of the output [Hz], default 8 f0.
    alpha0 : float
        Attenuation in dB/(cm MHz), applied at f0.
    depth : float, optional
        Record at least down to this depth [m].
    oversample : int
        Ratio of the internal sampling rate to fs.

    Returns
    -------
    rf : complex array (n_elements, n_samples)
        Analytic received signals; rf.real is the RF signal.
    fs : float
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    amp = np.asarray(amp, dtype=float).ravel()
    fs = 8 * pulse.f0 if fs is None else float(fs)
    fh = fs * oversample
    e, n = _element_frame(array)
    a = float(attenuation_np(alpha0, pulse.f0, 1.0))

    t_tx, path_tx, amp_tx = transmit.travel(pos, e, c)
    em = e.mean(axis=0)
    rmax = np.linalg.norm(pos - em, axis=1).max() + np.linalg.norm(e - em, axis=1).max()
    t_end = t_tx.max() + rmax / c + pulse.duration
    if depth is not None:
        t_end = max(t_end, 2 * depth / c + pulse.duration)
    ns = int(np.ceil(t_end * fs)) + 1
    nh = ns * oversample

    base = amp * amp_tx * np.exp(-a * path_tx)
    imp = np.zeros((array.n_elements, nh))
    for i0 in range(0, array.n_elements, element_chunk):
        ee = e[i0:i0 + element_chunk]
        nn = n[i0:i0 + element_chunk]
        d = pos[None, :, :] - ee[:, None, :]
        r = np.linalg.norm(d, axis=2)
        cos_t = np.einsum("eij,ej->ei", d, nn) / r
        sin_t = np.sqrt(np.clip(1 - cos_t ** 2, 0, 1))
        g = base * _directivity(sin_t, cos_t, array.width, pulse.f0, c) * np.exp(-a * r) / r
        g = np.where(cos_t > 0, g, 0.0)
        s = (t_tx[None, :] + r / c) * fh
        k0 = np.floor(s).astype(int)
        fr = s - k0
        for j in range(ee.shape[0]):
            ok = (k0[j] >= 0) & (k0[j] < nh - 1)
            k = k0[j][ok]
            w = g[j][ok]
            imp[i0 + j] = (np.bincount(k, w * (1 - fr[j][ok]), nh)
                           + np.bincount(k + 1, w * fr[j][ok], nh))

    half = int(np.ceil(0.5 * pulse.duration * fh))
    kern = pulse.analytic(np.arange(-half, half + 1) / fh)
    nfft = 1 << int(np.ceil(np.log2(nh + kern.size)))
    spec = np.fft.fft(imp, nfft, axis=1) * np.fft.fft(kern, nfft)[None, :]
    full = np.fft.ifft(spec, axis=1)[:, half:half + nh]
    return full[:, ::oversample].copy(), fs


def beamform(rf, fs, array, transmit, x, z, c=1540.0, f_number=1.5, window="hann", y=0.0):
    """Delay-and-sum image on the grid x by z (2-D, at elevation y).

    Returns the complex image of shape (len(z), len(x)); its modulus is the
    envelope.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    pts = np.stack([xx.ravel(), np.full(xx.size, y), zz.ravel()], 1)
    e, n = _element_frame(array)
    t_tx, _, _ = transmit.travel(pts, e, c)

    img = np.zeros(pts.shape[0], dtype=complex)
    ns = rf.shape[1]
    for i in range(array.n_elements):
        d = pts - e[i]
        r = np.linalg.norm(d, axis=1)
        ax = d @ n[i]
        lat = np.sqrt(np.maximum(r * r - ax * ax, 0))
        u = np.where(ax > 0, lat * f_number / np.maximum(ax, 1e-12), np.inf)
        if window == "hann":
            w = np.where(u <= 0.5, np.cos(np.pi * u) ** 2, 0.0)
        else:
            w = (u <= 0.5).astype(float)
        if not w.any():
            continue
        s = (t_tx + r / c) * fs
        k0 = np.floor(s).astype(int)
        fr = s - k0
        ok = (k0 >= 0) & (k0 < ns - 1) & (w > 0)
        k = k0[ok]
        v = rf[i, k] * (1 - fr[ok]) + rf[i, k + 1] * fr[ok]
        img[ok] += w[ok] * v
    return img.reshape(zz.shape)


def compound(rfs, fs, array, transmits, x, z, c=1540.0, **kw):
    """Sum of beamformed images over several transmits."""
    out = 0
    for rf, tx in zip(rfs, transmits):
        out = out + beamform(rf, fs, array, tx, x, z, c, **kw)
    return out


def log_compress(image, dynamic_range=60.0):
    """Envelope in dB relative to its maximum, clipped at -dynamic_range."""
    env = np.abs(image)
    db = 20 * np.log10(np.maximum(env / env.max(), 1e-12))
    return np.maximum(db, -dynamic_range)


def plane_wave_angles(n, span):
    """n steering angles [rad] spread evenly over +-span/2."""
    if n == 1:
        return np.array([0.0])
    return np.linspace(-span / 2, span / 2, n)

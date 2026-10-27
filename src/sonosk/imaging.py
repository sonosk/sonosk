"""Pulse-echo simulation and image formation.

The fast forward model is linear (first-order Born scattering):

- the transmit wave is described by a virtual source: a plane wave with a
  steering angle, a diverging wave from a point behind the array, or a
  wave focused on a point in front of it;
- each receive element is a point receiver with the directivity of a
  rectangular element in a soft baffle, sinc(w sin(theta) / lambda) cos(theta);
- spreading is 1/R on the receive path and 1/R from a diverging source;
- in a homogeneous medium the echo of a scatterer is delayed by the path
  length over c and attenuated with alpha0 f^y along both paths. With a
  heterogeneous medium the delay and the attenuation are integrals along
  the straight transmit and receive paths (sonosk.paths), which gives
  phase aberration, acoustic shadowing and posterior enhancement;
- frequency-dependent attenuation changes the spectrum of each echo, which
  lowers its centre frequency with depth. Echoes are grouped by their total
  attenuation and each group is convolved with its own filtered pulse.

The echo of every scatterer is a copy of the pulse delayed by its travel
time. Signals are kept in analytic (complex) form, so the envelope is the
modulus of the beamformed image.

Beamforming is delay-and-sum with a receive f-number. It can use the
nominal speed of the scanner or the travel times through a known medium
(sound-speed-aware beamforming). Images from several transmits can be
summed (coherent compounding).

References
----------
Montaldo G., Tanter M., Bercoff J., Benech N., Fink M. Coherent plane-wave
compounding for very high frame rate ultrasonography and transient
elastography. IEEE Trans. Ultrason. Ferroelectr. Freq. Control
56(3):489-506, 2009.

Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
Part I: theory & examples. Comput. Methods Programs Biomed. 218:106726, 2022.

Perrot V., Polichetti M., Varray F., Garcia D. So you think you can DAS? A
viewpoint on delay-and-sum beamforming. Ultrasonics 111:106309, 2021.
"""

from dataclasses import dataclass

import numpy as np

from .medium import attenuation_np


# --------------------------------------------------------------------------
# pulses


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

    @staticmethod
    def from_stack(stack, f_max=None, n=4096, cycles=1, f_drive=None, r_source=50.0):
        """Two-way pulse of a transducer stack (sonosk.stack.Stack).

        The electrical drive is a burst of the given number of cycles at
        f_drive (default: the half-wave frequency of the plate). The
        pulse-echo transfer function of the stack is applied twice, as for
        transmit and receive.
        """
        f_half = stack.piezo.f_half_wave
        f_drive = f_half if f_drive is None else float(f_drive)
        f_max = 4 * f_half if f_max is None else float(f_max)
        f = np.linspace(0, f_max, n)
        f[0] = f[1] * 1e-3
        h = stack.transmit(f, r_source) ** 2
        # spectrum of a rectangular burst of `cycles` cycles
        T = cycles / f_drive
        w = 2 * np.pi * f
        wd = 2 * np.pi * f_drive
        burst = (np.exp(-1j * (w - wd) * T) - 1) / (-1j * (w - wd) + 1e-30)
        spec = h * burst
        spec[0] = 0.0
        return SampledPulse.from_spectrum(f, spec)


class SampledPulse:
    """Pulse given by samples of its analytic signal.

    Use SampledPulse.from_spectrum or Pulse.from_stack to build one. The
    time axis is shifted so that the envelope peak is at t = 0.
    """

    def __init__(self, t, signal):
        t = np.asarray(t, dtype=float)
        a = np.asarray(signal, dtype=complex)
        k = int(np.argmax(np.abs(a)))
        self.t = t - t[k]
        self.a = a / np.abs(a[k])
        env = np.abs(self.a)
        keep = env > 1e-2
        lo, hi = self.t[keep].min(), self.t[keep].max()
        self._dur = 2 * max(-lo, hi)
        spec = np.fft.fft(self.a)
        fr = np.fft.fftfreq(self.a.size, self.t[1] - self.t[0])
        pw = np.abs(spec) ** 2
        self.f0 = float(np.sum(fr * pw) / np.sum(pw))

    @classmethod
    def from_spectrum(cls, f, spec, fs=None):
        """Pulse from a one-sided spectrum spec at frequencies f (uniform)."""
        f = np.asarray(f, dtype=float)
        df = f[1] - f[0]
        n = 2 * (f.size - 1)
        fs = n * df if fs is None else fs
        full = np.zeros(n, dtype=complex)
        full[:f.size] = spec
        a = np.fft.ifft(full) * 2
        t = np.arange(n) / (n * df)
        a = np.roll(a, n // 2)
        t = t - t[n // 2]
        return cls(t, a)

    @property
    def duration(self):
        return self._dur

    def analytic(self, t):
        t = np.asarray(t, dtype=float)
        re = np.interp(t, self.t, self.a.real, left=0.0, right=0.0)
        im = np.interp(t, self.t, self.a.imag, left=0.0, right=0.0)
        return re + 1j * im

    def __call__(self, t):
        return self.analytic(t).real


# --------------------------------------------------------------------------
# transmits


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
        """Diverging wave from a virtual point source behind the array (z < 0).

        For a convex array, a source at the centre of curvature,
        (0, 0, -radius), gives a sector-shaped wave.
        """
        return cls("diverging", source=source)

    @classmethod
    def focused(cls, point):
        """Wave focused on point (z > 0)."""
        return cls("focused", source=point)

    def __repr__(self):
        if self.kind == "plane":
            return f"Transmit.plane({np.rad2deg(self.angle):.2f} deg)"
        return f"Transmit.{self.kind}({self.source.tolist()})"

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

    def _span(self, p, e):
        """1 where the ray from the virtual source through p crosses the
        aperture, else 0."""
        ang_e = np.arctan2(e[:, 0] - self.source[0], e[:, 2] - self.source[2])
        ang_p = np.arctan2(p[:, 0] - self.source[0], p[:, 2] - self.source[2])
        return ((ang_p >= ang_e.min()) & (ang_p <= ang_e.max())).astype(float)

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
            amp = d_e.min() / np.maximum(r, 1e-9) * self._span(p, e)
            return path / c, path, amp
        beyond = p[:, 2] > self.source[2]
        path = d_e.max() + np.where(beyond, r, -r)
        amp = np.ones(p.shape[0])
        return path / c, path, amp

    def entry(self, points, elements, c):
        """Where the transmit ray through each point crosses the plane
        z = 0, and the time [s] the wave is there.

        Used with a heterogeneous medium: the wave is timed with the
        nominal speed c up to z = 0 and followed through the medium after.
        """
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        e = np.asarray(elements, dtype=float)
        q = p.copy()
        q[:, 2] = 0.0
        if self.kind == "plane":
            n = self._dir()
            s0 = (e @ n).min()
            q[:, 0] = p[:, 0] - p[:, 2] * np.tan(self.angle)
            t0 = (q @ n - s0) / c
            return q, t0
        vs = self.source
        d_e = np.linalg.norm(e - vs, axis=1)
        dz = p[:, 2] - vs[2]
        dz = np.where(np.abs(dz) < 1e-12, 1e-12, dz)
        lam = (0.0 - vs[2]) / dz
        q = vs + lam[:, None] * (p - vs)
        q[:, 2] = 0.0
        if self.kind == "diverging":
            t0 = (np.linalg.norm(q - vs, axis=1) - d_e.min()) / c
        else:
            t0 = (d_e.max() - np.linalg.norm(q - vs, axis=1)) / c
        return q, t0


# --------------------------------------------------------------------------
# forward model


def _directivity(sin_t, cos_t, width, f0, c):
    return np.sinc(width * sin_t * f0 / c) * cos_t


def _element_frame(array):
    return array.centers, array.element_normals


def _shaped_kernels(pulse, fh, A_bins, f0, y):
    """Pulse kernels filtered by exp(-A ((f/f0)^y - 1)) for each A in A_bins."""
    half = int(np.ceil(0.75 * pulse.duration * fh))
    t = np.arange(-half, half + 1) / fh
    base = pulse.analytic(t)
    n = 1 << int(np.ceil(np.log2(4 * t.size)))
    centred = np.zeros(n, dtype=complex)      # t = 0 at index 0
    centred[:half + 1] = base[half:]
    centred[-half:] = base[:half]
    spec = np.fft.fft(centred)
    f = np.fft.fftfreq(n, 1 / fh)
    fpos = np.clip(f, 0, None) / f0
    out = []
    for A in A_bins:
        h = np.where(f > 0, np.exp(-A * (fpos ** y - 1.0)), 0.0)
        k = np.fft.ifft(spec * h)
        k = np.concatenate([k[-half:], k[:half + 1]])
        out.append(k)
    return np.array(out), half


def simulate(array, transmit, pos, amp, pulse, c=1540.0, fs=None, alpha0=0.0,
             depth=None, oversample=4, element_chunk=16, medium=None, f_dependent=True,
             bin_step=0.5, element_step=4, ray_step=None, interfaces=False):
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
    pulse : Pulse or SampledPulse
        Two-way pulse (the electrical excitation convolved with both
        transducer responses).
    c : float
        Sound speed [m/s] of a homogeneous medium, and the nominal speed
        used to fire the elements when a medium is given.
    fs : float, optional
        Sampling frequency of the output [Hz], default 8 f0.
    alpha0 : float
        Attenuation of a homogeneous medium in dB/(cm MHz).
    depth : float, optional
        Record at least down to this depth [m].
    oversample : int
        Ratio of the internal sampling rate to fs.
    medium : Medium, optional
        2-D heterogeneous medium. Delays and attenuation are then
        integrated along the straight paths (c and alpha0 are not used for
        propagation).
    f_dependent : bool
        Apply the frequency dependence of the attenuation to the spectrum
        of each echo (centre-frequency downshift).
    bin_step : float
        Spacing [Np] of the attenuation groups used for the spectral
        shaping.
    element_step : int
        With a medium, receive paths are integrated for every
        element_step-th element and interpolated in between.
    ray_step : float, optional
        Integration step along the rays [m].
    interfaces : bool
        With a medium, also apply the transmission loss 1 - R^2 of every
        impedance step crossed on the way in and out. Echoes from the
        interfaces themselves come from phantom.interface_scatterers.

    Returns
    -------
    rf : complex array (n_elements, n_samples)
        Analytic received signals; rf.real is the RF signal.
    fs : float
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    amp = np.asarray(amp, dtype=float).ravel()
    f0 = pulse.f0
    fs = 8 * f0 if fs is None else float(fs)
    fh = fs * oversample
    e, n = _element_frame(array)
    n_el = e.shape[0]

    if medium is None:
        a = float(attenuation_np(alpha0, f0, 1.0))
        t_tx, path_tx, amp_tx = transmit.travel(pos, e, c)
        A_tx = a * path_tx
        y = 1.0
        rays = None
    else:
        from .paths import ArrayPaths, RayModel, transmit_paths
        rays = RayModel(medium, f0, c_nominal=c, step=ray_step, interfaces=interfaces)
        _, _, amp_tx = transmit.travel(pos, e, c)
        t_tx, A_tx, *L_tx = transmit_paths(rays, transmit, pos, e)
        y = float(np.median(medium.y))
        recv = ArrayPaths(rays, e, n, None, element_step)

    em = e.mean(axis=0)
    rmax = np.linalg.norm(pos - em, axis=1).max() + np.linalg.norm(e - em, axis=1).max()
    t_end = t_tx.max() + 1.2 * rmax / c + pulse.duration
    if depth is not None:
        t_end = max(t_end, 2 * depth / c + pulse.duration)
    ns = int(np.ceil(t_end * fs)) + 1
    nh = ns * oversample

    base = amp * amp_tx
    if medium is not None and interfaces:
        base = base * np.exp(-L_tx[0])
    keep = base != 0
    pos, base, t_tx, A_tx = pos[keep], base[keep], t_tx[keep], A_tx[keep]

    if medium is not None:
        excess, A_rx_all, *L_rx = recv(pos)
        if interfaces:
            loss_rx = np.exp(-L_rx[0])
    acc = np.zeros((n_el, ns), dtype=complex)
    if base.size == 0:
        return acc, fs

    # receive geometry and total attenuation, per element chunk
    chunks = []
    A_max = 0.0
    for i0 in range(0, n_el, element_chunk):
        ee = e[i0:i0 + element_chunk]
        nn = n[i0:i0 + element_chunk]
        d = pos[None, :, :] - ee[:, None, :]
        r = np.linalg.norm(d, axis=2)
        cos_t = np.einsum("eij,ej->ei", d, nn) / r
        sin_t = np.sqrt(np.clip(1 - cos_t ** 2, 0, 1))
        g = base * _directivity(sin_t, cos_t, array.width, f0, c) / r
        g = np.where(cos_t > 0, g, 0.0)
        if medium is not None and interfaces:
            g = g * loss_rx[i0:i0 + element_chunk]
        if medium is None:
            A = A_tx[None, :] + a * r
            t = t_tx[None, :] + r / c
        else:
            A = A_tx[None, :] + A_rx_all[i0:i0 + element_chunk]
            t = t_tx[None, :] + r / c + excess[i0:i0 + element_chunk]
        chunks.append((i0, g, A, t))
        A_max = max(A_max, float(A.max()))

    if f_dependent and A_max > 0:
        nb = int(np.ceil(A_max / bin_step)) + 1
        A_bins = np.linspace(0, A_max, nb) if nb > 1 else np.array([0.0])
    else:
        A_bins = np.array([0.0])
    kernels, half = _shaped_kernels(pulse, fh, A_bins, f0, y)
    nfft = 1 << int(np.ceil(np.log2(nh + kernels.shape[1])))
    K = np.fft.fft(kernels, nfft, axis=1)

    for i0, g, A, t in chunks:
        w = g * np.exp(-A)
        s = t * fh
        k0 = np.floor(s).astype(int)
        fr = s - k0
        ok = (k0 >= 0) & (k0 < nh - 1)
        ne = g.shape[0]
        rows = np.broadcast_to(np.arange(ne)[:, None], k0.shape)
        flat = (rows * nh + k0)[ok]
        val = w[ok]
        frv = fr[ok]
        if A_bins.size > 1:
            u = A[ok] / (A_bins[1] - A_bins[0])
            b0 = np.clip(np.floor(u).astype(int), 0, A_bins.size - 2)
            wb = np.clip(u - b0, 0, 1)
        else:
            b0 = np.zeros(val.size, int)
            wb = np.zeros(val.size)
        order = np.argsort(b0, kind="stable")
        flat, val, frv, b0, wb = flat[order], val[order], frv[order], b0[order], wb[order]
        edges = np.searchsorted(b0, np.arange(A_bins.size + 1))
        out = np.zeros((ne, nfft), dtype=complex)
        size = ne * nh
        for b in range(A_bins.size):
            parts = []
            lower = slice(edges[b], edges[b + 1])       # entries whose lower bin is b
            if lower.stop > lower.start:
                parts.append((lower, 1 - wb[lower]))
            if b > 0:
                upper = slice(edges[b - 1], edges[b])   # entries whose upper bin is b
                if upper.stop > upper.start:
                    parts.append((upper, wb[upper]))
            if not parts:
                continue
            imp = np.zeros(size)
            for sl, ww in parts:
                v = val[sl] * ww
                imp += (np.bincount(flat[sl], v * (1 - frv[sl]), size)
                        + np.bincount(flat[sl] + 1, v * frv[sl], size))
            out += np.fft.fft(imp.reshape(ne, nh), nfft, axis=1) * K[b][None, :]
        full = np.fft.ifft(out, axis=1)[:, half:half + nh]
        acc[i0:i0 + ne] = full[:, ::oversample][:, :ns]
    return acc, fs


def add_noise(rf, snr_db, rng=None):
    """Add complex white noise whose RMS is snr_db below the largest
    sample of rf (electronic noise of the receiver)."""
    rng = np.random.default_rng(rng)
    sigma = np.abs(rf).max() * 10 ** (-snr_db / 20) / np.sqrt(2)
    noise = rng.normal(size=rf.shape) + 1j * rng.normal(size=rf.shape)
    return rf + sigma * noise


# --------------------------------------------------------------------------
# beamforming


def _pixels(x, z, y):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    pts = np.stack([xx.ravel(), np.full(xx.size, y), zz.ravel()], 1)
    return pts, zz.shape


def receive_excess(array, x, z, medium, c=1540.0, f0=None, y=0.0, element_step=4,
                   ray_step=None):
    """Extra receive delays through a medium, compared with the nominal
    speed c, for every element at every pixel (a paths.PathTable). Pass it
    to beamform() to reuse it for several transmits."""
    from .paths import ArrayPaths, RayModel
    pts, _ = _pixels(x, z, y)
    e, n = _element_frame(array)
    rays = RayModel(medium, 5e6 if f0 is None else f0, c_nominal=c, step=ray_step)
    return ArrayPaths(rays, e, n, None, element_step).table(pts)


def beamform(rf, fs, array, transmit, x, z, c=1540.0, f_number=1.5, window="hann", y=0.0,
             medium=None, element_step=4, ray_step=None, rx_excess=None):
    """Delay-and-sum image on the grid x by z (2-D, at elevation y).

    With a medium, the delays are the travel times through it along
    straight rays (sound-speed-aware beamforming); otherwise they use the
    nominal speed c. rx_excess from receive_excess() can be given to avoid
    recomputing the receive delays.

    Returns the complex image of shape (len(z), len(x)); its modulus is the
    envelope.
    """
    pts, shape = _pixels(x, z, y)
    e, n = _element_frame(array)
    if medium is None:
        t_tx, _, _ = transmit.travel(pts, e, c)
        excess = None
    else:
        from .paths import RayModel, transmit_paths
        rays = RayModel(medium, fs / 8, c_nominal=c, step=ray_step)
        t_tx, _ = transmit_paths(rays, transmit, pts, e)
        excess = rx_excess
        if excess is None:
            excess = receive_excess(array, x, z, medium, c, fs / 8, y, element_step, ray_step)

    img = np.zeros(pts.shape[0], dtype=complex)
    ns = rf.shape[1]
    for i in range(array.n_elements):
        d = pts - e[i]
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        ax = d @ n[i]
        lat2 = np.maximum(r * r - ax * ax, 0)
        u = np.where(ax > 0, np.sqrt(lat2) * f_number / np.maximum(ax, 1e-12), np.inf)
        inside = u <= 0.5
        w = np.zeros(u.shape)
        if window == "hann":
            w[inside] = np.cos(np.pi * u[inside]) ** 2
        else:
            w[inside] = 1.0
        sel = np.flatnonzero(w > 0)
        if sel.size == 0:
            continue
        tr = r[sel] / c if excess is None else r[sel] / c + excess.excess(i)[sel]
        s = (t_tx[sel] + tr) * fs
        k0 = np.floor(s).astype(int)
        fr = s - k0
        ok = (k0 >= 0) & (k0 < ns - 1)
        sel, k, fr = sel[ok], k0[ok], fr[ok]
        img[sel] += w[sel] * (rf[i, k] * (1 - fr) + rf[i, k + 1] * fr)
    return img.reshape(shape)


def compound(rfs, fs, array, transmits, x, z, c=1540.0, medium=None, **kw):
    """Sum of beamformed images over several transmits."""
    if medium is not None and kw.get("rx_excess") is None:
        kw["rx_excess"] = receive_excess(array, x, z, medium, c, fs / 8, kw.get("y", 0.0),
                                         kw.get("element_step", 4), kw.get("ray_step"))
    out = 0
    for rf, tx in zip(rfs, transmits):
        out = out + beamform(rf, fs, array, tx, x, z, c, medium=medium, **kw)
    return out


# --------------------------------------------------------------------------
# display


def log_compress(image, dynamic_range=60.0, reference=None):
    """Envelope in dB relative to its maximum (or to reference), clipped at
    -dynamic_range."""
    env = np.abs(image)
    ref = env.max() if reference is None else reference
    db = 20 * np.log10(np.maximum(env / ref, 1e-12))
    return np.maximum(db, -dynamic_range)


def tgc(image, z, alpha0, f0, gain_db=0.0):
    """Time-gain compensation.

    Multiplies each row of the image (depth z [m] along the first axis) by
    exp(2 alpha z), which undoes the mean round-trip attenuation of a
    tissue with alpha0 [dB/(cm MHz)] at f0 [Hz], plus a constant gain.
    """
    a = float(attenuation_np(alpha0, f0, 1.0))
    g = np.exp(2 * a * np.asarray(z, dtype=float)) * 10 ** (gain_db / 20)
    return image * g[:, None]


def normalize_depth(image, n_smooth=31, floor=1e-12):
    """Equalize the mean brightness with depth (an automatic TGC).

    Each row is divided by a smoothed median of the envelope over depth.
    """
    env = np.abs(image)
    m = np.median(env, axis=1)
    k = np.hanning(n_smooth + 2)[1:-1]
    k /= k.sum()
    pad = n_smooth // 2
    ms = np.convolve(np.pad(m, pad, mode="edge"), k, mode="valid")
    return image / np.maximum(ms, floor * ms.max())[:, None]


def sector_mask(x, z, apex, half_angle, r_min=0.0, r_max=np.inf):
    """Boolean mask (len(z), len(x)) of a sector with its apex at
    (x, z) = apex, half opening angle [rad] around the depth axis."""
    zz, xx = np.meshgrid(np.asarray(z), np.asarray(x), indexing="ij")
    dx = xx - apex[0]
    dz = zz - apex[1]
    r = np.hypot(dx, dz)
    th = np.arctan2(dx, dz)
    return (np.abs(th) <= half_angle) & (r >= r_min) & (r <= r_max)


def scan_convert(image, r, theta, x, z, apex=(0.0, 0.0)):
    """Resample an image on a polar grid (rows r [m], columns theta [rad])
    onto the Cartesian grid x by z. Points outside the sector are NaN."""
    image = np.asarray(image)
    zz, xx = np.meshgrid(np.asarray(z), np.asarray(x), indexing="ij")
    dx = xx - apex[0]
    dz = zz - apex[1]
    rr = np.hypot(dx, dz)
    th = np.arctan2(dx, dz)
    fr = (rr - r[0]) / (r[1] - r[0])
    ft = (th - theta[0]) / (theta[1] - theta[0])
    inside = (fr >= 0) & (fr <= len(r) - 1) & (ft >= 0) & (ft <= len(theta) - 1)
    fr = np.clip(fr, 0, len(r) - 1.000001)
    ft = np.clip(ft, 0, len(theta) - 1.000001)
    i = fr.astype(int)
    j = ft.astype(int)
    wr = fr - i
    wt = ft - j
    v = (image[i, j] * (1 - wr) * (1 - wt) + image[i + 1, j] * wr * (1 - wt)
         + image[i, j + 1] * (1 - wr) * wt + image[i + 1, j + 1] * wr * wt)
    out = np.where(inside, v, np.nan)
    return out


def plane_wave_angles(n, span):
    """n steering angles [rad] spread evenly over +-span/2."""
    if n == 1:
        return np.array([0.0])
    return np.linspace(-span / 2, span / 2, n)

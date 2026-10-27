"""Full-wave simulation with a k-space pseudospectral method.

The solver integrates the coupled first-order equations of nonlinear
acoustics in a heterogeneous, absorbing fluid,

    du/dt   = -grad(p) / rho0
    drho/dt = -(2 rho + rho0) div(u) - u . grad(rho0)
    p       = c0^2 (rho + d . grad(rho0) + B/(2A) rho^2 / rho0 - L rho),

where L is the loss operator that gives power-law absorption
alpha(f) = alpha0 f^y and the dispersion that goes with it through the
Kramers-Kronig relations,

    L = tau d/dt (-nabla^2)^(y/2 - 1) + eta (-nabla^2)^((y + 1)/2 - 1),
    tau = -2 alpha0 c0^(y - 1),  eta = 2 alpha0 c0^y tan(pi y / 2).

The term u . grad(rho0) is second order and dropped, as in k-Wave. The
divergence of u, known at half steps, is extrapolated to the time of the
pressure before it enters the absorption term; this keeps the absorption
and dispersion second-order accurate in time (with the plain half-step lag
the phase speed drifts by about 1 m/s at 5 MHz and a CFL number of 0.3).
Spatial derivatives are computed with FFTs on a staggered grid and
corrected by the k-space operator sinc(c_ref k dt / 2), which makes the
scheme exact for a homogeneous medium and lets it run at two to three
points per wavelength. A perfectly matched layer surrounds the grid.

By default one power-law exponent y is used for the whole grid, and each
tissue keeps its own attenuation at a reference frequency f_ref: alpha0
is rescaled so that alpha0' f_ref^y = alpha0_tissue f_ref^y_tissue. Away
from f_ref the attenuation of a tissue then changes as (f/f_ref)^(y -
y_tissue); exponent_error() gives this error. With y="tissue" every
distinct exponent gets its own pair of fractional Laplacian terms, so each
tissue follows its own power law at every frequency, at the cost of two
inverse FFTs per extra exponent and time step.

The grid of the medium is the region of interest; the absorbing layer is
added outside it. All units are SI.

References
----------
Tabei M., Mast T.D., Waag R.C. A k-space method for coupled first-order
acoustic propagation equations. J. Acoust. Soc. Am. 111(1):53-63, 2002.

Treeby B.E., Cox B.T. Modeling power law absorption and dispersion for
acoustic propagation using the fractional Laplacian. J. Acoust. Soc. Am.
127(5):2741-2748, 2010.

Treeby B.E., Jaros J., Rendell A.P., Cox B.T. Modeling nonlinear
ultrasound propagation in heterogeneous media with power law absorption
using a k-space pseudospectral method. J. Acoust. Soc. Am.
131(6):4324-4336, 2012.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from .medium import DB_TO_NP

try:  # scipy.fft is faster and runs on several cores; numpy.fft is the fallback
    import scipy.fft as _fft

    def _rfftn(a, axes):
        return _fft.rfftn(a, axes=axes, workers=-1)

    def _irfftn(a, s, axes):
        return _fft.irfftn(a, s=s, axes=axes, workers=-1)

    _next_fast = _fft.next_fast_len
except ImportError:  # pragma: no cover - depends on the installation
    def _rfftn(a, axes):
        return np.fft.rfftn(a, axes=axes)

    def _irfftn(a, s, axes):
        return np.fft.irfftn(a, s=s, axes=axes)

    def _next_fast(n, real=True):
        while True:
            m = n
            for p in (2, 3, 5):
                while m % p == 0:
                    m //= p
            if m == 1:
                return n
            n += 1


# --------------------------------------------------------------------------
# sources and sensors


def tone_burst(f, cycles, envelope="hann"):
    """Tone burst s(t) of the given number of cycles, starting at t = 0.

    envelope is "hann", "gauss" or "rect". Returns a function of t.
    """
    duration = cycles / f

    def s(t):
        t = np.asarray(t, dtype=float)
        x = t / duration
        inside = (x >= 0) & (x <= 1)
        if envelope == "hann":
            w = np.sin(np.pi * x) ** 2
        elif envelope == "gauss":
            w = np.exp(-0.5 * ((x - 0.5) / 0.15) ** 2)
        elif envelope == "rect":
            w = np.ones_like(x)
        else:
            raise ValueError("envelope must be hann, gauss or rect")
        return np.where(inside, w * np.sin(2 * np.pi * f * t), 0.0)

    s.duration = duration
    return s


def continuous_wave(f, ramp_cycles=3):
    """Sine wave switched on with a smooth ramp of ramp_cycles cycles."""
    t_ramp = ramp_cycles / f

    def s(t):
        t = np.asarray(t, dtype=float)
        w = np.where(t < t_ramp, np.sin(0.5 * np.pi * np.clip(t, 0, None) / t_ramp) ** 2, 1.0)
        return np.where(t >= 0, w * np.sin(2 * np.pi * f * t), 0.0)

    s.duration = np.inf
    return s


def gaussian_pulse(pulse, t0=None):
    """Real signal of an imaging.Pulse, centred at t0 (default 4 sigma)."""
    t0 = 4 * pulse.sigma if t0 is None else float(t0)

    def s(t):
        return pulse(np.asarray(t, dtype=float) - t0)

    s.duration = 2 * t0
    s.t0 = t0
    return s


@dataclass
class Source:
    """Pressure or velocity source on grid points.

    Parameters
    ----------
    index : int array (n, ndim)
        Grid indices of the source points.
    signal : callable
        signal(t) gives the source waveform; every point uses it with its
        own delay and weight.
    delay : array (n,), optional
        Delay of each point [s].
    weight : array (n,), optional
        Amplitude of each point. For a pressure source the weight is in Pa.
    kind : str
        "p" for an additive pressure (mass) source, "u" for an additive
        velocity source along the depth axis.
    """

    index: np.ndarray
    signal: object
    delay: np.ndarray = None
    weight: np.ndarray = None
    kind: str = "p"

    def __post_init__(self):
        self.index = np.atleast_2d(np.asarray(self.index, dtype=int))
        n = self.index.shape[0]
        self.delay = np.zeros(n) if self.delay is None else np.broadcast_to(
            np.asarray(self.delay, dtype=float), (n,)).copy()
        self.weight = np.ones(n) if self.weight is None else np.broadcast_to(
            np.asarray(self.weight, dtype=float), (n,)).copy()
        if self.kind not in ("p", "u"):
            raise ValueError("kind must be 'p' or 'u'")

    def values(self, t):
        return self.weight * self.signal(t - self.delay)


def grid_index(grid, points):
    """Nearest grid indices (n, ndim) of points given as (n, 3) [m].

    In 2-D the y coordinate is ignored, in 1-D only z is used.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if grid.ndim == 1:
        c = p[:, 2:3]
    elif grid.ndim == 2:
        c = p[:, [0, 2]]
    else:
        c = p
    idx = np.rint((c - grid.origin) / grid.spacing).astype(int)
    return idx


def _element_points(transducer, grid, y_tol=None):
    """Grid points covered by each element of a transducer.

    Returns (index (n, ndim), element number of each index). In 2-D only
    patches near the x-z plane are used.
    """
    pos = transducer.pos
    elem = transducer.elem
    if grid.ndim == 2:
        tol = (0.5 * transducer.height if transducer.height else 0.0) + 1e-9
        tol = tol if y_tol is None else y_tol
        keep = np.abs(pos[:, 1]) <= tol
        pos, elem = pos[keep], elem[keep]
    idx = grid_index(grid, pos)
    inside = np.all((idx >= 0) & (idx < np.array(grid.shape)), axis=1)
    idx, elem = idx[inside], elem[inside]
    key = np.concatenate([elem[:, None], idx], axis=1)
    key = np.unique(key, axis=0)
    return key[:, 1:], key[:, 0]


def array_source(transducer, grid, signal, delays=None, apodization=None, kind="p",
                 amplitude=1.0):
    """Source made of the elements of a transducer.

    Every grid point under an element gets the element's delay and
    apodization. delays default to transducer.delays and apodization to
    transducer.apod.
    """
    idx, el = _element_points(transducer, grid)
    if idx.shape[0] == 0:
        raise ValueError("no element lies on the grid; the grid must contain the face")
    d = transducer.delays if delays is None else np.asarray(delays, dtype=float)
    a = transducer.apod if apodization is None else np.asarray(apodization, dtype=float)
    return Source(idx, signal, delay=d[el], weight=amplitude * a[el], kind=kind)


@dataclass
class Sensor:
    """Points where a time series is recorded.

    quantity is "p" for the pressure or "u" for the particle velocity
    along the depth axis, which is recorded half a grid step deeper and
    half a time step earlier. The velocity responds to a wave arriving at
    angle theta from the depth axis as cos(theta), like a baffled element,
    so waves that run along the face of an array are not picked up.
    """

    index: np.ndarray
    group: np.ndarray = None   # element number of each point, for arrays
    quantity: str = "p"

    def __post_init__(self):
        self.index = np.atleast_2d(np.asarray(self.index, dtype=int))
        if self.quantity not in ("p", "u"):
            raise ValueError("quantity must be 'p' or 'u'")


def array_sensor(transducer, grid, quantity="p"):
    """Sensor on the grid points under each element of a transducer."""
    idx, el = _element_points(transducer, grid)
    return Sensor(idx, el, quantity)


# --------------------------------------------------------------------------
# solver


@dataclass
class WaveResult:
    """Output of simulate().

    dt time step [s]; t times of the recorded samples [s]; p recorded
    pressure (n_points, n_t) or None; p_max and p_min the largest and
    smallest pressure at each grid point; p_rms; cw dict mapping each
    requested frequency to the complex amplitude map; grid the grid.
    """

    dt: float
    t: np.ndarray
    p: np.ndarray = None
    p_max: np.ndarray = None
    p_min: np.ndarray = None
    p_rms: np.ndarray = None
    cw: dict = field(default_factory=dict)
    sensor: Sensor = None
    grid: object = None
    y: float = None

    def element_signals(self):
        """Mean recorded pressure over the points of each element."""
        if self.sensor is None or self.sensor.group is None:
            raise ValueError("the sensor has no element grouping")
        g = self.sensor.group
        n = g.max() + 1
        out = np.zeros((n, self.p.shape[1]))
        np.add.at(out, g, self.p)
        cnt = np.bincount(g, minlength=n)
        return out / np.maximum(cnt, 1)[:, None]


def _alpha_np_rad(alpha0, y):
    """dB/(cm MHz^y) to Np/(m (rad/s)^y)."""
    return np.asarray(alpha0) * 100.0 * DB_TO_NP / (2 * np.pi * 1e6) ** y


def stable_dt(grid, c_max, cfl=0.3):
    return cfl * float(np.min(grid.spacing)) / float(c_max)


def simulate(medium, sources, t_end, sensor=None, dt=None, cfl=0.3, pml=20, pml_alpha=2.0,
             nonlinear=False, absorbing=True, y=None, f_ref=None, record_max=False,
             record_rms=False, cw=None, cw_cycles=10, record_every=1, smooth_medium=False,
             source_correction=True, precision="double", progress=False, max_memory=None):
    """Run the k-space pseudospectral solver.

    Parameters
    ----------
    medium : Medium
        Uses c, rho, alpha0, y and ba (B/A).
    sources : Source or list of Source
    t_end : float
        Simulated time [s].
    sensor : Sensor, optional
        Points whose pressure is stored at every record_every-th step.
    dt : float, optional
        Time step [s], default cfl * min(spacing) / max(c).
    pml : int
        Smallest thickness of the perfectly matched layer in grid points,
        added on every side of the grid. It is enlarged a little on each
        axis so that the padded size suits the FFT.
    pml_alpha : float
        Absorption of the layer [Np per grid point].
    nonlinear : bool
        Include the B/A term and convective nonlinearity.
    absorbing : bool
        Include power-law absorption and dispersion.
    y : float or "tissue", optional
        Power-law exponent used on the whole grid; default the median of
        medium.y. "tissue" keeps the exponent of every tissue (see the
        module notes). Values within 0.01 of 1 are set to 1.01 because the
        dispersion term is singular at y = 1.
    f_ref : float, optional
        Reference frequency [Hz], usually the centre frequency of the
        source. Each tissue keeps its own attenuation at f_ref when y
        differs from its exponent, and medium.c is taken as the phase speed
        at f_ref rather than the low-frequency limit of the dispersion
        relation. Default: no rescaling.
    record_max, record_rms : bool
        Store the largest, smallest and RMS pressure at every grid point.
    cw : float or list of float, optional
        Frequencies [Hz] whose complex amplitude is extracted at every grid
        point over the last cw_cycles periods of the lowest frequency.
    source_correction : bool
        Filter the source terms with cos(c_ref k dt / 2), as k-Wave does
        for additive sources. Turning it off saves two FFTs per step.
    precision : str
        "double" or "single". Single precision halves the memory and, with
        scipy installed, roughly doubles the speed.
    progress : bool
        Print the progress.
    max_memory : float, optional
        Largest memory [bytes] the run may use (see memory_estimate); a
        larger run raises MemoryError before anything is allocated.
        Without it, runs above 8 GB give a warning.

    The run stops with FloatingPointError if the pressure stops being
    finite, which happens when the time step is too long for a strongly
    heterogeneous medium.
    """
    g = medium.grid
    nd = g.ndim
    sources = [sources] if isinstance(sources, Source) else list(sources)

    c0 = np.asarray(medium.c, dtype=float)
    rho0 = np.asarray(medium.rho, dtype=float)
    if not (np.all(np.isfinite(c0)) and np.all(np.isfinite(rho0))):
        raise ValueError("sound speed and density must be finite")
    if np.any(c0 <= 0) or np.any(rho0 <= 0):
        raise ValueError("sound speed and density must be positive")
    if not (np.isfinite(t_end) and t_end > 0):
        raise ValueError("t_end must be a positive number")
    if not (np.isfinite(cfl) and 0 < cfl <= 1):
        raise ValueError("cfl must be in (0, 1]")
    if dt is not None and not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt must be a positive number")
    if int(record_every) != record_every or record_every < 1:
        raise ValueError("record_every must be a positive integer")
    if precision not in ("single", "double"):
        raise ValueError("precision must be 'single' or 'double'")
    if int(pml) != pml or pml < 0:
        raise ValueError("pml must be a non-negative integer")
    if not sources:
        raise ValueError("no source given")
    c_ref = float(c0.max())
    dt = stable_dt(g, c_ref, cfl) if dt is None else float(dt)
    cfl_used = c_ref * dt / float(np.min(g.spacing))
    if cfl_used > 0.5:
        warnings.warn(f"CFL number {cfl_used:.2f}: the scheme is exact only in a homogeneous "
                      "medium; above about 0.5 a heterogeneous run can become unstable",
                      RuntimeWarning, stacklevel=2)
    nt = int(np.ceil(t_end / dt)) + 1
    need = memory_estimate(g, pml, precision, nonlinear=nonlinear,
                           n_record=0 if sensor is None else
                           np.atleast_2d(sensor.index).shape[0] * (nt // int(record_every) + 1))
    if max_memory is not None and need > max_memory:
        raise MemoryError(f"the run needs about {need / 1e9:.1f} GB, above max_memory")
    if max_memory is None and need > 8e9:
        warnings.warn(f"the run needs about {need / 1e9:.1f} GB of memory", RuntimeWarning,
                      stacklevel=2)
    if isinstance(y, str) and y != "tissue":
        raise ValueError("y must be a number or 'tissue'")

    real = np.float32 if precision == "single" else np.float64

    # padded domain, with a size that suits the FFT
    Ps = []
    for n in g.shape:
        tot = _next_fast(n + 2 * int(pml))
        Ps.append((tot - n + 1) // 2)
    padw = [(Pi, Pi) for Pi in Ps]
    shape = tuple(n + 2 * Pi for n, Pi in zip(g.shape, Ps))
    inner = tuple(slice(Pi, Pi + n) for n, Pi in zip(g.shape, Ps))
    offset = np.array(Ps)

    def pad(a):
        a = np.asarray(a, dtype=float)
        if smooth_medium:
            a = _smooth(a)
        return np.pad(a, padw, mode="edge")

    c0p = pad(c0)
    rho0p = pad(rho0)

    # wavenumbers (rfft along the last axis)
    ks = []
    for i in range(nd):
        if i == nd - 1:
            k = 2 * np.pi * np.fft.rfftfreq(shape[i], g.spacing[i])
        else:
            k = 2 * np.pi * np.fft.fftfreq(shape[i], g.spacing[i])
        sh = [1] * nd
        sh[i] = k.size
        ks.append(k.reshape(sh))
    kmag = np.sqrt(sum(k ** 2 for k in ks))
    kappa = np.sinc(c_ref * kmag * dt / (2 * np.pi))
    src_kappa = np.cos(c_ref * kmag * dt / 2)
    axes = tuple(range(nd))
    ddx_pos = [1j * k * kappa * np.exp(1j * k * g.spacing[i] / 2) for i, k in enumerate(ks)]
    ddx_neg = [1j * k * kappa * np.exp(-1j * k * g.spacing[i] / 2) for i, k in enumerate(ks)]

    def fft(a):
        return _rfftn(a, axes)

    def ifft(a):
        return _irfftn(a, shape, axes)

    cplx = np.complex64 if real is np.float32 else np.complex128
    ddx_pos = [d.astype(cplx) for d in ddx_pos]
    ddx_neg = [d.astype(cplx) for d in ddx_neg]
    src_kappa = src_kappa.astype(real)

    # density on the staggered grids
    rho_sg = []
    for i in range(nd):
        r = rho0p.copy()
        sl = [slice(None)] * nd
        sl[i] = slice(0, -1)
        sl2 = [slice(None)] * nd
        sl2[i] = slice(1, None)
        r[tuple(sl)] = 0.5 * (rho0p[tuple(sl)] + rho0p[tuple(sl2)])
        rho_sg.append(r)

    # perfectly matched layer (as k-Wave: alpha (c / dx) d^4, d the depth
    # into the layer from 1/P to 1)
    def profile(n, axis, staggered):
        P = Ps[axis]
        j = np.arange(n, dtype=float) + (0.5 if staggered else 0.0)
        d = np.zeros(n)
        if P > 0:
            left = j < P
            d[left] = (P - j[left]) / P
            right = j > n - 1 - P
            d[right] = (j[right] - (n - 1 - P)) / P
        prof = pml_alpha * (c_ref / g.spacing[axis]) * np.clip(d, 0, None) ** 4
        sh = [1] * nd
        sh[axis] = n
        return np.exp(-prof * dt / 2).reshape(sh).astype(real)

    pml_n = [profile(shape[i], i, False) for i in range(nd)]
    pml_s = [profile(shape[i], i, True) for i in range(nd)]

    # absorption: one group per power-law exponent
    groups = []
    y_used = None
    if absorbing and np.any(np.asarray(medium.alpha0) > 0):
        a0 = np.broadcast_to(np.asarray(medium.alpha0, dtype=float), g.shape)
        y_map = np.broadcast_to(np.asarray(medium.y, dtype=float), g.shape)
        if isinstance(y, str):
            keys = np.round(y_map, 3)
            members = [(float(v), keys == v) for v in np.unique(keys[a0 > 0])]
        else:
            y_one = float(np.median(y_map)) if y is None else float(y)
            if y is None and f_ref is None and np.ptp(y_map[a0 > 0]) > 0.2:
                warnings.warn("tissues with different power-law exponents but no f_ref: each "
                              "tissue keeps its alpha0 with the median exponent; give f_ref "
                              "or y='tissue'", RuntimeWarning, stacklevel=2)
            members = [(y_one, np.ones(g.shape, bool))]
        a_maps, c_shift = [], np.zeros(g.shape)
        for yg, mask in members:
            yg = 1.01 if abs(yg - 1.0) < 0.01 else yg
            a = np.where(mask, a0, 0.0)
            if f_ref is not None and not isinstance(y, str):
                a = a * (f_ref / 1e6) ** (y_map - yg)
            a_np = _alpha_np_rad(a, yg)
            if f_ref is not None:
                w_ref = 2 * np.pi * f_ref
                c_shift += a_np * np.tan(np.pi * yg / 2) * w_ref ** (yg - 1)
            a_maps.append((yg, a_np))
        a_np_pad = [(yg, pad(a)) for yg, a in a_maps]
        if f_ref is not None:
            # the model's c0 is the low-frequency limit; shift it so that the
            # phase speed at f_ref equals the tissue sound speed
            c0p = 1.0 / (1.0 / c0p - pad(c_shift))
        for yg, a_np in a_np_pad:
            tau = -2 * a_np * c0p ** (yg - 1)
            eta = 2 * a_np * c0p ** yg * np.tan(np.pi * yg / 2)
            with np.errstate(divide="ignore"):
                nab1 = np.where(kmag > 0, kmag ** (yg - 2), 0.0).astype(real)
                nab2 = np.where(kmag > 0, kmag ** (yg - 1), 0.0).astype(real)
            groups.append(((c0p ** 2 * tau).astype(real), (c0p ** 2 * eta).astype(real),
                           nab1, nab2))
        y_used = "tissue" if isinstance(y, str) else a_maps[0][0]
    lossy = bool(groups)

    c2 = (c0p ** 2).astype(real)
    nl_c2 = (c0p ** 2 * pad(medium.ba) / (2 * rho0p)).astype(real) if nonlinear else None
    dt_rho_sg = [(dt / r).astype(real) for r in rho_sg]
    dt_rho0 = (dt * rho0p).astype(real)
    rho0r = rho0p.astype(real)

    # fields
    u = [np.zeros(shape, real) for _ in range(nd)]
    rho = [np.zeros(shape, real) for _ in range(nd)]
    p = np.zeros(shape, real)

    # sources on the padded grid
    p_src = [s for s in sources if s.kind == "p"]
    u_src = [s for s in sources if s.kind == "u"]

    def flat(index):
        idx = np.asarray(index) + offset
        if np.any(idx < offset) or np.any(idx >= np.array(shape) - offset):
            raise ValueError("source or sensor point outside the grid")
        return np.ravel_multi_index(tuple(idx.T), shape)

    p_src_flat = [flat(s.index) for s in p_src]
    u_src_flat = [flat(s.index) for s in u_src]
    p_scale = [2 * dt / (nd * c0p.ravel()[f] * float(np.mean(g.spacing))) for f in p_src_flat]
    u_scale = [2 * c0p.ravel()[f] * dt / g.spacing[-1] for f in u_src_flat]

    sens_flat = flat(sensor.index) if sensor is not None else None
    n_rec = (nt + record_every - 1) // record_every
    rec = np.zeros((sens_flat.size, n_rec), real) if sensor is not None else None

    p_max = np.full(g.shape, -np.inf) if record_max else None
    p_min = np.full(g.shape, np.inf) if record_max else None
    p_sq = np.zeros(g.shape) if record_rms else None

    freqs = [] if cw is None else list(np.atleast_1d(cw).astype(float))
    if freqs:
        n_win = int(round(cw_cycles / min(freqs) / dt))
        n_win = min(n_win, nt)
        cw_start = nt - n_win
        cw_acc = {f: np.zeros(g.shape, dtype=complex) for f in freqs}
    else:
        cw_start = nt + 1

    def inject(srcs, flats, scales, t):
        term = np.zeros(shape, real)
        tr = term.ravel()
        for s, fl, sc in zip(srcs, flats, scales):
            np.add.at(tr, fl, s.values(t) * sc)
        if source_correction:
            term = ifft(src_kappa * fft(term)).astype(real, copy=False)
        return term

    report = max(nt // 20, 1)
    div_prev = 0.0
    for n in range(nt):
        t = n * dt
        # velocity update
        P_k = fft(p)
        for i in range(nd):
            u[i] = pml_s[i] * (pml_s[i] * u[i] - dt_rho_sg[i] * ifft(ddx_pos[i] * P_k))
        if u_src:
            u[-1] += inject(u_src, u_src_flat, u_scale, t + dt / 2)
        # density update
        divsum = 0
        for i in range(nd):
            d = ifft(ddx_neg[i] * fft(u[i]))
            if lossy:
                divsum = divsum + d
            if nonlinear:
                rho[i] = pml_n[i] * (pml_n[i] * rho[i] - dt_rho0 * d) / (1 + 2 * dt * d)
            else:
                rho[i] = pml_n[i] * (pml_n[i] * rho[i] - dt_rho0 * d)
        if p_src:
            term = inject(p_src, p_src_flat, p_scale, t + dt)
            for i in range(nd):
                rho[i] += term
        # equation of state
        rsum = rho[0] if nd == 1 else sum(rho)
        if lossy:
            # div(u) is known at n + 1/2; extrapolate it to n + 1, where rho
            # and p live, so the absorption term is centred in time
            div_c = 1.5 * divsum - 0.5 * div_prev if n > 0 else divsum
            div_prev = divsum
            F_div = fft(rho0r * div_c)
            F_r = fft(rsum)
            p = c2 * rsum
            for tau_c2, eta_c2, nab1, nab2 in groups:
                p = p + tau_c2 * ifft(nab1 * F_div) - eta_c2 * ifft(nab2 * F_r)
        else:
            p = c2 * rsum
        if nonlinear:
            p = p + nl_c2 * rsum ** 2

        # recording
        pin = p[inner]
        if rec is not None and n % record_every == 0:
            field_ = u[-1] if sensor.quantity == "u" else p
            rec[:, n // record_every] = field_.ravel()[sens_flat]
        if record_max:
            np.maximum(p_max, pin, out=p_max)
            np.minimum(p_min, pin, out=p_min)
        if record_rms:
            p_sq += pin ** 2
        if n >= cw_start:
            for f in freqs:
                cw_acc[f] += pin * np.exp(-2j * np.pi * f * (t + dt))
        if n % report == 0:
            if not np.all(np.isfinite(pin)):
                raise FloatingPointError(f"the solution diverged at step {n}; use a shorter "
                                         "time step (smaller cfl)")
            if progress:
                print(f"step {n}/{nt}", flush=True)

    cw_out = {}
    if freqs:
        for f in freqs:
            cw_out[f] = 2 * cw_acc[f] / (nt - cw_start)
    return WaveResult(
        dt=dt,
        t=np.arange(n_rec) * dt * record_every + dt,
        p=rec,
        p_max=p_max,
        p_min=p_min,
        p_rms=None if p_sq is None else np.sqrt(p_sq / nt),
        cw=cw_out,
        sensor=sensor,
        grid=g,
        y=y_used,
    )


def _smooth(a):
    """Hann filter in k-space, used to reduce staircasing of sharp
    interfaces before a run (as k-Wave's smooth option)."""
    A = np.fft.fftn(a)
    for i, n in enumerate(a.shape):
        w = 0.5 * (1 + np.cos(2 * np.pi * np.fft.fftfreq(n)))
        sh = [1] * a.ndim
        sh[i] = n
        A = A * w.reshape(sh)
    return np.real(np.fft.ifftn(A))


# --------------------------------------------------------------------------
# helpers for imaging and therapy


def _surface_medium(medium):
    """Medium whose every depth row equals the first one."""
    from .medium import Medium
    g = medium.grid
    first = (slice(None),) * (g.ndim - 1) + (slice(0, 1),)
    ref = Medium(g, np.broadcast_to(medium.labels[first], g.shape).copy(), medium.materials)
    ref.c = np.broadcast_to(np.asarray(medium.c)[first], g.shape).copy()
    ref.rho = np.broadcast_to(np.asarray(medium.rho)[first], g.shape).copy()
    return ref


def memory_estimate(grid, pml=20, precision="double", nonlinear=False, n_record=0):
    """Rough memory [bytes] used by simulate() on grid: the padded field
    arrays, the FFT work arrays and the recorded samples."""
    shape = [_next_fast(n + 2 * int(pml)) for n in grid.shape]
    n = float(np.prod(shape))
    nd = grid.ndim
    item = 4 if precision == "single" else 8
    # u, rho per axis, p, c2, operators, about 2 complex work arrays
    arrays = 2 * nd + 6 + 2 * nd + (1 if nonlinear else 0)
    return int(n * item * arrays + 2 * n * 2 * item + n_record * item)


def exponent_error(alpha0, y_tissue, f, f_ref, y_grid):
    """Error of the single-exponent model for one tissue.

    A tissue with attenuation alpha0 f^y_tissue [dB/(cm MHz^y)] run with
    the grid exponent y_grid and the reference frequency f_ref [Hz] gets
    the attenuation alpha0 f_ref^(y_tissue - y_grid) f^y_grid. Returns the
    ratio of the modelled to the true attenuation at f [Hz] and the error
    of the phase speed [m/s] for a tissue of 1540 m/s.
    """
    f = np.asarray(f, dtype=float)
    ratio = (f / f_ref) ** (y_grid - y_tissue)
    a_grid = alpha0 * (f_ref / 1e6) ** (y_tissue - y_grid)
    yg = 1.01 if abs(y_grid - 1) < 0.01 else y_grid
    yt = 1.01 if abs(y_tissue - 1) < 0.01 else y_tissue
    c_true = dispersion_speed(1540.0, alpha0, yt, f, f_ref)
    c_grid = dispersion_speed(1540.0, a_grid, yg, f, f_ref)
    return ratio, c_grid - c_true


def points_per_wavelength(medium, f):
    """Smallest number of grid points per wavelength at frequency f."""
    return float(np.min(medium.c) / f / np.max(medium.grid.spacing))


def dispersion_speed(c0, alpha0, y, f, f0=0.0):
    """Phase speed at f [Hz] of the power-law model, from the
    Kramers-Kronig relation (Treeby and Cox 2010):

        1/c(f) = 1/c0 + a tan(pi y / 2) (w^(y-1) - w0^(y-1)),

    with a the attenuation in Np/(m (rad/s)^y). c0 is the speed at f0; the
    default f0 = 0 makes c0 the low-frequency limit, which is the speed the
    solver uses when f_ref is not given (for 1 < y < 2)."""
    a = _alpha_np_rad(alpha0, y)
    w = 2 * np.pi * np.asarray(f, dtype=float)
    w0 = 2 * np.pi * float(f0)
    ref = w0 ** (y - 1) if w0 > 0 else 0.0
    return 1.0 / (1.0 / c0 + a * np.tan(np.pi * y / 2) * (w ** (y - 1) - ref))


def resample(signals, t, fs, t_start=0.0, n=None):
    """Resample signals (n_ch, n_t) recorded at times t [s] onto a grid of
    rate fs starting at t_start, by linear interpolation (the solver's
    step is several times shorter than 1/fs)."""
    signals = np.atleast_2d(signals)
    t = np.asarray(t, dtype=float)
    if n is None:
        n = int(np.floor((t[-1] - t_start) * fs)) + 1
    t_new = t_start + np.arange(n) / fs
    out = np.empty((signals.shape[0], n))
    for i, s in enumerate(signals):
        out[i] = np.interp(t_new, t, s, left=0.0, right=0.0)
    return out, t_new


def analytic_signal(x, axis=-1):
    """Analytic signal by the FFT (Hilbert transform)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    X = np.fft.fft(x, axis=axis)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    sh = [1] * x.ndim
    sh[axis] = n
    return np.fft.ifft(X * h.reshape(sh), axis=axis)


def pulse_echo(medium, array, transmit, pulse, c=1540.0, fs=None, t_end=None, blank=None,
               amplitude=1e5, subtract_direct=False, apodization=None, directional=True,
               band=(0.3, 2.0), **kw):
    """Full-wave pulse-echo signals of one transmit, ready for beamforming.

    The elements of array fire pulse (an imaging.Pulse used as the
    transmit waveform) with the delays of transmit computed at the nominal
    speed c, and the pressure is recorded on the same elements. The result
    uses the time origin of sonosk.imaging (the first firing at t = 0), so
    it can go straight to imaging.beamform or imaging.compound.

    Parameters
    ----------
    medium : Medium
        Usually with random fluctuations from phantom.fluctuations.
    array : Transducer
        Its face must lie on grid points of the medium (z = first row).
    transmit : imaging.Transmit
    pulse : imaging.Pulse
    fs : float, optional
        Output sampling rate, default 8 f0.
    t_end : float, optional
        Recording time [s], default the round trip over the grid depth.
    blank : float, optional
        Signals are set to zero before delay + blank [s] on each element,
        to remove the transmitted pulse; default the pulse duration plus
        the time to cross 1 mm.
    amplitude : float
        Source pressure [Pa]; only matters for nonlinear runs.
    apodization : array, optional
        Transmit weight of each element, default uniform. A tapered
        aperture (for example a Tukey window) weakens the edge waves of
        plane-wave transmits.
    directional : bool
        Drive the elements with a normal velocity and record the normal
        velocity (converted to pressure with the impedance of the first
        row), instead of pressure. Elements then transmit and receive with
        a cos(theta) directivity. With omnidirectional pressure points, in
        2-D above all, the transmitted wave that runs along the face of
        the array reaches every element for as long as it takes to cross
        the aperture, and beamforms into a bright band over the near
        field.
    band : (low, high) or None
        Receive band-pass filter, in multiples of the centre frequency,
        with raised-cosine edges 0.2 f0 wide. It removes the slow drift
        of the recorded velocity, as the receive filter of a scanner does.
    subtract_direct : bool
        Run a second simulation in a medium that repeats the first depth
        row everywhere and subtract its signals. This removes the
        transmitted pulse and the edge waves that run along the face of
        the array, which otherwise beamform into an X across the near
        field. It doubles the cost.
    kw : passed to simulate (cfl, pml, precision, f_ref, ...).

    Returns
    -------
    rf : complex array (n_elements, n_samples), analytic signals
    fs : float
    """
    f0 = pulse.f0
    fs = 8 * f0 if fs is None else float(fs)
    e = array.centers
    delays = transmit.element_delays(e, c)
    sig = gaussian_pulse(pulse)
    t0 = sig.t0
    g = medium.grid
    apo = np.ones(array.n_elements) if apodization is None else np.asarray(apodization, float)
    if directional:
        first = (slice(None),) * (g.ndim - 1) + (0,)
        z0 = float(np.mean((np.asarray(medium.rho) * np.asarray(medium.c))[first]))
        src = array_source(array, g, sig, delays=delays, apodization=apo, kind="u",
                           amplitude=amplitude / z0)
        sens = array_sensor(array, g, quantity="u")
    else:
        z0 = 1.0
        src = array_source(array, g, sig, delays=delays, apodization=apo,
                           amplitude=amplitude)
        sens = array_sensor(array, g)
    if t_end is None:
        depth = medium.grid.extent[-1]
        t_end = delays.max() + 2 * t0 + 2.2 * depth / float(np.min(medium.c))
    kw.setdefault("f_ref", f0)
    res = simulate(medium, src, t_end, sensor=sens, **kw)
    el = res.element_signals() * z0 / amplitude
    if subtract_direct:
        kw_ref = dict(kw, dt=res.dt)
        el = el - simulate(_surface_medium(medium), src, t_end, sensor=sens,
                           **kw_ref).element_signals() * z0 / amplitude
    if band is not None:
        n = el.shape[1]
        f = np.fft.rfftfreq(n, res.t[1] - res.t[0])
        lo, hi, edge = band[0] * f0, band[1] * f0, 0.2 * f0
        w = np.clip((f - lo + edge) / edge, 0, 1) * np.clip((hi + edge - f) / edge, 0, 1)
        el = np.fft.irfft(np.fft.rfft(el, axis=1) * np.sin(0.5 * np.pi * w) ** 2, n, axis=1)
    blank = (2 * t0 + 1e-3 / c) if blank is None else blank
    t = res.t
    if directional:
        # the velocity lives half a time step earlier and half a cell
        # deeper, for the source and the sensor alike
        t = t - 0.5 * res.dt + g.spacing[-1] / c
    for i in range(el.shape[0]):
        el[i, t < delays[i] + blank] = 0.0
    rf, _ = resample(el, t, fs, t_start=t0)
    return analytic_signal(rf, axis=1), fs

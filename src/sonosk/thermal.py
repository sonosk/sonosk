"""Heating by ultrasound, bioheat transfer and thermal dose.

The temperature T follows the Pennes bioheat equation,

    rho cp dT/dt = div(k grad T) + w_b c_b (T_a - T) + Q,

with w_b the blood perfusion [kg/(m^3 s)], c_b the specific heat of blood
and T_a the arterial temperature. The heat source of a CW field with
pressure amplitude p is Q = 2 alpha I = alpha p^2 / (rho c), where alpha
is the amplitude absorption coefficient [Np/m].

The equation is solved with an explicit finite-difference scheme on the
grid of the medium, in 1, 2 or 3 dimensions, with the outer boundary held
at the body temperature. The time step is chosen inside the stability
limit.

Thermal dose is the cumulative equivalent minutes at 43 C,

    CEM43 = sum dt R^(43 - T),  R = 0.5 for T >= 43 C, 0.25 below,

with dt in minutes. A dose of 240 min is the usual threshold for thermal
necrosis in soft tissue.

References
----------
Pennes H.H. Analysis of tissue and arterial blood temperatures in the
resting human forearm. J. Appl. Physiol. 1(2):93-122, 1948.

Sapareto S.A., Dewey W.C. Thermal dose determination in cancer therapy.
Int. J. Radiat. Oncol. Biol. Phys. 10(6):787-800, 1984.
"""

from dataclasses import dataclass

import numpy as np

C_BLOOD = 3617.0   # J/(kg K), IT'IS tissue database, blood
LESION_DOSE = 240.0


def heat_source(p, alpha, rho, c):
    """Absorbed power density [W/m^3] for pressure amplitude p [Pa].

    alpha is the absorption coefficient [Np/m]; with no separate
    absorption data the attenuation is often used, which overestimates
    heating where scattering is large.
    """
    return np.asarray(alpha) * np.abs(p) ** 2 / (np.asarray(rho) * np.asarray(c))


def dose_rate(T):
    """CEM43 accumulated per second [min/s] at temperature T [C]."""
    T = np.asarray(T, dtype=float)
    r = np.where(T >= 43.0, 0.5, 0.25)
    return r ** (43.0 - T) / 60.0


def cem43(T, dt):
    """Thermal dose [min] of a temperature history.

    T has time on its first axis, sampled every dt seconds.
    """
    return np.sum(dose_rate(T), axis=0) * dt


@dataclass
class BioheatResult:
    t: np.ndarray            # times of the snapshots [s]
    T: np.ndarray            # snapshots, shape (len(t),) + grid.shape [C]
    T_max: np.ndarray        # highest temperature reached at each point [C]
    dose: np.ndarray         # CEM43 [min]
    dt: float                # time step used [s]
    probe: np.ndarray = None  # temperature history at the probe points

    def lesion(self, threshold=LESION_DOSE):
        return self.dose >= threshold


def _laplacian_flux(T, k, h):
    """div(k grad T) with face conductivities from the arithmetic mean and
    the boundary handled by the caller (fixed values)."""
    out = np.zeros_like(T)
    for ax, dx in enumerate(h):
        sl0 = [slice(None)] * T.ndim
        sl1 = [slice(None)] * T.ndim
        sl0[ax] = slice(0, -1)
        sl1[ax] = slice(1, None)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        kf = 0.5 * (k[sl0] + k[sl1])
        flux = kf * (T[sl1] - T[sl0]) / dx
        out[sl0] += flux / dx
        out[sl1] -= flux / dx
    return out


def bioheat(medium, q, duration, schedule=None, T0=37.0, T_blood=37.0, c_blood=C_BLOOD,
            dt=None, max_dt=0.1, snapshots=None, probe=None):
    """Solve the Pennes equation on the grid of medium.

    Parameters
    ----------
    medium : Medium
        Uses rho, cp, k and perfusion.
    q : array
        Heat source [W/m^3] with the grid shape, at full power.
    duration : float
        Simulated time [s].
    schedule : callable, optional
        schedule(t) gives the power fraction between 0 and 1 at time t, for
        pulsed or interrupted exposures. Default: on all the time.
    T0, T_blood : float
        Initial and arterial temperature [C]. The grid boundary stays at T0.
    dt : float, optional
        Time step [s]; by default 0.9 of the stability limit, at most max_dt.
        The schedule is sampled once per step, so dt must be short against
        the on and off times of a pulsed exposure.
    snapshots : array, optional
        Times [s] at which to store the temperature field.
    probe : list of index tuples, optional
        Grid points whose temperature is stored at every step.
    """
    g = medium.grid
    rc = medium.rho * medium.cp
    if np.any(rc <= 0):
        raise ValueError("rho * cp must be positive everywhere")
    k = medium.k
    wb = medium.perfusion * c_blood
    q = np.asarray(q, dtype=float)
    if q.shape != g.shape:
        raise ValueError("q must have the grid shape")

    lim = 1.0 / (np.max(2 * k / rc) * np.sum(1.0 / g.spacing ** 2) + np.max(wb / rc) + 1e-300)
    dt = min(0.9 * lim, max_dt) if dt is None else float(dt)
    if dt > lim:
        raise ValueError(f"dt above the stability limit {lim:.3g} s")
    n = int(np.ceil(duration / dt))
    dt = duration / n
    schedule = (lambda t: 1.0) if schedule is None else schedule

    snaps = np.array([] if snapshots is None else snapshots, dtype=float)
    k_snap = set(np.rint(snaps / dt).astype(int).tolist())
    T = np.full(g.shape, float(T0))
    Tmax = T.copy()
    dose = np.zeros(g.shape)
    stored, times = [], []
    hist = [] if probe else None
    inner = tuple(slice(1, -1) for _ in range(g.ndim))

    if 0 in k_snap:
        stored.append(T.copy())
        times.append(0.0)
    for i in range(n):
        t = i * dt
        rhs = _laplacian_flux(T, k, g.spacing) + wb * (T_blood - T) + schedule(t) * q
        Tn = T + dt * rhs / rc
        Tb = np.full(g.shape, float(T0))
        Tb[inner] = Tn[inner]
        T = Tb
        np.maximum(Tmax, T, out=Tmax)
        dose += dose_rate(T) * dt
        if hist is not None:
            hist.append([T[p] for p in probe])
        if i + 1 in k_snap:
            stored.append(T.copy())
            times.append((i + 1) * dt)
    return BioheatResult(np.array(times), np.array(stored), Tmax, dose, dt,
                         None if hist is None else np.array(hist))


def treatment_report(lesion, medium, target):
    """Volumes treated and spared.

    Parameters
    ----------
    lesion : bool array
        Points above the dose threshold.
    medium : Medium
    target : str or list of str
        Names of the target materials (for example the tumour).

    Returns
    -------
    dict with the target volume, ablated and residual target volume,
    fraction of the target ablated, damaged volume outside the target and
    the damaged volume of each material. Volumes in m^3 (3-D), m^2 (2-D)
    or m (1-D).
    """
    names = [target] if isinstance(target, str) else list(target)
    tmask = np.zeros(medium.grid.shape, bool)
    for nm in names:
        tmask |= medium.mask(nm)
    cell = float(np.prod(medium.grid.spacing))
    lesion = np.asarray(lesion, bool)
    tv = tmask.sum() * cell
    abl = (lesion & tmask).sum() * cell
    per = {}
    for m in {m.name for m in medium.materials}:
        per[m] = float((lesion & medium.mask(m)).sum() * cell)
    return {
        "target_volume": float(tv),
        "ablated": float(abl),
        "residual": float(tv - abl),
        "fraction_ablated": float(abl / tv) if tv > 0 else 0.0,
        "healthy_damaged": float((lesion & ~tmask).sum() * cell),
        "per_material": per,
    }

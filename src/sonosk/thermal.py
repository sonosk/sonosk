"""Heating by ultrasound, bioheat transfer, thermal dose and tissue damage.

The temperature T follows the Pennes bioheat equation,

    rho cp dT/dt = div(k grad T) + w_b c_b (T_a - T) + Q,

with w_b the blood perfusion [kg/(m^3 s)], c_b the specific heat of blood
and T_a the arterial temperature. The heat source of a CW field with
pressure amplitude p is Q = 2 alpha I = alpha p^2 / (rho c), where alpha
is the amplitude absorption coefficient [Np/m]. With harmonics from a
nonlinear field each harmonic adds its own term at its own frequency.

The equation is solved with an explicit finite-difference scheme on the
grid of the medium, in 1, 2 or 3 dimensions, with the outer boundary held
at the body temperature. The time step is chosen inside the stability
limit.

Two measures of thermal damage are computed:

- the thermal dose in cumulative equivalent minutes at 43 C,
  CEM43 = sum dt R^(43 - T), R = 0.5 for T >= 43 C and 0.25 below, with
  dt in minutes; 240 min is the usual threshold for necrosis;
- the Arrhenius damage integral Omega = integral A exp(-Ea / (R_gas T)) dt
  with T in kelvin; Omega = 1 means 63 % of the cells or proteins are
  damaged.

Blood flow stops in coagulated tissue. With shutdown set, the perfusion of
a point drops to zero once its dose or damage passes the threshold, which
lets the lesion grow faster in well-perfused tissue.

A treatment is a sequence of sonications (focus positions, powers, on and
cooling times); treat() runs them one after another on the same tissue.

References
----------
Pennes H.H. Analysis of tissue and arterial blood temperatures in the
resting human forearm. J. Appl. Physiol. 1(2):93-122, 1948.

Sapareto S.A., Dewey W.C. Thermal dose determination in cancer therapy.
Int. J. Radiat. Oncol. Biol. Phys. 10(6):787-800, 1984.

Moritz A.R., Henriques F.C. Studies of thermal injury II. The relative
importance of time and surface temperature in the causation of cutaneous
burns. Am. J. Pathol. 23(5):695-720, 1947.

Rossmann C., Haemmerich D. Review of temperature dependence of thermal
properties, dielectric properties, and perfusion of biological tissues at
hyperthermic and ablation temperatures. Crit. Rev. Biomed. Eng.
42(6):467-492, 2014.

Worthington A.E., Trachtenberg J., Sherar M.D. Ultrasound properties of
human prostate tissue during heating. Ultrasound Med. Biol.
28(10):1311-1318, 2002.

Köhler M.O., Mougenot C., Quesson B., et al. Volumetric HIFU ablation under
3D guidance of rapid MRI thermometry. Med. Phys. 36(8):3521-3535, 2009.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from .medium import Medium

C_BLOOD = 3617.0   # J/(kg K), blood, IT'IS V5.0
LESION_DOSE = 240.0
R_GAS = 8.314462618


# --------------------------------------------------------------------------
# heat sources


def heat_source(p, alpha, rho, c, fraction=1.0):
    """Absorbed power density [W/m^3] for pressure amplitude p [Pa].

    alpha is the attenuation [Np/m] and fraction the part of it that is
    absorption (a number or a map, between 0 and 1). The default 1 treats
    all of the attenuation as absorption, which overestimates heating where
    scattering is large; no tissue in the database has a measured
    absorption fraction yet, so the choice is left to the user.
    """
    fr = np.asarray(fraction, dtype=float)
    if np.any(fr < 0) or np.any(fr > 1):
        raise ValueError("fraction must lie between 0 and 1")
    return fr * np.asarray(alpha) * np.abs(p) ** 2 / (np.asarray(rho) * np.asarray(c))


def heat_source_harmonics(cw, medium, fraction=1.0):
    """Heat source [W/m^3] of a nonlinear field.

    cw maps each frequency [Hz] to the complex pressure amplitude map, as
    returned by sonosk.fullwave.simulate(..., cw=[f, 2f, 3f, ...]). Each
    harmonic is absorbed with the attenuation of the medium at its own
    frequency, times fraction (see heat_source).
    """
    q = 0.0
    for f, p in cw.items():
        q = q + heat_source(p, medium.attenuation(f), medium.rho, medium.c, fraction)
    return q


def shift(q, grid, offset):
    """Field q translated by offset (one value per grid axis, or (x, y, z))
    [m], with linear interpolation and zero fill. Used to move a focal heat
    source to another target point without recomputing the field, which is
    accurate for small electronic or mechanical steering."""
    q = np.asarray(q, dtype=float)
    off = np.asarray(offset, dtype=float).ravel()
    if off.size == 3 and grid.ndim < 3:
        off = off[[0, 2]] if grid.ndim == 2 else off[[2]]
    out = q
    for ax in range(grid.ndim):
        s = off[ax] / grid.spacing[ax]
        if s == 0:
            continue
        k = int(np.floor(s))
        w = s - k
        out = (1 - w) * _roll0(out, k, ax) + w * _roll0(out, k + 1, ax)
    return out


def _roll0(a, k, axis):
    out = np.zeros_like(a)
    n = a.shape[axis]
    if abs(k) >= n:
        return out
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if k >= 0:
        src[axis] = slice(0, n - k)
        dst[axis] = slice(k, n)
    else:
        src[axis] = slice(-k, n)
        dst[axis] = slice(0, n + k)
    out[tuple(dst)] = a[tuple(src)]
    return out


# --------------------------------------------------------------------------
# damage models


def dose_rate(T):
    """CEM43 accumulated per second [min/s] at temperature T [C].

    The exponent is capped at 700 (natural log), so temperatures above
    about 1000 C give a finite, very large rate instead of an overflow.
    """
    T = np.asarray(T, dtype=float)
    lnr = np.where(T >= 43.0, np.log(0.5), np.log(0.25))
    return np.exp(np.minimum((43.0 - T) * lnr, 700.0)) / 60.0


def cem43(T, dt, axis=0):
    """Thermal dose [min] of a temperature history sampled every dt [s].

    Time runs along axis (default the first). Rectangle rule: each sample
    stands for the interval dt that follows it, as in bioheat().
    """
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt must be a positive number")
    return np.sum(dose_rate(T), axis=axis) * dt


@dataclass
class Arrhenius:
    """Arrhenius damage model, possibly with a different rate above a
    breakpoint temperature.

    A [1/s] and Ea [J/mol] apply up to T_break [C]; A_high and Ea_high
    above it (None for a single law).
    """

    A: float
    Ea: float
    T_break: float = None
    A_high: float = None
    Ea_high: float = None
    name: str = ""

    def rate(self, T):
        """dOmega/dt [1/s] at temperature T [C]."""
        Tk = np.asarray(T, dtype=float) + 273.15
        r = np.exp(np.log(self.A) - self.Ea / (R_GAS * Tk))
        if self.T_break is not None:
            r2 = np.exp(np.log(self.A_high) - self.Ea_high / (R_GAS * Tk))
            r = np.where(np.asarray(T) > self.T_break, r2, r)
        return r


# Skin burn model of Henriques and Moritz (1947). It is the only damage model
# shipped with the library; for other tissues it is a stand-in, and the
# result should be read as a damage index. Pass tissue-specific Arrhenius
# objects to bioheat() and treat() where they are known.
#
# Skin burn model of Henriques and Moritz (1947), with the high-temperature
# branch of Weaver and Stoll: Ea/R = 7.5e4 K, A = 3.1e98 1/s up to 55 C;
# Ea/R = 3.54e4 K, A = 5.0e45 1/s above (values as tabulated by Ma et al.,
# Sci. Rep. 11:9958, 2021, Table 3).
HENRIQUES = Arrhenius(A=3.1e98, Ea=7.5e4 * R_GAS, T_break=55.0, A_high=5.0e45,
                      Ea_high=3.54e4 * R_GAS, name="Henriques-Moritz skin")


# --------------------------------------------------------------------------
# bioheat solver


@dataclass
class BioheatResult:
    t: np.ndarray            # times of the snapshots [s]
    T: np.ndarray            # snapshots, shape (len(t),) + grid.shape [C]
    T_max: np.ndarray        # highest temperature reached at each point [C]
    dose: np.ndarray         # CEM43 [min]
    dt: float                # time step used [s]
    probe: np.ndarray = None  # temperature history at the probe points
    T_final: np.ndarray = None    # temperature at the end [C]
    damage: np.ndarray = None     # Arrhenius damage integral Omega
    perfusion: np.ndarray = None  # perfusion map at the end [kg/(m^3 s)]
    time: float = 0.0             # total simulated time, including earlier runs [s]
    notes: list = field(default_factory=list)   # model-validity notes

    def lesion(self, threshold=LESION_DOSE):
        """Points whose thermal dose reached threshold [min]."""
        return self.dose >= threshold

    def damaged(self, threshold=1.0):
        """Points whose Arrhenius damage reached threshold."""
        if self.damage is None:
            raise ValueError("no Arrhenius model was used")
        return self.damage >= threshold


def bioheat(medium, q, duration, schedule=None, T0=37.0, T_blood=37.0, c_blood=C_BLOOD,
            dt=None, max_dt=0.1, snapshots=None, probe=None, initial=None, arrhenius=None,
            shutdown=None, T_surface=None, k_coeff=0.0, cp_coeff=0.0):
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
        Times [s], from the start of this call, at which to store the field.
    probe : list of index tuples, optional
        Grid points whose temperature is stored at every step.
    initial : BioheatResult, optional
        Continue from the end of an earlier run: its final temperature,
        dose, damage and perfusion are the starting state.
    arrhenius : Arrhenius or dict, optional
        Also integrate the Arrhenius damage with this model, or with one
        model per material: {material name: Arrhenius}, with the key "*"
        for all other materials (without it they get no damage).
    shutdown : tuple, optional
        ("dose", minutes) or ("damage", omega): perfusion stops at points
        past this threshold.
    T_surface : float, optional
        Temperature [C] held on the first depth plane (z = grid start), for
        a cooled coupling balloon or skin cooling. The other faces stay at
        T0.
    k_coeff, cp_coeff : float
        Linear temperature dependence of the thermal conductivity and the
        specific heat, as fractions per degree from 37 C:
        k(T) = k (1 + k_coeff (T - 37)). The default 0 keeps them constant.
        No default values are given because the published coefficients
        differ between tissues and studies (Rossmann and Haemmerich 2014).

    Temperatures above 100 C are reported in notes and with a warning:
    boiling and cavitation are not modelled, so results there are not
    physical.
    """
    g = medium.grid
    rc = medium.rho * medium.cp
    if np.any(rc <= 0):
        raise ValueError("rho * cp must be positive everywhere")
    k = medium.k
    q = np.asarray(q, dtype=float)
    if q.shape != g.shape:
        raise ValueError("q must have the grid shape")
    if shutdown is not None and shutdown[0] not in ("dose", "damage"):
        raise ValueError("shutdown must be ('dose', minutes) or ('damage', omega)")
    if shutdown is not None and shutdown[0] == "damage" and arrhenius is None:
        raise ValueError("shutdown on damage needs an Arrhenius model")
    models = _damage_models(medium, arrhenius)
    for name, cf in (("k_coeff", k_coeff), ("cp_coeff", cp_coeff)):
        if not np.isfinite(cf) or 1 + min(cf, 0.0) * 63.0 <= 0 or 1 + cf * -36.0 <= 0:
            raise ValueError(f"{name} makes the property negative between 1 and 100 C")
    varying = k_coeff != 0 or cp_coeff != 0

    if initial is None:
        T = np.full(g.shape, float(T0))
        dose = np.zeros(g.shape)
        dmg = np.zeros(g.shape) if arrhenius is not None else None
        perf = np.array(medium.perfusion, dtype=float)
        Tmax = T.copy()
        t_before = 0.0
    else:
        T = np.array(initial.T_final, dtype=float)
        dose = np.array(initial.dose, dtype=float)
        dmg = (np.array(initial.damage, dtype=float) if initial.damage is not None
               else np.zeros(g.shape)) if arrhenius is not None else None
        perf = np.array(initial.perfusion if initial.perfusion is not None
                        else medium.perfusion, dtype=float)
        Tmax = np.array(initial.T_max, dtype=float)
        t_before = initial.time

    wb = perf * c_blood
    fk = max(1.0, 1.0 + k_coeff * 63.0, 1.0 - k_coeff * 36.0)
    fc = min(1.0, 1.0 + cp_coeff * 63.0, 1.0 - cp_coeff * 36.0)
    lim = 1.0 / (np.max(2 * k * fk / (rc * fc)) * np.sum(1.0 / g.spacing ** 2)
                 + np.max(medium.perfusion * c_blood / (rc * fc)) + 1e-300)
    dt = min(0.9 * lim, max_dt) if dt is None else float(dt)
    if dt > lim:
        raise ValueError(f"dt above the stability limit {lim:.3g} s")
    n = max(int(np.ceil(duration / dt)), 1)
    dt = duration / n
    schedule = (lambda t: 1.0) if schedule is None else schedule

    snaps = np.array([] if snapshots is None else snapshots, dtype=float)
    k_snap = set(np.rint(snaps / dt).astype(int).tolist())
    stored, times = [], []
    hist = [] if probe else None
    inner = tuple(slice(1, -1) for _ in range(g.ndim))

    # face conductivities over dx^2, computed once
    faces = []
    for ax, h in enumerate(g.spacing):
        sl0 = [slice(None)] * g.ndim
        sl1 = [slice(None)] * g.ndim
        sl0[ax] = slice(0, -1)
        sl1[ax] = slice(1, None)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        faces.append((sl0, sl1, 0.5 * (k[sl0] + k[sl1]) / h ** 2))
    dt_rc = dt / rc
    ln_half, ln_quarter = np.log(0.5), np.log(0.25)
    rhs = np.empty(g.shape)

    surface = (slice(None),) * (g.ndim - 1) + (0,)
    if T_surface is not None:
        T[surface] = T_surface
    if 0 in k_snap:
        stored.append(T.copy())
        times.append(t_before)
    for i in range(n):
        t = i * dt
        rhs[...] = wb * (T_blood - T)
        rhs += schedule(t) * q
        if varying:
            kT = k * (1.0 + k_coeff * (T - 37.0))
            dt_rc = dt / (rc * (1.0 + cp_coeff * (T - 37.0)))
            for ax, (sl0, sl1, _) in enumerate(faces):
                kf = 0.5 * (kT[sl0] + kT[sl1]) / g.spacing[ax] ** 2
                flux = kf * (T[sl1] - T[sl0])
                rhs[sl0] += flux
                rhs[sl1] -= flux
        else:
            for sl0, sl1, kf in faces:
                flux = kf * (T[sl1] - T[sl0])
                rhs[sl0] += flux
                rhs[sl1] -= flux
        T[inner] += (rhs * dt_rc)[inner]
        if T_surface is not None:
            T[surface] = T_surface
        np.maximum(Tmax, T, out=Tmax)
        d43 = 43.0 - T
        dose += np.exp(np.minimum(d43 * np.where(T >= 43.0, ln_half, ln_quarter), 700.0)) \
            * (dt / 60.0)
        if dmg is not None:
            for mask, model in models:
                if mask is None:
                    dmg += model.rate(T) * dt
                else:
                    dmg[mask] += model.rate(T[mask]) * dt
        if shutdown is not None:
            stop = (dose >= shutdown[1]) if shutdown[0] == "dose" else (dmg >= shutdown[1])
            wb = np.where(stop, 0.0, wb)
        if hist is not None:
            hist.append([T[p] for p in probe])
        if i + 1 in k_snap:
            stored.append(T.copy())
            times.append(t_before + (i + 1) * dt)
    notes = list(initial.notes) if initial is not None else []
    if Tmax.max() > 100.0 and not any(n.startswith("above 100 C") for n in notes):
        msg = ("above 100 C: boiling and cavitation are not modelled, so temperature, dose "
               "and damage are not physical where this is reached")
        notes.append(msg)
        warnings.warn(f"peak temperature {Tmax.max():.0f} C; {msg}", RuntimeWarning,
                      stacklevel=2)
    return BioheatResult(np.array(times), np.array(stored), Tmax, dose, dt,
                         None if hist is None else np.array(hist),
                         T_final=T, damage=dmg, perfusion=wb / c_blood,
                         time=t_before + duration, notes=notes)


def _damage_models(medium, arrhenius):
    """[(mask or None, Arrhenius)] for bioheat()."""
    if arrhenius is None:
        return []
    if isinstance(arrhenius, Arrhenius):
        return [(None, arrhenius)]
    if not isinstance(arrhenius, dict):
        raise TypeError("arrhenius must be an Arrhenius or a dict of them")
    names = [m.name for m in medium.materials]
    out, used = [], np.zeros(medium.grid.shape, bool)
    for name, model in arrhenius.items():
        if name == "*":
            continue
        if name not in names:
            raise ValueError(f"no material named '{name}' in the medium")
        mask = medium.mask(name)
        out.append((mask, model))
        used |= mask
    if "*" in arrhenius:
        out.append((~used, arrhenius["*"]))
    return out


# --------------------------------------------------------------------------
# treatment plans


@dataclass
class Sonication:
    """One exposure of a treatment.

    q heat source at full power [W/m^3]; duration on-time [s]; power
    fraction of full power; cooling off-time after it [s]; schedule an
    optional on/off pattern within the on-time.
    """

    q: np.ndarray
    duration: float
    power: float = 1.0
    cooling: float = 0.0
    schedule: object = None
    label: str = ""


@dataclass
class Treatment:
    """Result of treat(): the final BioheatResult and per-sonication data."""

    result: BioheatResult
    log: list = field(default_factory=list)

    def lesion(self, threshold=LESION_DOSE):
        return self.result.lesion(threshold)


def treat(medium, sonications, T0=37.0, arrhenius=None, shutdown=None, max_dt=0.1,
          progress=False, **kw):
    """Run a sequence of sonications on the same tissue.

    Temperature, dose, damage and the perfusion state carry over from one
    sonication to the next. Returns a Treatment with the final state and a
    log of (label, peak temperature, lesion volume after the sonication).
    """
    state = None
    log = []
    cell = float(np.prod(medium.grid.spacing))
    for i, s in enumerate(sonications):
        sched = s.schedule
        state = bioheat(medium, s.power * np.asarray(s.q), s.duration, schedule=sched, T0=T0,
                        initial=state, arrhenius=arrhenius, shutdown=shutdown, max_dt=max_dt,
                        **kw)
        peak = float(state.T_final.max())
        if s.cooling > 0:
            state = bioheat(medium, np.zeros(medium.grid.shape), s.cooling, T0=T0,
                            initial=state, arrhenius=arrhenius, shutdown=shutdown,
                            max_dt=max_dt, **kw)
        vol = float(state.lesion().sum() * cell)
        log.append((s.label or f"sonication {i + 1}", peak, vol))
        if progress:
            print(f"{log[-1][0]}: peak {peak:.1f} C, lesion {vol * 1e9:.1f} mm^3", flush=True)
    return Treatment(state, log)


def raster(target_mask, grid, spacing, margin=0.0):
    """Focus positions [m] on a regular raster covering a target.

    Points spaced by spacing [m] that lie inside the target shrunk by
    margin [m], one row per point with grid.ndim columns in physical
    coordinates. Subtract the position of the focus of a heat source to get
    the offsets for shift().
    """
    target_mask = np.asarray(target_mask, bool)
    axes = grid.axes()
    idx = np.argwhere(target_mask)
    if idx.size == 0:
        return np.zeros((0, grid.ndim))
    pts = np.stack([axes[d][idx[:, d]] for d in range(grid.ndim)], 1)
    lo, hi = pts.min(0), pts.max(0)
    ranges = [np.arange(lo[d], hi[d] + 1e-12, spacing) for d in range(grid.ndim)]
    cand = np.stack([m.ravel() for m in np.meshgrid(*ranges, indexing="ij")], 1)
    keep = []
    for p in cand:
        ii = np.rint((p - grid.origin) / grid.spacing).astype(int)
        if np.any(ii < 0) or np.any(ii >= np.array(grid.shape)):
            continue
        if not target_mask[tuple(ii)]:
            continue
        if margin > 0:
            r = np.ceil(margin / grid.spacing).astype(int)
            sl = tuple(slice(max(i - rr, 0), i + rr + 1) for i, rr in zip(ii, r))
            if not target_mask[sl].all():
                continue
        keep.append(p)
    return np.array(keep).reshape(-1, grid.ndim)


def circles(radii, points_per_circle=None, spacing=1e-3):
    """Focus offsets [m] (x, z) on concentric circles in the focal plane,
    the trajectory of volumetric sonications (Köhler et al. 2009). In 2-D
    the circle lies in the x-y plane, so only its x extent is kept."""
    pts = [(0.0, 0.0, 0.0)]
    for r in radii:
        n = points_per_circle or max(int(np.ceil(2 * np.pi * r / spacing)), 6)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts += [(r * np.cos(t), r * np.sin(t), 0.0) for t in a]
    return np.array(pts)


def trajectory_source(q, grid, offsets):
    """Mean heat source of a focus moved quickly over offsets ((n, 3) in
    x, y, z) [m], as in volumetric sonication where the focus visits every
    point many times per second."""
    total = np.zeros(grid.shape)
    for off in offsets:
        total += shift(q, grid, off)
    return total / len(offsets)


# --------------------------------------------------------------------------
# reports and tissue changes


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


def coagulate(medium, T_max, levels=((60.0, 5.0, 1.25), (65.0, 9.0, 2.7))):
    """Medium after heating, for imaging the treated region.

    Points whose highest temperature reached a level get a backscatter and
    an attenuation multiplied by the factors of that level. The default
    levels are the changes measured in human prostate tissue heated to
    60 C (backscatter x5, attenuation x1.25) and 65 C (x9 and x2.7) by
    Worthington et al. (2002); below 55 C they saw little change. Their
    samples were held at the final temperature for 30 min and measured at
    5 MHz; a HIFU lesion forms in seconds at higher temperatures, so the
    factors are an approximation for it. Each affected material gets a
    copy named "coagulated <name>".
    """
    T_max = np.asarray(T_max, dtype=float)
    labels = medium.labels.copy()
    mats = list(medium.materials)
    for temp, f_bsc, f_att in sorted(levels):
        hot = T_max >= temp
        for i, m in enumerate(medium.materials):
            sel = hot & (medium.labels == i)
            if not sel.any():
                continue
            base = m
            name = f"coagulated {m.name} ({temp:g} C)"
            new = base.with_(name=name, alpha0=base.alpha0 * f_att,
                             echo=base.echo + 10 * np.log10(f_bsc),
                             source=base.source + "; coagulated (Worthington et al. 2002)")
            mats.append(new)
            labels[sel] = len(mats) - 1
    return Medium(medium.grid, labels, mats)

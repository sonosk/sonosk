"""Acoustic output quantities and regulatory limits for diagnostic ultrasound.

For research and teaching. The functions compute the quantities defined
in the standard from simulated or measured waveforms; they do not replace
the measurements and the review that a regulatory submission needs.

Derated (".3") values use the 0.3 dB/(cm MHz) model of IEC 62359 and the
FDA guidance. Inputs are water values on the beam axis.

References
----------
IEC 62359:2010+A1:2017. Ultrasonics - Field characterization - Test
methods for the determination of thermal and mechanical indices related
to medical diagnostic ultrasonic fields.

FDA. Marketing Clearance of Diagnostic Ultrasound Systems and Transducers.
Guidance for Industry and FDA Staff, February 2023 (FDA-2017-D-5372).
"""

import numpy as np

DERATING = 0.3  # dB / (cm MHz)

# FDA Track 3 global maximum derated values.
# ispta3 in mW/cm^2, isppa3 in W/cm^2.
FDA_LIMITS = {
    "general": {"ispta3": 720.0, "mi": 1.9, "isppa3": 190.0},
    "ophthalmic": {"ispta3": 50.0, "mi": 0.23, "ti": 1.0},
}


def derate(p, f_mhz, z_cm, alpha=DERATING):
    """Derated pressure for frequency f_mhz [MHz] at depth z_cm [cm]."""
    return np.asarray(p, dtype=float) * 10.0 ** (-alpha * f_mhz * np.asarray(z_cm) / 20.0)


def derate_intensity(i, f_mhz, z_cm, alpha=DERATING):
    """Derated intensity (or PII) at depth z_cm [cm]."""
    return np.asarray(i, dtype=float) * 10.0 ** (-alpha * f_mhz * np.asarray(z_cm) / 10.0)


def _waveform(p, dt):
    p = np.asarray(p, dtype=float)
    if p.ndim == 0 or p.shape[-1] < 2:
        raise ValueError("p must have at least two samples along its last axis")
    if not np.all(np.isfinite(p)):
        raise ValueError("p must be finite")
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt must be a positive number")
    return p


def pulse_intensity_integral(p, dt, rho=1000.0, c=1500.0):
    """Pulse intensity integral [J/m^2] of pressure waveform p [Pa].

    The time axis is the last axis of p; dt is the sample interval [s].
    Several waveforms can be given as rows of a 2-D array.
    """
    p = _waveform(p, dt)
    return np.sum(p * p, axis=-1) * dt / (rho * c)


def _crossing(e, t, level):
    # first time the non-decreasing curve e reaches level
    k = int(np.searchsorted(e, level))
    if k == 0:
        return t[0]
    return t[k - 1] + (level - e[k - 1]) * (t[k] - t[k - 1]) / (e[k] - e[k - 1])


def pulse_duration(p, dt):
    """Pulse duration [s]: 1.25 times the 10 % to 90 % interval of the
    running pulse intensity integral (IEC 62359).

    The time axis is the last axis of p; for several waveforms the result
    is an array with one value per waveform. A waveform of zeros has a
    duration of 0.
    """
    p = _waveform(p, dt)
    flat = p.reshape(-1, p.shape[-1])
    t = np.arange(flat.shape[1]) * dt
    out = np.zeros(flat.shape[0])
    for i, row in enumerate(flat):
        e = np.cumsum(row * row)
        if e[-1] <= 0:
            continue
        e = e / e[-1]
        out[i] = 1.25 * (_crossing(e, t, 0.9) - _crossing(e, t, 0.1))
    if p.ndim == 1:
        return float(out[0])
    return out.reshape(p.shape[:-1])


def isppa(p, dt, rho=1000.0, c=1500.0):
    """Pulse-average intensity [W/m^2]: PII over the pulse duration.

    The time axis is the last axis of p. Waveforms of zeros give 0.
    """
    pd = pulse_duration(p, dt)
    pii = pulse_intensity_integral(p, dt, rho, c)
    out = np.where(np.asarray(pd) > 0, pii / np.where(np.asarray(pd) > 0, pd, 1.0), 0.0)
    return float(out) if np.ndim(out) == 0 else out


def ispta(pii, prf):
    """Temporal-average intensity [W/m^2] for an unscanned beam."""
    return np.asarray(pii) * prf


def mechanical_index(z_cm, pr_mpa, f_mhz, pii=None):
    """Mechanical index.

    MI = pr.3(z_sp) / sqrt(fc), where z_sp is the point on the beam axis
    where the derated pulse intensity integral is largest.

    Parameters
    ----------
    z_cm : array
        Depths on the beam axis [cm].
    pr_mpa : array
        Peak rarefactional pressure in water at z_cm [MPa], positive.
    f_mhz : float
        Centre (acoustic working) frequency [MHz].
    pii : array, optional
        Pulse intensity integral in water at z_cm. If omitted, pr_mpa**2
        is used, which assumes the pulse shape does not change with depth.

    Returns
    -------
    mi : float
    z_sp : float
        Depth used [cm].
    """
    z = np.atleast_1d(np.asarray(z_cm, dtype=float))
    pr = np.abs(np.atleast_1d(np.asarray(pr_mpa, dtype=float)))
    if z.ndim != 1 or pr.shape != z.shape:
        raise ValueError("z_cm and pr_mpa must be 1-D arrays of the same length")
    if not (np.isfinite(f_mhz) and f_mhz > 0):
        raise ValueError("f_mhz must be a positive number")
    if pii is None:
        pii = pr ** 2
    pii = np.asarray(pii, dtype=float)
    if pii.shape != z.shape:
        raise ValueError("pii must have the shape of z_cm")
    k = int(np.argmax(derate_intensity(pii, f_mhz, z)))
    pr3 = derate(pr[k], f_mhz, z[k])
    return float(pr3 / np.sqrt(f_mhz)), float(z[k])


def w_m2_to_mw_cm2(i):
    return np.asarray(i) * 0.1


def w_m2_to_w_cm2(i):
    return np.asarray(i) * 1e-4


def _value(name, v):
    v = float(v)
    if not np.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return v


def check_limits(mi=None, ispta3=None, isppa3=None, ti=None, application="general"):
    """Screen values against the FDA Track 3 global maximum limits.

    This is a screening aid for research. It is not a regulatory
    determination: the guidance also sets measurement and reporting
    conditions that a simulation does not cover.

    Intensities are given in clinical units: ispta3 in mW/cm^2, isppa3 in
    W/cm^2. The rules follow Section 5.2.8 of the FDA guidance (February
    2023): for general use ISPTA.3 <= 720 mW/cm^2 and either MI <= 1.9 or
    ISPPA.3 <= 190 W/cm^2; for ophthalmic use ISPTA.3 <= 50 mW/cm^2,
    MI <= 0.23 and TI <= 1. For general use the guidance sets no TI limit
    but asks for an explanation of any TI above 6.

    Returns a dict name -> (value, limit, within_limit) for every value
    given, "complete" (all values the rules need were given), "ok" (complete
    and within the limits) and "notes" (list of strings).
    """
    if application not in FDA_LIMITS:
        raise ValueError(f"unknown application '{application}'")
    lim = FDA_LIMITS[application]
    given = {"mi": mi, "ispta3": ispta3, "isppa3": isppa3, "ti": ti}
    given = {k: _value(k, v) for k, v in given.items() if v is not None}

    out = {}
    for name, value in given.items():
        if name in lim:
            out[name] = (value, lim[name], value <= lim[name])

    notes = []
    ok = True
    if application == "general":
        complete = "ispta3" in out and ("mi" in out or "isppa3" in out)
        if "ispta3" in out:
            ok &= out["ispta3"][2]
        pressure = [out[k][2] for k in ("mi", "isppa3") if k in out]
        if pressure:
            ok &= any(pressure)
        if "ti" in given and given["ti"] > 6.0:
            notes.append("TI above 6: the guidance asks for an explanation")
    else:
        complete = all(k in out for k in ("ispta3", "mi", "ti"))
        for k in ("ispta3", "mi", "ti"):
            if k in out:
                ok &= out[k][2]
    if not complete:
        notes.append("not all values needed by the rules were given")
    out["complete"] = bool(complete)
    out["ok"] = bool(ok and complete)
    out["notes"] = notes
    return out

"""Acoustic output quantities and regulatory limits for diagnostic ultrasound.

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


def pulse_intensity_integral(p, dt, rho=1000.0, c=1500.0):
    """Pulse intensity integral [J/m^2] of pressure waveform p [Pa].

    The time axis is the last axis of p; dt is the sample interval [s].
    """
    p = np.asarray(p, dtype=float)
    return np.sum(p * p, axis=-1) * dt / (rho * c)


def _crossing(e, t, level):
    # first time the non-decreasing curve e reaches level
    k = int(np.searchsorted(e, level))
    if k == 0:
        return t[0]
    return t[k - 1] + (level - e[k - 1]) * (t[k] - t[k - 1]) / (e[k] - e[k - 1])


def pulse_duration(p, dt):
    """Pulse duration [s]: 1.25 times the 10 % to 90 % interval of the
    running pulse intensity integral (IEC 62359)."""
    e = np.cumsum(np.asarray(p, dtype=float) ** 2)
    if e.size == 0 or e[-1] <= 0:
        return 0.0
    e = e / e[-1]
    t = np.arange(e.size) * dt
    return 1.25 * (_crossing(e, t, 0.9) - _crossing(e, t, 0.1))


def isppa(p, dt, rho=1000.0, c=1500.0):
    """Pulse-average intensity [W/m^2] of a single waveform."""
    pd = pulse_duration(p, dt)
    if pd == 0:
        return 0.0
    return pulse_intensity_integral(p, dt, rho, c) / pd


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
    z = np.asarray(z_cm, dtype=float)
    pr = np.abs(np.asarray(pr_mpa, dtype=float))
    if pii is None:
        pii = pr ** 2
    k = int(np.argmax(derate_intensity(pii, f_mhz, z)))
    pr3 = derate(pr[k], f_mhz, z[k])
    return float(pr3 / np.sqrt(f_mhz)), float(z[k])


def w_m2_to_mw_cm2(i):
    return np.asarray(i) * 0.1


def w_m2_to_w_cm2(i):
    return np.asarray(i) * 1e-4


def check_limits(mi=None, ispta3=None, isppa3=None, ti=None, application="general"):
    """Compare values with the FDA Track 3 limits.

    Intensities are given in clinical units: ispta3 in mW/cm^2, isppa3 in
    W/cm^2. For general use either MI or ISPPA.3 must be within its limit.

    Returns a dict name -> (value, limit, within_limit) and an overall flag
    under the key "ok".
    """
    if application not in FDA_LIMITS:
        raise ValueError(f"unknown application '{application}'")
    lim = FDA_LIMITS[application]
    given = {"mi": mi, "ispta3": ispta3, "isppa3": isppa3, "ti": ti}

    out = {}
    for name, value in given.items():
        if value is None or name not in lim:
            continue
        out[name] = (float(value), lim[name], float(value) <= lim[name])

    ok = True
    if "ispta3" in out:
        ok &= out["ispta3"][2]
    if application == "general":
        pressure = [out[k][2] for k in ("mi", "isppa3") if k in out]
        if pressure:
            ok &= any(pressure)
    else:
        for k in ("mi", "ti"):
            if k in out:
                ok &= out[k][2]
    out["ok"] = bool(ok)
    return out

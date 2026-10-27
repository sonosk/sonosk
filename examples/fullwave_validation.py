"""Checks of the full-wave solver against closed-form results.

Three 1-D runs of sonosk.fullwave.simulate, each compared with theory:

- absorption and dispersion: a broadband pulse in a medium with
  alpha0 = 0.75 dB/(cm MHz^1.5), y = 1.5. The attenuation measured between
  two sensors follows alpha0 f^y and the phase speed follows the
  Kramers-Kronig relation of the fractional Laplacian model (Treeby and Cox
  2010);
- nonlinear propagation: a 1 MHz plane wave of 1 MPa in a lossless fluid
  with B/A = 5. The first three harmonics agree with the Fubini solution
  up to half the shock distance;
- an interface between two fluids: the reflected amplitude equals
  (Z2 - Z1) / (Z2 + Z1).
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
from sonosk import fullwave as fw, imaging
from sonosk.medium import attenuation_np


def bessel_j(n, x):
    """J_n(x) from its integral representation."""
    tau = np.linspace(0, np.pi, 4001)
    return np.trapezoid(np.cos(n * tau - x * np.sin(tau)), tau) / np.pi


def absorption():
    dx = 0.05e-3
    g = sk.Grid(3000, dx)
    a0, y, c = 0.75, 1.5, 1540.0
    med = sk.Medium.homogeneous(g, sk.Material("tissue", c, 1000, alpha0=a0, y=y))
    src = fw.Source([[100]], fw.gaussian_pulse(imaging.Pulse(3e6, 0.8)))
    x1, x2 = 400, 1400
    res = fw.simulate(med, src, 120e-6, sensor=fw.Sensor([[x1], [x2]]))
    n = 1 << 16
    P = np.fft.rfft(res.p, n=n, axis=1)
    f = np.fft.rfftfreq(n, res.dt)
    d = (x2 - x1) * dx
    band = (f > 0.8e6) & (f < 5.5e6)
    alpha = np.log(np.abs(P[0, band]) / np.abs(P[1, band])) / d
    phase = np.unwrap(np.angle(P[1]) - np.angle(P[0]))[band]
    c_ph = 2 * np.pi * f[band] * d / (-phase)
    return f[band], alpha, attenuation_np(a0, f[band], y), c_ph, (c, a0, y)


def fubini():
    c, rho, ba, f, p0 = 1500.0, 1000.0, 5.0, 1e6, 1e6
    beta = 1 + ba / 2
    x_shock = rho * c ** 3 / (beta * 2 * np.pi * f * p0)
    dx = 0.04e-3
    n = int(0.1 / dx)
    g = sk.Grid(n, dx)
    med = sk.Medium.homogeneous(g, sk.Material("fluid", c, rho, ba=ba))
    src = fw.Source([[20]], fw.continuous_wave(f, 3), weight=p0)
    res = fw.simulate(med, src, 0.1 / c + 14 / f, nonlinear=True, cw=[f, 2 * f, 3 * f],
                      cw_cycles=8, pml=40)
    x = (np.arange(n) - 20) * dx
    keep = (x > 1e-3) & (x < 0.085)
    sigma = x[keep] / x_shock
    meas = [np.abs(res.cw[k * f][keep]) / p0 for k in (1, 2, 3)]
    theory = [np.array([2 * bessel_j(k, k * s) / (k * s) for s in sigma]) for k in (1, 2, 3)]
    return sigma, meas, theory


def reflection():
    dx = 0.05e-3
    g = sk.Grid(2000, dx)
    lab = np.zeros(2000, int)
    lab[1000:] = 1
    mats = [sk.Material("a", 1500, 1000), sk.Material("b", 1800, 1600)]
    med = sk.Medium(g, lab, mats)
    res = fw.simulate(med, fw.Source([[300]], fw.tone_burst(1e6, 3)), 45e-6,
                      sensor=fw.Sensor([[600]]))
    p, t = res.p[0], res.t
    incident = np.abs(p[t < 15e-6]).max()
    reflected = np.abs(p[t > 30e-6]).max()
    z1, z2 = 1500 * 1000, 1800 * 1600
    return reflected / incident, (z2 - z1) / (z2 + z1)


def compute():
    return absorption(), fubini(), reflection()


def figure():
    (f, a_meas, a_th, c_ph, (c0, a0, y)), (sigma, meas, theory), (r_meas, r_th) = compute()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    ax[0].plot(f / 1e6, a_th, "k-", label="alpha0 f^y")
    ax[0].plot(f[::40] / 1e6, a_meas[::40], "o", ms=4, label="k-space solver")
    ax[0].set_xlabel("Frequency [MHz]")
    ax[0].set_ylabel("Attenuation [Np/m]")
    ax[0].set_title("Power-law absorption, y = 1.5")
    ax[0].legend()
    ax0b = ax[0].twinx()
    kk = fw.dispersion_speed(c0, a0, y, f, f[0]) - fw.dispersion_speed(c0, a0, y, f[0], f[0])
    ax0b.plot(f / 1e6, c_ph - c_ph[0], "C2.", ms=2)
    ax0b.plot(f / 1e6, kk, "C2-", lw=1)
    ax0b.set_ylabel("Phase speed change [m/s] (green)")
    for k, (m, th) in enumerate(zip(meas, theory)):
        ax[1].plot(sigma, th, "k-", lw=1)
        ax[1].plot(sigma[::60], m[::60], "o", ms=4, label=f"harmonic {k + 1}")
    ax[1].set_xlabel("Distance / shock distance")
    ax[1].set_ylabel("Amplitude / source amplitude")
    ax[1].set_title("Nonlinear plane wave against Fubini")
    ax[1].legend()
    ax[2].bar(["solver", "(Z2-Z1)/(Z2+Z1)"], [r_meas, r_th], color=["C0", "0.6"])
    ax[2].set_title(f"Reflection: {r_meas:.4f} vs {r_th:.4f}")
    ax[2].set_ylabel("Reflected / incident amplitude")
    return fig


if __name__ == "__main__":
    (f, a_meas, a_th, c_ph, _), (sigma, meas, theory), (r_meas, r_th) = compute()
    print("largest attenuation error [%]", 100 * np.max(np.abs(a_meas / a_th - 1)))
    print("largest Fubini error, harmonics 1-3:",
          [float(np.max(np.abs(m - t))) for m, t in zip(meas, theory)])
    print("reflection", r_meas, "theory", r_th)
    figure()
    plt.show()

"""Nonlinear propagation of a focused beam and its effect on heating.

A 1 MHz source focused at 40 mm (a 2-D arc of 30 mm aperture) sends a
continuous wave through 10 mm of water into myometrium (sound speed,
attenuation and B/A from sonosk.tissues). The k-space solver runs twice,
without and with the nonlinear terms. At high pressure the waveform at the
focus steepens: the peak positive pressure grows, the peak negative
pressure shrinks, and energy moves into harmonics that tissue absorbs more
strongly, so the heat source at the focus is larger than linear theory
predicts.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
from sonosk import fullwave as fw, phantom, thermal, tissues

F = 1e6
FOCAL = 40e-3
APERTURE = 30e-3
AMPLITUDE = 2.0e6      # source pressure [Pa]


def compute():
    dx = 0.075e-3
    g = sk.Grid.centered((481, 667), dx, depth_start=0.0)
    lab = phantom.paint(np.zeros(g.shape, int), phantom.halfspace(g, 10e-3), 1)
    med = sk.Medium(g, lab, [tissues.material("water"), tissues.material("myometrium")])

    xs = np.arange(-APERTURE / 2, APERTURE / 2 + dx / 2, dx)
    zs = 1e-3 + FOCAL - np.sqrt(FOCAL ** 2 - xs ** 2)
    pts = np.stack([xs, np.zeros_like(xs), zs], 1)
    idx = np.unique(fw.grid_index(g, pts), axis=0)
    src = fw.Source(idx, fw.continuous_wave(F, 3), weight=AMPLITUDE)
    focus = g.index((0.0, 1e-3 + FOCAL))
    sens = fw.Sensor([focus])
    freqs = [F * k for k in range(1, 6)]
    t_end = (FOCAL + 12e-3) / 1500 + 14 / F
    runs = {}
    for nl in (False, True):
        runs[nl] = fw.simulate(med, src, t_end, sensor=sens, nonlinear=nl, cw=freqs,
                               cw_cycles=6, f_ref=F, precision="single", pml=30)
    q_lin = thermal.heat_source_harmonics({F: runs[False].cw[F]}, med)
    q_nl = thermal.heat_source_harmonics(runs[True].cw, med)
    return g, med, runs, q_lin, q_nl, focus


def figure():
    g, med, runs, q_lin, q_nl, focus = compute()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    n_per = int(round(1 / (F * runs[True].dt)))
    for nl, name in ((False, "linear"), (True, "nonlinear")):
        r = runs[nl]
        p = r.p[0, -2 * n_per:] / 1e6
        t = (np.arange(p.size)) * r.dt * 1e6
        ax[0].plot(t, p, label=f"{name}: +{p.max():.1f} / {p.min():.1f} MPa")
    ax[0].set_xlabel("Time [us]")
    ax[0].set_ylabel("Pressure at the focus [MPa]")
    ax[0].set_title("Focal waveform")
    ax[0].legend()

    z = g.axis(1) * 1e3
    i = focus[0]
    for k, f in enumerate(sorted(runs[True].cw)):
        ax[1].plot(z, np.abs(runs[True].cw[f][i]) / 1e6, label=f"{k + 1} MHz")
    ax[1].set_xlabel("Depth on axis [mm]")
    ax[1].set_ylabel("Amplitude [MPa]")
    ax[1].set_title("Harmonics on the axis (nonlinear run)")
    ax[1].legend()

    ax[2].plot(z, q_lin[i] / 1e6, label="linear")
    ax[2].plot(z, q_nl[i] / 1e6, label="nonlinear, five harmonics")
    ax[2].set_xlabel("Depth on axis [mm]")
    ax[2].set_ylabel("Heat source [W/cm^3]")
    ax[2].set_title(f"Heating at the focus x{q_nl[focus] / q_lin[focus]:.2f}")
    ax[2].legend()
    return fig


if __name__ == "__main__":
    g, med, runs, q_lin, q_nl, focus = compute()
    for nl in (False, True):
        p = runs[nl].p[0]
        print("nonlinear" if nl else "linear", "p+ %.2f MPa, p- %.2f MPa" % (p.max() / 1e6,
                                                                          p.min() / 1e6))
    print("heat source ratio at the focus", q_nl[focus] / q_lin[focus])
    figure()
    plt.show()

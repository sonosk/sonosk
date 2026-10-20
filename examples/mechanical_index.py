"""Mechanical index of a focused bowl from its on-axis pressure.

A 3 MHz bowl, 15 mm aperture radius and 40 mm radius of curvature, is
driven so that the peak pressure in water is 2 MPa. The on-axis field is
the O'Neil solution, derated at 0.3 dB/(cm MHz).
"""

import matplotlib.pyplot as plt
import numpy as np

from sonosk import analytic, safety

F = 3e6
A = 15e-3
ROC = 40e-3
P_PEAK = 2.0  # MPa in water


def compute():
    z = np.linspace(2e-3, 80e-3, 2000)
    p = analytic.bowl_axial(z, A, ROC, F)
    pr = P_PEAK * p / p.max()
    z_cm = z * 100.0
    mi, z_sp = safety.mechanical_index(z_cm, pr, F / 1e6)
    pr3 = safety.derate(pr, F / 1e6, z_cm)
    return z_cm, pr, pr3, mi, z_sp


def figure():
    z_cm, pr, pr3, mi, z_sp = compute()

    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    ax.plot(z_cm, pr, label="water")
    ax.plot(z_cm, pr3, label="derated 0.3 dB/(cm MHz)")
    ax.axvline(z_sp, color="k", lw=0.8, ls="--")
    ax.text(0.03, 0.95, f"MI = {mi:.2f} at z = {z_sp:.2f} cm", transform=ax.transAxes, va="top")
    ax.set_xlabel("Depth [cm]")
    ax.set_ylabel("Peak rarefactional pressure [MPa]")
    ax.set_title("3 MHz focused bowl, a = 15 mm, R = 40 mm")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return fig


if __name__ == "__main__":
    z_cm, pr, pr3, mi, z_sp = compute()
    print(f"MI = {mi:.3f} at z = {z_sp:.2f} cm")
    print(safety.check_limits(mi=mi, ispta3=300.0))
    figure()
    plt.show()

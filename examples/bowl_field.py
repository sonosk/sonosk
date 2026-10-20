"""Pressure field of a focused bowl and comparison with O'Neil's solution.

A 1 MHz bowl with 20 mm aperture radius and 50 mm radius of curvature in
water. Left: field amplitude in the x-z plane from the Rayleigh integral.
Right: on-axis amplitude against the closed-form result.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import analytic, field, transducer

F = 1e6
C = 1500.0
A = 20e-3
ROC = 50e-3


def compute():
    src = transducer.bowl(A, ROC, dx=C / F / 6)
    g = sk.Grid.centered((121, 181), 0.25e-3, depth_start=20e-3)
    p = field.rayleigh_grid(src, g, F, C)
    z = np.linspace(10e-3, 90e-3, 600)
    pz = np.abs(field.rayleigh(src, np.stack([0 * z, 0 * z, z], 1), F, C))
    return g, p, z, pz, analytic.bowl_axial(z, A, ROC, F, C)


def figure():
    g, p, z, pz, pa = compute()
    x, zz = g.axes()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1.4]})
    h = splt.field_map(p, x, zz, ax=ax[0], title="|p| [dB]")
    fig.colorbar(h, ax=ax[0], shrink=0.8)
    ax[1].plot(z * 1e3, pz, label="Rayleigh integral")
    ax[1].plot(z * 1e3, pa, "--", label="O'Neil (1949)")
    ax[1].set_xlabel("z [mm]")
    ax[1].set_ylabel("|p| / (rho c u0)")
    ax[1].set_title(f"On axis, focal gain kh = {analytic.focal_gain(A, ROC, F, C):.1f}")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    return fig


if __name__ == "__main__":
    figure()
    plt.show()

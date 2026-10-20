"""CW beams of a linear array: focused, focused and steered, and with a
Hann apodization.

128 elements, 0.3 mm pitch, 5 mm high with a 20 mm elevation lens, at
5 MHz in a medium with 1540 m/s and 0.5 dB/(cm MHz). Field amplitude in
the x-z plane, normalised in each panel.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import field, transducer

F = 5e6
C = 1540.0


def compute():
    arr = transducer.linear_array(128, 0.3e-3, height=5e-3, elev_focus=20e-3, dx=0.15e-3)
    g = sk.Grid.centered((121, 151), 0.2e-3, depth_start=2e-3)
    out = []
    arr.focus((0, 0, 20e-3), C)
    out.append(("focus 20 mm", field.rayleigh_grid(arr, g, F, C, alpha0=0.5)))
    fp = 20e-3 * np.array([np.sin(np.deg2rad(20)), 0, np.cos(np.deg2rad(20))])
    arr.focus(fp, C)
    out.append(("focus 20 mm, steered 20 deg", field.rayleigh_grid(arr, g, F, C, alpha0=0.5)))
    arr.focus((0, 0, 20e-3), C).set_apodization(transducer.apodization(128, "hann"))
    out.append(("focus 20 mm, Hann apodization", field.rayleigh_grid(arr, g, F, C, alpha0=0.5)))
    return g, out


def figure():
    g, out = compute()
    x, z = g.axes()
    fig, ax = plt.subplots(1, 3, figsize=(11, 4.2), constrained_layout=True)
    for a, (title, p) in zip(ax, out):
        h = splt.field_map(p, x, z, ax=a, title=title, dynamic_range=40)
    fig.colorbar(h, ax=ax, shrink=0.8, label="dB")
    return fig


if __name__ == "__main__":
    figure()
    plt.show()

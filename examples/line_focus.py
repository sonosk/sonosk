"""Field of a cylindrically focused source, the geometry of line-focused
aesthetic HIFU transducers.

A 4 MHz source with 8 mm chord, 12 mm length and 6 mm radius of curvature
focuses to a line 6 mm under the face. The values are an example, not the
data of a commercial device. Left: x-z plane through the middle of the
line. Right: y-z plane through the focus line.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import field, transducer

F = 4e6
C = 1540.0


def compute():
    src = transducer.cylindrical(8e-3, 12e-3, 6e-3, dx=0.08e-3)
    gx = sk.Grid.centered((121, 121), 0.06e-3, depth_start=1e-3)
    pxz = field.rayleigh_grid(src, gx, F, C, alpha0=0.5)
    y = np.linspace(-9e-3, 9e-3, 121)
    z = gx.axis(1)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    pyz = field.rayleigh(src, np.stack([0 * yy.ravel(), yy.ravel(), zz.ravel()], 1),
                         F, C, alpha0=0.5).reshape(yy.shape)
    return gx, pxz, y, pyz


def figure():
    gx, pxz, y, pyz = compute()
    x, z = gx.axes()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    splt.field_map(pxz, x, z, ax=ax[0], title="x-z plane", dynamic_range=30)
    h = splt.field_map(pyz, y, z, ax=ax[1], title="y-z plane", dynamic_range=30)
    ax[1].set_xlabel("y [mm]")
    fig.colorbar(h, ax=ax, shrink=0.8, label="dB")
    return fig


if __name__ == "__main__":
    figure()
    plt.show()

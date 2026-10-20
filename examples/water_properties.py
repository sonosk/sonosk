"""Sound speed and density of water between 0 and 95 C."""

import matplotlib.pyplot as plt
import numpy as np

from sonosk import water


def figure():
    t = np.linspace(0.0, 95.0, 400)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax[0].plot(t, water.sound_speed(t), color="C0")
    ax[0].set_xlabel("Temperature [C]")
    ax[0].set_ylabel("Sound speed [m/s]")
    ax[0].set_title("Marczak (1997)")

    ax[1].plot(t, water.density(t), color="C1")
    ax[1].set_xlabel("Temperature [C]")
    ax[1].set_ylabel("Density [kg/m$^3$]")
    ax[1].set_title("Kell (1975)")

    for a in ax:
        a.grid(alpha=0.3)
    return fig


if __name__ == "__main__":
    figure()
    plt.show()

"""Design of a 5.7 MHz thickness-mode element with the Mason model.

A 0.4 mm piezoceramic plate (10 mm x 10 mm) on a 3 MRayl backing radiates
into water, without and with a quarter-wave matching layer whose
impedance follows Desilets et al. (1978). The plate constants are typical
of a soft PZT and are only an example; use the data sheet of the material
you build with.
"""

import matplotlib.pyplot as plt
import numpy as np

from sonosk import stack

PLATE = stack.Piezo(thickness=0.4e-3, c=4560.0, rho=7500.0, kt=0.5, eps_r=1470.0,
                    area=1e-4)
F = np.linspace(1e6, 10e6, 2000)


def compute():
    plain = stack.Stack(PLATE, backing=3e6, load=1.5e6)
    zm = stack.matching_impedance(PLATE.impedance, 1.5e6)
    cm = 2500.0
    lay = stack.Layer(stack.quarter_wave(cm, PLATE.f_half_wave), cm, zm / cm)
    matched = stack.Stack(PLATE, backing=3e6, layers=[lay], load=1.5e6)
    return {"no matching layer": plain, f"one layer, {zm / 1e6:.1f} MRayl": matched}


def figure():
    stacks = compute()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for name, s in stacks.items():
        z = s.impedance(F)
        ax[0].semilogy(F / 1e6, np.abs(z), label=name)
        h = s.pulse_echo(F)
        fc, bw = stack.bandwidth(F, h)
        ax[1].plot(F / 1e6, 20 * np.log10(np.abs(h)),
                   label=f"{name}: {fc / 1e6:.2f} MHz, {100 * bw:.0f} %")
    ax[0].set_xlabel("Frequency [MHz]")
    ax[0].set_ylabel("|Z| [ohm]")
    ax[0].set_title("Electrical impedance")
    ax[1].set_xlabel("Frequency [MHz]")
    ax[1].set_ylabel("Pulse-echo response [dB]")
    ax[1].set_ylim(-40, 2)
    ax[1].set_title("Two-way response, -6 dB bandwidth")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    return fig


if __name__ == "__main__":
    for name, s in compute().items():
        fc, bw = stack.bandwidth(F, s.pulse_echo(F))
        print(f"{name}: centre {fc / 1e6:.2f} MHz, bandwidth {100 * bw:.0f} %")
    figure()
    plt.show()

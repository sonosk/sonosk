"""Aberration study with full-wave pulse-echo data (2-D).

This example is about the delays that fat and glandular tissue put on the
echoes, not about realistic lesion appearance: in 2-D every inclusion is
an infinite cylinder, so a cyst wall reflects and focuses coherently and
fills the cyst with reverberation that a 3-D cyst would not have. The
fast imaging model shows the same kind of cyst as a dark hole.

The fast imaging model follows straight rays. Here the echo data are
computed with the k-space pseudospectral solver (sonosk.fullwave),
which includes refraction, diffraction, multiple scattering and
power-law absorption. Speckle comes from small random variations of
density whose strength follows the echo level of each tissue
(sonosk.phantom.fluctuations); the speed map is left smooth so that the
aberration comes from the tissue layers alone. The elements transmit and
receive a normal velocity, which gives them the directivity of a baffled
element.

A 96-element linear array at 4 MHz on a breast phantom with a cyst and a
carcinoma sends five plane waves. The recorded signals are beamformed
twice, with 1540 m/s and through the true speed map (an oracle case),
smoothed over 0.5 mm since sharp edges such as the thin ligaments would
put steps in the delays. The full-wave data contain refraction that the
straight-ray delays do not model. The script measures the depth of the
pectoralis fascia in both images against its true depth: the error left
after correction is the part of the aberration that is not a delay error
along straight rays.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import fullwave, imaging, organs, phantom, transducer

F0 = 4e6
C_SCANNER = 1540.0


def smoothed(medium, sigma):
    """Copy of a medium with its speed map smoothed by a Gaussian of width sigma."""
    g = medium.grid
    c = np.asarray(medium.c, dtype=float)
    k = [2 * np.pi * np.fft.fftfreq(n, d) for n, d in zip(g.shape, g.spacing)]
    kk = np.meshgrid(*k, indexing="ij")
    w = np.exp(-0.5 * sigma ** 2 * sum(q ** 2 for q in kk))
    out = sk.Medium(g, medium.labels, medium.materials)
    out.c = np.real(np.fft.ifftn(np.fft.fftn(c) * w))
    return out


def compute(seed=11):
    g = sk.Grid.centered((321, 341), 0.1e-3, depth_start=0.0)
    ph = organs.breast(g, lesions=[organs.Cyst((-6e-3, 0.0, 14e-3), a=3e-3, b=2.5e-3),
                                   organs.Carcinoma((7e-3, 0.0, 16e-3), radius=3e-3)],
                       fat=4e-3, gland=14e-3, ligaments=True, rng=seed)
    med = phantom.fluctuations(ph.medium, level=0.01, speed=0.0, rng=seed)
    arr = transducer.linear_array(96, 0.3e-3, height=4e-3, dx=0.1e-3)
    pulse = imaging.Pulse(F0, 0.7)
    txs = [imaging.Transmit.plane(a) for a in imaging.plane_wave_angles(5, np.deg2rad(12))]
    rfs = []
    for tx in txs:
        rf, fs = fullwave.pulse_echo(med, arr, tx, pulse, C_SCANNER, precision="single")
        rfs.append(rf)
    x = np.arange(-14e-3, 14e-3, 0.08e-3)
    z = np.arange(3e-3, 33e-3, 0.08e-3)
    nominal = imaging.compound(rfs, fs, arr, txs, x, z, C_SCANNER, f_number=1.4)
    aware = imaging.compound(rfs, fs, arr, txs, x, z, C_SCANNER, f_number=1.4,
                             medium=smoothed(ph.medium, 0.5e-3))
    nominal = imaging.tgc(nominal, z, 0.75, F0)
    aware = imaging.tgc(aware, z, 0.75, F0)
    stats = _fascia_error(ph.medium, x, z, {"1540 m/s": nominal, "speed map": aware})
    return ph, x, z, nominal, aware, stats


def _fascia_error(med, x, z, images):
    """Mean depth error [mm] of the top of the pectoralis over |x| < 10 mm."""
    g = med.grid
    musc = med.mask("pectoralis")
    cols = np.where(np.abs(x) < 10e-3)[0]
    true = []
    for i in cols:
        ix = g.index((x[i], 0.0))[0]
        true.append(g.axis(1)[np.argmax(musc[ix])])
    true = np.array(true)
    out = {}
    for name, img in images.items():
        env = np.abs(img)
        found = []
        for k, i in enumerate(cols):
            win = (z > true[k] - 3e-3) & (z < true[k] + 3e-3)
            found.append(z[win][np.argmax(env[win, i])])
        out[f"fascia depth error, {name} [mm]"] = float(np.mean(np.array(found) - true) * 1e3)
    return out


def figure():
    ph, x, z, nominal, aware, stats = compute()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    splt.property_map(ph.medium, "c", ax=ax[0], title="Sound speed", label="m/s")
    ref = np.abs(aware).max()
    e = list(stats.values())
    splt.bmode(nominal, x, z, 50, ax=ax[1], reference=ref,
               title=f"2-D full-wave, beamformed at 1540 m/s\nfascia depth error {e[0]:+.2f} mm")
    splt.bmode(aware, x, z, 50, ax=ax[2], reference=ref,
               title=f"2-D full-wave, true speed map (oracle)\nfascia depth error {e[1]:+.2f} mm")
    for a in ax:
        a.set_xlim(-14, 14)
        a.set_ylim(33, 0)
    return fig


if __name__ == "__main__":
    ph, x, z, nominal, aware, stats = compute()
    for k, v in stats.items():
        print(f"{k:<40s}{v:8.3f}")
    figure()
    plt.show()

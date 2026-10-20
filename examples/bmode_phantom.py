"""B-mode image of a test phantom with plane-wave compounding.

The phantom has a background of fully developed speckle, an anechoic cyst,
a -10 dB lesion, a +6 dB lesion and a row of wire targets. A 128-element
linear array at 5 MHz sends 9 plane waves between -12 and +12 degrees.
The table lists the quality measures of the image.

The material values are generic soft-tissue numbers for the example.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import imaging, phantom, quality, transducer

F0 = 5e6
C = 1540.0
CYST = (-5e-3, 0, 16e-3)
DARK = (5e-3, 0, 16e-3)
BRIGHT = (0.0, 0, 29e-3)
R = 2.5e-3


def compute(seed=0):
    g = sk.Grid.centered((241, 401), 0.1e-3, depth_start=4e-3)
    mats = [sk.Material("background", C, 1000, alpha0=0.5, echo=0.0),
            sk.Material("cyst", C, 1000, alpha0=0.5, echo=-np.inf),
            sk.Material("hypoechoic", C, 1000, alpha0=0.5, echo=-10.0),
            sk.Material("hyperechoic", C, 1000, alpha0=0.5, echo=6.0)]
    lab = np.zeros(g.shape, int)
    lab = phantom.paint(lab, phantom.sphere(g, CYST, R), 1)
    lab = phantom.paint(lab, phantom.sphere(g, DARK, R), 2)
    lab = phantom.paint(lab, phantom.sphere(g, BRIGHT, R), 3)
    med = sk.Medium(g, lab, mats)

    pos, amp = phantom.scatterers(med, density=60, rng=seed)
    wires, wamp = phantom.point_targets([(x, 0, 38e-3) for x in (-6e-3, -3e-3, 0, 3e-3, 6e-3)],
                                        amplitude=30.0)
    pos = np.vstack([pos, wires])
    amp = np.concatenate([amp, wamp])

    arr = transducer.linear_array(128, 0.3e-3, height=5e-3, dx=0.15e-3)
    pulse = imaging.Pulse(F0, 0.6)
    txs = [imaging.Transmit.plane(a) for a in imaging.plane_wave_angles(9, np.deg2rad(24))]
    rfs = [imaging.simulate(arr, tx, pos, amp, pulse, C, alpha0=0.5)[0] for tx in txs]
    x = np.arange(-11.5e-3, 11.5e-3, 0.075e-3)
    z = np.arange(5e-3, 43e-3, 0.075e-3)
    img = imaging.compound(rfs, 8 * F0, arr, txs, x, z, C, f_number=1.5)

    # depth gain for display: undo the mean attenuation of the round trip
    a = sk.medium.attenuation_np(0.5, F0)
    img = img * np.exp(2 * a * z)[:, None]

    env = np.abs(img)
    zz, xx = np.meshgrid(z, x, indexing="ij")

    def disc(c, r):
        return np.hypot(xx - c[0], zz - c[2]) < r

    near = (np.abs(zz - 16e-3) < 3.5e-3) & ~disc(CYST, R + 1e-3) & ~disc(DARK, R + 1e-3) \
        & (np.abs(xx) < 9e-3)
    stats = {
        "speckle SNR": quality.speckle_snr(env[near]),
        "cyst contrast [dB]": quality.contrast_db(env[disc(CYST, 0.8 * R)], env[near]),
        "cyst gCNR": quality.gcnr(env[disc(CYST, 0.8 * R)], env[near]),
        "-10 dB lesion [dB]": quality.contrast_db(env[disc(DARK, 0.8 * R)], env[near]),
        "-10 dB lesion gCNR": quality.gcnr(env[disc(DARK, 0.8 * R)], env[near]),
    }
    iz = np.argmin(np.abs(z - 38e-3))
    win = slice(max(iz - 40, 0), iz + 40)
    k = np.argmin(np.abs(x))
    sub = env[win, :]
    jz = np.argmax(sub[:, k])
    stats["wire lateral -6 dB [mm]"] = 1e3 * quality.width(sub[jz, k - 25:k + 26], x[k - 25:k + 26])
    stats["wire axial -6 dB [mm]"] = 1e3 * quality.width(sub[:, k], z[win])
    return g, med, x, z, img, stats


def figure():
    g, med, x, z, img, stats = compute()
    fig, ax = plt.subplots(1, 3, figsize=(12, 5.0), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1, 0.8]})
    xg, zg = g.axes()
    splt.labels_map(med.labels, xg, zg, names=[m.name for m in med.materials], ax=ax[0],
                    title="Phantom")
    splt.bmode(img, x, z, dynamic_range=55, ax=ax[1], title="B-mode, 9 plane waves")
    ax[2].axis("off")
    rows = [f"{k:<26s}{v:7.2f}" for k, v in stats.items()]
    ax[2].text(0.0, 0.98, "\n".join(rows), family="monospace", va="top", fontsize=9)
    return fig


if __name__ == "__main__":
    *_, stats = compute()
    for k, v in stats.items():
        print(f"{k:<26s}{v:8.3f}")
    figure()
    plt.show()

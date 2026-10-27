"""Transrectal image of the prostate with a tumour and calcifications.

The phantom (sonosk.organs.prostate) is a transverse section through a
25 mL gland seen from the rectum: rectal wall, periprostatic fat,
peripheral zone, transition zone around the urethra, anterior
fibromuscular stroma, two small calcifications in the transition zone and
a hypoechoic tumour in the right peripheral zone.

The probe is an endocavity convex array of 128 elements on a 10 mm radius
at 6.5 MHz. Five diverging waves, from virtual sources at and beside the
centre of curvature, are compounded, and the image is shown as a sector.
The calcifications cast acoustic shadows because their attenuation is
several times that of the gland.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import imaging, organs, phantom, quality, transducer

F0 = 6.5e6
C_SCANNER = 1540.0
RADIUS = 10e-3
HALF_ANGLE = np.deg2rad(68)
TUMOUR = (9e-3, 0.0, 11e-3)


def compute(seed=2):
    g = sk.Grid.centered((561, 451), 0.1e-3, depth_start=0.0)
    ph = organs.prostate(
        g,
        tumors=[organs.Tumor(TUMOUR, radius=3.5e-3, roughness=0.2, tissue="prostate carcinoma",
                             label="tumour")],
        calcifications=[(-3e-3, 0.0, 17.5e-3), (2.5e-3, 0.0, 18.5e-3)],
        rng=seed)
    med = ph.medium
    pos, amp = phantom.scatterers(med, density=40, rng=seed)
    # specular echoes of the interfaces, scaled like the speckle (see the docstring
    # of phantom.interface_scatterers)
    pos_i, amp_i = phantom.interface_scatterers(med, gain=30 * np.sqrt(40 / 50)
                                                * 0.1e-3 / 0.1e-3)
    pos, amp = np.concatenate([pos, pos_i]), np.concatenate([amp, amp_i])

    n = 128
    pitch = 2 * HALF_ANGLE * RADIUS / n
    arr = transducer.convex_array(n, pitch, RADIUS, height=4e-3, dx=0.1e-3)
    pulse = imaging.Pulse(F0, 0.65)
    txs = [imaging.Transmit.diverging((xs, 0.0, -RADIUS)) for xs in (-2e-3, -1e-3, 0, 1e-3, 2e-3)]
    rfs = []
    for i, tx in enumerate(txs):
        rf, fs = imaging.simulate(arr, tx, pos, amp, pulse, C_SCANNER, medium=med, depth=45e-3,
                                  interfaces=True)
        rfs.append(imaging.add_noise(rf, 90, rng=i))

    x = np.arange(-28e-3, 28e-3, 0.1e-3)
    z = np.arange(0.5e-3, 44e-3, 0.1e-3)
    img = imaging.compound(rfs, fs, arr, txs, x, z, C_SCANNER, f_number=0.7)
    img = imaging.tgc(img, z, 0.7, F0)
    sector = imaging.sector_mask(x, z, (0.0, -RADIUS), HALF_ANGLE - np.deg2rad(2), RADIUS,
                                 45e-3 + RADIUS)
    img = np.where(sector, img, np.nan)

    env = np.abs(img)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    r = np.hypot(xx - TUMOUR[0], zz - TUMOUR[2])
    inside = r < 2.5e-3
    around = (r > 5e-3) & (r < 7e-3) & np.isfinite(env)
    stats = {"tumour contrast [dB]": quality.contrast_db(env[inside], env[around]),
             "tumour gCNR": quality.gcnr(env[inside], env[around])}
    return ph, x, z, img, stats


def figure():
    ph, x, z, img, stats = compute()
    med = ph.medium
    xg, zg = med.grid.axes()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    splt.labels_map(med.labels, xg, zg, names=[m.name for m in med.materials], ax=ax[0],
                    title="Prostate phantom (transverse)")
    splt.bmode(img, x, z, 55, ax=ax[1], title="Transrectal B-mode, 6.5 MHz convex array")
    for a in ax:
        a.set_xlim(-28, 28)
        a.set_ylim(44, 0)
    return fig


if __name__ == "__main__":
    ph, x, z, img, stats = compute()
    print(ph.description)
    for k, v in stats.items():
        print(f"{k:<24s}{v:8.2f}")
    figure()
    plt.show()

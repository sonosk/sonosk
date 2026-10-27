"""Transabdominal image of the uterus with an intramural fibroid.

The phantom (sonosk.organs.uterus) is a sagittal section: skin,
subcutaneous fat, rectus muscle, a full bladder, and behind it an
anteverted uterus with myometrium, endometrium, cervix and a fibroid of
24 mm diameter. The full bladder is the acoustic window, as in a clinical
pelvic scan.

A convex array of 128 elements on a 60 mm radius at 3.5 MHz sends three
diverging waves from virtual sources near its centre of curvature. The
image is formed on a Cartesian grid inside the sector. The urine in the
bladder attenuates almost nothing, so the tissue behind it is brighter
than the rest of the image at the same depth (posterior enhancement).
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import imaging, organs, phantom, quality, transducer

F0 = 3.5e6
C_SCANNER = 1540.0
RADIUS = 60e-3
FIBROID = (12e-3, 0.0, 96e-3)


def compute(seed=5):
    g = sk.Grid.centered((651, 651), 0.2e-3, depth_start=0.0)
    ph = organs.uterus(
        g, fibroids=[organs.Tumor(FIBROID, radius=12e-3, roughness=0.06, tissue="fibroid",
                                  label="fibroid")],
        rng=seed)
    med = ph.medium
    pos, amp = phantom.scatterers(med, density=10, rng=seed)
    # specular echoes of the interfaces, scaled like the speckle (see the docstring
    # of phantom.interface_scatterers)
    pos_i, amp_i = phantom.interface_scatterers(med, gain=30 * np.sqrt(10 / 50)
                                                * 0.2e-3 / 0.1e-3)
    pos, amp = np.concatenate([pos, pos_i]), np.concatenate([amp, amp_i])

    n = 128
    arr = transducer.convex_array(n, 0.5e-3, RADIUS, height=12e-3, dx=0.25e-3)
    half = 0.5 * n * 0.5e-3 / RADIUS
    pulse = imaging.Pulse(F0, 0.65)
    txs = [imaging.Transmit.diverging((xs, 0.0, -RADIUS)) for xs in (-4e-3, 0.0, 4e-3)]
    rfs = []
    for i, tx in enumerate(txs):
        rf, fs = imaging.simulate(arr, tx, pos, amp, pulse, C_SCANNER, medium=med,
                                  depth=130e-3, interfaces=True)
        rfs.append(imaging.add_noise(rf, 95, rng=i))

    x = np.arange(-64e-3, 64e-3, 0.2e-3)
    z = np.arange(1e-3, 129e-3, 0.2e-3)
    img = imaging.compound(rfs, fs, arr, txs, x, z, C_SCANNER, f_number=1.0)
    img = imaging.tgc(img, z, 0.6, F0)
    img = np.where(imaging.sector_mask(x, z, (0.0, -RADIUS), half, RADIUS, RADIUS + 130e-3),
                   img, np.nan)

    env = np.abs(img)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    r = np.hypot(xx - FIBROID[0], zz - FIBROID[2])
    inside = r < 9e-3
    around = (r > 14e-3) & (r < 18e-3) & med_mask(med, xx, zz, "myometrium") & np.isfinite(env)
    stats = {"fibroid contrast [dB]": quality.contrast_db(env[inside], env[around]),
             "fibroid gCNR": quality.gcnr(env[inside], env[around])}
    return ph, x, z, img, stats


def med_mask(med, xx, zz, name):
    g = med.grid
    ix = np.clip(np.rint((xx - g.origin[0]) / g.spacing[0]).astype(int), 0, g.shape[0] - 1)
    iz = np.clip(np.rint((zz - g.origin[1]) / g.spacing[1]).astype(int), 0, g.shape[1] - 1)
    return med.mask(name)[ix, iz]


def figure():
    ph, x, z, img, stats = compute()
    med = ph.medium
    xg, zg = med.grid.axes()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.2), constrained_layout=True)
    splt.labels_map(med.labels, xg, zg, names=[m.name for m in med.materials], ax=ax[0],
                    title="Uterus phantom (sagittal)")
    splt.bmode(img, x, z, 60, ax=ax[1], title="Transabdominal B-mode, 3.5 MHz convex array")
    for a in ax:
        a.set_xlim(-64, 64)
        a.set_ylim(129, 0)
    return fig


if __name__ == "__main__":
    ph, x, z, img, stats = compute()
    print(ph.description)
    for k, v in stats.items():
        print(f"{k:<24s}{v:8.2f}")
    figure()
    plt.show()

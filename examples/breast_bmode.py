"""B-mode image of an anatomical breast phantom.

The phantom (sonosk.organs.breast) has skin, subcutaneous fat crossed by
Cooper's ligaments, fibroglandular tissue mixed with fat lobules,
retromammary fat and the pectoralis muscle, with three lesions: a simple
cyst, a fibroadenoma and a spiculated carcinoma. Every tissue takes its
properties from sonosk.tissues, where each value has a published source.

A 128-element linear array at 7.5 MHz sends 21 plane waves between -8
and +8 degrees. Echo delays and attenuation are integrated along straight
paths through the phantom, so the image shows the effects of the real
tissue: the fat (1422 m/s) and the glandular tissue (1487 m/s) are slower
than the 1540 m/s the scanner assumes, the cyst has posterior enhancement
and the carcinoma, which attenuates more, casts a shadow. Interfaces give
specular echoes (phantom.interface_scatterers) and transmission losses
(interfaces=True). Receiver noise is added 90 dB below the strongest echo.

The image is formed twice: with the usual constant speed of 1540 m/s, and
with delays computed through the true speed map. The second is an oracle
case, the upper bound of what sound-speed correction can give.

With seven plane waves instead of 21 the cyst contrast was about -12 dB;
removing the noise did not change it, flattening all echo levels improved
it by 1 dB, and 21 angles brought it to about -17 dB. The cyst is filled
by the off-axis clutter of plane-wave compounding, which falls as angles
are added (Montaldo et al. 2009).
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import imaging, organs, phantom, quality, transducer

F0 = 7.5e6
C_SCANNER = 1540.0
CYST = (-11e-3, 0.0, 13e-3)
FA = (0.5e-3, 0.0, 14.5e-3)
CA = (11e-3, 0.0, 16e-3)


def compute(seed=4):
    g = sk.Grid.centered((401, 401), 0.1e-3, depth_start=0.0)
    ph = organs.breast(
        g,
        lesions=[organs.Cyst(CYST, a=3e-3, b=2.4e-3),
                 organs.Fibroadenoma(FA, a=5e-3, b=3e-3),
                 organs.Carcinoma(CA, radius=3.2e-3)],
        rng=seed)
    med = ph.medium

    pos, amp = phantom.scatterers(med, density=50, rng=seed)
    pos_i, amp_i = phantom.interface_scatterers(med)
    pos, amp = np.concatenate([pos, pos_i]), np.concatenate([amp, amp_i])
    arr = transducer.linear_array(128, 0.3e-3, height=4e-3, dx=0.15e-3)
    pulse = imaging.Pulse(F0, 0.65)
    txs = [imaging.Transmit.plane(a) for a in imaging.plane_wave_angles(21, np.deg2rad(16))]
    rfs = []
    for i, tx in enumerate(txs):
        rf, fs = imaging.simulate(arr, tx, pos, amp, pulse, C_SCANNER, medium=med, depth=40e-3,
                                  interfaces=True)
        rfs.append(imaging.add_noise(rf, 90, rng=seed + i))

    x = np.arange(-18e-3, 18e-3, 0.08e-3)
    z = np.arange(1e-3, 39e-3, 0.08e-3)
    nominal = imaging.compound(rfs, fs, arr, txs, x, z, C_SCANNER, f_number=1.4)
    aware = imaging.compound(rfs, fs, arr, txs, x, z, C_SCANNER, f_number=1.4, medium=med)
    nominal = imaging.tgc(nominal, z, 0.75, F0)
    aware = imaging.tgc(aware, z, 0.75, F0)
    stats = _measure(med, x, z, aware)
    return ph, x, z, nominal, aware, stats


def _measure(med, x, z, img):
    env = np.abs(img)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    g = med.grid

    def mask_of(name):
        m = med.mask(name)
        ix = np.clip(np.rint((xx - g.origin[0]) / g.spacing[0]).astype(int), 0, g.shape[0] - 1)
        iz = np.clip(np.rint((zz - g.origin[1]) / g.spacing[1]).astype(int), 0, g.shape[1] - 1)
        return m[ix, iz]

    def ring(c, r0, r1):
        r = np.hypot(xx - c[0], zz - c[2])
        return (r > r0) & (r < r1)

    out = {}
    for name, c, r in (("breast cyst", CYST, 3e-3), ("fibroadenoma", FA, 5e-3),
                       ("breast carcinoma", CA, 3.5e-3)):
        inside = mask_of(name) & (np.hypot(xx - c[0], zz - c[2]) < 0.8 * r)
        around = ring(c, r + 1.5e-3, r + 4e-3) & (np.abs(zz - c[2]) < r)
        out[f"{name}: contrast [dB]"] = quality.contrast_db(env[inside], env[around])
        out[f"{name}: gCNR"] = quality.gcnr(env[inside], env[around])
        below = (np.abs(xx - c[0]) < 0.6 * r) & (zz > c[2] + 1.3 * r) & (zz < c[2] + 1.3 * r + 5e-3)
        beside = (np.abs(np.abs(xx - c[0]) - 2.2 * r) < 0.6 * r) & \
                 (zz > c[2] + 1.3 * r) & (zz < c[2] + 1.3 * r + 5e-3)
        out[f"{name}: posterior [dB]"] = quality.contrast_db(env[below], env[beside])
    return out


def figure():
    ph, x, z, nominal, aware, stats = compute()
    med = ph.medium
    xg, zg = med.grid.axes()
    fig, ax = plt.subplots(2, 2, figsize=(11, 10.5), constrained_layout=True)
    splt.labels_map(med.labels, xg, zg, names=[m.name for m in med.materials], ax=ax[0, 0],
                    title="Breast phantom")
    splt.property_map(med, "c", ax=ax[0, 1], title="Sound speed", label="m/s")
    ref = np.abs(aware).max()
    splt.bmode(nominal, x, z, 55, ax=ax[1, 0], reference=ref,
               title="B-mode, beamformed at 1540 m/s")
    splt.bmode(aware, x, z, 55, ax=ax[1, 1], reference=ref,
               title="B-mode, beamformed through the true speed map (oracle)")
    for a in ax.ravel():
        a.set_xlim(-18, 18)
        a.set_ylim(39, 0)
    return fig


if __name__ == "__main__":
    ph, x, z, nominal, aware, stats = compute()
    print(ph.description)
    for k, v in stats.items():
        print(f"{k:<34s}{v:8.2f}")
    figure()
    plt.show()

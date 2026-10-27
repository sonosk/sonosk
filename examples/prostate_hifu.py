"""Transrectal focused ultrasound ablation of a prostate tumour.

The source has the geometry of a clinical transrectal HIFU probe: a
spherical cap with 40 mm focal length and 50 mm aperture, truncated to
31 mm across, driven at 3 MHz (Pichardo et al. 2008, Ultrasound Med.
Biol. 34:1105). It sits in a water balloon in the rectum and focuses on a
tumour of 4 mm radius in the peripheral zone of a 3-D prostate phantom
(sonosk.organs.prostate).

The field reaches the tissue through the hybrid angular spectrum method.
The focus is moved over a raster of points 1.7 mm apart covering the
tumour; each shot lasts 4 s with 10 s of cooling, and the balloon keeps
the rectal surface at 20 C (a schedule and a coolant temperature chosen
for the example). The report gives the peak temperature and the thermal
dose in the rectal wall, the structure that limits transrectal treatments.

Finally the transverse plane is imaged before and after the treatment
with a 7.5 MHz linear array in the rectum. Tissue that reached 60 C gets
the higher backscatter and attenuation measured in heated human prostate
(Worthington et al. 2002): the front of the lesion turns brighter, and
its higher attenuation darkens the tissue behind it.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import field, imaging, organs, phantom, thermal, transducer, water

F = 3e6
ROC = 40e-3
APERTURE = 50e-3
WIDTH = 31e-3
POWER = 3.8             # keeps the peak below 100 C (no boiling model)
ON, OFF = 4.0, 10.0
COOLANT = 20.0          # temperature of the balloon at the rectal wall [C]
TUMOUR = (5e-3, 0.0, 10.5e-3)


def probe():
    c_w = float(water.sound_speed(37.0))
    cap = transducer.bowl(APERTURE / 2, ROC, dx=c_w / F / 2)
    cap = cap.subset(np.abs(cap.pos[:, 0]) <= WIDTH / 2)
    cap.translate((TUMOUR[0], TUMOUR[1], TUMOUR[2] - ROC))
    return cap, c_w


def compute(seed=3):
    g = sk.Grid.centered((181, 181, 126), 0.2e-3, depth_start=0.0)
    ph = organs.prostate(g, tumors=[organs.Tumor(TUMOUR, radius=4e-3, roughness=0.15,
                                                 tissue="prostate carcinoma", label="tumour")],
                         rng=seed)
    med = ph.medium
    cap, c_w = probe()
    plane = sk.Grid(g.shape[:2] + (1,), g.spacing, g.origin)
    p0 = field.rayleigh_grid(cap, plane, F, c_w)[..., 0]
    p0 = field.scale_to_pressure(p0, float(water.density(37.0)), c_w, power=POWER,
                                 area=cap.total_area)
    p = field.hybrid_angular_spectrum(p0, med, F)
    q = thermal.heat_source(p, med.attenuation(F), med.rho, med.c)

    lo = (TUMOUR[0] - 9e-3, TUMOUR[1] - 9e-3, 0.0)
    hi = (TUMOUR[0] + 9e-3, TUMOUR[1] + 9e-3, 24e-3)
    sub, sl = med.crop(lo, hi)
    qs = q[sl]
    k = np.unravel_index(np.argmax(qs), qs.shape)
    focus = np.array([sub.grid.axis(i)[k[i]] for i in range(3)])
    # raster of focal points in the plane of the tumour centre
    kz = sub.grid.index(TUMOUR)[2]
    layer = sk.Grid(sub.grid.shape[:2], sub.grid.spacing[:2], sub.grid.origin[:2])
    xy = thermal.raster(sub.mask("tumour")[:, :, kz], layer, 1.7e-3, margin=0.5e-3)
    pts = np.column_stack([xy, np.full(len(xy), TUMOUR[2])]) if len(xy) else np.array([TUMOUR])
    shots = [thermal.Sonication(thermal.shift(qs, sub.grid, pt - focus), ON, cooling=OFF,
                                label=f"shot {i + 1}") for i, pt in enumerate(pts)]
    plan = thermal.treat(sub, shots, shutdown=("dose", thermal.LESION_DOSE),
                         T_surface=COOLANT)
    rep = thermal.treatment_report(plan.lesion(), sub, "tumour")
    wall = sub.mask("rectal wall")
    out = {
        "tumour volume [mm3]": rep["target_volume"] * 1e9,
        "ablated [mm3]": rep["ablated"] * 1e9,
        "fraction ablated [%]": 100 * rep["fraction_ablated"],
        "healthy prostate damaged [mm3]": rep["healthy_damaged"] * 1e9,
        "shots": len(shots),
        "peak focal pressure [MPa]": np.abs(p).max() / 1e6,
        "peak temperature [C]": float(plan.result.T_max.max()),
        "rectal wall peak [C]": float(plan.result.T_max[wall].max()),
        "rectal wall max dose [min]": float(plan.result.dose[wall].max()),
    }

    # images of the transverse plane before and after
    T_full = np.full(g.shape, 37.0)
    T_full[sl] = plan.result.T_max
    j = g.index((0, 0, 0))[1]
    before = med.slice2d(0.0)
    after = thermal.coagulate(before, T_full[:, j, :])
    arr = transducer.linear_array(128, 0.25e-3, height=4e-3, dx=0.1e-3)
    pulse = imaging.Pulse(7.5e6, 0.65)
    txs = [imaging.Transmit.plane(a) for a in imaging.plane_wave_angles(5, np.deg2rad(12))]
    x = np.arange(-15e-3, 15e-3, 0.08e-3)
    z = np.arange(0.5e-3, 24.5e-3, 0.08e-3)
    images = []
    for m in (before, after):
        pos, amp = phantom.scatterers(m, density=50, rng=seed)
        pos_i, amp_i = phantom.interface_scatterers(m)
        pos, amp = np.concatenate([pos, pos_i]), np.concatenate([amp, amp_i])
        rfs = []
        for i, tx in enumerate(txs):
            rf, fs = imaging.simulate(arr, tx, pos, amp, pulse, 1540.0, medium=m, depth=25e-3,
                                      interfaces=True)
            rfs.append(imaging.add_noise(rf, 90, rng=i))
        img = imaging.compound(rfs, fs, arr, txs, x, z, 1540.0, f_number=1.4)
        images.append(imaging.tgc(img, z, 0.7, pulse.f0))
    return ph, sub, plan, pts, out, (x, z, images)


def figure():
    ph, sub, plan, pts, out, (x, z, images) = compute()
    sj = sub.grid.index((0, 0, 0))[1]
    sx, sz = sub.grid.axis(0), sub.grid.axis(2)
    fig, ax = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    h = splt.temperature_map(plan.result.T_max[:, sj, :], sx, sz, ax=ax[0, 0],
                             lesion=plan.lesion()[:, sj, :],
                             title="Peak temperature and lesion (cyan)")
    for name, style in (("tumour", "--"), ("rectal wall", ":")):
        ax[0, 0].contour(sx * 1e3, sz * 1e3, sub.mask(name)[:, sj, :].T.astype(float),
                         levels=[0.5], colors="white", linestyles=style, linewidths=1)
    fig.colorbar(h, ax=ax[0, 0], shrink=0.8, label="C")
    ax[0, 1].axis("off")
    rows = [f"{k:<32s}{v:10.4g}" for k, v in out.items()]
    ax[0, 1].text(0.0, 0.95, "\n".join(rows), family="monospace", va="top", fontsize=9)
    ref = np.abs(images[0]).max()
    splt.bmode(images[0], x, z, 55, ax=ax[1, 0], reference=ref, title="Before treatment")
    splt.bmode(images[1], x, z, 55, ax=ax[1, 1], reference=ref, title="After treatment")
    return fig


if __name__ == "__main__":
    ph, sub, plan, pts, out, _ = compute()
    print(ph.description)
    for row in plan.log:
        print(f"{row[0]}: peak {row[1]:.1f} C, lesion {row[2] * 1e9:.1f} mm3")
    for k, v in out.items():
        print(f"{k:<32s}{v:9.2f}")
    figure()
    plt.show()

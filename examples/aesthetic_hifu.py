"""Micro-focused ultrasound in facial skin: thermal coagulation points.

Aesthetic devices place small thermal coagulation points at fixed depths
under the skin: 1.5 mm with a 10 MHz transducer, 3.0 mm with 7 MHz and
4.5 mm with 4 MHz, the last one in the superficial musculoaponeurotic
system (SMAS) (Lowe 2021, Plast. Reconstr. Surg. Glob. Open 9:e3662).
Exposures last tens of milliseconds (50-200 ms in Laubach et al. 2008,
Dermatol. Surg. 34:727).

Here a 3-D face phantom (sonosk.organs.face) receives a line of five
points, 1.5 mm apart, from each of three small focused transducers. Their
geometry is generic (a spherical cap behind a 1.5 mm water standoff); it
is not that of a commercial device. The field goes through epidermis,
dermis, fat and SMAS with the hybrid angular spectrum method, and the
damage is computed with the Arrhenius model of Henriques and Moritz for
skin. The figure shows where each transducer coagulates tissue and checks
that the epidermis is spared.

The deepest transducer is aimed at 5.0 mm rather than 4.5 mm. Fat has a
lower heat capacity per volume than the SMAS (rho cp of 2.1 against
3.8 MJ/(m^3 K)), so for the same absorbed power the fat just above the
SMAS heats faster and pulls the coagulation point up. Aimed at 4.5 mm,
the points formed in the fat and missed the SMAS.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
from sonosk import field, organs, thermal, transducer, water

STANDOFF = 1.5e-3
PULSE = 0.05        # s
GAP = 0.15          # s between points
SPACING = 1.5e-3
TRANSDUCERS = [     # frequency, focal depth in tissue, cap radius, acoustic power
    (10e6, 1.5e-3, 2.0e-3, 0.7),
    (7e6, 3.0e-3, 3.0e-3, 2.3),
    (4e6, 5.0e-3, 4.0e-3, 24.0),
]


def run(f, depth, radius, power, med):
    g = med.grid
    c_w = float(water.sound_speed(37.0))
    cap = transducer.bowl(radius, STANDOFF + depth, dx=c_w / f / 4)
    cap.translate((0.0, 0.0, -STANDOFF))
    plane = sk.Grid(g.shape[:2] + (1,), g.spacing, g.origin)
    p0 = field.rayleigh_grid(cap, plane, f, c_w)[..., 0]
    p0 = field.scale_to_pressure(p0, float(water.density(37.0)), c_w, power=power,
                                 area=cap.total_area)
    p = field.hybrid_angular_spectrum(p0, med, f)
    q = thermal.heat_source(p, med.attenuation(f), med.rho, med.c)
    xs = (np.arange(5) - 2) * SPACING
    shots = [thermal.Sonication(thermal.shift(q, g, (x0, 0.0, 0.0)), PULSE, cooling=GAP)
             for x0 in xs]
    plan = thermal.treat(med, shots, arrhenius=thermal.HENRIQUES, max_dt=0.005)
    return p, plan


def compute(seed=1):
    g = sk.Grid.centered((161, 101, 141), 0.05e-3, depth_start=0.0)
    ph = organs.face(g, rng=seed)
    med = ph.medium
    results = []
    for f, depth, radius, power in TRANSDUCERS:
        p, plan = run(f, depth, radius, power, med)
        dmg = plan.result.damaged()
        z = g.axis(2)
        hit = np.where(dmg.any(axis=(0, 1)))[0]
        row = {
            "frequency [MHz]": f / 1e6,
            "focal depth [mm]": depth * 1e3,
            "energy per point [J]": power * PULSE,
            "peak temperature [C]": float(plan.result.T_max.max()),
            "damage from [mm]": z[hit[0]] * 1e3 if hit.size else np.nan,
            "damage to [mm]": z[hit[-1]] * 1e3 if hit.size else np.nan,
            "epidermis peak [C]": float(plan.result.T_max[med.mask("epidermis")].max()),
            "SMAS damaged [mm3]": float((dmg & med.mask("smas")).sum()
                                        * np.prod(g.spacing) * 1e9),
        }
        results.append((p, plan, row))
    return ph, results


def figure():
    ph, results = compute()
    med = ph.medium
    g = med.grid
    j = g.shape[1] // 2
    x, z = g.axis(0) * 1e3, g.axis(2) * 1e3
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.6), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for a, (p, plan, row) in zip(ax[:3], results):
        h = a.imshow(plan.result.T_max[:, j, :].T, cmap="hot", vmin=37, vmax=80,
                     extent=[x[0], x[-1], z[-1], z[0]], aspect="equal")
        a.contour(x, z, plan.result.damaged()[:, j, :].T.astype(float), levels=[0.5],
                  colors="cyan", linewidths=1)
        a.contour(x, z, med.labels[:, j, :].T, levels=np.arange(6) + 0.5, colors="white",
                  linewidths=0.4)
        a.set_title(f"{row['frequency [MHz]']:.0f} MHz, focus {row['focal depth [mm]']:.1f} mm")
        a.set_xlabel("x [mm]")
        a.set_ylabel("depth [mm]")
    fig.colorbar(h, ax=ax[2], shrink=0.8, label="peak temperature [C]")
    ax[3].axis("off")
    lines = []
    for _, _, row in results:
        lines += [f"{k:<22s}{v:7.2f}" for k, v in row.items()] + [""]
    ax[3].text(0.0, 1.0, "\n".join(lines), family="monospace", va="top", fontsize=8)
    return fig


if __name__ == "__main__":
    ph, results = compute()
    print(ph.description)
    for _, _, row in results:
        for k, v in row.items():
            print(f"{k:<22s}{v:8.2f}")
        print()
    figure()
    plt.show()

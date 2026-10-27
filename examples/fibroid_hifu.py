"""Planning a focused ultrasound treatment of a uterine fibroid.

A 3-D uterus phantom (sonosk.organs.uterus) has a fibroid of 24 mm
diameter behind the abdominal wall and the bladder. The source has the
geometry of a clinical MR-guided system, a spherical cap of 135.9 mm
aperture and 140 mm radius of curvature (Kothapalli et al. 2018,
J. Ther. Ultrasound 6:5), driven at 1.2 MHz and coupled through water.

The field is computed in two steps: the Rayleigh integral from the cap to
the skin, then the hybrid angular spectrum method through skin, fat,
muscle, bladder and uterus, which accounts for the speed, attenuation and
impedance of each tissue. The heat source of the focus is moved to the
centres of five volumetric cells, and in each cell the focus sweeps a
circle of 4 mm radius (the volumetric sonication of Köhler et al. 2009).
The Pennes equation runs through the whole plan with perfusion stopping
in coagulated tissue.

The report gives the ablated and residual fibroid volume, the damage to
healthy tissue, the energy efficiency in mm^3 of lethal dose per joule,
and the peak temperature in the skin and fat for one cell.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import field, organs, thermal, transducer, water

F = 1.2e6
APERTURE = 135.9e-3
ROC = 140e-3
POWER = 110.0          # acoustic power [W]
ON, OFF = 24.0, 36.0   # seconds per cell
FIBROID = (6e-3, 0.0, 96e-3)
CELLS = [(0, 0), (6e-3, 0), (-6e-3, 0), (0, 6e-3), (0, -6e-3)]


def source():
    c_w = float(water.sound_speed(37.0))
    cap = transducer.bowl(APERTURE / 2, ROC, dx=c_w / F / 2)
    cap.translate((FIBROID[0], FIBROID[1], FIBROID[2] - ROC))
    return cap, c_w


def compute(seed=7):
    g = sk.Grid.centered((209, 209, 231), 0.5e-3, depth_start=0.0)
    ph = organs.uterus(
        g, fibroids=[organs.Tumor(FIBROID, radius=12e-3, roughness=0.06, tissue="fibroid",
                                  label="fibroid")],
        rng=seed)
    med = ph.medium

    cap, c_w = source()
    rho_w = float(water.density(37.0))
    plane = sk.Grid(g.shape[:2] + (1,), g.spacing, g.origin)
    p0 = field.rayleigh_grid(cap, plane, F, c_w)[..., 0]
    p0 = field.scale_to_pressure(p0, rho_w, c_w, power=POWER, area=cap.total_area)
    p = field.hybrid_angular_spectrum(p0, med, F)
    q = thermal.heat_source(p, med.attenuation(F), med.rho, med.c)

    # thermal problem around the fibroid
    lo = (FIBROID[0] - 20e-3, FIBROID[1] - 20e-3, FIBROID[2] - 22e-3)
    hi = (FIBROID[0] + 20e-3, FIBROID[1] + 20e-3, FIBROID[2] + 22e-3)
    sub, sl = med.crop(lo, hi)
    qs = q[sl]
    k = np.unravel_index(np.argmax(qs), qs.shape)
    focus = np.array([sub.grid.axis(i)[k[i]] for i in range(3)])
    sonics = []
    for i, (dx, dy) in enumerate(CELLS):
        offs = thermal.circles([2e-3, 4e-3], spacing=1.5e-3)
        offs = offs + np.array([FIBROID[0] + dx - focus[0], FIBROID[1] + dy - focus[1],
                                FIBROID[2] - focus[2]])
        sonics.append(thermal.Sonication(thermal.trajectory_source(qs, sub.grid, offs), ON,
                                         cooling=OFF, label=f"cell {i + 1}"))
    plan = thermal.treat(sub, sonics, shutdown=("dose", thermal.LESION_DOSE))
    rep = thermal.treatment_report(plan.lesion(), sub, "fibroid")
    energy = POWER * ON * len(CELLS)

    # near field: skin and fat during one cell
    near, sln = med.crop((g.origin[0], g.origin[1], 0.0),
                         (g.origin[0] + g.extent[0], g.origin[1] + g.extent[1], 30e-3))
    coarse = (slice(None, None, 2), slice(None, None, 2), slice(None))
    qn = q[sln][coarse]
    gn = sk.Grid(qn.shape, near.grid.spacing * np.array([2, 2, 1]), near.grid.origin)
    near = sk.Medium(gn, near.labels[coarse], near.materials)
    res_near = thermal.bioheat(near, qn, ON, max_dt=0.1)
    skinfat = near.mask("skin") | near.mask("subcutaneous fat")
    out = {
        "fibroid volume [mL]": rep["target_volume"] * 1e6,
        "ablated [mL]": rep["ablated"] * 1e6,
        "fraction ablated [%]": 100 * rep["fraction_ablated"],
        "healthy tissue damaged [mL]": rep["healthy_damaged"] * 1e6,
        "energy [kJ]": energy / 1e3,
        "efficiency [mm3/J]": rep["ablated"] * 1e9 / energy,
        "peak focal pressure [MPa]": np.abs(p).max() / 1e6,
        "peak temperature [C]": float(plan.result.T_max.max()),
        "skin/fat peak, one cell [C]": float(res_near.T_max[skinfat].max()),
    }
    return ph, p, sub, sl, plan, out


def figure():
    ph, p, sub, sl, plan, out = compute()
    med = ph.medium
    g = med.grid
    j = g.index((0, FIBROID[1], 0))[1]
    x, z = g.axis(0), g.axis(2)
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.4), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1, 0.8]})
    splt.field_map(p[:, j, :], x, z, dynamic_range=30, ax=ax[0],
                   title="Pressure through the abdominal wall [dB]")
    ax[0].contour(x * 1e3, z * 1e3, med.labels[:, j, :].T, levels=np.arange(20) + 0.5,
                  colors="white", linewidths=0.4)
    sj = sub.grid.index((0, FIBROID[1], 0))[1]
    sx, sz = sub.grid.axis(0), sub.grid.axis(2)
    h = splt.temperature_map(plan.result.T_max[:, sj, :], sx, sz, ax=ax[1],
                             lesion=plan.lesion()[:, sj, :],
                             title="Peak temperature and lesion (cyan)")
    ax[1].contour(sx * 1e3, sz * 1e3, sub.mask("fibroid")[:, sj, :].T.astype(float),
                  levels=[0.5], colors="white", linestyles="--", linewidths=1)
    fig.colorbar(h, ax=ax[1], shrink=0.8, label="C")
    ax[2].axis("off")
    rows = [f"{k:<30s}{v:10.4g}" for k, v in out.items()]
    ax[2].text(0.0, 0.98, "\n".join(rows), family="monospace", va="top", fontsize=9)
    return fig


if __name__ == "__main__":
    ph, p, sub, sl, plan, out = compute()
    print(ph.description)
    for row in plan.log:
        print(f"{row[0]}: peak {row[1]:.1f} C, lesion {row[2] * 1e9:.0f} mm3")
    for k, v in out.items():
        print(f"{k:<30s}{v:9.3f}")
    figure()
    plt.show()

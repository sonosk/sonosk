"""How much the fibroid treatment result depends on the tissue values.

The fibroid plan of fibroid_hifu.py (same phantom, source, power, cells
and schedule) is run again with one input changed at a time:

- perfusion of the fibroid halved, and tripled (the value of 100 mL/(min
  kg) is a modelling choice, and fibroids vary widely);
- only 70 % of the attenuation taken as absorption (the default treats all
  of it as absorption, which overestimates heating);
- density of fibroid and myometrium at the minimum and at the maximum of
  the IT'IS V5.0 range for uterus (1052 and 1157 kg/m^3), which changes
  the heat capacity per volume;
- no perfusion shutdown.

The field is computed once. The figure and the printed table give the
ablated fraction of the fibroid, the damaged healthy volume and the peak
temperature of every run, which shows which assumptions dominate the
predicted outcome. These are model sensitivities, not clinical ranges.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
from sonosk import field, organs, thermal, tissues, transducer, water

F = 1.2e6
APERTURE = 135.9e-3
ROC = 140e-3
POWER = 110.0
ON, OFF = 24.0, 36.0
FIBROID = (6e-3, 0.0, 96e-3)
CELLS = [(0, 0), (6e-3, 0), (-6e-3, 0), (0, 6e-3), (0, -6e-3)]


def field_and_medium(seed=7):
    g = sk.Grid.centered((209, 209, 231), 0.5e-3, depth_start=0.0)
    ph = organs.uterus(
        g, fibroids=[organs.Tumor(FIBROID, radius=12e-3, roughness=0.06, tissue="fibroid",
                                  label="fibroid")],
        rng=seed)
    med = ph.medium
    c_w = float(water.sound_speed(37.0))
    cap = transducer.bowl(APERTURE / 2, ROC, dx=c_w / F / 2)
    cap.translate((FIBROID[0], FIBROID[1], FIBROID[2] - ROC))
    plane = sk.Grid(g.shape[:2] + (1,), g.spacing, g.origin)
    p0 = field.rayleigh_grid(cap, plane, F, c_w)[..., 0]
    p0 = field.scale_to_pressure(p0, float(water.density(37.0)), c_w, power=POWER,
                                 area=cap.total_area)
    p = field.hybrid_angular_spectrum(p0, med, F)
    lo = (FIBROID[0] - 20e-3, FIBROID[1] - 20e-3, FIBROID[2] - 22e-3)
    hi = (FIBROID[0] + 20e-3, FIBROID[1] + 20e-3, FIBROID[2] + 22e-3)
    sub, sl = med.crop(lo, hi)
    return sub, p[sl]


def changed(sub, **per_material):
    """Copy of sub with some material properties replaced:
    per_material = {material name: {property: value}}."""
    mats = [m.with_(**per_material[m.name]) if m.name in per_material else m
            for m in sub.materials]
    return sk.Medium(sub.grid, sub.labels, mats)


def plan(sub, p, fraction=1.0, shutdown=("dose", thermal.LESION_DOSE)):
    q = thermal.heat_source(p, sub.attenuation(F), sub.rho, sub.c, fraction)
    k = np.unravel_index(np.argmax(q), q.shape)
    focus = np.array([sub.grid.axis(i)[k[i]] for i in range(3)])
    sonics = []
    for dx, dy in CELLS:
        offs = thermal.circles([2e-3, 4e-3], spacing=1.5e-3)
        offs = offs + np.array([FIBROID[0] + dx - focus[0], FIBROID[1] + dy - focus[1],
                                FIBROID[2] - focus[2]])
        sonics.append(thermal.Sonication(thermal.trajectory_source(q, sub.grid, offs), ON,
                                         cooling=OFF))
    res = thermal.treat(sub, sonics, shutdown=shutdown)
    rep = thermal.treatment_report(res.lesion(), sub, "fibroid")
    return {"ablated [%]": 100 * rep["fraction_ablated"],
            "healthy damaged [mL]": rep["healthy_damaged"] * 1e6,
            "peak temperature [C]": float(res.result.T_max.max())}


def compute():
    sub, p = field_and_medium()
    rho = {n: m.rho for n, m in ((m.name, m) for m in sub.materials)}
    perf = tissues.perfusion_si
    runs = {"base": plan(sub, p)}
    runs["fibroid perfusion x0.5"] = plan(changed(
        sub, fibroid={"perfusion": perf(50.0, rho["fibroid"])}), p)
    runs["fibroid perfusion x3"] = plan(changed(
        sub, fibroid={"perfusion": perf(300.0, rho["fibroid"])}), p)
    runs["absorption 70 % of attenuation"] = plan(sub, p, fraction=0.7)
    for name, r in (("density 1052 kg/m3", 1052.0), ("density 1157 kg/m3", 1157.0)):
        runs[name] = plan(changed(sub, fibroid={"rho": r}, myometrium={"rho": r}), p)
    runs["no perfusion shutdown"] = plan(sub, p, shutdown=None)
    return runs


def figure():
    runs = compute()
    names = list(runs)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for a, key in zip(ax, ["ablated [%]", "healthy damaged [mL]", "peak temperature [C]"]):
        v = [runs[n][key] for n in names]
        a.barh(range(len(names)), v, color=["0.3"] + ["tab:blue"] * (len(names) - 1))
        a.axvline(runs["base"][key], color="0.3", lw=0.8, ls="--")
        for i, vi in enumerate(v):
            a.text(vi, i, f" {vi:.3g}", va="center", fontsize=8)
        a.set_yticks(range(len(names)))
        a.set_yticklabels(names if a is ax[0] else [])
        a.invert_yaxis()
        a.set_xlabel(key)
    ax[0].set_title("Fibroid ablated")
    ax[1].set_title("Healthy tissue damaged")
    ax[2].set_title("Peak temperature")
    return fig


if __name__ == "__main__":
    runs = compute()
    for name, r in runs.items():
        print(f"{name:<32s}" + "  ".join(f"{k} {v:7.2f}" for k, v in r.items()))
    figure()
    plt.show()

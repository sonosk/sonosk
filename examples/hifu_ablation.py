"""Thermal ablation of a spherical tumour with a focused bowl.

A 1 MHz bowl (20 mm aperture radius, 50 mm radius of curvature) heats a
tumour of 4 mm radius centred at the focus. The pressure field is the
Rayleigh integral in a homogeneous attenuating medium, scaled to a focal
amplitude of 3 MPa and treated as linear. The source is on for 8 s and the
tissue then cools for 20 s. The Pennes equation is solved in 3-D and the
lesion is the region with CEM43 of 240 min or more.

The tissue and tumour values below are generic numbers for the example and
not measured data.
"""

import matplotlib.pyplot as plt
import numpy as np

import sonosk as sk
import sonosk.plot as splt
from sonosk import field, phantom, thermal, transducer

F = 1e6
A = 20e-3
ROC = 50e-3
P_FOCUS = 3e6
TISSUE = sk.Material("tissue", 1540, 1050, alpha0=0.5, k=0.5, cp=3600, perfusion=0.5)
TUMOUR = sk.Material("tumour", 1540, 1050, alpha0=0.7, k=0.5, cp=3600, perfusion=0.3)


def _axisymmetric_field(r_max, z0, z1, h):
    """|p| of the bowl on an (r, z) grid; the bowl field is symmetric about z."""
    src = transducer.bowl(A, ROC, dx=TISSUE.c / F / 4)
    r = np.arange(0, r_max + h, h)
    z = np.arange(z0, z1 + h, h)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    pts = np.stack([rr.ravel(), 0 * rr.ravel(), zz.ravel()], 1)
    p = np.abs(field.rayleigh(src, pts, F, TISSUE.c, alpha0=TISSUE.alpha0)).reshape(rr.shape)
    return r, z, p


def _interp_rz(r, z, p, R, Z):
    # bilinear interpolation of p(r, z) at the points (R, Z)
    fr = np.clip((R - r[0]) / (r[1] - r[0]), 0, r.size - 1.001)
    fz = np.clip((Z - z[0]) / (z[1] - z[0]), 0, z.size - 1.001)
    i, j = fr.astype(int), fz.astype(int)
    a, b = fr - i, fz - j
    return ((1 - a) * (1 - b) * p[i, j] + a * (1 - b) * p[i + 1, j]
            + (1 - a) * b * p[i, j + 1] + a * b * p[i + 1, j + 1])


def compute():
    g = sk.Grid.centered((41, 41, 66), 0.4e-3, depth_start=36e-3)
    x, y, z = g.mesh()
    lab = phantom.paint(np.zeros(g.shape, int), phantom.sphere(g, (0, 0, ROC), 4e-3), 1)
    med = sk.Medium(g, lab, [TISSUE, TUMOUR])

    r, zr, pr = _axisymmetric_field(np.hypot(*g.extent[:2]) / 2 + 1e-3, 35e-3, 63e-3, 0.2e-3)
    p = _interp_rz(r, zr, pr, np.hypot(x, y), z)
    p *= P_FOCUS / p.max()
    q = thermal.heat_source(p, med.attenuation(F), med.rho, med.c)

    on = 8.0
    res = thermal.bioheat(med, q, duration=28.0, schedule=lambda t: 1.0 if t < on else 0.0,
                          probe=[tuple(np.array(g.shape) // 2)])
    rep = thermal.treatment_report(res.lesion(), med, "tumour")
    return g, med, res, rep


def figure():
    g, med, res, rep = compute()
    xg, yg, zg = g.axes()
    j = g.shape[1] // 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1.2, 0.9]})
    h = splt.temperature_map(res.T_max[:, j, :], xg, zg, ax=ax[0], lesion=res.lesion()[:, j, :],
                             title="Peak temperature, lesion in cyan")
    ax[0].contour(xg * 1e3, zg * 1e3, med.mask("tumour")[:, j, :].T.astype(float), levels=[0.5],
                  colors="white", linestyles="--", linewidths=1.0)
    fig.colorbar(h, ax=ax[0], shrink=0.8, label="C")

    t = np.arange(res.probe.shape[0]) * res.dt
    ax[1].plot(t, res.probe[:, 0])
    ax[1].axhline(43, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Temperature at the focus [C]")
    ax[1].set_title("Heating 8 s, cooling 20 s")
    ax[1].grid(alpha=0.3)

    mm3 = 1e9
    rows = [f"{'tumour volume':<22s}{rep['target_volume'] * mm3:7.1f} mm3",
            f"{'ablated':<22s}{rep['ablated'] * mm3:7.1f} mm3",
            f"{'residual':<22s}{rep['residual'] * mm3:7.1f} mm3",
            f"{'fraction ablated':<22s}{100 * rep['fraction_ablated']:7.1f} %",
            f"{'healthy tissue lesion':<22s}{rep['healthy_damaged'] * mm3:7.1f} mm3",
            f"{'peak temperature':<22s}{res.T_max.max():7.1f} C"]
    ax[2].axis("off")
    ax[2].text(0.0, 0.98, "\n".join(rows), family="monospace", va="top", fontsize=9)
    return fig


if __name__ == "__main__":
    g, med, res, rep = compute()
    print(f"peak temperature {res.T_max.max():.1f} C, time step {res.dt:.3f} s")
    for k, v in rep.items():
        print(k, v)
    figure()
    plt.show()

"""Convergence studies and reference cases with machine-readable results.

Each function runs one study and returns a dict of plain numbers, so the
results can be saved as JSON and compared from release to release
(benchmarks/run_benchmarks.py in the repository does this and draws the
figure of the documentation). The studies are:

- fullwave_convergence: k-space solver, a plane wave on a cylinder,
  traces at 3 to 12 points per wavelength against the finest grid;
- pml_reflection: reflection from the perfectly matched layer;
- has_dz_convergence: hybrid angular spectrum through a smooth impedance
  gradient and a sharp step, for several slice spacings;
- bioheat_convergence: Pennes solver against the analytic spreading of a
  Gaussian temperature profile, for several grid steps;
- exponent_errors: error of the single power-law exponent of the
  full-wave solver for tissues of the database.

The comparison with k-Wave on shared cases needs k-Wave itself and is in
benchmarks/kwave_comparison.py of the repository.
"""

import time

import numpy as np

from . import field, thermal, tissues
from . import fullwave as fw
from .grid import Grid
from .medium import Material, Medium


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def fullwave_convergence(ppw=(3, 4, 6, 8, 12), f0=1e6, c=1500.0):
    """Plane wave (3-cycle burst at f0) from a line source onto a fluid
    cylinder (radius 2 wavelengths, 1650 m/s, 1150 kg/m^3), lossless.

    Pressure traces at a transmitted, a reflected and a side point, all on
    multiples of the wavelength so that they fall on grid points of every
    grid. Returns the relative L2 error of each trace against the finest
    grid and the run time.
    """
    lam = c / f0
    sensors = {"transmitted": (0.0, 16 * lam), "reflected": (0.0, 4 * lam),
               "side": (6 * lam, 12 * lam)}
    t_end = 30 * lam / c
    t_common = np.arange(0, t_end, 1 / (40 * f0))
    traces, times = {}, {}
    for n in sorted(ppw):
        dx = lam / n
        nx, nz = 24 * n + 1, 20 * n + 1
        g = Grid((nx, nz), dx, origin=(-12 * lam, 0.0))
        xx, zz = g.mesh()
        lab = (np.hypot(xx, zz - 8 * lam) < 2 * lam).astype(int)
        med = Medium(g, lab, [Material("water", c, 1000.0),
                              Material("cylinder", 1650.0, 1150.0)])
        k_src = int(round(2 * lam / dx))
        src = fw.Source([(i, k_src) for i in range(nx)], fw.tone_burst(f0, 3))
        idx = [g.index(p, strict=True) for p in sensors.values()]
        t0 = time.time()
        res = fw.simulate(med, src, t_end, sensor=fw.Sensor(idx), absorbing=False)
        times[n] = time.time() - t0
        traces[n] = np.array([np.interp(t_common, res.t, p) for p in res.p])
    ref = traces[max(ppw)]
    out = {"ppw": [], "error": {k: [] for k in sensors}, "seconds": []}
    for n in sorted(ppw):
        if n == max(ppw):
            continue
        out["ppw"].append(n)
        out["seconds"].append(round(times[n], 2))
        for i, k in enumerate(sensors):
            out["error"][k].append(_rel(traces[n][i], ref[i]))
    out["reference_ppw"] = max(ppw)
    return out


def pml_reflection(thickness=(10, 20, 30, 40), pml_alpha=2.0, f0=2e6):
    """Amplitude of the wave reflected by the perfectly matched layer, in
    dB relative to the incident wave, for a 1-D pulse at normal incidence."""
    dx = 0.05e-3
    c = 1500.0
    n = 1200
    g = Grid((n,), dx)
    med = Medium.homogeneous(g, Material("water", c, 1000.0))
    src = fw.Source([[200]], fw.tone_burst(f0, 3))
    sensor = 1100
    t_inc = (sensor - 200) * dx / c
    t_ref = (2 * (n - 1) - sensor - 200) * dx / c
    out = {"thickness": [], "reflection_db": [], "pml_alpha": pml_alpha}
    for P in thickness:
        res = fw.simulate(med, src, t_ref + 4e-6, sensor=fw.Sensor([[sensor]]), pml=P,
                          pml_alpha=pml_alpha, absorbing=False)
        p, t = res.p[0], res.t
        inc = np.abs(p[(t > t_inc) & (t < t_inc + 3e-6)]).max()
        ref = np.abs(p[(t > t_ref - 0.5e-6) & (t < t_ref + 3.5e-6)]).max()
        out["thickness"].append(int(P))
        out["reflection_db"].append(float(20 * np.log10(ref / inc + 1e-30)))
    return out


def has_dz_convergence(dz=(0.4e-3, 0.2e-3, 0.1e-3, 0.05e-3), f=1e6):
    """Hybrid angular spectrum through a linear density gradient
    (1000 to 1500 kg/m^3 over 10 mm) and through a sharp step of the same
    size. The smooth case should converge to the WKB amplitude sqrt(Z2/Z1)
    as the slices get thinner; the sharp case should give the transmission
    coefficient 2 Z2 / (Z1 + Z2) for every slice spacing."""
    out = {"dz": [], "smooth_error": [], "sharp_error": [],
           "smooth_reference": float(np.sqrt(1.5)), "sharp_reference": 1.2}
    for d in dz:
        nz = int(round(30e-3 / d)) + 1
        g = Grid((32, nz), (0.3e-3, d))
        z = g.axis(1)
        errs = []
        for kind in ("smooth", "sharp"):
            if kind == "smooth":
                rho = 1000 + 500 * np.clip((z - 10e-3) / 10e-3, 0, 1)
            else:
                rho = np.where(z >= 15e-3, 1500.0, 1000.0)
            med = Medium.homogeneous(g, Material("w", 1500.0, 1000.0))
            med.rho = np.broadcast_to(rho, g.shape).copy()
            p = field.hybrid_angular_spectrum(np.ones(32, complex), med, f, taper=0.0, pad=1)
            ref = np.sqrt(1.5) if kind == "smooth" else 1.2
            errs.append(float(abs(np.abs(p[16, -1]) - ref) / ref))
        out["dz"].append(float(d))
        out["smooth_error"].append(errs[0])
        out["sharp_error"].append(errs[1])
    return out


def bioheat_convergence(dx=(0.4e-3, 0.2e-3, 0.1e-3), duration=20.0):
    """Spreading of a Gaussian temperature rise (10 K, sigma 1 mm) by
    conduction alone (k = 0.5 W/(m K), rho cp = 4 MJ/(m^3 K)) against the
    analytic solution; largest error after duration [s] and the observed
    order of convergence. The time step is a fixed fraction of the
    stability limit, so it shrinks with the square of the grid step and
    the order covers space and time together."""
    k, rc, s0, a = 0.5, 4e6, 1e-3, 10.0
    D = k / rc
    out = {"dx": [], "max_error": []}
    for h in dx:
        n = int(round(30e-3 / h)) + 1
        g = Grid((n,), h, origin=(-15e-3,))
        x = g.axis(0)
        med = Medium.homogeneous(g, Material("t", 1540.0, 1000.0, k=k, cp=rc / 1000.0))
        T0 = 37.0 + a * np.exp(-x ** 2 / (2 * s0 ** 2))
        start = thermal.BioheatResult(np.array([]), np.zeros((0, n)), T0.copy(), np.zeros(n),
                                      0.0, T_final=T0.copy(), perfusion=np.zeros(n))
        dt = 0.2 * h * h * rc / (2 * k)          # time step shrinks with h^2
        res = thermal.bioheat(med, np.zeros(n), duration, initial=start, dt=dt)
        s2 = s0 ** 2 + 2 * D * duration
        exact = 37.0 + a * s0 / np.sqrt(s2) * np.exp(-x ** 2 / (2 * s2))
        out["dx"].append(float(h))
        out["max_error"].append(float(np.max(np.abs(res.T_final - exact))))
    e = out["max_error"]
    out["order"] = [float(np.log(e[i] / e[i + 1]) / np.log(dx[i] / dx[i + 1]))
                    for i in range(len(e) - 1)]
    return out


def exponent_errors(names=("fat", "muscle", "dermis", "myometrium", "fibroid", "cervix",
                           "prostate"), f_ref=3e6, freqs=(1.5e6, 3e6, 6e6), y_grid=None):
    """Error of the single-exponent full-wave model for each tissue, when
    the grid uses y_grid (default the median of the tissues) and the
    reference frequency f_ref. Returns, per tissue, the ratio of the
    modelled to the true attenuation and the phase speed error [m/s] at
    each frequency. y="tissue" in fullwave.simulate removes this error."""
    mats = {n: tissues.material(n) for n in names}
    yg = float(np.median([m.y for m in mats.values()])) if y_grid is None else float(y_grid)
    out = {"f_ref": f_ref, "freqs": list(freqs), "y_grid": yg, "tissues": {}}
    for n, m in mats.items():
        ratio, dc = fw.exponent_error(m.alpha0, m.y, np.array(freqs), f_ref, yg)
        out["tissues"][n] = {"y": m.y, "attenuation_ratio": [float(r) for r in ratio],
                             "speed_error": [float(d) for d in dc]}
    return out


def run_all(quick=False):
    """Run every study; quick uses fewer and coarser cases (for tests)."""
    from . import run_info
    if quick:
        return {"run_info": run_info(),
                "fullwave_convergence": fullwave_convergence(ppw=(3, 4, 6)),
                "pml_reflection": pml_reflection(thickness=(10, 20)),
                "has_dz_convergence": has_dz_convergence(dz=(0.4e-3, 0.2e-3)),
                "bioheat_convergence": bioheat_convergence(dx=(0.4e-3, 0.2e-3)),
                "exponent_errors": exponent_errors()}
    return {"run_info": run_info(),
            "fullwave_convergence": fullwave_convergence(),
            "pml_reflection": pml_reflection(),
            "has_dz_convergence": has_dz_convergence(),
            "bioheat_convergence": bioheat_convergence(),
            "exponent_errors": exponent_errors()}

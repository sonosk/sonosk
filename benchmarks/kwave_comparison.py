"""Shared-case comparison of the SonoSK full-wave solver with k-Wave.

Both codes run the same 2-D problem: the same grid, time step, perfectly
matched layer outside the grid (20 points, alpha 2), an additive pressure
source at one grid point driven by the same samples, and the same medium:
an absorbing background with a slow layer and a fast, dense cylinder
(power law alpha0 f^1.5). k-Wave runs through k-wave-python with its
compiled C++ (OMP) binary, which is an implementation independent of
SonoSK. The pressure is recorded at five points: transmitted through the
cylinder, reflected back towards the source, to the side, and inside the
layer.

Two runs are made, linear and nonlinear (B/A = 6, a source strong enough
for a few MPa at the sensors; the JSON gives how much the nonlinear traces
of SonoSK differ from the scaled linear ones, to show that the nonlinear
terms matter in this case). The
script writes kwave_comparison.json with the relative L2 error of each
trace, the error of the peak amplitude and the time shift that best aligns
the traces (a shift of one time step comes from the two codes' different
indexing of the source samples and is removed before the errors are
computed), and a figure of the traces.

Needs k-wave-python (pip install k-wave-python) and, for its C++ binary
on Linux, the HDF5 high-level library (libhdf5-hl-100t64 on Ubuntu 24.04).
Tested with k-wave-python 0.6.2, whose kspaceFirstOrder() passes the
source samples to the binary without the scaling of additive sources that
k-Wave applies, 2 dt / (N c0 dx); the script applies it.
"""

import json
import pathlib
import sys
import time

import numpy as np

import sonosk as sk
from sonosk import fullwave as fw

HERE = pathlib.Path(__file__).resolve().parent
N = 216                 # plus 2 x 20 PML points = 256 in both codes
DX = 0.1e-3
F0 = 1.5e6
CYCLES = 3
NT = 1800


def medium_maps():
    x = (np.arange(N) - N // 2) * DX
    z = np.arange(N) * DX
    xx, zz = np.meshgrid(x, z, indexing="ij")
    c = np.full((N, N), 1500.0)
    rho = np.full((N, N), 1000.0)
    a0 = np.full((N, N), 0.5)
    layer = (zz > 6e-3) & (zz < 9e-3)
    c[layer], rho[layer], a0[layer] = 1450.0, 950.0, 0.6
    cyl = np.hypot(xx - 1e-3, zz - 14e-3) < 3e-3
    c[cyl], rho[cyl], a0[cyl] = 1600.0, 1100.0, 1.0
    return c, rho, a0


def signal(dt):
    t = np.arange(NT) * dt
    n = CYCLES / F0
    return np.where(t < n, np.sin(2 * np.pi * F0 * t) * np.sin(np.pi * t / n) ** 2, 0.0)


SOURCE = (N // 2, 20)
SENSORS = {"transmitted": (N // 2 + 10, 190), "reflected": (N // 2, 40),
           "side": (N // 2 + 70, 140), "in layer": (N // 2 - 30, 75),
           "behind layer": (N // 2 - 50, 110)}


def run_sonosk(nonlinear, amp, dt):
    c, rho, a0 = medium_maps()
    g = sk.Grid((N, N), DX)
    med = sk.Medium.homogeneous(g, sk.Material("m", 1500.0, 1000.0, alpha0=0.5, y=1.5, ba=6.0))
    med.c, med.rho, med.alpha0 = c, rho, a0
    s = amp * signal(dt)
    tk = np.arange(NT) * dt

    def sig(t):
        return np.interp(t, tk, s, right=0.0)

    src = fw.Source([SOURCE], sig)
    sens = fw.Sensor(list(SENSORS.values()))
    res = fw.simulate(med, src, (NT - 1) * dt, sensor=sens, dt=dt, pml=20, nonlinear=nonlinear,
                      y=1.5)
    return res.p[:, :NT]


def run_kwave(nonlinear, amp, dt):
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder
    c, rho, a0 = medium_maps()
    kgrid = kWaveGrid((N, N), (DX, DX))
    kgrid.setTime(NT, dt)
    kw = dict(sound_speed=c, density=rho, alpha_coeff=a0, alpha_power=1.5)
    if nonlinear:
        kw["BonA"] = 6.0 * np.ones((N, N))
    medium = kWaveMedium(**kw)
    source = kSource()
    m = np.zeros((N, N), bool)
    m[SOURCE] = True
    source.p_mask = m
    # k-Wave scales an additive pressure source by 2 dt / (N c0 dx) before the
    # C++ binary sees it; the unified kspaceFirstOrder() of k-wave-python 0.6
    # writes the samples unscaled, so the scaling is applied here
    scale = 2 * dt / (2 * c[SOURCE] * DX)
    source.p = (scale * amp * signal(dt))[None, :]
    sensor = kSensor()
    sm = np.zeros((N, N), bool)
    for ij in SENSORS.values():
        sm[ij] = True
    sensor.mask = sm
    r = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", pml_size=20,
                         pml_alpha=2.0, quiet=True)
    p = np.asarray(r["p"])
    if p.ndim == 1:
        p = p[None, :]
    if p.shape[0] != len(SENSORS):
        p = p.T
    # k-Wave orders sensor points column-major (x fastest)
    order = sorted(range(len(SENSORS)), key=lambda i: (list(SENSORS.values())[i][1],
                                                         list(SENSORS.values())[i][0]))
    out = np.empty_like(p)
    for row, i in enumerate(order):
        out[i] = p[row]
    return out[:, :NT]


def compare(a, b):
    """Relative L2 error of b against a after the best integer shift."""
    best = None
    for lag in range(-3, 4):
        bb = np.roll(b, lag)
        if lag > 0:
            bb[:lag] = 0
        elif lag < 0:
            bb[lag:] = 0
        e = np.linalg.norm(bb - a) / np.linalg.norm(a)
        if best is None or e < best[0]:
            best = (e, lag, bb)
    e, lag, bb = best
    peak = abs(np.abs(bb).max() / np.abs(a).max() - 1)
    return float(e), int(lag), float(peak)


def main():
    dt = 0.3 * DX / 1600.0
    results = {"grid": [N, N], "dx_m": DX, "dt_s": dt, "steps": NT, "f0_Hz": F0,
               "run_info": sk.run_info(), "cases": {}}
    traces = {}
    for name, nl, amp in (("linear", False, 1e5), ("nonlinear", True, 1e8)):
        t0 = time.time()
        ps = run_sonosk(nl, amp, dt)
        t1 = time.time()
        pk = run_kwave(nl, amp, dt)
        t2 = time.time()
        case = {"sonosk_s": round(t1 - t0, 1), "kwave_s": round(t2 - t1, 1), "sensors": {}}
        for i, sname in enumerate(SENSORS):
            e, lag, peak = compare(pk[i], ps[i])
            case["sensors"][sname] = {"rel_l2": e, "shift_steps": lag, "peak_error": peak}
        if nl:
            lin = traces["linear"][0] * (amp / 1e5)
            case["nonlinear_vs_scaled_linear"] = [
                float(np.linalg.norm(ps[i] - lin[i]) / np.linalg.norm(lin[i]))
                for i in range(len(SENSORS))]
        results["cases"][name] = case
        traces[name] = (ps, pk)
        print(name, json.dumps(case["sensors"], indent=1), flush=True)
    (HERE / "kwave_comparison.json").write_text(json.dumps(results, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(NT) * dt * 1e6
        fig, ax = plt.subplots(2, len(SENSORS), figsize=(16, 5.5), constrained_layout=True)
        for r, name in enumerate(traces):
            ps, pk = traces[name]
            for i, sname in enumerate(SENSORS):
                lag = results["cases"][name]["sensors"][sname]["shift_steps"]
                ax[r, i].plot(t, pk[i], lw=1.6, label="k-Wave (C++)")
                ax[r, i].plot(t, np.roll(ps[i], lag), "--", lw=1.0, label="SonoSK")
                err = results["cases"][name]["sensors"][sname]["rel_l2"]
                ax[r, i].set_title(f"{name}, {sname}: {100 * err:.2f} %", fontsize=9)
                ax[r, i].set_xlabel("t [us]")
        ax[0, 0].legend(fontsize=8)
        fig.savefig(HERE.parent / "docs" / "figures" / "kwave_comparison.png", dpi=110)
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())

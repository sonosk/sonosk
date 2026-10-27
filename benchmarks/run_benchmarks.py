"""Run the convergence studies of sonosk.benchmarks and save the results.

Writes benchmarks/results.json (numbers and the software versions) and
docs/figures/benchmarks.png. Takes about two minutes on a laptop. Run it
before each release and compare the JSON with the one of the previous
release.
"""

import json
import pathlib
import sys

import numpy as np

from sonosk import benchmarks

HERE = pathlib.Path(__file__).resolve().parent


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    fc = res["fullwave_convergence"]
    for k, v in fc["error"].items():
        ax[0].loglog(fc["ppw"], 100 * np.array(v), "o-", label=k)
    ax[0].set_xlabel("points per wavelength")
    ax[0].set_ylabel("relative L2 error [%]")
    ax[0].set_title(f"Full-wave, cylinder; reference {fc['reference_ppw']} ppw")
    ax[0].legend(fontsize=8)
    pm = res["pml_reflection"]
    ax[1].plot(pm["thickness"], pm["reflection_db"], "o-")
    ax[1].set_xlabel("PML thickness [grid points]")
    ax[1].set_ylabel("reflection [dB]")
    ax[1].set_title("Perfectly matched layer, normal incidence")
    hd = res["has_dz_convergence"]
    ax[2].loglog(np.array(hd["dz"]) * 1e3, hd["smooth_error"], "o-", label="gradient vs WKB")
    ax[2].set_xlabel("slice spacing [mm]")
    ax[2].set_ylabel("relative amplitude error")
    ax[2].set_title("Hybrid angular spectrum")
    ax[2].legend(fontsize=8)
    bh = res["bioheat_convergence"]
    ax[3].loglog(np.array(bh["dx"]) * 1e3, bh["max_error"], "o-")
    ax[3].set_xlabel("grid step [mm]")
    ax[3].set_ylabel("largest error [K]")
    ax[3].set_title(f"Bioheat, Gaussian diffusion (order {np.mean(bh['order']):.2f})")
    fig.savefig(HERE.parent / "docs" / "figures" / "benchmarks.png", dpi=110)


def main():
    res = benchmarks.run_all()
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "exponent_errors"}, indent=1))
    try:
        figure(res)
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())

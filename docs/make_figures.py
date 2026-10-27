"""Run the example scripts and save their figures for the documentation.

    python docs/make_figures.py                 # all examples
    python docs/make_figures.py breast_bmode    # only some of them

The anatomical, full-wave and 3-D therapy examples take a few minutes
each, the therapy sensitivity study about half an hour; all of them together
take about an hour on a laptop.
"""

import pathlib
import runpy
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "figures"


def main(names=None):
    OUT.mkdir(exist_ok=True)
    paths = sorted((ROOT.parent / "examples").glob("*.py"))
    if names:
        paths = [p for p in paths if p.stem in names]
    for path in paths:
        t0 = time.time()
        ns = runpy.run_path(str(path), run_name="figures")
        fig = ns["figure"]()
        fig.savefig(OUT / f"{path.stem}.png", dpi=120)
        plt.close(fig)
        print(f"saved {path.stem} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

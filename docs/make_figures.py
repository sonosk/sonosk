"""Run the example scripts and save their figures for the documentation."""

import pathlib
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "figures"


def main():
    OUT.mkdir(exist_ok=True)
    for path in sorted((ROOT.parent / "examples").glob("*.py")):
        ns = runpy.run_path(str(path), run_name="figures")
        fig = ns["figure"]()
        fig.savefig(OUT / f"{path.stem}.png", dpi=120)
        plt.close(fig)
        print("saved", path.stem)


if __name__ == "__main__":
    main()

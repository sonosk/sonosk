# Benchmarks

`run_benchmarks.py` runs the convergence studies of `sonosk.benchmarks`
(full-wave solver against grid resolution, reflection of the perfectly
matched layer, hybrid angular spectrum against slice spacing, bioheat
solver against an analytic solution, error of the single power-law
exponent) and writes `results.json` and `docs/figures/benchmarks.png`.

`kwave_comparison.py` runs a shared 2-D case, linear and nonlinear, with
SonoSK and with k-Wave (the C++ binary called through k-wave-python) and
writes `kwave_comparison.json` and `docs/figures/kwave_comparison.png`.

The JSON files of each release are kept in the repository, so a change
in the numbers between releases can be seen in the history.

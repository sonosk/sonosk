# Contributing to SonoSK

Bug reports and questions are welcome on the issue tracker. Please include
the SonoSK version, your Python version and a short script that shows the
problem.

## Working on the code

```
git clone https://github.com/sonosk/sonosk.git
cd sonosk
pip install -e ".[test,plot,docs,fast]"
pytest                                   # the test suite
ruff check --select E,F,W --line-length 100 src tests examples benchmarks
python docs/make_tables.py && sphinx-build -W -b html docs docs/_build/html
python benchmarks/run_benchmarks.py      # convergence studies, writes JSON
```

`pytest` also works from a fresh checkout without installing, because the
test configuration adds `src` to the path.

## Code

- Units are SI inside the library. Function arguments in clinical units
  carry the unit in their name, for example `f_mhz` or `z_cm`.
- Every physical constant or tissue value needs a published source, given
  in the module docstring.
- New functions come with tests. Run `pytest` before opening a pull request.

## Copyright of contributions

SonoSK is maintained by Soheil Kooklan, who holds the copyright. By opening
a pull request you confirm that the work is your own, that you license it
under GPL-3.0-or-later, and that you allow Soheil Kooklan to distribute it
under other licence terms as well. Please say so in the pull request if you
do not agree, and the change will not be merged.

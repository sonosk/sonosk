# Installation

SonoSK needs Python 3.11 or newer. The only required package is NumPy
(version 2 or newer). It runs on Windows, Linux and macOS.

## From PyPI

```
pip install sonosk
```

With Matplotlib for the figures and the example scripts:

```
pip install "sonosk[plot]"
```

With SciPy, which makes the full-wave solver about twice as fast through
its multi-threaded FFT:

```
pip install "sonosk[plot,fast]"
```

Check the installed version:

```
python -c "import sonosk; print(sonosk.__version__)"
```

## In a virtual environment

A virtual environment keeps SonoSK and its dependencies apart from other
projects.

On Windows (PowerShell):

```
python -m venv sonosk-env
sonosk-env\Scripts\activate
pip install "sonosk[plot]"
```

On Linux and macOS:

```
python -m venv sonosk-env
source sonosk-env/bin/activate
pip install "sonosk[plot]"
```

## In Google Colab or Kaggle

Run in a notebook cell:

```
!pip install "sonosk[plot]"
```

Large 3-D simulations need more memory than most laptops have, and a
notebook with 16 GB or more is a simple way to run them.

## From the source code

To work on the code or run the tests:

```
git clone https://github.com/sonosk/sonosk.git
cd sonosk
pip install -e ".[test,plot]"
pytest
```

## Building this documentation

```
pip install -e ".[docs,fast]"
python docs/make_tables.py
python docs/make_figures.py
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. `make_tables.py` writes
the tissue table from `sonosk.tissues`; `make_figures.py` runs every
example script and takes about half an hour (`python docs/make_figures.py
breast_bmode` runs one).

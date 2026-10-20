# SonoSK

[![PyPI](https://img.shields.io/pypi/v/sonosk)](https://pypi.org/project/sonosk/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23517377.svg)](https://doi.org/10.5281/zenodo.23517377)
[![tests](https://github.com/sonosk/sonosk/actions/workflows/tests.yml/badge.svg)](https://github.com/sonosk/sonosk/actions/workflows/tests.yml)

SonoSK is a Python library for simulating medical ultrasound. A probe can
be designed, placed in front of a numerical phantom and used to make a
B-mode image, or to heat a target and map the temperature, the thermal
dose and the lesion that remains. The same objects are used for imaging
and therapy, in 1, 2 and 3 dimensions.

Documentation, tutorials and examples: https://sonosk.github.io/sonosk/

![B-mode image of a simulated phantom with a cyst, two lesions and wire targets](https://raw.githubusercontent.com/sonosk/sonosk/main/docs/figures/bmode_phantom.png)

## What it simulates

- Transducers: flat and focused single elements, annular arrays, linear and
  phased arrays with an elevation lens, convex and endocavity arrays,
  matrix arrays, cylindrical line-focus sources. Focusing, steering and
  apodization per element.
- Pressure fields: Rayleigh integral in an attenuating medium and angular
  spectrum propagation, on lines, planes or volumes.
- Phantoms: layered media, spheres, ellipsoids, vessels and boxes built
  from materials with acoustic and thermal properties; point scatterers
  with an echo level per material; wire targets.
- Imaging: plane-wave, diverging-wave and focused transmits, pulse-echo
  signals, delay-and-sum beamforming, coherent compounding, B-mode display.
- Therapy: heat source from the pressure field, Pennes bioheat equation
  with pulsed exposures, CEM43 thermal dose, lesion maps and a treatment
  report with ablated and residual target and damaged healthy tissue.
- Probe design: Mason model of a piezoelectric element with backing and
  matching layers; electrical impedance, transmit and pulse-echo responses.
- Acoustic output: derated pressure and intensity, ISPPA, ISPTA, mechanical
  index, FDA Track 3 limits.
- Image quality: resolution, contrast, CNR, gCNR and speckle SNR.

Each solver is checked against a closed-form result in the test suite (see
the validation page of the documentation). The plan for the next releases,
including the tissue property database and the breast, prostate and uterus
phantoms, is on the roadmap page.

## Installation

SonoSK needs Python 3.11 or newer and NumPy.

```
pip install sonosk
```

For the figures and example scripts:

```
pip install "sonosk[plot]"
```

## Example

Focused field of a 128-element linear array and the mechanical index on
the beam axis:

```python
import numpy as np
import sonosk as sk
from sonosk import field, safety, transducer

probe = transducer.linear_array(128, pitch=0.3e-3, height=5e-3, elev_focus=20e-3)
probe.focus((0, 0, 20e-3), c=1540.0)

g = sk.Grid.centered((121, 151), 0.2e-3, depth_start=2e-3)
p = field.rayleigh_grid(probe, g, f=5e6, c=1540.0, alpha0=0.5)    # p / (rho c u0)

x, z = g.axes()
axis = np.abs(p[60])                          # on the beam axis, x = 0
pr = 2.0 * axis / axis.max()                  # scale to a 2 MPa peak
mi, z_mi = safety.mechanical_index(z * 100, pr, f_mhz=5.0)
print(f"MI = {mi:.2f} at {z_mi * 10:.1f} mm")
```

More scripts, from B-mode phantoms to HIFU ablation of a tumour, are in
the [examples](https://github.com/sonosk/sonosk/tree/main/examples) folder.

## Citing

If you use SonoSK in your work, please cite it:

Kooklan S. SonoSK: medical ultrasound simulation in Python.
Zenodo, 2026. https://doi.org/10.5281/zenodo.23517377

This DOI always points to the latest release. Each release also has its
own DOI on Zenodo; version 0.1.0 is https://doi.org/10.5281/zenodo.23517378.
The full metadata is in
[CITATION.cff](https://github.com/sonosk/sonosk/blob/main/CITATION.cff).

## Licence

SonoSK is free software under the GNU General Public License, version 3 or
later. See [LICENSE](https://github.com/sonosk/sonosk/blob/main/LICENSE).

## Disclaimer

SonoSK is research software. It is not a medical device, and its results
must not be used for clinical decisions or for the safety assessment of a
device on the market.

## Author

Soheil Kooklan ([ORCID 0009-0003-5035-7833](https://orcid.org/0009-0003-5035-7833))

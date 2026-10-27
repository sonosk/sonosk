# SonoSK

[![PyPI](https://img.shields.io/pypi/v/sonosk)](https://pypi.org/project/sonosk/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23517377.svg)](https://doi.org/10.5281/zenodo.23517377)
[![tests](https://github.com/sonosk/sonosk/actions/workflows/tests.yml/badge.svg)](https://github.com/sonosk/sonosk/actions/workflows/tests.yml)

SonoSK is a Python library for simulating medical ultrasound. A probe can
be designed, placed on an anatomical phantom of the breast, prostate,
uterus or face, and used to make a B-mode image, or to heat a target and
map the temperature, the thermal dose and the lesion that remains. Every
tissue value comes with its published source. The same objects are used
for imaging and therapy, in 1, 2 and 3 dimensions.

Documentation, tutorials and examples: https://sonosk.github.io/sonosk/

SonoSK is for research and teaching. Its models rest on stated
assumptions: soft tissue and bone are fluids, the fast imaging model
follows straight rays, the full-wave solver uses one power-law exponent
unless asked for one per tissue, 2-D full-wave images treat every
inclusion as a cylinder, heating uses the attenuation as absorption unless
told otherwise, and boiling and cavitation are not modelled. Echo levels
are modelling choices, and the tissue values differ in how they were
obtained (`tissues.provenance()`). The
[Scope and limits](https://sonosk.github.io/sonosk/limitations.html) page
lists them all.

![Breast phantom, its sound speed, and simulated B-mode images beamformed at 1540 m/s and through the speed map](https://raw.githubusercontent.com/sonosk/sonosk/main/docs/figures/breast_bmode.png)

## What it simulates

- Transducers: flat and focused single elements, annular arrays, linear and
  phased arrays with an elevation lens, convex and endocavity arrays,
  matrix arrays, cylindrical line-focus sources and spherical HIFU phased
  arrays. Focusing, steering and apodization per element.
- Propagation: Rayleigh integral, angular spectrum, the hybrid angular
  spectrum method through heterogeneous tissue, pulsed fields, and a
  k-space pseudospectral full-wave solver for heterogeneous, absorbing and
  nonlinear media in 1, 2 and 3 dimensions.
- Tissues: a property database (speed of sound, density, attenuation,
  B/A, thermal properties, perfusion, echo level) with the reference of
  every value.
- Anatomical phantoms: breast with cysts, fibroadenomas and spiculated
  carcinomas; prostate seen from the rectum with zones, tumour and
  calcifications; uterus with fibroids behind the abdominal wall and the
  bladder; facial skin and SMAS.
- Imaging: plane-wave, diverging-wave and focused transmits through
  heterogeneous tissue (aberration, shadowing, enhancement),
  frequency-dependent attenuation, delay-and-sum and sound-speed-aware
  beamforming, compounding, sector images, full-wave pulse-echo imaging.
- Therapy: heat sources including harmonics, Pennes bioheat equation,
  CEM43 dose and Arrhenius damage, perfusion shutdown, treatment plans with
  many sonications or volumetric trajectories, and a report of the ablated
  and residual target and the damaged healthy tissue.
- Probe design: Mason model of a piezoelectric element with backing and
  matching layers, whose pulse goes straight into the imaging model.
- Acoustic output: derated pressure and intensity, ISPPA, ISPTA, mechanical
  index, and a screening check against the FDA Track 3 limits.
- Image quality: resolution, contrast, CNR, gCNR and speckle SNR.

Each solver is checked against a closed-form result or an independent
method in the test suite (see the validation page of the documentation).
Convergence studies and a comparison of the full-wave solver with k-Wave
on a shared case are on the benchmarks page, with the numbers in
`benchmarks/*.json`. The plan for the next releases is on the roadmap
page.

## Installation

SonoSK needs Python 3.11 or newer and NumPy 2. With SciPy installed
(`pip install "sonosk[fast]"`) the full-wave solver uses its faster
multi-threaded FFT.

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

A transrectal image of a prostate phantom with a tumour:

```python
import numpy as np
import sonosk as sk
from sonosk import imaging, organs, phantom, transducer

g = sk.Grid.centered((561, 451), 0.1e-3, depth_start=0.0)
ph = organs.prostate(g, tumors=[organs.Tumor((9e-3, 0, 11e-3), radius=3.5e-3,
                                             tissue="prostate carcinoma")], rng=2)
pos, amp = phantom.scatterers(ph.medium, density=40, rng=2)
pos_i, amp_i = phantom.interface_scatterers(ph.medium, gain=27.0)   # interface echoes
pos, amp = np.concatenate([pos, pos_i]), np.concatenate([amp, amp_i])
probe = transducer.convex_array(128, 2 * np.deg2rad(68) * 10e-3 / 128, 10e-3)
pulse = imaging.Pulse(6.5e6, 0.65)
tx = imaging.Transmit.diverging((0, 0, -10e-3))
rf, fs = imaging.simulate(probe, tx, pos, amp, pulse, medium=ph.medium, interfaces=True)
x = np.arange(-28e-3, 28e-3, 0.1e-3)
z = np.arange(0.5e-3, 44e-3, 0.1e-3)
img = imaging.beamform(rf, fs, probe, tx, x, z, f_number=0.7)
```

More scripts, from breast and uterus images to fibroid and prostate HIFU
treatment plans, are in the
[examples](https://github.com/sonosk/sonosk/tree/main/examples) folder.

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

SonoSK is research and teaching software. It is not a medical device, and
its results must not be used for clinical decisions, for planning the
treatment of a patient, or for the safety assessment of a device on the
market. The acoustic output check is a screening aid, not a regulatory
determination.

## Author

Soheil Kooklan ([ORCID 0009-0003-5035-7833](https://orcid.org/0009-0003-5035-7833))

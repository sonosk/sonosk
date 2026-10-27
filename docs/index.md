---
myst:
  html_meta:
    "description lang=en": "SonoSK is a Python library for simulating medical ultrasound: transducers and arrays, full-wave and heterogeneous propagation, B-mode imaging of anatomical breast, prostate and uterus phantoms, tissue properties with sources, HIFU treatment planning, thermal dose and lesion maps, and piezoelectric stack design."
    "keywords": "ultrasound simulation, Python, k-space pseudospectral, full-wave, hybrid angular spectrum, transducer, linear array, convex array, beamforming, sound speed, aberration, B-mode, plane wave imaging, breast phantom, prostate, uterine fibroid, HIFU, focused ultrasound, treatment planning, bioheat, thermal dose, Arrhenius, tissue properties, mechanical index, Mason model"
---

# SonoSK

SonoSK is a Python library for simulating medical ultrasound. A probe can
be designed, placed on an anatomical phantom of the breast, prostate,
uterus or face, and used to make a B-mode image, or to heat a target and
map the temperature, the thermal dose and the lesion that remains. Every
tissue value comes with its published source. The same transducer, medium
and phantom objects serve imaging and therapy, in 1, 2 and 3 dimensions.

```{image} figures/breast_bmode.png
:alt: Breast phantom, its sound speed, and B-mode images beamformed at 1540 m/s and through the speed map
```

```{image} figures/fibroid_hifu.png
:alt: Pressure field through the abdominal wall, peak temperature and lesion over a uterine fibroid, and the treatment report
```

## Contents of version 0.2.0

| Part | What it does |
|------|--------------|
| [Transducers](guide/transducers.md) | piston, focused bowl, annular, linear, phased, convex and endocavity, matrix, cylindrical line-focus, spherical HIFU phased arrays; focusing, steering, apodization |
| [Pressure fields](guide/fields.md) | Rayleigh integral, angular spectrum, hybrid angular spectrum through heterogeneous tissue, pulsed fields |
| [Full-wave simulation](guide/fullwave.md) | k-space pseudospectral solver: heterogeneous, absorbing and nonlinear media, full-wave imaging, harmonics |
| [Tissue properties](guide/tissues.md) | acoustic and thermal properties with a reference for every value |
| [Anatomical phantoms](guide/organs.md) | breast, prostate, uterus and face, with cysts, fibroadenomas, carcinomas, tumours, fibroids and calcifications |
| [Media and phantoms](guide/phantoms.md) | materials, property maps, shapes, random structure, scatterers |
| [Imaging](guide/imaging.md) | pulse-echo signals through heterogeneous tissue, frequency-dependent attenuation, compounding, sound-speed-aware beamforming, sector images |
| [Therapy](guide/therapy.md) | heat sources, bioheat equation, CEM43 and Arrhenius damage, perfusion shutdown, multi-point treatment plans, treatment report |
| [Probe design](guide/probe-design.md) | Mason model of a piezoelectric element; its pulse used in the imaging model |
| [Acoustic output](guide/acoustic-output.md) | derating, ISPPA, ISPTA, mechanical index, FDA limits |
| [Image quality](guide/image-quality.md) | resolution, contrast, CNR, gCNR, speckle SNR |

What the models assume and leave out is on the
[Scope and limits](limitations.md) page; read it before using a number from
SonoSK. How each solver was checked is on the [Validation](validation.md)
page, and the convergence studies and the comparison with k-Wave are on
the [Benchmarks](benchmarks.md) page.
What comes next, with the version it is planned for, is on the
[Roadmap](roadmap.md).

## Getting started

```
pip install "sonosk[plot]"
```

Then follow [Getting started](quickstart.md). Every figure in this guide
comes from a script in the [Examples](examples.md).

SonoSK is research and teaching software. It is not a medical device, and
its results must not be used for clinical decisions, for planning the
treatment of a patient or for regulatory claims.

```{toctree}
:hidden:
:caption: User guide

installation
quickstart
guide/grid
guide/transducers
guide/fields
guide/fullwave
guide/tissues
guide/organs
guide/phantoms
guide/imaging
guide/therapy
guide/probe-design
guide/acoustic-output
guide/image-quality
guide/analytic
guide/water
examples
```

```{toctree}
:hidden:
:caption: About

limitations
validation
benchmarks
roadmap
citing
changelog
contributing
api
```

---
myst:
  html_meta:
    "description lang=en": "SonoSK is a Python library for simulating medical ultrasound: transducers and arrays, pressure fields, B-mode imaging of tissue phantoms, HIFU heating, thermal dose and lesion maps, and piezoelectric stack design."
    "keywords": "ultrasound simulation, Python, transducer, linear array, phased array, beamforming, B-mode, plane wave imaging, HIFU, focused ultrasound, bioheat, thermal dose, mechanical index, Rayleigh integral, angular spectrum, Mason model"
---

# SonoSK

SonoSK is a Python library for simulating medical ultrasound. A probe can
be designed, placed in front of a numerical phantom and used to make a
B-mode image, or to heat a target and map the temperature, the thermal
dose and the lesion that remains. The same transducer, medium and phantom
objects serve imaging and therapy, in 1, 2 and 3 dimensions.

```{image} figures/bmode_phantom.png
:alt: Phantom label map, simulated B-mode image with a cyst, two lesions and wire targets, and image quality measures
```

```{image} figures/hifu_ablation.png
:alt: Peak temperature around a focused ultrasound exposure with the lesion outline, temperature at the focus and the treatment report
```

## Contents of version 0.1.0

| Part | What it does |
|------|--------------|
| [Transducers](guide/transducers.md) | piston, focused bowl, annular, linear, phased, convex and endocavity, matrix and cylindrical line-focus sources; focusing, steering, apodization |
| [Pressure fields](guide/fields.md) | Rayleigh integral in an attenuating medium, angular spectrum method |
| [Media and phantoms](guide/phantoms.md) | materials, property maps, layers and inclusions, point scatterers |
| [Imaging](guide/imaging.md) | pulse-echo signals, plane, diverging and focused transmits, delay-and-sum, compounding, B-mode |
| [Therapy](guide/therapy.md) | heat source, Pennes bioheat equation, CEM43 dose, lesion and treatment report |
| [Probe design](guide/probe-design.md) | Mason model of a piezoelectric element with backing and matching layers |
| [Acoustic output](guide/acoustic-output.md) | derating, ISPPA, ISPTA, mechanical index, FDA limits |
| [Image quality](guide/image-quality.md) | resolution, contrast, CNR, gCNR, speckle SNR |

How each solver was checked is on the [Validation](validation.md) page.
What comes next, with the version it is planned for, is on the
[Roadmap](roadmap.md).

## Getting started

```
pip install "sonosk[plot]"
```

Then follow [Getting started](quickstart.md). Every figure in this guide
comes from a script in the [Examples](examples.md).

SonoSK is research software. It is not a medical device, and its results
must not be used for clinical decisions.

```{toctree}
:hidden:
:caption: User guide

installation
quickstart
guide/grid
guide/transducers
guide/fields
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

validation
roadmap
citing
changelog
contributing
api
```

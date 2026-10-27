# Getting started

This page goes through one complete simulation: a probe, a phantom, an
image and its quality, then a heating exposure of the same kind of
medium.

## Units and axes

Inside the library all quantities are SI: metres, seconds, pascals,
kilograms per cubic metre. Where a function takes clinical units, the unit
is part of the argument name, as in `f_mhz`, `z_cm` or `pr_mpa`.
Attenuation `alpha0` is always in dB/(cm MHz^y).

The probe face is near z = 0 and the beam goes towards +z. x is the array
direction and y the elevation. On a `Grid` the last axis is depth, so a
2-D grid is (x, z) and a 3-D grid is (x, y, z).

## 1. A probe

```python
import numpy as np
import sonosk as sk
from sonosk import transducer

probe = transducer.linear_array(128, pitch=0.3e-3, height=5e-3, dx=0.15e-3)
print(probe)                 # Transducer('linear', elements=128, patches=...)
```

`dx` is the patch size used to split each element; see
[Transducers](guide/transducers.md).

## 2. A phantom

A phantom is a label map on a grid plus one `Material` per label.

```python
from sonosk import phantom

g = sk.Grid.centered((201, 301), 0.1e-3, depth_start=5e-3)
mats = [sk.Material("background", 1540, 1000, alpha0=0.5, echo=0.0),
        sk.Material("cyst", 1540, 1000, alpha0=0.5, echo=-np.inf)]
labels = np.zeros(g.shape, int)
labels = phantom.paint(labels, phantom.sphere(g, (0, 0, 20e-3), 3e-3), 1)
medium = sk.Medium(g, labels, mats)

pos, amp = phantom.scatterers(medium, density=60, rng=0)
```

`echo` sets the backscatter level of each material in dB; `-inf` makes it
anechoic.

## 3. An image

```python
from sonosk import imaging

pulse = imaging.Pulse(5e6, bandwidth=0.6)
transmits = [imaging.Transmit.plane(a)
             for a in imaging.plane_wave_angles(7, np.deg2rad(12))]
rfs = [imaging.simulate(probe, tx, pos, amp, pulse, alpha0=0.5)[0]
       for tx in transmits]

x = np.arange(-9e-3, 9e-3, 0.075e-3)
z = np.arange(6e-3, 34e-3, 0.075e-3)
img = imaging.compound(rfs, 8 * pulse.f0, probe, transmits, x, z)
```

`img` is complex; its modulus is the envelope. To show it:

```python
import matplotlib.pyplot as plt
import sonosk.plot as splt

splt.bmode(img, x, z, dynamic_range=50)
plt.show()
```

## 4. Image quality

```python
from sonosk import quality

env = np.abs(img)
zz, xx = np.meshgrid(z, x, indexing="ij")
r = np.hypot(xx, zz - 20e-3)
inside = env[r < 2.4e-3]
background = env[(r > 4e-3) & (np.abs(zz - 20e-3) < 4e-3)]
print(quality.contrast_db(inside, background), quality.gcnr(inside, background))
```

## 5. Heating

For therapy the field is computed as a continuous wave, turned into a heat
source and passed to the bioheat solver.

```python
from sonosk import field, thermal

src = transducer.bowl(radius=20e-3, roc=50e-3, dx=0.4e-3)
g3 = sk.Grid.centered((31, 31, 51), 0.5e-3, depth_start=38e-3)
tissue = sk.Material("tissue", 1540, 1050, alpha0=0.5, k=0.5, cp=3600, perfusion=0.5)
med3 = sk.Medium.homogeneous(g3, tissue)

p = field.rayleigh_grid(src, g3, f=1e6, c=tissue.c, alpha0=tissue.alpha0)
p = 3e6 * np.abs(p) / np.abs(p).max()              # 3 MPa at the focus
q = thermal.heat_source(p, med3.attenuation(1e6), med3.rho, med3.c)

res = thermal.bioheat(med3, q, duration=20.0,
                      schedule=lambda t: 1.0 if t < 8.0 else 0.0)
print(res.T_max.max(), res.lesion().sum())
```

The tissue numbers here are only an example. The
[Therapy](guide/therapy.md) page explains each step, and the HIFU example
adds a tumour and the treatment report.

## 6. Real tissue and anatomy

The same steps work on an anatomical phantom with measured tissue values:

```python
from sonosk import organs, tissues

print(tissues.info("breast carcinoma"))       # values, units and sources
g = sk.Grid.centered((401, 401), 0.1e-3, depth_start=0.0)
ph = organs.breast(g, lesions=[organs.Carcinoma((5e-3, 0, 15e-3), radius=3e-3)], rng=1)
pos, amp = phantom.scatterers(ph.medium, density=50, rng=1)
probe = transducer.linear_array(128, pitch=0.3e-3, height=4e-3, dx=0.15e-3)
pulse = imaging.Pulse(7.5e6, 0.65)
tx = imaging.Transmit.plane(0.0)
rf, fs = imaging.simulate(probe, tx, pos, amp, pulse, medium=ph.medium)
x = np.arange(-18e-3, 18e-3, 0.08e-3)
z = np.arange(1e-3, 39e-3, 0.08e-3)
img = imaging.beamform(rf, fs, probe, tx, x, z, medium=ph.medium)
img = imaging.tgc(img, z, 0.75, pulse.f0)
```

With `medium=` the echoes travel through the speeds and attenuations of
the tissues, and the beamformer can use the same speed map. See
[Anatomical phantoms](guide/organs.md) and [Imaging](guide/imaging.md).

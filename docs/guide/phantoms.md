# Media and phantoms

## Materials

A `Material` holds the acoustic and thermal properties of one tissue or
phantom material:

| Field | Meaning | Unit |
|-------|---------|------|
| `c` | sound speed | m/s |
| `rho` | density | kg/m³ |
| `alpha0`, `y` | attenuation $\alpha_0 f^y$ | dB/(cm MHz^y) |
| `ba` | nonlinearity B/A | |
| `k` | thermal conductivity | W/(m K) |
| `cp` | specific heat | J/(kg K) |
| `perfusion` | blood mass flow per volume | kg/(m³ s) |
| `echo` | backscatter level relative to a reference | dB |
| `source` | where the values come from | |

```python
import numpy as np
import sonosk as sk

gel = sk.Material("gel", c=1540, rho=1000, alpha0=0.5, k=0.55, cp=4000)
gel.impedance            # rho c
gel.attenuation(5e6)     # Np/m at 5 MHz
soft = gel.with_(alpha0=0.7, name="soft")
```

Measured values for real tissues come from the
[tissue database](tissues.md), `sonosk.tissues.material("fat")`, where
every number carries its source. For your own materials, fill in
`source`.

## Media

A `Medium` turns a label map into property maps on a grid:

```python
from sonosk import phantom

g = sk.Grid.centered((201, 301), 0.1e-3, depth_start=0.0)
labels = phantom.layers(g, [2e-3, 8e-3])    # 0 above, 1, 2, then 3 below
mats = [water, skin, fat, gland]            # one Material per label
med = sk.Medium(g, labels, mats)

med.c, med.rho, med.alpha0                   # arrays with the grid shape (computed on demand)
med.rho = rho_map                            # replace one map, e.g. to add variations
sub, sl = med.crop(low, high)                # sub-medium and the slices that cut it
med2d = med3d.slice2d(y=0.0)                 # x-z plane of a 3-D medium
med.mask("fat")                              # where the material is fat
med.volume("gland")                          # area in 2-D, volume in 3-D
```

## Shapes

Every shape returns a boolean mask on the grid. Positions are given as
(x, y, z); on a 2-D grid y is ignored.

```python
phantom.sphere(g, center=(0, 0, 20e-3), radius=3e-3)
phantom.ellipsoid(g, center=(0, 0, 25e-3), semi_axes=(5e-3, 5e-3, 2e-3), angle=0.3)
phantom.cylinder(g, point=(0, 0, 15e-3), direction=(1, 0, 0), radius=1e-3)
phantom.box(g, low=(-5e-3, -5e-3, 10e-3), high=(5e-3, 5e-3, 12e-3))
phantom.halfspace(g, depth=30e-3)
```

`paint` writes a label into a mask, so phantoms are built from the outside
in:

```python
labels = np.zeros(g.shape, int)
labels = phantom.paint(labels, phantom.halfspace(g, 2e-3), 1)
labels = phantom.paint(labels, phantom.sphere(g, (0, 0, 20e-3), 4e-3), 2)
```

## Scatterers for imaging

```python
pos, amp = phantom.scatterers(med, density=60, rng=0)
```

`density` is the number of scatterers per mm² on a 2-D grid and per mm³ on
a 3-D grid. Ten or more per resolution cell give fully developed speckle.
The amplitude of each scatterer is a normal random number times
$10^{\mathrm{echo}/20}$ of its material. `point_targets` adds wires:

```python
wires, wamp = phantom.point_targets([(0, 0, 30e-3), (5e-3, 0, 30e-3)], amplitude=20.0)
```

## Interface echoes

```python
pos_i, amp_i = phantom.interface_scatterers(med, gain=30.0)
```

In the weak-contrast (Born) limit the reflection of a layered medium is
the depth derivative of $\tfrac12\ln Z$, so a step from $Z_1$ to $Z_2$
reflects about $(Z_2-Z_1)/(Z_2+Z_1)$. Each grid point gets a reflector of
amplitude `gain` times $\tfrac12\Delta\ln Z$ over one grid step in depth,
with $Z$ smoothed over one grid step to soften the staircase of the label
map. The reflectors are dense, so their echoes add up into a specular
echo. `gain` sets the level of these echoes against the diffuse
scatterers and is a modelling choice: the default 30 was set by eye on
the breast example (0.1 mm grid, 50 scatterers per mm², 7.5 MHz). The
speckle grows with the square root of the scatterer density and the
interface echo with the number of reflectors per unit length, so for
other settings use `gain = 30 * sqrt(density / 50) * dx / 0.1e-3`.

## Displaying a phantom

```python
import sonosk.plot as splt
x, z = g.axes()
splt.labels_map(med.labels, x, z, names=[m.name for m in med.materials])
```

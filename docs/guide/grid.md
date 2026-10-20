# Grid

A `Grid` is a set of equally spaced points in 1, 2 or 3 dimensions. Fields,
phantoms and temperature maps in SonoSK are stored on one.

The last axis is always depth, measured away from the transducer:

| Dimensions | Axes      |
|------------|-----------|
| 1          | (z)       |
| 2          | (x, z)    |
| 3          | (x, y, z) |

```python
import sonosk

g = sonosk.Grid((128, 128, 256), spacing=(0.1e-3, 0.1e-3, 0.05e-3))
g.ndim            # 3
g.extent          # length along each axis [m]
x, y, z = g.axes()
X, Y, Z = g.mesh()
pts = g.points()  # (N, 3) array of coordinates
```

`Grid.centered` puts the lateral axes symmetric about zero and starts the
depth axis at a given value, which is the usual layout for a probe at the
top of the image:

```python
g = sonosk.Grid.centered((256, 512), 50e-6, depth_start=5e-3)
```

## Sampling

`points_per_wavelength(c, f)` gives the number of grid points per
wavelength for the slowest sound speed `c` at frequency `f`, using the
coarsest spacing. The Rayleigh integral can be evaluated on any grid, so
the spacing there only sets how finely the field is shown. The bioheat
solver needs the heated region resolved by several points across, and
the scatterer generator needs a spacing small against the resolution
cell, since each scatterer takes the label of its nearest grid point.

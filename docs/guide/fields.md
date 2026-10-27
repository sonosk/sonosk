# Pressure fields

`sonosk.field` computes continuous-wave (CW) pressure fields. The time
convention is $e^{j\omega t}$.

## Rayleigh integral

For a source with normal velocity amplitude $u_0$, the pressure at a point
$\mathbf{r}$ is

$$
\frac{p(\mathbf{r})}{\rho c u_0} = \frac{jk}{2\pi} \sum_n w_n A_n
\frac{e^{-(jk + \alpha) R_n}}{R_n},
$$

where $A_n$ is the area of patch $n$, $R_n$ its distance to the point,
$w_n$ its complex weight from the apodization and the firing delay, and
$\alpha$ the amplitude attenuation of a homogeneous medium in Np/m. For a
curved source each patch is treated as part of a baffled plane, the usual
approximation behind the O'Neil solution.

```python
import numpy as np
import sonosk as sk
from sonosk import field, transducer

bowl = transducer.bowl(20e-3, 50e-3, dx=0.25e-3)
pts = np.array([[0, 0, 45e-3], [0, 0, 50e-3], [1e-3, 0, 50e-3]])
p = field.rayleigh(bowl, pts, f=1e6, c=1500.0)              # p / (rho c u0)

g = sk.Grid.centered((121, 181), 0.25e-3, depth_start=20e-3)
pg = field.rayleigh_grid(bowl, g, f=1e6, c=1500.0, alpha0=0.5)
```

`rayleigh_grid` evaluates a 1-D grid on the z axis, a 2-D grid in the x-z
plane (at `plane_y`) and a 3-D grid everywhere. The cost grows with the
number of points times the number of patches; `max_pairs` limits the
memory used at once.

```{image} ../figures/bowl_field.png
:alt: Field of a focused bowl in the x-z plane and on axis, with O'Neil's solution
```

## Angular spectrum

When the field is known on a plane, the angular spectrum method carries
it to other depths with fast Fourier transforms:

```python
x = np.arange(-30e-3, 30e-3, 0.5e-3)
X, Y = np.meshgrid(x, x, indexing="ij")
plane = field.rayleigh(bowl, np.stack([X.ravel(), Y.ravel(), np.full(X.size, 5e-3)], 1),
                       1e6, 1500.0).reshape(X.shape)
vol = field.angular_spectrum(plane, dx=0.5e-3, f=1e6, z=[10e-3, 20e-3, 30e-3], c=1500.0)
```

A 1-D input is a field that does not change along y, which turns the
problem into a 2-D one. Use `pad` to reduce wrap-around from the periodic
transform, and a plane large enough to hold the beam.

## Physical units

The solvers return $p / (\rho c u_0)$. `scale_to_pressure` gives pascals
from the surface velocity, or from the acoustic power of a uniform plane
source, and `intensity` gives the plane-wave intensity
$|p|^2 / (2 \rho c)$.

```python
p_pa = field.scale_to_pressure(p, rho=1000.0, c=1500.0, u0=0.05)
I = field.intensity(p_pa, rho=1000.0, c=1500.0)                # W/m^2
```

## Heterogeneous media: hybrid angular spectrum

`hybrid_angular_spectrum` carries a field given on the first depth plane
of a medium through all its tissues, slice by slice. Each step propagates
the angular spectrum at the mean wavenumber of the slice, then corrects
the phase and amplitude point by point for the local speed and
attenuation, and multiplies by the pressure transmission coefficient
$2Z_2/(Z_1+Z_2)$ between slices (Vyas and Christensen 2012). It is the
method behind several MR-guided focused ultrasound planning studies
(Johnson et al. 2018) and is fast enough for 3-D fields of HIFU
treatments: a 209 by 209 by 231 grid takes well under a minute.

```python
plane = sk.Grid(grid.shape[:2] + (1,), grid.spacing, grid.origin)   # first depth plane
p0 = field.rayleigh_grid(bowl, plane, f, c_water)[..., 0]
p = field.hybrid_angular_spectrum(p0, medium, f)                     # grid shape
```

In water it agrees with the Rayleigh integral to better than 0.3 % of the
peak on the axis of a focused bowl, and through an interface it
reproduces the transmission coefficient and the attenuation (see
[Validation](../validation.md)). The transmission coefficient is applied
between every pair of slices. At a sharp step this gives
$2Z_2/(Z_1+Z_2)$ for any slice spacing; through a smooth impedance
gradient the product of the many small factors converges, as the slices
get thinner, to $\sqrt{Z_2/Z_1}$, the amplitude that conserves the
energy flux. The error is first order in the slice spacing (0.08 % at
0.4 mm and 0.01 % at 0.05 mm for a gradient over 10 mm; see
[Benchmarks](../benchmarks.md)). The method is one-way: it neglects
reflected waves, scattering and nonlinearity; use the
[full-wave solver](fullwave.md) for those.

## Pulsed fields

`pulsed` synthesises the transient field of a pulsed excitation from
Rayleigh integrals over the pulse spectrum, so each frequency is
attenuated with its own power law:

```python
t, p = field.pulsed(array, points, imaging.Pulse(5e6, 0.6), c=1540.0, alpha0=0.5)
```

The peak rarefactional pressure and the pulse intensity integral of `p`
feed the acoustic output measures of [`sonosk.safety`](acoustic-output.md).

## References

O'Neil H.T. Theory of focusing radiators. J. Acoust. Soc. Am.
21(5):516-526, 1949.

Johnson S.L., Christensen D.A., Dillon C.R., Payne A. Validation of hybrid
angular spectrum acoustic and thermal modelling in phantoms. Int. J.
Hyperthermia 35(1):578-590, 2018.

Vyas U., Christensen D. Ultrasound beam simulations in inhomogeneous tissue
geometries using the hybrid angular spectrum method. IEEE Trans. Ultrason.
Ferroelectr. Freq. Control 59(6):1093-1100, 2012.

Zeng X., McGough R.J. Evaluation of the angular spectrum approach for
simulations of near-field pressures. J. Acoust. Soc. Am. 123(1):68-76,
2008.

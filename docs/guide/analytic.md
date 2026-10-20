# Analytic fields

Closed-form solutions are the first check for any numerical field code.
`sonosk.analytic` has the on-axis pressure of two circular sources in a
lossless fluid. Results are amplitudes normalised by $p_0 = \rho c u_0$,
where $u_0$ is the normal velocity of the source surface.

## Flat piston

For a baffled piston of radius $a$, at distance $z$ on its axis
(Kinsler et al. 2000):

$$
\left|\frac{p(z)}{p_0}\right| = 2\left|\sin\left[\frac{k}{2}\left(\sqrt{z^2 + a^2} - z\right)\right]\right|
$$

with $k = 2\pi f / c$. The amplitude reaches 2 at each maximum. The last
maximum, which marks the end of the near field, lies at
$z = a^2/\lambda - \lambda/4$.

```python
import numpy as np
from sonosk import analytic

z = np.linspace(1e-4, 0.1, 5000)
p = analytic.piston_axial(z, a=5e-3, f=2e6)
analytic.piston_last_maximum(5e-3, 2e6)     # [m]
```

## Focused bowl

For a spherical bowl with aperture radius $a$ and radius of curvature $R$,
O'Neil (1949) gives

$$
\left|\frac{p(z)}{p_0}\right| = \left|\frac{2}{1 - z/R}
\sin\left[\frac{k}{2}\left(\sqrt{(z - h)^2 + a^2} - z\right)\right]\right|,
\qquad h = R - \sqrt{R^2 - a^2}
$$

where $z$ is measured from the apex of the bowl and $h$ is the depth of
the bowl. At the geometric focus, $z = R$, the expression tends to the
focal gain $kh$, and `bowl_axial` returns that value there.

```python
p = analytic.bowl_axial(z, a=15e-3, roc=40e-3, f=3e6)
analytic.focal_gain(15e-3, 40e-3, 3e6)      # k h
```

The peak of the on-axis pressure lies a little closer to the bowl than the
geometric focus. The shift grows as $kh$ gets smaller.

## References

Kinsler L.E., Frey A.R., Coppens A.B., Sanders J.V. Fundamentals of
Acoustics, 4th ed. Wiley, 2000.

O'Neil H.T. Theory of focusing radiators. J. Acoust. Soc. Am.
21(5):516-526, 1949.

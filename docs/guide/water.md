# Water

Water is the reference medium for transducer measurements and for the
derated acoustic output values, so its properties are given as functions
of temperature.

## Sound speed

Marczak (1997) fitted a fifth-order polynomial to measured sound speed in
pure water between 0 and 95 °C:

$$
c(T) = 1402.385 + 5.038813\,T - 5.799136\times10^{-2}\,T^2
+ 3.287156\times10^{-4}\,T^3 - 1.398845\times10^{-6}\,T^4
+ 2.787860\times10^{-9}\,T^5
$$

with $T$ in °C and $c$ in m/s. At 20 °C it gives 1482.4 m/s and at 37 °C
1523.7 m/s.

## Density

The density follows Kell (1975), valid from 0 to 150 °C:

$$
\rho(T) = \frac{999.83952 + 16.945176\,T - 7.9870401\times10^{-3}\,T^2
- 46.170461\times10^{-6}\,T^3 + 105.56302\times10^{-9}\,T^4
- 280.54253\times10^{-12}\,T^5}{1 + 16.879850\times10^{-3}\,T}
$$

in kg/m³.

```python
from sonosk import water

water.sound_speed(20.0)    # m/s
water.density(20.0)        # kg/m^3
water.impedance(20.0)      # kg/(m^2 s)
```

Temperatures outside the range of each fit raise a `ValueError`.

## References

Marczak W. Water as a standard in the measurements of speed of sound in
liquids. J. Acoust. Soc. Am. 102(5):2776-2779, 1997.

Kell G.S. Density, thermal expansivity, and compressibility of liquid
water from 0 to 150 °C. J. Chem. Eng. Data 20(1):97-105, 1975.

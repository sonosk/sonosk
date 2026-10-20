# Validation

Every solver in SonoSK is compared with a closed-form result or with an
independent method. These comparisons are part of the test suite in the
`tests` folder and run on every change of the code, on Linux and Windows.

| Quantity | Reference | Agreement in the tests |
|----------|-----------|------------------------|
| On-axis field of a flat piston, Rayleigh integral | closed form (Kinsler et al. 2000) | better than 0.5 % beyond 20 mm |
| On-axis field of a focused bowl, Rayleigh integral | O'Neil (1949) | better than 0.5 % of the peak |
| Attenuation in the Rayleigh integral | $\alpha_0 f z$ in dB | 1 % |
| Angular spectrum propagation | Rayleigh integral on the target plane | 4 % of the peak |
| Area of discretised piston and bowl | $\pi a^2$, $2\pi R h$ | exact |
| Focusing delays | equal arrival time at the focus | exact |
| Axial resolution of a simulated point target | $-6$ dB width of the two-way Gaussian pulse | 5 % |
| Position of a simulated point target | true position | within $\lambda/8$ axially |
| Cyst contrast in a simulated image | anechoic region | below $-15$ dB, gCNR above 0.8 |
| Speckle of a Rayleigh envelope | $\sqrt{\pi/(4-\pi)} = 1.913$ (Burckhardt 1978) | 1 % |
| Heating without conduction or perfusion | $\Delta T = Q t / (\rho c_p)$ | exact |
| Heating with perfusion only | $\Delta T = \frac{Q}{w_b c_b}\left(1 - e^{-t w_b c_b / \rho c_p}\right)$ | 0.1 % |
| Conduction of a Gaussian temperature profile | analytic diffusion | 0.1 % of the peak |
| CEM43 at constant temperature | 43 °C for 60 min and 44 °C for 30 min both give 60 min; 42 °C for 60 min gives 15 min | exact |
| Electrical impedance of an unloaded plate, Mason model | closed form | $10^{-6}$ |
| Energy balance of a loaded stack | electrical power = acoustic power | $10^{-9}$ |
| Matching layer | Desilets et al. (1978) formula; wider band than without | |
| Pulse duration, PII, ISPPA, MI | IEC 62359 and FDA definitions on test signals | exact |
| Water sound speed and density | Marczak (1997), Kell (1975) | published values |

To run the tests yourself, from a copy of the repository:

```
git clone https://github.com/sonosk/sonosk.git
cd sonosk
pip install -e ".[test]"
pytest
```

## Comparison with other simulators

Comparisons with established tools on shared test cases are planned as a
benchmark set (see the [Roadmap](roadmap.md)). They will use the
published plane-wave imaging challenge data (PICMUS) for beamforming and
the transcranial benchmark problems of Aubry et al. (2022) for the
full-wave solver.

## References

Aubry J.-F. et al. Benchmark problems for transcranial ultrasound
simulation: intercomparison of compressional wave models. J. Acoust. Soc.
Am. 152(2):1003-1019, 2022.

Burckhardt C.B. Speckle in ultrasound B-mode scans. IEEE Trans. Sonics
Ultrason. 25(1):1-6, 1978.

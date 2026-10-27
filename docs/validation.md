# Validation

Every solver in SonoSK is compared with a closed-form result or with an
independent method. These comparisons are part of the test suite in the
`tests` folder (131 tests in version 0.2.0) and run on every change of the
code, on Linux and Windows.

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

## Added in 0.2.0

| Quantity | Reference | Agreement in the tests |
|----------|-----------|------------------------|
| Full-wave: arrival time and amplitude of a 1-D pulse | $x/c$, lossless transport | within 2 time steps, 0.2 % |
| Full-wave: power-law absorption, $y = 1.5$, 1 to 5 MHz | $\alpha_0 f^y$ | 2.5 % |
| Full-wave: phase speed | Kramers-Kronig relation (Treeby and Cox 2010) | 0.15 m/s |
| Full-wave: speed at the reference frequency `f_ref` | the tissue value | 1 m/s |
| Full-wave: reflection at a fluid interface | $(Z_2-Z_1)/(Z_2+Z_1)$ | 0.5 % |
| Full-wave: harmonics 1-3 of a plane wave, $B/A = 5$ | Fubini solution | 0.002 of the source amplitude up to half the shock distance |
| Full-wave: 2-D point source | cylindrical spreading $1/\sqrt{r}$ | 1 % |
| Full-wave: single against double precision | | 0.1 % |
| Full-wave: point target in a beamformed image | true position | 0.2 mm lateral, 0.3 mm axial |
| Full-wave: velocity sensor on a plane wave | $u = p/(\rho c)$ | 1 % |
| Full-wave: directional elements in an empty medium | omnidirectional elements | residual below 10 % |
| Hybrid angular spectrum in water | Rayleigh integral on the axis of a bowl | 0.3 % of the peak |
| Hybrid angular spectrum through an interface | transmission $2Z_2/(Z_1+Z_2)$ times $e^{-\alpha d}$ | 1 % |
| Pulsed field | CW Rayleigh integral at the centre frequency; arrival time | 1 %; half a period |
| Ray model: transmit times in a uniform medium | plane, diverging and focused travel times | 2 ns |
| Ray model: delay through a layer | slowness integral | $10^{-4}$; receive delays of interpolated elements 5 ns |
| Fast imaging model with a uniform medium | homogeneous model | 3 % of the largest sample |
| Sound-speed-aware beamforming under a 1440 m/s layer | true target position | 0.05 mm |
| Centre frequency of an echo from 60 mm | downshift $-A\sigma_f^2/f_0$ of a Gaussian spectrum | 2 % |
| Speckle with frequency-dependent attenuation and TGC | 1.913 | 0.15 |
| Arrhenius damage at constant temperature | $A e^{-E_a/RT} t$ | $10^{-6}$ |
| Bioheat run continued from an earlier one | one uninterrupted run | exact |
| Perfusion shutdown | zero perfusion in the lesion, unchanged outside | exact |
| Shift of a heat source | centroid of a Gaussian source | 10 µm |
| Heated-tissue echo and attenuation | factors of Worthington et al. (2002) | exact |
| Spherical phased array | steered focus position | 0.2 mm |
| Tissue database | every value has a reference; uterine power laws through the published points; selected values against their papers | 2 % |
| Prostate and calcification | impedances 1.63 and 3.20 MRayl of Suomi et al. (2018) | 0.005 MRayl |
| IT'IS V5.0 attenuation | conversion from Np/(m MHz$^y$) | 0.2 % |
| Anatomical phantoms | layer order, lesion areas, prostate volume 20-30 mL, SMAS depth 4-5 mm | |

## Added after the review of the release candidate

| Quantity | Reference | Agreement in the tests |
|----------|-----------|------------------------|
| Full-wave: one exponent per tissue (`y="tissue"`), slab of y = 1.8 in a grid of median y = 1.2 | $\alpha_0 f^{1.8}$ of the slab, 1.5 to 4.5 MHz | 4 % (single exponent: off by more than 30 %) |
| Full-wave: argument checks, CFL warning, memory guard, stop on a non-finite field | | errors and warnings raised |
| Hybrid angular spectrum through a density gradient | $\sqrt{Z_2/Z_1}$, first order in the slice spacing | $3 \times 10^{-4}$ at 0.1 mm |
| Hybrid angular spectrum at a sharp step, any slice spacing | $2Z_2/(Z_1+Z_2)$ | $10^{-12}$ |
| Ray model: interface loss of one step | $-\tfrac12\ln(1-R^2)$ | $10^{-6}$ (line), $10^{-3}$ (polar map) |
| Interface reflectors across a step | $\tfrac12\ln(Z_2/Z_1)$ | 2 % |
| Arrhenius damage on a temperature ramp | independent integration of the rate | 1 % |
| Damage model per material | rate of each model | exact |
| Bioheat solver, Gaussian diffusion | analytic solution | second order in space and time |
| Temperature-dependent conduction; warning above 100 °C | lower peak; warning and note | |
| Dose rate at 2000 °C | finite, no overflow | |
| FDA Track 3 rules, six combinations, incomplete input, TI note | Section 5.2.8 of the guidance | exact |
| Pulse measures on several waveforms at once | the same measures row by row | exact |
| Tissue database: provenance kinds, no inheritance cycles, values inside their IT'IS range, CSV export | | |
| Grid: strict index and crop | out-of-grid points raise | |

The convergence studies and the comparison with k-Wave are on the
[Benchmarks](benchmarks.md) page.

To run the tests yourself, from a copy of the repository:

```
git clone https://github.com/sonosk/sonosk.git
cd sonosk
pip install -e ".[test]"
pytest
```

## Comparison with other simulators

The full-wave solver has been compared with k-Wave on a shared 2-D
heterogeneous case, linear and nonlinear (see [Benchmarks](benchmarks.md)).
Comparisons of the fast imaging model with Field II or FOCUS, and of the
beamformers with the published plane-wave imaging challenge data
(PICMUS), are planned for version 0.3 (see the [Roadmap](roadmap.md)). The
transcranial benchmark problems of Aubry et al. (2022) need high-contrast
bone and will follow the elastic model.

## References

Aubry J.-F. et al. Benchmark problems for transcranial ultrasound
simulation: intercomparison of compressional wave models. J. Acoust. Soc.
Am. 152(2):1003-1019, 2022.

Burckhardt C.B. Speckle in ultrasound B-mode scans. IEEE Trans. Sonics
Ultrason. 25(1):1-6, 1978.

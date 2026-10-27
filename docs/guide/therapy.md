# Therapy

`sonosk.thermal` follows a therapeutic exposure from the pressure field to
the lesion: heat source, temperature, thermal dose, Arrhenius damage,
treatment plans with several sonications, and the volumes treated and
spared.

## Field in tissue

For a source coupled through water into layered tissue, compute the field
on the skin plane with the Rayleigh integral and carry it through the
tissue with the hybrid angular spectrum method (see
[Pressure fields](fields.md)):

```python
from sonosk import field, thermal, transducer, water

cap = transducer.spherical_array(256, 3.3e-3, 135.9e-3, 140e-3)   # or transducer.bowl(...)
cap.translate((0, 0, target_depth - 140e-3))
plane = sk.Grid(grid.shape[:2] + (1,), grid.spacing, grid.origin)  # skin plane
p0 = field.rayleigh_grid(cap, plane, f, c_water)[..., 0]
p0 = field.scale_to_pressure(p0, rho_water, c_water, power=110.0, area=cap.total_area)
p = field.hybrid_angular_spectrum(p0, medium, f)                   # Pa, grid shape
```

## Heat source

For a CW field with pressure amplitude $p$, the absorbed power density is

$$
Q = 2\alpha I = \frac{\alpha |p|^2}{\rho c},
$$

with $\alpha$ the amplitude absorption coefficient in Np/m. When only the
attenuation is known it is used in place of the absorption, which
overestimates heating in strongly scattering tissue. With the harmonics of
a nonlinear full-wave run, `heat_source_harmonics` adds each one at its
own attenuation.

```python
q = thermal.heat_source(p, medium.attenuation(f), medium.rho, medium.c)
q = thermal.heat_source(p, medium.attenuation(f), medium.rho, medium.c, fraction=0.7)
q = thermal.heat_source_harmonics(result.cw, medium)
```

`fraction` is the part of the attenuation that is absorption. The default
1 treats all of it as absorption; no tissue in the database has a
measured absorption fraction yet, so a lower value is the user's choice
and should be stated with the results.

## Bioheat equation

The temperature follows the Pennes equation

$$
\rho c_p \frac{\partial T}{\partial t} = \nabla\cdot(k \nabla T)
+ w_b c_b (T_a - T) + Q,
$$

solved with an explicit finite-difference scheme on the grid of the
medium, in 1, 2 or 3 dimensions, with the outer boundary held at the
initial temperature.

```python
res = thermal.bioheat(medium, q, duration=28.0,
                      schedule=lambda t: 1.0 if t < 8.0 else 0.0,
                      arrhenius=thermal.HENRIQUES, shutdown=("dose", 240.0))
res.T_max, res.dose, res.damage, res.perfusion, res.T_final
res2 = thermal.bioheat(medium, q2, 10.0, initial=res)      # continue from res
```

`k_coeff` and `cp_coeff` make the conductivity and the specific heat
change linearly with temperature, $k(T) = k\,(1 + k_\mathrm{coeff}(T-37))$.
The defaults keep them constant; no default coefficients are given
because the published values differ between tissues and studies
(Rossmann and Haemmerich 2014). Boiling and cavitation are not modelled:
a run that passes 100 °C gives a warning and a line in `res.notes`, and its
results are not physical where that happens.

`schedule(t)` gives the power fraction at time t. `T_surface` holds the
first depth plane at a fixed temperature, for a cooled coupling balloon or
skin cooling. `initial` continues from
the end of an earlier run, with its temperature, dose, damage and
perfusion. `medium.crop(low, high)` cuts the thermal problem to the
region around the target.

## Thermal dose and Arrhenius damage

$$
\mathrm{CEM}_{43} = \sum \Delta t\, R^{43 - T},\qquad
\Omega = \int A\, e^{-E_a / (R_\mathrm{gas} T)}\,dt .
$$

`res.lesion(240.0)` gives the points with a dose of at least 240 min
(Sapareto and Dewey 1984); `res.damaged(1.0)` the points with
$\Omega \ge 1$. `thermal.HENRIQUES` is the skin model of Moritz and
Henriques (1947) with a second branch above 55 °C (coefficients as
tabulated by Ma et al. 2021); `thermal.Arrhenius(A, Ea)`
takes any other coefficients. It is the only damage model shipped with
the library, and for tissues other than skin its result is a damage index.
Tissue-specific models can be given per material:

```python
res = thermal.bioheat(medium, q, 10.0,
                      arrhenius={"dermis": thermal.HENRIQUES, "*": my_prostate_model})
```

The key `"*"` covers all other materials; without it they get no damage.

## Perfusion shutdown

Blood flow stops in coagulated tissue. With `shutdown=("dose", 240.0)` or
`("damage", 1.0)` the perfusion of a point drops to zero once it passes the
threshold, so the lesion grows faster in well-perfused tissue. In human
tumours, perfusion is not seen to fall below about 44 °C (Rossmann and
Haemmerich 2014), which a dose threshold of 240 min respects.

## Treatment plans

```python
shots = [thermal.Sonication(thermal.shift(q, grid, pt - focus), 4.0, cooling=6.0)
         for pt in thermal.raster(target_mask, grid, spacing=1.7e-3)]
plan = thermal.treat(medium, shots, shutdown=("dose", 240.0))
plan.log                     # (label, peak temperature, lesion volume) per sonication
plan.result.lesion()
```

`shift` moves a focal heat source to another target point, which is
accurate for small steering; for large steering compute a new field with
`transducer.focus`. `raster` covers a target with focal points, `circles`
builds the concentric trajectory of volumetric sonication (Köhler et al.
2009) and `trajectory_source` averages the heat source over it.

```{image} ../figures/fibroid_hifu.png
:alt: Pressure through the abdominal wall, peak temperature with the lesion over a fibroid, and the treatment report
```

```{image} ../figures/prostate_hifu.png
:alt: Peak temperature and lesion of a transrectal prostate treatment, report, and B-mode images before and after
```

```{image} ../figures/aesthetic_hifu.png
:alt: Thermal coagulation points of 10, 7 and 4 MHz transducers in facial skin
```

## Treatment report and imaging the result

```python
rep = thermal.treatment_report(plan.lesion(), medium, target="tumour")
after = thermal.coagulate(medium_2d, T_max_2d)    # for a B-mode image after treatment
```

The report gives the target volume, the ablated and residual target, the
damaged volume outside the target and the damaged volume of every
material. `coagulate` gives heated tissue the higher backscatter and
attenuation measured in human prostate heated to 60 and 65 °C
(Worthington et al. 2002), so the treated region shows on a simulated
image. Those samples were held at temperature for 30 min and measured at
5 MHz, while a HIFU lesion forms in seconds at higher temperatures, so
the factors are an approximation for HIFU.

## How much the inputs matter

```{image} ../figures/therapy_sensitivity.png
:alt: Ablated fraction, healthy volume damaged and peak temperature of the fibroid plan when one tissue value or model choice is changed at a time
```

`examples/therapy_sensitivity.py` repeats the fibroid plan with one input
changed at a time: fibroid perfusion halved and tripled, 70 % of the
attenuation taken as absorption, density at the ends of the IT'IS range
for uterus, and no perfusion shutdown. `tissues.sample()` draws whole
materials within the published spread for Monte Carlo runs.

## Limits

Thermal properties are constant unless coefficients are given. Cavitation
and boiling, which matter at the highest HIFU pressures, are not modelled.
The energy efficiency of the fibroid example is of the same order as
clinical reports (Kim et al. 2012), but the plan shares heat between
cells and the phantom fibroid has a low perfusion, so the comparison is a
plausibility check, not a validation.

## References

Moritz A.R., Henriques F.C. Studies of thermal injury II. The relative
importance of time and surface temperature in the causation of cutaneous
burns. Am. J. Pathol. 23(5):695-720, 1947.

Köhler M.O., Mougenot C., Quesson B., et al. Volumetric HIFU ablation under
3D guidance of rapid MRI thermometry. Med. Phys. 36(8):3521-3535, 2009.

Kim Y.S., Keserci B., Partanen A., Rhim H., Lim H.K., Park M.J., Köhler M.O.
Volumetric MR-HIFU ablation of uterine fibroids: role of treatment cell size
in the improvement of energy efficiency. Eur. J. Radiol. 81(11):3652-3659,
2012.

Ma J., Yang X., Sun Y., Yang J. Theoretical analysis on thermal treatment
of skin with repetitive pulses. Sci. Rep. 11:9958, 2021.

Pennes H.H. Analysis of tissue and arterial blood temperatures in the
resting human forearm. J. Appl. Physiol. 1(2):93-122, 1948.

Rossmann C., Haemmerich D. Review of temperature dependence of thermal
properties, dielectric properties, and perfusion of biological tissues at
hyperthermic and ablation temperatures. Crit. Rev. Biomed. Eng.
42(6):467-492, 2014.

Sapareto S.A., Dewey W.C. Thermal dose determination in cancer therapy.
Int. J. Radiat. Oncol. Biol. Phys. 10(6):787-800, 1984.

Worthington A.E., Trachtenberg J., Sherar M.D. Ultrasound properties of
human prostate tissue during heating. Ultrasound Med. Biol.
28(10):1311-1318, 2002.

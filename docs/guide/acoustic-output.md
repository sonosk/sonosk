# Acoustic output

`sonosk.safety` computes the acoustic output quantities that are reported
for diagnostic ultrasound systems and compares them with the limits in the
FDA guidance (February 2023). The definitions follow IEC 62359 and the
same guidance. Inputs are values measured, or simulated, in water on the
beam axis.

## Derating

Tissue attenuation is modelled by a fixed 0.3 dB/(cm MHz). A derated
pressure, marked with the subscript ".3", is

$$
p_{.3}(z) = p(z)\,10^{-0.3 f z / 20}
$$

and a derated intensity or pulse intensity integral is

$$
I_{.3}(z) = I(z)\,10^{-0.3 f z / 10}
$$

with $f$ in MHz and $z$ in cm.

```python
from sonosk import safety

safety.derate(2.0, f_mhz=5.0, z_cm=3.0)
safety.derate_intensity(1.0, f_mhz=5.0, z_cm=3.0)
```

## Pulse quantities

For a pressure waveform $p(t)$ in a medium of density $\rho$ and sound
speed $c$:

$$
\mathrm{PII} = \frac{1}{\rho c}\int p^2(t)\,dt
$$

The pulse duration PD is 1.25 times the time between the points where the
running integral of $p^2$ reaches 10 % and 90 % of its final value. Then

$$
I_\mathrm{SPPA} = \frac{\mathrm{PII}}{\mathrm{PD}}, \qquad
I_\mathrm{SPTA} = \mathrm{PII}\cdot\mathrm{PRF}
$$

where the second form holds for an unscanned beam with pulse repetition
frequency PRF.

```python
pii = safety.pulse_intensity_integral(p, dt, rho=1000.0, c=1500.0)   # J/m^2
pd = safety.pulse_duration(p, dt)                                    # s
i_ppa = safety.isppa(p, dt)                                          # W/m^2
i_pta = safety.ispta(pii, prf=4000.0)                                # W/m^2
safety.w_m2_to_mw_cm2(i_pta)
```

## Mechanical index

$$
\mathrm{MI} = \frac{p_{r.3}(z_{sp})}{\sqrt{f_c}}
$$

with $p_{r.3}$ the derated peak rarefactional pressure in MPa and $f_c$ the
centre frequency in MHz. The depth $z_{sp}$ is the point on the beam axis
where the derated pulse intensity integral is largest. When the PII along
the axis is not given, `mechanical_index` uses $p_r^2$ in its place, which
assumes the pulse shape does not change with depth.

```python
mi, z_sp = safety.mechanical_index(z_cm, pr_mpa, f_mhz=3.0, pii=pii_axis)
```

## Limits

`check_limits` uses the Track 3 global maximum values:

| Application   | ISPTA.3 [mW/cm²] | MI   | ISPPA.3 [W/cm²] | TI  |
|---------------|------------------|------|-----------------|-----|
| general       | 720              | 1.9  | 190             |     |
| ophthalmic    | 50               | 0.23 |                 | 1.0 |

For general use the guidance asks for MI ≤ 1.9 or ISPPA.3 ≤ 190 W/cm², so
the check passes if either one is within its limit.

```python
r = safety.check_limits(mi=1.2, ispta3=430.0)
r["ok"]                       # True
r["mi"]                       # (1.2, 1.9, True)

safety.check_limits(mi=0.2, ispta3=40.0, ti=0.8, application="ophthalmic")
```

The thermal indices (TIS, TIB, TIC) are defined in IEC 62359 and are not in
this release.

## References

IEC 62359:2010+A1:2017. Ultrasonics. Field characterization. Test methods
for the determination of thermal and mechanical indices related to
medical diagnostic ultrasonic fields.

FDA. Marketing Clearance of Diagnostic Ultrasound Systems and Transducers.
Guidance for Industry and Food and Drug Administration Staff, February
2023. Docket FDA-2017-D-5372.

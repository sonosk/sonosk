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

## From a simulated transmit

`field.pulsed` gives the pressure waveform of a pulsed transmit at points
on the beam axis, in water, which is what the measures above expect:

```python
z = np.linspace(5e-3, 60e-3, 111)
array.focus((0, 0, 30e-3), c=1480.0)
t, p = field.pulsed(array, np.stack([0 * z, 0 * z, z], 1), imaging.Pulse(3e6, 0.6),
                    c=1480.0)
p = field.scale_to_pressure(p, 1000.0, 1480.0, u0=0.2)          # Pa
pr = -p.min(axis=1) / 1e6                                       # MPa
pii = np.array([safety.pulse_intensity_integral(s, t[1] - t[0]) for s in p])
mi, z_sp = safety.mechanical_index(z * 100, pr, f_mhz=3.0, pii=pii)
```

## Limits

`check_limits` is a screening aid for research, not a regulatory
determination: the guidance also sets measurement and reporting
conditions that a simulation does not cover. It uses the Track 3 global
maximum values of Section 5.2.8 of the FDA guidance (February 2023):

| Application   | ISPTA.3 [mW/cm²] | MI   | ISPPA.3 [W/cm²] | TI  |
|---------------|------------------|------|-----------------|-----|
| general       | 720              | 1.9  | 190             |     |
| ophthalmic    | 50               | 0.23 |                 | 1.0 |

For general use the guidance asks for ISPTA.3 within its limit and
either MI ≤ 1.9 or ISPPA.3 ≤ 190 W/cm², so either one is enough. It sets
no TI limit for general use but asks for an explanation of any TI above 6;
the result then carries a note. For ophthalmic use all three limits apply.
`r["complete"]` says whether all the values the rules need were given,
and `r["ok"]` is true only when they were and all are within the limits.
Negative or non-finite inputs raise an error.

```python
r = safety.check_limits(mi=1.2, ispta3=430.0)
r["ok"], r["complete"]        # True, True
r["mi"]                       # (1.2, 1.9, True)
safety.check_limits(mi=0.5)["ok"]          # False: ISPTA.3 missing
safety.check_limits(mi=0.2, ispta3=40.0, ti=0.8, application="ophthalmic")
```

The pulse measures take the time along the last axis, so several
waveforms (one per row) are measured in one call.

The thermal indices (TIS, TIB, TIC) are defined in IEC 62359 and are not in
this release; they will be added once the formulas have been checked
against the standard itself.

## References

IEC 62359:2010+A1:2017. Ultrasonics. Field characterization. Test methods
for the determination of thermal and mechanical indices related to
medical diagnostic ultrasonic fields.

FDA. Marketing Clearance of Diagnostic Ultrasound Systems and Transducers.
Guidance for Industry and Food and Drug Administration Staff, February
2023. Docket FDA-2017-D-5372.

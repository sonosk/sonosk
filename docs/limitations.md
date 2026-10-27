# Scope and limits

SonoSK is research and teaching software. Its results come from models
with the assumptions listed on this page. They are not validated for
clinical decisions, for planning the treatment of a patient or for
regulatory claims about a device.

## What SonoSK is not

- Not an elastic solver: soft tissue and bone are fluids, so shear waves,
  mode conversion in bone and elastography are outside version 0.2.
- Not a GPU code: everything runs on the CPU with NumPy, which limits
  full-wave runs in 3-D to small grids.
- Not a real-time scanner simulator, and the fast imaging model is 2-D in
  heterogeneous tissue.
- Not a tool for regulatory acoustic output measurements: the output
  measures are computed from waveforms as defined in the standard, and the
  limit check is a screening aid.

## Assumptions that most affect results

**Fast imaging model.** Echoes follow straight rays: phase aberration from
speed differences is included, ray bending is not. Interface echoes come
from `phantom.interface_scatterers`, a first-order (Born) model whose
level relative to the speckle is a modelling choice, and the transmission
loss at interfaces is applied with `interfaces=True`. Diffuse scatterer
amplitudes follow the echo level of each tissue, which is a modelling
choice for every tissue except the reference (breast fat). Plane-wave
compounding with few angles leaves clutter in anechoic regions: in the
breast example the cyst contrast is about -12 dB with 7 angles and about
-17 dB with 21 (-15.4 dB in the final example, which also has interface
echoes). Sound-speed-aware beamforming in the examples uses the
true speed map, so it shows the best case.

**Full-wave solver.** By default one power-law exponent is used on the
grid and each tissue keeps its own attenuation only at `f_ref`; on a grid
with y = 1.6 the attenuation of fat (y = 1) is 34 % too low at half of
`f_ref` and 52 % too high at twice `f_ref` (see `fullwave.exponent_error` and the
[Benchmarks](benchmarks.md) page). `y="tissue"` removes this error at
extra cost. Sharp interfaces are staircased on the grid; for a plane
wave on a sharp cylinder the error of the transmitted pressure trace is
8 % at 4 points per wavelength and 3 % at 8, and of the weaker reflected
trace 19 % and 4 %. The statement that 2 to 3 points per
wavelength are enough holds for smooth media and linear propagation, not
for scattering by grid-scale structure or for shocks. In 2-D every
inclusion is an infinite cylinder, which fills cysts with reverberation
in full-wave images; the 2-D full-wave breast example is an aberration
study, not a realistic B-mode image. On a shared heterogeneous case the
solver agrees with k-Wave to 0.2 to 2.6 % in the linear regime and to 4 to
6.5 % in a strongly nonlinear run, where the cause of the difference is
still open.

**Hybrid angular spectrum.** One-way propagation: reflections, scattering
and nonlinearity are neglected. Through a smooth impedance gradient the
amplitude converges, as the slices get thinner, to the energy-conserving
value sqrt(Z2/Z1); at a sharp step it is the transmission coefficient
2 Z2/(Z1 + Z2).

**Thermal model.** The heat source uses the attenuation as absorption
unless `fraction` is given, which overestimates heating where scattering
is large. Thermal properties are constant unless `k_coeff` and
`cp_coeff` are given (no default values are shipped). Perfusion shutdown
is immediate and permanent. Boiling and cavitation are not modelled: runs
that pass 100 C give a warning and a note in the result, and their
temperatures, doses and damage are not physical there. The only damage
model shipped is the Henriques-Moritz skin model; for other tissues its
output is a damage index unless tissue-specific Arrhenius coefficients are
passed (`arrhenius={material: Arrhenius(...)}`). The change of echo and
attenuation of heated tissue uses factors measured after 30 min of
heating, applied by peak temperature.

**Tissue values.** Every value has a source, but the sources differ in
kind: direct measurements, power-law fits, database averages,
values used in simulation studies, generic soft-tissue values, values
inherited from a similar tissue and modelling choices.
`tissues.provenance()` gives the kind of every value and
`tissues.uncertainty()` the spread where one is known. Use
`tissues.sample()` to repeat a run over the spread of the values (see the
sensitivity example) before reading a single number as a prediction.

## Checks that exist and checks still missing

The [Validation](validation.md) page lists the closed-form and
self-consistency tests, and the [Benchmarks](benchmarks.md) page the
convergence studies and the comparison with k-Wave on a shared case. Not
done yet: comparisons with measured fields and temperatures, with Field II
or FOCUS for the fast imaging model, and with public imaging data such as
PICMUS.

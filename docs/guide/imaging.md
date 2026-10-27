# Imaging

`sonosk.imaging` simulates the echo signals of an array and forms B-mode
images from them. It is the fast imaging model: linear, with echoes
delayed and attenuated along straight paths. For refraction and multiple
scattering use the [full-wave solver](fullwave.md).

## The forward model

The model is linear (first-order Born scattering):

- the transmit wave comes from a virtual source: a plane wave with a
  steering angle, a diverging wave from a point behind the array, or a
  wave focused on a point in front of it;
- each receive element is a point receiver with the directivity of a
  rectangular element in a soft baffle,
  $\mathrm{sinc}(w \sin\theta / \lambda)\cos\theta$;
- spreading is $1/R$ on the receive path, and from the virtual source for
  a diverging wave;
- in a homogeneous medium the echo of each scatterer is delayed by its
  path over $c$ and attenuated with $\alpha_0 f^y$ along both paths;
- in a heterogeneous medium (`medium=`) the delay is the integral of the
  slowness $1/c$ along the straight transmit and receive paths and the
  attenuation is integrated along them too (`sonosk.paths`). Layers of
  fat delay the echoes (phase aberration), an attenuating lesion casts a
  shadow and a cyst lets more energy through (posterior enhancement);
- with `interfaces=True` every impedance step crossed on the way in and
  out lowers the echo by $1 - R^2$, with $R$ the pressure reflection
  coefficient of the step. The echoes of the interfaces themselves come
  from `phantom.interface_scatterers`: dense point reflectors whose
  amplitudes add up to $\tfrac12\,\Delta\ln Z \approx R$ across a step
  (the weak-contrast limit), so a boundary gives a coherent, specular echo.
  Their level relative to the speckle is a modelling choice (see
  [Media and phantoms](phantoms.md));
- frequency-dependent attenuation filters each echo,
  $P(f)\,e^{-A (f/f_0)^y}$, which lowers its centre frequency with depth.
  Echoes are grouped by their total attenuation $A$ and each group is
  convolved with its own filtered pulse.

The echoes are placed on a time axis four times finer than the output
rate, convolved with the pulse and decimated. The signals are kept in
analytic (complex) form, so the envelope of an image is its modulus.

## Pulse

```python
from sonosk import imaging

pulse = imaging.Pulse(f0=5e6, bandwidth=0.6)     # Gaussian, 60 % -6 dB bandwidth
pulse = imaging.Pulse.from_stack(stack)          # two-way pulse of a designed element
```

`Pulse.from_stack` takes a `sonosk.stack.Stack` (see
[Probe design](probe-design.md)), drives it with a one-cycle burst at its
half-wave frequency and applies its transmit response twice, so a probe
design goes straight into an image.

## Transmits

```python
tx = imaging.Transmit.plane(angle=0.0)
tx = imaging.Transmit.diverging(source=(0, 0, -10e-3))
tx = imaging.Transmit.focused(point=(0, 0, 30e-3))
angles = imaging.plane_wave_angles(9, span=np.deg2rad(24))
```

A plane wave insonifies the strip in front of the aperture. A diverging
wave from the centre of curvature of a convex array, `(0, 0, -radius)`,
insonifies its sector.

## Simulation and beamforming

```python
rf, fs = imaging.simulate(probe, tx, pos, amp, pulse, c=1540.0, alpha0=0.5)
rf, fs = imaging.simulate(probe, tx, pos, amp, pulse, c=1540.0, medium=medium)
rf = imaging.add_noise(rf, snr_db=90)
img = imaging.beamform(rf, fs, probe, tx, x, z, c=1540.0, f_number=1.5)
```

With a medium, `c` is the speed the scanner assumes when it fires the
elements; propagation follows the medium. `beamform` is delay-and-sum on
the grid `x` by `z`, with a receive aperture set by `f_number` around each
element's normal and a Hann window across it. For a strongly curved array
a smaller f-number (0.7) lets more of the arc contribute. `compound` adds
the images of several transmits:

```python
transmits = [imaging.Transmit.plane(a) for a in angles]
rfs = [imaging.simulate(probe, t, pos, amp, pulse, medium=medium)[0] for t in transmits]
img = imaging.compound(rfs, fs, probe, transmits, x, z)
img = imaging.tgc(img, z, alpha0=0.75, f0=pulse.f0)      # time-gain compensation
db = imaging.log_compress(img, dynamic_range=60)
```

## Sound-speed-aware beamforming

Passing the medium to `beamform` or `compound` computes the delays through
its speed map instead of with a constant speed. With the true map, as in
the examples, this is an oracle: the best result that sound-speed
correction can give, not what an estimated map would give. In the breast
example the targets move back to their true places and the lateral
resolution under a fat layer improves by up to a factor of two.

```python
pos, amp = phantom.scatterers(medium, density=50, rng=1)
pos_i, amp_i = phantom.interface_scatterers(medium)          # specular echoes
pos, amp = np.concatenate([pos, pos_i]), np.concatenate([amp, amp_i])
rf, fs = imaging.simulate(probe, tx, pos, amp, pulse, medium=medium, interfaces=True)
```

The contrast of an anechoic region depends on the sequence. In the breast
example the cyst contrast was about -12 dB with 7 plane waves; it did not
change without receiver noise, improved by 1 dB when all echo levels were
made equal, and reached about -17 dB with 21 plane waves. The cyst is
filled by the off-axis clutter of plane-wave compounding, so the example
uses 21 angles; with the interface echoes added, the example measures
-15.4 dB (gCNR 0.70) for the cyst, -9.8 dB for the fibroadenoma and
-11.0 dB for the carcinoma.

```{image} ../figures/breast_bmode.png
:alt: Breast phantom, sound speed map, and B-mode images beamformed at 1540 m/s and through the speed map
```

## Sector images

```python
mask = imaging.sector_mask(x, z, apex=(0, -radius), half_angle=np.deg2rad(30))
img = np.where(mask, img, np.nan)                  # drawn black outside
out = imaging.scan_convert(polar_img, r, theta, x, z, apex=(0, 0))
```

```{image} ../figures/uterus_transabdominal.png
:alt: Uterus phantom and transabdominal sector image with the bladder window and a fibroid
```

## Run time

The simulation costs about elements times scatterers operations per
transmit, plus the ray integrals when a medium is given (polar maps for
every fourth element, interpolated in between). A 128-element array with
80 000 scatterers in a heterogeneous 2-D medium takes about 15 s per
transmit on a laptop; beamforming a 450 by 475 pixel image takes a few
seconds per transmit.

## Limits

Paths are straight: refraction and multiple scattering need the full-wave
solver. The interface echoes use the depth derivative of the impedance
only, which suits beams near the depth axis. The ray model works on 2-D media. Volume imaging with matrix
arrays is planned for version 0.3.

## References

Montaldo G., Tanter M., Bercoff J., Benech N., Fink M. Coherent plane-wave
compounding for very high frame rate ultrasonography and transient
elastography. IEEE Trans. Ultrason. Ferroelectr. Freq. Control
56(3):489-506, 2009.

Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
Part I: theory & examples. Comput. Methods Programs Biomed. 218:106726,
2022.

Perrot V., Polichetti M., Varray F., Garcia D. So you think you can DAS? A
viewpoint on delay-and-sum beamforming. Ultrasonics 111:106309, 2021.

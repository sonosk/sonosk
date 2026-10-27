# Changelog

## 0.2.0 (2026-10-27)

Tissues, anatomical phantoms, full-wave and heterogeneous propagation,
and treatment planning.

New modules

- `tissues`: tissue property database. Every value of speed of sound,
  density, attenuation and its exponent, B/A, thermal conductivity,
  specific heat, perfusion and echo level carries its reference and a note
  on how it was measured; `material()`, `info()`, `cite()`, `table()` and
  `pending()`. Values from in vivo measurements published since 2009 where
  they exist (breast: Li et al. 2009, Nam et al. 2013, Nasief et al. 2015),
  prostate and calcification values of Suomi et al. (2018), and densities,
  thermal properties and perfusions from the IT'IS tissue property
  database V5.0 (2025). The few modelling choices are marked as such.
- `organs`: anatomical phantoms of the breast (skin, fat, Cooper's
  ligaments, fibroglandular tissue with fat lobules, retromammary fat,
  pectoralis), the prostate seen from the rectum (rectal wall, peripheral
  and transition zones, urethra, anterior stroma, calcifications), the
  uterus (abdominal wall, bladder, myometrium, endometrium, cervix;
  transabdominal or transvaginal) and the face (epidermis, dermis, fat,
  SMAS, muscle), in 2-D and 3-D, with layer sizes from published
  measurements; lesion classes `Cyst`, `Fibroadenoma`, `Carcinoma`,
  `Tumor` and `Calcification`.
- `fullwave`: k-space pseudospectral solver for heterogeneous, absorbing
  and nonlinear media in 1, 2 and 3 dimensions, with power-law absorption
  and dispersion through the fractional Laplacian, a perfectly matched
  layer, array sources and sensors, peak, RMS and harmonic maps, single
  precision, and `pulse_echo()` for full-wave imaging with directional
  elements (velocity source and sensor), a receive band-pass filter and
  optional subtraction of the direct wave. The absorption term
  is centred in time, which keeps the phase speed within 0.15 m/s of the
  Kramers-Kronig relation.
- `paths`: straight-ray integrals of slowness and attenuation on polar and
  plane-wave grids, used by the imaging model for heterogeneous media.

Imaging

- `simulate(..., medium=)`: echoes delayed and attenuated along the paths
  through a heterogeneous medium (aberration, shadowing, enhancement).
- Frequency-dependent attenuation shapes the spectrum of every echo
  (downshift of the centre frequency with depth).
- `beamform(..., medium=)` and `compound(..., medium=)`: sound-speed-aware
  beamforming through a known speed map; `receive_excess()` to reuse the
  receive delays.
- `Pulse.from_stack()` and `SampledPulse`: the pulse of a designed
  piezoelectric stack goes straight into the image.
- `add_noise`, `tgc`, `normalize_depth`, `sector_mask`, `scan_convert`;
  `log_compress` takes a reference level.

Fields and transducers

- `field.hybrid_angular_spectrum()`: propagation through heterogeneous
  tissue, slice by slice (Vyas and Christensen 2012), for 3-D therapy
  fields.
- `field.pulsed()`: transient fields synthesised over the pulse spectrum.
- `transducer.spherical_array()`: phased HIFU array of circular elements
  on a spherical cap; `Transducer.translate`, `rotate` and `subset`.

Therapy

- Arrhenius damage (`Arrhenius`, `HENRIQUES`), perfusion shutdown on dose
  or damage, continuation of a run (`initial=`), a surface held at a fixed
  temperature for a cooling balloon or skin cooling (`T_surface=`).
- Treatment plans: `Sonication`, `treat()`, `raster()`, `circles()`,
  `trajectory_source()` and `shift()`.
- `heat_source_harmonics()` for nonlinear fields; `coagulate()` for imaging
  after treatment with the echo and attenuation changes of heated tissue.

Other changes

- `Medium` computes its property maps on demand and accepts replaced maps;
  `crop()` and `slice2d()`. `Grid.crop()` and `Grid.index()`.
- `phantom`: `random_field`, `wavy_below`, `irregular`, `curve` and
  `fluctuations`.
- `plot`: `property_map`; `bmode` draws NaN pixels black and takes a
  reference level; `labels_map` shows up to 20 labels.
- New examples: breast B-mode with sound-speed-aware beamforming,
  transrectal prostate image, transabdominal uterus image, fibroid and
  prostate HIFU treatment plans, aesthetic micro-focused ultrasound,
  nonlinear HIFU, full-wave breast imaging and full-wave checks.
- New guide pages: tissue properties, anatomical phantoms, full-wave
  simulation; imaging, therapy, fields and transducers pages extended.

Changes after the review of the release candidate

- `fullwave.simulate`: `y="tissue"` gives each tissue its own power-law
  exponent; `exponent_error()` gives the error of a single exponent;
  argument checks, a warning above CFL 0.5, a memory estimate
  (`memory_estimate`, `max_memory`) and a stop with `FloatingPointError`
  when the field stops being finite.
- Fast imaging model: `phantom.interface_scatterers()` gives the specular
  echoes of impedance interfaces, and `imaging.simulate(...,
  interfaces=True)` the transmission loss 1 - R^2 of each step crossed.
- `thermal`: absorption fraction in `heat_source`; damage models per
  material; linear temperature dependence of conduction and heat capacity
  (`k_coeff`, `cp_coeff`); a warning and a note above 100 C; no overflow
  of the dose rate at very high temperatures.
- `safety.check_limits` follows Section 5.2.8 of the FDA guidance with a
  `complete` flag and notes, rejects invalid input and is described as a
  screening aid; the pulse measures work on several waveforms at once.
- `tissues`: `provenance()`, `uncertainty()` (spread of the IT'IS V5.0
  values), `sample()` for Monte Carlo runs and `to_csv()`; the table is
  published as CSV.
- `Grid.index` and `Grid.crop` take `strict=True`; transducer elements
  whose patch normals cancel are rejected.
- New module `benchmarks` (convergence studies with JSON results), the
  benchmark scripts and a comparison with k-Wave on a shared case; new
  `run_info()` for result metadata.
- Examples: 21 plane waves in the breast example (cyst contrast), interface
  echoes in the imaging examples, the 2-D full-wave breast example recast
  as an aberration study, the prostate HIFU power lowered so that the peak
  stays below 100 C, and a new sensitivity study of the fibroid plan.
- Documentation: Scope and limits page, Benchmarks page, research-only
  notice, provenance of tissue values, and qualified statements on grid
  resolution, oracle beamforming and clinical comparisons.
- Security: workflow actions pinned to commit hashes, Dependabot,
  least-privilege workflow permissions, a check of the built wheel before
  upload, and a security policy. `pytest` runs from a fresh checkout.

## 0.1.0 (2026-10-20)

First release.

- `Grid`: regular grids in 1, 2 and 3 dimensions, depth on the last axis.
- `transducer`: piston, focused bowl, annular array, linear and phased
  arrays with optional elevation lens, convex and endocavity arrays, matrix
  arrays and cylindrical line-focus sources; focusing, steering and
  apodization.
- `field`: CW pressure from the Rayleigh integral in an attenuating
  medium, angular spectrum propagation, intensity and pressure scaling.
- `medium`: materials with acoustic and thermal properties and power-law
  attenuation; property maps from label maps.
- `phantom`: layers, spheres, ellipsoids, cylinders, boxes, point
  scatterers with echo levels and wire targets.
- `imaging`: Gaussian pulse, plane, diverging and focused transmits,
  pulse-echo simulation, delay-and-sum beamforming, coherent compounding
  and log compression.
- `thermal`: heat source from the pressure field, Pennes bioheat solver in
  1, 2 and 3 dimensions with pulsed schedules, CEM43 dose, lesion maps and
  a treatment report (ablated and residual target, damaged healthy tissue).
- `stack`: Mason model of a thickness-mode piezoelectric element with
  backing, matching layers and load; impedance, transmit and pulse-echo
  responses, Desilets matching layer impedances.
- `safety`: derating, pulse intensity integral, pulse duration, ISPPA,
  ISPTA, mechanical index and the FDA Track 3 limits.
- `quality`: -6 dB width, contrast, CNR, gCNR, speckle SNR and SNR.
- `analytic`: on-axis pressure of a flat piston and of a focused bowl.
- `water`: sound speed, density and impedance of water.
- `plot`: B-mode images, field maps, temperature maps with lesion outlines,
  label maps and orthogonal slices.
- User guide, validation page and example gallery.

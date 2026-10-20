# Changelog

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

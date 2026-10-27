# Examples

Each figure in this guide comes from a script in the `examples` folder of
the repository. Run a script to see the figure and the printed numbers:

```
pip install "sonosk[plot]"
python examples/breast_bmode.py
```

The anatomical examples take their tissue values from `sonosk.tissues`,
where every number has a source. The older test-phantom examples further
down use generic numbers chosen to show a method. Run times are for a
laptop with two cores; the full-wave and 3-D therapy examples take a few
minutes.

## Breast B-mode with sound-speed-aware beamforming

A breast phantom with a cyst, a fibroadenoma and a spiculated carcinoma, imaged by a 7.5 MHz linear array with 21 plane waves. The echoes travel through fat, glandular tissue and lesions with their own speeds and attenuations, the interfaces give specular echoes and transmission losses. Beamforming at 1540 m/s blurs and shifts the lesions; beamforming through the true speed map (an oracle case) restores them.

```{image} figures/breast_bmode.png
:alt: Breast phantom labels, sound speed map, and B-mode images beamformed at 1540 m/s and through the speed map
```

```{literalinclude} ../examples/breast_bmode.py
:language: python
```

## Transrectal image of the prostate

A 6.5 MHz endocavity convex array images a transverse section with a peripheral zone tumour and two calcifications in the transition zone.

```{image} figures/prostate_trus.png
:alt: Prostate phantom labels and transrectal sector image
```

```{literalinclude} ../examples/prostate_trus.py
:language: python
```

## Transabdominal image of the uterus

A 3.5 MHz convex array images a uterus with a fibroid through the abdominal wall and a full bladder.

```{image} figures/uterus_transabdominal.png
:alt: Uterus phantom labels and transabdominal sector image
```

```{literalinclude} ../examples/uterus_transabdominal.py
:language: python
```

## Treatment plan for a uterine fibroid

A spherical cap with the aperture and focal length of a clinical MR-guided system heats a 24 mm fibroid through skin, fat, muscle and bladder in five volumetric cells, with perfusion shutdown. The report gives the ablated volume, the energy efficiency and the near-field temperature. The plan ablates about 79 % of the fibroid with 0.41 mm³ of lethal dose per joule, the same order as the 0.42 ± 0.25 mm³/J of 236 clinical sonications in ten women (Kim et al. 2012, Eur. J. Radiol. 81:3652). It is above their 0.29 ± 0.12 mm³/J for 8 mm cells, the size used here; in the plan the heat left by earlier cells helps the later ones, and the fibroid of the phantom has a low perfusion. The comparison is a check of the order of magnitude, not a validation; the sensitivity example below shows how much the result moves with the tissue values.

```{image} figures/fibroid_hifu.png
:alt: Pressure through the abdominal wall, peak temperature and lesion, and treatment report
```

```{literalinclude} ../examples/fibroid_hifu.py
:language: python
```

## Transrectal HIFU of a prostate tumour

A truncated 3 MHz bowl in the rectum treats a peripheral zone tumour point by point at a power that keeps the peak below 100 °C (boiling is not modelled), reports the rectal wall temperature and dose, and the plane is imaged before and after with the echo changes of heated prostate tissue.

```{image} figures/prostate_hifu.png
:alt: Peak temperature and lesion, report, and B-mode images before and after treatment
```

```{literalinclude} ../examples/prostate_hifu.py
:language: python
```

## Micro-focused ultrasound in facial skin

Lines of thermal coagulation points from 10, 7 and 4 MHz transducers aimed at 1.5, 3.0 and 5.0 mm, with Arrhenius skin damage and the SMAS as the target of the deepest one. The deepest one is aimed 0.5 mm below the 4.5 mm of clinical devices, because the fat above the SMAS heats faster and pulls the coagulation point up.

```{image} figures/aesthetic_hifu.png
:alt: Peak temperature and damaged region for three transducers in a face phantom
```

```{literalinclude} ../examples/aesthetic_hifu.py
:language: python
```

## Nonlinear focused beam

The k-space solver with and without nonlinearity: focal waveform, harmonics on the axis and the extra heating they cause in myometrium.

```{image} figures/nonlinear_hifu.png
:alt: Focal waveforms, harmonics on the axis and heat source with and without nonlinearity
```

```{literalinclude} ../examples/nonlinear_hifu.py
:language: python
```

## Aberration study with full-wave data (2-D)

Pulse-echo signals from the k-space solver, with speckle from small random variations of density, beamformed at 1540 m/s and through the true speed map. The script measures how far the pectoralis fascia is displaced in each image. In 2-D every inclusion is an infinite cylinder, so the cyst fills with reverberation: this is not a realistic lesion image.

```{image} figures/fullwave_breast.png
:alt: Sound speed map and 2-D full-wave images with the depth error of the fascia
```

```{literalinclude} ../examples/fullwave_breast.py
:language: python
```

## How much a treatment result depends on the tissue values

The fibroid plan run again with one input changed at a time: fibroid perfusion halved and tripled, 70 % of the attenuation taken as absorption, density at the ends of its published range, and no perfusion shutdown.

```{image} figures/therapy_sensitivity.png
:alt: Ablated fraction, damaged healthy volume and peak temperature of the fibroid plan for each changed input
```

```{literalinclude} ../examples/therapy_sensitivity.py
:language: python
```

## Checks of the full-wave solver

Power-law absorption and dispersion, harmonics against the Fubini solution and the reflection coefficient of an interface.

```{image} figures/fullwave_validation.png
:alt: Attenuation and phase speed against theory, harmonics against Fubini, reflection coefficient
```

```{literalinclude} ../examples/fullwave_validation.py
:language: python
```

## B-mode image of a test phantom

Background speckle, an anechoic cyst, a -10 dB and a +6 dB lesion and a row
of wire targets, imaged by a 128-element linear array at 5 MHz with nine
compounded plane waves.

```{image} figures/bmode_phantom.png
:alt: Phantom label map, B-mode image and quality measures
```

```{literalinclude} ../examples/bmode_phantom.py
:language: python
```

## HIFU ablation of a tumour

A 1 MHz bowl heats a tumour of 4 mm radius for 8 s. The single exposure
leaves most of the tumour untreated and a small lesion in front of and
behind it, which is what a treatment plan with several points has to fix.

```{image} figures/hifu_ablation.png
:alt: Peak temperature with lesion and tumour outlines, temperature at the focus, treatment report
```

```{literalinclude} ../examples/hifu_ablation.py
:language: python
```

## Field of a focused bowl

```{image} figures/bowl_field.png
:alt: Field of a focused bowl and its on-axis amplitude against O'Neil's solution
```

```{literalinclude} ../examples/bowl_field.py
:language: python
```

## Linear array beams

```{image} figures/array_beams.png
:alt: Focused, steered and apodized beams of a linear array
```

```{literalinclude} ../examples/array_beams.py
:language: python
```

## Line-focus source

```{image} figures/line_focus.png
:alt: Field of a cylindrically focused source in two planes
```

```{literalinclude} ../examples/line_focus.py
:language: python
```

## Element design with the Mason model

```{image} figures/transducer_stack.png
:alt: Electrical impedance and pulse-echo response with and without a matching layer
```

```{literalinclude} ../examples/transducer_stack.py
:language: python
```

## Mechanical index

```{image} figures/mechanical_index.png
:alt: On-axis pressure in water and derated, with the depth used for the MI
```

```{literalinclude} ../examples/mechanical_index.py
:language: python
```

## Water properties

```{image} figures/water_properties.png
:alt: Sound speed and density of water against temperature
```

```{literalinclude} ../examples/water_properties.py
:language: python
```

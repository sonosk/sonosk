# Roadmap

SonoSK aims at one Python library that covers medical ultrasound from the
design of a probe to the image it makes and to the thermal effect of a
therapeutic exposure, for every organ that ultrasound images or treats.
This page lists what is in each release and what is planned. Planned
versions can move as the work goes on.

## Transducers and fields

| Item | Status |
|------|--------|
| Piston, focused bowl, annular array, cylindrical line focus | 0.1.0 |
| Linear, phased, convex, endocavity and matrix arrays; elevation lens | 0.1.0 |
| Focusing, steering, apodization | 0.1.0 |
| Rayleigh integral and angular spectrum in a homogeneous medium | 0.1.0 |
| Pulsed (transient) fields of arrays | 0.2.0 |
| Phased HIFU arrays with many elements on a sphere | 0.2.0 |
| Full-wave k-space solver for heterogeneous media (refraction, reflection, aberration), power-law absorption | 0.2.0 |
| Hybrid angular spectrum method for 3-D therapy fields in layered tissue | 0.2.0 |
| Nonlinear propagation for HIFU (k-space solver with B/A, harmonic heating) | 0.2.0 |
| Axisymmetric full-wave solver for fast HIFU fields | 0.3 |
| Elastic model of bone and skull | 0.4 |
| One power-law exponent per tissue in the full-wave solver | 0.2.0 |
| Argument checks, memory estimate and stop on divergence in the full-wave solver | 0.2.0 |
| GPU computation through CuPy, with results equal to the NumPy path on test cases | 0.3 |

## Probe design

| Item | Status |
|------|--------|
| Mason model of a thickness-mode element, backing, matching layers, load | 0.1.0 |
| Electrical impedance, transmit and pulse-echo responses, bandwidth | 0.1.0 |
| Pulse of a designed element passed directly to the imaging simulation | 0.2.0 |
| Library of piezoelectric and passive materials with sources | 0.3 |
| Composite piezoelectric materials, lens with attenuation, electrical matching network | 0.3 |

## Tissues and phantoms

| Item | Status |
|------|--------|
| Materials, property maps, layers, inclusions, scatterers, wire targets | 0.1.0 |
| Tissue property database (acoustic and thermal) with a published source for every value | 0.2.0 |
| Provenance of every tissue value; spread of the IT'IS values and random draws for Monte Carlo runs | 0.2.0 |
| Measured backscatter coefficients in place of modelling-choice echo levels | 0.3 |
| Tissue database as a separate citable data record (CSV with provenance) | 0.3 |
| Breast phantom: skin, subcutaneous fat, Cooper's ligaments, glandular tissue, retromammary fat, muscle, cysts, fibroadenoma and carcinoma | 0.2.0 |
| Prostate phantom for transrectal imaging and therapy: rectal wall, zones of the gland, tumour, calcifications | 0.2.0 |
| Uterus phantom for transabdominal and transvaginal imaging and therapy: abdominal wall, bladder, myometrium, endometrium, cervix, fibroids | 0.2.0 |
| Skin and SMAS phantom for aesthetic HIFU | 0.2.0 |
| Phantoms from segmented CT and MRI label maps | 0.3 |
| Thyroid, liver, kidney, pancreas, heart, carotid and vessels, fetus, muscle and tendon, eye, neonatal brain, lung | 0.4 and later, one organ at a time |
| Tissue-mimicking phantom materials | 0.3 |

## Imaging

| Item | Status |
|------|--------|
| Plane-wave, diverging and focused transmits, delay-and-sum, compounding, B-mode in 2-D | 0.1.0 |
| Frequency-dependent attenuation of the pulse (spectral shift with depth) | 0.2.0 |
| Heterogeneous media in the fast model: aberration delays, shadowing, enhancement | 0.2.0 |
| Sound-speed-aware beamforming through a known speed map | 0.2.0 |
| Full-wave pulse-echo imaging | 0.2.0 |
| Specular interface echoes and transmission loss in the fast model | 0.2.0 |
| Refracted rays (eikonal) and heterogeneous paths in 3-D for the fast model | 0.3 |
| Sound-speed estimation from the echoes | 0.3 |
| Time-gain compensation, sector masks and scan conversion, receiver noise | 0.2.0 |
| 3-D volume imaging with matrix arrays | 0.3 |
| Doppler and flow in vessels | 0.4 |
| Harmonic imaging with the nonlinear full-wave solver | 0.3 |
| Elastography | later |

## Therapy

| Item | Status |
|------|--------|
| Heat source, Pennes bioheat equation in 1, 2 and 3 dimensions, pulsed schedules | 0.1.0 |
| CEM43 dose, lesion map, treatment report (ablated, residual, healthy tissue damaged) | 0.1.0 |
| Arrhenius damage model, perfusion shutdown in the lesion | 0.2.0 |
| Temperature-dependent conduction and heat capacity (user coefficients) | 0.2.0 |
| Temperature-dependent properties with published coefficients per tissue | 0.3 |
| Absorption fraction of the attenuation in the heat source | 0.2.0 |
| Damage models per material; tissue-specific Arrhenius sets with sources | 0.2.0 (mechanism), 0.3 (sets) |
| Change of echo and attenuation of heated tissue by thermal dose or damage | 0.3 |
| Boiling and cavitation model | later |
| Treatment plans with several sonication points, volumetric trajectories and cooling times | 0.2.0 |
| Change of echo and attenuation in heated tissue, imaging after treatment | 0.2.0 |
| Prostate and uterine fibroid ablation with the anatomical phantoms | 0.2.0 |
| Breast fibroadenoma ablation example | 0.3 |
| Aesthetic micro-focused ultrasound in the dermis, fat and SMAS with Arrhenius skin damage | 0.2.0 |

## Safety and quality

| Item | Status |
|------|--------|
| Derating, PII, pulse duration, ISPPA, ISPTA, mechanical index, FDA Track 3 limits | 0.1.0 |
| Thermal indices TIS, TIB, TIC (IEC 62359), after checking against the standard | 0.3 |
| Safety report for every imaging simulation | 0.3 |
| Resolution, contrast, CNR, gCNR, speckle SNR | 0.1.0 |
| Automatic measurement of resolution and contrast on standard phantoms | 0.3 |

## Design optimisation and machine learning

| Item | Status |
|------|--------|
| Optimiser for probe and sequence parameters with image quality and safety as goals | 0.3 |
| Optimiser for treatment plans: full target coverage with least damage to healthy tissue | 0.3 |
| Fast surrogate models trained on SonoSK simulations | 0.4 |
| Generator of simulated image sets with exact labels for training | 0.3 |

## Validation and documentation

| Item | Status |
|------|--------|
| Tests against closed-form results for every solver | 0.1.0 |
| User guide, example gallery, API reference | 0.1.0 |
| Tests of the full-wave solver against Fubini, Kramers-Kronig and reflection theory | 0.2.0 |
| Convergence studies (full-wave, absorbing layer, hybrid angular spectrum, bioheat) with JSON results | 0.2.0 |
| Comparison of the full-wave solver with k-Wave on a shared heterogeneous case | 0.2.0 |
| Comparison of the fast imaging model with Field II or FOCUS; PICMUS data | 0.3 |
| Comparison with measured fields and HIFU temperatures | 0.3 |
| Type hints on the public API and a static type check | 0.3 |
| Software paper | after the benchmarks |

## Display

| Item | Status |
|------|--------|
| B-mode, field, temperature, lesion and label maps; orthogonal slices of volumes | 0.1.0 |
| Export of phantoms, fields and lesions as 3-D models | later |

Requests and suggestions are welcome on the
[issue tracker](https://github.com/sonosk/sonosk/issues).

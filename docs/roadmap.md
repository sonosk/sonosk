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
| Pulsed (transient) fields of arrays | 0.2 |
| Phased HIFU arrays with many elements on a sphere | 0.2 |
| Full-wave solver for heterogeneous media (refraction, reflection, aberration by fat and bone) | 0.3 |
| Nonlinear propagation for HIFU (KZK and Westervelt equations) | 0.3 |
| Elastic model of bone and skull | 0.4 |
| GPU computation through CuPy | 0.3 |

## Probe design

| Item | Status |
|------|--------|
| Mason model of a thickness-mode element, backing, matching layers, load | 0.1.0 |
| Electrical impedance, transmit and pulse-echo responses, bandwidth | 0.1.0 |
| Pulse of a designed element passed directly to the imaging simulation | 0.2 |
| Library of piezoelectric and passive materials with sources | 0.2 |
| Composite piezoelectric materials, lens with attenuation, electrical matching network | 0.3 |

## Tissues and phantoms

| Item | Status |
|------|--------|
| Materials, property maps, layers, inclusions, scatterers, wire targets | 0.1.0 |
| Tissue property database (acoustic and thermal), a published source for every value, ranges as well as means | 0.2 |
| Breast phantom: skin, subcutaneous fat, glandular tissue, retromammary fat, muscle, cysts, fibroadenoma and carcinoma | 0.2 |
| Prostate phantom for transrectal imaging and therapy: rectal wall, zones of the gland, tumour, bladder | 0.2 |
| Uterus phantom for transabdominal and transvaginal imaging and therapy: abdominal wall, bladder, myometrium, endometrium, fibroids | 0.2 |
| Skin and SMAS phantom for aesthetic HIFU | 0.2 |
| Phantoms from segmented CT and MRI label maps | 0.3 |
| Thyroid, liver, kidney, pancreas, heart, carotid and vessels, fetus, muscle and tendon, eye, neonatal brain, lung | 0.4 and later, one organ at a time |
| Tissue-mimicking phantom materials | 0.3 |

## Imaging

| Item | Status |
|------|--------|
| Plane-wave, diverging and focused transmits, delay-and-sum, compounding, B-mode in 2-D | 0.1.0 |
| Frequency-dependent attenuation of the pulse (spectral shift with depth) | 0.2 |
| Time-gain compensation, scan conversion for convex and phased arrays | 0.2 |
| 3-D volume imaging with matrix arrays | 0.3 |
| Doppler and flow in vessels | 0.4 |
| Harmonic imaging | with the nonlinear solver, 0.3 or later |
| Elastography | later |

## Therapy

| Item | Status |
|------|--------|
| Heat source, Pennes bioheat equation in 1, 2 and 3 dimensions, pulsed schedules | 0.1.0 |
| CEM43 dose, lesion map, treatment report (ablated, residual, healthy tissue damaged) | 0.1.0 |
| Arrhenius damage model, perfusion shutdown in the lesion, temperature-dependent properties | 0.2 |
| Treatment plans with several sonication points and cooling times | 0.2 |
| Breast fibroadenoma, prostate and uterine fibroid ablation with the 0.2 phantoms | 0.2 |
| Aesthetic HIFU with line-focus cartridges at several depths | 0.2 |
| Cavitation and histotripsy | later |

## Safety and quality

| Item | Status |
|------|--------|
| Derating, PII, pulse duration, ISPPA, ISPTA, mechanical index, FDA Track 3 limits | 0.1.0 |
| Thermal indices TIS, TIB, TIC (IEC 62359) | 0.2 |
| Safety report for every imaging simulation | 0.2 |
| Resolution, contrast, CNR, gCNR, speckle SNR | 0.1.0 |
| Automatic measurement of resolution and contrast on standard phantoms | 0.2 |

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
| Benchmarks against established simulators and public data (PICMUS, transcranial benchmarks) | 0.2 and 0.3 |
| Software paper | after the benchmarks |

## Display

| Item | Status |
|------|--------|
| B-mode, field, temperature, lesion and label maps; orthogonal slices of volumes | 0.1.0 |
| Export of phantoms, fields and lesions as 3-D models | later |

Requests and suggestions are welcome on the
[issue tracker](https://github.com/sonosk/sonosk/issues).

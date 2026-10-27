# Anatomical phantoms

`sonosk.organs` builds phantoms of the breast, prostate, uterus and facial
skin on 1-, 2- or 3-D grids. Every region takes its properties from
[`sonosk.tissues`](tissues.md), layer thicknesses default to published
means, and the interfaces undulate with smooth random profiles, so every
seed gives a different but plausible anatomy.

```python
import sonosk as sk
from sonosk import organs

g = sk.Grid.centered((401, 401), 0.1e-3, depth_start=0.0)     # x, z in metres
ph = organs.breast(g, lesions=[organs.Cyst((-11e-3, 0, 13e-3), a=3e-3, b=2.4e-3),
                               organs.Fibroadenoma((0, 0, 14.5e-3)),
                               organs.Carcinoma((11e-3, 0, 16e-3), radius=3.2e-3)],
                   rng=4)
ph.medium          # Medium with labels and materials
ph.targets         # names of the lesions
ph.description     # sizes used
ph.sources()       # citations of all tissue values
```

The depth axis points away from the probe, which sits at z = 0. A 3-D grid
(x, y, z) gives a 3-D phantom with the same call.

## Lesions

| Class | Shape |
|-------|-------|
| `Cyst(center, a, b)` | ellipsoid, anechoic |
| `Fibroadenoma(center, a, b)` | oval, wider than tall, smooth lobulated margin |
| `Carcinoma(center, radius, roughness, spicules, taller)` | irregular, spiculated, taller than wide |
| `Tumor(center, radius, roughness, tissue=...)` | generic irregular mass of any tissue, for example a fibroid or a prostate tumour |
| `Calcification(center, radius)` | small bright inclusion with high attenuation |

The shapes follow the BI-RADS descriptors of benign and suspicious masses.
Each lesion can be given a `label`, its name in the medium.

## Breast

`organs.breast(grid, lesions, skin=1.45e-3, fat=5e-3, gland=15e-3,
retro_fat=2e-3, muscle=8e-3, density=0.6, ligaments=True, ribs=False)`

Skin, subcutaneous fat with Cooper's ligaments, fibroglandular tissue
mixed with fat lobules (`density` is the glandular fraction), retromammary
fat, pectoralis and chest wall. The default skin thickness is the mean of
1.44-1.45 mm measured by dedicated breast CT (Shi et al. 2013; Huang et al.
2008).

```{image} ../figures/breast_bmode.png
:alt: Breast phantom labels, sound speed map, and B-mode images beamformed at 1540 m/s and through the speed map
```

## Prostate

`organs.prostate(grid, tumors, calcifications, rectal_wall=3e-3, fat=1.5e-3,
size=(45e-3, 35e-3, 30e-3), tz_fraction=0.45)`

A transverse section seen from the rectum: rectal wall, periprostatic fat,
peripheral zone facing the probe, transition zone around the urethra,
anterior fibromuscular stroma and levator ani. The default gland volume is
about 25 mL, near the mean of 26.9 mL of patients treated by transrectal
HIFU (Pichardo et al. 2008); the peripheral zone is about 9 mm thick behind
the transition zone, inside the 1.6-15 mm range measured by MRI (Sellers et
al. 2022).

```{image} ../figures/prostate_trus.png
:alt: Prostate phantom labels and transrectal sector image with a peripheral zone tumour and calcifications
```

## Uterus

`organs.uterus(grid, fibroids, approach="transabdominal", skin=2e-3,
fat=15e-3, muscle=10e-3, bladder=40e-3, size=(80e-3, 50e-3, 45e-3),
tilt=20 deg, endometrium=8e-3)`

A sagittal section with skin, subcutaneous fat, rectus muscle, a full
bladder (wall 3 mm, Hakenberg et al. 2000) and an anteverted uterus with
myometrium, endometrium and cervix. The default uterus lies within the
mean sizes measured by ultrasound, 72.8 x 42.8 x 32.4 mm in nulliparous to
90.8 x 51.7 x 43.0 mm in multiparous women (Esmaelzadeh et al. 2004).
`approach="transvaginal"` removes the abdominal wall and the bladder.

```{image} ../figures/uterus_transabdominal.png
:alt: Uterus phantom labels and transabdominal sector image with a fibroid behind the full bladder
```

## Face

`organs.face(grid, epidermis=0.1e-3, dermis=1.3e-3, fat=2.6e-3,
smas=1.0e-3, muscle=8e-3, bone=False)`

Epidermis, dermis, superficial fat, SMAS and facial muscle for aesthetic
focused ultrasound. The full skin thickness of 1.4 mm lies in the
1.31-1.64 mm range measured over eight facial sites (Jeong et al. 2023),
and the SMAS starts at 4.0 mm, the depth reached by the 4.5 mm
micro-focused transducer (Chaves Bellote and Miot 2021).

## Random structure

`phantom.random_field`, `phantom.wavy_below`, `phantom.irregular` and
`phantom.curve` are the building blocks behind the organs and can be used
for new ones. `phantom.fluctuations(medium, level)` adds sub-resolution
variations of density and speed whose strength follows the echo level of
each tissue; they make the speckle of [full-wave images](fullwave.md).

## References

Chaves Bellote T.P., Miot H.A. Microfocused ultrasound with visualization
for face slimming: preliminary results in four women. Clin. Cosmet.
Investig. Dermatol. 14:1613-1619, 2021.

Esmaelzadeh S., Rezaei N., HajiAhmadi M. Normal uterine size in women of
reproductive age in northern Islamic Republic of Iran. East. Mediterr.
Health J. 10(3):437-441, 2004.

Hakenberg O.W., Linne C., Manseck A., Wirth M.P. Bladder wall thickness in
normal adults and men with mild lower urinary tract symptoms and benign
prostatic enlargement. Neurourol. Urodyn. 19(5):585-593, 2000.

Huang S.Y., Boone J.M., Yang K., Kwan A.L., Packard N.J. The effect of skin
thickness determined using breast CT on mammographic dosimetry. Med.
Phys. 35(4):1199-1206, 2008.

Jeong K.M., Seo J.Y., Kim A., et al. Ultrasonographic analysis of facial
skin thickness in relation to age, site, sex, and body mass index. Skin
Res. Technol. 29(8):e13426, 2023.

Pichardo S., Gelet A., Curiel L., Chesnais S., Chapelon J.Y. New
integrated imaging high intensity focused ultrasound probe for transrectal
prostate cancer treatment. Ultrasound Med. Biol. 34(7):1105-1116, 2008.

Sellers J., Wagstaff R., Helo N., de Riese W.T.W. Association between
prostate size and MRI determined quantitative prostate zonal
measurements. Res. Rep. Urol. 14:265-274, 2022.

Shi L., Vedantham S., Karellas A., O'Connell A.M. Skin thickness
measurements using high-resolution flat-panel cone-beam dedicated breast
CT. Med. Phys. 40(3):031913, 2013.

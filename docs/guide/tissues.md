# Tissue properties

`sonosk.tissues` is a database of acoustic and thermal properties of the
tissues used by the anatomical phantoms. Every value carries the paper it
comes from and a note on how it was measured or derived, so a simulation
can always be traced back to its inputs.

```python
from sonosk import tissues

fat = tissues.material("fat")                       # a Material
fat.c, fat.alpha0, fat.source
tissues.material("subcutaneous fat")                 # alias, labelled as given
tissues.material("fibroadenoma", echo=-6.0)          # one property changed
tissues.info("breast carcinoma")                     # (property, value, unit, ref, note)
tissues.cite("myometrium")                           # full citations
tissues.names("uterus"), tissues.groups()
```

## How the values were chosen

The rules, in order:

1. a measurement in human tissue, in vivo if possible, published since
   2009 when one exists (for example breast sound speeds from ultrasound
   tomography in 61 women, Li et al. 2009, and attenuation of breast fat in
   24 women, Nasief et al. 2015);
2. otherwise the most specific ex vivo human measurement (myometrium,
   fibroid and cervix, Keshavarzi et al. 2001 and Kiss et al. 2011; skin,
   Moran et al. 1995);
3. otherwise a value used and cited in a recent peer-reviewed simulation
   study of the same organ (prostate attenuation, Suomi et al. 2018), or a
   reviewed compilation (ICRU Report 61 through Culjat et al. 2010);
4. otherwise the average of the IT'IS tissue property database V5.0
   (2025), which pools many studies for each tissue; most densities,
   thermal properties and perfusions come from it;
5. where nothing fits, a modelling choice, marked `*` in the table and
   explained in its note (perfusion of fibroids and of blood in vessels,
   B/A of bone).

Where a paper gives two points of a power law, the exponent is fitted
through them; the notes say so. Echo levels are the least certain
numbers: they are relative to breast fat (the BI-RADS standard of
comparison) and, apart from fat, are modelling choices that reproduce the
usual B-mode appearance of each tissue.

## Provenance and spread

A reference per value does not make every value equally strong.
`tissues.provenance(name)` says how each value was obtained: measured,
fitted, database average, used in a simulation study, review
compilation, generic soft-tissue value, inherited from a similar tissue,
reference level, or modelling choice. For the values taken from IT'IS
V5.0, `tissues.uncertainty(name, prop)` returns the standard deviation,
the minimum, the maximum and the number of samples or studies (a
standard deviation of 0 with one study means a single source), and
`tissues.sample(name, rng)` draws a material with every such value
within its spread, for Monte Carlo runs.

```python
tissues.provenance("prostate carcinoma")   # {'c': 'inherited', ..., 'perfusion': 'used in a simulation study', ...}
tissues.uncertainty("muscle", "rho")       # (55.14, 1041, 1178, 14)
m = tissues.sample("muscle", rng=1)
tissues.to_csv("tissues.csv")              # the whole table with provenance and spread
```

The table, with the provenance and the spread of every value, is also
available as {download}`tissues.csv <../tissues.csv>`, generated from the
code.

## Properties and units

| Name | Meaning | Unit |
|------|---------|------|
| `c` | speed of sound | m/s |
| `rho` | density | kg/m³ |
| `alpha0`, `y` | attenuation $\alpha(f) = \alpha_0 f^y$ | dB/(cm MHz$^y$) |
| `ba` | nonlinearity parameter $B/A$ | |
| `k` | thermal conductivity | W/(m K) |
| `cp` | specific heat | J/(kg K) |
| `perfusion` | blood perfusion | mL/(min kg), converted to kg/(m³ s) |
| `echo` | backscatter level relative to breast fat | dB |

## The table

```{include} tissue-table.md
```

## Missing data you can help with

In vivo measurements would improve several entries: sound speed and
attenuation of the normal prostate, perfusion of fibroids, and the
thermal properties of breast lesions. They are welcome through the
[tissue data issue form](https://github.com/sonosk/sonosk/issues/new/choose).

# Transducers

`sonosk.transducer` builds sources and arrays. A `Transducer` is a list of
elements, and each element is split into small patches with a centre, a
normal pointing into the medium and an area. The field and imaging code
sum over these patches.

## Coordinates

The face of a probe is at or near z = 0 and the beam goes towards +z. For
arrays, x is the array direction (azimuth) and y the elevation. All sizes
are in metres.

## Patch size

The argument `dx` of every constructor is the largest patch size. For the
field close to the source, patches of a quarter wavelength or less are a
safe choice; one sixth of a wavelength reproduces the on-axis field of a
piston and a bowl to better than 0.5 % (see [Validation](../validation.md)).
For far-field quantities and for imaging, larger patches are enough.

## Single elements

```python
from sonosk import transducer

disk = transducer.piston(radius=5e-3, dx=0.1e-3)
bowl = transducer.bowl(radius=20e-3, roc=50e-3, dx=0.25e-3)       # focus at z = 50 mm
line = transducer.cylindrical(aperture=8e-3, length=12e-3, roc=6e-3)
```

`bowl` has its apex at the origin and its focus at (0, 0, roc).
`cylindrical` focuses to a line along y at depth roc, the geometry of
line-focused aesthetic HIFU transducers.

## Annular arrays

```python
rings = [(0, 4e-3), (4.2e-3, 7e-3), (7.2e-3, 9e-3)]
ann = transducer.annular(rings, roc=40e-3)
```

Each (inner, outer) pair is one element. With `roc=None` the rings are
flat.

## Linear, phased and convex arrays

```python
lin = transducer.linear_array(128, pitch=0.3e-3, height=5e-3, elev_focus=20e-3)
pha = transducer.phased_array(64, pitch=0.15e-3, height=10e-3)
cvx = transducer.convex_array(128, pitch=0.4e-3, radius=60e-3, height=10e-3)
tr = transducer.convex_array(96, pitch=0.2e-3, radius=10e-3)       # endocavity
```

The element width defaults to 0.9 of the pitch. `elev_focus` curves each
element in elevation, which is the effect of the acoustic lens; `None`
leaves it flat. A convex array has its centre of curvature at
(0, 0, -radius), so the middle of the array touches z = 0. A small radius
gives the transrectal and transvaginal probes used for the prostate and
the uterus.

## Matrix arrays

```python
mat = transducer.matrix_array(32, 32, pitch_x=0.3e-3)
```

## Phased HIFU arrays on a sphere

```python
cap = transducer.spherical_array(256, element_radius=3.3e-3, aperture=135.9e-3,
                                 roc=140e-3)
cap.focus((10e-3, 0, 140e-3), c=1500.0)      # electronic steering
```

Circular elements sit on a spherical cap in a sunflower pattern, which
avoids the grating lobes of a regular layout, each facing the centre of
curvature. The example numbers are the aperture and radius of curvature of
a clinical MR-guided system (Kothapalli et al. 2018).

## Moving and cutting a transducer

```python
src.translate((0, 0, -20e-3))                # move by (x, y, z)
src.rotate(np.deg2rad(15), axis="y")         # rotate about an axis
cut = src.subset(np.abs(src.pos[:, 0]) <= 15.5e-3)   # truncated bowl
```

`subset` keeps the patches where a condition holds, for example the
truncated bowl of a transrectal HIFU probe.

## Delays and apodization

Each element has a firing delay and a weight.

```python
lin.focus((0, 0, 30e-3), c=1540.0)       # focus at 30 mm
lin.steer(np.deg2rad(10), c=1540.0)      # plane wave at 10 degrees
lin.set_apodization(transducer.apodization(128, "hann"))
lin.reset()                              # no delays, unit weights
```

`centers` and `element_normals` give the element positions and
directions; `n_elements`, `n_patches` and `total_area` describe the
geometry.

## Your own geometry

Any source can be built directly from patch arrays:

```python
src = transducer.Transducer(pos, normal, area, elem, width=w, height=h, name="mine")
```

`pos` and `normal` are (N, 3) arrays, `area` and `elem` have length N.

## References

Kothapalli S.V.V.N., Partanen A., Zhu L., et al. A convenient, reliable,
and fast acoustic pressure field measurement method for magnetic
resonance-guided high-intensity focused ultrasound systems with phased
array transducers. J. Ther. Ultrasound 6:5, 2018.

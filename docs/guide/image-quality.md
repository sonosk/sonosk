# Image quality

`sonosk.quality` measures resolution and contrast on envelope data. All
functions take envelope amplitudes. Values from a log-compressed image
give wrong results.

## Resolution

`width(profile, x, level_db=-6)` returns the width of the main lobe of an
amplitude profile at `level_db` below its peak. The edges are found by
linear interpolation between samples. A lateral profile through a point
target gives the lateral resolution, and an axial profile gives the axial
resolution.

```python
from sonosk import quality

w6 = quality.width(profile, x)                  # -6 dB width
w20 = quality.width(profile, x, level_db=-20)   # -20 dB width
```

## Contrast

For envelope samples $s_i$ inside a region and $s_o$ in the background:

$$
C = 20\log_{10}\frac{\bar{s}_i}{\bar{s}_o}, \qquad
\mathrm{CNR} = \frac{|\bar{s}_i - \bar{s}_o|}{\sqrt{\sigma_i^2 + \sigma_o^2}}
$$

```python
quality.contrast_db(inside, background)
quality.cnr(inside, background)
```

## Generalized CNR

The gCNR of Rodriguez-Molares et al. (2020) is one minus the overlap of the
two sample distributions:

$$
\mathrm{gCNR} = 1 - \sum_k \min\left(h_i(k),\, h_o(k)\right)
$$

where $h_i$ and $h_o$ are the normalised histograms on common bins. It
lies between 0 (the regions cannot be told apart) and 1 (no overlap). In
principle it does not change under a monotonic transform of the image,
such as log compression, while the CNR does.

```python
quality.gcnr(inside, background, bins=256)
```

## Speckle

In fully developed speckle the envelope has a Rayleigh distribution and its
mean over standard deviation is

$$
\mathrm{SNR}_s = \sqrt{\frac{\pi}{4 - \pi}} \approx 1.91
$$

(Burckhardt 1978). A simulated background region far from this value
points to too few scatterers per resolution cell or to a coherent
component.

```python
quality.speckle_snr(background)
quality.RAYLEIGH_SNR          # 1.913
```

`snr_db(signal, noise)` is the ratio of RMS values in decibels.

## References

Burckhardt C.B. Speckle in ultrasound B-mode scans. IEEE Trans. Sonics
Ultrason. 25(1):1-6, 1978.

Rodriguez-Molares A. et al. The generalized contrast-to-noise ratio: a
formal definition for lesion detectability. IEEE Trans. Ultrason.
Ferroelectr. Freq. Control 67(4):745-759, 2020.

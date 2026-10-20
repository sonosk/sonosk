"""Image quality measures.

Inputs are envelope (amplitude) values, not log-compressed images.

References
----------
Burckhardt C.B. Speckle in ultrasound B-mode scans. IEEE Trans. Sonics
Ultrason. 25(1):1-6, 1978.

Rodriguez-Molares A. et al. The generalized contrast-to-noise ratio: a
formal definition for lesion detectability. IEEE Trans. Ultrason.
Ferroelectr. Freq. Control 67(4):745-759, 2020.
"""

import numpy as np

RAYLEIGH_SNR = np.sqrt(np.pi / (4.0 - np.pi))  # 1.913 for fully developed speckle


def width(profile, x, level_db=-6.0):
    """Width of the main lobe of profile at level_db below its peak.

    profile is an amplitude profile sampled at positions x. Edges are found
    by linear interpolation. Use level_db=-6 for the usual -6 dB
    resolution and -20 dB for the beam width at that level.
    """
    a = np.abs(np.asarray(profile, dtype=float))
    x = np.asarray(x, dtype=float)
    k = int(np.argmax(a))
    thr = a[k] * 10.0 ** (level_db / 20.0)

    i = k
    while i > 0 and a[i] > thr:
        i -= 1
    j = k
    while j < a.size - 1 and a[j] > thr:
        j += 1
    if a[i] > thr or a[j] > thr:
        raise ValueError("profile does not fall below the level on both sides")

    left = x[i] + (thr - a[i]) * (x[i + 1] - x[i]) / (a[i + 1] - a[i])
    right = x[j - 1] + (thr - a[j - 1]) * (x[j] - x[j - 1]) / (a[j] - a[j - 1])
    return float(right - left)


def contrast_db(inside, outside):
    """Contrast of a region against background [dB]."""
    return float(20.0 * np.log10(np.mean(inside) / np.mean(outside)))


def cnr(inside, outside):
    """Contrast-to-noise ratio."""
    a = np.asarray(inside, dtype=float).ravel()
    b = np.asarray(outside, dtype=float).ravel()
    return float(abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var()))


def gcnr(inside, outside, bins=256):
    """Generalized contrast-to-noise ratio, between 0 and 1."""
    a = np.asarray(inside, dtype=float).ravel()
    b = np.asarray(outside, dtype=float).ravel()
    edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return float(1.0 - np.minimum(ha, hb).sum())


def speckle_snr(region):
    """Mean over standard deviation of envelope amplitude."""
    r = np.asarray(region, dtype=float).ravel()
    return float(r.mean() / r.std())


def snr_db(signal, noise):
    """Ratio of RMS values [dB]."""
    s = np.sqrt(np.mean(np.asarray(signal, dtype=float) ** 2))
    n = np.sqrt(np.mean(np.asarray(noise, dtype=float) ** 2))
    return float(20.0 * np.log10(s / n))

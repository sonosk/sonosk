import numpy as np

from sonosk import quality


def test_width_gaussian():
    x = np.linspace(-5e-3, 5e-3, 2001)
    s = 0.5e-3
    a = np.exp(-x ** 2 / (2 * s ** 2))
    # -6.02 dB on amplitude is half maximum
    w = quality.width(a, x, level_db=20 * np.log10(0.5))
    assert np.isclose(w, 2 * np.sqrt(2 * np.log(2)) * s, rtol=1e-3)


def test_rayleigh_speckle():
    rng = np.random.default_rng(1)
    env = np.abs(rng.normal(size=200000) + 1j * rng.normal(size=200000))
    assert np.isclose(quality.speckle_snr(env), quality.RAYLEIGH_SNR, rtol=0.01)


def test_gcnr_limits():
    rng = np.random.default_rng(2)
    a = rng.rayleigh(1.0, 50000)
    b = rng.rayleigh(1.0, 50000)
    assert quality.gcnr(a, b) < 0.05
    assert np.isclose(quality.gcnr(a, a + 100.0), 1.0)


def test_contrast_and_cnr():
    inside = np.full(100, 0.1)
    outside = np.full(100, 1.0)
    assert np.isclose(quality.contrast_db(inside, outside), -20.0)
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, 100000)
    b = rng.normal(2.0, 1.0, 100000)
    assert np.isclose(quality.cnr(a, b), 2.0 / np.sqrt(2.0), rtol=0.02)


def test_snr_db():
    assert np.isclose(quality.snr_db(np.ones(10) * 10.0, np.ones(10)), 20.0)

import numpy as np

from sonosk import imaging as im, phantom, quality, transducer as tr

C = 1540.0
F0 = 5e6


def _image(transmits, pos, amp, x, z):
    arr = tr.linear_array(64, 0.3e-3, dx=0.15e-3)
    pulse = im.Pulse(F0, 0.6)
    rfs = [im.simulate(arr, tx, pos, amp, pulse, C)[0] for tx in transmits]
    return im.compound(rfs, 8 * F0, arr, transmits, x, z, C)


def test_pulse_bandwidth():
    p = im.Pulse(5e6, 0.6)
    fs = 200e6
    t = np.arange(-2e-6, 2e-6, 1 / fs)
    s = np.abs(np.fft.fft(p(t), 1 << 16))
    f = np.fft.fftfreq(1 << 16, 1 / fs)
    pos = f > 0
    lo = f[pos][s[pos] >= 0.5 * s[pos].max()].min()
    hi = f[pos][s[pos] >= 0.5 * s[pos].max()].max()
    assert np.isclose((hi - lo) / 5e6, 0.6, rtol=0.01)


def test_point_target_position_and_axial_resolution():
    lam = C / F0
    pos, amp = phantom.point_targets([(1e-3, 0, 20e-3)])
    x = np.arange(-3e-3, 5e-3, lam / 8)
    z = np.arange(17e-3, 23e-3, lam / 16)
    env = np.abs(_image([im.Transmit.plane(0.0)], pos, amp, x, z))
    iz, ix = np.unravel_index(np.argmax(env), env.shape)
    assert abs(x[ix] - 1e-3) < lam / 4
    assert abs(z[iz] - 20e-3) < lam / 8
    # two-way Gaussian pulse: axial -6 dB width = c * fwhm_time / 2
    sigma = im.Pulse(F0, 0.6).sigma
    expected = C * 2 * sigma * np.sqrt(2 * np.log(2)) / 2
    assert np.isclose(quality.width(env[:, ix], z), expected, rtol=0.05)


def test_transmit_kinds():
    e = np.stack([np.linspace(-5e-3, 5e-3, 11), np.zeros(11), np.zeros(11)], 1)
    p = np.array([[0.0, 0.0, 20e-3]])
    t, _, _ = im.Transmit.plane(0.0).travel(p, e, C)
    assert np.isclose(t[0], 20e-3 / C)
    src = np.array([0.0, 0.0, -10e-3])
    t, _, a = im.Transmit.diverging(src).travel(p, e, C)
    assert np.isclose(t[0], 20e-3 / C)
    assert np.isclose(a[0], 10e-3 / 30e-3)
    fp = np.array([0.0, 0.0, 30e-3])
    tx = im.Transmit.focused(fp)
    d = tx.element_delays(e, C)
    arrive = d + np.linalg.norm(e - fp, axis=1) / C
    assert np.allclose(arrive, arrive[0])


def test_cyst_contrast():
    rng = np.random.default_rng(3)
    n = 4000
    pos = np.stack([rng.uniform(-4e-3, 4e-3, n), np.zeros(n), rng.uniform(14e-3, 26e-3, n)], 1)
    keep = np.hypot(pos[:, 0], pos[:, 2] - 20e-3) > 2e-3
    pos = pos[keep]
    amp = rng.normal(size=pos.shape[0])
    x = np.arange(-3e-3, 3e-3, 0.05e-3)
    z = np.arange(15e-3, 25e-3, 0.05e-3)
    env = np.abs(_image([im.Transmit.plane(a) for a in (-0.1, 0.0, 0.1)], pos, amp, x, z))
    zz, xx = np.meshgrid(z, x, indexing="ij")
    r = np.hypot(xx, zz - 20e-3)
    inside, out = env[r < 1.2e-3], env[(r > 2.5e-3) & (np.abs(xx) < 2.5e-3)]
    assert quality.contrast_db(inside, out) < -15
    assert quality.gcnr(inside, out) > 0.8

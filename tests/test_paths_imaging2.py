import numpy as np

import sonosk as sk
from sonosk import imaging as im, paths, phantom, quality, stack, transducer as tr


def _layered(c_top, depth, c_bottom=1540.0):
    g = sk.Grid.centered((241, 301), 0.1e-3, depth_start=0.0)
    lab = (g.mesh()[1] < depth).astype(int)
    return sk.Medium(g, lab, [sk.Material("deep", c_bottom, 1000, alpha0=0.5),
                              sk.Material("top", c_top, 950, alpha0=0.5)])


def test_transmit_paths_match_homogeneous_travel_times():
    med = _layered(1540.0, 5e-3)
    model = paths.RayModel(med, 5e6, 1540.0)
    e = tr.linear_array(48, 0.3e-3).centers
    rng = np.random.default_rng(0)
    p = np.stack([rng.uniform(-7e-3, 7e-3, 500), np.zeros(500), rng.uniform(1e-3, 29e-3, 500)], 1)
    for tx in (im.Transmit.plane(0.15), im.Transmit.diverging((1e-3, 0, -8e-3)),
               im.Transmit.focused((0.5e-3, 0, 15e-3))):
        t_ref, _, _ = tx.travel(p, e, 1540.0)
        t, _ = paths.transmit_paths(model, tx, p, e)
        assert np.abs(t - t_ref).max() < 2e-9


def test_layer_delay_matches_slowness_integral():
    med = _layered(1440.0, 8e-3)
    model = paths.RayModel(med, 5e6, 1540.0)
    tau, att = model.line(np.array([[0.0, 0.0]]), np.array([[0.0, 20e-3]]))
    assert np.isclose(tau[0], 8e-3 / 1440 + 12e-3 / 1540, rtol=1e-4)
    assert np.isclose(att[0], sk.medium.attenuation_np(0.5, 5e6) * 20e-3, rtol=1e-4)
    arr = tr.linear_array(64, 0.3e-3)
    ap = paths.ArrayPaths(model, arr.centers, arr.element_normals, element_step=8)
    pts = np.array([[1e-3, 0, 20e-3]])
    ex, _ = ap(pts)
    for i in (0, 20, 63):
        L = np.hypot(1e-3 - arr.centers[i, 0], 20e-3)
        exact = L * (8 / 20 / 1440 + 12 / 20 / 1540) - L / 1540
        assert abs(ex[i, 0] - exact) < 5e-9


def test_simulation_with_uniform_medium_matches_homogeneous_model():
    g = sk.Grid.centered((161, 251), 0.1e-3, depth_start=0.0)
    med = sk.Medium.homogeneous(g, sk.Material("b", 1540, 1000, alpha0=0.5))
    arr = tr.linear_array(32, 0.3e-3, dx=0.15e-3)
    pulse = im.Pulse(5e6, 0.6)
    pos, amp = phantom.point_targets([(1e-3, 0, 12e-3), (-2e-3, 0, 20e-3)])
    tx = im.Transmit.plane(0.0)
    a, _ = im.simulate(arr, tx, pos, amp, pulse, 1540.0, alpha0=0.5)
    b, _ = im.simulate(arr, tx, pos, amp, pulse, 1540.0, medium=med)
    assert np.abs(a - b).max() < 0.03 * np.abs(a).max()


def test_sound_speed_aware_beamforming_corrects_a_fat_layer():
    med = _layered(1440.0, 8e-3)
    arr = tr.linear_array(64, 0.3e-3, dx=0.15e-3)
    pulse = im.Pulse(5e6, 0.6)
    pos, amp = phantom.point_targets([(2e-3, 0, 22e-3)])
    tx = im.Transmit.plane(0.0)
    rf, fs = im.simulate(arr, tx, pos, amp, pulse, 1540.0, medium=med)
    x = np.arange(-2e-3, 6e-3, 0.025e-3)
    z = np.arange(18e-3, 26e-3, 0.025e-3)
    for m, tol in ((None, None), (med, 0.05e-3)):
        env = np.abs(im.beamform(rf, fs, arr, tx, x, z, 1540.0, medium=m))
        iz, ix = np.unravel_index(np.argmax(env), env.shape)
        if m is None:
            assert z[iz] > 22.4e-3       # the slow layer pushes the target deeper
        else:
            assert abs(z[iz] - 22e-3) < tol and abs(x[ix] - 2e-3) < tol


def test_frequency_downshift_with_depth():
    arr = tr.linear_array(16, 0.3e-3, dx=0.15e-3)
    pulse = im.Pulse(5e6, 0.6)
    pos, amp = phantom.point_targets([(0, 0, 60e-3)])
    rf, fs = im.simulate(arr, im.Transmit.plane(0.0), pos, amp, pulse, 1540.0, alpha0=0.5)
    s = np.abs(np.fft.fft(rf[8].real, 1 << 15)) ** 2
    f = np.fft.fftfreq(1 << 15, 1 / fs)
    keep = f > 0
    fc = np.sum(f[keep] * s[keep]) / np.sum(s[keep])
    # Gaussian spectrum times exp(-A f / f0): shift of -A sigma_f^2 / f0
    a = sk.medium.attenuation_np(0.5, 5e6) * 2 * 60e-3
    sigma_f = 0.6 * 5e6 / (2 * np.sqrt(2 * np.log(2)))
    assert np.isclose(fc, 5e6 - a * sigma_f ** 2 / 5e6, rtol=0.02)


def test_scan_conversion_and_sector_mask():
    r = np.linspace(0, 40e-3, 201)
    th = np.linspace(-0.6, 0.6, 121)
    img = np.outer(r, np.ones_like(th))          # value = radius
    x = np.linspace(-20e-3, 20e-3, 81)
    z = np.linspace(1e-3, 39e-3, 77)
    out = im.scan_convert(img, r, th, x, z)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    ok = np.isfinite(out)
    assert np.allclose(out[ok], np.hypot(xx, zz)[ok], atol=1e-6)
    m = im.sector_mask(x, z, (0, 0), 0.6, 0, 40e-3)
    assert np.array_equal(m, ok | (np.hypot(xx, zz) > 40e-3) & False | ok)


def test_tgc_and_noise():
    z = np.linspace(0, 0.05, 11)
    img = np.ones((11, 3))
    out = im.tgc(img, z, 0.5, 5e6)
    assert np.isclose(out[-1, 0], np.exp(2 * sk.medium.attenuation_np(0.5, 5e6) * 0.05))
    rf = np.zeros((4, 4000), complex)
    rf[0, 0] = 1.0
    n = im.add_noise(rf, 40, rng=0) - rf
    assert np.isclose(np.std(n.real) * np.sqrt(2), 1e-2, rtol=0.05)


def test_pulse_from_stack_has_the_stack_bandwidth():
    piezo = stack.Piezo(0.4e-3, 4350, 7500, 0.49, 1200, 1e-6)
    s = stack.Stack(piezo, backing=3e6, layers=[stack.Layer(stack.quarter_wave(2500, 5.4e6),
                                                         2500, 1720)], load=1.5e6)
    p = im.Pulse.from_stack(s)
    assert 4e6 < p.f0 < 7e6
    t = np.linspace(-2e-6, 2e-6, 4001)
    env = np.abs(p.analytic(t))
    assert env.max() == env[np.argmin(np.abs(t))] or np.isclose(env.max(), 1.0)
    assert p.duration < 4e-6


def test_speckle_statistics_with_frequency_dependent_attenuation():
    g = sk.Grid.centered((121, 201), 0.1e-3, depth_start=10e-3)
    med = sk.Medium.homogeneous(g, sk.Material("b", 1540, 1000, alpha0=0.5))
    pos, amp = phantom.scatterers(med, 60, rng=4)
    arr = tr.linear_array(64, 0.3e-3, dx=0.15e-3)
    pulse = im.Pulse(5e6, 0.6)
    txs = [im.Transmit.plane(a) for a in (-0.08, 0.0, 0.08)]
    rfs = [im.simulate(arr, tx, pos, amp, pulse, 1540.0, alpha0=0.5)[0] for tx in txs]
    x = np.arange(-4e-3, 4e-3, 0.05e-3)
    z = np.arange(14e-3, 26e-3, 0.05e-3)
    env = np.abs(im.tgc(im.compound(rfs, 8 * pulse.f0, arr, txs, x, z, 1540.0), z, 0.5, 5e6))
    assert abs(quality.speckle_snr(env) - 1.91) < 0.15

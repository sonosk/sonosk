import numpy as np
import pytest

import sonosk as sk
from sonosk import fullwave as fw, imaging, phantom, transducer
from sonosk.medium import attenuation_np


def _bessel_j(n, x):
    tau = np.linspace(0, np.pi, 4001)
    return np.trapezoid(np.cos(n * tau - x * np.sin(tau)), tau) / np.pi


def test_arrival_time_and_amplitude_1d():
    dx = 0.05e-3
    g = sk.Grid(1600, dx)
    med = sk.Medium.homogeneous(g, sk.Material("w", 1500, 1000))
    burst = fw.tone_burst(1e6, 3)
    res = fw.simulate(med, fw.Source([[200]], burst, weight=2.0), 45e-6,
                      sensor=fw.Sensor([[600], [1200]]))
    peak_src = 2.0 * np.abs(burst(np.linspace(0, 3e-6, 20001))).max()
    for i, x in enumerate((600, 1200)):
        env = np.abs(fw.analytic_signal(res.p[i]))
        t_peak = res.t[np.argmax(env)]
        assert abs(t_peak - ((x - 200) * dx / 1500 + 1.5e-6)) < 2 * res.dt
        assert np.isclose(np.abs(res.p[i]).max(), peak_src, rtol=2e-3)


def test_power_law_absorption_and_dispersion():
    dx = 0.05e-3
    g = sk.Grid(2400, dx)
    a0, y, c = 0.75, 1.5, 1540.0
    med = sk.Medium.homogeneous(g, sk.Material("t", c, 1000, alpha0=a0, y=y))
    src = fw.Source([[100]], fw.gaussian_pulse(imaging.Pulse(3e6, 0.8)))
    res = fw.simulate(med, src, 90e-6, sensor=fw.Sensor([[400], [1400]]))
    n = 1 << 16
    P = np.fft.rfft(res.p, n=n, axis=1)
    f = np.fft.rfftfreq(n, res.dt)
    d = 1000 * dx
    phase = np.unwrap(np.angle(P[1]) - np.angle(P[0]))
    for fq in (1e6, 2e6, 3e6, 4e6, 5e6):
        i = np.argmin(np.abs(f - fq))
        alpha = np.log(np.abs(P[0, i]) / np.abs(P[1, i])) / d
        assert np.isclose(alpha, attenuation_np(a0, f[i], y), rtol=0.025)
        # phase speed against the Kramers-Kronig relation, c the low-frequency limit
        c_ph = 2 * np.pi * f[i] * d / (-phase[i])
        assert abs(c_ph - fw.dispersion_speed(c, a0, y, f[i], 1e-9)) < 0.15


def test_reference_frequency_sets_phase_speed():
    dx = 0.05e-3
    g = sk.Grid(2000, dx)
    med = sk.Medium.homogeneous(g, sk.Material("t", 1540.0, 1000, alpha0=0.75, y=1.5))
    src = fw.Source([[100]], fw.gaussian_pulse(imaging.Pulse(3e6, 0.8)))
    res = fw.simulate(med, src, 70e-6, sensor=fw.Sensor([[400], [1400]]), f_ref=3e6)
    n = 1 << 16
    P = np.fft.rfft(res.p, n=n, axis=1)
    f = np.fft.rfftfreq(n, res.dt)
    i = np.argmin(np.abs(f - 3e6))
    phase = np.unwrap(np.angle(P[1]) - np.angle(P[0]))[i]
    assert np.isclose(2 * np.pi * f[i] * 1000 * dx / (-phase), 1540.0, atol=1.0)


def test_reflection_coefficient():
    dx = 0.05e-3
    g = sk.Grid(2000, dx)
    lab = np.zeros(2000, int)
    lab[1000:] = 1
    med = sk.Medium(g, lab, [sk.Material("a", 1500, 1000), sk.Material("b", 1800, 1600)])
    res = fw.simulate(med, fw.Source([[300]], fw.tone_burst(1e6, 3)), 45e-6,
                      sensor=fw.Sensor([[600]]))
    p, t = res.p[0], res.t
    r = np.abs(p[t > 30e-6]).max() / np.abs(p[t < 15e-6]).max()
    assert np.isclose(r, (2.88e6 - 1.5e6) / (2.88e6 + 1.5e6), rtol=0.005)


def test_fubini_harmonics():
    c, rho, ba, f, p0 = 1500.0, 1000.0, 5.0, 1e6, 1e6
    x_shock = rho * c ** 3 / ((1 + ba / 2) * 2 * np.pi * f * p0)
    dx = 0.04e-3
    n = 2000
    g = sk.Grid(n, dx)
    med = sk.Medium.homogeneous(g, sk.Material("fluid", c, rho, ba=ba))
    src = fw.Source([[20]], fw.continuous_wave(f, 3), weight=p0)
    res = fw.simulate(med, src, n * dx / c + 12 / f, nonlinear=True, cw=[f, 2 * f, 3 * f],
                      cw_cycles=6, pml=40)
    for xs in (0.03, 0.07):
        i = int(round(xs / dx)) + 20
        s = xs / x_shock
        for k in (1, 2, 3):
            fub = 2 * _bessel_j(k, k * s) / (k * s)
            assert abs(np.abs(res.cw[k * f][i]) / p0 - fub) < 2e-3


def test_point_source_2d_cylindrical_spreading():
    g = sk.Grid((200, 200), 0.1e-3)
    med = sk.Medium.homogeneous(g, sk.Material("w", 1500, 1000))
    res = fw.simulate(med, fw.Source([[100, 100]], fw.tone_burst(1e6, 2)), 12e-6,
                      sensor=fw.Sensor([[100, 125], [100, 150], [150, 100]]),
                      precision="single")
    env = np.abs(fw.analytic_signal(res.p, axis=1)).max(axis=1)
    assert np.isclose(env[0] / env[1], np.sqrt(2), rtol=0.01)
    assert np.isclose(env[1], env[2], rtol=1e-3)


def test_single_precision_matches_double():
    g = sk.Grid((120, 140), 0.1e-3)
    lab = phantom.paint(np.zeros(g.shape, int), phantom.halfspace(g, 7e-3), 1)
    med = sk.Medium(g, lab, [sk.Material("a", 1500, 1000, alpha0=0.5, y=1.5),
                             sk.Material("b", 1600, 1100, alpha0=0.8, y=1.5)])
    src = fw.Source([[60, 10]], fw.tone_burst(1.5e6, 2))
    kw = dict(sensor=fw.Sensor([[60, 100]]))
    a = fw.simulate(med, src, 12e-6, **kw).p
    b = fw.simulate(med, src, 12e-6, precision="single", **kw).p
    assert np.abs(a - b).max() < 1e-3 * np.abs(a).max()


def test_pulse_echo_locates_a_point_target():
    g = sk.Grid.centered((161, 221), 0.1e-3, depth_start=0.0)
    lab = phantom.paint(np.zeros(g.shape, int), phantom.sphere(g, (1.5e-3, 0, 14e-3), 0.15e-3), 1)
    med = sk.Medium(g, lab, [sk.Material("w", 1540, 1000), sk.Material("t", 1540, 2000)])
    arr = transducer.linear_array(48, 0.3e-3, dx=0.1e-3)
    tx = imaging.Transmit.plane(0.0)
    rf, fs = fw.pulse_echo(med, arr, tx, imaging.Pulse(3e6, 0.7), precision="single")
    x = np.arange(-5e-3, 5e-3, 0.05e-3)
    z = np.arange(6e-3, 20e-3, 0.05e-3)
    img = np.abs(imaging.beamform(rf, fs, arr, tx, x, z))
    iz, ix = np.unravel_index(np.argmax(img), img.shape)
    assert abs(x[ix] - 1.5e-3) < 0.2e-3
    assert abs(z[iz] - 14e-3) < 0.3e-3


def test_source_outside_grid_is_rejected():
    g = sk.Grid((20, 20), 0.1e-3)
    med = sk.Medium.homogeneous(g, sk.Material("w", 1500, 1000))
    with pytest.raises(ValueError):
        fw.simulate(med, fw.Source([[25, 5]], fw.tone_burst(1e6, 2)), 1e-6)


def test_velocity_sensor_of_a_plane_wave():
    # 1-D plane wave: u = p / (rho c), recorded half a cell deeper
    dx = 0.05e-3
    g = sk.Grid(1600, dx)
    med = sk.Medium.homogeneous(g, sk.Material("w", 1500, 1000))
    src = fw.Source([[200]], fw.tone_burst(1e6, 3))
    p = fw.simulate(med, src, 30e-6, sensor=fw.Sensor([[800]])).p[0]
    u = fw.simulate(med, src, 30e-6, sensor=fw.Sensor([[800]], quantity="u")).p[0]
    assert np.isclose(np.abs(u).max() * 1500 * 1000, np.abs(p).max(), rtol=0.01)


def test_directional_elements_do_not_hear_the_transmit():
    g = sk.Grid.centered((161, 121), 0.1e-3, depth_start=0.0)
    med = sk.Medium.homogeneous(g, sk.Material("w", 1540, 1000))
    arr = transducer.linear_array(48, 0.3e-3, height=4e-3, dx=0.1e-3)
    pulse = imaging.Pulse(4e6, 0.7)
    tx = imaging.Transmit.plane(0.0)
    omni, _ = fw.pulse_echo(med, arr, tx, pulse, directional=False, band=None)
    direc, _ = fw.pulse_echo(med, arr, tx, pulse)
    # an empty medium has no echoes: what is left is the wave along the face
    assert np.abs(direc).max() < 0.1 * np.abs(omni).max()


def test_fluctuations_without_speed_variation():
    g = sk.Grid((20, 20), 0.1e-3)
    med = sk.Medium.homogeneous(g, sk.Material("t", 1540, 1000, echo=0.0))
    f = phantom.fluctuations(med, level=0.02, rng=0, speed=0.0)
    assert np.all(f.c == 1540) and 0.01 < np.std(f.rho) / 1000 < 0.03

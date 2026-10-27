import numpy as np

import sonosk as sk
from sonosk import field, thermal, tissues, transducer as tr

TISSUE = sk.Material("t", 1540, 1050, alpha0=0.5, k=0.5, cp=3600, perfusion=1.0)


def test_arrhenius_at_constant_temperature():
    g = sk.Grid((11,), 1e-3)
    med = sk.Medium.homogeneous(g, TISSUE)
    res = thermal.bioheat(med, np.zeros(g.shape), 20.0, T0=60.0, T_blood=60.0,
                          arrhenius=thermal.HENRIQUES)
    expected = thermal.HENRIQUES.rate(60.0) * 20.0
    assert np.allclose(res.damage, expected, rtol=1e-6)
    # the two branches meet at the break temperature within a factor of a few
    lo = thermal.HENRIQUES.A * np.exp(-thermal.HENRIQUES.Ea / (thermal.R_GAS * 328.15))
    hi = thermal.HENRIQUES.A_high * np.exp(-thermal.HENRIQUES.Ea_high / (thermal.R_GAS * 328.15))
    assert 0.2 < lo / hi < 5


def test_continuing_a_run_equals_one_long_run():
    g = sk.Grid((41, 41), 0.5e-3)
    med = sk.Medium.homogeneous(g, TISSUE)
    x, z = g.mesh()
    q = 2e7 * np.exp(-((x - 10e-3) ** 2 + (z - 10e-3) ** 2) / (2 * 1e-3 ** 2))
    one = thermal.bioheat(med, q, 4.0, dt=0.01)
    a = thermal.bioheat(med, q, 2.0, dt=0.01)
    b = thermal.bioheat(med, q, 2.0, dt=0.01, initial=a)
    assert np.allclose(b.T_final, one.T_final)
    assert np.allclose(b.dose, one.dose)
    assert b.time == 4.0


def test_perfusion_stops_in_the_lesion():
    g = sk.Grid((41, 41), 0.5e-3)
    med = sk.Medium.homogeneous(g, TISSUE)
    x, z = g.mesh()
    q = 1e8 * np.exp(-((x - 10e-3) ** 2 + (z - 10e-3) ** 2) / (2 * 1e-3 ** 2))
    res = thermal.bioheat(med, q, 6.0, shutdown=("dose", 240.0))
    les = res.lesion()
    assert les.any() and (~les).any()
    assert np.all(res.perfusion[les] == 0)
    assert np.allclose(res.perfusion[~les], med.perfusion[~les])
    free = thermal.bioheat(med, q, 6.0)
    assert res.lesion().sum() >= free.lesion().sum()


def test_treatment_of_two_points_and_report():
    g = sk.Grid((41, 41), 0.5e-3)
    lab = np.zeros(g.shape, int)
    x, z = g.mesh()
    lab[np.hypot(x - 10e-3, z - 10e-3) < 3e-3] = 1
    med = sk.Medium(g, lab, [TISSUE, TISSUE.with_(name="tumour")])
    q0 = 1.5e8 * np.exp(-((x - 10e-3) ** 2 + (z - 10e-3) ** 2) / (2 * 0.6e-3 ** 2))
    shots = [thermal.Sonication(thermal.shift(q0, g, (dx, 0.0, 0.0)), 3.0, cooling=3.0)
             for dx in (-1.5e-3, 1.5e-3)]
    plan = thermal.treat(med, shots)
    assert len(plan.log) == 2 and plan.log[1][2] >= plan.log[0][2]
    rep = thermal.treatment_report(plan.lesion(), med, "tumour")
    assert 0 < rep["fraction_ablated"] <= 1
    assert np.isclose(rep["ablated"] + rep["residual"], rep["target_volume"])


def test_shift_moves_the_centroid():
    g = sk.Grid((64, 64), 0.25e-3)
    x, z = g.mesh()
    q = np.exp(-((x - 8e-3) ** 2 + (z - 8e-3) ** 2) / (2 * 1e-3 ** 2))
    s = thermal.shift(q, g, (1.3e-3, 0.0, -0.7e-3))
    cx = np.sum(s * x) / s.sum()
    cz = np.sum(s * z) / s.sum()
    assert np.isclose(cx, 9.3e-3, atol=1e-5) and np.isclose(cz, 7.3e-3, atol=1e-5)


def test_raster_and_circles():
    g = sk.Grid((81, 81), 0.25e-3)
    x, z = g.mesh()
    disk = np.hypot(x - 10e-3, z - 10e-3) < 5e-3
    pts = thermal.raster(disk, g, 2e-3)
    assert len(pts) > 10
    assert np.all(np.hypot(pts[:, 0] - 10e-3, pts[:, 1] - 10e-3) < 5e-3)
    c = thermal.circles([2e-3, 4e-3], spacing=1e-3)
    assert np.allclose(np.linalg.norm(c[1:13], axis=1), 2e-3)


def test_coagulation_changes_echo_and_attenuation():
    g = sk.Grid((10, 10), 1e-3)
    med = sk.Medium.homogeneous(g, tissues.material("prostate"))
    T = np.full(g.shape, 37.0)
    T[2, 2] = 62.0
    T[5, 5] = 70.0
    out = thermal.coagulate(med, T)
    base = med.materials[0]
    assert np.isclose(out.echo[2, 2] - base.echo, 10 * np.log10(5))
    assert np.isclose(out.alpha0[2, 2], base.alpha0 * 1.25)
    assert np.isclose(out.echo[5, 5] - base.echo, 10 * np.log10(9))
    assert np.isclose(out.alpha0[5, 5], base.alpha0 * 2.7)
    assert out.echo[0, 0] == base.echo


def test_harmonic_heat_source():
    g = sk.Grid((5,), 1e-3)
    med = sk.Medium.homogeneous(g, TISSUE)
    p1 = np.full(5, 1e6, complex)
    p2 = np.full(5, 3e5, complex)
    q = thermal.heat_source_harmonics({1e6: p1, 2e6: p2}, med)
    ref = (thermal.heat_source(p1, med.attenuation(1e6), med.rho, med.c)
           + thermal.heat_source(p2, med.attenuation(2e6), med.rho, med.c))
    assert np.allclose(q, ref)


def test_hybrid_angular_spectrum_matches_rayleigh_in_water():
    f, c = 1e6, 1500.0
    src = tr.bowl(8e-3, 25e-3, dx=0.3e-3)
    dx = 0.375e-3
    g = sk.Grid.centered((81, 81, 61), dx, depth_start=8e-3)
    med = sk.Medium.homogeneous(g, sk.Material("w", c, 1000))
    plane = sk.Grid.centered((81, 81, 1), dx, depth_start=8e-3)
    p0 = field.rayleigh_grid(src, plane, f, c)[..., 0]
    p = field.hybrid_angular_spectrum(p0, med, f)
    z = g.axis(2)
    ref = field.rayleigh(src, np.stack([0 * z, 0 * z, z], 1), f, c)
    assert np.abs(np.abs(p[40, 40]) - np.abs(ref)).max() < 3e-3 * np.abs(ref).max()


def test_hybrid_angular_spectrum_transmission_and_attenuation():
    f = 1e6
    g = sk.Grid.centered((64, 200), 0.3e-3, depth_start=0.0)
    lab = (g.mesh()[1] >= 30e-3).astype(int)
    m1 = sk.Material("a", 1500, 1000)
    m2 = sk.Material("b", 1600, 1200, alpha0=0.6, y=1.0)
    med = sk.Medium(g, lab, [m1, m2])
    p = field.hybrid_angular_spectrum(np.ones(64, complex), med, f, taper=0.0, pad=1)
    z = g.axis(1)
    k = np.argmin(np.abs(z - 50e-3))
    T = 2 * m2.impedance / (m1.impedance + m2.impedance)
    depth_in_b = z[k] - z[np.argmax(z >= 30e-3)]
    expected = T * np.exp(-sk.medium.attenuation_np(0.6, f) * depth_in_b)
    assert np.isclose(np.abs(p[32, k]), expected, rtol=0.01)


def test_pulsed_field_spectrum_matches_cw():
    from sonosk import imaging
    pist = tr.piston(4e-3, dx=0.15e-3)
    pulse = imaging.Pulse(1.5e6, 0.6)
    pt = np.array([[0.0, 0.0, 40e-3]])
    t, p = field.pulsed(pist, pt, pulse, c=1500.0, alpha0=0.5, y=1.2)
    dt = t[1] - t[0]
    u = pulse.analytic(t - 0.5 * pulse.duration).real
    f0 = 1.5e6
    w = np.exp(-2j * np.pi * f0 * t)
    ratio = np.sum(p[0] * w) / np.sum(u * w)
    cw = field.rayleigh(pist, pt, f0, 1500.0, alpha0=0.5, y=1.2)[0]
    assert np.isclose(ratio, cw, rtol=0.01)
    k = np.argmax(np.abs(p[0]))
    assert abs(t[k] - (40e-3 / 1500 + 0.5 * pulse.duration)) < 0.5 / f0 + dt


def test_cooled_surface():
    g = sk.Grid((5, 40), 0.5e-3)
    med = sk.Medium.homogeneous(g, TISSUE.with_(perfusion=0.0))
    res = thermal.bioheat(med, np.zeros(g.shape), 60.0, T_surface=20.0)
    assert np.all(res.T_final[:, 0] == 20.0)
    col = res.T_final[2]
    assert np.all(np.diff(col[:-1]) >= 0) and col[1] < 37.0 and col[-1] == 37.0

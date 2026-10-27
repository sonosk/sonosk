"""Tests added after the review of the 0.2.0 release candidate."""

import warnings

import numpy as np
import pytest

import sonosk as sk
from sonosk import benchmarks, paths, phantom, safety, thermal, tissues
from sonosk import fullwave as fw
from sonosk import transducer as tr
from sonosk.medium import attenuation_np

# ---------------------------------------------------------------- safety


@pytest.mark.parametrize("mi, isppa3, ispta3, ok", [
    (1.5, None, 500.0, True),      # MI within, ISPTA within
    (2.5, None, 500.0, False),     # MI above, no ISPPA given
    (2.5, 150.0, 500.0, True),     # MI above but ISPPA within: allowed
    (2.5, 250.0, 500.0, False),    # both above
    (1.5, 250.0, 500.0, True),     # ISPPA above but MI within: allowed
    (1.5, None, 800.0, False),     # ISPTA above
])
def test_track3_general_rules(mi, isppa3, ispta3, ok):
    r = safety.check_limits(mi=mi, isppa3=isppa3, ispta3=ispta3)
    assert r["ok"] is ok and r["complete"]


def test_track3_incomplete_input_is_not_ok():
    r = safety.check_limits(mi=0.5)
    assert not r["complete"] and not r["ok"] and r["notes"]
    r = safety.check_limits(mi=0.2, ispta3=40.0, application="ophthalmic")
    assert not r["complete"] and not r["ok"]
    r = safety.check_limits(mi=1.0, ispta3=100.0, ti=7.0)
    assert r["ok"] and any("TI" in n for n in r["notes"])


def test_safety_input_checks():
    with pytest.raises(ValueError):
        safety.check_limits(mi=-1.0, ispta3=10.0)
    with pytest.raises(ValueError):
        safety.check_limits(mi=float("nan"), ispta3=10.0)
    with pytest.raises(ValueError):
        safety.pulse_duration(np.ones(10), 0.0)
    with pytest.raises(ValueError):
        safety.mechanical_index([1.0, 2.0], [1.0], 3.0)


def test_pulse_measures_work_row_by_row():
    fs = 100e6
    t = np.arange(400) / fs
    a = np.sin(2 * np.pi * 3e6 * t) * (t < 1e-6)
    b = 2 * np.sin(2 * np.pi * 3e6 * t) * (t < 2e-6)
    both = np.stack([a, b])
    pd = safety.pulse_duration(both, 1 / fs)
    assert np.allclose(pd, [safety.pulse_duration(a, 1 / fs), safety.pulse_duration(b, 1 / fs)])
    ip = safety.isppa(both, 1 / fs)
    assert np.allclose(ip, [safety.isppa(a, 1 / fs), safety.isppa(b, 1 / fs)])
    assert np.allclose(safety.pulse_intensity_integral(both, 1 / fs),
                       [safety.pulse_intensity_integral(r, 1 / fs) for r in both])
    assert safety.isppa(np.zeros(10), 1 / fs) == 0.0

# ---------------------------------------------------------------- geometry


def test_grid_strict_index_and_crop():
    g = sk.Grid((11, 21), 1e-3)
    assert g.index((20e-3, 5e-3)) == (10, 5)
    with pytest.raises(ValueError):
        g.index((20e-3, 5e-3), strict=True)
    with pytest.raises(ValueError):
        g.index((1e-3,))
    with pytest.raises(ValueError):
        g.crop((0, 0), (12e-3, 5e-3), strict=True)
    with pytest.raises(ValueError):
        g.crop((5e-3, 5e-3), (1e-3, 9e-3))
    sub, sl = g.crop((2e-3, 3e-3), (4e-3, 6e-3))
    assert sub.shape == (3, 4) and sl == (slice(2, 5), slice(3, 7))


def test_element_with_cancelling_normals_is_rejected():
    pos = np.zeros((2, 3))
    normal = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    t = tr.Transducer(pos, normal, np.ones(2), np.zeros(2, int))
    with pytest.raises(ValueError):
        t.element_normals

# ---------------------------------------------------------------- full-wave


def _water_1d(n=400, dx=0.05e-3):
    g = sk.Grid((n,), dx)
    return sk.Medium.homogeneous(g, sk.Material("w", 1500, 1000))


def test_fullwave_argument_checks():
    med = _water_1d()
    src = fw.Source([[50]], fw.tone_burst(1e6, 2))
    for kw in ({"t_end": 0.0}, {"t_end": 1e-6, "dt": -1.0}, {"t_end": 1e-6, "cfl": 1.5},
               {"t_end": 1e-6, "precision": "half"}, {"t_end": 1e-6, "record_every": 0},
               {"t_end": 1e-6, "pml": -2}, {"t_end": 1e-6, "y": "per tissue"}):
        t_end = kw.pop("t_end")
        with pytest.raises(ValueError):
            fw.simulate(med, src, t_end, **kw)
    with pytest.warns(RuntimeWarning):
        fw.simulate(med, src, 1e-6, cfl=0.8)
    with pytest.raises(MemoryError):
        fw.simulate(med, src, 1e-6, max_memory=1000)
    assert fw.memory_estimate(sk.Grid((100, 100), 1e-4), precision="single") > 0


def test_fullwave_stops_when_the_field_is_not_finite():
    med = _water_1d()
    src = fw.Source([[50]], lambda t: np.where(t > 1e-6, np.inf, 0.0))
    with pytest.raises(FloatingPointError):
        fw.simulate(med, src, 5e-6)


def test_per_tissue_exponents():
    # attenuation measured inside a slab whose exponent differs from the
    # median of the grid: y="tissue" follows the tissue law, a single
    # exponent does not
    dx = 0.05e-3
    g = sk.Grid(3000, dx)
    lab = np.zeros(3000, int)
    lab[1800:] = 1
    med = sk.Medium(g, lab, [sk.Material("A", 1540, 1000, alpha0=0.5, y=1.2),
                             sk.Material("B", 1540, 1000, alpha0=0.4, y=1.8)])
    from sonosk import imaging
    src = fw.Source([[100]], fw.gaussian_pulse(imaging.Pulse(3e6, 0.8)))
    errs = {}
    for mode in ("tissue", None):
        res = fw.simulate(med, src, 120e-6, sensor=fw.Sensor([[2050], [2850]]), f_ref=3e6,
                          y=mode)
        n = 1 << 16
        P = np.fft.rfft(res.p, n=n, axis=1)
        f = np.fft.rfftfreq(n, res.dt)
        band = (f > 1.5e6) & (f < 4.5e6)
        a = -np.log(np.abs(P[1, band]) / np.abs(P[0, band])) / (800 * dx)
        errs[mode] = np.max(np.abs(a / attenuation_np(0.4, f[band], 1.8) - 1))
    assert errs["tissue"] < 0.04 and errs[None] > 0.3


def test_exponent_error_helper():
    r, dc = fw.exponent_error(0.5, 1.0, np.array([1e6, 3e6, 9e6]), 3e6, 1.5)
    assert np.isclose(r[1], 1.0) and np.isclose(dc[1], 0.0, atol=1e-9)
    assert np.allclose(r, (np.array([1e6, 3e6, 9e6]) / 3e6) ** 0.5)

# ---------------------------------------------------------------- hybrid angular spectrum


def test_has_converges_in_a_gradient_and_is_exact_at_a_step():
    out = benchmarks.has_dz_convergence(dz=(0.4e-3, 0.2e-3, 0.1e-3))
    e = out["smooth_error"]
    assert e[0] > e[1] > e[2] and e[2] < 3e-4
    assert np.isclose(e[0] / e[1], 2.0, rtol=0.05)     # first order in dz
    assert max(out["sharp_error"]) < 1e-12

# ---------------------------------------------------------------- imaging


def test_interface_loss_of_one_step():
    g = sk.Grid.centered((41, 81), 0.25e-3, depth_start=0.0)
    lab = (g.mesh()[1] > 10e-3).astype(int)
    m1, m2 = sk.Material("a", 1500, 1000), sk.Material("b", 1600, 1600)
    med = sk.Medium(g, lab, [m1, m2])
    model = paths.RayModel(med, 5e6, interfaces=True)
    tau, att, loss = model.line(np.array([[0.0, 2e-3]]), np.array([[0.0, 18e-3]]))
    r = (m2.impedance - m1.impedance) / (m2.impedance + m1.impedance)
    assert np.isclose(loss[0], -0.5 * np.log(1 - r * r), rtol=1e-6)
    pm = model.polar_map((0.0, 0.0))
    assert np.isclose(pm(0.0, 18e-3)[2], -0.5 * np.log(1 - r * r), rtol=1e-3)


def test_interface_scatterers_sum_to_the_reflection():
    g = sk.Grid.centered((21, 201), 0.1e-3, depth_start=0.0)
    lab = (g.mesh()[1] > 10e-3).astype(int)
    m1, m2 = sk.Material("a", 1500, 1000), sk.Material("b", 1600, 1100)
    med = sk.Medium(g, lab, [m1, m2])
    pos, amp = phantom.interface_scatterers(med, gain=1.0, threshold=0.0)
    col = np.abs(pos[:, 0]) < 1e-9
    total = amp[col].sum()
    assert np.isclose(total, 0.5 * np.log(m2.impedance / m1.impedance), rtol=0.02)
    assert np.all(np.abs(pos[col][np.argmax(amp[col])][2] - 10e-3) < 0.3e-3)

# ---------------------------------------------------------------- thermal


TISSUE = sk.Material("t", 1540, 1000, alpha0=0.5, k=0.5, cp=4000, perfusion=0.0)


def test_absorption_fraction():
    q1 = thermal.heat_source(1e6, 10.0, 1000.0, 1500.0)
    q2 = thermal.heat_source(1e6, 10.0, 1000.0, 1500.0, fraction=0.7)
    assert np.isclose(q2, 0.7 * q1)
    with pytest.raises(ValueError):
        thermal.heat_source(1e6, 10.0, 1000.0, 1500.0, fraction=1.2)


def test_arrhenius_on_a_temperature_ramp():
    # no conduction, no perfusion: the middle cell heats linearly and its
    # damage must match an independent integration of the rate
    g = sk.Grid((3,), 1e-3)
    med = sk.Medium.homogeneous(g, TISSUE.with_(k=0.0))
    q = np.zeros(3)
    q[1] = 4e6 * 30.0 / 10.0                  # 30 K in 10 s
    res = thermal.bioheat(med, q, 10.0, dt=1e-3, arrhenius=thermal.HENRIQUES)
    t = np.linspace(0, 10, 200001)
    T = 37.0 + 3.0 * t
    Tk = T + 273.15
    rate = np.where(T > 55.0, 5.0e45 * np.exp(-3.54e4 / Tk), 3.1e98 * np.exp(-7.5e4 / Tk))
    omega = np.trapezoid(rate, t)
    assert np.isclose(res.T_final[1], 67.0, atol=1e-6)
    assert np.isclose(res.damage[1], omega, rtol=0.01)


def test_damage_model_per_material():
    g = sk.Grid((7,), 1e-3)
    lab = np.array([0, 0, 0, 1, 1, 1, 1])
    med = sk.Medium(g, lab, [TISSUE.with_(name="skin"), TISSUE.with_(name="fat")])
    slow = thermal.Arrhenius(A=1.0, Ea=1e3)
    res = thermal.bioheat(med, np.zeros(7), 1.0, T0=60.0, T_blood=60.0,
                          arrhenius={"skin": thermal.HENRIQUES, "*": slow})
    assert np.isclose(res.damage[1], thermal.HENRIQUES.rate(60.0))
    assert np.isclose(res.damage[5], slow.rate(60.0))
    with pytest.raises(ValueError):
        thermal.bioheat(med, np.zeros(7), 1.0, arrhenius={"bone": slow})


def test_warning_above_100_c_and_temperature_dependent_conduction():
    g = sk.Grid((41,), 0.5e-3)
    med = sk.Medium.homogeneous(g, TISSUE)
    x = g.axis(0)
    q = 3e8 * np.exp(-(x - 10e-3) ** 2 / (2 * 1e-3 ** 2))
    with pytest.warns(RuntimeWarning):
        hot = thermal.bioheat(med, q, 4.0)
    assert hot.notes and hot.T_max.max() > 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast = thermal.bioheat(med, q, 4.0, k_coeff=0.01)
    assert fast.T_max.max() < hot.T_max.max()
    with pytest.raises(ValueError):
        thermal.bioheat(med, q, 1.0, k_coeff=-0.05)


def test_dose_does_not_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = thermal.dose_rate(np.array([2000.0, 37.0]))
    assert np.all(np.isfinite(r))
    assert np.isclose(thermal.cem43(np.full((60, 2), 43.0), 1.0)[0], 1.0)

# ---------------------------------------------------------------- tissues


def test_provenance_and_uncertainty():
    kinds = {"measured", "fitted", "database average", "used in a simulation study",
             "review compilation", "generic soft-tissue value", "inherited", "reference level",
             "modelling choice"}
    for n in tissues.names():
        assert set(tissues.provenance(n).values()) <= kinds
        # no inheritance cycles
        key, seen = n, set()
        while "like" in tissues.TISSUES[key]:
            assert key not in seen
            seen.add(key)
            key = tissues.TISSUES[key]["like"]
    for (n, p), (sd, lo, hi, cnt) in tissues.RANGES.items():
        v = tissues.properties(n)[p][0]
        assert sd >= 0 and lo <= v * 1.001 + 1e-9 and v <= hi * 1.001 + 1e-9 and cnt >= 1
    assert tissues.uncertainty("rectal wall", "c") == tissues.RANGES[("muscle", "c")]
    assert tissues.uncertainty("fat", "c") is None
    m = tissues.sample("muscle", rng=3)
    lo, hi = tissues.RANGES[("muscle", "rho")][1:3]
    assert lo <= m.rho <= hi and m.c != tissues.material("muscle").c


def test_csv_export(tmp_path):
    p = tmp_path / "tissues.csv"
    tissues.to_csv(p)
    rows = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(rows) == 1 + len(tissues.names()) * len(tissues.PROPERTIES)

# ---------------------------------------------------------------- benchmarks


def test_benchmark_studies():
    pml = benchmarks.pml_reflection(thickness=(20,))
    assert pml["reflection_db"][0] < -60
    bh = benchmarks.bioheat_convergence(dx=(0.4e-3, 0.2e-3))
    assert 1.8 < bh["order"][0] < 2.2
    fw_c = benchmarks.fullwave_convergence(ppw=(3, 6))
    assert fw_c["error"]["transmitted"][0] < 0.5
    ex = benchmarks.exponent_errors()
    assert ex["tissues"]["fat"]["attenuation_ratio"][1] == 1.0
    info = sk.run_info()
    assert info["sonosk"] == sk.__version__ and "numpy" in info

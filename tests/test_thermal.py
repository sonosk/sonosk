import numpy as np

from sonosk import Grid, Material, Medium, thermal


def test_heating_without_losses():
    g = Grid((21, 21), 1e-3)
    md = Medium.homogeneous(g, Material("gel", 1540, 1000, k=0.0, cp=4000.0))
    r = thermal.bioheat(md, np.full(g.shape, 1e6), duration=10.0, dt=0.01)
    assert np.isclose(r.T_max[10, 10] - 37.0, 1e6 * 10 / 4e6)


def test_perfusion_relaxation():
    g = Grid((11, 11), 1e-3)
    md = Medium.homogeneous(g, Material("t", 1540, 1000, k=0.0, cp=4000.0, perfusion=1.0))
    r = thermal.bioheat(md, np.full(g.shape, 1e5), duration=60.0)
    tau = 4e6 / thermal.C_BLOOD
    expected = 1e5 / thermal.C_BLOOD * (1 - np.exp(-60.0 / tau))
    assert np.isclose(r.T_max[5, 5] - 37.0, expected, rtol=1e-3)


def test_conduction_gaussian():
    g = Grid((401,), 0.1e-3, origin=-20e-3)
    x = g.axis(0)
    md = Medium.homogeneous(g, Material("w", 1500, 1000, k=0.6, cp=4000.0))
    s0 = 1e-3
    q = 1e7 * np.exp(-x ** 2 / (2 * s0 ** 2))
    r = thermal.bioheat(md, q, duration=20.0, dt=0.01, snapshots=[20.0],
                        schedule=lambda t: 1.0 if t < 0.1 - 1e-9 else 0.0)
    D = 0.6 / 4e6
    s2 = s0 ** 2 + 2 * D * (20 - 0.05)
    Ta = 1e7 * 0.1 / 4e6 * s0 / np.sqrt(s2) * np.exp(-x ** 2 / (2 * s2))
    assert np.max(np.abs(r.T[-1] - 37.0 - Ta)) / Ta.max() < 1e-3


def test_dose():
    assert np.isclose(thermal.cem43(np.full(3600, 43.0), 1.0), 60.0)
    assert np.isclose(thermal.cem43(np.full(1800, 44.0), 1.0), 60.0)
    assert np.isclose(thermal.cem43(np.full(3600, 42.0), 1.0), 15.0)


def test_heat_source():
    q = thermal.heat_source(1e6, alpha=5.0, rho=1000.0, c=1500.0)
    assert np.isclose(q, 5.0 * 1e12 / 1.5e6)


def test_treatment_report():
    g = Grid((10, 10), 1e-3)
    lab = np.zeros(g.shape, int)
    lab[2:6, 2:6] = 1
    md = Medium(g, lab, [Material("normal", 1540, 1000), Material("tumour", 1540, 1000)])
    lesion = np.zeros(g.shape, bool)
    lesion[3:7, 3:7] = True
    rep = thermal.treatment_report(lesion, md, "tumour")
    cell = 1e-6
    assert np.isclose(rep["target_volume"], 16 * cell)
    assert np.isclose(rep["ablated"], 9 * cell)
    assert np.isclose(rep["residual"], 7 * cell)
    assert np.isclose(rep["healthy_damaged"], 7 * cell)
    assert np.isclose(rep["fraction_ablated"], 9 / 16)

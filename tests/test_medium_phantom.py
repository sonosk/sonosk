import numpy as np
import pytest

from sonosk import Grid, Material, Medium, medium, phantom


def test_attenuation_units():
    # 1 dB/(cm MHz) at 1 MHz is 100 dB/m = 11.51 Np/m
    assert np.isclose(medium.attenuation_np(1.0, 1e6), 100 * np.log(10) / 20)
    m = Material("t", 1540, 1000, alpha0=0.5, y=1.1)
    assert np.isclose(m.attenuation(3e6), 0.5 * 3 ** 1.1 * 100 * np.log(10) / 20)
    assert m.impedance == 1540 * 1000
    assert m.with_(c=1500).c == 1500


def test_medium_maps():
    g = Grid((10, 20), 1e-3)
    lab = phantom.layers(g, [5e-3, 5e-3])
    mats = [Material("water", 1480, 1000), Material("fat", 1450, 950, alpha0=0.5),
            Material("muscle", 1580, 1050, alpha0=1.0), Material("deep", 1540, 1040)]
    md = Medium(g, lab, mats)
    assert md.c.shape == g.shape
    assert np.allclose(md.c[:, 2], 1450)
    assert np.allclose(md.c[:, 7], 1580)
    assert md.mask("muscle").sum() == 10 * 5
    with pytest.raises(KeyError):
        md.mask("bone")
    with pytest.raises(ValueError):
        Medium(g, lab, mats[:2])


def test_shapes():
    g = Grid.centered((201, 201), 0.1e-3, depth_start=0.0)
    disc = phantom.sphere(g, (0, 0, 10e-3), 3e-3)
    assert np.isclose(disc.sum() * 1e-8, np.pi * 9e-6, rtol=0.02)
    e = phantom.ellipsoid(g, (0, 0, 10e-3), (4e-3, 1, 2e-3))
    assert np.isclose(e.sum() * 1e-8, np.pi * 8e-6, rtol=0.02)
    cyl = phantom.cylinder(g, (0, 0, 10e-3), (1, 0, 0), 1e-3)
    assert np.isclose(cyl.sum() * 1e-8, 20e-3 * 2e-3, rtol=0.06)
    b = phantom.box(g, (-1.05e-3, 0, 4.95e-3), (1.05e-3, 0, 6.05e-3))
    assert b.sum() == 21 * 11


def test_scatterers_density_and_echo():
    g = Grid.centered((101, 101), 0.1e-3, depth_start=0.0)
    lab = phantom.paint(np.zeros(g.shape, int), phantom.halfspace(g, 5e-3), 1)
    md = Medium(g, lab, [Material("a", 1540, 1000, echo=0.0),
                         Material("b", 1540, 1000, echo=-np.inf)])
    pos, amp = phantom.scatterers(md, density=50, rng=0)
    assert pos.shape[1] == 3
    assert np.all(pos[:, 2] < 5e-3 + 0.1e-3)          # none in the anechoic half
    assert np.isclose(len(amp), 50 * 10 * 5, rtol=0.05)
    assert np.allclose(pos[:, 1], 0.0)

import numpy as np

from sonosk import Grid, analytic, field, transducer as tr


def _axis(z):
    return np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)


def test_piston_on_axis():
    f, c = 1e6, 1500.0
    t = tr.piston(5e-3, dx=c / f / 8)
    z = np.linspace(20e-3, 0.1, 100)
    p = np.abs(field.rayleigh(t, _axis(z), f, c))
    pa = analytic.piston_axial(z, 5e-3, f, c)
    assert np.max(np.abs(p - pa) / pa) < 5e-3


def test_bowl_on_axis():
    f, c = 1e6, 1500.0
    b = tr.bowl(15e-3, 40e-3, dx=c / f / 6)
    z = np.linspace(15e-3, 70e-3, 200)
    p = np.abs(field.rayleigh(b, _axis(z), f, c))
    pa = analytic.bowl_axial(z, 15e-3, 40e-3, f, c)
    assert np.max(np.abs(p - pa)) / pa.max() < 5e-3


def test_attenuation():
    f, c = 2e6, 1540.0
    t = tr.piston(3e-3, dx=0.1e-3)
    z = np.array([60e-3])
    p0 = np.abs(field.rayleigh(t, _axis(z), f, c))
    p1 = np.abs(field.rayleigh(t, _axis(z), f, c, alpha0=0.5))
    # far from a small piston the path length is close to z
    assert np.isclose(20 * np.log10(p0 / p1), 0.5 * 2 * 6.0, rtol=0.01)


def test_angular_spectrum_against_rayleigh():
    f, c = 1e6, 1500.0
    lam = c / f
    t = tr.piston(5e-3, dx=lam / 6)
    dx = lam / 3
    x = np.arange(-30e-3, 30e-3 + 1e-9, dx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    p0 = field.rayleigh(t, np.stack([X.ravel(), Y.ravel(), np.full(X.size, 5e-3)], 1),
                        f, c).reshape(X.shape)
    p1 = field.angular_spectrum(p0, dx, f, [25e-3], c)[0]
    pr = field.rayleigh(t, np.stack([x, 0 * x, np.full(x.size, 30e-3)], 1), f, c)
    i = np.argmin(np.abs(x))
    assert np.max(np.abs(np.abs(p1[:, i]) - np.abs(pr))) / np.abs(pr).max() < 0.04


def test_grid_helpers():
    t = tr.piston(2e-3, dx=0.2e-3)
    g1 = Grid((50,), 1e-3, origin=5e-3)
    assert field.rayleigh_grid(t, g1, 1e6).shape == (50,)
    g2 = Grid.centered((11, 21), 1e-3, depth_start=5e-3)
    assert field.rayleigh_grid(t, g2, 1e6).shape == (11, 21)


def test_scaling():
    p = field.scale_to_pressure(2.0, rho=1000.0, c=1500.0, u0=0.1)
    assert np.isclose(p, 2.0 * 1000 * 1500 * 0.1)
    assert np.isclose(field.intensity(1e6, 1000.0, 1500.0), 1e12 / 3e6)

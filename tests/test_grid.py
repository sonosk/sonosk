import numpy as np
import pytest

from sonosk import Grid


def test_dimensions():
    for shape in [(10,), (10, 20), (10, 20, 30)]:
        g = Grid(shape, 1e-4)
        assert g.ndim == len(shape)
        assert g.size == np.prod(shape)
        assert g.points().shape == (g.size, g.ndim)


def test_bad_input():
    with pytest.raises(ValueError):
        Grid((2, 2, 2, 2), 1e-4)
    with pytest.raises(ValueError):
        Grid((10, 10), 0.0)


def test_axes_and_extent():
    g = Grid((11, 21), (1e-4, 2e-4), origin=(0.0, 5e-3))
    assert np.isclose(g.axis(0)[-1], 1e-3)
    assert np.isclose(g.axis(1)[0], 5e-3)
    assert np.allclose(g.extent, [1e-3, 4e-3])


def test_centered():
    g = Grid.centered((101, 201), 1e-4, depth_start=2e-3)
    x, z = g.axes()
    assert np.isclose(x[50], 0.0)
    assert np.isclose(z[0], 2e-3)


def test_points_per_wavelength():
    g = Grid((10, 10), 1e-4)
    # 1540 m/s at 1.54 MHz gives a 1 mm wavelength
    assert np.isclose(g.points_per_wavelength(1540.0, 1.54e6), 10.0)

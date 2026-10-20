import numpy as np
import pytest

from sonosk import analytic


def test_piston_peak_and_last_maximum():
    a, f, c = 5e-3, 2e6, 1500.0
    z = np.linspace(1e-4, 0.1, 200001)
    p = analytic.piston_axial(z, a, f, c)
    assert np.isclose(p.max(), 2.0, atol=1e-6)
    zmax = analytic.piston_last_maximum(a, f, c)
    far = z > 0.5 * zmax
    assert np.isclose(z[far][np.argmax(p[far])], zmax, rtol=1e-3)


def test_piston_far_field():
    # far from the piston |p/p0| -> k a^2 / (2 z)
    a, f, c = 3e-3, 1e6, 1500.0
    z = 1.0
    k = analytic.wavenumber(f, c)
    assert np.isclose(analytic.piston_axial(z, a, f, c), k * a * a / (2 * z), rtol=1e-3)


def test_bowl_focus():
    a, roc, f, c = 15e-3, 40e-3, 3e6, 1500.0
    g = analytic.focal_gain(a, roc, f, c)
    assert np.isclose(analytic.bowl_axial(roc, a, roc, f, c), g)
    # continuous through the focus
    dz = 1e-7
    side = analytic.bowl_axial([roc - dz, roc + dz], a, roc, f, c)
    assert np.allclose(side, g, rtol=1e-3)


def test_bowl_depth():
    assert np.isclose(analytic.bowl_depth(3.0, 5.0), 1.0)
    with pytest.raises(ValueError):
        analytic.bowl_depth(6.0, 5.0)

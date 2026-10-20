import numpy as np
import pytest

from sonosk import analytic, transducer as tr


def test_piston_and_bowl_area():
    p = tr.piston(5e-3, dx=0.2e-3)
    assert np.isclose(p.total_area, np.pi * 25e-6)
    b = tr.bowl(15e-3, 40e-3, dx=0.3e-3)
    h = analytic.bowl_depth(15e-3, 40e-3)
    assert np.isclose(b.total_area, 2 * np.pi * 40e-3 * h)
    # bowl normals point at the centre of curvature
    to_focus = np.array([0, 0, 40e-3]) - b.pos
    to_focus /= np.linalg.norm(to_focus, axis=1, keepdims=True)
    assert np.allclose(np.sum(to_focus * b.normal, axis=1), 1.0)


def test_annular_rings():
    a = tr.annular([(0, 3e-3), (3.2e-3, 5e-3), (5.2e-3, 7e-3)], roc=50e-3, dx=0.2e-3)
    assert a.n_elements == 3
    assert np.all(np.bincount(a.elem) > 0)


def test_linear_array_layout():
    arr = tr.linear_array(64, 0.3e-3, height=4e-3, dx=0.1e-3)
    c = arr.centers
    assert arr.n_elements == 64
    assert np.isclose(c[1, 0] - c[0, 0], 0.3e-3)
    assert np.isclose(c[:, 0].mean(), 0.0, atol=1e-12)
    assert np.isclose(arr.total_area, 64 * 0.27e-3 * 4e-3)
    with pytest.raises(ValueError):
        tr.linear_array(8, 0.3e-3, width=0.4e-3)


def test_elevation_lens():
    arr = tr.linear_array(4, 0.3e-3, height=4e-3, elev_focus=20e-3, dx=0.1e-3)
    # every patch normal points at the elevation focus line
    d = np.stack([np.zeros(arr.n_patches), -arr.pos[:, 1], 20e-3 - arr.pos[:, 2]], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert np.allclose(np.sum(d * arr.normal, axis=1), 1.0)


def test_convex_array_on_arc():
    arr = tr.convex_array(33, 0.5e-3, radius=12e-3, dx=0.1e-3)
    c = arr.centers
    r = np.linalg.norm(c - np.array([0, 0, -12e-3]), axis=1)
    assert np.allclose(r, 12e-3, rtol=1e-3)
    n = arr.element_normals
    radial = (c - np.array([0, 0, -12e-3])) / r[:, None]
    assert np.allclose(np.sum(n * radial, axis=1), 1.0, atol=1e-6)


def test_focus_and_steer_delays():
    arr = tr.linear_array(32, 0.3e-3, dx=0.1e-3)
    arr.focus((0, 0, 20e-3), c=1540.0)
    d = np.linalg.norm(arr.centers - np.array([0, 0, 20e-3]), axis=1)
    # arrival at the focus is the same for all elements
    assert np.allclose(arr.delays + d / 1540.0, (arr.delays + d / 1540.0)[0])
    arr.steer(np.deg2rad(10), c=1540.0)
    assert arr.delays.min() == 0.0
    assert np.all(np.diff(arr.delays) > 0)


def test_matrix_and_cylindrical():
    m = tr.matrix_array(8, 8, 0.4e-3, dx=0.2e-3)
    assert m.n_elements == 64
    cyl = tr.cylindrical(10e-3, 8e-3, 15e-3, dx=0.2e-3)
    to_line = np.stack([-cyl.pos[:, 0], np.zeros(cyl.n_patches), 15e-3 - cyl.pos[:, 2]], 1)
    to_line /= np.linalg.norm(to_line, axis=1, keepdims=True)
    assert np.allclose(np.sum(to_line * cyl.normal, axis=1), 1.0)

import numpy as np

import sonosk as sk
from sonosk import field, tissues, transducer as tr


def test_spherical_array_layout_and_steering():
    cap = tr.spherical_array(64, 2e-3, 60e-3, 80e-3, dx=0.5e-3)
    assert cap.n_elements == 64
    d = np.linalg.norm(cap.centers - np.array([0, 0, 80e-3]), axis=1)
    assert np.allclose(d, 80e-3)
    assert np.isclose(cap.total_area, 64 * np.pi * 2e-3 ** 2, rtol=0.02)
    cap.focus((6e-3, 0, 80e-3), c=1500.0)
    x = np.linspace(-4e-3, 12e-3, 161)
    p = np.abs(field.rayleigh(cap, np.stack([x, 0 * x, np.full_like(x, 80e-3)], 1), 1e6, 1500.0))
    assert abs(x[np.argmax(p)] - 6e-3) < 0.2e-3


def test_translate_rotate_subset():
    b = tr.bowl(10e-3, 30e-3, dx=0.5e-3)
    n0 = b.n_patches
    z = np.linspace(15e-3, 45e-3, 301)
    pts = np.stack([0 * z, 0 * z, z], 1)
    p0 = np.abs(field.rayleigh(b, pts, 1e6, 1500.0))
    b.translate((0, 0, 5e-3))
    p1 = np.abs(field.rayleigh(b, pts, 1e6, 1500.0))
    assert np.isclose(z[np.argmax(p1)] - z[np.argmax(p0)], 5e-3, atol=0.15e-3)
    b.rotate(np.pi / 2, axis="y", center=(0, 0, 5e-3))
    mean_n = (b.normal * b.area[:, None]).sum(0)
    mean_n /= np.linalg.norm(mean_n)
    assert np.allclose(np.abs(mean_n), [1, 0, 0], atol=1e-9)
    cut = tr.bowl(10e-3, 30e-3, dx=0.5e-3)
    half = cut.subset(cut.pos[:, 0] > 0)
    assert 0.4 * n0 < half.n_patches < 0.6 * n0 and half.n_elements == 1


def test_grid_and_medium_crop_and_slice():
    g = sk.Grid.centered((21, 11, 31), 1e-3, depth_start=0.0)
    lab = np.zeros(g.shape, int)
    lab[:, :, 15:] = 1
    med = sk.Medium(g, lab, [tissues.material("fat"), tissues.material("muscle")])
    sub, sl = med.crop((-3e-3, -2e-3, 10e-3), (3e-3, 2e-3, 20e-3))
    assert sub.grid.shape == (7, 5, 11)
    assert np.allclose(sub.c, med.c[sl])
    assert np.isclose(sub.grid.origin[2], 10e-3)
    m2 = med.slice2d(0.0)
    assert m2.grid.shape == (21, 31)
    assert np.allclose(m2.c, med.c[:, 5, :])
    assert g.index((0.0, 0.0, 15e-3)) == (10, 5, 15)


def test_medium_maps_can_be_replaced():
    g = sk.Grid((4, 4), 1e-3)
    med = sk.Medium.homogeneous(g, tissues.material("fat"))
    med.rho = np.full(g.shape, 900.0)
    assert np.all(med.rho == 900.0)
    assert np.all(med.impedance == 900.0 * med.c)

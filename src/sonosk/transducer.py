"""Transducer geometries.

A transducer is a set of elements. Each element is split into small
patches; a patch has a centre, an inward normal pointing into the medium
and an area. Field and imaging code sum over the patches, so the patch
size sets the accuracy. Patches no larger than a quarter wavelength are
a safe choice for fields close to the source.

Coordinates are in metres. The array face is near z = 0 and the beam goes
towards +z. For arrays, x is the array (azimuth) direction and y the
elevation direction.
"""

import numpy as np


class Transducer:
    """Collection of elements made of patches.

    Parameters
    ----------
    pos : (N, 3) array
        Patch centres [m].
    normal : (N, 3) array
        Unit normals pointing into the medium.
    area : (N,) array
        Patch areas [m^2].
    elem : (N,) int array
        Element index of each patch.
    width, height : float
        Element size along x and y [m], used for element directivity.
    name : str
    """

    def __init__(self, pos, normal, area, elem, width=0.0, height=0.0, name=""):
        self.pos = np.asarray(pos, dtype=float).reshape(-1, 3)
        self.normal = np.asarray(normal, dtype=float).reshape(-1, 3)
        self.area = np.asarray(area, dtype=float).ravel()
        self.elem = np.asarray(elem, dtype=int).ravel()
        n = self.pos.shape[0]
        if not (self.normal.shape[0] == n and self.area.size == n and self.elem.size == n):
            raise ValueError("pos, normal, area and elem must have the same length")
        self.n_elements = int(self.elem.max()) + 1 if n else 0
        self.width = float(width)
        self.height = float(height)
        self.name = name
        self.delays = np.zeros(self.n_elements)
        self.apod = np.ones(self.n_elements)

    def __repr__(self):
        return (f"Transducer({self.name!r}, elements={self.n_elements}, "
                f"patches={self.pos.shape[0]})")

    @property
    def n_patches(self):
        return self.pos.shape[0]

    @property
    def total_area(self):
        return float(self.area.sum())

    @property
    def centers(self):
        """Area-weighted element centres, (E, 3)."""
        w = np.bincount(self.elem, self.area, self.n_elements)
        return np.stack([np.bincount(self.elem, self.area * self.pos[:, i], self.n_elements) / w
                         for i in range(3)], axis=1)

    @property
    def element_normals(self):
        """Mean unit normal of each element, (E, 3)."""
        n = np.stack([np.bincount(self.elem, self.area * self.normal[:, i], self.n_elements)
                      for i in range(3)], axis=1)
        size = np.linalg.norm(n, axis=1, keepdims=True)
        area = np.bincount(self.elem, self.area, self.n_elements)[:, None]
        if np.any(size <= 1e-9 * area):
            raise ValueError("an element has no mean normal (its patch normals cancel)")
        return n / size

    def reset(self):
        self.delays = np.zeros(self.n_elements)
        self.apod = np.ones(self.n_elements)
        return self

    def set_apodization(self, w):
        w = np.asarray(w, dtype=float)
        if w.shape != (self.n_elements,):
            raise ValueError("one weight per element is needed")
        self.apod = w.copy()
        return self

    def focus(self, point, c=1540.0):
        """Delays that focus all elements at point [m]."""
        d = np.linalg.norm(self.centers - np.asarray(point, dtype=float), axis=1)
        self.delays = (d.max() - d) / c
        return self

    def steer(self, angle, c=1540.0):
        """Plane-wave delays for a steering angle [rad] in the x-z plane."""
        n = np.array([np.sin(angle), 0.0, np.cos(angle)])
        s = self.centers @ n
        self.delays = (s - s.min()) / c
        return self

    def subset(self, keep):
        """New transducer made of the patches where keep is true, for
        example a truncated bowl. Elements are renumbered."""
        keep = np.asarray(keep, bool)
        old = self.elem[keep]
        uniq, new = np.unique(old, return_inverse=True)
        t = Transducer(self.pos[keep], self.normal[keep], self.area[keep], new, self.width,
                       self.height, self.name)
        t.delays = self.delays[uniq].copy()
        t.apod = self.apod[uniq].copy()
        return t

    def translate(self, offset):
        """Move the transducer by offset (x, y, z) [m]."""
        self.pos = self.pos + np.asarray(offset, dtype=float)
        return self

    def rotate(self, angle, axis="y", center=(0.0, 0.0, 0.0)):
        """Rotate the transducer by angle [rad] about an axis through center."""
        c, s = np.cos(angle), np.sin(angle)
        if axis == "y":
            r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == "x":
            r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "z":
            r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError("axis must be x, y or z")
        c0 = np.asarray(center, dtype=float)
        self.pos = (self.pos - c0) @ r.T + c0
        self.normal = self.normal @ r.T
        return self

    def weights(self, f):
        """Complex source weight of each patch at frequency f [Hz]."""
        e = self.elem
        return self.apod[e] * np.exp(-2j * np.pi * f * self.delays[e])


def _window(name, n):
    if name in (None, "rect", "boxcar"):
        return np.ones(n)
    if name == "hann":
        return np.hanning(n + 2)[1:-1]
    if name == "hamming":
        return np.hamming(n)
    raise ValueError(f"unknown window '{name}'")


def apodization(n, window="hann"):
    """Element weights for a window over n elements."""
    return _window(window, n)


# ---------------------------------------------------------------- circular

def _rings(r_in, r_out, roc, dx):
    """Patches of an annulus on a flat (roc=None) or spherical surface."""
    pos, nrm, area = [], [], []
    if roc is None:
        nr = max(1, int(np.ceil((r_out - r_in) / dx)))
        edges = np.linspace(r_in, r_out, nr + 1)
        for r0, r1 in zip(edges[:-1], edges[1:]):
            r = 0.5 * (r0 + r1)
            nphi = max(1, int(round(2 * np.pi * r / dx))) if r > 0 else 1
            phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
            a = np.pi * (r1 ** 2 - r0 ** 2) / nphi
            pos.append(np.stack([r * np.cos(phi), r * np.sin(phi), np.zeros(nphi)], 1))
            nrm.append(np.tile([0.0, 0.0, 1.0], (nphi, 1)))
            area.append(np.full(nphi, a))
    else:
        if r_out > roc:
            raise ValueError("aperture radius larger than radius of curvature")
        t_in, t_out = np.arcsin(r_in / roc), np.arcsin(r_out / roc)
        nt = max(1, int(np.ceil(roc * (t_out - t_in) / dx)))
        edges = np.linspace(t_in, t_out, nt + 1)
        for t0, t1 in zip(edges[:-1], edges[1:]):
            t = 0.5 * (t0 + t1)
            r = roc * np.sin(t)
            nphi = max(1, int(round(2 * np.pi * r / dx))) if r > 0 else 1
            phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
            a = 2 * np.pi * roc ** 2 * (np.cos(t0) - np.cos(t1)) / nphi
            p = np.stack([r * np.cos(phi), r * np.sin(phi),
                          np.full(nphi, roc - roc * np.cos(t))], 1)
            n = np.array([0.0, 0.0, roc]) - p
            pos.append(p)
            nrm.append(n / np.linalg.norm(n, axis=1, keepdims=True))
            area.append(np.full(nphi, a))
    return np.concatenate(pos), np.concatenate(nrm), np.concatenate(area)


def piston(radius, dx=0.1e-3):
    """Flat circular piston of given radius at z = 0."""
    p, n, a = _rings(0.0, radius, None, dx)
    return Transducer(p, n, a, np.zeros(a.size, int), 2 * radius, 2 * radius, "piston")


def bowl(radius, roc, dx=0.1e-3):
    """Spherically focused bowl, apex at the origin, focus at (0, 0, roc)."""
    p, n, a = _rings(0.0, radius, roc, dx)
    return Transducer(p, n, a, np.zeros(a.size, int), 2 * radius, 2 * radius, "bowl")


def annular(radii, roc=None, dx=0.1e-3):
    """Annular array. radii is a list of (inner, outer) radii, one per ring."""
    ps, ns, as_, es = [], [], [], []
    for i, (r0, r1) in enumerate(radii):
        p, n, a = _rings(r0, r1, roc, dx)
        ps.append(p)
        ns.append(n)
        as_.append(a)
        es.append(np.full(a.size, i))
    w = 2 * max(r for _, r in radii)
    return Transducer(np.concatenate(ps), np.concatenate(ns), np.concatenate(as_),
                      np.concatenate(es), w, w, "annular")


# ---------------------------------------------------------- rectangular

def _rect_patches(width, height, dx, elev_focus=None):
    """Patches of one rectangle centred at the origin, facing +z."""
    nx = max(1, int(np.ceil(width / dx)))
    ny = max(1, int(np.ceil(height / dx)))
    x = (np.arange(nx) + 0.5) * width / nx - width / 2
    if elev_focus is None:
        y = (np.arange(ny) + 0.5) * height / ny - height / 2
        xx, yy = np.meshgrid(x, y, indexing="ij")
        p = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], 1)
        n = np.tile([0.0, 0.0, 1.0], (xx.size, 1))
        a = np.full(xx.size, width * height / (nx * ny))
        return p, n, a
    # cylindrical curvature in elevation, focus line at z = elev_focus
    tmax = np.arcsin(0.5 * height / elev_focus)
    t = (np.arange(ny) + 0.5) * 2 * tmax / ny - tmax
    xx, tt = np.meshgrid(x, t, indexing="ij")
    p = np.stack([xx.ravel(), elev_focus * np.sin(tt.ravel()),
                  elev_focus * (1 - np.cos(tt.ravel()))], 1)
    n = np.stack([np.zeros(xx.size), -np.sin(tt.ravel()), np.cos(tt.ravel())], 1)
    a = np.full(xx.size, width / nx * elev_focus * 2 * tmax / ny)
    return p, n, a


def _assemble(centres, frames, patches, width, height, name):
    p0, n0, a0 = patches
    ps, ns, as_, es = [], [], [], []
    for i, (c, rot) in enumerate(zip(centres, frames)):
        ps.append(p0 @ rot.T + c)
        ns.append(n0 @ rot.T)
        as_.append(a0)
        es.append(np.full(a0.size, i))
    return Transducer(np.concatenate(ps), np.concatenate(ns), np.concatenate(as_),
                      np.concatenate(es), width, height, name)


def linear_array(n, pitch, width=None, height=5e-3, elev_focus=None, dx=0.1e-3):
    """Linear array of n elements along x, centred on the origin.

    width is the element width (default 0.9 pitch, so kerf 0.1 pitch).
    elev_focus is the radius of the elevation lens [m]; None for a flat
    element.
    """
    width = 0.9 * pitch if width is None else width
    if width > pitch:
        raise ValueError("element width larger than pitch")
    xc = (np.arange(n) - (n - 1) / 2) * pitch
    centres = np.stack([xc, np.zeros(n), np.zeros(n)], 1)
    frames = [np.eye(3)] * n
    t = _assemble(centres, frames, _rect_patches(width, height, dx, elev_focus),
                  width, height, "linear")
    t.pitch = pitch
    return t


def phased_array(n, pitch, width=None, height=10e-3, elev_focus=None, dx=0.1e-3):
    """Phased array: a linear array with a small pitch, usually half a wavelength."""
    t = linear_array(n, pitch, width, height, elev_focus, dx)
    t.name = "phased"
    return t


def convex_array(n, pitch, radius, width=None, height=5e-3, elev_focus=None, dx=0.1e-3):
    """Convex (curved) array on an arc of given radius.

    The centre of curvature is at (0, 0, -radius), so the middle of the
    array touches z = 0. Small radii give endocavity probes (transrectal,
    transvaginal).
    """
    width = 0.9 * pitch if width is None else width
    dphi = pitch / radius
    phi = (np.arange(n) - (n - 1) / 2) * dphi
    centres = np.stack([radius * np.sin(phi), np.zeros(n),
                        radius * np.cos(phi) - radius], 1)
    frames = []
    for f in phi:
        c, s = np.cos(f), np.sin(f)
        frames.append(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
    t = _assemble(centres, frames, _rect_patches(width, height, dx, elev_focus),
                  width, height, "convex")
    t.pitch = pitch
    t.radius = radius
    return t


def matrix_array(nx, ny, pitch_x, pitch_y=None, width_x=None, width_y=None, dx=0.1e-3):
    """2-D matrix array of nx by ny elements in the z = 0 plane."""
    pitch_y = pitch_x if pitch_y is None else pitch_y
    width_x = 0.9 * pitch_x if width_x is None else width_x
    width_y = 0.9 * pitch_y if width_y is None else width_y
    xc = (np.arange(nx) - (nx - 1) / 2) * pitch_x
    yc = (np.arange(ny) - (ny - 1) / 2) * pitch_y
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    centres = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], 1)
    frames = [np.eye(3)] * xx.size
    t = _assemble(centres, frames, _rect_patches(width_x, width_y, dx), width_x, width_y,
                  "matrix")
    t.shape = (nx, ny)
    return t


def cylindrical(aperture, length, roc, dx=0.1e-3):
    """Cylindrically focused source, focusing to a line along y at z = roc.

    aperture is the chord width in x, length the size along y. This is the
    geometry of line-focused transducers used in aesthetic HIFU.
    """
    tmax = np.arcsin(0.5 * aperture / roc)
    nt = max(1, int(np.ceil(2 * tmax * roc / dx)))
    ny = max(1, int(np.ceil(length / dx)))
    t = (np.arange(nt) + 0.5) * 2 * tmax / nt - tmax
    y = (np.arange(ny) + 0.5) * length / ny - length / 2
    tt, yy = np.meshgrid(t, y, indexing="ij")
    p = np.stack([roc * np.sin(tt.ravel()), yy.ravel(), roc * (1 - np.cos(tt.ravel()))], 1)
    n = np.stack([-np.sin(tt.ravel()), np.zeros(tt.size), np.cos(tt.ravel())], 1)
    a = np.full(tt.size, roc * 2 * tmax / nt * length / ny)
    return Transducer(p, n, a, np.zeros(tt.size, int), aperture, length, "cylindrical")


def spherical_array(n, element_radius, aperture, roc, dx=0.25e-3, seed=None):
    """Phased array of n circular elements on a spherical cap.

    The element centres follow a sunflower (golden-angle) pattern over the
    cap of given aperture diameter and radius of curvature, which avoids
    the grating lobes of a regular layout. Each element is a small flat
    disc of element_radius, tilted to face the centre of curvature at
    (0, 0, roc). This is the layout of many MR-guided HIFU systems, which
    steer the focus electronically with focus().

    seed rotates the pattern by a random angle (None for no rotation).
    """
    a = aperture / 2
    if a + element_radius > roc:
        raise ValueError("aperture too large for the radius of curvature")
    t_max = np.arcsin((a - element_radius) / roc)
    # equal-area spacing on the cap: cos(t) uniform between 1 and cos(t_max)
    k = np.arange(n) + 0.5
    cos_t = 1 - k / n * (1 - np.cos(t_max))
    t = np.arccos(cos_t)
    rot = 0.0 if seed is None else np.random.default_rng(seed).uniform(0, 2 * np.pi)
    phi = k * np.pi * (3 - np.sqrt(5)) + rot
    centres = np.stack([roc * np.sin(t) * np.cos(phi), roc * np.sin(t) * np.sin(phi),
                        roc - roc * np.cos(t)], 1)
    p, nrm, ar = _rings(0.0, element_radius, None, dx)
    frames = []
    for c in centres:
        nz = (np.array([0.0, 0.0, roc]) - c) / roc
        ref = np.array([1.0, 0.0, 0.0]) if abs(nz[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ex = np.cross(ref, nz)
        ex /= np.linalg.norm(ex)
        ey = np.cross(nz, ex)
        frames.append(np.stack([ex, ey, nz], 1))
    t_ = _assemble(centres, frames, (p, nrm, ar), 2 * element_radius, 2 * element_radius,
                   "spherical array")
    t_.roc = roc
    return t_

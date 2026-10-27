"""Travel time and attenuation along straight rays in a heterogeneous medium.

The fast imaging model (sonosk.imaging.simulate) treats each echo as a
copy of the pulse. In a heterogeneous medium the delay of the echo is the
integral of the slowness 1/c along the transmit and receive paths, and its
amplitude falls with the integral of the attenuation. This module computes
those integrals along straight lines, which gives the arrival-time errors
(phase aberration) caused by layers with a different sound speed, the
acoustic shadow behind an attenuating lesion and the enhancement behind a
cyst. Refraction and multiple scattering are not included; the full-wave
solver in sonosk.fullwave covers them.

All integrals are computed as cumulative sums on ray-aligned grids and then
interpolated: a polar grid around each receive element or virtual source,
and a sheared grid of parallel rays for a plane wave. Receive maps are made
for a subset of the elements, and the excess delay over the straight path
at the nominal speed, which varies slowly from element to element, is
interpolated between them.

With interfaces=True a third integral is kept: the loss of amplitude at
the impedance steps crossed by the ray, sum of -ln(1 - R^2)/2 with R the
pressure reflection coefficient of each step. Over a two-way path this is
the factor T12 T21 = 1 - R^2 of every interface crossed. It does not
depend on frequency, so it is kept apart from the attenuation.

Only 2-D media (x, z) are supported. Points outside the grid, such as the
coupling gel between the probe and the first grid row, take the nominal
speed and no attenuation.
"""

import numpy as np


def _nearest(img, x0, dx, z0, dz, x, z, outside):
    """Nearest-neighbour value of img (nx, nz) at points (x, z)."""
    nx, nz = img.shape
    i = np.rint((np.asarray(x, dtype=float) - x0) / dx).astype(np.intp)
    k = np.rint((np.asarray(z, dtype=float) - z0) / dz).astype(np.intp)
    inside = (i >= 0) & (i < nx) & (k >= 0) & (k < nz)
    v = img[np.clip(i, 0, nx - 1), np.clip(k, 0, nz - 1)]
    return np.where(inside, v, outside)


def _step_loss(z_imp, axis):
    """Cumulative interface loss along axis from impedance samples, with a
    leading zero so that it matches the cumulative integrals."""
    a = np.take(z_imp, np.arange(z_imp.shape[axis] - 1), axis=axis)
    b = np.take(z_imp, np.arange(1, z_imp.shape[axis]), axis=axis)
    r = (b - a) / (b + a)
    loss = -0.5 * np.log1p(-np.minimum(r * r, 0.999999))
    shape = list(loss.shape)
    shape[axis] = 2
    return np.concatenate([np.zeros(shape), np.cumsum(loss, axis=axis)], axis=axis)


def _bilinear(imgs, x0, dx, z0, dz, x, z, outside):
    """Bilinear interpolation of each image in imgs (k, nx, nz) at points
    (x, z). Points outside get the matching value of outside (k,)."""
    _, nx, nz = imgs.shape
    fx = (np.asarray(x, dtype=float) - x0) / dx
    fz = (np.asarray(z, dtype=float) - z0) / dz
    inside = (fx >= 0) & (fx <= nx - 1) & (fz >= 0) & (fz <= nz - 1)
    fx = np.clip(fx, 0, nx - 1.000001)
    fz = np.clip(fz, 0, nz - 1.000001)
    i = fx.astype(np.intp)
    k = fz.astype(np.intp)
    wx = fx - i
    wz = fz - k
    w00 = (1 - wx) * (1 - wz)
    w10 = wx * (1 - wz)
    w01 = (1 - wx) * wz
    w11 = wx * wz
    flat = i * nz + k
    out = []
    for m, o in zip(imgs, outside):
        mf = m.ravel()
        v = (mf[flat] * w00 + mf[flat + nz] * w10 + mf[flat + 1] * w01 + mf[flat + nz + 1] * w11)
        out.append(np.where(inside, v, o))
    return out


class RayModel:
    """Straight-ray slowness and attenuation integrals in a 2-D medium.

    Parameters
    ----------
    medium : Medium
        2-D medium (x, z).
    f0 : float
        Frequency [Hz] at which the attenuation is evaluated.
    c_nominal : float
        Speed assumed outside the grid [m/s].
    step : float, optional
        Integration step along the rays [m], default the smallest grid
        spacing.
    interfaces : bool
        Also integrate the transmission loss at impedance steps.
    """

    def __init__(self, medium, f0, c_nominal=1540.0, step=None, interfaces=False):
        g = medium.grid
        if g.ndim != 2:
            raise ValueError("the ray model needs a 2-D medium (x, z)")
        self.grid = g
        self.f0 = float(f0)
        self.c_nominal = float(c_nominal)
        self.step = float(np.min(g.spacing)) if step is None else float(step)
        self.maps = np.stack([1.0 / np.asarray(medium.c, dtype=float),
                              np.asarray(medium.attenuation(f0), dtype=float)])
        self.outside = (1.0 / self.c_nominal, 0.0)
        self.x0, self.z0 = g.origin
        self.dx, self.dz = g.spacing
        self.interfaces = bool(interfaces)
        if self.interfaces:
            self.z_imp = np.asarray(medium.rho, dtype=float) * np.asarray(medium.c, dtype=float)
            self.z_out = float(np.mean(self.z_imp[:, 0]))

    def impedance(self, x, z):
        """Impedance at points (x, z), nearest grid value (steps stay sharp)."""
        return _nearest(self.z_imp, self.x0, self.dx, self.z0, self.dz, x, z, self.z_out)

    def sample(self, x, z):
        """Slowness [s/m] and attenuation [Np/m] at points (x, z)."""
        return _bilinear(self.maps, self.x0, self.dx, self.z0, self.dz, x, z, self.outside)

    def line(self, a, b, chunk=400_000):
        """Integrals of slowness [s] and attenuation [Np] from points a to
        points b, both (N, 2) arrays of (x, z); with interfaces also the
        interface loss [Np]."""
        a = np.asarray(a, dtype=float).reshape(-1, 2)
        b = np.asarray(b, dtype=float).reshape(-1, 2)
        L = np.linalg.norm(b - a, axis=1)
        m = max(int(np.ceil(L.max() / self.step)), 1) if L.size else 1
        tau = np.empty(L.size)
        att = np.empty(L.size)
        los = np.zeros(L.size)
        frac = (np.arange(m) + 0.5) / m
        per = max(chunk // m, 1)
        for i0 in range(0, L.size, per):
            sl = slice(i0, i0 + per)
            d = b[sl] - a[sl]
            x = a[sl, 0:1] + d[:, 0:1] * frac
            z = a[sl, 1:2] + d[:, 1:2] * frac
            s, al = self.sample(x, z)
            ds = (L[sl] / m)
            tau[sl] = s.sum(axis=1) * ds
            att[sl] = al.sum(axis=1) * ds
            if self.interfaces and m > 1:
                los[sl] = _step_loss(self.impedance(x, z), 1)[:, -1]
        if self.interfaces:
            return tau, att, los
        return tau, att

    def polar_map(self, center, normal=(0.0, 1.0), r_max=None, half_angle=np.pi / 2,
                  n_theta=None):
        """Integrals from center along every direction, on a polar grid.

        normal is the (x, z) direction of theta = 0; half_angle the largest
        abs(theta) covered (pi for all directions). Returns a PolarMap.
        """
        cx, cz = float(center[0]), float(center[1])
        if r_max is None:
            g = self.grid
            corners = np.array([[g.origin[0], g.origin[1]],
                                [g.origin[0] + g.extent[0], g.origin[1]],
                                [g.origin[0], g.origin[1] + g.extent[1]],
                                [g.origin[0] + g.extent[0], g.origin[1] + g.extent[1]]])
            r_max = np.max(np.hypot(corners[:, 0] - cx, corners[:, 1] - cz)) + self.step
        phi0 = float(np.arctan2(normal[0], normal[1]))
        nr = max(int(np.ceil(r_max / self.step)), 2)
        dr = r_max / nr
        if n_theta is None:
            arc = 2 * half_angle * r_max
            n_theta = max(int(np.ceil(arc / (2 * min(self.dx, self.dz)))) + 1, 16)
        th = np.linspace(-half_angle, half_angle, n_theta)
        rm = (np.arange(nr) + 0.5) * dr
        ang = phi0 + th[None, :]
        x = cx + rm[:, None] * np.sin(ang)
        z = cz + rm[:, None] * np.cos(ang)
        s, a = self.sample(x, z)
        zero = np.zeros((1, n_theta))
        maps = [np.concatenate([zero, np.cumsum(s, axis=0) * dr]),
                np.concatenate([zero, np.cumsum(a, axis=0) * dr])]
        if self.interfaces:
            maps.append(_step_loss(self.impedance(x, z), 0))
        return PolarMap(cx, cz, phi0, dr, th, np.stack(maps), self.c_nominal)

    def plane_map(self, angle, x_range, z_max):
        """Integrals along parallel rays leaving the plane z = 0 at the given
        angle [rad] from the depth axis. Returns a PlaneMap."""
        ct, st = np.cos(angle), np.sin(angle)
        s_max = z_max / ct + self.step
        ns = max(int(np.ceil(s_max / self.step)), 2)
        ds = s_max / ns
        x_lo = x_range[0] - z_max * max(np.tan(angle), 0) - self.step
        x_hi = x_range[1] - z_max * min(np.tan(angle), 0) + self.step
        nx = max(int(np.ceil((x_hi - x_lo) / self.dx)) + 1, 2)
        x0 = np.linspace(x_lo, x_hi, nx)
        sm = (np.arange(ns) + 0.5) * ds
        x = x0[:, None] + sm[None, :] * st
        z = np.broadcast_to(sm[None, :] * ct, x.shape)
        s, a = self.sample(x, z)
        zero = np.zeros((nx, 1))
        maps = [np.concatenate([zero, np.cumsum(s, axis=1) * ds], axis=1),
                np.concatenate([zero, np.cumsum(a, axis=1) * ds], axis=1)]
        if self.interfaces:
            maps.append(_step_loss(self.impedance(x, z), 1))
        return PlaneMap(angle, x0, ds, np.stack(maps))


class PolarMap:
    """Integrals of slowness and attenuation from a centre on a polar grid."""

    def __init__(self, cx, cz, phi0, dr, theta, maps, c_nominal):
        self.cx, self.cz, self.phi0 = cx, cz, phi0
        self.dr, self.theta = dr, theta
        self.maps = maps
        self.c_nominal = c_nominal

    def __call__(self, x, z):
        """Slowness and attenuation integrals from the centre to (x, z).
        Points beyond the map take the nominal speed and no attenuation."""
        dx = np.asarray(x, dtype=float) - self.cx
        dz = np.asarray(z, dtype=float) - self.cz
        rr = np.hypot(dx, dz)
        th = np.angle(np.exp(1j * (np.arctan2(dx, dz) - self.phi0)))
        dth = self.theta[1] - self.theta[0]
        vals = _bilinear(self.maps, 0.0, self.dr, self.theta[0], dth, rr, th,
                         (np.nan,) * len(self.maps))
        bad = ~np.isfinite(vals[0])
        if bad.any():
            vals[0] = np.where(bad, rr / self.c_nominal, vals[0])
            for i in range(1, len(vals)):
                vals[i] = np.where(bad, 0.0, vals[i])
        return tuple(vals)


class PlaneMap:
    """Integrals along the parallel rays of a plane wave, from z = 0."""

    def __init__(self, angle, x0, ds, maps):
        self.angle = angle
        self.x0 = x0
        self.ds = ds
        self.maps = maps

    def __call__(self, x, z):
        ct = np.cos(self.angle)
        s = np.asarray(z, dtype=float) / ct
        xs = np.asarray(x, dtype=float) - np.asarray(z, dtype=float) * np.tan(self.angle)
        return tuple(_bilinear(self.maps, self.x0[0], self.x0[1] - self.x0[0], 0.0, self.ds,
                               xs, s, (np.nan,) * len(self.maps)))


class ArrayPaths:
    """Receive-path integrals from every element of an array to points.

    Polar maps are built for every element_step-th element (and the last
    one); other elements are interpolated.
    """

    def __init__(self, model, centers, normals, r_max=None, element_step=4):
        self.model = model
        self.centers = np.asarray(centers, dtype=float)
        self.normals = np.asarray(normals, dtype=float)
        n = self.centers.shape[0]
        idx = list(range(0, n, max(int(element_step), 1)))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        self.idx = np.array(idx)
        self.maps = [model.polar_map(self.centers[i, [0, 2]], self.normals[i, [0, 2]], r_max)
                     for i in self.idx]

    def table(self, points):
        """PathTable with the excess delay and the attenuation of the
        sampled elements at the points, interpolated on request."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        cn = self.model.c_nominal
        ex_sub = np.empty((self.idx.size, p.shape[0]), dtype=np.float32)
        at_sub = np.empty_like(ex_sub)
        lo_sub = np.empty_like(ex_sub) if self.model.interfaces else None
        for j, (i, mp) in enumerate(zip(self.idx, self.maps)):
            vals = mp(p[:, 0], p[:, 2])
            r = np.hypot(p[:, 0] - self.centers[i, 0], p[:, 2] - self.centers[i, 2])
            ex_sub[j] = vals[0] - r / cn
            at_sub[j] = vals[1]
            if lo_sub is not None:
                lo_sub[j] = vals[2]
        return PathTable(self.idx, ex_sub, at_sub, self.centers.shape[0], lo_sub)

    def __call__(self, points):
        """Excess delay [s] over the straight path at the nominal speed and
        attenuation [Np], both (n_elements, n_points); with interfaces also
        the interface loss [Np]."""
        return self.table(points).full()


class PathTable:
    """Receive-path values of a subset of elements, interpolated linearly
    over the element index."""

    def __init__(self, idx, ex_sub, at_sub, n_elements, lo_sub=None):
        self.idx, self.ex_sub, self.at_sub = idx, ex_sub, at_sub
        self.lo_sub = lo_sub
        self.n = n_elements

    def _weights(self, i):
        if self.idx.size == 1:
            return 0, 0, 0.0
        k = int(np.clip(np.searchsorted(self.idx, i, side="right") - 1, 0, self.idx.size - 2))
        w = (i - self.idx[k]) / (self.idx[k + 1] - self.idx[k])
        return k, k + 1, w

    def excess(self, i):
        k0, k1, w = self._weights(i)
        return self.ex_sub[k0] * (1 - w) + self.ex_sub[k1] * w

    def attenuation(self, i):
        k0, k1, w = self._weights(i)
        return self.at_sub[k0] * (1 - w) + self.at_sub[k1] * w

    def loss(self, i):
        k0, k1, w = self._weights(i)
        return self.lo_sub[k0] * (1 - w) + self.lo_sub[k1] * w

    def full(self):
        ex = np.array([self.excess(i) for i in range(self.n)], dtype=float)
        at = np.array([self.attenuation(i) for i in range(self.n)], dtype=float)
        if self.lo_sub is not None:
            return ex, at, np.array([self.loss(i) for i in range(self.n)], dtype=float)
        return ex, at


def transmit_paths(model, transmit, points, elements):
    """Arrival time [s] of a transmit wave at points through the medium and
    the attenuation [Np] on the way, with time zero at the first firing;
    with model.interfaces also the interface loss [Np].

    The elements fire with delays computed for the nominal speed; the wave
    then follows straight rays (parallel for a plane wave, through the
    virtual source otherwise).
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    e = np.asarray(elements, dtype=float)
    cn = model.c_nominal
    if transmit.kind == "plane":
        q, t0 = transmit.entry(p, e, cn)
        x_range = (p[:, 0].min(), p[:, 0].max())
        mp = model.plane_map(transmit.angle, x_range, p[:, 2].max())
        vals = list(mp(p[:, 0], p[:, 2]))
        bad = ~np.isfinite(vals[0])
        if bad.any():
            fix = model.line(q[bad][:, [0, 2]], p[bad][:, [0, 2]])
            for v, fv in zip(vals, fix):
                v[bad] = fv
        vals[0] = t0 + vals[0]
        return tuple(vals)
    vs = transmit.source
    d_e = np.linalg.norm(e - vs, axis=1)
    if transmit.kind == "diverging":
        mp = model.polar_map(vs[[0, 2]], (0.0, 1.0), half_angle=np.pi / 2)
        vals = list(mp(p[:, 0], p[:, 2]))
        vals[0] = vals[0] - d_e.min() / cn
        return tuple(vals)
    # focused: all directions around the focus
    mp = model.polar_map(vs[[0, 2]], (0.0, -1.0), half_angle=np.pi)
    q, t0 = transmit.entry(p, e, cn)
    vq = mp(q[:, 0], q[:, 2])
    vp = mp(p[:, 0], p[:, 2])
    before = p[:, 2] < vs[2]
    out = [np.where(before, a - b, a + b) for a, b in zip(vq, vp)]
    out[0] = t0 + out[0]
    for i in range(1, len(out)):
        out[i] = np.maximum(out[i], 0.0)
    return tuple(out)

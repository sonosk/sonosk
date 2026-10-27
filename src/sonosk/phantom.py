"""Building blocks for numerical phantoms.

Shapes return boolean masks on a Grid. Label maps are integer arrays where
each value selects a Material (see sonosk.medium). Point scatterers for
imaging are drawn from a label map, with an amplitude set by the echo level
of each material.

The depth axis is the last grid axis.
"""

import numpy as np


def _coords(grid):
    m = grid.mesh()
    if grid.ndim == 1:
        return 0.0, 0.0, m[0]
    if grid.ndim == 2:
        return m[0], 0.0, m[1]
    return m[0], m[1], m[2]


def layers(grid, thickness, start=0.0):
    """Label map of planar layers stacked in depth.

    thickness is a list of layer thicknesses [m]. Points above start get
    label 0, the n-th layer gets label n, and everything below the last
    layer gets label len(thickness) + 1. Use paint() to change labels.
    """
    _, _, z = _coords(grid)
    edges = start + np.concatenate([[0.0], np.cumsum(thickness)])
    return np.searchsorted(edges, z, side="right").astype(int)


def halfspace(grid, depth):
    """Points deeper than depth."""
    _, _, z = _coords(grid)
    return z >= depth


def sphere(grid, center, radius):
    """Sphere in 3-D, disc in the x-z plane in 2-D, segment in 1-D.

    center is (x, y, z) [m]; y is ignored in 2-D and x, y in 1-D.
    """
    x, y, z = _coords(grid)
    cx, cy, cz = center
    return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2


def ellipsoid(grid, center, semi_axes, angle=0.0):
    """Ellipsoid with semi-axes (ax, ay, az), rotated by angle [rad] about y."""
    x, y, z = _coords(grid)
    cx, cy, cz = center
    ax, ay, az = semi_axes
    dx, dz = x - cx, z - cz
    c, s = np.cos(angle), np.sin(angle)
    u = c * dx + s * dz
    w = -s * dx + c * dz
    return (u / ax) ** 2 + ((y - cy) / ay) ** 2 + (w / az) ** 2 <= 1.0


def cylinder(grid, point, direction, radius):
    """Infinite cylinder (a vessel) through point along direction."""
    x, y, z = _coords(grid)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    rx, ry, rz = x - point[0], y - point[1], z - point[2]
    t = rx * d[0] + ry * d[1] + rz * d[2]
    q = (rx - t * d[0]) ** 2 + (ry - t * d[1]) ** 2 + (rz - t * d[2]) ** 2
    return q <= radius ** 2


def box(grid, low, high):
    """Axis-aligned box between corners low and high, given as (x, y, z)."""
    x, y, z = _coords(grid)
    ok = (z >= low[2]) & (z <= high[2])
    if grid.ndim >= 2:
        ok &= (x >= low[0]) & (x <= high[0])
    if grid.ndim == 3:
        ok &= (y >= low[1]) & (y <= high[1])
    return ok


def paint(labels, mask, label):
    """Copy of labels with label written where mask is true."""
    out = np.array(labels, dtype=int, copy=True)
    out[np.broadcast_to(mask, out.shape)] = label
    return out


def scatterers(medium, density, rng=None, region=None):
    """Random point scatterers for pulse-echo simulation.

    Parameters
    ----------
    medium : Medium
        Its grid sets the region; the echo level of each material sets the
        mean scattering amplitude, 10**(echo/20).
    density : float
        Scatterers per mm^2 for a 2-D grid, per mm^3 for a 3-D grid. Ten or
        more per resolution cell give fully developed speckle.
    rng : numpy Generator or int, optional
    region : bool array, optional
        Only place scatterers where this mask is true.

    Returns
    -------
    pos : (N, 3) array [m]
    amp : (N,) array
    """
    grid = medium.grid
    if grid.ndim == 1:
        raise ValueError("scatterers need a 2-D or 3-D grid")
    rng = np.random.default_rng(rng)
    lo = grid.origin
    hi = grid.origin + grid.extent
    size_mm = np.prod((hi - lo) * 1e3)
    n = rng.poisson(density * size_mm)
    u = lo + rng.random((n, grid.ndim)) * (hi - lo)

    idx = np.rint((u - grid.origin) / grid.spacing).astype(int)
    idx = np.clip(idx, 0, np.array(grid.shape) - 1)
    lab = medium.labels[tuple(idx.T)]
    keep = np.ones(n, bool) if region is None else np.asarray(region)[tuple(idx.T)]
    level = np.array([m.echo for m in medium.materials])[lab]
    amp = rng.normal(size=n) * 10.0 ** (level / 20.0)

    if grid.ndim == 2:
        pos = np.stack([u[:, 0], np.zeros(n), u[:, 1]], 1)
    else:
        pos = u
    keep &= amp != 0
    return pos[keep], amp[keep]


def point_targets(positions, amplitude=1.0):
    """Wire targets at the given (x, y, z) positions [m]."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    return pos, np.full(pos.shape[0], float(amplitude))


# --------------------------------------------------------------------------
# random structure


def random_field(grid, correlation, rng=None, axes=None):
    """Smooth Gaussian random field with zero mean and unit variance.

    correlation is the correlation length [m] (the standard deviation of
    the Gaussian filter), a float or one value per grid axis. axes limits
    the field to some axes (for example (0,) for a lateral profile that
    is constant in depth).
    """
    rng = np.random.default_rng(rng)
    shape = list(grid.shape)
    use = range(grid.ndim) if axes is None else axes
    for i in range(grid.ndim):
        if i not in use:
            shape[i] = 1
    noise = rng.normal(size=shape)
    corr = np.broadcast_to(np.asarray(correlation, dtype=float), (grid.ndim,))
    F = np.fft.fftn(noise)
    for i in range(grid.ndim):
        if shape[i] == 1:
            continue
        k = 2 * np.pi * np.fft.fftfreq(shape[i], grid.spacing[i])
        h = np.exp(-0.5 * (k * corr[i]) ** 2)
        sh = [1] * grid.ndim
        sh[i] = shape[i]
        F = F * h.reshape(sh)
    f = np.real(np.fft.ifftn(F))
    f = f - f.mean()
    sd = f.std()
    f = f / sd if sd > 0 else f
    return np.broadcast_to(f, grid.shape)


def wavy_below(grid, depth, amplitude, correlation, rng=None):
    """Points deeper than a randomly undulating surface.

    The surface lies at depth + amplitude * F(x, y), with F a smooth random
    profile of unit variance and correlation length correlation [m].
    """
    _, _, z = _coords(grid)
    if grid.ndim == 1:
        return z >= depth
    lateral = tuple(range(grid.ndim - 1))
    f = random_field(grid, correlation, rng, axes=lateral)
    return z >= depth + amplitude * f


def irregular(grid, center, radius, roughness=0.15, correlation=None, spicules=0,
              spicule_length=None, spicule_width=None, rng=None, aspect=(1.0, 1.0, 1.0)):
    """Mass with an irregular margin and optional spicules.

    The surface is at radius * (1 + roughness * F) along each direction,
    with F a smooth random function. aspect stretches the mass along
    (x, y, z), for example (1, 1, 1.3) for a taller-than-wide lesion.
    Spicules are thin tapering spikes from the surface outwards.
    """
    rng = np.random.default_rng(rng)
    x, y, z = _coords(grid)
    c = np.asarray(center, dtype=float)
    a = np.asarray(aspect, dtype=float)
    dx, dy, dz = (x - c[0]) / a[0], (y - c[1]) / a[1], (z - c[2]) / a[2]
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    corr = radius / 2 if correlation is None else correlation
    f = random_field(grid, corr, rng)
    # value of the random field on the reference sphere along each direction
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(r > 0, radius / r, 0.0)
    px = c[0] + dx * u * a[0]
    pz = c[2] + dz * u * a[2]
    idx = [np.clip(np.rint((px - grid.origin[0]) / grid.spacing[0]).astype(int), 0,
                   grid.shape[0] - 1)]
    if grid.ndim == 3:
        py = c[1] + dy * u * a[1]
        idx.append(np.clip(np.rint((py - grid.origin[1]) / grid.spacing[1]).astype(int), 0,
                           grid.shape[1] - 1))
    if grid.ndim >= 2:
        idx.append(np.clip(np.rint((pz - grid.origin[-1]) / grid.spacing[-1]).astype(int), 0,
                           grid.shape[-1] - 1))
    fs = f[tuple(idx)] if grid.ndim >= 2 else 0.0
    mask = r <= radius * (1 + roughness * fs)

    if spicules:
        L = 1.2 * radius if spicule_length is None else spicule_length
        w = 0.12 * radius if spicule_width is None else spicule_width
        for _ in range(int(spicules)):
            if grid.ndim == 3:
                d = rng.normal(size=3)
            else:
                ang = rng.uniform(0, 2 * np.pi)
                d = np.array([np.sin(ang), 0.0, np.cos(ang)])
            d = d / np.linalg.norm(d)
            length = L * rng.uniform(0.6, 1.0)
            p0 = c + d * radius * 0.8 * a
            rx, ry, rz = x - p0[0], y - p0[1], z - p0[2]
            t = rx * d[0] + ry * d[1] + rz * d[2]
            q = np.sqrt(np.maximum(rx ** 2 + ry ** 2 + rz ** 2 - t ** 2, 0))
            mask |= (t >= 0) & (t <= length) & (q <= w * (1 - t / length))
    return mask


def curve(grid, points, width):
    """Thin band of the given width [m] along a polyline of (x, y, z)
    points, for ligaments, septa or vessel walls in 2-D or 3-D."""
    x, y, z = _coords(grid)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    mask = np.zeros(grid.shape, bool)
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L2 = float(d @ d)
        if L2 == 0:
            continue
        t = ((x - a[0]) * d[0] + (y - a[1]) * d[1] + (z - a[2]) * d[2]) / L2
        t = np.clip(t, 0, 1)
        q2 = (x - a[0] - t * d[0]) ** 2 + (y - a[1] - t * d[1]) ** 2 + (z - a[2] - t * d[2]) ** 2
        mask |= q2 <= (0.5 * width) ** 2
    return mask


def interface_scatterers(medium, gain=30.0, smooth=None, threshold=2e-4, region=None):
    """Point reflectors that give the echoes of impedance interfaces.

    In the weak-contrast (Born) limit the plane-wave reflection of a
    layered medium is the depth derivative of ln(Z)/2, so a step from Z1 to
    Z2 reflects about (Z2 - Z1)/(Z2 + Z1). Each grid point gets a
    reflector of amplitude gain * d(ln Z)/2 over one grid step in depth,
    where Z is smoothed over smooth [m] (default one grid step) to soften
    the staircase of the label map. Points where d(ln Z)/2 is below
    threshold are left out. The reflectors are dense (one per grid point),
    so their echoes add coherently into specular echoes; their level
    relative to the diffuse scatterers of phantom.scatterers is set by
    gain, which is a modelling choice. The default 30 was set by eye on the
    breast example (0.1 mm grid, 50 scatterers per mm^2, 7.5 MHz): skin and
    the fascia of the pectoralis stand out from the speckle, the fat and
    glandular boundaries stay faint. The speckle grows with the square root
    of the scatterer density, so gain should follow it. Only the depth
    derivative is used, which suits beams close to the depth axis.

    Returns pos (N, 3) [m] and amp (N,), to be concatenated with the
    output of scatterers().
    """
    grid = medium.grid
    if grid.ndim == 1:
        raise ValueError("interface scatterers need a 2-D or 3-D grid")
    lnz = np.log(np.asarray(medium.rho, dtype=float) * np.asarray(medium.c, dtype=float))
    sigma = float(np.min(grid.spacing)) if smooth is None else float(smooth)
    if sigma > 0:
        # pad with edge values so that the periodic transform does not join
        # the top and the bottom of the grid into a false interface
        npad = [int(np.ceil(4 * sigma / h)) + 1 for h in grid.spacing]
        big = np.pad(lnz, [(n, n) for n in npad], mode="edge")
        F = np.fft.fftn(big)
        for ax in range(grid.ndim):
            k = 2 * np.pi * np.fft.fftfreq(big.shape[ax], grid.spacing[ax])
            sh = [1] * grid.ndim
            sh[ax] = k.size
            F = F * np.exp(-0.5 * (sigma * k) ** 2).reshape(sh)
        lnz = np.real(np.fft.ifftn(F))[tuple(slice(n, n + m) for n, m in zip(npad, grid.shape))]
    d = np.zeros(grid.shape)
    d[..., :-1] = 0.5 * np.diff(lnz, axis=-1)
    amp = gain * d
    keep = np.abs(d) > threshold
    if region is not None:
        keep &= np.asarray(region, bool)
    idx = np.argwhere(keep)
    pts = grid.origin + idx * grid.spacing
    pts[:, -1] += 0.5 * grid.spacing[-1]
    if grid.ndim == 2:
        pos = np.stack([pts[:, 0], np.zeros(len(pts)), pts[:, 1]], 1)
    else:
        pos = pts
    return pos, amp[keep]


def fluctuations(medium, level=0.02, rng=None, correlation=None, max_level=0.15, speed=0.5):
    """Copy of a medium with random sub-resolution variations of density
    and sound speed, which act as scatterers in the full-wave solver.

    The relative standard deviation is level times 10**(echo/20) of each
    material, so the echo levels of the materials carry over to the
    full-wave images (backscatter grows with the square of the variation).
    correlation [m] smooths the variations; by default they are
    independent at every grid point. The relative variation is capped at
    max_level. The sound speed varies by speed times the density
    variation; speed=0 keeps the speed map, which avoids the random phase
    errors that speed variations add along the path.
    """
    from .medium import Medium
    rng = np.random.default_rng(rng)
    g = medium.grid
    if correlation is None:
        n1, n2 = rng.normal(size=g.shape), rng.normal(size=g.shape)
    else:
        n1 = np.asarray(random_field(g, correlation, rng))
        n2 = np.asarray(random_field(g, correlation, rng))
    lev = np.array([0.0 if np.isinf(m.echo) and m.echo < 0 else 10 ** (m.echo / 20)
                    for m in medium.materials])[medium.labels] * level
    lev = np.minimum(lev, max_level)
    out = Medium(g, medium.labels, medium.materials)
    out.rho = medium.rho * (1 + lev * n1)
    out.c = medium.c * (1 + speed * lev * n2)
    return out

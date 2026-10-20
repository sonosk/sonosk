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

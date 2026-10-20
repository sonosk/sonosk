"""Regular computational grids in 1, 2 and 3 dimensions.

Units are SI (metres). The last axis is depth (z), so a 2-D grid is
(x, z) and a 3-D grid is (x, y, z).
"""

import numpy as np


class Grid:
    """Regular grid.

    Parameters
    ----------
    shape : int or tuple of int
        Number of points along each axis.
    spacing : float or sequence of float
        Point spacing along each axis [m].
    origin : float or sequence of float, optional
        Coordinate of the first point on each axis [m]. Default 0.
    """

    def __init__(self, shape, spacing, origin=0.0):
        shape = tuple(int(n) for n in np.atleast_1d(shape))
        nd = len(shape)
        if nd < 1 or nd > 3:
            raise ValueError("grid must have 1, 2 or 3 dimensions")
        if min(shape) < 1:
            raise ValueError("shape must be positive")

        spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (nd,)).copy()
        if np.any(spacing <= 0):
            raise ValueError("spacing must be positive")
        origin = np.broadcast_to(np.asarray(origin, dtype=float), (nd,)).copy()

        self.shape = shape
        self.spacing = spacing
        self.origin = origin

    @classmethod
    def centered(cls, shape, spacing, depth_start=0.0):
        """Grid centred on the depth axis; depth starts at depth_start."""
        g = cls(shape, spacing)
        n = np.array(g.shape)
        origin = -0.5 * g.spacing * (n - 1)
        origin[-1] = depth_start
        g.origin = origin
        return g

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def extent(self):
        """Length covered along each axis [m]."""
        return self.spacing * (np.array(self.shape) - 1)

    def axis(self, i):
        return self.origin[i] + self.spacing[i] * np.arange(self.shape[i])

    def axes(self):
        return [self.axis(i) for i in range(self.ndim)]

    def mesh(self):
        return np.meshgrid(*self.axes(), indexing="ij")

    def points(self):
        """All grid points as an (N, ndim) array."""
        return np.stack([m.ravel() for m in self.mesh()], axis=1)

    def points_per_wavelength(self, c, f):
        """Smallest number of points per wavelength for speed c and frequency f."""
        return float(np.min(c) / f / self.spacing.max())

    def __repr__(self):
        sp = ", ".join(f"{s * 1e3:g}" for s in self.spacing)
        return f"Grid(shape={self.shape}, spacing=({sp}) mm)"

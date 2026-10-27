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

    def index(self, point, strict=False):
        """Nearest grid index of a point given with one coordinate per axis.

        Points outside the grid are moved to the nearest edge; with
        strict=True they raise ValueError instead.
        """
        p = np.asarray(point, dtype=float)
        if p.shape != (self.ndim,):
            raise ValueError(f"point must have {self.ndim} coordinates")
        f = np.rint((p - self.origin) / self.spacing)
        hi = np.array(self.shape) - 1
        if strict and (np.any(f < 0) or np.any(f > hi)):
            raise ValueError("point outside the grid")
        return tuple(int(i) for i in np.clip(f, 0, hi))

    def crop(self, low, high, strict=False):
        """Sub-grid covering low..high (one coordinate per axis) [m] and the
        slices that cut it from this grid.

        A box that reaches outside the grid is cut at the grid edges; with
        strict=True it raises ValueError instead.
        """
        lo = np.asarray(low, dtype=float)
        hi = np.asarray(high, dtype=float)
        if lo.shape != (self.ndim,) or hi.shape != (self.ndim,):
            raise ValueError(f"low and high must have {self.ndim} coordinates")
        if np.any(hi < lo):
            raise ValueError("high must not be below low")
        a = np.floor((lo - self.origin) / self.spacing).astype(int)
        b = np.ceil((hi - self.origin) / self.spacing).astype(int)
        top = np.array(self.shape) - 1
        if strict and (np.any(a < 0) or np.any(b > top)):
            raise ValueError("box reaches outside the grid")
        i0 = np.clip(a, 0, top)
        i1 = np.clip(b, 0, top)
        sl = tuple(slice(a, b + 1) for a, b in zip(i0, i1))
        sub = Grid(tuple(b - a + 1 for a, b in zip(i0, i1)), self.spacing,
                   self.origin + i0 * self.spacing)
        return sub, sl

    def points_per_wavelength(self, c, f):
        """Smallest number of points per wavelength for speed c and frequency f."""
        return float(np.min(c) / f / self.spacing.max())

    def __repr__(self):
        sp = ", ".join(f"{s * 1e3:g}" for s in self.spacing)
        return f"Grid(shape={self.shape}, spacing=({sp}) mm)"

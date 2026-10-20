"""Materials and property maps.

Attenuation follows a power law, alpha(f) = alpha0 f^y, with alpha0 in the
clinical unit dB/(cm MHz^y). Inside the solvers it is converted to an
amplitude coefficient in Np/m.
"""

from dataclasses import dataclass, replace

import numpy as np

DB_TO_NP = np.log(10.0) / 20.0   # 1 dB = 0.1151 Np


def attenuation_np(alpha0, f, y=1.0):
    """Amplitude attenuation [Np/m] at frequency f [Hz].

    alpha0 is in dB/(cm MHz^y).
    """
    return np.asarray(alpha0) * (f / 1e6) ** np.asarray(y) * 100.0 * DB_TO_NP


@dataclass(frozen=True)
class Material:
    """Acoustic and thermal properties of one material.

    c [m/s], rho [kg/m^3], alpha0 [dB/(cm MHz^y)], y power-law exponent,
    ba nonlinearity B/A, k thermal conductivity [W/(m K)], cp specific heat
    [J/(kg K)], perfusion blood mass flow per unit volume [kg/(m^3 s)],
    echo backscatter level relative to a reference tissue [dB], source a
    reference for the values.
    """

    name: str
    c: float
    rho: float
    alpha0: float = 0.0
    y: float = 1.0
    ba: float = 0.0
    k: float = 0.0
    cp: float = 0.0
    perfusion: float = 0.0
    echo: float = 0.0
    source: str = ""

    @property
    def impedance(self):
        return self.rho * self.c

    def attenuation(self, f):
        """Amplitude attenuation [Np/m] at f [Hz]."""
        return float(attenuation_np(self.alpha0, f, self.y))

    def with_(self, **kw):
        """Copy with some properties changed."""
        return replace(self, **kw)


_FIELDS = ("c", "rho", "alpha0", "y", "ba", "k", "cp", "perfusion", "echo")


class Medium:
    """Property maps on a grid, built from a label map.

    Parameters
    ----------
    grid : Grid
    labels : int array with grid.shape
        Material index of every grid point.
    materials : sequence of Material
        materials[i] is used where labels == i.
    """

    def __init__(self, grid, labels, materials):
        labels = np.asarray(labels)
        if labels.shape != grid.shape:
            raise ValueError("labels must have the grid shape")
        if labels.min() < 0 or labels.max() >= len(materials):
            raise ValueError("label without a material")
        self.grid = grid
        self.labels = labels.astype(int)
        self.materials = list(materials)
        for name in _FIELDS:
            table = np.array([getattr(m, name) for m in self.materials], dtype=float)
            setattr(self, name, table[self.labels])

    @classmethod
    def homogeneous(cls, grid, material):
        return cls(grid, np.zeros(grid.shape, int), [material])

    def __repr__(self):
        names = ", ".join(m.name for m in self.materials)
        return f"Medium({self.grid!r}, materials=[{names}])"

    def mask(self, name):
        """Boolean map of the points made of the named material."""
        idx = [i for i, m in enumerate(self.materials) if m.name == name]
        if not idx:
            raise KeyError(name)
        return np.isin(self.labels, idx)

    def attenuation(self, f):
        """Map of amplitude attenuation [Np/m] at f [Hz]."""
        return attenuation_np(self.alpha0, f, self.y)

    @property
    def impedance(self):
        return self.rho * self.c

    def volume(self, name=None):
        """Volume (3-D), area (2-D) or length (1-D) of a material, or of all."""
        cell = float(np.prod(self.grid.spacing))
        if name is None:
            return self.grid.size * cell
        return float(self.mask(name).sum()) * cell

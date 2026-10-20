"""One-dimensional model of a thickness-mode piezoelectric transducer.

The piezoelectric plate is described by the Mason equivalent circuit.
Backing and load are semi-infinite media given by their acoustic
impedance. Matching layers and a lens are transmission lines. The model
gives the electrical input impedance, the transmit response and the shape
of the pulse-echo response of a stack, which is the first step in
designing a probe element.

Impedances of media are specific acoustic impedances in Rayl
(kg m^-2 s^-1). Forces and velocities on the plate are per unit area.

References
----------
Mason W.P. Electromechanical Transducers and Wave Filters, 2nd ed.
Van Nostrand, 1948.

Kino G.S. Acoustic Waves: Devices, Imaging, and Analog Signal Processing.
Prentice-Hall, 1987.

Desilets C.S., Fraser J.D., Kino G.S. The design of efficient broad-band
piezoelectric transducers. IEEE Trans. Sonics Ultrason. 25(3):115-125, 1978.
"""

from dataclasses import dataclass, field

import numpy as np

from .medium import attenuation_np

EPS0 = 8.8541878128e-12


@dataclass
class Layer:
    """Passive layer: thickness [m], sound speed [m/s], density [kg/m^3],
    attenuation alpha0 [dB/(cm MHz^y)] and exponent y."""

    thickness: float
    c: float
    rho: float
    alpha0: float = 0.0
    y: float = 1.0

    @property
    def impedance(self):
        return self.rho * self.c


@dataclass
class Piezo:
    """Piezoelectric plate in thickness mode.

    thickness [m], c stiffened longitudinal sound speed [m/s], rho
    [kg/m^3], kt thickness coupling factor, eps_r clamped relative
    permittivity eps33^S / eps0, area electrode area [m^2].
    """

    thickness: float
    c: float
    rho: float
    kt: float
    eps_r: float
    area: float

    @property
    def impedance(self):
        return self.rho * self.c

    @property
    def c0(self):
        """Clamped capacitance [F]."""
        return self.eps_r * EPS0 * self.area / self.thickness

    @property
    def f_half_wave(self):
        """Frequency at which the plate is half a wavelength thick [Hz]."""
        return self.c / (2 * self.thickness)


@dataclass
class Stack:
    """Piezo plate with backing, front layers (from the plate outwards) and load."""

    piezo: Piezo
    backing: float = 400.0          # Rayl, air by default
    layers: list = field(default_factory=list)
    load: float = 1.5e6             # Rayl, water

    def _front(self, f):
        """Impedance seen by the plate front face, and the layer matrices."""
        z = self.load + 0j
        mats = []
        for lay in reversed(self.layers):
            w = 2 * np.pi * f
            kl = w / lay.c - 1j * attenuation_np(lay.alpha0, f, lay.y)
            zl = lay.impedance
            ct, st = np.cos(kl * lay.thickness), np.sin(kl * lay.thickness)
            z = zl * (z * ct + 1j * zl * st) / (zl * ct + 1j * z * st)
            mats.append((ct, st, zl))
        return z, mats[::-1]

    def _solve(self, f):
        """Face velocities and voltage for a unit current, all per unit area
        on the acoustic side."""
        p = self.piezo
        f = np.atleast_1d(np.asarray(f, dtype=float))
        w = 2 * np.pi * f
        bd = w * p.thickness / p.c
        z0 = p.impedance
        h = p.kt * np.sqrt(p.rho * p.c ** 2 / (p.eps_r * EPS0))
        A = p.area

        out = []
        for i in range(f.size):
            zf, mats = self._front(f[i])
            z11 = z0 / (1j * np.tan(bd[i]))
            z12 = z0 / (1j * np.sin(bd[i]))
            src = h / (1j * w[i]) / A        # force per area per unit current
            # face 1 is the back, face 2 the front; v inward, F = -Z v
            m = np.array([[z11 + self.backing, z12], [z12, z11 + zf]])
            v1, v2 = np.linalg.solve(m, -np.array([src, src]))
            V = h / (1j * w[i]) * (v1 + v2) + 1 / (1j * w[i] * p.c0)
            out.append((v1, v2, V, zf, mats))
        return out

    def impedance(self, f):
        """Electrical input impedance [ohm]."""
        return np.array([V for _, _, V, _, _ in self._solve(f)])

    def transmit(self, f, r_source=50.0):
        """Pressure in the load per volt of the source [Pa/V].

        The source has internal resistance r_source [ohm].
        """
        res = []
        for v1, v2, V, zf, mats in self._solve(f):
            u = -v2                  # outward velocity at the front face
            F = zf * u               # pressure entering the first layer
            for ct, st, zl in mats:
                F, u = F * ct - 1j * zl * st * u, u * ct - 1j * st / zl * F
            i = 1.0 / (V + r_source)  # current for one volt
            res.append(F * i)
        return np.array(res)

    def pulse_echo(self, f, r_source=50.0):
        """Shape of the two-way response, transmit squared, normalised to 1."""
        h = self.transmit(f, r_source) ** 2
        return h / np.abs(h).max()

    def power_balance(self, f):
        """Electrical input power and acoustic power into backing and front,
        for a unit current (all in W). They agree for lossless layers."""
        rows = []
        A = self.piezo.area
        for v1, v2, V, zf, _ in self._solve(f):
            p_el = 0.5 * V.real
            p_b = 0.5 * self.backing * abs(v1) ** 2 * A
            p_f = 0.5 * zf.real * abs(v2) ** 2 * A
            rows.append((p_el, p_b, p_f))
        return np.array(rows)


def matching_impedance(z_piezo, z_load, n_layers=1):
    """Matching layer impedances after Desilets et al. (1978)."""
    if n_layers == 1:
        return (z_piezo * z_load ** 2) ** (1 / 3)
    if n_layers == 2:
        return (z_piezo ** (4 / 7) * z_load ** (3 / 7), z_piezo ** (1 / 7) * z_load ** (6 / 7))
    raise ValueError("one or two layers")


def quarter_wave(c, f0):
    """Thickness of a quarter-wave layer [m]."""
    return c / (4 * f0)


def bandwidth(f, response, level_db=-6.0):
    """Centre frequency [Hz] and fractional bandwidth of a response."""
    a = np.abs(np.asarray(response))
    f = np.asarray(f, dtype=float)
    k = int(np.argmax(a))
    thr = a[k] * 10 ** (level_db / 20)
    i = k
    while i > 0 and a[i] > thr:
        i -= 1
    j = k
    while j < a.size - 1 and a[j] > thr:
        j += 1
    lo = np.interp(thr, [a[i], a[i + 1]], [f[i], f[i + 1]])
    hi = np.interp(thr, [a[j], a[j - 1]], [f[j], f[j - 1]])
    fc = 0.5 * (lo + hi)
    return fc, (hi - lo) / fc

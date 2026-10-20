import numpy as np

from sonosk import stack

PLATE = stack.Piezo(thickness=0.4e-3, c=4560.0, rho=7500.0, kt=0.5, eps_r=1470.0,
                    area=1e-4)
F = np.linspace(1e6, 10e6, 1500)


def test_unloaded_plate_closed_form():
    s = stack.Stack(PLATE, backing=0.0, load=0.0)
    z = s.impedance(F)
    bd = 2 * np.pi * F * PLATE.thickness / PLATE.c
    zc = 1 / (1j * 2 * np.pi * F * PLATE.c0) * (1 - PLATE.kt ** 2 * np.tan(bd / 2) / (bd / 2))
    assert np.max(np.abs(z - zc) / np.abs(zc)) < 1e-6


def test_power_balance_and_passivity():
    lay = stack.Layer(0.15e-3, 2500.0, 1800.0)
    s = stack.Stack(PLATE, backing=5e6, layers=[lay], load=1.5e6)
    pb = s.power_balance(F)
    assert np.allclose(pb[:, 0], pb[:, 1] + pb[:, 2], rtol=1e-9)
    assert np.all(s.impedance(F).real > 0)


def test_matching_layer_widens_band():
    plain = stack.Stack(PLATE, backing=3e6, load=1.5e6)
    zm = stack.matching_impedance(PLATE.impedance, 1.5e6)
    cm = 2500.0
    lay = stack.Layer(stack.quarter_wave(cm, PLATE.f_half_wave), cm, zm / cm)
    matched = stack.Stack(PLATE, backing=3e6, layers=[lay], load=1.5e6)
    _, bw0 = stack.bandwidth(F, plain.pulse_echo(F))
    _, bw1 = stack.bandwidth(F, matched.pulse_echo(F))
    assert bw1 > 2 * bw0


def test_desilets_values():
    zp, zl = 34e6, 1.5e6
    assert np.isclose(stack.matching_impedance(zp, zl), (zp * zl ** 2) ** (1 / 3))
    z1, z2 = stack.matching_impedance(zp, zl, 2)
    assert zp > z1 > z2 > zl

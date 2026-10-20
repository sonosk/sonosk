import numpy as np
import pytest

from sonosk import safety


def test_derate():
    # 3 MHz at 2 cm: 1.8 dB loss in pressure, 1.8 dB in intensity too
    assert np.isclose(safety.derate(1.0, 3.0, 2.0), 10 ** (-1.8 / 20))
    assert np.isclose(safety.derate_intensity(1.0, 3.0, 2.0), 10 ** (-1.8 / 10))


def test_pulse_duration_tone_burst():
    # rectangular-envelope burst: 10-90 % of energy spans 0.8 of its length
    fs = 200e6
    n = 400
    t = np.arange(n) / fs
    p = np.sin(2 * np.pi * 5e6 * t)
    pd = safety.pulse_duration(p, 1 / fs)
    assert np.isclose(pd, 1.25 * 0.8 * n / fs, rtol=0.02)


def test_pulse_duration_with_silence():
    # zeros before and after the pulse must not change the result
    fs = 200e6
    t = np.arange(400) / fs
    p = np.sin(2 * np.pi * 5e6 * t)
    padded = np.concatenate([np.zeros(300), p, np.zeros(500)])
    assert np.isclose(safety.pulse_duration(padded, 1 / fs),
                      safety.pulse_duration(p, 1 / fs), rtol=1e-6)
    assert safety.pulse_duration(np.zeros(10), 1 / fs) == 0.0


def test_pii_and_isppa():
    fs = 100e6
    p = np.full(100, 1e6)  # 1 MPa for 1 us
    pii = safety.pulse_intensity_integral(p, 1 / fs, rho=1000.0, c=1500.0)
    assert np.isclose(pii, 1e12 * 1e-6 / 1.5e6)
    assert np.isclose(safety.isppa(p, 1 / fs), pii / safety.pulse_duration(p, 1 / fs))
    assert np.isclose(safety.ispta(pii, 1000.0), pii * 1000.0)


def test_mechanical_index():
    z = np.linspace(0.5, 10.0, 200)
    pr = 2.0 * np.exp(-((z - 4.0) / 1.0) ** 2)
    mi, z_sp = safety.mechanical_index(z, pr, 4.0)
    k = np.argmax(safety.derate_intensity(pr ** 2, 4.0, z))
    assert np.isclose(z_sp, z[k])
    assert np.isclose(mi, safety.derate(pr[k], 4.0, z[k]) / 2.0)
    assert z_sp < 4.0  # derating moves the point towards the probe


def test_limits_general():
    r = safety.check_limits(mi=1.2, ispta3=300.0)
    assert r["ok"]
    r = safety.check_limits(mi=2.5, ispta3=300.0)
    assert not r["ok"]
    # MI too high, but ISPPA.3 within limit: allowed by Track 3
    r = safety.check_limits(mi=2.5, isppa3=150.0, ispta3=300.0)
    assert r["ok"]


def test_limits_ophthalmic():
    r = safety.check_limits(mi=0.2, ispta3=40.0, ti=0.8, application="ophthalmic")
    assert r["ok"]
    r = safety.check_limits(mi=0.3, ispta3=40.0, application="ophthalmic")
    assert not r["ok"]
    with pytest.raises(ValueError):
        safety.check_limits(mi=1.0, application="cardiac")

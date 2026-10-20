import numpy as np
import pytest

from sonosk import water


def test_sound_speed():
    # values from Marczak (1997) fit
    assert np.isclose(water.sound_speed(20.0), 1482.38, atol=0.05)
    assert np.isclose(water.sound_speed(37.0), 1523.6, atol=0.2)


def test_density():
    # Kell (1975)
    assert np.isclose(water.density(4.0), 999.97, atol=0.01)
    assert np.isclose(water.density(20.0), 998.21, atol=0.01)


def test_array_input():
    t = np.array([10.0, 20.0, 30.0])
    c = water.sound_speed(t)
    assert c.shape == (3,)
    assert np.all(np.diff(c) > 0)


def test_range():
    with pytest.raises(ValueError):
        water.sound_speed(100.0)
    with pytest.raises(ValueError):
        water.density(-1.0)

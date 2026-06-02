import numpy as np
import pytest

from app.simulation.vegetation.itu_p833 import excess_loss_db


# Golden values independently computed for 1.8 GHz in-leaf:
#   R = 0.20 * 1.8^0.30 ≈ 0.2385 dB/m; A_max = 40 dB


def test_zero_depth_in_leaf():
    assert excess_loss_db(0.0, 1.8e9, "in_leaf") == pytest.approx(0.0)


def test_zero_depth_out_of_leaf():
    assert excess_loss_db(0.0, 1.8e9, "out_of_leaf") == pytest.approx(0.0)


def test_in_leaf_1800mhz_10m():
    result = excess_loss_db(10.0, 1.8e9, "in_leaf")
    assert result == pytest.approx(2.316, abs=0.01)


def test_in_leaf_1800mhz_100m():
    result = excess_loss_db(100.0, 1.8e9, "in_leaf")
    assert result == pytest.approx(17.969, abs=0.01)


def test_out_of_leaf_1800mhz_10m():
    # R = 0.16 * 1.8^0.18 ≈ 0.1779 dB/m; A_max = 25 dB
    result = excess_loss_db(10.0, 1.8e9, "out_of_leaf")
    assert result == pytest.approx(1.716, abs=0.01)


def test_saturation_approaches_amax_in_leaf():
    result = excess_loss_db(1e6, 1.8e9, "in_leaf")
    assert result > 39.9


def test_saturation_approaches_amax_out_of_leaf():
    result = excess_loss_db(1e6, 1.8e9, "out_of_leaf")
    assert result > 24.9


def test_frequency_above_limit_raises():
    with pytest.raises(ValueError, match="7.125 GHz"):
        excess_loss_db(10.0, 8e9)


def test_invalid_leaf_state_raises():
    with pytest.raises(ValueError, match="leaf_state"):
        excess_loss_db(10.0, 1.8e9, "summer")


def test_numpy_array_input_preserves_shape():
    depths = np.array([0.0, 10.0, 100.0])
    result = excess_loss_db(depths, 1.8e9, "in_leaf")
    assert result.shape == (3,)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(2.316, abs=0.01)
    assert result[2] == pytest.approx(17.969, abs=0.01)

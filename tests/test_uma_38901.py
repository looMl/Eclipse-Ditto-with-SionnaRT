"""Unit tests for the 3GPP TR 38.901 UMa closed-form RSRP baseline."""

import numpy as np
import pytest

from app.simulation.baselines.uma_38901 import (
    _d_breakpoint,
    _tr38901_element_gain_db,
    antenna_gain_dbi,
    los_probability_uma,
    pl_los_uma,
    pl_nlos_uma,
    predict_rsrp_uma,
)


def test_los_pathloss_short_range_golden():
    # d_2d = 50 m is below the breakpoint, so the short-range LOS formula applies:
    # 28 + 22*log10(d_3d) + 20*log10(f_ghz)
    pl = pl_los_uma(d_2d=50.0, d_3d=50.0, h_bs=25.0, h_ut=1.5, f_ghz=1.8)
    assert pl == pytest.approx(70.483, abs=0.01)


def test_los_pathloss_increases_with_distance():
    near = pl_los_uma(50.0, 50.0, 25.0, 1.5, 1.8)
    far = pl_los_uma(500.0, 500.0, 25.0, 1.5, 1.8)
    assert far > near


def test_nlos_never_below_los():
    args = (200.0, 201.0, 25.0, 1.5, 1.8)
    assert pl_nlos_uma(*args) >= pl_los_uma(*args)


def test_los_probability_within_18m_is_one():
    assert los_probability_uma(10.0, 1.5) == 1.0


def test_los_probability_decreases_with_distance():
    near = los_probability_uma(200.0, 1.5)
    far = los_probability_uma(2000.0, 1.5)
    assert 0.0 < far < near < 1.0


def test_breakpoint_scales_linearly_with_frequency():
    low = _d_breakpoint(25.0, 1.5, 1.8e9)
    high = _d_breakpoint(25.0, 1.5, 3.6e9)
    assert high == pytest.approx(2.0 * low)


def test_element_gain_peaks_at_boresight():
    boresight = _tr38901_element_gain_db(90.0, 0.0, max_gain_dbi=8.0)
    off_axis = _tr38901_element_gain_db(90.0, 60.0, max_gain_dbi=8.0)
    assert boresight == pytest.approx(8.0)
    assert off_axis < boresight


def test_antenna_gain_degenerate_direction_uses_array_gain():
    gain = antenna_gain_dbi(
        np.array([0.0, 0.0, 0.0]), (0.0, 0.0, 0.0), num_rows=4, num_cols=1
    )
    assert gain == pytest.approx(8.0 + 10.0 * np.log10(4))


def test_predict_rsrp_structure_and_sanity():
    out = predict_rsrp_uma(
        tx_position=np.array([0.0, 0.0, 25.0]),
        tx_orientation_rad=(0.0, 0.0, 0.0),
        tx_power_dbm=46.0,
        rx_position=np.array([100.0, 0.0, 1.5]),
        f_ghz=1.8,
    )
    expected_keys = {
        "rsrp_dbm",
        "pl_los_db",
        "pl_nlos_db",
        "p_los",
        "pl_effective_db",
        "tx_gain_dbi",
        "d_2d_m",
        "d_3d_m",
    }
    assert expected_keys <= set(out)
    assert out["d_2d_m"] == pytest.approx(100.0, abs=1.0)
    assert 0.0 <= out["p_los"] <= 1.0
    assert out["rsrp_dbm"] < 46.0  # path loss reduces transmit power

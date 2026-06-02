"""Unit tests for GPR-analysis statistics and projection helpers."""

import json

import numpy as np
import pytest

from app.services.gpr_analysis import _load_records, _stats, _to_metres


def test_stats_zero_bias_unit_error():
    out = _stats(np.array([1.0, -1.0]))
    assert out["bias"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(1.0)
    assert out["crmse"] == pytest.approx(1.0)
    assert out["mae"] == pytest.approx(1.0)


def test_stats_constant_error_has_zero_centered_rmse():
    out = _stats(np.array([2.0, 2.0]))
    assert out["bias"] == pytest.approx(2.0)
    assert out["rmse"] == pytest.approx(2.0)
    assert out["crmse"] == pytest.approx(0.0)


def test_to_metres_centres_on_mean():
    lons = np.array([10.0, 10.0])
    lats = np.array([44.0, 46.0])
    xy = _to_metres(lons, lats)
    assert xy.shape == (2, 2)
    np.testing.assert_allclose(xy[:, 0], [0.0, 0.0], atol=1e-6)  # same lon -> x == 0
    assert xy[0, 1] == pytest.approx(-xy[1, 1])  # symmetric about mean latitude


def test_load_records_filters_errors_and_nulls(tmp_path):
    data = {
        "valid": {"best_rt_error_db": 1.5},
        "errored": {"error": "no signal", "best_rt_error_db": 2.0},
        "missing": {"best_rt_error_db": None},
    }
    path = tmp_path / "records.json"
    path.write_text(json.dumps(data))
    records = _load_records(path)
    assert len(records) == 1
    assert records[0]["best_rt_error_db"] == 1.5

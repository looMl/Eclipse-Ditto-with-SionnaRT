"""Unit tests for DemProcessor coordinate and elevation helpers."""

import numpy as np
import pytest
from rasterio.transform import from_bounds

from app.geomap_processor.processors.dem_processor import DemProcessor


def test_utm_crs_northern_hemisphere():
    # lon 11, lat 45 (northern Italy) -> UTM zone 32N -> EPSG:32632
    assert DemProcessor._get_utm_crs(11.0, 45.0) == "EPSG:32632"


def test_utm_crs_southern_hemisphere():
    assert DemProcessor._get_utm_crs(11.0, -45.0) == "EPSG:32732"


def test_local_global_roundtrip():
    origin_lon, origin_lat = 11.0, 45.0
    x, y = DemProcessor.global_to_local(11.01, 45.01, origin_lon, origin_lat)
    lon, lat = DemProcessor.local_to_global(x, y, origin_lon, origin_lat)
    assert lon == pytest.approx(11.01, abs=1e-5)
    assert lat == pytest.approx(45.01, abs=1e-5)


def test_global_to_local_origin_is_zero():
    x, y = DemProcessor.global_to_local(11.0, 45.0, 11.0, 45.0)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)


def _grid():
    elevation = np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32")
    transform = from_bounds(0.0, 0.0, 2.0, 2.0, 2, 2)
    return elevation, transform


def test_sample_elevation_reads_correct_cell():
    elevation, transform = _grid()
    assert DemProcessor.sample_elevation(elevation, transform, 0.5, 1.5) == 10.0
    assert DemProcessor.sample_elevation(elevation, transform, 1.5, 0.5) == 40.0


def test_sample_elevation_applies_normalize_value():
    elevation, transform = _grid()
    assert DemProcessor.sample_elevation(elevation, transform, 0.5, 1.5, 4.0) == 6.0


def test_sample_elevation_out_of_bounds_returns_zero():
    elevation, transform = _grid()
    assert DemProcessor.sample_elevation(elevation, transform, 50.0, 50.0) == 0.0


def test_normalize_elevation_without_origin_uses_minimum():
    elevation, transform = _grid()
    flat, ref = DemProcessor._normalize_elevation(elevation, transform, None)
    assert ref == 10.0
    np.testing.assert_array_equal(flat, [0.0, 10.0, 20.0, 30.0])


def test_normalize_elevation_with_origin_uses_that_cell():
    elevation, transform = _grid()
    _, ref = DemProcessor._normalize_elevation(elevation, transform, (1.5, 0.5))
    assert ref == 40.0

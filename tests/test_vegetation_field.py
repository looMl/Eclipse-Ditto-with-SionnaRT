"""Unit tests for VegetationField bilinear sampling and npz round-trip."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.geomap_processor.utils.vegetation_field import VegetationField, _bilinear


def test_bilinear_at_integer_nodes():
    arr = np.array([[0.0, 10.0], [20.0, 30.0]], dtype="float32")
    assert _bilinear(arr, 0, 0) == pytest.approx(0.0)
    assert _bilinear(arr, 0, 1) == pytest.approx(10.0)
    assert _bilinear(arr, 1, 0) == pytest.approx(20.0)


def test_bilinear_at_cell_centre_averages_corners():
    arr = np.array([[0.0, 10.0], [20.0, 30.0]], dtype="float32")
    assert _bilinear(arr, 0.5, 0.5) == pytest.approx(15.0)
    assert _bilinear(arr, 0.0, 0.5) == pytest.approx(5.0)


def test_save_load_roundtrip(tmp_path):
    field = VegetationField(
        tcd=np.array([[0.0, 0.5], [1.0, 0.25]], dtype="float32"),
        chm=np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
        transform=Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
        crs=CRS.from_epsg(4326),
        origin_lon=11.0,
        origin_lat=45.0,
    )
    path = tmp_path / "veg.npz"
    field.save(path)
    loaded = VegetationField.load(path)

    np.testing.assert_array_equal(loaded.tcd, field.tcd)
    np.testing.assert_array_equal(loaded.chm, field.chm)
    assert loaded.origin_lon == pytest.approx(11.0)
    assert loaded.origin_lat == pytest.approx(45.0)
    assert loaded.crs.to_epsg() == 4326
    assert loaded.transform.a == pytest.approx(1.0)

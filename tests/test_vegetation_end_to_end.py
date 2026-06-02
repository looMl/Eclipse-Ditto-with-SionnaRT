"""
End-to-end tests for the vegetation attenuation pipeline.

Exercises the data chain: synthetic VegetationField → PathDepthIntegrator →
excess_loss_db, without requiring Sionna or Mitsuba.
Includes a VegetationField npz round-trip test.
"""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.geomap_processor.processors.dem_processor import DemProcessor
from app.geomap_processor.utils.vegetation_field import VegetationField
from app.simulation.itu_p833 import excess_loss_db
from app.simulation.vegetation_path_integrator import PathDepthIntegrator


_ORIGIN_LON = 11.0
_ORIGIN_LAT = 45.0
_CELL_M = 1.0
_GRID = 400  # ±200 m scene extent


# --------------------------------------------------------------------------- #
# Shared helper (mirrors test_vegetation_path_integrator.py)
# --------------------------------------------------------------------------- #


def _make_field(tcd: np.ndarray, chm: np.ndarray) -> VegetationField:
    utm_crs_str = DemProcessor._get_utm_crs(_ORIGIN_LON, _ORIGIN_LAT)
    utm_crs = CRS.from_string(utm_crs_str)
    ox, oy = rasterio.warp.transform("EPSG:4326", utm_crs, [_ORIGIN_LON], [_ORIGIN_LAT])
    utm_ox, utm_oy = ox[0], oy[0]
    H, W = tcd.shape
    transform = Affine(
        _CELL_M, 0, utm_ox - W / 2 * _CELL_M,
        0, -_CELL_M, utm_oy + H / 2 * _CELL_M,
    )
    return VegetationField(
        tcd=tcd, chm=chm, transform=transform, crs=utm_crs,
        origin_lon=_ORIGIN_LON, origin_lat=_ORIGIN_LAT,
    )


# --------------------------------------------------------------------------- #
# Test 1: VegetationField npz round-trip
# --------------------------------------------------------------------------- #


def test_field_save_load_roundtrip(tmp_path):
    tcd = np.random.default_rng(0).random((_GRID, _GRID)).astype("float32")
    chm = np.random.default_rng(1).random((_GRID, _GRID)).astype("float32") * 20
    field = _make_field(tcd, chm)

    path = tmp_path / "test_field.npz"
    field.save(path)
    loaded = VegetationField.load(path)

    np.testing.assert_array_equal(loaded.tcd, tcd)
    np.testing.assert_array_equal(loaded.chm, chm.astype("float32"))
    assert loaded.origin_lon == pytest.approx(_ORIGIN_LON)
    assert loaded.origin_lat == pytest.approx(_ORIGIN_LAT)
    assert loaded.crs.to_epsg() == field.crs.to_epsg()
    assert loaded.transform == field.transform


# --------------------------------------------------------------------------- #
# Test 2: Full pipeline — uniform TCD × path length → physically expected dB
# --------------------------------------------------------------------------- #


def test_depth_to_attenuation_pipeline_uniform_tcd():
    """
    Uniform TCD=0.8, 100 m horizontal path, 1.8 GHz in-leaf.

    Expected effective depth: 0.8 × 100 m = 80 m
    Expected attenuation (P.833 MED): excess_loss_db(80, 1.8e9) ≈ 15.2 dB
    """
    tcd = np.full((_GRID, _GRID), 0.8, dtype="float32")
    chm = np.full((_GRID, _GRID), 20.0, dtype="float32")
    field = _make_field(tcd, chm)
    integrator = PathDepthIntegrator(field, step_m=_CELL_M)

    tx = np.array([-50.0, 0.0, 5.0], dtype="float32")
    rx = np.array([[50.0, 0.0, 5.0]], dtype="float32")  # 100 m path

    depth = integrator.integrate(tx, rx)
    assert depth[0] == pytest.approx(80.0, abs=1.0)  # 0.8 × 100 m

    attenuation = excess_loss_db(float(depth[0]), freq_hz=1.8e9, leaf_state="in_leaf")
    expected = excess_loss_db(80.0, freq_hz=1.8e9, leaf_state="in_leaf")
    assert attenuation == pytest.approx(expected, abs=0.1)
    assert attenuation > 10.0  # sanity: non-trivial attenuation for 80 m of foliage


# --------------------------------------------------------------------------- #
# Test 3: Zero TCD anywhere on the path → depth = 0 → attenuation = 0
# --------------------------------------------------------------------------- #


def test_zero_tcd_pipeline_gives_zero_attenuation():
    tcd = np.zeros((_GRID, _GRID), dtype="float32")
    chm = np.full((_GRID, _GRID), 20.0, dtype="float32")
    field = _make_field(tcd, chm)
    integrator = PathDepthIntegrator(field, step_m=_CELL_M)

    tx = np.array([-50.0, 0.0, 5.0], dtype="float32")
    rx = np.array([[50.0, 0.0, 5.0]], dtype="float32")

    depth = integrator.integrate(tx, rx)
    attenuation = excess_loss_db(float(depth[0]), freq_hz=1.8e9)
    assert attenuation == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Test 4: Monotonicity — longer path through uniform TCD → more attenuation
# --------------------------------------------------------------------------- #


def test_attenuation_increases_with_path_length():
    tcd = np.ones((_GRID, _GRID), dtype="float32")
    chm = np.full((_GRID, _GRID), 20.0, dtype="float32")
    field = _make_field(tcd, chm)
    integrator = PathDepthIntegrator(field, step_m=_CELL_M)

    tx = np.array([-100.0, 0.0, 5.0], dtype="float32")
    rx_short = np.array([[0.0, 0.0, 5.0]], dtype="float32")   # 100 m
    rx_long = np.array([[100.0, 0.0, 5.0]], dtype="float32")  # 200 m

    depth_short = integrator.integrate(tx, rx_short)[0]
    depth_long = integrator.integrate(tx, rx_long)[0]

    att_short = excess_loss_db(float(depth_short), freq_hz=1.8e9)
    att_long = excess_loss_db(float(depth_long), freq_hz=1.8e9)

    assert depth_long > depth_short
    assert att_long > att_short


# --------------------------------------------------------------------------- #
# Test 5: VegetationField.sample returns expected (tcd, chm) at scene centre
# --------------------------------------------------------------------------- #


def test_field_sample_at_origin():
    """Local (0, 0) should sample the centre pixel of the raster."""
    tcd = np.zeros((_GRID, _GRID), dtype="float32")
    chm = np.zeros((_GRID, _GRID), dtype="float32")
    # Set a distinctive value only at the centre pixel
    tcd[_GRID // 2, _GRID // 2] = 0.75
    chm[_GRID // 2, _GRID // 2] = 12.0
    field = _make_field(tcd, chm)

    sampled_tcd, sampled_chm = field.sample(0.0, 0.0)
    assert sampled_tcd == pytest.approx(0.75, abs=0.01)
    assert sampled_chm == pytest.approx(12.0, abs=0.5)

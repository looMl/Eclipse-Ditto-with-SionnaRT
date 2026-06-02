"""
Tests for PathDepthIntegrator using synthetic TCD/CHM rasters.
All coordinates are scene-local metres; TCD values are normalised [0, 1].
"""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.geomap_processor.processors.dem_processor import DemProcessor
from app.geomap_processor.utils.vegetation_field import VegetationField
from app.simulation.vegetation.vegetation_path_integrator import (
    DemSampler,
    PathDepthIntegrator,
)


_ORIGIN_LON = 11.0
_ORIGIN_LAT = 45.0
_CELL_M = 1.0   # 1 m/cell → precise depth assertions
_GRID = 400     # 400×400 → scene covers ±200 m in both axes


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_field(tcd: np.ndarray, chm: np.ndarray) -> VegetationField:
    """Wraps TCD/CHM arrays in a UTM-projected VegetationField centred at the origin."""
    utm_crs_str = DemProcessor._get_utm_crs(_ORIGIN_LON, _ORIGIN_LAT)
    utm_crs = CRS.from_string(utm_crs_str)
    ox, oy = rasterio.warp.transform("EPSG:4326", utm_crs, [_ORIGIN_LON], [_ORIGIN_LAT])
    utm_ox, utm_oy = ox[0], oy[0]
    H, W = tcd.shape
    # Raster centred on (utm_ox, utm_oy): local (0,0) maps to centre pixel
    transform = Affine(_CELL_M, 0, utm_ox - W / 2 * _CELL_M, 0, -_CELL_M, utm_oy + H / 2 * _CELL_M)
    return VegetationField(
        tcd=tcd,
        chm=chm,
        transform=transform,
        crs=utm_crs,
        origin_lon=_ORIGIN_LON,
        origin_lat=_ORIGIN_LAT,
    )


def _flat_dem() -> DemSampler:
    """1×1 flat DEM at elevation 0 m; clips all sample coordinates to cell (0, 0)."""
    return DemSampler(
        elevation=np.zeros((1, 1), dtype="float32"),
        transform=Affine(1.0, 0, _ORIGIN_LON, 0, -1.0, _ORIGIN_LAT),
        origin_lon=_ORIGIN_LON,
        origin_lat=_ORIGIN_LAT,
        ref_elev=0.0,
    )


# --------------------------------------------------------------------------- #
# Test 1: All-zero TCD → integrator is a no-op
# --------------------------------------------------------------------------- #


def test_zero_tcd_returns_zero_depth():
    tcd = np.zeros((_GRID, _GRID), dtype="float32")
    chm = np.zeros((_GRID, _GRID), dtype="float32")
    field = _make_field(tcd, chm)
    integrator = PathDepthIntegrator(field, step_m=_CELL_M)

    tx = np.array([0.0, 0.0, 5.0], dtype="float32")
    rx = np.array([[100.0, 0.0, 5.0], [-50.0, 50.0, 5.0]], dtype="float32")
    depth = integrator.integrate(tx, rx)
    np.testing.assert_array_equal(depth, 0.0)


# --------------------------------------------------------------------------- #
# Test 2: Full TCD=1 slab, single horizontal path → depth ≈ path length
# --------------------------------------------------------------------------- #


def test_full_tcd_depth_equals_path_length():
    tcd = np.ones((_GRID, _GRID), dtype="float32")
    chm = np.full((_GRID, _GRID), 20.0, dtype="float32")
    field = _make_field(tcd, chm)
    integrator = PathDepthIntegrator(field, step_m=_CELL_M)

    tx = np.array([-50.0, 0.0, 0.0], dtype="float32")
    rx = np.array([[50.0, 0.0, 0.0]], dtype="float32")  # 100 m path
    depth = integrator.integrate(tx, rx)
    assert depth[0] == pytest.approx(100.0, abs=1.0)


# --------------------------------------------------------------------------- #
# Test 3: AABB pre-filter rejects segments entirely outside the non-zero bbox
# --------------------------------------------------------------------------- #


def test_aabb_filter_outside_nz_bbox_returns_zero():
    # Non-zero TCD only in upper-left raster quadrant → local x ∈ [-200, 0], y ∈ [0, 200]
    tcd = np.zeros((_GRID, _GRID), dtype="float32")
    tcd[: _GRID // 2, : _GRID // 2] = 1.0
    chm = np.full((_GRID, _GRID), 20.0, dtype="float32")
    field = _make_field(tcd, chm)
    integrator = PathDepthIntegrator(field, step_m=_CELL_M)

    # Segment entirely in the opposite quadrant (local x > 0, y < 0)
    tx = np.array([50.0, -50.0, 5.0], dtype="float32")
    rx = np.array([[150.0, -150.0, 5.0]], dtype="float32")
    depth = integrator.integrate(tx, rx)
    assert depth[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Test 4: Partial path — only a central strip has TCD=1
# --------------------------------------------------------------------------- #


def test_partial_path_depth_matches_vegetation_width():
    # TCD=1 in a 40 m-wide central strip: local x ∈ [-20, +20]
    tcd = np.zeros((_GRID, _GRID), dtype="float32")
    col_lo = _GRID // 2 - 20
    col_hi = _GRID // 2 + 20
    tcd[:, col_lo:col_hi] = 1.0
    chm = np.full((_GRID, _GRID), 20.0, dtype="float32")
    field = _make_field(tcd, chm)
    integrator = PathDepthIntegrator(field, step_m=_CELL_M)

    tx = np.array([-100.0, 0.0, 5.0], dtype="float32")
    rx = np.array([[100.0, 0.0, 5.0]], dtype="float32")  # 200 m total, 40 m in TCD zone
    depth = integrator.integrate(tx, rx)
    assert depth[0] == pytest.approx(40.0, abs=1.0)


# --------------------------------------------------------------------------- #
# Test 5: 3-D DEM check — ray above canopy → zero depth
# --------------------------------------------------------------------------- #


def test_dem_ray_above_canopy_returns_zero():
    tcd = np.ones((_GRID, _GRID), dtype="float32")
    chm = np.full((_GRID, _GRID), 5.0, dtype="float32")  # 5 m canopy
    field = _make_field(tcd, chm)
    # flat DEM at z=0 → canopy top = 0 + 5 = 5 m
    integrator = PathDepthIntegrator(field, dem=_flat_dem(), step_m=_CELL_M)

    tx = np.array([0.0, 0.0, 10.0], dtype="float32")   # ray at z=10 m
    rx = np.array([[50.0, 0.0, 10.0]], dtype="float32")
    depth = integrator.integrate(tx, rx)
    assert depth[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Test 6: 3-D DEM check — ray below canopy top → depth ≈ path length
# --------------------------------------------------------------------------- #


def test_dem_ray_below_canopy_returns_full_depth():
    tcd = np.ones((_GRID, _GRID), dtype="float32")
    chm = np.full((_GRID, _GRID), 20.0, dtype="float32")  # 20 m canopy
    field = _make_field(tcd, chm)
    # flat DEM at z=0 → canopy top = 0 + 20 = 20 m
    integrator = PathDepthIntegrator(field, dem=_flat_dem(), step_m=_CELL_M)

    tx = np.array([-50.0, 0.0, 2.0], dtype="float32")   # ray at z=2 m, below 20 m canopy
    rx = np.array([[50.0, 0.0, 2.0]], dtype="float32")   # 100 m path
    depth = integrator.integrate(tx, rx)
    assert depth[0] == pytest.approx(100.0, abs=1.0)

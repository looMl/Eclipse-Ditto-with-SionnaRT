"""Unit tests for VegetationRasterDownloader tile-grid and cache-path helpers."""

from app.geomap_processor.data.vegetation_raster_downloader import (
    VegetationRasterDownloader,
)


def test_single_tile_id_northern_eastern(tmp_path):
    dl = VegetationRasterDownloader(tmp_path)
    assert dl._tile_ids((11.0, 45.0, 11.5, 45.5)) == ["N45E009"]


def test_tile_id_formatting_southern_western(tmp_path):
    dl = VegetationRasterDownloader(tmp_path)
    assert dl._tile_ids((-5.0, -10.0, -4.6, -9.6)) == ["S12W006"]


def test_bbox_spanning_grid_boundary_yields_multiple_tiles(tmp_path):
    dl = VegetationRasterDownloader(tmp_path)
    ids = dl._tile_ids((8.0, 44.0, 13.0, 46.0))
    assert len(ids) > 1
    assert "N45E009" in ids


def test_cache_path_is_deterministic(tmp_path):
    dl = VegetationRasterDownloader(tmp_path)
    bbox = (11.0, 45.0, 11.5, 45.5)
    p1 = dl._cache_path("tcd", "esa_worldcover", bbox)
    p2 = dl._cache_path("tcd", "esa_worldcover", bbox)
    assert p1 == p2
    assert p1.parent == tmp_path
    assert p1.suffix == ".tif"
    assert "tcd" in p1.name and "esa_worldcover" in p1.name


def test_cache_path_differs_for_different_bbox(tmp_path):
    dl = VegetationRasterDownloader(tmp_path)
    a = dl._cache_path("tcd", "esa_worldcover", (11.0, 45.0, 11.5, 45.5))
    b = dl._cache_path("tcd", "esa_worldcover", (12.0, 45.0, 12.5, 45.5))
    assert a != b

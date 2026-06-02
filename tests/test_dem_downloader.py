"""Unit tests for DemDownloader pure path/dimension/parameter helpers."""

from app.geomap_processor.data.dem_downloader import DemDownloader


def test_file_path_is_deterministic_and_cached(tmp_path):
    dl = DemDownloader(tmp_path)
    bbox = (11.0, 45.0, 11.5, 45.5)
    p1 = dl._get_file_path(bbox)
    p2 = dl._get_file_path(bbox)
    assert p1 == p2
    assert p1.parent == tmp_path
    assert p1.name.startswith("tinitaly_")
    assert p1.suffix == ".tif"


def test_file_path_differs_for_different_bbox(tmp_path):
    dl = DemDownloader(tmp_path)
    a = dl._get_file_path((11.0, 45.0, 11.5, 45.5))
    b = dl._get_file_path((12.0, 45.0, 12.5, 45.5))
    assert a != b


def test_calculate_dimensions_scale_with_span(tmp_path):
    dl = DemDownloader(tmp_path)
    width, height = dl._calculate_dimensions((0.0, 0.0, 0.09, 0.18))
    assert width > 0 and height > 0
    # latitude span is twice the longitude span
    assert abs(height - 2 * width) <= 1


def test_build_params_formats_request(tmp_path):
    dl = DemDownloader(tmp_path)
    params = dl._build_params((1.0, 2.0, 3.0, 4.0), 100, 200)
    assert params["service"] == "WCS"
    assert params["request"] == "GetCoverage"
    assert params["crs"] == "EPSG:4326"
    assert params["bbox"] == "1.0,2.0,3.0,4.0"
    assert params["width"] == "100"
    assert params["height"] == "200"

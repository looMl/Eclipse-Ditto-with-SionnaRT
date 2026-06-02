"""Unit tests for TelecomManager sector generation and Ditto export."""

import json

import pytest
from shapely.geometry import Point, Polygon

from app.geomap_processor.managers.telecom_manager import TelecomManager
from app.geomap_processor.utils.geometry_utils import BoundingBox


@pytest.fixture
def manager():
    return TelecomManager(BoundingBox(11.0, 45.0, 11.5, 45.5))


def test_create_site_builds_three_sectors(manager):
    sectors = manager._create_site_transmitters("123", 45.0, 11.0, 10.0, 20.0)
    assert [t.id for t in sectors] == ["123_s0", "123_s1", "123_s2"]
    for t in sectors:
        assert t.height == TelecomManager.DEFAULT_HEIGHT
        assert t.frequency == 1.8e9
        assert 43.0 <= t.power_dbm <= 46.0
        assert 2.0 <= t.tilt <= 6.0
        assert 0 <= t.active_users <= 33


def test_sectors_are_120_degrees_apart(manager):
    az = [
        t.azimuth for t in manager._create_site_transmitters("1", 45.0, 11.0, 0.0, 0.0)
    ]
    assert az[1] - az[0] == pytest.approx(120.0)
    assert az[2] - az[1] == pytest.approx(120.0)


def test_geometry_center_point(manager):
    assert manager._get_geometry_center(Point(11.0, 45.0)) == (11.0, 45.0)


def test_geometry_center_polygon_uses_centroid(manager):
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    assert manager._get_geometry_center(square) == (1.0, 1.0)


def test_save_transmitters_json_groups_site(manager, tmp_path):
    manager.transmitters = manager._create_site_transmitters(
        "123", 45.0, 11.0, 0.0, 0.0
    )
    out = tmp_path / "tx.json"
    manager.save_transmitters_json(out)

    items = json.loads(out.read_text())
    assert len(items) == 1
    assert items[0]["thingId"] == "com.sionna:antenna_123"
    assert set(items[0]["features"]) == {"sector_0", "sector_1", "sector_2"}

"""Unit tests for bounding-box, material-config, and material-index helpers."""

import pytest

from app.geomap_processor.utils.geometry_utils import (
    BoundingBox,
    MaterialConfig,
    resolve_material,
)


def test_valid_bbox_passes_validation():
    BoundingBox(11.0, 45.0, 11.5, 45.5).validate()  # should not raise


def test_invalid_longitude_order_raises():
    with pytest.raises(ValueError, match="min_lon"):
        BoundingBox(12.0, 45.0, 11.0, 45.5).validate()


def test_invalid_latitude_order_raises():
    with pytest.raises(ValueError, match="min_lat"):
        BoundingBox(11.0, 46.0, 11.5, 45.5).validate()


def test_center_is_midpoint():
    assert BoundingBox(10.0, 40.0, 12.0, 44.0).center == (11.0, 42.0)


def test_polygon_points_form_closed_ccw_loop():
    bbox = BoundingBox(10.0, 40.0, 12.0, 44.0)
    pts = bbox.polygon_points
    assert len(pts) == 5
    assert pts[0] == pts[-1]  # closed loop
    assert pts[0] == [10.0, 40.0]


def test_to_dict_roundtrips_fields():
    bbox = BoundingBox(10.0, 40.0, 12.0, 44.0)
    assert bbox.to_dict() == {
        "min_lon": 10.0,
        "min_lat": 40.0,
        "max_lon": 12.0,
        "max_lat": 44.0,
    }


def test_material_config_defaults():
    cfg = MaterialConfig()
    assert (cfg.ground_idx, cfg.rooftop_idx, cfg.wall_idx) == (1, 2, 1)


def test_resolve_material_returns_a_key():
    assert isinstance(resolve_material(0), str)
    assert resolve_material(0)


@pytest.mark.parametrize("idx", [-1, 10_000])
def test_resolve_material_out_of_range_raises(idx):
    with pytest.raises(ValueError, match="Invalid material index"):
        resolve_material(idx)

"""Unit tests for VegetationManager per-tag TCD/height lookups."""

import pytest

from app.geomap_processor.managers.vegetation_manager import VegetationManager
from app.geomap_processor.utils.geometry_utils import BoundingBox


@pytest.fixture
def manager():
    return VegetationManager(BoundingBox(11.0, 45.0, 11.5, 45.5))


def test_tag_tcd_known_tags(manager):
    assert manager._tag_tcd({"natural": "wood"}) == 0.9
    assert manager._tag_tcd({"landuse": "forest"}) == 0.9
    assert manager._tag_tcd({"leisure": "park"}) == 0.5


def test_tag_tcd_unknown_tag_is_zero(manager):
    assert manager._tag_tcd({"natural": "water"}) == 0.0
    assert manager._tag_tcd({}) == 0.0


def test_tag_height_known_tags(manager):
    assert manager._tag_height({"natural": "wood"}) == 18.0
    assert manager._tag_height({"leisure": "park"}) == 12.0


def test_tag_height_unknown_tag_is_zero(manager):
    assert manager._tag_height({"landuse": "industrial"}) == 0.0

"""Unit tests for configuration schema models and defaults."""

import pytest
from pydantic import ValidationError

from app.config import (
    CameraSettings,
    CoverageSettings,
    MaterialSettings,
    Settings,
    VegetationSettings,
    get_settings,
)


def test_coverage_settings_default_path_budget():
    cov = CoverageSettings(samples_per_tx=10, max_depth=3, metric="rss")
    assert cov.max_num_paths_per_src == 200_000


def test_vegetation_settings_defaults():
    veg = VegetationSettings()
    assert veg.enabled is False
    assert veg.leaf_state == "in_leaf"
    assert veg.mode == "per_link"


def test_material_settings_rejects_negative_index():
    with pytest.raises(ValidationError):
        MaterialSettings(ground_idx=-1)


def test_camera_settings_requires_all_vectors():
    with pytest.raises(ValidationError):
        CameraSettings(position=[0.0, 0.0, 0.0])  # missing orientation / look_at


def test_module_settings_singleton_is_loaded():
    assert isinstance(get_settings(), Settings)

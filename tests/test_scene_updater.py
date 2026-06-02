"""Unit tests for SceneXMLUpdater shape/material/projection handling."""

import pytest

from app.geomap_processor.data.scene_updater import SceneXMLUpdater

_SCENE_XML = """<scene version="2.0.0">
  <default name="scenegen_center_lat" value="45.0"/>
  <default name="scenegen_center_lon" value="11.0"/>
  <default name="scenegen_UTM_zone" value="EPSG:32632"/>
  <bsdf type="diffuse" id="mat-ground">
    <rgb name="reflectance" value="0.5 0.4 0.3"/>
  </bsdf>
  <shape type="ply" id="ground">
    <string name="filename" value="mesh/ground.ply"/>
    <ref name="bsdf" id="mat-ground"/>
  </shape>
</scene>
"""


@pytest.fixture
def updater(tmp_path):
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text(_SCENE_XML)
    return SceneXMLUpdater(scene_path)


def test_get_projection_info(updater):
    info = updater.get_projection_info()
    assert info == {"center_lat": 45.0, "center_lon": 11.0, "utm_zone": "EPSG:32632"}


def test_get_material_colors(updater):
    colors = updater.get_material_colors()
    assert colors == {"mat-ground": (0.5, 0.4, 0.3)}


def test_remove_shapes_returns_bsdf_and_deletes(updater):
    bsdf_id = updater.remove_shapes_by_filenames({"mesh/ground.ply"})
    assert bsdf_id == "mat-ground"
    assert updater.root.find(".//shape[@id='ground']") is None


def test_remove_unknown_filename_returns_none(updater):
    assert updater.remove_shapes_by_filenames({"mesh/nope.ply"}) is None


def test_add_mesh_shape_is_idempotent(updater):
    updater.add_mesh_shape("mesh/terrain.ply", "mesh-terrain", "mat-ground")
    updater.add_mesh_shape("mesh/terrain.ply", "mesh-terrain", "mat-ground")
    matches = updater.root.findall(".//shape[@id='mesh-terrain']")
    assert len(matches) == 1


def test_save_roundtrip(updater, tmp_path):
    updater.add_mesh_shape("mesh/terrain.ply", "mesh-terrain", "mat-ground")
    updater.save()
    reloaded = SceneXMLUpdater(tmp_path / "scene.xml")
    assert reloaded.root.find(".//shape[@id='mesh-terrain']") is not None

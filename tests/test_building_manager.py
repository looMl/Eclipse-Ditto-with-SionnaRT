"""Unit tests for BuildingMesher mesh merging and cleanup."""

import trimesh

from app.geomap_processor.managers.building_manager import BuildingMesher


def _write_box(path, extents=(1.0, 1.0, 1.0)):
    trimesh.creation.box(extents=extents).export(str(path))


def test_merge_meshes_writes_combined_output(tmp_path):
    f1 = tmp_path / "building_1_wall.ply"
    f2 = tmp_path / "building_2_wall.ply"
    _write_box(f1)
    _write_box(f2)

    mesher = BuildingMesher(tmp_path)
    assert mesher.merge_meshes([f1, f2], "walls.ply") is True

    output = tmp_path / "walls.ply"
    assert output.exists()
    merged = trimesh.load(output, force="mesh")
    assert len(merged.faces) > 0


def test_merge_meshes_empty_list_returns_false(tmp_path):
    assert BuildingMesher(tmp_path).merge_meshes([], "out.ply") is False


def test_get_building_files_globs_walls_and_rooftops(tmp_path):
    _write_box(tmp_path / "building_1_wall.ply")
    _write_box(tmp_path / "building_1_rooftop.ply")
    walls, rooftops = BuildingMesher(tmp_path).get_building_files()
    assert [p.name for p in walls] == ["building_1_wall.ply"]
    assert [p.name for p in rooftops] == ["building_1_rooftop.ply"]


def test_cleanup_files_removes_them(tmp_path):
    f = tmp_path / "building_1_wall.ply"
    _write_box(f)
    BuildingMesher(tmp_path).cleanup_files([f])
    assert not f.exists()


def test_apply_height_offset_translates_in_z(tmp_path):
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    z_before = mesh.centroid[2]
    BuildingMesher(tmp_path)._apply_height_offset(mesh, lambda x, y: 5.0)
    assert mesh.centroid[2] == z_before + 5.0

"""Unit tests for the mesh subdivision utility."""

import trimesh

from app.geomap_processor.utils.mesh_utils import subdivide_mesh


def test_subdivide_increases_face_count(tmp_path):
    src = tmp_path / "in.ply"
    dst = tmp_path / "out.ply"
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    box.export(str(src))

    assert subdivide_mesh(src, dst, max_edge_len=2.0) is True
    assert dst.exists()
    assert len(trimesh.load(dst, force="mesh").faces) > len(box.faces)


def test_subdivide_already_fine_mesh_passes_through(tmp_path):
    src = tmp_path / "in.ply"
    dst = tmp_path / "out.ply"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(str(src))

    # max_edge_len far above any edge -> no subdivision needed, still succeeds
    assert subdivide_mesh(src, dst, max_edge_len=1000.0) is True
    assert dst.exists()

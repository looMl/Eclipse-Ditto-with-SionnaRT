import logging
import trimesh
from pathlib import Path
from typing import List, Tuple, Callable, Optional

logger = logging.getLogger(__name__)


class BuildingMesher:
    """Service responsible for merging building meshes together."""

    def __init__(self, mesh_dir: Path):
        self.mesh_dir = mesh_dir

    def get_building_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Scans the mesh directory for building wall and rooftop files.
        """
        wall_files = list(self.mesh_dir.glob("building_*_wall.ply"))
        rooftop_files = list(self.mesh_dir.glob("building_*_rooftop.ply"))
        return wall_files, rooftop_files

    def merge_meshes(
        self,
        files: List[Path],
        output_filename: str,
        height_callback: Optional[Callable[[float, float], float]] = None,
    ) -> bool:
        """
        Merges provided mesh files into a single file, optionally applying a height offset.
        """
        if not files:
            return False

        logger.info(f"Merging {len(files)} meshes into {output_filename}...")
        meshes = []
        try:
            for f in files:
                mesh = trimesh.load(f)

                if isinstance(mesh, trimesh.Scene):
                    if len(mesh.geometry) == 0:
                        continue
                    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

                if height_callback:
                    self._apply_height_offset(mesh, height_callback)

                meshes.append(mesh)

            if not meshes:
                return False

            combined = trimesh.util.concatenate(meshes)
            output_path = self.mesh_dir / output_filename
            combined.export(str(output_path))
            logger.info(f"Exported {output_filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to merge meshes into {output_filename}: {e}")
            return False

    def _apply_height_offset(
        self, mesh, height_callback: Callable[[float, float], float]
    ) -> None:
        """Applies vertical offset to a mesh based on its centroid."""
        try:
            cx, cy = mesh.centroid[0], mesh.centroid[1]
            z_offset = height_callback(cx, cy)
            if z_offset != 0:
                mesh.apply_translation([0, 0, z_offset])
        except Exception as e:
            logger.warning(f"Failed to apply height offset to mesh: {e}")

    def cleanup_files(self, files: List[Path]) -> None:
        """Deletes the provided files from the filesystem."""
        for f in files:
            try:
                f.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete {f}: {e}")
        logger.info(f"Deleted {len(files)} individual mesh files.")

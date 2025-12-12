import logging
import trimesh
from pathlib import Path
from typing import List, Tuple

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

    def merge_meshes(self, files: List[Path], output_filename: str) -> bool:
        """
        Merges provided mesh files into a single file.
        """
        if not files:
            return False

        logger.info(f"Merging {len(files)} meshes into {output_filename}...")
        try:
            meshes = [trimesh.load(f) for f in files]
            combined = trimesh.util.concatenate(meshes)

            output_path = self.mesh_dir / output_filename
            combined.export(str(output_path))
            logger.info(f"Exported {output_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to merge meshes into {output_filename}: {e}")
            return False

    def cleanup_files(self, files: List[Path]) -> None:
        """Deletes the provided files from the filesystem."""
        for f in files:
            try:
                f.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete {f}: {e}")
        logger.info(f"Deleted {len(files)} individual mesh files.")

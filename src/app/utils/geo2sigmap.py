import logging
import trimesh
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import xml.etree.ElementTree as ET
from scene_generation.core import Scene
from scene_generation.itu_materials import ITU_MATERIALS

from app.utils.config import settings, get_project_root
from app.utils.telecom_manager import TelecomManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache material list for index-based access
_MATERIALS_LIST = list(ITU_MATERIALS.items())


def resolve_material(idx: int) -> str:
    """Resolves material index to name safely."""
    try:
        return _MATERIALS_LIST[idx][0]
    except IndexError:
        logger.warning(f"Invalid material index {idx}. Using default.")
        return _MATERIALS_LIST[0][0]


@dataclass(frozen=True)
class BoundingBox:
    """Represents the geographical bounding box."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def validate(self) -> None:
        """Ensures coordinates form a valid bounding box."""
        if self.min_lon >= self.max_lon:
            raise ValueError(
                f"min_lon ({self.min_lon}) must be less than max_lon ({self.max_lon})"
            )
        if self.min_lat >= self.max_lat:
            raise ValueError(
                f"min_lat ({self.min_lat}) must be less than max_lat ({self.max_lat})"
            )

    def to_dict(self) -> Dict[str, float]:
        """Returns the bbox as a dictionary for compatibility."""
        return {
            "min_lon": self.min_lon,
            "min_lat": self.min_lat,
            "max_lon": self.max_lon,
            "max_lat": self.max_lat,
        }

    @property
    def polygon_points(self) -> List[List[float]]:
        """
        Returns the counter-clockwise polygon points for the bbox.
        Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left -> Top-Left (Closed Loop)
        """
        return [
            [self.min_lon, self.min_lat],
            [self.min_lon, self.max_lat],
            [self.max_lon, self.max_lat],
            [self.max_lon, self.min_lat],
            [self.min_lon, self.min_lat],
        ]


@dataclass(frozen=True)
class MaterialConfig:
    """Configuration for material indices."""

    ground_idx: int = 14  # Default: wet ground
    rooftop_idx: int = 2  # Default: brick
    wall_idx: int = 1  # Default: concrete


class SceneBuilder:
    """Service responsible for generating the 3D scene."""

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir

    def _ensure_output_directory(self) -> None:
        if not self._output_dir.exists():
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ValueError(
                    f"Could not create output directory '{self._output_dir}': {e}"
                )

    def _process_telecom_infrastructure(
        self, bbox: BoundingBox
    ) -> Tuple[Optional[TelecomManager], Optional[str]]:
        """
        Fetches and processes telecom data, exporting the mesh.
        Returns the manager instance and the relative mesh path.
        """
        logger.info("Starting Telecom Infrastructure Generation...")
        telecom_mgr = TelecomManager(bbox=bbox.to_dict())
        telecom_mgr.fetch_and_process()

        mesh = telecom_mgr.get_mesh()
        if mesh:
            mesh_dir = self._output_dir / "mesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)

            ply_path = mesh_dir / "transmitters.ply"
            mesh.export(str(ply_path))
            logger.info(f"Exported mesh to {ply_path}")
            return telecom_mgr, "mesh/transmitters.ply"

        return None, None

    def _merge_building_meshes(self) -> None:
        """
        Optimizes building meshes by merging them into single files for walls and rooftops.
        """
        mesh_dir = self._output_dir / "mesh"

        logger.info("Optimizing building meshes...")

        wall_files = list(mesh_dir.glob("building_*_wall.ply"))
        rooftop_files = list(mesh_dir.glob("building_*_rooftop.ply"))

        if not wall_files and not rooftop_files:
            return

        self._merge_and_save_meshes(wall_files, "buildings_walls.ply")
        self._merge_and_save_meshes(rooftop_files, "buildings_rooftops.ply")

        self._update_scene_xml(wall_files, rooftop_files)
        self._cleanup_files(wall_files + rooftop_files)

    def _merge_and_save_meshes(self, files: List[Path], output_filename: str) -> None:
        if not files:
            return

        logger.info(f"Merging {len(files)} meshes into {output_filename}...")
        try:
            meshes = [trimesh.load(f) for f in files]
            combined = trimesh.util.concatenate(meshes)

            output_path = self._output_dir / "mesh" / output_filename
            combined.export(str(output_path))
            logger.info(f"Exported {output_filename}")
        except Exception as e:
            logger.error(f"Failed to merge meshes into {output_filename}: {e}")

    def _cleanup_files(self, files: List[Path]) -> None:
        for f in files:
            try:
                f.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete {f}: {e}")
        logger.info("Deleted individual mesh files.")

    def _update_scene_xml(
        self, wall_files: List[Path], rooftop_files: List[Path]
    ) -> None:
        """Updates scene.xml to point to merged meshes and remove old ones."""
        scene_path = self._output_dir / "scene.xml"

        try:
            tree = ET.parse(scene_path)
            root = tree.getroot()

            # Prepare sets for fast lookup (relative paths)
            wall_filenames = {f"mesh/{f.name}" for f in wall_files}
            rooftop_filenames = {f"mesh/{f.name}" for f in rooftop_files}

            shapes_to_remove = []
            wall_bsdf_id = None
            rooftop_bsdf_id = None

            # Identify shapes to remove and extract BSDF IDs
            for shape in root.findall("shape"):
                filename_node = shape.find("string[@name='filename']")
                if filename_node is None:
                    continue

                fname = filename_node.get("value")
                if fname in wall_filenames:
                    shapes_to_remove.append(shape)
                    if not wall_bsdf_id:
                        wall_bsdf_id = self._get_bsdf_id(shape)
                elif fname in rooftop_filenames:
                    shapes_to_remove.append(shape)
                    if not rooftop_bsdf_id:
                        rooftop_bsdf_id = self._get_bsdf_id(shape)

            # Remove old shapes
            for shape in shapes_to_remove:
                root.remove(shape)

            logger.info(
                f"Removed {len(shapes_to_remove)} individual building shapes from scene.xml"
            )

            # Add new merged shapes
            if wall_files and wall_bsdf_id:
                self._add_shape_to_xml(
                    root,
                    "mesh/buildings_walls.ply",
                    "mesh-buildings-walls",
                    wall_bsdf_id,
                )

            if rooftop_files and rooftop_bsdf_id:
                self._add_shape_to_xml(
                    root,
                    "mesh/buildings_rooftops.ply",
                    "mesh-buildings-rooftops",
                    rooftop_bsdf_id,
                )

            tree.write(scene_path, encoding="utf-8", xml_declaration=True)
            logger.info("Updated scene.xml with merged meshes.")

        except ET.ParseError as e:
            logger.error(f"Failed to parse scene.xml: {e}")
        except Exception as e:
            logger.error(f"Error updating scene.xml: {e}")

    def _get_bsdf_id(self, shape: ET.Element) -> Optional[str]:
        ref = shape.find("ref[@name='bsdf']")
        return ref.get("id") if ref is not None else None

    def _add_shape_to_xml(
        self, root: ET.Element, filename: str, shape_id: str, bsdf_id: str
    ) -> None:
        new_shape = ET.SubElement(root, "shape")
        new_shape.set("type", "ply")
        new_shape.set("id", shape_id)

        fn = ET.SubElement(new_shape, "string")
        fn.set("name", "filename")
        fn.set("value", filename)

        ref = ET.SubElement(new_shape, "ref")
        ref.set("name", "bsdf")
        ref.set("id", bsdf_id)

    def generate(self, bbox: BoundingBox, materials: MaterialConfig) -> None:
        """Orchestrates the scene generation process."""
        self._ensure_output_directory()
        bbox.validate()

        logger.info(
            f"Generating scene for BBOX: {bbox.min_lon}, {bbox.min_lat}, "
            f"{bbox.max_lon}, {bbox.max_lat}"
        )
        logger.info(f"Output Directory: {self._output_dir}")

        # Resolve materials
        ground_mat = resolve_material(materials.ground_idx)
        rooftop_mat = resolve_material(materials.rooftop_idx)
        wall_mat = resolve_material(materials.wall_idx)

        try:
            scene_instance = Scene()

            # Explicitly disabling LiDAR and DEM features of the library
            scene_instance(
                points=bbox.polygon_points,
                data_dir=str(self._output_dir),
                hag_tiff_path=None,
                osm_server_addr="https://overpass-api.de/api/interpreter",
                lidar_calibration=False,
                generate_building_map=False,
                ground_material_type=ground_mat,
                rooftop_material_type=rooftop_mat,
                wall_material_type=wall_mat,
                lidar_terrain=False,
                dem_terrain=False,
                gen_lidar_terrain_only=False,
            )

            logger.info("Scene generation completed successfully.")

            self._merge_building_meshes()

            telecom_mgr, mesh_rel_path = self._process_telecom_infrastructure(bbox)
            if telecom_mgr and mesh_rel_path:
                scene_file = self._output_dir / "scene.xml"
                telecom_mgr.update_scene_xml(scene_file, mesh_rel_path)

            logger.info("Telecom Infrastructure Generation completed.")

        except Exception as e:
            logger.error(f"Error during scene generation: {e}")
            raise RuntimeError(f"Scene generation failed: {e}")


def main():
    try:
        bbox = BoundingBox(
            min_lon=settings.geo2sigmap.min_lon,
            min_lat=settings.geo2sigmap.min_lat,
            max_lon=settings.geo2sigmap.max_lon,
            max_lat=settings.geo2sigmap.max_lat,
        )

        material_config = MaterialConfig()
        output_dir = get_project_root() / "scene"

        builder = SceneBuilder(output_dir=output_dir)

        builder.generate(bbox, material_config)

    except Exception as e:
        logger.critical(f"Application failed: {e}")


if __name__ == "__main__":
    main()

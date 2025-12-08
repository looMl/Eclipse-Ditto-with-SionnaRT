import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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

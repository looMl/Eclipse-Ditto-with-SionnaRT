import sys
from typing import Tuple, Optional, Any, Callable
from pathlib import Path
from loguru import logger
from scene_generation.core import Scene

from app.config import settings, get_project_root
from app.geomap_processor.managers.telecom_manager import TelecomManager
from app.geomap_processor.managers.building_manager import BuildingMesher
from app.geomap_processor.data.scene_updater import SceneXMLUpdater
from app.geomap_processor.data.dem_downloader import DemDownloader
from app.geomap_processor.processors.dem_processor import DemProcessor
from app.geomap_processor.utils.geometry_utils import (
    BoundingBox,
    MaterialConfig,
    resolve_material,
)
from app.core.ditto_manager import DittoManager


logger.remove()
logger.add(sys.stdout, level=settings.logging.level)


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

    def generate(self, bbox: BoundingBox, materials: MaterialConfig) -> None:
        """Orchestrates the scene generation process."""
        self._ensure_output_directory()
        bbox.validate()

        logger.info(
            f"Generating scene for BBOX: {bbox.min_lon}, {bbox.min_lat}, "
            f"{bbox.max_lon}, {bbox.max_lat}"
        )
        logger.info(f"Output Directory: {self._output_dir}")

        try:
            self._generate_core_scene(bbox, materials)
            logger.success("Scene generation completed successfully.")

            # Process terrain first to get elevation data
            elev_data, transform, ref_elev = self._process_terrain(bbox)

            # Define height callback for adjusting buildings meshes
            height_callback = self._create_height_callback(
                elev_data, transform, ref_elev, *bbox.center
            )

            self._optimize_buildings(height_callback)
            self._process_telecom_infrastructure(bbox, height_callback)

        except Exception as e:
            logger.error(f"Error during scene generation: {e}")
            raise RuntimeError(f"Scene generation failed: {e}")

    def _generate_core_scene(
        self, bbox: BoundingBox, materials: MaterialConfig
    ) -> None:
        """Generates the base scene using the core library."""
        # Resolve materials
        ground_mat = resolve_material(materials.ground_idx)
        rooftop_mat = resolve_material(materials.rooftop_idx)
        wall_mat = resolve_material(materials.wall_idx)

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

    def _create_height_callback(
        self,
        elev_data: Any,
        transform: Any,
        ref_elev: float,
        center_lon: float,
        center_lat: float,
    ) -> Optional[Callable[[float, float], float]]:
        """Creates a callback function for height adjustment."""
        if elev_data is None or transform is None:
            return None

        def _cb(x: float, y: float) -> float:
            lon, lat = DemProcessor.local_to_global(x, y, center_lon, center_lat)
            return DemProcessor.sample_elevation(
                elev_data, transform, lon, lat, ref_elev
            )

        return _cb

    def _optimize_buildings(
        self, height_callback: Optional[Callable[[float, float], float]]
    ) -> None:
        """Merges the generated building meshes into one using BuildingMesher"""
        logger.info("Optimizing building meshes...")
        mesh_dir = self._output_dir / "mesh"

        mesher = BuildingMesher(mesh_dir)

        wall_files, rooftop_files = mesher.get_building_files()

        if not wall_files and not rooftop_files:
            logger.info("No building meshes found to optimize.")
            return

        # Merge meshes with optional height adjustment
        walls_merged = mesher.merge_meshes(
            wall_files, "buildings_walls.ply", height_callback
        )
        rooftops_merged = mesher.merge_meshes(
            rooftop_files, "buildings_rooftops.ply", height_callback
        )

        scene_path = self._output_dir / "scene.xml"
        updater = SceneXMLUpdater(scene_path)

        # Remove old shapes and capture BSDF IDs
        wall_filenames = {f"mesh/{f.name}" for f in wall_files}
        wall_bsdf_id = updater.remove_shapes_by_filenames(wall_filenames)

        rooftop_filenames = {f"mesh/{f.name}" for f in rooftop_files}
        rooftop_bsdf_id = updater.remove_shapes_by_filenames(rooftop_filenames)

        # Add merged shapes
        if walls_merged and wall_bsdf_id:
            updater.add_mesh_shape(
                "mesh/buildings_walls.ply", "mesh-buildings-walls", wall_bsdf_id
            )

        if rooftops_merged and rooftop_bsdf_id:
            updater.add_mesh_shape(
                "mesh/buildings_rooftops.ply",
                "mesh-buildings-rooftops",
                rooftop_bsdf_id,
            )

        updater.save()

        # Cleanup the old individual mesh files
        mesher.cleanup_files(wall_files + rooftop_files)

    def _process_telecom_infrastructure(
        self,
        bbox: BoundingBox,
        height_callback: Optional[Callable[[float, float], float]],
    ) -> None:
        """
        Fetches and processes telecom data, exporting the mesh and updating scene.xml.
        """
        logger.info("Starting Telecom Infrastructure Generation...")
        telecom_mgr = TelecomManager(bbox=bbox)
        telecom_mgr.fetch_and_process()

        # Export transmitters to JSON for Eclipse Ditto
        json_path = get_project_root() / "things" / "transmitters.json"
        telecom_mgr.save_transmitters_json(json_path)

        # Provision things in Eclipse Ditto
        DittoManager().provision_simulation(json_path)

        mesh = telecom_mgr.get_mesh(height_callback)
        if mesh:
            mesh_dir = self._output_dir / "mesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)

            ply_path = mesh_dir / "transmitters.ply"
            mesh.export(str(ply_path))
            logger.info(f"Exported mesh to {ply_path}")

            scene_path = self._output_dir / "scene.xml"
            updater = SceneXMLUpdater(scene_path)

            # Using standard ITU metal for transmitters
            updater.add_mesh_shape(
                "mesh/transmitters.ply", "mesh-transmitters", "mat-itu_metal"
            )
            updater.save()
            logger.info("Telecom Infrastructure added to scene.")
        else:
            logger.info("No telecom infrastructure found or mesh generation failed.")

    def _process_terrain(self, bbox: BoundingBox) -> Tuple[Any, Any, float]:
        """Generates terrain mesh from DEM and updates the scene."""
        logger.info("Processing terrain from DEM...")

        bbox_tuple = (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
        center_lon, center_lat = bbox.center

        # Fetch DEM from TINITALY
        downloader = DemDownloader(get_project_root() / "geotiffs")
        dem_path = downloader.fetch(bbox_tuple)

        if not dem_path or not dem_path.exists():
            logger.warning("Could not fetch DEM file. Skipping terrain generation.")
            return None, None, 0.0

        try:
            elevation, transform = DemProcessor.process_dem(dem_path, bbox_tuple)

            mesh_dir = self._output_dir / "mesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            terrain_path = mesh_dir / "terrain.ply"

            ref_elev = DemProcessor.generate_terrain_mesh(
                elevation, transform, terrain_path, mesh_origin=(center_lon, center_lat)
            )

            scene_path = self._output_dir / "scene.xml"
            updater = SceneXMLUpdater(scene_path)

            # Remove old ground from scene.xml and get its material
            ground_files = {"mesh/ground.ply"}
            ground_bsdf = updater.remove_shapes_by_filenames(ground_files)

            if ground_bsdf:
                updater.add_mesh_shape("mesh/terrain.ply", "mesh-terrain", ground_bsdf)
                updater.save()
                logger.info("Replaced ground.ply with terrain.ply in scene.xml")
                # Remove the old ground mesh file
                ground_ply_path = self._output_dir / "mesh" / "ground.ply"
                ground_ply_path.unlink()
            else:
                logger.warning("Could not find existing ground shape to replace.")

            return elevation, transform, ref_elev

        except Exception as e:
            logger.error(f"Failed to process terrain: {e}")
            return None, None, 0.0


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

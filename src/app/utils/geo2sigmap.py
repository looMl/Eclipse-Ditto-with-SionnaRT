import os
import logging
from scene_generation.core import Scene
from scene_generation.itu_materials import ITU_MATERIALS
from config import settings, get_project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_scene_from_coords(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    output_dir: str,
    ground_material_idx: int = 14,  # Default: wet ground
    rooftop_material_idx: int = 2,  # Default: brick
    wall_material_idx: int = 1,  # Default: concrete
) -> None:
    """
    Generates a 3D scene using the 'scene_generation' library based on four corner points.
    This function isolates the specific 'bbox' functionality of the geo2sigmap library,
    disabling LiDAR, DEM, and other interactive features as they are not needed.

    Args:
        min_lon (float): Minimum longitude.
        min_lat (float): Minimum latitude.
        max_lon (float): Maximum longitude.
        max_lat (float): Maximum latitude.
        output_dir (str): Directory where the scene files (XML, meshes) will be saved.
        ground_material_idx (int): Index of the ground material in ITU_MATERIALS.
        rooftop_material_idx (int): Index of the rooftop material in ITU_MATERIALS.
        wall_material_idx (int): Index of the wall material in ITU_MATERIALS.
    """

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Could not create output directory '{output_dir}': {e}")

    # Validate coordinates
    if min_lon >= max_lon:
        raise ValueError(f"min_lon ({min_lon}) must be less than max_lon ({max_lon})")
    if min_lat >= max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be less than max_lat ({max_lat})")

    # Construct the polygon points (Counter-Clockwise)
    # Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left -> Top-Left (Closed Loop)

    points = [
        [min_lon, min_lat],
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat],
    ]

    # Material validation and retrieval
    materials_list = list(ITU_MATERIALS.items())

    def get_material_type(idx, name):
        if 0 <= idx < len(materials_list):
            return materials_list[idx][0]
        else:
            logger.warning(f"Invalid {name} material index {idx}. Using default.")
            return materials_list[0][0]

    ground_mat = get_material_type(ground_material_idx, "ground")
    rooftop_mat = get_material_type(rooftop_material_idx, "rooftop")
    wall_mat = get_material_type(wall_material_idx, "wall")

    logger.info(
        f"Generating scene for BBOX: {min_lon}, {min_lat}, {max_lon}, {max_lat}"
    )
    logger.info(f"Output Directory: {output_dir}")

    try:
        scene_instance = Scene()

        # Explicitly disabling LiDAR and DEM features
        scene_instance(
            points=points,
            data_dir=output_dir,
            hag_tiff_path=None,
            osm_server_addr="https://overpass-api.de/api/interpreter",
            lidar_calibration=False,
            generate_building_map=False,  # Optional, set to False for pure scene generation
            ground_material_type=ground_mat,
            rooftop_material_type=rooftop_mat,
            wall_material_type=wall_mat,
            lidar_terrain=False,
            dem_terrain=False,
            gen_lidar_terrain_only=False,
        )

        logger.info("Scene generation completed successfully.")

    except Exception as e:
        logger.error(f"Error during scene generation: {e}")
        raise RuntimeError(f"Scene generation failed: {e}")


if __name__ == "__main__":
    try:
        output_dir = get_project_root() / "scene"

        generate_scene_from_coords(
            min_lon=settings.geo2sigmap.min_lon,
            min_lat=settings.geo2sigmap.min_lat,
            max_lon=settings.geo2sigmap.max_lon,
            max_lat=settings.geo2sigmap.max_lat,
            output_dir=str(output_dir),
        )
    except Exception as e:
        print(e)

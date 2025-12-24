import numpy as np
import mitsuba as mi
from loguru import logger
from sionna.rt import RadioMapSolver, RadioMap, transform_mesh
from sionna.rt.scene import Scene
from app.config import settings, get_project_root
from app.geomap_processor.utils.mesh_utils import subdivide_mesh


class CoverageProcessor:
    """
    Handles computation of radio coverage maps on the terrain.
    """

    def __init__(self, scene: Scene):
        self.scene = scene
        self.settings = settings.sionnart.coverage

    def compute_coverage_map(self) -> RadioMap:
        """
        Computes the radio map using the terrain mesh as the measurement surface.
        """
        logger.info("Starting coverage map computation...")

        measurement_surface = self._prepare_measurement_surface()
        if measurement_surface is None:
            raise ValueError(
                "Could not create measurement surface. Ensure 'mesh-terrain' exists."
            )

        solver = RadioMapSolver()

        try:
            radio_map = solver(
                self.scene,
                measurement_surface=measurement_surface,
                max_depth=self.settings.max_depth,
                samples_per_tx=self.settings.samples_per_tx,
            )
            logger.success("Coverage map computation completed.")
            return radio_map
        except Exception as e:
            logger.error(f"Failed to compute coverage map: {e}")
            raise

    def _prepare_measurement_surface(self):
        """
        Finds the terrain mesh, creates a high-res version if needed, and loads it.
        """
        scene_dir = get_project_root() / "scene"
        mesh_dir = scene_dir / "mesh"
        original_ply = mesh_dir / "terrain.ply"
        subdivided_ply = mesh_dir / "terrain_subdivided.ply"

        # Check if we need to generate high-res mesh
        if not subdivided_ply.exists():
            logger.info(
                "Generating high-resolution terrain mesh for smoother coverage..."
            )
            success = subdivide_mesh(original_ply, subdivided_ply)
            if not success:
                logger.warning("Falling back to original terrain mesh.")
                subdivided_ply = original_ply
        else:
            logger.info("Using existing high-resolution terrain mesh.")

        try:
            ply_dict = {
                "type": "ply",
                "filename": str(subdivided_ply),
                "bsdf": {"type": "diffuse"},
            }
            surface = mi.load_dict(ply_dict)

            # Shift upwards by 1.5m to simulate user height
            transform_mesh(surface, translation=np.array([0, 0, 1.5]))

            return surface

        except Exception as e:
            logger.error(f"Error loading measurement surface: {e}")
            return None

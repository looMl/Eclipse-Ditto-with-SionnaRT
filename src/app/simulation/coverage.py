import numpy as np
import mitsuba as mi
from loguru import logger
from typing import Optional, Tuple
from sionna.rt import RadioMapSolver, RadioMap, transform_mesh
from sionna.rt.scene import Scene
from app.config import settings, get_project_root
from app.geomap_processor.utils.mesh_utils import subdivide_mesh
from app.geomap_processor.utils.vegetation_field import VegetationField
from app.simulation.vegetation_path_integrator import PathDepthIntegrator
from app.simulation.vegetation_attenuator import VegetationAttenuator


class CoverageProcessor:
    """
    Handles computation of radio coverage maps on the terrain.
    """

    def __init__(self, scene: Scene):
        self.scene = scene
        self.settings = settings.sionnart.coverage

    def compute_coverage_map(
        self, skip_vegetation: bool = False
    ) -> Tuple[RadioMap, Optional[np.ndarray]]:
        """
        Computes the radio map using the terrain mesh as the measurement surface.

        Returns
        -------
        radio_map : RadioMap
        attenuation_db : np.ndarray of shape [num_tx, num_faces], or None when
            vegetation correction is disabled or the vegetation field is absent.
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
        except Exception as e:
            logger.error(f"Failed to compute coverage map: {e}")
            raise

        attenuation_db = (
            None if skip_vegetation else self._compute_vegetation_attenuation(radio_map)
        )
        return radio_map, attenuation_db

    def _compute_vegetation_attenuation(self, radio_map) -> Optional[np.ndarray]:
        veg_cfg = getattr(settings.sionnart, "vegetation", None)
        if veg_cfg is None or not getattr(veg_cfg, "enabled", False):
            return None

        npz_path = get_project_root() / "scene" / "mesh" / "vegetation_field.npz"
        if not npz_path.exists():
            logger.warning(
                f"Vegetation field not found at {npz_path}; skipping attenuation."
            )
            return None

        field = VegetationField.load(npz_path)
        step_m = getattr(veg_cfg, "raster_step_m", 1.0)
        integrator = PathDepthIntegrator(field, step_m=step_m)
        attenuator = VegetationAttenuator(self.scene, field, integrator)

        freq_hz = getattr(veg_cfg, "frequency_hz", 1.8e9)
        leaf_state = getattr(veg_cfg, "leaf_state", "in_leaf")
        return attenuator.compute_attenuation_field(radio_map, freq_hz, leaf_state)

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

            # Translate by z 0.1 to avoid Z-fighting with the terrain
            transform_mesh(surface, translation=np.array([0, 0, 0.1]))

            return surface

        except Exception as e:
            logger.error(f"Error loading measurement surface: {e}")
            return None

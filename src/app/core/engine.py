from sionna.rt import PathSolver
from loguru import logger
from app.core.scene_manager import SceneManager
from app.core.renderer import SimulationRenderer
from app.utils.config import Settings


class SionnaRTEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.scene_manager = SceneManager(settings)
        self.renderer = SimulationRenderer(settings)
        self.solver = PathSolver()

    def run_simulation(self, rx_position: list, rx_orientation: list):
        """
        Runs a single simulation for a given receiver position and orientation.
        """
        # 1. Setup Scene
        self.scene_manager.add_receiver(rx_position, rx_orientation)

        try:
            # 2. Compute Paths
            paths = self._compute_paths()

            # 3. Render
            self.renderer.render(
                self.scene_manager.scene, self.scene_manager.camera, paths
            )
        finally:
            # 4. Cleanup
            self.scene_manager.remove_receiver("rx")

    def _compute_paths(self):
        try:
            sim_settings = self.settings.sionnart.simulation
            logger.info(
                f"Computing paths with max_depth={sim_settings.max_depth}, "
                f"num_samples={sim_settings.num_samples:.1e}..."
            )
            paths = self.solver(
                self.scene_manager.scene,
                max_depth=sim_settings.max_depth,
                samples_per_src=int(sim_settings.num_samples),
            )
            logger.info("Path computation finished.")
            return paths
        except Exception as e:
            logger.error(f"Critical error during path computation: {e}", exc_info=True)
            raise

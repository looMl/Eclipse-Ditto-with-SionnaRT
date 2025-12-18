from loguru import logger
from app.simulation.scene_manager import SceneManager
from app.simulation.renderer import SimulationRenderer
from app.config import Settings


class SionnaRTEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.scene_manager = SceneManager(settings)
        self.renderer = SimulationRenderer(settings)

    def run_visualization(self):
        """
        Renders the scene with all loaded transmitters.
        """
        logger.info("Running scene visualization...")
        self.renderer.render(self.scene_manager.scene, self.scene_manager.camera)
        logger.info("Visualization complete.")

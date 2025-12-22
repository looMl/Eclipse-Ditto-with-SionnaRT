from loguru import logger
from app.simulation.engine import SimulationEngine
from app.simulation.scene_manager import SceneManager
from app.simulation.renderer import SimulationRenderer


class SimulatorCLI:
    """Entry point for running simulations and renderings."""

    def run_visual_render(self):
        try:
            # 1. Initialize Engine
            SimulationEngine.initialize()

            # 2. Prepare Scene
            manager = SceneManager()
            scene = manager.load_scene()

            # 3. Render
            renderer = SimulationRenderer()
            renderer.render_rgb(scene)

        except Exception as e:
            logger.exception(f"Simulation execution failed: {e}")


if __name__ == "__main__":
    cli = SimulatorCLI()
    cli.run_visual_render()

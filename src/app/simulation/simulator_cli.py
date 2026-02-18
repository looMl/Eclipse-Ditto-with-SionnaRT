import argparse
import sys
from loguru import logger
from app.simulation.engine import SimulationEngine
from app.simulation.scene_manager import SceneManager
from app.simulation.renderer import SimulationRenderer
from app.simulation.coverage import CoverageProcessor


class SimulatorCLI:
    """
    Command-line interface for the SionnaRT simulation environment.
    Supports standard visual rendering and coverage map analysis.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SionnaRT Simulation CLI")
        self.parser.add_argument(
            "mode",
            choices=["render", "coverage"],
            nargs="?",
            default="render",
            help="Operation mode: 'render' for visual output, 'coverage' for signal analysis.",
        )

    def execute(self):
        args = self.parser.parse_args()

        try:
            # 1. Initialize Engine
            SimulationEngine.initialize()

            # 2. Load Scene
            manager = SceneManager()
            scene = manager.load_scene()

            # 3. Execute Mode
            renderer = SimulationRenderer()

            if args.mode == "coverage":
                self._run_coverage(scene, renderer)
            else:
                self._run_render(scene, renderer)

        except Exception as e:
            logger.exception(f"Simulation failed: {e}")
            sys.exit(1)

    def _run_render(self, scene, renderer: SimulationRenderer):
        logger.info("Mode: Visual Render")
        renderer.render_visual(scene)

    def _run_coverage(self, scene, renderer: SimulationRenderer):
        logger.info("Mode: Coverage Analysis")
        processor = CoverageProcessor(scene)
        radio_map = processor.compute_coverage_map()
        renderer.render_coverage(scene, radio_map)


if __name__ == "__main__":
    cli = SimulatorCLI()
    cli.execute()

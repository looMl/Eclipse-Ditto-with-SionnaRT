import argparse
from loguru import logger
from app.simulation.engine import SimulationEngine
from app.simulation.scene_manager import SceneManager
from app.simulation.rendering.renderer import SimulationRenderer
from app.simulation.coverage import CoverageProcessor


class SimulatorCLI:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SionnaRT Simulation CLI")
        self.parser.add_argument(
            "mode",
            choices=["render", "coverage"],
            nargs="?",
            default="render",
            help="Operation mode: 'render' for visual output, 'coverage' for signal analysis.",
        )
        self.parser.add_argument(
            "--no-vegetation",
            action="store_true",
            help="Skip vegetation attenuation correction (coverage mode only).",
        )

    def execute(self) -> None:
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
                self._run_coverage(scene, renderer, skip_vegetation=args.no_vegetation)
            else:
                self._run_render(scene, renderer)

        except Exception as e:
            logger.exception(f"Simulation failed: {e}")
            raise

    def _run_render(self, scene, renderer: SimulationRenderer):
        logger.info("Mode: Visual Render")
        renderer.render_visual(scene)

    def _run_coverage(
        self, scene, renderer: SimulationRenderer, skip_vegetation: bool = False
    ):
        logger.info("Mode: Coverage Analysis")
        processor = CoverageProcessor(scene)
        radio_map, attenuation_db = processor.compute_coverage_map(
            skip_vegetation=skip_vegetation
        )
        renderer.render_coverage(scene, radio_map, attenuation_db)


if __name__ == "__main__":
    cli = SimulatorCLI()
    cli.execute()

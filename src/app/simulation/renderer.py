from pathlib import Path
from loguru import logger
from app.config import get_project_root, Settings


class SimulationRenderer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.renders_dir = get_project_root() / "app" / "renders"
        self.renders_dir.mkdir(parents=True, exist_ok=True)

    def render(self, scene, camera, paths=None):
        logger.info("Starting scene rendering.")
        output_path = self._get_next_filename(self.renders_dir, "paths_render", "png")

        logger.info(f"Rendering scene to {output_path}...")

        render_settings = self.settings.sionnart.rendering

        scene.render_to_file(
            camera=camera,
            filename=str(output_path),
            paths=paths,
            show_devices=render_settings.show_devices,
            num_samples=render_settings.num_samples,
            resolution=render_settings.resolution,
        )
        logger.info("Rendering complete.")

    @staticmethod
    def _get_next_filename(directory: Path, base_name: str, extension: str) -> Path:
        i = 1
        while True:
            filename = f"{base_name}_{i}.{extension}"
            file_path = directory / filename
            if not file_path.exists():
                return file_path
            i += 1

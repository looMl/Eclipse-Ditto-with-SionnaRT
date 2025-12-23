import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sionna.rt import Camera
from sionna.rt.renderer import visual_scene_from_wireless_scene
from app.config import settings, get_project_root


class SimulationRenderer:
    """
    Handles RGB visual rendering of the scene.
    """

    def __init__(self):
        self.renders_dir = get_project_root() / "renders"
        self._ensure_renders_dir()

    def render_rgb(self, scene):
        """Performs a manual RGB render pass using Mitsuba's rendering."""
        render_settings = settings.sionnart.rendering
        logger.info(
            f"Visual Rendering: {render_settings.resolution}, samples={render_settings.num_samples}"
        )

        with mi.util.scoped_set_variant("cuda_ad_rgb"):
            sensor = self._create_sensor(scene, render_settings)
            visual_scene = self._build_visual_scene(scene, sensor)

            image_tensor = mi.render(visual_scene, spp=render_settings.num_samples)

            data = np.array(image_tensor)

        output_path = self._get_next_filename()
        self._save_image(data, output_path)

    def _save_image(self, data, path: Path):
        plt.imsave(str(path), np.clip(data, 0, 1))
        logger.success(f"Render saved to {path}")

    def _ensure_renders_dir(self):
        self.renders_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_filename(self) -> Path:
        idx = len(list(self.renders_dir.glob("render_*.png"))) + 1
        return self.renders_dir / f"render_{idx}.png"

    def _create_sensor(self, scene, render_settings) -> mi.Sensor:
        my_cam = Camera(
            position=settings.sionnart.camera.position,
            look_at=settings.sionnart.camera.look_at,
        )

        matrix = my_cam.world_transform.matrix.numpy().squeeze().reshape(4, 4)

        return mi.load_dict(
            {
                "type": "perspective",
                "to_world": mi.ScalarTransform4f(matrix.tolist()),
                "fov": 45.0,
                "film": {
                    "type": "hdrfilm",
                    "width": render_settings.resolution[0],
                    "height": render_settings.resolution[1],
                    "pixel_format": "rgb",
                    "rfilter": {"type": "gaussian"},
                },
            }
        )

    def _build_visual_scene(self, scene, sensor):
        """Converts wireless scene to visual scene and adds transmitter markers."""
        visual_dict = visual_scene_from_wireless_scene(scene, sensor, max_depth=8)

        # Ensure the sky is white and add artificaial sun light
        if "integrator" in visual_dict:
            visual_dict["integrator"]["hide_emitters"] = False

        visual_dict["sun"] = {
            "type": "directional",
            "direction": [0.5, 0.5, -1.0],
            "irradiance": {
                "type": "rgb",
                "value": [1.0, 1.0, 1.0],
            },
        }

        # Add visual markers for transmitters
        marker_bsdf = {
            "type": "twosided",
            "nested": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [1.0, 0.0, 0.0]},
            },
        }
        for i, tx in enumerate(scene.transmitters.values()):
            p = tx.position.numpy().squeeze()
            visual_dict[f"marker-{i}"] = {
                "type": "sphere",
                "center": [float(p[0]), float(p[1]), float(p[2])],
                "radius": 15.0,
                "bsdf": marker_bsdf,
            }
        return mi.load_dict(visual_dict)

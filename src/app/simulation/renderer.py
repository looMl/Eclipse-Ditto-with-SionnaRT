import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import sionna.rt as rt
from sionna.rt.renderer import (
    get_overlay_scene,
    unmultiply_alpha,
    scoped_set_log_level,
)

from app.config import settings, get_project_root
from app.simulation.visual_builder import VisualSceneBuilder
from app.simulation.shading_utils import prepare_gouraud_shading_for_radio_map


class SimulationRenderer:
    """
    Handles rendering of the scene, including standard RGB passes and
    complex coverage map overlays using custom shading pipelines.
    """

    def __init__(self):
        self.renders_dir = get_project_root() / "renders"
        self._ensure_renders_dir()

    def render_visual(self, scene):
        """
        Standard visual render (RGB).
        """
        settings_render = settings.sionnart.rendering
        logger.info(f"Starting Visual Render ({settings_render.resolution})...")

        with mi.util.scoped_set_variant("cuda_ad_rgb"):
            # Build Scene
            sensor = VisualSceneBuilder.create_sensor(
                settings_render.resolution, pixel_format="rgb"
            )
            visual_scene = VisualSceneBuilder.build_visual_scene(scene, sensor)

            # Render
            image = mi.render(visual_scene, spp=settings_render.num_samples)

            # Save
            self._save_image(np.array(image), self._get_next_filename("render_"))

    def render_coverage(self, scene, radio_map):
        """
        Renders the scene with a high-quality radio map overlay using Gouraud shading.
        """
        settings_render = settings.sionnart.rendering
        settings_cov = settings.sionnart.coverage
        logger.info("Starting Coverage Render with Gouraud shading...")

        with mi.util.scoped_set_variant("cuda_ad_rgb"):
            # 1. Setup
            sensor = VisualSceneBuilder.create_sensor(
                settings_render.resolution, pixel_format="rgba"
            )
            visual_scene = VisualSceneBuilder.build_visual_scene(
                scene, sensor, max_depth=settings_cov.max_depth
            )

            # 2. Render Base Scene
            main_image = mi.render(
                visual_scene, spp=settings_render.num_samples
            ).numpy()

            # 3. Create Overlay Scene (Radio Map)
            overlay_dict = get_overlay_scene(
                scene,
                sensor,
                show_sources=settings_render.show_devices,
                show_targets=settings_render.show_devices,
                radio_map=radio_map,
                rm_metric=settings_cov.metric,
                rm_vmin=settings_cov.vmin,
                rm_vmax=settings_cov.vmax,
                rm_cmap="viridis",
            )

            if not overlay_dict:
                logger.warning("No overlay content generated.")
                self._save_image(main_image, self._get_next_filename("coverage_"))
                return

            # 4. Apply Custom Gouraud Shading
            if isinstance(radio_map, rt.MeshRadioMap):
                logger.debug("Applying custom Gouraud shading to coverage mesh.")
                overlay_dict["radio-map"] = prepare_gouraud_shading_for_radio_map(
                    radio_map,
                    settings_cov.metric,
                    settings_cov.vmin,
                    settings_cov.vmax,
                    "viridis",
                )

            # 5. Render Overlay & Composite
            result = self._render_and_composite_overlay(
                visual_scene,
                overlay_dict,
                sensor,
                main_image,
                settings_cov.max_depth,
                settings_render.num_samples,
            )

            # Save
            self._save_image(result, self._get_next_filename("coverage_"))

    def _render_and_composite_overlay(
        self, visual_scene, overlay_dict, sensor, main_image, max_depth, spp
    ):
        """
        Renders the overlay scene and composites it onto the main image using depth occlusion.
        """
        # Load Overlay Scene (suppress warnings)
        with scoped_set_log_level(mi.LogLevel.Error):
            overlay_scene = mi.load_dict(overlay_dict)

        # Integrators
        depth_integrator = mi.load_dict({"type": "depth"})
        overlay_integrator = mi.load_dict(
            {
                "type": "path",
                "max_depth": max_depth,
                "hide_emitters": False,
            }
        )

        # Render Passes
        depth_main = unmultiply_alpha(
            mi.render(
                visual_scene, sensor=sensor, integrator=depth_integrator, spp=4
            ).numpy()
        )
        overlay_image = mi.render(
            overlay_scene, sensor=sensor, integrator=overlay_integrator, spp=spp
        ).numpy()
        depth_overlay = unmultiply_alpha(
            mi.render(
                overlay_scene, sensor=sensor, integrator=depth_integrator, spp=spp
            ).numpy()
        )

        # Composition Logic
        alpha_main = main_image[:, :, 3]
        alpha_overlay = overlay_image[:, :, 3]
        composite = overlay_image + main_image * (1 - alpha_overlay[:, :, None])

        # Prefer overlay if it's closer or if the main scene is transparent
        prefer_overlay = (alpha_main[:, :, None] < 0.1) & (
            depth_main < 2 * depth_overlay
        )
        # Also prefer overlay if depths are very similar (z-fighting fix for terrain overlay)
        prefer_overlay |= np.abs(depth_main - depth_overlay) < 0.01 * np.abs(depth_main)

        result = np.where(
            (alpha_main[:, :, None] > 0)
            & (depth_main < depth_overlay)
            & (~prefer_overlay),
            main_image,
            composite,
        )
        result[:, :, 3] = np.maximum(alpha_main, composite[:, :, 3])

        return result

    def _save_image(self, data, path: Path):
        plt.imsave(str(path), np.clip(data, 0, 1))
        logger.success(f"Render saved to {path}")

    def _ensure_renders_dir(self):
        self.renders_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_filename(self, prefix: str) -> Path:
        idx = len(list(self.renders_dir.glob(f"{prefix}*.png"))) + 1
        return self.renders_dir / f"{prefix}{idx}.png"

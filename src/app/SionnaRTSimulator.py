import os
import sys
import json
import argparse
from pathlib import Path
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, PathSolver
from loguru import logger
from app.utils.config import settings as cfg, get_project_root

import matplotlib

# -1: CPU Only execution - 0: GPU only if compatible
# Newer versions of SionnaRT use Dr.Jit which requires your GPU to have a Compute Capability (SM) > 7.0
# In my case, my GPU is not supported so I have to rely on the CPU (slower)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

matplotlib.use("Agg")


class SionnaRTSimulator:
    def __init__(self, scene_filename: str = "povo_scene.xml"):
        self._scene = self._load_scene(scene_filename)
        self._camera = self._setup_camera()
        self._configure_antenna_arrays()
        self._add_transmitter()

    def _load_scene(self, scene_filename: str):
        filepath = get_project_root() / "scene" / scene_filename
        try:
            logger.info(f"Loading scene from: {filepath}")
            return load_scene(str(filepath))
        except FileNotFoundError:
            logger.critical(
                f"Scene file not found at '{filepath}'. Cannot initialize simulator."
            )
            raise

    def _setup_camera(self):
        try:
            camera = Camera(
                position=cfg.sionnart.camera.position,
                look_at=cfg.sionnart.camera.look_at,
            )
            logger.info("Camera object created successfully.")
            return camera
        except Exception as e:
            logger.error(f"Failed to initialize Camera: {e}", exc_info=True)
            raise

    def _configure_antenna_arrays(self):
        # Antenna parameters (rows, cols, spacing, pattern, polarization) significantly affect simulation results.
        logger.info("Configuring TX/RX antenna arrays.")

        self._scene.tx_array = PlanarArray(
            num_rows=4,
            num_cols=4,  # 4x4 array
            vertical_spacing=0.5,
            horizontal_spacing=0.5,  # Spacing in wavelengths
            pattern="tr38901",  # Standard 3GPP antenna pattern
            polarization="V",  # Vertical polarization
        )

        self._scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,  # Single element
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",  # Simple dipole pattern
            polarization="cross",  # Cross-polarized to capture different signal components
        )
        logger.info("Antenna arrays configured.")

    def _add_transmitter(self):
        pos = cfg.sionnart.transmitter.position
        logger.info(f"Added TX at {pos}.")
        self._tx = Transmitter("tx", pos)
        self._scene.add(self._tx)

        # Set frequency (2.14 GHz) and synthetic array mode
        self._scene.frequency = 2.14e9
        self._scene.synthetic_array = True

        logger.info(
            f"Scene frequency set to {(self._scene.frequency / 1e9).numpy().item():.2f} GHz."
        )

    def _compute_paths(self):
        try:
            sim_settings = cfg.sionnart.simulation
            logger.info(
                f"Computing paths with max_depth={sim_settings.max_depth}, "
                f"num_samples={sim_settings.num_samples:.1e}..."
            )
            solver = PathSolver()
            paths = solver(
                self._scene,
                max_depth=sim_settings.max_depth,
                samples_per_src=int(sim_settings.num_samples),
            )
            logger.info("Path computation finished.")
            return paths
        except Exception as e:
            logger.error(f"Critical error during path computation: {e}", exc_info=True)
            raise

    def _render_and_save(self, paths):
        logger.info("Starting scene rendering.")

        renders_dir = get_project_root() / "app" / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)

        output_path = self._get_next_filename(renders_dir, "paths_render", "png")

        logger.info(f"Rendering scene to {output_path}...")
        self._scene.render_to_file(
            camera=self._camera,
            filename=str(output_path),
            paths=paths,
            show_devices=cfg.sionnart.rendering.show_devices,
            num_samples=cfg.sionnart.rendering.num_samples,
            resolution=cfg.sionnart.rendering.resolution,
        )
        logger.info("Rendering complete.")

    def _get_next_filename(
        self, directory: Path, base_name: str, extension: str
    ) -> Path:
        i = 1
        while True:
            filename = f"{base_name}_{i}.{extension}"
            file_path = directory / filename
            if not file_path.exists():
                return file_path
            i += 1

    def run_simulation(self, rx_position: list, rx_orientation: list):
        """
        Runs a single simulation for a given receiver position and orientation.
        """
        logger.info(f"Added RX at {rx_position} with orientation {rx_orientation}.")
        rx = Receiver("rx", rx_position, rx_orientation)
        self._scene.add(rx)
        self._tx.look_at(rx)

        try:
            paths = self._compute_paths()
            self._render_and_save(paths)
        finally:
            # Ensure the dynamic receiver is removed after each simulation to prevent scene pollution and memory leaks.
            self._scene.remove("rx")
            logger.debug("Receiver 'rx' removed from the scene.")


if __name__ == "__main__":

    def _validate_coordinate(coord_list: list, name: str) -> None:
        """
        Validates that the input is a list of 3 numbers.
        """
        if (
            not isinstance(coord_list, list)
            or len(coord_list) != 3
            or not all(isinstance(x, (int, float)) for x in coord_list)
        ):
            raise ValueError(
                f"Argument '{name}' must be a JSON list of 3 numbers [x, y, z]."
            )

    def parse_arguments():
        """Parses command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Run a single SionnaRT simulation."
        )

        parser.add_argument(
            "--position", type=str, required=True, help="JSON list [x, y, z]"
        )
        parser.add_argument(
            "--orientation", type=str, required=True, help="JSON list [x, y, z]"
        )

        args = parser.parse_args()
        logger.info("Parsing arguments...")

        try:
            rx_pos = json.loads(args.position)
            _validate_coordinate(rx_pos, "position")

            rx_ori = json.loads(args.orientation)
            _validate_coordinate(rx_ori, "orientation")

            logger.info(f"Parsed arguments - Position: {rx_pos}, Orientation: {rx_ori}")
            return rx_pos, rx_ori
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid arguments provided: {e}", exc_info=True)
            raise

    logger.info("--- Running SionnaRT Simulation ---")
    try:
        rx_pos, rx_ori = parse_arguments()
        simulator = SionnaRTSimulator()
        simulator.run_simulation(rx_pos, rx_ori)
        logger.info("--- Standalone simulation finished successfully. ---")
    except Exception as e:
        logger.critical(
            f"A critical error occurred during standalone execution: {e}", exc_info=True
        )
        sys.exit(1)

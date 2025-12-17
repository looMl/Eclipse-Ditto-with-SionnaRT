import sys
import json
import argparse
import os
import matplotlib
from loguru import logger

from app.config import settings as cfg
from app.simulation.engine import SionnaRTEngine

# -1: CPU Only execution - 0: GPU only if compatible
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

matplotlib.use("Agg")


class SionnaRTSimulator:
    """
    The primary orchestration controller for SionnaRT simulations.

    This class serves as the main entry point for initializing and executing
    simulation tasks. It encapsulates the complexity of the sionna engine
    setup (Scene Manager, Renderer, Path Solver) and makes sure that the simulation
    runs accordingly to the parameters defined in the global configuration.
    """

    def __init__(self):
        self.engine = SionnaRTEngine(cfg)

    def run_simulation(self, rx_position: list, rx_orientation: list):
        self.engine.run_simulation(rx_position, rx_orientation)


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


def _parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a single SionnaRT simulation.")

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


def run_cli():
    logger.info("--- Running SionnaRT Simulation ---")
    try:
        rx_pos, rx_ori = _parse_arguments()
        simulator = SionnaRTSimulator()
        simulator.run_simulation(rx_pos, rx_ori)
        logger.info("--- Standalone simulation finished successfully. ---")
    except Exception as e:
        logger.critical(
            f"A critical error occurred during standalone execution: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    run_cli()

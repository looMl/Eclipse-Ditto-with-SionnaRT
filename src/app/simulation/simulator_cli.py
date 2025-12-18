import sys
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
    The primary orchestration controller for SionnaRT simulations tests.
    """

    def __init__(self):
        self.engine = SionnaRTEngine(cfg)

    def run_visualization(self):
        self.engine.run_visualization()


def run_cli():
    logger.info("--- Running SionnaRT Visualization ---")
    try:
        simulator = SionnaRTSimulator()
        simulator.run_visualization()
        logger.info("--- Standalone visualization finished successfully. ---")
    except Exception as e:
        logger.critical(
            f"A critical error occurred during standalone execution: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    run_cli()

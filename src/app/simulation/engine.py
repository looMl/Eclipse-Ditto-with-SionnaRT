import mitsuba as mi
from loguru import logger
from sionna.rt.radio_materials import ITURadioMaterial, RadioMaterial


class SimulationEngine:
    """
    Manages the core SionnaRT/Mitsuba engine variant initialization.
    """

    @staticmethod
    def initialize(variant: str = "cuda_ad_mono_polarized"):
        try:
            logger.debug(f"Initializing Simulation Engine with variant: {variant}")
            mi.set_variant(variant)
            SimulationEngine._register_plugins()
        except Exception as e:
            logger.error(f"Failed to initialize Simulation Engine: {e}")
            raise

    @staticmethod
    def _register_plugins():
        """Registers custom SionnaRT BSDF plugins for the current variant."""
        plugins = {
            "itu-radio-material": ITURadioMaterial,
            "radio-material": RadioMaterial,
        }
        for name, cls in plugins.items():
            try:
                mi.register_bsdf(name, lambda props, c=cls: c(props=props))
            except Exception as e:
                logger.debug(f"Plugin '{name}' registration: {e}")

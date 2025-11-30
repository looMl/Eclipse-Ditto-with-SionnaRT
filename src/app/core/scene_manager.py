from loguru import logger
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from app.utils.config import get_project_root, Settings


class SceneManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.scene = self._load_scene()
        self.camera = self._setup_camera()
        self.tx = None

        self._configure_antenna_arrays()
        self._setup_transmitter()

    def _load_scene(self):
        scene_filename = self.settings.sionnart.scene_name
        filepath = get_project_root() / "scene" / scene_filename
        try:
            logger.info(f"Loading scene from: {filepath}")
            return load_scene(str(filepath))
        except FileNotFoundError:
            logger.critical(
                f"Scene file not found at '{filepath}'. Cannot initialize simulator."
            )
            raise

    def _setup_camera(self) -> Camera:
        try:
            camera = Camera(
                position=self.settings.sionnart.camera.position,
                look_at=self.settings.sionnart.camera.look_at,
            )
            logger.info("Camera object created successfully.")
            return camera
        except Exception as e:
            logger.error(f"Failed to initialize Camera: {e}", exc_info=True)
            raise

    def _configure_antenna_arrays(self):
        logger.info("Configuring TX/RX antenna arrays.")

        self.scene.tx_array = PlanarArray(
            num_rows=4,
            num_cols=4,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V",
        )

        self.scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="cross",
        )
        logger.info("Antenna arrays configured.")

    def _setup_transmitter(self):
        pos = self.settings.sionnart.transmitter.position
        logger.info(f"Added TX at {pos}.")
        self.tx = Transmitter("tx", pos)
        self.scene.add(self.tx)

        # Set frequency and synthetic array mode
        self.scene.frequency = 2.14e9
        self.scene.synthetic_array = True

        freq_ghz = (self.scene.frequency / 1e9).numpy().item()
        logger.info(f"Scene frequency set to {freq_ghz:.2f} GHz.")

    def add_receiver(self, position: list, orientation: list) -> Receiver:
        logger.debug(f"Adding RX at {position} with orientation {orientation}.")
        rx = Receiver("rx", position, orientation)
        self.scene.add(rx)
        self.tx.look_at(rx)
        return rx

    def remove_receiver(self, name: str = "rx"):
        self.scene.remove(name)
        logger.debug(f"Receiver '{name}' removed from the scene.")

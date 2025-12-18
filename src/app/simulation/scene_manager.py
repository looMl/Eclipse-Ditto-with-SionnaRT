import json
from loguru import logger
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from app.config import get_project_root, Settings
from app.geomap_processor.processors.dem_processor import DemProcessor


class SceneManager:
    # Antenna Configuration
    TX_ROWS = 4
    TX_COLS = 4
    RX_ROWS = 1
    RX_COLS = 1
    SPACING = 0.5
    FREQUENCY = 3.5e9

    def __init__(self, settings: Settings):
        self.settings = settings
        self.scene = self._load_scene()
        self.camera = self._setup_camera()
        self.transmitters = []

        self._configure_antenna_arrays()

        # Configure global scene parameters
        self.scene.frequency = self.FREQUENCY
        self.scene.synthetic_array = True

        # Load dynamic transmitters
        self.load_transmitters()

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
            num_rows=self.TX_ROWS,
            num_cols=self.TX_COLS,
            vertical_spacing=self.SPACING,
            horizontal_spacing=self.SPACING,
            pattern="tr38901",
            polarization="V",
        )

        self.scene.rx_array = PlanarArray(
            num_rows=self.RX_ROWS,
            num_cols=self.RX_COLS,
            vertical_spacing=self.SPACING,
            horizontal_spacing=self.SPACING,
            pattern="dipole",
            polarization="cross",
        )
        logger.info("Antenna arrays configured.")

    def load_transmitters(self):
        """Loads transmitters from JSON and adds them to the scene."""
        json_path = get_project_root() / "ditto" / "things" / "transmitters.json"
        if not json_path.exists():
            logger.warning(f"Transmitters file not found at {json_path}")
            return

        # Calculate scene origin from config
        g = self.settings.geo2sigmap
        origin_lon = (g.min_lon + g.max_lon) / 2.0
        origin_lat = (g.min_lat + g.max_lat) / 2.0

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            for item in data:
                tid = item.get("thingId", "unknown")
                loc = item.get("attributes", {}).get("location", {})
                lat = loc.get("latitude")
                lon = loc.get("longitude")
                h = loc.get("height_m")

                if lat is not None and lon is not None:
                    x, y = DemProcessor.global_to_local(
                        lon, lat, origin_lon, origin_lat
                    )
                    pos = [x, y, h]
                    # Sanitize name for Sionna/Mitsuba
                    safe_name = tid.replace(".", "_").replace(":", "_")
                    tx = Transmitter(name=safe_name, position=pos)
                    self.scene.add(tx)
                    self.transmitters.append(tx)
                    logger.debug(f"Added Transmitter {safe_name} at {pos}")

            logger.info(f"Loaded {len(self.transmitters)} transmitters into the scene.")

        except Exception as e:
            logger.error(f"Failed to load transmitters: {e}")

    def add_receiver(self, position: list, orientation: list) -> Receiver:
        logger.debug(f"Adding RX at {position} with orientation {orientation}.")
        rx = Receiver("rx", position, orientation)
        self.scene.add(rx)
        self.tx.look_at(rx)
        return rx

    def remove_receiver(self, name: str = "rx"):
        self.scene.remove(name)
        logger.debug(f"Receiver '{name}' removed from the scene.")

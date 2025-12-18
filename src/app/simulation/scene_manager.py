import json
import math
from typing import Tuple, Dict, Any
from loguru import logger
from sionna.rt import load_scene, Transmitter, PlanarArray, Camera
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
        self.transmitters = []

        # Configure global scene parameters
        self.scene.frequency = self.FREQUENCY
        self.scene.synthetic_array = True
        self._configure_antenna_arrays()

        try:
            self.camera = Camera(
                position=self.settings.sionnart.camera.position,
                look_at=self.settings.sionnart.camera.look_at,
            )
            logger.info("Camera initialized from config settings.")
        except Exception as e:
            logger.error(f"Failed to initialize Camera: {e}", exc_info=True)
            raise

        # Load dynamic transmitters
        self.load_transmitters()

    def _get_scene_origin(self) -> Tuple[float, float]:
        """Calculates the scene origin."""
        g = self.settings.geo2sigmap
        return (g.min_lon + g.max_lon) / 2.0, (g.min_lat + g.max_lat) / 2.0

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

    def load_transmitters(self):
        """Loads transmitters from JSON and adds them to the scene."""
        json_path = get_project_root() / "ditto" / "things" / "transmitters.json"
        if not json_path.exists():
            logger.warning(f"Transmitters file not found at {json_path}")
            return

        origin_lon, origin_lat = self._get_scene_origin()

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            for item in data:
                self._add_single_transmitter(item, origin_lon, origin_lat)

            logger.info(f"Loaded {len(self.transmitters)} transmitters into the scene.")

        except Exception as e:
            logger.error(f"Failed to load transmitters: {e}")

    def _add_single_transmitter(
        self, item: Dict[str, Any], origin_lon: float, origin_lat: float
    ):
        """Processes a single transmitter item and adds it to the scene."""
        loc = item.get("attributes", {}).get("location", {})
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        h = loc.get("height_m")

        if lat is None or lon is None:
            return

        # 1. Calculate Position
        x, y = DemProcessor.global_to_local(lon, lat, origin_lon, origin_lat)
        pos = [x, y, h]

        # 2. Calculate Orientation
        props = item.get("features", {}).get("configuration", {}).get("properties", {})
        azimuth = props.get("azimuth_deg", 0.0)
        tilt = props.get("mechanical_tilt", 0.0)

        yaw = math.radians(90.0 - azimuth)
        pitch = math.radians(-tilt)
        orientation = [yaw, pitch, 0.0]

        # 3. Create and Add Transmitter
        tid = item.get("thingId", "unknown")
        # Sanitize name for Sionna/Mitsuba
        safe_name = tid.replace(".", "_").replace(":", "_")

        try:
            tx = Transmitter(name=safe_name, position=pos, orientation=orientation)
            self.scene.add(tx)
            self.transmitters.append(tx)
            logger.debug(
                f"Added TX '{safe_name}' at {pos} (Yaw: {math.degrees(yaw):.1f}, Pitch: {math.degrees(pitch):.1f})"
            )
        except Exception as e:
            logger.warning(f"Failed to add transmitter {safe_name}: {e}")

import json
import pyproj
import numpy as np
import sionna.rt
from loguru import logger
from app.config import settings, get_project_root
from app.geomap_processor.data.scene_updater import SceneXMLUpdater


class SceneManager:
    """
    Handles scene loading, patching, and asset management.
    """

    def __init__(self):
        self.scene_dir = get_project_root() / "scene"
        self.scene_path = self.scene_dir / settings.sionnart.scene_name
        self.transmitters_json = (
            get_project_root() / "ditto" / "things" / "transmitters.json"
        )

    def load_scene(self) -> sionna.rt.Scene:
        """Loads and prepares the SionnaRT scene."""
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Scene file not found: {self.scene_path}")

        logger.info(f"Loading scene: {self.scene_path}")
        scene = sionna.rt.load_scene(str(self.scene_path))

        self._patch_visual_colors(scene)
        self._load_transmitters(scene)

        return scene

    def _patch_visual_colors(self, scene: sionna.rt.Scene):
        """Restores visual colors from XML to the loaded Sionna materials."""
        updater = SceneXMLUpdater(self.scene_path)
        xml_colors = updater.get_material_colors()

        for mat in scene.radio_materials.values():
            if mat.id() in xml_colors:
                mat.color = xml_colors[mat.id()]

    def _load_transmitters(self, scene: sionna.rt.Scene):
        """Loads transmitters from json file and positions them in the scene."""
        if not self.transmitters_json.exists():
            logger.warning("Transmitters registry not found.")
            return

        updater = SceneXMLUpdater(self.scene_path)
        proj_info = updater.get_projection_info()

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", proj_info["utm_zone"], always_xy=True
        )
        ox, oy = transformer.transform(proj_info["center_lon"], proj_info["center_lat"])

        with open(self.transmitters_json, "r") as f:
            data = json.load(f)

        # Set default array if needed
        if scene.tx_array is None:
            scene.tx_array = sionna.rt.PlanarArray(
                num_rows=8,
                num_cols=2,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="tr38901",
                polarization="VH",
            )

        for item in data:
            loc = item.get("attributes", {}).get("location", {})
            height = float(loc.get("height_m"))

            px, py = transformer.transform(loc["longitude"], loc["latitude"])

            feat = (
                item.get("features", {}).get("configuration", {}).get("properties", {})
            )
            azimuth = float(feat.get("azimuth_deg", 0.0))
            tilt = float(feat.get("mechanical_tilt", 0.0))

            tx = sionna.rt.Transmitter(
                name=str(item["thingId"]).replace(".", "_").replace(":", "_"),
                position=[px - ox, py - oy, height],
                orientation=[
                    (90.0 - azimuth) * np.pi / 180.0,  # Yaw
                    tilt * np.pi / 180.0,  # Pitch
                    0.0,
                ],
                power_dbm=float(feat.get("transmit_power_dbm", 40.0)),
            )
            tx.display_radius = 15.0
            tx.color = (1.0, 0.0, 0.0)
            scene.add(tx)

        logger.info(f"Loaded {len(data)} transmitters into scene.")

import json
from typing import Tuple
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
        self._apply_scattering(scene)
        self._load_transmitters(scene)

        return scene

    def get_transformer(self) -> Tuple[pyproj.Transformer, Tuple[float, float]]:
        """Returns the transformer and origin offset for coordinate conversion."""
        updater = SceneXMLUpdater(self.scene_path)
        proj_info = updater.get_projection_info()

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", proj_info["utm_zone"], always_xy=True
        )
        ox, oy = transformer.transform(proj_info["center_lon"], proj_info["center_lat"])
        return transformer, (ox, oy)

    def _apply_scattering(self, scene: sionna.rt.Scene):
        ds_cfg = getattr(settings.sionnart, "diffuse_scattering", None)
        if ds_cfg is None or not getattr(ds_cfg, "enabled", False):
            return
        nu = float(getattr(ds_cfg, "scattering_coefficient", 0.25))
        for mat in scene.radio_materials.values():
            mat.scattering_coefficient = nu
        logger.info(
            f"Applied diffuse scattering coefficient v={nu} to {len(scene.radio_materials)} materials."
        )

    def _patch_visual_colors(self, scene: sionna.rt.Scene):
        """Restores visual colors from XML to the loaded Sionna materials."""
        updater = SceneXMLUpdater(self.scene_path)
        xml_colors = updater.get_material_colors()

        for mat in scene.radio_materials.values():
            if mat.id() in xml_colors:
                mat.color = xml_colors[mat.id()]

    def _load_transmitters(self, scene: sionna.rt.Scene):
        """Loads transmitters from json file and positions them in the scene.

        Each JSON item represents one antenna site with sector_N features.
        One Sionna transmitter is created per sector so the simulation uses
        all 3 cells per site.
        """
        if not self.transmitters_json.exists():
            logger.warning("Transmitters registry not found.")
            return

        transformer, (ox, oy) = self.get_transformer()

        with open(self.transmitters_json, "r") as f:
            data = json.load(f)

        # Standard 4G/5G Sub-6 GHz Sector Antenna (e.g., 2T2R configuration)
        if scene.tx_array is None:
            scene.tx_array = sionna.rt.PlanarArray(
                num_rows=4,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="tr38901",
                polarization="VH",
            )

        tx_count = 0
        for item in data:
            loc = item.get("attributes", {}).get("location", {})
            height = float(loc.get("height_m", 30.0))
            px, py = transformer.transform(loc["longitude"], loc["latitude"])
            base_name = str(item["thingId"]).replace(".", "_").replace(":", "_")

            for sector_key, sector_data in item.get("features", {}).items():
                if not sector_key.startswith("sector_"):
                    continue
                props = sector_data.get("properties", {})
                azimuth = float(props.get("azimuth_deg", 0.0))
                mechanical_tilt = float(props.get("mechanical_tilt", 0.0))
                electrical_tilt = float(props.get("electrical_tilt_deg", 0.0))
                tilt = -(mechanical_tilt + electrical_tilt)

                tx = sionna.rt.Transmitter(
                    name=f"{base_name}__{sector_key}",
                    position=[px - ox, py - oy, height],
                    orientation=[
                        (90.0 - azimuth) * np.pi / 180.0,  # Yaw
                        tilt * np.pi / 180.0,  # Pitch
                        0.0,
                    ],
                    power_dbm=float(props.get("transmit_power_dbm", 40.0)),
                )
                tx.display_radius = 15.0
                tx.color = (1.0, 0.0, 0.0)
                scene.add(tx)
                tx_count += 1

        logger.info(f"Loaded {tx_count} transmitters from {len(data)} antenna sites.")

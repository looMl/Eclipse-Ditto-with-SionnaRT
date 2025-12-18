import json
import random
from typing import List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

import osmnx as ox
import trimesh
from shapely.geometry import Point
import geopandas as gpd

from app.geomap_processor.utils.geometry_utils import BoundingBox


@dataclass
class Transmitter:
    id: str
    lat: float
    lon: float
    height: float
    local_x: float
    local_y: float
    model: str
    type: str
    power_dbm: float
    tilt: float
    azimuth: float
    frequency: float
    active_users: int


class TelecomManager:
    """
    Manages fetching and processing of telecom infrastructure data.
    """

    DEFAULT_HEIGHT = 150.0
    CYLINDER_RADIUS = 2
    CYLINDER_SECTIONS = 16

    def __init__(self, bbox: BoundingBox):
        self.bbox = bbox
        self.transmitters: List[Transmitter] = []

        # Calculate center for local coordinate system
        self.center_lon, self.center_lat = bbox.center

    def fetch_and_process(self) -> None:
        # bbox for osmnx: (west, south, east, north)
        bbox_tuple = (
            self.bbox.min_lon,
            self.bbox.min_lat,
            self.bbox.max_lon,
            self.bbox.max_lat,
        )

        # OSM tags to look for
        tags = {"communication:mobile_phone": True, "tower:type": "communication"}

        logger.info(f"Fetching telecom data for bbox: {bbox_tuple}")

        try:
            gdf = ox.features_from_bbox(bbox=bbox_tuple, tags=tags)
            self._process_gdf(gdf)
        except Exception as e:
            logger.error(f"Telecom data fetch failed: {e}")

    def _process_gdf(self, gdf: gpd.GeoDataFrame) -> None:
        if gdf.empty:
            logger.warning("No telecom features found.")
            return

        logger.info(f"Found {len(gdf)} telecom features.")

        # Ensure CRS for projection
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)  # GCRS

        try:
            utm_crs = gdf.estimate_utm_crs()
        except Exception:
            utm_crs = "EPSG:3857"  # Web Mercator

        gdf_proj = gdf.to_crs(utm_crs)

        # Project center to calculate local offsets
        center_pt = gpd.GeoDataFrame(
            geometry=[Point(self.center_lon, self.center_lat)], crs="EPSG:4326"
        ).to_crs(utm_crs)

        cx, cy = center_pt.geometry[0].x, center_pt.geometry[0].y

        for idx, row in gdf_proj.iterrows():
            # Get metric coordinates
            x, y = self._get_geometry_center(row.geometry)
            geom_orig = gdf.loc[idx].geometry

            # Handle potential duplicate indices
            if isinstance(geom_orig, gpd.GeoSeries):
                geom_orig = geom_orig.iloc[0]
            lon, lat = self._get_geometry_center(geom_orig)

            # Populate Transmitter Data
            tx = self._create_transmitter(idx, lat, lon, x - cx, y - cy)
            self.transmitters.append(tx)

    def _get_geometry_center(self, geom: Any) -> Tuple[float, float]:
        """Extracts (x, y) from a Point or (centroid.x, centroid.y) from other geometries."""
        if geom.geom_type == "Point":
            return geom.x, geom.y
        return geom.centroid.x, geom.centroid.y

    def _create_transmitter(
        self, idx: Any, lat: float, lon: float, local_x: float, local_y: float
    ) -> Transmitter:
        """Creates a Transmitter object with synthetic simulation data."""

        # Clean up ID if it comes as a tuple (e.g. ('node', 12345))
        if isinstance(idx, tuple) and len(idx) > 1:
            tx_id = str(idx[1])
        else:
            tx_id = str(idx)

        return Transmitter(
            id=tx_id,
            lat=lat,
            lon=lon,
            height=self.DEFAULT_HEIGHT,
            local_x=local_x,
            local_y=local_y,
            model="Generic 5G Tower",
            type="Macro",
            power_dbm=random.uniform(43.0, 46.0),
            tilt=random.uniform(2, 6),
            azimuth=random.uniform(0, 360),
            frequency=3.5e9,
            active_users=random.randint(0, 100),
        )

    def get_mesh(
        self, height_callback: Optional[Callable[[float, float], float]] = None
    ) -> Optional[trimesh.Trimesh]:
        """Returns a combined mesh of all items, optionally adjusted to terrain height."""
        if not self.transmitters:
            return None

        meshes = []
        for tx in self.transmitters:
            c = trimesh.creation.cylinder(
                radius=self.CYLINDER_RADIUS,
                height=tx.height,
                sections=self.CYLINDER_SECTIONS,
            )

            z_ground = 0.0
            if height_callback:
                z_ground = height_callback(tx.local_x, tx.local_y)

            c.apply_translation([tx.local_x, tx.local_y, tx.height / 2.0 + z_ground])
            meshes.append(c)

        return trimesh.util.concatenate(meshes)

    def save_transmitters_json(self, output_path: Path) -> None:
        """Exports the transmitters to an Eclipse Ditto formatted JSON."""
        ditto_items = []

        for tx in self.transmitters:
            item = {
                "thingId": f"com.sionna:{tx.id}",
                "attributes": {
                    "location": {
                        "latitude": tx.lat,
                        "longitude": tx.lon,
                        "height_m": tx.height,
                    },
                    "physical": {"model": tx.model, "type": tx.type},
                },
                "features": {
                    "configuration": {
                        "properties": {
                            "transmit_power_dbm": tx.power_dbm,
                            "mechanical_tilt": round(tx.tilt, 2),
                            "azimuth_deg": round(tx.azimuth, 2),
                            "carrier_frequency_hz": tx.frequency,
                            "admin_state": "enabled",
                        }
                    },
                    "status": {
                        "properties": {
                            "operational_state": "up",
                            "active_users": tx.active_users,
                        }
                    },
                },
            }
            ditto_items.append(item)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(ditto_items, f, indent=2)
            logger.info(f"Exported {len(ditto_items)} transmitters to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export transmitters JSON: {e}")

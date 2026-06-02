import json
import random
from typing import List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

import osmnx as ox
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

    DEFAULT_HEIGHT = 30.0

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
        except (ValueError, RuntimeError):
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
            tx_list = self._create_site_transmitters(idx, lat, lon, x - cx, y - cy)
            self.transmitters.extend(tx_list)

    def _get_geometry_center(self, geom: Any) -> Tuple[float, float]:
        """Extracts (x, y) from a Point or (centroid.x, centroid.y) from other geometries."""
        if geom.geom_type == "Point":
            return geom.x, geom.y
        return geom.centroid.x, geom.centroid.y

    def _create_site_transmitters(
        self, idx: Any, lat: float, lon: float, local_x: float, local_y: float
    ) -> List[Transmitter]:
        """Creates 3 Transmitters (sectors) for a single cell site."""

        # Clean up ID if it comes as a tuple (e.g. ('node', 12345))
        if isinstance(idx, tuple) and len(idx) > 1:
            site_id = str(idx[1])
        else:
            site_id = str(idx)

        # Base orientation for the whole site so not all towers point True North
        base_azimuth = random.uniform(0, 119)
        sectors = []

        for sector_idx in range(3):
            azimuth = (base_azimuth + (sector_idx * 120)) % 360
            tx = Transmitter(
                id=f"{site_id}_s{sector_idx}",
                lat=lat,
                lon=lon,
                height=self.DEFAULT_HEIGHT,
                local_x=local_x,
                local_y=local_y,
                model="Generic 5G Sector",
                type="Macro",
                power_dbm=random.uniform(43.0, 46.0),
                tilt=random.uniform(2, 6),
                azimuth=azimuth,
                frequency=1.8e9,
                active_users=random.randint(0, 33),  # Divided by roughly 3 from old max
            )
            sectors.append(tx)

        return sectors

    def save_transmitters_json(self, output_path: Path) -> None:
        """Exports transmitters to Eclipse Ditto JSON — one Thing per antenna site.

        Each site's 3 sectors are nested as features (sector_0, sector_1, sector_2)
        so a single API call provisions the full antenna instead of 3 separate Things.
        """
        # Group the flat sector list back into sites keyed by site_id
        sites: dict[str, list[Transmitter]] = {}
        for tx in self.transmitters:
            site_id = tx.id.rsplit("_s", 1)[0]
            sites.setdefault(site_id, []).append(tx)

        ditto_items = []
        for site_id, sectors in sites.items():
            ref = sectors[0]  # shared location/physical attributes
            features = {}
            for tx in sectors:
                sector_idx = tx.id.rsplit("_s", 1)[1]
                features[f"sector_{sector_idx}"] = {
                    "properties": {
                        "transmit_power_dbm": tx.power_dbm,
                        "mechanical_tilt": round(tx.tilt, 2),
                        "azimuth_deg": round(tx.azimuth, 2),
                        "carrier_frequency_hz": tx.frequency,
                        "admin_state": "enabled",
                        "operational_state": "up",
                        "active_users": tx.active_users,
                    }
                }

            ditto_items.append(
                {
                    "thingId": f"com.sionna:antenna_{site_id}",
                    "attributes": {
                        "location": {
                            "latitude": ref.lat,
                            "longitude": ref.lon,
                            "height_m": ref.height,
                        },
                        "physical": {"model": ref.model, "type": ref.type},
                    },
                    "features": features,
                }
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(ditto_items, f, indent=2)
            logger.info(
                f"Exported {len(ditto_items)} antenna Things "
                f"({len(self.transmitters)} sectors) to {output_path}"
            )
        except (OSError, TypeError) as e:
            logger.error(f"Failed to export transmitters JSON: {e}")
            raise

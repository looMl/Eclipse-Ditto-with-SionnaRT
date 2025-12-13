import logging
from typing import Dict, List, Tuple, Optional, Callable

import osmnx as ox
import trimesh
from shapely.geometry import Point
import geopandas as gpd

logger = logging.getLogger(__name__)


class TelecomManager:
    """
    Manages fetching and processing of telecom infrastructure data.
    """

    def __init__(self, bbox: Dict[str, float]):
        self.bbox = bbox
        # (x, y, height) for each antenna
        self.locations: List[Tuple[float, float, float]] = []

        # Calculate center for local coordinate system
        self.center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2.0
        self.center_lon = (bbox["min_lon"] + bbox["max_lon"]) / 2.0

    def fetch_and_process(self) -> None:
        # bbox for osmnx: (west, south, east, north)
        bbox_tuple = (
            self.bbox["min_lon"],
            self.bbox["min_lat"],
            self.bbox["max_lon"],
            self.bbox["max_lat"],
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

        for _, row in gdf_proj.iterrows():
            geom = row.geometry
            if geom.geom_type == "Point":
                x, y = geom.x, geom.y
            else:
                x, y = geom.centroid.x, geom.centroid.y

            self.locations.append((x - cx, y - cy, 150.0))

    def get_mesh(
        self, height_callback: Optional[Callable[[float, float], float]] = None
    ) -> Optional[trimesh.Trimesh]:
        """Returns a combined mesh of all items, optionally adjusted to terrain height."""
        if not self.locations:
            return None

        meshes = []
        for x, y, h in self.locations:
            c = trimesh.creation.cylinder(radius=2, height=h, sections=16)

            z_ground = 0.0
            if height_callback:
                z_ground = height_callback(x, y)

            c.apply_translation([x, y, h / 2.0 + z_ground])
            meshes.append(c)

        return trimesh.util.concatenate(meshes)

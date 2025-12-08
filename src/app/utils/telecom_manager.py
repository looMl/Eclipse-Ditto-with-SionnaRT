import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import xml.etree.ElementTree as ET

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

    def get_mesh(self) -> Optional[trimesh.Trimesh]:
        """Returns a combined mesh of all items."""
        if not self.locations:
            return None

        meshes = []
        for x, y, h in self.locations:
            c = trimesh.creation.cylinder(radius=2, height=h, sections=16)
            # Translate to sit on z=0 (move up by h/2) and move to local pos
            c.apply_translation([x, y, h / 2.0])
            meshes.append(c)

        return trimesh.util.concatenate(meshes)

    def update_scene_xml(self, scene_path: Path, mesh_rel_path: str) -> None:
        """Injects the transmitters mesh into the scene XML."""
        logger.info(f"Injecting transmitters from {mesh_rel_path} into {scene_path}...")
        try:
            tree = ET.parse(scene_path)
            root = tree.getroot()

            # Check if already exists to avoid duplicates
            existing = root.find(".//shape[@id='mesh-transmitters']")
            if existing is None:
                shape = ET.SubElement(root, "shape")
                shape.set("type", "ply")
                shape.set("id", "mesh-transmitters")

                filename = ET.SubElement(shape, "string")
                filename.set("name", "filename")
                filename.set("value", mesh_rel_path)

                ref = ET.SubElement(shape, "ref")
                ref.set("name", "bsdf")
                ref.set("id", "mat-itu_metal")

                tree.write(scene_path, encoding="utf-8", xml_declaration=True)
                logger.info("Successfully added transmitters to scene.xml")
            else:
                logger.info("Transmitters shape already present in scene.xml")

        except Exception as e:
            logger.error(f"Failed to patch scene.xml: {e}")

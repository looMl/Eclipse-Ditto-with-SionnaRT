from pathlib import Path
from typing import Optional

import numpy as np
import osmnx as ox
import rasterio
from loguru import logger
from rasterio.features import rasterize

import geopandas as gpd

from app.geomap_processor.utils.geometry_utils import BoundingBox
from app.geomap_processor.utils.vegetation_field import VegetationField

# OSM tags defining the "where vegetation might exist"
_OSM_TAGS = {
    "landuse": ["forest", "meadow", "grass", "orchard", "vineyard"],
    "natural": ["wood", "tree", "tree_row", "scrub", "heath"],
    "leisure": ["park", "garden", "golf_course"],
}

# Tree cover density fraction per OSM tag — used when the satellite TCD raster
# has no tree signal (e.g. WorldCover classifies urban parks as Built-up).
_OSM_TCD_FRACTIONS = {
    "forest": 0.9,
    "wood": 0.9,
    "tree": 0.8,
    "tree_row": 0.7,
    "orchard": 0.5,
    "park": 0.5,
    "garden": 0.4,
    "scrub": 0.3,
    "golf_course": 0.2,
    "vineyard": 0.2,
    "heath": 0.1,
    "meadow": 0.0,
    "grass": 0.0,
}

# Heuristic canopy heights per OSM tag value (meters) — used when CHM raster is
# absent or has NaN pixels
_HEURISTIC_HEIGHTS = {
    "forest": 18.0,
    "wood": 18.0,
    "tree": 12.0,
    "orchard": 8.0,
    "vineyard": 4.0,
    "park": 12.0,
    "garden": 8.0,
    "golf_course": 4.0,
    "tree_row": 8.0,
    "scrub": 4.0,
    "heath": 2.0,
    "meadow": 0.0,
    "grass": 0.0,
}


class VegetationManager:
    """
    Fetches OSM vegetation polygons and builds a VegetationField from TCD/CHM rasters.
    """

    def __init__(self, bbox: BoundingBox):
        self.bbox = bbox

    def fetch_and_process(self) -> gpd.GeoDataFrame:
        """
        Returns OSM vegetation polygons as a GeoDataFrame (EPSG:4326).
        """
        bbox_tuple = (
            self.bbox.min_lon,
            self.bbox.min_lat,
            self.bbox.max_lon,
            self.bbox.max_lat,
        )
        logger.info(f"Fetching OSM vegetation for bbox: {bbox_tuple}")
        try:
            gdf = ox.features_from_bbox(bbox=bbox_tuple, tags=_OSM_TAGS)
        except Exception as e:
            logger.warning(f"OSM vegetation fetch failed: {e}")
            return gpd.GeoDataFrame()

        if gdf.empty:
            logger.warning("No OSM vegetation features found.")
            return gdf

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)

        logger.info(f"Found {len(gdf)} OSM vegetation features.")
        return gdf

    def build_density_field(
        self, tcd_path: Path, chm_path: Optional[Path]
    ) -> VegetationField:
        """
        Returns a VegetationField gated by the OSM polygon mask.
        - TCD pixels outside any OSM polygon are zeroed (the 'where' axis).
        - CHM NaN pixels are filled with per-tag heuristic heights.
        - If chm_path is None (heuristic source), the entire CHM is tag-derived.
        """
        gdf = self.fetch_and_process()
        osm_mask = self._burn_polygon_mask(gdf, tcd_path)

        with rasterio.open(tcd_path) as ds:
            tcd = ds.read(1).astype("float32")
            transform = ds.transform
            crs = ds.crs

        # WorldCover gate: keep satellite TCD only where OSM confirms vegetation
        tcd = tcd * osm_mask
        # Urban vegetation fallback: WorldCover misses park/garden trees (classifies
        # them as Built-up). Use per-tag TCD fractions so Piazza Bra etc. are modelled.
        osm_tcd = self._build_osm_tcd(gdf, tcd_path)
        tcd = np.maximum(tcd, osm_tcd)

        if chm_path is not None:
            with rasterio.open(chm_path) as ds:
                chm = ds.read(1).astype("float32")
            nan_mask = np.isnan(chm)
            if nan_mask.any():
                heuristic = self._build_heuristic_chm(gdf, tcd_path)
                chm = np.where(nan_mask, heuristic, chm)
        else:
            chm = self._build_heuristic_chm(gdf, tcd_path)

        origin_lon, origin_lat = self.bbox.center
        return VegetationField(
            tcd=tcd,
            chm=chm,
            transform=transform,
            crs=crs,
            origin_lon=origin_lon,
            origin_lat=origin_lat,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _burn_polygon_mask(
        self, gdf: gpd.GeoDataFrame, reference_path: Path
    ) -> np.ndarray:
        """Burns OSM polygons onto the TCD raster grid. Returns float32 mask (1/0)."""
        with rasterio.open(reference_path) as ds:
            out_shape = (ds.height, ds.width)
            transform = ds.transform
            raster_crs = ds.crs

        if gdf.empty:
            return np.zeros(out_shape, dtype="float32")

        gdf_proj = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf

        shapes = [
            (geom, 1.0)
            for geom in gdf_proj.geometry
            if geom is not None and geom.geom_type in ("Polygon", "MultiPolygon")
        ]

        if not shapes:
            return np.zeros(out_shape, dtype="float32")

        return rasterize(
            shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0.0,
            dtype="float32",
        )

    def _build_osm_tcd(self, gdf: gpd.GeoDataFrame, reference_path: Path) -> np.ndarray:
        """Rasterizes OSM features with per-tag TCD fractions onto the TCD grid.

        Points (natural=tree) are buffered to 5 m circles; LineStrings (tree_row)
        to 3 m strips so individual and aligned urban trees register on the raster.
        """
        with rasterio.open(reference_path) as ds:
            out_shape = (ds.height, ds.width)
            transform = ds.transform
            raster_crs = ds.crs

        if gdf.empty:
            return np.zeros(out_shape, dtype="float32")

        gdf_proj = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf

        shapes = []
        for _, row in gdf_proj.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            val = self._tag_tcd(row)
            if val == 0.0:
                continue
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                shapes.append((geom, val))
            elif geom.geom_type == "Point":
                shapes.append((geom.buffer(5.0), val))
            elif geom.geom_type == "LineString":
                shapes.append((geom.buffer(3.0), val))

        if not shapes:
            return np.zeros(out_shape, dtype="float32")

        return rasterize(
            shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0.0,
            dtype="float32",
        )

    def _build_heuristic_chm(
        self, gdf: gpd.GeoDataFrame, reference_path: Path
    ) -> np.ndarray:
        """Rasterizes OSM features with per-tag height values onto the TCD grid.

        Points and LineStrings receive the same buffer as _build_osm_tcd so CHM
        heights are consistent with the TCD mask.
        """
        with rasterio.open(reference_path) as ds:
            out_shape = (ds.height, ds.width)
            transform = ds.transform
            raster_crs = ds.crs

        if gdf.empty:
            return np.zeros(out_shape, dtype="float32")

        gdf_proj = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf

        shapes = []
        for _, row in gdf_proj.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            val = self._tag_height(row)
            if val == 0.0:
                continue
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                shapes.append((geom, val))
            elif geom.geom_type == "Point":
                shapes.append((geom.buffer(5.0), val))
            elif geom.geom_type == "LineString":
                shapes.append((geom.buffer(3.0), val))

        if not shapes:
            return np.zeros(out_shape, dtype="float32")

        return rasterize(
            shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0.0,
            dtype="float32",
        )

    def _tag_tcd(self, row) -> float:
        """Returns OSM-derived TCD fraction for a feature row."""
        for tag_key in ("natural", "landuse", "leisure"):
            val = row.get(tag_key)
            if val and isinstance(val, str) and val in _OSM_TCD_FRACTIONS:
                return _OSM_TCD_FRACTIONS[val]
        return 0.0

    def _tag_height(self, row) -> float:
        """Returns heuristic canopy height for an OSM feature row."""
        for tag_key in ("natural", "landuse", "leisure"):
            val = row.get(tag_key)
            if val and isinstance(val, str) and val in _HEURISTIC_HEIGHTS:
                return _HEURISTIC_HEIGHTS[val]
        return 0.0

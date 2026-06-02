import hashlib
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from loguru import logger
from rasterio.crs import CRS
from rasterio.errors import RasterioError
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from requests import RequestException, get as http_get
from shapely.geometry import box

# ESA WorldCover 2021 class value for "Tree cover"
_WC_TREE_CLASS = 10


class VegetationRasterDownloader:
    """
    Downloads and caches TCD (Tree Cover Density) and CHM (Canopy Height Model)
    rasters for a bounding box.
    """

    _WC_URL = (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
        "/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile_id}_Map.tif"
    )
    _PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        bbox: Tuple[float, float, float, float],
        tcd_source: str,
        chm_source: str,
        target_crs: Optional[CRS] = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Returns (tcd_path, chm_path) for bbox (min_lon, min_lat, max_lon, max_lat).
        Files are reprojected to target_crs when provided. Either path may be None
        on download failure.
        """
        tcd_path = self._fetch_tcd(bbox, tcd_source, target_crs)
        chm_path = self._fetch_chm(bbox, chm_source, target_crs)
        return tcd_path, chm_path

    # ------------------------------------------------------------------
    # TCD dispatch
    # ------------------------------------------------------------------

    def _fetch_tcd(
        self,
        bbox: Tuple[float, float, float, float],
        source: str,
        target_crs: Optional[CRS],
    ) -> Optional[Path]:
        out_path = self._cache_path("tcd", source, bbox)
        if out_path.exists():
            logger.info(f"Using cached TCD [{source}]: {out_path}")
            return out_path
        try:
            if source == "esa_worldcover":
                return self._worldcover_tcd(bbox, out_path, target_crs)
            if source == "worldcover+ndvi":
                return self._worldcover_ndvi_tcd(bbox, out_path, target_crs)
            if source == "copernicus_hrl":
                return self._copernicus_tcd(bbox, out_path, target_crs)
            raise ValueError(f"Unknown tcd_source: {source!r}")
        except (ValueError, OSError, RuntimeError, ImportError, RasterioError) as e:
            logger.error(f"TCD fetch failed [{source}]: {e}")
            out_path.unlink(missing_ok=True)
            return None

    def _fetch_chm(
        self,
        bbox: Tuple[float, float, float, float],
        source: str,
        target_crs: Optional[CRS],
    ) -> Optional[Path]:
        if source == "heuristic":
            return None  # No raster; VegetationField falls back to per-tag heights
        out_path = self._cache_path("chm", source, bbox)
        if out_path.exists():
            logger.info(f"Using cached CHM [{source}]: {out_path}")
            return out_path
        try:
            if source == "eth_global_2020":
                logger.warning(
                    "The ETH Global Canopy Height 2020 hosting (libdrive.ethz.ch) is no "
                    "longer available. Set chm_source: 'heuristic' in config.yaml to use "
                    "per-tag fallback heights instead."
                )
                out_path.unlink(missing_ok=True)
                return None
            raise ValueError(f"Unknown chm_source: {source!r}")
        except (ValueError, OSError) as e:
            logger.error(f"CHM fetch failed [{source}]: {e}")
            out_path.unlink(missing_ok=True)
            return None

    # ------------------------------------------------------------------
    # ESA WorldCover — binary tree mask, zero-auth
    # ------------------------------------------------------------------

    def _worldcover_tcd(
        self,
        bbox: Tuple[float, float, float, float],
        out_path: Path,
        target_crs: Optional[CRS],
    ) -> Path:
        tile_ids = self._tile_ids(bbox)
        logger.info(f"Fetching ESA WorldCover tiles: {tile_ids}")
        raw = self._download_tiles(tile_ids, self._WC_URL, "wc")

        # Nearest-neighbour preserves integer class values through reprojection
        self._merge_clip_reproject(
            raw, bbox, out_path, target_crs, resampling=Resampling.nearest
        )

        # Threshold in-place: class 10 (Tree cover) → 1.0, all else → 0.0
        with rasterio.open(out_path, "r+") as ds:
            data = ds.read(1)
            ds.write(np.where(data == _WC_TREE_CLASS, 1.0, 0.0).astype("float32"), 1)

        logger.info(f"WorldCover TCD saved: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # WorldCover + NDVI — continuous TCD, zero-auth
    # ------------------------------------------------------------------

    def _worldcover_ndvi_tcd(
        self,
        bbox: Tuple[float, float, float, float],
        out_path: Path,
        target_crs: Optional[CRS],
    ) -> Path:
        try:
            import planetary_computer
            import pystac_client
        except ImportError as exc:
            raise ImportError(
                "worldcover+ndvi requires: pip install pystac-client planetary-computer"
            ) from exc

        # Reuse binary WorldCover mask as spatial gate
        wc_path = self._cache_path("tcd", "esa_worldcover", bbox)
        if not wc_path.exists():
            self._worldcover_tcd(bbox, wc_path, target_crs)

        min_lon, min_lat, max_lon, max_lat = bbox
        catalog = pystac_client.Client.open(
            self._PLANETARY_COMPUTER_STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )
        items = list(
            catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=[min_lon, min_lat, max_lon, max_lat],
                query={"eo:cloud_cover": {"lt": 20}},
                max_items=1,
            ).items()
        )
        if not items:
            raise RuntimeError("No Sentinel-2 scene found for bbox.")

        item = items[0]
        with (
            rasterio.open(planetary_computer.sign(item.assets["B04"].href)) as b04_ds,
            rasterio.open(planetary_computer.sign(item.assets["B08"].href)) as b08_ds,
        ):
            b04 = b04_ds.read(1).astype("float32")
            b08 = b08_ds.read(1).astype("float32")
            profile = b04_ds.profile.copy()

        denom = b08 + b04
        ndvi = np.where(denom > 0, (b08 - b04) / denom, 0.0).astype("float32")

        # NDVI [0.2, 0.9] → TCD [0.0, 1.0], clamped; zeroed outside tree mask
        tcd = np.clip((ndvi - 0.2) / 0.7, 0.0, 1.0).astype("float32")
        with rasterio.open(wc_path) as wc_ds:
            tcd *= wc_ds.read(1)

        profile.update(dtype="float32", count=1)
        with rasterio.open(out_path, "w", **profile) as ds:
            ds.write(tcd, 1)

        logger.info(f"WorldCover+NDVI TCD saved: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Copernicus HRL — optional, requires CLMS_TOKEN
    # ------------------------------------------------------------------

    def _copernicus_tcd(
        self,
        bbox: Tuple[float, float, float, float],
        out_path: Path,
        target_crs: Optional[CRS],
    ) -> Path:
        if not os.environ.get("CLMS_TOKEN"):
            raise EnvironmentError(
                "copernicus_hrl requires the CLMS_TOKEN environment variable. "
                "Register at https://land.copernicus.eu/ to obtain one."
            )
        raise NotImplementedError(
            "Copernicus HRL WCS fetch not yet implemented. "
            "Use tcd_source='esa_worldcover' or 'worldcover+ndvi'."
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _tile_ids(self, bbox: Tuple[float, float, float, float]) -> List[str]:
        """3°×3° tile IDs (WorldCover / ETH CHM grid) covering the bbox."""
        min_lon, min_lat, max_lon, max_lat = bbox
        ids: List[str] = []
        lat = math.floor(min_lat / 3) * 3
        while lat <= max_lat:
            lon = math.floor(min_lon / 3) * 3
            while lon <= max_lon:
                lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
                lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
                ids.append(f"{lat_str}{lon_str}")
                lon += 3
            lat += 3
        return ids

    def _download_tiles(
        self, tile_ids: List[str], url_template: str, prefix: str
    ) -> List[Path]:
        paths: List[Path] = []
        for tile_id in tile_ids:
            dest = self.output_dir / f"veg_{prefix}_raw_{tile_id}.tif"
            if dest.exists():
                paths.append(dest)
                continue
            result = self._http_download(url_template.format(tile_id=tile_id), dest)
            if result:
                paths.append(result)
        if not paths:
            raise RuntimeError(f"No tiles could be downloaded for {tile_ids}")
        return paths

    def _merge_clip_reproject(
        self,
        tile_paths: List[Path],
        bbox: Tuple[float, float, float, float],
        out_path: Path,
        target_crs: Optional[CRS],
        resampling: Resampling = Resampling.bilinear,
    ) -> None:
        """Merges tiles, clips to bbox, optionally reprojects to target_crs."""
        datasets = [rasterio.open(p) for p in tile_paths]
        try:
            merged, merged_transform = rio_merge(datasets, resampling=resampling)
            src_crs = datasets[0].crs
            profile = datasets[0].profile.copy()
        finally:
            for ds in datasets:
                ds.close()

        merged = merged[0].astype("float32")
        profile.update(
            count=1,
            dtype="float32",
            transform=merged_transform,
            height=merged.shape[0],
            width=merged.shape[1],
            crs=src_crs,
        )

        clip_shape = [box(*bbox).__geo_interface__]
        with rasterio.MemoryFile() as tmp_mem:
            with tmp_mem.open(**profile) as tmp_ds:
                tmp_ds.write(merged, 1)
                clipped, clip_transform = rio_mask(tmp_ds, clip_shape, crop=True)

        clipped = clipped[0].astype("float32")
        profile.update(
            transform=clip_transform,
            height=clipped.shape[0],
            width=clipped.shape[1],
        )

        if target_crs is not None and target_crs != src_crs:
            clipped, profile = self._reproject(clipped, profile, target_crs, resampling)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as ds:
            ds.write(clipped, 1)

    def _reproject(
        self,
        data: np.ndarray,
        profile: dict,
        target_crs: CRS,
        resampling: Resampling,
    ) -> Tuple[np.ndarray, dict]:
        src_crs = profile["crs"]
        src_transform = profile["transform"]
        h, w = data.shape
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs,
            target_crs,
            w,
            h,
            *array_bounds(h, w, src_transform),
        )
        dst = np.zeros((dst_h, dst_w), dtype="float32")
        reproject(
            source=data,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=resampling,
        )
        new_profile = {
            **profile,
            "crs": target_crs,
            "transform": dst_transform,
            "width": dst_w,
            "height": dst_h,
        }
        return dst, new_profile

    def _http_download(self, url: str, dest: Path) -> Optional[Path]:
        logger.info(f"Downloading: {url}")
        try:
            response = http_get(url, stream=True, timeout=120)
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return dest
        except (RequestException, OSError) as e:
            logger.error(f"Download failed [{url}]: {e}")
            dest.unlink(missing_ok=True)
            return None

    def _cache_path(
        self,
        layer: str,
        source: str,
        bbox: Tuple[float, float, float, float],
    ) -> Path:
        bbox_hash = hashlib.md5(
            f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}".encode(), usedforsecurity=False
        ).hexdigest()
        return self.output_dir / f"veg_{layer}_{source}_{bbox_hash}.tif"

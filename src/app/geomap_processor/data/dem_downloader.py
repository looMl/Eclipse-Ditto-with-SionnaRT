import logging
import hashlib
import requests
import urllib3
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class DemDownloader:
    """Service responsible for downloading DEM data."""

    TINITALY_WCS_URL = "http://tinitaly.pi.ingv.it/TINItaly_1_1/wcs"
    COVERAGE_ID = "TINItaly_1_1:tinitaly_dem"
    RESOLUTION = 0.00009  # Approx 10m resolution (degrees)

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def fetch(self, bbox: Tuple[float, float, float, float]) -> Optional[Path]:
        """
        Downloads a DEM from TINITALY WCS for the given BBox (minx, miny, maxx, maxy).
        Returns the path to the cached or downloaded GeoTIFF file.
        """
        file_path = self._get_file_path(bbox)

        # Check cache
        if file_path.exists():
            logger.info(f"Using cached DEM file: {file_path}")
            return file_path

        logger.info(f"Downloading DEM from TINITALY WCS for bbox: {bbox}")

        try:
            # Disable SSL warnings for TINItaly server
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            width, height = self._calculate_dimensions(bbox)
            params = self._build_params(bbox, width, height)

            self._download_and_save(params, file_path)

            logger.info(f"DEM downloaded and saved to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to download DEM from TINITALY: {e}")
            if file_path.exists():
                file_path.unlink()
            return None

    def _get_file_path(self, bbox: Tuple[float, float, float, float]) -> Path:
        """Generates a unique file path based on the bounding box hash."""
        minx, miny, maxx, maxy = bbox
        bbox_str = f"{minx}_{miny}_{maxx}_{maxy}"
        bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()
        filename = f"tinitaly_{bbox_hash}.tif"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / filename

    def _calculate_dimensions(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[int, int]:
        """Calculates image dimensions based on target resolution."""
        minx, miny, maxx, maxy = bbox
        width = int(abs(maxx - minx) / self.RESOLUTION)
        height = int(abs(maxy - miny) / self.RESOLUTION)
        return width, height

    def _build_params(
        self, bbox: Tuple[float, float, float, float], width: int, height: int
    ) -> Dict[str, Any]:
        """Constructs WCS 1.0.0 request parameters."""
        minx, miny, maxx, maxy = bbox
        return {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": self.COVERAGE_ID,
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "crs": "EPSG:4326",
            "format": "image/tiff",
            "width": str(width),
            "height": str(height),
        }

    def _download_and_save(self, params: Dict[str, Any], file_path: Path) -> None:
        """Executes the download and saves the content to disk."""
        req = requests.Request("GET", self.TINITALY_WCS_URL, params=params)
        prepped = req.prepare()
        logger.info(f"Requesting URL: {prepped.url}")

        # Use requests with verify=False to bypass SSL errors
        response = requests.get(
            self.TINITALY_WCS_URL, params=params, verify=False, stream=True
        )
        response.raise_for_status()

        # Check if we got an XML error report instead of an image
        content_type = response.headers.get("Content-Type", "").lower()
        if "xml" in content_type:
            error_msg = response.text
            logger.error(f"WCS Error Response: {error_msg}")
            raise RuntimeError(f"WCS returned an XML error: {error_msg}")

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

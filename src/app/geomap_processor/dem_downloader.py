import logging
import hashlib
import requests
import urllib3
from pathlib import Path
from typing import Tuple, Optional

from app.utils.config import get_project_root

logger = logging.getLogger(__name__)

TINITALY_WCS_URL = "http://tinitaly.pi.ingv.it/TINItaly_1_1/wcs"
COVERAGE_ID = "TINItaly_1_1:tinitaly_dem"


def fetch_tinitaly_dem(bbox: Tuple[float, float, float, float]) -> Optional[Path]:
    """
    Downloads a DEM from TINITALY WCS for the given BBox (minx, miny, maxx, maxy).
    Returns the path to the cached or downloaded GeoTIFF file.
    """
    minx, miny, maxx, maxy = bbox

    # Generate unique filename based on bbox hash
    bbox_str = f"{minx}_{miny}_{maxx}_{maxy}"
    bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()
    filename = f"tinitaly_{bbox_hash}.tif"

    output_dir = get_project_root() / "geotiffs"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename

    # Check cache
    if file_path.exists():
        logger.info(f"Using cached DEM file: {file_path}")
        return file_path

    logger.info(f"Downloading DEM from TINITALY WCS for bbox: {bbox}")

    try:
        # Disable SSL warnings for TINItaly server
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Calculate width/height for approx 10m resolution (0.00009 deg)
        res = 0.00009
        width = int(abs(maxx - minx) / res)
        height = int(abs(maxy - miny) / res)

        params = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": COVERAGE_ID,
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "crs": "EPSG:4326",
            "format": "image/tiff",
            "width": str(width),
            "height": str(height),
        }

        req = requests.Request("GET", TINITALY_WCS_URL, params=params)
        prepped = req.prepare()
        logger.info(f"Requesting URL: {prepped.url}")

        # Use requests with verify=False to bypass SSL errors
        response = requests.get(
            TINITALY_WCS_URL, params=params, verify=False, stream=True
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

        logger.info(f"DEM downloaded and saved to: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to download DEM from TINITALY: {e}")
        if file_path.exists():
            file_path.unlink()
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        from app.utils.config import settings

        bbox = (
            settings.geo2sigmap.min_lon,
            settings.geo2sigmap.min_lat,
            settings.geo2sigmap.max_lon,
            settings.geo2sigmap.max_lat,
        )

        print(f"Testing DEM download for config bbox: {bbox}")
        result_path = fetch_tinitaly_dem(bbox)

        if result_path:
            print(f"SUCCESS: DEM available at {result_path}")
        else:
            print("FAILURE: Could not fetch DEM.")

    except ImportError:
        print("Incorrect import paths, cannot run test")
    except Exception as e:
        print(f"An error occurred: {e}")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.geomap_processor.processors.dem_processor import DemProcessor

_WGS84_EPSG = 4326


@dataclass
class VegetationField:
    """
    Aligned TCD [0,1] and CHM [m] arrays with raster georeferencing and scene origin.
    Serialisable to a .npz file so the simulation layer loads without re-fetching rasters.
    """

    tcd: np.ndarray  # [H, W] float32, canopy density in [0, 1]
    chm: np.ndarray  # [H, W] float32, canopy height in metres
    transform: Affine  # raster affine geotransform (native raster CRS)
    crs: CRS  # raster coordinate reference system
    origin_lon: float  # scene bbox centre longitude (for local ↔ global conversion)
    origin_lat: float  # scene bbox centre latitude

    def sample(self, x_local: float, y_local: float) -> tuple[float, float]:
        """
        Returns (tcd, chm) at scene-local (x, y) via bilinear interpolation.
        Out-of-bounds coordinates return (0.0, 0.0).
        """
        lon, lat = DemProcessor.local_to_global(
            x_local, y_local, self.origin_lon, self.origin_lat
        )

        # Convert lon/lat to the raster's native CRS when it's not EPSG:4326
        if self.crs.to_epsg() != _WGS84_EPSG:
            rx_list, ry_list = rasterio.warp.transform(
                "EPSG:4326", self.crs, [lon], [lat]
            )
            rx, ry = rx_list[0], ry_list[0]
        else:
            rx, ry = lon, lat

        col = (rx - self.transform.c) / self.transform.a
        row = (ry - self.transform.f) / self.transform.e

        h, w = self.tcd.shape
        if not (0.0 <= row < h and 0.0 <= col < w):
            return 0.0, 0.0

        return _bilinear(self.tcd, row, col), _bilinear(self.chm, row, col)

    def save(self, path: Path) -> None:
        """Saves to .npz. numpy appends .npz suffix if absent."""
        t = self.transform
        np.savez(
            path,
            tcd=self.tcd,
            chm=self.chm,
            transform=np.array([t.a, t.b, t.c, t.d, t.e, t.f], dtype="float64"),
            crs_wkt=np.array(self.crs.to_wkt()),
            origin_lon=np.float64(self.origin_lon),
            origin_lat=np.float64(self.origin_lat),
        )

    @classmethod
    def load(cls, path: Path) -> "VegetationField":
        """Loads from a .npz file produced by save()."""
        data = np.load(path, allow_pickle=False)
        return cls(
            tcd=data["tcd"],
            chm=data["chm"],
            transform=Affine(*data["transform"]),
            crs=CRS.from_wkt(str(data["crs_wkt"])),
            origin_lon=float(data["origin_lon"]),
            origin_lat=float(data["origin_lat"]),
        )


def _bilinear(arr: np.ndarray, row: float, col: float) -> float:
    """Bilinear interpolation at fractional (row, col) in a 2-D float32 array."""
    h, w = arr.shape
    r0, c0 = int(row), int(col)
    r1 = min(r0 + 1, h - 1)
    c1 = min(c0 + 1, w - 1)
    dr, dc = row - r0, col - c0
    return float(
        arr[r0, c0] * (1 - dr) * (1 - dc)
        + arr[r0, c1] * (1 - dr) * dc
        + arr[r1, c0] * dr * (1 - dc)
        + arr[r1, c1] * dr * dc
    )

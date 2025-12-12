import logging
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds, rowcol
import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple, Union, Optional

logger = logging.getLogger(__name__)


class DemProcessor:
    """
    Handles Digital Elevation Model (DEM) processing and terrain mesh generation.
    """

    @staticmethod
    def _get_utm_crs(lon: float, lat: float) -> str:
        """Calculates the EPSG code for the UTM zone of the given coordinate."""
        zone = int((lon + 180) / 6) + 1
        hemisphere = "6" if lat >= 0 else "7"  # 326xx (North) vs 327xx (South)
        return f"EPSG:32{hemisphere}{zone:02d}"

    @staticmethod
    def process_dem(
        dem_path: Union[str, Path], bbox: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Reads a DEM file, reprojects it to EPSG:4326, and crops it to the specified bounding box.
        """
        dst_crs = "EPSG:4326"
        west, south, east, north = bbox

        try:
            with rasterio.open(dem_path) as src:
                # Calculate optimal transform to determine target resolution
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )

                res_x, res_y = abs(transform[0]), abs(transform[4])

                # Calculate dimensions based on bbox and resolution
                dst_width = int((east - west) / res_x)
                dst_height = int((north - south) / res_y)

                dst_transform = from_bounds(
                    west, south, east, north, dst_width, dst_height
                )

                destination = np.zeros((dst_height, dst_width), dtype=src.meta["dtype"])

                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

                logger.info(
                    f"Processed DEM data: shape={destination.shape}, transform={dst_transform}"
                )
                return destination, dst_transform

        except Exception as e:
            logger.error(f"Failed to process DEM data from {dem_path}: {e}")
            raise

    @staticmethod
    def generate_terrain_mesh(
        elevation_data: np.ndarray,
        transform: rasterio.Affine,
        output_path: Union[str, Path],
        mesh_origin: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Generates a 3D mesh from elevation data and saves it as a PLY file.
        """
        try:
            height, width = elevation_data.shape

            # Create meshgrid of array indices
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))

            # Convert to map coordinates (EPSG:4326)
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
            xs, ys = np.array(xs).flatten(), np.array(ys).flatten()
            # Normalize elevation relative to mesh origin
            if mesh_origin:
                r, c = rowcol(transform, *mesh_origin)
                r, c = max(0, min(r, height - 1)), max(0, min(c, width - 1))
                ref_elev = elevation_data[r, c]
            else:
                ref_elev = np.min(elevation_data)

            zs = elevation_data.flatten() - ref_elev

            # Project to local coordinates if origin is provided
            if mesh_origin:
                origin_lon, origin_lat = mesh_origin
                dst_crs = DemProcessor._get_utm_crs(origin_lon, origin_lat)
                src_crs = "EPSG:4326"

                logger.info(
                    f"Projecting terrain mesh to {dst_crs} relative to {mesh_origin}"
                )

                xs_proj, ys_proj = rasterio.warp.transform(src_crs, dst_crs, xs, ys)
                ox, oy = rasterio.warp.transform(
                    src_crs, dst_crs, [origin_lon], [origin_lat]
                )
                ox, oy = ox[0], oy[0]

                xs = np.array(xs_proj) - ox
                ys = np.array(ys_proj) - oy

            vertices = np.column_stack((xs, ys, zs))

            # Create faces using vectorized indexing
            # Indices grid
            indices = np.arange(height * width).reshape((height, width))

            # Get corners for each quad (ignoring last row/col)
            tl = indices[:-1, :-1].flatten()  # Top-Left
            tr = indices[:-1, 1:].flatten()  # Top-Right
            bl = indices[1:, :-1].flatten()  # Bottom-Left
            br = indices[1:, 1:].flatten()  # Bottom-Right

            # Create two triangles per quad: (TL, BL, TR) and (TR, BL, BR)
            faces = np.column_stack(
                [
                    np.column_stack((tl, bl, tr)),
                    np.column_stack((tr, bl, br)),
                ]
            ).reshape(-1, 3)

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.export(str(output_path))
            logger.info(f"Terrain mesh saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate terrain mesh: {e}")
            raise

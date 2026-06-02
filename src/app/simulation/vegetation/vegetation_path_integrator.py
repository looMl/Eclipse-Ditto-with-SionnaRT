"""
Per-(TX, face) effective vegetation depth integrator.

Strategy: per-TX outer loop (memory-safe) + vectorised NumPy inner loop over
faces + AABB pre-filter that skips face batches whose segment bounding box
doesn't intersect any non-zero TCD pixel.

# TODO: Numba @njit escape hatch for production performance
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio.warp
from loguru import logger

from app.geomap_processor.processors.dem_processor import DemProcessor
from app.geomap_processor.utils.vegetation_field import VegetationField

_CHUNK_FACES = 10_000  # max faces per inner batch (~120 MB peak at 730 steps, float64)


@dataclass
class DemSampler:
    """Wraps DEM elevation data for optional 3-D canopy-height checks."""

    elevation: np.ndarray  # [H, W] float32, metres above reference
    transform: object  # rasterio Affine, EPSG:4326
    origin_lon: float
    origin_lat: float
    ref_elev: float = 0.0


class PathDepthIntegrator:
    """
    Walks LOS segments on the TCD/CHM raster grid to compute per-(TX, face)
    effective vegetation depth [m].

    Parameters
    ----------
    field : VegetationField
        Aligned TCD/CHM arrays in UTM CRS (output of VegetationManager).
    dem : DemSampler, optional
        Elevation model for 3-D in-canopy checks. When None, every LOS step
        that overlaps a non-zero TCD pixel is counted (conservative).
    step_m : float
        LOS sampling step in metres. Automatically capped to half the TCD
        raster cell size so there is no benefit in over-sampling.

    # TODO: Numba @njit escape hatch for production performance
    """

    def __init__(
        self,
        field: VegetationField,
        dem: Optional[DemSampler] = None,
        step_m: float = 1.0,
    ):
        self.field = field
        self.dem = dem

        # Cap step to half the raster cell size (cell width ≈ |transform.a| in UTM)
        cell_m = abs(field.transform.a)
        self._step_m = max(step_m, cell_m / 2.0)

        # Scene origin in absolute UTM: local (x, y) + (utm_ox, utm_oy) = absolute UTM
        utm_crs = DemProcessor._get_utm_crs(field.origin_lon, field.origin_lat)
        ox, oy = rasterio.warp.transform(
            "EPSG:4326", utm_crs, [field.origin_lon], [field.origin_lat]
        )
        self._utm_ox: float = ox[0]
        self._utm_oy: float = oy[0]

        # AABB of non-zero TCD in scene-local metres (None if TCD is all-zero)
        self._nz_bbox: Optional[Tuple[float, float, float, float]] = (
            self._nonzero_tcd_bbox()
        )
        if self._nz_bbox is None:
            logger.warning(
                "VegetationField has no non-zero TCD pixels — integrator is a no-op."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def integrate(self, tx_xyz: np.ndarray, rx_xyz_array: np.ndarray) -> np.ndarray:
        """
        Returns effective vegetation depth [m] for each of N face centroids.

        Parameters
        ----------
        tx_xyz : (3,) float
            TX position in scene-local metres [x, y, z].
        rx_xyz_array : (N, 3) float
            Face centroid positions in scene-local metres.

        Returns
        -------
        depth_eff : (N,) float32
        """
        N = len(rx_xyz_array)
        depth_eff = np.zeros(N, dtype="float32")

        if self._nz_bbox is None:
            return depth_eff

        active = self._aabb_filter(tx_xyz[:2], rx_xyz_array[:, :2])
        n_active = int(active.sum())
        if n_active == 0:
            return depth_eff

        logger.debug(f"AABB filter: {n_active}/{N} faces active for this TX.")
        depth_eff[active] = self._integrate_batch(tx_xyz, rx_xyz_array[active])
        return depth_eff

    def integrate_segments(
        self, p0_array: np.ndarray, p1_array: np.ndarray
    ) -> np.ndarray:
        """
        Returns effective vegetation depth [m] for each of N arbitrary segments.

        Per-segment counterpart to :meth:`integrate`: enables per-PATH vegetation
        attenuation (one call covers every segment across every reflected ray)
        rather than per-LINK (single TX→RX straight line).

        Parameters
        ----------
        p0_array : (N, 3) float
            Segment start points in scene-local metres.
        p1_array : (N, 3) float
            Segment end points in scene-local metres.

        Returns
        -------
        depth_eff : (N,) float32
        """
        p0 = np.asarray(p0_array, dtype="float64")
        p1 = np.asarray(p1_array, dtype="float64")
        N = len(p0)
        depth_eff = np.zeros(N, dtype="float32")

        if self._nz_bbox is None or N == 0:
            return depth_eff

        active = self._aabb_filter_segments(p0[:, :2], p1[:, :2])
        n_active = int(active.sum())
        if n_active == 0:
            return depth_eff

        p0_a, p1_a = p0[active], p1[active]
        depth_active = np.zeros(n_active, dtype="float32")
        for lo in range(0, n_active, _CHUNK_FACES):
            hi = min(lo + _CHUNK_FACES, n_active)
            depth_active[lo:hi] = self._integrate_segments_chunk(
                p0_a[lo:hi], p1_a[lo:hi]
            )
        depth_eff[active] = depth_active
        return depth_eff

    # ------------------------------------------------------------------
    # Chunked batch integrator (one TX, N filtered faces → loop over chunks)
    # ------------------------------------------------------------------

    def _integrate_batch(
        self, tx_xyz: np.ndarray, rx_xyz_array: np.ndarray
    ) -> np.ndarray:
        N = len(rx_xyz_array)
        depth = np.zeros(N, dtype="float32")
        for lo in range(0, N, _CHUNK_FACES):
            hi = min(lo + _CHUNK_FACES, N)
            depth[lo:hi] = self._integrate_chunk(tx_xyz, rx_xyz_array[lo:hi])
        return depth

    def _integrate_chunk(
        self, tx_xyz: np.ndarray, rx_xyz_array: np.ndarray
    ) -> np.ndarray:
        N = len(rx_xyz_array)
        tx_xy = tx_xyz[:2]
        rx_xy = rx_xyz_array[:, :2]

        delta_xy = rx_xy - tx_xy  # (N, 2)
        lengths = np.linalg.norm(delta_xy, axis=1)  # (N,)
        max_len = float(lengths.max())

        if max_len == 0.0:
            return np.zeros(N, dtype="float32")

        n_steps = max(2, int(np.ceil(max_len / self._step_m)))

        # Step centres: t ∈ (0, 1) uniformly spaced
        t = (np.arange(n_steps, dtype="float64") + 0.5) / n_steps  # (S,)

        # Sample positions: (N, S, 2)
        sample_xy = (
            tx_xy[np.newaxis, np.newaxis, :]
            + t[np.newaxis, :, np.newaxis] * delta_xy[:, np.newaxis, :]
        )

        tcd_s, chm_s = self._sample_field_batch(sample_xy)  # (N, S)

        # 3-D canopy check: count only steps where LOS height < ground_z + canopy_top
        if self.dem is not None:
            tx_z = float(tx_xyz[2])
            rx_z = rx_xyz_array[:, 2]
            z_link = tx_z + t[np.newaxis, :] * (rx_z[:, np.newaxis] - tx_z)  # (N, S)
            ground_z = self._sample_dem_batch(sample_xy)  # (N, S)
            in_canopy = z_link < (ground_z + chm_s)
        else:
            in_canopy = np.ones(tcd_s.shape, dtype=bool)

        # Mask steps that overshoot each face's actual path length
        t_dist = t[np.newaxis, :] * max_len  # distance along the max-length segment
        within_path = t_dist <= lengths[:, np.newaxis]  # (N, S)

        step_size = max_len / n_steps
        contribution = (in_canopy & within_path & (tcd_s > 0)).astype("float32")
        contribution *= tcd_s * step_size

        return contribution.sum(axis=1).astype("float32")

    def _integrate_segments_chunk(
        self, p0_array: np.ndarray, p1_array: np.ndarray
    ) -> np.ndarray:
        N = len(p0_array)
        p0_xy = p0_array[:, :2]
        p1_xy = p1_array[:, :2]

        delta_xy = p1_xy - p0_xy  # (N, 2)
        lengths = np.linalg.norm(delta_xy, axis=1)  # (N,)
        max_len = float(lengths.max())

        if max_len == 0.0:
            return np.zeros(N, dtype="float32")

        n_steps = max(2, int(np.ceil(max_len / self._step_m)))
        t = (np.arange(n_steps, dtype="float64") + 0.5) / n_steps  # (S,)

        # Sample positions: (N, S, 2)
        sample_xy = (
            p0_xy[:, np.newaxis, :]
            + t[np.newaxis, :, np.newaxis] * delta_xy[:, np.newaxis, :]
        )

        tcd_s, chm_s = self._sample_field_batch(sample_xy)

        if self.dem is not None:
            p0_z = p0_array[:, 2]
            p1_z = p1_array[:, 2]
            z_link = p0_z[:, np.newaxis] + t[np.newaxis, :] * (
                p1_z[:, np.newaxis] - p0_z[:, np.newaxis]
            )  # (N, S)
            ground_z = self._sample_dem_batch(sample_xy)
            in_canopy = z_link < (ground_z + chm_s)
        else:
            in_canopy = np.ones(tcd_s.shape, dtype=bool)

        t_dist = t[np.newaxis, :] * max_len
        within_path = t_dist <= lengths[:, np.newaxis]

        step_size = max_len / n_steps
        contribution = (in_canopy & within_path & (tcd_s > 0)).astype("float32")
        contribution *= tcd_s * step_size

        return contribution.sum(axis=1).astype("float32")

    # ------------------------------------------------------------------
    # Raster sampling (nearest-neighbour; 10 m grid makes bilinear redundant here)
    # ------------------------------------------------------------------

    def _sample_field_batch(
        self, sample_xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (tcd, chm) for (N, S, 2) scene-local XY points, shape (N, S).
        """
        t = self.field.transform
        H, W = self.field.tcd.shape

        abs_x = sample_xy[..., 0] + self._utm_ox
        abs_y = sample_xy[..., 1] + self._utm_oy

        col = np.round((abs_x - t.c) / t.a).astype(np.intp)
        row = np.round((abs_y - t.f) / t.e).astype(np.intp)
        np.clip(col, 0, W - 1, out=col)
        np.clip(row, 0, H - 1, out=row)

        return self.field.tcd[row, col], self.field.chm[row, col]

    def _sample_dem_batch(self, sample_xy: np.ndarray) -> np.ndarray:
        """
        Returns ground elevation [m] for (N, S, 2) scene-local XY points.
        Uses a linearised UTM→geographic conversion (error < 0.1 m for ≤ 10 km scenes).
        """
        dem = self.dem
        H, W = dem.elevation.shape
        t = dem.transform  # EPSG:4326

        lat_per_m = 1.0 / 111_320.0
        lon_per_m = 1.0 / (111_320.0 * np.cos(np.radians(dem.origin_lat)))

        lat = dem.origin_lat + sample_xy[..., 1] * lat_per_m
        lon = dem.origin_lon + sample_xy[..., 0] * lon_per_m

        col = np.round((lon - t.c) / t.a).astype(np.intp)
        row = np.round((lat - t.f) / t.e).astype(np.intp)
        np.clip(col, 0, W - 1, out=col)
        np.clip(row, 0, H - 1, out=row)

        return (dem.elevation[row, col] - dem.ref_elev).astype("float32")

    # ------------------------------------------------------------------
    # AABB pre-filter helpers
    # ------------------------------------------------------------------

    def _aabb_filter(self, tx_xy: np.ndarray, rx_xy_array: np.ndarray) -> np.ndarray:
        """Returns bool mask: True for faces whose segment AABB overlaps non-zero TCD bbox."""
        nz_min_x, nz_min_y, nz_max_x, nz_max_y = self._nz_bbox

        seg_min_x = np.minimum(tx_xy[0], rx_xy_array[:, 0])
        seg_max_x = np.maximum(tx_xy[0], rx_xy_array[:, 0])
        seg_min_y = np.minimum(tx_xy[1], rx_xy_array[:, 1])
        seg_max_y = np.maximum(tx_xy[1], rx_xy_array[:, 1])

        return (
            (seg_max_x >= nz_min_x)
            & (seg_min_x <= nz_max_x)
            & (seg_max_y >= nz_min_y)
            & (seg_min_y <= nz_max_y)
        )

    def _aabb_filter_segments(self, p0_xy: np.ndarray, p1_xy: np.ndarray) -> np.ndarray:
        """Per-segment counterpart to :meth:`_aabb_filter` (segments don't share an origin)."""
        nz_min_x, nz_min_y, nz_max_x, nz_max_y = self._nz_bbox

        seg_min_x = np.minimum(p0_xy[:, 0], p1_xy[:, 0])
        seg_max_x = np.maximum(p0_xy[:, 0], p1_xy[:, 0])
        seg_min_y = np.minimum(p0_xy[:, 1], p1_xy[:, 1])
        seg_max_y = np.maximum(p0_xy[:, 1], p1_xy[:, 1])

        return (
            (seg_max_x >= nz_min_x)
            & (seg_min_x <= nz_max_x)
            & (seg_max_y >= nz_min_y)
            & (seg_min_y <= nz_max_y)
        )

    def _nonzero_tcd_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Bounding box of non-zero TCD pixels in scene-local metres.
        Returns None when TCD is all-zero (no vegetation).
        """
        rows, cols = np.where(self.field.tcd > 0)
        if rows.size == 0:
            return None

        t = self.field.transform
        abs_x = t.c + (cols + 0.5) * t.a
        abs_y = t.f + (rows + 0.5) * t.e

        local_x = abs_x - self._utm_ox
        local_y = abs_y - self._utm_oy

        return (
            float(local_x.min()),
            float(local_y.min()),
            float(local_x.max()),
            float(local_y.max()),
        )

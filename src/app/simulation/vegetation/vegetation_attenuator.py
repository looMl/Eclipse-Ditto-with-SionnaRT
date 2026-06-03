import mitsuba as mi
import numpy as np
from loguru import logger
from tqdm import tqdm

from app.geomap_processor.utils.vegetation_field import VegetationField
from app.simulation.vegetation.itu_p833 import excess_loss_db
from app.simulation.vegetation.vegetation_path_integrator import PathDepthIntegrator


class VegetationAttenuator:
    """Computes per-(TX, face) vegetation attenuation [dB] from MeshRadioMap geometry.
    TX positions and face centroids are cached — both are fixed within a simulation run."""

    def __init__(
        self,
        scene,  # sionna.rt.Scene
        field: VegetationField,
        integrator: PathDepthIntegrator,
    ):
        self.scene = scene
        self.field = field
        self.integrator = integrator

        self._tx_positions: list[np.ndarray] = self._extract_tx_positions()
        self._face_centroids: np.ndarray | None = None  # cached on first use

    def compute_attenuation_field(
        self,
        radio_map,  # sionna.rt.MeshRadioMap
        freq_hz: float,
        leaf_state: str = "in_leaf",
    ) -> np.ndarray:
        """Returns attenuation_db [num_tx, num_faces] float32."""
        if self._face_centroids is None:
            self._face_centroids = self._extract_face_centroids(radio_map)

        face_centroids = self._face_centroids
        num_tx = len(self._tx_positions)
        num_faces = len(face_centroids)

        # Deduplicate to TCD raster resolution — many mesh triangles share the same
        # 10 m pixel, so we integrate once per unique pixel and broadcast back.
        pixel_reps, face_to_pixel = self._deduplicate_to_raster_pixels(face_centroids)
        n_unique = len(pixel_reps)

        logger.info(
            f"Vegetation attenuation: {num_tx} TXs × {n_unique} unique TCD pixels "
            f"(from {num_faces} faces) @ {freq_hz / 1e9:.3f} GHz [{leaf_state}]"
        )

        attenuation_db = np.zeros((num_tx, num_faces), dtype="float32")

        for i, tx_pos in enumerate(
            tqdm(self._tx_positions, desc="Vegetation attenuation", unit="TX")
        ):
            depths_unique = self.integrator.integrate(tx_pos, pixel_reps)
            attenuation_db[i] = excess_loss_db(
                depths_unique[face_to_pixel], freq_hz, leaf_state
            ).astype("float32")

        logger.info(
            f"Attenuation range: [{attenuation_db.min():.1f}, "
            f"{attenuation_db.max():.1f}] dB"
        )
        return attenuation_db

    def _deduplicate_to_raster_pixels(self, face_centroids: np.ndarray):
        """
        Maps N face centroids to their TCD raster pixel, returning:
          pixel_reps  : (n_unique, 3) mean centroid per pixel
          face_to_pixel: (N,) index mapping face → unique pixel row in pixel_reps
        """
        t = self.field.transform
        H, W = self.field.tcd.shape
        utm_ox, utm_oy = self.integrator.utm_origin

        abs_x = face_centroids[:, 0].astype("float64") + utm_ox
        abs_y = face_centroids[:, 1].astype("float64") + utm_oy

        col = np.clip(np.round((abs_x - t.c) / t.a).astype(np.intp), 0, W - 1)
        row = np.clip(np.round((abs_y - t.f) / t.e).astype(np.intp), 0, H - 1)

        pixel_ids = row * W + col  # (N,) — unique int per TCD cell
        _, face_to_pixel, counts = np.unique(
            pixel_ids, return_inverse=True, return_counts=True
        )

        n_unique = len(counts)
        pixel_reps = np.zeros((n_unique, 3), dtype="float32")
        np.add.at(pixel_reps, face_to_pixel, face_centroids)
        pixel_reps /= counts[:, np.newaxis]

        return pixel_reps, face_to_pixel

    def _extract_tx_positions(self) -> list[np.ndarray]:
        """Returns TX positions as list of (3,) float32 arrays in scene-local metres."""
        positions = []
        for tx in self.scene.transmitters.values():
            pos = np.array(tx.position, dtype="float32").flatten()
            positions.append(pos[:3])
        return positions

    def _extract_face_centroids(self, radio_map) -> np.ndarray:
        """Returns (F, 3) face centroid positions in scene-local metres."""
        mesh = radio_map.measurement_surface
        params = mi.traverse(mesh)

        # Mitsuba stores positions/faces as flat buffers → reshape before use
        verts = np.array(params["vertex_positions"], copy=False).reshape(-1, 3)
        faces = np.array(params["faces"], copy=False).reshape(-1, 3)

        centroids = verts[faces].mean(axis=1).astype("float32")  # (F, 3)
        logger.info(
            f"Extracted {len(centroids)} face centroids from measurement surface."
        )
        return centroids

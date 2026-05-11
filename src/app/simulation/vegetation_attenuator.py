from typing import List, Optional

import mitsuba as mi
import numpy as np
from loguru import logger
from tqdm import tqdm

from app.geomap_processor.utils.vegetation_field import VegetationField
from app.simulation.itu_p833 import excess_loss_db
from app.simulation.vegetation_path_integrator import PathDepthIntegrator


class VegetationAttenuator:
    """
    Computes per-(TX, face) vegetation attenuation in dB from the MeshRadioMap geometry.

    TX positions are extracted once from the scene and cached; face centroids are
    extracted on the first call to compute_attenuation_field and cached thereafter
    (both are fixed within a single simulation run).
    """

    def __init__(
        self,
        scene,  # sionna.rt.Scene
        field: VegetationField,
        integrator: PathDepthIntegrator,
    ):
        self.scene = scene
        self.field = field
        self.integrator = integrator

        self._tx_positions: List[np.ndarray] = self._extract_tx_positions()
        self._face_centroids: Optional[np.ndarray] = None  # cached on first use

    def compute_attenuation_field(
        self,
        radio_map,  # sionna.rt.MeshRadioMap
        freq_hz: float,
        leaf_state: str = "in_leaf",
    ) -> np.ndarray:
        """
        Returns attenuation in dB, shape [num_tx, num_faces].

        Outer loop over TXs is tracked with a tqdm progress bar.
        Inner vectorised loop over faces is handled by PathDepthIntegrator.

        Parameters
        ----------
        radio_map : MeshRadioMap
            Output of RadioMapSolver; provides the measurement surface mesh.
        freq_hz : float
            Carrier frequency in Hz.
        leaf_state : {"in_leaf", "out_of_leaf"}
            Foliage state passed to the P.833 model.

        Returns
        -------
        attenuation_db : np.ndarray, shape [num_tx, num_faces], float32
        """
        if self._face_centroids is None:
            self._face_centroids = self._extract_face_centroids(radio_map)

        face_centroids = self._face_centroids
        num_tx = len(self._tx_positions)
        num_faces = len(face_centroids)

        logger.info(
            f"Vegetation attenuation: {num_tx} TXs × {num_faces} faces "
            f"@ {freq_hz / 1e9:.3f} GHz [{leaf_state}]"
        )

        attenuation_db = np.zeros((num_tx, num_faces), dtype="float32")

        for i, tx_pos in enumerate(
            tqdm(self._tx_positions, desc="Vegetation attenuation", unit="TX")
        ):
            depth_eff = self.integrator.integrate(tx_pos, face_centroids)
            attenuation_db[i] = excess_loss_db(depth_eff, freq_hz, leaf_state).astype(
                "float32"
            )

        logger.info(
            f"Attenuation range: [{attenuation_db.min():.1f}, "
            f"{attenuation_db.max():.1f}] dB"
        )
        return attenuation_db

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_tx_positions(self) -> List[np.ndarray]:
        """Returns TX positions as list of (3,) float32 arrays in scene-local metres."""
        positions = []
        for tx in self.scene.transmitters.values():
            pos = np.array(tx.position, dtype="float32").flatten()
            positions.append(pos[:3])
        return positions

    def _extract_face_centroids(self, radio_map) -> np.ndarray:
        """
        Returns (F, 3) face centroid positions in scene-local metres.
        Extracted from the measurement surface via mi.traverse.
        """
        mesh = radio_map.measurement_surface
        params = mi.traverse(mesh)

        # vertex_positions: flat buffer (V*3,) → reshape to (V, 3)
        verts = np.array(params["vertex_positions"], copy=False).reshape(-1, 3)
        # faces: flat buffer (F*3,) → reshape to (F, 3)
        faces = np.array(params["faces"], copy=False).reshape(-1, 3)

        centroids = verts[faces].mean(axis=1).astype("float32")  # (F, 3)
        logger.info(
            f"Extracted {len(centroids)} face centroids from measurement surface."
        )
        return centroids

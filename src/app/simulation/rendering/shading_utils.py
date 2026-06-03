import numpy as np
import mitsuba as mi
from sionna.rt.utils.render import radio_map_texture
from sionna.rt.utils.meshes import clone_mesh
from sionna.rt import MeshRadioMap


def prepare_gouraud_shading_for_radio_map(
    radio_map: MeshRadioMap,
    metric: str,
    vmin: float,
    vmax: float,
    cmap: str,
    attenuation_db: np.ndarray | None = None,
) -> mi.Shape:
    """Converts a flat-shaded MeshRadioMap into a Mitsuba shape with per-vertex colors
    for Gouraud (smooth-gradient) shading instead of discrete triangle colors."""
    # 4G LTE 15 MHz = 900 subcarriers
    NUM_SUBCARRIERS = 900

    if attenuation_db is not None:
        # Subtract vegetation attenuation in dB before max-aggregating.
        # Subtraction in dB == division in linear: rm_attenuated = rm / 10^(A/10).
        num_tx = attenuation_db.shape[0]
        rm_values = np.zeros(attenuation_db.shape[1], dtype="float64")
        for k in range(num_tx):
            rm_k = radio_map.transmitter_radio_map(metric=metric, tx=k).numpy()
            if metric == "rss":
                rm_k = rm_k * NUM_SUBCARRIERS
            rm_k = rm_k / np.power(10.0, attenuation_db[k] / 10.0)
            np.maximum(rm_values, rm_k, out=rm_values)
    else:
        # tx=None: Sionna aggregates over all TXs with max internally.
        rm_values = radio_map.transmitter_radio_map(metric=metric, tx=None).numpy()
        if metric == "rss":
            rm_values *= NUM_SUBCARRIERS

    mesh = radio_map.measurement_surface
    num_vertices = mesh.vertex_count()

    # mi.traverse is safe for Dr.Jit/Mitsuba variants
    params = mi.traverse(mesh)
    faces = np.array(params["faces"], copy=False)

    vertex_values = np.zeros(num_vertices)
    vertex_counts = np.zeros(num_vertices)
    expanded_values = np.repeat(rm_values, 3)
    np.add.at(vertex_values, faces, expanded_values)
    np.add.at(vertex_counts, faces, 1)
    mask = vertex_counts > 0
    vertex_values[mask] /= vertex_counts[mask]

    texture, opacity = radio_map_texture(
        vertex_values, db_scale=True, rm_cmap=cmap, vmin=vmin, vmax=vmax
    )

    # 'mesh_attribute' tells Mitsuba to interpolate per-vertex data across the face.
    bsdf = {
        "type": "mask",
        "opacity": {
            "type": "mesh_attribute",
            "name": "vertex_opacity",
        },
        "nested": {
            "type": "diffuse",
            "reflectance": 0.0,
        },
    }

    emitter = {
        "type": "twosided_area",
        "nested": {
            "type": "area",
            "radiance": {
                "type": "mesh_attribute",
                "name": "vertex_rm_values",
            },
        },
    }

    props = mi.Properties()
    props["bsdf"] = mi.load_dict(bsdf)
    props["emitter"] = mi.load_dict(emitter)

    cloned_shape = clone_mesh(mesh, props=props)
    cloned_shape.add_attribute("vertex_opacity", 1, opacity.astype(np.float32))
    cloned_shape.add_attribute(
        "vertex_rm_values", 3, texture.ravel().astype(np.float32)
    )

    return cloned_shape

import mitsuba as mi
from sionna.rt import Camera
from sionna.rt.renderer import visual_scene_from_wireless_scene
from app.config import settings

class VisualSceneBuilder:
    """
    Responsible for constructing the visual representation of the scene 
    (geometry, lights, camera) for rendering.
    """

    @staticmethod
    def create_sensor(resolution: list[int], pixel_format: str = "rgba") -> mi.Sensor:
        """
        Creates a Mitsuba sensor (camera) based on configuration.
        """
        my_cam = Camera(
            position=settings.sionnart.camera.position,
            look_at=settings.sionnart.camera.look_at,
        )

        matrix = my_cam.world_transform.matrix.numpy().squeeze().reshape(4, 4)

        return mi.load_dict(
            {
                "type": "perspective",
                "to_world": mi.ScalarTransform4f(matrix.tolist()),
                "fov": 45.0,
                "film": {
                    "type": "hdrfilm",
                    "width": resolution[0],
                    "height": resolution[1],
                    "pixel_format": pixel_format,
                    "rfilter": {"type": "gaussian"},
                },
            }
        )

    @staticmethod
    def build_visual_scene(scene, sensor: mi.Sensor, max_depth: int = 8) -> mi.Scene:
        """
        Converts the wireless scene to a visual Mitsuba scene with custom lighting.
        """
        # 1. Base conversion from Sionna
        visual_dict = visual_scene_from_wireless_scene(scene, sensor, max_depth=max_depth)

        # 2. Inject Custom Lighting (Sun + Sky)
        # Enable emitters to allow sky visibility
        if "integrator" in visual_dict:
            visual_dict["integrator"]["hide_emitters"] = False

        # Add Directional Sun
        visual_dict["sun"] = {
            "type": "directional",
            "direction": [0.5, 0.5, -1.0],
            "irradiance": {
                "type": "rgb",
                "value": [1.0, 1.0, 1.0],
            },
        }

        # 3. Add Transmitter Markers
        VisualSceneBuilder._add_transmitter_markers(scene, visual_dict)

        return mi.load_dict(visual_dict)

    @staticmethod
    def _add_transmitter_markers(wireless_scene, visual_dict: dict):
        """Adds red sphere markers for transmitters to the visual scene dict."""
        marker_bsdf = {
            "type": "twosided",
            "nested": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [1.0, 0.0, 0.0]},
            },
        }
        for i, tx in enumerate(wireless_scene.transmitters.values()):
            p = tx.position.numpy().squeeze()
            visual_dict[f"marker-{i}"] = {
                "type": "sphere",
                "center": [float(p[0]), float(p[1]), float(p[2])],
                "radius": 15.0,
                "bsdf": marker_bsdf,
            }

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from loguru import logger


class SceneXMLUpdater:
    def __init__(self, scene_path: Path):
        self.scene_path = scene_path
        self.tree = None
        self.root = None
        self._load()

    def _load(self) -> None:
        try:
            self.tree = ET.parse(self.scene_path)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse scene.xml at {self.scene_path}: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"scene.xml not found at {self.scene_path}")
            raise

    def save(self) -> None:
        try:
            self.tree.write(self.scene_path, encoding="utf-8", xml_declaration=True)
            logger.success(f"Saved changes to {self.scene_path}")
        except Exception as e:
            logger.error(f"Failed to save scene.xml: {e}")
            raise

    def remove_shapes_by_filenames(self, filenames: set[str]) -> str | None:
        """
        Removes shapes referencing any of the given filenames.
        Returns the BSDF ID of the first removed shape, or None if nothing is removed.
        """
        shapes_to_remove = []
        captured_bsdf_id = None

        for shape in self.root.findall("shape"):
            filename_node = shape.find("string[@name='filename']")
            if filename_node is None:
                continue

            fname = filename_node.get("value")
            if fname in filenames:
                shapes_to_remove.append(shape)
                if not captured_bsdf_id:
                    captured_bsdf_id = self._get_bsdf_id(shape)

        if not shapes_to_remove:
            return None

        for shape in shapes_to_remove:
            self.root.remove(shape)

        logger.info(f"Removed {len(shapes_to_remove)} shapes from scene.xml")
        return captured_bsdf_id

    def add_mesh_shape(self, filename: str, shape_id: str, bsdf_id: str) -> None:
        """Adds a new PLY shape to the scene."""
        existing = self.root.find(f".//shape[@id='{shape_id}']")
        if existing is not None:
            logger.info(f"Shape with id '{shape_id}' already exists. Skipping add.")
            return

        new_shape = ET.SubElement(self.root, "shape")
        new_shape.set("type", "ply")
        new_shape.set("id", shape_id)

        fn = ET.SubElement(new_shape, "string")
        fn.set("name", "filename")
        fn.set("value", filename)

        ref = ET.SubElement(new_shape, "ref")
        ref.set("name", "bsdf")
        ref.set("id", bsdf_id)

        logger.info(
            f"Added shape '{shape_id}' pointing to '{filename}' with BSDF '{bsdf_id}'"
        )

    def get_material_colors(self) -> dict[str, tuple[float, float, float]]:
        """
        Extracts material RGB reflectance values from the XML as tuple.
        """
        xml_colors = {}
        for bsdf in self.root.findall(".//bsdf"):
            mat_id = bsdf.get("id")
            if not mat_id:
                continue

            rgb_node = bsdf.find(".//rgb[@name='reflectance']")
            if rgb_node is not None:
                val_str = rgb_node.get("value")
                if val_str:
                    try:
                        r, g, b = map(float, val_str.split())
                        xml_colors[mat_id] = (r, g, b)
                    except ValueError:
                        logger.debug(
                            f"Skipping malformed reflectance '{val_str}' for material '{mat_id}'"
                        )
                        continue
        return xml_colors

    def get_projection_info(self) -> dict[str, Any]:
        """Returns center_lat, center_lon, utm_zone from scene XML defaults.
        Raises ValueError if any key is missing."""
        info: dict[str, Any] = {}
        defaults = {
            "center_lat": "scenegen_center_lat",
            "center_lon": "scenegen_center_lon",
            "utm_zone": "scenegen_UTM_zone",
        }
        for key, xml_name in defaults.items():
            node = self.root.find(f".//default[@name='{xml_name}']")
            if node is not None:
                val = node.get("value")
                info[key] = float(val) if "center" in key else val
        missing = [k for k in defaults if k not in info]
        if missing:
            raise ValueError(
                f"scene.xml is missing projection metadata: {missing}. "
                "Ensure scenegen_center_lat, scenegen_center_lon, and "
                "scenegen_UTM_zone are present as <default> elements."
            )
        return info

    def _get_bsdf_id(self, shape: ET.Element) -> str | None:
        ref = shape.find("ref[@name='bsdf']")
        return ref.get("id") if ref is not None else None

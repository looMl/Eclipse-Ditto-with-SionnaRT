import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Set, Optional
from loguru import logger


class SceneXMLUpdater:
    """Service for updating the scene.xml file."""

    def __init__(self, scene_path: Path):
        self.scene_path = scene_path
        self.tree = None
        self.root = None
        self._load()

    def _load(self) -> None:
        """Parses the XML file."""
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
        """Writes changes back to the file."""
        try:
            self.tree.write(self.scene_path, encoding="utf-8", xml_declaration=True)
            logger.success(f"Saved changes to {self.scene_path}")
        except Exception as e:
            logger.error(f"Failed to save scene.xml: {e}")
            raise

    def remove_shapes_by_filenames(self, filenames: Set[str]) -> Optional[str]:
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

    def _get_bsdf_id(self, shape: ET.Element) -> Optional[str]:
        ref = shape.find("ref[@name='bsdf']")
        return ref.get("id") if ref is not None else None

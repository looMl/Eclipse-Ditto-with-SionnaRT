from loguru import logger
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scene_generation.itu_materials import ITU_MATERIALS

# Cache material list for index-based access
_MATERIALS_LIST = list(ITU_MATERIALS.items())


def resolve_material(idx: int) -> str:
    """Resolves material index to name safely."""
    try:
        return _MATERIALS_LIST[idx][0]
    except IndexError:
        logger.warning(f"Invalid material index {idx}. Using default.")
        return _MATERIALS_LIST[0][0]


@dataclass(frozen=True)
class BoundingBox:
    """Represents the geographical bounding box."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def validate(self) -> None:
        """Ensures coordinates form a valid bounding box."""
        if self.min_lon >= self.max_lon:
            raise ValueError(
                f"min_lon ({self.min_lon}) must be less than max_lon ({self.max_lon})"
            )
        if self.min_lat >= self.max_lat:
            raise ValueError(
                f"min_lat ({self.min_lat}) must be less than max_lat ({self.max_lat})"
            )

    def to_dict(self) -> Dict[str, float]:
        """Returns the bbox as a dictionary for compatibility."""
        return {
            "min_lon": self.min_lon,
            "min_lat": self.min_lat,
            "max_lon": self.max_lon,
            "max_lat": self.max_lat,
        }

    @property
    def polygon_points(self) -> List[List[float]]:
        """
        Returns the counter-clockwise polygon points for the bbox.
        Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left -> Top-Left (Closed Loop)
        """
        return [
            [self.min_lon, self.min_lat],
            [self.min_lon, self.max_lat],
            [self.max_lon, self.max_lat],
            [self.max_lon, self.min_lat],
            [self.min_lon, self.min_lat],
        ]

    @property
    def center(self) -> Tuple[float, float]:
        """Returns the center (lon, lat) of the bounding box."""
        return (self.min_lon + self.max_lon) / 2.0, (self.min_lat + self.max_lat) / 2.0


@dataclass(frozen=True)
class MaterialConfig:
    """Configuration for material indices."""

    ground_idx: int = 13  # Default: medium dry ground
    rooftop_idx: int = 2  # Default: brick
    wall_idx: int = 1  # Default: concrete

import yaml
from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from loguru import logger
import sys

# --- Models schemas ---


class CameraSettings(BaseModel):
    position: List[float]
    orientation: List[float]
    look_at: List[float]


class RenderingSettings(BaseModel):
    resolution: List[int]
    num_samples: int
    show_devices: bool


class CoverageSettings(BaseModel):
    samples_per_tx: int
    max_depth: int
    metric: str
    vmin: float | None = None
    vmax: float | None = None
    max_num_paths_per_src: int = 200_000


class DiffuseScatteringSettings(BaseModel):
    enabled: bool = False
    scattering_coefficient: float = 0.25


class VegetationSettings(BaseModel):
    enabled: bool = False
    frequency_hz: float = 1.8e9
    leaf_state: Literal["in_leaf", "out_of_leaf"] = "in_leaf"
    tcd_source: str = "esa_worldcover"
    chm_source: str = "eth_global_2020"
    raster_step_m: float = 1.0
    heuristic_heights: Dict[str, float] = Field(default_factory=dict)
    mode: Literal["per_link", "per_path"] = "per_link"


class ShadowingSettings(BaseModel):
    enabled: bool = True
    sigma_db: float = 6.0


class Geo2SigmapSettings(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class SionnartSettings(BaseModel):
    scene_name: str
    camera: CameraSettings
    rendering: RenderingSettings
    coverage: CoverageSettings
    vegetation: Optional[VegetationSettings] = None
    diffuse_scattering: Optional[DiffuseScatteringSettings] = None
    shadowing: Optional[ShadowingSettings] = None


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseModel):
    logging: LoggingSettings
    sionnart: SionnartSettings
    geo2sigmap: Geo2SigmapSettings


# --- Load logic ---


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_settings() -> Settings:
    """Loads the config file and validates it with Pydantic."""
    config_path = get_project_root() / "config.yaml"
    logger.info(f"Loading configuration from: {config_path}")

    if not config_path.exists():
        logger.critical(f"Config file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        return Settings(**raw_config)

    except Exception as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)


try:
    settings = load_settings()
except Exception:
    sys.exit(1)

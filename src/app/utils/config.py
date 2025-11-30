import yaml
from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel, Field, field_validator
from loguru import logger
import sys

# --- Models schemas ---


class MQTTPublisherSettings(BaseModel):
    client_id_prefix: str
    thing_id: str
    base_topic: str
    publish_interval_seconds: int
    num_messages: int
    initial_translation: float
    translation_increment: float


class MQTTWorkerSettings(BaseModel):
    client_id_prefix: str
    base_topic: str


class MQTTSettings(BaseModel):
    broker_host: str
    broker_port: int
    keepalive: int
    publisher: MQTTPublisherSettings
    worker: MQTTWorkerSettings


class TransmitterSettings(BaseModel):
    position: List[float]

    @field_validator("position")
    def check_len(cls, v):
        if len(v) != 3:
            raise ValueError("Position must be a list of 3 coordinates [x, y, z]")
        return v


class CameraSettings(BaseModel):
    position: List[float]
    orientation: List[float]
    look_at: List[float]


class SimulationSettings(BaseModel):
    max_depth: int
    num_samples: float


class RenderingSettings(BaseModel):
    resolution: List[int]
    num_samples: int
    show_devices: bool


class SionnartSettings(BaseModel):
    script_name: str
    transmitter: TransmitterSettings
    camera: CameraSettings
    simulation: SimulationSettings = Field(..., alias="paths_simulation")
    rendering: RenderingSettings


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseModel):
    logging: LoggingSettings
    mqtt: MQTTSettings
    sionnart: SionnartSettings


# --- Load logic ---


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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

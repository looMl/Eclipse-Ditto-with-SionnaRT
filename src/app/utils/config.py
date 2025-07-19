import yaml
import os
import sys
from loguru import logger
from dataclasses import dataclass

def get_project_root() -> str:
    # This navigates up three levels from app/utils/config.py to the project root.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CONFIG_PATH = os.path.join(get_project_root() , "config.yaml")

@dataclass(frozen=True)
class MQTTPublisherSettings:
    client_id_prefix: str
    thing_id: str
    base_topic: str
    publish_interval_seconds: int
    num_messages: int
    initial_translation: float
    translation_increment: float

@dataclass(frozen=True)
class MQTTSubscriberSettings:
    client_id_prefix: str
    base_topic: str

@dataclass(frozen=True)
class MQTTSettings:
    broker_host: str
    broker_port: int
    keepalive: int
    publisher: MQTTPublisherSettings
    subscriber: MQTTSubscriberSettings

@dataclass(frozen=True)
class SionnartSettings:
    script_name: str

@dataclass(frozen=True)
class LoggingSettings:
    level: str

@dataclass(frozen=True)
class Settings:
    logging: LoggingSettings
    mqtt: MQTTSettings
    sionnart: SionnartSettings

def _load_config(config_path: str) -> Settings:
    """
    Loads YAML config file and maps values on dataclass objects.
    Critical exception if file not found or incorrect.
    """
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return Settings(
            logging=LoggingSettings(**config_data['logging']),
            mqtt=MQTTSettings(
                **{k: v for k, v in config_data['mqtt'].items() if k not in ['publisher', 'subscriber']},
                publisher=MQTTPublisherSettings(**config_data['mqtt']['publisher']),
                subscriber=MQTTSubscriberSettings(**config_data['mqtt']['subscriber'])
            ),
            sionnart=SionnartSettings(**config_data['sionnart'])
        )
    except FileNotFoundError:
        logger.critical(f"Couldn't find config file in '{config_path}'. Cannot start.")
        sys.exit(1)
    except (yaml.YAMLError, TypeError, KeyError) as e:
        logger.critical(f"Error reading or parsing the config file '{config_path}': {e}", exc_info=True)
        sys.exit(1)

settings = _load_config(CONFIG_PATH)
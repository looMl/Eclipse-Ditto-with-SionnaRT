import json
import random
import time
import sys
from loguru import logger
from paho.mqtt import client as mqtt_client

from app.utils.config import settings as cfg

logger.remove()
logger.add(sys.stdout, level=cfg.logging.level)


def connect_mqtt() -> mqtt_client.Client:
    client_id = f"{cfg.mqtt.publisher.client_id_prefix}{random.randint(0, 1000)}"
    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id)

    try:
        client.connect(cfg.mqtt.broker_host, cfg.mqtt.broker_port, cfg.mqtt.keepalive)
        logger.info(f"Connected to MQTT Broker at {cfg.mqtt.broker_host}")
        return client
    except Exception as e:
        logger.critical(f"Failed to connect to MQTT Broker: {e}")
        sys.exit(1)


def get_next_position(current_x: float, current_y: float, increment: float):
    """
    Calculates the next position.
    - Y axis: moves forward by the configured increment value.
    - X axis: adds a small random 'wobble' to simulate natural walking.
    """
    new_y = current_y + increment
    wobble = increment * 0.2  # Lateral deviation
    new_x = current_x + random.uniform(-wobble, wobble)

    return new_x, new_y


def run():
    pub_settings = cfg.mqtt.publisher
    client = connect_mqtt()
    client.loop_start()

    # Topic structure: devices/in/<thing_id>
    topic = f"{pub_settings.base_topic}/{pub_settings.thing_id}"

    # Initial State
    current_y = float(pub_settings.initial_translation)
    current_x = 0.0
    fixed_z = 1.5

    logger.info(f"Starting Device Simulation. Target Topic: {topic}")
    logger.info(f"Initial Pos: [x={current_x}, y={current_y}, z={fixed_z}]")

    msg_count = 0
    limit = pub_settings.num_messages

    try:
        while True:
            if limit > 0 and msg_count >= limit:
                logger.info("Message limit reached. Stopping.")
                break

            current_x, current_y = get_next_position(
                current_x, current_y, pub_settings.translation_increment
            )

            payload = {
                "thingId": pub_settings.thing_id,
                "position": [current_x, current_y, fixed_z],
                "orientation": [0.0, 0.0, 0.0],
            }

            payload_str = json.dumps(payload)
            info = client.publish(topic, payload_str)

            if info.rc == mqtt_client.MQTT_ERR_SUCCESS:
                logger.info(f"Sent Telemetry: {payload_str}")
            else:
                logger.error(f"Failed to publish message. Return code: {info.rc}")

            msg_count += 1

            time.sleep(pub_settings.publish_interval_seconds)

    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        logger.info("Device disconnected.")


if __name__ == "__main__":
    run()

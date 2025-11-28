import json
import random
import time
import sys
from loguru import logger
from paho.mqtt import client as mqtt_client

from app.utils.config import settings as cfg
from app.core.mqtt import MQTTClientWrapper

logger.remove()
logger.add(sys.stdout, level=cfg.logging.level)


class DeviceSimulator:
    def __init__(self, settings):
        self.pub_settings = settings.mqtt.publisher
        self.mqtt_settings = settings.mqtt

        self.client = MQTTClientWrapper(
            self.mqtt_settings.broker_host,
            self.mqtt_settings.broker_port,
            self.mqtt_settings.keepalive,
            self.pub_settings.client_id_prefix,
        )

        self.current_x = 0.0
        self.current_y = float(self.pub_settings.initial_translation)
        self.fixed_z = 1.5
        self.topic = f"{self.pub_settings.base_topic}/{self.pub_settings.thing_id}"

    def get_next_position(self, increment: float):
        """
        Calculates the next position.
        - Y axis: moves forward by the configured increment value.
        - X axis: adds a small random 'wobble' to simulate natural walking.
        """
        self.current_y += increment
        wobble = increment * 0.2  # Lateral deviation
        self.current_x += random.uniform(-wobble, wobble)

    def run(self):
        try:
            self.client.connect()
            self.client.start()

            logger.info(f"Starting Device Simulation. Target Topic: {self.topic}")
            logger.info(
                f"Initial Pos: [x={self.current_x}, y={self.current_y}, z={self.fixed_z}]"
            )

            msg_count = 0
            limit = self.pub_settings.num_messages

            while True:
                if limit > 0 and msg_count >= limit:
                    logger.info("Message limit reached. Stopping.")
                    break

                self.get_next_position(self.pub_settings.translation_increment)

                payload = {
                    "thingId": self.pub_settings.thing_id,
                    "position": [self.current_x, self.current_y, self.fixed_z],
                    "orientation": [0.0, 0.0, 0.0],
                }

                payload_str = json.dumps(payload)
                info = self.client.publish(self.topic, payload_str)

                if info.rc == mqtt_client.MQTT_ERR_SUCCESS:
                    logger.info(f"Sent Telemetry: {payload_str}")
                else:
                    logger.error(f"Failed to publish message. Return code: {info.rc}")

                msg_count += 1

                time.sleep(self.pub_settings.publish_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Simulation stopped by user.")
        except Exception as e:
            logger.critical(f"Simulation failed: {e}")
        finally:
            self.client.stop()
            logger.info("Device disconnected.")


if __name__ == "__main__":
    simulator = DeviceSimulator(cfg)
    simulator.run()

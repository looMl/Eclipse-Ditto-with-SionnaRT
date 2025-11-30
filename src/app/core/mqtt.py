import random
from typing import Any
from paho.mqtt import client as mqtt_client
from loguru import logger


class MQTTClientWrapper:
    def __init__(
        self, broker_host: str, broker_port: int, keepalive: int, client_id_prefix: str
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.keepalive = keepalive
        self.client_id = f"{client_id_prefix}{random.randint(0, 1000)}"
        self.client = mqtt_client.Client(
            mqtt_client.CallbackAPIVersion.VERSION2, self.client_id
        )
        self._setup_callbacks()

    def _setup_callbacks(self):
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"Connected to MQTT Broker at {self.broker_host}")
        else:
            logger.error(f"Failed to connect to MQTT Broker, return code {rc}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        logger.info("Disconnected from MQTT Broker")

    def connect(self):
        try:
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
        except Exception as e:
            logger.critical(f"Failed to connect to MQTT Broker: {e}")
            raise

    def start(self):
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic: str, payload: str) -> Any:
        return self.client.publish(topic, payload)

    def subscribe(self, topic: str):
        self.client.subscribe(topic)
        logger.info(f"Subscribed to {topic}")

    def loop_forever(self):
        self.client.loop_forever()

    def set_user_data(self, data: Any):
        self.client.user_data_set(data)

    def set_on_message(self, callback):
        self.client.on_message = callback

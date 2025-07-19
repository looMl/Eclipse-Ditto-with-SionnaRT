import json
import sys
import random
from loguru import logger
from paho.mqtt import client as mqtt_client

from app.utils.config import settings
from app.SionnaRTSimulator import SionnaRTSimulator

logger.remove()
logger.add(sys.stdout, level=settings.logging.level)


def on_connect_factory(subscribe_topic: str):
    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0 and client.is_connected():
            logger.info(
                f"Successfully connected to MQTT Broker at {settings.mqtt.broker_host}:{settings.mqtt.broker_port}"
            )
            try:
                client.subscribe(subscribe_topic)
                logger.info(f"Subscribed to topic: {subscribe_topic}")
            except Exception as e:
                logger.error(
                    f"Failed to subscribe to topic {subscribe_topic}: {e}",
                    exc_info=True,
                )
        else:
            logger.error(
                f"Failed to connect to MQTT Broker, return code: {reason_code}"
            )

    return on_connect


def on_disconnect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        logger.info("Successfully disconnected from MQTT Broker.")
    else:
        logger.warning(
            f"Unexpectedly disconnected from MQTT Broker with result code: {reason_code}."
        )


def on_message_factory(simulator: SionnaRTSimulator):
    def on_message(client, userdata, message):
        logger.info(f"Received message on topic: {message.topic}")

        try:
            payload_str = message.payload.decode("utf-8")
            payload_data = json.loads(payload_str)
            logger.debug(f"Raw payload string: {payload_str}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning(
                f"Failed to decode or parse JSON payload on topic {message.topic}. Skipping.",
                exc_info=True,
            )
            return

        position = payload_data.get("position")
        orientation = payload_data.get("orientation")

        if not all((position, orientation)):
            logger.warning(
                f"Missing 'position' or 'orientation' in payload on topic {message.topic}. Skipping."
            )
            return

        logger.info(f"Extracted Position: {position}, Orientation: {orientation}")

        try:
            simulator.run_simulation(position, orientation)
        except Exception as e:
            logger.error(
                f"An error occurred during simulation for message on topic {message.topic}: {e}",
                exc_info=True,
            )

    return on_message


def connect_mqtt(simulator: SionnaRTSimulator) -> mqtt_client.Client | None:
    sub_settings = settings.mqtt.subscriber
    client_id = f"{sub_settings.client_id_prefix}{random.randint(0, 1000)}"
    subscribe_topic = f"{sub_settings.base_topic}/#"

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id)
    client.on_connect = on_connect_factory(subscribe_topic)
    client.on_disconnect = on_disconnect
    client.on_message = on_message_factory(simulator)

    try:
        logger.info(
            f"Attempting to connect to MQTT Broker at {settings.mqtt.broker_host}:{settings.mqtt.broker_port}..."
        )
        client.connect(
            settings.mqtt.broker_host,
            settings.mqtt.broker_port,
            settings.mqtt.keepalive,
        )
        return client
    except Exception as e:
        logger.critical(
            f"MQTT connection failed during connect() call: {e}", exc_info=True
        )
        return None


def run():
    """Initializes resources and starts the main application loop."""
    logger.info("--- Starting MQTT Subscriber ---")

    try:
        logger.info("Initializing SionnaRT Simulator...")
        simulator = SionnaRTSimulator()
        logger.success("SionnaRT Simulator initialized successfully.")
    except Exception as e:
        logger.critical(
            f"Failed to initialize SionnaRT Simulator: {e}. Exiting.", exc_info=True
        )
        sys.exit(1)

    client = connect_mqtt(simulator)

    if not client:
        logger.critical(
            "Could not establish initial connection to MQTT Broker. Exiting."
        )
        sys.exit(1)

    logger.info("Starting MQTT client loop...")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logger.critical(f"Unhandled exception in MQTT loop: {e}", exc_info=True)
    finally:
        logger.info("Disconnecting MQTT client...")
        client.disconnect()
        logger.info("--- MQTT Subscriber finished. ---")


if __name__ == "__main__":
    run()

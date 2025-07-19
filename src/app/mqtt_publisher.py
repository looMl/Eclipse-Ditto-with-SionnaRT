import json
import random
import time
import sys
from loguru import logger
from paho.mqtt import client as mqtt_client

from app.utils.config import settings

logger.remove()
logger.add(sys.stdout, level=settings.logging.level)


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0 and client.is_connected():
        logger.info(
            f"Successfully connected to MQTT Broker at {settings.mqtt.broker_host}:{settings.mqtt.broker_port}"
        )
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code: {reason_code}")


def on_disconnect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        logger.info("Successfully disconnected from MQTT Broker.")
    else:
        logger.warning(
            f"Unexpectedly disconnected from MQTT Broker with result code: {reason_code}"
        )


def connect_mqtt() -> mqtt_client.Client | None:
    client_id = f"{settings.mqtt.publisher.client_id_prefix}{random.randint(0, 1000)}"
    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    try:
        logger.info(
            f"Attempting to connect to MQTT Broker at {settings.mqtt.broker_host}:{settings.mqtt.broker_port}..."
        )
        client.connect(settings.mqtt.broker_host, settings.mqtt.broker_port)
        return client
    except Exception as e:
        logger.critical(
            f"MQTT connection failed during connect() call: {e}", exc_info=True
        )
        return None


def publish(client: mqtt_client.Client, translation: float) -> bool:
    """Constructs and publishes a fake GPS data message to the MQTT broker."""
    pub_settings = settings.mqtt.publisher
    topic = f"{pub_settings.base_topic}/{pub_settings.thing_id}"

    position = [0.0, round(translation, 2), 0.1]
    orientation = [0.0, 0.0, 0.0]

    gps_data = {
        "position": position,
        "orientation": orientation,
        "thingId": pub_settings.thing_id,
    }
    payload = json.dumps(gps_data, ensure_ascii=False)

    if not client.is_connected():
        logger.error("Publish attempt failed: MQTT client is not connected.")
        return False

    try:
        msg_info = client.publish(
            topic, payload
        )  # publish() returns MqttMessageInfo object
        if msg_info.rc == mqtt_client.MQTT_ERR_SUCCESS:
            logger.info(f"Successfully published `{payload}` to topic `{topic}`")
            return True
        else:
            logger.error(
                f"Failed to publish message to topic `{topic}`. Return code: {msg_info.rc}"
            )
            return False
    except Exception as e:
        logger.error(f"Exception during publish to topic `{topic}`: {e}", exc_info=True)
        return False


def run():
    logger.info("Starting MQTT Publisher...")
    client = connect_mqtt()

    if not client:
        logger.critical(
            "Could not establish initial connection to MQTT Broker. Exiting."
        )
        sys.exit(1)

    client.loop_start()

    logger.info("Waiting for connection to establish...")
    time.sleep(2)
    if not client.is_connected():
        logger.error(
            "Client failed to connect after starting loop. Check broker/network. Exiting."
        )
        client.loop_stop()
        sys.exit(1)

    pub_settings = settings.mqtt.publisher
    translation = float(pub_settings.initial_translation)
    for i in range(pub_settings.num_messages):
        logger.info(f"Publishing message {i + 1}/{pub_settings.num_messages}...")
        success = publish(client, translation)

        if success:
            translation += float(pub_settings.translation_increment)
        else:
            logger.warning(
                "Publish failed for current message. Continuing with next message."
            )

        if i < pub_settings.num_messages - 1:
            logger.info(
                f"Waiting for {pub_settings.publish_interval_seconds} seconds..."
            )
            time.sleep(pub_settings.publish_interval_seconds)

    logger.info("Finished publishing messages.")

    logger.info("Stopping MQTT loop...")
    client.loop_stop()
    logger.info("Disconnecting from MQTT Broker...")
    client.disconnect()
    time.sleep(1)
    logger.info("MQTT Publisher finished.")


if __name__ == "__main__":
    run()

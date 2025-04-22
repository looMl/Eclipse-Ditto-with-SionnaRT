import logging
import json
import random
import time
import sys

from paho.mqtt import client as mqtt_client

# --- Configuration Constants ---
BROKER = "localhost"
PORT = 1883
THING_ID = "org.povo:phone"
BASE_TOPIC = "devices/in"
PUBLISH_TOPIC = f"{BASE_TOPIC}/{THING_ID}"
CLIENT_ID = f'publish-{random.randint(0, 1000)}'
PUBLISH_INTERVAL_SECONDS = 25
NUM_MESSAGES = 5  # Number of messages to send
INITIAL_TRANSLATION = -30
TRANSLATION_INCREMENT = 10

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def on_connect(client, userdata, flags, reason_code, properties=None):
  """Callback function executed when the client connects to the MQTT broker."""
  if reason_code == 0 and client.is_connected():
    logger.info(f"Successfully connected to MQTT Broker at {BROKER}:{PORT}")
  else:
    logger.error(f"Failed to connect to MQTT Broker, return code: {reason_code}")

def on_disconnect(client, userdata, flags, reason_code, properties):
  """Callback function executed when the client disconnects from the MQTT broker."""
  if reason_code == 0:
    logger.info("Successfully disconnected from MQTT Broker.")
  else:
    logger.warning(f"Unexpectedly disconnected from MQTT Broker with result code: {reason_code}")

def connect_mqtt() -> mqtt_client.Client | None:
  """Establishes a connection to the MQTT broker."""
  client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, CLIENT_ID)
  client.on_connect = on_connect
  client.on_disconnect = on_disconnect
  
  try:
    logger.info(f"Attempting to connect to MQTT Broker at {BROKER}:{PORT}...")
    client.connect(BROKER, PORT) 
    return client
  except Exception as e:
    logger.critical(f"MQTT connection failed during connect() call: {e}", exc_info=True)
    return None

def publish(client: mqtt_client.Client, translation: float) -> bool:
  """Constructs and publishes a fake GPS data message to the MQTT broker."""
  # Construct the payload
  position = [0.0, round(translation, 2), 0.1]
  orientation = [0.0, 0.0, 0.0]

  gps_data = {
    "position": position,
    "orientation": orientation,
    "thingId": THING_ID
  }
  payload = json.dumps(gps_data, ensure_ascii=False) 

  # Check connection before publishing
  if not client.is_connected():
    logger.error("Publish attempt failed: MQTT client is not connected.")
    return False

  # Publish the message
  try:
    # publish() returns MqttMessageInfo object
    msg_info = client.publish(PUBLISH_TOPIC, payload) 
    
    if msg_info.rc == mqtt_client.MQTT_ERR_SUCCESS:
      logger.info(f"Successfully published `{payload}` to topic `{PUBLISH_TOPIC}`")
      return True
    else:
      logger.error(f"Failed to publish message to topic `{PUBLISH_TOPIC}`. Return code: {msg_info.rc}")
      return False
  except Exception as e:
    logger.error(f"Exception during publish to topic `{PUBLISH_TOPIC}`: {e}", exc_info=True)
    return False

def run():
  logger.info("Starting MQTT Publisher...")
  client = connect_mqtt()

  if not client:
    logger.critical("Could not establish initial connection to MQTT Broker. Exiting.")
    sys.exit(1)

  client.loop_start()

  logger.info("Waiting for connection to establish...")
  time.sleep(2) 
  if not client.is_connected():
    logger.error("Client failed to connect after starting loop. Check broker/network. Exiting.")
    client.loop_stop()
    sys.exit(1)

  translation = float(INITIAL_TRANSLATION)
  for i in range(NUM_MESSAGES):
    logger.info(f"Publishing message {i+1}/{NUM_MESSAGES}...")
    success = publish(client, translation)
    
    if success:
      translation += float(TRANSLATION_INCREMENT)
    else:
      logger.warning("Publish failed for current message. Continuing with next message.")
    
    if i < NUM_MESSAGES - 1:
      logger.info(f"Waiting for {PUBLISH_INTERVAL_SECONDS} seconds...")
      time.sleep(PUBLISH_INTERVAL_SECONDS)

  logger.info("Finished publishing messages.")

  logger.info("Stopping MQTT loop...")
  client.loop_stop()
  logger.info("Disconnecting from MQTT Broker...")
  client.disconnect() 
  time.sleep(1) 
  logger.info("MQTT Publisher finished.")

if __name__ == '__main__':
  run()
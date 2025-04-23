import logging
import json
import random
import subprocess
import os
import sys

from paho.mqtt import client as mqtt_client

# --- Configuration Constants ---
BROKER = "localhost"
PORT = 1883
BASE_TOPIC_OUT = "devices/out" 
SUBSCRIBE_TOPIC = f"{BASE_TOPIC_OUT}/#" 
CLIENT_ID = f'subscribe-{random.randint(0, 1000)}'
# Gets absolute path to sionnart script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIONNART_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "sionnart.py") 

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def on_connect(client, userdata, flags, reason_code, properties=None):
  """Callback function executed when the client connects to the MQTT broker."""
  if reason_code == 0 and client.is_connected():
    logger.info(f"Successfully connected to MQTT Broker at {BROKER}:{PORT}")
    try:
      client.subscribe(SUBSCRIBE_TOPIC)
      logger.info(f"Subscribed to topic: {SUBSCRIBE_TOPIC}")
    except Exception as e:
      logger.error(f"Failed to subscribe to topic {SUBSCRIBE_TOPIC}: {e}", exc_info=True)
  else:
    logger.error(f"Failed to connect to MQTT Broker, return code: {reason_code}")

def on_disconnect(client, userdata, flags, reason_code, properties):
  """Callback function executed when the client disconnects from the MQTT broker."""
  if reason_code == 0:
    logger.info("Successfully disconnected from MQTT Broker.")
  else:
    logger.warning(f"Unexpectedly disconnected from MQTT Broker with result code: {reason_code}.")

def on_message(client, userdata, message):
  """Callback function executed when a message is received on a subscribed topic."""
  logger.info(f"Received message on topic: {message.topic}")
  
  # 1. Decode Payload
  try:
    payload_str = message.payload.decode('utf-8')
    logger.debug(f"Raw payload string: {payload_str}")
  except UnicodeDecodeError as e:
    logger.error(f"Failed to decode payload (UTF-8) on topic {message.topic}: {e}. Skipping message.", exc_info=True)
    return

  # 2. Parse JSON
  try:
    payload_data = json.loads(payload_str)
  except json.JSONDecodeError as e:
    logger.error(f"Failed to parse JSON payload on topic {message.topic}: {e}. Payload: '{payload_str}'. Skipping message.", exc_info=True)
    return

  # 3. Extract Data (Add checks for key existence)
  # Ditto messages often have a specific structure (e.g., 'value' or 'features')
  # Adjust the extraction logic based on the actual message format from Ditto
  # Assuming the relevant data is directly in the payload for now, as per original code
  # It's safer to use .get() with defaults or check with 'in'
  position = payload_data.get("position")
  orientation = payload_data.get("orientation")

  if position is None or orientation is None:
    logger.warning(f"Missing 'position' or 'orientation' in payload on topic {message.topic}. Payload: {payload_data}. Skipping execution.")
    return

  logger.info(f"Extracted Position: {position}")
  logger.info(f"Extracted Orientation: {orientation}")

  # 4. Execute External Script
  try:
    # Convert lists back to JSON strings for command-line arguments
    position_arg = json.dumps(position)
    orientation_arg = json.dumps(orientation)

    logger.info(f"Executing SionnaRT script: {SIONNART_SCRIPT_PATH} with position={position_arg} orientation={orientation_arg}")
    
    # Using check=True raises CalledProcessError on non-zero exit code
    # Capture output for potential debugging
    result = subprocess.run(["python", SIONNART_SCRIPT_PATH, 
                              "--position", position_arg, 
                              "--orientation", orientation_arg], 
                            check=True) 
    
    logger.info(f"SionnaRT script executed successfully!")
    if result.stdout:
      logger.info(f"SionnaRT stdout:\n{result.stdout}")
    if result.stderr:
      logger.warning(f"SionnaRT stderr:\n{result.stderr}")

  except FileNotFoundError:
    logger.error(f"Error executing script: '{SIONNART_SCRIPT_PATH}' not found. Please check the path.", exc_info=True)
  except subprocess.CalledProcessError as e:
    logger.error(f"Error during SionnaRT script execution (non-zero exit code: {e.returncode}).", exc_info=True)
    logger.error(f"SionnaRT stdout:\n{e.stdout}")
    logger.error(f"SionnaRT stderr:\n{e.stderr}")
  except Exception as e:
    logger.error(f"An unexpected error occurred while running the SionnaRT script: {e}", exc_info=True)

def connect_mqtt() -> mqtt_client.Client | None:
  """Connects to the MQTT broker and sets up callbacks."""
  client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, CLIENT_ID)
  client.on_connect = on_connect
  client.on_disconnect = on_disconnect
  client.on_message = on_message
  
  try:
    logger.info(f"Attempting to connect to MQTT Broker at {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, keepalive=60)
    return client
  except Exception as e:
    logger.critical(f"MQTT connection failed during connect() call: {e}", exc_info=True)
    return None
  
def run():
  logger.info("Starting MQTT Subscriber...")
  client = connect_mqtt()

  if not client:
    logger.critical("Could not establish initial connection to MQTT Broker. Exiting.")
    sys.exit(1)

  logger.info("Starting MQTT loop (blocking)...")
  try:
    client.loop_forever()
  except KeyboardInterrupt:
    logger.info("KeyboardInterrupt received. Shutting down...")
  except Exception as e:
    logger.critical(f"Unhandled exception in MQTT loop: {e}", exc_info=True)
  finally:
    logger.info("Disconnecting MQTT client...")
    client.disconnect()
    logger.info("MQTT Subscriber finished.")

if __name__ == '__main__':
  run()
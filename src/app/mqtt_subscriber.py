import json
import subprocess
import sys
import random
import os
from loguru import logger
from paho.mqtt import client as mqtt_client

from app.utils.config import settings, get_project_root

logger.remove()
logger.add(sys.stdout, level=settings.logging.level)

def on_connect(client, userdata, flags, reason_code, properties=None):
  sub_settings = settings.mqtt.subscriber
  subscribed_topic = f"{sub_settings.base_topic}/#"

  if reason_code == 0 and client.is_connected():
    logger.info(f"Successfully connected to MQTT Broker at {settings.mqtt.broker_host}:{settings.mqtt.broker_port}")
    try:
      client.subscribe(subscribed_topic)
      logger.info(f"Subscribed to topic: {subscribed_topic}")
    except Exception as e:
      logger.error(f"Failed to subscribe to topic {subscribed_topic}: {e}", exc_info=True)
  else:
    logger.error(f"Failed to connect to MQTT Broker, return code: {reason_code}")

def on_disconnect(client, userdata, flags, reason_code, properties):
  if reason_code == 0:
    logger.info("Successfully disconnected from MQTT Broker.")
  else:
    logger.warning(f"Unexpectedly disconnected from MQTT Broker with result code: {reason_code}.")

def on_message(client, userdata, message):
  logger.info(f"Received message on topic: {message.topic}")
  
  try:
    payload_str = message.payload.decode('utf-8')
    payload_data = json.loads(payload_str)
    logger.debug(f"Raw payload string: {payload_str}")
  except (UnicodeDecodeError, json.JSONDecodeError) as e:
    logger.error(f"Failed to decode or parse payload on topic {message.topic}: {e}. Skipping message.", exc_info=True)
    return

  # Ditto messages often have a specific structure (e.g., 'value' or 'features')
  # Adjust the extraction logic based on the actual message format from Ditto
  position = payload_data.get("position")
  orientation = payload_data.get("orientation")

  if position is None or orientation is None:
    logger.warning(f"Missing 'position' or 'orientation' in payload on topic {message.topic}. Payload: {payload_data}. Skipping execution.")
    return

  logger.info(f"Extracted Position: {position}, Orientation: {orientation}")

  execute_sionnart_script(position, orientation)

def execute_sionnart_script(position: list, orientation: list):
  sionnart_script_path = os.path.join(get_project_root(), "app", settings.sionnart.script_name)

  try:
    command = [
      sys.executable, # Uses same python interpreter that is running the subscriber
      sionnart_script_path,
      "--position", json.dumps(position),
      "--orientation", json.dumps(orientation)
    ]
    logger.info(f"Executing SionnaRT with command: {' '.join(command)}")
    
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      encoding='utf-8',
      check=True  # Will raise CalledProcessError for non-zero exit codes
    )
        
    if result.stdout:
      for line in result.stdout.strip().splitlines():
        logger.info(f"[SionnaRT stdout] {line}")
    if result.stderr:
      for line in result.stderr.strip().splitlines():
        logger.warning(f"[SionnaRT stderr] {line}")
    
    logger.info("SionnaRT script executed successfully!")

  except FileNotFoundError:
    logger.error(f"Error executing script: '{sionnart_script_path}' not found.", exc_info=True)
  except subprocess.CalledProcessError as e:
    logger.error(f"Error during SionnaRT script execution (exit code {e.returncode}):\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}", exc_info=True)
  except Exception as e:
    logger.error(f"An unexpected error occurred while running the SionnaRT script: {e}", exc_info=True)

def connect_mqtt() -> mqtt_client.Client | None:
  client_id = f"{settings.mqtt.subscriber.client_id_prefix}{random.randint(0, 1000)}"
  client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id)
  client.on_connect = on_connect
  client.on_disconnect = on_disconnect
  client.on_message = on_message
  
  try:
    logger.info(f"Attempting to connect to MQTT Broker at {settings.mqtt.broker_host}:{settings.mqtt.broker_port}...")
    client.connect(settings.mqtt.broker_host, settings.mqtt.broker_port, settings.mqtt.keepalive)
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
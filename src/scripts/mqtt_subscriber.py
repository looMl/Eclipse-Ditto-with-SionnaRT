from paho.mqtt import client as mqtt_client
import json
import random
import subprocess
import os

# Define MQTT broker settings
BROKER = "localhost"
PORT = 1883
TOPIC = "devices/out/#"
CLIENT_ID = f'subscribe-{random.randint(0, 1000)}'

# If connection to broker works then subscribe to topic
def on_connect(client, userdata, flags, reason_code, properties=None):
  if reason_code == 0 and client.is_connected():
      print("Connected to MQTT Broker!")
      client.subscribe(TOPIC)
  else:
      print(f'Failed to connect, return code {reason_code}')

def on_disconnect(client, userdata, flags, reason_code, properties):
  print("Disconnected with result code:", reason_code)

# Gets the payload from message
def on_message(client, userdata, message):
  payload = json.loads(message.payload.decode())
  position = payload["position"]
  orientation = payload["orientation"]

  print("Position:", position)
  print("Orientation:", orientation)
  print("Topic:", message.topic)


  script_path = "./sionnart.py"
  try:
    subprocess.run(["python", script_path, 
                      "--position", json.dumps(position), 
                      "--orientation", json.dumps(orientation)], check=True)
    print("Sionnart executed successfully!")
  except subprocess.CalledProcessError as e:
    print(f"Error during script's execution: {e}")

# Connects to the mqtt broker
def connect_mqtt():
  client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, CLIENT_ID)
  client.on_connect = on_connect
  client.on_message = on_message
  client.connect(BROKER, PORT)
  client.on_disconnect = on_disconnect
  return client
  
def run():
  client = connect_mqtt()
  client.loop_forever()

if __name__ == '__main__':
  run()
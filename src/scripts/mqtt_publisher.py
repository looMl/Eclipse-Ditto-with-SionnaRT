from paho.mqtt import client as mqtt_client
import json
import random
import time

# Define MQTT broker settings
BROKER = "localhost"
PORT = 1883
TOPIC = "devices/in"
# Generate a Client ID with the publish prefix.
CLIENT_ID = f'publish-{random.randint(0, 1000)}'


def on_connect(client, userdata, flags, reason_code, properties=None):
  if reason_code == 0 and client.is_connected():
      print("Connected to MQTT Broker!")
  else:
      print("Failed to connect, return code %d\n", reason_code)

def on_disconnect(client, userdata, flags, reason_code, properties):
  print("Disconnected with result code:", reason_code)

def connect_mqtt():
  client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, CLIENT_ID)
  client.on_connect = on_connect
  client.on_disconnect = on_disconnect
  client.connect(BROKER, PORT)
  return client

# Data and publish
def publish(client, translation):
  #position = [round(random.uniform(-15, 0), 2), round(random.uniform(-42, 0), 2), 0.1]
  position = [0, translation, 0.1]
  orientation = [0.0, 0.0, 0.0]
  thingId = "org.povo:phone"

  gps_data = {
    "position": position,
    "orientation": orientation,
    "thingId": thingId
  }
  payload = json.dumps(gps_data, ensure_ascii=False)

  if not client.is_connected():
    print("publish: MQTT client is not connected!")
    time.sleep(1)
    exit

  result = client.publish(f"{TOPIC}/{thingId}", payload) 
  # result: [0, 1]
  status = result[0]
  if status == 0:
    print(f"Sent `{payload}` to topic `{TOPIC}/{thingId}`")
  else:
    print(f"Failed to send message to topic {TOPIC}/{thingId}")

def run():
  client = connect_mqtt()
  client.loop_start()
  translation = -30

  for i in range(5):
    publish(client, translation)
    translation += 10
    time.sleep(25)

  client.loop_stop()
  client.disconnect()

if __name__ == '__main__':
  run()
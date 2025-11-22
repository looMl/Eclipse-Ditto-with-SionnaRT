import json
import sys
import time
import multiprocessing
import queue
from loguru import logger
from paho.mqtt import client as mqtt_client

from app.utils.config import settings as cfg
from app.SionnaRTSimulator import SionnaRTSimulator

logger.remove()
logger.add(sys.stdout, level=cfg.logging.level)


def run_simulation_process(task_queue: multiprocessing.Queue):
    logger.info("Worker: Initializing SionnaRT Engine...")

    try:
        simulator = SionnaRTSimulator()
        logger.success("Worker: SionnaRT ready. Waiting for coordinates...")
    except Exception as e:
        logger.critical(f"Worker: Failed to initialize SionnaRT: {e}")
        return

    while True:
        try:
            # Block until a new task is available
            task = task_queue.get()

            if task is None:
                logger.info("Worker: Received stop signal.")
                break

            position, orientation = task

            logger.info(f"Worker: Starting simulation for Pos={position}")
            start_time = time.time()

            simulator.run_simulation(position, orientation)

            duration = time.time() - start_time
            logger.info(f"Worker: Simulation completed in {duration:.2f}s")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Worker: Error during simulation step: {e}")
            continue


def on_message(client, userdata, message):
    """Parses the message and puts it into the queue if the worker is free."""
    task_queue = userdata

    try:
        payload = json.loads(message.payload.decode())

        pos = payload.get("position")
        ori = payload.get("orientation")

        if pos and ori:
            try:
                # If the queue is full (Sionna is busy), this raises queue.Full as Drop Oldest strategy
                task_queue.put_nowait((pos, ori))
                logger.debug("Main: Task queued successfully.")
            except queue.Full:
                logger.warning(
                    "Main: Simulation busy! Dropping incoming frame to maintain real-time."
                )
        else:
            logger.warning(f"Main: Received invalid payload: {payload}")

    except json.JSONDecodeError:
        logger.error("Main: Failed to decode JSON payload.")
    except Exception as e:
        logger.error(f"Main: Unexpected error in MQTT callback: {e}")


def run():
    """Sets up the worker process and the MQTT listener."""
    # maxsize=1 ensures we only process the latest available data
    task_queue = multiprocessing.Queue(maxsize=1)

    worker = multiprocessing.Process(
        target=run_simulation_process, args=(task_queue,), name="SionnaWorker"
    )
    worker.start()

    work_settings = cfg.mqtt.worker
    client_id = f"{work_settings.client_id_prefix}bridge"

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id)

    # Pass the queue to the callback as userdata
    client.user_data_set(task_queue)
    client.on_message = on_message

    try:
        logger.info(f"Main: Connecting to MQTT Broker at {cfg.mqtt.broker_host}...")
        client.connect(cfg.mqtt.broker_host, cfg.mqtt.broker_port, cfg.mqtt.keepalive)

        topic = f"{work_settings.base_topic}/#"
        client.subscribe(topic)
        logger.info(f"Main: Subscribed to {topic}")

        client.loop_forever()

    except KeyboardInterrupt:
        logger.info("Main: Stopping application...")
    except Exception as e:
        logger.critical(f"Main: MQTT Error: {e}")
    finally:
        client.disconnect()

        # Send stop signal to worker
        task_queue.put(None)
        worker.join(timeout=5)

        if worker.is_alive():
            logger.warning("Main: Worker did not stop gracefully, forcing termination.")
            worker.terminate()

        logger.info("System Shutdown Complete.")


if __name__ == "__main__":
    run()

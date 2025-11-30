import json
import sys
import time
import multiprocessing
import queue
from loguru import logger

from app.utils.config import settings as cfg
from app.SionnaRTSimulator import SionnaRTSimulator
from app.core.mqtt import MQTTClientWrapper

logger.remove()
logger.add(sys.stdout, level=cfg.logging.level)


def run_simulation_process(task_queue: multiprocessing.Queue):
    """
    Worker process function that runs the simulation.
    """
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


class BridgeService:
    def __init__(self, settings):
        self.settings = settings
        self.task_queue = multiprocessing.Queue(maxsize=1)
        self.worker_process = None
        self.mqtt_client = None

    def _on_message(self, client, userdata, message):
        """Parses the message and puts it into the queue if the worker is free."""
        try:
            payload = json.loads(message.payload.decode())

            pos = payload.get("position")
            ori = payload.get("orientation")

            if pos and ori:
                try:
                    # If the queue is full (Sionna is busy), this raises queue.Full as Drop Oldest strategy
                    self.task_queue.put_nowait((pos, ori))
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

    def start(self):
        # Start Worker
        self.worker_process = multiprocessing.Process(
            target=run_simulation_process, args=(self.task_queue,), name="SionnaWorker"
        )
        self.worker_process.start()

        # Start MQTT
        work_settings = self.settings.mqtt.worker
        self.mqtt_client = MQTTClientWrapper(
            self.settings.mqtt.broker_host,
            self.settings.mqtt.broker_port,
            self.settings.mqtt.keepalive,
            work_settings.client_id_prefix + "bridge",
        )

        self.mqtt_client.set_on_message(self._on_message)

        try:
            self.mqtt_client.connect()
            topic = f"{work_settings.base_topic}/#"
            self.mqtt_client.subscribe(topic)

            logger.info("Main: Bridge Service Started. Press Ctrl+C to stop.")
            self.mqtt_client.loop_forever()

        except KeyboardInterrupt:
            logger.info("Main: Stopping application...")
        except Exception as e:
            logger.critical(f"Main: MQTT Error: {e}")
        finally:
            self.stop()

    def stop(self):
        if self.mqtt_client:
            self.mqtt_client.stop()

        # Send stop signal to worker
        if self.worker_process and self.worker_process.is_alive():
            self.task_queue.put(None)
            self.worker_process.join(timeout=5)

            if self.worker_process.is_alive():
                logger.warning(
                    "Main: Worker did not stop gracefully, forcing termination."
                )
                self.worker_process.terminate()

        logger.info("System Shutdown Complete.")


if __name__ == "__main__":
    service = BridgeService(cfg)
    service.start()

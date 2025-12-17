import json
import requests
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from requests.auth import HTTPBasicAuth


class DittoManager:
    """
    Manages the lifecycle of Eclipse Ditto Things for the simulation.
    """

    DEFAULT_API_URL = "http://localhost:8080/api/2"
    DEFAULT_USERNAME = "ditto"
    DEFAULT_PASSWORD = "ditto"
    DEFAULT_NAMESPACE = "com.sionna"
    DEFAULT_POLICY_ID = "com.sionna:policy"

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
    ):
        self.base_url = api_url
        self.auth = HTTPBasicAuth(username, password)
        self.headers = {"Content-Type": "application/json"}

    def provision_simulation(
        self, transmitters_json_path: Path, namespace: str = DEFAULT_NAMESPACE
    ):
        """
        1. Delete all existing things in the namespace.
        2. Create new things from the transmitters JSON file.
        """
        if not transmitters_json_path.exists():
            logger.error(
                f"DittoManager: JSON file not found at {transmitters_json_path}"
            )
            return

        logger.info("DittoManager: Starting provisioning process...")

        self.create_policy()
        self.delete_namespace_things(namespace)

        transmitters = self._load_transmitters(transmitters_json_path)
        self._create_things(transmitters)

        logger.success("DittoManager: Provisioning complete.")

    def delete_namespace_things(self, namespace: str) -> None:
        """Deletes all things in the specified namespace."""
        logger.info(f"DittoManager: Cleaning up namespace '{namespace}*'...")

        # Search for all things in the namespace
        search_url = f"{self.base_url}/search/things"
        filter_query = f'like(thingId,"{namespace}:*")'
        params = {"filter": filter_query, "fields": "thingId"}

        try:
            response = requests.get(
                search_url, auth=self.auth, params=params, timeout=10
            )
            response.raise_for_status()

            things = response.json().get("items", [])

            for thing in things:
                thing_id = thing.get("thingId")
                self._delete_thing(thing_id)

        except requests.RequestException as e:
            logger.error(f"DittoManager: Failed to search/delete things: {e}")

    def _delete_thing(self, thing_id: str) -> None:
        """Deletes a single thing by ID."""
        try:
            url = f"{self.base_url}/things/{thing_id}"
            response = requests.delete(url, auth=self.auth, timeout=5)
            if response.status_code in [200, 204]:
                logger.debug(f"DittoManager: Deleted {thing_id}")
            else:
                logger.warning(
                    f"DittoManager: Failed to delete {thing_id}: {response.text}"
                )
        except requests.RequestException as e:
            logger.error(f"DittoManager: Error deleting {thing_id}: {e}")

    def _load_transmitters(self, path: Path) -> List[Dict[str, Any]]:
        """Parses the transmitters JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"DittoManager: Failed to load JSON: {e}")
            return []

    def _create_things(self, items: List[Dict[str, Any]]) -> None:
        """Iterates through the list and creates Things in Ditto."""
        logger.info("DittoManager: Creating new Things...")
        success_count = 0

        for item in items:
            thing_id = item.get("thingId")
            if not thing_id:
                logger.warning("DittoManager: Skipping item without thingId")
                continue

            if self._create_single_thing(thing_id, item):
                success_count += 1

        logger.info(
            f"DittoManager: Successfully created {success_count}/{len(items)} things."
        )

    def _create_single_thing(self, thing_id: str, payload: Dict[str, Any]) -> bool:
        """Creates or updates a single Thing."""
        url = f"{self.base_url}/things/{thing_id}"

        if "policyId" not in payload:
            payload["policyId"] = self.DEFAULT_POLICY_ID

        try:
            response = requests.put(
                url,
                auth=self.auth,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=5,
            )

            if response.status_code in [201, 204]:
                logger.debug(f"DittoManager: Created {thing_id}")
                return True
            else:
                logger.error(
                    f"DittoManager: Failed to create {thing_id}. Status: {response.status_code}, Body: {response.text}"
                )
                return False

        except requests.RequestException as e:
            logger.error(f"DittoManager: Connection error creating {thing_id}: {e}")
            return False

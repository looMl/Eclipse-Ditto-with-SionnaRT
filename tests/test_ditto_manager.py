"""Unit tests for DittoManager's transmitter-file loading."""

import json

from app.services.ditto_manager import DittoManager


def test_load_transmitters_reads_list(tmp_path):
    path = tmp_path / "tx.json"
    payload = [{"thingId": "a"}, {"thingId": "b"}]
    path.write_text(json.dumps(payload))
    assert DittoManager()._load_transmitters(path) == payload


def test_load_transmitters_missing_file_returns_empty(tmp_path):
    assert DittoManager()._load_transmitters(tmp_path / "nope.json") == []


def test_load_transmitters_invalid_json_returns_empty(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    assert DittoManager()._load_transmitters(path) == []

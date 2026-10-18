import json
import logging
import sqlite3
import time

import pytest

from app.devices.registry import parse_registry
from app.sensors.ingest import SensorIngest, parse_topic
from app.storage.event_store import EventDatabase

REGISTRY = parse_registry(
    {
        "site": "home",
        "devices": [
            {"id": "cam1", "type": "camera", "host": "10.0.0.2"},
            {"id": "radar-1", "type": "radar", "location": "living-room", "zone": "indoor"},
            {"id": "tw-1", "type": "tripwire", "location": "front-wall"},
            {"id": "tw-2", "type": "tripwire"},
        ],
    }
)


@pytest.fixture
def ingest(tmp_path):
    db = EventDatabase(tmp_path / "events.db")
    yield SensorIngest(REGISTRY, db, tmp_path / "events", logging.getLogger("test"))
    db.close()


def test_parse_topic():
    assert parse_topic("hes/home/radar-1/event", "hes").device_id == "radar-1"
    assert parse_topic("hes/home/radar-1/cmd", "hes") is None
    assert parse_topic("other/home/radar-1/event", "hes") is None
    assert parse_topic("hes/home/radar-1", "hes") is None


def test_tripwire_event_is_stored_with_state(ingest, tmp_path):
    now = time.time()
    record = ingest.handle_message(
        "hes/home/tw-1/event", json.dumps({"event": "beam_broken", "ts": now, "rssi": -60})
    )
    assert record is not None
    assert record["device_type"] == "tripwire"
    assert record["data"]["trigger"] is True
    assert record["data"]["location"] == "front-wall"

    events = ingest.event_db.query_events(device_types=["tripwire"])
    assert [event["event_type"] for event in events] == ["beam_broken"]
    state = ingest.event_db.device_states()["tw-1"]
    assert state["state"]["beam"] == "broken"
    assert state["telemetry"]["rssi"] == -60
    assert state["status"] == "online"
    assert (tmp_path / "events" / "tw-1.jsonl").exists()

    ingest.handle_message("hes/home/tw-1/event", '{"event": "beam_restored", "duration_ms": 300}')
    state = ingest.event_db.device_states()["tw-1"]
    assert state["state"] == {"beam": "ok", "duration_ms": 300}
    assert state["last_event_type"] == "beam_restored"


def test_radar_presence_and_status(ingest):
    ingest.handle_message(
        "hes/home/radar-1/event", '{"event": "presence_start", "distance_cm": 210, "energy": 80}'
    )
    state = ingest.event_db.device_states()["radar-1"]["state"]
    assert state == {"presence": True, "distance_cm": 210, "energy": 80}

    ingest.handle_message("hes/home/radar-1/status", "offline")
    assert ingest.event_db.device_states()["radar-1"]["status"] == "offline"
    ingest.handle_message("hes/home/radar-1/telemetry", '{"rssi": -55, "fw": "1.0"}')
    record = ingest.event_db.device_states()["radar-1"]
    assert record["status"] == "online"
    assert record["telemetry"]["fw"] == "1.0"


def test_many_devices_of_one_type_stay_separate(ingest):
    ingest.handle_message("hes/home/tw-1/event", '{"event": "beam_broken"}')
    ingest.handle_message("hes/home/tw-2/event", '{"event": "fault"}')
    states = ingest.event_db.device_states()
    assert states["tw-1"]["state"]["beam"] == "broken"
    assert states["tw-2"]["state"]["beam"] == "fault"
    assert len(ingest.event_db.query_events(device_ids=["tw-2"])) == 1


def test_unregistered_and_foreign_messages_keep_no_events(ingest):
    assert ingest.handle_message("hes/home/mystery/event", '{"event": "x", "type": "door"}') is None
    assert ingest.handle_message("hes/other-site/tw-1/event", '{"event": "beam_broken"}') is None
    assert ingest.handle_message("hes/home/tw-1/event", '{"no_event": 1}') is None
    assert ingest.handle_message("hes/home/tw-1/event", "not json") is None
    assert ingest.event_db.count() == 0
    mystery = ingest.event_db.device_states()["mystery"]
    assert mystery["registered"] is False
    assert mystery["device_type"] == "door"


def test_device_clock_skew_falls_back_to_hub_time(ingest):
    record = ingest.handle_message("hes/home/tw-1/event", '{"event": "beam_broken", "ts": 12345}')
    assert record is not None
    assert record["event_at_utc"].startswith(time.strftime("%Y", time.gmtime()))


def test_legacy_camera_database_is_migrated(tmp_path):
    path = tmp_path / "events.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            camera_id TEXT NOT NULL,
            camera_host TEXT,
            event_at_utc TEXT NOT NULL,
            event_at_epoch REAL NOT NULL,
            frame_index INTEGER,
            detections_count INTEGER NOT NULL DEFAULT 0,
            counts_json TEXT NOT NULL DEFAULT '{}',
            detections_json TEXT NOT NULL DEFAULT '[]',
            snapshot_path TEXT
        );
        INSERT INTO events (event_id, camera_id, event_at_utc, event_at_epoch, counts_json)
        VALUES ('cam1-1', 'cam1', '2026-10-01T00:00:00+00:00', 1790000000, '{"cat": 1}');
        """
    )
    conn.commit()
    conn.close()

    db = EventDatabase(path)
    events = db.query_events(device_ids=["cam1"])
    assert len(events) == 1
    assert events[0]["device_type"] == "camera"
    assert events[0]["event_type"] == "detection"
    assert events[0]["counts"] == {"cat": 1}
    db.close()
    EventDatabase(path).close()  # second open must not re-migrate or fail

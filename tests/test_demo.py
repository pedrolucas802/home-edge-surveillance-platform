import json
import logging
import random
from pathlib import Path

from app.demo.world import SiteWorld
from app.devices.registry import load_registry
from app.sensors import rd03d
from app.sensors.ingest import SensorIngest, topic_for
from app.storage.event_store import EventDatabase

DEMO = Path(__file__).resolve().parents[1] / "configs" / "devices.demo.yaml"


def test_rd03d_frame_round_trip():
    targets = [
        rd03d.RadarTarget(x_mm=-782, y_mm=1713, speed_cms=-16),
        rd03d.RadarTarget(x_mm=1200, y_mm=4000, speed_cms=35, resolution_mm=320),
    ]
    frame = rd03d.encode_frame(targets)
    assert len(frame) == rd03d.FRAME_LENGTH
    assert frame[:4] == bytes.fromhex("AAFF0300") and frame[-2:] == bytes.fromhex("55CC")
    assert rd03d.decode_frame(frame) == targets
    assert rd03d.decode_frame(rd03d.encode_frame([])) == []


def test_rd03d_decodes_datasheet_style_bytes():
    # x = -782 mm (sign bit clear), y = 1713 mm, speed = -16 cm/s, resolution 360 mm
    frame = bytes.fromhex("AAFF03000E03B18610006801" + "00" * 16 + "55CC")
    (target,) = rd03d.decode_frame(frame)
    assert (target.x_mm, target.y_mm, target.speed_cms, target.resolution_mm) == (
        -782,
        1713,
        -16,
        360,
    )
    assert target.distance_mm == 1883
    assert target.angle_deg < 0


def test_coverage():
    assert rd03d.in_coverage(0, 3000)
    assert not rd03d.in_coverage(0, 9000)
    assert not rd03d.in_coverage(5000, 1000)  # outside the 120 degree cone


def test_one_visit_drives_the_zone_devices(tmp_path):
    registry = load_registry(DEMO)
    world = SiteWorld(registry, random.Random(3), mean_gap_seconds=1e9)
    world._next_visit_at = float("inf")  # only the forced visit below
    db = EventDatabase(tmp_path / "events.db")
    ingest = SensorIngest(registry, db, None, logging.getLogger("test"))

    def send(outputs):
        for output in outputs:
            body = output.body if isinstance(output.body, str) else json.dumps(output.body)
            ingest.handle_message(
                topic_for("hes", registry.site_id, output.device.id, output.channel), body
            )

    send(world.boot())
    t0 = world.started_at + 10
    visit = world.force_visit("front", t0)
    seen_on_camera = False
    now = t0
    while now < t0 + visit.duration + 1:
        send(world.tick(now))
        camera = registry.get("cam-front")
        seen_on_camera |= bool(world.camera_people(camera, now))
        assert not world.camera_people(registry.get("cam-back"), now)
        now += 0.1

    kinds = [(e["device_id"], e["event_type"]) for e in db.query_events(limit=100)]
    assert ("tripwire-front-wall", "beam_broken") in kinds
    assert ("radar-porch", "presence_start") in kinds
    assert ("radar-porch", "presence_end") in kinds
    assert all(device_id in {"tripwire-front-wall", "radar-porch"} for device_id, _ in kinds)
    assert seen_on_camera
    states = db.device_states()
    assert states["radar-porch"]["state"]["presence"] is False
    assert states["tripwire-front-wall"]["status"] == "online"

"""Pretend to be the site's ESP32 sensors until the real hardware exists.

Every radar and tripwire in the devices file gets a virtual board that speaks
the same MQTT contract the firmware will: retained online/offline status with
a Last Will, periodic telemetry, and events.

    python -m app.sensors.simulator              # publish to the MQTT broker
    python -m app.sensors.simulator --direct     # no broker, write straight to the store
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from threading import Event

from app.config.settings import Settings
from app.devices.registry import DeviceSpec
from app.devices.types import RADAR, TRIPWIRE
from app.sensors.ingest import SensorIngest, topic_for
from app.storage.dashboard_store import resolve_dashboard_dir
from app.storage.event_store import EventDatabase


@dataclass(slots=True)
class VirtualDevice:
    spec: DeviceSpec
    rng: random.Random
    started_at: float = field(default_factory=time.monotonic)
    next_action_at: float = 0.0
    active: bool = False
    pending: list[tuple[float, dict[str, object]]] = field(default_factory=list)

    def uptime_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)

    def telemetry(self) -> dict[str, object]:
        return {
            "rssi": self.rng.randint(-78, -48),
            "uptime_ms": self.uptime_ms(),
            "fw": "sim-0.1",
            "heap_free": self.rng.randint(150_000, 200_000),
        }

    def due_events(self, now: float, mean_gap_seconds: float) -> list[dict[str, object]]:
        """Return the events this device emits at time 'now'."""
        ready = [event for at, event in self.pending if at <= now]
        self.pending = [(at, event) for at, event in self.pending if at > now]
        if now < self.next_action_at or self.pending:
            return ready
        self.next_action_at = now + self.rng.expovariate(1.0 / mean_gap_seconds)
        if self.spec.type == RADAR:
            ready.extend(self._radar_cycle(now))
        elif self.spec.type == TRIPWIRE:
            ready.extend(self._tripwire_cycle(now))
        return ready

    def _radar_cycle(self, now: float) -> list[dict[str, object]]:
        distance = self.rng.randint(80, 600)
        start = {
            "event": "presence_start",
            "distance_cm": distance,
            "energy": self.rng.randint(35, 100),
            "moving": self.rng.random() < 0.7,
        }
        end_at = now + self.rng.uniform(3.0, 15.0)
        self.pending.append((end_at, {"event": "presence_end", "distance_cm": distance}))
        return [start]

    def _tripwire_cycle(self, now: float) -> list[dict[str, object]]:
        if self.rng.random() < 0.03:
            return [{"event": "fault", "reason": "beam blocked > 30 s"}]
        duration_ms = self.rng.randint(80, 1500)
        restore_at = now + duration_ms / 1000.0
        self.pending.append((restore_at, {"event": "beam_restored", "duration_ms": duration_ms}))
        return [{"event": "beam_broken"}]


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate the site's radar and tripwire boards.")
    parser.add_argument("--direct", action="store_true", help="Skip MQTT and write to the store.")
    parser.add_argument("--mqtt-host", default=defaults.mqtt_host)
    parser.add_argument("--mqtt-port", type=int, default=defaults.mqtt_port)
    parser.add_argument(
        "--mean-gap",
        type=float,
        default=20.0,
        help="Average seconds between activity bursts per device.",
    )
    parser.add_argument("--telemetry-interval", type=float, default=30.0)
    parser.add_argument(
        "--duration", type=float, default=0.0, help="Stop after N seconds (0 = forever)."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help="Only simulate these device ids (default: every radar and tripwire).",
    )
    return parser


class _MqttSink:
    def __init__(self, settings: Settings, args: argparse.Namespace, site_id: str) -> None:
        import paho.mqtt.client as mqtt

        self._mqtt = mqtt
        self.settings = settings
        self.args = args
        self.site_id = site_id
        self.clients: dict[str, object] = {}

    def connect(self, device: DeviceSpec) -> None:
        mqtt = self._mqtt
        prefix = self.settings.mqtt_topic_prefix
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{device.id}")
        if self.settings.mqtt_username:
            client.username_pw_set(self.settings.mqtt_username, self.settings.mqtt_password or None)
        status_topic = topic_for(prefix, self.site_id, device.id, "status")
        client.will_set(status_topic, "offline", qos=1, retain=True)
        client.connect(self.args.mqtt_host, self.args.mqtt_port, keepalive=30)
        client.loop_start()
        client.publish(status_topic, "online", qos=1, retain=True)
        self.clients[device.id] = client

    def publish(self, device: DeviceSpec, channel: str, body: object, retain: bool = False) -> None:
        topic = topic_for(self.settings.mqtt_topic_prefix, self.site_id, device.id, channel)
        payload = body if isinstance(body, str) else json.dumps(body)
        self.clients[device.id].publish(topic, payload, qos=1, retain=retain)

    def close(self) -> None:
        for device_id, client in self.clients.items():
            topic = topic_for(self.settings.mqtt_topic_prefix, self.site_id, device_id, "status")
            client.publish(topic, "offline", qos=1, retain=True).wait_for_publish(2.0)
            client.loop_stop()
            client.disconnect()


class _DirectSink:
    def __init__(self, settings: Settings, ingest: SensorIngest, site_id: str) -> None:
        self.settings = settings
        self.ingest = ingest
        self.site_id = site_id

    def connect(self, device: DeviceSpec) -> None:
        self.publish(device, "status", "online")

    def publish(self, device: DeviceSpec, channel: str, body: object, retain: bool = False) -> None:
        topic = topic_for(self.settings.mqtt_topic_prefix, self.site_id, device.id, channel)
        payload = body if isinstance(body, str) else json.dumps(body)
        self.ingest.handle_message(topic, payload)

    def close(self) -> None:
        for device in self.ingest.registry.enabled():
            if device.device_type.transport == "mqtt":
                self.publish(device, "status", "offline")


def main() -> int:
    settings = Settings.from_env()
    args = build_parser(settings).parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("sensor_simulator")

    registry = settings.load_registry()
    devices = [
        device
        for device in registry.enabled()
        if device.type in {RADAR, TRIPWIRE} and (not args.devices or device.id in args.devices)
    ]
    if not devices:
        logger.error(
            "No radar or tripwire devices in %s. Copy configs/devices.example.yaml to "
            "configs/devices.yaml and add some.",
            registry.source,
        )
        return 2

    rng = random.Random(args.seed)
    event_db: EventDatabase | None = None
    if args.direct:
        base_dir = resolve_dashboard_dir(settings.dashboard_dir)
        event_db = EventDatabase(base_dir / "events.db", logger=logger)
        ingest = SensorIngest(
            registry, event_db, base_dir / "events", logger, settings.mqtt_topic_prefix
        )
        sink: _MqttSink | _DirectSink = _DirectSink(settings, ingest, registry.site_id)
    else:
        sink = _MqttSink(settings, args, registry.site_id)

    virtual = [VirtualDevice(spec=device, rng=random.Random(rng.random())) for device in devices]
    for device in virtual:
        sink.connect(device.spec)
        # Stagger the first burst so devices do not all fire at once.
        device.next_action_at = time.monotonic() + device.rng.uniform(1.0, args.mean_gap)
    logger.info(
        "Simulating %s device(s) %s: %s",
        len(virtual),
        "directly into the store" if args.direct else f"on {args.mqtt_host}:{args.mqtt_port}",
        ", ".join(device.spec.id for device in virtual),
    )

    stop_event = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())

    started = time.monotonic()
    last_telemetry = 0.0
    try:
        while not stop_event.wait(0.1):
            now = time.monotonic()
            if args.duration and now - started >= args.duration:
                break
            if now - last_telemetry >= args.telemetry_interval:
                last_telemetry = now
                for device in virtual:
                    sink.publish(device.spec, "telemetry", device.telemetry())
            for device in virtual:
                for event in device.due_events(now, args.mean_gap):
                    event["ts"] = time.time()
                    event["rssi"] = device.telemetry()["rssi"]
                    event["uptime_ms"] = device.uptime_ms()
                    sink.publish(device.spec, "event", event)
                    logger.info("%s %s", device.spec.id, event["event"])
    finally:
        sink.close()
        if event_db is not None:
            event_db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

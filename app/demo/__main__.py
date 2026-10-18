"""Run the hub with mock hardware: cameras, RD-03D radars and tripwires.

    python -m app.demo                      # everything, written straight to the store
    python -m app.demo --transport mqtt     # sensors go through Mosquitto + the bridge
    python -m app.demo --no-cameras         # sensors only (what app.sensors.simulator runs)

Start the dashboard alongside it (scripts/run_demo.sh does both).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import time
from threading import Event

from app.config.settings import Settings
from app.demo.world import Output, SiteWorld
from app.devices.registry import DeviceSpec
from app.sensors.ingest import SensorIngest, topic_for
from app.storage.dashboard_store import resolve_dashboard_dir
from app.storage.event_store import EventDatabase

TICK_SECONDS = 0.1


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hub with simulated devices.")
    parser.add_argument(
        "--transport",
        choices=["direct", "mqtt"],
        default="direct",
        help="direct writes sensor messages straight to the store; mqtt publishes them.",
    )
    parser.add_argument("--no-cameras", action="store_true", help="Simulate sensors only.")
    parser.add_argument("--mqtt-host", default=defaults.mqtt_host)
    parser.add_argument("--mqtt-port", type=int, default=defaults.mqtt_port)
    parser.add_argument(
        "--mean-gap",
        type=float,
        default=12.0,
        help="Average seconds between people showing up somewhere on the site.",
    )
    parser.add_argument("--camera-fps", type=float, default=4.0)
    parser.add_argument("--telemetry-interval", type=float, default=30.0)
    parser.add_argument(
        "--duration", type=float, default=0.0, help="Stop after N seconds (0 = forever)."
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser


class MqttSink:
    """One MQTT client per virtual board, each with its own Last Will."""

    def __init__(self, settings: Settings, host: str, port: int, site_id: str) -> None:
        import paho.mqtt.client as mqtt

        self._mqtt = mqtt
        self.settings = settings
        self.host = host
        self.port = port
        self.site_id = site_id
        self.clients: dict[str, object] = {}

    def _client(self, device: DeviceSpec):
        client = self.clients.get(device.id)
        if client is not None:
            return client
        mqtt = self._mqtt
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{device.id}")
        if self.settings.mqtt_username:
            client.username_pw_set(self.settings.mqtt_username, self.settings.mqtt_password or None)
        status = topic_for(self.settings.mqtt_topic_prefix, self.site_id, device.id, "status")
        client.will_set(status, "offline", qos=1, retain=True)
        client.connect(self.host, self.port, keepalive=30)
        client.loop_start()
        self.clients[device.id] = client
        return client

    def send(self, output: Output) -> None:
        topic = topic_for(
            self.settings.mqtt_topic_prefix, self.site_id, output.device.id, output.channel
        )
        payload = output.body if isinstance(output.body, str) else json.dumps(output.body)
        retain = output.channel in {"status", "state"}
        self._client(output.device).publish(topic, payload, qos=1, retain=retain)

    def close(self) -> None:
        for client in self.clients.values():
            client.loop_stop()
            client.disconnect()


class DirectSink:
    """Skip the broker and hand messages to the same ingest code the bridge uses."""

    def __init__(self, settings: Settings, ingest: SensorIngest, site_id: str) -> None:
        self.settings = settings
        self.ingest = ingest
        self.site_id = site_id

    def send(self, output: Output) -> None:
        topic = topic_for(
            self.settings.mqtt_topic_prefix, self.site_id, output.device.id, output.channel
        )
        payload = output.body if isinstance(output.body, str) else json.dumps(output.body)
        self.ingest.handle_message(topic, payload)

    def close(self) -> None:
        return None


def run(
    argv: list[str] | None = None,
    *,
    default_transport: str = "direct",
    default_cameras: bool = True,
) -> int:
    settings = Settings.from_env()
    parser = build_parser(settings)
    parser.set_defaults(transport=default_transport, no_cameras=not default_cameras)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("demo")

    registry = settings.load_registry()
    world = SiteWorld(
        registry,
        random.Random(args.seed),
        mean_gap_seconds=args.mean_gap,
        telemetry_interval=args.telemetry_interval,
    )
    if not world.sensors and (args.no_cameras or not world.cameras):
        logger.error(
            "Nothing to simulate in %s. Use configs/devices.demo.yaml or add radars and "
            "tripwires to configs/devices.yaml.",
            registry.source,
        )
        return 2

    base_dir = resolve_dashboard_dir(settings.dashboard_dir)
    event_db: EventDatabase | None = None
    if args.transport == "direct":
        event_db = EventDatabase(base_dir / "events.db", logger=logger)
        ingest = SensorIngest(
            registry, event_db, base_dir / "events", logger, settings.mqtt_topic_prefix
        )
        sink: MqttSink | DirectSink = DirectSink(settings, ingest, registry.site_id)
    else:
        sink = MqttSink(settings, args.mqtt_host, args.mqtt_port, registry.site_id)

    cameras = []
    if not args.no_cameras:
        from app.demo.camera import MockCamera

        cameras = [
            MockCamera(device, settings, logger, fps=args.camera_fps) for device in world.cameras
        ]

    logger.info(
        "Demo site '%s': %s camera(s), %s sensor(s) via %s. Data: %s",
        registry.site_id,
        len(cameras),
        len(world.sensors),
        args.transport,
        base_dir,
    )

    stop_event = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())

    started = time.monotonic()
    for output in world.boot():
        sink.send(output)
    try:
        while not stop_event.wait(TICK_SECONDS):
            now = time.monotonic()
            if args.duration and now - started >= args.duration:
                break
            for output in world.tick(now):
                sink.send(output)
                if output.channel == "event":
                    logger.info("%s %s", output.device.id, output.body.get("event"))
            for camera in cameras:
                camera.tick(now, world.camera_people(camera.device, now))
    finally:
        for output in world.shutdown():
            sink.send(output)
        time.sleep(0.3)
        sink.close()
        if event_db is not None:
            event_db.close()
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())

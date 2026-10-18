"""Subscribe to the site's MQTT topics and store sensor events.

Run with: python -m app.sensors.bridge
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from threading import Event

import paho.mqtt.client as mqtt

from app.config.settings import Settings
from app.sensors.ingest import SensorIngest
from app.storage.dashboard_store import resolve_dashboard_dir
from app.storage.event_store import EventDatabase, run_retention


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge site MQTT sensors into the event store.")
    parser.add_argument("--mqtt-host", default=defaults.mqtt_host)
    parser.add_argument("--mqtt-port", type=int, default=defaults.mqtt_port)
    parser.add_argument("--dashboard-dir", default=defaults.dashboard_dir)
    return parser


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger("sensor_bridge")


def main() -> int:
    settings = Settings.from_env()
    args = build_parser(settings).parse_args()
    logger = configure_logging()

    try:
        registry = settings.load_registry()
    except ValueError as exc:
        logger.error("Devices config error: %s", exc)
        return 2

    base_dir = resolve_dashboard_dir(args.dashboard_dir)
    event_db = EventDatabase(base_dir / "events.db", logger=logger)
    ingest = SensorIngest(
        registry=registry,
        event_db=event_db,
        events_dir=base_dir / "events",
        logger=logger,
        topic_prefix=settings.mqtt_topic_prefix,
    )
    for device in registry.enabled():
        if device.device_type.transport == "mqtt":
            event_db.upsert_device_state(device.id, device.type, seen_at_epoch=0.0)

    subscription = f"{settings.mqtt_topic_prefix}/{registry.site_id}/+/+"
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"hub-bridge-{registry.site_id}",
        clean_session=False,
    )
    if settings.mqtt_username:
        client.username_pw_set(settings.mqtt_username, settings.mqtt_password or None)

    def on_connect(client: mqtt.Client, _userdata, _flags, reason_code, _properties) -> None:
        if reason_code.is_failure:
            logger.error("MQTT connect failed: %s", reason_code)
            return
        client.subscribe(subscription, qos=1)
        logger.info(
            "Connected to %s:%s, subscribed to %s", args.mqtt_host, args.mqtt_port, subscription
        )

    def on_message(_client: mqtt.Client, _userdata, message: mqtt.MQTTMessage) -> None:
        try:
            record = ingest.handle_message(message.topic, message.payload)
        except Exception:  # noqa: BLE001 - one bad message must not stop the bridge
            logger.exception("Failed to handle message on %s", message.topic)
            return
        if record is not None:
            logger.info("%s %s", record["device_id"], record["event_type"])

    def on_disconnect(_client, _userdata, _flags, reason_code, _properties) -> None:
        logger.warning("MQTT disconnected (%s); reconnecting.", reason_code)

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    stop_event = Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received %s, stopping.", signal.Signals(signum).name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    logger.info(
        "Sensor bridge for site '%s' with %s device(s): %s",
        registry.site_id,
        len(registry.enabled()),
        ", ".join(f"{key}={len(items)}" for key, items in registry.by_type().items()) or "none",
    )
    client.connect_async(args.mqtt_host, args.mqtt_port, keepalive=30)
    client.loop_start()
    last_retention_at = 0.0
    try:
        while not stop_event.wait(1.0):
            now = time.monotonic()
            if now - last_retention_at >= settings.event_retention_check_interval_seconds:
                last_retention_at = now
                run_retention(event_db, base_dir, settings.event_retention_days, logger)
    finally:
        client.loop_stop()
        client.disconnect()
        event_db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

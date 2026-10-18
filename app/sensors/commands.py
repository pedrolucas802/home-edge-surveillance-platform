"""Send commands from the hub to the ESP32 boards over MQTT (the `cmd` channel)."""

from __future__ import annotations

import json
import time

from app.config.settings import Settings
from app.sensors.ingest import topic_for


def send_command(
    settings: Settings,
    site_id: str,
    device_ids: list[str],
    payload: dict[str, object],
    timeout_seconds: float = 4.0,
) -> int:
    """Publish one command to each device; return how many were delivered to the broker.

    Raises OSError when the broker cannot be reached.
    """
    import paho.mqtt.client as mqtt

    if not device_ids:
        return 0
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"hub-cmd-{site_id}-{int(time.time())}")
    if settings.mqtt_username:
        client.username_pw_set(settings.mqtt_username, settings.mqtt_password or None)
    client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=10)
    client.loop_start()
    try:
        body = json.dumps({**payload, "ts": time.time()})
        infos = [
            client.publish(topic_for(settings.mqtt_topic_prefix, site_id, device_id, "cmd"), body, qos=1)
            for device_id in device_ids
        ]
        deadline = time.monotonic() + timeout_seconds
        for info in infos:
            info.wait_for_publish(max(deadline - time.monotonic(), 0.1))
        return sum(1 for info in infos if info.is_published())
    finally:
        client.loop_stop()
        client.disconnect()

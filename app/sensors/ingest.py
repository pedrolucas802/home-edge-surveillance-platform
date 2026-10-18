from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.devices.registry import DeviceRegistry, DeviceSpec
from app.devices.types import get_device_type
from app.storage.event_store import EventDatabase

CHANNELS = frozenset({"event", "state", "status", "telemetry"})
# Payload keys that describe the device rather than the event itself.
TELEMETRY_KEYS = frozenset(
    {"rssi", "uptime_ms", "fw", "firmware", "heap_free", "battery_v", "voltage"}
)
_ENVELOPE_KEYS = frozenset({"device", "event", "ts", "type"})
# Device clocks come from NTP and can be wrong right after boot, so a
# timestamp further than this from the hub's clock is replaced by the hub's.
_MAX_CLOCK_SKEW_SECONDS = 300.0


@dataclass(slots=True, frozen=True)
class Topic:
    site_id: str
    device_id: str
    channel: str


def parse_topic(topic: str, prefix: str) -> Topic | None:
    """Split 'hes/<site>/<device>/<channel>' into its parts."""
    parts = topic.split("/")
    if len(parts) != 4 or parts[0] != prefix:
        return None
    _, site_id, device_id, channel = parts
    if not site_id or not device_id or channel not in CHANNELS:
        return None
    return Topic(site_id=site_id, device_id=device_id, channel=channel)


def topic_for(prefix: str, site_id: str, device_id: str, channel: str) -> str:
    return f"{prefix}/{site_id}/{device_id}/{channel}"


class SensorIngest:
    """Turns MQTT messages from site devices into events and device state.

    Kept free of any MQTT client so the bridge, the simulator's direct mode
    and the tests all share one code path.
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        event_db: EventDatabase,
        events_dir: Path | None,
        logger: logging.Logger,
        topic_prefix: str = "hes",
    ) -> None:
        self.registry = registry
        self.event_db = event_db
        self.events_dir = events_dir
        self.logger = logger.getChild("sensor_ingest")
        self.topic_prefix = topic_prefix
        self._warned_unregistered: set[str] = set()

    def handle_message(self, topic: str, payload: bytes | str) -> dict[str, object] | None:
        """Process one message. Returns the stored event record, if any."""
        parsed = parse_topic(topic, self.topic_prefix)
        if parsed is None or parsed.site_id != self.registry.site_id:
            return None

        body = _decode_payload(payload)
        device = self.registry.get(parsed.device_id)
        now = time.time()
        if device is None or not device.enabled:
            self._record_unregistered(parsed, body, now)
            return None

        if parsed.channel == "status":
            self.event_db.upsert_device_state(
                device.id, device.type, status=_parse_status(body), seen_at_epoch=now
            )
            return None
        if parsed.channel == "telemetry":
            self.event_db.upsert_device_state(
                device.id,
                device.type,
                status="online",
                telemetry=body if isinstance(body, dict) else {},
                seen_at_epoch=now,
            )
            return None
        if parsed.channel == "state":
            self.event_db.upsert_device_state(
                device.id,
                device.type,
                status="online",
                state_patch=body if isinstance(body, dict) else {"value": body},
                seen_at_epoch=now,
            )
            return None
        return self._handle_event(device, body, now)

    def _handle_event(
        self,
        device: DeviceSpec,
        body: object,
        now: float,
    ) -> dict[str, object] | None:
        if not isinstance(body, dict) or not str(body.get("event", "")).strip():
            self.logger.warning(
                "Ignoring event from %s without an 'event' field: %r", device.id, body
            )
            return None

        event_type = str(body["event"]).strip().lower()
        event_epoch = _parse_timestamp(body.get("ts"), now)
        data = {key: value for key, value in body.items() if key not in _ENVELOPE_KEYS}
        record = build_sensor_event(device, event_type, event_epoch, data)

        self.event_db.insert_event(record)
        if self.events_dir is not None:
            _append_jsonl(self.events_dir / f"{device.id}.jsonl", record)

        device_type = device.device_type
        state_patch = dict(device_type.event_state.get(event_type, {}))
        state_patch.update(
            {key: value for key, value in data.items() if key in device_type.state_fields}
        )
        self.event_db.upsert_device_state(
            device.id,
            device.type,
            status="online",
            state_patch=state_patch,
            telemetry={key: value for key, value in data.items() if key in TELEMETRY_KEYS},
            last_event_type=event_type,
            last_event_at_epoch=event_epoch,
            seen_at_epoch=now,
        )
        return record

    def _record_unregistered(self, parsed: Topic, body: object, now: float) -> None:
        if parsed.device_id not in self._warned_unregistered:
            self._warned_unregistered.add(parsed.device_id)
            self.logger.warning(
                "Message from unregistered device '%s'. Add it to the devices file.",
                parsed.device_id,
            )
        reported_type = body.get("type") if isinstance(body, dict) else None
        self.event_db.upsert_device_state(
            parsed.device_id,
            str(reported_type or "unknown"),
            status=_parse_status(body) if parsed.channel == "status" else "online",
            seen_at_epoch=now,
            registered=False,
        )


def build_sensor_event(
    device: DeviceSpec,
    event_type: str,
    event_epoch: float,
    data: dict[str, object],
) -> dict[str, object]:
    device_type = get_device_type(device.type)
    return {
        "event_id": f"{device.id}-{event_type}-{int(event_epoch * 1000)}",
        "device_id": device.id,
        "device_type": device.type,
        "event_type": event_type,
        "camera_id": device.id,
        "event_at_utc": datetime.fromtimestamp(event_epoch, timezone.utc).isoformat(),
        "detections_count": 0,
        "counts": {},
        "detections": [],
        "data": {
            **data,
            "trigger": event_type in device_type.trigger_events,
            "location": device.location,
            "zone": device.zone,
        },
    }


def _decode_payload(payload: bytes | str) -> object:
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_status(body: object) -> str:
    raw = body.get("status") if isinstance(body, dict) else body
    value = str(raw or "").strip().lower()
    return "offline" if value in {"offline", "0", "false", "lost"} else "online"


def _parse_timestamp(raw: object, now: float) -> float:
    epoch: float | None = None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        epoch = float(raw) / 1000.0 if raw > 1e12 else float(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = datetime.fromisoformat(raw.strip())
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            epoch = parsed.timestamp()
        except ValueError:
            epoch = None
    if epoch is None or abs(epoch - now) > _MAX_CLOCK_SKEW_SECONDS:
        return now
    return epoch


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

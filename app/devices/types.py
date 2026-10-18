from __future__ import annotations

from dataclasses import dataclass, field

CAMERA = "camera"
RADAR = "radar"
TRIPWIRE = "tripwire"


@dataclass(slots=True, frozen=True)
class DeviceType:
    """Describes one kind of device the hub knows how to talk to.

    Adding a new kind of device (a door reed switch, a siren, a smoke sensor)
    means registering one more DeviceType. The registry, MQTT bridge, event
    store and dashboard all work from these descriptions instead of
    hard-coding device kinds.
    """

    key: str
    label: str
    plural: str
    transport: str
    icon: str
    # Event name -> short description shown in docs and the dashboard.
    events: dict[str, str] = field(default_factory=dict)
    # Events that mean "something may be happening" (inputs for alarm fusion).
    trigger_events: frozenset[str] = frozenset()
    # Event name -> patch applied to the device's last known state.
    event_state: dict[str, dict[str, object]] = field(default_factory=dict)
    # State keys worth showing on a device card, in display order.
    state_fields: tuple[str, ...] = ()


_DEVICE_TYPES: dict[str, DeviceType] = {}


def register_device_type(device_type: DeviceType) -> DeviceType:
    _DEVICE_TYPES[device_type.key] = device_type
    return device_type


def get_device_type(key: str) -> DeviceType:
    """Return the registered type, or a generic MQTT type for unknown keys.

    Unknown types still load, ingest and show on the dashboard, so a new
    sensor can be wired up before it gets a dedicated description here.
    """
    normalized = key.strip().lower()
    known = _DEVICE_TYPES.get(normalized)
    if known is not None:
        return known
    return DeviceType(
        key=normalized,
        label=normalized.replace("_", " ").title(),
        plural=f"{normalized.replace('_', ' ').title()}s",
        transport="mqtt",
        icon="🔌",
    )


def registered_device_types() -> list[DeviceType]:
    return list(_DEVICE_TYPES.values())


register_device_type(
    DeviceType(
        key=CAMERA,
        label="Camera",
        plural="Cameras",
        transport="rtsp",
        icon="📷",
        events={"detection": "YOLO detected one or more objects"},
        trigger_events=frozenset({"detection"}),
    )
)

register_device_type(
    DeviceType(
        key=RADAR,
        label="Motion radar",
        plural="Motion radars",
        transport="mqtt",
        icon="📡",
        # Modelled on the Ai-Thinker RD-03D: up to 3 tracked people, each
        # with x/y position and speed (see app/sensors/rd03d.py).
        events={
            "presence_start": "A person entered the radar's range",
            "presence_end": "The radar's range is empty again",
            "targets": "Positions of up to 3 tracked people",
            "fault": "The radar stopped reporting valid frames",
        },
        trigger_events=frozenset({"presence_start"}),
        event_state={
            "presence_start": {"presence": True},
            "presence_end": {"presence": False, "target_count": 0, "nearest_m": None, "targets": []},
            "fault": {"fault": True},
        },
        state_fields=("presence", "target_count", "nearest_m", "fault", "targets"),
    )
)

register_device_type(
    DeviceType(
        key=TRIPWIRE,
        label="Tripwire",
        plural="Tripwires",
        transport="mqtt",
        icon="🚧",
        events={
            "beam_broken": "Something crossed the beam",
            "beam_restored": "The beam is clear again",
            "fault": "Beam blocked too long or misaligned",
            "tamper": "Enclosure opened",
        },
        trigger_events=frozenset({"beam_broken", "tamper"}),
        event_state={
            "beam_broken": {"beam": "broken"},
            "beam_restored": {"beam": "ok"},
            "fault": {"beam": "fault"},
            "tamper": {"tamper": True},
        },
        state_fields=("beam", "duration_ms", "signal", "tamper"),
    )
)

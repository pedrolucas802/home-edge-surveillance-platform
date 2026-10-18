"""Site preferences edited from the dashboard's Configuration page.

Lives next to the site's data (``<DASHBOARD_DIR>/hub_config.yaml``, git-ignored)
so every client site keeps its own copy. Device definitions stay in the
devices file; this holds the owner, alert switches, the Wi-Fi the boards
should join and the default detection model.
"""

from __future__ import annotations

import copy
import os
from datetime import datetime, time as dt_time
from pathlib import Path

import yaml

HUB_CONFIG_FILENAME = "hub_config.yaml"

# Alert kinds the owner can switch on and off, in display order.
ALERT_KINDS: dict[str, str] = {
    "camera_detection": "Camera sees a person",
    "radar_presence": "Radar detects someone in range",
    "tripwire_beam": "Tripwire beam is crossed",
    "tamper": "A device enclosure is opened",
    "fault": "A device reports a fault",
    "device_offline": "A device goes offline",
}

DEFAULTS: dict[str, object] = {
    "user": {"name": "", "email": "", "phone": ""},
    "alerts": {
        "enabled": True,
        "kinds": {kind: True for kind in ALERT_KINDS},
        "muted_devices": [],
        "quiet_hours": {"enabled": False, "start": "23:00", "end": "06:00"},
        "channels": {"dashboard": True, "email": False, "push": False},
    },
    "wifi": {"ssid": "", "password": "", "country": "BR", "hostname_prefix": "hes"},
    "yolo": {
        "enabled": False,
        "model": "",
        "device": "mps",
        "detect": True,
        "pose": False,
        "track": False,
        "tracker": "bytetrack.yaml",
        "classes": "person",
        "confidence": 0.35,
        "infer_every_n": 1,
    },
    "display": {"refresh_seconds": 2},
}


def resolve_hub_config_path(dashboard_dir: Path) -> Path:
    raw = os.getenv("HUB_CONFIG", "").strip()
    return Path(raw).expanduser() if raw else dashboard_dir / HUB_CONFIG_FILENAME


def load_hub_config(path: Path) -> dict[str, object]:
    """Return the saved preferences merged over the defaults."""
    config = copy.deepcopy(DEFAULTS)
    if not path.exists():
        return config
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return config
    if isinstance(raw, dict):
        _merge(config, raw)
    return config


def save_hub_config(path: Path, config: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        "# Written by the dashboard's Configuration page.\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if os.name == "posix":
        # Holds the Wi-Fi password.
        tmp_path.chmod(0o600)
    tmp_path.replace(path)


def alert_kind_for(device_type: str, event_type: str) -> str | None:
    """Map a stored event to the alert kind that covers it, if any."""
    if event_type == "tamper":
        return "tamper"
    if event_type == "fault":
        return "fault"
    if device_type == "camera" and event_type == "detection":
        return "camera_detection"
    if device_type == "radar" and event_type == "presence_start":
        return "radar_presence"
    if device_type == "tripwire" and event_type == "beam_broken":
        return "tripwire_beam"
    return None


def is_alert(event: dict[str, object], config: dict[str, object], now: datetime | None = None) -> bool:
    """True when the owner wants to be alerted about this event."""
    alerts = config.get("alerts", {})
    if not isinstance(alerts, dict) or not alerts.get("enabled", True):
        return False
    kind = alert_kind_for(str(event.get("device_type", "camera")), str(event.get("event_type", "")))
    if kind is None or not alerts.get("kinds", {}).get(kind, True):
        return False
    if str(event.get("device_id", "")) in (alerts.get("muted_devices") or []):
        return False
    return not in_quiet_hours(alerts.get("quiet_hours", {}), now)


def in_quiet_hours(quiet_hours: object, now: datetime | None = None) -> bool:
    if not isinstance(quiet_hours, dict) or not quiet_hours.get("enabled"):
        return False
    start = _parse_clock(quiet_hours.get("start"))
    end = _parse_clock(quiet_hours.get("end"))
    if start is None or end is None or start == end:
        return False
    current = (now or datetime.now()).time()
    if start < end:
        return start <= current < end
    return current >= start or current < end


def _parse_clock(value: object) -> dt_time | None:
    try:
        hours, minutes = str(value).split(":", 1)
        return dt_time(int(hours), int(minutes))
    except (ValueError, TypeError):
        return None


def _merge(base: dict[str, object], override: dict[str, object]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value

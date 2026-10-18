from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.devices.types import CAMERA, DeviceType, get_device_type

DEFAULT_DEVICES_CONFIG = "configs/devices.yaml"
DEFAULT_SITE_ID = "home"
_DEVICE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_RESERVED_DEVICE_IDS = frozenset({"system"})
_CORE_KEYS = frozenset({"id", "type", "name", "location", "zone", "enabled"})


@dataclass(slots=True, frozen=True)
class DeviceSpec:
    """One physical device installed at a client site."""

    id: str
    type: str
    name: str
    location: str = ""
    zone: str = ""
    enabled: bool = True
    # Type-specific settings, e.g. host for a camera or model for a radar.
    options: dict[str, object] = field(default_factory=dict)

    @property
    def device_type(self) -> DeviceType:
        return get_device_type(self.type)

    def option(self, key: str, default: object = None) -> object:
        return self.options.get(key, default)


@dataclass(slots=True, frozen=True)
class DeviceRegistry:
    """All devices for one site, loaded from the per-client devices file."""

    site_id: str
    devices: tuple[DeviceSpec, ...]
    source: str = ""

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for device in self.devices:
            if device.id in seen:
                raise ValueError(
                    f"Duplicate device id '{device.id}' in {self.source or 'registry'}."
                )
            seen.add(device.id)

    def get(self, device_id: str) -> DeviceSpec | None:
        for device in self.devices:
            if device.id == device_id:
                return device
        return None

    def enabled(self) -> list[DeviceSpec]:
        return [device for device in self.devices if device.enabled]

    def of_type(self, type_key: str) -> list[DeviceSpec]:
        return [device for device in self.enabled() if device.type == type_key]

    def by_type(self) -> dict[str, list[DeviceSpec]]:
        grouped: dict[str, list[DeviceSpec]] = {}
        for device in self.enabled():
            grouped.setdefault(device.type, []).append(device)
        return grouped

    def type_keys(self) -> list[str]:
        return list(self.by_type())

    def camera_hosts(self) -> dict[str, str]:
        return {
            device.id: str(device.option("host"))
            for device in self.of_type(CAMERA)
            if device.option("host")
        }


def resolve_devices_config_path(raw_path: str | None = None) -> Path:
    path = Path(raw_path or DEFAULT_DEVICES_CONFIG).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def load_registry(
    path: Path | str | None = None,
    *,
    fallback_camera_hosts: dict[str, str] | None = None,
    fallback_site_id: str = DEFAULT_SITE_ID,
) -> DeviceRegistry:
    """Load the site's device registry.

    Without a devices file the registry falls back to the cameras configured
    through CAMERA_HOSTS, so existing camera-only installs keep working.
    """
    config_path = resolve_devices_config_path(str(path) if path else None)
    if not config_path.exists():
        cameras = tuple(
            DeviceSpec(id=camera_id, type=CAMERA, name=camera_id, options={"host": host})
            for camera_id, host in (fallback_camera_hosts or {}).items()
        )
        return DeviceRegistry(site_id=fallback_site_id, devices=cameras, source="environment")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return parse_registry(raw, source=str(config_path), fallback_site_id=fallback_site_id)


def parse_registry(
    raw: object,
    *,
    source: str = "",
    fallback_site_id: str = DEFAULT_SITE_ID,
) -> DeviceRegistry:
    if not isinstance(raw, dict):
        raise ValueError(f"Devices config {source} must be a mapping with 'site' and 'devices'.")

    site_id = str(raw.get("site") or fallback_site_id).strip()
    if not _DEVICE_ID_PATTERN.match(site_id):
        raise ValueError(f"Site id '{site_id}' must be lowercase letters, digits, '-' or '_'.")

    raw_devices = raw.get("devices") or []
    if not isinstance(raw_devices, list):
        raise ValueError(f"'devices' in {source} must be a list.")

    devices = tuple(_parse_device(entry, index, source) for index, entry in enumerate(raw_devices))
    return DeviceRegistry(site_id=site_id, devices=devices, source=source)


def _parse_device(entry: object, index: int, source: str) -> DeviceSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"Device #{index + 1} in {source} must be a mapping.")

    device_id = str(entry.get("id", "")).strip()
    type_key = str(entry.get("type", "")).strip().lower()
    if not device_id or not _DEVICE_ID_PATTERN.match(device_id):
        raise ValueError(
            f"Device #{index + 1} in {source} needs a lowercase 'id' (letters, digits, '-', '_')."
        )
    if device_id in _RESERVED_DEVICE_IDS:
        raise ValueError(f"Device id '{device_id}' is reserved.")
    if not type_key:
        raise ValueError(f"Device '{device_id}' in {source} needs a 'type'.")

    options = {key: value for key, value in entry.items() if key not in _CORE_KEYS}
    return DeviceSpec(
        id=device_id,
        type=type_key,
        name=str(entry.get("name") or device_id),
        location=str(entry.get("location") or ""),
        zone=str(entry.get("zone") or ""),
        enabled=bool(entry.get("enabled", True)),
        options=options,
    )

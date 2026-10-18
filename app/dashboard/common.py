"""Data loading, formatting and shared widgets used by every dashboard page."""

from __future__ import annotations

import html
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image

from app.config.hub_config import is_alert, load_hub_config, resolve_hub_config_path
from app.config.settings import Settings
from app.devices.registry import DeviceRegistry, DeviceSpec
from app.devices.types import CAMERA, get_device_type
from app.storage.dashboard_store import resolve_dashboard_dir
from app.storage.event_store import EventDatabase


@dataclass(slots=True)
class HubContext:
    settings: Settings
    dashboard_dir: Path
    registry: DeviceRegistry
    hub_config: dict[str, object]
    hub_config_path: Path

    @property
    def refresh_seconds(self) -> float:
        display = self.hub_config.get("display", {})
        try:
            return max(float(display.get("refresh_seconds", 2)), 1.0)  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return 2.0

    @property
    def user_name(self) -> str:
        user = self.hub_config.get("user", {})
        return str(user.get("name", "")).strip() if isinstance(user, dict) else ""

    def cameras(self) -> list[DeviceSpec]:
        return self.registry.of_type(CAMERA)

    def camera_ids(self) -> list[str]:
        return [device.id for device in self.cameras()] or (
            list(self.settings.camera_hosts_map) or [self.settings.camera_id]
        )

    def devices_of(self, type_key: str) -> list[DeviceSpec]:
        return self.registry.of_type(type_key)

    def camera_host(self, camera_id: str) -> str:
        device = self.registry.get(camera_id)
        if device is not None and device.option("host"):
            return str(device.option("host"))
        try:
            return self.settings.resolve_camera_host(camera_id)
        except ValueError:
            return "n/a"


def get_context() -> HubContext:
    settings = Settings.from_env()
    dashboard_dir = resolve_dashboard_dir(settings.dashboard_dir)
    hub_config_path = resolve_hub_config_path(dashboard_dir)
    return HubContext(
        settings=settings,
        dashboard_dir=dashboard_dir,
        registry=settings.load_registry(),
        hub_config=load_hub_config(hub_config_path),
        hub_config_path=hub_config_path,
    )


# Data -------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _get_event_database(dashboard_dir: str) -> EventDatabase:
    return EventDatabase(Path(dashboard_dir) / "events.db")


def event_db(ctx: HubContext) -> EventDatabase | None:
    if not (ctx.dashboard_dir / "events.db").exists():
        return None
    return _get_event_database(str(ctx.dashboard_dir))


def load_status(ctx: HubContext, camera_id: str) -> dict[str, object] | None:
    path = ctx.dashboard_dir / "live" / f"{camera_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_statuses(ctx: HubContext, camera_ids: list[str] | None = None) -> dict[str, dict[str, object] | None]:
    return {camera_id: load_status(ctx, camera_id) for camera_id in (camera_ids or ctx.camera_ids())}


def load_device_states(ctx: HubContext) -> dict[str, dict[str, object]]:
    db = event_db(ctx)
    return db.device_states() if db else {}


def recent_events(
    ctx: HubContext,
    *,
    device_ids: list[str] | None = None,
    device_types: list[str] | None = None,
    since_epoch: float | None = None,
    limit: int = 50,
) -> list[dict[str, object]]:
    db = event_db(ctx)
    if db is None:
        return []
    return db.query_events(
        device_ids=device_ids,
        device_types=device_types,
        start_epoch=since_epoch,
        limit=limit,
    )


def recent_alerts(ctx: HubContext, *, since_epoch: float | None = None, limit: int = 200) -> list[dict[str, object]]:
    return [event for event in recent_events(ctx, since_epoch=since_epoch, limit=limit) if is_alert(event, ctx.hub_config)]


def load_display_image(path: Path, zoom_factor: float = 1.0) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        return None
    if zoom_factor <= 1.0:
        return image
    width, height = image.size
    crop_width = max(int(width / zoom_factor), 1)
    crop_height = max(int(height / zoom_factor), 1)
    left = max((width - crop_width) // 2, 0)
    top = max((height - crop_height) // 2, 0)
    return image.crop((left, top, left + crop_width, top + crop_height)).resize((width, height))


# Presence ---------------------------------------------------------------


def status_age_seconds(status: dict[str, object] | None) -> float | None:
    if not status:
        return None
    timestamp = status.get("updated_at_utc")
    if not isinstance(timestamp, str):
        return None
    try:
        updated_at = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - updated_at).total_seconds(), 0.0)


def camera_is_live(status: dict[str, object] | None, settings: Settings) -> bool:
    age = status_age_seconds(status)
    return age is not None and age <= settings.dashboard_publish_interval_seconds * 3


def sensor_presence(state_record: dict[str, object] | None, settings: Settings) -> str:
    if not state_record:
        return "never seen"
    last_seen = state_record.get("last_seen_at_epoch")
    if not isinstance(last_seen, (int, float)) or last_seen <= 0:
        return "never seen"
    if state_record.get("status") == "offline":
        return "offline"
    if time.time() - float(last_seen) > settings.sensor_offline_after_seconds:
        return "offline"
    return "online"


def device_presence(
    device: DeviceSpec,
    statuses: dict[str, dict[str, object] | None],
    device_states: dict[str, dict[str, object]],
    settings: Settings,
) -> str:
    if not device.enabled:
        return "disabled"
    if device.type == CAMERA:
        status = statuses.get(device.id)
        if status is None:
            return "never seen"
        return "online" if camera_is_live(status, settings) else "offline"
    return sensor_presence(device_states.get(device.id), settings)


def device_last_seen(
    device: DeviceSpec,
    statuses: dict[str, dict[str, object] | None],
    state_record: dict[str, object] | None,
) -> float | None:
    if device.type == CAMERA:
        age = status_age_seconds(statuses.get(device.id))
        return time.time() - age if age is not None else None
    last_seen = (state_record or {}).get("last_seen_at_epoch")
    return float(last_seen) if isinstance(last_seen, (int, float)) and last_seen > 0 else None


def sensor_alert_text(state: dict[str, object]) -> str:
    if state.get("fault") is True or state.get("beam") == "fault":
        return "Fault reported"
    if state.get("tamper") is True:
        return "Enclosure opened"
    if state.get("beam") == "broken":
        return "Beam broken"
    if state.get("presence") is True:
        count = state.get("target_count")
        if isinstance(count, int) and count > 0:
            return f"{count} {'person' if count == 1 else 'people'} in range"
        return "Presence detected"
    return ""


# Formatting -------------------------------------------------------------

PRESENCE_BADGE = {
    "online": ("badge-live", "Online"),
    "offline": ("badge-offline", "Offline"),
    "never seen": ("badge-stale", "Never seen"),
    "disabled": ("badge-muted", "Disabled"),
}
PRESENCE_ICON = {"online": "🟢", "offline": "🔴", "never seen": "⚪", "disabled": "⚫"}

_FIELD_LABELS = {
    "nearest_m": "Nearest (m)",
    "target_count": "People tracked",
    "duration_ms": "Last break (ms)",
    "signal": "Receiver level",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def badge(presence: str) -> str:
    css, text = PRESENCE_BADGE.get(presence, ("badge-stale", presence.title()))
    return f"<span class='status-badge {css}'>{text}</span>"


def humanize(key: str) -> str:
    if key in _FIELD_LABELS:
        return _FIELD_LABELS[key]
    return key.replace("_cm", " (cm)").replace("_ms", " (ms)").replace("_", " ").capitalize()


def format_state_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def format_age(epoch: object) -> str:
    if not isinstance(epoch, (int, float)) or epoch <= 0:
        return "never"
    seconds = max(time.time() - float(epoch), 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def format_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "n/a"
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return value
    return timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def format_time_short(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "n/a"
    try:
        timestamp = datetime.fromisoformat(value).astimezone()
    except ValueError:
        return value
    if timestamp.date() == datetime.now().astimezone().date():
        return timestamp.strftime("%H:%M:%S")
    return timestamp.strftime("%d %b %H:%M")


def format_detection_label(detection: object) -> str:
    if not isinstance(detection, dict):
        return str(detection)
    label = str(detection.get("class_name", "object"))
    confidence = detection.get("confidence")
    track_id = detection.get("track_id")
    if track_id is not None:
        label = f"{label} #{track_id}"
    if isinstance(confidence, (float, int)):
        label = f"{label} {confidence:.2f}"
    return label


def counts_pills(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return "<span class='pill pill-muted'>nothing detected</span>"
    return "".join(
        f"<span class='pill'>{esc(name)}: {esc(count)}</span>" for name, count in counts.items()
    )


def describe_event(event: dict[str, object], registry: DeviceRegistry) -> str:
    device_id = str(event.get("device_id") or event.get("camera_id") or "")
    device = registry.get(device_id)
    device_type = get_device_type(str(event.get("device_type", CAMERA)))
    name = device.name if device else device_id
    event_type = str(event.get("event_type", "event"))
    if device_type.key == CAMERA:
        counts = event.get("counts") or {}
        what = ", ".join(f"{count} {label}" for label, count in counts.items()) if isinstance(counts, dict) else ""
        return f"{device_type.icon} {name}: {what or 'detection'}"
    label = device_type.events.get(event_type, event_type.replace("_", " "))
    return f"{device_type.icon} {name}: {label}"


def page_header(title: str, subtitle: str, icon: str) -> None:
    st.markdown(
        f"<div class='page-header'><div class='page-icon'>{icon}</div>"
        f"<div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div></div>",
        unsafe_allow_html=True,
    )


def stat_tiles(tiles: list[tuple[str, object, str]]) -> None:
    """Row of (label, value, hint) tiles."""
    cells = "".join(
        f"<div class='stat-tile'><div class='stat-label'>{esc(label)}</div>"
        f"<div class='stat-value'>{esc(value)}</div>"
        f"<div class='stat-hint'>{esc(hint)}</div></div>"
        for label, value, hint in tiles
    )
    st.markdown(f"<div class='stat-row'>{cells}</div>", unsafe_allow_html=True)


def card_header(title: str, subtitle: str, presence: str) -> str:
    return (
        f"<div class='hub-card-header'><div><div class='hub-card-title'>{esc(title)}</div>"
        f"<div class='hub-card-subtitle'>{esc(subtitle)}</div></div>{badge(presence)}</div>"
    )


def card(title: str, subtitle: str, presence: str, body_html: str = "", extra_class: str = "") -> str:
    return f"<div class='hub-card {extra_class}'>{card_header(title, subtitle, presence)}{body_html}</div>"


def rows_html(rows: list[tuple[str, object]]) -> str:
    return "".join(
        f"<div class='kv-row'><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"
        for label, value in rows
    )


def radar_map_svg(targets: object, max_range_m: float = 8.0, height: int = 150) -> str:
    """Top-down view of an RD-03D's coverage (120 degrees, 8 m) with its targets."""
    width, range_mm = int(height * 1.75), 8000.0
    origin_x, origin_y = width / 2, height - 6
    scale = (height - 14) / range_mm
    half_fov = math.radians(60)

    def arc(radius_mm: float) -> tuple[float, float, float]:
        r = radius_mm * scale
        return math.sin(half_fov) * r, math.cos(half_fov) * r, r

    edge_x, edge_y, radius = arc(range_mm)
    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' class='radar-map' role='img' "
        "aria-label='Radar coverage map'>",
        (
            f"<path d='M{origin_x},{origin_y} L{origin_x - edge_x:.1f},{origin_y - edge_y:.1f} "
            f"A{radius:.1f},{radius:.1f} 0 0,1 {origin_x + edge_x:.1f},{origin_y - edge_y:.1f} Z' "
            "class='radar-cone'/>"
        ),
    ]
    for ring_m in (2, 4, 6):
        rx, ry, r = arc(ring_m * 1000)
        parts.append(
            f"<path d='M{origin_x - rx:.1f},{origin_y - ry:.1f} A{r:.1f},{r:.1f} 0 0,1 "
            f"{origin_x + rx:.1f},{origin_y - ry:.1f}' class='radar-ring'/>"
        )
        parts.append(
            f"<text x='{origin_x + 3}' y='{origin_y - r + 10:.1f}' class='radar-text'>{ring_m} m</text>"
        )
    if 0 < max_range_m < 8:
        rx, ry, r = arc(max_range_m * 1000)
        parts.append(
            f"<path d='M{origin_x - rx:.1f},{origin_y - ry:.1f} A{r:.1f},{r:.1f} 0 0,1 "
            f"{origin_x + rx:.1f},{origin_y - ry:.1f}' class='radar-limit'/>"
        )
    parts.append(f"<circle cx='{origin_x}' cy='{origin_y}' r='3.5' class='radar-origin'/>")
    for target in targets if isinstance(targets, list) else []:
        if not isinstance(target, dict):
            continue
        x = origin_x + float(target.get("x_mm", 0)) * scale
        y = origin_y - float(target.get("y_mm", 0)) * scale
        moving = abs(float(target.get("speed_cms", 0))) > 15
        css = "radar-target-moving" if moving else "radar-target-still"
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5.5' class='{css}'/>")
    parts.append("</svg>")
    return "".join(parts)


def beam_svg(beam: str) -> str:
    """Two posts and the beam between them, coloured by the beam state."""
    css = {"broken": "beam-broken", "fault": "beam-fault"}.get(beam, "beam-ok")
    dash = " stroke-dasharray='6 6'" if beam in {"broken", "fault"} else ""
    gap = (
        "<line x1='120' y1='30' x2='160' y2='30' class='beam-gap'/>" if beam == "broken" else ""
    )
    return (
        "<svg viewBox='0 0 280 60' width='100%' class='beam-svg' role='img' aria-label='Beam state'>"
        "<rect x='8' y='10' width='14' height='40' rx='3' class='beam-post'/>"
        "<rect x='258' y='10' width='14' height='40' rx='3' class='beam-post'/>"
        f"<line x1='22' y1='30' x2='258' y2='30' class='{css}'{dash}/>{gap}</svg>"
    )


def signal_bar(level: object, threshold: int = 500) -> str:
    if not isinstance(level, (int, float)):
        return "<div class='signal-bar'><div class='signal-fill' style='width:0%'></div></div>"
    percent = max(min(float(level) / 1023.0 * 100.0, 100.0), 0.0)
    css = "signal-good" if level >= threshold else "signal-weak"
    marker = threshold / 1023.0 * 100.0
    return (
        f"<div class='signal-bar'><div class='signal-fill {css}' style='width:{percent:.0f}%'></div>"
        f"<div class='signal-mark' style='left:{marker:.0f}%'></div></div>"
    )


def device_ip(device: DeviceSpec, state_record: dict[str, object] | None) -> str:
    telemetry = (state_record or {}).get("telemetry") or {}
    return str(device.option("host") or (telemetry.get("ip") if isinstance(telemetry, dict) else "") or "")


def option_float(device: DeviceSpec, key: str, default: float) -> float:
    try:
        return float(device.option(key, default))
    except (TypeError, ValueError):
        return default

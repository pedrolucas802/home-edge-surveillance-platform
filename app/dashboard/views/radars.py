from __future__ import annotations

import math
import time

import streamlit as st

from app.dashboard import common
from app.devices.types import RADAR


def render() -> None:
    ctx = common.get_context()
    common.page_header("Radars", "Who is where right now, from the 24 GHz RD-03D motion radars.", "📡")
    radars = ctx.devices_of(RADAR)
    if not radars:
        st.info("No radars in the devices file yet. Add them on the Configuration page.")
        return
    per_row = st.segmented_control("Per row", options=[2, 3, 4], default=2, required=True, key="radar_per_row")
    st.fragment(run_every=min(ctx.refresh_seconds, 1.0))(_live)(int(per_row or 2))

    devices_tab, events_tab = st.tabs(["Devices", "Presence events"])
    with devices_tab:
        st.fragment(run_every=max(ctx.refresh_seconds, 5.0))(_devices)()
    with events_tab:
        st.fragment(run_every=max(ctx.refresh_seconds, 5.0))(_events)()


def _live(per_row: int) -> None:
    ctx = common.get_context()
    radars = ctx.devices_of(RADAR)
    states = common.load_device_states(ctx)
    online = sum(1 for d in radars if common.sensor_presence(states.get(d.id), ctx.settings) == "online")
    occupied = [d for d in radars if (states.get(d.id) or {}).get("state", {}).get("presence")]
    people = sum(int((states.get(d.id) or {}).get("state", {}).get("target_count") or 0) for d in occupied)
    hour_events = common.recent_events(
        ctx, device_types=[RADAR], since_epoch=time.time() - 3600, limit=2000
    )
    common.stat_tiles(
        [
            ("Radars online", f"{online}/{len(radars)}", "reporting to the hub"),
            ("People tracked", people, "right now"),
            ("Rooms occupied", len(occupied), ", ".join(d.name for d in occupied) or "none"),
            ("Presence events", sum(1 for e in hour_events if e.get("event_type") == "presence_start"), "in the last hour"),
        ]
    )

    columns = st.columns(per_row)
    for idx, device in enumerate(radars):
        record = states.get(device.id) or {}
        state = record.get("state", {}) or {}
        presence = common.sensor_presence(record, ctx.settings)
        targets = state.get("targets") if state.get("presence") else []
        alert = common.sensor_alert_text(state)
        max_range = common.option_float(device, "max_range_m", 8.0)
        rows: list[tuple[str, object]] = [
            ("People tracked", state.get("target_count") or 0),
            ("Nearest", f"{state['nearest_m']} m" if state.get("nearest_m") is not None else "–"),
            ("Last event", f"{record.get('last_event_type') or '–'} ({common.format_age(record.get('last_event_at_epoch'))})"),
        ]
        target_rows = "".join(
            f"<div class='kv-row'><span>Target {n}</span><strong>{t.get('distance_mm', 0) / 1000:.2f} m · "
            f"{_angle(t):+.0f}° · {t.get('speed_cms', 0)} cm/s</strong></div>"
            for n, t in enumerate(targets if isinstance(targets, list) else [], start=1)
            if isinstance(t, dict)
        )
        body = (
            (f"<div class='card-alert-text'>{common.esc(alert)}</div>" if alert else "")
            + common.radar_map_svg(targets, max_range_m=max_range, height=170)
            + common.rows_html(rows)
            + target_rows
        )
        subtitle = " · ".join(p for p in (device.location, str(device.option("model") or "")) if p)
        css = "card-alert" if state.get("presence") else ("card-offline" if presence == "offline" else "")
        with columns[idx % per_row]:
            st.markdown(common.card(device.name, subtitle, presence, body, css), unsafe_allow_html=True)


def _angle(target: dict[str, object]) -> float:
    if "angle_deg" in target:
        try:
            return float(target["angle_deg"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    return math.degrees(math.atan2(float(target.get("x_mm", 0)), float(target.get("y_mm", 1) or 1)))


def _devices() -> None:
    ctx = common.get_context()
    states = common.load_device_states(ctx)
    rows = []
    for device in ctx.devices_of(RADAR):
        record = states.get(device.id) or {}
        telemetry = record.get("telemetry", {}) or {}
        rows.append(
            {
                "status": f"{common.PRESENCE_ICON.get(common.sensor_presence(record, ctx.settings), '⚪')} "
                f"{common.sensor_presence(record, ctx.settings)}",
                "name": device.name,
                "id": device.id,
                "location": device.location,
                "zone": device.zone,
                "model": device.option("model") or telemetry.get("model"),
                "ip": common.device_ip(device, record),
                "mount_height_m": device.option("mount_height_m"),
                "max_range_m": device.option("max_range_m", 8.0),
                "rssi_dbm": telemetry.get("rssi"),
                "firmware": telemetry.get("fw") or telemetry.get("firmware"),
                "last_seen": common.format_age(record.get("last_seen_at_epoch")),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("Mounting height and range limit are set under Configuration → Alignment.")


def _events() -> None:
    ctx = common.get_context()
    events = common.recent_events(ctx, device_types=[RADAR], limit=60)
    if not events:
        st.info("No radar events yet.")
        return
    rows = []
    for event in events:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        device = ctx.registry.get(str(event.get("device_id")))
        rows.append(
            {
                "time": common.format_timestamp(event.get("event_at_utc")),
                "radar": device.name if device else event.get("device_id"),
                "event": str(event.get("event_type", "")).replace("_", " "),
                "people": data.get("target_count"),
                "nearest_m": data.get("nearest_m"),
                "zone": device.zone if device else "",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

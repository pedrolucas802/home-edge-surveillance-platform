from __future__ import annotations

import time

import streamlit as st

from app.dashboard import common
from app.devices.types import TRIPWIRE


def render() -> None:
    ctx = common.get_context()
    common.page_header("Tripwires", "Laser and IR beams on the walls, gates and hallways.", "🚧")
    tripwires = ctx.devices_of(TRIPWIRE)
    if not tripwires:
        st.info("No tripwires in the devices file yet. Add them on the Configuration page.")
        return
    st.fragment(run_every=min(ctx.refresh_seconds, 1.0))(_live)()

    events_tab, devices_tab = st.tabs(["Crossings and events", "Devices"])
    with events_tab:
        st.fragment(run_every=max(ctx.refresh_seconds, 5.0))(_events)()
    with devices_tab:
        st.fragment(run_every=max(ctx.refresh_seconds, 5.0))(_devices)()


def _live() -> None:
    ctx = common.get_context()
    tripwires = ctx.devices_of(TRIPWIRE)
    states = common.load_device_states(ctx)
    online = sum(1 for d in tripwires if common.sensor_presence(states.get(d.id), ctx.settings) == "online")
    beams = {d.id: str((states.get(d.id) or {}).get("state", {}).get("beam") or "ok") for d in tripwires}
    hour = common.recent_events(ctx, device_types=[TRIPWIRE], since_epoch=time.time() - 3600, limit=2000)
    common.stat_tiles(
        [
            ("Tripwires online", f"{online}/{len(tripwires)}", "reporting to the hub"),
            ("Beams broken", sum(1 for b in beams.values() if b == "broken"), "right now"),
            ("Crossings", sum(1 for e in hour if e.get("event_type") == "beam_broken"), "in the last hour"),
            ("Faults", sum(1 for b in beams.values() if b == "fault"), "need attention"),
        ]
    )

    columns = st.columns(min(len(tripwires), 4))
    for idx, device in enumerate(tripwires):
        record = states.get(device.id) or {}
        state = record.get("state", {}) or {}
        presence = common.sensor_presence(record, ctx.settings)
        beam = beams[device.id]
        alert = common.sensor_alert_text(state)
        threshold = int(common.option_float(device, "signal_threshold", 500))
        body = (
            (f"<div class='card-alert-text'>{common.esc(alert)}</div>" if alert else "")
            + common.beam_svg(beam)
            + common.rows_html(
                [
                    ("Beam", {"ok": "clear", "broken": "broken", "fault": "fault"}.get(beam, beam)),
                    ("Last crossing", common.format_age(record.get("last_event_at_epoch"))
                     if record.get("last_event_type") in {"beam_broken", "beam_restored"} else "–"),
                    ("Last break", f"{state['duration_ms']} ms" if state.get("duration_ms") else "–"),
                    ("Receiver level", state.get("signal", "–")),
                ]
            )
            + common.signal_bar(state.get("signal"), threshold)
        )
        subtitle = " · ".join(p for p in (device.location, str(device.option("variant") or "")) if p)
        css = "card-alert" if beam in {"broken", "fault"} or state.get("tamper") else (
            "card-offline" if presence == "offline" else ""
        )
        with columns[idx % len(columns)]:
            st.markdown(common.card(device.name, subtitle, presence, body, css), unsafe_allow_html=True)


def _events() -> None:
    ctx = common.get_context()
    names = {d.id: d.name for d in ctx.devices_of(TRIPWIRE)}
    filter_col, kind_col = st.columns([2, 2])
    chosen = filter_col.multiselect(
        "Tripwires", options=list(names), format_func=lambda i: names.get(i, i), placeholder="All tripwires", key="tw_filter"
    )
    kinds = kind_col.multiselect(
        "Events", options=["beam_broken", "beam_restored", "fault", "tamper"], placeholder="All events",
        format_func=lambda k: k.replace("_", " "), key="tw_kinds",
    )
    events = common.recent_events(ctx, device_ids=chosen or None, device_types=[TRIPWIRE], limit=300)
    if kinds:
        events = [e for e in events if e.get("event_type") in kinds]
    if not events:
        st.info("No tripwire events yet.")
        return
    rows = []
    for event in events[:120]:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        device = ctx.registry.get(str(event.get("device_id")))
        icon = {"beam_broken": "🔴", "fault": "🟠", "tamper": "🟣"}.get(str(event.get("event_type")), "🟢")
        rows.append(
            {
                "time": common.format_timestamp(event.get("event_at_utc")),
                "tripwire": device.name if device else event.get("device_id"),
                "event": f"{icon} {str(event.get('event_type', '')).replace('_', ' ')}",
                "duration_ms": data.get("duration_ms"),
                "signal": data.get("signal"),
                "detail": data.get("reason", ""),
                "zone": device.zone if device else "",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def _devices() -> None:
    ctx = common.get_context()
    states = common.load_device_states(ctx)
    rows = []
    for device in ctx.devices_of(TRIPWIRE):
        record = states.get(device.id) or {}
        telemetry = record.get("telemetry", {}) or {}
        presence = common.sensor_presence(record, ctx.settings)
        rows.append(
            {
                "status": f"{common.PRESENCE_ICON.get(presence, '⚪')} {presence}",
                "name": device.name,
                "id": device.id,
                "location": device.location,
                "zone": device.zone,
                "variant": device.option("variant"),
                "ip": common.device_ip(device, record),
                "signal_threshold": device.option("signal_threshold", 500),
                "rssi_dbm": telemetry.get("rssi"),
                "firmware": telemetry.get("fw") or telemetry.get("firmware"),
                "last_seen": common.format_age(record.get("last_seen_at_epoch")),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

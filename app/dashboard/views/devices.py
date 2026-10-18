from __future__ import annotations

import streamlit as st

from app.dashboard import common, nav
from app.devices.types import get_device_type


def render() -> None:
    ctx = common.get_context()
    common.page_header("All devices", "Every camera, radar and tripwire at this site, with network details.", "🗂️")
    type_keys = [device.type for device in ctx.registry.devices]
    type_keys = list(dict.fromkeys(type_keys))
    type_col, status_col, search_col = st.columns([2, 2, 2])
    types = type_col.multiselect(
        "Type", options=type_keys, format_func=lambda k: get_device_type(k).plural, placeholder="All types"
    )
    statuses_wanted = status_col.multiselect(
        "Status", options=["online", "offline", "never seen", "disabled"], placeholder="Any status"
    )
    search = search_col.text_input("Search", placeholder="Name, id, IP, location…").strip().lower()
    st.fragment(run_every=max(ctx.refresh_seconds, 5.0))(_table)(types, statuses_wanted, search)


def _table(types: list[str], statuses_wanted: list[str], search: str) -> None:
    ctx = common.get_context()
    statuses = common.load_statuses(ctx)
    states = common.load_device_states(ctx)
    rows = []
    counts = {"online": 0, "offline": 0, "never seen": 0, "disabled": 0}
    for device in ctx.registry.devices:
        record = states.get(device.id) or {}
        telemetry = record.get("telemetry", {}) or {}
        status = statuses.get(device.id) or {}
        presence = common.device_presence(device, statuses, states, ctx.settings)
        counts[presence] = counts.get(presence, 0) + 1
        ip = common.device_ip(device, record) or (ctx.camera_host(device.id) if device.type == "camera" else "")
        row = {
            "status": f"{common.PRESENCE_ICON.get(presence, '⚪')} {presence}",
            "type": f"{device.device_type.icon} {device.device_type.label}",
            "name": device.name,
            "id": device.id,
            "ip": ip,
            "mac": telemetry.get("mac", ""),
            "location": device.location,
            "zone": device.zone,
            "transport": device.device_type.transport.upper(),
            "model": device.option("model") or device.option("variant") or status.get("yolo_model") or "",
            "rssi_dbm": telemetry.get("rssi"),
            "firmware": telemetry.get("fw") or telemetry.get("firmware") or "",
            "uptime": _uptime(telemetry.get("uptime_ms")),
            "last_seen": common.format_age(common.device_last_seen(device, statuses, record)),
            "last_event": record.get("last_event_type") or "",
        }
        if types and device.type not in types:
            continue
        if statuses_wanted and presence not in statuses_wanted:
            continue
        if search and search not in " ".join(str(value).lower() for value in row.values()):
            continue
        rows.append(row)

    common.stat_tiles(
        [
            ("Devices", len(ctx.registry.devices), f"site {ctx.registry.site_id}"),
            ("Online", counts["online"], "reachable now"),
            ("Offline", counts["offline"], "not reporting"),
            ("Never seen", counts["never seen"], "waiting for first contact"),
        ]
    )
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "rssi_dbm": st.column_config.NumberColumn("Wi-Fi (dBm)", format="%d"),
            "ip": st.column_config.TextColumn("IP address"),
        },
    )
    st.caption(f"{len(rows)} shown · devices file: {ctx.registry.source or 'n/a'}")

    unregistered = [record for record in states.values() if not record.get("registered", True)]
    if unregistered:
        st.warning(
            "These devices are talking to the hub but are not in the devices file, so their events "
            "are not kept: " + ", ".join(str(record["device_id"]) for record in unregistered)
        )
        nav.link("configuration", "Add them in configuration", icon=":material/add:")


def _uptime(value: object) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    seconds = int(value / 1000)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    return f"{days}d {hours}h" if days else f"{hours}h {minutes}m"

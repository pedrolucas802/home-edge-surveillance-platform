from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from app.dashboard import common, nav, runtime
from app.devices.types import CAMERA, RADAR, TRIPWIRE


def render() -> None:
    st.fragment(run_every=5.0)(_body)()


def _body() -> None:
    ctx = common.get_context()
    statuses = common.load_statuses(ctx)
    states = common.load_device_states(ctx)
    devices = ctx.registry.enabled()
    presence = {
        device.id: common.device_presence(device, statuses, states, ctx.settings) for device in devices
    }
    online = sum(1 for value in presence.values() if value == "online")
    day_ago = time.time() - 86400
    alerts_today = common.recent_alerts(ctx, since_epoch=day_ago, limit=500)
    detection_running, _ = runtime.managed_publishers_status()
    model_on = detection_running or any(
        bool(status and status.get("yolo_model")) for status in statuses.values()
    )

    _hero(ctx, online, len(devices), len(alerts_today), model_on)

    st.markdown("<div class='section-title'>Jump to</div>", unsafe_allow_html=True)
    radars = ctx.devices_of(RADAR)
    tripwires = ctx.devices_of(TRIPWIRE)
    people_in_range = sum(
        int((states.get(device.id) or {}).get("state", {}).get("target_count") or 0)
        for device in radars
        if (states.get(device.id) or {}).get("state", {}).get("presence")
    )
    beams_broken = sum(
        1 for device in tripwires if (states.get(device.id) or {}).get("state", {}).get("beam") == "broken"
    )
    cameras_live = sum(1 for status in statuses.values() if common.camera_is_live(status, ctx.settings))
    offline = [device for device in devices if presence[device.id] == "offline"]

    cards = [
        ("cameras", "📷", "Cameras", "Live views, person detections and snapshots.",
         f"{cameras_live}/{len(ctx.cameras())} live"),
        ("radars", "📡", "Radars", "Where people are right now, from the mmWave radars.",
         f"{people_in_range} {'person' if people_in_range == 1 else 'people'} in range"),
        ("tripwires", "🚧", "Tripwires", "Beam status and every crossing at the walls and gates.",
         f"{beams_broken} beam{'s' if beams_broken != 1 else ''} broken now"),
        ("devices", "🗂️", "All devices", "Every device with its IP, signal, firmware and last seen.",
         f"{len(offline)} offline" if offline else "All reachable"),
        ("configuration", "⚙️", "Configuration", "Devices, alerts, Wi-Fi, alignment, model and your profile.",
         "Detection model on" if model_on else "Detection model off"),
    ]
    columns = st.columns(len(cards))
    for column, (key, icon, title, text, stat) in zip(columns, cards):
        with column, st.container(border=True):
            st.markdown(
                f"<div class='link-card-icon'>{icon}</div>"
                f"<div class='link-card-title'>{common.esc(title)}</div>"
                f"<div class='link-card-text'>{common.esc(text)}</div>"
                f"<div class='link-card-stat'><strong>{common.esc(stat)}</strong></div>",
                unsafe_allow_html=True,
            )
            nav.link(key, "Open", icon=":material/arrow_forward:")

    left, right = st.columns([1.35, 1.0], gap="large")
    with left:
        st.markdown("<div class='section-title'>How it works</div>", unsafe_allow_html=True)
        features = [
            ("🧠", "On-site AI", "Camera feeds run through a YOLO model on this hub. Video never leaves the house."),
            ("📡", "Radar presence", "24 GHz RD-03D radars track up to 3 people each, through the dark and behind curtains."),
            ("🚧", "Perimeter beams", "Laser and IR tripwires on walls and gates catch anyone crossing the line."),
            ("🔔", "One alert stream", "Every device reports to the hub, so one person walking in shows up everywhere at once."),
        ]
        feature_cols = st.columns(2)
        for idx, (icon, title, text) in enumerate(features):
            feature_cols[idx % 2].markdown(
                f"<div class='feature'><div style='font-size:1.4rem'>{icon}</div>"
                f"<h4>{common.esc(title)}</h4><p>{common.esc(text)}</p></div>",
                unsafe_allow_html=True,
            )
    with right:
        st.markdown("<div class='section-title'>Latest alerts</div>", unsafe_allow_html=True)
        with st.container(border=True):
            if not ctx.hub_config.get("alerts", {}).get("enabled", True):
                st.caption("Alerts are switched off on the Configuration page.")
            elif not alerts_today:
                st.caption("Nothing in the last 24 hours. All quiet.")
            else:
                rows = "".join(
                    f"<div class='feed-row'><span>{common.esc(common.describe_event(event, ctx.registry))}</span>"
                    f"<span class='feed-time'>{common.esc(common.format_time_short(event.get('event_at_utc')))}</span></div>"
                    for event in alerts_today[:6]
                )
                st.markdown(rows, unsafe_allow_html=True)


def _hero(ctx: common.HubContext, online: int, total: int, alerts_today: int, model_on: bool) -> None:
    hour = datetime.now().hour
    greeting = "Good morning" if 5 <= hour < 12 else "Good afternoon" if 12 <= hour < 18 else "Good evening"
    name = ctx.user_name
    counts = ctx.registry.by_type()
    chips = [
        f"<strong>{online}/{total}</strong> devices online",
        f"<strong>{alerts_today}</strong> alert{'s' if alerts_today != 1 else ''} today",
        f"<strong>{len(counts.get(CAMERA, []))}</strong> cameras",
        f"<strong>{len(counts.get(RADAR, []))}</strong> radars",
        f"<strong>{len(counts.get(TRIPWIRE, []))}</strong> tripwires",
        f"Detection model <strong>{'on' if model_on else 'off'}</strong>",
    ]
    st.markdown(
        "<div class='hero'>"
        f"<div class='hero-kicker'>Home Edge Surveillance · site {common.esc(ctx.registry.site_id)}</div>"
        f"<h1>{greeting}{', ' + common.esc(name) if name else ''}.</h1>"
        "<p>Cameras, motion radars and tripwires working together on a hub inside your home. "
        "Everything is processed locally and private by design.</p>"
        f"<div class='hero-chips'>{''.join(f'<span class=hero-chip>{chip}</span>' for chip in chips)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

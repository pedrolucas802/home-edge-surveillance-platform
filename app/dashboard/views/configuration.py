from __future__ import annotations

import copy
import os
from collections.abc import Callable
from datetime import time as dt_time

import streamlit as st

from app.config.hub_config import ALERT_KINDS, save_hub_config
from app.dashboard import common, runtime
from app.devices.registry import device_to_dict, resolve_devices_config_path, save_registry
from app.devices.types import CAMERA, RADAR, TRIPWIRE, get_device_type, registered_device_types
from app.sensors.commands import send_command

_FLASH_KEY = "config_flash"


def render() -> None:
    ctx = common.get_context()
    common.page_header("Configuration", "Devices, alerts, network, alignment, detection model and your profile.", "⚙️")
    flash = st.session_state.pop(_FLASH_KEY, None)
    if flash:
        (st.success if flash[0] == "ok" else st.error)(flash[1])

    tabs = st.tabs(
        ["🗂️ Devices", "📷 Cameras", "🧠 Detection model", "🔔 Alerts", "📶 Wi-Fi", "🎯 Alignment", "👤 User"],
        key="config_tab",
        on_change="rerun",
    )
    with tabs[0]:
        _devices_tab(ctx)
    with tabs[1]:
        _cameras_tab(ctx)
    with tabs[2]:
        _model_tab(ctx)
    with tabs[3]:
        _alerts_tab(ctx)
    with tabs[4]:
        _wifi_tab(ctx)
    with tabs[5]:
        _alignment_tab(ctx)
    with tabs[6]:
        _user_tab(ctx)


# Helpers ------------------------------------------------------------------


def _done(kind: str, message: str) -> None:
    st.session_state[_FLASH_KEY] = (kind, message)
    st.rerun()


def _devices_path(ctx: common.HubContext) -> str:
    if ctx.registry.source and ctx.registry.source != "environment":
        return ctx.registry.source
    return str(resolve_devices_config_path(ctx.settings.devices_config))


def _save_devices(ctx: common.HubContext, entries: list[dict[str, object]], message: str) -> None:
    try:
        save_registry(_devices_path(ctx), ctx.registry.site_id, entries)
    except ValueError as exc:
        st.error(f"Not saved: {exc}")
        return
    _done("ok", message)


def _update_device_options(ctx: common.HubContext, device_id: str, changes: dict[str, object], message: str) -> None:
    entries = [device_to_dict(device) for device in ctx.registry.devices]
    for entry in entries:
        if entry["id"] == device_id:
            for key, value in changes.items():
                if value in (None, ""):
                    entry.pop(key, None)
                else:
                    entry[key] = value
    _save_devices(ctx, entries, message)


def _save_config(ctx: common.HubContext, mutate: Callable[[dict[str, object]], None], message: str) -> None:
    config = copy.deepcopy(ctx.hub_config)
    mutate(config)
    try:
        save_hub_config(ctx.hub_config_path, config)
    except OSError as exc:
        st.error(f"Not saved: {exc}")
        return
    _done("ok", message)


def _send(ctx: common.HubContext, device_ids: list[str], payload: dict[str, object], what: str) -> None:
    try:
        sent = send_command(ctx.settings, ctx.registry.site_id, device_ids, payload)
    except Exception as exc:  # noqa: BLE001 - broker down, DNS, refused, ...
        st.error(
            f"Could not reach the MQTT broker at {ctx.settings.mqtt_host}:{ctx.settings.mqtt_port} ({exc}). "
            "Start Mosquitto (docker compose up -d mosquitto) and try again."
        )
        return
    st.success(f"Sent {what} to {sent} of {len(device_ids)} device(s).")


def _mqtt_devices(ctx: common.HubContext) -> list:
    return [device for device in ctx.registry.enabled() if device.device_type.transport == "mqtt"]


# Devices ------------------------------------------------------------------


def _devices_tab(ctx: common.HubContext) -> None:
    st.markdown("#### Devices at this site")
    st.caption(
        f"Site **{ctx.registry.site_id}** · saved to `{_devices_path(ctx)}`. Add a row for a new device, "
        "select rows and press Delete to remove them. IP is the camera address, or a fixed IP for a board."
    )
    type_options = sorted({t.key for t in registered_device_types()} | {d.type for d in ctx.registry.devices})
    rows = [
        {
            "id": device.id,
            "type": device.type,
            "name": device.name,
            "location": device.location,
            "zone": device.zone,
            "ip": str(device.option("host") or ""),
            "enabled": device.enabled,
        }
        for device in ctx.registry.devices
    ]
    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="devices_editor",
        column_config={
            "id": st.column_config.TextColumn("ID", required=True, help="Lowercase letters, digits, - and _"),
            "type": st.column_config.SelectboxColumn(
                "Type", options=type_options, required=True,
                help=", ".join(f"{k}: {get_device_type(k).label}" for k in type_options),
            ),
            "name": st.column_config.TextColumn("Name"),
            "location": st.column_config.TextColumn("Location"),
            "zone": st.column_config.TextColumn("Zone", help="Devices watching the same area share a zone."),
            "ip": st.column_config.TextColumn("IP"),
            "enabled": st.column_config.CheckboxColumn("Enabled", default=True),
        },
    )
    if st.button("Save devices", type="primary", key="save_devices"):
        existing = {device.id: device_to_dict(device) for device in ctx.registry.devices}
        entries: list[dict[str, object]] = []
        for row in edited:
            device_id = str(row.get("id") or "").strip()
            if not device_id:
                continue
            entry = existing.get(device_id, {}).copy()
            entry.update(
                {
                    "id": device_id,
                    "type": str(row.get("type") or "").strip(),
                    "name": str(row.get("name") or "").strip() or device_id,
                    "location": str(row.get("location") or "").strip(),
                    "zone": str(row.get("zone") or "").strip(),
                    "enabled": bool(row.get("enabled", True)),
                }
            )
            ip = str(row.get("ip") or "").strip()
            if ip:
                entry["host"] = ip
            else:
                entry.pop("host", None)
            entries.append(entry)
        _save_devices(ctx, entries, f"Saved {len(entries)} device(s). Restart the hub services to apply them to running sensors.")

    unregistered = [r for r in common.load_device_states(ctx).values() if not r.get("registered", True)]
    if unregistered:
        st.markdown("#### Devices asking to join")
        for record in unregistered:
            col_text, col_button = st.columns([3, 1])
            col_text.write(f"`{record['device_id']}` · {get_device_type(str(record['device_type'])).label}")
            if col_button.button("Add", key=f"add_{record['device_id']}"):
                entries = [device_to_dict(device) for device in ctx.registry.devices]
                entries.append({"id": record["device_id"], "type": record["device_type"], "name": record["device_id"]})
                _save_devices(ctx, entries, f"Added {record['device_id']}.")


# Cameras ------------------------------------------------------------------


def _cameras_tab(ctx: common.HubContext) -> None:
    cameras = [device for device in ctx.registry.devices if device.type == CAMERA]
    st.markdown("#### Camera addresses")
    if not cameras:
        st.info("No cameras yet. Add a device of type camera in the Devices tab.")
    else:
        with st.form("camera_ips"):
            values: dict[str, tuple[str, str, bool]] = {}
            for device in cameras:
                name_col, ip_col, on_col = st.columns([2, 2, 1])
                name = name_col.text_input("Name", value=device.name, key=f"cam_name_{device.id}")
                ip = ip_col.text_input(f"IP ({device.id})", value=str(device.option("host") or ""), key=f"cam_ip_{device.id}")
                on_col.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
                enabled = on_col.toggle("Enabled", value=device.enabled, key=f"cam_on_{device.id}")
                values[device.id] = (name.strip(), ip.strip(), enabled)
            if st.form_submit_button("Save cameras", type="primary"):
                entries = [device_to_dict(device) for device in ctx.registry.devices]
                for entry in entries:
                    if entry["id"] in values:
                        name, ip, enabled = values[str(entry["id"])]
                        entry["name"] = name or entry["id"]
                        entry["enabled"] = enabled
                        if ip:
                            entry["host"] = ip
                        else:
                            entry.pop("host", None)
                _save_devices(ctx, entries, "Saved camera settings. Restart the cameras to connect to new IPs.")

    s = ctx.settings
    st.markdown("#### Stream settings")
    st.caption("Shared by every camera; set in `.env`.")
    with st.container(border=True):
        st.markdown(
            common.rows_html(
                [
                    ("RTSP port · CAM_PORT", s.camera_port),
                    ("Stream path · CAM_STREAM", s.camera_stream),
                    ("Transport · RTSP_TRANSPORT", s.rtsp_transport),
                    ("User · CAM_USER", s.camera_user),
                    ("Password · CAM_PASS", "set" if s.camera_pass else "not set"),
                    ("Frames · FRAME_WIDTH/HEIGHT/FPS", f"{s.width}x{s.height} @ {s.output_fps} fps"),
                ]
            ),
            unsafe_allow_html=True,
        )
    st.caption("Finding a new camera's IP: `python scripts/discover_onvif.py` lists ONVIF cameras on the network.")

    st.markdown("#### Restart cameras")
    camera_ids = ctx.camera_ids()
    chosen = st.multiselect("Cameras to restart", options=camera_ids, default=camera_ids, key="restart_cams")
    if st.button("Restart cameras", key="restart_cams_button"):
        ok, message = runtime.restart_cameras(settings=s, cameras=chosen or camera_ids)
        _done("ok" if ok else "error", message)


# Detection model ------------------------------------------------------------


def _model_tab(ctx: common.HubContext) -> None:
    s = ctx.settings
    running, state = runtime.managed_publishers_status()
    workers = runtime.list_dashboard_publisher_worker_pids()
    saved = ctx.hub_config.get("yolo", {}) or {}
    source = state if running and state else saved

    if running and state:
        st.success(
            f"Detection is **on**: {state.get('yolo_model')} on {state.get('yolo_device')} for "
            f"{', '.join(map(str, state.get('cameras') or [])) or 'all cameras'} "
            f"(started {common.format_timestamp(state.get('started_at_utc'))})."
        )
    elif workers:
        st.warning(f"Camera publishers are running outside the dashboard (PIDs {', '.join(map(str, workers))}).")
    else:
        st.info("Detection is **off**. Pick a model and turn it on.")
    if os.getenv("HES_DEMO"):
        st.caption("Demo mode: cameras are simulated, so turning the model on will try to reach real RTSP cameras.")

    def pick(key: str, default: object) -> object:
        value = source.get(key) if isinstance(source, dict) else None
        return default if value in (None, "") else value

    model_options = runtime.discover_model_options(s.yolo_model)
    current_model = str(pick("yolo_model", None) or saved.get("model") or s.yolo_model)
    if current_model not in model_options:
        model_options.insert(0, current_model)

    with st.form("model_form"):
        model_col, device_col = st.columns([2, 1])
        model = model_col.selectbox("Model", options=model_options, index=model_options.index(current_model))
        custom = model_col.text_input("Or a model path", value="", placeholder="models/my-model.pt").strip()
        device_options = ["mps", "cpu", "cuda", "auto"]
        current_device = str(pick("yolo_device", saved.get("device") or s.yolo_device)).lower()
        if current_device not in device_options:
            device_options.insert(0, current_device)
        device = device_col.selectbox("Runs on", options=device_options, index=device_options.index(current_device),
                                      help="mps uses the Apple Silicon GPU.")
        t1, t2, t3 = st.columns(3)
        detect = t1.toggle("Detect objects", value=bool(pick("yolo_detect", saved.get("detect", True))))
        pose = t2.toggle("Pose skeletons", value=bool(pick("yolo_pose", saved.get("pose", False))))
        track = t3.toggle("Track IDs", value=bool(pick("yolo_tracking", saved.get("track", False))))
        c1, c2, c3 = st.columns(3)
        tracker_options = ["bytetrack.yaml", "botsort.yaml"]
        current_tracker = str(pick("yolo_tracker", saved.get("tracker") or s.yolo_tracker))
        if current_tracker not in tracker_options:
            tracker_options.insert(0, current_tracker)
        tracker = c1.selectbox("Tracker", options=tracker_options, index=tracker_options.index(current_tracker))
        classes = c2.text_input("Classes", value=str(pick("yolo_classes", saved.get("classes") or "person")),
                                help="Comma-separated, or all.").strip() or "all"
        infer_every = c3.slider("Run on every Nth frame", 1, 6, int(pick("yolo_infer_every_n", saved.get("infer_every_n") or s.yolo_infer_every_n)))
        confidence = st.slider("Confidence", 0.05, 0.95, float(pick("yolo_confidence", saved.get("confidence") or s.yolo_confidence)), 0.01)
        cameras = st.multiselect("Cameras", options=ctx.camera_ids(),
                                 default=[c for c in (state or {}).get("cameras", []) if c in ctx.camera_ids()] or ctx.camera_ids())
        on_col, off_col, save_col = st.columns(3)
        turn_on = on_col.form_submit_button("Apply and restart" if running else "Turn on", type="primary", width="stretch")
        turn_off = off_col.form_submit_button("Turn off", width="stretch", disabled=not (running or workers))
        save_only = save_col.form_submit_button("Save as default", width="stretch")

    choice = {
        "model": custom or model, "device": device, "detect": detect, "pose": pose, "track": track,
        "tracker": tracker, "classes": classes, "confidence": round(confidence, 2), "infer_every_n": infer_every,
    }
    if turn_on or save_only:
        config = copy.deepcopy(ctx.hub_config)
        config["yolo"] = {**choice, "enabled": bool(turn_on or saved.get("enabled"))}
        save_hub_config(ctx.hub_config_path, config)
    if save_only:
        _done("ok", f"Saved {choice['model']} as the default model.")
    if turn_on:
        if not cameras:
            st.error("Pick at least one camera.")
            return
        with st.spinner("Starting camera publishers…"):
            ok, message = runtime.start_managed_publishers(
                cameras=cameras, yolo_model=str(choice["model"]), yolo_device=device,
                yolo_detect_enabled=detect, yolo_pose_enabled=pose, yolo_classes=classes,
                yolo_confidence=confidence, yolo_infer_every_n=infer_every, yolo_tracking=track,
                yolo_tracker=tracker, launch_delay=3,
            )
        _done("ok" if ok else "error", message)
    if turn_off:
        ok, message = runtime.stop_managed_publishers()
        config = copy.deepcopy(ctx.hub_config)
        config.setdefault("yolo", {})["enabled"] = False
        save_hub_config(ctx.hub_config_path, config)
        _done("ok" if ok else "error", message)
    st.caption(f"Publisher log: `{runtime.PUBLISHERS_SUPERVISOR_LOG}`")


# Alerts -------------------------------------------------------------------


def _alerts_tab(ctx: common.HubContext) -> None:
    alerts = ctx.hub_config.get("alerts", {}) or {}
    kinds = alerts.get("kinds", {}) or {}
    quiet = alerts.get("quiet_hours", {}) or {}
    channels = alerts.get("channels", {}) or {}
    devices = ctx.registry.devices
    with st.form("alerts_form"):
        enabled = st.toggle("Alerts on", value=bool(alerts.get("enabled", True)),
                            help="Master switch. Off silences everything below.")
        st.markdown("**Alert me when…**")
        cols = st.columns(2)
        chosen_kinds = {
            kind: cols[idx % 2].toggle(label, value=bool(kinds.get(kind, True)), key=f"alert_{kind}")
            for idx, (kind, label) in enumerate(ALERT_KINDS.items())
        }
        muted = st.multiselect(
            "Muted devices", options=[d.id for d in devices],
            default=[d for d in alerts.get("muted_devices", []) if ctx.registry.get(d)],
            format_func=lambda i: f"{ctx.registry.get(i).name} ({i})" if ctx.registry.get(i) else i,
            help="These devices still record events but never raise alerts.",
        )
        st.markdown("**Quiet hours**")
        q1, q2, q3 = st.columns([1, 1, 1])
        quiet_on = q1.toggle("Quiet hours", value=bool(quiet.get("enabled", False)))
        start = q2.time_input("From", value=_clock(quiet.get("start"), dt_time(23, 0)))
        end = q3.time_input("Until", value=_clock(quiet.get("end"), dt_time(6, 0)))
        st.markdown("**Deliver to**")
        d1, d2, d3 = st.columns(3)
        dashboard = d1.toggle("Dashboard", value=bool(channels.get("dashboard", True)))
        email = d2.toggle("Email", value=bool(channels.get("email", False)))
        push = d3.toggle("Phone push", value=bool(channels.get("push", False)))
        st.caption("Email and push are stored now and used once the notification service is added.")
        submitted = st.form_submit_button("Save alerts", type="primary")
    if submitted:
        def mutate(config: dict[str, object]) -> None:
            config["alerts"] = {
                "enabled": enabled,
                "kinds": chosen_kinds,
                "muted_devices": muted,
                "quiet_hours": {"enabled": quiet_on, "start": start.strftime("%H:%M"), "end": end.strftime("%H:%M")},
                "channels": {"dashboard": dashboard, "email": email, "push": push},
            }
        _save_config(ctx, mutate, "Alert settings saved.")


def _clock(value: object, default: dt_time) -> dt_time:
    try:
        hours, minutes = str(value).split(":", 1)
        return dt_time(int(hours), int(minutes))
    except (TypeError, ValueError):
        return default


# Wi-Fi --------------------------------------------------------------------


def _wifi_tab(ctx: common.HubContext) -> None:
    wifi = ctx.hub_config.get("wifi", {}) or {}
    st.markdown("#### Network the boards join")
    st.caption("Radars and tripwires (ESP32) connect to this Wi-Fi and to the hub's MQTT broker.")
    with st.form("wifi_form"):
        c1, c2, c3 = st.columns([2, 2, 1])
        ssid = c1.text_input("Network name (SSID)", value=str(wifi.get("ssid", "")))
        password = c2.text_input(
            "Password", value="", type="password",
            placeholder="unchanged" if wifi.get("password") else "Wi-Fi password",
        )
        country = c3.text_input("Country", value=str(wifi.get("country", "BR")), max_chars=2)
        prefix = st.text_input("Board hostname prefix", value=str(wifi.get("hostname_prefix", "hes")),
                               help="Boards appear on the router as <prefix>-<device id>.")
        submitted = st.form_submit_button("Save Wi-Fi", type="primary")
    if submitted:
        def mutate(config: dict[str, object]) -> None:
            current = config.setdefault("wifi", {})
            current.update({"ssid": ssid.strip(), "country": country.strip().upper(), "hostname_prefix": prefix.strip()})
            if password:
                current["password"] = password
        _save_config(ctx, mutate, "Wi-Fi settings saved on the hub.")

    s = ctx.settings
    st.markdown("#### Hub connection")
    with st.container(border=True):
        st.markdown(
            common.rows_html(
                [
                    ("MQTT broker", f"{s.mqtt_host}:{s.mqtt_port}"),
                    ("MQTT user", s.mqtt_username or "none"),
                    ("Topics", f"{s.mqtt_topic_prefix}/{ctx.registry.site_id}/<device>/…"),
                ]
            ),
            unsafe_allow_html=True,
        )

    st.markdown("#### Send to the boards")
    boards = _mqtt_devices(ctx)
    targets = st.multiselect("Boards", options=[d.id for d in boards], default=[d.id for d in boards],
                             format_func=lambda i: ctx.registry.get(i).name if ctx.registry.get(i) else i, key="wifi_targets")
    st.caption("Boards save the new network and reboot onto it. Those that cannot join fall back to their setup hotspot.")
    if st.button("Send Wi-Fi to boards", disabled=not (wifi.get("ssid") and targets)):
        _send(
            ctx, targets,
            {"cmd": "wifi", "ssid": wifi.get("ssid"), "password": wifi.get("password", ""),
             "country": wifi.get("country", ""), "hostname_prefix": wifi.get("hostname_prefix", "hes")},
            "the Wi-Fi settings",
        )

    states = common.load_device_states(ctx)
    rows = [
        {
            "board": d.name, "id": d.id,
            "ip": common.device_ip(d, states.get(d.id)),
            "rssi_dbm": ((states.get(d.id) or {}).get("telemetry") or {}).get("rssi"),
            "status": common.sensor_presence(states.get(d.id), s),
        }
        for d in boards
    ]
    if rows:
        st.markdown("#### Signal per board")
        st.dataframe(rows, width="stretch", hide_index=True)


# Alignment ------------------------------------------------------------------


def _alignment_tab(ctx: common.HubContext) -> None:
    tripwire_col, radar_col = st.columns(2, gap="large")
    with tripwire_col:
        st.markdown("#### Tripwire beams")
        tripwires = [d for d in ctx.registry.devices if d.type == TRIPWIRE]
        if not tripwires:
            st.caption("No tripwires.")
        else:
            device_id = st.selectbox("Tripwire", [d.id for d in tripwires],
                                     format_func=lambda i: ctx.registry.get(i).name, key="align_tw")
            device = ctx.registry.get(device_id)
            threshold = int(common.option_float(device, "signal_threshold", 500))
            st.fragment(run_every=1.0)(_tripwire_live)(device_id, threshold)
            st.caption(
                "Point the emitter at the receiver until the level stays above the marker. "
                "Alignment mode makes the board stream its level every 200 ms for a minute."
            )
            new_threshold = st.slider("Alarm threshold (receiver level)", 50, 1000, threshold, 10, key="tw_threshold",
                                      help="Below this level the beam counts as broken.")
            b1, b2 = st.columns(2)
            if b1.button("Save threshold", key="tw_save"):
                _update_device_options(ctx, device_id, {"signal_threshold": new_threshold}, f"Saved threshold for {device.name}.")
            if b2.button("Start alignment mode", key="tw_align"):
                _send(ctx, [device_id], {"cmd": "align", "seconds": 60}, "alignment mode")

    with radar_col:
        st.markdown("#### Radar placement")
        radars = [d for d in ctx.registry.devices if d.type == RADAR]
        if not radars:
            st.caption("No radars.")
            return
        device_id = st.selectbox("Radar", [d.id for d in radars], format_func=lambda i: ctx.registry.get(i).name, key="align_radar")
        device = ctx.registry.get(device_id)
        max_range = common.option_float(device, "max_range_m", 8.0)
        st.fragment(run_every=1.0)(_radar_live)(device_id, max_range)
        st.caption("Walk straight away from the radar: your dot should travel up the middle of the cone.")
        with st.form(f"radar_form_{device_id}"):
            r1, r2 = st.columns(2)
            height = r1.number_input("Mounting height (m)", 0.5, 3.0, common.option_float(device, "mount_height_m", 1.5), 0.1)
            new_range = r2.slider("Range limit (m)", 1.0, 8.0, max_range, 0.5,
                                  help="Targets further than this are ignored.")
            facing = st.text_input("Facing", value=str(device.option("facing") or ""), placeholder="towards the front door")
            if st.form_submit_button("Save radar placement"):
                _update_device_options(
                    ctx, device_id,
                    {"mount_height_m": round(height, 2), "max_range_m": round(new_range, 1), "facing": facing.strip()},
                    f"Saved placement for {device.name}.",
                )


def _tripwire_live(device_id: str, threshold: int) -> None:
    ctx = common.get_context()
    record = common.load_device_states(ctx).get(device_id) or {}
    state = record.get("state", {}) or {}
    level = state.get("signal")
    beam = str(state.get("beam") or "ok")
    verdict = (
        "No reading yet" if not isinstance(level, (int, float))
        else "Aligned" if level >= threshold else "Too weak: adjust the emitter"
    )
    st.markdown(
        common.beam_svg(beam)
        + common.signal_bar(level, threshold)
        + common.rows_html([("Receiver level", level if level is not None else "–"), ("Threshold", threshold), ("Verdict", verdict)]),
        unsafe_allow_html=True,
    )


def _radar_live(device_id: str, max_range: float) -> None:
    ctx = common.get_context()
    record = common.load_device_states(ctx).get(device_id) or {}
    state = record.get("state", {}) or {}
    targets = state.get("targets") if state.get("presence") else []
    st.markdown(common.radar_map_svg(targets, max_range_m=max_range, height=190), unsafe_allow_html=True)


# User ---------------------------------------------------------------------


def _user_tab(ctx: common.HubContext) -> None:
    user = ctx.hub_config.get("user", {}) or {}
    display = ctx.hub_config.get("display", {}) or {}
    with st.form("user_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Your name", value=str(user.get("name", "")))
        email = c2.text_input("Email", value=str(user.get("email", "")))
        phone = st.text_input("Phone", value=str(user.get("phone", "")), help="For alerts by phone, later.")
        refresh = st.slider("Live pages refresh every (seconds)", 1, 10, int(display.get("refresh_seconds", 2)))
        submitted = st.form_submit_button("Save profile", type="primary")
    if submitted:
        def mutate(config: dict[str, object]) -> None:
            config["user"] = {"name": name.strip(), "email": email.strip(), "phone": phone.strip()}
            config.setdefault("display", {})["refresh_seconds"] = refresh
        _save_config(ctx, mutate, "Profile saved.")
    st.caption(f"Settings file: `{ctx.hub_config_path}`")

from __future__ import annotations

import time
from collections import Counter

import streamlit as st

from app.dashboard import common, nav
from app.devices.types import CAMERA


def render() -> None:
    ctx = common.get_context()
    common.page_header("Cameras", "Live views from every camera, with what the detector sees.", "📷")
    camera_ids = ctx.camera_ids()
    if not camera_ids:
        st.info("No cameras configured yet. Add one on the Configuration page.")
        nav.link("configuration", "Open configuration", icon=":material/settings:")
        return

    view_col, zoom_col, layout_col = st.columns([2.2, 1.2, 1.0])
    with view_col:
        selected = st.multiselect(
            "Cameras",
            options=camera_ids,
            default=camera_ids,
            format_func=lambda camera_id: _camera_name(ctx, camera_id),
        )
    with zoom_col:
        zoom = st.select_slider("Zoom", options=[1.0, 1.5, 2.0, 2.5, 3.0], value=1.0, format_func=lambda z: f"{z:.1f}x")
    with layout_col:
        per_row = st.segmented_control("Per row", options=[1, 2, 3], default=2, required=True)

    st.fragment(run_every=ctx.refresh_seconds)(_live)(selected or camera_ids, zoom, int(per_row or 2))

    events_tab, info_tab = st.tabs(["Detection events", "Camera info"])
    with events_tab:
        st.fragment(run_every=max(ctx.refresh_seconds, 5.0))(_events)(selected or camera_ids)
    with info_tab:
        st.fragment(run_every=max(ctx.refresh_seconds, 5.0))(_info)(camera_ids)


def _camera_name(ctx: common.HubContext, camera_id: str) -> str:
    device = ctx.registry.get(camera_id)
    return device.name if device and device.name != camera_id else camera_id


def _live(camera_ids: list[str], zoom: float, per_row: int) -> None:
    ctx = common.get_context()
    statuses = common.load_statuses(ctx, camera_ids)
    live = sum(1 for status in statuses.values() if common.camera_is_live(status, ctx.settings))
    people_now = sum(int((status or {}).get("counts", {}).get("person", 0)) for status in statuses.values())
    hour_events = common.recent_events(
        ctx, device_ids=camera_ids, device_types=[CAMERA], since_epoch=time.time() - 3600, limit=1000
    )
    models = sorted({str(status.get("yolo_model")) for status in statuses.values() if status and status.get("yolo_model")})
    common.stat_tiles(
        [
            ("Cameras live", f"{live}/{len(camera_ids)}", "publishing frames"),
            ("People in view", people_now, "across all cameras"),
            ("Detections", len(hour_events), "in the last hour"),
            ("Detection model", "on" if models else "off", ", ".join(models) or "turn it on in Configuration"),
        ]
    )

    columns = st.columns(per_row)
    for idx, camera_id in enumerate(camera_ids):
        status = statuses.get(camera_id)
        presence = "online" if common.camera_is_live(status, ctx.settings) else ("never seen" if status is None else "offline")
        device = ctx.registry.get(camera_id)
        subtitle = " · ".join(part for part in (ctx.camera_host(camera_id), device.location if device else "") if part)
        with columns[idx % per_row], st.container(border=True):
            st.markdown(common.card_header(_camera_name(ctx, camera_id), subtitle, presence), unsafe_allow_html=True)
            image = common.load_display_image(ctx.dashboard_dir / "live" / f"{camera_id}.jpg", zoom)
            if image is not None:
                st.image(image, width="stretch")
            else:
                st.info("No live picture yet.")
            if status:
                age = common.status_age_seconds(status)
                st.markdown(
                    f"<div>{common.counts_pills(status.get('counts'))}</div>"
                    + common.rows_html(
                        [
                            ("FPS / detector FPS", f"{status.get('fps', 0)} / {status.get('detector_fps', 0)}"),
                            ("Latency", f"{status.get('latency_ms', 0)} ms"),
                            ("Updated", f"{age:.1f}s ago" if age is not None else "n/a"),
                        ]
                    ),
                    unsafe_allow_html=True,
                )


def _events(camera_ids: list[str]) -> None:
    ctx = common.get_context()
    limit_col, class_col = st.columns([1, 2])
    limit = limit_col.selectbox("Show", options=[6, 12, 24, 48], index=1, key="camera_event_limit")
    events = common.recent_events(ctx, device_ids=camera_ids, device_types=[CAMERA], limit=200)
    classes = sorted({label for event in events for label in (event.get("counts") or {})})
    chosen = class_col.multiselect("Classes", options=classes, default=[], placeholder="All classes", key="camera_event_classes")
    if chosen:
        events = [event for event in events if set(event.get("counts") or {}) & set(chosen)]
    events = events[: int(limit)]
    if not events:
        st.info("No detection events yet. Turn the detection model on in Configuration to record them.")
        return

    columns = st.columns(3)
    for idx, event in enumerate(events):
        with columns[idx % 3], st.container(border=True):
            snapshot = event.get("snapshot_path")
            snapshot_file = ctx.dashboard_dir / str(snapshot) if snapshot else None
            if snapshot_file and snapshot_file.exists():
                st.image(str(snapshot_file), width="stretch")
            counts = event.get("counts") or {}
            labels = ", ".join(common.format_detection_label(d) for d in event.get("detections") or [])
            st.markdown(
                f"**{common.esc(_camera_name(ctx, str(event.get('camera_id'))))}** · "
                f"{common.esc(common.format_time_short(event.get('event_at_utc')))}  \n"
                + (", ".join(f"{count} {name}" for name, count in counts.items()) or "detection")
            )
            if labels:
                st.caption(labels)


def _info(camera_ids: list[str]) -> None:
    ctx = common.get_context()
    statuses = common.load_statuses(ctx, camera_ids)
    day_events = common.recent_events(ctx, device_types=[CAMERA], since_epoch=time.time() - 86400, limit=5000)
    per_camera = Counter(str(event.get("device_id")) for event in day_events)
    rows = []
    for camera_id in camera_ids:
        status = statuses.get(camera_id) or {}
        device = ctx.registry.get(camera_id)
        age = common.status_age_seconds(status) if status else None
        rows.append(
            {
                "status": "🟢 live" if common.camera_is_live(status, ctx.settings) else "🔴 offline",
                "camera": _camera_name(ctx, camera_id),
                "id": camera_id,
                "ip": ctx.camera_host(camera_id),
                "location": device.location if device else "",
                "zone": device.zone if device else "",
                "stream": status.get("stream", ctx.settings.camera_stream),
                "fps": status.get("fps"),
                "detector_fps": status.get("detector_fps"),
                "latency_ms": status.get("latency_ms"),
                "model": status.get("yolo_model") or "off",
                "track": bool(status.get("tracking_enabled", False)),
                "events_24h": per_camera.get(camera_id, 0),
                "updated": f"{age:.1f}s ago" if age is not None else "never",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("Change camera IPs and the detection model on the Configuration page.")

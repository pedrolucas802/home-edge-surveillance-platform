from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.config.settings import Settings
from app.storage.dashboard_store import resolve_dashboard_dir


def main() -> None:
    settings = Settings.from_env()
    dashboard_dir = resolve_dashboard_dir(settings.dashboard_dir)
    camera_ids = list(settings.camera_hosts_map) or [settings.camera_id]

    st.set_page_config(
        page_title="Home Edge Surveillance Dashboard",
        layout="wide",
    )
    _inject_styles()
    st.title("Home Edge Surveillance Dashboard")
    st.caption(
        "Live camera snapshots and recent detections published by the running preview processes."
    )

    with st.sidebar:
        st.header("Controls")
        selected_cameras = st.multiselect(
            "Cameras",
            options=camera_ids,
            default=camera_ids,
        )
        refresh_seconds = st.slider("Refresh interval (sec)", min_value=1, max_value=10, value=2)
        auto_refresh = st.toggle("Auto refresh", value=True)
        event_limit = st.slider("Recent events", min_value=5, max_value=50, value=12)
        with st.expander("Zoom", expanded=True):
            zoom_levels = [1.0, 1.5, 2.0, 2.5, 3.0]
            zoom_by_camera = {
                camera_id: st.select_slider(
                    f"{camera_id} zoom",
                    options=zoom_levels,
                    value=1.0,
                )
                for camera_id in selected_cameras
            }
        st.markdown(
            "Run the camera preview processes first so this dashboard has live data to read."
        )

    statuses = {
        camera_id: _load_status(dashboard_dir, camera_id)
        for camera_id in selected_cameras
    }
    events = _load_recent_events(dashboard_dir, selected_cameras, limit=event_limit)

    _render_overview(statuses, events, settings)
    live_tab, events_tab, health_tab = st.tabs(["Live Monitor", "Recent Events", "System Health"])

    with live_tab:
        _render_live_cameras(dashboard_dir, statuses, settings, zoom_by_camera)
    with events_tab:
        _render_events(dashboard_dir, events)
    with health_tab:
        _render_health(statuses, settings, dashboard_dir)

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()


def _render_overview(
    statuses: dict[str, dict[str, object] | None],
    events: list[dict[str, object]],
    settings: Settings,
) -> None:
    active_cameras = 0
    stale_cameras = 0
    current_detections = 0
    for status in statuses.values():
        if not status:
            continue
        age_seconds = _status_age_seconds(status)
        if age_seconds is not None and age_seconds <= settings.dashboard_publish_interval_seconds * 3:
            active_cameras += 1
        else:
            stale_cameras += 1
        current_detections += int(status.get("detections_count", 0))

    last_event = events[0]["event_at_utc"] if events else "none yet"
    class_counter = Counter()
    for event in events:
        class_counter.update(event.get("counts", {}))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Configured Cameras", len(statuses))
    col2.metric("Active Cameras", active_cameras)
    col3.metric("Stale Cameras", stale_cameras)
    col4.metric("Detections Now", current_detections)
    col5.metric("Recent Events", len(events))
    st.caption(f"Last event: {_format_timestamp(last_event)}")

    if class_counter:
        counts_text = ", ".join(
            f"{class_name}: {count}"
            for class_name, count in class_counter.most_common()
        )
        st.caption(f"Recent event counts: {counts_text}")
    else:
        st.caption("No detection events logged yet.")


def _render_live_cameras(
    dashboard_dir: Path,
    statuses: dict[str, dict[str, object] | None],
    settings: Settings,
    zoom_by_camera: dict[str, float],
) -> None:
    if not statuses:
        st.info("No cameras selected.")
        return

    columns = st.columns(min(2, max(1, len(statuses))))
    for idx, (camera_id, status) in enumerate(statuses.items()):
        column = columns[idx % len(columns)]
        with column:
            image_path = dashboard_dir / "live" / f"{camera_id}.jpg"
            age_seconds = _status_age_seconds(status) if status else None
            stale = (
                age_seconds is None
                or age_seconds > settings.dashboard_publish_interval_seconds * 3
            )
            status_badge = "Stale" if stale else "Live"
            badge_class = "badge-stale" if stale else "badge-live"
            st.markdown(
                (
                    f"<div class='camera-card'><div class='camera-card-header'>"
                    f"<div><div class='camera-title'>{camera_id}</div>"
                    f"<div class='camera-subtitle'>{settings.resolve_camera_host(camera_id)}</div></div>"
                    f"<span class='status-badge {badge_class}'>{status_badge}</span>"
                    f"</div></div>"
                ),
                unsafe_allow_html=True,
            )

            zoom_factor = zoom_by_camera.get(camera_id, 1.0)
            display_image = _load_display_image(image_path, zoom_factor)
            if display_image is not None:
                st.image(display_image, use_container_width=True)
            else:
                st.info("No live snapshot published yet.")

            if not status:
                st.caption("Waiting for status data from this camera.")
                continue

            host = status.get("camera_host", settings.resolve_camera_host(camera_id))
            counts = status.get("counts", {})
            count_html = _format_counts_badges(counts)

            metric_cols = st.columns(4)
            metric_cols[0].metric("FPS", status.get("fps", 0.0))
            metric_cols[1].metric("Det FPS", status.get("detector_fps", 0.0))
            metric_cols[2].metric("Latency ms", status.get("latency_ms", 0.0))
            metric_cols[3].metric("Infer ms", status.get("detector_inference_ms", 0.0))
            st.caption(
                f"Host {host} | Stream {status.get('stream', settings.camera_stream)} | "
                f"Age {age_seconds:.1f}s | Zoom {zoom_factor:.1f}x"
            )
            st.markdown(
                (
                    f"<div class='counts-row'><strong>Current classes:</strong> "
                    f"{count_html}</div>"
                ),
                unsafe_allow_html=True,
            )


def _render_events(dashboard_dir: Path, events: list[dict[str, object]]) -> None:
    if not events:
        st.info("No events have been recorded yet. Start a preview with YOLO enabled.")
        return

    for event in events:
        snapshot_path = event.get("snapshot_path")
        snapshot_file = dashboard_dir / str(snapshot_path) if snapshot_path else None
        detections = event.get("detections", [])
        counts = event.get("counts", {})
        count_text = ", ".join(
            f"{class_name}: {count}" for class_name, count in counts.items()
        ) or "none"

        image_col, info_col = st.columns([1.2, 1.0])
        with image_col:
            if snapshot_file and snapshot_file.exists():
                st.image(str(snapshot_file), use_container_width=True)
        with info_col:
            st.markdown(
                f"**{event.get('camera_id', 'camera')}**  \n"
                f"{_format_timestamp(event.get('event_at_utc'))}"
            )
            st.write(f"Detections: {event.get('detections_count', 0)}")
            st.write(f"Classes: {count_text}")
            if detections:
                labels = ", ".join(
                    _format_detection_label(detection) for detection in detections
                )
                st.caption(labels)
        st.divider()


def _render_health(
    statuses: dict[str, dict[str, object] | None],
    settings: Settings,
    dashboard_dir: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for camera_id, status in statuses.items():
        if not status:
            rows.append(
                {
                    "camera": camera_id,
                    "host": settings.resolve_camera_host(camera_id),
                    "status": "no data",
                    "fps": 0.0,
                    "det_fps": 0.0,
                    "latency_ms": 0.0,
                    "age_s": None,
                    "frame_index": None,
                }
            )
            continue

        age_seconds = _status_age_seconds(status)
        rows.append(
            {
                "camera": camera_id,
                "host": status.get("camera_host", settings.resolve_camera_host(camera_id)),
                "status": "live"
                if age_seconds is not None
                and age_seconds <= settings.dashboard_publish_interval_seconds * 3
                else "stale",
                "fps": status.get("fps", 0.0),
                "det_fps": status.get("detector_fps", 0.0),
                "latency_ms": status.get("latency_ms", 0.0),
                "age_s": round(age_seconds, 1) if age_seconds is not None else None,
                "frame_index": status.get("frame_index"),
            }
        )

    st.caption(f"Dashboard directory: {dashboard_dir}")
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _load_status(dashboard_dir: Path, camera_id: str) -> dict[str, object] | None:
    path = dashboard_dir / "live" / f"{camera_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_recent_events(
    dashboard_dir: Path,
    camera_ids: list[str],
    limit: int,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for camera_id in camera_ids:
        path = dashboard_dir / "events" / f"{camera_id}.jsonl"
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        except OSError:
            continue
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    events.sort(key=lambda item: str(item.get("event_at_utc", "")), reverse=True)
    return events[:limit]


def _load_display_image(path: Path, zoom_factor: float) -> Image.Image | None:
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
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height))


def _status_age_seconds(status: dict[str, object]) -> float | None:
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


def _format_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "n/a"
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return value
    return timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _format_detection_label(detection: object) -> str:
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


def _format_counts_badges(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return "<span class='pill'>none</span>"
    parts = [
        f"<span class='pill'>{class_name}: {count}</span>"
        for class_name, count in counts.items()
    ]
    return "".join(parts)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .camera-card {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 18px;
            padding: 0.9rem 1rem 1rem 1rem;
            background: linear-gradient(180deg, rgba(248,249,252,0.96), rgba(255,255,255,0.98));
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .camera-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.6rem;
        }
        .camera-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #122033;
        }
        .camera-subtitle {
            font-size: 0.82rem;
            color: #5f6b7a;
        }
        .status-badge {
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .badge-live {
            background: #e8fff1;
            color: #0f8a4b;
        }
        .badge-stale {
            background: #fff1e8;
            color: #b45309;
        }
        .counts-row {
            margin-top: 0.5rem;
            color: #243244;
        }
        .pill {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            margin: 0.15rem 0.2rem 0.1rem 0;
            border-radius: 999px;
            background: #eef4ff;
            color: #214f98;
            font-size: 0.8rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

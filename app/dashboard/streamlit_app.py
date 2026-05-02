from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
PUBLISHERS_STATE_PATH = RUNTIME_DIR / "publishers-runtime-state.json"
PUBLISHERS_SUPERVISOR_LOG = PROJECT_ROOT / "data" / "logs" / "publishers-supervisor.log"


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
        cameras_to_restart = selected_cameras or camera_ids
        if st.button("Restart Cameras", width="stretch"):
            if not cameras_to_restart:
                st.error("Pick at least one camera before restarting.")
            else:
                ok, message = _restart_cameras(
                    settings=settings,
                    cameras=cameras_to_restart,
                )
                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
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
        _render_runtime_controls(
            settings=settings,
            selected_cameras=selected_cameras,
            all_camera_ids=camera_ids,
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


def _render_runtime_controls(
    settings: Settings,
    selected_cameras: list[str],
    all_camera_ids: list[str],
) -> None:
    st.markdown("---")
    st.subheader("Runtime Control")
    st.caption("This panel manages camera publishers started from this dashboard session.")

    managed_running, managed_state = _managed_publishers_status()
    unmanaged_worker_pids = _list_dashboard_publisher_worker_pids()
    if managed_running and managed_state:
        model = managed_state.get("yolo_model", "n/a")
        device = str(managed_state.get("yolo_device", settings.yolo_device))
        detect_text = "on" if bool(managed_state.get("yolo_detect", True)) else "off"
        pose_text = "on" if bool(managed_state.get("yolo_pose", False)) else "off"
        track_text = "on" if bool(managed_state.get("yolo_tracking", False)) else "off"
        tracker_name = str(managed_state.get("yolo_tracker", settings.yolo_tracker))
        raw_cameras = managed_state.get("cameras", [])
        camera_text = (
            ", ".join(str(camera_id) for camera_id in raw_cameras)
            if isinstance(raw_cameras, list)
            else "all configured"
        ) or "all configured"
        started_at = _format_timestamp(managed_state.get("started_at_utc"))
        st.success(
            f"Managed publishers are running (PID {managed_state.get('pid')})."
        )
        st.caption(
            f"Model: {model} | Device: {device} | Detect: {detect_text} | Pose: {pose_text} | "
            f"Track: {track_text} ({tracker_name}) | Cameras: {camera_text} | Started: {started_at}"
        )
    else:
        if unmanaged_worker_pids:
            pid_text = ", ".join(str(pid) for pid in unmanaged_worker_pids)
            st.warning(
                "Publishers are running outside managed mode "
                f"(worker PIDs: {pid_text}). Stop Model can still stop them."
            )
        else:
            st.info("Managed publishers are stopped.")

    model_options = _discover_model_options(settings.yolo_model)
    active_model = str(
        managed_state.get("yolo_model", settings.yolo_model)
        if managed_state
        else settings.yolo_model
    )
    if active_model not in model_options:
        model_options.insert(0, active_model)
    model_index = model_options.index(active_model) if model_options else 0

    selected_model = st.selectbox(
        "YOLO model",
        options=model_options,
        index=model_index,
        help="Pick one of the local model checkpoints, or set a custom path below.",
    )
    custom_model = st.text_input(
        "Custom model path (optional)",
        value="",
        placeholder="models/home-surveillance-yolo26m-best.pt",
    ).strip()
    effective_model = custom_model or selected_model
    default_device = str(
        managed_state.get("yolo_device", settings.yolo_device)
        if managed_state
        else settings.yolo_device
    ).strip().lower()
    device_options = ["mps", "cpu", "auto", "cuda"]
    if default_device not in device_options:
        device_options.insert(0, default_device)
    yolo_device = st.selectbox(
        "YOLO device",
        options=device_options,
        index=device_options.index(default_device) if default_device in device_options else 0,
        help="Use mps on Apple Silicon to run inference on the Mac GPU.",
    )

    detect_enabled = st.toggle(
        "Detect",
        value=bool(managed_state.get("yolo_detect", settings.yolo_enabled))
        if managed_state
        else settings.yolo_enabled,
        help="Draw class boxes and labels, and publish class-based events.",
    )
    pose_enabled = st.toggle(
        "Pose",
        value=bool(managed_state.get("yolo_pose", settings.yolo_pose_enabled))
        if managed_state
        else settings.yolo_pose_enabled,
        help="Draw keypoint skeletons when the selected model supports pose outputs.",
    )
    tracking_enabled = st.toggle(
        "Track",
        value=bool(managed_state.get("yolo_tracking", settings.yolo_tracking))
        if managed_state
        else settings.yolo_tracking,
        help="Attach persistent IDs to detections and suppress duplicate tracked events.",
    )
    tracker_options = ["bytetrack.yaml", "botsort.yaml"]
    default_tracker = str(
        managed_state.get("yolo_tracker", settings.yolo_tracker)
        if managed_state
        else settings.yolo_tracker
    ).strip() or settings.yolo_tracker
    if default_tracker not in tracker_options:
        tracker_options.insert(0, default_tracker)
    selected_tracker = st.selectbox(
        "Tracker",
        options=tracker_options,
        index=tracker_options.index(default_tracker) if default_tracker in tracker_options else 0,
        help="Tracking uses detector boxes plus a tracker algorithm, not a separate YOLO checkpoint.",
    )
    if tracking_enabled and not detect_enabled:
        st.caption("Tracking requires detections. Detect will be enabled automatically at runtime.")

    default_classes = (
        str(managed_state.get("yolo_classes", "all"))
        if managed_state
        else (
            ",".join(settings.yolo_classes)
            if settings.yolo_classes
            else "car,person,cat"
        )
    )
    yolo_classes = st.text_input(
        "YOLO classes",
        value=default_classes,
        help="Use all for no class filter, or comma-separated names like car,person,cat.",
    ).strip() or "all"
    yolo_confidence = st.slider(
        "YOLO confidence",
        min_value=0.05,
        max_value=0.95,
        value=float(managed_state.get("yolo_confidence", settings.yolo_confidence))
        if managed_state
        else float(settings.yolo_confidence),
        step=0.01,
    )
    infer_every_n = st.slider(
        "Infer every N frames",
        min_value=1,
        max_value=6,
        value=int(managed_state.get("yolo_infer_every_n", settings.yolo_infer_every_n))
        if managed_state
        else int(settings.yolo_infer_every_n),
    )
    launch_delay = st.slider("Camera launch delay (sec)", min_value=1, max_value=8, value=3)

    cameras_to_start = selected_cameras or all_camera_ids
    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("Start Model", width="stretch", type="primary"):
            if not cameras_to_start:
                st.error("Pick at least one camera before starting publishers.")
            else:
                ok, message = _start_managed_publishers(
                    cameras=cameras_to_start,
                    yolo_model=effective_model,
                    yolo_device=yolo_device,
                    yolo_detect_enabled=detect_enabled,
                    yolo_pose_enabled=pose_enabled,
                    yolo_classes=yolo_classes,
                    yolo_confidence=yolo_confidence,
                    yolo_infer_every_n=infer_every_n,
                    yolo_tracking=tracking_enabled,
                    yolo_tracker=selected_tracker,
                    launch_delay=launch_delay,
                )
                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    with stop_col:
        if st.button("Stop Model", width="stretch"):
            ok, message = _stop_managed_publishers()
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    st.caption(f"Supervisor log: {PUBLISHERS_SUPERVISOR_LOG}")


def _discover_model_options(default_model: str) -> list[str]:
    candidates: set[str] = {default_model}
    patterns = ("yolo*.pt", "models/**/*.pt")
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                relative = path.as_posix()
            candidates.add(relative)

    return sorted(candidates)


def _managed_publishers_status() -> tuple[bool, dict[str, object] | None]:
    state = _load_managed_publishers_state()
    if not state:
        return False, None

    pid = state.get("pid")
    if not isinstance(pid, int):
        _clear_managed_publishers_state()
        return False, None

    if _pid_is_running(pid):
        return True, state

    _clear_managed_publishers_state()
    return False, None


def _load_managed_publishers_state() -> dict[str, object] | None:
    if not PUBLISHERS_STATE_PATH.exists():
        return None
    try:
        state = json.loads(PUBLISHERS_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    return state


def _save_managed_publishers_state(state: dict[str, object]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    PUBLISHERS_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=True),
        encoding="utf-8",
    )


def _clear_managed_publishers_state() -> None:
    try:
        PUBLISHERS_STATE_PATH.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_managed_publishers(
    cameras: list[str],
    yolo_model: str,
    yolo_device: str,
    yolo_detect_enabled: bool,
    yolo_pose_enabled: bool,
    yolo_classes: str,
    yolo_confidence: float,
    yolo_infer_every_n: int,
    yolo_tracking: bool,
    yolo_tracker: str,
    launch_delay: int,
) -> tuple[bool, str]:
    running, state = _managed_publishers_status()
    did_restart = False
    unmanaged_worker_pids = _list_dashboard_publisher_worker_pids()
    if (running and state) or unmanaged_worker_pids:
        stop_ok, stop_message = _stop_managed_publishers()
        if not stop_ok:
            return (
                False,
                "Could not restart camera publishers before starting the model. "
                f"{stop_message}"
            )
        did_restart = True

    command = [
        "bash",
        "scripts/run_dashboard_stack.sh",
        "--publishers-only",
        "--launch-delay",
        str(launch_delay),
    ]
    if cameras:
        command.append("--cameras")
        command.extend(cameras)

    effective_detect_enabled = bool(yolo_detect_enabled or yolo_tracking)
    effective_yolo_enabled = bool(effective_detect_enabled or yolo_pose_enabled)
    env = os.environ.copy()
    env.update(
        {
            "CAMERA_PUBLISHERS_ENABLED": "true",
            "YOLO_ENABLED": "true" if effective_yolo_enabled else "false",
            "YOLO_POSE_ENABLED": "true" if yolo_pose_enabled else "false",
            "YOLO_MODEL": yolo_model,
            "YOLO_DEVICE": yolo_device,
            "YOLO_CLASSES": yolo_classes,
            "YOLO_CONFIDENCE": f"{yolo_confidence:.2f}",
            "YOLO_INFER_EVERY_N": str(max(yolo_infer_every_n, 1)),
            "YOLO_TRACKING": "true" if yolo_tracking else "false",
            "YOLO_TRACKER": yolo_tracker,
        }
    )

    PUBLISHERS_SUPERVISOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with PUBLISHERS_SUPERVISOR_LOG.open("ab") as log_handle:
            log_handle.write(
                (
                    f"\n[{datetime.now(timezone.utc).isoformat()}] "
                    f"Starting publishers with model={yolo_model} cameras={','.join(cameras)}\n"
                ).encode("utf-8")
            )
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        return False, f"Failed to start publishers: {exc}"

    time.sleep(1.0)
    if process.poll() is not None:
        return (
            False,
            "Publisher supervisor exited during startup. Check data/logs/publishers-supervisor.log and camera logs.",
        )

    _save_managed_publishers_state(
        {
            "pid": process.pid,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "yolo_model": yolo_model,
            "yolo_device": yolo_device,
            "yolo_detect": effective_detect_enabled,
            "yolo_pose": bool(yolo_pose_enabled),
            "yolo_classes": yolo_classes,
            "yolo_confidence": round(yolo_confidence, 2),
            "yolo_infer_every_n": int(max(yolo_infer_every_n, 1)),
            "yolo_tracking": bool(yolo_tracking),
            "yolo_tracker": yolo_tracker,
            "cameras": list(cameras),
            "launch_delay_seconds": launch_delay,
            "command": command,
        }
    )
    feature_text = (
        f"detect={'on' if effective_detect_enabled else 'off'}, "
        f"pose={'on' if yolo_pose_enabled else 'off'}, "
        f"track={'on' if yolo_tracking else 'off'}"
    )
    action_text = "Restarted cameras and started" if did_restart else "Started"
    return (
        True,
        f"{action_text} model {yolo_model} for {', '.join(cameras)} ({feature_text}).",
    )


def _restart_cameras(
    settings: Settings,
    cameras: list[str],
) -> tuple[bool, str]:
    managed_state = _load_managed_publishers_state() or {}
    default_classes = ",".join(settings.yolo_classes) if settings.yolo_classes else "all"

    model = str(managed_state.get("yolo_model", settings.yolo_model))
    device = str(managed_state.get("yolo_device", settings.yolo_device)).strip().lower() or settings.yolo_device
    detect_enabled = bool(managed_state.get("yolo_detect", settings.yolo_enabled))
    pose_enabled = bool(managed_state.get("yolo_pose", settings.yolo_pose_enabled))
    tracking_enabled = bool(managed_state.get("yolo_tracking", settings.yolo_tracking))
    tracker = str(managed_state.get("yolo_tracker", settings.yolo_tracker)).strip() or settings.yolo_tracker

    raw_classes = managed_state.get("yolo_classes", default_classes)
    yolo_classes = str(raw_classes).strip() or default_classes

    try:
        yolo_confidence = float(managed_state.get("yolo_confidence", settings.yolo_confidence))
    except (TypeError, ValueError):
        yolo_confidence = float(settings.yolo_confidence)

    try:
        infer_every_n = int(managed_state.get("yolo_infer_every_n", settings.yolo_infer_every_n))
    except (TypeError, ValueError):
        infer_every_n = int(settings.yolo_infer_every_n)

    try:
        launch_delay = int(managed_state.get("launch_delay_seconds", 3))
    except (TypeError, ValueError):
        launch_delay = 3

    ok, message = _start_managed_publishers(
        cameras=cameras,
        yolo_model=model,
        yolo_device=device,
        yolo_detect_enabled=detect_enabled,
        yolo_pose_enabled=pose_enabled,
        yolo_classes=yolo_classes,
        yolo_confidence=max(0.05, min(0.95, yolo_confidence)),
        yolo_infer_every_n=max(1, infer_every_n),
        yolo_tracking=tracking_enabled,
        yolo_tracker=tracker,
        launch_delay=max(1, launch_delay),
    )
    if not ok:
        return False, message
    return True, f"Restarted cameras ({', '.join(cameras)}). {message}"


def _stop_managed_publishers() -> tuple[bool, str]:
    running, state = _managed_publishers_status()
    unmanaged_worker_pids = _list_dashboard_publisher_worker_pids()
    if not state and not unmanaged_worker_pids:
        return False, "No managed or unmanaged publisher process found."

    if state:
        pid = state.get("pid")
        if not isinstance(pid, int):
            _clear_managed_publishers_state()
            return False, "Runtime state was invalid and has been reset."
        if running:
            if not _terminate_managed_process_group(pid):
                return False, "Failed to stop managed publishers. Please stop them manually from the terminal."
        _clear_managed_publishers_state()

    worker_pids = _list_dashboard_publisher_worker_pids()
    if worker_pids and not _terminate_processes(worker_pids):
        pid_text = ", ".join(str(pid) for pid in worker_pids)
        return (
            False,
            "Failed to stop publisher workers. "
            f"Please stop these PIDs manually from the terminal: {pid_text}."
        )

    stubborn_workers = _list_dashboard_publisher_worker_pids()
    if stubborn_workers:
        pid_text = ", ".join(str(pid) for pid in stubborn_workers)
        return False, f"Publisher workers are still running: {pid_text}. Please stop them manually."

    _clear_managed_publishers_state()
    return True, "Stopped publishers."


def _terminate_managed_process_group(pid: int) -> bool:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    deadline = time.time() + 8.0
    while time.time() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    time.sleep(0.2)
    return not _pid_is_running(pid)


def _terminate_processes(pids: list[int]) -> bool:
    target_pids = sorted({pid for pid in pids if isinstance(pid, int) and pid > 0})
    if not target_pids:
        return True

    for pid in target_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            return False

    deadline = time.time() + 8.0
    while time.time() < deadline:
        if all(not _pid_is_running(pid) for pid in target_pids):
            return True
        time.sleep(0.2)

    for pid in target_pids:
        if not _pid_is_running(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            return False

    time.sleep(0.2)
    return all(not _pid_is_running(pid) for pid in target_pids)


def _list_dashboard_publisher_worker_pids() -> list[int]:
    try:
        process_list = subprocess.run(
            ["ps", "-axo", "pid=,args="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []

    if process_list.returncode != 0:
        return []

    current_pid = os.getpid()
    pids: list[int] = []
    for raw_line in process_list.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if "--dashboard-publish" not in command:
            continue
        if "app.main" not in command:
            continue
        pids.append(pid)

    return sorted(set(pids))


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
                st.image(display_image, width="stretch")
            else:
                st.info("No live snapshot published yet.")

            if not status:
                st.caption("Waiting for status data from this camera.")
                continue

            host = status.get("camera_host", settings.resolve_camera_host(camera_id))
            counts = status.get("counts", {})
            count_html = _format_counts_badges(counts)
            model_name = status.get("yolo_model")
            device_name = status.get("yolo_device")
            detect_on = bool(status.get("yolo_detect_enabled", status.get("yolo_enabled", False)))
            pose_on = bool(status.get("yolo_pose_enabled", False))
            track_on = bool(status.get("tracking_enabled", False))
            model_suffix = (
                (
                    f" | Model {model_name} | Device {device_name or 'n/a'} | Detect {'on' if detect_on else 'off'}"
                    f" | Pose {'on' if pose_on else 'off'} | Track {'on' if track_on else 'off'}"
                )
                if isinstance(model_name, str) and model_name
                else ""
            )

            metric_cols = st.columns(4)
            metric_cols[0].metric("FPS", status.get("fps", 0.0))
            metric_cols[1].metric("Det FPS", status.get("detector_fps", 0.0))
            metric_cols[2].metric("Latency ms", status.get("latency_ms", 0.0))
            metric_cols[3].metric("Infer ms", status.get("detector_inference_ms", 0.0))
            st.caption(
                f"Host {host} | Stream {status.get('stream', settings.camera_stream)} | "
                f"Age {age_seconds:.1f}s | Zoom {zoom_factor:.1f}x{model_suffix}"
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
                st.image(str(snapshot_file), width="stretch")
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
                    "model": None,
                    "detect": False,
                    "pose": False,
                    "track": False,
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
                "model": status.get("yolo_model"),
                "detect": bool(status.get("yolo_detect_enabled", status.get("yolo_enabled", False))),
                "pose": bool(status.get("yolo_pose_enabled", False)),
                "track": bool(status.get("tracking_enabled", False)),
            }
        )

    st.caption(f"Dashboard directory: {dashboard_dir}")
    st.dataframe(rows, width="stretch", hide_index=True)


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

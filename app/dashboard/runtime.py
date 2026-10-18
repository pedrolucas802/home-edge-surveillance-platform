"""Start, stop and inspect the camera publisher processes (YOLO on the RTSP feeds)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
PUBLISHERS_STATE_PATH = RUNTIME_DIR / "publishers-runtime-state.json"
PUBLISHERS_SUPERVISOR_LOG = PROJECT_ROOT / "data" / "logs" / "publishers-supervisor.log"


def discover_model_options(default_model: str) -> list[str]:
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


def managed_publishers_status() -> tuple[bool, dict[str, object] | None]:
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


def start_managed_publishers(
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
    running, state = managed_publishers_status()
    did_restart = False
    unmanaged_worker_pids = list_dashboard_publisher_worker_pids()
    if (running and state) or unmanaged_worker_pids:
        stop_ok, stop_message = stop_managed_publishers()
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


def restart_cameras(
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

    ok, message = start_managed_publishers(
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


def stop_managed_publishers() -> tuple[bool, str]:
    running, state = managed_publishers_status()
    unmanaged_worker_pids = list_dashboard_publisher_worker_pids()
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

    worker_pids = list_dashboard_publisher_worker_pids()
    if worker_pids and not _terminate_processes(worker_pids):
        pid_text = ", ".join(str(pid) for pid in worker_pids)
        return (
            False,
            "Failed to stop publisher workers. "
            f"Please stop these PIDs manually from the terminal: {pid_text}."
        )

    stubborn_workers = list_dashboard_publisher_worker_pids()
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


def list_dashboard_publisher_worker_pids() -> list[int]:
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

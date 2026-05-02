from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote

_DOTENV_LOADED = False


def _load_dotenv_if_present() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    candidate_paths: list[Path] = []
    explicit_path = os.getenv("SURVEILLANCE_ENV_FILE", "").strip() or os.getenv(
        "PTZ_ENV_FILE", ""
    ).strip()
    if explicit_path:
        candidate_paths.append(Path(explicit_path).expanduser())

    candidate_paths.append(Path.cwd() / ".env")
    project_root = Path(__file__).resolve().parents[2]
    candidate_paths.append(project_root / ".env")

    seen: set[Path] = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists():
            continue
        _read_env_file(path)
        break

    _DOTENV_LOADED = True


def _read_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default

    normalized = raw.strip().lower()
    if normalized in {"all", "*"}:
        return ()

    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or default


def _get_mapping(name: str) -> tuple[tuple[str, str], ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()

    pairs: list[tuple[str, str]] = []
    for part in raw.split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            pairs.append((key, value))
    return tuple(pairs)


@dataclass(slots=True, frozen=True)
class Settings:
    camera_id: str
    camera_host: str
    camera_hosts: tuple[tuple[str, str], ...]
    camera_port: int
    camera_user: str
    camera_pass: str
    camera_stream: str
    rtsp_url_override: str
    rtsp_transport: str
    ffmpeg_bin: str
    ffmpeg_loglevel: str
    width: int
    height: int
    output_fps: int
    source_fps: int
    use_hwaccel: bool
    reconnect_delay_seconds: float
    read_timeout_seconds: float
    startup_timeout_seconds: float
    startup_attempts: int
    window_name: str
    window_scale: float
    dashboard_enabled: bool
    dashboard_dir: str
    dashboard_publish_interval_seconds: float
    dashboard_event_cooldown_seconds: float
    yolo_enabled: bool
    yolo_pose_enabled: bool
    yolo_model: str
    yolo_device: str
    yolo_confidence: float
    yolo_iou: float
    yolo_imgsz: int
    yolo_classes: tuple[str, ...]
    yolo_infer_every_n: int
    yolo_tracking: bool
    yolo_tracker: str
    yolo_track_history: int

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must not be empty.")
        if not self.camera_host:
            raise ValueError("camera_host must not be empty.")
        if self.camera_port < 1:
            raise ValueError("camera_port must be greater than 0.")
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be greater than 0.")
        if self.output_fps < 1:
            raise ValueError("output_fps must be at least 1.")
        if self.source_fps < 1:
            raise ValueError("source_fps must be at least 1.")
        if self.reconnect_delay_seconds <= 0:
            raise ValueError("reconnect_delay_seconds must be greater than 0.")
        if self.read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be greater than 0.")
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be greater than 0.")
        if self.startup_attempts < 1:
            raise ValueError("startup_attempts must be at least 1.")
        if self.window_scale <= 0:
            raise ValueError("window_scale must be greater than 0.")
        if self.dashboard_publish_interval_seconds <= 0:
            raise ValueError("dashboard_publish_interval_seconds must be greater than 0.")
        if self.dashboard_event_cooldown_seconds <= 0:
            raise ValueError("dashboard_event_cooldown_seconds must be greater than 0.")
        if self.rtsp_transport not in {"udp", "tcp"}:
            raise ValueError("rtsp_transport must be either 'udp' or 'tcp'.")
        if self.yolo_imgsz < 32:
            raise ValueError("yolo_imgsz must be at least 32.")
        if self.yolo_infer_every_n < 1:
            raise ValueError("yolo_infer_every_n must be at least 1.")
        if self.yolo_track_history < 1:
            raise ValueError("yolo_track_history must be at least 1.")

    @property
    def rtsp_url(self) -> str:
        if self.rtsp_url_override:
            return self.rtsp_url_override

        safe_user = quote(self.camera_user, safe="")
        safe_pass = quote(self.camera_pass, safe="")
        return (
            f"rtsp://{safe_user}:{safe_pass}"
            f"@{self.camera_host}:{self.camera_port}/{self.camera_stream}"
        )

    @property
    def camera_hosts_map(self) -> dict[str, str]:
        return dict(self.camera_hosts)

    @property
    def effective_output_fps(self) -> int:
        return min(self.output_fps, self.source_fps)

    def resolve_camera_host(self, camera_id: str | None = None) -> str:
        selected_id = camera_id or self.camera_id
        hosts = self.camera_hosts_map
        if not hosts:
            return self.camera_host
        if selected_id in hosts:
            return hosts[selected_id]
        if selected_id == self.camera_id:
            return self.camera_host
        raise ValueError(
            f"Unknown camera id '{selected_id}'. Available cameras: {', '.join(sorted(hosts))}"
        )

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv_if_present()
        camera_hosts = _get_mapping("CAMERA_HOSTS")
        camera_id = os.getenv("CAMERA_ID", "cam1")
        fallback_host = os.getenv("CAM_HOST", "192.168.0.90")
        camera_host = dict(camera_hosts).get(camera_id, fallback_host)
        return cls(
            camera_id=camera_id,
            camera_host=camera_host,
            camera_hosts=camera_hosts,
            camera_port=_get_int("CAM_PORT", 554),
            camera_user=os.getenv("CAM_USER", "admin"),
            camera_pass=os.getenv("CAM_PASS", ""),
            camera_stream=os.getenv("CAM_STREAM", "onvif2"),
            rtsp_url_override=os.getenv("RTSP_URL", ""),
            rtsp_transport=os.getenv("RTSP_TRANSPORT", "udp"),
            ffmpeg_bin=os.getenv("FFMPEG_BIN", "/opt/homebrew/bin/ffmpeg"),
            ffmpeg_loglevel=os.getenv("FFMPEG_LOGLEVEL", "error"),
            width=_get_int("FRAME_WIDTH", 640),
            height=_get_int("FRAME_HEIGHT", 360),
            output_fps=_get_int("FRAME_FPS", 15),
            source_fps=_get_int("SOURCE_FPS", 15),
            use_hwaccel=_get_bool("USE_HWACCEL", True),
            reconnect_delay_seconds=_get_float("RECONNECT_DELAY_SECONDS", 1.0),
            read_timeout_seconds=_get_float("READ_TIMEOUT_SECONDS", 2.0),
            startup_timeout_seconds=_get_float("STARTUP_TIMEOUT_SECONDS", 8.0),
            startup_attempts=_get_int("STARTUP_ATTEMPTS", 3),
            window_name=os.getenv("WINDOW_NAME", "Home Edge Surveillance Preview"),
            window_scale=_get_float("WINDOW_SCALE", 2.0),
            dashboard_enabled=_get_bool("DASHBOARD_ENABLED", True),
            dashboard_dir=os.getenv("DASHBOARD_DIR", "data/dashboard"),
            dashboard_publish_interval_seconds=_get_float("DASHBOARD_PUBLISH_INTERVAL_SECONDS", 1.0),
            dashboard_event_cooldown_seconds=_get_float("DASHBOARD_EVENT_COOLDOWN_SECONDS", 3.0),
            yolo_enabled=_get_bool("YOLO_ENABLED", False),
            yolo_pose_enabled=_get_bool("YOLO_POSE_ENABLED", False),
            yolo_model=os.getenv("YOLO_MODEL", "models/home-surveillance-yolo26m-best.pt"),
            yolo_device=os.getenv("YOLO_DEVICE", "mps"),
            yolo_confidence=_get_float("YOLO_CONFIDENCE", 0.35),
            yolo_iou=_get_float("YOLO_IOU", 0.45),
            yolo_imgsz=_get_int("YOLO_IMGSZ", 960),
            yolo_classes=_get_csv("YOLO_CLASSES", ("person", "cat", "dog", "car", "car_plate")),
            yolo_infer_every_n=_get_int("YOLO_INFER_EVERY_N", 1),
            yolo_tracking=_get_bool("YOLO_TRACKING", False),
            yolo_tracker=os.getenv("YOLO_TRACKER", "bytetrack.yaml"),
            yolo_track_history=_get_int("YOLO_TRACK_HISTORY", 30),
        )

    def with_overrides(self, **overrides: object) -> "Settings":
        valid_overrides = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **valid_overrides)

    def ensure_credentials(self) -> None:
        if not self.rtsp_url_override and not self.camera_pass:
            raise ValueError(
                "CAM_PASS is required unless RTSP_URL is provided. "
                "Set CAM_PASS, or pass --password, or pass --rtsp-url."
            )

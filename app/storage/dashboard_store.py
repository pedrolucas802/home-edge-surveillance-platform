from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from app.config.settings import Settings
from app.detection.async_runner import DetectorStats
from app.detection.yolo_detector import Detection
from app.stream.reader import FramePacket


def resolve_dashboard_dir(raw_path: str) -> Path:
    base_dir = Path(raw_path).expanduser()
    if base_dir.is_absolute():
        return base_dir
    project_root = Path(__file__).resolve().parents[2]
    return project_root / base_dir


class DashboardStore:
    """Persist live camera artifacts for the Streamlit dashboard."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger.getChild("dashboard_store")
        self.base_dir = resolve_dashboard_dir(settings.dashboard_dir)
        self.live_dir = self.base_dir / "live"
        self.events_dir = self.base_dir / "events"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        (self.snapshots_dir / settings.camera_id).mkdir(parents=True, exist_ok=True)
        self._last_publish_at = 0.0
        self._last_event_at = 0.0
        self._last_event_signature: tuple[tuple[str, int | None], ...] | None = None

    def publish(
        self,
        packet: FramePacket,
        frame: np.ndarray,
        fps: float,
        detections: list[Detection],
        detector_stats: DetectorStats | None = None,
    ) -> None:
        now_monotonic = time.monotonic()
        should_publish_live = (
            bool(detections)
            or now_monotonic - self._last_publish_at >= self.settings.dashboard_publish_interval_seconds
        )

        event_record: dict[str, object] | None = None
        if detections and self._should_log_event(now_monotonic, detections):
            event_record = self._build_event_record(packet, detections)
            snapshot_rel_path = self._write_event_snapshot(event_record["event_id"], frame)
            event_record["snapshot_path"] = snapshot_rel_path
            self._append_jsonl(
                self.events_dir / f"{self.settings.camera_id}.jsonl",
                event_record,
            )
            self._last_event_at = now_monotonic
            self._last_event_signature = self._build_event_signature(detections)
            should_publish_live = True

        if not should_publish_live:
            return

        status_record = self._build_status_record(packet, fps, detections, detector_stats)
        self._write_json_atomic(
            self.live_dir / f"{self.settings.camera_id}.json",
            status_record,
        )
        self._write_jpeg_atomic(
            self.live_dir / f"{self.settings.camera_id}.jpg",
            frame,
        )
        self._last_publish_at = now_monotonic

    def _build_status_record(
        self,
        packet: FramePacket,
        fps: float,
        detections: list[Detection],
        detector_stats: DetectorStats | None,
    ) -> dict[str, object]:
        labels = [detection.class_name for detection in detections]
        counts = dict(sorted(Counter(labels).items()))
        return {
            "camera_id": self.settings.camera_id,
            "camera_host": self.settings.camera_host,
            "stream": self.settings.camera_stream,
            "updated_at_utc": _utc_now_iso(),
            "frame_index": packet.index,
            "fps": round(fps, 1),
            "latency_ms": round(
                (time.monotonic() - packet.captured_at_monotonic) * 1_000.0,
                1,
            ),
            "detector_fps": round(detector_stats.inference_fps, 1) if detector_stats else 0.0,
            "detector_inference_ms": round(detector_stats.last_inference_ms, 1) if detector_stats else 0.0,
            "detector_processed_frames": detector_stats.processed_frames if detector_stats else 0,
            "detections_count": len(detections),
            "counts": counts,
            "detections": [self._serialize_detection(detection) for detection in detections],
            "tracking_enabled": self.settings.yolo_tracking,
        }

    def _build_event_record(
        self,
        packet: FramePacket,
        detections: list[Detection],
    ) -> dict[str, object]:
        labels = [detection.class_name for detection in detections]
        counts = dict(sorted(Counter(labels).items()))
        event_timestamp = _utc_now_iso()
        return {
            "event_id": f"{self.settings.camera_id}-{packet.index}-{int(time.time() * 1000)}",
            "camera_id": self.settings.camera_id,
            "camera_host": self.settings.camera_host,
            "event_at_utc": event_timestamp,
            "frame_index": packet.index,
            "detections_count": len(detections),
            "counts": counts,
            "detections": [self._serialize_detection(detection) for detection in detections],
        }

    def _write_event_snapshot(self, event_id: object, frame: np.ndarray) -> str:
        relative_path = Path("snapshots") / self.settings.camera_id / f"{event_id}.jpg"
        self._write_jpeg_atomic(self.base_dir / relative_path, frame)
        return relative_path.as_posix()

    def _should_log_event(
        self,
        now_monotonic: float,
        detections: list[Detection],
    ) -> bool:
        signature = self._build_event_signature(detections)
        if self._last_event_signature != signature:
            return True
        return (
            now_monotonic - self._last_event_at
            >= self.settings.dashboard_event_cooldown_seconds
        )

    @staticmethod
    def _build_event_signature(
        detections: list[Detection],
    ) -> tuple[tuple[str, int | None], ...]:
        return tuple(
            sorted((detection.class_name, detection.track_id) for detection in detections)
        )

    @staticmethod
    def _serialize_detection(detection: Detection) -> dict[str, object]:
        return {
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": round(detection.confidence, 3),
            "bbox_xyxy": list(detection.bbox_xyxy),
            "track_id": detection.track_id,
        }

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    @staticmethod
    def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(json.dumps(record, ensure_ascii=True), encoding="utf-8")
        temp_path.replace(path)

    @staticmethod
    def _write_jpeg_atomic(path: Path, frame: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            return
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_bytes(encoded.tobytes())
        temp_path.replace(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

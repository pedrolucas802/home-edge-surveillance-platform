from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from app.detection.yolo_detector import BaseDetector, Detection


@dataclass(slots=True)
class DetectionSnapshot:
    frame_index: int
    detections: list[Detection]
    processed_at_monotonic: float


@dataclass(slots=True)
class DetectorStats:
    inference_fps: float = 0.0
    last_inference_ms: float = 0.0
    processed_frames: int = 0
    last_frame_index: int | None = None


class AsyncDetectorRunner:
    """Run inference on the newest submitted frame without blocking preview."""

    def __init__(self, detector: BaseDetector, logger: logging.Logger) -> None:
        self.detector = detector
        self.logger = logger.getChild("async_detector")
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._pending_frame: np.ndarray | None = None
        self._pending_frame_index: int | None = None
        self._latest_result: DetectionSnapshot | None = None
        self._stats = DetectorStats()

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run,
            name="yolo-async-detector",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def submit(self, frame: np.ndarray, frame_index: int) -> None:
        with self._condition:
            self._pending_frame = frame
            self._pending_frame_index = frame_index
            self._condition.notify()

    def get_latest(self) -> DetectionSnapshot | None:
        with self._condition:
            return self._latest_result

    def get_stats(self) -> DetectorStats:
        with self._condition:
            return DetectorStats(
                inference_fps=self._stats.inference_fps,
                last_inference_ms=self._stats.last_inference_ms,
                processed_frames=self._stats.processed_frames,
                last_frame_index=self._stats.last_frame_index,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while (
                    not self._stop_event.is_set()
                    and self._pending_frame is None
                    and self._pending_frame_index is None
                ):
                    self._condition.wait(timeout=0.2)

                if self._stop_event.is_set():
                    return

                frame = self._pending_frame
                frame_index = self._pending_frame_index
                self._pending_frame = None
                self._pending_frame_index = None

            if frame is None or frame_index is None:
                continue

            started_at = time.monotonic()
            try:
                detections = self.detector.infer(frame)
            except Exception as exc:
                self.logger.exception("YOLO inference failed: %s", exc)
                detections = []
            finished_at = time.monotonic()
            inference_ms = (finished_at - started_at) * 1_000.0
            inference_fps = 0.0 if inference_ms <= 0 else 1_000.0 / inference_ms

            snapshot = DetectionSnapshot(
                frame_index=frame_index,
                detections=detections,
                processed_at_monotonic=finished_at,
            )
            with self._condition:
                self._latest_result = snapshot
                self._stats = DetectorStats(
                    inference_fps=inference_fps,
                    last_inference_ms=inference_ms,
                    processed_frames=self._stats.processed_frames + 1,
                    last_frame_index=frame_index,
                )

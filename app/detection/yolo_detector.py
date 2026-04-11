from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Protocol

import cv2
import numpy as np


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    track_id: int | None = None


class BaseDetector(Protocol):
    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Run inference for a single frame."""

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> None:
        """Draw detections onto an existing frame."""


class YoloDetector:
    """Ultralytics YOLO adapter. Keep it isolated from stream ingestion details."""

    def __init__(
        self,
        model_name: str = "models/home-surveillance-yolo26m-best.pt",
        device: str = "cpu",
        confidence: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 960,
        target_classes: tuple[str, ...] = (),
        tracking: bool = False,
        tracker: str = "bytetrack.yaml",
        track_history_length: int = 30,
    ) -> None:
        from ultralytics import YOLO

        if tracking:
            self._ensure_tracking_dependencies()

        self.model = YOLO(model_name)
        self.device = self._resolve_device(device)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.names = self._normalize_names(self.model.names)
        self.use_class_filter = bool(target_classes)
        self.target_class_ids = self._resolve_target_class_ids(target_classes)
        self.tracking = tracking
        self.tracker = tracker
        self.track_history_length = track_history_length
        self.track_history: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.track_history_length)
        )

    def infer(self, frame: np.ndarray) -> list[Detection]:
        if self.use_class_filter and not self.target_class_ids:
            return []

        inference_kwargs = dict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.target_class_ids if self.use_class_filter else None,
            device=self.device,
            verbose=False,
        )
        if self.tracking:
            results = self.model.track(
                persist=True,
                tracker=self.tracker,
                **inference_kwargs,
            )
        else:
            results = self.model.predict(**inference_kwargs)
        result = results[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        track_ids: list[int | None] = [None] * len(result.boxes)
        if getattr(result.boxes, "is_track", False) and result.boxes.id is not None:
            track_ids = [int(track_id) for track_id in result.boxes.id.int().cpu().tolist()]

        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = track_ids[idx] if idx < len(track_ids) else None
            if track_id is not None:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.track_history[track_id].append((center_x, center_y))
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.names.get(class_id, str(class_id)),
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                    track_id=track_id,
                )
            )
        return detections

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> None:
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.track_id is not None:
                label = f"{detection.class_name} #{detection.track_id} {detection.confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            if detection.track_id is None:
                continue

            points = list(self.track_history.get(detection.track_id, ()))
            if len(points) < 2:
                continue

            trail = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame,
                [trail],
                isClosed=False,
                color=(180, 180, 180),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    @staticmethod
    def _normalize_names(names: dict[int, str] | Iterable[str]) -> dict[int, str]:
        if isinstance(names, dict):
            return names
        return {idx: name for idx, name in enumerate(names)}

    def _resolve_target_class_ids(self, target_classes: tuple[str, ...]) -> list[int]:
        normalized = {name.lower() for name in target_classes}
        return [
            class_id
            for class_id, class_name in self.names.items()
            if class_name.lower() in normalized
        ]

    @staticmethod
    def _ensure_tracking_dependencies() -> None:
        try:
            import lap  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ultralytics tracking requires the 'lap' package in the active project environment. "
                "Install it with: .venv/bin/python -m pip install 'lap>=0.5.12' "
                "or reinstall from requirements.txt."
            ) from exc

    @staticmethod
    def _resolve_device(device: str) -> str | None:
        normalized = device.strip().lower()
        if normalized == "auto":
            try:
                import torch
            except ModuleNotFoundError:
                return None
            if torch.backends.mps.is_available():
                return "mps"
            return None

        if normalized == "mps":
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "MPS was requested but PyTorch is not installed in the active environment."
                ) from exc
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS was requested but is not available in this runtime. "
                    "Use the local macOS .venv with YOLO_DEVICE=mps, or set YOLO_DEVICE=cpu."
                )
            return "mps"

        return normalized

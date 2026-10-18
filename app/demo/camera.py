"""Mock RTSP camera: draws a synthetic scene with the people the simulated
world puts in front of it, and publishes through the real DashboardStore so
the dashboard reads exactly what a live camera publisher would write.
"""

from __future__ import annotations

import logging
import time
import zlib
from datetime import datetime

import cv2
import numpy as np

from app.config.settings import Settings
from app.detection.async_runner import DetectorStats
from app.detection.yolo_detector import Detection
from app.devices.registry import DeviceSpec
from app.storage.dashboard_store import DashboardStore
from app.stream.reader import FramePacket

PERSON_CLASS_ID = 0
BOX_COLOR = (0, 200, 255)


class MockCamera:
    def __init__(
        self,
        device: DeviceSpec,
        settings: Settings,
        logger: logging.Logger,
        fps: float = 4.0,
    ) -> None:
        self.device = device
        self.fps = fps
        self.width = settings.width
        self.height = settings.height
        self.settings = settings.with_overrides(
            camera_id=device.id,
            camera_host=str(device.option("host") or "mock"),
            camera_stream="mock",
            yolo_enabled=True,
            yolo_tracking=True,
            yolo_model="demo: mock person detections",
            yolo_device="none",
            yolo_classes=("person",),
        )
        self.store = DashboardStore(settings=self.settings, logger=logger)
        self.rng = np.random.default_rng(zlib.crc32(device.id.encode()))
        self._background = self._draw_background()
        self._frame_index = 0
        self._last_frame_at = 0.0
        self._track_history: dict[int, list[tuple[int, int]]] = {}

    def tick(self, now: float, people: list[tuple[int, float, float, float]]) -> None:
        if now - self._last_frame_at < 1.0 / self.fps:
            return
        self._last_frame_at = now
        self._frame_index += 1
        captured_at = time.monotonic()
        frame, detections = self._render(people)
        stats = DetectorStats(
            inference_fps=self.fps,
            last_inference_ms=float(self.rng.uniform(38, 62)),
            processed_frames=self._frame_index,
            last_frame_index=self._frame_index,
        )
        self.store.publish(
            FramePacket(self._frame_index, frame, captured_at),
            frame,
            self.fps,
            detections,
            stats,
        )

    # Rendering ----------------------------------------------------------

    def _render(
        self, people: list[tuple[int, float, float, float]]
    ) -> tuple[np.ndarray, list[Detection]]:
        frame = self._background.copy()
        night = _is_night()
        detections: list[Detection] = []
        for track_id, cx, foot, scale in sorted(people, key=lambda p: p[2]):
            box = self._draw_person(frame, cx, foot, scale, night)
            confidence = float(np.clip(self.rng.normal(0.86, 0.04), 0.55, 0.97))
            detections.append(
                Detection(
                    class_id=PERSON_CLASS_ID,
                    class_name="person",
                    confidence=confidence,
                    bbox_xyxy=box,
                    track_id=track_id,
                )
            )
        if night:
            # IR night mode: grayscale with a little sensor noise.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        noise = self.rng.normal(0, 4 if not night else 7, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        self._draw_detections(frame, detections)
        self._draw_osd(frame)
        return frame, detections

    def _draw_background(self) -> np.ndarray:
        w, h = self.width, self.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        horizon = int(h * 0.42)
        night = _is_night()
        sky_top = (60, 40, 30) if night else (230, 190, 140)
        sky_bottom = (90, 70, 55) if night else (245, 225, 200)
        for y in range(horizon):
            t = y / max(horizon - 1, 1)
            frame[y, :] = [int(a + (b - a) * t) for a, b in zip(sky_top, sky_bottom)]
        ground = (50, 70, 55) if night else (90, 140, 105)
        frame[horizon:, :] = ground
        # Paving towards the camera.
        cv2.fillPoly(
            frame,
            [
                np.array(
                    [[w * 0.35, horizon], [w * 0.55, horizon], [w * 0.95, h], [w * 0.05, h]],
                    np.int32,
                )
            ],
            (120, 125, 130) if not night else (75, 78, 80),
        )
        # Perimeter wall with a gate gap.
        wall_top, wall_bottom = int(h * 0.30), horizon + 6
        wall_color = (170, 180, 190) if not night else (95, 100, 105)
        gate_x = int(w * self.rng.uniform(0.55, 0.75))
        cv2.rectangle(frame, (0, wall_top), (gate_x - 40, wall_bottom), wall_color, -1)
        cv2.rectangle(frame, (gate_x + 40, wall_top), (w, wall_bottom), wall_color, -1)
        for x in range(gate_x - 40, gate_x + 41, 8):
            cv2.line(frame, (x, wall_top + 4), (x, wall_bottom), (60, 60, 60), 2)
        # A house front behind the wall.
        house_x = int(w * self.rng.uniform(0.05, 0.3))
        cv2.rectangle(
            frame,
            (house_x, int(h * 0.08)),
            (house_x + int(w * 0.38), wall_top),
            (150, 160, 200),
            -1,
        )
        for i in range(3):
            wx = house_x + 20 + i * int(w * 0.12)
            lit = night and self.rng.random() < 0.5
            cv2.rectangle(
                frame,
                (wx, int(h * 0.13)),
                (wx + 34, int(h * 0.22)),
                (90, 200, 240) if lit else (90, 80, 70),
                -1,
            )
        # A tree and a light post for depth.
        tree_x = int(w * self.rng.uniform(0.8, 0.92))
        cv2.line(frame, (tree_x, horizon + 30), (tree_x, int(h * 0.2)), (40, 60, 80), 7)
        cv2.circle(frame, (tree_x, int(h * 0.18)), 38, (40, 110, 60), -1)
        return frame

    def _draw_person(
        self, frame: np.ndarray, cx: float, foot: float, scale: float, night: bool
    ) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        height_px = int(h * scale * 0.85)
        width_px = max(int(height_px * 0.38), 6)
        x_center = int(cx * w)
        y_foot = int(foot * h)
        x1, y1 = x_center - width_px // 2, y_foot - height_px
        x2, y2 = x_center + width_px // 2, y_foot
        cloth = (60, 50, 140) if not night else (180, 180, 180)
        skin = (120, 150, 200) if not night else (220, 220, 220)
        head_r = max(height_px // 9, 3)
        cv2.circle(frame, (x_center, y1 + head_r), head_r, skin, -1, cv2.LINE_AA)
        torso_top = y1 + head_r * 2
        torso_bottom = y1 + int(height_px * 0.58)
        cv2.rectangle(
            frame, (x1 + width_px // 6, torso_top), (x2 - width_px // 6, torso_bottom), cloth, -1
        )
        leg_w = max(width_px // 5, 2)
        stride = int(width_px * 0.25 * np.sin(time.monotonic() * 6 + cx * 10))
        for side in (-1, 1):
            cv2.line(
                frame,
                (x_center + side * leg_w, torso_bottom),
                (x_center + side * (leg_w + stride), y2),
                (40, 40, 40) if not night else (150, 150, 150),
                max(leg_w, 2),
                cv2.LINE_AA,
            )
        pad = 4
        return (
            max(x1 - pad, 0),
            max(y1 - pad, 0),
            min(x2 + pad, w - 1),
            min(y2 + pad, h - 1),
        )

    def _draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> None:
        # Same look as YoloDetector.draw, so mock frames match live ones.
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            label = f"{detection.class_name} #{detection.track_id} {detection.confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                BOX_COLOR,
                2,
                cv2.LINE_AA,
            )
            history = self._track_history.setdefault(detection.track_id or 0, [])
            history.append(((x1 + x2) // 2, y2))
            del history[:-30]
            if len(history) >= 2:
                cv2.polylines(
                    frame,
                    [np.array(history, np.int32).reshape((-1, 1, 2))],
                    False,
                    (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
        active = {detection.track_id for detection in detections}
        for track_id in list(self._track_history):
            if track_id not in active:
                del self._track_history[track_id]

    def _draw_osd(self, frame: np.ndarray) -> None:
        text = f"{self.device.name}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (11, 23), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 22), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "DEMO", (frame.shape[1] - 62, 22), font, 0.55, (0, 80, 255), 2, cv2.LINE_AA)


def _is_night() -> bool:
    hour = datetime.now().hour
    return hour >= 18 or hour < 6

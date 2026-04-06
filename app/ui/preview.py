from __future__ import annotations

import logging
import time
from threading import Event

import cv2
import numpy as np

from app.config.settings import Settings
from app.detection.async_runner import AsyncDetectorRunner
from app.detection.yolo_detector import BaseDetector, Detection
from app.storage.dashboard_store import DashboardStore
from app.stream.reader import FramePacket, StreamReader


def run_preview(
    settings: Settings,
    reader: StreamReader,
    logger: logging.Logger,
    stop_event: Event,
    detector: BaseDetector | None = None,
    dashboard_store: DashboardStore | None = None,
    show_window: bool = True,
    infer_every_n: int = 1,
) -> int:
    logger = logger.getChild("preview")
    if show_window:
        cv2.namedWindow(settings.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            settings.window_name,
            int(settings.width * settings.window_scale),
            int(settings.height * settings.window_scale),
        )

    last_index: int | None = None
    last_frame_at = time.monotonic()
    fps_ema = 0.0
    last_display_frame: np.ndarray | None = None
    detector_runner: AsyncDetectorRunner | None = None

    try:
        if detector is not None:
            detector_runner = AsyncDetectorRunner(detector=detector, logger=logger)
            detector_runner.start()
        reader.start()
        while not stop_event.is_set():
            packet = reader.read(
                timeout_seconds=settings.read_timeout_seconds,
                since_index=last_index,
            )

            if packet is None:
                stalled_seconds = time.monotonic() - last_frame_at
                if stalled_seconds > settings.read_timeout_seconds:
                    if not show_window:
                        continue
                    if last_display_frame is None:
                        waiting = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
                        cv2.putText(
                            waiting,
                            "Waiting for stream frames...",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                        )
                        cv2.imshow(settings.window_name, waiting)
                    else:
                        stalled_frame = last_display_frame.copy()
                        _draw_stall_overlay(stalled_frame, stalled_seconds)
                        cv2.imshow(settings.window_name, stalled_frame)
                if show_window and cv2.waitKey(1) & 0xFF == 27:
                    stop_event.set()
                continue

            now = time.monotonic()
            frame_interval = max(now - last_frame_at, 1e-6)
            instant_fps = 1.0 / frame_interval
            fps_ema = instant_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * instant_fps)
            last_frame_at = now
            last_index = packet.index

            display_frame = packet.frame.copy()
            detections_count = 0
            latest_detections: list[Detection] = []
            detector_inference_fps = 0.0
            detector_inference_ms = 0.0
            if detector_runner is not None:
                if packet.index % infer_every_n == 0:
                    detector_runner.submit(packet.frame, packet.index)
                latest_detection = detector_runner.get_latest()
                if latest_detection is not None:
                    latest_detections = latest_detection.detections
                    detections_count = len(latest_detections)
                    detector.draw(display_frame, latest_detections)
                detector_stats = detector_runner.get_stats()
                detector_inference_fps = detector_stats.inference_fps
                detector_inference_ms = detector_stats.last_inference_ms
            else:
                detector_stats = None

            _draw_status_overlay(
                frame=display_frame,
                packet=packet,
                current_time=now,
                fps=fps_ema,
                detections_count=detections_count,
                detector_fps=detector_inference_fps,
                detector_ms=detector_inference_ms,
            )

            if dashboard_store is not None:
                dashboard_store.publish(
                    packet=packet,
                    frame=display_frame,
                    fps=fps_ema,
                    detections=latest_detections,
                    detector_stats=detector_stats,
                )

            last_display_frame = display_frame.copy()
            if show_window:
                cv2.imshow(settings.window_name, display_frame)
            if show_window and cv2.waitKey(1) & 0xFF == 27:
                stop_event.set()

        return 0
    finally:
        if detector_runner is not None:
            detector_runner.stop()
        reader.stop()
        if show_window:
            cv2.destroyAllWindows()


def _draw_status_overlay(
    frame: np.ndarray,
    packet: FramePacket,
    current_time: float,
    fps: float,
    detections_count: int,
    detector_fps: float,
    detector_ms: float,
) -> None:
    latency_ms = (current_time - packet.captured_at_monotonic) * 1_000.0
    cv2.putText(
        frame,
        (
            f"FPS {fps:5.1f} | Latency {latency_ms:6.1f} ms | "
            f"DetFPS {detector_fps:4.1f} | Infer {detector_ms:5.1f} ms | "
            f"Frame {packet.index} | Dets {detections_count}"
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _draw_stall_overlay(frame: np.ndarray, stalled_seconds: float) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (15, 15, 15), -1)
    cv2.putText(
        frame,
        f"Stream stalled, showing last frame | reconnecting... {stalled_seconds:4.1f}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

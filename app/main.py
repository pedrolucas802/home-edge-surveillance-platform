from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from threading import Event

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.config.settings import Settings
from app.detection.yolo_detector import YoloDetector
from app.storage.dashboard_store import DashboardStore
from app.stream.ffmpeg_reader import FFmpegLatestFrameReader
from app.ui.preview import run_preview


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Home Edge Surveillance low-latency local preview")
    parser.add_argument("--rtsp-url", default=defaults.rtsp_url_override or None, help="Optional full RTSP URL override.")
    parser.add_argument("--camera", default=defaults.camera_id, help="Logical camera id, e.g. cam1 or cam2.")
    parser.add_argument("--host", default=None, help="Explicit camera host/IP override.")
    parser.add_argument("--port", type=int, default=defaults.camera_port, help="Camera RTSP port.")
    parser.add_argument("--user", default=defaults.camera_user, help="Camera username.")
    parser.add_argument(
        "--password",
        default=defaults.camera_pass or None,
        help="Camera password. If omitted, CAM_PASS is used.",
    )
    parser.add_argument("--stream", default=defaults.camera_stream, help="Camera stream path, e.g. onvif2.")
    parser.add_argument("--transport", default=defaults.rtsp_transport, choices=["udp", "tcp"], help="RTSP transport mode.")
    parser.add_argument("--width", type=int, default=defaults.width, help="Output frame width.")
    parser.add_argument("--height", type=int, default=defaults.height, help="Output frame height.")
    parser.add_argument("--fps", type=int, default=defaults.output_fps, help="Output frame rate after ffmpeg filters.")
    parser.add_argument("--ffmpeg-bin", default=defaults.ffmpeg_bin, help="Path to ffmpeg binary.")
    parser.add_argument("--ffmpeg-loglevel", default=defaults.ffmpeg_loglevel, help="ffmpeg loglevel, e.g. error or warning.")
    parser.add_argument("--window-name", default=defaults.window_name, help="Display window title.")
    parser.add_argument("--headless", action="store_true", help="Run without an OpenCV preview window.")
    parser.add_argument(
        "--dashboard-publish",
        action=argparse.BooleanOptionalAction,
        default=defaults.dashboard_enabled,
        help="Publish live artifacts for the Streamlit dashboard.",
    )
    parser.add_argument("--dashboard-dir", default=defaults.dashboard_dir, help="Directory for dashboard live data.")
    parser.add_argument("--disable-hwaccel", action="store_true", help="Disable videotoolbox hardware acceleration.")
    parser.add_argument("--enable-yolo", action="store_true", default=defaults.yolo_enabled, help="Enable YOLO inference in preview.")
    parser.add_argument(
        "--pose",
        action=argparse.BooleanOptionalAction,
        default=defaults.yolo_pose_enabled,
        help="Enable YOLO keypoint pose overlay when the selected model supports pose outputs.",
    )
    parser.add_argument("--yolo-model", default=defaults.yolo_model, help="YOLO model name/path when enabled.")
    parser.add_argument("--yolo-device", default=defaults.yolo_device, help="YOLO device: cpu, mps, cuda, auto.")
    parser.add_argument("--yolo-confidence", type=float, default=defaults.yolo_confidence, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=defaults.yolo_iou, help="YOLO IOU threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=defaults.yolo_imgsz, help="YOLO inference image size.")
    parser.add_argument("--yolo-classes", nargs="*", default=list(defaults.yolo_classes), help="Optional class-name filters.")
    parser.add_argument(
        "--infer-every-n",
        type=int,
        default=defaults.yolo_infer_every_n,
        help="Run YOLO every Nth frame to trade detection frequency for responsiveness.",
    )
    parser.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=defaults.yolo_tracking,
        help="Enable Ultralytics object tracking with persistent IDs.",
    )
    parser.add_argument("--tracker", default=defaults.yolo_tracker, help="Ultralytics tracker YAML, e.g. bytetrack.yaml.")
    parser.add_argument("--track-history", type=int, default=defaults.yolo_track_history, help="How many past centers to keep per track.")
    return parser


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("surveillance_platform")


def install_signal_handlers(stop_event: Event, logger: logging.Logger) -> None:
    def _handle_signal(signum: int, _frame: object) -> None:
        signal_name = signal.Signals(signum).name
        logger.info("Received %s, stopping preview loop.", signal_name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)


def build_settings(defaults: Settings, args: argparse.Namespace) -> Settings:
    selected_camera_id = args.camera or defaults.camera_id
    resolved_host = args.host or defaults.resolve_camera_host(selected_camera_id)
    return defaults.with_overrides(
        camera_id=selected_camera_id,
        rtsp_url_override=args.rtsp_url or "",
        camera_host=resolved_host,
        camera_port=args.port,
        camera_user=args.user,
        camera_pass=args.password or defaults.camera_pass,
        camera_stream=args.stream,
        rtsp_transport=args.transport,
        ffmpeg_bin=args.ffmpeg_bin,
        ffmpeg_loglevel=args.ffmpeg_loglevel,
        width=args.width,
        height=args.height,
        output_fps=args.fps,
        use_hwaccel=not args.disable_hwaccel,
        window_name=args.window_name,
        dashboard_enabled=args.dashboard_publish,
        dashboard_dir=args.dashboard_dir,
        yolo_enabled=args.enable_yolo,
        yolo_pose_enabled=args.pose,
        yolo_tracking=args.tracking,
    )


def main() -> int:
    defaults = Settings.from_env()
    args = build_parser(defaults).parse_args()
    try:
        settings = build_settings(defaults, args)
        settings.ensure_credentials()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        print("Example:")
        print("  CAM_PASS='your_password' python -m app.main --stream onvif2")
        print("  python -m app.main --password 'your_password' --stream onvif2")
        return 2

    logger = configure_logging()
    stop_event = Event()
    install_signal_handlers(stop_event, logger)

    logger.info(
        "Starting surveillance preview. camera=%s host=%s stream=%s transport=%s output=%sx%s@%sfps",
        settings.camera_id,
        settings.camera_host,
        settings.camera_stream if not settings.rtsp_url_override else "custom-rtsp-url",
        settings.rtsp_transport,
        settings.width,
        settings.height,
        settings.effective_output_fps,
    )

    if settings.output_fps > settings.source_fps:
        logger.warning(
            "Requested fps=%s exceeds source fps=%s. Clamping output fps to %s.",
            settings.output_fps,
            settings.source_fps,
            settings.effective_output_fps,
        )

    reader = FFmpegLatestFrameReader(settings=settings, logger=logger)
    dashboard_store = DashboardStore(settings=settings, logger=logger) if settings.dashboard_enabled else None
    detector = None
    run_yolo = bool(args.enable_yolo or args.pose or args.tracking)
    if run_yolo:
        try:
            detector = YoloDetector(
                model_name=args.yolo_model,
                device=args.yolo_device,
                confidence=args.yolo_confidence,
                iou=args.yolo_iou,
                imgsz=args.yolo_imgsz,
                target_classes=tuple(args.yolo_classes),
                detect_enabled=args.enable_yolo,
                pose_enabled=args.pose,
                tracking=args.tracking,
                tracker=args.tracker,
                track_history_length=args.track_history,
            )
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 2
        logger.info(
            (
                "YOLO enabled with model=%s device=%s imgsz=%s classes=%s infer_every_n=%s "
                "detect=%s pose=%s tracking=%s tracker=%s"
            ),
            args.yolo_model,
            detector.device or "cpu-auto",
            args.yolo_imgsz,
            ",".join(args.yolo_classes) if args.yolo_classes else "all",
            args.infer_every_n,
            "on" if args.enable_yolo else "off",
            "on" if args.pose else "off",
            "on" if args.tracking else "off",
            args.tracker,
        )

    return run_preview(
        settings=settings,
        reader=reader,
        logger=logger,
        stop_event=stop_event,
        detector=detector,
        dashboard_store=dashboard_store,
        show_window=not args.headless,
        infer_every_n=max(args.infer_every_n, 1),
    )


if __name__ == "__main__":
    raise SystemExit(main())

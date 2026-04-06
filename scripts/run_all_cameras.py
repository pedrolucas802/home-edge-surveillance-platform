from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

# Support direct execution from the scripts/ directory.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.config.settings import Settings


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch one preview process per configured camera."
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(defaults.camera_hosts_map) or [defaults.camera_id],
        help="Camera ids to launch, e.g. cam1 cam2.",
    )
    parser.add_argument("--stream", default=defaults.camera_stream, help="RTSP stream path.")
    parser.add_argument("--width", type=int, default=defaults.width, help="Output frame width.")
    parser.add_argument("--height", type=int, default=defaults.height, help="Output frame height.")
    parser.add_argument("--fps", type=int, default=defaults.output_fps, help="Requested output FPS.")
    parser.add_argument("--transport", default=defaults.rtsp_transport, choices=["udp", "tcp"], help="RTSP transport.")
    parser.add_argument(
        "--launch-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between launching camera processes.",
    )
    parser.add_argument(
        "--enable-yolo",
        action="store_true",
        default=defaults.yolo_enabled,
        help="Enable YOLO in each camera process.",
    )
    parser.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=defaults.yolo_tracking,
        help="Enable Ultralytics tracking in each camera process.",
    )
    parser.add_argument(
        "--yolo-classes",
        nargs="*",
        default=list(defaults.yolo_classes),
        help="Class filter forwarded to each camera process.",
    )
    return parser


def build_command(args: argparse.Namespace, camera_id: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "app.main",
        "--camera",
        camera_id,
        "--stream",
        args.stream,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--transport",
        args.transport,
        "--window-name",
        f"Surveillance {camera_id}",
    ]
    if args.enable_yolo:
        command.append("--enable-yolo")
    if args.tracking:
        command.append("--tracking")
    else:
        command.append("--no-tracking")
    if args.yolo_classes:
        command.extend(["--yolo-classes", *args.yolo_classes])
    return command


def terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 3.0
    for process in processes:
        if process.poll() is None:
            remaining = max(deadline - time.monotonic(), 0.0)
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    defaults = Settings.from_env()
    args = build_parser(defaults).parse_args()

    available_cameras = defaults.camera_hosts_map
    if available_cameras:
        unknown = [camera_id for camera_id in args.cameras if camera_id not in available_cameras]
        if unknown:
            print(
                "Unknown camera id(s): "
                f"{', '.join(unknown)}. Available cameras: {', '.join(sorted(available_cameras))}"
            )
            return 2

    processes: list[subprocess.Popen[bytes]] = []

    def _handle_signal(signum: int, _frame: object) -> None:
        signal_name = signal.Signals(signum).name
        print(f"Received {signal_name}, stopping all camera previews...")
        terminate_processes(processes)
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    try:
        for camera_id in args.cameras:
            command = build_command(args, camera_id)
            print(f"Launching {camera_id}: {' '.join(command)}")
            processes.append(subprocess.Popen(command))
            time.sleep(max(args.launch_delay, 0.0))

        while processes:
            for process in processes:
                return_code = process.poll()
                if return_code is None:
                    continue
                if return_code != 0:
                    print(
                        f"Camera process exited with code {return_code}. "
                        "Stopping the remaining previews."
                    )
                terminate_processes(processes)
                return return_code
            time.sleep(0.5)
        return 0
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())

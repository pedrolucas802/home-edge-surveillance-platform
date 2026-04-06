from __future__ import annotations

from collections import deque
import logging
import re
import subprocess
import threading
import time

import numpy as np

from app.config.settings import Settings
from app.stream.reader import FramePacket


class FFmpegLatestFrameReader:
    """Read frames from ffmpeg stdout and keep only the freshest frame."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger.getChild("ffmpeg_reader")
        self.frame_size = settings.width * settings.height * 3
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._latest_frame: FramePacket | None = None
        self._frame_index = 0
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._last_error: str | None = None
        self._stderr_lines: deque[str] = deque(maxlen=20)

    def start(self) -> None:
        if self.is_running:
            return

        final_detail = ""
        for attempt in range(1, self.settings.startup_attempts + 1):
            self._stop_event.clear()
            self._latest_frame = None
            self._last_error = None
            self._spawn_process()
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name="ffmpeg-latest-frame-reader",
                daemon=True,
            )
            self._reader_thread.start()

            deadline = time.monotonic() + self.settings.startup_timeout_seconds
            while time.monotonic() < deadline and not self._stop_event.is_set():
                first_frame = self.read(timeout_seconds=0.2)
                if first_frame is not None:
                    return

            final_detail = self._build_error_detail()
            self.logger.warning(
                "Initial ffmpeg startup attempt %s/%s failed%s",
                attempt,
                self.settings.startup_attempts,
                final_detail,
            )
            self.stop()
            if attempt < self.settings.startup_attempts:
                time.sleep(self.settings.reconnect_delay_seconds)

        raise RuntimeError(f"Could not receive initial frames from ffmpeg stream.{final_detail}")

    def read(self, timeout_seconds: float, since_index: int | None = None) -> FramePacket | None:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while not self._stop_event.is_set():
                if self._latest_frame is not None:
                    if since_index is None or self._latest_frame.index > since_index:
                        return self._latest_frame

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
        return None

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        self._terminate_process()

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None

    @property
    def is_running(self) -> bool:
        return self._reader_thread is not None and self._reader_thread.is_alive()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._process is None or self._process.poll() is not None:
                self._restart("ffmpeg process not running")
                continue

            raw_frame = self._read_exact_frame()
            if raw_frame is None:
                self._restart(self._build_error_message("failed to read full raw frame"))
                continue

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (self.settings.height, self.settings.width, 3)
            )
            self._frame_index += 1
            packet = FramePacket(
                index=self._frame_index,
                frame=frame,
                captured_at_monotonic=time.monotonic(),
            )

            # Keep only the newest frame to avoid backlog and reduce latency.
            with self._condition:
                self._latest_frame = packet
                self._condition.notify_all()

    def _read_exact_frame(self) -> bytes | None:
        if self._process is None or self._process.stdout is None:
            return None

        buffer = bytearray()
        while len(buffer) < self.frame_size and not self._stop_event.is_set():
            chunk = self._process.stdout.read(self.frame_size - len(buffer))
            if not chunk:
                return None
            buffer.extend(chunk)

        if len(buffer) != self.frame_size:
            return None
        return bytes(buffer)

    def _restart(self, reason: str) -> None:
        if self._stop_event.is_set():
            return

        self._last_error = reason
        self.logger.warning("Restarting ffmpeg reader: %s", reason)
        self._terminate_process()
        time.sleep(self.settings.reconnect_delay_seconds)
        self._spawn_process()

    def _spawn_process(self) -> None:
        command = self._build_ffmpeg_command()
        self.logger.info("Starting ffmpeg reader process.")
        self._stderr_lines.clear()

        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ffmpeg binary not found at '{self.settings.ffmpeg_bin}'. "
                "Set FFMPEG_BIN to the correct path."
            ) from exc

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="ffmpeg-stderr-reader",
            daemon=True,
        )
        self._stderr_thread.start()

    def _terminate_process(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)

        self._process = None

    def _build_ffmpeg_command(self) -> list[str]:
        video_filters = [f"scale={self.settings.width}:{self.settings.height}"]

        command = [
            self.settings.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            self.settings.ffmpeg_loglevel,
            "-rtsp_transport",
            self.settings.rtsp_transport,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-avioflags",
            "direct",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-an",
        ]

        if self.settings.use_hwaccel:
            command.extend(["-hwaccel", "videotoolbox"])

        command.extend(
            [
                "-i",
                self.settings.rtsp_url,
                "-vf",
                ",".join(video_filters),
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        return command

    def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return

        while not self._stop_event.is_set():
            line = self._process.stderr.readline()
            if not line:
                break
            decoded = self._sanitize_stderr_line(
                line.decode("utf-8", errors="replace").strip()
            )
            if decoded:
                self._stderr_lines.append(decoded)

    def _build_error_message(self, prefix: str) -> str:
        if not self._stderr_lines:
            return prefix
        return f"{prefix}. ffmpeg: {self._stderr_lines[-1]}"

    def _build_error_detail(self) -> str:
        if self._last_error:
            return f" Last error: {self._last_error}"
        if self._stderr_lines:
            return f" ffmpeg: {self._stderr_lines[-1]}"
        return ""

    @staticmethod
    def _sanitize_stderr_line(line: str) -> str:
        return re.sub(r"(rtsp://[^:\s]+:)[^@\s]+(@)", r"\1***\2", line)

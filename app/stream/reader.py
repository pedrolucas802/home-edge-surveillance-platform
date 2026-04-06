from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class FramePacket:
    index: int
    frame: np.ndarray
    captured_at_monotonic: float


class StreamReader(Protocol):
    def start(self) -> None:
        """Start stream ingestion resources."""

    def read(self, timeout_seconds: float, since_index: int | None = None) -> FramePacket | None:
        """Get the most recent frame, optionally waiting for a newer frame index."""

    def stop(self) -> None:
        """Release stream ingestion resources."""


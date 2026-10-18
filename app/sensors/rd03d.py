"""Ai-Thinker RD-03D 24 GHz radar: UART frame format and target maths.

The RD-03D tracks up to 3 people and reports each one about 10 times per
second over UART (256000 baud, 8N1) in a 30-byte frame:

    AA FF 03 00 | target 1 (8 bytes) | target 2 | target 3 | 55 CC

Each target is four little-endian uint16 values: X and Y position in mm,
speed in cm/s, and the distance gate resolution in mm. X, Y and speed use
sign-magnitude with an inverted sign bit: bit 15 set means positive. An
all-zero target slot means "no target". The ESP32 firmware decodes these
frames and publishes the targets over MQTT; the simulator and tests use the
same codec so the hub sees exactly what the firmware will send.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

FRAME_HEADER = bytes((0xAA, 0xFF, 0x03, 0x00))
FRAME_FOOTER = bytes((0x55, 0xCC))
FRAME_LENGTH = 30
MAX_TARGETS = 3
MAX_RANGE_MM = 8000
FIELD_OF_VIEW_DEG = 120.0
DEFAULT_RESOLUTION_MM = 360


@dataclass(slots=True, frozen=True)
class RadarTarget:
    x_mm: int
    y_mm: int
    speed_cms: int
    resolution_mm: int = DEFAULT_RESOLUTION_MM

    @property
    def distance_mm(self) -> int:
        return int(round(math.hypot(self.x_mm, self.y_mm)))

    @property
    def angle_deg(self) -> float:
        # 0 degrees is straight ahead; negative is to the radar's left.
        return round(math.degrees(math.atan2(self.x_mm, self.y_mm)), 1)

    def to_payload(self) -> dict[str, object]:
        return {
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "speed_cms": self.speed_cms,
            "distance_mm": self.distance_mm,
            "angle_deg": self.angle_deg,
        }


def in_coverage(x_mm: float, y_mm: float) -> bool:
    if y_mm <= 0:
        return False
    if math.hypot(x_mm, y_mm) > MAX_RANGE_MM:
        return False
    return abs(math.degrees(math.atan2(x_mm, y_mm))) <= FIELD_OF_VIEW_DEG / 2


def encode_frame(targets: list[RadarTarget]) -> bytes:
    if len(targets) > MAX_TARGETS:
        raise ValueError(f"RD-03D reports at most {MAX_TARGETS} targets.")
    body = b""
    for slot in range(MAX_TARGETS):
        if slot < len(targets):
            target = targets[slot]
            body += struct.pack(
                "<HHHH",
                _encode_signed(target.x_mm),
                _encode_signed(target.y_mm),
                _encode_signed(target.speed_cms),
                target.resolution_mm & 0xFFFF,
            )
        else:
            body += bytes(8)
    return FRAME_HEADER + body + FRAME_FOOTER


def decode_frame(frame: bytes) -> list[RadarTarget]:
    if len(frame) != FRAME_LENGTH:
        raise ValueError(f"RD-03D frame must be {FRAME_LENGTH} bytes, got {len(frame)}.")
    if frame[:4] != FRAME_HEADER or frame[-2:] != FRAME_FOOTER:
        raise ValueError("Not an RD-03D target frame (bad header or footer).")
    targets: list[RadarTarget] = []
    for slot in range(MAX_TARGETS):
        raw = frame[4 + slot * 8 : 12 + slot * 8]
        if raw == bytes(8):
            continue
        x_raw, y_raw, speed_raw, resolution = struct.unpack("<HHHH", raw)
        targets.append(
            RadarTarget(
                x_mm=_decode_signed(x_raw),
                y_mm=_decode_signed(y_raw),
                speed_cms=_decode_signed(speed_raw),
                resolution_mm=resolution,
            )
        )
    return targets


def _encode_signed(value: int) -> int:
    magnitude = min(abs(int(value)), 0x7FFF)
    return magnitude | 0x8000 if value >= 0 else magnitude


def _decode_signed(raw: int) -> int:
    magnitude = raw & 0x7FFF
    return magnitude if raw & 0x8000 else -magnitude

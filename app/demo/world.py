"""A small simulated house: people walk through zones and every device in
that zone reacts the way the real hardware would.

The tripwire in a zone breaks when someone crosses it, the RD-03D radar in
the zone tracks them, and the camera in the zone sees them. That gives the
dashboard correlated, believable activity before any hardware exists.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from app.devices.registry import DeviceRegistry, DeviceSpec
from app.devices.types import CAMERA, RADAR, TRIPWIRE
from app.sensors import rd03d

# Seconds between radar position updates sent to the hub (the RD-03D itself
# reports at about 10 Hz; the firmware forwards a thinned stream).
RADAR_STATE_INTERVAL = 0.5
TRIPWIRE_FAULT_CHANCE = 0.02
INDOOR_ZONES = frozenset({"indoor", "inside", "house"})


@dataclass(slots=True)
class Visit:
    """One person moving through one zone."""

    track_id: int
    zone: str
    started_at: float
    duration: float
    # Radar path in RD-03D coordinates (mm): from start to end and back out.
    start_mm: tuple[float, float]
    turn_mm: tuple[float, float]
    # Camera path in normalized image coordinates (0..1).
    cam_start: tuple[float, float]
    cam_end: tuple[float, float]
    crossings: dict[str, float] = field(default_factory=dict)

    def progress(self, now: float) -> float:
        return min(max((now - self.started_at) / self.duration, 0.0), 1.0)

    def active(self, now: float) -> bool:
        return self.started_at <= now < self.started_at + self.duration

    def radar_position(self, now: float) -> tuple[float, float, float]:
        """Return (x_mm, y_mm, speed_cms) at time now."""
        p = self.progress(now)
        # Walk in for the first 60% of the visit, then walk back out.
        if p < 0.6:
            a, b, f, leg = self.start_mm, self.turn_mm, p / 0.6, 0.6
        else:
            a, b, f, leg = self.turn_mm, self.start_mm, (p - 0.6) / 0.4, 0.4
        x = a[0] + (b[0] - a[0]) * f
        y = a[1] + (b[1] - a[1]) * f
        leg_seconds = self.duration * leg
        dist_mm = math.hypot(b[0] - a[0], b[1] - a[1])
        speed = (dist_mm / 10.0) / leg_seconds if leg_seconds > 0 else 0.0
        # RD-03D convention: negative speed means moving toward the radar.
        approaching = math.hypot(b[0], b[1]) < math.hypot(a[0], a[1])
        return x, y, -speed if approaching else speed

    def camera_box(self, now: float) -> tuple[float, float, float] | None:
        """Return (cx, foot_y, scale) in normalized image coordinates."""
        p = self.progress(now)
        if p <= 0.05 or p >= 0.95:
            return None
        f = (p - 0.05) / 0.9
        cx = self.cam_start[0] + (self.cam_end[0] - self.cam_start[0]) * f
        foot = self.cam_start[1] + (self.cam_end[1] - self.cam_start[1]) * f
        # People lower in the frame are closer, so they look bigger.
        scale = 0.06 + 0.4 * foot * foot
        return cx, foot, scale


@dataclass(slots=True)
class Output:
    device: DeviceSpec
    channel: str
    body: object


class SiteWorld:
    def __init__(
        self,
        registry: DeviceRegistry,
        rng: random.Random,
        mean_gap_seconds: float = 25.0,
        telemetry_interval: float = 30.0,
    ) -> None:
        self.registry = registry
        self.rng = rng
        self.mean_gap = mean_gap_seconds
        self.telemetry_interval = telemetry_interval
        self.started_at = time.monotonic()
        self.visits: list[Visit] = []
        self._next_track_id = 1
        self._next_visit_at = self.started_at + 2.0
        self._last_telemetry = 0.0
        self._last_radar_state: dict[str, float] = {}
        self._radar_present: dict[str, bool] = {}
        self._beam_broken_until: dict[str, tuple[float, float]] = {}
        self._fault_until: dict[str, float] = {}
        self.sensors = [device for device in registry.enabled() if device.type in {RADAR, TRIPWIRE}]
        self.cameras = registry.of_type(CAMERA)
        self.zones = sorted({device.zone or "default" for device in registry.enabled()})

    # Public API ---------------------------------------------------------

    def boot(self) -> list[Output]:
        outputs = [Output(device, "status", "online") for device in self.sensors]
        for device in self.sensors:
            if device.type == TRIPWIRE:
                outputs.append(
                    Output(device, "state", {"beam": "ok", "signal": self._beam_signal(device)})
                )
            elif device.type == RADAR:
                outputs.append(
                    Output(device, "state", {"presence": False, "target_count": 0, "targets": []})
                )
        return outputs

    def shutdown(self) -> list[Output]:
        return [Output(device, "status", "offline") for device in self.sensors]

    def tick(self, now: float) -> list[Output]:
        outputs: list[Output] = []
        self._maybe_start_visit(now)
        self.visits = [visit for visit in self.visits if now < visit.started_at + visit.duration]
        if now - self._last_telemetry >= self.telemetry_interval:
            self._last_telemetry = now
            outputs.extend(Output(d, "telemetry", self._telemetry(d, now)) for d in self.sensors)
        for device in self.sensors:
            if device.type == TRIPWIRE:
                outputs.extend(self._tick_tripwire(device, now))
            elif device.type == RADAR:
                outputs.extend(self._tick_radar(device, now))
        return outputs

    def camera_people(
        self, camera: DeviceSpec, now: float
    ) -> list[tuple[int, float, float, float]]:
        """People this camera sees now: (track_id, cx, foot_y, scale)."""
        zone = camera.zone or "default"
        people = []
        for visit in self.visits:
            if visit.zone != zone or not visit.active(now):
                continue
            box = visit.camera_box(now)
            if box is not None:
                people.append((visit.track_id, *box))
        return people

    def force_visit(self, zone: str, now: float) -> Visit:
        visit = self._new_visit(zone, now)
        self.visits.append(visit)
        return visit

    # Visits -------------------------------------------------------------

    def _maybe_start_visit(self, now: float) -> None:
        if now < self._next_visit_at or not self.zones:
            return
        self._next_visit_at = now + self.rng.expovariate(1.0 / self.mean_gap)
        busy = {visit.zone for visit in self.visits}
        # Indoor zones see more (resident) activity than the perimeter.
        weights = [3.0 if zone in INDOOR_ZONES else 1.0 for zone in self.zones]
        zone = self.rng.choices(self.zones, weights=weights)[0]
        if zone in busy and len([v for v in self.visits if v.zone == zone]) >= rd03d.MAX_TARGETS:
            return
        self.visits.append(self._new_visit(zone, now))

    def _new_visit(self, zone: str, now: float) -> Visit:
        rng = self.rng
        indoor = zone in INDOOR_ZONES
        duration = rng.uniform(14.0, 30.0) if indoor else rng.uniform(10.0, 20.0)
        start = (rng.uniform(-2500, 2500), rng.uniform(5500, 7200))
        turn = (rng.uniform(-1200, 1200), rng.uniform(700, 2200))
        left_to_right = rng.random() < 0.5
        cam_start = (0.12 if left_to_right else 0.88, rng.uniform(0.55, 0.7))
        cam_end = (0.85 if left_to_right else 0.15, rng.uniform(0.75, 0.95))
        visit = Visit(
            track_id=self._next_track_id,
            zone=zone,
            started_at=now,
            duration=duration,
            start_mm=start,
            turn_mm=turn,
            cam_start=cam_start,
            cam_end=cam_end,
        )
        self._next_track_id += 1
        for device in self.sensors:
            if device.type == TRIPWIRE and (device.zone or "default") == zone:
                # The person crosses the beam early in the visit.
                visit.crossings[device.id] = now + duration * rng.uniform(0.1, 0.3)
        return visit

    # Tripwire -----------------------------------------------------------

    def _tick_tripwire(self, device: DeviceSpec, now: float) -> list[Output]:
        outputs: list[Output] = []
        fault_until = self._fault_until.get(device.id)
        if fault_until is not None:
            if now >= fault_until:
                del self._fault_until[device.id]
                outputs.append(
                    Output(
                        device,
                        "event",
                        {
                            "event": "beam_restored",
                            "duration_ms": 35_000,
                            "signal": self._beam_signal(device),
                        },
                    )
                )
            return outputs

        broken = self._beam_broken_until.get(device.id)
        if broken is not None and now >= broken[0]:
            del self._beam_broken_until[device.id]
            outputs.append(
                Output(
                    device,
                    "event",
                    {
                        "event": "beam_restored",
                        "duration_ms": int(broken[1]),
                        "signal": self._beam_signal(device),
                    },
                )
            )

        for visit in self.visits:
            crossing_at = visit.crossings.get(device.id)
            if crossing_at is None or now < crossing_at:
                continue
            del visit.crossings[device.id]
            if self.rng.random() < TRIPWIRE_FAULT_CHANCE:
                self._fault_until[device.id] = now + 35.0
                outputs.append(Output(device, "event", {"event": "beam_broken", "signal": 12}))
                outputs.append(
                    Output(device, "event", {"event": "fault", "reason": "beam blocked > 30 s"})
                )
                continue
            duration_ms = self.rng.randint(180, 900)
            self._beam_broken_until[device.id] = (now + duration_ms / 1000.0, duration_ms)
            outputs.append(
                Output(
                    device,
                    "event",
                    {"event": "beam_broken", "signal": self.rng.randint(5, 40)},
                )
            )
        return outputs

    def _beam_signal(self, device: DeviceSpec) -> int:
        # Receiver level on a 0..1023 ADC: a laser on an LDR reads high when
        # the beam is on it; an IR barrier relay just reads as full scale.
        if device.option("variant") == "laser":
            return self.rng.randint(820, 960)
        return 1023

    # Radar --------------------------------------------------------------

    def _tick_radar(self, device: DeviceSpec, now: float) -> list[Output]:
        zone = device.zone or "default"
        targets: list[rd03d.RadarTarget] = []
        for visit in self.visits:
            if visit.zone != zone or not visit.active(now):
                continue
            x, y, speed = visit.radar_position(now)
            # Measurement noise of a few centimetres, like the real module.
            x += self.rng.gauss(0, 40)
            y += self.rng.gauss(0, 40)
            if rd03d.in_coverage(x, y) and len(targets) < rd03d.MAX_TARGETS:
                targets.append(rd03d.RadarTarget(int(x), int(y), int(speed)))
        # Round-trip through the UART frame format, as the firmware would.
        targets = rd03d.decode_frame(rd03d.encode_frame(targets))

        outputs: list[Output] = []
        present = bool(targets)
        was_present = self._radar_present.get(device.id, False)
        nearest_m = round(min(t.distance_mm for t in targets) / 1000.0, 2) if targets else None
        if present and not was_present:
            outputs.append(
                Output(
                    device,
                    "event",
                    {
                        "event": "presence_start",
                        "target_count": len(targets),
                        "nearest_m": nearest_m,
                        "targets": [t.to_payload() for t in targets],
                    },
                )
            )
        elif was_present and not present:
            outputs.append(Output(device, "event", {"event": "presence_end"}))
        self._radar_present[device.id] = present

        last_state = self._last_radar_state.get(device.id, 0.0)
        if present and now - last_state >= RADAR_STATE_INTERVAL:
            self._last_radar_state[device.id] = now
            outputs.append(
                Output(
                    device,
                    "state",
                    {
                        "presence": True,
                        "target_count": len(targets),
                        "nearest_m": nearest_m,
                        "targets": [t.to_payload() for t in targets],
                    },
                )
            )
        return outputs

    # Telemetry ----------------------------------------------------------

    def _telemetry(self, device: DeviceSpec, now: float) -> dict[str, object]:
        return {
            "rssi": self.rng.randint(-78, -48),
            "uptime_ms": int((now - self.started_at) * 1000),
            "fw": "sim-0.2",
            "heap_free": self.rng.randint(150_000, 200_000),
            "model": str(device.option("model") or device.option("variant") or ""),
        }

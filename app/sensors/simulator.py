"""Pretend to be the site's ESP32 sensors until the real hardware exists.

Every RD-03D radar and tripwire in the devices file gets a virtual board that
speaks the same MQTT contract the firmware will: retained online/offline
status with a Last Will, periodic telemetry, state and events.

    python -m app.sensors.simulator              # publish to the MQTT broker
    python -m app.sensors.simulator --direct     # no broker, write straight to the store

For cameras too, run the full demo: python -m app.demo
"""

from __future__ import annotations

import sys

from app.demo.__main__ import run


def main() -> int:
    argv = ["--transport" if arg == "--direct" else arg for arg in sys.argv[1:]]
    if "--direct" in sys.argv[1:]:
        argv.insert(argv.index("--transport") + 1, "direct")
    return run(argv, default_transport="mqtt", default_cameras=False)


if __name__ == "__main__":
    sys.exit(main())

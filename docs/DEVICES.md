# Devices: cameras, motion radars, tripwires

A client site can have any number of each device type. Every device is listed
once in the site's devices file, and the hub works out what to do from its
`type`.

| Type | Talks to the hub over | Events |
| --- | --- | --- |
| `camera` | RTSP (FFmpeg + YOLO publishers) | `detection` |
| `radar` | MQTT (ESP32 + Ai-Thinker RD-03D 24 GHz) | `presence_start`, `presence_end`, `fault` |
| `tripwire` | MQTT (ESP32 + laser or IR barrier) | `beam_broken`, `beam_restored`, `fault`, `tamper` |

```text
ESP32 radars/tripwires --MQTT--> Mosquitto --> sensor-bridge --+
                                                               +--> data/dashboard/events.db --> Streamlit
RTSP cameras --> FFmpeg --> YOLO --> camera-publishers --------+     (events + device_state)
```

## 1. Describe the site

```bash
cp configs/devices.example.yaml configs/devices.yaml
```

Edit `configs/devices.yaml` (git-ignored, one per client). Each device needs a
unique lowercase `id` and a `type`; `name`, `location`, `zone` and `enabled`
are optional, and any other key is a type-specific option (`host` for a
camera, `model` for a radar). Once this file exists, its camera `host`
entries replace `CAMERA_HOSTS` from `.env`. Without it, the hub runs
camera-only from `CAMERA_HOSTS`, exactly as before.

`zone` groups devices that watch the same area (e.g. the front-wall tripwire
and the front-yard camera). Nothing uses it yet; the fusion service will.

## 2. Run it without hardware

Quickest: `bash scripts/run_demo.sh`. It uses `configs/devices.demo.yaml`
(3 cameras, 4 RD-03D radars, 4 tripwires in 4 zones) and its own data
folder `data/demo/`, and starts mock devices plus the dashboard on
http://localhost:8501. People walk through zones; each one breaks that
zone's tripwire, shows up as a target on its radar, and appears as a tracked
`person` on its camera, with snapshots and events saved as a real camera
publisher would. `bash scripts/run_demo.sh --mqtt` sends the sensor traffic
through Mosquitto and the bridge instead (start Mosquitto first).

To simulate only the sensors of your own devices file:

You need an MQTT broker. Either:

```bash
docker compose up -d mosquitto        # uses deploy/mosquitto/mosquitto.conf
# or on macOS: brew install mosquitto && mosquitto -p 1883
```

Then, in three terminals:

```bash
python -m app.sensors.bridge          # MQTT -> events.db
python -m app.sensors.simulator       # virtual boards for every radar and tripwire
bash scripts/run_dashboard_stack.sh --dashboard-only
```

The dashboard's **Sensors** tab shows a card per radar and tripwire, grouped
by type; **Devices** lists every device with online/offline, last seen, RSSI
and firmware; **Recent Events** mixes camera and sensor events (🔴 marks a
trigger event such as a broken beam).

No broker at all? `python -m app.sensors.simulator --direct` writes straight
into the event store, skipping MQTT and the bridge.

With Docker: `docker compose --profile sim up` runs Mosquitto, the bridge, the
simulator, the camera publishers and the dashboard together.

## 3. MQTT contract for the firmware

Topics follow `hes/<site>/<device-id>/<channel>`, where `<site>` is `site:` in
the devices file and `<device-id>` is the device's `id`.

| Channel | Direction | Retained | Payload |
| --- | --- | --- | --- |
| `event` | device to hub | no | `{"event": "beam_broken", "ts": 1792300000.5, ...}` |
| `state` | device to hub | yes | `{"presence": true, "distance_cm": 210}` (merged into last state) |
| `status` | device to hub | yes | `online` / `offline`; set `offline` as the MQTT Last Will |
| `telemetry` | device to hub | no | `{"rssi": -60, "uptime_ms": 123456, "fw": "1.0.0"}` every 30 to 60 s |
| `cmd` | hub to device | no | reserved for siren and arm/disarm commands |

Radars (RD-03D) also publish `state` about twice a second while someone is
in range, with the decoded targets (the module tracks up to 3 people, 8 m,
120 degree field of view; frame format in `app/sensors/rd03d.py`):

```json
{"presence": true, "target_count": 2, "nearest_m": 1.88,
 "targets": [{"x_mm": -782, "y_mm": 1713, "speed_cms": -16, "distance_mm": 1883, "angle_deg": -24.5}, ...]}
```

Positions are not stored as events, only as the radar's last state; the
dashboard draws them on a small top-down map. Events are only
`presence_start` (with the first targets) and `presence_end`.

Tripwires send `beam_broken` (with `signal`, the receiver level 0..1023) and
`beam_restored` (with `duration_ms`); a beam blocked for over 30 s reports
`fault`.

Event payloads need `event`. `ts` is optional (epoch seconds or ms, or ISO
8601); if it is missing or more than 5 minutes off the hub's clock, the hub's
receive time is used. Every other field is kept with the event, and fields
listed in the type's `state_fields` also update the device card.

A sensor is shown offline when its status is `offline` or nothing has
arrived from it for `SENSOR_OFFLINE_AFTER_SECONDS` (default 90 s). Messages
from ids that are not in the devices file are not stored as events; the
Devices tab lists them so you can add them.

## 4. Adding a new device type

Register it in `app/devices/types.py`:

```python
register_device_type(
    DeviceType(
        key="door",
        label="Door contact",
        plural="Door contacts",
        transport="mqtt",
        icon="🚪",
        events={"opened": "Door opened", "closed": "Door closed"},
        trigger_events=frozenset({"opened"}),
        event_state={"opened": {"open": True}, "closed": {"open": False}},
        state_fields=("open",),
    )
)
```

Then add devices with `type: door` to the devices file. The bridge, event
store and dashboard pick it up with no other change. A type that is not
registered still works as a generic MQTT device, just without the state
mapping and nicer labels.

## Storage

Events from all device types go to `data/dashboard/events.db` (SQLite, WAL),
`events` table, with `device_id`, `device_type`, `event_type` and a
`data_json` column for sensor fields. Older camera-only databases are
migrated in place on first open. Last known state per device lives in the
`device_state` table. Retention (`EVENT_RETENTION_DAYS`, default 30) is
applied by both the camera publishers and the sensor bridge. JSONL logs per
device are still written to `data/dashboard/events/` as a raw backup.

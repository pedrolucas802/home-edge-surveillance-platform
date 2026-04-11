# Home Edge Surveillance Analytics Platform Runbook

## Purpose

This runbook is the day-to-day operations guide for the `Home Edge Surveillance Analytics Platform`.

For dataset preparation and model fine-tuning, use [docs/TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md).

Use it for:

- starting the local surveillance stack
- running the Streamlit dashboard
- launching one or both cameras
- troubleshooting RTSP / YOLO / dashboard issues
- understanding where runtime artifacts and logs are written

This runbook reflects the current Phase 1 setup:

- local development on MacBook
- RTSP cameras on the same LAN
- FFmpeg-based ingest
- Ultralytics YOLO + tracking
- Streamlit dashboard reading published artifacts

## Current Camera Inventory

- `cam1` -> `192.168.0.90`
- `cam2` -> `192.168.0.89`
- shared camera password via `CAM_PASS`
- default stream: `onvif2`
- transport: `udp`

## Key Files

- `.env`
- `README.md`
- `models/home-surveillance-yolo26m-best.pt`
- `docs/TRAINING_RUNBOOK.md`
- `scripts/run_dashboard_stack.sh`
- `scripts/run_all_cameras.py`
- `app/main.py`
- `app/dashboard/streamlit_app.py`
- `docker-compose.yml`

## Prerequisites

- Python virtual environment at `.venv`
- dependencies installed from `requirements.txt`
- cameras reachable on the LAN
- valid `CAM_PASS` in `.env`
- FFmpeg available

## Environment

Minimum required `.env` values:

```dotenv
CAMERA_ID=cam1
CAMERA_HOSTS=cam1=192.168.0.90,cam2=192.168.0.89
CAM_PASS=your-real-camera-password
CAM_STREAM=onvif2
RTSP_TRANSPORT=udp
YOLO_ENABLED=true
YOLO_MODEL=models/home-surveillance-yolo26m-best.pt
YOLO_CLASSES=person,cat,dog,car,car_plate
YOLO_TRACKING=true
DASHBOARD_ENABLED=true
DASHBOARD_DIR=data/dashboard
CAMERA_PUBLISHERS_ENABLED=true
```

## Standard Startup

### 1. Activate environment

```bash
cd /path/to/your/surveillance-project
source .venv/bin/activate
```

### 2. Start full local stack

```bash
bash scripts/run_dashboard_stack.sh
```

This does:

- starts one headless publisher per configured camera
- writes live frames and event artifacts into `data/dashboard`
- launches the Streamlit dashboard

### 3. Open dashboard

```text
http://localhost:8501
```

## Common Startup Modes

### Single camera preview with window

```bash
.venv/bin/python -m app.main --camera cam1
```

### Single camera, headless publisher only

```bash
.venv/bin/python -m app.main --camera cam1 --headless --dashboard-publish
```

### Both cameras with OpenCV windows

```bash
.venv/bin/python scripts/run_all_cameras.py --enable-yolo --tracking --launch-delay 3.0
```

### Publishers only

```bash
bash scripts/run_dashboard_stack.sh --publishers-only
```

### Dashboard only

```bash
bash scripts/run_dashboard_stack.sh --dashboard-only
```

### Away from the camera LAN

When you are not on the same LAN as the cameras, disable RTSP publishers and just run the dashboard:

```bash
CAMERA_PUBLISHERS_ENABLED=false bash scripts/run_dashboard_stack.sh
```

For Docker:

```bash
CAMERA_PUBLISHERS_ENABLED=false docker compose up
```

Current safe setting for this repo while off-LAN:

- keep `CAMERA_PUBLISHERS_ENABLED=false` in `.env`
- use Docker for dashboard access only
- use local `.venv` if you want Apple `mps` acceleration later when back on the camera LAN
- keep the promoted baseline checkpoint as the default runtime model unless you are intentionally testing a smaller fallback

## Shutdown

### Local stack

- press `Ctrl+C` in the terminal running `scripts/run_dashboard_stack.sh`

### Single preview window

- press `ESC` in the OpenCV window
- or press `Ctrl+C` in the terminal

### Docker stack

```bash
docker compose down
```

### Docker rebuild after dependency changes

```bash
docker compose build --no-cache
docker compose up
```

### Docker startup recommended right now

Since you are currently away from the camera LAN, use detached mode so the dashboard comes up without RTSP workers:

```bash
docker compose down
docker compose up -d --build
docker compose ps
docker compose logs -f dashboard
```

## Data and Logs

### Dashboard artifacts

Written under:

```text
data/dashboard/
```

Important paths:

- `data/dashboard/live/cam1.jpg`
- `data/dashboard/live/cam1.json`
- `data/dashboard/live/cam2.jpg`
- `data/dashboard/live/cam2.json`
- `data/dashboard/events/cam1.jsonl`
- `data/dashboard/events/cam2.jsonl`
- `data/dashboard/snapshots/cam1/`
- `data/dashboard/snapshots/cam2/`

### Launcher logs

Written under:

```text
data/logs/
```

Examples:

- `data/logs/cam1.log`
- `data/logs/cam2.log`

## Health Checks

### Check that config loads

```bash
.venv/bin/python -m app.main --help
```

### Check Streamlit is installed

```bash
.venv/bin/streamlit --version
```

### Check camera publishing output exists

```bash
ls data/dashboard/live
ls data/dashboard/events
```

### Check one camera directly

```bash
.venv/bin/python -m app.main --camera cam1
```

## Troubleshooting

### Problem: `CAM_PASS is required unless RTSP_URL is provided`

Cause:

- password missing from `.env`
- wrong working directory
- `.env` not loaded

Fix:

- confirm `.env` exists in repo root
- confirm `CAM_PASS=...` is set
- run from repo root

### Problem: dashboard opens but shows no live frames

Cause:

- camera publisher is not running
- dashboard artifact files were not created yet
- publisher failed during stream startup
- camera publishers are intentionally disabled because you are away from the LAN

Checks:

```bash
ls data/dashboard/live
ls data/logs
tail -n 50 data/logs/cam1.log
tail -n 50 data/logs/cam2.log
```

Fix:

- start publishers again with `bash scripts/run_dashboard_stack.sh --publishers-only`
- make sure cameras are reachable

### Problem: second camera fails to start

Known current issue:

- dual-camera startup is more fragile than single-camera startup
- this is likely due to WiFi + H.265 + UDP startup sensitivity

Mitigations:

- increase startup staggering:

```bash
.venv/bin/python scripts/run_all_cameras.py --launch-delay 3.0
```

- increase `.env` retry count:

```dotenv
STARTUP_ATTEMPTS=5
```

- prefer headless publishers + dashboard over dual OpenCV windows

### Problem: FFmpeg startup error or no initial frames

Checks:

- confirm camera is online
- confirm `CAM_STREAM=onvif2`
- confirm `RTSP_TRANSPORT=udp`
- confirm password is still valid

Fixes:

- retry single-camera startup first
- keep `SOURCE_FPS=15`
- do not force TCP for this camera family

### Problem: tracking fails with missing `lap`

Fix:

```bash
.venv/bin/python -m pip install "lap>=0.5.12"
```

Then retry:

```bash
.venv/bin/python -m app.main --camera cam1
```

### Problem: dashboard works but detections are poor

Likely causes:

- `onvif2` is low resolution
- cat is too small in frame
- low light or IR mode

What to try:

- increase YOLO image size
- try `onvif1` for evaluation
- save difficult cases for fine-tuning later

### Problem: Docker stack starts but inference is slow

Expected reason:

- Compose forces `YOLO_DEVICE=cpu`
- containers do not use macOS `mps`

What to do:

- use Docker mainly for dashboard/publisher packaging right now
- use local `.venv` runtime for faster Apple Silicon inference

### Problem: I want MPS on my Mac

Important distinction:

- local macOS `.venv` runs can use `YOLO_DEVICE=mps`
- Docker containers on macOS do not provide PyTorch MPS access

Use local host execution for Apple GPU acceleration:

```bash
.venv/bin/python -m app.main --camera cam1
```

Keep Docker for:

- dashboard packaging
- headless CPU publishers
- off-LAN dashboard access to previously collected artifacts

### Problem: Docker build appears stuck on `pip install -r requirements.txt`

Likely cause:

- PyTorch is downloading large Linux wheels
- older builds may also try to pull NVIDIA CUDA packages that are not useful for this project

What changed:

- the Dockerfile now preinstalls CPU-only PyTorch before `ultralytics`
- Compose reuses a single built image for both services
- the container image excludes `onvif-zeep`, which is only needed for local discovery scripts and was slowing or breaking runtime image builds

Recommended retry:

```bash
docker compose build --no-cache
docker compose up
```

## Recovery Procedures

### Restart local stack cleanly

```bash
pkill -f "app.main" || true
pkill -f "streamlit run app/dashboard/streamlit_app.py" || true
bash scripts/run_dashboard_stack.sh
```

### Clear dashboard artifacts

Only do this when you want a fresh dashboard state.

```bash
rm -rf data/dashboard/live/*
rm -rf data/dashboard/events/*
rm -rf data/dashboard/snapshots/cam1/*
rm -rf data/dashboard/snapshots/cam2/*
```

### Reinstall Python dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Recommended Operational Flow

For normal local use:

1. Start with `bash scripts/run_dashboard_stack.sh`
2. Open Streamlit dashboard
3. Confirm both camera publishers are producing live frames
4. Check `data/logs/*.log` if one camera is missing
5. Use single-camera preview only when debugging stream behavior

## Known Limitations

- multi-camera startup can still be flaky on this WiFi/H.265 camera family
- event storage is file-based, not yet a real database
- dashboard is reading published artifacts, not a proper API backend
- Docker path is operationally useful, but local Apple Silicon runtime is still better for inference performance

## Next Operational Improvements

- add process supervisor / watchdog behavior
- add explicit health endpoint or heartbeat JSON
- add event retention and cleanup policy
- add SQLite-backed event storage
- add dashboard filters by camera, class, and time range

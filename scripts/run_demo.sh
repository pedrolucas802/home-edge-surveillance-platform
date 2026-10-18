#!/usr/bin/env bash
# Run the hub with mock hardware: 3 cameras, 4 RD-03D radars, 4 tripwires.
# Uses its own devices file and data folder, so real data is untouched.
#
#   bash scripts/run_demo.sh            # sensors written straight to the store
#   bash scripts/run_demo.sh --mqtt     # through Mosquitto + the sensor bridge
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export DEVICES_CONFIG="${DEVICES_CONFIG:-configs/devices.demo.yaml}"
export DASHBOARD_DIR="${DASHBOARD_DIR:-data/demo}"
export CAMERA_PUBLISHERS_ENABLED=false

TRANSPORT="direct"
if [[ "${1:-}" == "--mqtt" ]]; then
  TRANSPORT="mqtt"
fi

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="$(command -v python3 || command -v python)"
STREAMLIT_BIN="${ROOT_DIR}/.venv/bin/streamlit"
[[ -x "${STREAMLIT_BIN}" ]] || STREAMLIT_BIN="$(command -v streamlit)"

LOG_DIR="${ROOT_DIR}/data/logs"
mkdir -p "${LOG_DIR}"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "${TRANSPORT}" == "mqtt" ]]; then
  echo "Starting sensor bridge (log: data/logs/demo-bridge.log). Needs Mosquitto on ${MQTT_HOST:-localhost}:${MQTT_PORT:-1883}."
  "${PYTHON_BIN}" -m app.sensors.bridge >>"${LOG_DIR}/demo-bridge.log" 2>&1 &
  PIDS+=("$!")
  sleep 1
fi

echo "Starting mock devices (log: data/logs/demo.log)"
"${PYTHON_BIN}" -m app.demo --transport "${TRANSPORT}" >>"${LOG_DIR}/demo.log" 2>&1 &
PIDS+=("$!")

echo "Dashboard: http://localhost:8501"
"${STREAMLIT_BIN}" run app/dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  echo "Missing .env at ${ROOT_DIR}/.env. Create it from .env.example first." >&2
  exit 2
fi

export CAMERA_PUBLISHERS_ENABLED="${CAMERA_PUBLISHERS_ENABLED:-true}"
export YOLO_DEVICE="${YOLO_DEVICE:-mps}"

echo "Starting full surveillance stack..."
echo "CAMERA_PUBLISHERS_ENABLED=${CAMERA_PUBLISHERS_ENABLED}"
if [[ -n "${RTSP_TRANSPORT:-}" ]]; then
  echo "RTSP_TRANSPORT=${RTSP_TRANSPORT} (environment override)"
else
  echo "RTSP_TRANSPORT=<from .env>"
fi
echo "YOLO_DEVICE=${YOLO_DEVICE}"
echo "Dashboard URL: http://localhost:8501"

exec bash scripts/run_dashboard_stack.sh "$@"

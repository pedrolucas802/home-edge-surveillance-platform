#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODE="all"
LAUNCH_DELAY="${LAUNCH_DELAY:-3}"
CAMERA_IDS=()
PIDS=()
CAMERA_PUBLISHERS_ENABLED="${CAMERA_PUBLISHERS_ENABLED:-}"

read_env_value() {
  local key="$1"
  local env_file="${ROOT_DIR}/.env"
  if [[ ! -f "${env_file}" ]]; then
    return 1
  fi

  local line
  line="$(grep -E "^${key}=" "${env_file}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    return 1
  fi

  line="${line#*=}"
  line="${line%\"}"
  line="${line#\"}"
  line="${line%\'}"
  line="${line#\'}"
  printf '%s' "${line}"
}

if [[ -z "${CAMERA_PUBLISHERS_ENABLED}" ]]; then
  CAMERA_PUBLISHERS_ENABLED="$(read_env_value CAMERA_PUBLISHERS_ENABLED || true)"
fi
CAMERA_PUBLISHERS_ENABLED="${CAMERA_PUBLISHERS_ENABLED:-true}"

usage() {
  cat <<'EOF'
Usage: scripts/run_dashboard_stack.sh [--publishers-only | --dashboard-only] [--launch-delay SECONDS] [--cameras cam1 cam2 ...]

Default mode launches all configured camera publishers in headless mode and then starts the Streamlit dashboard.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --publishers-only)
      MODE="publishers"
      shift
      ;;
    --dashboard-only)
      MODE="dashboard"
      shift
      ;;
    --launch-delay)
      LAUNCH_DELAY="${2:?Missing value for --launch-delay}"
      shift 2
      ;;
    --cameras)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CAMERA_IDS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

STREAMLIT_BIN="${ROOT_DIR}/.venv/bin/streamlit"
if [[ ! -x "${STREAMLIT_BIN}" ]]; then
  STREAMLIT_BIN="$(command -v streamlit)"
fi

if [[ ${#CAMERA_IDS[@]} -eq 0 ]]; then
  camera_output="$("${PYTHON_BIN}" -c 'from app.config.settings import Settings; s = Settings.from_env(); ids = list(s.camera_hosts_map) or [s.camera_id]; print("\n".join(ids))')"
  while IFS= read -r camera_id; do
    if [[ -n "${camera_id}" ]]; then
      CAMERA_IDS+=("${camera_id}")
    fi
  done <<< "${camera_output}"
fi

LOG_DIR="${ROOT_DIR}/data/logs"
mkdir -p "${LOG_DIR}"

cleanup() {
  if [[ ${#PIDS[@]} -eq 0 ]]; then
    return
  fi

  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}

wait_for_publishers() {
  while true; do
    for pid in "${PIDS[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "A camera publisher exited unexpectedly. Check data/logs for details." >&2
        return 1
      fi
    done
    sleep 2
  done
}

start_publishers() {
  if [[ "${CAMERA_PUBLISHERS_ENABLED}" != "true" ]]; then
    echo "Camera publishers are disabled. Starting dashboard without RTSP camera workers."
    return
  fi

  local camera_id
  for camera_id in "${CAMERA_IDS[@]}"; do
    local log_file="${LOG_DIR}/${camera_id}.log"
    echo "Starting headless publisher for ${camera_id} (log: ${log_file})"
    "${PYTHON_BIN}" -m app.main \
      --camera "${camera_id}" \
      --headless \
      --dashboard-publish \
      >>"${log_file}" 2>&1 &
    PIDS+=("$!")
    sleep "${LAUNCH_DELAY}"
  done
}

trap cleanup EXIT INT TERM

case "${MODE}" in
  publishers)
    if [[ "${CAMERA_PUBLISHERS_ENABLED}" != "true" ]]; then
      echo "Camera publishers are disabled. Idling publisher container."
      exec sleep infinity
    fi
    start_publishers
    wait_for_publishers
    ;;
  dashboard)
    exec "${STREAMLIT_BIN}" run app/dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
    ;;
  all)
    start_publishers
    "${STREAMLIT_BIN}" run app/dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
    ;;
esac

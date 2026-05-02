#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -gt 0 ]]; then
  LOG_FILES=("$@")
else
  LOG_FILES=(
    "data/logs/publishers-supervisor.log"
    "data/logs/cam1.log"
    "data/logs/cam2.log"
  )
fi

echo "Tailing important log lines..."
echo "Files: ${LOG_FILES[*]}"

tail -F "${LOG_FILES[@]}" 2>/dev/null \
  | rg --line-buffered -i "error|warning|runtimeerror|exited unexpectedly|starting surveillance preview|yolo enabled|received sig" \
  | rg --line-buffered -v "use_container_width|^\s*raise RuntimeError\(" \
  | awk '
    {
      if ($0 == prev) {
        count += 1
        next
      }
      if (count > 0) {
        print prev " [repeated " count "x]"
        count = 0
      }
      print
      prev = $0
    }
    END {
      if (count > 0) {
        print prev " [repeated " count "x]"
      }
    }
  '

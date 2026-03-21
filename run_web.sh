#!/usr/bin/env bash
set -euo pipefail

BIND_HOST="127.0.0.1"
BACKEND_PORT=8000
FRONTEND_PORT=3000
RUN_EXPERIMENT=0
RUN_ID_SUFFIX="main"
DEVICE="cpu"
VERBOSE=0
BACKEND_ONLY=0
FRONTEND_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bind-host) BIND_HOST="${2:-}"; shift 2 ;;
    --backend-port) BACKEND_PORT="${2:-8000}"; shift 2 ;;
    --frontend-port) FRONTEND_PORT="${2:-3000}"; shift 2 ;;
    --run-experiment) RUN_EXPERIMENT=1; shift ;;
    --device) DEVICE="${2:-cpu}"; shift 2 ;;
    --run-id-suffix) RUN_ID_SUFFIX="${2:-main}"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    --backend-only) BACKEND_ONLY=1; shift ;;
    --frontend-only) FRONTEND_ONLY=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ "${FRONTEND_ONLY}" -eq 0 ]]; then
  BACKEND_ARGS=(web_main.py --host "$BIND_HOST" --port "$BACKEND_PORT")
  if [[ "${RUN_EXPERIMENT}" -eq 1 ]]; then
    BACKEND_ARGS+=(--run-experiment --run-id-suffix "$RUN_ID_SUFFIX" --device "$DEVICE")
  fi
  if [[ "${VERBOSE}" -eq 1 ]]; then
    BACKEND_ARGS+=(--verbose)
  fi
  echo "[run_web] Starting backend..."
  poetry run python "${BACKEND_ARGS[@]}" &
fi

if [[ "${BACKEND_ONLY}" -eq 0 ]]; then
  echo "[run_web] Starting frontend..."
  (
    cd www
    if [[ "${FRONTEND_PORT}" != "3000" ]]; then
      PORT="${FRONTEND_PORT}" npm run dev
    else
      npm run dev
    fi
  )
fi

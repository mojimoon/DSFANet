#!/usr/bin/env bash
set -euo pipefail

CUDA="cu130"
PYTHON_VER="3.13"
SKIP_TORCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_VER="${2:-}"
      shift 2
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "[setup] Creating/using Poetry environment (Python ${PYTHON_VER}) ..."
poetry env use "${PYTHON_VER}"

echo "[setup] Installing Python dependencies via Poetry ..."
poetry install

if [[ "${SKIP_TORCH}" -eq 0 ]]; then
  if [[ "${CUDA}" == "cpu" ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cpu"
  else
    INDEX_URL="https://download.pytorch.org/whl/${CUDA}"
  fi

  echo "[setup] Installing torch from ${INDEX_URL} ..."
  poetry run pip install --index-url "${INDEX_URL}" torch

  echo "[setup] Verifying torch backend ..."
  poetry run python -c 'import torch; print(torch.__version__); print(torch.version.cuda)'
fi

echo "[setup] Installing frontend dependencies ..."
(
  cd www
  npm install
)

echo "[setup] Done."

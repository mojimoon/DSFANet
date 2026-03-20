#!/usr/bin/env bash
set -euo pipefail

SINGLE=0
RUN_ID="unsw-main"
RUN_ID_SUFFIX="main"
BASE_DATASET="NF-UNSW-NB15-v3.csv"
DEVICE="cuda"
STEPS="1,2,3,4,5,6,7,8"
EPOCHS="10,10,20"
SIZE_LIMIT=0
OOD_DATASET="NF-BoT-IoT-v3.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --single) SINGLE=1; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --run-id-suffix) RUN_ID_SUFFIX="${2:-}"; shift 2 ;;
    --base-dataset) BASE_DATASET="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --size-limit) SIZE_LIMIT="${2:-0}"; shift 2 ;;
    --ood-dataset) OOD_DATASET="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

run_one() {
  local run_id="$1"
  local dataset="$2"

  local args=(
    experiments_main.py
    --run-id "$run_id"
    --steps "$STEPS"
    --epochs "$EPOCHS"
    --base-dataset "$dataset"
    --ood-dataset "$OOD_DATASET"
    --device "$DEVICE"
  )

  if [[ "${SIZE_LIMIT}" -gt 0 ]]; then
    args+=(
      --test-size "$SIZE_LIMIT"
      --max-train-samples "$SIZE_LIMIT"
      --drift-subset-size "$SIZE_LIMIT"
      --natural-shift-size "$SIZE_LIMIT"
      --max-benign-for-attacks "$SIZE_LIMIT"
      --step5-train-max-samples "$SIZE_LIMIT"
      --step5-eval-max-samples "$SIZE_LIMIT"
      --step6-val-max-samples "$SIZE_LIMIT"
      --step6-eval-max-samples "$SIZE_LIMIT"
    )
  fi

  echo "[run_experiments] Running: ${run_id} (${dataset})"
  poetry run python "${args[@]}"
}

if [[ "${SINGLE}" -eq 1 ]]; then
  run_one "$RUN_ID" "$BASE_DATASET"
else
  run_one "unsw-${RUN_ID_SUFFIX}" "NF-UNSW-NB15-v3.csv"
  run_one "ton-${RUN_ID_SUFFIX}" "NF-ToN-IoT-v3.csv"
  run_one "ids2018-${RUN_ID_SUFFIX}" "NF-CICIDS2018-v3.csv"
fi

#!/usr/bin/env bash
# Train baseline -> predict -> (optionally) submit.
# Usage:
#   bash scripts/run_baseline.sh                       # train + predict
#   SUBMIT=1 bash scripts/run_baseline.sh "msg"        # also submit to Kaggle
set -euo pipefail

CONFIG="${CONFIG:-configs/baseline.yaml}"
COMP="the-freuid-challenge-2026-ijcai-ecai"
MSG="${1:-baseline $(date +%F_%T)}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fraud

echo "[run] training with ${CONFIG}"
RUN_DIR=$(python -m freuid.train --config "${CONFIG}" | tail -1)
echo "[run] run_dir=${RUN_DIR}"

echo "[run] predicting"
SUB=$(python -m freuid.predict --config "${CONFIG}" --run "${RUN_DIR}" | tail -1)
echo "[run] submission=${SUB}"

if [[ "${SUBMIT:-0}" == "1" ]]; then
  echo "[run] submitting to Kaggle: ${MSG}"
  kaggle competitions submit "${COMP}" -f "${SUB}" -m "${MSG}"
else
  echo "[run] dry run (set SUBMIT=1 to upload). file ready: ${SUB}"
fi

#!/usr/bin/env bash
# Fast pipeline check: tiny image size, 1 epoch, few workers. Verifies train+predict end-to-end.
# Usage: bash scripts/smoke_test.sh
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fraud

RUN_DIR=$(python -m freuid.train --config configs/baseline.yaml --set \
  exp_name=smoke train.epochs=1 train.warmup_epochs=0 data.img_size=128 \
  data.batch_size=32 data.num_workers=4 train.log_every=20 | tail -1)
echo "[smoke] run_dir=${RUN_DIR}"
python -m freuid.predict --config configs/baseline.yaml --run "${RUN_DIR}"
echo "[smoke] OK"

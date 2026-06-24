#!/usr/bin/env bash
# Set up the `fraud` conda env for the FREUID baseline.
# Usage: bash scripts/setup_env.sh
set -euo pipefail

ENV_NAME="fraud"
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "[setup] creating conda env '${ENV_NAME}' (python 3.11)"
  conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"

# RTX 3080 -> CUDA 12.1 wheels
echo "[setup] installing torch (cu121)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[setup] installing project requirements"
pip install -r requirements.txt

python - <<'PY'
import torch, timm
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
print("timm", timm.__version__)
PY
echo "[setup] done. activate with: conda activate ${ENV_NAME}"

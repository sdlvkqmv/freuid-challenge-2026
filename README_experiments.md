# FREUID Baseline — Experiment Setup

Clean, reproducible image-classification baseline for binary ID-document fraud detection.

## Layout

```
configs/baseline.yaml     # all hyperparameters (single source of truth)
freuid/                   # package
  config.py               # YAML load + dotted CLI overrides + resolved-config dump
  data.py                 # dataset, transforms, stratified / group-holdout split
  model.py                # timm backbone, single-logit fraud head
  metrics.py              # FREUID score = 1 - HM(1-AuDET, 1-APCER@1%BPCER)
  train.py                # train loop (AMP, cosine+warmup, early stop)
  predict.py              # inference -> Kaggle submission.csv
  utils.py                # seed, logging, git hash, run-dir
scripts/
  setup_env.sh            # conda `fraud` env + torch(cu121) + requirements
  run_baseline.sh         # train -> predict -> (optional) submit
  smoke_test.sh           # 1-epoch tiny-res end-to-end check
experiments/<run>/        # per-run outputs (gitignored)
```

Each run folder is self-contained and inspectable:
`config.yaml` (resolved) · `meta.json` (git hash, split stats, val doc-types) ·
`train.log` · `metrics.json` (best + per-epoch history) · `oof_val.csv` · `checkpoints/best.pt` · `submission.csv`.

## Setup (once)

```bash
bash scripts/setup_env.sh
conda activate fraud
```

## Run

```bash
# verify pipeline end-to-end (~minutes)
bash scripts/smoke_test.sh

# full baseline (train + build submission, no upload)
bash scripts/run_baseline.sh

# train + upload to Kaggle
SUBMIT=1 bash scripts/run_baseline.sh "effb0 384 5ep stratified f0"
```

Override anything without editing the YAML:

```bash
python -m freuid.train --config configs/baseline.yaml \
  --set model.name=convnext_tiny data.img_size=512 train.epochs=10 gpu=1
```

## Key design notes

- **Metric** mirrors the competition: positive class = fraud (label 1); lower FREUID is better.
  Model selection saves the checkpoint with the lowest validation FREUID.
- **Validation schemes** (`val.scheme`):
  - `stratified` — StratifiedKFold on `(type, label)`; in-distribution estimate.
  - `group_holdout` — hold out whole `type`(s) via `val.holdout_types`; simulates the
    private leaderboard's **unseen document types** (OOD). Recommended as a second check.
- **Test gap**: only `public_test` images (7,821) exist now; `sample_submission.csv` lists
  142,818 ids. `predict.py` scores available images and fills the rest with `predict.fill_value`.
  Re-run prediction once the full test set is released.
- **Class imbalance**: `train.pos_weight=null` auto-sets `n_neg/n_pos` in BCE.
- **Augmentation**: `data.hflip=false` by default — horizontal flip mirrors document text.
- **Data paths**: handles the doubled directory layout (`train/train/<id>.jpeg`).

## Reproducibility

`seed`, resolved `config.yaml`, and git short-hash (`meta.json`) are recorded per run.
Set `data.num_workers=0` for fully deterministic data order if needed.

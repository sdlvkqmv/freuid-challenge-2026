# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Workspace for the **FREUID Challenge 2026** Kaggle competition (IJCAI-ECAI) — binary detection of forged/manipulated identity documents. Lower competition score is better. There is **no model code yet**: the repo currently holds only the competition data (gitignored) and a Korean participation guide (`FREUID_Challenge_2026_가이드.md`). New training/inference code is expected to be added here.

- Competition: https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai
- Submission deadline: **2026-07-14 11:59 UTC**, max **5 submissions/day**.
- Final deliverables: predictions + OSI-licensed source code + technical report.

## Environment & CLI

Use the conda env `fraud` for all work (`conda activate fraud`). The Kaggle CLI is the interface for data and submissions:

```bash
# download full data / a single file
kaggle competitions download the-freuid-challenge-2026-ijcai-ecai
kaggle competitions download the-freuid-challenge-2026-ijcai-ecai -f sample_submission.csv

# submit predictions, then check status / leaderboard
kaggle competitions submit the-freuid-challenge-2026-ijcai-ecai -f submission.csv -m "message"
kaggle competitions submissions the-freuid-challenge-2026-ijcai-ecai
kaggle competitions leaderboard the-freuid-challenge-2026-ijcai-ecai -s
```

## Data layout

Everything under `data/` is gitignored (large). Extracted structure:

```
data/extracted/
  train/train/<hash>.jpeg              # 69,352 training images
  train_sample/train_sample/<hash>.jpeg # 13 demo images
  public_test/public_test/<hash>.jpeg  # 7,821 test images
  train_labels.csv                     # 69,352 rows
  train_sample_labels.csv              # 13 rows
  sample_submission.csv                # 142,818 ids
```

Note the **doubled directory name** (`train/train/`, `public_test/public_test/`) — build paths accordingly. The image filename `<hash>` matches the CSV `id` (no extension).

### Label schema (`train_labels.csv`)

`id, image_path, label, is_digital, type`

- `label` — **1 = fraud, 0 = bona-fide** (target).
- `is_digital` — `True`/`False`; digital edit vs print-and-capture origin. Useful stratification axis.
- `type` — document class as `COUNTRY/DOCTYPE` (e.g. `EGYPT/DL`, `MAURITIUS/ID`). 7 doc types across Asian/African countries, Latin + Arabic scripts.

The forgery families (physical manipulation, GenAI multimodal edits, print-and-capture) deliberately blur domain boundaries — `is_digital` and `type` are the levers for stratified validation.

## Submission format

`sample_submission.csv` → `id,label` where `label` is a **continuous fraud score/probability** (higher = more likely fraud). The metric (FREUID score = harmonic-mean combination of AuDET and APCER@1%BPCER) needs the full DET curve, so output calibrated continuous scores, not hard 0/1 labels.

⚠️ `sample_submission.csv` lists **142,818 ids** but `public_test/` has only **7,821 images** — verify which id set a submission must cover before generating predictions (re-download / re-check the test set rather than assuming).

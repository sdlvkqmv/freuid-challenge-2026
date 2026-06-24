# FREUID 2026 — Experiment Summary (entry point)

> Always read this first. Scoreboard + one-line conclusions + drill-down links.
> Update this table every session. New experiment → add a row here + write `attempts/NN_*.md`.

## Problem

Binary forged-ID detection. **Lower competition score is better.**
Metric = **FREUID score** = `1 - HM(1-AuDET, 1-APCER@1%BPCER)` (DET-curve based, threshold-swept).

- Train: 69,352 images, labels `id,image_path,label,is_digital,type`. label 1=fraud.
- 5 doc types: EGYPT/DL, GUINEA/DL, BENIN/DL, MOZAMBIQUE/DL (DL) + MAURITIUS/ID.
- `is_digital`: 69,332 True / 20 False (train is ~all digital).
- Test: public_test has **7,821 images** but `sample_submission.csv` lists **142,818 ids**
  → 134,997 ids have no image yet (full test set staged). Submit real preds for 7,821, fill rest.
- **Private LB adds 2 unseen doc types** → OOD generalization is the real objective (research §1,§6).

## Root findings (highest-confidence)

0. **🔴 THE PROXY IS BROKEN — train/test domain gap is the whole game.**
   In-domain stratified val FREUID **0.00011** → Kaggle public LB **0.18471** (1700× worse).
   Mechanism: **train is 69,332 digital / only 20 recaptured** (`is_digital=False`), but the
   public/private test **emphasizes print-and-capture (recaptured) examples** (research §4 +
   dataset desc). The model never sees the recapture domain → collapses on it. Near-perfect
   in-domain numbers are a shortcut on the digital domain, not fraud understanding.
   **⇒ Stratified CV is worthless as a proxy. Must validate on a recapture/OOD holdout.**
   (LB top ~0.00063; our 0.18 is far down.) → [[eval_harness]], [[attempts/02_effb3_srm]]

1. **Calibration is a no-op for this metric.** AuDET and APCER@1%BPCER are computed by
   sweeping thresholds over submitted scores → any *monotonic* transform (isotonic/Platt/
   temperature) leaves both unchanged. The research doc's "calibration is the bottleneck"
   applies to fixed-0.5-threshold F1 benchmarks, NOT to DET-based AuDET. **The lever is
   ranking quality + behavior near the 1%BPCER operating point, not calibration.** → [[eval_harness]]
2. **Generalization gap is THE problem** (research §1): models hit ~100% in-domain, collapse
   to ~50% cross-type. Private LB has unseen types → validate with `group_holdout`, not only
   in-domain stratified CV.
3. **EfficientNet-B3 > ViT** on ID forgery (research §1, SIDTD 0.994 vs 0.552). Started here.
4. **Noise-residual streams add tamper signal** RGB misses; multi-stream beats either (§2,§6).

## Scoreboard (local proxy = val FREUID, stratified fold0; lower better)

| Rank | Attempt | val FREUID (in-domain) | Kaggle public LB | Notes |
|---|---|---|---|---|
| ✅ submitted | 02 effb3 rgb+srm | 0.00011 | **0.18471** | best in-domain; **collapsed on real test** |
| — | 01 effb3 rgb       | 0.00013 | (not submitted) | doc baseline; OOF tied w/ 02 |
| — | 03 convnext rgb    | 0.18008 | — | **under-converged** (loss 0.33, lr too high); excluded |
| — | 04 ensemble (rank) | 01+02=0.000117; +03=0.000190 | — | fusion **no in-domain gain** (saturated); convnext hurts |

**val ≠ LB by 1700× (0.00011 vs 0.18471)** — the proxy is broken (finding #0). The single
submission burned tells us: this domain gap, not model capacity, is the bottleneck.
SRM marginally beat RGB in-domain (AuDET 5.9e-5 vs 9.2e-5) but that signal is untrustworthy now.

## Local proxy formula

```
FREUID = 1 - 2*g1*g2/(g1+g2),  g1 = 1-AuDET,  g2 = 1-APCER@1%BPCER
```
Computed by `freuid/metrics.py` on held-out val. Selection: lowest val FREUID. → [[eval_harness]]

## Attempts (drill-down)

| # | File | Status |
|---|---|---|
| 01 | [effb3 rgb baseline](attempts/01_effb3_rgb.md) | running |
| 02 | [effb3 + SRM noise stream](attempts/02_effb3_srm.md) | running |
| 03 | [convnext_tiny rgb](attempts/03_convnext_rgb.md) | running |
| 04 | [rank-fusion ensemble](attempts/04_ensemble.md) | pending |

## Remaining directions (re-prioritized after finding #0)

**Top priority — close the digital→recapture domain gap (this is what lost 0.18 vs 0.0001):**
1. **Heavy print-recapture / JPEG / moiré / resize augmentation** on the all-digital train set so
   the model sees recapture-like artifacts (research §4). Cheapest lever against the gap.
2. **Build a recapture/OOD validation holdout** so the proxy tracks LB. Options: the 20
   `is_digital=False` train rows as a (tiny) probe; or hold out a whole doc `type`
   (`val.scheme=group_holdout`) to at least catch cross-type collapse. Current stratified CV lies.
3. **Forensic streams that target recapture** (NoisePrint++/PRNU camera-fingerprint, DCT double-JPEG)
   — recapture = a second capture pipeline, exactly what these detect (§2, §4).

**Then (capacity / standard):**
- DCT/JPEG-artifact stream (CAT-Net) · ROI face/text crops (YOLO, §6 biggest single jump) ·
  diffusion reconstruction-error branch (DIRE/FIRE, §3 — GenAI family).
- Re-converge convnext (lower lr / layer-decay) only if a *trustworthy* proxy says diversity helps.

## Pointers

- Research survey: [[research/forgery_detection]]
- Eval harness detail: [[eval_harness]]
- Code: `freuid/` (train/predict/ensemble), configs `configs/baseline.yaml`, runner `scripts/run_baseline.sh`.
- Run outputs: `experiments/<name>_<ts>/` (config.yaml, meta.json, metrics.json, oof_val.csv, checkpoints/best.pt, submission.csv).

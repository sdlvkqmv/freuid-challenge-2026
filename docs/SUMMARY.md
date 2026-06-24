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

1. **🟢 SRM × recapture synergy (best lever so far).** Recapture augmentation improved the SRM
   model (0.18471→**0.15185**) but *hurt* plain RGB (0.17920→0.18433). Neither piece alone helps;
   together they do. Interpretation: SRM noise-residual cues are only discriminative once the input
   distribution includes recapture artifacts, and recapture aug needs a forensic stream to be
   exploited. → push this combo (stronger/longer). [[attempts/06_effb3_srm_recap]]

2. **Calibration is a no-op for this metric.** AuDET and APCER@1%BPCER are computed by
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

| Rank (by LB) | Attempt | val FREUID (in-domain) | recap-val | **Kaggle public LB** | Notes |
|---|---|---|---|---|---|
| 🥇 1 | 06 effb3 srm + recapture | 0.00018 | 0.00042 | **0.15185** | **best — SRM×recapture synergy** |
| 2 | 01 effb3 rgb       | 0.00013 | — | 0.17920 | prior best |
| 3 | 05 effb3 rgb + recapture | 0.00012 | 0.00022 | 0.18433 | recapture **didn't help RGB** |
| 4 | 02 effb3 rgb+srm   | 0.00011 | — | 0.18471 | SRM w/o recapture: no gain |
| 5 | 03 convnext rgb    | 0.18008 | — | 0.35407 | under-converged |
| — | 04 ensemble (rank) | 01+02=0.000117 | — | (not submitted) | no in-domain gain |

**Best 0.17920 → 0.15185 (~15% rel) via recapture augmentation on the SRM stream.**
- **Recapture aug helps SRM, not RGB**: 02→06 (0.18471→0.15185) improved; 01→05 (0.17920→0.18433)
  regressed. **Synergy (root finding #1):** the noise stream only pays off once the model sees the
  recapture domain; recapture aug only pays off if there's a forensic stream to exploit it.
- Proxies still mis-rank at fine grain: in-domain rank 02<05<01<06, recap-val rank 05<06 — but LB
  rank 06<01<05<02. **Coarsely** the recapture *family* won; absolute proxy values remain untrustworthy.
- Still far from LB top ~0.00063 → keep closing the domain gap.

## Local proxy formula

```
FREUID = 1 - 2*g1*g2/(g1+g2),  g1 = 1-AuDET,  g2 = 1-APCER@1%BPCER
```
Computed by `freuid/metrics.py` on held-out val. Selection: lowest val FREUID. → [[eval_harness]]

## Attempts (drill-down)

| # | File | Status |
|---|---|---|
| 01 | [effb3 rgb baseline](attempts/01_effb3_rgb.md) | done · LB 0.17920 |
| 02 | [effb3 + SRM noise stream](attempts/02_effb3_srm.md) | done · LB 0.18471 |
| 03 | [convnext_tiny rgb](attempts/03_convnext_rgb.md) | done · LB 0.35407 |
| 04 | [rank-fusion ensemble](attempts/04_ensemble.md) | not submitted |
| 05 | [effb3 rgb + recapture](attempts/05_effb3_rgb_recap.md) | done · LB 0.18433 |
| 06 | [effb3 SRM + recapture](attempts/06_effb3_srm_recap.md) | **done · LB 0.15185 🥇** |

## Remaining directions (re-prioritized after finding #0)

**✅ DONE: recapture augmentation → new best 0.15185 (attempt 06, SRM stream only).**

**Top priority — push the SRM×recapture winner (finding #1):**
1. **Stronger / longer recapture training** on the SRM model: more epochs, higher recapture `prob`,
   wider artifact ranges, maybe higher resolution. 06 was only 6 epochs / 384px.
2. **Add a DCT / JPEG-artifact stream** (CAT-Net, §2) alongside RGB+SRM — double-JPEG is a core
   recapture signature; complementary to SRM.
3. **A real recapture validation probe** so the proxy stops lying: use the 20 `is_digital=False`
   train rows held out, and/or `group_holdout` by doc type. recap-sim proxy is circular.

**Then:**
- ROI face/text crops (YOLO, §6 biggest single jump) · diffusion reconstruction-error branch
  (DIRE/FIRE, §3 GenAI family) · ensemble of diverse recapture-trained streams.
- Re-converge convnext (lower lr / layer-decay) only if a trustworthy proxy says diversity helps.

## Pointers

- Research survey: [[research/forgery_detection]]
- Eval harness detail: [[eval_harness]]
- Code: `freuid/` (train/predict/ensemble), configs `configs/baseline.yaml`, runner `scripts/run_baseline.sh`.
- Run outputs: `experiments/<name>_<ts>/` (config.yaml, meta.json, metrics.json, oof_val.csv, checkpoints/best.pt, submission.csv).

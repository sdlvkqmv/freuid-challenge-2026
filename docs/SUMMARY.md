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
| 🥇 1 | 06 effb3 srm + recapture | 0.00018 | 0.00042 | **0.15185** | **best — SRM×recapture synergy, moderate aug** |
| 2 | 10 fusion 2×06 + 1×07b | — | — | 0.15564 | fusion < 06 (07b correlated, weaker) |
| 3 | 07b srm+recap 448 hi-res | 0.00022 | 0.00032 | 0.16440 | hi-res didn't help |
| 4 | 01 effb3 rgb       | 0.00013 | — | 0.17920 | prior best |
| 5 | 05 effb3 rgb + recapture | 0.00012 | 0.00022 | 0.18433 | recapture **didn't help RGB** |
| 6 | 02 effb3 rgb+srm   | 0.00011 | — | 0.18471 | SRM w/o recapture: no gain |
| 7 | 07a srm+recap STRONG/WIDE | 0.00022 | 0.00058 | 0.19546 | **over-aug (prob0.85) hurts** |
| 8 | 08 rgb+srm+dct (+strong aug) | 0.00032 | 0.00063 | 0.24078 | DCT confounded by bad aug → see 09 |
| 9 | 09 rgb+srm+dct FAIR (06 aug) | 0.00011 | 0.00021 | 0.21476 | **DCT stream genuinely hurts** (clean test) |
| 10 | 03 convnext rgb    | 0.18008 | — | 0.35407 | under-converged |
| — | 04 ensemble (rank) | 01+02=0.000117 | — | (not submitted) | no in-domain gain |

**Best still 0.15185 (attempt 06).** Session 2 (2026-06-25), all 5 subs spent, **everything regressed**:
stronger/wider aug (07a 0.19546), hi-res (07b 0.16440), DCT+bad-aug (08 0.24078), DCT-fair (09 0.21476),
fusion (10 0.15564). → **06's moderate-aug SRM+recapture is a local optimum; neither hyperparameter
pushes nor extra pixel streams (DCT) beat it. Next lever must change the signal family —
field-consistency D / ROI crops.**
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
| 07 | [push SRM×recapture (harder aug + hi-res)](attempts/07_srm_recap_push.md) | done · 07a 0.19546 / 07b 0.16440 |
| 08 | [DCT block-frequency stream](attempts/08_dct_stream.md) | done · LB 0.24078 (confounded) |
| 09 | [DCT-fair (06 moderate aug)](attempts/09_dct_fair.md) | pending |
| 10 | [rank-fusion 06×07b](attempts/10_ensemble_06_07b.md) | done · LB 0.15564 |

## Remaining directions (re-prioritized after finding #0)

**✅ DONE: recapture augmentation → new best 0.15185 (attempt 06, SRM stream only).**

**❌ EXHAUSTED (session 2, all regressed vs 06) — parameter pushes don't beat the winner:**
1. ~~Stronger/longer/wider recapture aug~~ → 07a 0.19546 (over-aug HURTS), 07b 448px 0.16440 (hi-res
   neutral-worse). **06's moderate prob-0.7 aug is a local optimum.** [[attempts/07_srm_recap_push]]
2. ~~DCT/JPEG stream~~ → 08 0.24078 (confounded), **09 FAIR test 0.21476** with 06's moderate aug →
   **DCT stream genuinely hurts, dropped.** SRM is the forensic stream that works. [[attempts/09_dct_fair]]
3. ~~Real recapture validation probe~~ → built (`freuid/probe.py`) but **also broken**: n=20 too
   small for FREUID, AUROC inverted vs LB. No local proxy ranks at the top. [[eval_harness]]

**⇒ Next must change the SIGNAL FAMILY, not the knobs. Tomorrow's order ([[research/directions]] F–J):**
1. **G — test-time BN / Tent** (cheapest, hours): recompute BN stats / entropy-min on the unlabeled
   test batch to re-align 06 to the recapture domain. No retraining, bolt onto the 06 checkpoint.
   Free insurance — do first.
2. **F — frozen foundation features** (biggest OOD upside): frozen CLIP-ViT-L/14 or DINOv2-L +
   light head (linear probe → LoRA). Precedent (UnivFD) literally generalizes to *unseen* generators
   = the private-LB objective. ~1 day.
3. **D — field-consistency / MRZ checks** (strongest orthogonal net-new, domain-gap-robust); fuse late.
4. **I — operating-point hard-negative mining** (metric squeeze on APCER@1%BPCER) once a backbone wins.
5. **H / J** (one-class on bona-fide · domain-adversarial) — hedges for unseen-type private LB if F/G plateau.

**Then:**
- ROI face/text crops (YOLO, §6 biggest single jump) · diffusion reconstruction-error branch
  (DIRE/FIRE, §3 GenAI family) · ensemble of diverse recapture-trained streams.
- Re-converge convnext (lower lr / layer-decay) only if a trustworthy proxy says diversity helps.

**Net-new (see [[research/directions]]):**
- **D. Field-consistency / cross-field semantic checks** (MRZ ICAO-9303 check digits, font/baseline,
  logical issue≤expiry) — *strongest net-new bet*: a non-pixel signal orthogonal to RGB/SRM/DCT/DIRE,
  **robust to the recapture domain gap** (keys on semantic breakage, not camera artifacts). Late
  score-fuse with the SRM×recapture winner. Cost: OCR+MRZ infra (next big build, not today).
- **Per-doc-type score normalization** — a *per-group* (non-global) transform is NOT a monotonic
  no-op, so it CAN move the combined DET / APCER@1%BPCER by realigning per-type operating points.
  Needs a doc-type classifier on test (test has no type labels); in-domain val is near-perfect so
  **only LB-testable** → costs a submission. Slot in once a winner is fixed.
- **E. Tiny-region max-aggregation** — per-patch/field score → max-pool (tampered region is tiny;
  global pooling dilutes it). Bolt-on once ROI crops exist.

## Pointers

- Research survey: [[research/forgery_detection]] · net-new ideas: [[research/directions]]
- Eval harness detail: [[eval_harness]]
- Code: `freuid/` (train/predict/ensemble), configs `configs/baseline.yaml`, runner `scripts/run_baseline.sh`.
- Run outputs: `experiments/<name>_<ts>/` (config.yaml, meta.json, metrics.json, oof_val.csv, checkpoints/best.pt, submission.csv).

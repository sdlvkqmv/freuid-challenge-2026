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
| 3 | 11 G-AdaBN (06 ckpt + test-BN) | — | — | 0.15598 | test-time BN recompute ≈ no-op (06 already recap-trained) |
| 4 | 11 G-Tent (06 ckpt, BN affine) | — | — | 0.15854 | entropy ~0 (overconfident) → drift, worse |
| 5 | 13 I-focal (hard-neg, gp2/gn3) | 0.00012 | 0.00033 | 0.16161 | hard-neg mining regresses (amplifies digital shortcut) |
| 6 | 07b srm+recap 448 hi-res | 0.00022 | 0.00032 | 0.16440 | hi-res didn't help |
| 4 | 01 effb3 rgb       | 0.00013 | — | 0.17920 | prior best |
| 5 | 05 effb3 rgb + recapture | 0.00012 | 0.00022 | 0.18433 | recapture **didn't help RGB** |
| 6 | 02 effb3 rgb+srm   | 0.00011 | — | 0.18471 | SRM w/o recapture: no gain |
| 7 | 15 multicrop max-agg (06) | — | — | 0.18077 | spatial crop discards global recapture cue |
| 8 | 14 I-OHEM (hard-neg, top50%) | 0.00013 | 0.00032 | 0.18855 | OHEM most aggressive → worst hard-neg variant |
| 9 | 07a srm+recap STRONG/WIDE | 0.00022 | 0.00058 | 0.19546 | **over-aug (prob0.85) hurts** |
| 10 | 18 recap-realism V2 (full) | 0.00015 | 0.00030 | 0.20846 | synthetic-realism overfit; **best recap-val→near-worst LB** |
| 11 | 17 recap-realism V1 (chroma-moiré) | 0.00014 | 0.00034 | 0.21221 | faithful artifacts overfit synthetic domain |
| 12 | 16 multicrop topk3-agg (06) | — | — | 0.21898 | finer grid + more crop-reliance → worse |
| 8 | 08 rgb+srm+dct (+strong aug) | 0.00032 | 0.00063 | 0.24078 | DCT confounded by bad aug → see 09 |
| 9 | 09 rgb+srm+dct FAIR (06 aug) | 0.00011 | 0.00021 | 0.21476 | **DCT stream genuinely hurts** (clean test) |
| 12 | 03 convnext rgb    | 0.18008 | — | 0.35407 | under-converged |
| — | 12 F frozen CLIP/DINOv2 | 0.67 / 0.40 | — | (killed, not submitted) | **frozen semantic feats fail** — wrong signal family |
| — | 04 ensemble (rank) | 01+02=0.000117 | — | (not submitted) | no in-domain gain |

**Best still 0.15185 (attempt 06).** Session 2 (2026-06-25), all 5 subs spent, **everything regressed**:
stronger/wider aug (07a 0.19546), hi-res (07b 0.16440), DCT+bad-aug (08 0.24078), DCT-fair (09 0.21476),
fusion (10 0.15564). → **06's moderate-aug SRM+recapture is a local optimum; neither hyperparameter
pushes nor extra pixel streams (DCT) beat it. Next lever must change the signal family —
field-consistency D / ROI crops.**

**Session 3 (2026-06-27), directions G + F — both FAILED, 06 still champion:**
- **G test-time adaptation** (AdaBN 0.15598, Tent 0.15854): re-aligning BN to the test domain at
  inference does **not** help — 06's recapture-*aug training already broadened BN stats*, leaving no
  slack; Tent's entropy is ~0 (overconfident model). [[attempts/11_tta_bn_tent]]
- **F frozen foundation features** (CLIP linear / DINOv2 mlp): **killed at ep2**, in-domain FREUID
  stuck 0.67 / 0.40 (vs 06's 0.00018). Frozen CLIP/DINOv2 encode **semantics** and discard the
  **low-level forensic** signal (noise/JPEG-grid/recapture traces) ID-forgery needs. UnivFD precedent
  is for *semantic* GAN/diffusion artifacts; physical+recapture ID forgery is a different family.
  Confirms research §1 (effb3 full-finetune ≫ frozen ViT). [[attempts/12_frozen_foundation]]
- **I operating-point hard-neg mining** (focal 0.16161, OHEM 0.18855): also FAILED. recap-val said
  *better* than 06 yet LB worse — proxy lied again. Hard examples live in the **digital** train
  domain → concentrating gradient there **amplifies the domain shortcut** and widens the train/test
  gap. Metric-squeeze presumes good ranking; ranking is gated by the domain gap, not the operating
  point. [[attempts/13_hardneg_mining]]
- **⇒ Three directions (G, F, I) all confirm the same lesson: the winning recipe is a FULL fine-tune
  of a CNN with a forensic stream (SRM) + recapture aug (06). Backbone swaps (F), post-hoc adaptation
  (G), and loss reshaping (I) all fail. The ONLY remaining lever is a NEW signal family that is
  itself less domain-gap-sensitive — ROI face/text crops + max-agg, or field-consistency — added
  late-fused. Everything that reshapes the same digital-domain model is exhausted.**
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
| 09 | [DCT-fair (06 moderate aug)](attempts/09_dct_fair.md) | done · LB 0.21476 |
| 10 | [rank-fusion 06×07b](attempts/10_ensemble_06_07b.md) | done · LB 0.15564 |
| 11 | [G test-time AdaBN + Tent](attempts/11_tta_bn_tent.md) | done · AdaBN 0.15598 / Tent 0.15854 |
| 12 | [F frozen CLIP/DINOv2](attempts/12_frozen_foundation.md) | killed ep2 · not submitted (frozen feats fail) |
| 13/14 | [I hard-neg mining (focal/OHEM)](attempts/13_hardneg_mining.md) | done · focal 0.16161 / OHEM 0.18855 |
| 15/16 | [E multicrop max-agg](attempts/15_multicrop_maxagg.md) | done · max 0.18077 / topk 0.21898 |
| 17/18 | [recap-sim realism (chroma-moiré/ab/vignette)](attempts/17_recap_realism.md) | done · V1 0.21221 / V2 0.20846 |

## Remaining directions (re-prioritized after finding #0)

**✅ DONE: recapture augmentation → new best 0.15185 (attempt 06, SRM stream only).**

**❌ EXHAUSTED (session 2, all regressed vs 06) — parameter pushes don't beat the winner:**
1. ~~Stronger/longer/wider recapture aug~~ → 07a 0.19546 (over-aug HURTS), 07b 448px 0.16440 (hi-res
   neutral-worse). **06's moderate prob-0.7 aug is a local optimum.** [[attempts/07_srm_recap_push]]
2. ~~DCT/JPEG stream~~ → 08 0.24078 (confounded), **09 FAIR test 0.21476** with 06's moderate aug →
   **DCT stream genuinely hurts, dropped.** SRM is the forensic stream that works. [[attempts/09_dct_fair]]
3. ~~Real recapture validation probe~~ → built (`freuid/probe.py`) but **also broken**: n=20 too
   small for FREUID, AUROC inverted vs LB. No local proxy ranks at the top. [[eval_harness]]

**⇒ Next must change the SIGNAL FAMILY, not the knobs. Updated order ([[research/directions]]):**
- ~~**G — test-time BN / Tent**~~ → DONE, FAILED (AdaBN 0.15598, Tent 0.15854). [[attempts/11_tta_bn_tent]]
- ~~**F — frozen foundation features**~~ → DONE, FAILED (killed ep2, frozen semantic feats are the
  wrong signal family for low-level ID forgery). [[attempts/12_frozen_foundation]]
- ~~**I — operating-point hard-neg mining**~~ → DONE, FAILED (focal 0.16161, OHEM 0.18855; amplifies
  digital shortcut). [[attempts/13_hardneg_mining]]
- ~~**E — multicrop max-agg (detector-free spatial focus)**~~ → DONE, FAILED (max 0.18077, topk
  0.21898). Spatial cropping discards the dominant *global* recapture cue. [[attempts/15_multicrop_maxagg]]
- ~~**Recapture-realism retrain**~~ → DONE, FAILED (V1 0.21221, V2 0.20846). Faithful synthetic
  artifacts OVERFIT the synthetic domain; 06's generic moderate aug is optimal. Also proved **recap-val
  is anti-correlated for aug changes** (V2 best recap-val → near-worst LB). [[attempts/17_recap_realism]]
1. **D — field-consistency** (font/baseline/layout; MRZ weak — dataset is 4 DLs + 1 ID, DLs lack MRZ).
   Orthogonal **non-pixel** signal, **domain-gap-robust** (keys on semantic breakage, not camera
   artifacts) → THE one untried family that does NOT depend on the recapture texture 06 already owns.
   Now the only high-EV lever left. Cost: OCR + layout infra (real build, next session).
2. **Additive face-ROI stream** (research §6) — keeps global context (unlike failed multicrop),
   late-fused. Lower priority: E shows global recapture cue dominates. Needs face-detector infra.
- **🔴 9 CONSECUTIVE DIRECTIONS REGRESS vs 06** (07 pushes, 08/09 DCT, 10 fusion, 11 BN-adapt, 12
  frozen ViT, 13/14 hard-neg, 15/16 multicrop, 17/18 realism). 06's moderate-aug effb3+SRM is an
  extremely robust local optimum. **Every lever that touches pixels / the digital-domain model
  overfits.** Anti-pattern confirmed: adding specificity (streams, realism, hard examples, frozen
  semantic feats, spatial crops) all overfit; only generic moderate recapture aug generalizes.
  → **STOP blind LB-probing** (each just re-confirms 06). The single remaining bet is the **non-pixel
  field-consistency family (D)**, which is independent of the recapture domain gap; build it properly
  (OCR + layout consistency) and late-fuse with 06. The LB top (~0.00063) implies such a fundamentally
  different signal exists.

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

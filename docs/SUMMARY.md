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
| — | 24 ens 06:resnext50 3:1 | — | — | 0.17064 | diverse-backbone fuse regresses (effb3 uniquely OOD-robust) |
| — | 29 ens 06+effb3-seed123 | — | — | 0.17400 | **same-backbone reseed fuse also regresses** (reseeds worse solo) |
| — | 25 ens 06:resnext50 2:1 | — | — | 0.17409 | " (monotonic in 06 weight → asymptotes to 06 from above) |
| — | 23 ens 06+resnet50 equal | — | — | 0.17868 | " |
| — | 28 ens 3× effb3 seeds (06+123+456) | — | — | 0.18088 | 3-way reseed fuse worse than 2-way (all reseeds worse than 06) |
| — | 27 effb3 SEED456 solo | 0.00017 | 0.00009 | 0.19178 | **06's exact recipe, diff seed → 0.19 (06 is a lucky seed)** |
| — | 26 effb3 SEED123 solo | 0.00009 | 0.00009 | 0.21497 | " → 0.21. recipe LB seed-variance 0.152-0.215; recap-val can't select |
| 4 | 01 effb3 rgb       | 0.00013 | — | 0.17920 | prior best |
| 5 | 05 effb3 rgb + recapture | 0.00012 | 0.00022 | 0.18433 | recapture **didn't help RGB** |
| 6 | 02 effb3 rgb+srm   | 0.00011 | — | 0.18471 | SRM w/o recapture: no gain |
| 7 | 15 multicrop max-agg (06) | — | — | 0.18077 | spatial crop discards global recapture cue |
| 8 | 14 I-OHEM (hard-neg, top50%) | 0.00013 | 0.00032 | 0.18855 | OHEM most aggressive → worst hard-neg variant |
| 9 | 07a srm+recap STRONG/WIDE | 0.00022 | 0.00058 | 0.19546 | **over-aug (prob0.85) hurts** |
| — | 21 resnet50 srm+recap solo | 0.00000 | 0.00043 | 0.21612 | diverse backbone: fits digital identically, **generalizes worse to recapture** |
| — | 22 resnext50 srm+recap solo | 0.00000 | 0.00053 | (in 24/25) | same family as r50 (spearman0.95); most decorrelated vs 06 (0.82) |
| — | 20 T1 VLM zero-shot (Qwen2-VL-2B) | — | — | (not submitted) | **AUROC 0.4452 < random** — VLM can't judge these forgeries zero-shot |
| 10 | 19 D face-consistency fusion (06+face) | — | — | 0.20334 | validated AUROC0.88 but **redundant with 06** (r=0.66) |
| 11 | 18 recap-realism V2 (full) | 0.00015 | 0.00030 | 0.20846 | synthetic-realism overfit; **best recap-val→near-worst LB** |
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
| 19 | [D face-consistency (main↔ghost)](attempts/19_face_consistency.md) | done · fusion 0.20334 (signal valid AUROC0.88 but redundant w/ 06) |
| 20-29 | [T1 VLM external-prior + T2/T3 diverse & reseed ensembles](attempts/20_vlm_and_diverse_ensemble.md) | done · VLM AUROC0.44 (not sub) / ensembles 0.170-0.215 (**12th direction; 06 is a LUCKY SEED — recipe LB var 0.152-0.215**) |

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
- ~~**D — field-consistency (photo identity: main↔ghost)**~~ → DONE. Signal VALIDATED (Mauritius
  AUROC 0.88, 95% cov) but **REDUNDANT with 06** (pearson 0.66; 06 already flags 90% of swaps at mean
  score 0.88). Fusion regressed (0.20334) — injects the face signal's 12% error across 1,695 images to
  recover only 26 swaps 06 missed. 06's SRM stream already detects photo substitution.
  [[attempts/19_face_consistency]]
1. **D — OCR date-logic** (issue≤expiry, DOB plausibility) — the one untried D sub-signal, universal
   across types. BUT likely also redundant (field-value edits leave pixel traces 06 sees) + needs OCR
   infra. Low EV after D-photo proved redundant.
2. **Confident-only face boost** (push only incons>0.8 swaps up, never move others) — marginal, ceiling
   ~26 test images. Cheap if revisited.
- **🔴🔴 06 IS A LUCKY SEED (new, highest-impact this session).** 06's exact recipe (effb3+SRM+recap)
  re-trained at seeds 123 & 456 scored **0.21497 / 0.19178** on LB — vs 06's 0.15185. Identical
  in-domain (0.0000) and *better* recap-val (0.00009 < 0.00042), yet far worse LB. So the recipe has a
  **wide LB seed-variance band (~0.152-0.215)** and 06 sits at the lucky low end. **No local proxy
  selects the good seed** (recap-val anti-correlated even across seeds; in-domain useless) → the good
  seed is only findable by spending public-LB submissions. Two implications: reseed *ensembles* all
  regress (members individually 0.19-0.21, drag the fuse to 0.174-0.181); and **06's 0.15185 is partly
  public-LB luck** — on the private LB (142k, +2 unseen types) it may regress toward the band mean, so a
  lower-variance effb3-reseed *ensemble* is a candidate hedge for the FINAL submission (revisit before
  2026-07-14). [[attempts/20_vlm_and_diverse_ensemble]]
- **🔴 12 CONSECUTIVE DIRECTIONS DO NOT BEAT 06** (07 pushes, 08/09 DCT, 10 fusion, 11 BN-adapt, 12
  frozen ViT, 13/14 hard-neg, 15/16 multicrop, 17/18 realism, 19 face-consistency, **20-29 VLM +
  diverse-backbone + reseed ensembles**). Confirmed reasons: **(a)** adding specificity overfits
  (streams/realism/hard-neg/frozen/spatial all worsen); **(b)** 06's SRM forensic stream is **more
  semantically complete than expected** — even an orthogonal-by-design signal (face identity) is 65%
  correlated with it and adds nothing; **(c)** [[attempts/20_vlm_and_diverse_ensemble]] — the
  **backbone itself (effb3) carries the OOD-robustness**: resnet50/resnext50 on 06's EXACT recipe fit
  the digital train identically (in-domain 0.0000) yet generalize far worse to recapture (solo 0.216 vs
  0.152), so rank-fusion at any 06-weight regresses monotonically (0.179→0.171→0.170, asymptote 0.152
  from above); **(d)** the external-prior escape hatch is also closed — a zero-shot VLM (Qwen2-VL-2B)
  scores **AUROC 0.44 < random**, so world-knowledge can't substitute for forensic pixels either.
  → The pixel-forensic + recapture-aug recipe has captured the accessible signal, AND the accessible
  zero-shot external prior is too weak. **Beating 06 now needs new DATA or a fine-tuned strong VLM, not
  a same-day bolt-on** (LB top ~0.00063 implies a gain exists). Realistic options: **accept 06 as the
  final submission**, or a from-scratch domain-gap rethink (genuinely real recapture training data, or
  fine-tune a 7B+ VLM on recapture). Stop bolt-on LB-probing.

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

# Attempt 03 — ConvNeXt-Tiny, RGB

Back to [[SUMMARY]] · proxy detail [[eval_harness]] · research [[research/forgery_detection]]

## Hypothesis
ConvNeXt is a strong modern conv backbone (research §6 lists ConvNeXt/Swin alongside EfficientNet).
Primary purpose here: **architecture diversity for the ensemble** — a backbone that errs
differently from EfficientNet so rank-fusion (attempt 04) gains from decorrelated mistakes.

## Setup
- Backbone `convnext_tiny` (timm, pretrained), single-logit head, RGB 3ch.
- Same training recipe as [01](01_effb3_rgb.md); fit bs 48 at 384px on 10GB (ran on GPU2).
- Val: stratified fold0 (identical split to 01/02 → OOF predictions are directly fusible).

## Result
| metric | value |
|---|---|
| val FREUID (in-domain, stratified f0) | 0.18008 (best epoch 4) |
| val AuDET | 0.0774 |
| val APCER@1%BPCER | 0.262 |
| train_loss (final) | 0.326 (vs effb3 0.006) |
| **Kaggle public LB** | **0.35407** (`submission 54013119`) — worst of all attempts |

**Under-converged** — final train_loss 0.326 vs effb3's 0.006. `convnext_tiny` needs a lower LR /
layer-wise decay than the shared `lr=3e-4` recipe; at this LR it barely trained. Excluded from the
ensemble (it drags rank-fusion: 01+02 0.000117 → 01+02+03 0.000190).

## Notes
- Same fold as 01/02 by construction (same seed/scheme) → `oof_val.csv` rows align for stacking.
- Re-run with lr≈5e-5 + `layer_decay` before trusting convnext as an ensemble member.

# Attempt 01 — EfficientNet-B3, RGB baseline

Back to [[SUMMARY]] · proxy detail [[eval_harness]] · research [[research/forgery_detection]]

## Hypothesis
EfficientNet-B3 is the research-recommended ID-forgery backbone (SIDTD: 0.994 acc / 1.000 ROC-AUC,
crushing ViT-L/16 at 0.552 / 0.501, research §1). Establish a clean full-image RGB baseline to lock
the pipeline and the local FREUID proxy before adding forensic streams.

## Setup
- Backbone `tf_efficientnet_b3` (timm, pretrained), single-logit head, `in_chans=3`.
- 384×384, RandomResizedCrop(0.7–1.0), color-jitter 0.1, **no hflip** (mirrors document text).
- BCEWithLogits, auto `pos_weight = n_neg/n_pos` (train pos-rate 0.4231).
- AdamW lr 3e-4, wd 0.05, cosine + 1ep warmup, AMP, grad-clip 1.0, 6 epochs.
- Val: stratified fold0 on (type,label); n_train 55,481 / n_val 13,871.
- GPU3 (RTX 3080 10GB), bs 24 (bs 48 OOMs at 384), `expandable_segments:True`.

## Result
| metric | value |
|---|---|
| val FREUID (in-domain, stratified f0) | 0.00013 (best epoch 4) |
| val AuDET | 9.17e-5 |
| val APCER@1%BPCER | 1.70e-4 |
| Kaggle LB | not submitted (02 was the better single; 1 sub/day spent on it) |

Slightly worse than [02 effb3+SRM](02_effb3_srm.md) in-domain (0.00013 vs 0.00011). Subject to the
same domain-gap caveat — in-domain near-zero is untrustworthy (finding #0 in [[SUMMARY]]).

## Notes
- bs had to drop 48→24 for 10GB at 384px. If throughput-bound, try 320px bs48 or grad-accum.
- This is the in-domain (optimistic) proxy; OOD via group_holdout still pending (see SUMMARY remaining).

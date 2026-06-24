# Attempt 05 — EfficientNet-B3, RGB + recapture augmentation

Back to [[SUMMARY]] · proxy [[eval_harness]] · research [[research/forgery_detection]]

## Hypothesis
Apply recapture augmentation (finding #0) to the best LB config so far — plain RGB ([01](01_effb3_rgb.md),
LB 0.17920). Expectation: simulating print-and-capture closes the train→test domain gap and improves LB.

## Setup
- `tf_efficientnet_b3`, RGB 3ch, 384px, bs24, 6ep, GPU3.
- Recapture aug `prob=0.7` (same params as [06](06_effb3_srm_recap.md)); selection by `recap_freuid`.

## Result
| metric | value |
|---|---|
| in-domain clean val FREUID | 0.00012 |
| recapture-sim val FREUID | 0.00022 (best epoch 4) |
| **Kaggle public LB** | **0.18433** (`submission 54014453`) |

**Recapture aug did NOT help plain RGB** — 0.17920 ([01](01_effb3_rgb.md)) → 0.18433, slightly worse.
Contrast with [06](06_effb3_srm_recap.md) where the same aug *helped* the SRM stream (0.18471→0.15185).

## Takeaway
Recapture augmentation is only useful **with a forensic noise stream** (root finding #1 in [[SUMMARY]]).
On RGB alone it mostly adds distortion without a channel that turns recapture artifacts into signal.
The recap-sim proxy (0.00022) looked better than 06's (0.00042) but lost on the LB → proxy unreliable.

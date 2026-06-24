# Attempt 06 — EfficientNet-B3 + SRM + recapture augmentation 🥇

Back to [[SUMMARY]] · proxy [[eval_harness]] · research [[research/forgery_detection]]

## Hypothesis
Finding #0: train is ~all digital, test emphasizes print-and-capture → models collapse. Simulate the
recapture pipeline on training images (research §4) so the model learns recapture-surviving cues.
Pair it with the SRM noise stream ([02](02_effb3_srm.md)) since recapture is fundamentally a second
capture pipeline — exactly what camera/noise forensics target.

## Setup
- `tf_efficientnet_b3`, `streams=[rgb, srm]` (6ch), 384px, bs24, 6ep, GPU4.
- Recapture aug (`freuid/augment.py`, `prob=0.7`): down-up resample, defocus blur, moiré,
  gamma/brightness/white-balance shift, sensor noise, JPEG recompress.
- **Model selection by `recap_freuid`** (recapture-simulated val), not clean val.

## Result
| metric | value |
|---|---|
| in-domain clean val FREUID | 0.00018 |
| recapture-sim val FREUID | 0.00042 (best epoch 3) |
| **Kaggle public LB** | **0.15185** (`submission 54014459`) — **best of all attempts** |

**Recapture aug + SRM is the winning combo.** vs the no-recapture SRM model ([02](02_effb3_srm.md)
LB 0.18471) this is a clear gain (0.18471 → 0.15185, ~18% rel), and it beats the prior best plain-RGB
([01](01_effb3_rgb.md) 0.17920).

## Why it worked (vs RGB+recapture [05](05_effb3_rgb_recap.md) which did NOT)
- 05 (rgb+recapture) regressed to 0.18433 — recapture aug alone, without a forensic stream, didn't help.
- 06 (srm+recapture) improved. → **synergy**: SRM cues are discriminative only when the input
  distribution contains recapture artifacts; recapture aug is exploitable only with a noise stream.
- This is root finding #1 in [[SUMMARY]].

## Notes / next
- recap-sim proxy mis-ranked 05 (0.00022) vs 06 (0.00042) relative to LB — still circular (same sim).
  Don't trust its absolute ordering; it only flags the right *family*.
- Push: stronger/longer recapture training, add DCT/JPEG stream, real held-out recapture probe
  (the 20 `is_digital=False` rows), higher resolution. See [[SUMMARY]] remaining directions.

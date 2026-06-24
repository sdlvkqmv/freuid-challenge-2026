# Attempt 02 — EfficientNet-B3, RGB + SRM noise stream

Back to [[SUMMARY]] · proxy detail [[eval_harness]] · research [[research/forgery_detection]]

## Hypothesis
Noise-residual streams expose tamper / camera-fingerprint inconsistencies that raw RGB misses;
multi-stream (RGB + residual) beats either alone (research §2, §6: EdgeDoc, TruFor, guided-noise).
SRM high-pass residual is the cheapest such stream (fixed conv, no extra training data).

## Setup
- Same as [01](01_effb3_rgb.md) but `model.streams=[rgb, srm]` → `in_chans=6`.
- SRM = 3 fixed Fridrich/Kodovsky high-pass kernels (5×5), applied per channel, concatenated to RGB.
  Implemented in `freuid/model.py:SRMResidual` (computed on-GPU in forward).
- GPU4, bs 24, 384px, 6 epochs.

## Result
| metric | value |
|---|---|
| val FREUID (in-domain, stratified f0) | **0.00011** (best epoch 4) |
| val AuDET | 5.90e-5 |
| val APCER@1%BPCER | 1.70e-4 |
| **Kaggle public LB** | **0.18471** (`submission 54012138`) — 2nd, beaten by plain RGB (01=0.17920) |

**In-domain best, but collapsed on the real public test (0.00011 → 0.18471, ~1700×) — and on the
LB it LOST to plain RGB ([01](01_effb3_rgb.md) 0.17920).** SRM's in-domain edge was on digital-domain
noise that does not survive print-and-capture. SRM is not helping here; revisit only with recapture
augmentation + an OOD proxy.
This single submission produced the project's central finding (#0 in [[SUMMARY]]):
- Train is 69,332 digital / **20 recaptured**; the test emphasizes **print-and-capture** images.
- The model never learned the recapture domain → its near-perfect in-domain score is a shortcut.
- SRM beat RGB *in-domain* (AuDET 5.9e-5 vs 9.2e-5) but that gain is on the wrong domain; do not
  trust it. SRM may still help once recapture augmentation/validation is in place — re-test then.

OOF rank-fusion (computed on shared val fold): 02 alone 0.000115 · 01+02 0.000117 · 01+02+03 0.000190
→ fusion gives **no in-domain gain** (saturated) and convnext hurts. See [04](04_ensemble.md).

## Notes
- SRM computed on the ImageNet-normalized tensor; high-pass removes the DC/mean shift so the
  normalization offset is harmless, but absolute residual scale differs from a raw-pixel SRM.
  If this underperforms, try computing SRM on un-normalized pixels (needs a 2nd dataset output).
- Next forensic stream to add if this helps: DCT/JPEG-artifact (CAT-Net style), research §2.

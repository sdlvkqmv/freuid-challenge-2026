# Attempt 04 — Rank-fusion ensemble (01 + 02 + 03)

Back to [[SUMMARY]] · proxy detail [[eval_harness]] · research [[research/forgery_detection]]

## Hypothesis
Score-level fusion of complementary detectors beats any single model on the private split
(research §6: EdgeDoc+TruFor 0.59/0.71 → 0.79). Because FREUID is ranking-based ([[eval_harness]]),
fuse by **rank-average** (scale-free) rather than raw-probability mean.

## Setup
- `freuid/ensemble.py` rank-averages the per-model `submission.csv` (and/or `oof_val.csv` for the
  local proxy). Equal weights to start; tune weights on OOF if it helps.
- Members: 01 effb3-rgb, 02 effb3-srm, 03 convnext-rgb (same val fold → OOF directly comparable).

## Result
OOF rank-fusion on the shared stratified val fold (lower better):

| combo | OOF FREUID |
|---|---|
| 02 srm (best single) | 0.000115 |
| 01 + 02 | 0.000117 |
| 01 + 02 + 03 | 0.000190 |

**No in-domain gain from fusion** — single 02 wins; 01+02 essentially tied; adding the
under-converged convnext (03) hurts. Reason: 01 and 02 are highly correlated (same backbone,
in-domain saturated near zero → no room), and 03 is just weak. **Not submitted.**

Bigger picture: in-domain OOF can't show fusion's real value because the proxy is saturated AND
broken (finding #0). Re-evaluate ensembling once a recapture/OOD validation exists and members are
genuinely diverse (e.g. RGB + noise-stream + DCT, each re-trained with recapture augmentation).

## Notes
- If a member hurts the OOF fusion, drop it. Diversity (different backbone / stream) matters more
  than individual strength for fusion gain.

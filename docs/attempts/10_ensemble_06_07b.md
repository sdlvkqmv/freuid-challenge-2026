# Attempt 10 — rank-fusion ensemble (06 × 07b)

Back to [[SUMMARY]] · members [[attempts/06_effb3_srm_recap]] · [[attempts/07_srm_recap_push]]

## Hypothesis
Rank-fuse the two best recapture-trained models for ranking diversity (scale-free, targets
AuDET/APCER@1%BPCER). 06 (srm+recap 384, LB 0.15185) + 07b (srm+recap 448 hi-res, LB 0.16440),
weighted 2:1 toward the stronger 06. `freuid/ensemble.py`.

## Result
| | **Kaggle public LB** |
|---|---|
| 06 alone | 0.15185 |
| 07b alone | 0.16440 |
| **10 fusion 2×06 + 1×07b** | **0.15564** |

## Takeaway
Fusion beats 07b but **does not beat 06 alone** (0.15564 > 0.15185). 07b is correlated with 06
(same SRM+recapture recipe, only resolution differs) and weaker, so it dilutes rather than
diversifies. → For ensembling to help, members must be **genuinely diverse** (different signal
family — e.g. a working DCT/field-consistency model), not resolution variants of the same recipe.
06 remains the single best.
